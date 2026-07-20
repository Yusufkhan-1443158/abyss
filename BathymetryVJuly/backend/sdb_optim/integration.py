# INTEGRATION FACADE — orchestrator wiring for the sdb_optim optimization package.
#
# This module is the single seam where the four delivered optimization tracks
# (T1 physics-informed architecture, T2 feature engineering/fusion, T3
# validation/calibration, T4 HPC IO) are composed against the LIVE pipeline
# primitives in `backend/sdb_cnn_baseline.py`, `backend/unet_sdb.py`,
# `backend/icesat2_bathy.py` and `backend/very_hr_engine.py`.
#
# Per OPTIMIZE_ORCHESTRATOR.md, the individual tracks never edit pipeline files;
# they ship drop-in primitives and document their integration point. The
# orchestrator wires them here, by COMPOSITION rather than mutation, so the
# production fast path and any in-flight ablation that depend on the existing
# files keep running unchanged. Opt in to the optimized behaviour by calling
# these factories from a training/eval driver; default production stays as-is.
#
# Every symbol referenced in WIRING below is asserted importable by
# tests/test_integration.py, so the documented wiring can never silently drift
# from the real pipeline API.
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- Track modules (guarded so a partial build still imports) ---------------
from .physics_loss import PhysicsInformedLoss
from .hybrid_losses import HybridStructuralLoss
from .feature_engineering import compute_optical_indices, compute_temporal_stats
from .spatial_cv import spatial_block_kfold, fold_min_distance
from .stratified_metrics import stratified_report
from .uncertainty import total_uncertainty
from .subsystem_interface import InferenceConfig, run_inference_job

__all__ = [
    "recommended_config", "WIRING",
    "build_physics_criterion", "build_structural_criterion",
    "extend_feature_cube", "temporal_features",
    "cross_validate", "depth_stratified_report", "combined_uncertainty",
    "InferenceConfig", "run_inference_job",
]


# ---------------------------------------------------------------------------
# Wiring registry: optim symbol  ->  live pipeline symbol it augments/feeds.
# Used as documentation AND as a verifiable contract (see test_integration).
# ---------------------------------------------------------------------------
WIRING: Tuple[Tuple[str, str, str], ...] = (
    ("sdb_optim.physics_loss.PhysicsInformedLoss",
     "sdb_cnn_baseline.HeteroNLLLoss",
     "T1: wraps the pointwise NLL with Stumpf/Lyzenga/monotonicity regularisers."),
    ("sdb_optim.hybrid_losses.HybridStructuralLoss",
     "sdb_cnn_baseline.HeteroNLLLoss",
     "T3: augments the raster U-Net loss with SSIM + gradient structural terms."),
    ("sdb_optim.feature_engineering.compute_optical_indices",
     "sdb_cnn_baseline.build_feature_cube",
     "T2: extra optical-index channels concatenated onto the A0 12-channel cube."),
    ("sdb_optim.feature_engineering.compute_temporal_stats",
     "sdb_cnn_baseline.build_feature_cube",
     "T2: per-pixel temporal median/var/count from the multi-date reflectance stack."),
    ("sdb_optim.spatial_cv.spatial_block_kfold",
     "sdb_cnn_baseline.make_spatial_block_centers",
     "T1: full K-fold extension of the single spatial-block split (>=500 m buffer)."),
    ("sdb_optim.stratified_metrics.stratified_report",
     "sdb_cnn_baseline.evaluate",
     "T3: IHO-aligned depth-stratified report superseding inline per-bin printing."),
    ("sdb_optim.uncertainty.total_uncertainty",
     "sdb_cnn_baseline.PatchCNN",
     "T3: combines the hetero sigma head (aleatoric) with MC-dropout (epistemic)."),
    ("sdb_optim.subsystem_interface.run_inference_job",
     "icesat2_bathy.process_granule",
     "T4: windowed/parallel inference IO scaffolding for grid-scale prediction."),
)


def recommended_config() -> Dict[str, Any]:
    """Central, citable defaults the orchestrator recommends when enabling the
    optimized path. All regularisers default low so the optimized path stays a
    near-no-op perturbation of the canonical A0 model (mussafah 0.9237 m) until
    a driver deliberately raises them in an ablation.

    Returns a plain dict so it can be JSON-logged into a run's provenance header.
    """
    return {
        # T1 physics-informed loss (pointwise PatchCNN head)
        "lambda_stumpf": 0.10,      # Stumpf log-ratio affine consistency (Stumpf 2003)
        "lambda_lyzenga": 0.05,     # per-batch Lyzenga OLS regulariser (Lyzenga 2006)
        "lambda_monotonic": 0.10,   # deeper-water => darker reflectance monotonicity hinge
        # T3 structural loss (raster U-Net head)
        "lambda_ssim": 0.15,        # structural similarity (Wang 2004)
        "lambda_grad": 0.10,        # Sobel gradient / bathymetric edge fidelity
        # T1 spatial CV honesty contract
        "kfold": 5,
        "min_test_train_dist_m": 500.0,   # S2 reflectance autocorrelation length
        "block_size_m": 2000.0,
    }


# ---------------------------------------------------------------------------
# T1 / T3 — loss composition (wraps the live HeteroNLLLoss)
# ---------------------------------------------------------------------------
def build_physics_criterion(
    base: Optional[Any] = None,
    *,
    lambda_stumpf: float = 0.10,
    lambda_lyzenga: float = 0.05,
    lambda_monotonic: float = 0.10,
) -> PhysicsInformedLoss:
    """Compose the T1 physics-informed loss around the live pointwise base loss.

    `base` defaults to `backend.sdb_cnn_baseline.HeteroNLLLoss()`. The returned
    criterion is called exactly like HeteroNLLLoss — ``crit(mu, logsigma,
    target, stumpf_ratio=..., ln_blue=..., weights=...)`` — so it drops into the
    PatchCNN/ResidualCNN training loop in `train_model`/`train_default_model`.
    """
    if base is None:
        from backend.sdb_cnn_baseline import HeteroNLLLoss  # live primitive
        base = HeteroNLLLoss()
    return PhysicsInformedLoss(
        base, lambda_stumpf=lambda_stumpf,
        lambda_lyzenga=lambda_lyzenga, lambda_monotonic=lambda_monotonic,
    )


def build_structural_criterion(
    base: Optional[Any] = None,
    *,
    lambda_ssim: float = 0.15,
    lambda_grad: float = 0.10,
) -> HybridStructuralLoss:
    """Compose the T3 structural loss around the live base loss for the raster
    U-Net path (`backend.unet_sdb`). Called as ``crit(mu, logsigma, target,
    pred_raster=..., true_raster=...)``; with the raster kwargs omitted it
    reduces to the pointwise base loss, so it is safe to drop in unconditionally.
    """
    if base is None:
        from backend.sdb_cnn_baseline import HeteroNLLLoss
        base = HeteroNLLLoss()
    return HybridStructuralLoss(base, lambda_ssim=lambda_ssim, lambda_grad=lambda_grad)


# ---------------------------------------------------------------------------
# T2 — feature engineering (extends build_feature_cube's A0 cube)
# ---------------------------------------------------------------------------
def extend_feature_cube(
    blue: np.ndarray, green: np.ndarray, red: np.ndarray, nir: np.ndarray,
    *,
    coastal_aerosol: Optional[np.ndarray] = None,
    swir: Optional[np.ndarray] = None,
    deep_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return the (H, W, C) optical-index channels + names that EXTEND the
    canonical A0 12-channel cube from `build_feature_cube`. Concatenate the
    returned cube onto the A0 cube along the channel axis before building the
    SDBPatchDataset.
    """
    return compute_optical_indices(
        blue, green, red, nir,
        coastal_aerosol=coastal_aerosol, swir=swir, deep_mask=deep_mask,
    )


def temporal_features(
    time_series: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_valid_scenes: int = 2,
) -> Tuple[np.ndarray, List[str]]:
    """Per-pixel temporal median/variance/count from a multi-date reflectance
    stack — the glint/cloud-suppression channels that feed `build_feature_cube`.
    """
    return compute_temporal_stats(time_series, valid_mask=valid_mask,
                                  min_valid_scenes=min_valid_scenes)


# ---------------------------------------------------------------------------
# T1 / T3 — validation & calibration (honesty contract)
# ---------------------------------------------------------------------------
def cross_validate(
    coords: np.ndarray, k: int = 5, *,
    block_size_m: float = 2000.0,
    min_test_train_dist_m: float = 500.0,
    seed: int = 42, verbose: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leakage-safe spatial K-fold over (lat, lon) coords — the honesty-contract
    splitter. Drop-in K-fold extension of the single `SPLIT_MODE=spatial_block`
    split used in `very_hr_engine.py`.
    """
    return spatial_block_kfold(
        coords, k=k, block_size_m=block_size_m,
        min_test_train_dist_m=min_test_train_dist_m, seed=seed, verbose=verbose,
    )


def depth_stratified_report(
    pred: np.ndarray, truth: np.ndarray, *, print_table: bool = False,
):
    """IHO-aligned depth-stratified RMSE/MAE/R²/MBE report (0–2/2–5/5–10/10+ m).
    Pass the held-out test arrays from `evaluate` straight in.
    """
    return stratified_report(pred, truth, print_table=print_table)


def combined_uncertainty(aleatoric: np.ndarray, epistemic: np.ndarray) -> np.ndarray:
    """Total predictive σ = sqrt(σ_aleatoric² + σ_epistemic²): the hetero sigma
    head combined with the MC-dropout/ensemble spread.
    """
    return total_uncertainty(aleatoric, epistemic)
