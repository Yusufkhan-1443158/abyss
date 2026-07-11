# INTEGRATION: Extends the heteroscedastic sigma head already in PatchCNN /
# ResidualCNN (backend/sdb_cnn_baseline.py) and the unet hetero_nll path in
# backend/unet_sdb.py to produce calibrated, decomposed uncertainty maps.
#
# Primary integration points:
#   from backend.sdb_optim.uncertainty import (
#       mc_dropout_predict, deep_ensemble_predict, calibration_curve,
#       total_uncertainty,
#   )
#
# mc_dropout_predict wraps any PatchCNN / ResidualCNN / unet model that has
#   Dropout layers; call with model in training mode (dropout active).
#
# deep_ensemble_predict averages across a list of models (e.g. 5-member
#   ensemble trained from different random seeds).
#
# calibration_curve checks how well nominal coverage intervals (10%,20%,...,95%)
#   match empirical coverage — used to validate the σ-temperature calibration
#   from calibrate_sigma() in sdb_cnn_baseline.
#
# total_uncertainty combines aleatoric (σ_a from log-sigma head) and epistemic
#   (σ_e from MC-dropout / ensemble std) in quadrature: σ_total = √(σ_a² + σ_e²).
"""
Predictive uncertainty quantification for Satellite-Derived Bathymetry.

Depth prediction errors have two distinct sources that should be reported separately:

1. **Aleatoric uncertainty** (irreducible, data-driven):
   Captured by the heteroscedastic log-sigma head present in PatchCNN and
   ResidualCNN (backend/sdb_cnn_baseline.py) and in the unet hetero-NLL loss
   (backend/unet_sdb.py).  High aleatoric σ flags pixels where the spectral
   signal is ambiguous — e.g., turbid water, sun-glint patches, macro-algae.
   The aleatoric σ is output by the model's logvar head as:
       σ_a = exp(0.5 * log_var)

2. **Epistemic uncertainty** (reducible, model/data coverage):
   Captured by MC-Dropout (Gal & Ghahramani 2016) or by a deep ensemble
   (Lakshminarayanan et al. 2017).  High epistemic σ flags pixels or depth
   ranges where the training set lacks coverage — e.g., deep basins with few
   ICESat-2 transects, shallow areas masked as land, highly turbid margins.
   Epistemic uncertainty is reduced by adding more training data or better
   regularisation.

Combined total uncertainty (independent σ addition in quadrature):
   σ_total = √(σ_a² + σ_e²)

This decomposition supports:
  - CATZOC depth-confidence layers (IHO S-52): flag pixels with σ_total > THR.
  - Active-learning sounding prioritisation: direct survey resources to high-σ_e zones.
  - Post-hoc bias correction: use aleatoric σ as observation noise in kriging.

References
----------
Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation:
  Representing model uncertainty in deep learning. ICML 2016.
Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable
  predictive uncertainty estimation using deep ensembles. NeurIPS 2017.
Kendall, A. & Gal, Y. (2017). What uncertainties do we need in Bayesian deep
  learning for computer vision? NeurIPS 2017.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "mc_dropout_predict",
    "deep_ensemble_predict",
    "calibration_curve",
    "total_uncertainty",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _enable_dropout(model: nn.Module) -> None:
    """Set all Dropout (and variants) layers to training mode while leaving
    BatchNorm in eval mode.  This is the standard MC-Dropout inference recipe
    (Gal & Ghahramani 2016): dropout is active → stochastic forward pass.

    BatchNorm stays in eval mode to use population statistics, not batch stats,
    which would destabilise predictions at small batch sizes.
    """
    model.eval()  # start from eval (fixes BatchNorm)
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()  # re-enable dropout only


def _forward_pass(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int = 512,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run one forward pass; return (mu, sigma_aleatoric).

    Handles models with two outputs (mu, log_var) and models with three
    outputs (mu, log_var, residual) — the ResidualCNN signature.
    If the model returns a single tensor, sigma_aleatoric is None.

    Parameters
    ----------
    model : nn.Module in correct mode (eval or MC-dropout-active)
    x     : (N, ...) input tensor (patches or feature vectors)
    batch_size : inference batch size (default 512)

    Returns
    -------
    mu    : (N,) float32 depth predictions
    sigma : (N,) float32 aleatoric sigma (or None if not available)
    """
    device = next(model.parameters()).device
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_mu: List[np.ndarray] = []
    all_sig: List[Optional[np.ndarray]] = []

    with torch.no_grad():
        for (batch_x,) in loader:
            out = model(batch_x.to(device))
            if isinstance(out, (tuple, list)):
                mu_t  = out[0]
                lv_t  = out[1]
                sigma_t = torch.exp(0.5 * torch.clamp(lv_t, -6.0, 6.0))
                all_mu.append(mu_t.cpu().numpy().ravel())
                all_sig.append(sigma_t.cpu().numpy().ravel())
            else:
                all_mu.append(out.cpu().numpy().ravel())
                all_sig.append(None)

    mu_arr = np.concatenate(all_mu)
    if all_sig[0] is not None:
        sig_arr: Optional[np.ndarray] = np.concatenate(
            [s for s in all_sig if s is not None]
        )
    else:
        sig_arr = None

    return mu_arr, sig_arr


# ─────────────────────────────────────────────────────────────────────────────
# MC-Dropout prediction
# ─────────────────────────────────────────────────────────────────────────────

def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 30,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Monte Carlo Dropout prediction: mean + epistemic uncertainty.

    Activates Dropout (keeps BatchNorm in eval mode) and runs `n_passes`
    stochastic forward passes.  The std across passes estimates epistemic
    uncertainty (model uncertainty due to limited training coverage).

    Parameters
    ----------
    model    : nn.Module with at least one Dropout layer (PatchCNN, ResidualCNN,
               or any U-Net with MC-Dropout support)
    x        : (N, ...) input tensor on CPU or GPU
    n_passes : number of stochastic forward passes (default 30).
               Literature recommendation: 30–100 passes; 30 is a practical
               minimum for reasonable variance estimates.
    batch_size : inference batch size

    Returns
    -------
    mean_pred      : (N,) posterior mean depth (m)
    epistemic_std  : (N,) epistemic standard deviation (m) — std across passes
    aleatoric_mean : (N,) mean aleatoric σ across passes (or None if unavailable)

    Notes
    -----
    If the model has no Dropout layers, epistemic_std will be near zero.
    This is expected behaviour (not a bug); the user should use deep ensembles.
    """
    if n_passes < 2:
        raise ValueError(f"n_passes must be >= 2 (got {n_passes})")

    _enable_dropout(model)

    all_mu:  List[np.ndarray] = []
    all_sig: List[np.ndarray] = []

    for _ in range(n_passes):
        mu_i, sig_i = _forward_pass(model, x, batch_size=batch_size)
        all_mu.append(mu_i)
        if sig_i is not None:
            all_sig.append(sig_i)

    mu_stack = np.stack(all_mu, axis=0)           # (n_passes, N)
    mean_pred = mu_stack.mean(axis=0)             # (N,)
    epistemic_std = mu_stack.std(axis=0)          # (N,)

    if all_sig:
        aleatoric_mean: Optional[np.ndarray] = np.stack(all_sig, axis=0).mean(axis=0)
    else:
        aleatoric_mean = None

    # Restore eval mode
    model.eval()

    return mean_pred, epistemic_std, aleatoric_mean


# ─────────────────────────────────────────────────────────────────────────────
# Deep ensemble prediction
# ─────────────────────────────────────────────────────────────────────────────

def deep_ensemble_predict(
    models: List[nn.Module],
    x: torch.Tensor,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Deep ensemble prediction: mean + epistemic uncertainty across members.

    Combines predictions from multiple independently-trained models (ensemble
    members).  The std across member means estimates epistemic uncertainty.
    Aleatoric uncertainty is the mean σ_a across all members (if available).

    Parameters
    ----------
    models  : list of trained nn.Module instances (≥ 2 recommended; ≥ 5 is ideal).
              Each should be independently trained (different seed / data order).
    x       : (N, ...) input tensor
    batch_size : inference batch size

    Returns
    -------
    mean_pred      : (N,) ensemble mean depth (m)
    epistemic_std  : (N,) std of member means — inter-member disagreement (m)
    aleatoric_mean : (N,) mean aleatoric σ across members (or None)

    Notes
    -----
    For M=1 ensemble, epistemic_std=0 everywhere.  Use M ≥ 5 for reliable
    uncertainty decomposition (Lakshminarayanan et al. 2017).
    """
    if len(models) == 0:
        raise ValueError("models list is empty")

    member_mu:  List[np.ndarray] = []
    member_sig: List[np.ndarray] = []

    for m in models:
        m.eval()
        mu_i, sig_i = _forward_pass(m, x, batch_size=batch_size)
        member_mu.append(mu_i)
        if sig_i is not None:
            member_sig.append(sig_i)

    mu_stack = np.stack(member_mu, axis=0)       # (M, N)
    mean_pred = mu_stack.mean(axis=0)
    epistemic_std = mu_stack.std(axis=0)

    if member_sig:
        aleatoric_mean: Optional[np.ndarray] = np.stack(member_sig, axis=0).mean(axis=0)
    else:
        aleatoric_mean = None

    return mean_pred, epistemic_std, aleatoric_mean


# ─────────────────────────────────────────────────────────────────────────────
# Calibration curve
# ─────────────────────────────────────────────────────────────────────────────

def calibration_curve(
    pred: np.ndarray,
    sigma: np.ndarray,
    truth: np.ndarray,
    n_levels: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reliability / calibration curve: nominal vs empirical coverage.

    For each nominal confidence level p ∈ {0.1, 0.2, ..., 1.0}, compute the
    empirical fraction of truth values that fall within the symmetric interval
    [pred − z_p * sigma, pred + z_p * sigma], where z_p = scipy.stats.norm.ppf
    (or approximated here without scipy).

    A well-calibrated model yields a roughly diagonal curve (empirical ≈ nominal).
    Under-confident model: empirical > nominal (sigma too large).
    Over-confident model:  empirical < nominal (sigma too small).

    Parameters
    ----------
    pred    : (N,) predicted depths (m)
    sigma   : (N,) predicted standard deviation (m); must be > 0
    truth   : (N,) true depths (m)
    n_levels: number of nominal levels to evaluate (default 10 → 10%-100%)

    Returns
    -------
    nominal_levels  : (n_levels,) array in (0, 1]
    empirical_cov   : (n_levels,) empirical coverage fraction at each level

    Example
    -------
    >>> nom, emp = calibration_curve(pred, sigma, truth)
    >>> # Perfect calibration: emp ≈ nom (diagonal)
    >>> # Under-confident: emp > nom (sigma too large → easy to cover truth)
    """
    pred  = np.asarray(pred,  dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    if np.any(sigma <= 0):
        n_bad = int(np.sum(sigma <= 0))
        sigma = np.where(sigma <= 0, 1e-6, sigma)
        import warnings
        warnings.warn(
            f"calibration_curve: {n_bad} sigma values <= 0 clipped to 1e-6.",
            RuntimeWarning, stacklevel=2,
        )

    ok = np.isfinite(pred) & np.isfinite(sigma) & np.isfinite(truth)
    pred, sigma, truth = pred[ok], sigma[ok], truth[ok]

    if len(pred) == 0:
        return np.linspace(0.1, 1.0, n_levels), np.full(n_levels, float("nan"))

    abs_err = np.abs(pred - truth)

    # Nominal confidence levels
    nominal_levels = np.linspace(1.0 / n_levels, 1.0, n_levels)

    # Normal quantile (probit) approximation without scipy
    # Using the Beasley-Springer-Moro rational approximation for Phi^{-1}
    def _ppf(p: np.ndarray) -> np.ndarray:
        """Probit (normal quantile) function, vectorised."""
        # Clamp to valid range for the approximation
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        # Abramowitz & Stegun (26.2.17) rational approximation
        # Accurate to |error| < 4.5e-4
        c = np.array([2.515517, 0.802853, 0.010328])
        d = np.array([1.432788, 0.189269, 0.001308])
        sign = np.where(p >= 0.5, 1.0, -1.0)
        pp   = np.where(p >= 0.5, p, 1.0 - p)
        t = np.sqrt(-2.0 * np.log(1.0 - pp))
        num   = c[0] + c[1] * t + c[2] * t ** 2
        denom = 1.0 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3
        return sign * (t - num / denom)

    # Two-sided interval: |err| <= z_{(1+p)/2} * sigma
    # For symmetric Gaussian: nominal coverage p → z = ppf((1+p)/2)
    half_p = 0.5 + nominal_levels / 2.0         # (1+p)/2
    z_vals  = _ppf(half_p)                        # z-score for each level

    empirical_cov = np.array([
        float(np.mean(abs_err <= z * sigma))
        for z in z_vals
    ])

    return nominal_levels, empirical_cov


# ─────────────────────────────────────────────────────────────────────────────
# Total uncertainty composition
# ─────────────────────────────────────────────────────────────────────────────

def total_uncertainty(
    aleatoric_sigma: np.ndarray,
    epistemic_sigma: np.ndarray,
) -> np.ndarray:
    """Combine aleatoric and epistemic σ in quadrature.

    For independent Gaussian noise sources, the total standard deviation is:
        σ_total = √(σ_aleatoric² + σ_epistemic²)

    This is valid under the assumption that aleatoric and epistemic errors are
    uncorrelated — a standard assumption in Bayesian deep learning
    (Kendall & Gal 2017).

    Parameters
    ----------
    aleatoric_sigma : (N,) σ_a from the heteroscedastic head
    epistemic_sigma : (N,) σ_e from MC-Dropout or ensemble std

    Returns
    -------
    sigma_total : (N,) total predictive σ (m)
    """
    sa = np.asarray(aleatoric_sigma, dtype=np.float64)
    se = np.asarray(epistemic_sigma, dtype=np.float64)
    return np.sqrt(sa ** 2 + se ** 2).astype(np.float32)
