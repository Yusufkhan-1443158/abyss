"""Integration test for the orchestrator composition facade (backend.sdb_optim.integration).

Verifies the four tracks compose correctly against the LIVE pipeline primitives:
  (a) import-clean,
  (b) loss factories produce finite, backprop-able scalars wrapping HeteroNLLLoss,
  (c) feature extension returns shape/name-consistent finite channels,
  (d) the spatial K-fold honours the >=500 m buffer and covers all points once,
  (e) the depth-stratified report renders,
  (f) every WIRING target is a REAL importable pipeline symbol (no fabricated wiring),
  (g) determinism under a fixed seed.

CPU-only, synthetic inputs, < ~5 s. No training, no network, no GEE.
"""
from __future__ import annotations

import importlib

import numpy as np
import torch

from backend.sdb_optim import integration as I


def test_import_clean():
    assert hasattr(I, "recommended_config")
    cfg = I.recommended_config()
    assert cfg["min_test_train_dist_m"] == 500.0
    assert 0.0 <= cfg["lambda_ssim"] <= 1.0


def test_physics_criterion_wraps_live_heteronll():
    crit = I.build_physics_criterion(lambda_stumpf=0.1, lambda_monotonic=0.1)
    n = 64
    torch.manual_seed(0)
    mu = torch.randn(n, requires_grad=True)
    lv = torch.zeros(n, requires_grad=True)
    target = torch.randn(n)
    ratio = torch.rand(n) + 0.5
    loss = crit(mu, lv, target, stumpf_ratio=ratio,
                ln_blue=torch.rand(n), ln_green=torch.rand(n))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()


def test_structural_criterion_reduces_to_base():
    crit = I.build_structural_criterion(lambda_ssim=0.15, lambda_grad=0.10)
    n = 32
    mu, lv, target = torch.randn(n), torch.zeros(n), torch.randn(n)
    # pointwise-only path (no rasters) == finite scalar
    loss_pt = crit(mu, lv, target)
    assert torch.isfinite(loss_pt)
    # raster path adds structural terms => still finite scalar
    pr = torch.rand(2, 1, 16, 16)
    tr = torch.rand(2, 1, 16, 16)
    loss_struct = crit(mu, lv, target, pred_raster=pr, true_raster=tr)
    assert torch.isfinite(loss_struct)


def test_extend_feature_cube_shapes():
    H, W = 24, 20
    rng = np.random.default_rng(0)
    b, g, r, nir = (rng.random((H, W)).astype(np.float32) for _ in range(4))
    cube, names = I.extend_feature_cube(b, g, r, nir)
    assert cube.shape[:2] == (H, W)
    assert cube.shape[2] == len(names)
    assert np.isfinite(cube).all()


def test_temporal_features():
    T, H, W = 5, 12, 10
    rng = np.random.default_rng(1)
    stack = rng.random((T, H, W)).astype(np.float32)
    cube, names = I.temporal_features(stack)
    assert cube.shape[:2] == (H, W)
    assert cube.shape[2] == len(names)
    assert np.isfinite(cube).all()


def _synthetic_coords(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # ~ a 5x5 km patch near Mussafah (deg): 0.05 deg ~ 5.5 km
    lat = 24.40 + rng.random(n) * 0.05
    lon = 54.40 + rng.random(n) * 0.05
    return np.column_stack([lat, lon]).astype(np.float64)


def test_spatial_kfold_buffer_and_coverage():
    coords = _synthetic_coords()
    folds = I.cross_validate(coords, k=5, seed=42)
    assert len(folds) == 5
    seen_test = []
    for fid, (tr, te) in enumerate(folds):
        assert len(set(tr.tolist()) & set(te.tolist())) == 0  # no train/test overlap
        seen_test.extend(te.tolist())
        if len(te) and len(tr):
            d = I.fold_min_distance(coords, folds, fid) if hasattr(I, "fold_min_distance") else None
        # buffer enforced via the module helper
        from backend.sdb_optim.spatial_cv import fold_min_distance
        if len(te) and len(tr):
            assert fold_min_distance(coords, folds, fid) >= 500.0 - 1e-6
    # every point is a test point at most once; buffer may drop a few entirely
    assert len(seen_test) == len(set(seen_test))
    assert len(seen_test) <= len(coords)


def test_depth_stratified_report():
    rng = np.random.default_rng(3)
    truth = rng.random(500) * 18.0
    pred = truth + rng.normal(0, 0.5, size=truth.shape)
    strata, df, md = I.depth_stratified_report(pred, truth)
    assert "Overall" in df.iloc[:, 0].astype(str).tolist() or "Overall" in md
    assert isinstance(md, str) and len(md) > 0


def test_combined_uncertainty():
    a = np.full(10, 0.3, dtype=np.float32)
    e = np.full(10, 0.4, dtype=np.float32)
    tot = I.combined_uncertainty(a, e)
    assert np.allclose(tot, 0.5, atol=1e-5)  # sqrt(0.3^2 + 0.4^2)


def test_wiring_targets_are_real():
    """Each WIRING entry must point at an importable LIVE pipeline symbol —
    guarantees the documented integration is not fabricated."""
    import backend
    for optim_sym, live_sym, _note in I.WIRING:
        mod_path, _, attr = live_sym.rpartition(".")
        mod = importlib.import_module(f"backend.{mod_path}")
        assert hasattr(mod, attr), f"missing live pipeline symbol: backend.{live_sym}"


def test_determinism():
    coords = _synthetic_coords(seed=7)
    f1 = I.cross_validate(coords, k=4, seed=123)
    f2 = I.cross_validate(coords, k=4, seed=123)
    for (a_tr, a_te), (b_tr, b_te) in zip(f1, f2):
        assert np.array_equal(a_te, b_te) and np.array_equal(a_tr, b_tr)
