"""
T3 test battery — Validation & Calibration Suite
=================================================
Tests: T3a (imports) | T3b (HybridStructuralLoss) | T3c (stratified_report)
       T3d (error_distribution_plots) | T3e (uncertainty + calibration)

Run:
    /home/wassi/Bathymetry_VMarch/.venv/bin/python3 \
        /home/wassi/Bathymetry_VMarch/backend/sdb_optim/tests/test_t3.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

# Fixed seed for determinism
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal test harness
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []   # (test_id, passed, details)


def _run(test_id: str, fn):
    try:
        details = fn()
        _results.append((test_id, True, details or "OK"))
        print(f"  PASS  {test_id}: {details or 'OK'}")
    except Exception as exc:
        tb = traceback.format_exc()
        _results.append((test_id, False, str(exc)))
        print(f"  FAIL  {test_id}: {exc}")
        print("        " + tb.replace("\n", "\n        "))


# ─────────────────────────────────────────────────────────────────────────────
# T3a — Import-clean
# ─────────────────────────────────────────────────────────────────────────────

def _t3a():
    from backend.sdb_optim.hybrid_losses import (
        HybridStructuralLoss, ssim_loss, gradient_loss,
    )
    from backend.sdb_optim.stratified_metrics import (
        stratified_report, error_distribution_plots, DEPTH_BINS_DEFAULT,
    )
    from backend.sdb_optim.uncertainty import (
        mc_dropout_predict, deep_ensemble_predict,
        calibration_curve, total_uncertainty,
    )
    return "all T3 symbols imported cleanly"


# ─────────────────────────────────────────────────────────────────────────────
# T3b — HybridStructuralLoss
# ─────────────────────────────────────────────────────────────────────────────

def _t3b_finite_grad():
    """Hybrid loss is finite and has non-zero gradient on synthetic rasters.

    We test via a simple linear model whose parameters ARE leaves, so
    .grad is populated after backward().
    """
    from backend.sdb_optim.hybrid_losses import HybridStructuralLoss, ssim_loss, gradient_loss

    torch.manual_seed(SEED)
    B, H, W = 2, 32, 32

    # Learnable weight (the actual leaf we check grad on)
    w = nn.Parameter(torch.ones(1))
    true_r = (torch.rand(B, 1, H, W) * 10.0).detach()
    # pred = w * some_raster → leaf is w
    raw = torch.rand(B, 1, H, W).detach()
    pred_r = w * raw   # non-leaf; grad flows to w

    # SSIM and gradient losses alone (no base needed for this sub-test)
    l_ssim = ssim_loss(pred_r, true_r)
    l_grad = gradient_loss(pred_r, true_r)
    loss = l_ssim + l_grad

    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    loss.backward()
    assert w.grad is not None, "Gradient on parameter w is None"
    grad_val = abs(float(w.grad.item()))
    assert grad_val > 0, f"Gradient on w is zero"

    # Also verify HybridStructuralLoss wraps correctly
    base = nn.MSELoss()
    criterion = HybridStructuralLoss(base, lambda_ssim=0.10, lambda_grad=0.05)
    w2 = nn.Parameter(torch.ones(1))
    pred2 = w2 * raw
    loss2 = criterion(pred2.view(-1), true_r.view(-1),
                      pred_raster=pred2, true_raster=true_r)
    assert torch.isfinite(loss2), f"HybridStructuralLoss not finite: {loss2.item()}"
    loss2.backward()
    assert w2.grad is not None, "Gradient on w2 is None"

    return (f"ssim_loss={l_ssim.item():.4f}, grad_loss={l_grad.item():.4f}, "
            f"hybrid_loss={loss2.item():.4f}, dL/dw={grad_val:.4f}")


def _t3b_ssim_converges():
    """SSIM term → 0 as pred → truth."""
    from backend.sdb_optim.hybrid_losses import ssim_loss

    torch.manual_seed(SEED)
    B, H, W = 2, 32, 32
    true_r = torch.rand(B, 1, H, W) * 10.0

    # pred == truth → SSIM loss should be near 0
    loss_identical = ssim_loss(true_r, true_r).item()
    # pred totally random
    pred_random = torch.rand(B, 1, H, W) * 10.0
    loss_random = ssim_loss(pred_random, true_r).item()

    assert loss_identical < 0.02, f"SSIM loss on identical rasters = {loss_identical:.4f} (expected < 0.02)"
    assert loss_random > loss_identical, f"Random pred SSIM loss {loss_random:.4f} should exceed identical {loss_identical:.4f}"
    return f"SSIM(identical)={loss_identical:.4f}, SSIM(random)={loss_random:.4f}"


def _t3b_grad_penalises_smooth():
    """Gradient loss penalises a smoothed (blurred) prediction vs the sharp truth."""
    from backend.sdb_optim.hybrid_losses import gradient_loss

    torch.manual_seed(SEED)
    B, H, W = 2, 64, 64
    true_r = torch.rand(B, 1, H, W) * 10.0

    # "Smooth" prediction = low-pass of truth (loses edge info)
    kernel = torch.ones(1, 1, 7, 7) / 49.0
    smooth_pred = nn.functional.conv2d(true_r, kernel, padding=3)

    loss_smooth = gradient_loss(smooth_pred, true_r).item()
    loss_identical = gradient_loss(true_r, true_r).item()

    assert loss_smooth > loss_identical, (
        f"Gradient loss for smooth pred ({loss_smooth:.4f}) should exceed "
        f"identical ({loss_identical:.4f})"
    )
    assert loss_identical < 1e-3, f"Gradient loss on identical rasters = {loss_identical:.6f}"
    return f"grad_loss(smooth)={loss_smooth:.4f}, grad_loss(identical)={loss_identical:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# T3c — stratified_report correctness
# ─────────────────────────────────────────────────────────────────────────────

def _t3c_four_bands():
    """stratified_report returns correct n, metrics for all 4 depth bands.

    stratified_report(pred, truth, ...) — pred is first argument.
    """
    from backend.sdb_optim.stratified_metrics import stratified_report, DEPTH_BINS_DEFAULT
    import warnings

    np.random.seed(SEED)
    N_PER = 200   # points per band
    truth = np.concatenate([
        np.random.uniform(0,   2,  N_PER),
        np.random.uniform(2,   5,  N_PER),
        np.random.uniform(5,  10,  N_PER),
        np.random.uniform(10, 20,  N_PER),
    ])
    pred = truth + np.random.normal(0, 0.5, 4 * N_PER)   # unbiased noise

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Correct call order: pred first, truth second
        strata, df, md_str = stratified_report(pred, truth, print_table=False)

    assert len(strata) == 5, f"Expected 5 strata (4 bands + overall), got {len(strata)}"
    for sr in strata[:-1]:   # exclude overall
        assert sr.n == N_PER, f"Stratum {sr.label} n={sr.n}, expected {N_PER}"
    for sr in strata:
        if sr.n > 0:
            assert math.isfinite(sr.rmse), f"RMSE not finite for {sr.label}"
            assert math.isfinite(sr.mae),  f"MAE not finite for {sr.label}"
            assert math.isfinite(sr.r2),   f"R2 not finite for {sr.label}"
            assert math.isfinite(sr.mbe),  f"MBE not finite for {sr.label}"

    ns = [sr.n for sr in strata[:-1]]
    return (f"n per band={ns}, "
            f"overall RMSE={strata[-1].rmse:.3f} m, R²={strata[-1].r2:.3f}")


def _t3c_bias_sign():
    """MBE sign is correct for injected +0.5 m positive offset.

    MBE = mean(pred − truth).  pred = truth + 0.5 → MBE = +0.5.
    Call order: stratified_report(pred, truth).
    """
    from backend.sdb_optim.stratified_metrics import stratified_report
    import warnings

    np.random.seed(SEED)
    N = 500
    truth = np.random.uniform(0, 15, N)
    OFFSET = 0.5
    pred = truth + OFFSET   # constant positive bias: pred > truth

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # pred first, truth second
        strata, df, md_str = stratified_report(pred, truth, print_table=False)

    overall = strata[-1]
    assert abs(overall.mbe - OFFSET) < 0.01, (
        f"MBE={overall.mbe:.4f} expected ≈ +{OFFSET} m"
    )
    assert overall.mbe > 0, f"MBE should be positive for +{OFFSET} m injection, got {overall.mbe:.4f}"
    for sr in strata[:-1]:
        if sr.n > 0:
            assert sr.mbe > 0, f"Per-band MBE sign wrong for {sr.label}: {sr.mbe:.4f}"
    return f"MBE overall={overall.mbe:+.4f} m (injected +{OFFSET} m) — sign correct"


# ─────────────────────────────────────────────────────────────────────────────
# T3d — error_distribution_plots writes PNGs
# ─────────────────────────────────────────────────────────────────────────────

def _t3d_plots():
    """error_distribution_plots writes ≥ 1 readable PNG with non-zero size."""
    from backend.sdb_optim.stratified_metrics import error_distribution_plots

    np.random.seed(SEED)
    N = 300
    truth = np.random.uniform(0, 15, N)
    pred  = truth + np.random.normal(0, 0.8, N)

    # Use a persistent temp dir so we can stat after the context manager closes
    tmpdir = tempfile.mkdtemp()
    try:
        out = Path(tmpdir)
        paths = error_distribution_plots(pred, truth, out_path=out, dpi=72)

        assert len(paths) >= 1, "No paths returned"
        sizes = []
        for p in paths:
            assert p.exists(), f"PNG not written: {p}"
            size = p.stat().st_size
            assert size > 1000, f"PNG suspiciously small ({size} bytes): {p}"
            sizes.append(size)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return (f"wrote {len(paths)} PNGs "
            f"(sizes: {', '.join(str(s) for s in sizes)} bytes)")


# ─────────────────────────────────────────────────────────────────────────────
# T3e — Uncertainty: MC-Dropout, ensemble, calibration_curve
# ─────────────────────────────────────────────────────────────────────────────

class _TinyModel(nn.Module):
    """Minimal model with heteroscedastic head and dropout (for testing)."""
    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.drop = nn.Dropout(p=0.3)
        self.mu_head  = nn.Linear(16, 1)
        self.lv_head  = nn.Linear(16, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.drop(h)
        mu = torch.sigmoid(self.mu_head(h)).squeeze(1) * 10.0
        lv = torch.clamp(self.lv_head(h).squeeze(1), -4.0, 4.0)
        return mu, lv


def _t3e_mc_dropout():
    """MC-Dropout produces std > 0 across multiple passes (fixed seed)."""
    from backend.sdb_optim.uncertainty import mc_dropout_predict

    torch.manual_seed(SEED)
    model = _TinyModel()

    N = 100
    x = torch.rand(N, 4)

    mean1, epist1, aleat1 = mc_dropout_predict(model, x, n_passes=15)
    # Run again with same seed — should be identical
    torch.manual_seed(SEED + 1)
    mean2, epist2, aleat2 = mc_dropout_predict(model, x, n_passes=15)

    assert mean1.shape == (N,), f"mean shape wrong: {mean1.shape}"
    assert epist1.shape == (N,), f"epistemic std shape wrong: {epist1.shape}"
    assert float(epist1.mean()) > 0.0, f"epistemic std is zero (MC-Dropout not active)"
    assert aleat1 is not None, "aleatoric sigma should be returned by TinyModel"
    assert float(aleat1.mean()) > 0.0, f"aleatoric std is zero"

    # Determinism check: same model + same seed → same mean (within float tolerance)
    torch.manual_seed(SEED)
    mean_det1, _, _ = mc_dropout_predict(model, x, n_passes=5)
    torch.manual_seed(SEED)
    mean_det2, _, _ = mc_dropout_predict(model, x, n_passes=5)
    assert np.allclose(mean_det1, mean_det2, atol=1e-6), "MC-Dropout not deterministic on fixed seed"

    return (f"mean_pred range=[{mean1.min():.2f},{mean1.max():.2f}] m, "
            f"epistemic_std mean={epist1.mean():.4f}, aleatoric_std mean={aleat1.mean():.4f}")


def _t3e_ensemble():
    """Deep ensemble gives std > 0 across 3 differently-seeded models."""
    from backend.sdb_optim.uncertainty import deep_ensemble_predict

    models = []
    for seed_i in [1, 2, 3]:
        torch.manual_seed(seed_i)
        m = _TinyModel()
        models.append(m)

    N = 100
    x = torch.rand(N, 4)

    mean_e, epist_e, aleat_e = deep_ensemble_predict(models, x)

    assert mean_e.shape == (N,), f"mean shape: {mean_e.shape}"
    assert float(epist_e.mean()) > 0.0, "Ensemble epistemic std is zero (all members identical?)"
    assert aleat_e is not None, "aleatoric sigma should be returned"

    return (f"ensemble mean range=[{mean_e.min():.2f},{mean_e.max():.2f}] m, "
            f"epistemic_std mean={epist_e.mean():.4f}")


def _t3e_calibration():
    """calibration_curve returns monotone-ish coverage for well-calibrated sigma.

    Uses n_levels=9 so the final level = 90% (not 100%), keeping the ppf
    finite and the test assertion meaningful.
    """
    from backend.sdb_optim.uncertainty import calibration_curve

    np.random.seed(SEED)
    N = 2000
    truth = np.random.uniform(0, 10, N)
    sigma = np.random.uniform(0.3, 1.0, N)
    # Well-calibrated: pred = truth + Normal(0, sigma)
    pred = truth + np.random.normal(0, 1.0, N) * sigma

    # n_levels=9 → levels = 1/9, 2/9, …, 9/9=1.0; use 10 and check index 8 (90%)
    # Actually use n_levels=19 so level[17]=18/19≈0.947 is close to 0.95
    nom, emp = calibration_curve(pred, sigma, truth, n_levels=19)

    assert nom.shape == (19,), f"nominal_levels shape: {nom.shape}"
    assert emp.shape == (19,), f"empirical_cov shape: {emp.shape}"
    # Monotone-ish: emp[0] (10%) < emp[-2] (90%ish)
    assert float(emp[0]) < float(emp[-2]) + 0.05, (
        "Calibration curve should be roughly non-decreasing "
        f"(emp[0]={emp[0]:.3f}, emp[-2]={emp[-2]:.3f})"
    )
    # Level index 17 = 18/19 ≈ 0.947 — empirical coverage should be ~0.95
    idx95 = 17
    assert abs(float(emp[idx95]) - 0.95) < 0.12, (
        f"~95% empirical coverage = {emp[idx95]:.3f}, expected 0.95 ± 0.12 "
        f"for calibrated sigma (nominal={nom[idx95]:.3f})"
    )
    return (f"coverage at nom={nom[0]:.2f}→{emp[0]:.3f}, "
            f"nom={nom[9]:.2f}→{emp[9]:.3f}, "
            f"nom={nom[idx95]:.2f}→{emp[idx95]:.3f} (monotone-ish)")


def _t3e_total_uncertainty():
    """total_uncertainty composes aleatoric + epistemic in quadrature."""
    from backend.sdb_optim.uncertainty import total_uncertainty

    sa = np.array([0.3, 0.4, 0.5], dtype=np.float64)
    se = np.array([0.4, 0.3, 0.0], dtype=np.float64)
    st = total_uncertainty(sa, se)

    expected = np.sqrt(sa ** 2 + se ** 2)
    assert np.allclose(st, expected, atol=1e-6), f"total_uncertainty mismatch: {st} vs {expected}"
    return (f"σ_total = {st.tolist()!r} (quadrature: {expected.tolist()!r})")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("T3 Test Battery — Validation & Calibration Suite")
    print("=" * 70)

    _run("T3a  imports",               _t3a)
    print()
    _run("T3b1 finite+grad",           _t3b_finite_grad)
    _run("T3b2 SSIM->0 as pred->truth",_t3b_ssim_converges)
    _run("T3b3 grad penalises smooth", _t3b_grad_penalises_smooth)
    print()
    _run("T3c1 4-band n + metrics",    _t3c_four_bands)
    _run("T3c2 MBE sign +0.5 m",      _t3c_bias_sign)
    print()
    _run("T3d  plots PNG written",     _t3d_plots)
    print()
    _run("T3e1 MC-Dropout std>0",      _t3e_mc_dropout)
    _run("T3e2 ensemble std>0",        _t3e_ensemble)
    _run("T3e3 calibration curve",     _t3e_calibration)
    _run("T3e4 total_uncertainty",     _t3e_total_uncertainty)

    print()
    print("=" * 70)
    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    print(f"RESULT: {passed}/{total} tests passed")
    print("=" * 70)

    if passed < total:
        sys.exit(1)
    sys.exit(0)
