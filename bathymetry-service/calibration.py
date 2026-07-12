"""Local calibration of a depth grid to user-supplied soundings.

Vendored minimal from Bathymetry_Production/backend/depth_calibration.py
(spatial shift search + bias + IDW residual field), adapted for honesty:
the points are split 80/20 BEFORE any fitting, every step is fitted on the
80% train split only, and the holdout RMSE/MAE/bias before vs after is
reported on the untouched 20%. The plain mean-bias step is replaced by a
robust Huber (IRLS) linear fit of observed vs predicted depth.

    calibrate_local(depth, rows, cols, obs_depths, ...) -> (depth_cal, info)

rows/cols are the observed points' pixel indices on the depth grid (the
caller handles CRS/georeferencing). The original grid is never modified.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

L = logging.getLogger("bathymetry.calibration")

MAX_DEPTH_M = 25.0
MIN_POINTS = 10


class CalibrationError(ValueError):
    """Bad calibration input — maps to HTTP 400."""


def _stats(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(pred) & np.isfinite(obs) & (obs > 0)
    pv, yv = pred[ok], obs[ok]
    if len(yv) == 0:
        return {"n": 0, "rmse_m": None, "mae_m": None, "bias_m": None, "r2": None}
    r = pv - yv
    ss_res = float(np.sum(r ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    return {
        "n": int(len(yv)),
        "rmse_m": round(float(np.sqrt(np.mean(r ** 2))), 3),
        "mae_m": round(float(np.mean(np.abs(r))), 3),
        "bias_m": round(float(np.mean(r)), 3),
        "r2": round(float(1 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else 0.0, 4),
    }


def _sample_shift(depth, r, c, dr, dc):
    H, W = depth.shape
    return depth[np.clip(r + dr, 0, H - 1), np.clip(c + dc, 0, W - 1)]


def _find_best_shift(depth, obs, r, c, max_shift_px=5) -> Tuple[int, int]:
    best, best_rmse = (0, 0), float("inf")
    for dr in range(-max_shift_px, max_shift_px + 1):
        for dc in range(-max_shift_px, max_shift_px + 1):
            s = _stats(_sample_shift(depth, r, c, dr, dc), obs)
            if s["n"] < 5 or s["rmse_m"] is None:
                continue
            if s["rmse_m"] < best_rmse:
                best_rmse, best = s["rmse_m"], (dr, dc)
    return best


def _apply_shift(depth: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Shift by (dr, dc) pixels, NaN-padding the edges (never fabricates data)."""
    H, W = depth.shape
    out = np.full_like(depth, np.nan)
    src_r0, src_r1 = max(0, -dr), H - max(0, dr)
    src_c0, src_c1 = max(0, -dc), W - max(0, dc)
    out[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)] = \
        depth[src_r0:src_r1, src_c0:src_c1]
    return out


def _huber_fit(pred: np.ndarray, obs: np.ndarray, k: float = 1.345,
               iters: int = 50) -> Tuple[float, float]:
    """Robust (Huber IRLS) linear fit obs ≈ a*pred + b. Slope clamped to a
    physically plausible range so a degenerate fit can't invert the grid."""
    x = np.asarray(pred, np.float64)
    y = np.asarray(obs, np.float64)
    a, b = 1.0, 0.0
    for _ in range(iters):
        r = y - (a * x + b)
        s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
        w = np.clip(k * s / np.maximum(np.abs(r), 1e-9), None, 1.0)
        sw = w.sum()
        mx, my = (w * x).sum() / sw, (w * y).sum() / sw
        vx = (w * (x - mx) ** 2).sum()
        if vx < 1e-9:
            break
        a_new = float((w * (x - mx) * (y - my)).sum() / vx)
        b_new = float(my - a_new * mx)
        if abs(a_new - a) < 1e-6 and abs(b_new - b) < 1e-6:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    a = float(np.clip(a, 0.3, 3.0))
    return a, b


def _idw_residual_field(shape, r, c, residual, grid_h=80, grid_w=80,
                        power=2.0, clamp_m=2.0):
    """IDW-interpolated additive residual field on a coarse grid, upsampled."""
    H, W = shape
    GH, GW = min(grid_h, H), min(grid_w, W)
    if len(residual) == 0:
        return np.zeros((H, W), dtype=np.float32)
    rr_c = np.asarray(r) * GH / H
    cc_c = np.asarray(c) * GW / W
    Y, X = np.meshgrid(np.arange(GH), np.arange(GW), indexing="ij")
    dy = Y[..., None] - rr_c[None, None, :]
    dx = X[..., None] - cc_c[None, None, :]
    d2 = dy * dy + dx * dx + 1e-6
    w = 1.0 / (d2 ** (power / 2))
    coarse = (w * np.asarray(residual)[None, None, :]).sum(-1) / w.sum(-1)
    coarse = np.clip(coarse, -clamp_m, clamp_m).astype(np.float32)
    try:
        from scipy.ndimage import zoom, gaussian_filter
        field = zoom(coarse, (H / GH, W / GW), order=1, mode="nearest")
        field = gaussian_filter(field, sigma=2.0)
    except Exception:
        field = np.kron(coarse, np.ones((H // GH + 1, W // GW + 1)))[:H, :W]
    return field.astype(np.float32)


def calibrate_local(depth: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                    obs_depths: np.ndarray, holdout_frac: float = 0.2,
                    max_shift_px: int = 5, clamp_local_m: float = 2.0,
                    max_depth: float = MAX_DEPTH_M, seed: int = 42
                    ) -> Tuple[np.ndarray, Dict]:
    """Fit shift + Huber linear + local IDW residual field on 80% of the
    points; report holdout metrics before/after on the untouched 20%."""
    depth = np.asarray(depth, np.float32).copy()
    H, W = depth.shape
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    obs = np.abs(np.asarray(obs_depths, np.float64))
    keep = (np.isfinite(obs) & (obs > 0.2) & (obs <= max_depth) &
            (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W))
    keep &= np.isfinite(depth[np.clip(rows, 0, H - 1), np.clip(cols, 0, W - 1)])
    rows, cols, obs = rows[keep], cols[keep], obs[keep]
    n = len(obs)
    if n < MIN_POINTS:
        raise CalibrationError(
            f"only {n} usable soundings fall on valid depth pixels "
            f"(need >= {MIN_POINTS})")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_hold = max(3, int(round(holdout_frac * n)))
    hold, train = order[:n_hold], order[n_hold:]
    rt, ct, yt = rows[train], cols[train], obs[train]
    rh, ch, yh = rows[hold], cols[hold], obs[hold]

    hold_before = _stats(depth[rh, ch], yh)
    train_before = _stats(depth[rt, ct], yt)

    dr, dc = _find_best_shift(depth, yt, rt, ct, max_shift_px=max_shift_px)
    if (dr, dc) != (0, 0):
        depth = _apply_shift(depth, dr, dc)

    pv = depth[rt, ct]
    ok = np.isfinite(pv)
    a, b = _huber_fit(pv[ok], yt[ok])
    depth = np.where(np.isfinite(depth),
                     np.clip(a * depth + b, 0.0, max_depth), depth)

    pv = depth[rt, ct]
    ok = np.isfinite(pv)
    field_stats = None
    if ok.sum() >= MIN_POINTS:
        resid = (yt[ok] - pv[ok]).astype(np.float32)
        field = _idw_residual_field(depth.shape, rt[ok], ct[ok], resid,
                                    clamp_m=clamp_local_m)
        depth = np.where(np.isfinite(depth),
                         np.clip(depth + field, 0.0, max_depth), depth)
        field_stats = {
            "min_m": round(float(np.nanmin(field)), 3),
            "max_m": round(float(np.nanmax(field)), 3),
            "std_m": round(float(np.nanstd(field)), 3),
        }

    hold_after = _stats(depth[rh, ch], yh)
    train_after = _stats(depth[rt, ct], yt)

    info = {
        "provenance": "local_points",
        "n_points": n,
        "n_train": int(len(train)),
        "n_holdout": int(len(hold)),
        "steps": ["shift", "huber_linear", "idw_local"],
        "shift_px": {"dr": int(dr), "dc": int(dc)},
        "linear": {"a": round(a, 4), "b": round(b, 4)},
        "local_field": field_stats,
        "clamp_local_m": clamp_local_m,
        "holdout": {"before": hold_before, "after": hold_after},
        "train": {"before": train_before, "after": train_after},
        "improvement_holdout_rmse_m": (
            None if hold_before["rmse_m"] is None or hold_after["rmse_m"] is None
            else round(hold_before["rmse_m"] - hold_after["rmse_m"], 3)),
        "note": ("Calibrated to user-supplied soundings inside the layer "
                 "extent; holdout metrics are computed on a 20% split never "
                 "used for fitting. The original layer is preserved."),
    }
    L.info("calibrate_local: n=%d holdout RMSE %.3f -> %.3f m", n,
           hold_before["rmse_m"] or -1, hold_after["rmse_m"] or -1)
    return depth, info
