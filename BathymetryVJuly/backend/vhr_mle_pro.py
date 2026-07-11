"""Very-HR + Maximum-Likelihood PRO pipeline.

Stacks 10 Sentinel-2 scenes spread across the calendar year (each scene is
itself a median composite of a ±15 d window through ``fetch_s2`` /
``fetch_s2_gee``) and fuses them via per-pixel inverse-variance Maximum
Likelihood Estimation. The MLE depth grid is then aligned onto a Mapbox
Very-HR (~2 m/px) mosaic so the final raster is delivered at VHR
resolution but with multi-temporal MLE noise control.

Pipeline stages (and their share of the progress bar):

    Stage                                          weight
    ------------------------------------------------------
    01  Fetch Mapbox VHR mosaic                     8 %
    02  Generate scene calendar (10 dates)          1 %
    03..12  Fetch + compute scene i (10 × 7 %)     70 %
    13  Inverse-variance MLE fusion                 6 %
    14  Resample MLE depth → VHR grid               5 %
    15  Calibration / metrics                       4 %
    16  Render PNGs + save                          6 %

A progress callback ``progress_cb(pct, stage, idx, total)`` is invoked
after every stage so the frontend's job poll returns a real percentage
instead of the legacy time-based estimate.
"""
from __future__ import annotations

import io
import logging
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

L = logging.getLogger("vhr_mle_pro")
if not L.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    L.addHandler(h)
    L.setLevel(logging.INFO)

MAX_DEPTH_M = 25.0

# Stage weights (must sum to 100). Tuned so the bar moves visibly across
# every scene rather than sitting at 8 % through the long S2 phase.
_STAGE_WEIGHTS = {
    "vhr": 8,
    "calendar": 1,
    "scenes": 70,   # split evenly across N_SCENES
    "mle": 6,
    "resample": 5,
    "calibrate": 4,
    "render": 6,
}


def _emit(progress_cb: Optional[Callable], pct: float, stage: str,
          idx: Optional[int] = None, total: Optional[int] = None) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(float(pct), stage, idx, total)
    except Exception:
        # Never let progress reporting crash the pipeline.
        pass


def _is_cancelled(cancel_cb: Optional[Callable]) -> bool:
    if cancel_cb is None:
        return False
    try:
        return bool(cancel_cb())
    except Exception:
        return False


def _scene_windows(year: int, n_scenes: int = 10, pad_days: int = 15
                   ) -> List[Dict[str, str]]:
    """Pick ``n_scenes`` evenly-spaced ±``pad_days`` windows across a calendar
    year. Each window centre lands at the midpoint of an equal-width slot so
    seasonal coverage is balanced (no clumping in summer)."""
    centres = [
        datetime(int(year), 1, 1) + timedelta(days=int((i + 0.5) * 365 / n_scenes))
        for i in range(n_scenes)
    ]
    out = []
    for c in centres:
        out.append({
            "date": c.strftime("%Y-%m-%d"),
            "start": (c - timedelta(days=pad_days)).strftime("%Y-%m-%d"),
            "end":   (c + timedelta(days=pad_days)).strftime("%Y-%m-%d"),
        })
    return out


def _inv_variance_mle(grids: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Per-pixel inverse-variance MLE across N depth grids.

        z_hat[r,c] = Σ_i z_i / σ_i²  /  Σ_i 1 / σ_i²
        σ_hat[r,c] = 1 / √( Σ_i 1 / σ_i² )

    σ_i is the scene's held-out RMSE clamped to [0.5, 5] m so a single
    near-perfect-fit scene cannot dominate.
    """
    if not grids:
        raise RuntimeError("MLE fusion received zero scenes")
    H = min(g["depth"].shape[0] for g in grids)
    W = min(g["depth"].shape[1] for g in grids)
    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)
    n_valid = np.zeros((H, W), dtype=np.int16)
    union_water = np.zeros((H, W), dtype=bool)
    for g in grids:
        d = g["depth"][:H, :W]
        w = g["water"][:H, :W]
        sigma = float(np.clip(g.get("sigma", 1.5), 0.5, 5.0))
        inv = 1.0 / (sigma * sigma)
        ok = w & np.isfinite(d) & (d > 0)
        num[ok] += d[ok] * inv
        den[ok] += inv
        n_valid[ok] += 1
        union_water |= w
    z = np.full((H, W), np.nan, dtype=np.float32)
    s = np.full((H, W), np.nan, dtype=np.float32)
    finite = den > 0
    z[finite] = (num[finite] / den[finite]).astype(np.float32)
    s[finite] = (1.0 / np.sqrt(den[finite])).astype(np.float32)
    z = np.clip(z, 0.0, MAX_DEPTH_M)
    return {
        "depth": z, "sigma": s, "n_valid": n_valid,
        "water": union_water, "H": H, "W": W,
    }


def _resample_to_vhr(arr: np.ndarray, H_vhr: int, W_vhr: int,
                     nearest: bool = False) -> np.ndarray:
    """Resample an arbitrary 2-D float array onto the VHR pixel grid via PIL.

    NaNs are preserved by routing them through a zero-fill with a companion
    valid mask, then re-inserted as NaN after resampling.
    """
    from PIL import Image
    a = np.asarray(arr, dtype=np.float32)
    if a.shape == (H_vhr, W_vhr):
        return a.copy()
    nan_mask = ~np.isfinite(a)
    if nan_mask.any():
        a_safe = np.where(nan_mask, 0.0, a)
    else:
        a_safe = a
    resample = Image.NEAREST if nearest else Image.BILINEAR
    img = Image.fromarray(a_safe).resize((W_vhr, H_vhr), resample)
    out = np.array(img, dtype=np.float32)
    if nan_mask.any():
        m = Image.fromarray(nan_mask.astype(np.uint8) * 255).resize(
            (W_vhr, H_vhr), Image.NEAREST)
        out[np.array(m) > 0] = np.nan
    return out


def _depth_png(depth_vhr: np.ndarray, water_vhr: np.ndarray) -> bytes:
    """Render the VHR depth grid to an RGBA PNG (blue→cyan ramp, alpha=water)."""
    from PIL import Image
    H, W = depth_vhr.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    d = np.clip(np.where(np.isfinite(depth_vhr), depth_vhr, 0), 0, MAX_DEPTH_M)
    t = d / MAX_DEPTH_M
    # Shallow water = cyan, deep water = navy
    rgba[..., 0] = (255 * (1 - t) * 0.30).astype(np.uint8)
    rgba[..., 1] = (255 * (1 - t * 0.6)).astype(np.uint8)
    rgba[..., 2] = (255 * (0.45 + 0.55 * (1 - t))).astype(np.uint8)
    rgba[..., 3] = np.where(water_vhr & np.isfinite(depth_vhr), 220, 0).astype(np.uint8)
    bio = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def _sigma_png(sigma_vhr: np.ndarray, water_vhr: np.ndarray) -> bytes:
    """Yellow→red colour-ramp PNG on σ ∈ [0, 3] m for the uncertainty layer."""
    from PIL import Image
    H, W = sigma_vhr.shape
    u = np.where(np.isfinite(sigma_vhr), np.clip(sigma_vhr, 0, 3), 0) / 3.0
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., 0] = (255 * np.clip(u * 2.0, 0, 1)).astype(np.uint8)
    rgba[..., 1] = (255 * np.clip(2.0 - u * 2.0, 0, 1)).astype(np.uint8)
    rgba[..., 2] = 0
    rgba[..., 3] = np.where(water_vhr & np.isfinite(sigma_vhr), 200, 0).astype(np.uint8)
    bio = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def run_vhr_mle_pro(
    bbox: List[float],
    *,
    year: int = 2024,
    n_scenes: int = 10,
    max_cloud: int = 20,
    target_res_m: float = 2.0,
    max_tiles_per_side: int = 4,
    user_pts: Optional[List[Dict[str, float]]] = None,
    fetch_sliderule: bool = False,
    fetch_vhr_fn: Optional[Callable] = None,
    compute_scene_fn: Optional[Callable] = None,
    progress_cb: Optional[Callable] = None,
    cancel_cb: Optional[Callable] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full Very-HR + MLE PRO pipeline.

    ``fetch_vhr_fn(bbox, target_res_m, max_tiles_per_side)`` and
    ``compute_scene_fn(bbox, start, end, user_pts, max_cloud, fetch_sliderule)``
    are injected to avoid an ``app.py`` ↔ ``vhr_mle_pro`` import cycle.

    The pipeline keeps writing progress so the UI bar advances scene-by-scene.
    """
    t0 = time.time()
    n_scenes = max(2, min(int(n_scenes), 12))
    pct = 0.0
    _emit(progress_cb, pct, "Starting Very-HR + MLE PRO …", 0, n_scenes + 5)

    # ── Stage 1: Mapbox VHR mosaic ─────────────────────────────────────────
    if fetch_vhr_fn is None:
        raise RuntimeError("fetch_vhr_fn must be provided")
    if _is_cancelled(cancel_cb):
        raise RuntimeError("cancelled")
    _emit(progress_cb, pct, "Fetching Mapbox VHR mosaic (~2 m/px) …", 1, n_scenes + 5)
    vhr = fetch_vhr_fn(bbox, target_res_m, max_tiles_per_side)
    rgb_vhr = vhr["rgb"]
    H_vhr, W_vhr = rgb_vhr.shape[:2]
    res_vhr = float(vhr.get("resolution_m", target_res_m))
    pct += _STAGE_WEIGHTS["vhr"]
    L.info(f"VHR: {W_vhr}×{H_vhr} px @ {res_vhr:.2f} m/px")
    _emit(progress_cb, pct, f"VHR ready · {W_vhr}×{H_vhr} px @ {res_vhr:.2f} m/px",
          1, n_scenes + 5)

    # ── Stage 2: 10-scene calendar ─────────────────────────────────────────
    if _is_cancelled(cancel_cb):
        raise RuntimeError("cancelled")
    windows = _scene_windows(year, n_scenes=n_scenes, pad_days=15)
    pct += _STAGE_WEIGHTS["calendar"]
    _emit(progress_cb, pct, f"Scheduled {n_scenes} S2 median windows for {year}",
          2, n_scenes + 5)

    # ── Stage 3: per-scene fetch + Lyzenga/Stumpf depth grid ──────────────
    if compute_scene_fn is None:
        raise RuntimeError("compute_scene_fn must be provided")
    grids: List[Dict[str, Any]] = []
    scene_log: List[Dict[str, Any]] = []
    per_scene_w = _STAGE_WEIGHTS["scenes"] / float(n_scenes)
    for i, win in enumerate(windows):
        if _is_cancelled(cancel_cb):
            raise RuntimeError("cancelled")
        idx = i + 1
        _emit(progress_cb, pct,
              f"Scene {idx}/{n_scenes} · S2 median {win['start']} → {win['end']}",
              idx, n_scenes + 5)
        try:
            res = compute_scene_fn(
                bbox, win["start"], win["end"],
                user_pts=user_pts or [],
                max_cloud=max_cloud,
                fetch_sliderule=fetch_sliderule,
            )
            d = res.get("depth_grid")
            w = res.get("water_mask")
            if d is None or w is None:
                raise RuntimeError("scene returned no depth grid")
            m = res.get("metrics") or {}
            bc = (res.get("augmentation") or {}).get("bias_correction") or {}
            sigma = (m.get("rmse_m")
                     or bc.get("post_rmse_m")
                     or bc.get("pre_rmse_m") or 1.5)
            grids.append({
                "depth": np.asarray(d, dtype=np.float32),
                "water": np.asarray(w, dtype=bool),
                "sigma": float(sigma),
                "date": win["date"],
                "window": [win["start"], win["end"]],
            })
            scene_log.append({
                "scene": idx, "date": win["date"], "sigma_m": round(float(sigma), 3),
                "mean_depth_m": (res.get("stats") or {}).get("mean_depth"),
                "rmse_m": m.get("rmse_m"),
                "ok": True,
            })
        except Exception as ex:
            scene_log.append({
                "scene": idx, "date": win["date"], "ok": False, "error": str(ex)[:200],
            })
            L.info(f"MLE-PRO: scene {idx} ({win['date']}) skipped — {ex}")
        pct += per_scene_w
        _emit(progress_cb, pct,
              f"Scene {idx}/{n_scenes} done · σ ≈ {scene_log[-1].get('sigma_m','—')} m",
              idx, n_scenes + 5)

    if not grids:
        raise RuntimeError("All 10 S2 scenes failed to return a depth grid")

    # ── Stage 4: per-pixel inverse-variance MLE ───────────────────────────
    if _is_cancelled(cancel_cb):
        raise RuntimeError("cancelled")
    _emit(progress_cb, pct,
          f"Fusing {len(grids)} scenes · per-pixel inverse-variance MLE",
          n_scenes + 1, n_scenes + 5)
    mle = _inv_variance_mle(grids)
    pct += _STAGE_WEIGHTS["mle"]
    _emit(progress_cb, pct,
          f"MLE done · grid {mle['W']}×{mle['H']} px", n_scenes + 1, n_scenes + 5)

    # ── Stage 5: resample MLE depth + σ + water → VHR grid ────────────────
    if _is_cancelled(cancel_cb):
        raise RuntimeError("cancelled")
    _emit(progress_cb, pct, "Resampling MLE depth → VHR grid", n_scenes + 2, n_scenes + 5)
    depth_vhr = _resample_to_vhr(mle["depth"], H_vhr, W_vhr)
    sigma_vhr = _resample_to_vhr(mle["sigma"], H_vhr, W_vhr)
    water_vhr = _resample_to_vhr(
        mle["water"].astype(np.float32), H_vhr, W_vhr, nearest=True) > 0.5
    depth_vhr = np.where(water_vhr, depth_vhr, np.nan).astype(np.float32)
    sigma_vhr = np.where(water_vhr, sigma_vhr, np.nan).astype(np.float32)
    pct += _STAGE_WEIGHTS["resample"]
    _emit(progress_cb, pct,
          f"VHR raster {W_vhr}×{H_vhr} px @ {res_vhr:.2f} m/px ready",
          n_scenes + 2, n_scenes + 5)

    # ── Stage 6: in-situ calibration if user_pts supplied ─────────────────
    if _is_cancelled(cancel_cb):
        raise RuntimeError("cancelled")
    cal = {"alpha": 0.0, "beta": 1.0, "applied": False, "n_refs": 0}
    if user_pts:
        try:
            w_, s_, e_, n_ = bbox
            lats = np.asarray([float(p["lat"]) for p in user_pts])
            lons = np.asarray([float(p["lon"]) for p in user_pts])
            dgt = np.abs(np.asarray([float(p["depth"]) for p in user_pts]))
            cols = ((lons - w_) / (e_ - w_) * W_vhr).astype(int)
            rows = ((n_ - lats) / (n_ - s_) * H_vhr).astype(int)
            ok = (rows >= 0) & (rows < H_vhr) & (cols >= 0) & (cols < W_vhr)
            rows, cols, dgt = rows[ok], cols[ok], dgt[ok]
            preds = depth_vhr[rows, cols]
            ok = np.isfinite(preds) & np.isfinite(dgt) & (dgt > 0)
            if ok.sum() >= 10:
                x = preds[ok]; y = dgt[ok]
                # Linear least squares y = α + β·x
                X = np.vstack([np.ones_like(x), x]).T
                (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
                depth_vhr = np.clip(a + b * depth_vhr, 0, MAX_DEPTH_M)
                cal = {"alpha": float(a), "beta": float(b),
                       "applied": True, "n_refs": int(ok.sum())}
        except Exception as ex:
            L.info(f"MLE-PRO calibration skipped: {ex}")
    pct += _STAGE_WEIGHTS["calibrate"]
    _emit(progress_cb, pct,
          (f"Calibration α={cal['alpha']:.3f}, β={cal['beta']:.3f} "
           f"({cal['n_refs']} refs)" if cal["applied"] else
           "Calibration skipped (no in-situ refs)"),
          n_scenes + 3, n_scenes + 5)

    # ── Stage 7: PNG render + save ────────────────────────────────────────
    if _is_cancelled(cancel_cb):
        raise RuntimeError("cancelled")
    _emit(progress_cb, pct, "Rendering depth + uncertainty PNGs",
          n_scenes + 4, n_scenes + 5)
    depth_png = _depth_png(depth_vhr, water_vhr)
    sigma_png = _sigma_png(sigma_vhr, water_vhr)
    paths: Dict[str, str] = {}
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # SPEC-2: deterministic content/param key instead of int(time.time())
        # so identical (bbox, year, n_scenes, target_res_m) -> identical artifact.
        import hashlib as _hl
        _key = _hl.sha1(repr([
            [round(float(v), 9) for v in bbox], int(year), int(n_scenes),
            round(float(target_res_m), 6),
        ]).encode()).hexdigest()[:12]
        dp = save_dir / f"vhr_mle_pro_depth_{_key}.png"
        sp = save_dir / f"vhr_mle_pro_sigma_{_key}.png"
        dp.write_bytes(depth_png); sp.write_bytes(sigma_png)
        paths = {"depth_png": str(dp), "sigma_png": str(sp)}

    pct += _STAGE_WEIGHTS["render"]
    _emit(progress_cb, min(pct, 99.0),
          f"Done — {len(grids)} scenes fused @ {res_vhr:.2f} m/px",
          n_scenes + 5, n_scenes + 5)

    # ── Stats ─────────────────────────────────────────────────────────────
    val = depth_vhr[np.isfinite(depth_vhr) & (depth_vhr > 0)]
    stats = {
        "mean_depth_m": round(float(np.mean(val)), 2) if val.size else 0.0,
        "min_depth_m":  round(float(np.min(val)),  2) if val.size else 0.0,
        "max_depth_m":  round(float(np.max(val)),  2) if val.size else 0.0,
        "std_depth_m":  round(float(np.std(val)),  2) if val.size else 0.0,
        "mean_sigma_m": (round(float(np.nanmean(sigma_vhr)), 2)
                        if np.any(np.isfinite(sigma_vhr)) else 0.0),
        "grid_points": int(val.size),
        "resolution_m": round(res_vhr, 3),
    }
    elapsed = round(time.time() - t0, 1)
    L.info(f"VHR-MLE PRO done: {len(grids)}/{n_scenes} scenes, "
           f"{elapsed}s, mean depth {stats['mean_depth_m']} m, "
           f"mean σ {stats['mean_sigma_m']} m")

    import base64 as _b64
    return {
        "method": (f"Very-HR + MLE PRO · {len(grids)} S2 median scenes "
                   f"(year {year}) + Mapbox VHR @ {res_vhr:.2f} m/px"),
        "bbox": list(bbox),
        "resolution_m": round(res_vhr, 3),
        "n_scenes_attempted": n_scenes,
        "n_scenes_used": len(grids),
        "year": int(year),
        "scenes": scene_log,
        "stats": stats,
        "calibration": cal,
        "elapsed_s": elapsed,
        "depth_png_b64": _b64.b64encode(depth_png).decode(),
        "uncertainty_png_b64": _b64.b64encode(sigma_png).decode(),
        "raster_bounds": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "paths": paths,
    }
