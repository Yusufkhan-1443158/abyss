"""Multi-scene inverse-variance depth compositing.

Vendored math from Bathymetry_Production/backend/app.py::_run_s2_mle
(per-pixel inverse-variance accumulation, sigma clamp [0.5, 5] m, physical
glint gate NIR > 0.030 — Kay et al. 2009). Stage-B shared calibration and
EOT20 per-scene tide are deliberately excluded (no reference soundings in
Abyss; Open-Meteo tide is applied per scene upstream instead).

    z_hat = sum_i z_i/sigma_i^2 / sum_i 1/sigma_i^2
    sigma_hat = (sum_i 1/sigma_i^2)^-1/2
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

SIGMA_CLAMP = (0.5, 5.0)
GLINT_MAX = 0.030  # mean NIR reflectance over water above which a scene is glinty


def compose(results: List[dict], glint_max: float = GLINT_MAX) -> Dict:
    """Inverse-variance composite of per-scene depth grids.

    Each result: {depth: (H,W) float, sigma: (H,W) float | scalar,
                  scene_id, acquired, cloud_cover, glint: float | None,
                  tide_m: float | None}.
    Scenes with glint > glint_max are dropped (always keeping >=1: the
    lowest-glint scene survives a total wipe-out).

    Returns {depth, sigma, agreement} where agreement carries n_scenes_used,
    a per-scene table and pixel-level cross-scene spread stats."""
    if not results:
        raise ValueError("compose() needs at least one scene result")

    glints = [r.get("glint") for r in results]
    kept = [g is None or float(g) <= glint_max for g in glints]
    if not any(kept):
        order = np.argsort([np.inf if g is None else float(g) for g in glints])
        kept = [False] * len(results)
        kept[int(order[0])] = True

    kept_results = [r for r, k in zip(results, kept) if k]
    H = min(np.asarray(r["depth"]).shape[0] for r in kept_results)
    W = min(np.asarray(r["depth"]).shape[1] for r in kept_results)

    num = np.zeros((H, W), np.float64)
    den = np.zeros((H, W), np.float64)
    cnt = np.zeros((H, W), np.int32)
    zsum = np.zeros((H, W), np.float64)
    zsq = np.zeros((H, W), np.float64)
    per_scene = []
    for r, k, g in zip(results, kept, glints):
        depth = np.asarray(r["depth"], np.float32)[:H, :W]
        sigma = r.get("sigma")
        if sigma is None:
            sigma = np.full((H, W), SIGMA_CLAMP[0], np.float32)
        elif np.ndim(sigma) == 0:
            sigma = np.full((H, W), float(sigma), np.float32)
        else:
            sigma = np.asarray(sigma, np.float32)[:H, :W]
        sigma = np.clip(sigma, *SIGMA_CLAMP)
        ok = np.isfinite(depth) & np.isfinite(sigma)
        sig_med = float(np.median(sigma[ok])) if ok.any() else None
        per_scene.append({
            "scene_id": r.get("scene_id"),
            "acquired": r.get("acquired"),
            "cloud_cover": r.get("cloud_cover"),
            "sigma_med": None if sig_med is None else round(sig_med, 3),
            "glint": None if g is None else round(float(g), 5),
            "tide_m": r.get("tide_m"),
            "kept": bool(k),
        })
        if not k:
            continue
        iv = np.zeros((H, W), np.float64)
        iv[ok] = 1.0 / (sigma[ok].astype(np.float64) ** 2)
        num[ok] += depth[ok] * iv[ok]
        den[ok] += iv[ok]
        cnt[ok] += 1
        zsum[ok] += depth[ok]
        zsq[ok] += depth[ok].astype(np.float64) ** 2

    fin = den > 0
    z_hat = np.full((H, W), np.nan, np.float32)
    z_hat[fin] = (num[fin] / den[fin]).astype(np.float32)
    sigma_hat = np.full((H, W), np.nan, np.float32)
    sigma_hat[fin] = (1.0 / np.sqrt(den[fin])).astype(np.float32)

    multi = cnt >= 2
    px_std = np.full((H, W), np.nan, np.float64)
    if multi.any():
        m = zsum[multi] / cnt[multi]
        var = np.maximum(zsq[multi] / cnt[multi] - m ** 2, 0.0)
        px_std[multi] = np.sqrt(var)
    std_vals = px_std[np.isfinite(px_std)]

    agreement = {
        "n_scenes_used": int(sum(kept)),
        "n_scenes_requested": len(results),
        "glint_max": glint_max,
        "per_scene": per_scene,
        "px_std_median_m": (round(float(np.median(std_vals)), 3)
                            if std_vals.size else None),
        "px_std_p95_m": (round(float(np.percentile(std_vals, 95)), 3)
                         if std_vals.size else None),
        "n_scenes_grid_median": (float(np.median(cnt[cnt > 0]))
                                 if (cnt > 0).any() else 0.0),
    }
    return {"depth": z_hat, "sigma": sigma_hat, "agreement": agreement}
