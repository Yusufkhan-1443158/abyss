"""NDWI-based land/water separation (McFeeters 1996).

`ndwi_water_mask` is the primary land cut: NDWI = (green - nir)/(green + nir),
water where NDWI > 0. NaN NDWI (nodata/cloud) is never classified as water.

`ndwi_land_mask` is a conservative ADDITIVE cut for deployments that also
carry a vector land mask: it only flags confidently dry pixels (NDWI < -0.5,
far below any water/land split) that are attached to known land (reclaimed
pads) or form small isolated ship-sized blobs. A naive Otsu-threshold cut was
measured unsafe over turbid shallows (depressed NDWI), hence the fixed
conservative threshold and the additive-only gating.
"""
from __future__ import annotations

import numpy as np

NDWI_LAND_THRESHOLD = -0.5
NDWI_BUFFER_M = 20.0
NDWI_MAX_SHIP_BLOB_M2 = 1000.0


def ndwi_water_mask(green_dn: np.ndarray, nir_dn: np.ndarray,
                    threshold: float = 0.0, eps: float = 1e-9):
    """Water mask + NDWI grid from green/NIR DN bands."""
    g = np.asarray(green_dn, dtype=np.float64)
    n = np.asarray(nir_dn, dtype=np.float64)
    ndwi = (g - n) / (g + n + eps)
    water = np.isfinite(ndwi) & (ndwi > threshold)
    return water, ndwi.astype(np.float32)


def ndwi_otsu_threshold(ndwi: np.ndarray) -> float | None:
    """Diagnostic-only Otsu split of the scene's NDWI histogram (surfaced in
    metadata for transparency; not used as the cut threshold)."""
    try:
        from skimage.filters import threshold_otsu
        finite = np.isfinite(ndwi)
        if finite.sum() < 100:
            return None
        return float(threshold_otsu(ndwi[finite]))
    except Exception:
        return None


def ndwi_land_mask(ndwi: np.ndarray, vec_land: np.ndarray, res_m: float = 10.0,
                   threshold: float = NDWI_LAND_THRESHOLD,
                   buffer_m: float = NDWI_BUFFER_M,
                   max_ship_blob_m2: float = NDWI_MAX_SHIP_BLOB_M2):
    """Conservative additive land layer gated to `vec_land` (known land).

    Returns (extra_land bool (H, W), stats dict); `extra_land` never overlaps
    `vec_land`, never cuts NaN pixels and never floods into open water.
    """
    from scipy import ndimage

    ndwi = np.asarray(ndwi, dtype=np.float64)
    vec_land = np.asarray(vec_land, dtype=bool)

    otsu = ndwi_otsu_threshold(ndwi)
    finite = np.isfinite(ndwi)
    raw_land = (ndwi < threshold) & finite
    land_clean = (ndimage.binary_opening(raw_land, structure=np.ones((3, 3)))
                  if raw_land.any() else raw_land)

    buf_px = max(1, int(round(buffer_m / max(res_m, 1e-6))))
    buffered_vec = (ndimage.binary_dilation(vec_land, iterations=buf_px)
                    if vec_land.any() else vec_land)
    near_vec = land_clean & buffered_vec & ~vec_land

    isolated_candidates = land_clean & ~buffered_vec
    max_blob_px = max(1, int(round(max_ship_blob_m2 / (res_m ** 2))))
    ship_mask = np.zeros_like(land_clean)
    n_blobs = 0
    if isolated_candidates.any():
        lbl, n_blobs = ndimage.label(isolated_candidates)
        if n_blobs:
            sizes = ndimage.sum(isolated_candidates, lbl, range(1, n_blobs + 1))
            for i, sz in enumerate(np.atleast_1d(sizes), start=1):
                if sz <= max_blob_px:
                    ship_mask |= (lbl == i)

    extra_land = near_vec | ship_mask
    stats = {
        "threshold_used": threshold, "otsu_threshold_diagnostic": otsu,
        "buffer_m": buffer_m, "max_ship_blob_m2": max_ship_blob_m2,
        "raw_land_px": int(land_clean.sum()),
        "near_vector_extra_px": int(near_vec.sum()),
        "ship_blob_px": int(ship_mask.sum()), "n_isolated_blobs": int(n_blobs),
        "extra_land_px": int(extra_land.sum()),
        "land_pct": round(100.0 * float(land_clean.mean()), 3) if land_clean.size else None,
    }
    return extra_land, stats
