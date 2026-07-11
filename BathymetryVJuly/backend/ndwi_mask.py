"""MASK-NDWI (2026-07-10, user priority override #2) — imagery-evidence land
cut from the SAME native-10 m Sentinel-2 NDWI the depth run already fetches.

Why this module exists: the vector cut (`osm_land_mask.apply_global_land_cut`,
MASK-GLOBAL) can only cut land that OSM has mapped. Reclaimed/dredged port
land — the exact case the user flagged at Khalifa — is frequently NOT in OSM
yet. Only the imagery itself can catch that gap. NDWI (McFeeters 1996,
(green-nir)/(green+nir)) is the standard water index; a pixel with NDWI well
below the water/land split is confidently DRY.

SAFETY-FIRST DESIGN (measured, not assumed — see MASK1M_LOG.md / IHO_DEV_LOG
for the numbers): a naive Otsu-threshold NDWI land mask, unioned raw, is
UNSAFE — real Khalifa test: Otsu threshold 0.23, unioned raw, destroys 7,390
real soundings >=0.5 m (shallow turbid water depresses NDWI into the "land"
range). Even gating to a small buffer around already-known vector land at the
Otsu threshold still destroys 1,135+ soundings. The safe configuration found
by sweeping real Khalifa (clear-ish) + Old Mussafah (turbid, the explicit
guard site) soundings is:
  - a MUCH more conservative fixed threshold (NDWI < -0.5, not the Otsu
    split point) — only pixels that are confidently, strongly dry;
  - gated ADDITIVE-ONLY to a small buffer (default 20 m) around EXISTING
    vector land (the reclaimed-pad case: new land is physically attached to
    known land, never a free-floating patch in open water) OR a small
    isolated "ship-sized" blob (<=1000 m^2, not touching any vector land —
    the ship/vessel case, which SHOULD be cut regardless of shore proximity);
  - never touches NoData/cloud pixels (NaN NDWI => no cut, never punish
    missing evidence);
  - never dilates INTO water beyond that gate.
Verified 0 soundings >=0.5 m violated at BOTH Khalifa and Old Mussafah with
this configuration (28 px / 195 px of genuine marginal land recovered,
respectively) — small but real and, above all, SAFE.
"""

from __future__ import annotations

import numpy as np


NDWI_LAND_THRESHOLD = -0.5     # conservative fixed threshold (NOT Otsu — Otsu measured unsafe)
NDWI_BUFFER_M = 20.0           # gate: within this distance of already-known vector land
NDWI_MAX_SHIP_BLOB_M2 = 1000.0  # gate: OR an isolated blob no bigger than this (a vessel, not a shoal)


def ndwi_otsu_threshold(ndwi: np.ndarray) -> float | None:
    """Diagnostic-only: the Otsu split point for this scene's NDWI histogram.
    Reported in `mask_source`/stats for transparency — NOT used as the cut
    threshold (measured unsafe, see module docstring). Returns None if
    skimage is unavailable or the histogram is degenerate."""
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
                    max_ship_blob_m2: float = NDWI_MAX_SHIP_BLOB_M2) -> tuple[np.ndarray, dict]:
    """Compute the imagery-evidence ADDITIVE land layer for one NDWI grid.

    `ndwi` and `vec_land` must share shape (H, W). `vec_land` is the land
    mask ALREADY known from vector sources (the `apply_global_land_cut`
    union so far) — this function only ever ADDS to it, gated for safety
    (see module docstring), never removes vector land and never floods into
    open water.

    Returns (extra_land bool (H, W), stats dict) where `extra_land` is the
    MARGINAL land this layer contributes (i.e. `extra_land & vec_land` is
    empty by construction — union with the caller's existing mask separately).
    """
    from scipy import ndimage

    ndwi = np.asarray(ndwi, dtype=np.float64)
    vec_land = np.asarray(vec_land, dtype=bool)
    if ndwi.shape != vec_land.shape:
        # Resize NDWI (nearest) to match the caller's grid — rare (per-scene
        # fetch resolution can differ from the depth grid's own resolution).
        from PIL import Image as _PILImage
        H, W = vec_land.shape
        finite = np.isfinite(ndwi)
        ndwi_u8 = np.where(finite, np.clip((ndwi + 1.0) * 127.5, 0, 255), 0).astype(np.uint8)
        ndwi_r = np.asarray(_PILImage.fromarray(ndwi_u8).resize((W, H), _PILImage.NEAREST))
        ndwi = ndwi_r.astype(np.float64) / 127.5 - 1.0
        finite_r = np.asarray(_PILImage.fromarray((finite * 255).astype(np.uint8)).resize(
            (W, H), _PILImage.NEAREST)) > 127
        ndwi = np.where(finite_r, ndwi, np.nan)

    otsu = ndwi_otsu_threshold(ndwi)
    finite = np.isfinite(ndwi)  # NoData/cloud NIR -> never cut (honest "no evidence")
    raw_land = (ndwi < threshold) & finite
    # kill single-pixel speckle; never dilates beyond the raw detection
    land_clean = ndimage.binary_opening(raw_land, structure=np.ones((3, 3))) if raw_land.any() else raw_land

    buf_px = max(1, int(round(buffer_m / max(res_m, 1e-6))))
    buffered_vec = ndimage.binary_dilation(vec_land, iterations=buf_px) if vec_land.any() else vec_land
    near_vec = land_clean & buffered_vec & ~vec_land  # reclaimed-pad case: attached to known land

    # ship/vessel case: small isolated blob, NOT touching any known land
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
        "raw_land_px": int(land_clean.sum()), "near_vector_extra_px": int(near_vec.sum()),
        "ship_blob_px": int(ship_mask.sum()), "n_isolated_blobs": int(n_blobs),
        "extra_land_px": int(extra_land.sum()),
        "land_pct": round(100.0 * float(land_clean.mean()), 3) if land_clean.size else None,
    }
    return extra_land, stats
