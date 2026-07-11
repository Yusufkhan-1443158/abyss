"""infer_depth(): per-pixel water depth from an ingested raster.

Two honest paths, selected by the available bands:

* Multispectral (blue/green/red + NIR, Sentinel-2-like): the calibrated
  UAE cluster ensemble (Random Forest + MLP where torch is available) over
  the 13-feature spectral stack, NDWI land cut, clipped to 0-25 m, with a
  per-pixel uncertainty channel (per-cluster calibration RMSE + ensemble
  disagreement).
* Plain RGB (3 bands, no NIR): Stumpf blue/green log-ratio pseudo-depth,
  percentile-stretched — RELATIVE, UNCALIBRATED, flagged as low confidence.
  Land cut via a blue-water index ((blue-red)/(blue+red)).

Input contract (stub-compatible): bands (C, H, W); returns (H, W) float32
depth in metres, NaN = nodata/land. `infer()` additionally returns the
uncertainty grid and provenance metadata.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .features import stumpf_log_ratio, percentile_stumpf, MAX_DEPTH_M
from .ndwi_mask import ndwi_water_mask, ndwi_otsu_threshold

L = logging.getLogger(__name__)

MODEL_NAME = "uae-sdb-ensemble"
MODEL_REGISTRY_VERSION = 1
FALLBACK_NAME = "stumpf-log-ratio"

CALIBRATION = {
    "domain": "UAE coastal waters (8 calibration regions)",
    "registry_version": MODEL_REGISTRY_VERSION,
    "in_sample_r2": 0.978,
    "in_sample_rmse_m": 0.735,
    "max_depth_m": MAX_DEPTH_M,
    "note": ("Calibrated on UAE Sentinel-2 scenes and soundings; outside the "
             "UAE the output is indicative only. Depths are relative to the "
             "calibration soundings' survey datum (approx. MSL/chart datum), "
             "not tide-corrected."),
}

_BAND_ALIASES = {
    "coastal": "coastal", "b01": "coastal", "b1": "coastal",
    "blue": "blue", "b02": "blue", "b2": "blue",
    "green": "green", "b03": "green", "b3": "green",
    "red": "red", "b04": "red", "b4": "red",
    "nir": "nir", "b08": "nir", "b8": "nir", "nir08": "nir",
}


def _map_bands(bands: np.ndarray, band_names: Optional[List[str]]) -> Dict[str, np.ndarray]:
    """Resolve (C, H, W) -> named bands via metadata, else positional
    convention: 3 bands = R,G,B; 4 = B,G,R,NIR (S2 B02-B03-B04-B08);
    5+ = coastal,B,G,R,NIR (S2 B01-B02-B03-B04-B08)."""
    C = bands.shape[0]
    named: Dict[str, np.ndarray] = {}
    if band_names:
        for i, nm in enumerate(band_names[:C]):
            key = _BAND_ALIASES.get(str(nm or "").strip().lower())
            if key and key not in named:
                named[key] = bands[i]
    if {"blue", "green", "red"} <= set(named):
        return named
    if C >= 5:
        order = ["coastal", "blue", "green", "red", "nir"]
    elif C == 4:
        order = ["blue", "green", "red", "nir"]
    elif C == 3:
        order = ["red", "green", "blue"]
    else:
        raise ValueError(f"need at least 3 bands, got {C}")
    return {k: bands[i] for i, k in enumerate(order)}


def _to_dn(bands: Dict[str, np.ndarray], src_dtype):
    """Normalise pixel values to the S2 L2A DN scale (reflectance*10000)
    the models were trained on: 0-1 float reflectance and 8-bit imagery are
    rescaled; uint16-scale data passes through."""
    sample = np.concatenate([np.asarray(v, dtype=np.float64).ravel()[::97]
                             for v in bands.values()])
    sample = sample[np.isfinite(sample)]
    mx = float(sample.max()) if sample.size else 0.0
    if mx <= 1.5:
        scale = 10000.0
    elif mx <= 255.0 and np.dtype(src_dtype).itemsize == 1:
        scale = 10000.0 / 255.0
    else:
        scale = 1.0
    return {k: np.asarray(v, dtype=np.float32) * scale for k, v in bands.items()}, scale


def _cluster_sigma(cluster: np.ndarray, meta: Dict) -> np.ndarray:
    """Per-pixel sigma from each cluster's calibration RMSE."""
    sigma = np.full(cluster.shape, np.nan, dtype=np.float32)
    per = meta.get("per_cluster", {}) or {}
    for c, m in per.items():
        rmse = m.get("in_sample_rmse_m") or m.get("val_rmse_m")
        if rmse is not None:
            sigma[cluster == int(c)] = float(rmse)
    return sigma


def _coastline_cut(water: np.ndarray, bbox4326) -> tuple:
    """Union the vendored vector-coastline LAND into `water` (land-authoritative:
    only removes water, never adds). Returns (water, info_or_None)."""
    if bbox4326 is None:
        return water, None
    from .coastline_mask import coastline_land_for
    land, info = coastline_land_for(bbox4326, water.shape)
    if land is not None:
        water = water & ~land
    return water, info


def infer(bands: np.ndarray, band_names: Optional[List[str]] = None,
          max_depth: float = MAX_DEPTH_M, resolution_m: float = 10.0,
          src_dtype=None, bbox4326=None) -> Dict:
    """Full inference: {'depth', 'sigma', 'method', 'calibrated', ...}.

    `src_dtype` is the on-disk dtype of the source raster (callers often cast
    to float before handing bands over); it disambiguates 8-bit imagery.
    `bbox4326` ([w, s, e, n]) enables the vector-coastline land cut where a
    committed regional GPKG covers the area."""
    bands = np.asarray(bands)
    if bands.ndim != 3:
        raise ValueError(f"expected (C, H, W) bands, got shape {bands.shape}")
    if src_dtype is None:
        src_dtype = bands.dtype
    named = _map_bands(bands, band_names)
    named, dn_scale = _to_dn(named, src_dtype)

    has_nir = "nir" in named
    if has_nir:
        return _infer_multispectral(named, max_depth, dn_scale, resolution_m,
                                    bbox4326=bbox4326)
    return _infer_rgb_fallback(named, max_depth, dn_scale, bbox4326=bbox4326)


def _infer_multispectral(named, max_depth, dn_scale, resolution_m,
                         bbox4326=None) -> Dict:
    from . import uae_rf, uae_cnn

    water, ndwi = ndwi_water_mask(named["green"], named["nir"])
    water, coast_info = _coastline_cut(water, bbox4326)
    s2 = {
        "blue": named["blue"], "green": named["green"], "red": named["red"],
        "coastal": named.get("coastal", named["blue"]), "nir": named["nir"],
        "ndwi": ndwi, "water_mask": water,
    }

    rf = uae_rf.load_model()
    if rf is None:
        raise RuntimeError("UAE RF model bundle missing "
                           "(sdb_engine/models/uae_clustered_rf.pkl)")
    rf_pred = rf.predict(s2, water_mask=water, clip_max=max_depth)
    depth = rf_pred["depth"]
    cluster = rf_pred["cluster"]
    members = ["rf"]

    spread = None
    cnn = uae_cnn.load_model()
    if cnn is not None:
        try:
            cnn_pred = cnn.predict(s2, water_mask=water, clip_max=max_depth)
            d2 = cnn_pred["depth"]
            both = np.isfinite(depth) & np.isfinite(d2)
            spread = np.full(depth.shape, np.nan, dtype=np.float32)
            spread[both] = np.abs(depth[both] - d2[both])
            merged = np.where(both, 0.5 * (depth + d2), depth)
            merged = np.where(np.isfinite(depth), merged, d2)
            depth = np.clip(merged, 0.0, max_depth).astype(np.float32)
            members.append("mlp")
        except Exception as ex:
            L.warning("MLP member skipped: %s", ex)

    sigma = _cluster_sigma(cluster, rf.meta)
    if spread is not None:
        sigma = np.fmax(sigma, spread)
    sigma = np.where(np.isfinite(depth), sigma, np.nan).astype(np.float32)

    valid = np.isfinite(depth)
    mask = {
        "kind": "ndwi",
        "water_pct": round(100.0 * float(water.mean()), 1),
        "otsu_threshold_diagnostic": ndwi_otsu_threshold(ndwi),
    }
    if coast_info is not None:
        mask["coastline"] = coast_info
    return {
        "depth": depth.astype(np.float32),
        "sigma": sigma,
        "water_mask": water,
        "method": ("UAE cluster ensemble (" + "+".join(members) +
                   "), 13 spectral features, NDWI land cut"),
        "model": MODEL_NAME,
        "model_version": f"registry-v{MODEL_REGISTRY_VERSION}",
        "calibrated": True,
        "confidence": "calibrated (UAE domain); indicative elsewhere",
        "calibration": dict(CALIBRATION),
        "max_depth_m": float(max_depth),
        "band_mapping": sorted(named.keys()),
        "dn_scale_applied": dn_scale,
        "mask": mask,
        "n_valid": int(valid.sum()),
    }


def _infer_rgb_fallback(named, max_depth, dn_scale, bbox4326=None) -> Dict:
    b = np.asarray(named["blue"], dtype=np.float64)
    r = np.asarray(named["red"], dtype=np.float64)
    bwi = (b - r) / (b + r + 1e-9)
    water = np.isfinite(bwi) & (bwi > 0.05)
    water, coast_info = _coastline_cut(water, bbox4326)

    ratio = stumpf_log_ratio(named["blue"], named["green"])
    depth = percentile_stumpf(ratio, water, max_depth=max_depth)
    depth = np.where(water, depth, np.nan).astype(np.float32)
    valid = np.isfinite(depth)

    mask = {
        "kind": "blue_water_index",
        "water_pct": round(100.0 * float(water.mean()), 1),
    }
    if coast_info is not None:
        mask["coastline"] = coast_info
    return {
        "depth": depth,
        "sigma": None,
        "water_mask": water,
        "method": ("Stumpf blue/green log-ratio pseudo-depth "
                   "(percentile-stretched), blue-water-index land cut"),
        "model": FALLBACK_NAME,
        "model_version": "uncalibrated",
        "calibrated": False,
        "confidence": ("LOW — relative pseudo-depth from RGB only "
                       "(no NIR/multispectral bands); NOT metric depth"),
        "calibration": None,
        "max_depth_m": float(max_depth),
        "band_mapping": sorted(named.keys()),
        "dn_scale_applied": dn_scale,
        "mask": mask,
        "n_valid": int(valid.sum()),
    }


def infer_depth(bands: np.ndarray, max_depth: float = MAX_DEPTH_M,
                band_names: Optional[List[str]] = None) -> np.ndarray:
    """Stub-compatible entry point: (C, H, W) bands -> (H, W) float32 depth
    in metres, NaN = nodata/land."""
    return infer(bands, band_names=band_names, max_depth=max_depth)["depth"]


def model_info() -> Dict:
    from . import uae_rf, uae_cnn
    rf = uae_rf.load_model()
    info = {
        "model": MODEL_NAME,
        "registry_version": MODEL_REGISTRY_VERSION,
        "fallback": FALLBACK_NAME,
        "calibration": dict(CALIBRATION),
        "members": [],
    }
    if rf is not None:
        info["members"].append("rf")
        info["n_train"] = rf.meta.get("n_train")
        info["regions"] = [r.get("region") for r in rf.meta.get("regions", [])]
    if uae_cnn.load_model() is not None:
        info["members"].append("mlp")
    return info
