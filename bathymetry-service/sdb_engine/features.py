"""Spectral feature engineering for SDB.

13-feature stack (must match training): raw reflectances (blue, green, red,
coastal, nir), Lyzenga log-bands (lnB, lnG, lnR, lnC), Stumpf log-ratios
(B/G, B/R), NDWI and the Lyzenga depth-invariant index.

References: Lyzenga (1985), Stumpf et al. (2003), Caballero & Stumpf (2020),
McFeeters (1996).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

FEATURE_NAMES = [
    "blue", "green", "red", "coastal", "nir",
    "lnB", "lnG", "lnR", "lnC",
    "stumpf_BG", "stumpf_BR",
    "ndwi", "lyzenga_DII",
]
N_FEATURES = len(FEATURE_NAMES)

MAX_DEPTH_M = 25.0


def _safe_reflectance(band: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """DN (reflectance*10000) -> reflectance with a strictly positive floor."""
    return np.clip(band.astype(np.float64) / 10000.0, eps, None)


def build_feature_stack(s2: Dict, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (H, W, N_FEATURES) stack + water mask from an S2-like dict.

    Expects keys blue/green/red (uint16-scale DN); coastal defaults to blue,
    nir to red, ndwi/water_mask computed when absent.
    """
    blue = _safe_reflectance(s2["blue"], eps)
    green = _safe_reflectance(s2["green"], eps)
    red = _safe_reflectance(s2["red"], eps)
    coastal = _safe_reflectance(s2.get("coastal", s2["blue"]), eps)
    nir = _safe_reflectance(s2.get("nir", s2["red"]), eps)
    ndwi = np.asarray(s2.get("ndwi", (green - nir) / (green + nir + eps)),
                      dtype=np.float64)

    n = 1000.0
    stumpf_bg = np.log(n * blue) / np.log(n * green + eps)
    stumpf_br = np.log(n * blue) / np.log(n * red + eps)

    lnB = np.log(blue)
    lnG = np.log(green)
    lnR = np.log(red)
    lnC = np.log(coastal)

    water = s2.get("water_mask")
    if water is None:
        water = ndwi > 0
    water = np.asarray(water, dtype=bool)
    if water.sum() > 50:
        std_b = float(np.std(lnB[water]))
        std_g = float(np.std(lnG[water]))
        ratio = (std_b / std_g) if std_g > 1e-6 else 1.0
    else:
        ratio = 1.0
    dii = lnB - ratio * lnG

    feats = np.stack([blue, green, red, coastal, nir,
                      lnB, lnG, lnR, lnC,
                      stumpf_bg, stumpf_br,
                      ndwi, dii], axis=-1).astype(np.float32)
    return feats, water


def stumpf_log_ratio(blue_dn: np.ndarray, green_dn: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    """Stumpf (2003) blue/green log-ratio pSDB."""
    b = _safe_reflectance(blue_dn, eps)
    g = _safe_reflectance(green_dn, eps)
    n = 1000.0
    return (np.log(n * b) / np.log(n * g + eps)).astype(np.float32)


def percentile_stumpf(ratio: np.ndarray, water: np.ndarray,
                      ref_depths=None, max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """Uncalibrated Stumpf pseudo-depth: percentile stretch of the log-ratio
    to a literature-anchored range. Relative shape only — NOT metric depth."""
    rv = ratio[water]
    rv = rv[np.isfinite(rv)]
    if len(rv) < 20:
        return np.full_like(ratio, np.nan, dtype=np.float32)
    p2, p98 = np.percentile(rv, 2), np.percentile(rv, 98)
    if not np.isfinite(p2) or not np.isfinite(p98) or p98 - p2 < 1e-6:
        return np.full_like(ratio, np.nan, dtype=np.float32)
    if ref_depths is not None and len(ref_depths) >= 5:
        d_max = float(np.clip(np.percentile(ref_depths, 95), 5.0, max_depth))
    else:
        d_max = max_depth * 0.8
    z = d_max * (np.clip(ratio, p2, p98) - p2) / (p98 - p2)
    return z.astype(np.float32)
