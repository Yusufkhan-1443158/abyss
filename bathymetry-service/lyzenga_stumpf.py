"""
Lyzenga + Stumpf weighted fusion — robust Sentinel-2 SDB (vendored verbatim
from Bathymetry_Production/backend/lyzenga_sliderule.py)
============================================================

Combines:
  • Lyzenga (1985, 2006) multi-band log-linear regression on
    deep-water-subtracted reflectance:
        z = a0 + sum_i a_i * ln(R_i - R_inf_i)
  • Stumpf (2003) / Caballero & Stumpf (2020) log-ratio:
        pSDB_BG = ln(n*Rrs_blue) / ln(n*Rrs_green)
        pSDB_BR = ln(n*Rrs_blue) / ln(n*Rrs_red)
  • SlideRule ICESat-2 ATL03 bathymetric photons as the highest-weight
    calibration source (sigma_z ~ 0.6-1.0 m after refraction correction).

Robustness guarantees:
  • Never returns an all-NaN grid over a non-trivial water mask.
  • If calibration fails (too few training points or degenerate WLS),
    falls back to a percentile-scaled uncalibrated Stumpf estimator
    bounded to [0, MAX_DEPTH_M].
  • Ensembles Lyzenga + Stumpf using weights derived from training-fit
    residuals (inverse-RMSE), so the better-fitting model dominates.

References
----------
Lyzenga, D. R. (1985). Shallow-water bathymetry using combined lidar
and passive multispectral scanner data. IJRS 6(1), 115-125.
Lyzenga, D. R. et al. (2006). Multispectral bathymetry using a simple
physically based algorithm. IEEE TGARS 44(8), 2251-2259.
Stumpf, R. P. et al. (2003). Determination of water depth with
high-resolution satellite imagery over variable bottom types.
L&O 48(1, part 2), 547-556.
Caballero, I. & Stumpf, R. P. (2020). Towards routine mapping of
shallow bathymetry in environments with variable turbidity. Remote
Sensing 12(3), 451.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

L = logging.getLogger(__name__)

# Hard global cap (matches the rest of the pipeline)
MAX_DEPTH_M = 25.0

# Reflectance scaling: S2 L2A DN → reflectance is DN/10000
DN_SCALE = 10000.0

# Per-source weights for the calibration WLS solve. Higher = more trusted.
# These mirror the values already used elsewhere in the codebase
# (backend/swin_unet_sdb.py, backend/very_hr_augment.py).
W_OBSERVED = 6.0   # in-situ survey, single-beam echo sounder, multibeam
W_SLIDERULE = 5.0  # ICESat-2 ATL03 bathymetric photons (refraction corrected)
W_CHART = 2.0      # nautical chart soundings extracted via Gemini Vision
W_GEBCO = 1.0      # 450 m GEBCO interpolation


# ────────────────────────────────────────────────────────────────────
# Feature engineering
# ────────────────────────────────────────────────────────────────────
def _safe_reflectance(band: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """DN → reflectance with strictly positive floor (avoid log(<=0))."""
    r = band.astype(np.float64) / DN_SCALE
    return np.clip(r, eps, None)


def _deep_water_subtract(band: np.ndarray, water: np.ndarray,
                         deep_mask: np.ndarray, eps: float) -> Tuple[np.ndarray, float]:
    """Subtract Rw_inf (deep-water reflectance) using the dark-pixel
    method (Lyzenga 1978). Returns the corrected band and the Rw_inf
    estimate. If deep-water pixels are scarce, returns the original band.
    """
    if deep_mask.sum() >= 30:
        rw_inf = float(np.median(band[deep_mask]))
    elif water.sum() >= 200:
        rw_inf = float(np.percentile(band[water], 5))
    else:
        rw_inf = 0.0
    corr = np.clip(band - rw_inf, eps, None)
    return corr, rw_inf


def build_features(s2: Dict, eps: float = 1e-5) -> Dict[str, np.ndarray]:
    """Compute Lyzenga + Stumpf feature stack from a Sentinel-2 bands dict.

    Returns
    -------
    dict with keys: 'lnB', 'lnG', 'lnR', 'lnC', 'pSDB_BG', 'pSDB_BR',
    'water', 'rw_inf'
    """
    blue_raw = _safe_reflectance(s2['blue'], eps)
    green_raw = _safe_reflectance(s2['green'], eps)
    red_raw = _safe_reflectance(s2['red'], eps)
    coastal_raw = _safe_reflectance(s2.get('coastal', s2['blue']), eps)
    nir_raw = _safe_reflectance(s2.get('nir', s2['red']), eps)

    water = s2.get('water_mask')
    if water is None:
        water = s2.get('ndwi', np.zeros_like(blue_raw)) > 0
    water = np.asarray(water, dtype=bool)

    # Deep, clear-water pixels: high NDWI, very low NIR, low red (no glint)
    deep = water & (nir_raw < 0.02) & (red_raw < 0.05)

    blue, rw_b = _deep_water_subtract(blue_raw, water, deep, eps)
    green, rw_g = _deep_water_subtract(green_raw, water, deep, eps)
    red, rw_r = _deep_water_subtract(red_raw, water, deep, eps)
    coastal, rw_c = _deep_water_subtract(coastal_raw, water, deep, eps)

    # Lyzenga log-bands (Lyzenga 1985 eq. 2)
    ln_b = np.log(blue)
    ln_g = np.log(green)
    ln_r = np.log(red)
    ln_c = np.log(coastal)

    # Stumpf log-ratios (Stumpf 2003; Caballero & Stumpf 2020 eq. 1-2)
    n = 1000.0
    pSDB_BG = np.log(n * blue_raw) / np.log(n * green_raw + eps)
    pSDB_BR = np.log(n * blue_raw) / np.log(n * red_raw + eps)

    return {
        'lnB': ln_b.astype(np.float32),
        'lnG': ln_g.astype(np.float32),
        'lnR': ln_r.astype(np.float32),
        'lnC': ln_c.astype(np.float32),
        'pSDB_BG': pSDB_BG.astype(np.float32),
        'pSDB_BR': pSDB_BR.astype(np.float32),
        'water': water,
        'rw_inf': {'blue': rw_b, 'green': rw_g, 'red': rw_r, 'coastal': rw_c},
    }


# ────────────────────────────────────────────────────────────────────
# Calibration training set — fuse all sources with weights
# ────────────────────────────────────────────────────────────────────
def _latlon_to_pixel(lat: float, lon: float, bbox, H: int, W: int) -> Tuple[int, int]:
    w_b, s_b, e_b, n_b = bbox
    r_px = int((n_b - lat) / (n_b - s_b + 1e-10) * H)
    c_px = int((lon - w_b) / (e_b - w_b + 1e-10) * W)
    return max(0, min(H - 1, r_px)), max(0, min(W - 1, c_px))


def collect_training(feats: Dict, bbox,
                     ref_lats, ref_lons, ref_depths,
                     ref_weights=None,
                     sliderule_pts: Optional[List[Dict]] = None) -> Dict:
    """Map every reference depth to a feature vector + weight.

    sliderule_pts is a list of dicts with keys 'lat', 'lon', 'depth',
    optionally 'confidence' (high/medium/low).
    """
    H, W = feats['lnB'].shape
    water = feats['water']
    lnB, lnG, lnR, lnC = feats['lnB'], feats['lnG'], feats['lnR'], feats['lnC']
    pBG, pBR = feats['pSDB_BG'], feats['pSDB_BR']

    Xl, ys, ws = [], [], []  # Lyzenga features
    Xs_bg, Xs_br = [], []    # Stumpf features

    def push(lat, lon, depth, weight):
        if not (0 < depth <= MAX_DEPTH_M):
            return
        r, c = _latlon_to_pixel(lat, lon, bbox, H, W)
        if not water[r, c]:
            return
        feats_l = (lnC[r, c], lnB[r, c], lnG[r, c], lnR[r, c])
        feats_s = (pBG[r, c], pBR[r, c])
        if not all(np.isfinite(f) for f in feats_l + feats_s):
            return
        Xl.append(feats_l)
        Xs_bg.append(feats_s[0])
        Xs_br.append(feats_s[1])
        ys.append(float(depth))
        ws.append(float(weight))

    # In-situ / GEBCO / chart / RAG points (caller-supplied weights)
    if ref_lats is not None and ref_depths is not None:
        n = len(ref_depths)
        for i in range(n):
            wt = float(ref_weights[i]) if ref_weights is not None else W_GEBCO
            push(float(ref_lats[i]), float(ref_lons[i]), float(ref_depths[i]), wt)

    # SlideRule ICESat-2 photons — confidence-weighted
    if sliderule_pts:
        conf_w = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
        for p in sliderule_pts:
            d = float(p.get('depth', 0))
            if d <= 0.3 or p.get('photon_class') != 'bathymetry':
                continue
            cw = conf_w.get(p.get('confidence', 'medium'), 0.6)
            push(float(p['lat']), float(p['lon']), d, W_SLIDERULE * cw)

    return {
        'X_lyz': np.asarray(Xl, dtype=np.float64) if Xl else np.zeros((0, 4)),
        'X_bg': np.asarray(Xs_bg, dtype=np.float64) if Xs_bg else np.zeros(0),
        'X_br': np.asarray(Xs_br, dtype=np.float64) if Xs_br else np.zeros(0),
        'y': np.asarray(ys, dtype=np.float64) if ys else np.zeros(0),
        'w': np.asarray(ws, dtype=np.float64) if ws else np.zeros(0),
    }


# ────────────────────────────────────────────────────────────────────
# Weighted least squares with Tikhonov regularisation
# ────────────────────────────────────────────────────────────────────
def _wls_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray,
             ridge: float = 1e-3) -> Tuple[Optional[np.ndarray], float]:
    """Solve min ||W^{1/2}(X b - y)||^2 + ridge * ||b||^2.

    Returns (coefficients, training RMSE). Adds a bias column to X
    automatically. Returns (None, inf) if the solve fails.
    """
    if len(y) < 3:
        return None, float('inf')
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    sw = np.sqrt(np.maximum(w, 1e-6))
    Xw = Xb * sw[:, None]
    yw = y * sw
    # Normal equations with Tikhonov: (Xw^T Xw + lam I) b = Xw^T yw
    A = Xw.T @ Xw + ridge * np.eye(Xb.shape[1])
    b = Xw.T @ yw
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, float('inf')
    if not np.all(np.isfinite(coef)):
        return None, float('inf')
    pred = Xb @ coef
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return coef, rmse


def fit_lyzenga(train: Dict) -> Optional[Dict]:
    X = train['X_lyz']
    y = train['y']
    w = train['w']
    if X.shape[0] < 6:
        return None
    coef, rmse = _wls_fit(X, y, w)
    if coef is None:
        return None
    return {'coef': coef, 'rmse': rmse, 'n': X.shape[0]}


def fit_stumpf(train: Dict, ratio_key: str = 'X_bg') -> Optional[Dict]:
    X = train[ratio_key].reshape(-1, 1)
    y = train['y']
    w = train['w']
    if X.shape[0] < 4:
        return None
    coef, rmse = _wls_fit(X, y, w)
    if coef is None:
        return None
    return {'coef': coef, 'rmse': rmse, 'n': X.shape[0]}


# ────────────────────────────────────────────────────────────────────
# Apply fitted models to the full grid
# ────────────────────────────────────────────────────────────────────
def _apply_lyzenga(feats: Dict, fit: Dict) -> np.ndarray:
    a_c, a_b, a_g, a_r, a0 = fit['coef']
    z = a_c * feats['lnC'] + a_b * feats['lnB'] + \
        a_g * feats['lnG'] + a_r * feats['lnR'] + a0
    return z.astype(np.float32)


def _apply_stumpf(ratio: np.ndarray, fit: Dict) -> np.ndarray:
    m1, m0 = float(fit['coef'][0]), float(fit['coef'][1])
    return (m1 * ratio + m0).astype(np.float32)


def _percentile_stumpf(ratio: np.ndarray, water: np.ndarray,
                        ref_depths=None) -> np.ndarray:
    """Uncalibrated Stumpf — percentile stretch to a literature-based
    depth range. Used as a last-resort fallback so the pipeline never
    returns all-NaN over water.
    """
    rv = ratio[water]
    rv = rv[np.isfinite(rv)]
    if len(rv) < 20:
        return np.full_like(ratio, np.nan, dtype=np.float32)
    p2, p98 = np.percentile(rv, 2), np.percentile(rv, 98)
    if not np.isfinite(p2) or not np.isfinite(p98) or p98 - p2 < 1e-6:
        return np.full_like(ratio, np.nan, dtype=np.float32)
    if ref_depths is not None and len(ref_depths) >= 5:
        d_max = float(np.clip(np.percentile(ref_depths, 95), 5.0, MAX_DEPTH_M))
    else:
        d_max = MAX_DEPTH_M * 0.8  # default literature anchor
    z = d_max * (np.clip(ratio, p2, p98) - p2) / (p98 - p2)
    return z.astype(np.float32)


# ────────────────────────────────────────────────────────────────────
# Ensemble — combine Lyzenga + Stumpf BG + Stumpf BR
# ────────────────────────────────────────────────────────────────────
def _inv_rmse_weights(rmses: List[float]) -> List[float]:
    inv = [1.0 / max(r, 0.1) for r in rmses]
    s = sum(inv)
    return [v / s for v in inv]


def estimate_depth(s2: Dict, bbox,
                   ref_lats=None, ref_lons=None, ref_depths=None,
                   ref_weights=None,
                   sliderule_pts: Optional[List[Dict]] = None) -> Dict:
    """Top-level entry point: returns a depth grid plus diagnostics.

    The output 'depth' grid is guaranteed to contain finite values over
    a non-trivial water mask (unless the entire ROI is land). NaN is
    used only to mark land/cloud/non-water pixels.
    """
    feats = build_features(s2)
    water = feats['water']
    H, W = feats['lnB'].shape

    # If barely any water → return all-NaN depth (nothing meaningful to do)
    if int(water.sum()) < 50:
        return {
            'depth': np.full((H, W), np.nan, dtype=np.float32),
            'method': 'Lyzenga+Stumpf (no water)',
            'fit': {}, 'sources': [], 'water_pixels': int(water.sum()),
        }

    train = collect_training(feats, bbox, ref_lats, ref_lons, ref_depths,
                             ref_weights=ref_weights,
                             sliderule_pts=sliderule_pts)
    n_train = len(train['y'])
    n_sliderule = int(np.sum(train['w'] >= W_SLIDERULE * 0.3)) if n_train else 0
    L.info(f"Lyzenga+SlideRule: {n_train} training pts ({n_sliderule} from SlideRule)")

    fit_l = fit_lyzenga(train) if n_train >= 6 else None
    fit_bg = fit_stumpf(train, 'X_bg') if n_train >= 4 else None
    fit_br = fit_stumpf(train, 'X_br') if n_train >= 4 else None

    grids: List[Tuple[str, np.ndarray, float]] = []  # (name, grid, rmse)
    if fit_l is not None:
        g = _apply_lyzenga(feats, fit_l)
        grids.append(('Lyzenga', g, fit_l['rmse']))
        L.info(f"Lyzenga fit: RMSE={fit_l['rmse']:.2f}m on {fit_l['n']} pts, "
               f"coefs={[round(c, 3) for c in fit_l['coef']]}")
    if fit_bg is not None:
        g = _apply_stumpf(feats['pSDB_BG'], fit_bg)
        grids.append(('Stumpf_BG', g, fit_bg['rmse']))
        L.info(f"Stumpf B/G fit: RMSE={fit_bg['rmse']:.2f}m on {fit_bg['n']} pts, "
               f"m1={fit_bg['coef'][0]:.3f}, m0={fit_bg['coef'][1]:.3f}")
    if fit_br is not None:
        g = _apply_stumpf(feats['pSDB_BR'], fit_br)
        grids.append(('Stumpf_BR', g, fit_br['rmse']))
        L.info(f"Stumpf B/R fit: RMSE={fit_br['rmse']:.2f}m on {fit_br['n']} pts, "
               f"m1={fit_br['coef'][0]:.3f}, m0={fit_br['coef'][1]:.3f}")

    if grids:
        weights = _inv_rmse_weights([g[2] for g in grids])
        depth = np.zeros((H, W), dtype=np.float32)
        wsum = np.zeros((H, W), dtype=np.float32)
        for (name, grid, _), wt in zip(grids, weights):
            valid = water & np.isfinite(grid)
            depth[valid] += wt * grid[valid]
            wsum[valid] += wt
        good = wsum > 0
        out = np.full((H, W), np.nan, dtype=np.float32)
        out[good] = depth[good] / wsum[good]
        out = np.clip(out, 0.0, MAX_DEPTH_M)
        out[~water] = np.nan
        method = ' + '.join(f"{g[0]}×{w:.2f}" for g, w in zip(grids, weights))
        # Final NaN sweep over water — fill any rare gaps with uncalibrated Stumpf
        nan_water = water & ~np.isfinite(out)
        if int(nan_water.sum()) > 0:
            uc = _percentile_stumpf(feats['pSDB_BG'], water,
                                     ref_depths=ref_depths)
            fill = water & ~np.isfinite(out) & np.isfinite(uc)
            out[fill] = np.clip(uc[fill], 0.0, MAX_DEPTH_M)
            L.info(f"Filled {int(fill.sum())} residual NaN water pixels "
                   f"with uncalibrated Stumpf")
        return {
            'depth': out, 'method': method,
            'fit': {'lyzenga': fit_l, 'stumpf_bg': fit_bg, 'stumpf_br': fit_br},
            'sources': [g[0] for g in grids], 'water_pixels': int(water.sum()),
            'n_train': n_train, 'n_sliderule': n_sliderule,
        }

    # No usable fit → uncalibrated Stumpf (always finite over water)
    L.warning(f"Lyzenga+SlideRule: no calibration possible "
              f"(n_train={n_train}), using percentile-scaled Stumpf B/G")
    uc_bg = _percentile_stumpf(feats['pSDB_BG'], water, ref_depths=ref_depths)
    uc_br = _percentile_stumpf(feats['pSDB_BR'], water, ref_depths=ref_depths)
    if np.any(np.isfinite(uc_bg)) and np.any(np.isfinite(uc_br)):
        both = np.isfinite(uc_bg) & np.isfinite(uc_br)
        depth = np.where(both, 0.6 * uc_bg + 0.4 * uc_br,
                         np.where(np.isfinite(uc_bg), uc_bg, uc_br))
    else:
        depth = uc_bg if np.any(np.isfinite(uc_bg)) else uc_br
    depth = np.where(water, np.clip(depth, 0.0, MAX_DEPTH_M), np.nan)
    return {
        'depth': depth.astype(np.float32),
        'method': 'Stumpf (uncalibrated, percentile-scaled)',
        'fit': {}, 'sources': ['Stumpf_BG_uncal'],
        'water_pixels': int(water.sum()),
        'n_train': n_train, 'n_sliderule': n_sliderule,
    }
