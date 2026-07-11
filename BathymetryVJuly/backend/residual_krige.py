"""S1 — Leakage-safe residual co-kriging of the CNN prior against in-situ anchors.

KHALIFA_SUB30_LOG.md request S1 (the only physically-honest path to <0.30 m the
scientist found; ATL24 is surface-only/dead in the deep basin, S2-optical & VHR
cannot cross 0.30 m).

Method (gated behind env RESIDUAL_KRIGE=1, default OFF — raw CNN path untouched):
  1. Load the frozen baseline CNN (backend/models/sdb_cnn_default.pt, PatchCNN) and
     predict mu_pred at every Khalifa multibeam anchor pixel (9x9 patches, same norm).
  2. residual r = truth - mu_pred at all anchors.
  3. Split anchors with the project's KMeans spatial-block protocol + >=500 m buffer
     (make_spatial_block_centers from sdb_cnn_baseline — the SAME leakage-safe surface).
  4. Fit a Matheron empirical variogram on TRAIN residuals; fit a spherical model
     (range/sill/nugget). Report the RANGE.
  5. Ordinary-krige r from TRAIN anchors onto the held-out TEST anchors.
  6. corrected depth = mu_pred + r_krige; evaluate RMSE/bias/decile-slope overall,
     per-2 m bin, and on the dense deep band 16-22 m.

Leakage controls (the scientist's tripwire):
  - report fitted variogram range; if range > buffer the buffered split STILL leaks.
  - 0 m-buffer control (leakage ceiling — proves the method CAN cheat).
  - buffer sweep 0/250/500/1000 m → RMSE vs buffer table.

Self-contained kriging (no skgstat): Matheron estimator + spherical model + OK.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.sdb_cnn_baseline import (  # noqa: E402
    SDBPatchDataset,
    build_feature_cube,
    fetch_s2_for_site,
    load_default_model,
    load_truth_points,
    make_spatial_block_centers,
    _predict_iter1,
    SITE_CFG,
    PATCH_RADIUS,
)

L = logging.getLogger("residual_krige")

# Deep dense-anchor band of interest (the scientist's pass-bar subset)
DEEP_LO, DEEP_HI = 16.0, 22.0
PER2M_BANDS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12),
               (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), (22, 25)]


# ──────────────────────────────────────────────────────────────────────────────
# Local geographic coordinates (metres) — equirectangular about the anchor cloud
# ──────────────────────────────────────────────────────────────────────────────
def latlon_to_local_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0 = float(np.mean(lat))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.radians(lat0))
    x = (lon - np.mean(lon)) * m_lon
    y = (lat - np.mean(lat)) * m_lat
    return np.column_stack([x, y]).astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Matheron empirical variogram + spherical model fit
# ──────────────────────────────────────────────────────────────────────────────
def empirical_variogram(
    xy: np.ndarray, vals: np.ndarray, n_lags: int = 15, max_dist: float = None,
    max_pairs: int = 400000, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matheron semivariance gamma(h)=0.5*mean((z_i-z_j)^2) over distance bins.

    Subsamples pairs (random point subset) for tractability on ~30k anchors.
    Returns (lag_centres, gamma, n_pairs_per_lag).
    """
    rng = np.random.default_rng(seed)
    n = len(vals)
    # cap the number of points used so the full pair matrix is tractable
    max_pts = int(np.sqrt(2 * max_pairs)) + 1
    if n > max_pts:
        sel = rng.choice(n, size=max_pts, replace=False)
        xy = xy[sel]; vals = vals[sel]
        n = max_pts
    # all unique pairs
    iu, ju = np.triu_indices(n, k=1)
    d = np.sqrt(np.sum((xy[iu] - xy[ju]) ** 2, axis=1))
    sv = 0.5 * (vals[iu] - vals[ju]) ** 2
    if max_dist is None:
        max_dist = np.percentile(d, 90)
    edges = np.linspace(0, max_dist, n_lags + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.full(n_lags, np.nan)
    npairs = np.zeros(n_lags, dtype=np.int64)
    idx = np.digitize(d, edges) - 1
    for k in range(n_lags):
        m = idx == k
        if m.sum() >= 30:
            gamma[k] = float(np.mean(sv[m]))
            npairs[k] = int(m.sum())
    ok = np.isfinite(gamma)
    return centres[ok], gamma[ok], npairs[ok]


def fit_spherical(centres: np.ndarray, gamma: np.ndarray) -> Tuple[float, float, float]:
    """Fit spherical variogram gamma(h)=nugget+(sill-nugget)*[1.5 h/r - 0.5 (h/r)^3], h<r.

    Returns (nugget, sill, range_m). Falls back to robust guesses if curve_fit fails.
    """
    from scipy.optimize import curve_fit

    def sph(h, nugget, psill, rng):
        h = np.asarray(h, dtype=np.float64)
        out = nugget + psill * (1.5 * h / rng - 0.5 * (h / rng) ** 3)
        out = np.where(h >= rng, nugget + psill, out)
        return out

    sill0 = float(np.nanmax(gamma))
    rng0 = float(centres[np.argmax(np.cumsum(gamma) >= 0.9 * np.sum(gamma))]) or float(np.max(centres) / 2)
    nug0 = float(max(gamma[0] * 0.3, 1e-4))
    p0 = [nug0, max(sill0 - nug0, 1e-3), max(rng0, 50.0)]
    bounds = ([0.0, 0.0, 10.0], [sill0 * 2 + 1e-3, sill0 * 4 + 1e-3, float(np.max(centres) * 3)])
    try:
        popt, _ = curve_fit(sph, centres, gamma, p0=p0, bounds=bounds, maxfev=20000)
        nugget, psill, rng = popt
        return float(nugget), float(nugget + psill), float(rng)
    except Exception as e:  # noqa: BLE001
        L.warning(f"spherical fit failed ({e}); using moment estimates")
        return nug0, sill0, max(rng0, 50.0)


def spherical_cov(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    """Covariance C(h)=sill-gamma(h) for the spherical model (used by OK)."""
    h = np.asarray(h, dtype=np.float64)
    psill = sill - nugget
    g = np.where(
        h <= 0, 0.0,
        np.where(h >= rng, psill, psill * (1.5 * h / rng - 0.5 * (h / rng) ** 3)),
    )
    gamma = np.where(h <= 0, 0.0, nugget + g)
    return sill - gamma


# ──────────────────────────────────────────────────────────────────────────────
# Ordinary kriging — local neighbourhood (k nearest train anchors per test point)
# ──────────────────────────────────────────────────────────────────────────────
def ordinary_krige(
    xy_train: np.ndarray, r_train: np.ndarray, xy_test: np.ndarray,
    nugget: float, sill: float, rng: float, k_neighbors: int = 48,
) -> np.ndarray:
    """Local ordinary kriging of residual field. Returns r_hat at xy_test."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xy_train)
    k = min(k_neighbors, len(xy_train))
    dist, idx = tree.query(xy_test, k=k)
    if k == 1:
        dist = dist[:, None]; idx = idx[:, None]

    out = np.empty(len(xy_test), dtype=np.float64)
    mean_r = float(np.mean(r_train))
    for i in range(len(xy_test)):
        nb = idx[i]
        pxy = xy_train[nb]
        pr = r_train[nb]
        m = len(nb)
        # pairwise distances among neighbours
        dd = np.sqrt(np.sum((pxy[:, None, :] - pxy[None, :, :]) ** 2, axis=2))
        K = spherical_cov(dd, nugget, sill, rng)
        # OK system with Lagrange multiplier
        A = np.ones((m + 1, m + 1))
        A[:m, :m] = K
        A[m, m] = 0.0
        d0 = np.sqrt(np.sum((pxy - xy_test[i]) ** 2, axis=1))
        b = np.ones(m + 1)
        b[:m] = spherical_cov(d0, nugget, sill, rng)
        try:
            w = np.linalg.solve(A + np.eye(m + 1) * 1e-9, b)
            out[i] = float(np.dot(w[:m], pr))
        except np.linalg.LinAlgError:
            # fall back to inverse-distance
            iw = 1.0 / (d0 + 1e-6)
            out[i] = float(np.dot(iw / iw.sum(), pr))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def _decile_slope(pred: np.ndarray, truth: np.ndarray) -> float:
    edges = np.percentile(truth, np.linspace(0, 100, 11))
    dp, dt = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (truth >= lo) & (truth < hi)
        if m.sum() >= 3:
            dp.append(pred[m].mean()); dt.append(truth[m].mean())
    if len(dp) < 3:
        return float("nan")
    dp = np.array(dp); dt = np.array(dt)
    dtc = dt - dt.mean(); dpc = dp - dp.mean()
    den = (dtc ** 2).sum()
    return float((dtc * dpc).sum() / den) if den > 1e-12 else 0.0


def _metrics(pred: np.ndarray, truth: np.ndarray) -> Dict:
    n = len(truth)
    if n == 0:
        return {"n": 0, "rmse": None, "bias": None, "r2": None, "slope": None}
    e = pred - truth
    rmse = float(np.sqrt(np.mean(e ** 2)))
    bias = float(np.mean(e))
    sstot = np.sum((truth - truth.mean()) ** 2)
    r2 = float(1 - np.sum(e ** 2) / max(sstot, 1e-12))
    return {"n": int(n), "rmse": rmse, "bias": bias, "r2": r2,
            "slope": _decile_slope(pred, truth)}


# ──────────────────────────────────────────────────────────────────────────────
# Build CNN preds at all anchor pixels (frozen baseline model)
# ──────────────────────────────────────────────────────────────────────────────
def cnn_predict_anchors(site_key: str, cache_dir: Path):
    """Return (lat, lon, truth, mu_pred) at water anchor pixels for the site."""
    s2 = fetch_s2_for_site(site_key, cache_dir)
    H, W = int(s2["height"]), int(s2["width"])
    bbox = SITE_CFG[site_key]["bbox"]
    lat_t, lon_t, dep_t = load_truth_points(site_key)
    cube, water_mask, names = build_feature_cube(s2)

    w_b, s_b, e_b, n_b = bbox
    rows = np.clip(((n_b - lat_t) / (n_b - s_b) * H).astype(int), 0, H - 1)
    cols = np.clip(((lon_t - w_b) / (e_b - w_b) * W).astype(int), 0, W - 1)
    is_water = water_mask[rows, cols]
    rows, cols, dep = rows[is_water], cols[is_water], dep_t[is_water]
    lat, lon = lat_t[is_water], lon_t[is_water]
    L.info(f"[{site_key}] water anchors: {len(dep)}")

    model, meta = load_default_model()
    # The baseline CNN was trained with POOLED normalisation (per-site norm gives the
    # wrong RMSE — verified: pooled→1.09 m all-anchor, per-site→3.75 m). Use pooled.
    mean_ = np.array(meta["normalisation"]["mean"], dtype=np.float32)
    std_ = np.array(meta["normalisation"]["std"], dtype=np.float32)

    ds = SDBPatchDataset(cube, rows, cols, dep, patch_radius=PATCH_RADIUS,
                         mean=mean_, std=std_)
    mu_pred, _ = _predict_iter1(model, ds)
    return lat, lon, dep.astype(np.float64), mu_pred.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# One buffered run: split → variogram → krige → metrics
# ──────────────────────────────────────────────────────────────────────────────
def run_one(lat, lon, truth, mu_pred, buffer_m: float, n_blocks: int, seed: int,
            k_neighbors: int = 48) -> Dict:
    train_mask, test_mask = make_spatial_block_centers(
        lat, lon, n_blocks=n_blocks, buffer_m=buffer_m, seed=seed)
    xy = latlon_to_local_m(lat, lon)
    r = truth - mu_pred  # CNN residual at anchors

    xy_tr, r_tr = xy[train_mask], r[train_mask]
    xy_te = xy[test_mask]
    truth_te = truth[test_mask]
    mu_te = mu_pred[test_mask]

    # Guard against degenerate splits (too-aggressive buffer starves TRAIN)
    if len(r_tr) < 100 or len(xy_te) < 10:
        L.warning(f"degenerate split buf={buffer_m} seed={seed}: "
                  f"n_train={len(r_tr)} n_test={len(xy_te)} → skip")
        return {
            "buffer_m": buffer_m, "seed": seed,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "variogram": {"nugget": None, "sill": None, "range_m": None},
            "cnn_only": {"overall": {"rmse": None}, "deep_16_22": {"rmse": None}},
            "krige": {"overall": {"rmse": None, "bias": None, "slope": None},
                      "deep_16_22": {"rmse": None}, "per_bin": {}},
            "degenerate": True,
        }

    # variogram on TRAIN residuals
    centres, gamma, npairs = empirical_variogram(xy_tr, r_tr, seed=seed)
    if len(centres) < 3:
        L.warning(f"empty variogram buf={buffer_m} seed={seed} → skip")
        return {
            "buffer_m": buffer_m, "seed": seed,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "variogram": {"nugget": None, "sill": None, "range_m": None},
            "cnn_only": {"overall": {"rmse": None}, "deep_16_22": {"rmse": None}},
            "krige": {"overall": {"rmse": None, "bias": None, "slope": None},
                      "deep_16_22": {"rmse": None}, "per_bin": {}},
            "degenerate": True,
        }
    nugget, sill, vrange = fit_spherical(centres, gamma)

    # ordinary-krige residual onto TEST anchors
    r_hat = ordinary_krige(xy_tr, r_tr, xy_te, nugget, sill, vrange,
                           k_neighbors=k_neighbors)
    corrected = mu_te + r_hat

    res = {
        "buffer_m": buffer_m,
        "seed": seed,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "variogram": {"nugget": nugget, "sill": sill, "range_m": vrange},
        "cnn_only": {
            "overall": _metrics(mu_te, truth_te),
            "deep_16_22": _metrics(mu_te[(truth_te >= DEEP_LO) & (truth_te < DEEP_HI)],
                                   truth_te[(truth_te >= DEEP_LO) & (truth_te < DEEP_HI)]),
        },
        "krige": {
            "overall": _metrics(corrected, truth_te),
            "deep_16_22": _metrics(
                corrected[(truth_te >= DEEP_LO) & (truth_te < DEEP_HI)],
                truth_te[(truth_te >= DEEP_LO) & (truth_te < DEEP_HI)]),
            "per_bin": {},
        },
    }
    for lo, hi in PER2M_BANDS:
        m = (truth_te >= lo) & (truth_te < hi)
        res["krige"]["per_bin"][f"{lo}-{hi}"] = _metrics(corrected[m], truth_te[m])
    return res
