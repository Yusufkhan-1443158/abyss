"""
Anomaly detection on SST time series for the Strait of Hormuz.

Objective: detect persistent, localized thermal anomalies that could indicate
underwater objects (mines, infrastructure, submarine thermal wakes).

Methods implemented (from remote sensing / oceanographic literature):

1. PIXEL-WISE Z-SCORE (Emery & Thomson, 2001 - "Data Analysis Methods in Physical Oceanography")
   ─────────────────
   For each pixel (i,j), compute:
     μ(i,j) = temporal mean of SST over N days
     σ(i,j) = temporal std of SST over N days
     z(i,j,t) = (SST(i,j,t) - μ(i,j)) / σ(i,j)

   Pixels with |z| > threshold are anomalous at time t.
   Natural ocean variability has σ ≈ 0.3-0.8°C. A mine or submarine wake
   creates a small but persistent signal (0.05-0.2°C) that stands out
   when σ_local is low (calm, stratified waters).

2. PERSISTENT ANOMALY SCORE (inspired by Merchant et al., 2014 - SST CCI)
   ─────────────────────────
   Count how many timesteps each pixel is anomalous (|z| > 2):
     P(i,j) = Σ_t 𝟙[|z(i,j,t)| > 2]

   Natural anomalies (eddies, fronts) are transient and move spatially.
   A fixed underwater object creates a STATIONARY anomaly → high P score.
   Normalize: P_norm = P / N_timesteps.

3. ROBUST PCA / BACKGROUND SUBTRACTION (Candès et al., 2011 - "Robust PCA")
   ──────────────────────────────────────
   Decompose the SST spatiotemporal matrix X (time × pixels) as:
     X = L + S
   where L = low-rank (smooth ocean background) and S = sparse (anomalies).

   Simplified implementation using truncated SVD:
     L = U_k Σ_k V_k^T  (keep top k singular values, typically k=3-5)
     S = X - L           (residual = anomalies)

   The sparse component S captures localized signals that don't fit
   the dominant ocean patterns (seasonal cycle, large-scale advection).

4. SPATIAL ISOLATION SCORE (Reed & Marks, 1999 - "Anomaly detection in hyperspectral")
   ─────────────────────────
   Natural ocean features are spatially coherent (eddies ~10-100 km).
   A mine creates a point-source anomaly (~1-10 pixels).

   For each pixel, compare its anomaly to its spatial neighborhood:
     I(i,j) = |S(i,j) - median(S in r-neighborhood)| / MAD(S in r-neighborhood)

   High I = the pixel is anomalous compared to its surroundings = isolated signal.

5. COMBINED DETECTION SCORE
   ─────────────────────────
   D(i,j) = w1 * |z_mean| + w2 * P_norm + w3 * |S_mean| + w4 * I

   Threshold at D > τ to flag potential detection sites.
   Weights tuned for sensitivity vs. false alarm trade-off.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import xarray as xr
import json
import os

from .land_mask import make_land_mask

HYCOM_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"


def fetch_monthly_sst(bbox, n_days=30, progress_cb=None):
    """
    Fetch daily SST snapshots from HYCOM for the last n_days.
    Returns: sst_cube (n_days, lat, lon), lats, lons
    """
    ds = xr.open_dataset(HYCOM_URL, engine="netcdf4", decode_times=False)
    sub = ds.sel(
        lon=slice(bbox["lon_min"], bbox["lon_max"]),
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
    ).sel(depth=0, method="nearest")

    lats = sub.lat.values.astype(np.float64)
    lons = sub.lon.values.astype(np.float64)

    # Fetch daily (every 8th 3-hourly step)
    step = 8
    indices = list(range(-n_days * step, 0, step))

    sst_cube = np.zeros((n_days, len(lats), len(lons)), dtype=np.float64)
    time_values = []

    for idx_out, idx_in in enumerate(indices):
        snapshot = sub.isel(time=idx_in)
        sst_cube[idx_out] = snapshot["water_temp"].values.astype(np.float64)
        time_values.append(float(snapshot.time.values))
        if progress_cb:
            progress_cb(idx_out + 1, n_days)

    ds.close()
    return sst_cube, lats, lons, time_values


def detect_anomalies(bbox, n_days=30, superres_factor=4):
    """
    Run the full anomaly detection pipeline.
    Returns results dict with all layers + detection metadata.
    """
    print(f"[DETECT] Fetching {n_days} days of HYCOM SST...")
    sst_cube, lats, lons, times = fetch_monthly_sst(
        bbox, n_days,
        progress_cb=lambda i, n: print(f"  [{i}/{n}]") if i % 5 == 0 else None,
    )
    ny, nx = len(lats), len(lons)
    n_t = sst_cube.shape[0]

    # Ocean mask from HYCOM (NaN = land across all timesteps)
    ocean_mask = np.all(~np.isnan(sst_cube), axis=0)
    # Also exclude land pixels identified by polygon coastline mask
    poly_land = make_land_mask(lons, lats)
    ocean_mask = ocean_mask & ~poly_land
    print(f"[DETECT] Grid: {ny}x{nx}, ocean pixels: {ocean_mask.sum()}/{ny*nx}")

    # ─── METHOD 1: Pixel-wise Z-score ───────────────────────
    print("[DETECT] Computing z-scores...")
    sst_mean = np.nanmean(sst_cube, axis=0)
    sst_std = np.nanstd(sst_cube, axis=0)
    sst_std[sst_std < 0.01] = 0.01  # avoid division by zero

    z_scores = (sst_cube - sst_mean[np.newaxis]) / sst_std[np.newaxis]

    # Mean absolute z-score over time (persistent deviations)
    z_mean_abs = np.nanmean(np.abs(z_scores), axis=0)
    # Latest z-score
    z_latest = z_scores[-1]

    # ─── METHOD 2: Persistence score ────────────────────────
    print("[DETECT] Computing persistence...")
    threshold_z = 2.0
    anomalous_count = np.nansum(np.abs(z_scores) > threshold_z, axis=0)
    persistence = anomalous_count / n_t  # fraction of time anomalous

    # ─── METHOD 3: Robust PCA (truncated SVD) ───────────────
    print("[DETECT] Running Robust PCA (SVD)...")
    # Reshape to (time, pixels) — only ocean pixels
    ocean_indices = np.where(ocean_mask.ravel())[0]
    X = sst_cube.reshape(n_t, -1)[:, ocean_indices]  # (n_t, n_ocean)

    # Fill any remaining NaN with column mean
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask_nan = np.isnan(X[:, j])
        X[mask_nan, j] = col_means[j]

    # Center
    X_centered = X - X.mean(axis=0)

    # Truncated SVD — keep top k components (background)
    k = min(5, n_t - 1)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    L = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]  # low-rank background
    Sparse = X_centered - L  # anomaly residual

    # Map back to spatial grid
    sparse_full = np.full((n_t, ny * nx), np.nan)
    sparse_full[:, ocean_indices] = Sparse

    sparse_map = sparse_full.reshape(n_t, ny, nx)
    sparse_mean_abs = np.nanmean(np.abs(sparse_map), axis=0)
    sparse_latest = sparse_map[-1]

    # Explained variance by first k components
    total_var = np.sum(S ** 2)
    explained_var = np.sum(S[:k] ** 2) / total_var
    print(f"[DETECT] SVD: {k} components explain {explained_var:.1%} of variance")

    # ─── METHOD 4: Spatial isolation ────────────────────────
    print("[DETECT] Computing spatial isolation...")
    # Use sparse_mean_abs as the anomaly field
    field = sparse_mean_abs.copy()
    field[~ocean_mask] = 0

    # Local median and MAD in 5x5 neighborhood
    local_median = median_filter(field, size=7)
    local_mad = median_filter(np.abs(field - local_median), size=7)
    local_mad[local_mad < 0.001] = 0.001

    isolation = np.abs(field - local_median) / local_mad
    isolation[~ocean_mask] = 0

    # ─── METHOD 5: Combined detection score ─────────────────
    print("[DETECT] Computing combined score...")
    # Normalize each component to [0, 1]
    def normalize(arr, mask):
        vals = arr[mask]
        if vals.max() == vals.min():
            return np.zeros_like(arr)
        out = (arr - vals.min()) / (vals.max() - vals.min())
        out[~mask] = 0
        return out

    n_z = normalize(z_mean_abs, ocean_mask)
    n_p = normalize(persistence, ocean_mask)
    n_s = normalize(sparse_mean_abs, ocean_mask)
    n_i = normalize(isolation, ocean_mask)

    # Weights: persistence and isolation are most diagnostic for fixed objects
    w = {"z_score": 0.15, "persistence": 0.35, "sparse_pca": 0.20, "isolation": 0.30}
    detection_score = (
        w["z_score"] * n_z +
        w["persistence"] * n_p +
        w["sparse_pca"] * n_s +
        w["isolation"] * n_i
    )
    detection_score[~ocean_mask] = 0

    # ─── Flag detections ────────────────────────────────────
    # Adaptive threshold: mean + 3*std of detection score
    det_vals = detection_score[ocean_mask]
    det_threshold = float(np.mean(det_vals) + 3 * np.std(det_vals))
    detected = detection_score > det_threshold

    # Cluster nearby detections
    detections = _extract_detections(
        detection_score, detected, lats, lons, ocean_mask,
        z_mean_abs, persistence, sparse_mean_abs, isolation, sst_mean, sst_std,
    )
    print(f"[DETECT] Found {len(detections)} detection sites (threshold={det_threshold:.3f})")

    # ─── Super-resolve for visualization ────────────────────
    print("[DETECT] Super-resolving...")
    layers = {
        "detection_score": _superres(detection_score, lats, lons, ocean_mask, bbox, superres_factor),
        "z_mean": _superres(z_mean_abs, lats, lons, ocean_mask, bbox, superres_factor),
        "persistence": _superres(persistence, lats, lons, ocean_mask, bbox, superres_factor),
        "sparse_pca": _superres(sparse_mean_abs, lats, lons, ocean_mask, bbox, superres_factor),
        "isolation": _superres(isolation, lats, lons, ocean_mask, bbox, superres_factor),
    }

    print("[DETECT] Done.")
    return {
        "layers": layers,
        "detections": detections,
        "metadata": {
            "n_days": n_days,
            "n_timesteps": n_t,
            "grid_raw": f"{ny}x{nx}",
            "grid_superres": f"{ny*superres_factor}x{nx*superres_factor}",
            "svd_components": k,
            "svd_explained_variance": round(explained_var, 4),
            "detection_threshold": round(det_threshold, 4),
            "n_detections": len(detections),
            "weights": w,
            "sst_mean_range": [round(float(np.nanmin(sst_mean)), 2),
                               round(float(np.nanmax(sst_mean)), 2)],
            "sst_std_range": [round(float(np.nanmin(sst_std[ocean_mask])), 3),
                              round(float(np.nanmax(sst_std[ocean_mask])), 3)],
        },
    }


def _extract_detections(score, detected, lats, lons, ocean_mask,
                        z_mean, persistence, sparse, isolation, sst_mean, sst_std):
    """Extract and cluster detection points into discrete sites."""
    from scipy.ndimage import label

    labeled, n_clusters = label(detected)
    detections = []

    for c in range(1, n_clusters + 1):
        cluster_mask = labeled == c
        cluster_scores = score[cluster_mask]

        # Peak location
        peak_idx = np.argmax(cluster_scores)
        rows, cols = np.where(cluster_mask)
        peak_row, peak_col = rows[peak_idx], cols[peak_idx]

        lat = float(lats[peak_row])
        lon = float(lons[peak_col])

        detections.append({
            "id": c,
            "lat": lat,
            "lon": lon,
            "score": round(float(score[peak_row, peak_col]), 4),
            "n_pixels": int(cluster_mask.sum()),
            "components": {
                "z_score": round(float(z_mean[peak_row, peak_col]), 3),
                "persistence": round(float(persistence[peak_row, peak_col]), 3),
                "sparse_pca": round(float(sparse[peak_row, peak_col]), 4),
                "isolation": round(float(isolation[peak_row, peak_col]), 3),
            },
            "sst_mean": round(float(sst_mean[peak_row, peak_col]), 2),
            "sst_std": round(float(sst_std[peak_row, peak_col]), 3),
        })

    # Sort by score descending
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def _superres(field, lats, lons, ocean_mask, bbox, factor):
    """Bicubic super-resolution + land masking, return as JSON matrix."""
    from scipy.ndimage import distance_transform_edt

    # Fill NaN for interpolation
    filled = field.copy()
    nan_mask = ~ocean_mask | np.isnan(filled)
    if nan_mask.any():
        ind = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        filled = filled[tuple(ind)]

    spline = RectBivariateSpline(lats, lons, filled, kx=3, ky=3)

    n_lat_hr = len(lats) * factor
    n_lon_hr = len(lons) * factor
    lats_hr = np.linspace(lats[0], lats[-1], n_lat_hr)
    lons_hr = np.linspace(lons[0], lons[-1], n_lon_hr)
    hr_field = spline(lats_hr, lons_hr)

    # Mask
    mask_interp = RegularGridInterpolator(
        (lats, lons), ocean_mask.astype(np.float64),
        method="nearest", bounds_error=False, fill_value=0
    )
    lon_hr_grid, lat_hr_grid = np.meshgrid(lons_hr, lats_hr)
    ocean_hr = mask_interp((lat_hr_grid, lon_hr_grid)) > 0.5
    poly_land = make_land_mask(lons_hr, lats_hr)
    hr_field[~ocean_hr] = np.nan
    hr_field[poly_land] = np.nan

    ny, nx = hr_field.shape
    matrix = []
    for i in range(ny):
        row = []
        for j in range(nx):
            v = hr_field[i, j]
            row.append(None if np.isnan(v) else round(float(v), 4))
        matrix.append(row)

    sea_vals = hr_field[~np.isnan(hr_field)]
    return {
        "lats": lats_hr.tolist(),
        "lons": lons_hr.tolist(),
        "matrix": matrix,
        "ny": ny,
        "nx": nx,
        "min_val": round(float(sea_vals.min()), 4) if len(sea_vals) > 0 else 0,
        "max_val": round(float(sea_vals.max()), 4) if len(sea_vals) > 0 else 0,
    }
