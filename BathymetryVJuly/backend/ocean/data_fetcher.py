"""
Data fetching module — real HYCOM data via OPeNDAP.

Sources:
  - HYCOM GLBy0.08/expt_93.0: currents (water_u, water_v), SST (water_temp)
  - ~0.08° resolution, 3-hourly
  - Super-resolution via bicubic interpolation for SST
  - Anomaly = latest - mean of recent N timesteps
"""

import json
import os
import numpy as np
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from .land_mask import make_land_mask

HYCOM_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"


# ---------------------------------------------------------------------------
# HYCOM fetch
# ---------------------------------------------------------------------------

def _open_hycom():
    """Open HYCOM dataset via OPeNDAP."""
    return xr.open_dataset(HYCOM_URL, engine="netcdf4", decode_times=False)


def fetch_hycom_currents(date_str, depth, bbox):
    """
    Fetch real U/V current components from HYCOM.
    Returns leaflet-velocity compatible JSON.
    """
    if not HAS_XARRAY:
        raise RuntimeError("xarray + netCDF4 required for real data")

    ds = _open_hycom()
    sub = ds.sel(
        lon=slice(bbox["lon_min"], bbox["lon_max"]),
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
    )
    sub = sub.sel(depth=depth, method="nearest")
    latest = sub.isel(time=-1)

    u = latest["water_u"].values.astype(np.float32)
    v = latest["water_v"].values.astype(np.float32)
    lats = latest.lat.values.astype(np.float32)
    lons = latest.lon.values.astype(np.float32)

    ds.close()

    # NaN → 0 for velocity JSON (NaN = land in HYCOM)
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)

    return _format_velocity_json(u, v, lats, lons)


def fetch_hycom_sst(date_str, bbox, superres_factor=4):
    """
    Fetch real SST from HYCOM, then super-resolve with bicubic interpolation.
    Returns grid with null on land.
    """
    if not HAS_XARRAY:
        raise RuntimeError("xarray + netCDF4 required for real data")

    ds = _open_hycom()
    sub = ds.sel(
        lon=slice(bbox["lon_min"], bbox["lon_max"]),
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
    )
    latest = sub.sel(depth=0, method="nearest").isel(time=-1)

    sst_raw = latest["water_temp"].values.astype(np.float64)
    lats_raw = latest.lat.values.astype(np.float64)
    lons_raw = latest.lon.values.astype(np.float64)

    ds.close()

    # Build ocean mask from HYCOM NaN (true land mask from data)
    ocean_mask_raw = ~np.isnan(sst_raw)

    # --- Super-resolution via bicubic interpolation ---
    # Fill NaN temporarily for interpolation (use nearest-neighbor fill)
    sst_filled = _fill_nan_nearest(sst_raw)

    # Bicubic spline on the filled grid
    spline = RectBivariateSpline(lats_raw, lons_raw, sst_filled, kx=3, ky=3)

    # High-res grid
    n_lat_hr = len(lats_raw) * superres_factor
    n_lon_hr = len(lons_raw) * superres_factor
    lats_hr = np.linspace(lats_raw[0], lats_raw[-1], n_lat_hr)
    lons_hr = np.linspace(lons_raw[0], lons_raw[-1], n_lon_hr)
    sst_hr = spline(lats_hr, lons_hr)

    # Build high-res land mask (interpolate ocean mask)
    from scipy.interpolate import RegularGridInterpolator
    mask_interp = RegularGridInterpolator(
        (lats_raw, lons_raw), ocean_mask_raw.astype(np.float64),
        method="nearest", bounds_error=False, fill_value=0
    )
    lon_hr_grid, lat_hr_grid = np.meshgrid(lons_hr, lats_hr)
    ocean_mask_hr = mask_interp((lat_hr_grid, lon_hr_grid)) > 0.5

    # Apply mask
    sst_hr[~ocean_mask_hr] = np.nan

    # Also apply our polygon land mask for clean edges
    poly_land = make_land_mask(lons_hr, lats_hr)
    sst_hr[poly_land] = np.nan

    # Build output
    ny, nx = sst_hr.shape
    sst_matrix = []
    for i in range(ny):
        row = []
        for j in range(nx):
            v = sst_hr[i, j]
            row.append(None if np.isnan(v) else round(float(v), 2))
        sst_matrix.append(row)

    sea_vals = sst_hr[~np.isnan(sst_hr)]

    return {
        "lats": lats_hr.tolist(),
        "lons": lons_hr.tolist(),
        "sst_matrix": sst_matrix,
        "ny": ny,
        "nx": nx,
        "min_sst": round(float(sea_vals.min()), 2),
        "max_sst": round(float(sea_vals.max()), 2),
        "resolution_km": round(float((lats_hr[1] - lats_hr[0]) * 111), 2),
        "source": "HYCOM GLBy0.08 + bicubic 4x super-resolution",
    }


def fetch_sst_anomaly(bbox, n_steps=8, superres_factor=4):
    """
    Compute SST anomaly: latest SST minus mean of last N time steps.
    Positive anomaly = warmer than recent mean (possible thermal wake).
    Returns high-res anomaly grid.
    """
    if not HAS_XARRAY:
        raise RuntimeError("xarray + netCDF4 required")

    ds = _open_hycom()
    sub = ds.sel(
        lon=slice(bbox["lon_min"], bbox["lon_max"]),
        lat=slice(bbox["lat_min"], bbox["lat_max"]),
    )
    sub = sub.sel(depth=0, method="nearest")

    # Get last N+1 time steps
    recent = sub.isel(time=slice(-n_steps - 1, None))
    sst_all = recent["water_temp"].values.astype(np.float64)  # (time, lat, lon)
    lats_raw = recent.lat.values.astype(np.float64)
    lons_raw = recent.lon.values.astype(np.float64)

    ds.close()

    # Latest and mean of previous N
    sst_latest = sst_all[-1]
    sst_mean = np.nanmean(sst_all[:-1], axis=0)
    anomaly_raw = sst_latest - sst_mean

    ocean_mask_raw = ~np.isnan(sst_latest)

    # Super-resolve anomaly
    anomaly_filled = _fill_nan_nearest(anomaly_raw)
    spline = RectBivariateSpline(lats_raw, lons_raw, anomaly_filled, kx=3, ky=3)

    n_lat_hr = len(lats_raw) * superres_factor
    n_lon_hr = len(lons_raw) * superres_factor
    lats_hr = np.linspace(lats_raw[0], lats_raw[-1], n_lat_hr)
    lons_hr = np.linspace(lons_raw[0], lons_raw[-1], n_lon_hr)
    anomaly_hr = spline(lats_hr, lons_hr)

    # Smooth slightly to reduce spline ringing
    anomaly_hr = gaussian_filter(anomaly_hr, sigma=1.0)

    # Masks
    from scipy.interpolate import RegularGridInterpolator
    mask_interp = RegularGridInterpolator(
        (lats_raw, lons_raw), ocean_mask_raw.astype(np.float64),
        method="nearest", bounds_error=False, fill_value=0
    )
    lon_hr_grid, lat_hr_grid = np.meshgrid(lons_hr, lats_hr)
    ocean_mask_hr = mask_interp((lat_hr_grid, lon_hr_grid)) > 0.5

    poly_land = make_land_mask(lons_hr, lats_hr)
    anomaly_hr[~ocean_mask_hr] = np.nan
    anomaly_hr[poly_land] = np.nan

    ny, nx = anomaly_hr.shape
    matrix = []
    for i in range(ny):
        row = []
        for j in range(nx):
            v = anomaly_hr[i, j]
            row.append(None if np.isnan(v) else round(float(v), 3))
        matrix.append(row)

    sea_vals = anomaly_hr[~np.isnan(anomaly_hr)]

    return {
        "lats": lats_hr.tolist(),
        "lons": lons_hr.tolist(),
        "anomaly_matrix": matrix,
        "ny": ny,
        "nx": nx,
        "min_val": round(float(sea_vals.min()), 3),
        "max_val": round(float(sea_vals.max()), 3),
        "std_val": round(float(sea_vals.std()), 3),
        "mean_val": round(float(sea_vals.mean()), 3),
        "n_baseline_steps": n_steps,
        "source": "HYCOM SST anomaly (latest - mean of last 24h)",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_nan_nearest(arr):
    """Fill NaN values with nearest non-NaN neighbor."""
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(arr)
    if not mask.any():
        return arr.copy()
    ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(ind)]


def _format_velocity_json(u, v, lats, lons):
    """Format into leaflet-velocity JSON (two header+data blocks, N→S order)."""
    ny, nx = u.shape
    lo1, lo2 = float(lons[0]), float(lons[-1])
    la1, la2 = float(lats[-1]), float(lats[0])  # N→S
    dx = float((lons[-1] - lons[0]) / max(nx - 1, 1))
    dy = float((lats[-1] - lats[0]) / max(ny - 1, 1))

    u_flipped = np.flipud(u)
    v_flipped = np.flipud(v)

    return [
        {
            "header": {
                "parameterCategory": 2,
                "parameterNumber": 2,
                "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
                "dx": abs(dx), "dy": abs(dy),
                "nx": nx, "ny": ny,
                "refTime": datetime.now().isoformat(),
            },
            "data": u_flipped.flatten().tolist(),
        },
        {
            "header": {
                "parameterCategory": 2,
                "parameterNumber": 3,
                "lo1": lo1, "la1": la1, "lo2": lo2, "la2": la2,
                "dx": abs(dx), "dy": abs(dy),
                "nx": nx, "ny": ny,
                "refTime": datetime.now().isoformat(),
            },
            "data": v_flipped.flatten().tolist(),
        },
    ]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def get_cached_or_fetch(data_type, date_str, depth, bbox, data_dir):
    """Check cache, then fetch from HYCOM."""
    cache_key = f"{data_type}_{date_str}_{int(depth)}"
    cache_file = os.path.join(data_dir, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    if data_type == "currents":
        data = fetch_hycom_currents(date_str, depth, bbox)
    elif data_type == "sst":
        data = fetch_hycom_sst(date_str, bbox)
    elif data_type == "sst_anomaly":
        data = fetch_sst_anomaly(bbox)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    with open(cache_file, "w") as f:
        json.dump(data, f)

    return data
