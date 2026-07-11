"""Tide correction via the Open-Meteo Marine API (CMEMS sea-level height).

Vendored from Bathymetry_Production/backend/tide_correction.py. The offline
M2+S2 harmonic fallback is deliberately dropped: on any API failure NO
correction is applied and the product is disclosed as uncorrected
(method="unavailable") instead of guessing site-specific amplitudes.

Depth-grid convention: positive-down water depth (metres). A positive tide
height (sea surface above MSL) is subtracted so the output is referenced to
Mean Sea Level.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import requests

L = logging.getLogger("bathymetry.tide")

OPEN_METEO_URL = "https://marine-api.open-meteo.com/v1/marine"

# In-process cache: (lat_r, lon_r, YYYY-MM-DD) -> list[{time, height}].
_CACHE: Dict[Tuple[float, float, str], list] = {}


def _fetch_day(lat: float, lon: float, date_iso: str) -> list:
    """Fetch hourly sea-level heights for one UTC calendar day."""
    key = (round(lat, 3), round(lon, 3), date_iso)
    if key in _CACHE:
        return _CACHE[key]
    params = {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "hourly": "sea_level_height_msl",
        "start_date": date_iso,
        "end_date": date_iso,
        "timezone": "UTC",
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    hours = r.json().get("hourly", {})
    times = hours.get("time", []) or []
    heights = hours.get("sea_level_height_msl", []) or []
    out = [{"time": t, "height": h}
           for t, h in zip(times, heights) if h is not None]
    _CACHE[key] = out
    return out


def get_tide_height(lat: float, lon: float,
                    utc_dt: Optional[datetime] = None) -> Tuple[float, Dict[str, Any]]:
    """Return (tide_height_m, info) at (lat, lon) for a UTC time; positive when
    the sea surface sits above MSL. On any failure returns
    (0.0, {"method": "unavailable", ...}) — the caller applies no correction."""
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)
    elif utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)

    date_iso = utc_dt.strftime("%Y-%m-%d")
    try:
        series = _fetch_day(lat, lon, date_iso)
        if not series:
            raise RuntimeError("empty series from Open-Meteo")
        xs, ys = [], []
        for row in series:
            try:
                ts = datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                xs.append(ts.timestamp())
                ys.append(float(row["height"]))
            except Exception:
                continue
        if not xs:
            raise RuntimeError("all samples unparseable")
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        h = float(np.interp(utc_dt.timestamp(), xs, ys))
        info = {
            "method": "open-meteo/cmems",
            "source_url": OPEN_METEO_URL,
            "samples_in_day": len(xs),
            "lat": lat, "lon": lon, "utc": utc_dt.isoformat(),
            "tide_range_that_day_m": round(float(np.ptp(ys)), 3),
        }
        L.info("tide %.3f,%.3f @ %s = %+.3f m", lat, lon, utc_dt.isoformat(), h)
        return h, info
    except Exception as ex:
        info = {
            "method": "unavailable",
            "reason": str(ex),
            "lat": lat, "lon": lon, "utc": utc_dt.isoformat(),
        }
        L.warning("tide API unavailable (%s); no correction applied", ex)
        return 0.0, info


def apply_tide_correction(depth_grid: np.ndarray, bbox,
                          utc_dt: Optional[datetime] = None,
                          max_depth_m: float = 25.0):
    """Subtract the tide height from a positive-down depth grid so the result
    is MSL-referenced. Returns (corrected_grid, tide_m, info); when the tide
    is unavailable the grid is returned unchanged and info discloses it.

    depth_grid : (H, W) float array, NaN for land.
    bbox       : [W, S, E, N] EPSG:4326.
    utc_dt     : acquisition time (UTC); None -> now.
    """
    w, s, e, n = bbox
    lat_c = 0.5 * (s + n)
    lon_c = 0.5 * (w + e)
    tide_m, info = get_tide_height(lat_c, lon_c, utc_dt)
    info["centre_lat"] = float(lat_c)
    info["centre_lon"] = float(lon_c)

    corr = np.asarray(depth_grid, dtype=np.float32).copy()
    if info.get("method") != "open-meteo/cmems":
        info["applied_m"] = 0.0
        return corr, 0.0, info

    valid = np.isfinite(corr)
    corr[valid] = np.clip(corr[valid] - tide_m, 0.0, max_depth_m)
    info["applied_m"] = float(tide_m)
    return corr, float(tide_m), info
