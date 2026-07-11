"""
Open-source tide correction via the Open-Meteo Marine Weather API.

Open-Meteo is a free, no-key, no-rate-limit (within reason) API that serves
global ocean sea-level height from the Copernicus Marine Environment Monitoring
Service (CMEMS) and ECMWF models. We use the hourly `sea_level_height_msl`
variable — height of the sea surface above mean sea level (m), positive up.

Why this and not pyTMD/FES2022:
  • pyTMD/FES2022 require downloading multi-hundred-MB NetCDF model files and
    a Copernicus Climate Data Store account. For a lightweight, offline-ready
    production app, a single HTTPS GET to Open-Meteo is far simpler and
    equally accurate at coastal points (Open-Meteo itself wraps CMEMS tide
    and surge products).
  • Fully free (no API key, no signup), CC-BY license.

Endpoint used:
    https://marine-api.open-meteo.com/v1/marine
       ?latitude=<lat>&longitude=<lon>
       &hourly=sea_level_height_msl
       &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
       &timezone=UTC

Public functions
----------------
    get_tide_height(lat, lon, utc_dt)           -> float (metres above MSL)
    apply_tide_correction(depth_grid, bbox, utc_dt) -> (corr_grid, tide_m, info)

Depth-grid convention: positive-down water depth (metres). Tide correction
reduces the predicted depth by the tide height so the output is referenced
to MSL (not instantaneous water surface).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode
from typing import Optional, Tuple, Dict, Any

import numpy as np
import requests

L = logging.getLogger("bathymetry.tide")

OPEN_METEO_URL = "https://marine-api.open-meteo.com/v1/marine"

# In-process cache: key = (lat_r, lon_r, YYYY-MM-DD) → list[{time, height}].
# Keeps us from hitting the API twice for the same site on the same day.
_CACHE: Dict[Tuple[float, float, str], list] = {}


def _fetch_day(lat: float, lon: float, date_iso: str) -> list:
    """Fetch hourly tide heights for one calendar day (UTC).
    Returns list[{time: ISO string, height: metres}]."""
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
    js = r.json()
    hours = js.get("hourly", {})
    times = hours.get("time", []) or []
    heights = hours.get("sea_level_height_msl", []) or []
    out = [{"time": t, "height": h}
           for t, h in zip(times, heights)
           if h is not None]
    _CACHE[key] = out
    return out


def get_wave_height(lat: float, lon: float,
                     utc_dt: Optional[datetime] = None) -> Tuple[float, Dict[str, Any]]:
    """Return (significant_wave_height_m, info) at (lat, lon) for UTC time.

    Uses the same Open-Meteo Marine endpoint. `wave_height` is the significant
    wave height (Hs ≈ mean of the highest third of waves); the rule-of-thumb
    bias correction for SDB is to subtract Hs/2 from the apparent water
    surface so the depth is referenced to still water.
    """
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)
    elif utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    date_iso = utc_dt.strftime("%Y-%m-%d")
    params = {
        "latitude": round(lat, 3), "longitude": round(lon, 3),
        "hourly": "wave_height",
        "start_date": date_iso, "end_date": date_iso,
        "timezone": "UTC",
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        hours = r.json().get("hourly", {})
        times = hours.get("time", []) or []
        heights = hours.get("wave_height", []) or []
        xs, ys = [], []
        for t, h in zip(times, heights):
            if h is None: continue
            try:
                ts = datetime.fromisoformat(t.replace("Z", "+00:00"))
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                xs.append(ts.timestamp()); ys.append(float(h))
            except Exception:
                continue
        if xs:
            target = utc_dt.timestamp()
            h = float(np.interp(target, xs, ys))
            info = {"method": "open-meteo/cmems",
                     "day_max_wave_m": round(float(np.max(ys)), 3),
                     "utc": utc_dt.isoformat()}
            return h, info
    except Exception as ex:
        return 0.0, {"method": "unavailable", "reason": str(ex)}
    return 0.0, {"method": "empty_series"}


def get_tide_height(lat: float, lon: float,
                     utc_dt: Optional[datetime] = None) -> Tuple[float, Dict[str, Any]]:
    """Return (tide_height_m, info) at the given lat/lon and UTC time.
    tide_height_m is positive if the sea surface is above MSL at that moment.

    If utc_dt is None → uses now.
    If the API is unreachable, falls back to a simple M2+S2 harmonic model
    so the pipeline never crashes for a transient network issue.
    """
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)
    elif utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)

    date_iso = utc_dt.strftime("%Y-%m-%d")
    try:
        series = _fetch_day(lat, lon, date_iso)
        if not series:
            raise RuntimeError("empty series from Open-Meteo")
        # Linear-interpolate to the exact hour
        target_ts = utc_dt.timestamp()
        xs, ys = [], []
        for row in series:
            try:
                ts = datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
                if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                xs.append(ts.timestamp()); ys.append(float(row["height"]))
            except Exception:
                continue
        if not xs:
            raise RuntimeError("all samples unparseable")
        xs = np.asarray(xs); ys = np.asarray(ys)
        h = float(np.interp(target_ts, xs, ys))
        info = {
            "method": "open-meteo/cmems",
            "source_url": OPEN_METEO_URL,
            "samples_in_day": len(xs),
            "lat": lat, "lon": lon, "utc": utc_dt.isoformat(),
            "tide_range_that_day_m": round(float(np.ptp(ys)), 3),
        }
        L.info(f"tide {lat:.3f},{lon:.3f} @ {utc_dt.isoformat()} = {h:+.3f} m "
               f"(day range {info['tide_range_that_day_m']:.2f} m)")
        return h, info
    except Exception as ex:
        # Offline fallback: M2+S2 harmonic (same formula as patent_mlp)
        ref = datetime(2000, 1, 1, tzinfo=timezone.utc)
        hours = (utc_dt - ref).total_seconds() / 3600.0
        M2_amp, M2_p = 0.8, 12.4206
        S2_amp, S2_p = 0.3, 12.0
        M2_phase = math.radians(lon)
        h = M2_amp * math.cos(2 * math.pi * hours / M2_p + M2_phase) + \
            S2_amp * math.cos(2 * math.pi * hours / S2_p)
        info = {
            "method": "harmonic_fallback",
            "reason": str(ex),
            "lat": lat, "lon": lon, "utc": utc_dt.isoformat(),
        }
        L.warning(f"tide API unreachable ({ex}); falling back to M2+S2 harmonic "
                  f"→ {h:+.3f} m at {lat:.3f},{lon:.3f}")
        return h, info


def apply_tide_correction(depth_grid: np.ndarray,
                           bbox,
                           utc_dt: Optional[datetime] = None,
                           max_depth_m: float = 25.0):
    """
    Subtract the tide height from the depth grid so the result is referenced
    to Mean Sea Level (MSL).

    depth_grid : (H, W) float array, NaN for land.
    bbox       : [W, S, E, N]  WGS84.
    utc_dt     : acquisition time (UTC). If None → now.
    """
    w, s, e, n = bbox
    lat_c = 0.5 * (s + n); lon_c = 0.5 * (w + e)
    tide_m, info = get_tide_height(lat_c, lon_c, utc_dt)

    corr = depth_grid.copy().astype(np.float32)
    valid = np.isfinite(corr)
    # Tide > 0 means the water surface sits above MSL, so the satellite saw
    # the sea bed from a higher datum → the raw depth over-estimates the
    # MSL-referenced depth by tide_m. Subtract.
    corr[valid] = np.clip(corr[valid] - tide_m, 0.0, max_depth_m)
    info["applied_m"] = float(tide_m)
    info["centre_lat"] = float(lat_c)
    info["centre_lon"] = float(lon_c)
    return corr, float(tide_m), info
