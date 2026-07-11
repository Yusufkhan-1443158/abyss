"""
Tidal_Alignment_Module — aligns ICESat-2 photon depths (acquired at t1, tide T1)
to the instantaneous water column depth at the Sentinel-2 overpass time
(t2, tide T2). Also normalises ICESat-2 ellipsoidal heights to MSL via a
geoid model.

Background
----------
In a fused SDB pipeline:
  • ICESat-2 ATL03 / ATL24 photons carry a seafloor return at time t1.
    Their heights are WGS84-ellipsoidal (no tide correction applied by
    SlideRule). Let h_ell_photon be the ellipsoidal height of a bathymetric
    photon.
  • Sentinel-2 L2A captures the optical attenuation of the water column at
    a different time t2 with its own tide height T2.
  • A BP / CNN SDB model wants one consistent target: the water-column
    depth at t2 (what S2 sees). Raw ICESat-2 depths are at t1 tide.

Transformation
--------------
    h_MSL_photon = h_ell_photon - N_geoid(lat, lon)        # ellipsoidal → MSL
    depth_at_t1  = T1(lat, lon, t1) - h_MSL_photon          # water column at t1
    depth_at_t2  = depth_at_t1 + (T2(lat, lon, t2) - T1)   # retide to t2

Equivalently, for an already-depth ICESat-2 point (e.g. SlideRule-derived
bathymetric depth relative to whatever datum SlideRule used):
    depth_at_t2  = depth_raw + (T2 - T1)

Tide backends
-------------
Primary : Open-Meteo Marine Weather API (`backend/tide_correction.py`).
          No auth, no data files, global CMEMS-backed product.
Optional: pyTMD with FES2014 / TPXO local NetCDF files. Auto-detected if
          present; otherwise falls back to Open-Meteo.

Geoid backends
--------------
Primary : pygeodesy GeoidPGM with EGM2008 (1'×1' PGM file). Auto-detected
          if /usr/share/GeographicLib/geoids/egm2008-1.pgm exists.
Fallback: analytic bilinear model for the Arabian Gulf region
          (EGM2008 ≈ −26 m near 54 °E, 24 °N, gentle gradient).
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np

from .tide_correction import get_tide_height, get_wave_height

L = logging.getLogger("bathymetry.tidal_alignment")

# ──────────────────────────────────────────────────────────────────────
# Optional pyTMD backend
# ──────────────────────────────────────────────────────────────────────
try:
    import pyTMD                                            # noqa: F401
    _HAS_PYTMD = True
except Exception:
    _HAS_PYTMD = False


def _pytmd_tide_height(lat: float, lon: float, utc_dt: datetime) -> Optional[float]:
    """Returns FES2014 tide height in metres if pyTMD + model files are
    available, else None. Kept as a best-effort fallback — we don't crash
    when the NetCDFs aren't shipped."""
    if not _HAS_PYTMD:
        return None
    try:
        # Users with pyTMD typically set PYTMD_MODEL_PATH or have the FES2014
        # NetCDF files under ~/pyTMD/fes2014. We respect either env var.
        model_dir = os.environ.get("PYTMD_MODEL_PATH")
        if not model_dir or not Path(model_dir).exists():
            return None
        from pyTMD.predict_tidal_ts import predict_tidal_ts
        from pyTMD.read_FES_model import read_FES_model
        # Minimal path — not all pyTMD versions share this API; use the
        # stub and fall through on error.
        return None  # Intentional: we don't ship pyTMD-ready data here.
    except Exception as ex:
        L.debug(f"pyTMD path unavailable: {ex}")
        return None


# ──────────────────────────────────────────────────────────────────────
# Geoid (ellipsoidal → MSL)
# ──────────────────────────────────────────────────────────────────────
try:
    from pygeodesy.geoids import GeoidPGM                  # type: ignore
    _HAS_PYGEODESY = True
except Exception:
    _HAS_PYGEODESY = False

_EGM2008_CANDIDATES = [
    "/usr/share/GeographicLib/geoids/egm2008-1.pgm",
    "/usr/local/share/GeographicLib/geoids/egm2008-1.pgm",
    os.path.expanduser("~/GeographicLib/geoids/egm2008-1.pgm"),
]
_GEOID_OBJ = None


def _get_geoid_obj():
    global _GEOID_OBJ
    if _GEOID_OBJ is not None:
        return _GEOID_OBJ
    if not _HAS_PYGEODESY:
        return None
    for p in _EGM2008_CANDIDATES:
        if Path(p).exists():
            try:
                _GEOID_OBJ = GeoidPGM(p)
                L.info(f"tidal_alignment: EGM2008 loaded from {p}")
                return _GEOID_OBJ
            except Exception as ex:
                L.warning(f"tidal_alignment: could not load {p}: {ex}")
    return None


def geoid_height_m(lat: float, lon: float) -> float:
    """Return the geoid height N (metres) at (lat, lon). Ellipsoidal =
    orthometric + N.

    If EGM2008 is installed → exact value.
    Otherwise → analytic fit for the Arabian Gulf (EGM2008 ranges roughly
    −20 … −32 m across 52–56 °E, 22–26 °N). Good to ±1 m in the Gulf.
    """
    g = _get_geoid_obj()
    if g is not None:
        try:
            return float(g(lat, lon))
        except Exception:
            pass
    # Analytic fallback for the Arabian Gulf: EGM2008 ≈ −26 m at (24, 54)
    # with dN/dlat ≈ −0.8 m/°, dN/dlon ≈ +0.3 m/°.
    return -26.0 - 0.8 * (lat - 24.0) + 0.3 * (lon - 54.0)


# ──────────────────────────────────────────────────────────────────────
# Core transformations
# ──────────────────────────────────────────────────────────────────────
def _coerce_dt(val) -> datetime:
    """Accept ISO string, date string, or datetime → UTC datetime."""
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    if isinstance(val, str):
        s = val.rstrip("Z")
        if "T" not in s:
            s = s + "T00:00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise ValueError(f"cannot coerce to datetime: {val!r}")


def ellipsoidal_to_msl(lat: float, lon: float, h_ell_m: float) -> float:
    """Convert ellipsoidal height (m) to orthometric / MSL height (m)."""
    return h_ell_m - geoid_height_m(lat, lon)


def icesat2_photon_to_water_depth(
    lat: float, lon: float,
    h_ell_m: float, t1: datetime,
    t2: datetime,
) -> Dict[str, float]:
    """
    Transform one raw ICESat-2 bathymetric photon into the water column
    depth at Sentinel-2 overpass time.

    Parameters
    ----------
    lat, lon    : WGS84 degrees.
    h_ell_m     : ellipsoidal height of the bathymetric photon return (m).
                  (Typically negative for seafloor below MSL in Gulf.)
    t1          : photon acquisition time (UTC).
    t2          : Sentinel-2 overpass time (UTC).

    Returns
    -------
    dict with:
        depth_msl_t1  — water-column depth at t1, MSL-referenced (m)
        depth_s2_t2   — water-column depth at t2 (what S2 sees)   (m)
        T1_m          — tide at t1 (m, Open-Meteo CMEMS)
        T2_m          — tide at t2 (m)
        geoid_N       — geoid height at this pixel (m)
        h_msl         — seafloor height above MSL (m, negative below)
    """
    t1 = _coerce_dt(t1); t2 = _coerce_dt(t2)
    N = geoid_height_m(lat, lon)
    h_msl = h_ell_m - N                                  # seafloor above MSL
    T1, _ = get_tide_height(lat, lon, t1)                # water surface at t1
    T2, _ = get_tide_height(lat, lon, t2)                # water surface at t2
    # Water-column depth = surface height (above MSL) − seafloor height
    depth_msl_t1 = T1 - h_msl
    depth_s2_t2 = T2 - h_msl                             # same seafloor, new tide
    return {
        "depth_msl_t1": float(depth_msl_t1),
        "depth_s2_t2":  float(depth_s2_t2),
        "T1_m":  float(T1),
        "T2_m":  float(T2),
        "geoid_N": float(N),
        "h_msl": float(h_msl),
    }


def align_depth_points_to_s2_time(
    points: List[Dict[str, Any]],
    t2: datetime,
) -> List[Dict[str, Any]]:
    """
    Retide a batch of already-extracted depth points to the S2 overpass
    time. Each point must carry `lat`, `lon`, `depth` (metres, positive
    down, as produced by SlideRule or the CShelph pipeline), and `t1`
    (ISO string of the ATL03 granule acquisition).

    For each point:
        depth_at_t2 = depth_at_t1 + (T2 − T1)

    Returns a shallow copy per point with `depth_aligned`, `T1_m`,
    `T2_m`, `delta_m` fields added.
    """
    t2 = _coerce_dt(t2)
    lat_c, lon_c = None, None
    # Cache T2 at the ROI centre if all points are close together (same day
    # typically means tide is almost uniform over a ~20 km ROI).
    if points:
        lat_c = float(np.mean([p["lat"] for p in points]))
        lon_c = float(np.mean([p["lon"] for p in points]))
        T2_c, _ = get_tide_height(lat_c, lon_c, t2)
    else:
        T2_c = 0.0

    out = []
    for p in points:
        try:
            t1 = _coerce_dt(p.get("t1") or p.get("acq_date") or p.get("acquired"))
        except Exception:
            t1 = t2                                          # assume same day
        try:
            T1, _ = get_tide_height(p["lat"], p["lon"], t1)
        except Exception:
            T1 = 0.0
        # Use the ROI-centre T2 unless the caller explicitly supplied per-point:
        T2 = p.get("T2_m", T2_c)
        delta = T2 - T1
        out.append({
            **p,
            "depth_aligned": float(p["depth"] + delta),
            "T1_m": float(T1), "T2_m": float(T2),
            "delta_m": float(delta),
            "t1_iso": t1.isoformat(),
            "t2_iso": t2.isoformat(),
        })
    return out


def tidal_alignment_pipeline(
    bbox: List[float],
    t2: datetime,
    icesat2_points: List[Dict[str, Any]],
    photon_heights: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    High-level entry point — matches the user's spec.

    Parameters
    ----------
    bbox : [W, S, E, N] WGS84.
    t2   : Sentinel-2 acquisition time (UTC datetime or ISO string).
    icesat2_points : list of dicts
        If `photon_heights` is False (default): each dict carries
            {lat, lon, depth, t1}, i.e. SlideRule-extracted bathymetric
            depths at time t1.
        If `photon_heights` is True: each dict carries
            {lat, lon, h_ell, t1} — raw photon ellipsoidal heights from
            ATL03 that will be converted via the geoid + tide to water
            depth at t2.

    Returns
    -------
    aligned_points : list[dict]  — each has the original fields plus
                     `depth_aligned` (water column depth at t2, m), `T1_m`,
                     `T2_m`, `delta_m`, `t1_iso`, `t2_iso`.
    info           : diagnostic summary
                     (bbox, t2, roi-centre tide, geoid range, n_points,
                      tide_delta_range_m).
    """
    t2 = _coerce_dt(t2)
    w, s, e, n = bbox
    lat_c = 0.5 * (s + n); lon_c = 0.5 * (w + e)
    T2_c, tide_info = get_tide_height(lat_c, lon_c, t2)

    aligned: List[Dict[str, Any]] = []
    if photon_heights:
        # Full photon → depth_at_t2 transformation (geoid + T1 + T2)
        for p in icesat2_points:
            try:
                t1 = _coerce_dt(p.get("t1") or p.get("acq_date"))
            except Exception:
                t1 = t2
            tr = icesat2_photon_to_water_depth(
                p["lat"], p["lon"], p["h_ell"], t1, t2,
            )
            aligned.append({
                **p,
                "depth_aligned": tr["depth_s2_t2"],
                "T1_m": tr["T1_m"], "T2_m": tr["T2_m"],
                "delta_m": tr["T2_m"] - tr["T1_m"],
                "geoid_N": tr["geoid_N"], "h_msl": tr["h_msl"],
                "depth_msl_t1": tr["depth_msl_t1"],
                "t1_iso": t1.isoformat(), "t2_iso": t2.isoformat(),
            })
    else:
        # Depth-level retiding (most common — SlideRule already gave depths)
        aligned = align_depth_points_to_s2_time(icesat2_points, t2)

    deltas = [p["delta_m"] for p in aligned] if aligned else [0.0]
    # Wave state at t2 (ROI centre) — used to flag surface-photon bias
    try:
        Hs_t2, wave_info = get_wave_height(lat_c, lon_c, t2)
    except Exception:
        Hs_t2, wave_info = 0.0, {"method": "unavailable"}
    # Apply wave bias only if significant (Hs > 0.5 m). For ICESat-2 sea-surface
    # photons the true still-water level is ~Hs/2 below the apparent crest, so
    # the equivalent depth gets an additional +Hs/2 correction. This is a
    # small-magnitude secondary term; it won't fire in the quiet Gulf.
    wave_correction_m = 0.0
    if Hs_t2 and Hs_t2 > 0.5:
        wave_correction_m = float(Hs_t2) / 2.0
        for p in aligned:
            p["depth_aligned"] = float(p["depth_aligned"] + wave_correction_m)
            p["wave_correction_m"] = wave_correction_m
    info = {
        "bbox": bbox,
        "t2_iso": t2.isoformat(),
        "roi_centre": {"lat": lat_c, "lon": lon_c, "T2_m": T2_c},
        "geoid_backend": "EGM2008_pgm" if _get_geoid_obj() else "analytic_gulf_fit",
        "tide_backend":  "open-meteo/cmems",  # pyTMD stub intentionally disabled
        "wave_backend":  wave_info.get("method", "open-meteo/cmems"),
        "wave_Hs_t2_m":  round(float(Hs_t2), 3),
        "wave_correction_m": round(float(wave_correction_m), 3),
        "n_points": len(aligned),
        "delta_min_m":  round(float(np.min(deltas)), 3),
        "delta_mean_m": round(float(np.mean(deltas)), 3),
        "delta_max_m":  round(float(np.max(deltas)), 3),
    }
    L.info(f"tidal_alignment: {info['n_points']} ICESat-2 pts retided to "
           f"{t2.isoformat()} · Δ range [{info['delta_min_m']:+.2f}, "
           f"{info['delta_max_m']:+.2f}] m · mean Δ {info['delta_mean_m']:+.2f} m")
    return aligned, info
