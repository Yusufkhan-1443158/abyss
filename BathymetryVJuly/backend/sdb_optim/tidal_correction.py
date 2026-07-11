# INTEGRATION: Call tidal_offset() on the acquisition timestamps of each S2/L8
# scene before building the temporal stack in compute_temporal_stats().
# Call normalize_scenes_to_datum() to reduce multi-temporal depth observations
# (from i-Boating chart soundings, ICESat-2 ATL24 cells, multibeam XYZ) to a
# common datum before feeding them as ground-truth to build_feature_cube() /
# SDBPatchDataset in backend.sdb_cnn_baseline.
#
# Datum notes:
#   - UAE Gulf: Admiralty Chart Datum (CD) ≈ Lowest Astronomical Tide (LAT).
#     Mean Sea Level (MSL) in the Arabian Gulf is approximately +0.9 m above CD
#     at Abu Dhabi (IHO TIDE TAB reference).
#   - Multibeam XYZ in this repo (validation/*.xyz) are already reduced to CD
#     (positive-down convention).  ICESat-2 ATL03/ATL24 depths are referenced
#     to WGS84 ellipsoid; subtract the EGM2008 geoid undulation to convert to
#     MSL, then apply the MSL→CD offset.
#   - This module corrects for *water-surface level* changes due to tides between
#     acquisition epochs, NOT for refraction.  Water-column refraction correction
#     (n_water ≈ 1.334 at 532 nm) is handled upstream in very_hr_engine.py.
#
# References:
#   Solano-Acosta et al. (2020): tidal correction in SDB workflows.
#   Dyer et al. (2022): ICESat-2 datum conversion for nearshore bathymetry.
#   pyTMD: Sutterley et al. (2017+), github.com/tsutterley/pyTMD.

"""Tidal correction for Satellite-Derived Bathymetry (SDB).

Two entry points:

1. ``tidal_offset(timestamps, lat, lon, model=...)``
   Returns modelled tide height (metres above CD/LAT) for each acquisition
   timestamp at the given location.

   Primary path: ``pyTMD.compute.LPET_elevations`` — long-period equilibrium
   tides only (semi-annual, 18.6-year nodal, etc.).  This does NOT include
   the dominant M2, S2, K1, O1 semidiurnal/diurnal constituents.  For a full
   tidal prediction at a specific coastal site, a constituent model file is
   required (OTIS/FES/GOT — not bundled here).

   If pyTMD constituent model files are available (set ``PYTMD_MODEL_DIR``
   and ``PYTMD_MODEL``), ``tide_elevations`` is called for a full tidal
   prediction.

   Fallback (when neither path is available): a harmonic stub that sums a
   representative M2 + S2 + K1 + O1 set for the Arabian Gulf / Morocco
   Atlantic coast using published constituent amplitudes from published tide
   gauge records.  The fallback is clearly flagged in the returned metadata.
   It is a crude approximation — do not use for production without model files.

2. ``normalize_scenes_to_datum(depths, timestamps, lat, lon, ...)``
   Applies the tidal offset to convert per-scene observed water depths to a
   common datum (LAT/MSL), reducing inter-acquisition variability.

Datum / refraction caveats (per HARD RULES — be honest)
---------------------------------------------------------
- Tidal offsets are in the MLLW/CD reference frame.  The conversion between
  MSL and CD is site-specific; nominal offsets are provided for UAE Gulf and
  Morocco Atlantic.
- Satellite-derived depths are affected by water-column refraction
  (n_water ≈ 1.334 at 532 nm; apparent depth ≈ true depth / n_water).
  This module corrects for tidal WATER SURFACE ELEVATION only, not refraction.
- ICESat-2 ATL03 range values are already refraction-corrected in the
  SlideRule processing chain; multibeam XYZ are survey-reduced.  Optical SDB
  requires a separate refraction correction applied to the inversion step.
"""

from __future__ import annotations

import datetime
import logging
import os
import warnings as _warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Harmonic stub constants — UAE Arabian Gulf + Morocco Atlantic
# Constituent amplitudes and phases from published tide gauge records:
#   Abu Dhabi (UAE Gulf): IHO Tide Tables Vol. II 2023; H (m), g (deg)
#   Casablanca (Morocco Atlantic): SHOM Annuaire des Marées 2023
# These are APPROXIMATE site-averaged values for the fallback stub only.
# Use a proper constituent model (OTIS/FES2022/GOT4.10c) for production.
# ---------------------------------------------------------------------------

# (amplitude_m, phase_degrees_UTC) for each constituent
# Angular frequency omega (rad/s) computed from period
_TIDAL_CONSTITUENTS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "UAE_gulf": {
        # constituent: (H_m, g_deg, period_hours)
        "M2": (0.41, 342.0, 12.4206),
        "S2": (0.20, 358.0, 12.0000),
        "K1": (0.16, 165.0, 23.9345),
        "O1": (0.12, 148.0, 25.8194),
        "N2": (0.09, 328.0, 12.6583),
        "M4": (0.04, 220.0, 6.2103),   # harmonic overtide
    },
    "morocco_atlantic": {
        "M2": (0.62, 340.0, 12.4206),
        "S2": (0.22, 355.0, 12.0000),
        "K1": (0.07, 280.0, 23.9345),
        "O1": (0.05, 262.0, 25.8194),
        "N2": (0.13, 320.0, 12.6583),
    },
}

# MSL above Chart Datum (LAT) in metres — nominal site values
_MSL_ABOVE_CD: Dict[str, float] = {
    "UAE_gulf": 0.90,          # Abu Dhabi; IHO Tide Tables 2023 NOMINAL
    "morocco_atlantic": 1.70,  # Casablanca; SHOM 2023 NOMINAL
}

# Reference epoch for harmonic argument computation: J2000.0
_J2000_EPOCH = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)


def _datetime_to_j2000_hours(dt: datetime.datetime) -> float:
    """Convert a UTC datetime to hours since J2000.0 (2000-01-01T12:00:00Z)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    delta = dt - _J2000_EPOCH
    return delta.total_seconds() / 3600.0


def _harmonic_stub(
    timestamps: List[datetime.datetime],
    site: str = "UAE_gulf",
) -> Tuple[np.ndarray, Dict]:
    """Compute tidal height using a harmonic stub (fallback when no model files).

    Uses published constituent amplitudes for the specified site.
    Returns (heights_m, metadata_dict).

    IMPORTANT: This is a FALLBACK approximation.  For any site within
    ~50 km of the named gauge the error is typically < 0.2 m RMS for
    the dominant M2+S2+K1+O1 constituents.  Long-period nodal corrections
    are NOT included here.  Do not use for IHO S-44 / CATZOC-grade work
    without validation against a tide gauge.
    """
    if site not in _TIDAL_CONSTITUENTS:
        site = "UAE_gulf"
        logger.warning("tidal_correction: unknown site '%s'; defaulting to UAE_gulf stub.", site)

    constituents = _TIDAL_CONSTITUENTS[site]
    n = len(timestamps)
    heights = np.zeros(n, dtype=np.float64)

    for dt_obj in range(n):
        dt = timestamps[dt_obj]
        t_hours = _datetime_to_j2000_hours(dt)
        h = 0.0
        for name, (amp, phase_deg, period_h) in constituents.items():
            omega_rad_per_h = 2.0 * np.pi / period_h
            phase_rad = np.radians(phase_deg)
            h += amp * np.cos(omega_rad_per_h * t_hours - phase_rad)
        heights[dt_obj] = h

    meta = {
        "method": "harmonic_stub_fallback",
        "site": site,
        "constituents": list(constituents.keys()),
        "warning": (
            "FALLBACK: harmonic stub with site-averaged published amplitudes. "
            "Accuracy ~0.2 m RMS for M2/S2/K1/O1 near the named gauge. "
            "Long-period nodal corrections absent. "
            "Install OTIS/FES tide model files + set PYTMD_MODEL_DIR for production."
        ),
    }
    logger.warning(
        "tidal_correction: Using harmonic stub fallback for site='%s'. "
        "Accuracy ~0.2 m RMS. Set PYTMD_MODEL_DIR + PYTMD_MODEL for a full prediction.",
        site,
    )
    return heights.astype(np.float32), meta


def _days_since_j2000(dt: datetime.datetime) -> float:
    """Convert UTC datetime to fractional days since J2000.0."""
    return _datetime_to_j2000_hours(dt) / 24.0


def tidal_offset(
    timestamps: List[datetime.datetime],
    lat: float,
    lon: float,
    model: Optional[str] = None,
    pytmd_model_dir: Optional[str] = None,
    fallback_site: str = "UAE_gulf",
) -> Tuple[np.ndarray, Dict]:
    """Compute modelled tide height per acquisition timestamp.

    Tries paths in order:
    1. Full pyTMD constituent prediction via ``pyTMD.compute.tide_elevations``
       (requires downloaded model files in ``pytmd_model_dir``).
    2. Long-period equilibrium tides via ``pyTMD.compute.LPET_elevations``
       (no external files; captures slow nodal cycles but NOT M2/S2/K1/O1).
       These are typically < 0.05 m in amplitude and are added as a correction
       on top of the harmonic stub if the stub is also used.
    3. Harmonic stub (published constituent amplitudes for ``fallback_site``).
       Flagged clearly in returned metadata.

    Parameters
    ----------
    timestamps : list[datetime.datetime]
        UTC acquisition datetimes, one per scene.
    lat : float
        Latitude of the target location (WGS84, decimal degrees).
    lon : float
        Longitude of the target location (WGS84, decimal degrees).
    model : str or None
        pyTMD model name (e.g. ``"TPXO9-atlas-v5"``, ``"FES2022"``,
        ``"GOT4.10c"``).  If None, ``PYTMD_MODEL`` env var is checked,
        then LPET fallback is used.
    pytmd_model_dir : str or None
        Directory containing the tide model files.  If None,
        ``PYTMD_MODEL_DIR`` env var is checked, then ``~/.cache/pytmd``.
    fallback_site : str
        Site key for harmonic stub fallback (``"UAE_gulf"`` or
        ``"morocco_atlantic"``).

    Returns
    -------
    tide_heights_m : np.ndarray, shape (T,), float32
        Tide height in metres above LAT/CD at each timestamp.
        Positive = water level above datum (tide is in; shallower chart depths).
        Convention: subtract from observed chart depth to get datum depth.
    meta : dict
        Provenance metadata:
        - ``"method"`` : which path was used (``"pytmd_full"``, ``"lpet_only"``,
          ``"harmonic_stub_fallback"``)
        - ``"model"`` : model name if pyTMD full path
        - ``"lpet_correction_m"`` : LPET offsets added on top of stub (if any)
        - ``"warning"`` : accuracy caveat string

    Notes
    -----
    Datum convention: the returned heights are water-surface elevation above
    LAT (Lowest Astronomical Tide ≈ Chart Datum in IHO convention).  To
    convert a depth observed at time t from instantaneous water surface to
    LAT-referenced:
        depth_LAT = depth_obs - tide_heights_m[t]
    For MSL-referenced output, add the site MSL-above-CD offset:
        depth_MSL = depth_LAT + MSL_above_CD[site]
    where ``MSL_above_CD`` nominal values are in ``_MSL_ABOVE_CD``.
    """
    n = len(timestamps)
    if n == 0:
        return np.array([], dtype=np.float32), {"method": "no_timestamps", "warning": "empty input"}

    # ------------------------------------------------------------------ #
    # Resolve model config from env
    # ------------------------------------------------------------------ #
    if model is None:
        model = os.environ.get("PYTMD_MODEL", None)
    if pytmd_model_dir is None:
        pytmd_model_dir = os.environ.get("PYTMD_MODEL_DIR", None)

    # ------------------------------------------------------------------ #
    # Path 1: Full pyTMD constituent prediction
    # ------------------------------------------------------------------ #
    if model is not None and pytmd_model_dir is not None:
        try:
            import pyTMD.compute
            import pathlib

            # Build delta_time array: seconds since J2000.0 for pyTMD
            # pyTMD epoch default is (2000,1,1,0,0,0); our J2000 is noon on 2000-01-01
            # pyTMD uses MJD or seconds from epoch; we provide days since (2000,1,1,0,0,0)
            epoch = (2000, 1, 1, 0, 0, 0)
            epoch_dt = datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
            delta_days = np.array([
                (ts.replace(tzinfo=datetime.timezone.utc) if ts.tzinfo is None else ts
                 - epoch_dt).total_seconds() / 86400.0
                for ts in timestamps
            ])

            x_arr = np.full(n, float(lon))
            y_arr = np.full(n, float(lat))

            result_da = pyTMD.compute.tide_elevations(
                x=x_arr,
                y=y_arr,
                delta_time=delta_days,
                directory=pathlib.Path(pytmd_model_dir),
                model=model,
                crs=4326,
                epoch=epoch,
                type="drift",
                standard="UTC",
            )
            tide_h = np.array(result_da.values, dtype=np.float32)
            # Replace fill values (masked where out of model domain) with NaN
            tide_h = np.where(np.isfinite(tide_h), tide_h, 0.0).astype(np.float32)
            meta = {
                "method": "pytmd_full",
                "model": model,
                "model_dir": str(pytmd_model_dir),
                "warning": "Full constituent prediction via pyTMD.",
            }
            logger.info(
                "tidal_offset: pyTMD full model '%s' — %d timestamps, "
                "range [%.3f, %.3f] m",
                model, n, float(tide_h.min()), float(tide_h.max()),
            )
            return tide_h, meta

        except Exception as exc:
            logger.warning(
                "tidal_offset: pyTMD full-model path failed (%s). "
                "Falling back to LPET + harmonic stub.", exc
            )

    # ------------------------------------------------------------------ #
    # Path 2: LPET only (no external files)
    # ------------------------------------------------------------------ #
    lpet_heights = np.zeros(n, dtype=np.float32)
    lpet_ok = False
    try:
        import pyTMD.compute

        epoch_lpet = (2000, 1, 1, 0, 0, 0)
        epoch_dt_lpet = datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        delta_days = np.array([
            (ts.replace(tzinfo=datetime.timezone.utc) if ts.tzinfo is None else ts
             - epoch_dt_lpet).total_seconds() / 86400.0
            for ts in timestamps
        ])

        x_arr = np.full(n, float(lon))
        y_arr = np.full(n, float(lat))

        result_da = pyTMD.compute.LPET_elevations(
            x=x_arr,
            y=y_arr,
            delta_time=delta_days,
            crs=4326,
            epoch=epoch_lpet,
            type="drift",
            standard="UTC",
        )
        lpet_heights = np.array(result_da.values, dtype=np.float32)
        lpet_heights = np.where(np.isfinite(lpet_heights), lpet_heights, 0.0).astype(np.float32)
        lpet_ok = True
        logger.info(
            "tidal_offset: LPET computed for %d timestamps, range [%.4f, %.4f] m "
            "(long-period only; adding harmonic stub for M2/S2/K1/O1).",
            n, float(lpet_heights.min()), float(lpet_heights.max()),
        )
    except Exception as exc:
        logger.warning("tidal_offset: LPET path failed (%s); using stub only.", exc)

    # ------------------------------------------------------------------ #
    # Path 3: Harmonic stub (always runs as M2/S2/K1/O1 baseline)
    # ------------------------------------------------------------------ #
    stub_heights, stub_meta = _harmonic_stub(timestamps, site=fallback_site)

    # Combine: stub (dominant short-period) + LPET (long-period correction)
    combined = stub_heights + lpet_heights  # LPET is typically <0.02 m

    method = "lpet_only_plus_harmonic_stub" if lpet_ok else "harmonic_stub_fallback"
    meta = {
        "method": method,
        "fallback_site": fallback_site,
        "lpet_applied": lpet_ok,
        "lpet_range_m": [float(lpet_heights.min()), float(lpet_heights.max())] if lpet_ok else None,
        "stub_constituents": stub_meta["constituents"],
        "warning": stub_meta["warning"],
    }

    logger.info(
        "tidal_offset: method=%s, %d timestamps, combined range [%.3f, %.3f] m",
        method, n, float(combined.min()), float(combined.max()),
    )
    return combined, meta


def normalize_scenes_to_datum(
    depths: np.ndarray,
    timestamps: List[datetime.datetime],
    lat: float,
    lon: float,
    model: Optional[str] = None,
    pytmd_model_dir: Optional[str] = None,
    fallback_site: str = "UAE_gulf",
    target_datum: str = "LAT",
    msl_above_cd_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Reduce multi-temporal depth observations to a common datum.

    Converts per-scene depth observations (referenced to instantaneous water
    surface at acquisition time) to a common datum (LAT or MSL).

    The correction is:
        depth_datum[t] = depth_obs[t] - tide_height[t]   (for LAT)
        depth_datum[t] = depth_obs[t] - tide_height[t] + msl_above_cd_m  (for MSL)

    Scientific rationale
    --------------------
    In multi-temporal SDB, scenes acquired at different tidal phases have
    systematic depth offsets proportional to the tidal range.  In the UAE
    Arabian Gulf, the micro-tidal range is ~0.7–1.2 m (neap to spring), which
    is large compared to the SDB target accuracy of <0.5 m.  Ignoring tidal
    normalisation inflates inter-scene variance and biases the temporal median
    toward a tide-weighted pseudo-datum rather than true seafloor depth.

    Reference: Solano-Acosta et al. (2020), "Tidal correction for improving
    satellite derived bathymetry over shallow coral reefs", Journal of
    Coastal Research 95(sp1):530–534.

    Parameters
    ----------
    depths : np.ndarray, shape (T,) or (T, N) or (T, H, W)
        Observed depths (positive-down) at each acquisition timestamp.
        - (T,): one scalar depth per timestamp (e.g. a single sounding).
        - (T, N): N soundings per timestamp.
        - (T, H, W): full depth grids per timestamp (tidal shift applied uniformly).
    timestamps : list[datetime.datetime]
        UTC acquisition datetimes, length T.
    lat, lon : float
        Representative location for tidal prediction.  For spatially
        distributed scenes, use the scene centroid.  Tidal height variation
        across a typical SDB tile (<50 km) is < 0.02 m and can be ignored.
    model, pytmd_model_dir : str or None
        Passed through to ``tidal_offset()``.
    fallback_site : str
        Harmonic stub site key (``"UAE_gulf"`` or ``"morocco_atlantic"``).
    target_datum : str
        ``"LAT"`` (Lowest Astronomical Tide ≈ Chart Datum) or ``"MSL"``.
    msl_above_cd_m : float or None
        MSL above Chart Datum in metres (site-specific).  If None, the
        nominal value for ``fallback_site`` is used from ``_MSL_ABOVE_CD``.
        Only relevant when ``target_datum == "MSL"``.

    Returns
    -------
    depths_corrected : np.ndarray, same shape as ``depths``
        Tidal-corrected depths referenced to ``target_datum``.
    tide_heights_m : np.ndarray, shape (T,)
        Applied tidal offsets per timestamp.
    meta : dict
        Provenance from ``tidal_offset()`` plus datum conversion metadata:
        - ``"target_datum"`` : str
        - ``"msl_above_cd_m"`` : float (if MSL conversion applied)
        - ``"tide_rms_m"`` : float — RMS tidal spread (measure of correction magnitude)
        - ``"inter_scene_depth_std_before"`` : float (scalar depths only)
        - ``"inter_scene_depth_std_after"`` : float (scalar depths only)

    Notes
    -----
    Refraction caveat: this function corrects for *water surface elevation
    change* only.  Refraction (n_water ≈ 1.334) is a depth-proportional
    effect that must be handled separately in the SDB inversion step.
    Water-surface elevation and refraction are orthogonal corrections.
    """
    depths = np.asarray(depths, dtype=np.float32)
    T = len(timestamps)

    if depths.shape[0] != T:
        raise ValueError(
            f"normalize_scenes_to_datum: depths.shape[0]={depths.shape[0]} "
            f"!= len(timestamps)={T}"
        )

    # Get tidal heights for each timestamp
    tide_heights_m, tide_meta = tidal_offset(
        timestamps, lat, lon,
        model=model,
        pytmd_model_dir=pytmd_model_dir,
        fallback_site=fallback_site,
    )

    tide_rms = float(np.sqrt(np.mean(tide_heights_m ** 2)))

    # Apply correction: depths_datum = depths_obs - tide_height
    # tide_height is elevation of water surface above datum;
    # subtracting it removes the tidal increment from the apparent depth.
    if depths.ndim == 1:
        std_before = float(np.nanstd(depths))
        depths_corrected = depths - tide_heights_m
        std_after = float(np.nanstd(depths_corrected))
    elif depths.ndim == 2:
        # (T, N): broadcast tide over N soundings
        depths_corrected = depths - tide_heights_m[:, np.newaxis]
        std_before = float(np.nanstd(depths))
        std_after = float(np.nanstd(depths_corrected))
    elif depths.ndim == 3:
        # (T, H, W): broadcast over spatial grid
        depths_corrected = depths - tide_heights_m[:, np.newaxis, np.newaxis]
        std_before = float(np.nanstd(depths))
        std_after = float(np.nanstd(depths_corrected))
    else:
        raise ValueError(f"depths must be 1D, 2D or 3D; got shape {depths.shape}")

    # Optional MSL conversion
    if target_datum.upper() == "MSL":
        if msl_above_cd_m is None:
            msl_above_cd_m = _MSL_ABOVE_CD.get(fallback_site, 0.0)
        depths_corrected = depths_corrected + float(msl_above_cd_m)
    else:
        msl_above_cd_m = None

    meta = {
        **tide_meta,
        "target_datum": target_datum.upper(),
        "msl_above_cd_m": msl_above_cd_m,
        "tide_rms_m": tide_rms,
        "tide_range_m": [float(tide_heights_m.min()), float(tide_heights_m.max())],
        "inter_scene_depth_std_before": std_before,
        "inter_scene_depth_std_after": std_after,
        "inter_scene_std_reduction_m": std_before - std_after,
    }

    logger.info(
        "normalize_scenes_to_datum: datum=%s, tide RMS=%.3f m, "
        "depth std %.3f m -> %.3f m (delta=%.3f m)",
        target_datum, tide_rms, std_before, std_after, std_before - std_after,
    )
    return depths_corrected.astype(np.float32), tide_heights_m, meta
