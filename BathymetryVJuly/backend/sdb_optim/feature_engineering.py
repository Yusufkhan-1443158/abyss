# INTEGRATION: Feed fuse_optical_sensors() output as the `s2` dict to
# backend.sdb_cnn_baseline.build_feature_cube(), OR pass the returned cube
# directly into backend.sdb_cnn_baseline.compute_lyzenga_features() for Iter-3.
# compute_optical_indices() and compute_temporal_stats() return channel arrays
# that EXTEND the canonical A0 12-channel cube defined in build_feature_cube().
# deepwater_baseline_prior() returns a (H,W) spatial prior that can seed the
# Lyzenga/Stumpf baseline where passive optics is attenuation-limited.
#
# Band conventions (Sentinel-2 MSI, 10 m resolution):
#   B1  coastal-aerosol  ~443 nm  (60 m, resampled when available)
#   B2  blue             ~490 nm  10 m
#   B3  green            ~560 nm  10 m
#   B4  red              ~665 nm  10 m
#   B8  NIR              ~842 nm  10 m
#
# Landsat 8/9 OLI equivalents (30 m):
#   B1  coastal-aerosol  ~443 nm
#   B2  blue             ~482 nm
#   B3  green            ~562 nm
#   B4  red              ~655 nm
#   B5  NIR              ~865 nm
#
# Cross-calibration: Pahlevan et al. (2022) show that after atmospheric
# correction, S2 MSI and Landsat 8/9 OLI surface reflectance agree to within
# ~0.003 SR units (RMSD) over coastal/aquatic targets.  The nominal offsets
# applied here are derived from their Table 1 (Rrs at coastal bands) and are
# labelled NOMINAL — they should be recalibrated from overlapping acquisitions
# before operational use.
#
# References (key):
#   Pahlevan et al. (2022) "Simultaneous retrieval of selected optical water
#       quality indicators from Landsat-8, Sentinel-2, and Sentinel-3"
#       Remote Sensing of Environment, 270, 112860.
#   Lyzenga (1978, 1985): passive optical SDB; log-transformed radiance ratio.
#   Stumpf et al. (2003): log-ratio bathymetry, Coastal Management 31:1.
#   McFeeters (1996): NDWI = (Green-NIR)/(Green+NIR), IJRS 17:7.
#   Xu (2006): MNDWI = (Green-SWIR)/(Green+SWIR), IJRS 27:14.
#   Lacaux et al. (2007): NDTI = (Red-Green)/(Red+Green), RSE 109:3-4.

"""Feature engineering for Satellite-Derived Bathymetry (SDB).

This module provides three functional groups:

1. **Multi-sensor fusion** (`fuse_optical_sensors`): harmonise Sentinel-2 MSI
   and Landsat 8/9 OLI reflectance bands to a common 5-channel set at the
   coarser of the two native resolutions (or a user-specified target).
   Cross-calibration offsets are NOMINAL (Pahlevan et al. 2022) — label them
   as such and do not represent fabricated numbers.

2. **Optical indices + temporal statistics** (`compute_optical_indices`,
   `compute_temporal_stats`): vectorised NDWI, MNDWI, Stumpf log-ratios,
   Lyzenga log-bands, coastal-aerosol ratio, NDTI turbidity, plus per-pixel
   temporal median/variance/valid-count from a multi-date stack.  Temporal
   median is the most glint- and cloud-robust single-scene substitute because
   sunglint and thin cloud artefacts are quasi-random across dates (especially
   with the solar zenith and wind variability in UAE/Morocco coastal waters),
   so the 50th percentile over N>=3 dates suppresses transient radiance spikes.

3. **Deep-water prior** (`deepwater_baseline_prior`): builds an
   optically-deep-water reflectance baseline and an ICESat-2-anchored spatial
   prior for depth regularisation beyond ~1–1.5 Secchi depths, where passive
   reflectance saturates.  IMPORTANT: this function does NOT incorporate true
   radar altimetry (radar altimeters are severely range-limited over the
   sub-kilometre coastal strip and cannot resolve individual depth soundings);
   the "altimetry trend" label refers exclusively to the ICESat-2 photon-
   counted depth anchor, interpolated spatially.  The output is labelled
   accordingly in the returned dictionary.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nominal cross-calibration offsets: S2 MSI → Landsat 8/9 OLI SR equivalents
# Derived from Pahlevan et al. (2022), Table 1, Rrs medians over coastal sites.
# Applied as: L8_SR_equiv ≈ S2_SR + offset
# Sign: positive = S2 reads higher than L8 in that band.
# IMPORTANT: these are NOMINAL scene-averaged values.  Per-scene recalibration
# from overlapping acquisitions is strongly recommended before operational use.
# ---------------------------------------------------------------------------
_S2_TO_L8_SR_OFFSETS: Dict[str, float] = {
    "coastal_aerosol": -0.001,  # S2 B1 vs L8 B1; Pahlevan et al. 2022 NOMINAL
    "blue":            +0.002,  # S2 B2 vs L8 B2; NOMINAL
    "green":           +0.001,  # S2 B3 vs L8 B3; NOMINAL
    "red":             -0.001,  # S2 B4 vs L8 B4; NOMINAL
    "nir":             -0.004,  # S2 B8 vs L8 B5; NOMINAL (band-width difference ~23 nm)
}

# Sentinel-2 band-pass effective wavelengths (nm) — ESA S2 SRF centres
_S2_CENTRE_NM: Dict[str, float] = {
    "coastal_aerosol": 442.7,
    "blue":            492.4,
    "green":           559.8,
    "red":             664.6,
    "nir":             832.8,
}

# Landsat 8/9 OLI band-pass effective wavelengths (nm) — USGS
_L8_CENTRE_NM: Dict[str, float] = {
    "coastal_aerosol": 442.0,
    "blue":            482.0,
    "green":           561.4,
    "red":             654.6,
    "nir":             864.7,
}


def fuse_optical_sensors(
    s2: Dict[str, np.ndarray],
    landsat: Optional[Dict[str, np.ndarray]] = None,
    target_res_m: float = 30.0,
    apply_crosscal: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Harmonise Sentinel-2 MSI and Landsat 8/9 OLI to a common band set.

    Scientific rationale
    --------------------
    S2 (10 m) and Landsat 8/9 (30 m) observe the same coastal waters in
    overlapping time windows.  Fusing them improves temporal sampling (both
    missions revisit ~5 days combined) and therefore the probability of
    obtaining a glint-free, cloud-free composite.

    Band-pass differences between S2 MSI and L8/9 OLI are real but small at
    the coastal blue/green bands used for SDB (Pahlevan et al. 2022 report
    RMSD ≈ 0.003 in surface reflectance after atmospheric correction over
    coastal targets).  A simple additive SR offset (nominal, from Table 1 of
    that paper) is applied here to bring L8/9 bands into the S2 SR frame.
    Operational users should derive per-scene offsets from spatially
    overlapping acquisitions.

    NIR band caveat: S2 B8 (central ~833 nm, width ~115 nm) vs L8 B5
    (central ~865 nm, width ~28 nm) have the largest band-pass mismatch.  The
    NIR is only used as a water/land mask discriminator in this pipeline, not
    as a depth predictor, so the small residual offset has negligible effect
    on SDB accuracy.  However, users combining NIR-based indices across sensors
    should apply a more careful band-correction.

    Resolution harmonisation
    ------------------------
    S2 blue/green/red/NIR are 10 m.  Landsat is 30 m.  If `target_res_m` ==
    30 (default), S2 bands are block-averaged 3x3 to 30 m using simple mean
    downsampling (equivalent to a low-pass anti-alias filter at 30 m).  If
    `target_res_m` == 10, Landsat bands are bilinearly upsampled (scipy
    zoom).  Using 30 m as the common frame is preferred to avoid
    artificially sharpening the lower-resolution sensor.

    Parameters
    ----------
    s2 : dict
        Sentinel-2 bands as float32 arrays in [0, ~0.5] surface reflectance.
        Required keys: ``"blue"`` (B2), ``"green"`` (B3), ``"red"`` (B4),
        ``"nir"`` (B8).
        Optional: ``"coastal_aerosol"`` (B1, 60 m — already resampled to 10 m
        if provided).
        Shape: ``(H_s2, W_s2)`` per band (10 m native).
    landsat : dict or None
        Landsat 8/9 OLI bands in the same SR units (after atmospheric
        correction, e.g. USGS Collection 2 SR).
        Required keys: ``"blue"`` (B2), ``"green"`` (B3), ``"red"`` (B4),
        ``"nir"`` (B5).
        Optional: ``"coastal_aerosol"`` (B1).
        Shape: ``(H_l8, W_l8)`` per band (30 m native).
        If ``None``, only S2 is returned (resampled to ``target_res_m``).
    target_res_m : float
        Common output resolution in metres.  Default 30 m (L8 native).
        If < 30, L8 is upsampled (bilinear zoom via scipy.ndimage.zoom).
    apply_crosscal : bool
        Apply nominal S2→L8 SR offsets (Pahlevan et al. 2022 NOMINAL) to
        bring sensors into a common radiometric frame.  Default True.
    verbose : bool
        Log band-mismatch warnings and cross-cal application.

    Returns
    -------
    fused : dict[str, np.ndarray]
        Common band set: ``{blue, green, red, nir, coastal_aerosol}`` at
        ``target_res_m``.  Values are float32 clipped to ``[1e-6, 0.5]``.
        ``coastal_aerosol`` is only present if both inputs provide it.
        ``"sensor"`` key: list of contributing sensor names.
        ``"n_scenes"`` key: int, total number of input scenes.
    meta : dict
        Provenance metadata:
        ``crosscal_applied`` (bool), ``crosscal_offsets`` (dict of NOMINAL
        offsets), ``target_res_m`` (float), ``warning`` (list[str]).

    Notes
    -----
    - NOMINAL label: the cross-calibration offsets in ``_S2_TO_L8_SR_OFFSETS``
      are scene-averaged medians from Pahlevan et al. (2022).  They represent
      a best estimate, not a precise physical measurement for a given scene.
    - This function does NOT handle BRDF or adjacency-effect corrections.
    - If Landsat and S2 are from different dates, the user must ensure they
      are temporally compatible (same tidal state, similar turbidity).
    """
    from scipy.ndimage import zoom as nd_zoom

    eps = 1e-6
    warnings: List[str] = []

    def _to_float32(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr.astype(np.float32), eps, 0.5)

    # ------------------------------------------------------------------ #
    # Step 1: Prepare S2 at native 10 m
    # ------------------------------------------------------------------ #
    s2_clean: Dict[str, np.ndarray] = {}
    for band in ("blue", "green", "red", "nir"):
        if band not in s2:
            raise KeyError(f"fuse_optical_sensors: S2 missing required band '{band}'")
        s2_clean[band] = _to_float32(s2[band])
    if "coastal_aerosol" in s2 and s2["coastal_aerosol"] is not None:
        s2_clean["coastal_aerosol"] = _to_float32(s2["coastal_aerosol"])

    H_s2, W_s2 = s2_clean["blue"].shape

    # Determine S2 output factor: how much to downsample S2 to reach target_res_m
    # Assume S2 native = 10 m
    s2_factor = 10.0 / target_res_m  # < 1 means downsample (e.g. 10/30 ≈ 0.333)

    def _resample_s2(arr: np.ndarray, factor: float) -> np.ndarray:
        """Resample S2 band using block-average (down) or zoom (up)."""
        if abs(factor - 1.0) < 0.01:
            return arr
        if factor < 1.0:
            # Downsample: block average
            block = max(1, round(1.0 / factor))
            h_new = arr.shape[0] // block
            w_new = arr.shape[1] // block
            truncated = arr[: h_new * block, : w_new * block]
            reshaped = truncated.reshape(h_new, block, w_new, block)
            return reshaped.mean(axis=(1, 3)).astype(np.float32)
        else:
            # Upsample: bilinear
            return nd_zoom(arr, factor, order=1).astype(np.float32)

    s2_rs: Dict[str, np.ndarray] = {
        k: _resample_s2(v, s2_factor) for k, v in s2_clean.items()
    }
    H_out, W_out = s2_rs["blue"].shape
    if verbose:
        logger.info(
            "fuse_optical_sensors: S2 resampled %dx%d -> %dx%d "
            "(factor=%.3f, target=%.0f m)",
            H_s2, W_s2, H_out, W_out, s2_factor, target_res_m,
        )

    if landsat is None:
        meta = {
            "crosscal_applied": False,
            "crosscal_offsets": {},
            "target_res_m": target_res_m,
            "sensors": ["S2"],
            "n_scenes": 1,
            "warning": ["No Landsat provided; single-sensor output"],
        }
        fused = dict(s2_rs)
        fused["sensor"] = ["S2"]
        fused["n_scenes"] = 1
        return fused, meta

    # ------------------------------------------------------------------ #
    # Step 2: Prepare Landsat at native 30 m
    # ------------------------------------------------------------------ #
    l8_clean: Dict[str, np.ndarray] = {}
    for band in ("blue", "green", "red", "nir"):
        if band not in landsat:
            raise KeyError(f"fuse_optical_sensors: Landsat missing required band '{band}'")
        l8_clean[band] = _to_float32(landsat[band])
    if "coastal_aerosol" in landsat and landsat["coastal_aerosol"] is not None:
        l8_clean["coastal_aerosol"] = _to_float32(landsat["coastal_aerosol"])

    # Landsat output factor: assume L8 native = 30 m
    l8_factor = 30.0 / target_res_m  # < 1 means downsample, > 1 upsample to 10 m

    def _resample_l8(arr: np.ndarray, factor: float) -> np.ndarray:
        if abs(factor - 1.0) < 0.01:
            return arr
        if factor < 1.0:
            block = max(1, round(1.0 / factor))
            h_new = arr.shape[0] // block
            w_new = arr.shape[1] // block
            truncated = arr[: h_new * block, : w_new * block]
            return truncated.reshape(h_new, block, w_new, block).mean(axis=(1, 3)).astype(np.float32)
        else:
            return nd_zoom(arr, factor, order=1).astype(np.float32)

    l8_rs: Dict[str, np.ndarray] = {
        k: _resample_l8(v, l8_factor) for k, v in l8_clean.items()
    }

    # ------------------------------------------------------------------ #
    # Step 3: Shape-align to the S2-derived output grid
    # ------------------------------------------------------------------ #
    def _crop_or_pad(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = arr.shape
        # Crop if larger, pad (edge) if smaller
        arr_c = arr[: min(h, target_h), : min(w, target_w)]
        if arr_c.shape[0] < target_h or arr_c.shape[1] < target_w:
            arr_c = np.pad(
                arr_c,
                ((0, max(0, target_h - arr_c.shape[0])),
                 (0, max(0, target_w - arr_c.shape[1]))),
                mode="edge",
            )
        return arr_c.astype(np.float32)

    l8_aligned: Dict[str, np.ndarray] = {
        k: _crop_or_pad(v, H_out, W_out) for k, v in l8_rs.items()
    }

    # ------------------------------------------------------------------ #
    # Step 4: Apply nominal cross-calibration (S2 → L8 frame)
    # ------------------------------------------------------------------ #
    crosscal_applied = False
    crosscal_log: Dict[str, float] = {}
    if apply_crosscal:
        for band, offset in _S2_TO_L8_SR_OFFSETS.items():
            if band in s2_rs:
                # Bring S2 into L8 SR frame: S2_corrected = S2 - offset
                # (offset is defined as S2 - L8, so subtract to equalise)
                s2_rs[band] = np.clip(
                    s2_rs[band] - float(offset), eps, 0.5
                ).astype(np.float32)
                crosscal_log[band] = -float(offset)  # applied delta
        crosscal_applied = True
        if verbose:
            logger.info(
                "fuse_optical_sensors: NOMINAL cross-cal applied (Pahlevan 2022). "
                "Applied SR deltas: %s. These are NOMINAL — recalibrate per-scene "
                "before operational use.",
                crosscal_log,
            )
        warnings.append(
            "Cross-cal offsets are NOMINAL (Pahlevan et al. 2022 Table 1). "
            "Per-scene recalibration from overlapping acquisitions recommended."
        )
    else:
        warnings.append("Cross-calibration NOT applied — sensors in different SR frames.")

    # ------------------------------------------------------------------ #
    # Step 5: Fuse by simple mean (equal weight)
    # ------------------------------------------------------------------ #
    fused: Dict[str, np.ndarray] = {}
    for band in ("blue", "green", "red", "nir"):
        fused[band] = (0.5 * s2_rs[band] + 0.5 * l8_aligned[band]).astype(np.float32)
    # Coastal aerosol: only if both provide it
    if "coastal_aerosol" in s2_rs and "coastal_aerosol" in l8_aligned:
        fused["coastal_aerosol"] = (
            0.5 * s2_rs["coastal_aerosol"] + 0.5 * l8_aligned["coastal_aerosol"]
        ).astype(np.float32)
    elif "coastal_aerosol" in s2_rs:
        fused["coastal_aerosol"] = s2_rs["coastal_aerosol"]
        warnings.append("coastal_aerosol from S2 only (Landsat not provided).")
    elif "coastal_aerosol" in l8_aligned:
        fused["coastal_aerosol"] = l8_aligned["coastal_aerosol"]
        warnings.append("coastal_aerosol from Landsat only (S2 not provided).")

    fused["sensor"] = ["S2", "Landsat8/9"]
    fused["n_scenes"] = 2

    meta: Dict[str, object] = {
        "crosscal_applied": crosscal_applied,
        "crosscal_offsets": crosscal_log,
        "target_res_m": target_res_m,
        "sensors": ["S2", "Landsat8/9"],
        "n_scenes": 2,
        "warning": warnings,
    }

    if verbose:
        logger.info(
            "fuse_optical_sensors: output shape %dx%d, bands=%s",
            H_out, W_out, list(fused.keys()),
        )
    return fused, meta


def deepwater_baseline_prior(
    coords: np.ndarray,
    depths_known: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    reflectance_cube: Optional[np.ndarray] = None,
    water_mask: Optional[np.ndarray] = None,
    icesat2_depths: Optional[np.ndarray] = None,
    icesat2_lats: Optional[np.ndarray] = None,
    icesat2_lons: Optional[np.ndarray] = None,
    smooth_sigma_px: float = 5.0,
    deep_percentile: float = 2.0,
    secchi_depth_m: float = 8.0,
) -> Dict[str, np.ndarray]:
    """Build a spatially-smooth deep-water prior for SDB regularisation.

    Scientific rationale
    --------------------
    Passive optical SDB saturates at approximately 1–1.5 Secchi depths (Zd).
    For UAE coastal waters, Zd ~ 5–10 m (clear open water) and 2–5 m in
    turbid nearshore zones, meaning optical information is effectively lost
    below 8–15 m.  A deep-water prior derived from:

    1. **Optically-deep reflectance floor**: the 2nd percentile of
       deglinted reflectance over pixels with NDWI > 0.3 — this is the
       R_infinity term in Lyzenga (1978, 1985).  Used to normalise shallow-
       bottom reflectance to be independent of water-column colour.

    2. **ICESat-2 spatial anchor** (when ``icesat2_depths`` are provided):
       ICESat-2 ATL03 photon-counted depths provide a sparse but accurate
       ground truth in the 0–25 m range along narrow ground tracks (2–4 Hz,
       ~0.7 m footprint, ~17 m along-track spacing for strong beams).  Here
       we interpolate the ICESat-2 depths onto the output pixel grid using
       Gaussian-weighted radial-basis interpolation, yielding a smooth
       ``icesat2_spatial_prior`` (H×W) that constrains the model in depth
       ranges where passive optics fails.

    3. **Dense known-point IDW fill**: all ``depths_known`` (from multibeam,
       chart soundings, ATL24 cells) are fused into the prior using inverse-
       distance-weighted interpolation (IDW, p=2) on a decimated grid, then
       smoothed with a Gaussian kernel (sigma = ``smooth_sigma_px``).

    IMPORTANT HONESTY NOTE
    ----------------------
    This function does NOT incorporate true **radar altimetry** data.  Radar
    altimeters (Jason-3, Sentinel-6, SWOT) operate in open-ocean or large-lake
    mode and cannot resolve individual coastal depth soundings at the 10–30 m
    pixel scale used here; their coastal waveform retracking is limited to
    ~5–20 km offshore footprints.  The ``altimetry_trend`` label in the
    returned dict refers exclusively to the ICESat-2 photon-counted depth
    anchor interpolated spatially.  ICESat-2 is a photon-counting laser
    altimeter (532 nm), not a radar, and it DOES resolve individual seafloor
    returns at the 1 m scale in clear coastal waters.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Known (lat, lon) pairs for ``depths_known`` soundings.
    depths_known : np.ndarray, shape (N,)
        Depth values (positive-down, metres) at ``coords`` positions.
    grid_lat : np.ndarray, shape (H, W)
        Grid latitude of output pixels.
    grid_lon : np.ndarray, shape (H, W)
        Grid longitude of output pixels.
    reflectance_cube : np.ndarray or None, shape (H, W, C)
        If provided, used to estimate the R_infinity deep-water floor per
        band (2nd percentile over deep pixels, analogous to Lyzenga 1978).
        Expects channel order from build_feature_cube(): ch4=B2_dg, ch5=B3_dg.
    water_mask : np.ndarray or None, shape (H, W) bool
        Water pixels (NDWI > 0 OR SCL==6).  Used to restrict deep-pixel
        detection.  If None, all pixels assumed water.
    icesat2_depths : np.ndarray or None, shape (M,)
        ICESat-2 (ATL03/ATL24) depth values along track.  Positive-down, m.
    icesat2_lats, icesat2_lons : np.ndarray or None, shape (M,)
        Geographic coordinates of the ICESat-2 points.
    smooth_sigma_px : float
        Standard deviation (pixels) for Gaussian smoothing of the IDW surface.
        Default 5 px ≈ 150 m at 30 m grid (suppresses tile-edge artefacts).
    deep_percentile : float
        Percentile of deep-water pixels used for R_infinity (Lyzenga 1978).
        Default 2.0 (matches build_feature_cube).
    secchi_depth_m : float
        Estimated Secchi depth for the site (m).  Used only to produce a
        metadata label indicating the approximate optical depth limit.
        Default 8.0 m (conservative UAE open-water estimate).

    Returns
    -------
    prior : dict
        Keys:
        - ``"idw_depth_prior"`` : (H, W) float32 — IDW-filled + Gaussian-
          smoothed depth prior from all known soundings.  np.nan where
          extrapolated far from any known point.
        - ``"idw_valid_mask"`` : (H, W) bool — pixels with at least one
          known sounding within the IDW search radius.
        - ``"icesat2_spatial_prior"`` : (H, W) float32 or None — ICESat-2
          photon-counted depths interpolated to output grid.  None when no
          ICESat-2 data provided.  LABEL: ICESat-2 laser altimetry anchor,
          NOT radar altimetry.
        - ``"r_infinity_floor"`` : dict[str, float] — per-band deep-water
          reflectance baseline (R_infinity in Lyzenga 1978), band names
          matching build_feature_cube channel order.  Empty dict if
          reflectance_cube not provided.
        - ``"optical_depth_limit_m"`` : float — 1.5 * secchi_depth_m, the
          estimated passive-optics saturation depth for this site.
        - ``"altimetry_anchor_label"`` : str — explicit provenance label
          ("ICESat-2 photon-counting laser; NOT radar altimetry").
    """
    H, W = grid_lat.shape
    eps = 1e-9

    # ------------------------------------------------------------------ #
    # 1. IDW depth prior from all known soundings
    # ------------------------------------------------------------------ #
    grid_flat_lat = grid_lat.ravel()  # (H*W,)
    grid_flat_lon = grid_lon.ravel()

    # Chunk size for memory-safe vectorised distance computation
    chunk_size = 4096

    N = len(depths_known)
    idw_surface = np.full(H * W, np.nan, dtype=np.float64)
    idw_valid = np.zeros(H * W, dtype=bool)

    if N > 0:
        pt_lat = coords[:, 0].astype(np.float64)
        pt_lon = coords[:, 1].astype(np.float64)
        dep = depths_known.astype(np.float64)

        # Vectorised IDW: process in chunks to avoid O(H*W*N) memory blow-up
        n_grid = H * W
        lat_mid = float(np.mean(pt_lat))
        m_per_deg_lat = 111000.0
        m_per_deg_lon = 111000.0 * np.cos(np.radians(lat_mid))
        # IDW search radius: 10 km
        search_deg_lat = 10000.0 / m_per_deg_lat
        search_deg_lon = 10000.0 / m_per_deg_lon

        for start in range(0, n_grid, chunk_size):
            end = min(start + chunk_size, n_grid)
            glat = grid_flat_lat[start:end][:, None]  # (chunk, 1)
            glon = grid_flat_lon[start:end][:, None]

            dlat = (glat - pt_lat[None, :])   # (chunk, N)
            dlon = (glon - pt_lon[None, :])

            dist_m = np.sqrt(
                (dlat * m_per_deg_lat) ** 2
                + (dlon * m_per_deg_lon) ** 2
            )  # (chunk, N)

            # IDW p=2; add eps to avoid division by zero at exact matches
            weights = 1.0 / (dist_m ** 2 + eps)

            # Only use points within 10 km
            in_radius = dist_m < 10000.0  # (chunk, N)
            weights = np.where(in_radius, weights, 0.0)

            w_sum = weights.sum(axis=1)  # (chunk,)
            valid_chunk = w_sum > 0
            idw_valid[start:end] = valid_chunk
            idw_surface[start:end] = np.where(
                valid_chunk,
                (weights * dep[None, :]).sum(axis=1) / (w_sum + eps),
                np.nan,
            )

    idw_surface_2d = idw_surface.reshape(H, W).astype(np.float32)
    idw_valid_2d = idw_valid.reshape(H, W)

    # Gaussian smoothing on valid pixels only (avoid NaN propagation)
    if idw_valid_2d.any():
        filled = np.where(idw_valid_2d, idw_surface_2d, 0.0)
        smoothed_num = gaussian_filter(filled, sigma=smooth_sigma_px)
        smoothed_den = gaussian_filter(idw_valid_2d.astype(np.float32), sigma=smooth_sigma_px)
        smoothed_den = np.maximum(smoothed_den, eps)
        idw_smooth = (smoothed_num / smoothed_den).astype(np.float32)
        idw_smooth = np.where(idw_valid_2d, idw_smooth, np.nan).astype(np.float32)
    else:
        idw_smooth = idw_surface_2d

    # ------------------------------------------------------------------ #
    # 2. ICESat-2 spatial prior (laser altimetry anchor, NOT radar)
    # ------------------------------------------------------------------ #
    icesat2_prior: Optional[np.ndarray] = None
    if (
        icesat2_depths is not None
        and icesat2_lats is not None
        and icesat2_lons is not None
        and len(icesat2_depths) > 0
    ):
        M = len(icesat2_depths)
        is_lats = icesat2_lats.astype(np.float64)
        is_lons = icesat2_lons.astype(np.float64)
        is_deps = icesat2_depths.astype(np.float64)

        lat_mid_is = float(np.mean(is_lats))
        mpl = 111000.0
        mplat = 111000.0 * np.cos(np.radians(lat_mid_is))

        is_prior_flat = np.full(H * W, np.nan, dtype=np.float64)
        is_weight_flat = np.zeros(H * W, dtype=np.float64)

        # Gaussian-weighted: sigma = 200 m for ICESat-2 tracks (~1 m footprint)
        sigma_m = 200.0

        for start in range(0, H * W, chunk_size):
            end = min(start + chunk_size, H * W)
            glat = grid_flat_lat[start:end][:, None]
            glon = grid_flat_lon[start:end][:, None]

            dlat = (glat - is_lats[None, :]) * mpl
            dlon = (glon - is_lons[None, :]) * mplat

            dist2 = dlat ** 2 + dlon ** 2
            gaus_w = np.exp(-dist2 / (2.0 * sigma_m ** 2))

            # Only use ICESat-2 within 2 km
            in_rad = dist2 < (2000.0 ** 2)
            gaus_w = np.where(in_rad, gaus_w, 0.0)

            w_sum = gaus_w.sum(axis=1)
            valid_chunk = w_sum > 1e-12
            is_prior_flat[start:end] = np.where(
                valid_chunk,
                (gaus_w * is_deps[None, :]).sum(axis=1) / (w_sum + eps),
                np.nan,
            )

        icesat2_prior_2d = is_prior_flat.reshape(H, W).astype(np.float32)

        # Smooth the ICESat-2 tracks (they are 1D; interpolated gaps are noisy)
        is_valid = np.isfinite(icesat2_prior_2d)
        if is_valid.any():
            filled_is = np.where(is_valid, icesat2_prior_2d, 0.0)
            sm_num = gaussian_filter(filled_is, sigma=smooth_sigma_px * 2)
            sm_den = gaussian_filter(is_valid.astype(np.float32), sigma=smooth_sigma_px * 2)
            sm_den = np.maximum(sm_den, eps)
            icesat2_prior = (sm_num / sm_den).astype(np.float32)
            icesat2_prior = np.where(is_valid, icesat2_prior, np.nan).astype(np.float32)
        else:
            icesat2_prior = icesat2_prior_2d

        logger.info(
            "deepwater_baseline_prior: ICESat-2 anchor: %d points, "
            "%.1f%% grid pixels covered. "
            "LABEL: ICESat-2 photon-counting laser altimetry; NOT radar altimetry.",
            M, 100.0 * np.isfinite(icesat2_prior).mean(),
        )

    # ------------------------------------------------------------------ #
    # 3. R_infinity deep-water reflectance floor (Lyzenga 1978)
    # ------------------------------------------------------------------ #
    r_inf: Dict[str, float] = {}
    if reflectance_cube is not None and reflectance_cube.ndim == 3:
        if water_mask is None:
            water_mask = np.ones((H, W), dtype=bool)
        # Deep pixels: NDWI proxy using ch0 (B2) and ch3 (B8) from cube
        # Use channel indices matching build_feature_cube A0:
        #   ch0=B2, ch1=B3, ch3=B8
        if reflectance_cube.shape[2] >= 4:
            green = reflectance_cube[:, :, 1].astype(np.float64)
            nir   = reflectance_cube[:, :, 3].astype(np.float64)
            ndwi_proxy = (green - nir) / (green + nir + 1e-6)
            deep_mask = (ndwi_proxy > 0.3) & water_mask
            if deep_mask.sum() < 50:
                deep_mask = water_mask

            band_names = {
                4: "B2_dg", 5: "B3_dg",
                6: "ln_B2", 7: "ln_B3", 8: "ln_B4",
            }
            for ch_idx, bname in band_names.items():
                if ch_idx < reflectance_cube.shape[2]:
                    band = reflectance_cube[:, :, ch_idx]
                    vals = band[deep_mask] if deep_mask.sum() >= 20 else band.ravel()
                    r_inf[bname] = float(np.percentile(vals, deep_percentile))

            logger.info(
                "deepwater_baseline_prior: R_infinity (deep_pct=%.1f%%): %s",
                deep_percentile, {k: f"{v:.5f}" for k, v in r_inf.items()},
            )

    optical_depth_limit = 1.5 * secchi_depth_m

    logger.info(
        "deepwater_baseline_prior: IDW prior %.1f%% valid, "
        "optical saturation limit ~%.1f m (1.5 × Secchi=%.1f m). "
        "ICESat-2 anchor: %s",
        100.0 * idw_valid_2d.mean(),
        optical_depth_limit, secchi_depth_m,
        "provided" if icesat2_prior is not None else "NOT provided — deep anchor absent",
    )

    return {
        "idw_depth_prior": idw_smooth,
        "idw_valid_mask": idw_valid_2d,
        "icesat2_spatial_prior": icesat2_prior,
        "r_infinity_floor": r_inf,
        "optical_depth_limit_m": optical_depth_limit,
        "altimetry_anchor_label": (
            "ICESat-2 photon-counting 532 nm laser altimetry; "
            "NOT radar altimetry (radar waveform retracking cannot resolve "
            "individual coastal depth soundings at 10–30 m pixel scale)."
        ),
    }


# ---------------------------------------------------------------------------
# Optical indices
# ---------------------------------------------------------------------------

def compute_optical_indices(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    coastal_aerosol: Optional[np.ndarray] = None,
    swir: Optional[np.ndarray] = None,
    deep_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute a stack of optical indices for SDB feature engineering.

    Returns an (H, W, C) float32 cube and a list of channel names.

    Indices computed
    ----------------
    1.  NDWI (McFeeters 1996): ``(Green - NIR) / (Green + NIR)``
        Water index; values > 0 indicate open water.  Bounded [-1, 1].

    2.  MNDWI (Xu 2006): ``(Green - SWIR) / (Green + SWIR)``
        Modified NDWI; SWIR better discriminates water from built surfaces.
        Only computed when ``swir`` is provided; otherwise set to NDWI.

    3.  Stumpf BG log-ratio (Stumpf et al. 2003):
        ``ln(1000 * Blue) / ln(1000 * Green)``
        Empirical depth proxy; ratio of log-transformed reflectance.  The
        1000 multiplier avoids log(<1) and is standard in coastal remote
        sensing (Stumpf 2003, Coastal Management 31:1, 23–36).

    4.  Stumpf GR log-ratio: ``ln(1000 * Green) / ln(1000 * Red)``
        Useful for shallower waters where blue is less penetrative.

    5.  Lyzenga ln_Blue: ``ln(Blue - Blue_deep)``
        Lyzenga (1985) linearised log-band, referenced to deep-water
        background.  This removes the additive water-column reflectance
        so that the signal is proportional to depth.

    6.  Lyzenga ln_Green: ``ln(Green - Green_deep)``

    7.  Lyzenga ln_Red: ``ln(Red - Red_deep)``
        Red penetrates less than blue/green but is useful at < 2 m.

    8.  Coastal-aerosol ratio: ``CA / Blue`` (B1/B2).
        B1 (~443 nm) is more sensitive to dissolved organic matter and
        very shallow bottom in clear water.  Band-pass differences between
        S2 and L8 are labelled in the meta output.

    9.  NDTI (Lacaux et al. 2007): ``(Red - Green) / (Red + Green)``
        Turbidity index; high NDTI = high sediment load = low Secchi depth.

    Parameters
    ----------
    blue, green, red, nir : np.ndarray, shape (H, W)
        Surface reflectance bands in [0, ~0.5], float32.  Values are
        clipped to [1e-6, 0.5] internally.
    coastal_aerosol : np.ndarray or None, shape (H, W)
        S2 B1 or L8 B1.  If None, CA/Blue ratio is set to 1.0 everywhere.
    swir : np.ndarray or None, shape (H, W)
        Short-wave infrared band (S2 B11 or L8 B6) for MNDWI.
        If None, MNDWI falls back to NDWI.
    deep_mask : np.ndarray or None, shape (H, W) bool
        Pixels used to estimate R_infinity (deep-water background).
        If None, estimated as NDWI > 0.3.

    Returns
    -------
    cube : np.ndarray, shape (H, W, 9), float32
        Channel stack in the order listed above.
    names : list[str], length 9
        Channel names: ``["NDWI","MNDWI","stumpf_BG","stumpf_GR",
        "ln_B2","ln_B3","ln_B4","CA_ratio","NDTI"]``.
    """
    eps = 1e-6

    blue  = np.clip(blue.astype(np.float64),  eps, 0.5)
    green = np.clip(green.astype(np.float64), eps, 0.5)
    red   = np.clip(red.astype(np.float64),   eps, 0.5)
    nir   = np.clip(nir.astype(np.float64),   eps, 0.5)

    H, W = blue.shape

    # Deep-water background (R_infinity per Lyzenga 1978, 1985)
    if deep_mask is None:
        ndwi_est = (green - nir) / (green + nir + eps)
        deep_mask = ndwi_est > 0.3
        if deep_mask.sum() < 50:
            deep_mask = np.ones((H, W), dtype=bool)

    def _pct2(band: np.ndarray) -> float:
        vals = band[deep_mask] if deep_mask.sum() >= 10 else band.ravel()
        return float(np.percentile(vals, 2.0))

    b_deep = _pct2(blue)
    g_deep = _pct2(green)
    r_deep = _pct2(red)

    # 1. NDWI (McFeeters 1996) — guaranteed in [-1, 1]
    ndwi = (green - nir) / (green + nir + eps)

    # 2. MNDWI (Xu 2006) — falls back to NDWI if SWIR absent
    if swir is not None:
        swir_c = np.clip(swir.astype(np.float64), eps, 0.5)
        mndwi = (green - swir_c) / (green + swir_c + eps)
    else:
        mndwi = ndwi.copy()

    # 3–4. Stumpf log-ratios (Stumpf et al. 2003)
    stumpf_bg = np.log(1000.0 * blue)  / (np.log(1000.0 * green) + eps)
    stumpf_gr = np.log(1000.0 * green) / (np.log(1000.0 * red)   + eps)

    # 5–7. Lyzenga log-bands (Lyzenga 1985)
    ln_b2 = np.log(np.clip(blue  - b_deep, eps, None))
    ln_b3 = np.log(np.clip(green - g_deep, eps, None))
    ln_b4 = np.log(np.clip(red   - r_deep, eps, None))

    # 8. Coastal-aerosol ratio
    if coastal_aerosol is not None:
        ca = np.clip(coastal_aerosol.astype(np.float64), eps, 0.5)
        ca_ratio = ca / (blue + eps)
    else:
        ca_ratio = np.ones((H, W), dtype=np.float64)

    # 9. NDTI (Lacaux et al. 2007)
    ndti = (red - green) / (red + green + eps)

    arrays = [ndwi, mndwi, stumpf_bg, stumpf_gr, ln_b2, ln_b3, ln_b4, ca_ratio, ndti]
    names = [
        "NDWI", "MNDWI",
        "stumpf_BG", "stumpf_GR",
        "ln_B2", "ln_B3", "ln_B4",
        "CA_ratio",
        "NDTI",
    ]

    cube = np.stack(arrays, axis=-1).astype(np.float32)  # (H, W, 9)
    return cube, names


def compute_temporal_stats(
    time_series: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_valid_scenes: int = 2,
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-pixel temporal statistics from a multi-date reflectance stack.

    Scientific rationale: temporal median
    --------------------------------------
    Sunglint, thin cloud, and aerosol contamination appear as positive
    radiance spikes in individual scenes.  Because:
    (a) glint occurs at specular-reflection angles that vary with wind and
        solar geometry — quasi-random across independent acquisition dates,
    (b) cloud shadow and thin cirrus are similarly date-specific,

    the **temporal median** over N >= 3 independent dates suppresses these
    artefacts far more effectively than a single-date acquisition (Lyzenga
    et al. 2006; Joyce et al. 2024).  For N=3 scenes the median is the
    middle value; for N=5+ it approaches the cloud- and glint-free
    bottom-of-atmosphere reflectance.  The per-pixel temporal **variance**
    quantifies residual instability (high variance ≈ persistent cloud,
    unresolved glint, or dynamic seafloor change) and can be used as a
    per-pixel uncertainty weight.  The valid-scene **count** flags pixels
    with insufficient temporal sampling.

    Parameters
    ----------
    time_series : np.ndarray, shape (T, H, W) or (T, H, W, C)
        Multi-date reflectance stack.  T = number of acquisition dates.
        Values should be in surface reflectance units [0, ~0.5].
    valid_mask : np.ndarray or None, shape (T, H, W) bool
        Per-date per-pixel validity mask (e.g. SCL water/cloud mask from
        GEE).  If None, all pixels in ``time_series`` are treated as valid.
    min_valid_scenes : int
        Minimum number of valid dates required for a pixel to have a finite
        output.  Default 2.  Pixels with fewer valid dates return NaN.

    Returns
    -------
    stats_cube : np.ndarray, shape (H, W, 3*C) or (H, W, 3) if T,H,W input
        For each band channel C, the output stack contains:
        - ``median[c]`` : temporal median (glint/cloud-robust central value)
        - ``variance[c]`` : temporal variance (instability indicator)
        - ``valid_count[c]`` : number of valid scenes per pixel (float32)
        Channel order: [med_ch0, var_ch0, cnt_ch0, med_ch1, var_ch1, cnt_ch1, ...]
    names : list[str]
        Channel names in the same order as ``stats_cube`` last axis.

    Notes
    -----
    - For SDB, feed the ``median`` channels into the feature cube as if they
      were single-scene SR bands.  Use ``variance`` as a per-pixel input
      uncertainty weight in the loss function.
    - A temporal stack of N=3–5 Sentinel-2 scenes covers ~15–25 days, which
      introduces sub-tidal depth variation (~0.1–0.3 m RMS in UAE Gulf
      waters).  Apply ``tidal_correction.normalize_scenes_to_datum()`` before
      stacking if tidal range exceeds ~0.3 m at the site.
    """
    ts = time_series.astype(np.float32)
    is_4d = ts.ndim == 4

    if ts.ndim == 3:
        # (T, H, W) -> add channel dim
        ts = ts[:, :, :, np.newaxis]  # (T, H, W, 1)

    T, H, W, C = ts.shape

    if valid_mask is None:
        vm = np.ones((T, H, W), dtype=bool)
    else:
        vm = valid_mask.astype(bool)
        assert vm.shape == (T, H, W), (
            f"valid_mask shape {vm.shape} != ({T},{H},{W})"
        )

    # Broadcast valid_mask to cover channels
    vm_c = vm[:, :, :, np.newaxis]  # (T, H, W, 1) -> broadcast to (T,H,W,C)

    # Mask invalid pixels as NaN
    ts_masked = np.where(vm_c, ts, np.nan)  # (T, H, W, C)

    # Valid count: max over C (same mask per channel assumed)
    valid_count = vm.astype(np.float32).sum(axis=0)  # (H, W)

    # Temporal median and variance using nan-aware numpy
    with np.errstate(all="ignore"):
        temp_median = np.nanmedian(ts_masked, axis=0)  # (H, W, C)
        temp_var    = np.nanvar(ts_masked, axis=0)     # (H, W, C)

    # Set pixels with fewer than min_valid_scenes to NaN
    insufficient = valid_count < min_valid_scenes  # (H, W)
    temp_median[insufficient] = np.nan
    temp_var[insufficient]    = np.nan

    # Build interleaved output: [med_c0, var_c0, cnt_c0, med_c1, ...]
    out_channels = []
    out_names: List[str] = []
    for c in range(C):
        out_channels.append(temp_median[:, :, c])
        out_channels.append(temp_var[:, :, c])
        out_channels.append(valid_count)  # same count for all channels (same mask)
        label = str(c) if is_4d else "0"
        out_names += [f"temporal_median_ch{label}", f"temporal_var_ch{label}", "temporal_valid_count"]

    stats_cube = np.stack(out_channels, axis=-1).astype(np.float32)  # (H, W, 3*C)
    return stats_cube, out_names
