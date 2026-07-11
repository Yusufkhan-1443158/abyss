"""
Wave-Dispersion Bathymetry — Almar et al. (S2Shores physics)
═════════════════════════════════════════════════════════════
Derives water depth from wave celerity measured via Sentinel-2
inter-band time offsets using the linear dispersion relation:

    ω² = g·k·tanh(k·h)  →  h = atanh(γ) / k
    where γ = 2π·c² / (g·L)

Sentinel-2 MSI acquires B02 (blue) and B04 (red) ~1.005s apart.
Waves propagating shoreward shift between acquisitions. By measuring
this phase shift via cross-spectral analysis (FFT), we recover
wavelength L and celerity c, then invert for depth h.

References:
  - Almar et al. (2024), Coastal Engineering 189, 104458
  - Bergsma et al. (2019), Remote Sensing 11, 1918
  - Binet et al. (2022), ISPRS Annals V-1-2022, 57-66

Uses only numpy + scipy — no external packages.
"""
from __future__ import annotations
import logging, math
import numpy as np

L = logging.getLogger("bathy.wave")

# ── Physical constants ─────────────────────────────────────
GRAVITY = 9.81                     # m/s²
DELTA_T_S2A = 1.005                # seconds between B02 and B04 (S2A)
DELTA_T_S2B = 1.005                # S2B — same detector layout
MAX_DEPTH_M = 30.0                 # SDB cap (raised for 0-30 m regions like Khalifa)

# ── Wave validity constraints (from S2Shores) ──────────────
MIN_PERIOD = 3.0                   # seconds
MAX_PERIOD = 25.0
# Phase-wrap ambiguity limit: c·Δt < λ/2 ⇒ λ > 2·c_max·Δt.
# For c_max = 17 m/s (√(g·30m)) and Δt = 1.005 s ⇒ λ > 34 m.
# Rounding up gives a safe floor that also rejects short-wind-sea ripples.
MIN_WAVELENGTH_M = 40.0
MAX_WAVELENGTH_M = 600.0
MIN_LINEARITY = 0.2                # γ = tanh(kh): small ⇒ shallow/breaking regime
MAX_LINEARITY = 0.95               # γ→1 ⇒ deep-water limit, depth undetermined
MIN_CELERITY = 1.0                 # m/s
MAX_CELERITY = 25.0


# ═══════════════════════════════════════════════════════════
# 1. PREPROCESSING — isolate wave signal from reflectance
# ═══════════════════════════════════════════════════════════

def _detrend_2d(band: np.ndarray) -> np.ndarray:
    """Remove large-scale trend (land/cloud gradients) via linear detrend."""
    H, W = band.shape
    y = np.arange(H, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)
    Y, X = np.meshgrid(y, x, indexing='ij')
    A = np.column_stack([X.ravel(), Y.ravel(), np.ones(H * W)])
    b = band.ravel().astype(np.float64)
    valid = np.isfinite(b)
    if valid.sum() < 100:
        return band.astype(np.float64)
    try:
        coef, _, _, _ = np.linalg.lstsq(A[valid], b[valid], rcond=None)
        trend = (A @ coef).reshape(H, W)
        return band.astype(np.float64) - trend
    except Exception:
        return band.astype(np.float64)


def _highpass_filter(band: np.ndarray, cutoff_px: int = 80) -> np.ndarray:
    """FFT-based high-pass to remove low-frequency background."""
    from scipy.fft import fft2, ifft2, fftfreq
    F = fft2(np.nan_to_num(band, nan=0.0))
    H, W = band.shape
    fy = fftfreq(H)
    fx = fftfreq(W)
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    freq = np.sqrt(FX ** 2 + FY ** 2)
    fc = 1.0 / cutoff_px
    # Butterworth high-pass, order=2
    filt = 1.0 / (1.0 + (fc / (freq + 1e-10)) ** 4)
    filt[0, 0] = 0  # kill DC
    return np.real(ifft2(F * filt))


def _preprocess_band(band: np.ndarray) -> np.ndarray:
    """Detrend + high-pass filter a single S2 band for wave analysis."""
    out = _detrend_2d(band)
    out = _highpass_filter(out)
    # Normalise to zero-mean unit-variance
    std = np.std(out)
    if std > 1e-10:
        out = (out - np.mean(out)) / std
    return out


# ═══════════════════════════════════════════════════════════
# LAND MASK + DISTANCE-TO-SHORE
# ═══════════════════════════════════════════════════════════
# Mirrors the GDAL reference pipeline used for Sentinel-2 bathymetry:
#   gdal.Translate + gdal.Warp reproject a land raster into the scene
#   grid, then `gdal_proximity.py -values 0` produces a distance-to-shore
#   raster. Here we do it with global_land_mask + B04 reflectance +
#   scipy.ndimage.distance_transform_edt — same maths, no GDAL dep.

def build_wave_water_mask(
    b02: np.ndarray,
    b04: np.ndarray,
    bbox: list,
    extra_erode_px: int = 1,
) -> np.ndarray:
    """Binary water=True mask on the S2 wave grid.

    Uses two independent land sources and ANDs the "not land" result:
      1. `global_land_mask` — coarse (~1 km) global coastline polygon
      2. B04 (red) reflectance DN — land/buildings are bright in red;
         water is dark. Adaptive percentile threshold with a 1500 DN
         floor handles both L1C TOA and L2A BOA inputs.

    Morphological erode shrinks the water footprint slightly so we do
    not admit mixed-pixel shoreline cells into the FFT window.
    """
    H, W = b02.shape
    w, s, e, n = bbox
    water = np.ones((H, W), dtype=bool)

    try:
        from global_land_mask import globe
        lats = np.linspace(n, s, H)
        lons = np.linspace(w, e, W)
        lon_g, lat_g = np.meshgrid(lons, lats)
        water &= ~globe.is_land(lat_g, lon_g)
        L.info(f"Wave water-mask (global): {int((~water).sum())} land px")
    except Exception as ex:
        L.warning(f"global_land_mask failed: {ex}")

    # Image-based refinement — catches breakwaters / ports / reclaimed
    # land that the 1 km coastline mask misses.
    b04_f = b04.astype(np.float32)
    if np.isfinite(b04_f).any():
        p65 = float(np.percentile(b04_f[np.isfinite(b04_f)], 65))
        thr = max(p65, 1500.0)
        land_img = b04_f > thr
        water &= ~land_img
        L.info(f"Wave water-mask (B04>{thr:.0f}): +{int(land_img.sum())} land px")

    try:
        from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
        struct = np.ones((3, 3))
        # Fill interior holes in the land mask (small spurious water speckle on land)
        land = ~water
        land = binary_fill_holes(land)
        water = ~land
        if extra_erode_px > 0:
            water = binary_erosion(water, struct, iterations=extra_erode_px)
    except Exception as ex:
        L.warning(f"mask cleanup: {ex}")

    n_water = int(water.sum())
    L.info(f"Wave water-mask final: {n_water}/{H*W} water "
           f"({100 - 100*n_water//(H*W)}% land)")
    return water


def distance_to_shore_m(water_mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Euclidean distance (metres) from each water pixel to nearest land.

    Equivalent to `gdal_proximity.py -values 0` on a land-coded raster.
    Land pixels return 0.
    """
    from scipy.ndimage import distance_transform_edt
    # distance_transform_edt measures distance to nearest 0 — feeding the
    # water mask (water=1, land=0) therefore yields distance from every
    # water pixel to the nearest land pixel.
    dist_px = distance_transform_edt(water_mask.astype(np.uint8))
    return (dist_px * float(pixel_size_m)).astype(np.float32)


# ═══════════════════════════════════════════════════════════
# 2. RADON TRANSFORM — detect dominant wave direction
# ═══════════════════════════════════════════════════════════

def _radon_variance(img: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """
    Simplified Radon: for each angle, project image along that direction
    and compute variance of the 1D projection. Peak variance = wave direction.
    """
    from scipy.ndimage import rotate
    H, W = img.shape
    variances = np.zeros(len(angles_deg))
    for i, angle in enumerate(angles_deg):
        rotated = rotate(img, angle, reshape=False, order=1, mode='constant', cval=0)
        projection = np.sum(rotated, axis=0)  # collapse rows
        variances[i] = np.var(projection)
    return variances


def _detect_wave_direction_fft(b02: np.ndarray, b04: np.ndarray,
                               min_wavelength_m: float = MIN_WAVELENGTH_M,
                               max_wavelength_m: float = MAX_WAVELENGTH_M,
                               pixel_size_m: float = 10.0) -> float:
    """
    Fast 2-D FFT direction detection.

    Builds the 2-D power spectrum of the averaged bands, zeros-out the
    DC cross and all bins outside the physically valid wavelength band,
    then picks the dominant (kx, ky) and returns its angle. Equivalent
    to a Radon variance peak but ~100× faster — no per-angle rotation.

    Returns angle in degrees (0 = horizontal propagation, 90 = vertical).
    """
    combined = (b02 + b04) / 2.0
    H, W = combined.shape
    F = np.fft.fft2(np.nan_to_num(combined, nan=0.0))
    P = np.abs(F) ** 2
    # Centred spatial frequencies (cycles per pixel); convert to m⁻¹.
    fy = np.fft.fftfreq(H) / pixel_size_m
    fx = np.fft.fftfreq(W) / pixel_size_m
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    freq_mag = np.sqrt(FX ** 2 + FY ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        wl = np.where(freq_mag > 0, 1.0 / freq_mag, 0.0)
    valid = (wl >= min_wavelength_m) & (wl <= max_wavelength_m)
    P_valid = np.where(valid, P, 0.0)
    if P_valid.max() <= 0:
        return 0.0
    iy, ix = np.unravel_index(np.argmax(P_valid), P_valid.shape)
    # atan2(ky, kx) gives angle of wavenumber vector (propagation direction).
    direction = float(np.degrees(np.arctan2(FY[iy, ix], FX[iy, ix])))
    # Fold to [0, 180) — waves are bidirectional in the power spectrum.
    if direction < 0:
        direction += 180.0
    if direction >= 180.0:
        direction -= 180.0
    return direction


def _detect_wave_direction(b02: np.ndarray, b04: np.ndarray,
                           pixel_size_m: float = 10.0) -> float:
    """Fast FFT path (default). Falls back to Radon variance on failure."""
    try:
        return _detect_wave_direction_fft(b02, b04, pixel_size_m=pixel_size_m)
    except Exception as ex:
        L.warning(f"FFT direction failed ({ex}); falling back to Radon")
        from scipy.ndimage import rotate  # noqa: F401 — import-on-demand
        combined = (b02 + b04) / 2.0
        angles = np.arange(0, 180, 1.0)
        var = _radon_variance(combined, angles)
        peak_idx = int(np.argmax(var))
        direction = float(angles[peak_idx])
        if 1 <= peak_idx <= len(angles) - 2:
            v0, v1, v2 = var[peak_idx - 1], var[peak_idx], var[peak_idx + 1]
            denom = 2.0 * (2 * v1 - v0 - v2)
            if abs(denom) > 1e-10:
                direction += (v0 - v2) / denom
        return direction


# ═══════════════════════════════════════════════════════════
# 3. SINOGRAM EXTRACTION along wave direction
# ═══════════════════════════════════════════════════════════

def _extract_sinogram(band: np.ndarray, direction_deg: float) -> np.ndarray:
    """
    Extract 1D sinogram (profile) along the detected wave direction
    by rotating the image so waves are vertical, then averaging columns.
    """
    from scipy.ndimage import rotate
    rotated = rotate(band, direction_deg, reshape=True, order=1, mode='constant', cval=0)
    # Sum along rows → 1D profile perpendicular to wave crests
    profile = np.mean(rotated, axis=0)
    # Trim zero-padded edges from rotation
    nonzero = np.where(np.abs(profile) > 1e-8)[0]
    if len(nonzero) > 10:
        profile = profile[nonzero[0]:nonzero[-1] + 1]
    return profile


# ═══════════════════════════════════════════════════════════
# 4. CROSS-SPECTRAL ANALYSIS — wavelength + celerity
# ═══════════════════════════════════════════════════════════

def _cross_spectral_celerity(
    sino_b02: np.ndarray,
    sino_b04: np.ndarray,
    pixel_size_m: float,
    delta_t: float = DELTA_T_S2A,
) -> tuple:
    """
    Compute wavelength and celerity from cross-spectral phase shift.

    Returns: (wavelength_m, celerity_m_s, period_s, peak_energy)
             or (None, None, None, 0) if no valid wave detected.
    """
    n = min(len(sino_b02), len(sino_b04))
    sino_b02 = sino_b02[:n]
    sino_b04 = sino_b04[:n]

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    s1 = sino_b02 * window
    s2 = sino_b04 * window

    # FFT
    F1 = np.fft.rfft(s1)
    F2 = np.fft.rfft(s2)
    freqs = np.fft.rfftfreq(n, d=pixel_size_m)  # spatial frequency (1/m)

    # Cross-spectrum
    cross = F1 * np.conj(F2)
    power = np.abs(F1) ** 2 + np.abs(F2) ** 2  # combined power

    # Ignore DC and very low frequencies (> 500m wavelength)
    min_freq = 1.0 / MAX_WAVELENGTH_M
    max_freq = 1.0 / MIN_WAVELENGTH_M
    valid = (freqs > min_freq) & (freqs < max_freq)

    if valid.sum() < 3:
        return None, None, None, 0

    # Find spectral peak(s) in valid range — not just the global argmax.
    # A long swell often dominates a shallow-water scene and masks the
    # wind-sea peak at shorter wavelengths that actually resolves the
    # bottom. We iterate candidate peaks shortest-λ first and return the
    # first one that yields a valid γ ∈ [MIN_LINEARITY, MAX_LINEARITY]
    # depth inversion below.
    power_valid = power.copy()
    power_valid[~valid] = 0
    global_peak_idx = int(np.argmax(power_valid))
    peak_energy_global = float(power_valid[global_peak_idx])

    if peak_energy_global < 1e-6:
        return None, None, None, 0

    # Scan for local maxima; accept any peak with ≥ MULTI_PEAK_MIN_FRAC
    # of the global peak energy so we don't chase noise floors.
    MULTI_PEAK_MIN_FRAC = 0.15
    thr = peak_energy_global * MULTI_PEAK_MIN_FRAC
    candidate_idx = []
    for i in range(1, len(power_valid) - 1):
        if not valid[i]:
            continue
        if (power_valid[i] > thr and
                power_valid[i] >= power_valid[i - 1] and
                power_valid[i] >= power_valid[i + 1]):
            candidate_idx.append(i)
    if not candidate_idx:
        candidate_idx = [global_peak_idx]
    # Shortest wavelength first (= highest frequency first). Short waves
    # see shallower bottoms, which is where we currently lose coverage.
    candidate_idx.sort(key=lambda i: -freqs[i])

    # Try each candidate peak (shortest λ first). Return the first one
    # whose celerity/period pass the validity gates AND whose γ falls in
    # the inversion-valid range — that check is done by the caller's
    # dispersion step, but we pre-screen here for MIN/MAX celerity & period.
    for peak_idx in candidate_idx:
        peak_energy = float(power_valid[peak_idx])

        # Wavelength from peak frequency — parabolic sub-pixel refinement.
        if 1 <= peak_idx < len(freqs) - 1:
            p0, p1, p2 = float(power[peak_idx - 1]), float(power[peak_idx]), float(power[peak_idx + 1])
            denom = (p0 - 2.0 * p1 + p2)
            offset = 0.5 * (p0 - p2) / denom if abs(denom) > 1e-12 else 0.0
            offset = max(-0.5, min(0.5, offset))
        else:
            offset = 0.0
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
        peak_freq = float(freqs[peak_idx]) + offset * df
        wavelength = 1.0 / (peak_freq + 1e-10)  # metres

        if wavelength < MIN_WAVELENGTH_M or wavelength > MAX_WAVELENGTH_M:
            continue

        # Phase at peak — interpolate to match sub-pixel λ offset.
        if 1 <= peak_idx < len(cross) - 1 and abs(offset) > 1e-3:
            if offset >= 0:
                phase_shift = (1 - offset) * np.angle(cross[peak_idx]) + offset * np.angle(cross[peak_idx + 1])
            else:
                phase_shift = (1 + offset) * np.angle(cross[peak_idx]) + (-offset) * np.angle(cross[peak_idx - 1])
        else:
            phase_shift = np.angle(cross[peak_idx])

        # Phase-wrap guard — at |phase|≈π the celerity is aliased.
        if abs(phase_shift) > 0.85 * np.pi:
            continue

        displacement = wavelength * phase_shift / (2 * np.pi)
        celerity_signed = displacement / delta_t
        celerity = abs(celerity_signed)

        if celerity < MIN_CELERITY or celerity > MAX_CELERITY:
            continue

        period = wavelength / celerity if celerity > 0 else 0
        if period < MIN_PERIOD or period > MAX_PERIOD:
            continue

        # Pre-screen with γ so we reject peaks whose depth inversion
        # would fail anyway — this lets the shortest-λ candidate carry
        # on to the next one if it fails, instead of killing the window.
        gamma_screen = 2.0 * np.pi * celerity ** 2 / (GRAVITY * wavelength)
        if gamma_screen < MIN_LINEARITY or gamma_screen > MAX_LINEARITY:
            continue

        return wavelength, celerity, period, peak_energy

    return None, None, None, 0


# ═══════════════════════════════════════════════════════════
# 5. DISPERSION RELATION INVERSION → depth
# ═══════════════════════════════════════════════════════════

def _dispersion_depth(wavelength: float, celerity: float, g: float = GRAVITY) -> float:
    """
    Invert the linear dispersion relation for depth.

    γ = 2π·c² / (g·L) = tanh(k·h)
      → γ → 0  (kh → 0)   shallow / near-breaking
      → γ → 1  (kh → ∞)   deep water — bottom undetermined
    depth = atanh(γ) · L / (2π)

    Returns depth in metres, or NaN if invalid.
    """
    k = 2.0 * np.pi / wavelength  # wavenumber
    gamma = 2.0 * np.pi * celerity ** 2 / (g * wavelength)

    if gamma < MIN_LINEARITY:
        return np.nan  # too shallow / breaking zone — linear theory fails
    if gamma > MAX_LINEARITY:
        return np.nan  # too deep — waves don't feel the bottom

    try:
        depth = np.arctanh(gamma) / k
    except (ValueError, FloatingPointError):
        return np.nan

    if not np.isfinite(depth) or depth <= 0:
        return np.nan

    return min(float(depth), MAX_DEPTH_M)


# ═══════════════════════════════════════════════════════════
# 6. FULL PIPELINE — sliding window across image
# ═══════════════════════════════════════════════════════════

def wave_bathymetry(
    s2_wave: dict,
    bbox: list,
    window_m: float = 400.0,
    step_m: float = 100.0,
    delta_t: float = DELTA_T_S2A,
    pixel_size_m: float = 10.0,
    water_mask: np.ndarray | None = None,
    dist_map_m: np.ndarray | None = None,
    min_water_frac: float = 0.7,
    min_shore_dist_m: float = 50.0,
    max_shore_dist_m: float = 0.0,
    shore_dist_dir: str | None = None,
    shore_dist_tag: str | None = None,
) -> tuple:
    """
    Full wave-dispersion bathymetry pipeline.

    Parameters
    ----------
    s2_wave : dict with "b02" and "b04" arrays (H, W) — raw single-orbit bands
    bbox : [west, south, east, north]
    window_m : analysis window size in metres
    step_m : sliding window step in metres
    delta_t : inter-band time offset in seconds
    pixel_size_m : pixel size in metres (10 for S2 at 10m)
    water_mask : optional boolean (H,W) mask (water=True). If None, the
        GDAL pipeline in `shore_distance.create_distance_raster()`
        (adapted from the original Bathy.py — gdal_translate / gdalwarp /
        gdal_calc.py / gdal_proximity.py) is used when `shore_dist_dir`
        and `shore_dist_tag` are given. Otherwise the numpy
        `build_wave_water_mask()` fallback runs.
    dist_map_m : optional (H,W) shore-distance raster in metres — if
        supplied together with `water_mask`, skips the GDAL call.
    min_water_frac : minimum fraction of water pixels required per window
        (default 0.7 — keep windows that are mostly open water).
    min_shore_dist_m : minimum distance-to-shore (metres) for a window's
        centre to be processed. Sheltered lagoon / inside-breakwater cells
        where waves do not propagate are rejected. Set to 0 to disable.
    max_shore_dist_m : optional upper bound on distance-to-shore (metres).
        If > 0, wave inversion is only attempted within this band — useful
        to focus on the coastal regime where the physics is valid. 0 disables.
    shore_dist_dir, shore_dist_tag : when given, the GDAL pipeline writes
        its intermediate rasters (B02.tiff, mask_clip.tiff, mask_warp.tiff,
        Inter.tiff, Distance.tiff) into `shore_dist_dir` with the prefix
        `shore_dist_tag`. Matches the Bathy.py on-disk convention.

    Returns
    -------
    (result_dict, error_string_or_None)
    result_dict has: depth (H_out, W_out), wavelength, celerity, direction, period grids
    """
    b02_raw = s2_wave["b02"].astype(np.float64)
    b04_raw = s2_wave["b04"].astype(np.float64)
    H, W = b02_raw.shape
    w, s, e, n = bbox

    L.info(f"Wave bathy: {W}x{H} @{pixel_size_m}m, window={window_m}m, step={step_m}m")

    # ── Land mask + distance-to-shore ──
    # Preferred path: the GDAL-CLI pipeline in `shore_distance.py`
    # (adapted from Bathy.py). Falls back to the numpy path if the
    # caller did not provide a scratch dir or GDAL is unavailable.
    dist_map = None
    if water_mask is not None and dist_map_m is not None \
            and water_mask.shape == (H, W) and dist_map_m.shape == (H, W):
        L.info("Wave shore-dist: using caller-supplied mask + distance raster")
        dist_map = dist_map_m.astype(np.float32)
    elif shore_dist_dir is not None and shore_dist_tag is not None:
        try:
            from backend.shore_distance import create_distance_raster
        except ImportError:
            from shore_distance import create_distance_raster
        try:
            L.info(f"Wave shore-dist: running GDAL pipeline (gdal_translate / "
                   f"gdalwarp / gdal_calc.py / gdal_proximity.py) tag={shore_dist_tag}")
            water_mask, dist_map = create_distance_raster(
                s2_wave["b02"], bbox, shore_dist_dir, shore_dist_tag,
                pixel_size_m=pixel_size_m,
            )
        except Exception as ex:
            L.warning(f"GDAL shore-dist failed ({ex}); falling back to numpy path")
            water_mask = None
    if water_mask is None:
        water_mask = build_wave_water_mask(s2_wave["b02"], s2_wave["b04"], bbox)
    if water_mask.shape != (H, W):
        L.warning(f"water_mask shape {water_mask.shape} != {(H, W)} — rebuilding")
        water_mask = build_wave_water_mask(s2_wave["b02"], s2_wave["b04"], bbox)
    if dist_map is None or dist_map.shape != (H, W):
        dist_map = distance_to_shore_m(water_mask, pixel_size_m)
    L.info(f"Wave shore-dist: max={float(dist_map.max()):.0f}m, "
           f"water cells={int(water_mask.sum())}")

    # Zero the raw bands on land BEFORE preprocessing so the detrend /
    # high-pass do not fit to port / building pixels.
    b02_raw = np.where(water_mask, b02_raw, 0.0)
    b04_raw = np.where(water_mask, b04_raw, 0.0)

    # Preprocess both bands
    b02 = _preprocess_band(b02_raw)
    b04 = _preprocess_band(b04_raw)

    win_px = max(16, int(window_m / pixel_size_m))
    # step_px floor was 4 — that silently inflated the output grid to
    # 4·pixel_size (≈80 m for a 20 m request). Honour the caller's
    # step_m down to 1 pixel; the caller is responsible for the
    # performance trade-off.
    step_px = max(1, int(round(step_m / pixel_size_m)))

    # Output grid
    ny_out = max(1, (H - win_px) // step_px + 1)
    nx_out = max(1, (W - win_px) // step_px + 1)

    depth_grid = np.full((ny_out, nx_out), np.nan, dtype=np.float32)
    wave_dir_grid = np.full((ny_out, nx_out), np.nan, dtype=np.float32)
    wavelength_grid = np.full((ny_out, nx_out), np.nan, dtype=np.float32)
    celerity_grid = np.full((ny_out, nx_out), np.nan, dtype=np.float32)
    period_grid = np.full((ny_out, nx_out), np.nan, dtype=np.float32)
    energy_grid = np.full((ny_out, nx_out), np.nan, dtype=np.float32)

    valid_count = 0
    skipped_land = 0
    skipped_sheltered = 0
    skipped_far = 0
    total_windows = ny_out * nx_out

    for iy in range(ny_out):
        y0 = iy * step_px
        y1 = min(y0 + win_px, H)
        for ix in range(nx_out):
            x0 = ix * step_px
            x1 = min(x0 + win_px, W)

            patch_b02 = b02[y0:y1, x0:x1]
            patch_b04 = b04[y0:y1, x0:x1]
            patch_mask = water_mask[y0:y1, x0:x1]

            # Land-fraction gate — reject windows dominated by land / port
            water_frac = float(patch_mask.sum()) / max(patch_mask.size, 1)
            if water_frac < min_water_frac:
                skipped_land += 1
                continue

            # Shore-distance gate — reject sheltered (lagoon / inside-breakwater)
            # windows where the linear dispersion assumption breaks down.
            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2
            d_shore = float(dist_map[cy, cx])
            if min_shore_dist_m > 0 and d_shore < min_shore_dist_m:
                skipped_sheltered += 1
                continue
            if max_shore_dist_m > 0 and d_shore > max_shore_dist_m:
                skipped_far += 1
                continue

            # Skip if too much zero/NaN
            if np.sum(np.abs(patch_b02) > 1e-6) < 0.5 * patch_b02.size:
                continue

            # Detect wave direction in this window (FFT path, ~100× faster)
            try:
                direction = _detect_wave_direction(patch_b02, patch_b04,
                                                   pixel_size_m=pixel_size_m)
            except Exception:
                continue

            # Extract sinograms along wave direction
            sino_b02 = _extract_sinogram(patch_b02, direction)
            sino_b04 = _extract_sinogram(patch_b04, direction)

            if len(sino_b02) < 20 or len(sino_b04) < 20:
                continue

            # Cross-spectral analysis
            wl, cel, per, energy = _cross_spectral_celerity(
                sino_b02, sino_b04, pixel_size_m, delta_t
            )

            if wl is None:
                continue

            # Depth inversion
            depth = _dispersion_depth(wl, cel)
            if np.isfinite(depth) and depth > 0:
                depth_grid[iy, ix] = depth
                wave_dir_grid[iy, ix] = direction
                wavelength_grid[iy, ix] = wl
                celerity_grid[iy, ix] = cel
                period_grid[iy, ix] = per
                energy_grid[iy, ix] = energy
                valid_count += 1

    L.info(f"Wave bathy: {valid_count}/{total_windows} valid "
           f"(skipped land={skipped_land}, sheltered={skipped_sheltered}, "
           f"far={skipped_far})")

    if valid_count < 3:
        return None, (f"Only {valid_count} valid wave depth windows "
                      f"(skipped land={skipped_land}, sheltered={skipped_sheltered})")

    # Downsample water mask to output grid for final masking
    out_water_mask = np.zeros((ny_out, nx_out), dtype=bool)
    out_dist_m = np.zeros((ny_out, nx_out), dtype=np.float32)
    for iy in range(ny_out):
        y0 = iy * step_px
        y1 = min(y0 + win_px, H)
        for ix in range(nx_out):
            x0 = ix * step_px
            x1 = min(x0 + win_px, W)
            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2
            out_water_mask[iy, ix] = bool(water_mask[cy, cx])
            out_dist_m[iy, ix] = float(dist_map[cy, cx])

    # Interpolate gaps using scipy if enough valid points
    try:
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter

        valid_mask = np.isfinite(depth_grid)
        if valid_mask.sum() >= 3:
            ys, xs = np.where(valid_mask)
            vals = depth_grid[valid_mask]
            all_y, all_x = np.meshgrid(
                np.arange(ny_out), np.arange(nx_out), indexing='ij'
            )
            # Linear interpolation to fill gaps
            filled = griddata(
                np.column_stack([ys, xs]), vals,
                np.column_stack([all_y.ravel(), all_x.ravel()]),
                method='linear', fill_value=np.nan
            ).reshape(ny_out, nx_out)

            # Smooth
            filled_clean = np.nan_to_num(filled, nan=0.0)
            smoothed = gaussian_filter(filled_clean, sigma=1.0)
            depth_grid = np.where(np.isfinite(filled), smoothed, np.nan)
            depth_grid = np.clip(depth_grid, 0, MAX_DEPTH_M)
    except Exception as ex:
        L.warning(f"Wave interpolation: {ex}")

    # ── Final mask: force land cells to NaN and honour the shore-dist band ──
    depth_grid = np.where(out_water_mask, depth_grid, np.nan)
    if min_shore_dist_m > 0:
        depth_grid = np.where(out_dist_m >= min_shore_dist_m, depth_grid, np.nan)
    if max_shore_dist_m > 0:
        depth_grid = np.where(out_dist_m <= max_shore_dist_m, depth_grid, np.nan)

    # Stats
    valid_depths = depth_grid[np.isfinite(depth_grid)]
    valid_wl = wavelength_grid[np.isfinite(wavelength_grid)]
    valid_cel = celerity_grid[np.isfinite(celerity_grid)]
    valid_per = period_grid[np.isfinite(period_grid)]

    stats = {
        "method": "Wave Dispersion (Almar et al.)",
        "valid_windows": valid_count,
        "total_windows": total_windows,
        "skipped_land": skipped_land,
        "skipped_sheltered": skipped_sheltered,
        "skipped_far": skipped_far,
        "min_water_frac": min_water_frac,
        "min_shore_dist_m": min_shore_dist_m,
        "max_shore_dist_m": max_shore_dist_m,
        "coverage_pct": round(100.0 * valid_count / max(total_windows, 1), 1),
        "depth_range": [round(float(np.nanmin(valid_depths)), 1),
                        round(float(np.nanmax(valid_depths)), 1)] if len(valid_depths) > 0 else [0, 0],
        "mean_wavelength_m": round(float(np.mean(valid_wl)), 1) if len(valid_wl) > 0 else 0,
        "mean_celerity_ms": round(float(np.mean(valid_cel)), 1) if len(valid_cel) > 0 else 0,
        "mean_period_s": round(float(np.mean(valid_per)), 1) if len(valid_per) > 0 else 0,
        "mean_direction_deg": round(float(np.nanmean(wave_dir_grid[np.isfinite(wave_dir_grid)])), 1) if valid_count > 0 else 0,
        "window_m": window_m,
        "step_m": step_m,
        "delta_t_s": delta_t,
    }

    L.info(f"Wave bathy result: {valid_count} pts, depth=[{stats['depth_range'][0]}-{stats['depth_range'][1]}]m, "
           f"λ={stats['mean_wavelength_m']}m, c={stats['mean_celerity_ms']}m/s, T={stats['mean_period_s']}s")

    return {
        "depth": depth_grid,
        "wavelength": wavelength_grid,
        "celerity": celerity_grid,
        "direction": wave_dir_grid,
        "period": period_grid,
        "energy": energy_grid,
        "water_mask_out": out_water_mask,
        "shore_dist_out_m": out_dist_m,
        "water_mask_src": water_mask,
        "shore_dist_src_m": dist_map,
        "stats": stats,
        "grid_shape": (ny_out, nx_out),
        "pixel_size_m": step_m,  # output pixel = step size
    }, None


# ═══════════════════════════════════════════════════════════
# 7. TILED PARALLEL WAVE BATHYMETRY
# ═══════════════════════════════════════════════════════════
# Splits the input scene into N_x × N_y overlapping pixel tiles,
# runs wave_bathymetry on each in parallel, then mosaics with a
# cosine-squared feather so seams do not show. Mirrors the feathered-
# mosaic pattern used in gpu_mosaic_engine.py; the overlap is expressed
# in metres and converted to pixels from `pixel_size_m`.

def _cosine_feather(H: int, W: int, margin: int) -> np.ndarray:
    """2-D cosine-squared feather: 1 inside the non-overlap core, falling
    smoothly to 0 across the `margin`-pixel overlap band. Cos² blends
    sum to 1 on pairs of adjacent tiles, so mosaic amplitude is preserved."""
    w = np.ones((H, W), dtype=np.float32)
    mH = min(margin, H // 2)
    mW = min(margin, W // 2)
    for i in range(mH):
        f = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / (mH + 1))
        w[i, :] *= f
        w[H - 1 - i, :] *= f
    for j in range(mW):
        f = 0.5 - 0.5 * np.cos(np.pi * (j + 1) / (mW + 1))
        w[:, j] *= f
        w[:, W - 1 - j] *= f
    return w


def _process_wave_tile(tile_idx, s2w_tile, bbox_tile,
                       window_m, step_m, pixel_size_m,
                       delta_t, min_water_frac,
                       min_shore_dist_m, max_shore_dist_m):
    """Worker — runs wave_bathymetry on a single tile. Returns
    (tile_idx, result_dict_or_None, error_str_or_None)."""
    try:
        res, err = wave_bathymetry(
            s2w_tile, bbox_tile,
            window_m=window_m, step_m=step_m,
            pixel_size_m=pixel_size_m, delta_t=delta_t,
            min_water_frac=min_water_frac,
            min_shore_dist_m=min_shore_dist_m,
            max_shore_dist_m=max_shore_dist_m,
        )
        return (tile_idx, res, err)
    except Exception as ex:
        return (tile_idx, None, f"{type(ex).__name__}: {ex}")


def tiled_wave_bathymetry(
    s2w: dict,
    bbox: list,
    *,
    tile_nx: int = 4,
    tile_ny: int = 4,
    overlap_m: float = 400.0,
    window_m: float = 800.0,
    step_m: float = 50.0,
    pixel_size_m: float = 20.0,
    delta_t: float = DELTA_T_S2A,
    min_water_frac: float = 0.6,
    min_shore_dist_m: float = 50.0,
    max_shore_dist_m: float = 0.0,
    n_workers: int = 4,
) -> tuple:
    """Run wave_bathymetry on a `tile_nx × tile_ny` grid of overlapping
    sub-tiles in parallel and feather-mosaic the per-tile depth grids.

    Parameters mirror wave_bathymetry; additions:
      tile_nx, tile_ny : number of column / row tiles
      overlap_m        : overlap width (metres) between adjacent tiles
      n_workers        : ThreadPool size (scipy FFT releases the GIL, so
                         threads give real speed-up)

    Returns (mosaic_result, err). mosaic_result has the same contract as
    wave_bathymetry but with added `tile_report` (per-tile stats).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    b02 = s2w["b02"]; b04 = s2w["b04"]
    H_in, W_in = b02.shape
    w, s, e, n = bbox
    L.info(f"[Tiled] grid={tile_nx}×{tile_ny} in={W_in}x{H_in}px "
           f"@{pixel_size_m}m overlap={overlap_m}m window={window_m}m "
           f"step={step_m}m workers={n_workers}")

    overlap_px_in = max(1, int(round(overlap_m / pixel_size_m)))
    core_w = W_in / tile_nx
    core_h = H_in / tile_ny

    tiles = []  # list of (idx, s2w_sub, bbox_sub, origin_out)
    # First, precompute each tile's slice in the INPUT grid and the
    # corresponding geographic sub-bbox.
    per_tile_slices = []
    for ty in range(tile_ny):
        for tx in range(tile_nx):
            y0 = int(round(ty * core_h))
            y1 = int(round((ty + 1) * core_h))
            x0 = int(round(tx * core_w))
            x1 = int(round((tx + 1) * core_w))
            # Extend with overlap, clipped to image bounds
            ey0 = max(0, y0 - overlap_px_in)
            ey1 = min(H_in, y1 + overlap_px_in)
            ex0 = max(0, x0 - overlap_px_in)
            ex1 = min(W_in, x1 + overlap_px_in)
            per_tile_slices.append((tx, ty, x0, y0, x1, y1, ex0, ey0, ex1, ey1))

    # Output mosaic resolution — each tile returns its grid at step_m
    # pixel size; the mosaic shares that resolution.
    out_px_m = float(step_m)
    cl = np.cos(np.radians((n + s) / 2))
    W_out = max(1, int(round(abs(e - w) * 111000 * cl / out_px_m)))
    H_out = max(1, int(round(abs(n - s) * 111000 / out_px_m)))
    L.info(f"[Tiled] mosaic output grid: {W_out}x{H_out} @{out_px_m}m")

    # Fire worker tasks
    task_payload = []
    for (tx, ty, x0, y0, x1, y1, ex0, ey0, ex1, ey1) in per_tile_slices:
        sub_b02 = b02[ey0:ey1, ex0:ex1]
        sub_b04 = b04[ey0:ey1, ex0:ex1]
        # Geographic bbox of the (extended) tile
        bx0 = w + (ex0 / W_in) * (e - w)
        bx1 = w + (ex1 / W_in) * (e - w)
        by0 = n - (ey1 / H_in) * (n - s)   # south edge (ey1 is max y)
        by1 = n - (ey0 / H_in) * (n - s)   # north edge (ey0 is min y)
        bbox_tile = [bx0, by0, bx1, by1]
        s2w_tile = {"b02": sub_b02, "b04": sub_b04,
                    "width": sub_b02.shape[1], "height": sub_b02.shape[0]}
        task_payload.append(((tx, ty), s2w_tile, bbox_tile))

    depth_mosaic = np.zeros((H_out, W_out), dtype=np.float32)
    weight_mosaic = np.zeros((H_out, W_out), dtype=np.float32)
    tile_report = []
    n_ok = 0

    def _paste(tx, ty, tile_res, bbox_tile):
        """Feather-blend a tile's depth grid into the mosaic."""
        nonlocal n_ok
        if tile_res is None:
            return
        tH, tW = tile_res["depth"].shape
        bx0, by0, bx1, by1 = bbox_tile
        # Map tile bbox → mosaic pixel indices
        cl_ = np.cos(np.radians((n + s) / 2))
        i0 = int(round((n - by1) * 111000 / out_px_m))
        i1 = int(round((n - by0) * 111000 / out_px_m))
        j0 = int(round((bx0 - w) * 111000 * cl_ / out_px_m))
        j1 = int(round((bx1 - w) * 111000 * cl_ / out_px_m))
        i0 = max(0, i0); j0 = max(0, j0)
        i1 = min(H_out, i1); j1 = min(W_out, j1)
        if i1 <= i0 or j1 <= j0:
            return
        # Resample tile's depth onto the mosaic slot (nearest is fine —
        # the grids are already at step_m). Use bilinear via zoom if
        # shapes do not match exactly.
        dH = i1 - i0
        dW = j1 - j0
        if (tH, tW) != (dH, dW):
            try:
                from scipy.ndimage import zoom as _zoom
                zH = dH / tH if tH > 0 else 1.0
                zW = dW / tW if tW > 0 else 1.0
                tile_dep = _zoom(np.nan_to_num(tile_res["depth"], nan=0.0), (zH, zW), order=1)
                tile_mask = _zoom((np.isfinite(tile_res["depth"])).astype(np.float32), (zH, zW), order=0) > 0.5
            except Exception:
                tile_dep = np.nan_to_num(tile_res["depth"], nan=0.0)
                tile_mask = np.isfinite(tile_res["depth"])
        else:
            tile_dep = np.nan_to_num(tile_res["depth"], nan=0.0)
            tile_mask = np.isfinite(tile_res["depth"])
        margin = max(1, int(round(overlap_m / out_px_m)))
        fw = _cosine_feather(tile_dep.shape[0], tile_dep.shape[1], margin)
        fw = np.where(tile_mask, fw, 0.0)
        depth_mosaic[i0:i1, j0:j1] += tile_dep * fw
        weight_mosaic[i0:i1, j0:j1] += fw
        n_ok += 1

    if n_workers <= 1 or len(task_payload) == 1:
        for idx, s2w_tile, bbox_tile in task_payload:
            _, res, err = _process_wave_tile(
                idx, s2w_tile, bbox_tile,
                window_m, step_m, pixel_size_m, delta_t,
                min_water_frac, min_shore_dist_m, max_shore_dist_m)
            stats = (res or {}).get("stats", {})
            tile_report.append({"tile": idx, "err": err,
                                "valid": stats.get("valid_windows", 0),
                                "total": stats.get("total_windows", 0)})
            _paste(idx[0], idx[1], res, bbox_tile)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            fut_map = {}
            for idx, s2w_tile, bbox_tile in task_payload:
                fut = pool.submit(
                    _process_wave_tile, idx, s2w_tile, bbox_tile,
                    window_m, step_m, pixel_size_m, delta_t,
                    min_water_frac, min_shore_dist_m, max_shore_dist_m)
                fut_map[fut] = (idx, bbox_tile)
            for fut in as_completed(fut_map):
                idx, bbox_tile = fut_map[fut]
                _, res, err = fut.result()
                stats = (res or {}).get("stats", {})
                tile_report.append({"tile": idx, "err": err,
                                    "valid": stats.get("valid_windows", 0),
                                    "total": stats.get("total_windows", 0)})
                _paste(idx[0], idx[1], res, bbox_tile)

    if n_ok == 0:
        return None, "All tiles failed"

    with np.errstate(invalid='ignore', divide='ignore'):
        mosaic = np.where(weight_mosaic > 1e-6, depth_mosaic / weight_mosaic, np.nan)
    mosaic = np.clip(mosaic, 0.0, MAX_DEPTH_M)

    valid_depths = mosaic[np.isfinite(mosaic)]
    stats_all = {
        "method": "Wave Dispersion (Almar) — tiled",
        "tiles_total": len(task_payload),
        "tiles_ok": n_ok,
        "tiles_failed": len(task_payload) - n_ok,
        "total_valid": int(np.sum(np.isfinite(mosaic))),
        "depth_range": [round(float(np.nanmin(valid_depths)), 1),
                        round(float(np.nanmax(valid_depths)), 1)] if len(valid_depths) else [0, 0],
        "window_m": window_m, "step_m": step_m,
        "pixel_size_m": pixel_size_m,
        "tile_nx": tile_nx, "tile_ny": tile_ny, "overlap_m": overlap_m,
    }
    L.info(f"[Tiled] mosaic done: {n_ok}/{len(task_payload)} tiles ok, "
           f"{stats_all['total_valid']} valid cells, "
           f"depth ∈ {stats_all['depth_range']}")

    return {
        "depth": mosaic,
        "stats": stats_all,
        "tile_report": tile_report,
        "grid_shape": (H_out, W_out),
        "pixel_size_m": out_px_m,
    }, None
