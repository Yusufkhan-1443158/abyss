"""
S2Shores adapter — physically-grounded WAVE-DISPERSION bathymetry
═════════════════════════════════════════════════════════════════
Wraps the GDAL-free scientific core of CNES/S2Shores (Apache-2.0,
vendored under third_party/s2shores/) so it can be driven directly
from a single-date Sentinel-2 RAW B02+B04 band pair (the dict shape
returned by app.fetch_s2_wave / fetch_s2_wave_gee) and produce a
depth grid at a requested resolution (default 50 m).

WHY THIS EXISTS, separate from backend/wave_bathy.py
----------------------------------------------------
backend/wave_bathy.py is the prior, simplified, fully-self-contained
attempt. It is kept and still wired into /api/s2shores-bathymetry.
This adapter is the PHYSICALLY-FAITHFUL path:

  * It reuses the authentic CNES S2Shores cross-spectral spatial-DFT
    method (Radon transform → sinogram DFT → cross-correlation
    spectrum  S1·conj(S2)  → |phase|·amplitude directional/wavenumber
    peaks → linear-dispersion inversion).  See
    third_party/s2shores/local_bathymetry/spatial_dft_bathy_estimator.py
  * It uses the vendored physics functions in
    third_party/s2shores/bathy_physics.py for the depth inversion
    (linearity_indicator, depth_from_dispersion, sensitivity_indicator).
  * It uses the per-detector inter-band Δt from the vendored CNES CSVs
    (third_party/s2shores/bathylauncher/config/CNES/S2{A,B}_delta_times.csv):
    B02→B04 |Δt| ≈ 0.998 s (S2A) / 0.996 s (S2B).

MANDATORY HONESTY GATES (the whole point — fetch-limited basins such
as the Arabian Gulf will FABRICATE 16-18 s "swell" from noise unless
every cell is forced through ALL of these). A cell that fails ANY gate
is returned as NO-DATA — never a fabricated depth:

  G1  coherence            γ² ≥ COH_MIN            (default 0.5)
  G2  wave period          T  ∈ [T_MIN, T_MAX]      (default [4, 16] s)
  G3  wavelength           λ  ∈ [LAM_MIN, LAM_MAX]  (default [40, 600] m)
  G4  dispersion linearity γ  ∈ [GAM_MIN, GAM_MAX]  (default [0.2, 0.95])
  G5  spectral energy      E  ≥ ENERGY_MULT × local-median  (default 3×)

The whole estimator is gated behind an env flag, default OFF:
  WAVE_DISPERSION_ENABLED=1   →  run; otherwise raise / return disabled.

Gate thresholds are overridable via env for experimentation but the
defaults above are the ones validated in WAVE_DISPERSION_LOG.md.

References:
  Almar R. et al. (2024) Coastal Engineering 189, 104458
  Bergsma E.W.J. et al. (2019) Remote Sensing 11, 1918
  Binet R. et al. (2022) ISPRS Annals V-1-2022, 57-66
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

L = logging.getLogger("bathy.s2shores")

# ── Make the vendored GDAL-free core importable ──────────────────────
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
_TP = _REPO / "third_party"
import sys as _sys
if str(_TP) not in _sys.path:
    _sys.path.insert(0, str(_TP))

# Authentic CNES physics (vendored, Apache-2.0).
try:
    from s2shores.bathy_physics import (  # type: ignore
        depth_from_dispersion,
        linearity_indicator,
        period_offshore,
    )
    _S2SHORES_PHYSICS = True
except Exception as _ex:  # pragma: no cover - import guard
    L.warning("s2shores.bathy_physics import failed (%s); using local fallback", _ex)
    _S2SHORES_PHYSICS = False

GRAVITY = 9.81


# ── Local re-implementations matching the vendored physics, used only
#    if the import above failed (keeps the adapter runnable stand-alone).
def _linearity_indicator(wavelength: float, celerity: float, g: float = GRAVITY) -> float:
    return 2.0 * np.pi * (celerity ** 2) / (g * wavelength)


def _depth_from_wavelength(wavelength: float, celerity: float, g: float = GRAVITY) -> float:
    """Depth from the linear dispersion relation given λ and c.

    γ = tanh(k·h) = 2π·c²/(g·λ);  h = atanh(γ)/k,  k = 2π/λ.
    Returns np.nan when |γ| ≥ 1 (deep-water — bottom undetermined).
    """
    if _S2SHORES_PHYSICS:
        gamma = linearity_indicator(wavelength, celerity, g)
    else:
        gamma = _linearity_indicator(wavelength, celerity, g)
    if not np.isfinite(gamma) or abs(gamma) >= 1.0:
        return np.nan
    k = 2.0 * np.pi / wavelength
    try:
        depth = np.arctanh(gamma) / k
    except (ValueError, FloatingPointError):
        return np.nan
    if not np.isfinite(depth) or depth <= 0:
        return np.nan
    return float(depth)


def _gamma_of(wavelength: float, celerity: float, g: float = GRAVITY) -> float:
    if _S2SHORES_PHYSICS:
        return float(linearity_indicator(wavelength, celerity, g))
    return float(_linearity_indicator(wavelength, celerity, g))


# ═════════════════════════════════════════════════════════════════════
# Inter-band Δt from the vendored CNES CSVs (per-detector, B02→B04)
# ═════════════════════════════════════════════════════════════════════
def delta_t_b02_b04(satellite: str = "S2A") -> float:
    """Mean magnitude of the B02→B04 inter-detector time offset (seconds)
    read from the vendored CNES delta-times CSV. The sign alternates per
    detector (odd/even detectors scan in opposite directions); only the
    magnitude matters for celerity since we take |phase|.

    Falls back to the published constants if the CSV is unavailable:
      S2A ≈ 0.998 s, S2B ≈ 0.996 s, S2C ≈ 0.998 s.
    """
    sat = (satellite or "S2A").upper()
    if sat.startswith("S2A"):
        sat = "S2A"
    elif sat.startswith("S2B"):
        sat = "S2B"
    elif sat.startswith("S2C"):
        sat = "S2C"
    else:
        sat = "S2A"
    csv_path = _TP / "s2shores" / "bathylauncher" / "config" / "CNES" / f"{sat}_delta_times.csv"
    fallback = {"S2A": 0.998, "S2B": 0.996, "S2C": 0.998}[sat]
    try:
        mags = []
        with open(csv_path, newline="") as fh:
            rdr = csv.DictReader(fh, delimiter=";")
            for row in rdr:
                if row.get("bande_src") == "B02" and row.get("bande_dst") == "B04":
                    mags.append(abs(float(row["delta_t"])))
        if mags:
            dt = float(np.mean(mags))
            L.info("Δt(B02→B04) %s = %.4f s (mean of %d detectors, CNES CSV)",
                   sat, dt, len(mags))
            return dt
    except Exception as ex:  # pragma: no cover
        L.warning("Δt CSV read failed (%s); using fallback %.3f s", ex, fallback)
    return fallback


# ═════════════════════════════════════════════════════════════════════
# Gate thresholds (env-overridable; validated defaults)
# ═════════════════════════════════════════════════════════════════════
def _gate_cfg() -> dict:
    def _f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except Exception:
            return default
    return {
        "coh_min": _f("WAVE_COH_MIN", 0.5),
        "t_min": _f("WAVE_T_MIN", 4.0),
        "t_max": _f("WAVE_T_MAX", 16.0),
        "lam_min": _f("WAVE_LAM_MIN", 40.0),
        "lam_max": _f("WAVE_LAM_MAX", 600.0),
        "gam_min": _f("WAVE_GAM_MIN", 0.2),
        "gam_max": _f("WAVE_GAM_MAX", 0.95),
        "energy_mult": _f("WAVE_ENERGY_MULT", 3.0),
        "c_min": _f("WAVE_C_MIN", 1.0),
        "c_max": _f("WAVE_C_MAX", 25.0),
        # Coherence spectral-averaging width (bins). 0 = auto (≥15, len/8).
        "coh_smooth": int(_f("WAVE_COH_SMOOTH", 0)) or 0,
        # Depth-band gate (G6): valid swell-SDB band. Depths outside this
        # are physically implausible for this method (γ→1 runaway) and are
        # returned as NO-DATA. Brief: usable band ≈ 5–35 m; we keep a small
        # margin [2, 40] m so genuine shallow/deep edges are not over-clipped.
        "depth_min": _f("WAVE_DEPTH_MIN", 2.0),
        "depth_max": _f("WAVE_DEPTH_MAX", 40.0),
    }


def is_enabled() -> bool:
    """Master env gate — default OFF (per brief)."""
    return os.environ.get("WAVE_DISPERSION_ENABLED", "0") in ("1", "true", "True", "yes")


# ═════════════════════════════════════════════════════════════════════
# Pre-processing (detrend + Hann taper) — matches the S2Shores
# `detrend` preprocessing filter applied before the Radon transform.
# ═════════════════════════════════════════════════════════════════════
def _detrend2d(img: np.ndarray) -> np.ndarray:
    H, W = img.shape
    y = np.arange(H, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)
    Y, X = np.meshgrid(y, x, indexing="ij")
    A = np.column_stack([X.ravel(), Y.ravel(), np.ones(H * W)])
    b = img.ravel().astype(np.float64)
    valid = np.isfinite(b)
    if valid.sum() < 16:
        return np.nan_to_num(img.astype(np.float64))
    try:
        coef, _, _, _ = np.linalg.lstsq(A[valid], b[valid], rcond=None)
        return img.astype(np.float64) - (A @ coef).reshape(H, W)
    except Exception:
        return np.nan_to_num(img.astype(np.float64))


def _hann2d(H: int, W: int) -> np.ndarray:
    return np.outer(np.hanning(H), np.hanning(W))


# ═════════════════════════════════════════════════════════════════════
# Radon transform (variance-direction probe) + sinogram along the
# dominant wave direction. This mirrors WavesRadon → sinogram extraction
# in S2Shores but uses scipy.ndimage.rotate (GDAL-free, no shapely).
# ═════════════════════════════════════════════════════════════════════
def _dominant_direction(img: np.ndarray, pixel_m: float,
                        lam_min: float, lam_max: float) -> float:
    """2-D power-spectrum direction finder restricted to the valid λ band.
    Returns the propagation angle in degrees (image-frame)."""
    F = np.fft.fft2(np.nan_to_num(img))
    P = np.abs(F) ** 2
    H, W = img.shape
    fy = np.fft.fftfreq(H) / pixel_m
    fx = np.fft.fftfreq(W) / pixel_m
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    fmag = np.sqrt(FX ** 2 + FY ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        wl = np.where(fmag > 0, 1.0 / fmag, 0.0)
    valid = (wl >= lam_min) & (wl <= lam_max)
    Pv = np.where(valid, P, 0.0)
    if Pv.max() <= 0:
        return 0.0
    iy, ix = np.unravel_index(int(np.argmax(Pv)), Pv.shape)
    ang = float(np.degrees(np.arctan2(FY[iy, ix], FX[iy, ix])))
    if ang < 0:
        ang += 180.0
    if ang >= 180.0:
        ang -= 180.0
    return ang


def _sinogram(img: np.ndarray, direction_deg: float) -> np.ndarray:
    from scipy.ndimage import rotate
    rot = rotate(img, direction_deg, reshape=True, order=1, mode="constant", cval=0.0)
    prof = np.mean(rot, axis=0)
    nz = np.where(np.abs(prof) > 1e-9)[0]
    if len(nz) > 10:
        prof = prof[nz[0]:nz[-1] + 1]
    return prof


# ═════════════════════════════════════════════════════════════════════
# Cross-spectral celerity + COHERENCE on the two band sinograms.
# This is the authentic S2Shores cross-correlation spectrum:
#     C(f) = S1(f) · conj(S2(f))      (sinograms_correlation_fft)
#     phase = angle(C),  amplitude = |C|
# We additionally compute the magnitude-squared coherence
#     γ²(f) = |C|² / (|S1|² · |S2|²)
# at the dominant wavenumber as the G1 honesty gate.
# ═════════════════════════════════════════════════════════════════════
def _cross_spectral(sino1: np.ndarray, sino2: np.ndarray, pixel_m: float,
                    dt: float, cfg: dict) -> Optional[dict]:
    """Returns a dict(wavelength, celerity, period, gamma, coherence,
    energy, energy_ratio) for the best valid peak, or None."""
    n = min(len(sino1), len(sino2))
    if n < 24:
        return None
    s1 = sino1[:n] * np.hanning(n)
    s2 = sino2[:n] * np.hanning(n)
    F1 = np.fft.rfft(s1)
    F2 = np.fft.rfft(s2)
    freqs = np.fft.rfftfreq(n, d=pixel_m)            # cycles / m
    cross = F1 * np.conj(F2)
    amp1 = np.abs(F1) ** 2
    amp2 = np.abs(F2) ** 2

    # ── Magnitude-squared coherence requires ENSEMBLE/SPECTRAL averaging.
    # From a single FFT realization γ²≡1 identically — that is the exact
    # mechanism that fabricates "waves" from noise. We smooth the cross-
    # and auto-spectra across a small band of adjacent frequency bins
    # (equivalent to Welch segment averaging) so that white noise, whose
    # phase is random bin-to-bin, decoheres while a narrowband swell —
    # coherent across neighbouring bins — keeps γ² ≈ 1.
    # Averaging width: ≥15 bins (validated to push white-noise coherence
    # below the 0.5 gate while a narrowband swell stays at γ²≈1), scaling
    # up with spectrum length. Must be odd for a symmetric 'same' convolve.
    nb_spec = len(freqs)
    cw = int(cfg.get("coh_smooth", max(15, nb_spec // 8)))
    if cw % 2 == 0:
        cw += 1
    cw = min(cw, max(3, nb_spec - 1))

    def _smooth(x, w=cw):
        # Reflect-pad before the box filter so band-edge bins are not
        # artificially coherent (zero-padding would inflate γ² near Nyquist
        # and the long-wave cut, exactly where spurious short-λ candidates
        # from noise tend to sit).
        h = w // 2
        xp = np.pad(x, h, mode="reflect")
        k = np.ones(w) / w
        return np.convolve(xp, k, mode="same")[h:h + len(x)]
    S11 = _smooth(amp1)
    S22 = _smooth(amp2)
    S12r = _smooth(np.real(cross))
    S12i = _smooth(np.imag(cross))
    cross_avg_mag2 = S12r ** 2 + S12i ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        coh_spec = np.where(S11 * S22 > 1e-20,
                            cross_avg_mag2 / (S11 * S22), 0.0)
    coh_spec = np.clip(coh_spec, 0.0, 1.0)
    # S2Shores directional/wavenumber selector: |phase|·amplitude.
    total_spectrum = np.abs(np.angle(cross)) * np.abs(cross)
    power = amp1 + amp2

    fmin = 1.0 / cfg["lam_max"]
    fmax = 1.0 / cfg["lam_min"]
    band = (freqs > fmin) & (freqs < fmax)
    if band.sum() < 3:
        return None

    ts = total_spectrum.copy()
    ts[~band] = 0.0
    if ts.max() <= 0:
        return None

    # Local-median spectral floor over the valid band → energy gate (G5).
    band_power = power[band]
    local_med = float(np.median(band_power)) if band_power.size else 0.0

    # Candidate peaks: shortest-λ first (short waves feel shallower bottoms).
    peak_thr = ts.max() * 0.15
    cand = []
    for i in range(1, len(ts) - 1):
        if not band[i]:
            continue
        if ts[i] > peak_thr and ts[i] >= ts[i - 1] and ts[i] >= ts[i + 1]:
            cand.append(i)
    if not cand:
        cand = [int(np.argmax(ts))]
    cand.sort(key=lambda i: -freqs[i])

    for idx in cand:
        # Sub-pixel parabolic refinement of the wavelength.
        if 1 <= idx < len(freqs) - 1:
            p0, p1, p2 = float(power[idx - 1]), float(power[idx]), float(power[idx + 1])
            den = p0 - 2.0 * p1 + p2
            off = 0.5 * (p0 - p2) / den if abs(den) > 1e-12 else 0.0
            off = max(-0.5, min(0.5, off))
        else:
            off = 0.0
        df = freqs[1] - freqs[0]
        f_pk = float(freqs[idx]) + off * df
        if f_pk <= 0:
            continue
        wavelength = 1.0 / f_pk
        if wavelength < cfg["lam_min"] or wavelength > cfg["lam_max"]:
            continue

        phase = float(np.angle(cross[idx]))
        # Phase-wrap guard: near ±π the celerity is aliased.
        if abs(phase) > 0.85 * np.pi:
            continue
        displacement = wavelength * phase / (2.0 * np.pi)
        celerity = abs(displacement / dt)
        if celerity < cfg["c_min"] or celerity > cfg["c_max"]:
            continue

        period = wavelength / celerity if celerity > 0 else 0.0

        # Spectrally-averaged magnitude-squared coherence at this bin (G1).
        # Enforced INSIDE the candidate loop so a high-frequency bin that
        # only looks coherent due to band-edge convolution effects is
        # skipped in favour of (or instead of) a genuinely coherent peak —
        # rather than being returned and silently rejected downstream.
        coherence = float(coh_spec[idx])
        if coherence < cfg["coh_min"]:
            continue

        energy = float(power[idx])
        energy_ratio = energy / local_med if local_med > 1e-20 else 0.0
        gamma = _gamma_of(wavelength, celerity)

        return {
            "wavelength": wavelength,
            "celerity": celerity,
            "period": period,
            "gamma": gamma,
            "coherence": coherence,
            "energy": energy,
            "energy_ratio": energy_ratio,
        }
    return None


# ═════════════════════════════════════════════════════════════════════
# Honesty gates
# ═════════════════════════════════════════════════════════════════════
def _passes_gates(m: dict, cfg: dict) -> Tuple[bool, str]:
    if m["coherence"] < cfg["coh_min"]:
        return False, "G1_coherence"
    if not (cfg["t_min"] <= m["period"] <= cfg["t_max"]):
        return False, "G2_period"
    if not (cfg["lam_min"] <= m["wavelength"] <= cfg["lam_max"]):
        return False, "G3_wavelength"
    if not (cfg["gam_min"] <= m["gamma"] <= cfg["gam_max"]):
        return False, "G4_linearity"
    if m["energy_ratio"] < cfg["energy_mult"]:
        return False, "G5_energy"
    return True, "ok"


# ═════════════════════════════════════════════════════════════════════
# Land / water mask (reuse wave_bathy's mask builder if available).
# ═════════════════════════════════════════════════════════════════════
def _water_mask(b02: np.ndarray, b04: np.ndarray, bbox: list) -> np.ndarray:
    try:
        try:
            from backend.wave_bathy import build_wave_water_mask
        except ImportError:
            from wave_bathy import build_wave_water_mask
        return build_wave_water_mask(b02, b04, bbox)
    except Exception as ex:
        L.warning("water mask builder unavailable (%s); using B04 threshold", ex)
        b04f = b04.astype(np.float32)
        finite = np.isfinite(b04f)
        if finite.any():
            thr = max(float(np.percentile(b04f[finite], 65)), 1500.0)
            return b04f <= thr
        return np.ones_like(b04, dtype=bool)


# ═════════════════════════════════════════════════════════════════════
# MAIN ENTRY — sliding-window spatial-DFT wave bathymetry @ resolution_m
# ═════════════════════════════════════════════════════════════════════
def run_wave_bathymetry(
    s2_wave: dict,
    bbox: list,
    *,
    resolution_m: float = 50.0,
    window_m: float = 800.0,
    pixel_size_m: float = 10.0,
    satellite: Optional[str] = None,
    delta_t: Optional[float] = None,
    min_water_frac: float = 0.7,
    water_mask: Optional[np.ndarray] = None,
    require_enabled: bool = True,
) -> Tuple[Optional[dict], Optional[str]]:
    """Physically-grounded wave-dispersion bathymetry on a single-date
    RAW S2 B02+B04 pair, gated by the 5 honesty gates, output at
    `resolution_m` (default 50 m).

    Parameters
    ----------
    s2_wave        dict with "b02","b04" (H,W) RAW DN arrays (single orbit)
    bbox           [west, south, east, north]
    resolution_m   OUTPUT grid pixel size (default 50 m)
    window_m       spatial analysis window (default 800 m → ≥1 wavelength)
    pixel_size_m   input pixel size in metres (10 for native S2)
    satellite      "S2A"/"S2B"/"S2C" — selects Δt from the CNES CSV
    delta_t        explicit inter-band Δt (s); overrides `satellite`
    min_water_frac minimum water fraction per window
    water_mask     optional (H,W) bool mask (water=True) overriding the
                   auto land/water detection (used by synthetic tests)
    require_enabled  if True, refuse unless WAVE_DISPERSION_ENABLED=1

    Returns (result_dict, err). result_dict has:
      depth (ny,nx)        NO-DATA = NaN
      coherence, period, wavelength, celerity, gamma, energy_ratio grids
      stats {...}          coverage + gate-rejection breakdown
      grid_shape, pixel_size_m (= resolution_m)
    """
    if require_enabled and not is_enabled():
        return None, ("WAVE_DISPERSION disabled — set WAVE_DISPERSION_ENABLED=1 "
                      "to run the wave-dispersion estimator")

    cfg = _gate_cfg()
    if delta_t is None:
        delta_t = delta_t_b02_b04(satellite or s2_wave.get("satellite") or "S2A")

    b02 = np.asarray(s2_wave["b02"], dtype=np.float64)
    b04 = np.asarray(s2_wave["b04"], dtype=np.float64)
    H, W = b02.shape

    L.info("S2Shores-adapter: %dx%d @%.0fm  window=%.0fm  out=%.0fm  Δt=%.4fs",
           W, H, pixel_size_m, window_m, resolution_m, delta_t)
    L.info("Gates: γ²≥%.2f T∈[%.0f,%.0f]s λ∈[%.0f,%.0f]m γ∈[%.2f,%.2f] E≥%.1f×med",
           cfg["coh_min"], cfg["t_min"], cfg["t_max"], cfg["lam_min"],
           cfg["lam_max"], cfg["gam_min"], cfg["gam_max"], cfg["energy_mult"])

    if water_mask is not None and np.shape(water_mask) == (H, W):
        water = np.asarray(water_mask, dtype=bool)
    else:
        water = _water_mask(b02, b04, bbox)
    if water.shape != (H, W):
        water = np.ones((H, W), dtype=bool)

    # Zero land before detrend so trends are fit on water only.
    b02 = np.where(water, b02, 0.0)
    b04 = np.where(water, b04, 0.0)

    win_px = max(16, int(round(window_m / pixel_size_m)))
    step_px = max(1, int(round(resolution_m / pixel_size_m)))
    ny = max(1, (H - win_px) // step_px + 1)
    nx = max(1, (W - win_px) // step_px + 1)

    depth = np.full((ny, nx), np.nan, np.float32)
    g_coh = np.full((ny, nx), np.nan, np.float32)
    g_per = np.full((ny, nx), np.nan, np.float32)
    g_lam = np.full((ny, nx), np.nan, np.float32)
    g_cel = np.full((ny, nx), np.nan, np.float32)
    g_gam = np.full((ny, nx), np.nan, np.float32)
    g_eng = np.full((ny, nx), np.nan, np.float32)

    rej = {"land": 0, "no_peak": 0, "G1_coherence": 0, "G2_period": 0,
           "G3_wavelength": 0, "G4_linearity": 0, "G5_energy": 0,
           "deep_undetermined": 0, "G6_depth_band": 0}
    n_valid = 0
    total = ny * nx

    for iy in range(ny):
        y0 = iy * step_px
        y1 = min(y0 + win_px, H)
        for ix in range(nx):
            x0 = ix * step_px
            x1 = min(x0 + win_px, W)
            pm = water[y0:y1, x0:x1]
            if float(pm.mean()) < min_water_frac:
                rej["land"] += 1
                continue
            p02 = _detrend2d(b02[y0:y1, x0:x1])
            p04 = _detrend2d(b04[y0:y1, x0:x1])
            taper = _hann2d(*p02.shape)
            p02 = p02 * taper
            p04 = p04 * taper

            direction = _dominant_direction(0.5 * (p02 + p04), pixel_size_m,
                                             cfg["lam_min"], cfg["lam_max"])
            s1 = _sinogram(p02, direction)
            s2 = _sinogram(p04, direction)
            m = _cross_spectral(s1, s2, pixel_size_m, delta_t, cfg)
            if m is None:
                rej["no_peak"] += 1
                continue

            ok, why = _passes_gates(m, cfg)
            if not ok:
                rej[why] += 1
                continue

            d = _depth_from_wavelength(m["wavelength"], m["celerity"])
            if not np.isfinite(d) or d <= 0:
                rej["deep_undetermined"] += 1
                continue
            # G6 depth-band gate — reject physically-implausible runaway
            # depths (γ→1) that the linearity gate alone lets through.
            if d < cfg["depth_min"] or d > cfg["depth_max"]:
                rej["G6_depth_band"] += 1
                continue

            depth[iy, ix] = d
            g_coh[iy, ix] = m["coherence"]
            g_per[iy, ix] = m["period"]
            g_lam[iy, ix] = m["wavelength"]
            g_cel[iy, ix] = m["celerity"]
            g_gam[iy, ix] = m["gamma"]
            g_eng[iy, ix] = m["energy_ratio"]
            n_valid += 1

    nodata_frac = 1.0 - (n_valid / total if total else 0.0)
    vd = depth[np.isfinite(depth)]
    vp = g_per[np.isfinite(g_per)]
    vl = g_lam[np.isfinite(g_lam)]

    stats = {
        "method": "S2Shores / Almar wave-dispersion (gated)",
        "delta_t_s": round(delta_t, 4),
        "resolution_m": resolution_m,
        "window_m": window_m,
        "input_pixel_m": pixel_size_m,
        "total_windows": total,
        "valid_windows": n_valid,
        "nodata_windows": total - n_valid,
        "nodata_frac": round(nodata_frac, 4),
        "coverage_pct": round(100.0 * n_valid / total, 2) if total else 0.0,
        "rejections": rej,
        "gates": cfg,
        "depth_range_m": [round(float(vd.min()), 2), round(float(vd.max()), 2)] if vd.size else None,
        "period_range_s": [round(float(vp.min()), 2), round(float(vp.max()), 2)] if vp.size else None,
        "period_mean_s": round(float(vp.mean()), 2) if vp.size else None,
        "wavelength_range_m": [round(float(vl.min()), 1), round(float(vl.max()), 1)] if vl.size else None,
        "wavelength_mean_m": round(float(vl.mean()), 1) if vl.size else None,
        "coherence_mean": round(float(g_coh[np.isfinite(g_coh)].mean()), 3) if vd.size else None,
    }
    L.info("S2Shores-adapter: %d/%d valid (%.1f%% coverage, NO-DATA=%.1f%%)",
           n_valid, total, stats["coverage_pct"], 100 * nodata_frac)
    L.info("Rejections: %s", rej)

    return {
        "depth": depth,
        "coherence": g_coh,
        "period": g_per,
        "wavelength": g_lam,
        "celerity": g_cel,
        "gamma": g_gam,
        "energy_ratio": g_eng,
        "water_mask": water,
        "stats": stats,
        "grid_shape": (ny, nx),
        "pixel_size_m": resolution_m,
        "bbox": bbox,
    }, None
