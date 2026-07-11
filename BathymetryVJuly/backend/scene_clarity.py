"""HY-5 / HY-6 Clarity-based scene selection for SDB.

Scores a pool of individual Sentinel-2 L2A scenes by water-column clarity
over the water mask, selects the clearest scene(s), and optionally builds
a top-2 inverse-variance weighted composite.

Physical basis
--------------
For optically shallow coastal water the key limiting factor on depth-of-
penetration is the per-scene attenuation length Ld = 1 / (2·Kd).  We proxy
Kd490 from the blue/green reflectance ratio, SPM from Nechad (2010) at B04,
glint from NIR reflectance, and cloud/cirrus from the S2 SCL class map.
A scalar clarity score (higher = better) is then:

    clarity = -w_turb * SPM_median
              -w_kd   * Kd490_median
              -w_glint* glint_frac
              -w_cloud* cloud_frac

all computed over water-mask pixels only.  Lower turbidity / lower Kd /
lower glint / lower cloud → higher clarity → selected first.

Gate
----
Activated by env var SCENE_SELECT=clarity  (default: OFF — existing median
composite path is untouched).

Env vars
--------
SCENE_SELECT        : "clarity" to enable; any other value (or unset) = off.
SCENE_SELECT_TOP_N  : 1 (default) → single clearest scene;
                      2 → inverse-variance weighted composite of top-2.
S2_POOL_SIZE        : number of candidate scenes to fetch from GEE (default 6).
S2_START / S2_END   : date window (default: same 2023-01-01 → 2024-12-31).
S2_CLOUD_PCT        : cloud filter passed to GEE (default 25).

References
----------
Nechad et al. 2010 (SPM from red reflectance).
Lee et al. 2002 / Morel & Maritorena 2001 (Kd490 empirical blue-green ratio).
Hedley et al. 2005 (NIR glint fraction).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

L = logging.getLogger("scene_clarity")
if not L.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    L.addHandler(h)
    L.setLevel(logging.INFO)

# ─── Physical constants ───────────────────────────────────────────────────────
# Nechad 2010 calibration at 665 nm (S2 B04 = red)
NECHAD_A_665 = 366.14    # g m⁻³
NECHAD_C_665 = 0.19563
EPS = 1e-7

# Kd490 empirical coefficients (Morel & Maritorena 2001 / QAA-v5 blue-green):
#   Kd490 = A0 + A1 * (rho_blue / rho_green) ^ exp
#
# NOTE on scale: The Morel formula was derived for open-ocean Rrs (sr⁻¹) which
# is ~π-fold smaller than S2 BOA reflectance ρ.  However, the *ratio* ρ_blue/ρ_green
# equals Rrs_blue/Rrs_green (π cancels), so the formula produces absolute Kd values
# calibrated for open-ocean blue-green ratios (typically 0.3–0.8 sr⁻¹/sr⁻¹).
# For turbid Arabian Gulf water (ratio ≈ 1.0), this gives Kd ≈ 0.8 m⁻¹, consistent
# with a diffuse-attenuation coefficient for 665 nm turbid Case-2 water.
# We use this proxy EXCLUSIVELY for relative ranking across scenes (lower = clearer),
# NOT as an absolute Kd(490) estimate.
# z_max is derived from the SPM-calibrated Beer-Lambert slope (see score_scene()).
KD490_A0 = 0.0166
KD490_A1 = 0.8358
KD490_EXP = -1.470

# SPM → Beer-Lambert Kd calibration (from HYDRO log §1.0 Khalifa bulk fit):
# Khalifa: SPM ≈ 12.3 g/m³, bulk Kd_B3 ≈ 0.082 m⁻¹
# Kd_BL ≈ SPM * (0.082 / 12.3) = SPM * 0.00667 m⁻¹/(g/m³)
# z_max_practical ≈ 2.3 / Kd_BL   (4.6 one-way optical depths)
SPM_TO_KD_COEFF = 0.00667   # m⁻¹ per g/m³

# Clarity score weights (tuned to give ~equal dynamic range to each component
# for the Arabian Gulf turbidity regime).
W_SPM   = 0.004   # 12 g/m³ SPM → −0.048 penalty
W_KD    = 2.0     # 0.1/m Kd → −0.2 penalty
W_GLINT = 5.0     # 0.04 NIR reflectance glint → −0.2 penalty
W_CLOUD = 5.0     # 0.04 cloud fraction → −0.2 penalty

# SCL water classes (Sentinel-2 Scene Classification Layer)
SCL_WATER_CLASSES = {6}   # class 6 = water
SCL_CLOUD_CLASSES = {8, 9, 10, 11}  # medium/high cloud + cirrus + snow


def _water_mask(s2_scene: Dict) -> np.ndarray:
    """Return boolean water mask from NDWI + SCL."""
    eps = EPS
    green = s2_scene["green"].astype(np.float32) / 10000.0
    nir   = s2_scene["nir"].astype(np.float32)   / 10000.0
    ndwi  = (green - nir) / (green + nir + eps)
    ndwi_mask = ndwi > 0.0

    scl = s2_scene.get("scl")
    if scl is not None:
        scl = np.asarray(scl).astype(np.int32)
        scl_water = np.isin(scl, list(SCL_WATER_CLASSES))
    else:
        scl_water = np.zeros_like(ndwi_mask)

    return ndwi_mask | scl_water


def nechad_spm(rho_red: np.ndarray) -> np.ndarray:
    """Nechad 2010 SPM (g m⁻³) from surface reflectance (dimensionless)."""
    r = np.clip(rho_red, EPS, NECHAD_C_665 - 0.005)
    return np.clip((NECHAD_A_665 * r) / (1.0 - r / NECHAD_C_665), 0.0, 200.0).astype(np.float32)


def kd490_proxy(rho_blue: np.ndarray, rho_green: np.ndarray) -> np.ndarray:
    """Empirical Kd(490) (m⁻¹) from blue/green reflectance ratio.

    Morel & Maritorena 2001 parameterisation adapted for S2 B2/B3:
        Kd490 = A0 + A1 * (Rrs_blue / Rrs_green) ^ exp

    Clamped to physically plausible range [0.01, 5.0] m⁻¹.
    """
    ratio = np.clip(rho_blue, EPS, None) / np.clip(rho_green, EPS, None)
    kd = KD490_A0 + KD490_A1 * np.power(np.clip(ratio, 0.05, 20.0), KD490_EXP)
    return np.clip(kd, 0.01, 5.0).astype(np.float32)


def score_scene(scene: Dict) -> Dict:
    """Compute per-scene clarity metrics over the water mask.

    Parameters
    ----------
    scene : dict with keys blue/green/red/nir/scl (raw DN) + metadata.

    Returns
    -------
    metrics dict:
        clarity_score   : scalar (higher = clearer = better)
        spm_median_g_m3 : Nechad SPM over water pixels
        kd490_median    : empirical Kd490 over water pixels
        glint_frac      : fraction of water pixels with NIR > 0.02
        cloud_frac      : fraction of ROI with SCL cloud class
        n_water_px      : number of water pixels used
        z_max_approx_m  : 1/(2*kd490_median) — approximate depth-of-penetration
    """
    # Reflectances (dimensionless)
    blue  = scene["blue"].astype(np.float32)  / 10000.0
    green = scene["green"].astype(np.float32) / 10000.0
    red   = scene["red"].astype(np.float32)   / 10000.0
    nir   = scene["nir"].astype(np.float32)   / 10000.0

    wm = _water_mask(scene)
    n_water = int(wm.sum())
    total_px = int(wm.size)

    if n_water < 50:
        # Degenerate: no water pixels → very low clarity
        return {
            "clarity_score": -999.0,
            "spm_median_g_m3": 999.0,
            "kd490_median": 5.0,
            "glint_frac": 1.0,
            "cloud_frac": 1.0,
            "n_water_px": n_water,
            "z_max_approx_m": 0.1,
        }

    spm   = nechad_spm(red[wm])
    kd490 = kd490_proxy(blue[wm], green[wm])

    spm_med   = float(np.median(spm))
    kd490_med = float(np.median(kd490))

    # Glint fraction over water pixels: NIR > threshold indicates sunglint.
    # Arabian Gulf turbid coastal median NIR ≈ 0.020–0.025 over water (non-glint),
    # so threshold 0.04 targets pixels with genuine glint above background.
    # (The 2-yr median composite already reduces episodic glint; individual scene
    #  glint will be higher — 0.04 is intentionally conservative for single scenes.)
    glint_thresh = 0.04
    glint_frac = float(np.mean(nir[wm] > glint_thresh))

    # Cloud fraction over entire ROI from SCL
    scl = scene.get("scl")
    if scl is not None:
        scl_arr = np.asarray(scl).astype(np.int32)
        cloud_frac = float(np.isin(scl_arr, list(SCL_CLOUD_CLASSES)).mean())
    else:
        # Fall back to metadata cloud_pct if SCL missing
        cloud_frac = float(scene.get("cloud_pct", 0)) / 100.0

    clarity = -(W_SPM * spm_med + W_KD * kd490_med
                + W_GLINT * glint_frac + W_CLOUD * cloud_frac)

    # Approximate depth-of-penetration via SPM-calibrated Beer-Lambert Kd:
    #   Kd_BL = SPM * SPM_TO_KD_COEFF  (calibrated against Khalifa bulk fit)
    #   z_max ≈ 2.3 / Kd_BL  (4.6 one-way optical depths at 10% bottom reflectance)
    # Clamped to [2, 30] m as a practical SDB range.
    kd_bl_approx = max(spm_med * SPM_TO_KD_COEFF, 0.005)
    z_max = float(np.clip(2.3 / kd_bl_approx, 2.0, 30.0))

    return {
        "clarity_score": float(clarity),
        "spm_median_g_m3": float(spm_med),
        "kd490_median": float(kd490_med),
        "glint_frac": float(glint_frac),
        "cloud_frac": float(cloud_frac),
        "n_water_px": n_water,
        "z_max_approx_m": float(z_max),
    }


def rank_scenes(scenes: List[Dict]) -> List[Dict]:
    """Score and rank a list of scenes by clarity (best first).

    Each scene dict is augmented with a "clarity_metrics" sub-dict.
    Returns the same list sorted descending by clarity_score.
    """
    scored = []
    for i, sc in enumerate(scenes):
        m = score_scene(sc)
        sc_copy = dict(sc)
        sc_copy["clarity_metrics"] = m
        sc_copy["clarity_score"] = m["clarity_score"]
        scored.append(sc_copy)
        L.info(
            f"  scene {i:02d}  date={sc.get('acq_date','?')[:13]}  "
            f"cloud={sc.get('cloud_pct', float('nan')):.1f}%  "
            f"SPM={m['spm_median_g_m3']:.1f} g/m³  "
            f"Kd490={m['kd490_median']:.3f}/m  "
            f"glint={m['glint_frac']:.3f}  "
            f"z_max≈{m['z_max_approx_m']:.1f}m  "
            f"clarity={m['clarity_score']:.3f}"
        )
    scored.sort(key=lambda s: s.get("clarity_score", -999), reverse=True)
    return scored


def select_best_scene(ranked_scenes: List[Dict], top_n: int = 1) -> Dict:
    """Select the clearest scene, or build an inverse-variance composite of top-n.

    Parameters
    ----------
    ranked_scenes : output of rank_scenes() — sorted best-first.
    top_n         : 1 = single best scene; 2 = inverse-variance combine of top-2.

    Returns
    -------
    scene dict with the same keys as a single-scene output plus:
      - "composite_mode": "single" or "iv_blend"
      - "selected_indices": list of original scene indices used
      - "clarity_score": score of the primary (best) scene
    """
    if not ranked_scenes:
        raise ValueError("select_best_scene: empty ranked_scenes list")

    top_n = min(top_n, len(ranked_scenes))

    if top_n == 1:
        sc = dict(ranked_scenes[0])
        sc["composite_mode"] = "single"
        sc["selected_indices"] = [sc.get("scene_idx", 0)]
        L.info(f"Selected scene: idx={sc.get('scene_idx')}  "
               f"date={sc.get('acq_date','?')[:13]}  "
               f"clarity={sc.get('clarity_score',float('nan')):.3f}")
        return sc

    # top_n == 2: inverse-variance weighted composite
    # Use 1/(SPM + 1) as proxy for precision (higher clarity → higher weight)
    sc_a = ranked_scenes[0]
    sc_b = ranked_scenes[1]

    w_a = 1.0 / max(sc_a["clarity_metrics"]["spm_median_g_m3"] + 1.0, 0.1)
    w_b = 1.0 / max(sc_b["clarity_metrics"]["spm_median_g_m3"] + 1.0, 0.1)
    w_total = w_a + w_b
    alpha = w_a / w_total   # weight for scene_a
    beta  = w_b / w_total   # weight for scene_b

    L.info(f"IV composite: scene_a={sc_a.get('scene_idx')} (w={alpha:.3f}) "
           f"+ scene_b={sc_b.get('scene_idx')} (w={beta:.3f})")

    def _blend(key: str) -> np.ndarray:
        a = sc_a[key].astype(np.float32)
        b = sc_b[key].astype(np.float32)
        return (alpha * a + beta * b).astype(np.float32)

    composite = {
        "blue":  _blend("blue"),
        "green": _blend("green"),
        "red":   _blend("red"),
        "nir":   _blend("nir"),
        # For SCL take the cleaner of the two (lower cloud class index = cleaner)
        # Use scene_a's SCL as primary (it has better clarity score)
        "scl":   sc_a["scl"],
        "height": sc_a["height"],
        "width":  sc_a["width"],
        "resolution_m": sc_a["resolution_m"],
        "composite_mode": "iv_blend",
        "selected_indices": [sc_a.get("scene_idx", 0), sc_b.get("scene_idx", 1)],
        "clarity_score": sc_a["clarity_score"],
        "clarity_metrics": sc_a["clarity_metrics"],
        "acq_date": f"{sc_a.get('acq_date','?')[:13]}_+_{sc_b.get('acq_date','?')[:13]}",
        "cloud_pct": float(alpha * sc_a.get("cloud_pct", 0) + beta * sc_b.get("cloud_pct", 0)),
        "glint_nir": float(alpha * sc_a.get("glint_nir", 0) + beta * sc_b.get("glint_nir", 0)),
    }
    return composite


def clarity_table_str(ranked_scenes: List[Dict]) -> str:
    """Format a compact clarity table string for logging / reporting."""
    header = (
        f"{'Rank':>4} {'ScnIdx':>6} {'Date':>13} {'Cloud%':>7} "
        f"{'SPM g/m³':>9} {'Kd490/m':>8} {'Glint':>6} {'z_max m':>8} "
        f"{'ClarityScore':>13}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for rank, sc in enumerate(ranked_scenes):
        m = sc.get("clarity_metrics", {})
        rows.append(
            f"{rank+1:>4} {sc.get('scene_idx', '?'):>6} "
            f"{str(sc.get('acq_date','?'))[:13]:>13} "
            f"{sc.get('cloud_pct', float('nan')):>7.1f} "
            f"{m.get('spm_median_g_m3', float('nan')):>9.2f} "
            f"{m.get('kd490_median', float('nan')):>8.4f} "
            f"{m.get('glint_frac', float('nan')):>6.4f} "
            f"{m.get('z_max_approx_m', float('nan')):>8.1f} "
            f"{sc.get('clarity_score', float('nan')):>13.4f}"
        )
    return "\n".join(rows)


def _scene_pixel_weight(scene: Dict) -> np.ndarray:
    """Per-pixel clarity weight for one scene.

    Weight = 1 / (SPM_px + SPM_floor) — SPM-inverse per pixel, so turbid
    patches within an otherwise clean scene are down-weighted relative to clear
    patches.  Glint pixels (NIR > glint_thresh) receive zero weight.
    Cloud pixels (SCL cloud class) receive zero weight.
    Output shape (H, W), dtype float32, ≥ 0.

    Physical rationale: the depth signal in log(R) decays as exp(−2·Kd·z).
    SPM ∝ Kd for sediment-dominated Case-2 water (Nechad 2010), so
    1/SPM ∝ 1/Kd ∝ depth-of-penetration.  Down-weighting turbid pixels
    preserves the deep-water radiance contribution in the weighted composite.
    """
    H = int(scene["height"]); W = int(scene["width"])
    red   = scene["red"].astype(np.float32).reshape(H, W)  / 10000.0
    nir   = scene["nir"].astype(np.float32).reshape(H, W)  / 10000.0
    green = scene["green"].astype(np.float32).reshape(H, W) / 10000.0

    spm_px  = nechad_spm(red)          # (H,W) g/m³
    SPM_FLOOR = 1.0                    # min SPM to avoid 1/0 (1 g/m³ ≈ clean-coastal)
    w_px = 1.0 / (spm_px + SPM_FLOOR)

    # Zero-weight: glint (NIR > 0.04 threshold)
    glint_mask = nir > 0.04
    w_px[glint_mask] = 0.0

    # Zero-weight: SCL cloud classes
    scl = scene.get("scl")
    if scl is not None:
        scl_arr = np.asarray(scl).astype(np.int32).reshape(H, W)
        cloud_mask = np.isin(scl_arr, list(SCL_CLOUD_CLASSES))
        w_px[cloud_mask] = 0.0

    return w_px.astype(np.float32)


# HY-6 turbidity/glint gate thresholds
HY6_SPM_MAD_FACTOR  = 1.5   # drop scene if SPM_median > baseline_median + 1.5×MAD
HY6_GLINT_FRAC_MAX  = 0.35  # drop if glint fraction > 35%
HY6_CLOUD_FRAC_MAX  = 0.20  # drop if cloud fraction > 20%


def _hy6_gate(ranked_scenes: List[Dict]) -> List[Dict]:
    """HY-6: reject scenes whose turbidity/glint exceeds the pool baseline.

    Gate criteria (all over water mask):
      - SPM_median > median_pool + 1.5 × MAD_pool  (turbidity spike)
      - glint_frac > HY6_GLINT_FRAC_MAX
      - cloud_frac > HY6_CLOUD_FRAC_MAX

    At least 2 scenes are always retained to avoid degenerate composites.
    Returns the kept sub-list (order preserved).
    """
    if len(ranked_scenes) <= 2:
        return ranked_scenes

    spms = np.array([s["clarity_metrics"]["spm_median_g_m3"] for s in ranked_scenes])
    spm_med_pool = float(np.median(spms))
    spm_mad_pool = float(np.median(np.abs(spms - spm_med_pool)))
    spm_thresh   = spm_med_pool + HY6_SPM_MAD_FACTOR * spm_mad_pool

    kept = []
    for sc in ranked_scenes:
        m = sc["clarity_metrics"]
        spm_   = m["spm_median_g_m3"]
        glint_ = m["glint_frac"]
        cloud_ = m["cloud_frac"]
        reason = []
        if spm_   > spm_thresh:     reason.append(f"SPM {spm_:.1f}>{spm_thresh:.1f}")
        if glint_ > HY6_GLINT_FRAC_MAX: reason.append(f"glint {glint_:.3f}>{HY6_GLINT_FRAC_MAX}")
        if cloud_ > HY6_CLOUD_FRAC_MAX: reason.append(f"cloud {cloud_:.3f}>{HY6_CLOUD_FRAC_MAX}")
        if reason:
            L.info(f"  HY-6 DROP scene {sc.get('scene_idx')}  date={sc.get('acq_date','?')[:13]}"
                   f"  reason: {', '.join(reason)}")
        else:
            kept.append(sc)

    # Must keep at least 2
    if len(kept) < 2:
        L.warning(f"  HY-6 gate dropped too many scenes ({len(ranked_scenes)-len(kept)} dropped); "
                  f"keeping best 2 regardless")
        kept = ranked_scenes[:2]

    L.info(f"  HY-6 gate: {len(kept)}/{len(ranked_scenes)} scenes kept")
    return kept


def build_clarity_weighted_composite(
    ranked_scenes: List[Dict],
    *,
    apply_hy6_gate: bool = True,
    weight_scheme: str = "spm_inv",
) -> Tuple[Dict, Dict]:
    """Build a clarity-weighted composite from K scenes (the corrected HY-5).

    Unlike the single-scene replacement, this function:
      1. Applies the HY-6 turbidity/glint gate (optional) to exclude the worst scenes.
      2. Assigns a per-pixel weight to each scene proportional to 1/SPM (or other scheme).
      3. Forms a pixel-wise weighted mean for B2/B3/B4/B8.
      4. Forms the SCL by taking the SCL class of the highest-weight scene at each pixel.

    The output has the SAME spatial extent and pixel count as any individual input scene.
    Training-data count is therefore PRESERVED vs the median composite.

    Weight schemes
    --------------
    "spm_inv" (default): w_px = 1 / (SPM_px + 1.0)  per pixel per scene.
    "scene_spm_inv"     : w_s = 1 / (SPM_median_s + 1.0)  scalar per scene,
                          broadcast to all pixels (faster, nearly as good).

    Parameters
    ----------
    ranked_scenes   : output of rank_scenes() — scored, sorted best-first.
    apply_hy6_gate  : whether to apply the HY-6 turbidity/glint gate first.
    weight_scheme   : weighting scheme (see above).

    Returns
    -------
    composite       : dict with same keys as fetch_s2_for_site output + composite_meta.
    composite_meta  : summary dict (n_scenes_used, weights, gate_result, etc.).
    """
    if not ranked_scenes:
        raise ValueError("build_clarity_weighted_composite: empty ranked_scenes")

    # Step 1: HY-6 gate
    gated = _hy6_gate(ranked_scenes) if apply_hy6_gate else list(ranked_scenes)
    n_scenes = len(gated)

    H = int(gated[0]["height"]); W = int(gated[0]["width"])

    L.info(f"build_clarity_weighted_composite: {n_scenes} scenes  ({weight_scheme})  {H}x{W}")

    # Step 2: compute per-scene weight arrays
    w_arrays: List[np.ndarray] = []
    for sc in gated:
        if weight_scheme == "spm_inv":
            w = _scene_pixel_weight(sc)           # (H, W) float32
        elif weight_scheme == "scene_spm_inv":
            spm_s = sc["clarity_metrics"]["spm_median_g_m3"]
            w_scalar = 1.0 / max(spm_s + 1.0, 0.01)
            w = np.full((H, W), w_scalar, dtype=np.float32)
            # Still zero-weight glint/cloud pixels
            nir_px = sc["nir"].astype(np.float32).reshape(H, W) / 10000.0
            w[nir_px > 0.04] = 0.0
            scl_ = sc.get("scl")
            if scl_ is not None:
                scl_arr = np.asarray(scl_).astype(np.int32).reshape(H, W)
                w[np.isin(scl_arr, list(SCL_CLOUD_CLASSES))] = 0.0
        else:
            raise ValueError(f"Unknown weight_scheme: {weight_scheme}")
        w_arrays.append(w)

    # Step 3: weighted mean per band
    W_sum = np.zeros((H, W), dtype=np.float64)
    comp_b = np.zeros((H, W), dtype=np.float64)
    comp_g = np.zeros((H, W), dtype=np.float64)
    comp_r = np.zeros((H, W), dtype=np.float64)
    comp_n = np.zeros((H, W), dtype=np.float64)

    scene_weights_scalar = []
    for sc, w in zip(gated, w_arrays):
        w64 = w.astype(np.float64)
        comp_b += w64 * sc["blue"].astype(np.float64).reshape(H, W)
        comp_g += w64 * sc["green"].astype(np.float64).reshape(H, W)
        comp_r += w64 * sc["red"].astype(np.float64).reshape(H, W)
        comp_n += w64 * sc["nir"].astype(np.float64).reshape(H, W)
        W_sum  += w64
        scene_weights_scalar.append(float(w.mean()))

    # Where W_sum==0 (every scene has glint+cloud at this pixel): fall back to
    # unweighted mean to avoid NaN / division-by-zero.
    zero_mask = W_sum < 1e-12
    fallback_count = int(zero_mask.sum())
    if fallback_count > 0:
        n_count = np.zeros((H, W), dtype=np.float64)
        fb_b = np.zeros((H, W), dtype=np.float64)
        fb_g = np.zeros((H, W), dtype=np.float64)
        fb_r = np.zeros((H, W), dtype=np.float64)
        fb_n_ = np.zeros((H, W), dtype=np.float64)
        for sc in gated:
            fb_b += sc["blue"].astype(np.float64).reshape(H, W)
            fb_g += sc["green"].astype(np.float64).reshape(H, W)
            fb_r += sc["red"].astype(np.float64).reshape(H, W)
            fb_n_ += sc["nir"].astype(np.float64).reshape(H, W)
            n_count += 1.0
        n_count = np.where(n_count < 1, 1, n_count)
        comp_b[zero_mask] = (fb_b / n_count)[zero_mask]
        comp_g[zero_mask] = (fb_g / n_count)[zero_mask]
        comp_r[zero_mask] = (fb_r / n_count)[zero_mask]
        comp_n[zero_mask] = (fb_n_ / n_count)[zero_mask]
        W_sum[zero_mask] = 1.0
        L.info(f"  {fallback_count} pixels had W_sum=0 — unweighted fallback applied")

    blue_out  = (comp_b / W_sum).astype(np.float32)
    green_out = (comp_g / W_sum).astype(np.float32)
    red_out   = (comp_r / W_sum).astype(np.float32)
    nir_out   = (comp_n / W_sum).astype(np.float32)

    # Step 4: SCL — pixel-wise best-scene SCL (highest weight scene wins each pixel)
    w_stack = np.stack(w_arrays, axis=0)          # (K, H, W)
    best_scene_idx = np.argmax(w_stack, axis=0)   # (H, W)
    scl_out = np.zeros((H, W), dtype=np.float32)
    for k, sc in enumerate(gated):
        mask_k = best_scene_idx == k
        scl_arr = np.asarray(sc["scl"]).astype(np.float32).reshape(H, W)
        scl_out[mask_k] = scl_arr[mask_k]

    # Composite metadata
    dates_used = [sc.get("acq_date", "?")[:13] for sc in gated]
    scene_idx_used = [sc.get("scene_idx", i) for i, sc in enumerate(gated)]
    spms_used  = [sc["clarity_metrics"]["spm_median_g_m3"] for sc in gated]

    composite_meta = {
        "mode":              f"clarity_weighted_{weight_scheme}",
        "n_scenes_used":     n_scenes,
        "n_scenes_gated_out": len(ranked_scenes) - n_scenes,
        "scene_indices_used": scene_idx_used,
        "dates_used":         dates_used,
        "scene_weights_mean": scene_weights_scalar,
        "spm_per_scene":      spms_used,
        "fallback_px_count":  fallback_count,
        "weight_scheme":      weight_scheme,
        "z_max_approx_m":     float(np.clip(2.3 / max(np.mean(spms_used) * SPM_TO_KD_COEFF, 0.005),
                                            2.0, 30.0)),
    }

    composite = {
        "blue":         blue_out,
        "green":        green_out,
        "red":          red_out,
        "nir":          nir_out,
        "scl":          scl_out,
        "height":       gated[0]["height"],
        "width":        gated[0]["width"],
        "resolution_m": gated[0]["resolution_m"],
        "composite_mode": composite_meta["mode"],
        "clarity_metrics": {
            "spm_median_g_m3": float(np.mean(spms_used)),
            "z_max_approx_m":   composite_meta["z_max_approx_m"],
        },
    }

    L.info(f"  Composite: {n_scenes} scenes, SPM weighted-mean={np.mean(spms_used):.2f} g/m³, "
           f"z_max≈{composite_meta['z_max_approx_m']:.1f} m, "
           f"fallback_px={fallback_count}")
    return composite, composite_meta


def fetch_and_build_weighted_composite(
    bbox: List[float],
    start_date: str,
    end_date: str,
    *,
    cloud: int = 25,
    pool_size: int = 6,
    weight_scheme: str = "spm_inv",
    apply_hy6_gate: bool = True,
    existing_median_s2: Optional[Dict] = None,
) -> Tuple[Dict, List[Dict], Dict]:
    """Full pipeline: fetch pool → score → HY-6 gate → clarity-weighted composite.

    This is the CORRECTED HY-5 form: all gated scenes contribute to the composite
    (training-data count preserved).  The per-pixel 1/SPM weights down-weight turbid
    patches without discarding them, so the CNN still sees full spatial coverage.

    Parameters
    ----------
    bbox             : [W, S, E, N] EPSG:4326
    start_date, end_date : ISO date strings
    cloud            : CLOUDY_PIXEL_PERCENTAGE threshold for GEE pre-filter
    pool_size        : number of candidate scenes to fetch from GEE
    weight_scheme    : "spm_inv" (per-pixel) or "scene_spm_inv" (per-scene scalar)
    apply_hy6_gate   : apply HY-6 turbidity/glint gate before compositing
    existing_median_s2 : fallback if GEE unavailable

    Returns
    -------
    composite        : clarity-weighted composite dict
    ranked_scenes    : full ranked list (before gate)
    meta             : composite_meta + clarity_table + gate info
    """
    from backend.very_hr_engine import fetch_s2_gee_scenes, _init_gee

    meta: Dict = {
        "mode": "median_fallback",
        "pool_size_requested": pool_size,
        "n_fetched": 0,
        "weight_scheme": weight_scheme,
        "apply_hy6_gate": apply_hy6_gate,
        "clarity_table": "",
    }

    if not _init_gee():
        L.warning("fetch_and_build_weighted_composite: GEE unavailable — fallback to median")
        if existing_median_s2 is None:
            raise RuntimeError("GEE unavailable and no fallback median_s2 provided")
        return existing_median_s2, [], meta

    L.info(f"Fetching scene pool for weighted composite: bbox={bbox} "
           f"{start_date}→{end_date} cloud<{cloud}% pool_size={pool_size}")
    scenes = fetch_s2_gee_scenes(bbox, start_date, end_date,
                                  cloud=cloud, max_scenes=pool_size)

    if not scenes:
        L.warning("fetch_and_build_weighted_composite: no scenes — fallback to median")
        if existing_median_s2 is None:
            raise RuntimeError("No scenes returned and no fallback provided")
        return existing_median_s2, [], meta

    meta["n_fetched"] = len(scenes)

    ranked = rank_scenes(scenes)
    table  = clarity_table_str(ranked)
    L.info(f"\nClarity ranking table:\n{table}")
    meta["clarity_table"] = table

    composite, comp_meta = build_clarity_weighted_composite(
        ranked,
        apply_hy6_gate=apply_hy6_gate,
        weight_scheme=weight_scheme,
    )
    meta.update(comp_meta)

    # Normalise output schema to match fetch_s2_for_site
    out = {
        "blue":  composite["blue"].astype(np.float32),
        "green": composite["green"].astype(np.float32),
        "red":   composite["red"].astype(np.float32),
        "nir":   composite["nir"].astype(np.float32),
        "scl":   np.asarray(composite["scl"]).astype(np.float32),
        "height": composite["height"],
        "width":  composite["width"],
        "resolution_m": composite["resolution_m"],
        "composite_mode": composite.get("composite_mode"),
        "clarity_metrics": composite.get("clarity_metrics", {}),
    }
    return out, ranked, meta


def fetch_and_select_scene(
    bbox: List[float],
    start_date: str,
    end_date: str,
    *,
    cloud: int = 25,
    pool_size: int = 6,
    top_n: int = 1,
    existing_median_s2: Optional[Dict] = None,
) -> Tuple[Dict, List[Dict], Dict]:
    """Full pipeline: fetch pool → score → select.

    If GEE is unavailable or no scenes returned, falls back to the supplied
    ``existing_median_s2`` (the cached median composite), so the caller never
    regresses to a hard failure.

    Parameters
    ----------
    bbox             : [W, S, E, N] EPSG:4326
    start_date       : ISO date
    end_date         : ISO date
    cloud            : CLOUDY_PIXEL_PERCENTAGE threshold
    pool_size        : number of candidate scenes to request from GEE
    top_n            : 1 = single clearest; 2 = IV blend of top-2
    existing_median_s2 : fallback dict (same shape as fetch_s2_for_site output)

    Returns
    -------
    selected_scene   : chosen scene dict (or median fallback)
    ranked_scenes    : full ranked list (or empty if GEE failed)
    selection_meta   : dict with clarity table, chosen idx, mode, z_max
    """
    from backend.very_hr_engine import fetch_s2_gee_scenes, _init_gee

    selection_meta: Dict = {
        "mode": "median_fallback",
        "pool_size_requested": pool_size,
        "n_fetched": 0,
        "top_n": top_n,
        "clarity_table": "",
        "chosen_scene_idx": None,
        "chosen_date": None,
        "z_max_approx_m": None,
    }

    if not _init_gee():
        L.warning("fetch_and_select_scene: GEE unavailable — falling back to median composite")
        if existing_median_s2 is None:
            raise RuntimeError("GEE unavailable and no fallback median_s2 provided")
        return existing_median_s2, [], selection_meta

    L.info(f"Fetching scene pool: bbox={bbox} {start_date}→{end_date} "
           f"cloud<{cloud}% pool_size={pool_size}")
    scenes = fetch_s2_gee_scenes(
        bbox, start_date, end_date,
        cloud=cloud, max_scenes=pool_size,
    )

    if not scenes:
        L.warning("fetch_and_select_scene: no scenes returned — falling back to median composite")
        if existing_median_s2 is None:
            raise RuntimeError("No scenes returned and no fallback provided")
        return existing_median_s2, [], selection_meta

    selection_meta["n_fetched"] = len(scenes)

    # Score and rank
    ranked = rank_scenes(scenes)
    table  = clarity_table_str(ranked)
    L.info(f"\nClarity ranking table:\n{table}")
    selection_meta["clarity_table"] = table

    # Select
    chosen = select_best_scene(ranked, top_n=top_n)
    selection_meta["mode"] = f"clarity_{'single' if top_n == 1 else 'iv_blend'}"
    selection_meta["chosen_scene_idx"] = chosen.get("selected_indices", [])
    selection_meta["chosen_date"] = chosen.get("acq_date", "?")
    selection_meta["z_max_approx_m"] = chosen.get("clarity_metrics", {}).get("z_max_approx_m")
    selection_meta["clarity_score"] = chosen.get("clarity_score")

    # Normalise chosen scene to match fetch_s2_for_site output schema
    # (scl must be float32 for downstream compatibility)
    out = {
        "blue":  chosen["blue"].astype(np.float32),
        "green": chosen["green"].astype(np.float32),
        "red":   chosen["red"].astype(np.float32),
        "nir":   chosen["nir"].astype(np.float32),
        "scl":   np.asarray(chosen["scl"]).astype(np.float32),
        "height": chosen["height"],
        "width":  chosen["width"],
        "resolution_m": chosen["resolution_m"],
        "composite_mode": chosen.get("composite_mode", "single"),
        "clarity_score":  chosen.get("clarity_score"),
        "clarity_metrics": chosen.get("clarity_metrics", {}),
        "acq_date": chosen.get("acq_date", "?"),
    }
    return out, ranked, selection_meta
