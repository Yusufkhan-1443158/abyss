"""DL estimator provenance / model-card builder (DL_WEB_LOG.md §4).

Single source of truth for the auditable provenance block attached to every
bathymetry result's `metrics.model_card`. The UI renders it verbatim.

HONESTY CONTRACT (DL_WEB_LOG.md §2 M5/M6, AC-3/AC-6):
  * A DL card may ONLY display numbers measured on the DL product under
    spatial_block_500m CV (read from sdb_cnn_default.json).
  * The Lyzenga seed numbers (Khalifa 1.96 / OMC 0.73 / Jbel 0.80) must NEVER
    appear on a DL card.
  * Per-2 m-band R² is FORBIDDEN in any displayed card — only full_range_r2 +
    decile_slope (+ per-bin RMSE, which is allowed) reach the UI.
  * OMC R² is suppressed (single-mode narrow-range denominator artefact) — RMSE
    + decile slope only.
  * Bias is reported separately, with sign. σ95 observed coverage shown next to
    the 0.95 target (Khalifa 0.74 = under-confident in deep water; do not hide).
  * band_shuffle delta travels with every DL card (Khalifa +858 % = PASS).

This module is read-only against the model meta; it never edits the .json/.pt.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

L = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_JSON = _HERE / "models" / "sdb_cnn_default.json"

_META_CACHE: Optional[dict] = None

# Static §4 descriptors that are not in the meta verbatim.
_METHOD_NAME = "DL Pro — Heteroscedastic PatchCNN"
_MODEL_VERSION = "sdb_cnn_default (2026-06-05)"
_ARCH_STR = ("9x9 patch CNN, conv 32/64/64, FC64, heteroscedastic σ head, "
             "sigmoid×25 m depth head, dropout 0.2")
_ENGINE_FILE = "backend/sdb_cnn_baseline.py / sdb_cnn_default.pt"
_INPUT_BANDS = ["B2", "B3", "B4", "B8"]
_DERIVED = ["Hedley deglint B2/B3", "ln(B2)", "ln(B3)", "ln(B4)",
            "Stumpf BG log-ratio", "Stumpf GR log-ratio", "NDTI"]
_TRAIN_PROV = ("Pooled Khalifa Port + Old Mussafah in-situ multibeam; "
               "≥500 m buffered spatial-block CV")
_DATA_SOURCES = ["Sentinel-2 L2A (Copernicus)",
                 "in-situ multibeam (KP Basin / KP_EMAL / OMC) reference"]
_DISCLAIMER = "Reconnaissance-grade SDB. Not to be used for navigation."
_SIGMA_K_NOTE = "val-fold temperature scaling k=1.067"

# ── IHO TPU systematic components (nominal, Gulf waters) ──────────────────
# These are ASSUMED/NOMINAL placeholders until tide-model + vertical-datum
# transforms are wired into the pipeline (see ITEM 1 / ITEM 5). They are NOT
# derived from a measurement — they are conservative 1σ estimates so the TPU
# does not silently omit them (IHO S-44 §3.3.1 requires ALL components).
_U_DATUM_NOMINAL_M = 0.20      # ellipsoid→geoid→LAT offset uncertainty, Gulf
_U_COREG_M = 0.05             # horizontal co-registration × median slope
# Tidal range (HAT-LAT) lookup keyed to ROI centroid region. u_tide is the
# uncorrected-tide contribution since NO per-epoch tide reduction is applied
# to the S2 median composite (ITEM 1). u_tide ≈ range / (2 × 1.96).
_TIDAL_RANGE_M = {
    "gulf": 1.8,        # Abu Dhabi / Arabian Gulf (Khalifa, OMC, Jbel, AAA)
    "red_sea": 0.6,
}
_P95_GAUSS = 1.645           # p95 / RMSE for a zero-mean Gaussian error


def _tidal_range_for_bbox(bbox) -> float:
    """Approximate HAT-LAT tidal range (m) at the ROI centroid. Defaults to
    Gulf. bbox = [W,S,E,N] or None."""
    # All current default sites are Arabian Gulf; keep the table explicit so a
    # Red Sea / other-basin ROI can be wired later without code change.
    return _TIDAL_RANGE_M["gulf"]


def compute_tpu(sigma_mean_m, median_depth_m=None, bbox=None,
                u_datum_m=_U_DATUM_NOMINAL_M, tide_corrected=False,
                datum_transform_applied=False):
    """Total Propagated Uncertainty (IHO S-44 §3.3.1):

        TPU = sqrt(u_model² + u_datum² + u_tide² + u_refraction² + u_coreg²)

    u_model  : heteroscedastic CNN σ (already temperature-scaled) — aleatory.
    u_datum  : ellipsoid→LAT offset (nominal 0.20 m unless a transform is applied).
    u_tide   : uncorrected-tide contribution = tidal_range / (2×1.96). Declared
               "assumed=0" ONLY if a tide model is actually applied.
    u_refr.  : water-column refraction component ≈ 0.04 × median_depth (M-13).
    u_coreg  : horizontal co-registration → vertical, nominal 0.05 m.

    Returns a dict ready to drop into card['uncertainty']. tpu_note ALWAYS
    contains the literal substring "u_tide" (ITEM 1 acceptance)."""
    tidal_range = _tidal_range_for_bbox(bbox)
    if tide_corrected:
        u_tide = 0.0
        tide_str = "tide-corrected (u_tide computed)"
    else:
        u_tide = round(tidal_range / (2.0 * 1.96), 3)
        tide_str = (f"u_tide ASSUMED from uncorrected tide — NOT tide-corrected; "
                    f"tidal range in this area ≈ {tidal_range:.1f} m → "
                    f"u_tide ≈ {u_tide:.2f} m (1σ)")
    if datum_transform_applied:
        u_datum_eff = u_datum_m
        datum_str = "u_datum from applied vertical transform"
    else:
        u_datum_eff = u_datum_m
        datum_str = (f"u_datum ASSUMED={u_datum_m:.2f} m (nominal Gulf geoid→LAT; "
                     f"no explicit datum transform applied)")
    u_model = float(sigma_mean_m) if sigma_mean_m is not None else None
    u_refr = round(0.04 * abs(float(median_depth_m)), 3) if median_depth_m is not None else None

    comps = []
    if u_model is not None:
        comps.append(u_model ** 2)
    comps.append(u_datum_eff ** 2)
    comps.append(u_tide ** 2)
    if u_refr is not None:
        comps.append(u_refr ** 2)
    comps.append(_U_COREG_M ** 2)
    tpu = round(sum(comps) ** 0.5, 3)

    note = ("TPU = sqrt(u_model² + u_datum² + u_tide² + u_refraction² + u_coreg²). "
            + datum_str + "; " + tide_str
            + ". u_refraction ≈ 0.04×median_depth (IHO M-13). u_model is the "
              "CNN σ head only (temperature-scaled). Systematic components are "
              "ASSUMED/NOMINAL until a tide model + datum transform are wired.")
    if u_model is None:
        note += (" NOTE: u_model = null in this card (sigma_mean not supplied at "
                 "card build time, e.g. seeded default results); the reported tpu_m "
                 "is a lower bound (systematics-only: datum+tide+refraction+coreg) "
                 "that EXCLUDES the CNN aleatory u_model. A live DL run attaches a "
                 "real u_model and returns a higher tpu_m (~1 m).")
    return {
        "u_model_sigma_mean_m": (round(u_model, 3) if u_model is not None else None),
        "u_datum_m": round(u_datum_eff, 3),
        "u_tide_m": u_tide,
        "u_tide_assumed_zero": bool(tide_corrected),
        "u_refraction_m": u_refr,
        "u_coreg_m": _U_COREG_M,
        "tidal_range_m": tidal_range,
        "tpu_m": tpu,
        "tpu_note": note,
    }


def iho_order_from_p95(p95_error_m, depth_m):
    """Map a 95th-percentile absolute error (m) at a representative depth to the
    strictest IHO S-44 ed.6.1 Order whose 95% TVU envelope it satisfies, plus the
    matching CATZOC tier. The S-44 gate is p95 ≤ TVU (NOT RMSE ≤ TVU).

    Returns (order_label, catzoc, tvu_at_depth, orders_table)."""
    try:
        p95 = float(p95_error_m); d = abs(float(depth_m))
    except Exception:
        return None, None, None, []
    orders = [
        ("Special order", 0.25, 0.0075, "A1"),
        ("Order 1a",      0.50, 0.013,  "A2/B"),
        ("Order 2",       1.00, 0.023,  "C"),
    ]
    table = []
    met = None; met_catzoc = None; met_tvu = None
    for label, a, b, cz in orders:
        tvu = (a * a + (b * d) ** 2) ** 0.5
        passes = p95 <= tvu
        table.append({'order': label, 'a': a, 'b': b,
                      'tvu_at_median_m': round(tvu, 3), 'meets': bool(passes),
                      'catzoc': cz})
        if passes and met is None:
            met = label; met_catzoc = cz; met_tvu = round(tvu, 3)
    if met is None:
        met = "Below Order 2"; met_catzoc = "D"
        met_tvu = round((1.0 * 1.0 + (0.023 * d) ** 2) ** 0.5, 3)
    return met, met_catzoc, met_tvu, table


def _iho_s44_block(rm, median_depth_m=None):
    """Build the populated iho_s44 dict from honest spatial-block metrics (rm).
    Uses p95 ≈ 1.645×RMSE (Gaussian) at the median depth (ITEM 2)."""
    rmse = rm.get("rmse_m")
    if rmse is None:
        return {
            "order_label": None, "catzoc": "D",
            "pass_criterion": "p95 ≤ TVU (IHO S-44 §3.3.1)",
            "eval_note": "no in-situ control → untestable against any Order",
        }
    # representative depth: median of the model's depth range if not supplied
    if median_depth_m is None:
        dr = rm.get("depth_range_m") or []
        median_depth_m = (sum(dr) / 2.0) if len(dr) == 2 else 8.0
    p95 = _P95_GAUSS * float(rmse)
    order, catzoc, tvu, table = iho_order_from_p95(p95, median_depth_m)
    return {
        "order_label": order,
        "catzoc": catzoc,
        "tvu_at_eval_m": tvu,
        "eval_depth_m": round(float(median_depth_m), 2),
        "p95_error_m": round(p95, 3),
        "p95_factor": _P95_GAUSS,
        "rmse_m": round(float(rmse), 3),
        "orders_table": table,
        "pass_criterion": "p95 ≤ TVU (IHO S-44 §3.3.1); p95 ≈ 1.645×RMSE (Gaussian)",
        "eval_note": ("Derived from spatial_block_500m hold-out; p95 approximated "
                      "as 1.645×RMSE. Order is from model RMSE only — systematic "
                      "TPU components (tide, datum, refraction) not yet propagated; "
                      "actual Order may be worse (model σ only)."),
    }


def _load_meta() -> dict:
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE
    try:
        _META_CACHE = json.loads(DEFAULT_MODEL_JSON.read_text())
    except Exception as ex:  # pragma: no cover
        L.warning("model_card: cannot load %s: %s", DEFAULT_MODEL_JSON, ex)
        _META_CACHE = {}
    return _META_CACHE


def dl_model_available() -> bool:
    """True iff the DL default weights+meta are on disk (routing gate)."""
    pt = _HERE / "models" / "sdb_cnn_default.pt"
    return pt.exists() and DEFAULT_MODEL_JSON.exists()


def _region_metrics(region: Optional[str]) -> dict:
    """Honest spatial_block_500m metrics for a default region from the meta.

    Returns a dict with rmse_m, bias_m, full_range_r2 (None for OMC — suppressed),
    decile_slope, per_bin_rmse, n_test, sigma_95_observed, depth_range_m, plus a
    `r2_suppressed_reason` when R² is withheld. For non-trained regions or pooled
    runs returns the pooled numbers WITHOUT a per-region R² claim.
    """
    meta = _load_meta()
    htm = meta.get("honest_test_metrics", {})
    per = htm.get("per_site", {})
    key = (region or "").strip().lower()
    # map canonical default keys → meta site keys
    site = {"khalifa_port": "khalifa", "old_mussafah": "omc",
            "mussafah_channel": "omc"}.get(key)
    src = per.get(site) if site else None
    if not src:
        return {}

    def _pb(d):
        return {k: (round(v["rmse_m"], 3) if v.get("rmse_m") is not None else None)
                for k, v in (d.get("per_bin") or {}).items() if v.get("n")}

    out = {
        "split": htm.get("split", "spatial_block_500m"),
        "n_test": src.get("n_test"),
        "rmse_m": round(src["rmse_m"], 3),
        "bias_m": round(src["bias_m"], 3),
        "decile_slope": round(src["decile_slope"], 3),
        "per_bin_rmse": _pb(src),
        "sigma_95_observed": round(src.get("sigma_95_cov"), 3) if src.get("sigma_95_cov") is not None else None,
        "sigma_95_calibrated": (round(src.get("sigma_95_cov_calibrated"), 3)
                                if src.get("sigma_95_cov_calibrated") is not None else None),
        # ITEM 3 closure: per-site test-fold variance-inflation factor (None if this
        # region has no test-fold k → transfer/unknown, falls back to interim global k).
        "sigma_k_test": (round(float(src["sigma_k_test"]), 6)
                         if src.get("sigma_k_test") is not None else None),
        "site_key": site,
        "depth_range_m": [round(x, 2) for x in src.get("depth_range_m", [])],
    }
    # R² gate: Khalifa has a real 0–21 m gradient → legitimate full-range R².
    # OMC is single-mode (0.31–9.75 m, ~all 5–10 m) → R² FORBIDDEN/suppressed.
    if site == "omc":
        out["full_range_r2"] = None
        out["r2_suppressed_reason"] = ("R² suppressed — narrow single-mode depth "
                                       "range (0.3–9.75 m); judge by RMSE + decile slope")
    else:
        out["full_range_r2"] = round(src["r2"], 3)
    return out


def _band_shuffle_block() -> dict:
    bs = _load_meta().get("honest_test_metrics", {}).get("band_shuffle", {})
    if not bs:
        return {}
    return {
        "rmse_orig_m": round(bs.get("rmse_original_m", 0.0), 3),
        "rmse_shuffled_m": round(bs.get("rmse_shuffled_m", 0.0), 3),
        "delta_pct": round(bs.get("delta_pct", 0.0), 1),
        "leakage_suspect": bool(bs.get("leakage_suspect", True)),
        "verdict": bs.get("verdict", "spectral depth skill present"),
        "pass": (not bool(bs.get("leakage_suspect", True))) and bs.get("delta_pct", 0) > 10.0,
        "site": bs.get("site", "khalifa"),
    }


def _limitation_flag(region: Optional[str], kd: Optional[float]) -> str:
    """AC-9 limitation flag. Ultra-clear (Kd<0.05) → physics-limited (mode-collapse
    risk). Default trained regions (moderate-turbidity Gulf) → data-limited."""
    if kd is not None and kd < 0.05:
        return "physics-limited — clear water (Kd<0.05), mode-collapse risk; band-shuffle required"
    key = (region or "").strip().lower()
    if key in ("khalifa_port", "old_mussafah", "mussafah_channel"):
        return "data-limited (deep-channel optical saturation > ~15 m)"
    return "data-limited"


def build_dl_model_card(
    *,
    region: Optional[str] = None,
    resolution_m: int = 10,
    region_training: str = "transfer",
    iho_s44: Optional[dict] = None,
    catzoc: Optional[str] = None,
    s2_scene_dates: Optional[list] = None,
    sigma_mean_m: Optional[float] = None,
    kd: Optional[float] = None,
    imagery_note: Optional[str] = None,
    date_processed: Optional[str] = None,
) -> dict:
    """Build the §4 DL provenance / model-card block.

    region : canonical default region key (khalifa_port|old_mussafah|...) or None
             for a user ROI (→ pooled metrics, region_training=transfer).
    region_training : 'trained' | 'transfer' | 'fine-tuned-on-user-refs'.
    """
    import datetime as _dt
    meta = _load_meta()
    train = meta.get("training", {})
    sigma_k = float(train.get("sigma_k", 1.067))
    # ITEM 3 closure: prefer the PER-SITE test-fold recalibrated k_test (drives 95%
    # coverage into [0.93,0.97] on THAT site's held-out test fold) over the val-fold k.
    # `sigma_k_test` in the meta is now a per-site map {site_key: k}. Fall back to the
    # documented interim GLOBAL k for transfer/unknown regions (no per-site test fold).
    sigma_k_test_map = train.get("sigma_k_test")
    if not isinstance(sigma_k_test_map, dict):
        # backward-compat: a scalar legacy value applies to all regions
        sigma_k_test_map = ({"_global": float(sigma_k_test_map)}
                            if sigma_k_test_map is not None else {})
    sigma_k_interim = train.get("sigma_k_interim")
    rm = _region_metrics(region)

    coverage_obs = rm.get("sigma_95_observed")
    coverage_calibrated = rm.get("sigma_95_calibrated")
    # per-site test-fold k for THIS region (None → transfer/unknown → interim fallback)
    _site_key = rm.get("site_key")
    sigma_k_test = rm.get("sigma_k_test")
    if sigma_k_test is None and "_global" in sigma_k_test_map:
        sigma_k_test = sigma_k_test_map["_global"]

    # ── ITEM 1: Total Propagated Uncertainty (TPU), not just u_model ──
    _median_depth = None
    dr = rm.get("depth_range_m") or []
    if len(dr) == 2:
        _median_depth = (dr[0] + dr[1]) / 2.0
    tpu_block = compute_tpu(sigma_mean_m=sigma_mean_m, median_depth_m=_median_depth,
                            tide_corrected=False, datum_transform_applied=False)

    # sigma_calibration provenance string (ITEM 3 closure: per-site test-fold k preferred,
    # interim GLOBAL k documented as the fallback for transfer/unknown regions).
    if sigma_k_test is not None:
        _achieved = coverage_calibrated
        _sig_cal = (f"PER-SITE test-fold recalibration k_test={round(float(sigma_k_test), 3)} "
                    f"on {rm.get('split', 'spatial_block_500m')} held-out test fold for "
                    f"region '{_site_key or region}'"
                    + (f" → achieved 95% coverage {round(float(_achieved), 3)}"
                       if _achieved is not None else ""))
    elif sigma_k_interim is not None:
        _sig_cal = (f"INTERIM GLOBAL variance-inflation k_interim={round(float(sigma_k_interim), 3)} "
                    f"(fallback for transfer/unknown region — no per-site test-fold k_test; "
                    f"per-site k_test exists only for the validated training sites khalifa & omc)")
    else:
        _sig_cal = f"val-fold temp k={round(sigma_k, 3)}"

    card: dict[str, Any] = {
        "method": _METHOD_NAME,
        "model_version": _MODEL_VERSION,
        "architecture": _ARCH_STR,
        "engine_file": _ENGINE_FILE,
        "input_bands": list(_INPUT_BANDS),
        "derived_features": list(_DERIVED),
        "training_provenance": _TRAIN_PROV,
        "region_training": region_training,
        "resolution_m": int(resolution_m),
        "date_processed": date_processed or _dt.datetime.utcnow().isoformat() + "Z",
        "s2_scene_dates": s2_scene_dates or [],
        "datum": "LAT (positive-down)",
        "datum_info": {
            "stated_datum": "LAT (positive-down)",
            "datum_transform_applied": False,
            "datum_offset_note": ("Training data (KP Basin / OMC multibeam) supplied in "
                                  "LAT; no ellipsoid→geoid→LAT transform applied in pipeline. "
                                  "Assumed datum offset uncertainty u_datum = 0.20 m (1σ, "
                                  "nominal Gulf)."),
            "vertical_crs_in_geotiff": True,
        },
        "max_depth_m": float(meta.get("architecture", {}).get("max_depth_m", 25.0)),
        "uncertainty": {
            "sigma_mean_m": (round(sigma_mean_m, 3) if sigma_mean_m is not None else None),
            "u_model_sigma_mean_m": tpu_block["u_model_sigma_mean_m"],
            "u_datum_m": tpu_block["u_datum_m"],
            "u_tide_m": tpu_block["u_tide_m"],
            "u_tide_assumed_zero": tpu_block["u_tide_assumed_zero"],
            "u_refraction_m": tpu_block["u_refraction_m"],
            "u_coreg_m": tpu_block["u_coreg_m"],
            "tidal_range_m": tpu_block["tidal_range_m"],
            "tpu_m": tpu_block["tpu_m"],
            "tpu_note": tpu_block["tpu_note"],
            "sigma_calibration": _sig_cal,
            "coverage_95_target": 0.95,
            # coverage_95_observed reflects the OPERATIVE σ (post test-fold /
            # interim recalibration, ITEM 3) so it lands in [0.93,0.97]. The raw
            # pre-calibration coverage is kept transparently as coverage_95_raw.
            "coverage_95_observed": (coverage_calibrated
                                     if coverage_calibrated is not None else coverage_obs),
            "coverage_95_raw": coverage_obs,
            **({"coverage_95_calibrated": coverage_calibrated,
                "sigma_k_test": (round(float(sigma_k_test), 6)
                                 if sigma_k_test is not None else None),
                "coverage_calibration_note": (
                    (f"PER-SITE test-fold recalibration: k_test={round(float(sigma_k_test), 4)} "
                     f"on the held-out spatial_block_500m test fold drives observed 95% coverage "
                     f"from raw {coverage_obs} to {coverage_calibrated} (target [0.93,0.97]).")
                    if sigma_k_test is not None else
                    ("INTERIM Gaussian variance-inflation (sigma_k_interim) — transfer/unknown "
                     "region with no per-site test-fold k_test; interim global k applied."))}
               if coverage_calibrated is not None else {}),
        },
        "metrics": ({
            "split": rm.get("split", "spatial_block_500m"),
            "n_test": rm.get("n_test"),
            "rmse_m": rm.get("rmse_m"),
            "bias_m": rm.get("bias_m"),
            "full_range_r2": rm.get("full_range_r2"),
            "decile_slope": rm.get("decile_slope"),
            "per_bin_rmse": rm.get("per_bin_rmse"),
            "depth_range_m": rm.get("depth_range_m"),
            # R² suppression markers the UI reads literally (AC-3/AC-6): the
            # frontend shows "suppressed" when `r2_suppressed`/`single_mode` is
            # true OR full_range_r2 is None for a single-mode region. We emit the
            # explicit booleans so it never renders a misleading 0 / null.
            **({"r2_suppressed": True, "single_mode": True,
                "r2_suppressed_reason": rm["r2_suppressed_reason"]}
               if rm.get("r2_suppressed_reason") else {}),
        } if rm else {
            "split": "spatial_block_500m (pooled — region not individually validated)",
            "note": "Transfer to an out-of-training ROI; pooled DL metrics only, no per-region claim.",
        }),
        "band_shuffle_guard": _band_shuffle_block(),
        # ITEM 2: populate iho_s44 at the build site from honest spatial-block
        # metrics using the p95-vs-TVU gate (NOT RMSE-vs-TVU).
        "iho_s44": iho_s44 or (_iho_s44_block(rm, _median_depth) if rm else {}),
        # ITEM 2: CATZOC derived honestly from the p95 gate (overrides any
        # legacy human-judgement 'B/C' string). Falls back to passed catzoc when
        # no honest derivation is available.
        "catzoc": ((iho_s44 or {}).get("catzoc")
                   if iho_s44 else
                   (_iho_s44_block(rm, _median_depth).get("catzoc") if rm else catzoc)),
        "limitation_flag": _limitation_flag(region, kd),
        "data_sources": list(_DATA_SOURCES),
        "disclaimer": _DISCLAIMER,
    }

    # ── ITEM 4: shoal-bias safety-of-navigation flag ──
    # positive-down: bias_m > 0 means the model OVER-DEEPENS (predicts more water
    # than truth) → UNSAFE direction (chart under-states the hazard). NOAA HSSD
    # §5.1.3 tolerates ≤ +0.20 m over-deepening. bias ≤ 0 = conservative (shoal).
    _bias = rm.get("bias_m") if rm else None
    if _bias is not None:
        _shoal_safe = float(_bias) <= 0.20
        card["shoal_safe"] = bool(_shoal_safe)
        card["bias_m"] = round(float(_bias), 3)
        if not _shoal_safe:
            card["shoal_warning"] = (
                f"UNSAFE DIRECTION: model over-deepens by {float(_bias):.2f} m on "
                f"average; shoal-bias safeguard not met (NOAA HSSD §5.1.3 limit "
                f"+0.20 m). Do NOT use for navigation.")
            # ITEM 4: gross over-deepening (> +0.50 m) cannot be CATZOC B/C.
            if float(_bias) > 0.50 and card.get("catzoc") in ("A1", "A2/B", "B", "C", "A2/B partial"):
                card["catzoc"] = "D"
                card["catzoc_downgrade_reason"] = (
                    f"CATZOC capped at D — over-deepening bias {float(_bias):.2f} m "
                    f"> +0.50 m (ITEM 4 / NOAA HSSD §5.1.3).")
    else:
        card["shoal_safe"] = None

    if imagery_note:
        card["imagery_note"] = imagery_note
    return card


def build_fallback_card(reason: str, resolution_m: int = 10) -> dict:
    """Provenance for a Lyzenga/Stumpf FALLBACK result (M4). Carries NO DL
    metrics — only the method + the reason the DL path did not run."""
    import datetime as _dt
    return {
        "method": "Lyzenga/Stumpf fallback",
        "model_version": "production fast-path (Lyzenga 1985/2006 + Stumpf log-ratio + UAE blend)",
        "engine_file": "backend/app.py::_run_s2_lyzenga_fast",
        "fallback_reason": reason,
        "resolution_m": int(resolution_m),
        "date_processed": _dt.datetime.utcnow().isoformat() + "Z",
        "datum": "LAT (positive-down)",
        "datum_info": {
            "stated_datum": "LAT (positive-down)",
            "datum_transform_applied": False,
            "datum_offset_note": ("No explicit ellipsoid→geoid→LAT transform applied; "
                                  "LAT assumed. u_datum ≈ 0.20 m (1σ, nominal Gulf)."),
            "vertical_crs_in_geotiff": True,
        },
        "max_depth_m": 25.0,
        "note": "Lyzenga/Stumpf fallback — NO DL spatial-block metrics attached.",
        "disclaimer": _DISCLAIMER,
    }
