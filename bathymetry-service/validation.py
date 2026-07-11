"""IHO S-44 reference-sounding validation for depth products.

Vendored from Bathymetry_Production/backend/model_card.py (iho_order_from_p95)
and backend/app.py (api_validate_xyz core math), stripped of Flask and photon
classification: Abyss predictions are grid samples, plain {lat, lon, depth}.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

L = logging.getLogger("bathymetry.validation")

_P95_GAUSS = 1.645  # p95 / RMSE for a zero-mean Gaussian error

CLASS_BINS = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 9e9]
CLASS_LABELS = ["0-2", "2-5", "5-10", "10-15", "15-20", ">20"]

_TPU_CAVEAT = ("RMSE/bias reported here are TVU-only (no THU assessment); "
               "tide/datum/refraction TPU terms not included unless the "
               "uploaded survey and the model output share a stated "
               "vertical datum — see export metadata.")


class ValidationError(ValueError):
    """Bad request payload — maps to HTTP 400."""


def iho_order_from_p95(p95_error_m, depth_m):
    """Map a 95th-percentile absolute error (m) at a representative depth to
    the strictest IHO S-44 ed.6.1 Order whose 95% TVU envelope it satisfies,
    plus the matching CATZOC tier. The S-44 gate is p95 <= TVU (NOT RMSE).

    Returns (order_label, catzoc, tvu_at_depth, orders_table)."""
    try:
        p95 = float(p95_error_m)
        d = abs(float(depth_m))
    except Exception:
        return None, None, None, []
    orders = [
        ("Special order", 0.25, 0.0075, "A1"),
        ("Order 1a",      0.50, 0.013,  "A2/B"),
        ("Order 2",       1.00, 0.023,  "C"),
    ]
    table = []
    met = None
    met_catzoc = None
    met_tvu = None
    for label, a, b, cz in orders:
        tvu = (a * a + (b * d) ** 2) ** 0.5
        passes = p95 <= tvu
        table.append({"order": label, "a": a, "b": b,
                      "tvu_at_median_m": round(tvu, 3), "meets": bool(passes),
                      "catzoc": cz})
        if passes and met is None:
            met = label
            met_catzoc = cz
            met_tvu = round(tvu, 3)
    if met is None:
        met = "Below Order 2"
        met_catzoc = "D"
        met_tvu = round((1.0 + (0.023 * d) ** 2) ** 0.5, 3)
    return met, met_catzoc, met_tvu, table


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _cls(d):
    for i in range(len(CLASS_BINS) - 1):
        if d < CLASS_BINS[i + 1]:
            return i
    return len(CLASS_LABELS) - 1


def validate_points(predicted: List[dict], observed: List[dict],
                    max_match_m: float = 50.0,
                    resolution_m: Optional[float] = None,
                    observed_vertical_datum: Optional[str] = None,
                    predicted_vertical_datum: str = "LAT") -> Dict[str, Any]:
    """Compare predicted depth samples against reference soundings.

    predicted/observed: lists of {lat, lon, depth} (metres, positive down).
    Returns surveyor-grade stats: RMSE/MAE/MedAE/bias/R2/Pearson/MAPE, per-order
    IHO S-44 TVU pass %, p95 -> Order/CATZOC, depth-stratified stats, residual
    histogram, 12x12 spatial residual grid, S-52 class confusion, datum flags.
    Raises ValidationError (-> HTTP 400) on malformed/empty inputs."""
    if not isinstance(predicted, list):
        raise ValidationError("'predicted' must be a list of {lat, lon, depth} objects")
    if not isinstance(observed, list):
        raise ValidationError("'observed' must be a list of {lat, lon, depth} objects")
    if not observed:
        raise ValidationError("No observed points")
    for i, p in enumerate(predicted):
        if not isinstance(p, dict):
            raise ValidationError(f"predicted[{i}] must be an object with lat/lon/depth, "
                                  f"got {type(p).__name__}")

    try:
        resolution_m = float(resolution_m) if resolution_m is not None else None
        if resolution_m is not None and not math.isfinite(resolution_m):
            resolution_m = None
    except Exception:
        resolution_m = None

    if isinstance(observed_vertical_datum, str):
        observed_vertical_datum = observed_vertical_datum.strip().upper() or None
    else:
        observed_vertical_datum = None
    predicted_vertical_datum = (str(predicted_vertical_datum or "LAT")).strip().upper()

    def _keep(p):
        try:
            return float(p.get("depth", 0)) > 0
        except Exception:
            return False

    kept = [p for p in predicted if _keep(p)]
    for i, p in enumerate(kept):
        if "lat" not in p or "lon" not in p:
            missing = "lat" if "lat" not in p else "lon"
            raise ValidationError(f"predicted[{i}] missing '{missing}'")
    pred_pts = kept
    if not pred_pts:
        raise ValidationError("No predicted depths")

    pred_lats = np.array([p["lat"] for p in pred_pts], dtype=np.float64)
    pred_lons = np.array([p["lon"] for p in pred_pts], dtype=np.float64)
    pred_depths = np.array([p["depth"] for p in pred_pts], dtype=np.float64)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([pred_lats, pred_lons]))
        use_tree = True
    except Exception:
        tree = None
        use_tree = False

    try:
        max_match = float(max_match_m or 50.0)
    except Exception:
        max_match = 50.0

    pairs_full = []
    match_dists_m = []
    n_excluded_by_distance = 0
    n_rejected_nonfinite = 0
    for op in observed:
        try:
            olat = float(op["lat"])
            olon = float(op["lon"])
            od = abs(float(op["depth"]))
        except Exception:
            continue
        if not (math.isfinite(olat) and math.isfinite(olon) and math.isfinite(od)):
            n_rejected_nonfinite += 1
            continue
        if od <= 0:
            continue
        if use_tree:
            _, idx = tree.query([olat, olon], k=1)
        else:
            d2 = (pred_lats - olat) ** 2 + (pred_lons - olon) ** 2
            idx = int(np.argmin(d2))
        idx = int(idx)
        dist_m = _haversine_m(olat, olon, float(pred_lats[idx]), float(pred_lons[idx]))
        if dist_m > max_match:
            n_excluded_by_distance += 1
            continue
        pd_val = float(pred_depths[idx])
        pairs_full.append({"lat": olat, "lon": olon, "obs": od, "pred": pd_val,
                           "diff": pd_val - od, "dist_m": round(dist_m, 2)})
        match_dists_m.append(dist_m)

    n_pairs = len(pairs_full)
    if n_pairs == 0:
        raise ValidationError(f"No matching points found (nearest neighbour > "
                              f"{max_match:.0f} m); "
                              f"{n_excluded_by_distance} excluded by distance")

    match_distance_stats_m = {
        "mean": round(float(np.mean(match_dists_m)), 2),
        "p95": round(float(np.percentile(match_dists_m, 95)), 2),
        "max": round(float(np.max(match_dists_m)), 2),
    }

    obs_arr = np.array([p["obs"] for p in pairs_full])
    pred_arr = np.array([p["pred"] for p in pairs_full])
    diffs = pred_arr - obs_arr
    abs_diffs = np.abs(diffs)

    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    mae = float(np.mean(abs_diffs))
    medae = float(np.median(abs_diffs))
    bias = float(np.mean(diffs))
    std = float(np.std(diffs))
    ss_res = float(np.sum(diffs ** 2))
    ss_tot = float(np.sum((obs_arr - obs_arr.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if n_pairs >= 2 and float(np.std(obs_arr)) > 0 and float(np.std(pred_arr)) > 0:
        pearson = float(np.corrcoef(obs_arr, pred_arr)[0, 1])
        if not math.isfinite(pearson):
            pearson = 0.0
    else:
        pearson = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_vals = np.where(obs_arr > 0.2, abs_diffs / obs_arr * 100, np.nan)
    mape = float(np.nanmean(mape_vals)) if np.any(np.isfinite(mape_vals)) else 0.0

    # Multi-order IHO S-44 TVU compliance (Special / 1a / 2).
    orders = {"special": (0.25, 0.0075), "order1a": (0.5, 0.013), "order2": (1.0, 0.023)}
    order_results = {}
    for k, (a, b) in orders.items():
        tvus = np.sqrt(a * a + (b * obs_arr) ** 2)
        passes = abs_diffs <= tvus
        order_results[k] = {
            "a": a, "b": b,
            "pass_pct": round(float(passes.sum()) / n_pairs * 100, 2),
            "n_pass": int(passes.sum()),
            "n_fail": int(n_pairs - int(passes.sum())),
        }

    tvu_1a = np.sqrt(0.5 ** 2 + (0.013 * obs_arr) ** 2)
    pass_1a = abs_diffs <= tvu_1a

    strat = []
    for i in range(len(CLASS_BINS) - 1):
        lo, hi = CLASS_BINS[i], CLASS_BINS[i + 1]
        mask = (obs_arr >= lo) & (obs_arr < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        sd = diffs[mask]
        sad = abs_diffs[mask]
        strat.append({
            "range": CLASS_LABELS[i],
            "n": n,
            "rmse": round(float(np.sqrt(np.mean(sd ** 2))), 3),
            "mae": round(float(np.mean(sad)), 3),
            "bias": round(float(np.mean(sd)), 3),
            "pass_1a": round(float(pass_1a[mask].sum()) / n * 100, 1),
        })

    # Residual histogram — 25 bins, symmetric range.
    lim = max(2.0, float(np.percentile(abs_diffs, 99)) * 1.1)
    lim = min(lim, 15.0)
    bins = 25
    hist, edges = np.histogram(diffs, bins=bins, range=(-lim, lim))
    hist_data = [{"x": round(float((edges[i] + edges[i + 1]) / 2), 3),
                  "count": int(hist[i])} for i in range(bins)]

    # 12x12 spatial mean-residual grid over the observed bbox.
    obs_lat_arr = np.array([p["lat"] for p in pairs_full])
    obs_lon_arr = np.array([p["lon"] for p in pairs_full])
    lat_lo, lat_hi = float(obs_lat_arr.min()), float(obs_lat_arr.max())
    lon_lo, lon_hi = float(obs_lon_arr.min()), float(obs_lon_arr.max())
    GH, GW = 12, 12
    sum_g = np.zeros((GH, GW), dtype=np.float64)
    cnt_g = np.zeros((GH, GW), dtype=np.int32)
    if lat_hi > lat_lo and lon_hi > lon_lo:
        for i in range(n_pairs):
            r = int((lat_hi - obs_lat_arr[i]) / (lat_hi - lat_lo + 1e-10) * (GH - 1))
            c = int((obs_lon_arr[i] - lon_lo) / (lon_hi - lon_lo + 1e-10) * (GW - 1))
            r = max(0, min(GH - 1, r))
            c = max(0, min(GW - 1, c))
            sum_g[r, c] += diffs[i]
            cnt_g[r, c] += 1
    with np.errstate(all="ignore"):
        mean_g = np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), np.nan)
    spatial_grid = {
        "bounds": {"south": lat_lo, "north": lat_hi, "west": lon_lo, "east": lon_hi},
        "h": GH, "w": GW,
        "cells": [[{"mean": None if not np.isfinite(mean_g[r, c]) else round(float(mean_g[r, c]), 3),
                    "n": int(cnt_g[r, c])}
                   for c in range(GW)] for r in range(GH)],
    }

    # IHO S-52 depth-class confusion matrix.
    cm = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int32)
    for i in range(n_pairs):
        cm[_cls(obs_arr[i])][_cls(pred_arr[i])] += 1
    class_agreement = round(float(np.trace(cm)) / n_pairs * 100, 1) if n_pairs else 0.0

    pairs_details = [{
        "lat": round(p["lat"], 6),
        "lon": round(p["lon"], 6),
        "obs_depth": round(p["obs"], 2),
        "pred_depth": round(p["pred"], 2),
        "diff": round(p["diff"], 2),
        "s44_pass": bool(pass_1a[i]),
        "match_dist_m": p["dist_m"],
    } for i, p in enumerate(pairs_full[:3000])]

    pair_stats = [{"obs": round(float(p["obs"]), 3), "pred": round(float(p["pred"]), 3),
                   "diff": round(float(p["diff"]), 3), "s44_pass": bool(pass_1a[i]),
                   "s44_max": round(float(tvu_1a[i]), 3)}
                  for i, p in enumerate(pairs_full[:3000])]

    p95_error_m = float(np.percentile(abs_diffs, 95))
    med_obs_depth = float(np.median(obs_arr))
    order_label, catzoc, tvu_at_median, orders_table = (
        iho_order_from_p95(p95_error_m, med_obs_depth) if n_pairs >= 10
        else (None, None, None, []))

    # Gridded SDB cannot claim Special/1a full-seafloor object detection.
    detection_capability_note = None
    special_pct = order_results.get("special", {}).get("pass_pct", 0)
    order1a_pct = order_results.get("order1a", {}).get("pass_pct", 0)
    if special_pct >= 95 or order1a_pct >= 95:
        res_txt = f"{resolution_m:g} m" if resolution_m is not None else "the selected"
        detection_capability_note = (
            "TVU pass-rate only; Special Order / Order 1a additionally require "
            "full-seafloor object-detection capability (cubic features >=1 m / "
            f">=2 m, S-44 Table 1) that a {res_txt} gridded SDB product cannot "
            "demonstrate — treat any 'Special'/'1a' badge above as "
            "vertical-accuracy-only, not full IHO Order compliance.")

    datum_mismatch = bool(observed_vertical_datum and
                          observed_vertical_datum != predicted_vertical_datum)
    datum_mismatch_note = None
    bias_note = None
    if datum_mismatch:
        datum_mismatch_note = (
            f"Observed survey datum '{observed_vertical_datum}' differs from the model "
            f"prediction datum '{predicted_vertical_datum}'. Regional MSL-LAT offsets "
            f"(e.g. Gulf 0.3-1.2 m) are NOT subtracted here.")
        bias_note = (
            f"bias includes an un-reconciled {observed_vertical_datum}->"
            f"{predicted_vertical_datum} datum offset; treat the constant component "
            f"as datum, not model error.")

    L.info("validate: %d pairs RMSE=%.2f bias=%.2f R2=%.3f S44-1a=%.0f%% catzoc=%s",
           n_pairs, rmse, bias, r2, order_results["order1a"]["pass_pct"], catzoc)

    return {
        "n_pairs": n_pairs,
        "n_observed": len(observed),
        "n_predicted": len(pred_pts),
        "n_rejected_nonfinite": n_rejected_nonfinite,
        "rmse": round(rmse, 3), "mae": round(mae, 3), "medae": round(medae, 3),
        "bias": round(bias, 3), "std": round(std, 3),
        "r2": round(r2, 4), "pearson_r": round(pearson, 4),
        "mape_pct": round(mape, 2),
        "s44_pass_pct": order_results["order1a"]["pass_pct"],
        "iho_orders": order_results,
        "catzoc": catzoc,
        "order_label": order_label,
        "p95_error_m": round(p95_error_m, 3),
        "pass_criterion": "p95 <= TVU (IHO S-44 §3.3.1)",
        "tpu_caveat": _TPU_CAVEAT,
        "iho_orders_table": orders_table,
        "detection_capability_note": detection_capability_note,
        "resolution_m": resolution_m,
        "match_distance_stats_m": match_distance_stats_m,
        "n_excluded_by_distance": n_excluded_by_distance,
        "max_match_m": max_match,
        "observed_vertical_datum": observed_vertical_datum,
        "predicted_vertical_datum": predicted_vertical_datum,
        "datum_mismatch": datum_mismatch,
        "datum_mismatch_note": datum_mismatch_note,
        "bias_note": bias_note,
        "stratified": strat,
        "residual_histogram": hist_data,
        "spatial_residuals": spatial_grid,
        "class_confusion": {
            "labels": CLASS_LABELS,
            "matrix": cm.tolist(),
            "agreement_pct": class_agreement,
        },
        "pairs": pairs_details,
        "pair_stats": pair_stats,
        "method": f"KDTree NN within {max_match:.0f} m (CATZOC-B-consistent)",
    }
