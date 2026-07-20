#!/usr/bin/env python3
"""R1 tide/wave harness — TIDE_WAVE_LOG.md R1.2 (σ guarantees) + R1.3 (scoped
re-test of MLE_TIDE_CORRECT in the REFERENCE-SPARSE / calibration-off regime).

This is NOT a silent reopen of the α≈0.04 kill (which was measured in the
CALIBRATED per-scene regime, where Stage-B isotonic pre-absorbs the tide).
New hypothesis: on the UN-calibrated per-scene depths (``scenes_pre_calib`` in
the run's npz) nothing absorbs the tide, so per-scene tide referencing should
reduce cross-scene std / bias — or, the pre-declared honest negative: the
upstream per-band median + region blend already flatten the datum → α stays ~0.

Reports (all vs KP multibeam, ≥500 m spatial-block pooling):
  A) R1.2 guarantees — byte-identical OFF depth (σ_tide floor 0→0.15) + σ grows.
  B) R1.3 — per-scene tide offsets + tide-table sanity; cross-scene per-pixel std
     BEFORE vs AFTER tide referencing; chart-only RMSE/bias/decile-slope BEFORE
     vs AFTER, depth-stratified, spatial-block aggregated.

Usage:  .venv/bin/python scripts/eval_tide_wave_mle.py
Env:    TIDE_HARNESS_YEAR (2024), TIDE_HARNESS_NSCENES (5)
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Khalifa Port bbox (_KNOWN_SITES / app.py ≈ L8634).
KHALIFA_BBOX = [54.636, 24.785, 54.690, 24.840]
YEAR = int(os.environ.get("TIDE_HARNESS_YEAR", "2024"))
N_SCENES = int(os.environ.get("TIDE_HARNESS_NSCENES", "5"))
BLOCK_DEG = 0.0055  # ~550 m at 24.8°N → ≥500 m spatial-block guard

DL_DIR = ROOT / "backend" / "ocean" / "downloads"


def _banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def run_mle(tide_correct, tide_floor):
    os.environ["MLE_TIDE_CORRECT"] = "1" if tide_correct else "0"
    os.environ["MLE_TIDE_SIGMA_FLOOR_M"] = str(tide_floor)
    import backend.app as A
    res = A._run_s2_mle(KHALIFA_BBOX, year=YEAR, n_scenes=N_SCENES, max_cloud=20)
    return res


def run_mle_wave(bbox, wave_qc, hs0=None, hs_reject=None, k=None):
    """R2.3: run _run_s2_mle with the wave-QC knobs set (tide OFF, floor 0.15)."""
    os.environ["MLE_TIDE_CORRECT"] = "0"
    os.environ["MLE_TIDE_SIGMA_FLOOR_M"] = "0.15"
    os.environ["MLE_WAVE_QC"] = "1" if wave_qc else "0"
    if hs0 is not None:
        os.environ["MLE_WAVE_HS0"] = str(hs0)
    else:
        os.environ.pop("MLE_WAVE_HS0", None)
    if hs_reject is not None:
        os.environ["MLE_HS_REJECT"] = str(hs_reject)
    else:
        os.environ.pop("MLE_HS_REJECT", None)
    if k is not None:
        os.environ["MLE_WAVE_K"] = str(k)
    else:
        os.environ.pop("MLE_WAVE_K", None)
    import backend.app as A
    return A._run_s2_mle(list(bbox), year=YEAR, n_scenes=N_SCENES, max_cloud=20)


def wave_sanity(lat_c, lon_c, per_scene):
    """R2.3 item 4: for the named scenes, pull the Open-Meteo wave day series and
    confirm the reported Hs sits inside the day's envelope with a physical
    magnitude (calm Gulf ~0.3-0.8 m; swell coast higher)."""
    from backend.tide_correction import get_wave_height
    from datetime import datetime
    out = []
    for ps in per_scene:
        acqs = ps.get("acquisition_datetimes") or []
        if not acqs:
            continue
        a0 = acqs[0]
        try:
            d = datetime.fromisoformat(str(a0).replace("Z", "+00:00"))
        except Exception:
            continue
        h, inf = get_wave_height(lat_c, lon_c, d)
        out.append({
            "scene": ps["date"], "acq_utc": a0,
            "reported_hs_mean_m": ps.get("hs_mean_m"),
            "reported_hs_max_m": ps.get("hs_max_m"),
            "openmeteo_pointcheck_hs_m": round(float(h), 3),
            "openmeteo_day_max_hs_m": (inf or {}).get("day_max_wave_m"),
            "source": (inf or {}).get("method"),
        })
        if len(out) >= 2:
            break
    return out


def wave_weights(stab):
    """Extract per-scene Hs / σ_wave / relative fusion weight from stability.wave."""
    w = (stab or {}).get("wave") or {}
    return {
        "wave_qc_applied": w.get("wave_qc_applied"),
        "source": w.get("source"), "k": w.get("k"), "hs0_m": w.get("hs0_m"),
        "hs_rejected_dates": w.get("hs_rejected_dates"),
        "per_scene": [
            {"date": p["date"], "hs_mean_m": p.get("hs_mean_m"),
             "hs_max_m": p.get("hs_max_m"), "sigma_wave_m": p.get("sigma_wave_m"),
             "rel_weight": p.get("rel_weight")}
            for p in (w.get("per_scene") or [])],
    }


def part_c_site(site_name, bbox, has_gt):
    """R2.3 for one site: MLE_WAVE_QC 0 vs 1, byte-identical-OFF no-op proof,
    σ-grows, induced weights, cross-scene std, and chart RMSE (if GT)."""
    lat_c = 0.5 * (bbox[1] + bbox[3]); lon_c = 0.5 * (bbox[0] + bbox[2])
    rep = {"site": site_name, "bbox": list(bbox), "has_gt": has_gt}
    try:
        res_off = run_mle_wave(bbox, wave_qc=False)
        res_on = run_mle_wave(bbox, wave_qc=True)
        # machinery no-op proof: WAVE_QC=1 but Hs0 huge → every σ_wave=0 → must
        # equal the OFF composite byte-for-byte (proves the flag path is additive).
        res_noop = run_mle_wave(bbox, wave_qc=True, hs0=100.0)
    except Exception as ex:
        import traceback
        rep["error"] = f"{ex}"
        rep["traceback"] = traceback.format_exc()[-1500:]
        return rep

    sid_off, sid_on, sid_noop = (res_off["stability_id"], res_on["stability_id"],
                                 res_noop["stability_id"])
    g_off = read_tif(sid_off, "composite")
    g_on = read_tif(sid_on, "composite")
    g_noop = read_tif(sid_noop, "composite")

    def _cmp(a, b):
        both = np.isfinite(a) & np.isfinite(b)
        o0 = int(np.sum(np.isfinite(a) & ~np.isfinite(b)))
        o1 = int(np.sum(np.isfinite(b) & ~np.isfinite(a)))
        mx = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        return mx, o0, o1

    mx_noop, o0_noop, o1_noop = _cmp(g_off, g_noop)
    mx_on, o0_on, o1_on = _cmp(g_off, g_on)
    uq_off = (res_off.get("ml_stats") or {}).get("uncertainty") or {}
    uq_on = (res_on.get("ml_stats") or {}).get("uncertainty") or {}
    sig_off = uq_off.get("median_sigma_post_tidefloor_m")
    sig_on = uq_on.get("median_sigma_post_tidefloor_m")

    rep["byte_identical_off"] = {
        "max_abs_depth_diff_m": mx_noop, "finite_only": [o0_noop, o1_noop],
        "identical": bool(mx_noop == 0.0 and o0_noop == 0 and o1_noop == 0),
        "note": "WAVE_QC=1 with Hs0=100 (zero excess) == WAVE_QC=0 composite",
    }
    rep["wave_on_effect"] = {
        "max_abs_depth_diff_vs_off_m": mx_on, "finite_only": [o0_on, o1_on],
        "median_sigma_off_m": sig_off, "median_sigma_on_m": sig_on,
        "sigma_grows": bool((sig_on or 0) >= (sig_off or 0)),
    }
    rep["weights_off"] = wave_weights(res_off.get("stability"))
    rep["weights_on"] = wave_weights(res_on.get("stability"))
    try:
        rep["hs_sanity"] = wave_sanity(
            lat_c, lon_c, (res_on.get("stability") or {}).get("per_scene", []))
    except Exception as ex:
        rep["hs_sanity"] = {"error": str(ex)}

    # cross-scene std from the retained per-scene stack (unchanged by re-weighting
    # unless the HS_REJECT cap drops a scene).
    try:
        npz_on = load_npz(sid_on)
        std_on, m_on = cross_scene_std(npz_on["scenes"])
        v = std_on[m_on & np.isfinite(std_on)]
        rep["cross_scene_std_m"] = {
            "median": round(float(np.median(v)), 4) if v.size else None,
            "p95": round(float(np.percentile(v, 95)), 4) if v.size else None,
            "note": ("std of retained per-scene stack; wave-QC re-weights the "
                     "fusion, it does not alter per-scene depths, so std changes "
                     "only if HS_REJECT drops a scene"),
        }
    except Exception as ex:
        rep["cross_scene_std_m"] = {"error": str(ex)}

    if has_gt:
        try:
            lon, lat, obs = load_kp()
            ev_off = eval_chart(g_off, list(bbox), lon, lat, obs, "WAVE_QC=0")
            ev_on = eval_chart(g_on, list(bbox), lon, lat, obs, "WAVE_QC=1")
            rep["chart_eval"] = {"off": ev_off, "on": ev_on}
            rmse_ok = (ev_off.get("rmse_m") is not None and ev_on.get("rmse_m") is not None
                       and ev_on["rmse_m"] <= ev_off["rmse_m"] + 1e-6)
            rep["verdict"] = {
                "chart_rmse_not_worse": bool(rmse_ok),
                "rmse_delta_m": (round(ev_on["rmse_m"] - ev_off["rmse_m"], 3)
                                 if ev_off.get("rmse_m") and ev_on.get("rmse_m") else None),
            }
        except Exception as ex:
            rep["chart_eval"] = {"error": str(ex)}
    return rep


def read_tif(stab_id, kind="composite"):
    import rasterio
    p = DL_DIR / f"{stab_id}_{kind}.tif"
    with rasterio.open(p) as ds:
        a = ds.read(1).astype(np.float64)
        nod = ds.nodata
    if nod is not None:
        a[a == nod] = np.nan
    return a


def load_npz(stab_id):
    return np.load(DL_DIR / f"{stab_id}_scenes.npz", allow_pickle=True)


def load_kp():
    from pyproj import Transformer, CRS
    tr = Transformer.from_crs(CRS.from_epsg(32640), CRS.from_epsg(4326), always_xy=True)
    lons, lats, deps = [], [], []
    for f in ["KP Basin Soundings 10m.xyz", "KP_EMAL_Soundings_10x_New.xyz"]:
        d = np.loadtxt(ROOT / "validation" / f, usecols=(0, 1, 2))
        lo, la = tr.transform(d[:, 0], d[:, 1])
        z = np.abs(d[:, 2])
        lons.append(lo); lats.append(la); deps.append(z)
    return np.concatenate(lons), np.concatenate(lats), np.concatenate(deps)


def fuse(stack, sigmas):
    """Inverse-variance fusion over the scene axis (bare σ, no tide σ)."""
    num = np.zeros(stack.shape[1:], np.float64)
    den = np.zeros(stack.shape[1:], np.float64)
    for i in range(stack.shape[0]):
        di = stack[i]
        ok = np.isfinite(di) & (di > 0)
        iv = 1.0 / (float(sigmas[i]) ** 2)
        num[ok] += di[ok] * iv
        den[ok] += iv
    fin = den > 0
    out = np.full(stack.shape[1:], np.nan, np.float32)
    out[fin] = (num[fin] / den[fin]).astype(np.float32)
    return out


def cross_scene_std(stack):
    with np.errstate(all="ignore"):
        cnt = np.sum(np.isfinite(stack), axis=0)
        multi = cnt >= 2
        std = np.full(stack.shape[1:], np.nan, np.float32)
        if multi.any():
            std[multi] = np.nanstd(stack[:, multi], axis=0).astype(np.float32)
    return std, multi


def sample_grid(grid, bbox, lon, lat):
    w, s, e, n = bbox
    H, W = grid.shape
    col = np.floor((lon - w) / (e - w) * W).astype(int)
    row = np.floor((n - lat) / (n - s) * H).astype(int)  # from_bounds → row0=north
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    out = np.full(lon.shape, np.nan)
    out[inb] = grid[row[inb], col[inb]]
    return out


def decile_slope(pred, obs):
    """Regression-to-mean detector: slope of mean(obs) vs mean(pred) across the
    predicted-depth deciles (1.0 = unbiased, <1 = regression to the mean)."""
    order = np.argsort(pred)
    p = pred[order]; o = obs[order]
    n = len(p)
    if n < 20:
        return None
    edges = np.linspace(0, n, 11).astype(int)
    mp, mo = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            mp.append(p[a:b].mean()); mo.append(o[a:b].mean())
    mp = np.asarray(mp); mo = np.asarray(mo)
    if len(mp) < 3 or np.ptp(mp) < 1e-6:
        return None
    return float(np.polyfit(mp, mo, 1)[0])


def eval_chart(grid, bbox, lon, lat, obs, tag):
    pred = sample_grid(grid, bbox, lon, lat)
    m = np.isfinite(pred) & np.isfinite(obs) & (obs > 0)
    pred, obs2, lon2, lat2 = pred[m], obs[m], lon[m], lat[m]
    if len(pred) < 20:
        return {"tag": tag, "n": int(len(pred)), "error": "too few matched soundings"}
    resid = pred - obs2
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    mae = float(np.mean(np.abs(resid)))
    # ≥500 m spatial-block aggregation (block-mean residual → pseudo-replication-free)
    bx = np.floor(lon2 / BLOCK_DEG).astype(int)
    by = np.floor(lat2 / BLOCK_DEG).astype(int)
    keys = bx * 100000 + by
    blk_res, blk_n = [], 0
    for k in np.unique(keys):
        sel = keys == k
        if sel.sum() >= 3:
            blk_res.append(float(np.mean(resid[sel]))); blk_n += 1
    blk_rmse = float(np.sqrt(np.mean(np.asarray(blk_res) ** 2))) if blk_res else None
    # depth-stratified RMSE
    strat = {}
    for name, lo, hi in [("0-2", 0, 2), ("2-5", 2, 5), ("5-10", 5, 10), ("10+", 10, 99)]:
        s = (obs2 >= lo) & (obs2 < hi)
        strat[name] = (round(float(np.sqrt(np.mean(resid[s] ** 2))), 3),
                       int(s.sum())) if s.sum() >= 10 else (None, int(s.sum()))
    return {
        "tag": tag, "n": int(len(pred)),
        "rmse_m": round(rmse, 3), "bias_m": round(bias, 3), "mae_m": round(mae, 3),
        "block_rmse_m": (round(blk_rmse, 3) if blk_rmse is not None else None),
        "n_blocks": blk_n, "block_deg": BLOCK_DEG,
        "decile_slope": (round(decile_slope(pred, obs2), 3)
                         if decile_slope(pred, obs2) is not None else None),
        "rmse_by_depth_m": strat,
    }


def tide_table_sanity(lat_c, lon_c, per_scene):
    """R1.1 item-4b: for ONE named scene, pull the Open-Meteo day series and
    confirm the reported per-scene tide sits inside the day's range with the
    right sign/magnitude."""
    from backend.tide_correction import _fetch_day, get_tide_height
    from datetime import datetime
    out = []
    for ps in per_scene:
        acqs = ps.get("acq") or []
        if not acqs:
            continue
        a0 = acqs[0]
        try:
            d = datetime.fromisoformat(str(a0).replace("Z", "+00:00"))
        except Exception:
            continue
        day = _fetch_day(round(lat_c, 3), round(lon_c, 3), d.strftime("%Y-%m-%d"))
        hs = [r["height"] for r in day if r.get("height") is not None]
        h, _ = get_tide_height(lat_c, lon_c, d)
        out.append({
            "scene": ps["date"], "acq_utc": a0,
            "reported_tide_m": ps["tide_m"],
            "openmeteo_pointcheck_m": round(float(h), 3),
            "day_min_m": round(float(np.min(hs)), 3) if hs else None,
            "day_max_m": round(float(np.max(hs)), 3) if hs else None,
            "day_range_m": round(float(np.ptp(hs)), 3) if hs else None,
        })
        if len(out) >= 2:
            break
    return out


def main():
    report = {"bbox": KHALIFA_BBOX, "year": YEAR, "n_scenes": N_SCENES}

    # ── PART A: R1.2 guarantees (byte-identical OFF depth + σ grows) ──
    _banner("PART A — R1.2: byte-identical OFF depth (σ_tide floor 0.0 → 0.15) + σ grows")
    res0 = run_mle(tide_correct=False, tide_floor=0.0)     # baseline: no σ_tide
    res1 = run_mle(tide_correct=False, tide_floor=0.15)    # floor folded (OFF)
    sid0, sid1 = res0["stability_id"], res1["stability_id"]
    g0 = read_tif(sid0, "composite")
    g1 = read_tif(sid1, "composite")
    both = np.isfinite(g0) & np.isfinite(g1)
    only0 = int(np.sum(np.isfinite(g0) & ~np.isfinite(g1)))
    only1 = int(np.sum(np.isfinite(g1) & ~np.isfinite(g0)))
    max_abs = float(np.max(np.abs(g0[both] - g1[both]))) if both.any() else 0.0
    identical = bool(max_abs == 0.0 and only0 == 0 and only1 == 0)
    uq0 = (res0.get("ml_stats") or {}).get("uncertainty") or {}
    uq1 = (res1.get("ml_stats") or {}).get("uncertainty") or {}
    report["R1_2_guarantees"] = {
        "byte_identical_off_depth": identical,
        "max_abs_depth_diff_m": max_abs,
        "finite_only_floor0": only0, "finite_only_floor015": only1,
        "median_sigma_floor0_m": uq0.get("median_sigma_post_tidefloor_m"),
        "median_sigma_floor015_m": uq1.get("median_sigma_post_tidefloor_m"),
        "sigma_grows": bool(
            (uq1.get("median_sigma_post_tidefloor_m") or 0)
            >= (uq0.get("median_sigma_post_tidefloor_m") or 0)),
        "pre_vs_post_floor015_m": [uq1.get("median_sigma_pre_tidefloor_m"),
                                   uq1.get("median_sigma_post_tidefloor_m")],
        "sigma_tide_m_reported": ((res1.get("stability") or {}).get("tide") or {}).get("sigma_tide_m"),
    }
    print(json.dumps(report["R1_2_guarantees"], indent=2), flush=True)

    # ── PART B: R1.3 tide re-test in the reference-sparse (pre_calib) regime ──
    _banner("PART B — R1.3: tide referencing on UN-calibrated (pre_calib) scenes")
    npz = load_npz(sid1)   # reuse the floor-0.15 OFF run (tide_m computed live)
    pre = npz["scenes_pre_calib"]      # (n, H, W) un-calibrated per-scene depths
    sig = npz["sigmas"]
    tide_m = npz["tide_m"]
    tsrc = [str(x) for x in npz["tide_source"]]
    dates = [str(x) for x in npz["dates"]]
    bbox = list(npz["bbox"])
    lat_c = 0.5 * (bbox[1] + bbox[3]); lon_c = 0.5 * (bbox[0] + bbox[2])

    finite_frac = [float(np.mean(np.isfinite(pre[i]))) for i in range(pre.shape[0])]
    have_pre = sum(1 for f in finite_frac if f > 0)
    tide_ok = np.isfinite(tide_m)
    h_ref = float(np.nanmedian(tide_m[tide_ok])) if tide_ok.any() else None
    shifts = (tide_m - h_ref) if h_ref is not None else np.zeros_like(tide_m)

    per_scene = (res1.get("stability") or {}).get("tide", {}).get("per_scene", [])
    # attach acq datetimes from the scene_table for the sanity lookup
    stab_scenes = {r["date"]: r for r in (res1.get("stability") or {}).get("per_scene", [])}
    for ps in per_scene:
        ps["acq"] = (stab_scenes.get(ps["date"], {}) or {}).get("acquisition_datetimes")

    report["R1_3_tide"] = {
        "tide_source": (res1.get("stability") or {}).get("tide", {}).get("source"),
        "h_ref_m": (round(h_ref, 4) if h_ref is not None else None),
        "n_scenes_with_pre_calib": have_pre,
        "per_scene": [
            {"date": dates[i], "tide_m": (round(float(tide_m[i]), 4) if tide_ok[i] else None),
             "shift_m": (round(float(shifts[i]), 4) if tide_ok[i] else None),
             "source": tsrc[i], "pre_calib_finite_frac": round(finite_frac[i], 4)}
            for i in range(pre.shape[0])],
    }

    # tide-table sanity (item 4b)
    try:
        report["R1_3_tide"]["tide_table_sanity"] = tide_table_sanity(lat_c, lon_c, per_scene)
    except Exception as ex:
        report["R1_3_tide"]["tide_table_sanity"] = {"error": str(ex)}

    if have_pre >= 2 and h_ref is not None:
        shifted = pre.copy()
        for i in range(pre.shape[0]):
            if tide_ok[i]:
                m = np.isfinite(shifted[i])
                shifted[i][m] = shifted[i][m] - float(shifts[i])
        std_before, mb = cross_scene_std(pre)
        std_after, ma = cross_scene_std(shifted)
        comp_before = fuse(pre, sig)
        comp_after = fuse(shifted, sig)

        lon, lat, obs = load_kp()
        ev_b = eval_chart(comp_before, bbox, lon, lat, obs, "BEFORE (no tide ref)")
        ev_a = eval_chart(comp_after, bbox, lon, lat, obs, "AFTER (tide ref)")

        def _med_p95(a, m):
            v = a[m & np.isfinite(a)]
            return (round(float(np.median(v)), 4), round(float(np.percentile(v, 95)), 4)) \
                if v.size else (None, None)
        cb = _med_p95(std_before, mb); ca = _med_p95(std_after, ma)
        report["R1_3_tide"]["cross_scene_std_m"] = {
            "before_median_p95": cb, "after_median_p95": ca,
            "median_delta": (round(ca[0] - cb[0], 4)
                             if cb[0] is not None and ca[0] is not None else None),
        }
        report["R1_3_tide"]["chart_eval"] = {"before": ev_b, "after": ev_a}
        # ADOPT bar
        std_drop = (cb[0] is not None and ca[0] is not None and ca[0] < cb[0])
        rmse_ok = (ev_b.get("rmse_m") is not None and ev_a.get("rmse_m") is not None
                   and ev_a["rmse_m"] <= ev_b["rmse_m"] + 1e-6)
        report["R1_3_tide"]["verdict"] = {
            "cross_scene_std_drops": bool(std_drop),
            "chart_rmse_not_worse": bool(rmse_ok),
            "recommend": ("ADOPT" if (std_drop and rmse_ok
                          and ev_a.get("rmse_m", 9) <= ev_b.get("rmse_m", 9) - 0.10)
                          else "REVERT"),
        }
    else:
        report["R1_3_tide"]["note"] = ("insufficient pre_calib scenes or tide → "
                                       "cannot run before/after (see per_scene)")

    print(json.dumps(report["R1_3_tide"], indent=2, default=str), flush=True)

    # ── PART C: R2.3 — Hs sea-state QC (MLE_WAVE_QC 0 vs 1) ──
    _banner("PART C — R2.3: wave-QC (MLE_WAVE_QC 0 vs 1) at Khalifa + swell site")
    report["R2_3_wave"] = {}
    # Primary: Khalifa (calm Gulf, chart GT). Never-worse gate; expect ~null.
    report["R2_3_wave"]["khalifa"] = part_c_site("khalifa_port", KHALIFA_BBOX, has_gt=True)
    print(json.dumps(report["R2_3_wave"]["khalifa"], indent=2, default=str), flush=True)
    # Swell site: Fujairah / Gulf-of-Oman east coast (Hs discrimination; no chart
    # GT — memory records passive-optical SDB as no-skill there, so the meaningful
    # signal is physical Hs + rough-scene down-weighting, NOT chart RMSE).
    if os.environ.get("WAVE_SKIP_SWELL", "0") != "1":
        swell_bbox = [float(x) for x in os.environ.get(
            "WAVE_SWELL_BBOX", "56.345,25.10,56.375,25.13").split(",")]
        report["R2_3_wave"]["swell"] = part_c_site("fujairah_eastcoast", swell_bbox,
                                                   has_gt=False)
        print(json.dumps(report["R2_3_wave"]["swell"], indent=2, default=str), flush=True)

    out = ROOT / "scripts" / "tide_wave_r1_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    r2out = ROOT / "scripts" / "tide_wave_r2_report.json"
    r2out.write_text(json.dumps(report.get("R2_3_wave", {}), indent=2, default=str))
    _banner(f"WROTE {out} + {r2out}")
    return report


if __name__ == "__main__":
    main()
