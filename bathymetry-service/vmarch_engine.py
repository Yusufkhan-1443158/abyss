"""VMarch SDB — full-fidelity port of the Railway production method.

Vendored from Bathymetry_Production/backend/app.py::_run_s2_lyzenga_fast
(default env-knob path), stages 2-4c:

  mask morphology → reference fusion (user pts w=6.0, GEBCO per-tile
  fall-through w=1.0) → Lyzenga(1985/2006) WLS + Stumpf log-ratio
  inverse-RMSE ensemble (lyzenga_stumpf.estimate_depth, verbatim) →
  UAE cluster ensemble blend with REGION-AWARE weights (w_uae=0.10 inside
  a calibrated region, 0.50 outside; −0.10 with ≥5 user points) →
  3-stage bias correction (Stage 1 global isotonic pred→truth with WLS
  linear fallback; Stage 2 per-pixel kNN-IDW residual field, k=6, 1/d²,
  0.05° cap, global-median far-field; Stage 3 residual clamp ±3 m and
  final 0-25 m clip) → per-pixel σ (TPU) grid.

The bundled in-situ surveys the Railway deploy ships (KP Basin / KP_EMAL /
OMC XYZ + SWOT_Dhanna shapefile) are vendored under data/insitu and ingested
exactly as upstream: 600-point budget, bbox-seeded deterministic 80/20
train/holdout split, weight 5.5; the held-out 20% + user points drive the
honest RMSE/MAE/bias/R²/S-44 metrics block. GEBCO is skipped whenever ≥30
in-situ anchors cover the ROI (upstream rule).

Deviations from the original, all documented and justified:
  * SlideRule ICESat-2 / ATL24 / VHR-Mapbox providers need external creds
    absent from Abyss — those arms are skipped (upstream skips them too
    when unavailable);
  * env-gated experimental knobs that default OFF upstream (PIECEWISE_CALIB,
    STRATIFY_SHALLOW, DEPTH_DENSITY_W, PER_BAND_RESID, HEDLEY_DEGLINT,
    S2_DSF) are not ported;
  * the in-region CNN fine-tune is torch-only and torch is not shipped in
    the Abyss service image — skipped (upstream skips without torch);
  * the σ (TPU) grid — upstream EMIT_SIGMA=1 — is always computed here
    (it feeds the multi-scene inverse-variance composite);
  * GEBCO GeoTIFFs are decoded with rasterio, the SWOT shapefile with
    fiona (instead of tifffile / geopandas).
"""
from __future__ import annotations

import io
import logging
import math
from typing import Dict, List, Optional

import numpy as np

L = logging.getLogger("bathymetry.vmarch")

MAX_DEPTH_M = 25.0

MODEL_NAME = "vmarch-sdb"
MODEL_VERSION = "registry-v1"
MODEL_LABEL = ("VMarch SDB (port) — Lyzenga+Stumpf+UAE ensemble, "
               "bias-corrected · registry-v1")
METHOD = ("VMarch SDB — Lyzenga(1985/2006) WLS + Stumpf log-ratio "
          "(inverse-RMSE ensemble) + UAE cluster ensemble region-aware blend "
          "+ 3-stage bias correction (isotonic → residual IDW → ±3 m clamp)")

# Calibration envelopes the UAE pretrained model was trained on
# (app.py::_PRESET_BBOXES, verbatim).
PRESET_BBOXES = [
    ("khalifa_port", [54.636, 24.785, 54.690, 24.840]),
    ("old_mussafah", [54.355, 24.412, 54.412, 24.466]),
    ("abu_al_abyad", [53.852, 24.213, 53.910, 24.267]),
    ("jbel_dhanna",  [52.5692, 24.1989, 52.6186, 24.2440]),
]


def bbox_inside_predefined(bbox, pad_deg: float = 0.005) -> bool:
    w, s, e, n = bbox
    cx = (w + e) / 2.0
    cy = (s + n) / 2.0
    for _key, pb in PRESET_BBOXES:
        if (pb[0] - pad_deg) <= cx <= (pb[2] + pad_deg) and \
           (pb[1] - pad_deg) <= cy <= (pb[3] + pad_deg):
            return True
    return False


def coverage_water_mask(bbox, H, W, s2=None):
    """app.py::make_water_mask verbatim: ~1 km global land DB + satellite
    brightness/blue-fraction land detection with NDWI rescue and NIR test."""
    w, s_b, e, n = bbox
    water = np.ones((H, W), dtype=bool)
    try:
        from global_land_mask import globe
        lats = np.linspace(n, s_b, H)
        lons = np.linspace(w, e, W)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        is_land = globe.is_land(lat_grid, lon_grid)
        water = water & (~is_land)
    except Exception as ex:
        L.warning("global_land_mask failed: %s", ex)
    if s2 is not None:
        try:
            rf = s2["red"].astype(np.float32)
            gf = s2["green"].astype(np.float32)
            bf = s2["blue"].astype(np.float32)
            nirf = s2.get("nir")
            if nirf is not None:
                nirf = nirf.astype(np.float32)
            if rf.shape != (H, W):
                from scipy.ndimage import zoom as ndizoom
                rf = ndizoom(rf, (H / rf.shape[0], W / rf.shape[1]), order=1)
                gf = ndizoom(gf, (H / gf.shape[0], W / gf.shape[1]), order=1)
                bf = ndizoom(bf, (H / bf.shape[0], W / bf.shape[1]), order=1)
                if nirf is not None:
                    nirf = ndizoom(nirf, (H / nirf.shape[0], W / nirf.shape[1]),
                                   order=1)
            brightness = rf + gf + bf
            max_b = max(brightness.max(), 1)
            bright_norm = brightness / max_b
            blue_frac = bf / (brightness + 1e-6)
            red_frac = rf / (brightness + 1e-6)
            sat_land = ((bright_norm > 0.55) & (blue_frac < 0.30)) | \
                       ((bright_norm > 0.65) & (red_frac > 0.38) & (blue_frac < 0.32))
            ndwi_local = s2.get("ndwi")
            if ndwi_local is not None:
                if ndwi_local.shape != (H, W):
                    from scipy.ndimage import zoom as ndizoom
                    ndwi_local = ndizoom(ndwi_local,
                                         (H / ndwi_local.shape[0],
                                          W / ndwi_local.shape[1]), order=1)
                sat_land = sat_land & ~(ndwi_local > 0.15)
            if nirf is not None:
                nir_norm = nirf / max(np.percentile(nirf, 99), 1.0)
                sat_land = sat_land | (nir_norm > 0.45)
            from scipy.ndimage import (binary_erosion, binary_dilation,
                                       binary_fill_holes)
            struct = np.ones((3, 3))
            sat_land = binary_erosion(sat_land, struct, iterations=1)
            sat_land = binary_dilation(sat_land, struct, iterations=1)
            sat_land = binary_fill_holes(sat_land)
            water = water & (~sat_land)
        except Exception as ex:
            L.warning("satellite land detection failed: %s", ex)
    return water


def fetch_gebco(bbox, timeout: float = 30.0) -> Optional[Dict]:
    """GEBCO/NOAA global DEM point sampler (app.py::fetch_gebco, rasterio
    decode). Returns {lats, lons, depths} or None."""
    import requests
    w, s, e, n = bbox
    wp = max(16, min(500, int((e - w) * 240)))
    hp = max(16, min(500, int((n - s) * 240)))
    for url in [
        f"https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage?bbox={w},{s},{e},{n}&bboxSR=4326&imageSR=4326&size={wp},{hp}&format=tiff&f=image",
        f"https://wms.gebco.net/mapserv?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage&COVERAGE=GEBCO_LATEST&CRS=EPSG:4326&FORMAT=GeoTIFF&BBOX={s},{w},{n},{e}&WIDTH={wp}&HEIGHT={hp}",
    ]:
        try:
            r = requests.get(url, timeout=timeout)
            if r.ok and len(r.content) > 200:
                from rasterio.io import MemoryFile
                with MemoryFile(r.content) as mem, mem.open() as src:
                    elev = src.read(1).astype(np.float32)
                grid = np.where(elev < 0, -elev, np.nan)
                gh, gw = grid.shape
                st = max(1, int(math.sqrt(gh * gw / 2000)))
                lats, lons, depths = [], [], []
                for i in range(0, gh, st):
                    lat = n - (i / gh) * (n - s)
                    for j in range(0, gw, st):
                        v = grid[i, j]
                        if np.isfinite(v) and 0.3 < v < MAX_DEPTH_M:
                            lats.append(lat)
                            lons.append(w + (j / gw) * (e - w))
                            depths.append(min(float(v), MAX_DEPTH_M))
                if len(depths) > 5:
                    L.info("GEBCO: %d pts", len(depths))
                    return {"lats": np.array(lats), "lons": np.array(lons),
                            "depths": np.array(depths)}
        except Exception as ex:
            L.warning("GEBCO: %s", ex)
    return None


_INSITU_CACHE = None


def _load_insitu():
    """Bundled in-situ surveys (app.py::_load_xyz_full): three UTM-40N XYZ
    files + the SWOT_Dhanna UTM-39N shapefile, fused to lat/lon/depth."""
    global _INSITU_CACHE
    if _INSITU_CACHE is not None:
        return _INSITU_CACHE
    import os
    from rasterio.warp import transform as rio_transform
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "insitu")
    all_lats, all_lons, all_deps = [], [], []
    for fname in ("KP Basin Soundings 10m.xyz",
                  "KP_EMAL_Soundings_10x_New.xyz",
                  "OMC_Soundings_20m.xyz"):
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            continue
        try:
            data = np.loadtxt(fpath, usecols=(0, 1, 2))
            lons, lats = rio_transform("EPSG:32640", "EPSG:4326",
                                       list(data[:, 0]), list(data[:, 1]))
            lons, lats = np.asarray(lons), np.asarray(lats)
            deps = np.abs(data[:, 2])
            valid = (deps > 0) & (deps < 50) & np.isfinite(lats) & np.isfinite(lons)
            all_lats.append(lats[valid])
            all_lons.append(lons[valid])
            all_deps.append(deps[valid])
            L.info("in-situ: loaded %d pts from %s", int(valid.sum()), fname)
        except Exception as ex:
            L.warning("in-situ: failed to load %s: %s", fname, ex)
    shp = os.path.join(base, "swot_dhanna", "SWOT_Dhanna.shp")
    if os.path.exists(shp):
        try:
            import fiona
            xs, ys, zs = [], [], []
            with fiona.open(shp) as coll:
                for f in coll:
                    g = f["geometry"]
                    if g["type"] != "Point":
                        continue
                    xs.append(float(g["coordinates"][0]))
                    ys.append(float(g["coordinates"][1]))
                    zs.append(abs(float(f["properties"].get("Z", 0) or 0)))
            lons, lats = rio_transform("EPSG:32639", "EPSG:4326", xs, ys)
            lons, lats = np.asarray(lons), np.asarray(lats)
            deps = np.asarray(zs)
            valid = (deps > 0) & (deps < 50) & np.isfinite(lats) & np.isfinite(lons)
            all_lats.append(lats[valid])
            all_lons.append(lons[valid])
            all_deps.append(deps[valid])
            L.info("in-situ: loaded %d pts from SWOT_Dhanna.shp", int(valid.sum()))
        except Exception as ex:
            L.warning("in-situ: failed to load SWOT_Dhanna: %s", ex)
    if all_lats:
        _INSITU_CACHE = (np.concatenate(all_lats), np.concatenate(all_lons),
                         np.concatenate(all_deps))
    else:
        _INSITU_CACHE = (np.array([]), np.array([]), np.array([]))
    return _INSITU_CACHE


def insitu_points_in_bbox(bbox):
    la, lo, de = _load_insitu()
    if len(de) == 0:
        return la, lo, de
    w, s, e, n = bbox
    m = (la >= s) & (la <= n) & (lo >= w) & (lo <= e)
    return la[m], lo[m], de[m]


def has_insitu(bbox, min_pts: int = 30) -> bool:
    """True when the ROI already carries a healthy in-situ pool — the
    upstream rule that skips the GEBCO fall-through (450 m / ±5 m noise)."""
    return len(insitu_points_in_bbox(bbox)[2]) >= min_pts


def fuse_references(bbox, user_pts=None, gebco: Optional[Dict] = None) -> Dict:
    """Reference fusion (app.py stage 3): user points w=6.0 always; bundled
    in-situ w=5.5 with a 600-pt budget and a bbox-seeded deterministic 80/20
    train/holdout split; GEBCO per-tile (4×4, ≤4 pts/tile) fall-through
    w=1.0 into undercovered tiles, skipped when ≥30 in-situ pts cover the ROI."""
    w_, s_, e_, n_ = bbox
    rl, rlo, rd, rw = [], [], [], []
    counts = {"in_situ": 0, "sliderule": 0, "gebco": 0, "user": 0}
    test_lats, test_lons, test_deps = [], [], []

    if user_pts:
        for p in user_pts:
            try:
                d = abs(float(p["depth"]))
            except Exception:
                continue
            if 0 < d <= MAX_DEPTH_M:
                rl.append(float(p["lat"]))
                rlo.append(float(p["lon"]))
                rd.append(d)
                rw.append(6.0)
                counts["user"] += 1

    ila, ilo, ide = insitu_points_in_bbox(bbox)
    if len(ide):
        budget = 600
        if len(ide) > budget:
            sel = np.linspace(0, len(ide) - 1, budget).astype(int)
            ila, ilo, ide = ila[sel], ilo[sel], ide[sel]
        rng = np.random.default_rng(
            int(abs(hash(tuple(round(x, 4) for x in bbox))) % 2 ** 31))
        idx = np.arange(len(ide))
        rng.shuffle(idx)
        n_test = max(1, int(round(0.20 * len(idx))))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        for j in train_idx:
            d = float(abs(ide[j]))
            if 0 < d <= MAX_DEPTH_M:
                rl.append(float(ila[j]))
                rlo.append(float(ilo[j]))
                rd.append(d)
                rw.append(5.5)
                counts["in_situ"] += 1
        for j in test_idx:
            d = float(abs(ide[j]))
            if 0 < d <= MAX_DEPTH_M:
                test_lats.append(float(ila[j]))
                test_lons.append(float(ilo[j]))
                test_deps.append(d)

    if counts["in_situ"] >= 30:
        gebco = None
        L.info("VMarch refs: skipping GEBCO (%d in-situ pts cover the ROI)",
               counts["in_situ"])

    N_PER_TILE = 4
    NX = NY = 4
    if gebco:
        gla = np.asarray(gebco["lats"])
        glo = np.asarray(gebco["lons"])
        gde = np.asarray(gebco["depths"])
        if rl:
            la_arr = np.asarray(rl)
            lo_arr = np.asarray(rlo)
            tile_x = np.clip(((lo_arr - w_) / max(e_ - w_, 1e-9) * NX).astype(int), 0, NX - 1)
            tile_y = np.clip(((n_ - la_arr) / max(n_ - s_, 1e-9) * NY).astype(int), 0, NY - 1)
            tile_counts = {(tx, ty): 0 for ty in range(NY) for tx in range(NX)}
            for tx, ty in zip(tile_x, tile_y):
                tile_counts[(int(tx), int(ty))] += 1
        else:
            tile_counts = {(tx, ty): 0 for ty in range(NY) for tx in range(NX)}
        added = 0
        for ty in range(NY):
            for tx in range(NX):
                if tile_counts[(tx, ty)] >= N_PER_TILE:
                    continue
                tw = w_ + tx * (e_ - w_) / NX
                te = w_ + (tx + 1) * (e_ - w_) / NX
                tn = n_ - ty * (n_ - s_) / NY
                ts = n_ - (ty + 1) * (n_ - s_) / NY
                m = ((glo >= tw) & (glo <= te) & (gla >= ts) & (gla <= tn)
                     & (gde > 0.5) & (gde <= MAX_DEPTH_M))
                if not m.any():
                    continue
                idx = np.where(m)[0]
                if len(idx) > N_PER_TILE:
                    idx = idx[np.linspace(0, len(idx) - 1, N_PER_TILE).astype(int)]
                for j in idx:
                    rl.append(float(gla[j]))
                    rlo.append(float(glo[j]))
                    rd.append(float(gde[j]))
                    rw.append(1.0)
                    added += 1
        counts["gebco"] = added
    L.info("VMarch refs: user=%d in_situ=%d gebco=%d total=%d",
           counts["user"], counts["in_situ"], counts["gebco"], len(rd))
    return {"lats": rl, "lons": rlo, "depths": rd, "weights": rw,
            "counts": counts,
            "test": {"lats": test_lats, "lons": test_lons,
                     "depths": test_deps}}


def _bias_correct(depth, bbox, refs, H, W):
    """3-stage bias correction (app.py stage 4c default path, verbatim
    logic): isotonic → kNN-IDW residual field → ±3 m clamp + 0-25 clip."""
    w_, s_, e_, n_ = bbox
    rl, rlo = refs["lats"], refs["lons"]
    rd, rw = refs["depths"], refs["weights"]
    bias_info = {"applied": False}
    post_residuals = None
    tr_lat = tr_lon = None
    dbg = {}
    if not rd:
        return depth, bias_info, post_residuals, tr_lat, tr_lon, dbg

    train_rows = np.array([
        max(0, min(H - 1, int((n_ - la) / (n_ - s_ + 1e-10) * H)))
        for la in rl], dtype=np.int32)
    train_cols = np.array([
        max(0, min(W - 1, int((lo - w_) / (e_ - w_ + 1e-10) * W)))
        for lo in rlo], dtype=np.int32)
    train_truth = np.asarray(rd, dtype=np.float64)
    train_pred = depth[train_rows, train_cols].astype(np.float64)
    ok = np.isfinite(train_pred) & (train_truth > 0) & (train_truth <= MAX_DEPTH_M)
    if int(ok.sum()) < 10:
        return depth, bias_info, post_residuals, tr_lat, tr_lon, dbg

    tr_pred = train_pred[ok]
    tr_truth = train_truth[ok]
    tr_lat = np.asarray([rl[i] for i, k in enumerate(ok) if k])
    tr_lon = np.asarray([rlo[i] for i, k in enumerate(ok) if k])
    pre_residual = tr_pred - tr_truth
    pre_rmse = float(np.sqrt(np.mean(pre_residual ** 2)))
    pre_bias = float(np.mean(pre_residual))

    # Stage 1: monotonic recalibration (isotonic; WLS linear fallback).
    a_lin, b_lin = 1.0, 0.0
    iso = None
    try:
        from sklearn.isotonic import IsotonicRegression
        w_arr = np.asarray(rw, dtype=np.float64)[ok]
        if len(np.unique(np.round(tr_pred, 2))) >= 8:
            iso = IsotonicRegression(out_of_bounds="clip",
                                     y_min=0.0, y_max=MAX_DEPTH_M)
            iso.fit(tr_pred, tr_truth, sample_weight=w_arr)
        else:
            raise RuntimeError("too few unique preds for isotonic")
    except Exception as _ix:
        try:
            w_arr = np.asarray(rw, dtype=np.float64)[ok]
            W_diag = w_arr / max(w_arr.sum(), 1e-9) * len(w_arr)
            X = np.column_stack([tr_pred, np.ones_like(tr_pred)])
            Xw = X * np.sqrt(W_diag)[:, None]
            yw = tr_truth * np.sqrt(W_diag)
            coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            a_lin, b_lin = float(coef[0]), float(coef[1])
            if not (0.4 <= a_lin <= 1.8 and -8.0 <= b_lin <= 8.0):
                a_lin, b_lin = 1.0, 0.0
        except Exception:
            a_lin, b_lin = 1.0, 0.0
        L.info("VMarch: Stage 1 = linear (a=%.3f, b=%.3f) — isotonic skipped: %s",
               a_lin, b_lin, _ix)

    if iso is not None:
        flat_in = depth.reshape(-1)
        finite = np.isfinite(flat_in)
        flat_out = flat_in.copy()
        if int(finite.sum()) > 0:
            flat_out[finite] = iso.predict(flat_in[finite].astype(np.float64))
        depth = flat_out.reshape(H, W).astype(np.float32)
        tr_pred_after_lin = iso.predict(tr_pred)
    else:
        depth = a_lin * depth + b_lin
        tr_pred_after_lin = a_lin * tr_pred + b_lin
    depth = np.clip(depth, 0.0, MAX_DEPTH_M)
    dbg["post_stage1"] = depth.copy()
    if iso is not None:
        try:
            dbg["iso_x"] = np.asarray(iso.X_thresholds_, np.float64)
            dbg["iso_y"] = np.asarray(iso.y_thresholds_, np.float64)
        except Exception:
            pass

    # Stage 2: per-pixel kNN-IDW residual correction; Stage 3: ±3 m clamp.
    post_lin_residuals = tr_pred_after_lin - tr_truth
    global_med = float(np.median(post_lin_residuals))
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([tr_lat, tr_lon]))
        step = max(1, min(H, W) // 96)
        rs = np.arange(0, H, step)
        cs = np.arange(0, W, step)
        Hc, Wc = len(rs), len(cs)
        lat_1d = n_ - (rs + 0.5) / H * (n_ - s_)
        lon_1d = w_ + (cs + 0.5) / W * (e_ - w_)
        grid_lat, grid_lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
        pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
        k = min(6, len(tr_lat))
        d_deg, idx = tree.query(pts, k=k)
        wts = 1.0 / (d_deg + 1e-3) ** 2
        far = d_deg > 0.05
        wts = np.where(far, 0.0, wts)
        wsum = wts.sum(axis=1, keepdims=True)
        close = wsum.squeeze() > 1e-9
        local_resid = np.full(pts.shape[0], global_med, dtype=np.float64)
        if k > 0:
            contribs = (wts * post_lin_residuals[idx]).sum(axis=1)
            local_resid[close] = contribs[close] / wsum.squeeze()[close]
        coarse = local_resid.reshape(Hc, Wc).astype(np.float32)
        try:
            from scipy.ndimage import zoom as ndizoom
            resid_grid = ndizoom(coarse, (H / coarse.shape[0], W / coarse.shape[1]),
                                 order=1, mode="nearest")[:H, :W]
        except Exception:
            resid_grid = np.full((H, W), global_med, dtype=np.float32)
        resid_grid = np.clip(resid_grid, -3.0, 3.0)
        depth = depth - resid_grid
        depth = np.clip(depth, 0.0, MAX_DEPTH_M)
        tr_pred_final = depth[train_rows[ok], train_cols[ok]].astype(np.float64)
        post_residuals = tr_pred_final - tr_truth
        post_rmse = float(np.sqrt(np.mean(post_residuals ** 2)))
        post_bias = float(np.mean(post_residuals))
        bias_info = {
            "applied": True,
            "stage1": "isotonic" if iso is not None else "linear",
            "pre_rmse_m": round(pre_rmse, 3),
            "pre_bias_m": round(pre_bias, 3),
            "post_rmse_m": round(post_rmse, 3),
            "post_bias_m": round(post_bias, 3),
            "lin_a": round(a_lin, 4),
            "lin_b": round(b_lin, 4),
            "global_resid_med_m": round(global_med, 3),
            "n_train_used": int(ok.sum()),
        }
        L.info("VMarch bias-correction: %s + IDW → RMSE %.2f→%.2f m, "
               "bias %+.2f→%+.2f m, n=%d",
               bias_info["stage1"], pre_rmse, post_rmse, pre_bias, post_bias,
               int(ok.sum()))
    except Exception as ex:
        L.info("VMarch: per-pixel residual correction skipped (%s)", ex)
    return depth, bias_info, post_residuals, tr_lat, tr_lon, dbg


def _sigma_grid(depth, bbox, post_residuals, tr_lat, tr_lon, H, W):
    """Per-pixel σ (TPU) grid (app.py EMIT_SIGMA block, defaults):
    σ² = σ_model² + σ_tide² + σ_refr² + σ_ref² + σ_georef_z²."""
    w_, s_, e_, n_ = bbox
    s_tide, s_refr_frac, s_ref, s_georef_h = 0.10, 0.005, 0.25, 5.0
    sigma_model = np.full((H, W), s_ref, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
        from scipy.ndimage import zoom
        if post_residuals is not None and tr_lat is not None and len(tr_lat) >= 4:
            tree = cKDTree(np.column_stack([tr_lat, tr_lon]))
            step = max(1, min(H, W) // 96)
            rs = np.arange(0, H, step)
            cs = np.arange(0, W, step)
            Hc, Wc = len(rs), len(cs)
            lat_1d = n_ - (rs + 0.5) / H * (n_ - s_)
            lon_1d = w_ + (cs + 0.5) / W * (e_ - w_)
            gla, glo = np.meshgrid(lat_1d, lon_1d, indexing="ij")
            pts = np.column_stack([gla.ravel(), glo.ravel()])
            kk = min(8, len(tr_lat))
            _dd, ii = tree.query(pts, k=kk)
            loc = np.sqrt(np.mean(post_residuals[ii] ** 2, axis=1))
            coarse = loc.reshape(Hc, Wc).astype(np.float32)
            sm = zoom(coarse, (H / Hc, W / Wc), order=1, mode="nearest")
            sigma_model = sm[:H, :W].astype(np.float32)
            sigma_model = np.maximum(sigma_model, s_ref)
    except Exception as se:
        L.info("VMarch: σ_model kNN failed (%s); flat σ_ref", se)
    d = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    sigma_refr = (s_refr_frac * d).astype(np.float32)
    try:
        gz_y, gz_x = np.gradient(d)
        px_m = (e_ - w_) / W * 111320.0 * math.cos(math.radians((n_ + s_) / 2))
        py_m = (n_ - s_) / H * 110570.0
        slope = np.sqrt((gz_x / max(px_m, 1e-6)) ** 2 +
                        (gz_y / max(py_m, 1e-6)) ** 2)
        sigma_georef_z = np.clip((s_georef_h * slope).astype(np.float32), 0.0, 2.0)
    except Exception:
        sigma_georef_z = np.zeros((H, W), dtype=np.float32)
    return np.sqrt(sigma_model ** 2 + np.float32(s_tide) ** 2 + sigma_refr ** 2
                   + np.float32(s_ref) ** 2 + sigma_georef_z ** 2).astype(np.float32)


def run_vmarch(s2: Dict, bbox, user_pts: Optional[List[dict]] = None,
               uae_model=None, gebco: Optional[Dict] = None,
               fetch_refs: bool = True) -> Dict:
    """The full VMarch chain on a fetched S2 scene dict (DN bands +
    water_mask). `uae_model` is the injected UAE cluster-ensemble arm (an
    object with .predict(s2, water_mask=...) and optionally .fine_tune();
    None disables the blend arm). `gebco` may be pre-fetched (multi-scene
    reuse); fetch_refs=False skips the network fall-through (tests)."""
    from lyzenga_stumpf import estimate_depth as ls_estimate_depth

    H, W = np.asarray(s2["red"]).shape
    w_, s_, e_, n_ = bbox

    # Composite water mask (app.py stage 2): spectral ∩ coverage with a
    # collapse fallback, then opening; NDWI fallback when nearly empty.
    spectral_water = s2.get("water_mask")
    coverage_water = coverage_water_mask(bbox, H, W, s2=s2)
    if spectral_water is not None and coverage_water is not None:
        water = np.asarray(spectral_water, bool) & np.asarray(coverage_water, bool)
    elif coverage_water is not None:
        water = np.asarray(coverage_water, bool)
    elif spectral_water is not None:
        water = np.asarray(spectral_water, bool)
    else:
        water = np.asarray(s2.get("ndwi", np.zeros((H, W)))) > 0
    if spectral_water is not None:
        surviving = float(water.sum() / max(np.asarray(spectral_water).sum(), 1))
        if surviving < 0.30:
            L.info("VMarch mask: dual mask collapsed (kept %.1f%%) — spectral only",
                   surviving * 100)
            water = np.asarray(spectral_water, bool)
    try:
        from scipy.ndimage import binary_opening
        water = binary_opening(water, structure=np.ones((3, 3), bool),
                               iterations=1)
    except Exception:
        pass
    if int(water.sum()) < 100:
        water = np.asarray(s2.get("ndwi", np.zeros((H, W)))) > 0
    s2 = dict(s2)
    s2["water_mask"] = water

    if gebco is None and fetch_refs:
        gebco = fetch_gebco(bbox)
    refs = fuse_references(bbox, user_pts=user_pts, gebco=gebco)
    rl, rlo = refs["lats"], refs["lons"]
    rd, rw = refs["depths"], refs["weights"]

    # 4a. Lyzenga + Stumpf inverse-RMSE ensemble.
    res_ls = ls_estimate_depth(s2, bbox, ref_lats=rl, ref_lons=rlo,
                               ref_depths=rd, ref_weights=rw,
                               sliderule_pts=None)
    depth = res_ls["depth"]
    ls_depth = depth.copy()

    # 4b. UAE cluster ensemble blend — region-aware weights, with the
    # in-region darkbox fine-tune when the model supports it.
    uae_used = False
    uae_grid = None
    inside = bbox_inside_predefined(bbox)
    w_uae = 0.0
    if uae_model is not None:
        try:
            if (inside and hasattr(uae_model, "fine_tune")
                    and refs["counts"].get("in_situ", 0) >= 30):
                try:
                    from sdb_engine.features import build_feature_stack
                    feats_full, _ = build_feature_stack(s2)
                    ft_feats, ft_deps, ft_wts = [], [], []
                    for la, lo, dt, wgt in zip(rl, rlo, rd, rw):
                        if wgt < 5.0:  # skip GEBCO; only in-situ / user truth
                            continue
                        r_px = max(0, min(H - 1, int((n_ - la) / (n_ - s_ + 1e-10) * H)))
                        c_px = max(0, min(W - 1, int((lo - w_) / (e_ - w_ + 1e-10) * W)))
                        v = feats_full[r_px, c_px]
                        if np.all(np.isfinite(v)):
                            ft_feats.append(v)
                            ft_deps.append(dt)
                            ft_wts.append(wgt)
                    if len(ft_deps) >= 30:
                        res_ft = uae_model.fine_tune(
                            np.asarray(ft_feats, dtype=np.float32),
                            np.asarray(ft_deps, dtype=np.float32),
                            weights=np.asarray(ft_wts, dtype=np.float32),
                            n_epochs=20, lr=5e-4)
                        L.info("VMarch: in-situ fine-tune updated clusters %s (n=%s)",
                               res_ft.get("updated", []), res_ft.get("n_used"))
                except Exception as ftex:
                    L.info("VMarch: fine-tune skipped (%s)", ftex)
            uae_pred = uae_model.predict(s2, water_mask=water)
            uae_grid = np.asarray(uae_pred["depth"], dtype=np.float32)
            w_uae = 0.10 if inside else 0.50
            if user_pts and len(user_pts) >= 5:
                w_uae = max(0.05, w_uae - 0.10)
            w_ls = 1.0 - w_uae
            both = water & np.isfinite(depth) & np.isfinite(uae_grid)
            only_uae = water & ~np.isfinite(depth) & np.isfinite(uae_grid)
            only_ls = water & np.isfinite(depth) & ~np.isfinite(uae_grid)
            blended = np.full_like(depth, np.nan, dtype=np.float32)
            blended[both] = w_uae * uae_grid[both] + w_ls * depth[both]
            blended[only_uae] = uae_grid[only_uae]
            blended[only_ls] = depth[only_ls]
            depth = np.clip(blended, 0.0, MAX_DEPTH_M)
            uae_used = True
            L.info("VMarch: UAE blend (w_UAE=%.2f, w_LS=%.2f, inside=%s)",
                   w_uae, 1.0 - w_uae, inside)
        except Exception as ex:
            L.info("VMarch: UAE arm unavailable (%s)", ex)

    # 4c. 3-stage bias correction.
    depth_pre_calib = depth.copy()
    depth, bias_info, post_residuals, tr_lat, tr_lon, stage_dbg = _bias_correct(
        depth, bbox, refs, H, W)

    sigma = _sigma_grid(depth, bbox, post_residuals, tr_lat, tr_lon, H, W)
    depth = np.where(water, depth, np.nan)
    sigma = np.where(water, sigma, np.nan).astype(np.float32)

    # Stage 5: honest metrics on the held-out 20% in-situ split + user pts.
    metrics = {"n_test": 0}
    test = refs.get("test") or {}
    test_la = list(test.get("lats") or [])
    test_lo = list(test.get("lons") or [])
    test_dp = list(test.get("depths") or [])
    if user_pts:
        for p in user_pts:
            try:
                la, lo = float(p["lat"]), float(p["lon"])
                dt = abs(float(p["depth"]))
            except Exception:
                continue
            if 0 < dt <= MAX_DEPTH_M:
                test_la.append(la)
                test_lo.append(lo)
                test_dp.append(dt)
    sp, st = [], []
    for la, lo, dt in zip(test_la, test_lo, test_dp):
        r_px = max(0, min(H - 1, int((n_ - la) / (n_ - s_ + 1e-10) * H)))
        c_px = max(0, min(W - 1, int((lo - w_) / (e_ - w_ + 1e-10) * W)))
        v = depth[r_px, c_px]
        if np.isfinite(v):
            sp.append(float(v))
            st.append(dt)
    if len(sp) >= 5:
        sp = np.asarray(sp)
        st = np.asarray(st)
        r = sp - st
        metrics = {
            "n_test": int(len(sp)),
            "rmse_m": round(float(np.sqrt(np.mean(r ** 2))), 3),
            "mae_m": round(float(np.mean(np.abs(r))), 3),
            "bias_m": round(float(np.mean(r)), 3),
            "r2": round(float(1 - np.sum(r ** 2) /
                              max(np.sum((st - st.mean()) ** 2), 1e-9)), 4),
            "s44_1a_pct": round(100.0 * float(np.mean(
                np.abs(r) <= np.sqrt(0.5 ** 2 + (0.013 * st) ** 2))), 1),
            "s44_order2_pct": round(100.0 * float(np.mean(
                np.abs(r) <= np.sqrt(1.0 ** 2 + (0.023 * st) ** 2))), 1),
        }

    return {
        "metrics": metrics,
        "depth": depth.astype(np.float32),
        "sigma": sigma,
        "depth_pre_calib": depth_pre_calib,
        "ls_depth": ls_depth,
        "uae_depth": uae_grid,
        "stage_debug": stage_dbg,
        "water_mask": water,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_label": MODEL_LABEL,
        "method": METHOD,
        "ls_method": res_ls.get("method"),
        "calibrated": True,
        "confidence": ("region-calibrated (UAE domain)" if inside
                       else "transfer-learning blend (outside UAE envelopes)"),
        "inside_predefined": inside,
        "uae_blend": {"used": uae_used, "w_uae": round(float(w_uae), 2),
                      "inside_predefined": inside},
        "bias_correction": bias_info,
        "refs": refs["counts"],
        "n_train": len(rd),
    }
