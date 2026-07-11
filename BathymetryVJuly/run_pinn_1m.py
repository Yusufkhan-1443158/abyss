#!/usr/bin/env python3
"""run_pinn_1m.py — PINN-1M SDB, Round-1 native-1 m ablation ladder at Khalifa Port.

Implements REQUEST 1 (PINN_1M_LOG.md), converged spec METHOD_SPEC_DL.md +
METHOD_SPEC_PHYSICS.md. Reuses backend/unet_sdb.py AttUNet (CBAM σ head,
hetero-NLL, SILog+rank, edge smoothness, RESIDUAL_CARRIER, UNET_BUFFER_TEST
500 m spatial-block CV, OBS_PRECISION_W). Does NOT rewrite the backbone.

Ablation ladder (env PINN_RUNGS, default A,B,C,D):
  A  render-1 m baseline   : AttUNet trained at NATIVE 10 m, depth guided-upsampled
                             (render) to the target grid, scored at 1 m.
  B  native-1 m VHR-guided : RESIDUAL_CARRIER over guided-upsampled S2 features
     SR, physics OFF          (VHR luminance guide) + 2 bounded VHR morphology ch.
  C  B + wave residual     : + wave channels (celerity/λ/d_disp/σ_disp/m_wave) —
                             ALL NO-DATA at Khalifa (m_wave≡0 → L_wave≡0). MUST
                             equal B within seed noise (graceful-degradation proof).
  D  C + full VHR-SR ch     : + 3 normalised VHR RGB channels. Gated on the
                             VHR-specific shuffle guard.

Truth = in-situ multibeam (validation/KP*.xyz, EPSG:32640, ~41.6k pts).
Leakage-safe: spatial_block_split (>=500 m buffer), min_train_depth 0.0,
full-channel + VHR-channel shuffle guards, decile slope reported.

Env knobs (all defaulted):
  PINN_RES_M=1.0  PINN_RUNGS=A,B,C,D  PINN_SEEDS=42,1337,2024
  PINN_EPOCHS=40  PINN_CROP1=256  PINN_CROP10=64  PINN_BASE=32
  PINN_MARGIN_M=150  PINN_TTA=0  PINN_TAG=r1
  S2_GEE_FALLBACK=1 ; set -a; source .env; set +a  (MAPBOX_TOKEN)
"""
from __future__ import annotations
import os, sys, json, time, math
from pathlib import Path
import numpy as np

ROOT = Path("/home/wassi/Bathymetry_VMarch")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("S2_GEE_FALLBACK", "1")
os.environ.setdefault("S2_GEE_RAW", "1")

from scipy.ndimage import (gaussian_filter, uniform_filter, binary_dilation,
                           sobel, zoom as ndzoom)
from pyproj import Transformer

import run_very_hr_sdb as VHR          # build_feature_image, _fetch_mapbox_rgb, _guided_filter
import DL_2.dl2_dlnb as D              # spatial_block_split
from backend import unet_sdb
from backend.grid_1m import utm_lattice_transform

# ── config ────────────────────────────────────────────────────────────────────
KHALIFA_BBOX = [54.64, 24.79, 54.685, 24.838]         # matches cached S2 cube
S2_NPZ = ROOT / "DL_2" / "s2_cache" / "khalifa_s2_10band.npz"
MB_FILES = [ROOT / "validation" / "KP Basin Soundings 10m.xyz",
            ROOT / "validation" / "KP_EMAL_Soundings_10x_New.xyz"]

RES_M   = float(os.environ.get("PINN_RES_M", "1.0"))
RUNGS   = [r.strip().upper() for r in os.environ.get("PINN_RUNGS", "A,B,C,D").split(",") if r.strip()]
SEEDS   = [int(s) for s in os.environ.get("PINN_SEEDS", "42,1337,2024").split(",")]
EPOCHS  = int(os.environ.get("PINN_EPOCHS", "40"))
CROP1   = int(os.environ.get("PINN_CROP1", "256"))
CROP10  = int(os.environ.get("PINN_CROP10", "64"))
BASE    = int(os.environ.get("PINN_BASE", "32"))
MARGIN_M= float(os.environ.get("PINN_MARGIN_M", "150"))
USE_TTA = os.environ.get("PINN_TTA", "0") == "1"
TAG     = os.environ.get("PINN_TAG", "r1")
N_BLOCKS= int(os.environ.get("PINN_N_BLOCKS", "20"))
BUFFER_M= float(os.environ.get("PINN_BUFFER_M", "500"))
CROPS_PE= int(os.environ.get("PINN_CROPS_PER_EPOCH", "80"))
BATCH   = int(os.environ.get("PINN_BATCH", "4"))

OUTDIR = ROOT / "Very_HR_Results" / "khalifa_1m" / f"pinn_iter_{TAG}"
OUTDIR.mkdir(parents=True, exist_ok=True)
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")

DEPTH_BANDS = [(0, 2), (2, 5), (5, 10), (10, 16), (16, 25)]

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}] [pinn1m]", *a, flush=True)


# ── metrics ────────────────────────────────────────────────────────────────────
def _metrics(pred, truth):
    m = np.isfinite(pred) & np.isfinite(truth)
    pred, truth = pred[m], truth[m]
    if pred.size < 3:
        return dict(n=int(pred.size), rmse=float("nan"), bias=float("nan"),
                    r2=float("nan"), slope=float("nan"), decile_slope=float("nan"))
    err = pred - truth
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # calibration slope truth ~ a + b*pred
    b = float(np.polyfit(pred, truth, 1)[0]) if pred.std() > 1e-6 else float("nan")
    # decile slope: bin by predicted quantile, regress mean(truth)~mean(pred)
    ds = float("nan")
    try:
        q = np.quantile(pred, np.linspace(0, 1, 11))
        q = np.unique(q)
        if len(q) >= 3:
            idx = np.clip(np.digitize(pred, q) - 1, 0, len(q) - 2)
            mp, mt = [], []
            for k in range(len(q) - 1):
                s = idx == k
                if s.sum() >= 2:
                    mp.append(pred[s].mean()); mt.append(truth[s].mean())
            if len(mp) >= 3:
                ds = float(np.polyfit(mp, mt, 1)[0])
    except Exception:
        pass
    return dict(n=int(pred.size), rmse=rmse, bias=bias, r2=r2, slope=b, decile_slope=ds)


def _banded(pred, truth):
    out = {"pooled": _metrics(pred, truth)}
    for lo, hi in DEPTH_BANDS:
        s = (truth >= lo) & (truth < hi)
        out[f"{lo}-{hi}"] = _metrics(pred[s], truth[s])
    # convenience aggregates
    s05 = (truth >= 0) & (truth < 5)
    out["shallow_0_5"] = _metrics(pred[s05], truth[s05])
    sdeep = (truth >= 12) & (truth < 22)
    out["deep_12_22"] = _metrics(pred[sdeep], truth[sdeep])
    return out


# ── grid helpers ───────────────────────────────────────────────────────────────
def _bbox_to_pix(lon, lat, bbox, W, H):
    """Fractional pixel (col,row) in a lon/lat raster (top-left = w,n)."""
    w, s, e, n = bbox
    col = (lon - w) / (e - w) * W
    row = (n - lat) / (n - s) * H
    return col, row


def _luminance(rgb):
    return (rgb.astype(np.float32).mean(axis=2) / 255.0)


def _gf64(guide_gray, src, r=8, eps=1e-2):
    """float64 guided filter (He 2013). The production run_very_hr._guided_filter
    uses float32 cumsum which loses catastrophic precision over ~18 Mpx grids
    (sum ~3e8, float32 ULP ~30 m) -> box-filter garbage at the native-1 m scale.
    This float64 + uniform_filter reimplementation is numerically stable at 1 m."""
    I = np.asarray(guide_gray, np.float64)
    p = np.asarray(src, np.float64)
    win = 2 * int(r) + 1
    mI = uniform_filter(I, win); mp = uniform_filter(p, win)
    mIp = uniform_filter(I * p, win); mII = uniform_filter(I * I, win)
    varI = mII - mI * mI; covIp = mIp - mI * mp
    a = covIp / (varI + float(eps)); b = mp - a * mI
    ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    return (ma * I + mb).astype(np.float32)


def _guided_up_feature(feat10, guide, r, eps):
    """Bilinear upsample a 10 m feature channel to guide shape, then guided-filter."""
    H1, W1 = guide.shape
    up = ndzoom(feat10, (H1 / feat10.shape[0], W1 / feat10.shape[1]), order=1)
    up = up[:H1, :W1]
    if up.shape != guide.shape:
        tmp = np.zeros_like(guide); tmp[:up.shape[0], :up.shape[1]] = up; up = tmp
    return _gf64(guide, up, r=int(r), eps=float(eps))


# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_all = time.time()
    log(f"RES_M={RES_M} RUNGS={RUNGS} SEEDS={SEEDS} EPOCHS={EPOCHS} "
        f"CROP1={CROP1} BASE={BASE} TTA={USE_TTA} TAG={TAG}")
    log(f"GPU: cuda unavailable (wedged) -> CPU. torch threads={os.cpu_count()}")

    # ── 1. load S2 cube + build 10 m features + water mask ──────────────────────
    d = np.load(str(S2_NPZ), allow_pickle=True)
    bbox = [float(x) for x in d["bbox"]]
    Hs, Ws = int(d["height"]), int(d["width"])
    cube = np.zeros((Hs, Ws, 10), dtype=np.float32)
    for slot, band in [(0, "B2"), (1, "B3"), (2, "B4"), (6, "B8"), (8, "B11")]:
        if band in d.files:
            cube[..., slot] = d[band]
    scl = d["scl"] if "scl" in d.files else None
    log(f"S2 cube {Hs}x{Ws} bbox={bbox}")

    # water mask @ 10 m: SCL water/no-cloud + OSM coastline land removed
    if scl is not None:
        water10 = np.isin(scl, [6]) | (np.isin(scl, [2, 4, 5, 7, 11]) & False)  # SCL 6 = water
        # broaden: treat non-land non-cloud dark pixels as candidate water via NDWI
    else:
        water10 = np.ones((Hs, Ws), bool)
    # NDWI fallback / union so shallow bright bottoms are kept
    b3 = cube[..., 1] / 1e4; b8 = cube[..., 6] / 1e4
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    water10 = water10 | (ndwi > 0.0)
    try:
        from backend.osm_land_mask import coastline_land_for
        osm_land, osm_info = coastline_land_for(bbox, (Hs, Ws))
        if osm_land is not None and np.asarray(osm_land).shape == (Hs, Ws):
            osm_land = np.asarray(osm_land, bool)
            water10 = water10 & (~osm_land)
            log(f"R7 OSM coastline land removed: {int(osm_land.sum())} px "
                f"(src={osm_info.get('source')}, vintage={osm_info.get('vintage')})")
        else:
            log(f"OSM coastline returned no land (src={osm_info.get('source')})")
    except Exception as ex:
        log(f"OSM mask unavailable ({ex}); SCL/NDWI water only")
    water10 = water10 & (cube[..., 0] > 0)  # drop nodata
    log(f"water10 px = {int(water10.sum())}/{Hs*Ws}")

    feat10_full, feat_names, rinf = VHR.build_feature_image(cube, water10)
    log(f"10 m features {feat10_full.shape} names={feat_names}")

    # ── 2. load multibeam, reproject to lon/lat ────────────────────────────────
    tf = Transformer.from_crs("EPSG:32640", "EPSG:4326", always_xy=True)
    pts = np.vstack([np.loadtxt(str(f)) for f in MB_FILES])
    lon, lat = tf.transform(pts[:, 0], pts[:, 1])
    depth = pts[:, 2].astype(np.float32)
    ok = np.isfinite(depth) & (depth > -1.0) & (depth < 25.0)
    lon, lat, depth = lon[ok], lat[ok], depth[ok]
    depth = np.clip(depth, 0.0, 25.0)
    log(f"multibeam N={len(depth)} depth mean={depth.mean():.2f} "
        f"[{depth.min():.2f},{depth.max():.2f}]")

    # ── 3. crop bbox to multibeam extent + margin ──────────────────────────────
    mlat = 0.5 * (lat.min() + lat.max())
    dlat = MARGIN_M / 110570.0
    dlon = MARGIN_M / (111320.0 * math.cos(math.radians(mlat)))
    cw = max(bbox[0], lon.min() - dlon); cs = max(bbox[1], lat.min() - dlat)
    ce = min(bbox[2], lon.max() + dlon); cn = min(bbox[3], lat.max() + dlat)
    cbbox = [cw, cs, ce, cn]
    # crop indices into the 10 m raster
    c0, r0 = _bbox_to_pix(cw, cn, bbox, Ws, Hs)   # top-left
    c1, r1 = _bbox_to_pix(ce, cs, bbox, Ws, Hs)   # bottom-right
    c0, r0 = int(max(0, math.floor(c0))), int(max(0, math.floor(r0)))
    c1, r1 = int(min(Ws, math.ceil(c1))), int(min(Hs, math.ceil(r1)))
    feat10 = feat10_full[r0:r1, c0:c1, :].copy()
    water10c = water10[r0:r1, c0:c1].copy()
    Hc, Wc = feat10.shape[:2]
    log(f"crop bbox={cbbox} 10 m grid {Hc}x{Wc}")

    # metric size of cropped bbox
    W_m = (ce - cw) * 111320.0 * math.cos(math.radians(mlat))
    H_m = (cn - cs) * 110570.0
    H1 = int(round(H_m / RES_M)); W1 = int(round(W_m / RES_M))
    log(f"target native grid {H1}x{W1} @ {RES_M} m ({H1*W1/1e6:.1f} Mpx)")

    # ── 4. VHR RGB at native res (cached) ──────────────────────────────────────
    vhr_cache = OUTDIR / f"vhr_rgb_{H1}x{W1}.npz"
    rgb = None
    if vhr_cache.exists():
        rgb = np.load(str(vhr_cache))["rgb"]
        log(f"VHR RGB from cache {rgb.shape}")
    elif MAPBOX_TOKEN:
        os.environ.setdefault("MAPBOX_MAX_TILES", "500")
        log(f"fetching Mapbox VHR RGB (max_tiles={os.environ['MAPBOX_MAX_TILES']}) ...")
        rgb = VHR._fetch_mapbox_rgb(cbbox, (H1, W1), MAPBOX_TOKEN)
        if rgb is not None:
            np.savez_compressed(str(vhr_cache), rgb=rgb.astype(np.uint8))
            log(f"VHR RGB fetched {rgb.shape}")
    if rgb is None:
        log("VHR RGB unavailable -> luminance guide = bilinear S2-green (degraded)")
        g = ndzoom(cube[r0:r1, c0:c1, 1] / 1e4, (H1 / Hc, W1 / Wc), order=1)[:H1, :W1]
        g = (g - np.nanpercentile(g, 2)) / (np.nanpercentile(g, 98) - np.nanpercentile(g, 2) + 1e-6)
        rgb = np.clip(np.stack([g, g, g], -1) * 255, 0, 255).astype(np.uint8)
    guide = _luminance(rgb)

    # ── 5. water mask @ native res (nearest upsample of water10c) ──────────────
    water1 = ndzoom(water10c.astype(np.float32), (H1 / Hc, W1 / Wc), order=0)[:H1, :W1] > 0.5
    log(f"water1 px = {int(water1.sum())}/{H1*W1}")

    # ── 6. guided-upsample the 6 S2 depth features to native res ───────────────
    r_gf = max(4, int(round(12 * (1.0 / RES_M))))   # ~12 px at 1 m
    S2_1m = np.zeros((H1, W1, 6), dtype=np.float32)
    t0 = time.time()
    for ci in range(6):
        S2_1m[..., ci] = _guided_up_feature(feat10[..., ci], guide, r_gf, 1e-2)
    log(f"guided-upsampled 6 S2 features in {time.time()-t0:.0f}s (r={r_gf}px)")

    # ── 7. VHR morphology channels (bounded, texture-guarded) ──────────────────
    lum = guide.astype(np.float32)
    gx = sobel(lum, axis=1); gy = sobel(lum, axis=0)
    grad = np.hypot(gx, gy)
    grad = grad / (np.percentile(grad, 98) + 1e-6)
    win = max(3, int(round(5 * (1.0 / RES_M))))
    lmean = uniform_filter(lum, win)
    lstd = np.sqrt(np.clip(uniform_filter(lum * lum, win) - lmean * lmean, 0, None))
    lstd = lstd / (np.percentile(lstd, 98) + 1e-6)
    VHR_MORPH = np.stack([grad.astype(np.float32), lstd.astype(np.float32)], -1)
    # normalised RGB (rung D)
    RGB_N = (rgb.astype(np.float32) / 255.0)

    # wave channels (rung C): NO-DATA at Khalifa (m_wave ≡ 0 -> L_wave ≡ 0)
    WAVE = np.zeros((H1, W1, 3), dtype=np.float32)   # d_disp, sigma_disp(=big), m_wave=0
    WAVE[..., 1] = 1.0   # sigma_disp large placeholder (irrelevant; m_wave=0)
    wave_coverage = 0.0
    log(f"wave coverage at Khalifa = {wave_coverage:.4f} (m_wave grid all zero) "
        f"-> lambda_phys -> 0, L_wave === 0 (graceful degradation by construction)")

    # multibeam -> native pixel coords (fractional)
    mcol, mrow = _bbox_to_pix(lon, lat, cbbox, W1, H1)
    mcoli = np.clip(np.round(mcol).astype(int), 0, W1 - 1)
    mrowi = np.clip(np.round(mrow).astype(int), 0, H1 - 1)
    # 10 m pixel coords (for rung A + carrier fit)
    mcol10, mrow10 = _bbox_to_pix(lon, lat, cbbox, Wc, Hc)
    mcoli10 = np.clip(np.round(mcol10).astype(int), 0, Wc - 1)
    mrowi10 = np.clip(np.round(mrow10).astype(int), 0, Hc - 1)

    # EOT20 LAT datum context (reported separately; cached median has no single epoch)
    lat_off = _eot20_lat_offset(0.5 * (cw + ce), mlat)

    # ── channel stacks per rung ────────────────────────────────────────────────
    def stack_for(rung, carrier1):
        """Assemble native-res channel stack; carrier is appended LAST."""
        if rung == "B":
            chans = [S2_1m, VHR_MORPH]
        elif rung == "C":
            chans = [S2_1m, VHR_MORPH, WAVE]
        elif rung == "D":
            chans = [S2_1m, VHR_MORPH, WAVE, RGB_N]
        else:
            raise ValueError(rung)
        chans.append(carrier1[..., None])
        return np.concatenate(chans, axis=-1).astype(np.float32)

    results = {}
    for rung in RUNGS:
        log(f"================  RUNG {rung}  ================")
        results[rung] = run_rung(
            rung, feat10, water10c, water1, guide, S2_1m,
            lon, lat, depth, mcoli, mrowi, mcoli10, mrowi10,
            Hc, Wc, H1, W1, cbbox, stack_for, lat_off)

    # ── aggregate + write ──────────────────────────────────────────────────────
    summary = {
        "tag": TAG, "res_m": RES_M, "bbox_crop": cbbox,
        "native_grid": [H1, W1], "n_multibeam": int(len(depth)),
        "seeds": SEEDS, "epochs": EPOCHS, "crop1": CROP1, "base": BASE,
        "buffer_m": BUFFER_M, "n_blocks": N_BLOCKS,
        "eot20_lat_offset_m": lat_off,
        "wall_clock_s": round(time.time() - t_all, 1),
        "rungs": results,
    }
    outjson = OUTDIR / f"pinn_1m_{TAG}.json"
    outjson.write_text(json.dumps(summary, indent=2, default=float))
    log(f"WROTE {outjson}")
    _print_table(results)
    log(f"TOTAL wall-clock {time.time()-t_all:.0f}s")


def _eot20_lat_offset(lon0, lat0):
    """LAT offset (min astronomical tide) at the site centroid via EOT20, m."""
    try:
        import pyTMD
        n_hours = 24 * 365 * 4
        dt64 = np.datetime64("2018-01-01T00:00:00") + np.arange(n_hours) * np.timedelta64(1, "h")
        tide = pyTMD.compute.tide_elevations(
            x=np.array([lon0]), y=np.array([lat0]), delta_time=dt64,
            directory=str(ROOT / "cache"), model="EOT20",
            type="time series", standard="datetime", crs=4326,
            extrapolate=True, cutoff=50.0)
        tide = np.asarray(tide).ravel()
        if np.isfinite(tide).any():
            lat_off = float(np.nanmin(tide)); hat = float(np.nanmax(tide))
            log(f"EOT20 datum context: LAT(min astro tide)={lat_off:.3f} m, "
                f"HAT={hat:.3f} m, range={hat-lat_off:.2f} m (MSL=0 ref)")
            return lat_off
    except Exception as ex:
        log(f"EOT20 LAT offset unavailable ({ex}); reporting NaN (bias reported empirically)")
    return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
def run_rung(rung, feat10, water10c, water1, guide, S2_1m,
             lon, lat, depth, mcoli, mrowi, mcoli10, mrowi10,
             Hc, Wc, H1, W1, cbbox, stack_for, lat_off):
    """Train + evaluate one ablation rung across all seeds; pooled OOF metrics."""
    is_native = rung in ("B", "C", "D")
    pooled_pred, pooled_truth = [], []
    guard_rows = []
    sigma_cov = []
    per_seed = []
    seed_time = []
    for seed in SEEDS:
        ts = time.time()
        tr_m, te_m, _ = D.spatial_block_split(lat, lon, depth,
                                              n_blocks=N_BLOCKS, test_frac=0.25,
                                              buffer_m=BUFFER_M, seed=seed)
        n_tr, n_te = int(tr_m.sum()), int(te_m.sum())
        log(f"[{rung} s{seed}] spatial split train={n_tr} test={n_te} (buffer {BUFFER_M} m)")

        # ---- carrier (TRAIN-only robust linear on 10 m S2 -> depth) ----
        carrier10 = _fit_carrier10(feat10, water10c, mcoli10[tr_m], mrowi10[tr_m], depth[tr_m])

        if is_native:
            # native-res build
            carrier1 = _guided_up_feature(carrier10, guide, max(4, int(round(12 / RES_M))), 1e-2)
            carrier1 = np.clip(carrier1, 0, unet_sdb.MAX_DEPTH_M)
            feats = stack_for(rung, carrier1)
            water = water1
            H, W = H1, W1
            col_tr, row_tr = mcoli[tr_m], mrowi[tr_m]
            col_te, row_te = mcoli[te_m], mrowi[te_m]
            crop = CROP1
            use_carrier = True
            # VHR channel index range for shuffle guard: S2(0..5) then morph/rgb/wave.
            s2_idx = list(range(6))
            vhr_idx = _vhr_channel_indices(rung)
        else:
            # rung A (render baseline): SAME residual-carrier + S2 machinery as B
            # but at NATIVE 10 m, then depth guided-upsampled (render) to 1 m.
            # Carrier common to A and B, so A->B isolates the native-1 m VHR-guided
            # SR contribution alone (a fair baseline, not a straw man).
            feats = np.concatenate([feat10, carrier10[..., None]], axis=-1).astype(np.float32)
            water = water10c
            H, W = Hc, Wc
            col_tr, row_tr = mcoli10[tr_m], mrowi10[tr_m]
            col_te, row_te = mcoli10[te_m], mrowi10[te_m]
            crop = CROP10
            use_carrier = True
            s2_idx = list(range(6)); vhr_idx = []

        C = feats.shape[-1]
        # ---- rasterize TRAIN labels + obs precision ----
        label_grid = np.zeros((H, W), np.float32)
        label_mask = np.zeros((H, W), bool)
        obs_prec = np.zeros((H, W), np.float32)
        label_grid[row_tr, col_tr] = depth[tr_m]
        label_mask[row_tr, col_tr] = True
        obs_prec[row_tr, col_tr] = 1.0 / (0.15 ** 2)   # multibeam sigma ~0.15 m
        label_mask &= water

        # ---- forbidden mask: TEST points dilated by BUFFER_M ----
        forbid = _forbidden_mask(row_te, col_te, H, W, Hc, Wc, BUFFER_M, RES_M if is_native else 10.0)

        # ---- env wiring (reuse backbone; default OFF knobs) ----
        env_bak = _set_env(use_carrier)
        try:
            model = unet_sdb.train_unet_sdb(
                feats, label_grid, label_mask,
                epochs=EPOCHS, crops_per_epoch=CROPS_PE, batch=BATCH, crop=crop,
                base=BASE, seed=seed,
                forbidden_mask=forbid, water_mask=water,
                obs_precision_grid=obs_prec)
            mu, sig = unet_sdb.predict_grid_tta(
                model, feats, water, tile=512, overlap=32,
                use_tta=USE_TTA, mc_passes=0)
        finally:
            _restore_env(env_bak)

        # ---- eval (render-upsample for rung A) ----
        if is_native:
            pred_te = mu[row_te, col_te]
            sig_te = sig[row_te, col_te]
        else:
            mu_up = ndzoom(np.nan_to_num(mu, nan=0.0), (H1 / Hc, W1 / Wc), order=1)[:H1, :W1]
            mu_up = _gf64(guide, mu_up, r=max(4, int(round(12 / RES_M))), eps=1e-2)
            sig_up = ndzoom(np.nan_to_num(sig, nan=0.0), (H1 / Hc, W1 / Wc), order=1)[:H1, :W1]
            pred_te = mu_up[mrowi[te_m], mcoli[te_m]]
            sig_te = sig_up[mrowi[te_m], mcoli[te_m]]
        truth_te = depth[te_m]

        mtr = _banded(pred_te, truth_te)
        per_seed.append({"seed": seed, "n_test": n_te, "metrics": mtr})
        pooled_pred.append(pred_te); pooled_truth.append(truth_te)

        # sigma coverage (fraction within +-1.96 sigma)
        good = np.isfinite(pred_te) & np.isfinite(sig_te) & (sig_te > 0)
        if good.sum() > 5:
            cov = float(np.mean(np.abs(pred_te[good] - truth_te[good]) <= 1.96 * sig_te[good]))
            sigma_cov.append(cov)

        # ---- shuffle guards (inference-only, use_tta off) ----
        gd = _shuffle_guards(model, feats, water, col_te, row_te, truth_te,
                             s2_idx, vhr_idx, is_native, guide, H1, W1, Hc, Wc,
                             mcoli, mrowi, te_m, seed)
        gd["seed"] = seed
        guard_rows.append(gd)
        seed_time.append(time.time() - ts)
        _vp = gd.get("vhr_pct")
        _vp = float("nan") if _vp is None else _vp
        log(f"[{rung} s{seed}] pooled RMSE={mtr['pooled']['rmse']:.3f} "
            f"0-5={mtr['shallow_0_5']['rmse']:.3f} decile05={mtr['shallow_0_5']['decile_slope']:.2f} "
            f"jointΔ={gd.get('joint_pct', float('nan')):.1f}% "
            f"vhrΔ={_vp:.1f}% {time.time()-ts:.0f}s")

        # save 1 m products from first seed
        if seed == SEEDS[0]:
            _save_products(rung, mu if is_native else mu_up, sig if is_native else sig_up,
                           cbbox, water1)

    pp = np.concatenate(pooled_pred); tt = np.concatenate(pooled_truth)
    pooled = _banded(pp, tt)
    return {
        "native": is_native, "n_channels": int(feats.shape[-1]),
        "pooled_oof": pooled,
        "per_seed": per_seed,
        "shuffle_guards": guard_rows,
        "sigma_coverage_mean": float(np.mean(sigma_cov)) if sigma_cov else float("nan"),
        "seed_time_s": [round(x, 1) for x in seed_time],
    }


def _vhr_channel_indices(rung):
    # layout: S2(0-5), MORPH(6-7), [WAVE(8-10) if C/D], [RGB(...) if D], carrier(last)
    if rung == "B":
        return [6, 7]
    if rung == "C":
        return [6, 7]                       # wave chans are zero; VHR = morph only
    if rung == "D":
        return [6, 7, 11, 12, 13]           # morph + 3 RGB
    return []


def _fit_carrier10(feat10, water10c, col, row, dep):
    """TRAIN-only robust linear depth prior at 10 m -> full grid (leakage-safe)."""
    from sklearn.linear_model import HuberRegressor
    X = feat10[row, col, :]
    okc = np.all(np.isfinite(X), 1) & np.isfinite(dep)
    if okc.sum() < 20:
        return np.full(feat10.shape[:2], float(np.nanmedian(dep)), np.float32)
    Hc, Wc = feat10.shape[:2]
    try:
        Xt = np.nan_to_num(X[okc], nan=0.0, posinf=0.0, neginf=0.0)
        hr = HuberRegressor(max_iter=300).fit(Xt, dep[okc])
        flat = np.nan_to_num(feat10.reshape(-1, feat10.shape[-1]), nan=0.0,
                             posinf=0.0, neginf=0.0)
        pred = hr.predict(flat).reshape(Hc, Wc)
        fit_pred = hr.predict(Xt)
        fit_rmse = float(np.sqrt(np.mean((fit_pred - dep[okc]) ** 2)))
        fit_r = float(np.corrcoef(fit_pred, dep[okc])[0, 1]) if fit_pred.std() > 1e-6 else 0.0
    except Exception as ex:
        pred = np.full((Hc, Wc), float(np.nanmedian(dep)), np.float32)
        fit_rmse, fit_r = float("nan"), 0.0
        log(f"  carrier Huber failed ({ex}); flat-median scaffold")
    # clip to the observed truth envelope so the coarse prior is not saturated at 25
    pred = np.clip(pred, 0.0, min(unet_sdb.MAX_DEPTH_M, float(np.nanpercentile(dep, 99)) + 2.0))
    pred = pred.astype(np.float32)
    pred[~water10c] = 0.0
    log(f"  carrier10 train-fit RMSE={fit_rmse:.2f}m r={fit_r:.2f} "
        f"grid median={float(np.median(pred[water10c])):.2f}m "
        f"p10/p90={float(np.percentile(pred[water10c],10)):.1f}/{float(np.percentile(pred[water10c],90)):.1f}m")
    return pred


def _forbidden_mask(row_te, col_te, H, W, Hc, Wc, buffer_m, res_m):
    """Test-point cells dilated by buffer_m. Computed at ~10 m then upsampled."""
    # rasterize test points at 10 m grid, dilate, upsample to (H,W)
    fr = np.zeros((Hc, Wc), bool)
    r10 = np.clip((row_te.astype(float) * Hc / H).astype(int), 0, Hc - 1)
    c10 = np.clip((col_te.astype(float) * Wc / W).astype(int), 0, Wc - 1)
    fr[r10, c10] = True
    rad = max(1, int(round(buffer_m / 10.0)))
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    struct = (xx * xx + yy * yy) <= rad * rad
    fr = binary_dilation(fr, structure=struct)
    if (Hc, Wc) != (H, W):
        fr = ndzoom(fr.astype(np.float32), (H / Hc, W / Wc), order=0)[:H, :W] > 0.5
    return fr.astype(bool)


def _set_env(use_carrier):
    bak = {k: os.environ.get(k) for k in
           ["RESIDUAL_CARRIER", "INTERP_INPUT", "CARRIER_NO_ZSCORE", "CARRIER_SCALE_M",
            "UNET_BUFFER_TEST", "OBS_PRECISION_W", "LOSS", "RANK_LOSS_W",
            "INTERP_SMOOTH_W", "RANGE_DEPTH_GATE_M", "NLL_WARMUP_EPOCHS", "TARGET_NORM"]}
    os.environ["UNET_BUFFER_TEST"] = "1"
    os.environ["OBS_PRECISION_W"] = "1"
    os.environ["LOSS"] = "nll+silog+range+interp"
    os.environ["RANK_LOSS_W"] = os.environ.get("PINN_RANK_W", "0.1")
    os.environ["INTERP_SMOOTH_W"] = "0.005"
    os.environ["RANGE_DEPTH_GATE_M"] = "12.0"
    os.environ["NLL_WARMUP_EPOCHS"] = os.environ.get("PINN_NLL_WARMUP", "8")
    if use_carrier:
        os.environ["RESIDUAL_CARRIER"] = "1"
        os.environ["INTERP_INPUT"] = "1"
        os.environ["CARRIER_NO_ZSCORE"] = "1"
        os.environ["CARRIER_SCALE_M"] = "25.0"
        os.environ["TARGET_NORM"] = "none"     # residual carrier needs metres
    else:
        for k in ["RESIDUAL_CARRIER", "INTERP_INPUT", "CARRIER_NO_ZSCORE"]:
            os.environ.pop(k, None)
        os.environ["TARGET_NORM"] = "none"
    return bak


def _restore_env(bak):
    for k, v in bak.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _shuffle_guards(model, feats, water, col_te, row_te, truth_te,
                    s2_idx, vhr_idx, is_native, guide, H1, W1, Hc, Wc,
                    mcoli, mrowi, te_m, seed):
    """Inference-only leakage guards. Returns pct RMSE degradation vs baseline,
    both POOLED and on the SHALLOW band (0-5 m) where spectral/VHR signal — not
    the deep-mean carrier scaffold — actually governs the metric. For a
    mean-matching carrier the pooled shuffle-Δ is weak by construction (E4 trap);
    the shallow-band Δ is the honest discriminator for the carrier rungs."""
    rng = np.random.default_rng(seed + 7)
    sh = (truth_te >= 0) & (truth_te < 5)

    def _pred(fz):
        mu, _ = unet_sdb.predict_grid_tta(model, fz, water, tile=1024, overlap=16,
                                          use_tta=False, mc_passes=0)
        if is_native:
            p = mu[row_te, col_te]
        else:
            mu_up = ndzoom(np.nan_to_num(mu, nan=0.0), (H1 / Hc, W1 / Wc), order=1)[:H1, :W1]
            p = mu_up[mrowi[te_m], mcoli[te_m]]
        return p

    def _rmse(p, sel):
        m = np.isfinite(p) & np.isfinite(truth_te) & sel
        return float(np.sqrt(np.mean((p[m] - truth_te[m]) ** 2))) if m.sum() > 3 else float("nan")

    p_base = _pred(feats)
    base = _rmse(p_base, np.ones_like(sh)); base_sh = _rmse(p_base, sh)

    def _shuffle(idx):
        if not idx:
            return float("nan"), float("nan")
        fz = feats.copy()
        Hh, Ww = fz.shape[:2]
        perm = rng.permutation(Hh * Ww)
        for ci in idx:
            fz[..., ci] = fz[..., ci].reshape(-1)[perm].reshape(Hh, Ww)
        p = _pred(fz)
        return _rmse(p, np.ones_like(sh)), _rmse(p, sh)

    joint, joint_sh = _shuffle(list(range(feats.shape[-1])))
    s2_sh_r, s2_sh_sh = _shuffle(s2_idx)
    vhr_r, vhr_sh_sh = _shuffle(vhr_idx) if vhr_idx else (float("nan"), float("nan"))

    def pct(x, b):
        return float((x - b) / b * 100.0) if (np.isfinite(x) and np.isfinite(b) and b > 0) else float("nan")
    out = {
        "base_rmse": round(base, 4), "base_rmse_shallow": round(base_sh, 4),
        "joint_rmse": round(joint, 4), "joint_pct": round(pct(joint, base), 2),
        "joint_pct_shallow": round(pct(joint_sh, base_sh), 2),
        "s2_rmse": round(s2_sh_r, 4), "s2_pct": round(pct(s2_sh_r, base), 2),
        "s2_pct_shallow": round(pct(s2_sh_sh, base_sh), 2),
    }
    if vhr_idx:
        out["vhr_rmse"] = round(vhr_r, 4)
        out["vhr_pct"] = round(pct(vhr_r, base), 2)
        out["vhr_pct_shallow"] = round(pct(vhr_sh_sh, base_sh), 2)
    else:
        out["vhr_pct"] = None; out["vhr_pct_shallow"] = None
    return out


def _save_products(rung, mu, sig, cbbox, water1):
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        H, W = mu.shape
        w, s, e, n = cbbox
        tr = from_bounds(w, s, e, n, W, H)
        for name, arr in [("depth", mu), ("sigma", sig)]:
            a = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
            p = OUTDIR / f"pinn_{rung}_{name}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1,
                               dtype="float32", crs=CRS.from_epsg(4326), transform=tr,
                               nodata=-9999.0, compress="deflate") as dst:
                dst.write(a, 1)
        log(f"[{rung}] wrote depth.tif + sigma.tif -> {OUTDIR}")
    except Exception as ex:
        log(f"[{rung}] product write failed: {ex}")


def _print_table(results):
    log("=========== ABLATION LADDER (pooled OOF) ===========")
    hdr = f"{'rung':4} {'ch':3} {'pooledRMSE':11} {'R2':7} {'0-5RMSE':8} {'0-5slope':8} " \
          f"{'decile':7} {'12-22R2':8} {'jointΔ%':8} {'vhrΔ%':7} {'σcov':6}"
    log(hdr)
    for rung, r in results.items():
        p = r["pooled_oof"]
        gd = r["shuffle_guards"]
        jp = np.nanmean([g.get("joint_pct", np.nan) for g in gd])
        vp = np.nanmean([g.get("vhr_pct", np.nan) if g.get("vhr_pct") is not None else np.nan for g in gd])
        log(f"{rung:4} {r['n_channels']:3d} "
            f"{p['pooled']['rmse']:11.3f} {p['pooled']['r2']:7.3f} "
            f"{p['shallow_0_5']['rmse']:8.3f} {p['shallow_0_5']['slope']:8.2f} "
            f"{p['shallow_0_5']['decile_slope']:7.2f} {p['deep_12_22']['r2']:8.3f} "
            f"{jp:8.1f} {vp:7.1f} {r['sigma_coverage_mean']:6.2f}")


if __name__ == "__main__":
    main()
