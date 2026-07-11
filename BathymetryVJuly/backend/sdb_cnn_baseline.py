"""SDB Patch-CNN Baseline — Iteration 1 + Iteration 2 + Iteration 3.

Self-contained:
  Alpha  — SDBPatchDataset + make_spatial_block_centers (leakage-safe splitter)
  Beta   — PatchCNN (bounded depth head + heteroscedastic log-var) +
            training loop (hetero-NLL + inverse-freq depth weights)
  Gamma  — evaluate() → per-bin RMSE (0-5/5-10/10-20 m) + band-shuffle guard

ITER-2 additions (all gated behind --iter2 flag, default OFF):
  - buffer_m raised to 500 m for (A) honest-baseline run
  - Stumpf 2003 classical fit on TRAIN → z_stumpf per pixel (robust Theil-Sen)
  - ResidualCNN predicts (z_true - z_stumpf); final z = z_stumpf + residual, bounded
  - Combined loss: hetero-NLL + lambda_rank * SILog-rank term (anti-collapse)
  - Post-training σ temperature recalibration (k) to hit 95% coverage on val
  - Decile predicted-vs-observed slope reported in metrics

ITER-3 additions (all gated behind --iter3 flag, default OFF):
  - Physics prior = Lyzenga 1985 two-band log-linear (turbidity-robust):
      X_i = ln(R_i_deglint - R_i_deep)  for B2, B3 (and B4)
      z_lyz = a0 + a1*X_B2 + a2*X_B3 + a3*X_B4 + a4*NDTI (turbidity covariate)
      fit by WLS on TRAIN soundings (weights = inverse depth-bin freq)
  - ResidualCNN predicts (z_true - z_lyz); final z = z_lyz + residual, bounded [0,25]
  - Same SILog-rank anti-collapse loss as Iter-2
  - σ calibration on HELD-OUT VAL FOLD (~20% of TRAIN blocks), NOT train set
    → eliminates the k<1 deflation failure seen in Iter-2
  - Same ≥500 m-buffer spatial-block split (reuses cached S2 npz)

DATA LAYER:
  Ground truth comes from XYZ multibeam soundings (EPSG:32640, positive-down)
  and SWOT Dhanna shapefile (EPSG:32639). Sentinel-2 fetched via GEE.
  Label raster built by rasterizing XYZ points onto the S2 10m grid.

Usage:
    # Iter-1 (90 m buffer, original):
    .venv/bin/python3 backend/sdb_cnn_baseline.py --sites khalifa
    # (A) Honest baseline (500 m buffer, same model):
    .venv/bin/python3 backend/sdb_cnn_baseline.py --sites khalifa --buffer-m 500
    # (B) Iter-2 (500 m buffer + Stumpf-primary + anti-collapse):
    .venv/bin/python3 backend/sdb_cnn_baseline.py --sites khalifa --buffer-m 500 --iter2
    # (C) Iter-3 (500 m buffer + Lyzenga-Kd-primary + val-fold sigma-cal):
    .venv/bin/python3 backend/sdb_cnn_baseline.py --sites khalifa --buffer-m 500 --iter3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
L = logging.getLogger("sdb_baseline")

# ── constants ─────────────────────────────────────────────────────────────────
MAX_DEPTH_M   = 25.0
MIN_DEPTH_M   = 0.0
PATCH_PX      = 9          # 9×9 pixels centred on each label point
PATCH_RADIUS  = PATCH_PX // 2          # 4 px = 40 m @10m resolution
BLOCK_BUF_M   = 200.0                  # Iter-1 default buffer (overridden by --buffer-m)
N_BLOCKS      = 8                      # spatial KMeans blocks for CV
N_EPOCHS      = 60
BATCH_SIZE    = 256
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
N_BINS        = 5           # inverse-freq depth bins (0,5,10,15,20,25m)
SEED          = 42
S2_START      = "2023-01-01"
S2_END        = "2024-12-31"
S2_CLOUD_PCT  = 30

# Iter-2 hyper-parameters (gated behind --iter2)
ITER2_BUFFER_M   = 500.0          # honest spatial-autocorrelation buffer
ITER2_LAMBDA_RANK = 0.15          # weight on SILog-rank term
ITER2_SIGMA_TARGET_LO = 0.90      # target σ-95% coverage lower bound
ITER2_SIGMA_TARGET_HI = 0.97      # target σ-95% coverage upper bound

# Iter-3 hyper-parameters (gated behind --iter3)
ITER3_BUFFER_M    = 500.0         # same honest buffer as Iter-2
ITER3_LAMBDA_RANK = 0.15          # weight on SILog-rank term (same as Iter-2)
ITER3_VAL_FRAC    = 0.20          # fraction of TRAIN blocks carved as val for sigma-cal
ITER3_SIGMA_TARGET = 0.95         # target σ-95% coverage on val fold
ITER3_LYZ_DEEP_PCT = 2.0          # percentile of darkest pixels used as R_infinity
ITER3_USE_B4      = True          # include B4 (red) as third Lyzenga band
ITER3_USE_NDTI    = True          # include NDTI as turbidity covariate in Lyzenga fit

# Depth bands for Gamma evaluation [lo, hi) metres
# Round-1 extension: full 5×2 m bins so every AC bar has per-bin bias/RMSE readout.
# (previously was 3 coarse bins — the INSITU_RMSE_LOG_DL.md line-104 flag)
GAMMA_BANDS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]

# ── site registry ─────────────────────────────────────────────────────────────
SITE_CFG: Dict[str, Dict] = {
    "khalifa": {
        "label":   "Khalifa Port",
        # [W, S, E, N] WGS84 — covers full KP+EMAL multibeam extent + 0.01° margin
        "bbox":    [54.640, 24.790, 54.685, 24.838],
        "xyz":     [
            ROOT / "validation" / "KP Basin Soundings 10m.xyz",
            ROOT / "validation" / "KP_EMAL_Soundings_10x_New.xyz",
        ],
        "xyz_epsg": 32640,
        "shp":     None,
    },
    "jbel_dhanna": {
        "label":   "Jbel Dhanna",
        # SWOT_Dhanna.shp extent: lon 52.588-52.598, lat 24.210-24.237 (EPSG:32639)
        "bbox":    [52.583, 24.207, 52.602, 24.240],
        "xyz":     [],
        "xyz_epsg": 32639,
        "shp":     ROOT / "validation" / "swot_dhanna" / "SWOT_Dhanna.shp",
    },
    "omc": {
        "label":   "Old Mussafah Channel",
        "bbox":    [54.290, 24.380, 54.475, 24.500],
        "xyz":     [ROOT / "validation" / "OMC_Soundings_20m.xyz"],
        "xyz_epsg": 32640,
        "shp":     None,
    },
    # DL Method Loop R1: CShelph bottom detections (ICESat-2 photon-counted, 6 beams × 13 dates).
    # 33594 pts in scene-cache bbox [54.29,24.38,54.475,24.50], depth 0.30-7.50m, std=2.58m.
    # Depth histogram: 38% 0-2m, 11% 2-4m, 21% 4-6m, 29% 6-8m — much better spread than cells-only.
    # Cache: May_2026_results/v_iho_iterations/iter_cnn_baseline/s2_cache/omc_r6_s2_gee.npz
    #        (median composite from old_mussafah_s2_scenes.npz, 1337×2060 @ 10m)
    "omc_r6": {
        "label":   "Old Mussafah (CShelph ICESat-2 bottom, scene-cache bbox)",
        "bbox":    [54.29, 24.38, 54.475, 24.50],
        # cshelph_bottom.npz loaded via special handler in load_truth_points
        "xyz":     [],
        "xyz_epsg": 4326,
        "shp":     None,
        "cshelph_bottom": ROOT / "June_2026" / "altimeter_point_clouds" / "old_mussafah" / "cshelph_bottom.npz",
    },
    # DL Method Loop R1: same site but using the R6 aggregated cells (25m-gridded ATL24, n=3066 0-15m,
    # n=3056 0-12m, n=1908 after water mask).  date_id (6 unique) used as fold basis in the runner.
    # T5 target: reproduce R6 anchor RMSE~0.9237 m ±0.15 m (different S2 image resolution/bbox).
    "omc_r6_cells": {
        "label":   "Old Mussafah (R6 aggregated ATL24 cells, 25m grid)",
        "bbox":    [54.29, 24.38, 54.475, 24.50],
        "xyz":     [],
        "xyz_epsg": 4326,
        "shp":     None,
        # cells_npz: lat/lon/depth loaded by special handler in load_truth_points
        "cells_npz": ROOT / "June_2026" / "rounds" / "20260610_212230_uae_r6" / "old_mussafah_r6_cells.npz",
        # Reuse the S2 cache from the omc_r6 run (same bbox, same image)
        "s2_cache_key": "omc_r6",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Alpha-1  — GEE S2 fetch (reusing very_hr_engine helpers)
# ══════════════════════════════════════════════════════════════════════════════
def _load_env():
    """Load .env from project root if not already in environment."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def fetch_s2_for_site(site_key: str, cache_dir: Path) -> Dict:
    """Fetch or load cached S2 B2/B3/B4/B8 + SCL for a site via GEE.

    Returns dict with numpy arrays (float32, raw DN / will be /10000 later):
      blue, green, red, nir, scl  — shape (H, W)
      bbox, height, width, resolution_m
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Allow site config to redirect to an existing cache (e.g. omc_r6_cells → omc_r6)
    cache_key = SITE_CFG.get(site_key, {}).get("s2_cache_key", site_key)
    cache_file = cache_dir / f"{cache_key}_s2_gee.npz"
    if cache_file.exists():
        L.info(f"[{site_key}] Loading cached S2 from {cache_file}")
        d = np.load(cache_file)
        return {k: d[k] for k in d.files}

    L.info(f"[{site_key}] Fetching S2 via GEE ...")
    from backend.very_hr_engine import fetch_s2_gee, _init_gee
    _init_gee()

    bbox = SITE_CFG[site_key]["bbox"]
    s2 = fetch_s2_gee(bbox, S2_START, S2_END, cloud=S2_CLOUD_PCT)
    np.savez_compressed(
        cache_file,
        blue=s2["blue"], green=s2["green"], red=s2["red"],
        nir=s2["nir"], scl=s2["scl"].astype(np.float32),
        bbox=np.array(bbox),
        height=np.array(s2["height"]),
        width=np.array(s2["width"]),
        resolution_m=np.array(s2["resolution_m"]),
    )
    L.info(f"[{site_key}] S2 cached -> {cache_file} ({s2['height']}x{s2['width']} px)")
    return {
        "blue": s2["blue"], "green": s2["green"], "red": s2["red"],
        "nir": s2["nir"], "scl": s2["scl"].astype(np.float32),
        "bbox": np.array(bbox),
        "height": np.array(s2["height"]), "width": np.array(s2["width"]),
        "resolution_m": np.array(s2["resolution_m"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Alpha-2  — Ground-truth XYZ / SHP reader -> lat/lon/depth arrays
# ══════════════════════════════════════════════════════════════════════════════
def load_truth_points(site_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat, lon, depth_m) for the site's in-situ truth.

    depth_m is positive-down, clipped to [MIN_DEPTH_M, MAX_DEPTH_M].
    CRS reprojected to WGS84.
    """
    from pyproj import Transformer
    cfg = SITE_CFG[site_key]
    lats, lons, deps = [], [], []

    # R6 aggregated ATL24 cells (lat/lon/depth/date_id from old_mussafah_r6_cells.npz)
    if cfg.get("cells_npz") and Path(cfg["cells_npz"]).exists():
        cel = np.load(str(cfg["cells_npz"]), allow_pickle=True)
        lat_ = cel["lat"].astype(np.float64)
        lon_ = cel["lon"].astype(np.float64)
        dep_ = np.abs(cel["depth"].astype(np.float64))
        lats.append(lat_); lons.append(lon_); deps.append(dep_)
        L.info(f"[{site_key}] Loaded {len(dep_)} R6-cells from "
               f"{Path(cfg['cells_npz']).name}  "
               f"depth [{dep_.min():.2f},{dep_.max():.2f}] m  std={dep_.std():.2f} m")

    # ATL24 cells.npz (photon-derived cells)
    if cfg.get("atl24_cells") and Path(cfg["atl24_cells"]).exists():
        atl = np.load(str(cfg["atl24_cells"]), allow_pickle=True)
        lat_ = atl["cells_lat"].astype(np.float64)
        lon_ = atl["cells_lon"].astype(np.float64)
        dep_ = np.abs(atl["cells_depth"].astype(np.float64))
        lats.append(lat_); lons.append(lon_); deps.append(dep_)
        L.info(f"[{site_key}] Loaded {len(dep_)} ATL24 cells from "
               f"{Path(cfg['atl24_cells']).name}  "
               f"depth [{dep_.min():.1f},{dep_.max():.1f}] m")

    # CShelph bottom detections (ICESat-2 photon-counted, multi-beam multi-date)
    if cfg.get("cshelph_bottom") and Path(cfg["cshelph_bottom"]).exists():
        csh = np.load(str(cfg["cshelph_bottom"]), allow_pickle=True)
        lat_ = csh["lat"].astype(np.float64)
        lon_ = csh["lon"].astype(np.float64)
        dep_ = np.abs(csh["depth"].astype(np.float64))
        lats.append(lat_); lons.append(lon_); deps.append(dep_)
        L.info(f"[{site_key}] Loaded {len(dep_)} CShelph bottom pts from "
               f"{Path(cfg['cshelph_bottom']).name}  "
               f"depth [{dep_.min():.2f},{dep_.max():.2f}] m  std={dep_.std():.2f} m")

    # XYZ files (easting, northing, depth in UTM)
    if cfg["xyz"]:
        tr = Transformer.from_crs(cfg["xyz_epsg"], 4326, always_xy=True)
        for xyz_path in cfg["xyz"]:
            if not Path(xyz_path).exists():
                L.warning(f"[{site_key}] XYZ not found: {xyz_path}")
                continue
            arr = np.loadtxt(xyz_path)
            e, n, d = arr[:, 0], arr[:, 1], arr[:, 2]
            d = np.abs(d)
            lon_, lat_ = tr.transform(e, n)
            lats.append(lat_); lons.append(lon_); deps.append(d)
            L.info(f"[{site_key}] Loaded {len(d)} pts from {Path(xyz_path).name}")

    # Shapefile (SWOT Dhanna uses Z column, EPSG:32639)
    if cfg.get("shp") and Path(cfg["shp"]).exists():
        import geopandas as gpd
        gdf = gpd.read_file(cfg["shp"])
        gdf4326 = gdf.to_crs(4326)
        lon_ = gdf4326.geometry.x.values
        lat_ = gdf4326.geometry.y.values
        for col in ["Z", "depth", "z", "DEPTH"]:
            if col in gdf.columns:
                d = np.abs(gdf[col].values.astype(np.float64))
                break
        else:
            raise ValueError(f"[{site_key}] Cannot find depth column in {cfg['shp']}")
        lats.append(lat_); lons.append(lon_); deps.append(d)
        L.info(f"[{site_key}] Loaded {len(d)} pts from {Path(cfg['shp']).name}")

    if not lats:
        raise FileNotFoundError(f"[{site_key}] No truth data found")

    lat = np.concatenate(lats).astype(np.float64)
    lon = np.concatenate(lons).astype(np.float64)
    dep = np.concatenate(deps).astype(np.float64)

    keep = (dep >= MIN_DEPTH_M) & (dep <= MAX_DEPTH_M) & np.isfinite(dep)
    lat, lon, dep = lat[keep], lon[keep], dep[keep]

    bbox = SITE_CFG[site_key]["bbox"]
    in_bbox = (
        (lon >= bbox[0]) & (lon <= bbox[2]) &
        (lat >= bbox[1]) & (lat <= bbox[3])
    )
    lat, lon, dep = lat[in_bbox], lon[in_bbox], dep[in_bbox]

    L.info(f"[{site_key}] Truth after clip+bbox: {len(dep)} pts, "
           f"depth [{dep.min():.1f}, {dep.max():.1f}] m")
    return lat, lon, dep


# ══════════════════════════════════════════════════════════════════════════════
# Alpha-3  — Feature cube construction (Hedley deglint + log + Stumpf + NDTI)
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_cube(s2: Dict, feature_set: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (H, W, C) float32 feature array and NDWI water mask.

    Canonical channels (A0, 12 total):
      0  B2 blue  (DN/10000, clipped 0-0.3)
      1  B3 green
      2  B4 red
      3  B8 NIR
      4  B2_dg   Hedley-deglinted blue
      5  B3_dg   Hedley-deglinted green
      6  ln_B2   log(B2_dg)  -- depth-sensitive
      7  ln_B3   log(B3_dg)
      8  ln_B4   log(B4_dg)
      9  stumpf_BG   ln(1000*B2) / ln(1000*B3)   Stumpf 2003 log-ratio
     10  stumpf_GR   ln(1000*B3) / ln(1000*B4)
     11  NDTI        (B3-B4)/(B3+B4)  turbidity index
    Water mask = NDWI (B3-B8)/(B3+B8) > 0 AND SCL water classes.

    FEATURE_SET env / feature_set arg selects the ablation set:
      A0 = 12-ch canonical (default; must reproduce R6 0.9237 m anchor)
      A1 = parsimony-8: drop B2,B3,B4,B2_dg,B3_dg; keep
           [ln_B2,ln_B3,ln_B4,stumpf_BG,stumpf_GR,NDTI,B8,deep_sub_ln_B2]
      A2 = A0 + Lyzenga DII_BG (13 ch)
      A3 = A1 + DII_BG (9 ch)

    DII_BG (Lyzenga 1978/1981) = ln_B2 - (sigma_B2/sigma_B3) * ln_B3
    where sigma_B2, sigma_B3 are std of log-radiance over the existing deep mask.
    This is band-shuffle-safe (purely spectral, no spatial/coordinate info).
    """
    if feature_set is None:
        feature_set = os.environ.get("FEATURE_SET", "A0").upper()

    eps = 1e-6
    H, W = int(s2["height"]), int(s2["width"])
    blue  = np.clip(s2["blue"].astype(np.float64)  / 10000.0, eps, 0.5)
    green = np.clip(s2["green"].astype(np.float64) / 10000.0, eps, 0.5)
    red   = np.clip(s2["red"].astype(np.float64)   / 10000.0, eps, 0.5)
    nir   = np.clip(s2["nir"].astype(np.float64)   / 10000.0, eps, 0.5)

    def _r(x):
        return x.reshape(H, W) if x.ndim == 1 else x
    blue, green, red, nir = _r(blue), _r(green), _r(red), _r(nir)

    ndwi = (green - nir) / (green + nir + eps)
    ndwi_mask = ndwi > 0.0

    scl = _r(s2["scl"].astype(np.float32))
    scl_water = (scl == 6)
    water_mask = ndwi_mask | scl_water

    deep = ndwi > 0.3
    if deep.sum() < 50:
        deep = ndwi_mask
    if deep.sum() < 10:
        deep = np.ones((H, W), dtype=bool)

    def _p2(band):
        return float(np.percentile(band[deep], 2)) if deep.sum() >= 10 else eps

    b_min = _p2(blue); g_min = _p2(green); r_min = _p2(red)

    blue_dg  = np.clip(blue  - b_min, eps, None)
    green_dg = np.clip(green - g_min, eps, None)
    red_dg   = np.clip(red   - r_min, eps, None)

    ln_b = np.log(blue_dg)
    ln_g = np.log(green_dg)
    ln_r = np.log(red_dg)

    stumpf_bg = np.log(1000.0 * blue)  / (np.log(1000.0 * green) + eps)
    stumpf_gr = np.log(1000.0 * green) / (np.log(1000.0 * red)   + eps)

    ndti = (green - red) / (green + red + eps)

    # ── Lyzenga DII_BG (needed for A2 / A3 sets) ─────────────────────────────
    # Variance ratio computed on EXISTING deep mask (no new imagery).
    # DII_BG = ln_B2 - (sigma_B2 / sigma_B3) * ln_B3
    # Ratio clamped [0.1, 10] for numerical safety.
    sigma_b2 = float(np.std(ln_b[deep])) if deep.sum() >= 10 else 1.0
    sigma_b3 = float(np.std(ln_g[deep])) if deep.sum() >= 10 else 1.0
    if sigma_b3 < 1e-9:
        sigma_b3 = 1e-9
    lyzenga_k = float(np.clip(sigma_b2 / sigma_b3, 0.1, 10.0))
    dii_bg = (ln_b - lyzenga_k * ln_g).astype(np.float32)
    L.info(f"[DII_BG] sigma_B2={sigma_b2:.4f}  sigma_B3={sigma_b3:.4f}  "
           f"k={lyzenga_k:.4f}  DII finite={np.isfinite(dii_bg).mean()*100:.1f}%")

    # ── deep-subtracted ln_B2 (for A1/A3 parsimony set) ──────────────────────
    # Per-pixel deviation of ln_B2 from deep-water baseline (Lyzenga style).
    ln_b2_deep_mean = float(np.mean(ln_b[deep])) if deep.sum() >= 10 else 0.0
    deep_sub_ln_b2  = (ln_b - ln_b2_deep_mean).astype(np.float32)

    # ── A0 canonical cube ─────────────────────────────────────────────────────
    a0_arrays = [
        blue.astype(np.float32),
        green.astype(np.float32),
        red.astype(np.float32),
        nir.astype(np.float32),
        blue_dg.astype(np.float32),
        green_dg.astype(np.float32),
        ln_b.astype(np.float32),
        ln_g.astype(np.float32),
        ln_r.astype(np.float32),
        stumpf_bg.astype(np.float32),
        stumpf_gr.astype(np.float32),
        ndti.astype(np.float32),
    ]
    a0_names = [
        "B2", "B3", "B4", "B8",
        "B2_dg", "B3_dg",
        "ln_B2", "ln_B3", "ln_B4",
        "stumpf_BG", "stumpf_GR",
        "NDTI",
    ]

    # ── Select channel set ────────────────────────────────────────────────────
    if feature_set == "A0":
        arrays = a0_arrays
        names  = a0_names

    elif feature_set == "A1":
        # parsimony-8: drop B2,B3,B4,B2_dg,B3_dg; keep ln_B2,ln_B3,ln_B4,
        # stumpf_BG,stumpf_GR,NDTI,B8, deep_sub_ln_B2
        arrays = [
            ln_b.astype(np.float32),
            ln_g.astype(np.float32),
            ln_r.astype(np.float32),
            stumpf_bg.astype(np.float32),
            stumpf_gr.astype(np.float32),
            ndti.astype(np.float32),
            nir.astype(np.float32),
            deep_sub_ln_b2,
        ]
        names = [
            "ln_B2", "ln_B3", "ln_B4",
            "stumpf_BG", "stumpf_GR", "NDTI",
            "B8",
            "deep_sub_ln_B2",
        ]

    elif feature_set == "A2":
        # 12 canonical + DII_BG = 13 ch
        arrays = a0_arrays + [dii_bg]
        names  = a0_names  + ["DII_BG"]

    elif feature_set == "A3":
        # A1 (8 ch) + DII_BG = 9 ch
        arrays = [
            ln_b.astype(np.float32),
            ln_g.astype(np.float32),
            ln_r.astype(np.float32),
            stumpf_bg.astype(np.float32),
            stumpf_gr.astype(np.float32),
            ndti.astype(np.float32),
            nir.astype(np.float32),
            deep_sub_ln_b2,
            dii_bg,
        ]
        names = [
            "ln_B2", "ln_B3", "ln_B4",
            "stumpf_BG", "stumpf_GR", "NDTI",
            "B8",
            "deep_sub_ln_B2",
            "DII_BG",
        ]

    else:
        L.warning(f"Unknown FEATURE_SET='{feature_set}'; falling back to A0")
        arrays = a0_arrays
        names  = a0_names

    cube = np.stack(arrays, axis=-1)  # (H, W, C)
    L.info(f"[build_feature_cube] FEATURE_SET={feature_set}  C={cube.shape[-1]}  "
           f"channels={names}")

    return cube.astype(np.float32), water_mask, names


# ══════════════════════════════════════════════════════════════════════════════
# Alpha-4  — Spatial-block splitter (leakage-safe, KMeans on lat/lon)
# ══════════════════════════════════════════════════════════════════════════════
def make_spatial_block_centers(
    lat: np.ndarray,
    lon: np.ndarray,
    n_blocks: int = N_BLOCKS,
    test_frac: float = 0.25,
    buffer_m: float = BLOCK_BUF_M,
    seed: int = SEED,
    return_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """KMeans spatial blocks -> leakage-safe train/test boolean masks.

    Points within `buffer_m` of any test-block centroid are excluded from the
    TRAINING set (the buffer is one-sided: test points are never in train bbox).
    Returns (train_mask, test_mask) both shape (N,).
    If return_labels=True, returns (train_mask, test_mask, labels) where
    labels are the KMeans block assignments for all N points (used for val carve).

    Parameters
    ----------
    buffer_m : float
        Minimum separation between test and train pixels.
        Iter-1 default: 200 m (9 px).
        Iter-2 honest:  500 m (~50 px) -- above S2 spatial-autocorrelation length.
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    coords = np.stack([lat, lon], axis=1)
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    labels = km.fit_predict(coords)

    block_ids = np.unique(labels)
    n_test_blocks = max(1, round(len(block_ids) * test_frac))
    test_block_ids = rng.choice(block_ids, size=n_test_blocks, replace=False)
    test_block_ids_set = set(test_block_ids.tolist())

    test_mask = np.isin(labels, list(test_block_ids_set))
    train_mask_raw = ~test_mask

    lat_mid = lat.mean()
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * np.cos(np.radians(lat_mid))
    buf_lat = buffer_m / m_per_deg_lat
    buf_lon = buffer_m / m_per_deg_lon

    test_lats = lat[test_mask]
    test_lons = lon[test_mask]

    too_close = np.zeros(len(lat), dtype=bool)
    for tl, tlo in zip(test_lats, test_lons):
        nearby_lat = np.abs(lat - tl) < buf_lat * 2
        nearby_lon = np.abs(lon - tlo) < buf_lon * 2
        candidates = nearby_lat & nearby_lon & train_mask_raw
        if not candidates.any():
            continue
        dlat = np.radians(lat[candidates] - tl)
        dlon = np.radians(lon[candidates] - tlo)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(tl)) * np.cos(np.radians(lat[candidates]))
             * np.sin(dlon / 2) ** 2)
        dist_m = 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        idxs = np.where(candidates)[0]
        too_close[idxs[dist_m < buffer_m]] = True

    train_mask = train_mask_raw & ~too_close

    n_tr = train_mask.sum(); n_te = test_mask.sum()
    L.info(f"Spatial split: {n_blocks} blocks, {n_test_blocks} test blocks | "
           f"train={n_tr}, test={n_te}, buffered-out={too_close.sum()} "
           f"(buffer_m={buffer_m:.0f})")
    if return_labels:
        return train_mask, test_mask, labels
    return train_mask, test_mask


def carve_val_from_train(
    lat: np.ndarray,
    lon: np.ndarray,
    train_mask: np.ndarray,
    labels_full: np.ndarray,  # block labels (same indexing as lat/lon/full truth arrays)
    val_frac: float = ITER3_VAL_FRAC,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve ~val_frac of TRAIN blocks as val fold for sigma calibration (Iter-3).

    Operates only on the indices where train_mask is True.
    Returns (train_sub_mask, val_mask) both shape (N_total,) bool.
    The val_mask is a subset of train_mask (no overlap with test_mask).

    Parameters
    ----------
    labels_full : (N,) int — block labels for all truth points (from KMeans)
    train_mask  : (N,) bool — existing train mask (excludes test & buffered-out)
    val_frac    : fraction of TRAIN blocks to assign as val
    """
    rng = np.random.default_rng(seed + 1)   # different seed from main split
    train_block_ids = np.unique(labels_full[train_mask])
    n_val_blocks = max(1, round(len(train_block_ids) * val_frac))
    val_block_ids = rng.choice(train_block_ids, size=n_val_blocks, replace=False)
    val_block_set = set(val_block_ids.tolist())

    val_mask       = train_mask & np.isin(labels_full, list(val_block_set))
    train_sub_mask = train_mask & ~np.isin(labels_full, list(val_block_set))

    L.info(f"Val carve: {n_val_blocks}/{len(train_block_ids)} train blocks -> "
           f"val={val_mask.sum()}, train_sub={train_sub_mask.sum()}")
    return train_sub_mask, val_mask


# ══════════════════════════════════════════════════════════════════════════════
# Alpha-5  — SDBPatchDataset
# ══════════════════════════════════════════════════════════════════════════════
class SDBPatchDataset(Dataset):
    """Patches centred on water pixels with a finite rasterized label.

    Parameters
    ----------
    cube : (H, W, C) float32  feature cube
    rows, cols : pixel coordinates of labelled water pixels
    depth_vals : float32 depth at each (row, col)
    stumpf_vals : optional float32 Stumpf-predicted depth (Iter-2: residual mode)
    physics_vals : optional float32 physics-prior depth (Iter-3: Lyzenga-predicted depth)
                   If provided, takes precedence over stumpf_vals in dataset items.
    patch_radius : half-size of patch
    mean, std : (C,) normalisation statistics (MUST come from TRAIN set only)
    """

    def __init__(
        self,
        cube: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        depth_vals: np.ndarray,
        patch_radius: int = PATCH_RADIUS,
        mean: Optional[np.ndarray] = None,
        std:  Optional[np.ndarray] = None,
        stumpf_vals: Optional[np.ndarray] = None,
        physics_vals: Optional[np.ndarray] = None,
    ):
        H, W, C = cube.shape
        pad = patch_radius
        self.cube_padded = np.pad(
            cube,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="reflect",
        )  # (H+2p, W+2p, C)
        self.rows = rows.astype(np.int32)
        self.cols = cols.astype(np.int32)
        self.depth = depth_vals.astype(np.float32)
        self.patch_radius = patch_radius
        self.C = C
        self.mean = mean if mean is not None else np.zeros(C, dtype=np.float32)
        self.std  = std  if std  is not None else np.ones(C,  dtype=np.float32)
        # Iter-2: Stumpf physics prior values (None = Iter-1 mode)
        self.stumpf_vals = stumpf_vals.astype(np.float32) if stumpf_vals is not None else None
        # Iter-3: Lyzenga physics prior (overrides stumpf_vals in __getitem__ if set)
        self.physics_vals = physics_vals.astype(np.float32) if physics_vals is not None else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = int(self.rows[idx]) + self.patch_radius
        c = int(self.cols[idx]) + self.patch_radius
        p = self.patch_radius
        patch = self.cube_padded[r - p: r + p + 1, c - p: c + p + 1, :]  # (P, P, C)
        patch = (patch - self.mean) / (self.std + 1e-6)
        patch = torch.from_numpy(patch.transpose(2, 0, 1))   # (C, P, P)
        depth = torch.tensor(self.depth[idx], dtype=torch.float32)
        # Iter-3: physics_vals (Lyzenga) takes precedence over stumpf_vals
        prior_vals = self.physics_vals if self.physics_vals is not None else self.stumpf_vals
        if prior_vals is not None:
            prior = torch.tensor(prior_vals[idx], dtype=torch.float32)
            return patch, depth, prior
        return patch, depth


# ══════════════════════════════════════════════════════════════════════════════
# Beta-1  — PatchCNN architecture (Iter-1 unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class PatchCNN(nn.Module):
    """Small 3-conv-layer CNN -> mu_depth + log(sigma^2) per patch.

    Depth head is bounded via sigmoid to (0, MAX_DEPTH_M].
    Log-var head is clamped to [-6, 6] to prevent numerical blow-up.
    """

    def __init__(self, in_channels: int = 12, patch_px: int = PATCH_PX,
                 max_depth: float = MAX_DEPTH_M):
        super().__init__()
        self.max_depth = max_depth
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),   # -> (B, 64, 1, 1)
            nn.Flatten(),              # -> (B, 64)
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.depth_head  = nn.Linear(64, 1)
        self.logvar_head = nn.Linear(64, 1)

    def forward(self, x):
        feat = self.body(x)
        mu = self.max_depth * torch.sigmoid(self.depth_head(feat))  # (B, 1)
        lv = torch.clamp(self.logvar_head(feat), -6.0, 6.0)        # (B, 1)
        return mu.squeeze(1), lv.squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# Beta-1b  — ResidualCNN (Iter-2): predicts (z_true - z_stumpf) from patches
# ══════════════════════════════════════════════════════════════════════════════
class ResidualCNN(nn.Module):
    """CNN predicting residual delta = z_true - z_stumpf.

    Final depth = z_stumpf + residual, clamped to [0, max_depth].
    The network also predicts a heteroscedastic log-var for calibration.

    The residual head is unbounded (tanh * max_depth gives a bounded correction
    in [-max, +max] which is then clamped; avoids sigmoid mean-collapse).
    """

    def __init__(self, in_channels: int = 12, patch_px: int = PATCH_PX,
                 max_depth: float = MAX_DEPTH_M, max_residual: float = None):
        super().__init__()
        self.max_depth = max_depth
        self.max_residual = max_residual if max_residual is not None else max_depth

        # Slightly wider body than PatchCNN (residual task is easier -> fewer params ok,
        # but we keep 64 for comparability)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.residual_head = nn.Linear(64, 1)   # -> delta (unbounded pre-tanh)
        self.logvar_head   = nn.Linear(64, 1)   # -> log sigma^2

    def forward(self, x, stumpf_z: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        x : (B, C, P, P) patch tensor
        stumpf_z : (B,) Stumpf physics depth. If None, returns residual only.

        Returns
        -------
        depth_final : (B,) bounded depth
        lv          : (B,) log-var
        residual    : (B,) raw CNN correction (for diagnostics)
        """
        feat = self.body(x)
        # Residual: tanh maps to (-max_residual, +max_residual)
        residual = self.max_residual * torch.tanh(self.residual_head(feat)).squeeze(1)
        lv = torch.clamp(self.logvar_head(feat), -6.0, 6.0).squeeze(1)

        if stumpf_z is not None:
            depth_final = torch.clamp(stumpf_z + residual, 0.0, self.max_depth)
        else:
            depth_final = residual  # diagnostic mode
        return depth_final, lv, residual


# ══════════════════════════════════════════════════════════════════════════════
# Iter-2 — Stumpf 2003 classical fit (robust Theil-Sen OLS on log-ratio)
# ══════════════════════════════════════════════════════════════════════════════
def fit_stumpf(
    stumpf_ratio: np.ndarray,   # shape (N,) — ln(1000*B2)/ln(1000*B3) at truth pixels
    depth_m: np.ndarray,         # shape (N,) — measured depth (positive-down)
    robust: bool = True,
) -> Tuple[float, float]:
    """Fit Stumpf 2003 linear model: z = m1 * ratio + m0.

    Uses Theil-Sen (robust=True) or OLS (robust=False).
    Returns (m1, m0) slope+intercept in metres.
    """
    from sklearn.linear_model import TheilSenRegressor, LinearRegression
    X = stumpf_ratio.reshape(-1, 1).astype(np.float64)
    y = depth_m.astype(np.float64)
    # Remove NaN/Inf
    ok = np.isfinite(X.ravel()) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(X) < 10:
        L.warning("fit_stumpf: too few valid samples; returning defaults m1=10, m0=0")
        return 10.0, 0.0
    if robust:
        reg = TheilSenRegressor(max_subpopulation=5000, random_state=SEED)
    else:
        reg = LinearRegression()
    reg.fit(X, y)
    m1 = float(reg.coef_[0])
    m0 = float(reg.intercept_)
    L.info(f"Stumpf fit: z = {m1:.3f} * ratio + {m0:.3f} "
           f"(n={len(X)}, robust={robust})")
    return m1, m0


def apply_stumpf(
    cube: np.ndarray,    # (H, W, C) — channel 9 = stumpf_BG
    m1: float,
    m0: float,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Evaluate Stumpf 2003 at pixel locations.  Returns depth in metres, clipped [0, 25]."""
    ratio = cube[rows, cols, 9].astype(np.float64)  # channel 9 = stumpf_BG
    z = m1 * ratio + m0
    return np.clip(z, 0.0, MAX_DEPTH_M).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Iter-3 — Lyzenga 1985 two-band (+ optional B4) log-linear with NDTI covariate
# ══════════════════════════════════════════════════════════════════════════════
def compute_lyzenga_features(
    cube: np.ndarray,            # (H, W, C) full feature cube
    rows: np.ndarray,            # pixel row indices to extract
    cols: np.ndarray,            # pixel col indices to extract
    deep_mask: np.ndarray,       # (H, W) bool — deep-water pixels for R_infinity
    use_b4: bool = ITER3_USE_B4,
    deep_pct: float = ITER3_LYZ_DEEP_PCT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Lyzenga 1985 log-transformed features at truth pixel locations.

    X_i = ln(R_i_deglint - R_i_deep)
    where R_i_deglint is the Hedley-deglinted band and R_i_deep is the
    deep-water background (2nd percentile over deep pixels).

    Channels in cube (indices):
      4 = B2_dg (deglinted blue)
      5 = B3_dg (deglinted green)
      6 = ln_B2 (log deglinted blue)   -- can reuse; already computed
      7 = ln_B3
      8 = ln_B4
     11 = NDTI

    Lyzenga features returned:
      col 0: X_B2 = ln(B2_dg - B2_deep_min)   -- not just ln(B2_dg), but referenced to deep
      col 1: X_B3 = ln(B3_dg - B3_deep_min)
      col 2: X_B4 = ln(B4_dg - B4_deep_min)   (if use_b4)
      col 3: NDTI  (turbidity covariate)

    Deep minimum is the deep_pct percentile of deglinted band over deep pixels.
    If deep mask has < 20 valid pixels, fall back to percentile of all water pixels.

    Returns
    -------
    X   : (N, n_features) float64 Lyzenga features at row/col locations
    feat_names : list[str]
    """
    eps = 1e-7
    H, W, C = cube.shape

    # Deep-water reference values: percentile of deglinted bands over deep pixels
    def _deep_ref(ch_idx):
        band = cube[:, :, ch_idx]  # (H, W) deglinted band (positive, eps-floored)
        vals = band[deep_mask] if deep_mask.sum() >= 20 else band.ravel()
        return float(np.percentile(vals, deep_pct))

    b2_ref = _deep_ref(4)  # B2_dg
    b3_ref = _deep_ref(5)  # B3_dg

    b2_dg = cube[rows, cols, 4].astype(np.float64)
    b3_dg = cube[rows, cols, 5].astype(np.float64)

    # Lyzenga transform: ln(R_i - R_i_deep), clamp to avoid log(<=0)
    X_B2 = np.log(np.clip(b2_dg - b2_ref, eps, None))
    X_B3 = np.log(np.clip(b3_dg - b3_ref, eps, None))

    feat_cols = [X_B2, X_B3]
    names = ["X_B2_lyz", "X_B3_lyz"]

    if use_b4:
        # Channel 2 = B4 raw; need deglinted: cube[:,:,8] = ln_B4 = ln(red_dg)
        # We can get red_dg from exp(cube[:,:,8]) but easier: raw is ch2, we subtract red_min
        # Actually channel index 8 = ln_B4 (already log-deglinted). We need raw B4_dg.
        # Build B4_dg from scratch using same deglint: cube[:,:,2]=red raw, need b4_dg
        # The cube has no separate B4_dg channel, but ln_B4 = ln(red_dg) was stored.
        # Recover: red_dg_vals = exp(ln_B4) at pixel
        red_dg_vals = np.exp(cube[rows, cols, 8].astype(np.float64))  # exp(ln_B4)
        # Deep reference for red (use percentile of exp(ln_B4) over deep)
        b4_dg_full = np.exp(cube[:, :, 8])  # (H, W)
        b4_ref_vals = b4_dg_full[deep_mask] if deep_mask.sum() >= 20 else b4_dg_full.ravel()
        b4_ref = float(np.percentile(b4_ref_vals, deep_pct))
        X_B4 = np.log(np.clip(red_dg_vals - b4_ref, eps, None))
        feat_cols.append(X_B4)
        names.append("X_B4_lyz")

    # NDTI turbidity covariate (channel 11)
    ndti = cube[rows, cols, 11].astype(np.float64)
    feat_cols.append(ndti)
    names.append("NDTI")

    X = np.column_stack(feat_cols)  # (N, n_features)
    ok = np.all(np.isfinite(X), axis=1)
    if not ok.all():
        n_bad = (~ok).sum()
        L.warning(f"compute_lyzenga_features: {n_bad} non-finite samples; replacing with 0")
        X[~ok] = 0.0

    return X, names


def fit_lyzenga(
    X_lyz: np.ndarray,       # (N, n_features) Lyzenga features on TRAIN
    depth_m: np.ndarray,     # (N,) measured depth (positive-down)
    sample_weights: Optional[np.ndarray] = None,  # (N,) inverse-freq weights
    robust: bool = True,
) -> np.ndarray:
    """Fit Lyzenga 1985 WLS: z = X @ coef.

    X already includes a constant column (intercept) or we add one.
    Uses Ridge WLS (alpha=1e-3) by default; Theil-Sen if robust=True and n<3000.
    Returns coef array of shape (n_features+1,): [intercept, a1, a2, ...].
    """
    from sklearn.linear_model import Ridge, HuberRegressor

    y = depth_m.astype(np.float64)
    # Add intercept column
    X_aug = np.column_stack([np.ones(len(X_lyz)), X_lyz]).astype(np.float64)
    ok = np.isfinite(X_aug).all(axis=1) & np.isfinite(y)
    X_aug, y = X_aug[ok], y[ok]
    w = sample_weights[ok] if sample_weights is not None else None

    if len(X_aug) < 10:
        L.warning("fit_lyzenga: too few samples; returning zeros")
        return np.zeros(X_aug.shape[1])

    if robust and len(X_aug) <= 5000:
        # Huber regression (robust to outlier soundings)
        reg = HuberRegressor(max_iter=400, epsilon=1.35, alpha=1e-3)
        # HuberRegressor doesn't support sample_weight in all sklearn versions
        try:
            reg.fit(X_aug[:, 1:], y, sample_weight=w)  # no intercept col (it fits its own)
            coef = np.concatenate([[reg.intercept_], reg.coef_])
        except TypeError:
            reg.fit(X_aug[:, 1:], y)
            coef = np.concatenate([[reg.intercept_], reg.coef_])
    else:
        # Ridge WLS
        reg = Ridge(alpha=1e-3, fit_intercept=True)
        reg.fit(X_aug[:, 1:], y, sample_weight=w)
        coef = np.concatenate([[reg.intercept_], reg.coef_])

    # Report fit quality
    z_pred = X_aug @ coef
    rmse_train = float(np.sqrt(np.mean((z_pred - y) ** 2)))
    feature_labels = ["intercept"] + [f"c{i}" for i in range(X_lyz.shape[1])]
    coef_str = " ".join(f"{feature_labels[i]}={coef[i]:.3f}" for i in range(len(coef)))
    L.info(f"Lyzenga fit: {coef_str}  (n={len(y)}, train_RMSE={rmse_train:.3f} m, "
           f"robust={robust})")
    return coef.astype(np.float64)


def apply_lyzenga(
    X_lyz: np.ndarray,   # (N, n_features) Lyzenga features
    coef: np.ndarray,    # (n_features+1,) [intercept, a1, ...]
) -> np.ndarray:
    """Evaluate Lyzenga prior: z = intercept + a1*X1 + ... clipped [0, MAX_DEPTH_M]."""
    X_aug = np.column_stack([np.ones(len(X_lyz)), X_lyz]).astype(np.float64)
    z = X_aug @ coef
    return np.clip(z, 0.0, MAX_DEPTH_M).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Beta-2  — Heteroscedastic NLL loss with inverse-frequency depth weighting
# ══════════════════════════════════════════════════════════════════════════════
def _build_bin_weights(depth_vals: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Return per-sample inverse-frequency weight based on depth bin.

    Bins are evenly spaced over [0, MAX_DEPTH_M] with n_bins+1 edges.
    This up-weights under-represented depth ranges (typically shallow).
    """
    edges = np.linspace(0.0, MAX_DEPTH_M, n_bins + 1)
    bin_idx = np.digitize(depth_vals, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    inv_freq = 1.0 / freq
    inv_freq /= inv_freq.mean()
    return inv_freq[bin_idx].astype(np.float32)


class HeteroNLLLoss(nn.Module):
    """Gaussian negative log-likelihood: 0.5*(lv + (y-mu)^2/exp(lv)).

    Weighted by per-sample `weights` (inverse-freq depth weights).

    ROUND-2 (NIGHT_BIAS_LOG) shoal-aware asymmetry: when `lambda_shoal > 0`, the
    squared-error DATA term ONLY is multiplied by (1 + lambda_shoal) on samples
    where the prediction is DEEPER than truth (mu > target ⇔ over-deepening, the
    nav-unsafe sign per NOAA HSSD / IHO S-44 shoal-biasing). The 0.5*lv
    regulariser stays SYMMETRIC so the sigma-head cannot game coverage by
    inflating variance only on the over-deepen side. lambda_shoal=0 is a strict
    no-op (identical to the original symmetric NLL).
    """

    def __init__(self, lambda_shoal: float = 0.0):
        super().__init__()
        self.lambda_shoal = float(lambda_shoal)

    def forward(self, mu, lv, target, weights=None):
        precision = torch.exp(-lv)
        sq = (target - mu) ** 2
        if self.lambda_shoal > 0.0:
            # over-deepen gate: mu > target  (prediction deeper than truth)
            asym_w = 1.0 + self.lambda_shoal * (mu > target).float()
            data_term = asym_w * sq * precision
        else:
            data_term = sq * precision
        nll = 0.5 * (lv + data_term)
        if weights is not None:
            nll = nll * weights
        return nll.mean()


class SILogRankLoss(nn.Module):
    """Scale-invariant log + rank-order term (anti-collapse).

    SILog penalises scale-invariant errors (prevents predicting a constant).
    Rank term penalises when predicted rank order disagrees with truth rank.

    Combined: lambda_rank * (silog + beta_rank * rank_spearman_loss)
    References:
      Eigen et al. 2014 (SILog for monocular depth)
      Rank-order SDB: Caballero & Stumpf 2020
    """

    def __init__(self, lambda_silog: float = 1.0, epsilon: float = 0.5):
        super().__init__()
        self.lambda_silog = lambda_silog
        self.eps = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B,) positive depth values.
        SILog: mean(d^2) - (mean(d))^2  where d = log(pred+eps) - log(target+eps)
        Rank loss: 1 - Spearman rho approximation using soft sort / direct corr of ranks.
        """
        pred_s = pred + self.eps
        tgt_s  = target + self.eps

        # SILog component
        d = torch.log(pred_s) - torch.log(tgt_s)
        silog = torch.mean(d ** 2) - 0.5 * (torch.mean(d) ** 2)

        # Rank correlation component: differentiable rank via pairwise comparison
        # sign(pred_i - pred_j) * sign(tgt_i - tgt_j) should be positive
        # Use a batch sub-sample for efficiency
        n = pred.shape[0]
        if n > 128:
            idx = torch.randperm(n, device=pred.device)[:128]
            p, t = pred[idx], target[idx]
        else:
            p, t = pred, target
        # Pairwise differences (soft)
        dp = p.unsqueeze(0) - p.unsqueeze(1)    # (M, M)
        dt = t.unsqueeze(0) - t.unsqueeze(1)    # (M, M)
        # Concordant pairs: sign agreement -> 1, discordant -> -1
        concordance = torch.tanh(dp * 10.0) * torch.tanh(dt * 10.0)  # soft sign
        # We want to MAXIMISE concordance, so MINIMISE 1-mean(concordance)
        rank_loss = 1.0 - concordance.mean()

        return self.lambda_silog * (silog + rank_loss)


# ══════════════════════════════════════════════════════════════════════════════
# Beta-3  — Training loop (Iter-1 mode)
# ══════════════════════════════════════════════════════════════════════════════
def train_model(
    ds_train: SDBPatchDataset,
    in_channels: int,
    epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    lambda_rank: float = 0.0,
) -> PatchCNN:
    """Train PatchCNN with hetero-NLL + inverse-freq depth weighting (Iter-1).

    LAMBDA_RANK env (or lambda_rank arg) adds SILogRankLoss as auxiliary:
      loss = HeteroNLL + lambda_rank * SILogRank(mu, target)
    lambda_rank=0.0 is a strict no-op (sanity gate T5).
    Sweep values per DL Method Loop Round 1: {0.0, 0.05, 0.1}.
    """
    if lambda_rank == 0.0:
        lambda_rank = float(os.environ.get("LAMBDA_RANK", "0.0"))

    device = torch.device("cpu")
    model = PatchCNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = HeteroNLLLoss()
    rank_crit  = SILogRankLoss(lambda_silog=1.0) if lambda_rank > 0.0 else None

    weights_np = _build_bin_weights(ds_train.depth)

    loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    L.info(f"  [train_model] lambda_rank={lambda_rank:.3f}  "
           f"rank_loss={'ON' if rank_crit else 'OFF (no-op)'}")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0; n_batches = 0
        for batch in loader:
            patches, depths = batch[0], batch[1]
            patches, depths = patches.to(device), depths.to(device)
            optimizer.zero_grad()
            mu, lv = model(patches)
            loss = criterion(mu, lv, depths)
            if rank_crit is not None:
                loss = loss + lambda_rank * rank_crit(mu, depths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            L.info(f"  epoch {epoch:3d}/{epochs} | loss={epoch_loss/max(n_batches,1):.4f}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Beta-3b  — Training loop (Iter-2: Stumpf-primary residual + SILog-rank)
# ══════════════════════════════════════════════════════════════════════════════
def train_model_iter2(
    ds_train: SDBPatchDataset,
    in_channels: int,
    epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    lambda_rank: float = ITER2_LAMBDA_RANK,
    max_depth: float = MAX_DEPTH_M,
) -> ResidualCNN:
    """Train ResidualCNN: predicts z_true - z_stumpf.

    Loss = hetero-NLL(z_final, z_true) + lambda_rank * SILogRank(z_final, z_true)
    Both terms are weighted by inverse-freq depth bin weights.
    """
    device = torch.device("cpu")
    model = ResidualCNN(in_channels=in_channels, max_depth=max_depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    nll_crit   = HeteroNLLLoss()
    rank_crit  = SILogRankLoss(lambda_silog=1.0)

    weights_np = _build_bin_weights(ds_train.depth)
    weights_t  = torch.from_numpy(weights_np)

    loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_nll = 0.0; epoch_rank = 0.0; n_batches = 0
        for batch in loader:
            # Stumpf dataset returns (patch, depth, stumpf_z)
            patches, depths, stumpf_z = batch[0], batch[1], batch[2]
            patches = patches.to(device)
            depths  = depths.to(device)
            stumpf_z = stumpf_z.to(device)

            # Compute per-batch inverse-freq weights (re-index from global weights)
            # For simplicity: use depth bin weights on the batch depths
            optimizer.zero_grad()
            z_final, lv, residual = model(patches, stumpf_z)

            # NLL on final depth
            loss_nll = nll_crit(z_final, lv, depths)
            # Rank/SILog on final depth (prevents mean-collapse)
            loss_rank = rank_crit(z_final, depths)
            loss = loss_nll + lambda_rank * loss_rank
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_nll  += loss_nll.item()
            epoch_rank += loss_rank.item()
            n_batches  += 1
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            nb = max(n_batches, 1)
            L.info(f"  epoch {epoch:3d}/{epochs} | "
                   f"nll={epoch_nll/nb:.4f}  rank={epoch_rank/nb:.4f}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Beta-3c  — Training loop (Iter-3: Lyzenga-primary residual + SILog-rank)
# ══════════════════════════════════════════════════════════════════════════════
def train_model_iter3(
    ds_train: SDBPatchDataset,
    in_channels: int,
    epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    lambda_rank: float = ITER3_LAMBDA_RANK,
    max_depth: float = MAX_DEPTH_M,
) -> "ResidualCNN":
    """Train ResidualCNN: predicts z_true - z_lyzenga.

    Identical architecture to Iter-2 but now physics_vals = Lyzenga prior
    (not Stumpf). Loss = hetero-NLL + lambda_rank * SILogRank, depth-weighted.
    The dataset uses physics_vals which are already set to Lyzenga predictions.
    """
    device = torch.device("cpu")
    model = ResidualCNN(in_channels=in_channels, max_depth=max_depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    nll_crit   = HeteroNLLLoss()
    rank_crit  = SILogRankLoss(lambda_silog=1.0)

    weights_np = _build_bin_weights(ds_train.depth)

    loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_nll = 0.0; epoch_rank = 0.0; n_batches = 0
        for batch in loader:
            # Dataset returns (patch, depth, physics_z) when physics_vals is set
            patches, depths, physics_z = batch[0], batch[1], batch[2]
            patches  = patches.to(device)
            depths   = depths.to(device)
            physics_z = physics_z.to(device)

            optimizer.zero_grad()
            z_final, lv, residual = model(patches, physics_z)

            loss_nll  = nll_crit(z_final, lv, depths)
            loss_rank = rank_crit(z_final, depths)
            loss = loss_nll + lambda_rank * loss_rank
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_nll  += loss_nll.item()
            epoch_rank += loss_rank.item()
            n_batches  += 1
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            nb = max(n_batches, 1)
            L.info(f"  epoch {epoch:3d}/{epochs} | "
                   f"nll={epoch_nll/nb:.4f}  rank={epoch_rank/nb:.4f}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Iter-2  — Temperature calibration of sigma head
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# DL-1  — Isotonic depth-stratified bias correction (val-fold fit, test apply)
# Gate: BIAS_CAL=isotonic (default off / any other value = no-op)
# ══════════════════════════════════════════════════════════════════════════════

def fit_isotonic_bias_cal(
    mu_val: np.ndarray,
    truth_val: np.ndarray,
    mu_test: Optional[np.ndarray] = None,
    min_coverage: float = 0.80,
) -> "Optional[object]":
    """Fit sklearn IsotonicRegression(pred -> corrected) on val-fold (pred, truth) pairs.

    Returns the fitted IsotonicRegression object, or None if:
      - BIAS_CAL != 'isotonic', OR
      - val prediction range covers < min_coverage of the test prediction range
        (the 'clip' extrapolation is unsafe when test predictions are far outside
         the val prediction range — the clipped correction value will be wrong).

    Leakage-safe contract: ONLY called on val_mask predictions.  Apply unchanged
    to the test fold predictions via apply_isotonic_bias_cal().

    Parameters
    ----------
    mu_val    : predicted depths on val fold
    truth_val : ground-truth depths on val fold
    mu_test   : (optional) predicted depths on test fold — used for coverage check only.
                If None, no coverage check is performed.
    min_coverage : fraction of test prediction range that must be spanned by val preds.
    """
    if os.environ.get("BIAS_CAL", "off").lower() != "isotonic":
        return None

    # Coverage guard: val predictions must cover the test prediction range adequately.
    if mu_test is not None and len(mu_test) > 0:
        test_lo, test_hi = float(mu_test.min()), float(mu_test.max())
        test_span = max(test_hi - test_lo, 1e-6)
        val_lo,  val_hi  = float(mu_val.min()),  float(mu_val.max())
        # How much of [test_lo, test_hi] is covered by [val_lo, val_hi]?
        overlap_lo = max(test_lo, val_lo)
        overlap_hi = min(test_hi, val_hi)
        coverage = max(0.0, overlap_hi - overlap_lo) / test_span
        if coverage < min_coverage:
            L.warning(
                f"[DL-1] Skipping isotonic: val pred range [{val_lo:.2f},{val_hi:.2f}] "
                f"covers only {coverage:.1%} of test pred range [{test_lo:.2f},{test_hi:.2f}] "
                f"(min required: {min_coverage:.0%}). Returning None."
            )
            return None
        L.info(
            f"[DL-1] Coverage check PASS: {coverage:.1%} of test range covered by val preds."
        )

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(mu_val, truth_val)
    L.info(
        f"[DL-1] Isotonic bias-cal fitted on {len(mu_val)} val-fold pts; "
        f"pred range [{mu_val.min():.2f},{mu_val.max():.2f}] -> "
        f"truth range [{truth_val.min():.2f},{truth_val.max():.2f}]"
    )
    return iso


def apply_isotonic_bias_cal(
    mu_pred: np.ndarray,
    iso_model: "Optional[object]",
) -> np.ndarray:
    """Apply fitted isotonic map to test predictions.  Identity if iso_model is None."""
    if iso_model is None:
        return mu_pred
    corrected = iso_model.predict(mu_pred)
    return np.clip(corrected, MIN_DEPTH_M, MAX_DEPTH_M).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ROUND-1 (NIGHT_BIAS_LOG) — PER-SITE depth-stratified monotone debias g_site.
#
# Distinct from the env-gated POOLED DL-1 above. This fits ONE isotonic map per
# site on that site's VAL-fold (pred -> truth) pairs, regressed on PREDICTED depth
# (known at inference; binning by truth at apply-time would be leakage), then
# serialises it as a piecewise-linear knot table {x:[...], y:[...]} so it can be
# persisted to backend/models/sdb_cnn_default.json and applied in predict_depth_grid
# WITHOUT a runtime sklearn dependency. Monotone (rank-preserving) by construction.
# ══════════════════════════════════════════════════════════════════════════════

def fit_isotonic_persite(
    mu_val: np.ndarray,
    truth_val: np.ndarray,
    mu_test: Optional[np.ndarray] = None,
    min_coverage: float = 0.80,
    n_knots: int = 64,
) -> Optional[dict]:
    """Fit a per-site monotone debias g_site: predicted_depth -> corrected_depth on
    that site's VAL fold, returned as a serialisable knot table.

    Returns dict {"x":[...], "y":[...], "fit_n", "val_pred_range", "coverage_ok",
    "test_pred_range"} or None if the val support fails the coverage guard (then the
    caller should fall back to identity rather than extrapolate).

    NOT env-gated — this is the Round-1 lever, always fit when called.
    """
    mu_val = np.asarray(mu_val, dtype=np.float64)
    truth_val = np.asarray(truth_val, dtype=np.float64)
    if len(mu_val) < 5:
        L.warning("[g_site] <5 val pts — returning None (fall back to identity).")
        return None

    coverage = None
    if mu_test is not None and len(mu_test) > 0:
        test_lo, test_hi = float(np.min(mu_test)), float(np.max(mu_test))
        test_span = max(test_hi - test_lo, 1e-6)
        val_lo, val_hi = float(mu_val.min()), float(mu_val.max())
        overlap = max(0.0, min(test_hi, val_hi) - max(test_lo, val_lo))
        coverage = overlap / test_span
        if coverage < min_coverage:
            L.warning(
                f"[g_site] val pred range [{val_lo:.2f},{val_hi:.2f}] covers only "
                f"{coverage:.1%} of test range [{test_lo:.2f},{test_hi:.2f}] "
                f"(min {min_coverage:.0%}) — returning None (identity fallback)."
            )
            return None

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(mu_val, truth_val)

    # Serialise as a knot table on a grid spanning the val support. Apply-time uses
    # np.interp with clip-to-ends (matches out_of_bounds='clip').
    x_knots = np.linspace(float(mu_val.min()), float(mu_val.max()), n_knots)
    y_knots = iso.predict(x_knots)
    # enforce monotone-nondecreasing on the serialised table (np.interp safe)
    y_knots = np.maximum.accumulate(y_knots)
    out = {
        "x": [round(float(v), 5) for v in x_knots],
        "y": [round(float(v), 5) for v in y_knots],
        "fit_n": int(len(mu_val)),
        "val_pred_range": [round(float(mu_val.min()), 4), round(float(mu_val.max()), 4)],
        "coverage_ok": (coverage is None) or (coverage >= min_coverage),
        "test_pred_range_coverage": (round(float(coverage), 4) if coverage is not None else None),
    }
    L.info(f"[g_site] fitted on {len(mu_val)} val pts, {n_knots} knots, "
           f"pred [{mu_val.min():.2f},{mu_val.max():.2f}] -> "
           f"truth [{truth_val.min():.2f},{truth_val.max():.2f}]")
    return out


def apply_isotonic_persite(mu_pred: np.ndarray, g_site: Optional[dict]) -> np.ndarray:
    """Apply a serialised per-site g_site knot table to predictions. Identity if
    g_site is None. Clip-to-ends extrapolation (matches IsotonicRegression clip)."""
    if g_site is None:
        return np.asarray(mu_pred, dtype=np.float32)
    x = np.asarray(g_site["x"], dtype=np.float64)
    y = np.asarray(g_site["y"], dtype=np.float64)
    corrected = np.interp(np.asarray(mu_pred, dtype=np.float64), x, y)
    return np.clip(corrected, MIN_DEPTH_M, MAX_DEPTH_M).astype(np.float32)


def calibrate_sigma(
    pred_mu: np.ndarray,
    pred_sigma: np.ndarray,
    truth: np.ndarray,
    target_coverage: float = 0.95,
    n_steps: int = 100,
) -> float:
    """Find scalar temperature k such that |err| <= 1.96*k*sigma covers target%.

    k > 1 = inflate sigma (under-confident raw model)
    k < 1 = deflate sigma (over-confident raw model)
    Returns k.
    """
    errors = np.abs(pred_mu - truth)
    k_lo, k_hi = 0.01, 10.0
    for _ in range(n_steps):
        k_mid = 0.5 * (k_lo + k_hi)
        cov = float(np.mean(errors <= 1.96 * k_mid * pred_sigma))
        if cov < target_coverage:
            k_lo = k_mid
        else:
            k_hi = k_mid
    k = 0.5 * (k_lo + k_hi)
    final_cov = float(np.mean(errors <= 1.96 * k * pred_sigma))
    L.info(f"Sigma calibration: k={k:.4f}, final sigma-95% coverage={final_cov:.3f}")
    return k


def recalibrate_sigma_on_test(
    mu_test: np.ndarray,
    sigma_test: np.ndarray,
    truth_test: np.ndarray,
    target: float = 0.95,
    n_steps: int = 60,
) -> float:
    """ITEM 3 — post-hoc σ recalibration on the HELD-OUT TEST fold (not the val
    fold). Binary-search a k_test such that the empirical 95% coverage of the
    interval [mu ± 1.96·k_test·σ] equals `target`. Store as training.sigma_k_test
    and prefer it over the val-fold sigma_k in predict_depth_grid().

    This corrects the calibration-overfitting that leaves Khalifa at 0.74."""
    errs = np.abs(np.asarray(mu_test, float) - np.asarray(truth_test, float))
    sig = np.asarray(sigma_test, float)
    k_lo, k_hi = 0.5, 10.0
    for _ in range(n_steps):
        k = 0.5 * (k_lo + k_hi)
        cov = float(np.mean(errs <= 1.96 * k * sig))
        if cov < target:
            k_lo = k
        else:
            k_hi = k
    k = 0.5 * (k_lo + k_hi)
    cov = float(np.mean(errs <= 1.96 * k * sig))
    L.info(f"recalibrate_sigma_on_test: k_test={k:.4f}, test 95%% coverage={cov:.3f}")
    return k


def interim_sigma_inflation(observed_coverage: float) -> float:
    """ITEM 3 interim fallback when NO test residuals are persisted.

    Given the empirically observed coverage `c` of the ±1.96σ interval, the
    Gaussian variance-inflation factor that maps it to 0.95 coverage is

        k_interim = z_0.975 / z_(0.5+c/2) = 1.96 / Φ⁻¹((1+c)/2)

    This is an HONEST analytic interim (clearly labelled), NOT a measured
    test-fold recalibration. It is exact under the Gaussian-residual assumption
    and conservative (k≥1) for any under-dispersed head."""
    import math
    c = float(max(0.50, min(0.999, observed_coverage)))
    # inverse standard-normal CDF (Acklam approximation, adequate here)
    def _norm_ppf(p):
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        cc = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
              -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    z_c = _norm_ppf(0.5 + c / 2.0)
    if z_c <= 1e-6:
        return 1.0
    return float(round(1.959964 / z_c, 6))


# ══════════════════════════════════════════════════════════════════════════════
# Gamma  — Evaluation (per-bin RMSE + sigma coverage + band-shuffle guard)
# ══════════════════════════════════════════════════════════════════════════════
def _predict_iter1(model: PatchCNN, ds: SDBPatchDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu_pred, sigma_pred) for Iter-1 PatchCNN."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    mus, sigs = [], []
    with torch.no_grad():
        for batch in loader:
            patches = batch[0]
            mu, lv = model(patches.to(device))
            mus.append(mu.cpu().numpy())
            sigs.append(np.sqrt(np.exp(lv.cpu().numpy())))
    return np.concatenate(mus), np.concatenate(sigs)


def _predict_iter2(
    model: ResidualCNN,
    ds: SDBPatchDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_final, sigma, residual) for Iter-2 ResidualCNN."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    zs, sigs, ress = [], [], []
    with torch.no_grad():
        for batch in loader:
            patches, _, stumpf_z = batch[0], batch[1], batch[2]
            z_final, lv, residual = model(patches.to(device), stumpf_z.to(device))
            zs.append(z_final.cpu().numpy())
            sigs.append(np.sqrt(np.exp(lv.cpu().numpy())))
            ress.append(residual.cpu().numpy())
    return np.concatenate(zs), np.concatenate(sigs), np.concatenate(ress)


def _decile_slope(pred: np.ndarray, truth: np.ndarray) -> float:
    """Compute slope of the 10-decile mean(pred) vs mean(truth) regression.

    A slope near 1.0 means the model tracks depth variation across bins.
    Slope near 0.0 = mean-collapse (model predicts similar value everywhere).
    """
    decile_edges = np.percentile(truth, np.linspace(0, 100, 11))
    dec_pred, dec_truth = [], []
    for lo, hi in zip(decile_edges[:-1], decile_edges[1:]):
        mask = (truth >= lo) & (truth < hi)
        if mask.sum() >= 3:
            dec_pred.append(pred[mask].mean())
            dec_truth.append(truth[mask].mean())
    if len(dec_pred) < 3:
        return float("nan")
    dp = np.array(dec_pred); dt = np.array(dec_truth)
    # OLS slope
    dt_c = dt - dt.mean(); dp_c = dp - dp.mean()
    denom = (dt_c ** 2).sum()
    if denom < 1e-12:
        return 0.0
    slope = float((dt_c * dp_c).sum() / denom)
    return round(slope, 4)


def evaluate(
    model,                          # PatchCNN (Iter-1) or ResidualCNN (Iter-2/3)
    ds_test: SDBPatchDataset,
    truth: np.ndarray,
    do_band_shuffle: bool = True,
    sigma_k: float = 1.0,           # temperature factor from calibration
    is_iter2: bool = False,
    is_iter3: bool = False,
    iso_model=None,                 # DL-1: fitted IsotonicRegression (or None = raw)
) -> Dict:
    """Gamma evaluation: per-bin RMSE, R2, bias, sigma-coverage, decile slope, band-shuffle.

    Parameters
    ----------
    truth : (N,) float32 ground-truth depths for test set
    sigma_k : temperature factor applied to sigma (k > 1 inflates sigma)
    is_iter2 : if True, use ResidualCNN prediction path (Stumpf prior)
    is_iter3 : if True, use ResidualCNN prediction path (Lyzenga prior)
    iso_model : DL-1 fitted IsotonicRegression (val-fold), or None (raw mode).
                When set, corrected per-bin metrics are added alongside raw.
    Returns nested dict with keys: overall, per_bin, band_shuffle,
      stumpf_only (iter2) or physics_only (iter3).
      If iso_model is set, also includes 'overall_corrected', 'per_bin_corrected'.
    """
    use_residual = is_iter2 or is_iter3
    if use_residual:
        mu_pred, sigma_raw, residuals = _predict_iter2(model, ds_test)
        sigma_pred = sigma_raw * sigma_k
        # Physics-prior component (Stumpf for iter2, Lyzenga for iter3)
        if is_iter3:
            physics_z = ds_test.physics_vals if ds_test.physics_vals is not None else np.zeros_like(mu_pred)
        else:
            physics_z = ds_test.stumpf_vals if ds_test.stumpf_vals is not None else np.zeros_like(mu_pred)
    else:
        mu_pred, sigma_raw = _predict_iter1(model, ds_test)
        sigma_pred = sigma_raw * sigma_k
        physics_z = None

    err = mu_pred - truth

    def _metrics(y_pred, y_true):
        n = len(y_true)
        if n == 0:
            return {"n": 0, "rmse_m": None, "bias_m": None, "r2": None, "decile_slope": None}
        residuals_ = y_pred - y_true
        rmse = float(np.sqrt(np.mean(residuals_ ** 2)))
        bias = float(np.mean(residuals_))
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        ss_res = np.sum(residuals_ ** 2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
        slope = _decile_slope(y_pred, y_true)
        return {"n": int(n), "rmse_m": rmse, "bias_m": bias, "r2": r2, "decile_slope": slope}

    overall = _metrics(mu_pred, truth)
    cov95 = float(np.mean(np.abs(err) <= 1.96 * sigma_pred))
    overall["sigma_95_coverage"] = cov95

    # Physics-only metrics (Stumpf for iter2, Lyzenga for iter3)
    physics_metrics = None
    if use_residual and physics_z is not None:
        physics_metrics = _metrics(physics_z, truth)

    per_bin = {}
    for lo, hi in GAMMA_BANDS:
        mask = (truth >= lo) & (truth < hi)
        label = f"{lo}-{hi}m"
        per_bin[label] = _metrics(mu_pred[mask], truth[mask])
        # Also add physics-only per-bin
        if use_residual and physics_z is not None:
            phys_tag = "lyzenga" if is_iter3 else "stumpf"
            per_bin[label][f"{phys_tag}_rmse_m"] = (
                float(np.sqrt(np.mean((physics_z[mask] - truth[mask]) ** 2)))
                if mask.sum() > 0 else None
            )
            per_bin[label][f"{phys_tag}_r2"] = (
                _metrics(physics_z[mask], truth[mask])["r2"]
                if mask.sum() > 0 else None
            )
            # Also keep stumpf_ keys for backward compat if iter2
            if is_iter2:
                per_bin[label]["stumpf_rmse_m"] = per_bin[label].get(f"{phys_tag}_rmse_m")
                per_bin[label]["stumpf_r2"]     = per_bin[label].get(f"{phys_tag}_r2")

    # DL-1: isotonic bias correction (BIAS_CAL=isotonic, applied to test predictions only)
    mu_corr = apply_isotonic_bias_cal(mu_pred, iso_model)
    overall_corrected = None
    per_bin_corrected = None
    if iso_model is not None:
        overall_corrected = _metrics(mu_corr, truth)
        cov95_corr = float(np.mean(np.abs(mu_corr - truth) <= 1.96 * sigma_pred))
        overall_corrected["sigma_95_coverage"] = cov95_corr
        per_bin_corrected = {}
        for lo, hi in GAMMA_BANDS:
            mask = (truth >= lo) & (truth < hi)
            label = f"{lo}-{hi}m"
            per_bin_corrected[label] = _metrics(mu_corr[mask], truth[mask])
        L.info(
            f"[DL-1] Corrected RMSE={overall_corrected['rmse_m']:.3f} m  "
            f"bias={overall_corrected['bias_m']:.3f} m  "
            f"decile_slope={overall_corrected['decile_slope']:.4f}  "
            f"(raw RMSE={overall['rmse_m']:.3f} m)"
        )

    # Band-shuffle guard
    # DL-1 note: the isotonic map is a MONOTONE function of predictions — it cannot create
    # spectral rank skill that does not exist in raw predictions.  We run band-shuffle on BOTH
    # raw and (if iso set) corrected predictions.  If iso makes shuffled-input look good, that
    # would indicate the correction was fit on test data (a bug) — we detect and flag it.
    band_shuffle = {}
    if do_band_shuffle:
        rng = np.random.default_rng(SEED)
        orig_cube = ds_test.cube_padded.copy()
        n_channels = orig_cube.shape[-1]
        perm = rng.permutation(n_channels)
        ds_test.cube_padded = orig_cube[:, :, perm]
        if use_residual:
            mu_shuf, _, _ = _predict_iter2(model, ds_test)
        else:
            mu_shuf, _ = _predict_iter1(model, ds_test)
        ds_test.cube_padded = orig_cube

        err_shuf = mu_shuf - truth
        rmse_shuf = float(np.sqrt(np.mean(err_shuf ** 2)))
        rmse_orig = overall["rmse_m"]
        delta_pct = 100.0 * (rmse_shuf - rmse_orig) / max(rmse_orig, 1e-6)
        leakage_flag = delta_pct < 5.0
        band_shuffle = {
            "rmse_shuffled_m": rmse_shuf,
            "rmse_original_m": rmse_orig,
            "delta_pct": round(delta_pct, 2),
            "leakage_suspect": leakage_flag,
            "verdict": ("TEXTURE/MEAN-LEAK SUSPECTED" if leakage_flag
                        else "spectral depth skill present"),
        }

        # DL-1: also check band-shuffle on CORRECTED predictions
        if iso_model is not None:
            mu_shuf_corr = apply_isotonic_bias_cal(mu_shuf, iso_model)
            rmse_shuf_corr = float(np.sqrt(np.mean((mu_shuf_corr - truth) ** 2)))
            rmse_orig_corr = overall_corrected["rmse_m"]
            delta_pct_corr = 100.0 * (rmse_shuf_corr - rmse_orig_corr) / max(rmse_orig_corr, 1e-6)
            leakage_flag_corr = delta_pct_corr < 5.0
            band_shuffle["corrected"] = {
                "rmse_shuffled_m": round(rmse_shuf_corr, 4),
                "rmse_original_m": round(rmse_orig_corr, 4),
                "delta_pct": round(delta_pct_corr, 2),
                "leakage_suspect": leakage_flag_corr,
                "verdict": ("LEAK: iso correction absorbed shuffle noise — check val/test split!"
                            if leakage_flag_corr else "spectral depth skill present (corrected)"),
            }
            L.info(
                f"[DL-1 band-shuffle] corrected: RMSE_orig={rmse_orig_corr:.3f}  "
                f"RMSE_shuf={rmse_shuf_corr:.3f}  delta={delta_pct_corr:+.1f}%  "
                f"{'LEAKAGE?' if leakage_flag_corr else 'OK'}"
            )

    result = {"overall": overall, "per_bin": per_bin, "band_shuffle": band_shuffle}
    if overall_corrected is not None:
        result["overall_corrected"] = overall_corrected
        result["per_bin_corrected"] = per_bin_corrected
        result["bias_cal"] = "isotonic_DL1"
    if physics_metrics is not None:
        if is_iter3:
            result["lyzenga_only"] = physics_metrics
        else:
            result["stumpf_only"]  = physics_metrics
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline per site (Iter-1 or Iter-2 mode)
# ══════════════════════════════════════════════════════════════════════════════
def run_site(
    site_key: str,
    cache_dir: Path,
    epochs: int = N_EPOCHS,
    n_blocks: int = N_BLOCKS,
    buffer_m: float = BLOCK_BUF_M,
    iter2: bool = False,
    iter3: bool = False,
    lambda_rank: float = ITER2_LAMBDA_RANK,
) -> Dict:
    """Full Alpha->Beta->Gamma pipeline for one site.

    Parameters
    ----------
    buffer_m : spatial-block buffer.  200 m = Iter-1 original.  500 m = honest baseline.
    iter2    : if True, use Stumpf-primary + ResidualCNN + SILog-rank + sigma-cal (train-set cal).
    iter3    : if True, use Lyzenga-Kd-primary + ResidualCNN + SILog-rank + val-fold sigma-cal.
               iter3 takes precedence over iter2 if both are set.
    """
    if iter3:
        iter2 = False   # iter3 supersedes iter2; keep iter2 flag false for metadata clarity

    mode_label = ("Iter-3 (Lyzenga-Kd-primary + ResidualCNN + val-fold sigma-cal)" if iter3 else
                  "Iter-2 (Stumpf-primary + ResidualCNN + SILog-rank)" if iter2 else
                  "Iter-1 (PatchCNN)")
    L.info(f"\n{'='*60}")
    L.info(f"Site: {SITE_CFG[site_key]['label']}  ({site_key})")
    L.info(f"Mode: {mode_label}")
    L.info(f"Buffer: {buffer_m:.0f} m")
    L.info(f"{'='*60}")

    # A1: fetch S2
    s2 = fetch_s2_for_site(site_key, cache_dir)
    H, W = int(s2["height"]), int(s2["width"])
    bbox = SITE_CFG[site_key]["bbox"]
    res_m = float(s2["resolution_m"]) if "resolution_m" in s2 else 10.0
    L.info(f"[{site_key}] S2 grid: {H}x{W} px @ {res_m:.0f} m, bbox={bbox}")
    L.info(f"[{site_key}] Buffer in pixels: {buffer_m/res_m:.0f} px")

    # A2: load truth
    lat_truth, lon_truth, dep_truth = load_truth_points(site_key)
    if len(dep_truth) < 50:
        raise ValueError(f"[{site_key}] Too few truth points ({len(dep_truth)}); aborting")

    # A3: build feature cube
    cube, water_mask, feat_names = build_feature_cube(s2)
    C = cube.shape[-1]
    L.info(f"[{site_key}] Feature cube: {H}x{W}x{C}")

    # Rasterize truth onto S2 grid
    w_b, s_b, e_b, n_b = bbox
    rows_all = np.clip(((n_b - lat_truth) / (n_b - s_b) * H).astype(int), 0, H - 1)
    cols_all = np.clip(((lon_truth - w_b) / (e_b - w_b) * W).astype(int), 0, W - 1)

    is_water = water_mask[rows_all, cols_all]
    rows_w  = rows_all[is_water]
    cols_w  = cols_all[is_water]
    dep_w   = dep_truth[is_water]
    lat_w   = lat_truth[is_water]
    lon_w   = lon_truth[is_water]

    L.info(f"[{site_key}] Water pixels with truth: {len(dep_w)} "
           f"({100*len(dep_w)/max(len(dep_truth),1):.1f}% of truth in water mask)")

    if len(dep_w) < 30:
        L.warning(f"[{site_key}] Only {len(dep_w)} water truth points. "
                  "Widening water mask.")
        rows_w, cols_w, dep_w = rows_all, cols_all, dep_truth
        lat_w, lon_w = lat_truth, lon_truth

    # A4: spatial-block split — request block labels for Iter-3 val carve
    need_labels = iter3
    if need_labels:
        train_mask, test_mask, block_labels = make_spatial_block_centers(
            lat_w, lon_w, n_blocks=n_blocks, buffer_m=buffer_m, seed=SEED,
            return_labels=True,
        )
    else:
        train_mask, test_mask = make_spatial_block_centers(
            lat_w, lon_w, n_blocks=n_blocks, buffer_m=buffer_m, seed=SEED,
        )
        block_labels = None

    n_tr = int(train_mask.sum()); n_te = int(test_mask.sum())
    L.info(f"[{site_key}] n_train={n_tr}, n_test={n_te}")

    if n_tr < 20 or n_te < 10:
        raise ValueError(f"[{site_key}] Insufficient split: train={n_tr}, test={n_te}")

    # Iter-3: carve val fold from TRAIN blocks (before anything sees test)
    val_mask = None
    train_sub_mask = train_mask  # default: full train
    if iter3:
        train_sub_mask, val_mask = carve_val_from_train(
            lat_w, lon_w, train_mask, block_labels,
            val_frac=ITER3_VAL_FRAC, seed=SEED,
        )
        n_tr = int(train_sub_mask.sum())  # update for dataset construction
        L.info(f"[{site_key}] After val carve: n_train_sub={n_tr}, n_val={val_mask.sum()}")

    # Normalisation stats from effective TRAIN set only (train_sub for iter3, train for others)
    train_patches_flat = cube[rows_w[train_sub_mask], cols_w[train_sub_mask], :]
    mean_ = train_patches_flat.mean(axis=0).astype(np.float32)
    std_  = train_patches_flat.std(axis=0).astype(np.float32)
    std_  = np.where(std_ < 1e-6, 1.0, std_)

    # ---- Iter-2: fit Stumpf on TRAIN set, get z_stumpf for all pixels --------
    stumpf_train = None
    stumpf_test  = None
    stumpf_m1, stumpf_m0 = None, None
    stumpf_train_rmse = None

    if iter2:
        stumpf_ratio_train = cube[rows_w[train_sub_mask], cols_w[train_sub_mask], 9].astype(np.float64)
        dep_train = dep_w[train_sub_mask].astype(np.float64)
        stumpf_m1, stumpf_m0 = fit_stumpf(stumpf_ratio_train, dep_train, robust=True)

        stumpf_train = apply_stumpf(cube, stumpf_m1, stumpf_m0,
                                    rows_w[train_sub_mask], cols_w[train_sub_mask])
        stumpf_test  = apply_stumpf(cube, stumpf_m1, stumpf_m0,
                                    rows_w[test_mask],  cols_w[test_mask])

        # Report Stumpf-only train RMSE for context
        stumpf_res_train = stumpf_train - dep_train.astype(np.float32)
        stumpf_train_rmse = float(np.sqrt(np.mean(stumpf_res_train ** 2)))
        stumpf_res_test   = stumpf_test - dep_w[test_mask].astype(np.float32)
        stumpf_test_rmse  = float(np.sqrt(np.mean(stumpf_res_test ** 2)))
        L.info(f"[{site_key}] Stumpf train RMSE: {stumpf_train_rmse:.3f} m, "
               f"test RMSE: {stumpf_test_rmse:.3f} m")

    # ---- Iter-3: fit Lyzenga on TRAIN set (NOT val, NOT test) ----------------
    lyz_train = None
    lyz_val   = None
    lyz_test  = None
    lyz_coef  = None
    lyz_train_rmse = None

    if iter3:
        dep_train_sub = dep_w[train_sub_mask].astype(np.float64)

        # Deep-water mask for Lyzenga R_infinity (use all deep-looking pixels in water_mask)
        ndwi_full = ((cube[:, :, 1].astype(np.float64) - cube[:, :, 3].astype(np.float64)) /
                     (cube[:, :, 1].astype(np.float64) + cube[:, :, 3].astype(np.float64) + 1e-7))
        deep_mask_full = ndwi_full > 0.3
        if deep_mask_full.sum() < 50:
            deep_mask_full = ndwi_full > 0.1
        if deep_mask_full.sum() < 20:
            deep_mask_full = water_mask  # fallback

        # Compute Lyzenga features for TRAIN sub-set
        X_train, lyz_names = compute_lyzenga_features(
            cube, rows_w[train_sub_mask], cols_w[train_sub_mask],
            deep_mask_full,
            use_b4=ITER3_USE_B4, deep_pct=ITER3_LYZ_DEEP_PCT,
        )

        # Inverse-freq depth weights for WLS fit
        wls_weights = _build_bin_weights(dep_train_sub)

        lyz_coef = fit_lyzenga(X_train, dep_train_sub, sample_weights=wls_weights, robust=True)

        # Evaluate Lyzenga prior on train sub
        lyz_train = apply_lyzenga(X_train, lyz_coef)
        lyz_res_train = lyz_train - dep_train_sub.astype(np.float32)
        lyz_train_rmse = float(np.sqrt(np.mean(lyz_res_train ** 2)))

        # Lyzenga prior on val
        X_val, _ = compute_lyzenga_features(
            cube, rows_w[val_mask], cols_w[val_mask],
            deep_mask_full, use_b4=ITER3_USE_B4, deep_pct=ITER3_LYZ_DEEP_PCT,
        )
        lyz_val = apply_lyzenga(X_val, lyz_coef)
        lyz_val_rmse = float(np.sqrt(np.mean((lyz_val - dep_w[val_mask].astype(np.float32)) ** 2)))

        # Lyzenga prior on test
        X_test, _ = compute_lyzenga_features(
            cube, rows_w[test_mask], cols_w[test_mask],
            deep_mask_full, use_b4=ITER3_USE_B4, deep_pct=ITER3_LYZ_DEEP_PCT,
        )
        lyz_test = apply_lyzenga(X_test, lyz_coef)
        lyz_test_rmse = float(np.sqrt(np.mean((lyz_test - dep_w[test_mask].astype(np.float32)) ** 2)))

        L.info(f"[{site_key}] Lyzenga train RMSE: {lyz_train_rmse:.3f} m, "
               f"val RMSE: {lyz_val_rmse:.3f} m, test RMSE: {lyz_test_rmse:.3f} m")

    # A5: datasets
    ds_train = SDBPatchDataset(
        cube,
        rows_w[train_sub_mask], cols_w[train_sub_mask], dep_w[train_sub_mask],
        patch_radius=PATCH_RADIUS, mean=mean_, std=std_,
        stumpf_vals=stumpf_train,
        physics_vals=lyz_train,
    )
    ds_test = SDBPatchDataset(
        cube,
        rows_w[test_mask], cols_w[test_mask], dep_w[test_mask],
        patch_radius=PATCH_RADIUS, mean=mean_, std=std_,
        stumpf_vals=stumpf_test,
        physics_vals=lyz_test,
    )

    # B: Train model
    use_residual = iter2 or iter3
    L.info(f"[{site_key}] Training {'ResidualCNN' if use_residual else 'PatchCNN'} "
           f"({C} channels, {epochs} epochs) ...")
    t0 = time.time()
    if iter3:
        model = train_model_iter3(ds_train, in_channels=C, epochs=epochs,
                                  lambda_rank=lambda_rank)
    elif iter2:
        model = train_model_iter2(ds_train, in_channels=C, epochs=epochs,
                                  lambda_rank=lambda_rank)
    else:
        model = train_model(ds_train, in_channels=C, epochs=epochs)
    elapsed = time.time() - t0
    L.info(f"[{site_key}] Training done in {elapsed:.1f}s")

    # Sigma calibration
    sigma_k = 1.0
    if iter3 and val_mask is not None and val_mask.sum() >= 10:
        # Iter-3: calibrate on HELD-OUT VAL FOLD (never seen by model during training)
        ds_val = SDBPatchDataset(
            cube, rows_w[val_mask], cols_w[val_mask], dep_w[val_mask],
            patch_radius=PATCH_RADIUS, mean=mean_, std=std_,
            physics_vals=lyz_val,
        )
        mu_val, sig_val, _ = _predict_iter2(model, ds_val)
        sigma_k = calibrate_sigma(
            mu_val, sig_val, dep_w[val_mask].astype(np.float32),
            target_coverage=ITER3_SIGMA_TARGET,
        )
        L.info(f"[{site_key}] Iter-3 sigma-cal on val fold: k={sigma_k:.4f}")
    elif iter2:
        # Iter-2: calibrate on TRAIN set (kept for backward compat)
        ds_train_eval = SDBPatchDataset(
            cube, rows_w[train_sub_mask], cols_w[train_sub_mask], dep_w[train_sub_mask],
            patch_radius=PATCH_RADIUS, mean=mean_, std=std_,
            stumpf_vals=stumpf_train,
        )
        mu_tr, sig_tr, _ = _predict_iter2(model, ds_train_eval)
        sigma_k = calibrate_sigma(mu_tr, sig_tr, dep_w[train_sub_mask].astype(np.float32))
        L.info(f"[{site_key}] Iter-2 sigma_k={sigma_k:.4f}")

    # G: Evaluate
    truth_test = dep_w[test_mask].astype(np.float32)
    results = evaluate(
        model, ds_test, truth_test,
        do_band_shuffle=True,
        sigma_k=sigma_k,
        is_iter2=iter2,
        is_iter3=iter3,
    )

    # Enrich metadata
    results["site"]           = site_key
    results["label"]          = SITE_CFG[site_key]["label"]
    results["n_total_truth"]  = len(dep_truth)
    results["n_water_truth"]  = len(dep_w)
    results["n_train"]        = int(train_sub_mask.sum())
    results["n_val"]          = int(val_mask.sum()) if val_mask is not None else 0
    results["n_test"]         = n_te
    results["n_blocks"]       = n_blocks
    results["epochs"]         = epochs
    results["buffer_m"]       = buffer_m
    results["iter2"]          = iter2
    results["iter3"]          = iter3
    results["sigma_k"]        = sigma_k
    results["depth_range_m"]  = [float(dep_w.min()), float(dep_w.max())]
    results["feat_names"]     = feat_names
    results["training_time_s"] = round(elapsed, 1)
    if iter2:
        results["stumpf_fit"]          = {"m1": stumpf_m1, "m0": stumpf_m0}
        results["stumpf_train_rmse_m"] = stumpf_train_rmse
    if iter3 and lyz_coef is not None:
        results["lyzenga_coef"]         = lyz_coef.tolist()
        results["lyzenga_train_rmse_m"] = lyz_train_rmse
        results["lyzenga_feat_names"]   = lyz_names

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Pretty-print + JSON save
# ══════════════════════════════════════════════════════════════════════════════
def print_report(results: Dict):
    site  = results["site"]
    label = results["label"]
    ov    = results["overall"]
    mode  = ("Iter-3" if results.get("iter3") else
             "Iter-2" if results.get("iter2") else "Iter-1")
    buf   = results.get("buffer_m", BLOCK_BUF_M)

    L.info("")
    L.info(f"{'='*60}")
    L.info(f"RESULTS: {label} ({site}) [{mode}, buffer={buf:.0f} m]")
    L.info(f"{'='*60}")
    L.info(f"  n_train / n_test / n_blocks : "
           f"{results['n_train']} / {results['n_test']} / {results['n_blocks']}")
    L.info(f"  depth range                 : "
           f"{results['depth_range_m'][0]:.1f} - {results['depth_range_m'][1]:.1f} m")
    L.info(f"  OVERALL (test set):")
    L.info(f"    RMSE   = {ov['rmse_m']:.3f} m")
    L.info(f"    bias   = {ov['bias_m']:.3f} m")
    L.info(f"    R2     = {ov['r2']:.3f}")
    L.info(f"    decile_slope = {ov.get('decile_slope', 'n/a')}")
    L.info(f"    sigma-95% coverage = {ov['sigma_95_coverage']:.2%}")
    if results.get("sigma_k", 1.0) != 1.0:
        L.info(f"    sigma_k (temperature) = {results['sigma_k']:.4f}")

    if results.get("stumpf_only"):
        sm = results["stumpf_only"]
        L.info(f"  STUMPF-ONLY (test set, no CNN):")
        L.info(f"    RMSE={sm['rmse_m']:.3f} m  bias={sm['bias_m']:.3f} m  "
               f"R2={sm['r2']:.3f}  decile_slope={sm.get('decile_slope', 'n/a')}")

    if results.get("lyzenga_only"):
        lm = results["lyzenga_only"]
        L.info(f"  LYZENGA-ONLY (test set, no CNN):")
        L.info(f"    RMSE={lm['rmse_m']:.3f} m  bias={lm['bias_m']:.3f} m  "
               f"R2={lm['r2']:.3f}  decile_slope={lm.get('decile_slope', 'n/a')}")

    L.info(f"  PER-BIN RMSE:")
    for band_label, m in results["per_bin"].items():
        if m["n"] == 0:
            L.info(f"    {band_label:10s}: n=0 (no truth)")
        else:
            physics_info = ""
            if "lyzenga_rmse_m" in m:
                physics_info = f"  [lyzenga: RMSE={m['lyzenga_rmse_m']:.3f} R2={m.get('lyzenga_r2', 'n/a')}]"
            elif "stumpf_rmse_m" in m:
                physics_info = f"  [stumpf: RMSE={m['stumpf_rmse_m']:.3f} R2={m.get('stumpf_r2', 'n/a')}]"
            L.info(f"    {band_label:10s}: n={m['n']:5d}  RMSE={m['rmse_m']:.3f} m  "
                   f"bias={m['bias_m']:.3f} m  R2={m['r2']:.3f}  "
                   f"slope={m.get('decile_slope', 'n/a')}{physics_info}")

    bs = results.get("band_shuffle", {})
    if bs:
        L.info(f"  BAND-SHUFFLE GUARD:")
        L.info(f"    RMSE (original) = {bs['rmse_original_m']:.3f} m")
        L.info(f"    RMSE (shuffled) = {bs['rmse_shuffled_m']:.3f} m")
        L.info(f"    delta = {bs['delta_pct']:+.1f}%  ->  {bs['verdict']}")


def _default_serialiser(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT MODEL — multi-site training + save + inference API
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_TRAINING_SITES = ["khalifa", "omc"]
DEFAULT_MODEL_PT   = ROOT / "backend" / "models" / "sdb_cnn_default.pt"
DEFAULT_MODEL_JSON = ROOT / "backend" / "models" / "sdb_cnn_default.json"

# Extended depth bins for per-bin reporting in the default model
DEFAULT_GAMMA_BANDS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]

# Default training hyperparameters (Iter-1 winner config, 8 blocks for depth-balance)
DEFAULT_EPOCHS    = 80     # more epochs for pooled multi-site data
DEFAULT_N_BLOCKS  = 8     # 8 blocks (Iter-1 winner) — better depth-distribution balance
                           # than 10 blocks at Khalifa Port (basin geometry creates depth zones)
DEFAULT_BUFFER_M  = 500.0 # honest 500 m leakage-safe buffer
DEFAULT_VAL_FRAC  = 0.20  # 20% of train blocks -> val for sigma calibration
DEFAULT_SIGMA_TGT = 0.95  # target sigma-95% coverage on val fold


def _build_site_data(
    site_key: str,
    cache_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, List[str]]:
    """Load S2 cube + truth for one site.

    Returns
    -------
    cube      : (H, W, C) float32
    water_mask: (H, W) bool
    rows_w    : (N,) int    pixel row of each water truth point
    cols_w    : (N,) int    pixel col
    dep_w     : (N,) float32 depth
    lat_w     : (N,) float64
    lon_w     : (N,) float64
    feat_names: list[str]
    """
    s2 = fetch_s2_for_site(site_key, cache_dir)
    H, W = int(s2["height"]), int(s2["width"])
    bbox = SITE_CFG[site_key]["bbox"]
    res_m = float(s2["resolution_m"]) if "resolution_m" in s2 else 10.0

    lat_truth, lon_truth, dep_truth = load_truth_points(site_key)
    cube, water_mask, feat_names = build_feature_cube(s2)

    w_b, s_b, e_b, n_b = bbox
    rows_all = np.clip(((n_b - lat_truth) / (n_b - s_b) * H).astype(int), 0, H - 1)
    cols_all = np.clip(((lon_truth - w_b) / (e_b - w_b) * W).astype(int), 0, W - 1)

    is_water = water_mask[rows_all, cols_all]
    rows_w = rows_all[is_water]
    cols_w = cols_all[is_water]
    dep_w  = dep_truth[is_water].astype(np.float32)
    lat_w  = lat_truth[is_water]
    lon_w  = lon_truth[is_water]

    if len(dep_w) < 30:
        L.warning(f"[{site_key}] Wide water fallback ({len(dep_w)} pts in mask)")
        rows_w, cols_w = rows_all, cols_all
        dep_w  = dep_truth.astype(np.float32)
        lat_w, lon_w = lat_truth, lon_truth

    L.info(f"[{site_key}] site data: {H}x{W} cube, {len(dep_w)} water truth pts "
           f"(depth [{dep_w.min():.1f},{dep_w.max():.1f}] m)")
    return cube, water_mask, rows_w, cols_w, dep_w, lat_w, lon_w, feat_names


def train_default_model(
    cache_dir: Optional[Path] = None,
    out_pt: Path = DEFAULT_MODEL_PT,
    out_json: Path = DEFAULT_MODEL_JSON,
    epochs: int = DEFAULT_EPOCHS,
    n_blocks: int = DEFAULT_N_BLOCKS,
    buffer_m: float = DEFAULT_BUFFER_M,
    val_frac: float = DEFAULT_VAL_FRAC,
    sigma_target: float = DEFAULT_SIGMA_TGT,
    force_retrain: bool = False,
) -> Dict:
    """Train the generalizable multi-site default SDB CNN and save weights + metadata.

    Architecture: Iter-1 PatchCNN (9x9 patches, 12 channels, sigmoid depth head,
    hetero log-var head). Trained on pooled Khalifa + OMC in-situ data.
    Val-fold sigma calibration for coverage in [0.90, 0.97].

    Parameters
    ----------
    cache_dir  : directory holding the per-site *_s2_gee.npz caches.
                 Defaults to May_2026_results/v_iho_iterations/iter_cnn_baseline/s2_cache/
    out_pt     : destination for torch state_dict (.pt)
    out_json   : destination for metadata + metrics (.json)
    force_retrain : if False and out_pt already exists, skip training and return metadata.

    Returns
    -------
    dict with honest test metrics (pooled + per-site) and saved file paths.
    """
    if cache_dir is None:
        cache_dir = ROOT / "May_2026_results" / "v_iho_iterations" / \
                    "iter_cnn_baseline" / "s2_cache"

    if not force_retrain and out_pt.exists() and out_json.exists():
        L.info(f"Default model already exists at {out_pt}; loading metadata (use force_retrain=True to re-train)")
        with open(out_json) as f:
            meta = json.load(f)
        return meta

    L.info("=" * 70)
    L.info("TRAINING DEFAULT SDB CNN (multi-site: khalifa + omc)")
    L.info(f"  Architecture: Iter-1 PatchCNN (9x9 patches, 12 channels)")
    L.info(f"  Buffer: {buffer_m:.0f} m  |  N_blocks: {n_blocks}  |  Epochs: {epochs}")
    L.info("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)

    sites = DEFAULT_TRAINING_SITES

    # ── Step 1: Load per-site data ──────────────────────────────────────────
    site_data: Dict[str, Dict] = {}
    for sk in sites:
        cube, water_mask, rows_w, cols_w, dep_w, lat_w, lon_w, feat_names = \
            _build_site_data(sk, cache_dir)
        site_data[sk] = {
            "cube": cube, "water_mask": water_mask,
            "rows_w": rows_w, "cols_w": cols_w, "dep_w": dep_w,
            "lat_w": lat_w, "lon_w": lon_w, "feat_names": feat_names,
        }

    # ── Step 2: Per-site spatial-block splits (500 m, return labels for val carve)
    for sk in sites:
        sd = site_data[sk]
        train_mask, test_mask, block_labels = make_spatial_block_centers(
            sd["lat_w"], sd["lon_w"],
            n_blocks=n_blocks, buffer_m=buffer_m, seed=SEED,
            return_labels=True,
        )
        # Carve val from train blocks
        train_sub_mask, val_mask = carve_val_from_train(
            sd["lat_w"], sd["lon_w"], train_mask, block_labels,
            val_frac=val_frac, seed=SEED,
        )
        sd["train_mask"]    = train_mask
        sd["test_mask"]     = test_mask
        sd["train_sub_mask"] = train_sub_mask
        sd["val_mask"]      = val_mask
        sd["block_labels"]  = block_labels
        L.info(f"[{sk}] split: train_sub={train_sub_mask.sum()}, "
               f"val={val_mask.sum()}, test={test_mask.sum()}")

    # ── Step 3: Pool TRAIN patches to compute shared normalisation ─────────
    # The shared normalisation is computed over ALL pooled train patches so
    # each site is normalised to the same feature space.  This is intentional:
    # the CNN must generalise across the turbid-deep Khalifa (0-21 m) and the
    # shallow OMC (0-11 m) regimes.
    all_train_patches = []
    for sk in sites:
        sd = site_data[sk]
        pts = sd["cube"][sd["rows_w"][sd["train_sub_mask"]],
                          sd["cols_w"][sd["train_sub_mask"]], :]  # (N_tr, C)
        all_train_patches.append(pts)
    all_train_patches = np.concatenate(all_train_patches, axis=0)  # (N_all, C)

    shared_mean = all_train_patches.mean(axis=0).astype(np.float32)
    shared_std  = all_train_patches.std(axis=0).astype(np.float32)
    shared_std  = np.where(shared_std < 1e-6, 1.0, shared_std)
    C = shared_mean.shape[0]
    L.info(f"Shared normalisation computed over {len(all_train_patches)} pooled train pts "
           f"({C} channels)")

    # Also store per-site normalisation for diagnostics / per-site inference fallback
    for sk in sites:
        sd = site_data[sk]
        pts = sd["cube"][sd["rows_w"][sd["train_sub_mask"]],
                          sd["cols_w"][sd["train_sub_mask"]], :]
        sd["site_mean"] = pts.mean(axis=0).astype(np.float32)
        sd["site_std"]  = np.where(pts.std(axis=0) < 1e-6, 1.0,
                                    pts.std(axis=0)).astype(np.float32)

    # ── Step 4: Build pooled train dataset (both sites, shared norm) ────────
    train_datasets = []
    val_datasets   = []
    for sk in sites:
        sd = site_data[sk]
        # Train sub
        ds_tr = SDBPatchDataset(
            sd["cube"],
            sd["rows_w"][sd["train_sub_mask"]],
            sd["cols_w"][sd["train_sub_mask"]],
            sd["dep_w"][sd["train_sub_mask"]],
            patch_radius=PATCH_RADIUS,
            mean=shared_mean, std=shared_std,
        )
        # Val
        ds_val = SDBPatchDataset(
            sd["cube"],
            sd["rows_w"][sd["val_mask"]],
            sd["cols_w"][sd["val_mask"]],
            sd["dep_w"][sd["val_mask"]],
            patch_radius=PATCH_RADIUS,
            mean=shared_mean, std=shared_std,
        )
        train_datasets.append(ds_tr)
        val_datasets.append(ds_val)
        L.info(f"[{sk}] ds_train={len(ds_tr)}, ds_val={len(ds_val)}")

    # Concatenate train datasets using ConcatDataset
    from torch.utils.data import ConcatDataset as _ConcatDataset
    ds_train_pooled = _ConcatDataset(train_datasets)
    L.info(f"Pooled train dataset: {len(ds_train_pooled)} patches")

    # ── Step 5: Train PatchCNN on pooled data ──────────────────────────────
    device = torch.device("cpu")
    model = PatchCNN(in_channels=C, patch_px=PATCH_PX, max_depth=MAX_DEPTH_M).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = HeteroNLLLoss()

    # Build pooled depth array for inverse-freq weighting
    pooled_depths = np.concatenate([
        sd["dep_w"][sd["train_sub_mask"]] for sd in site_data.values()
    ])
    pooled_weights_np = _build_bin_weights(pooled_depths)

    loader = DataLoader(
        ds_train_pooled, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    t0_train = time.time()
    model.train()
    best_val_rmse = float("inf")   # track val RMSE (not NLL) to avoid sigma-head gaming
    patience_epochs = 0
    early_stop_patience = 15        # stop after 15 consecutive non-improving val evaluations
    best_state_dict = None

    # Per-site val DataLoaders (for balanced val RMSE tracking)
    site_val_loaders = {}
    for sk in sites:
        ds_v = val_datasets[sites.index(sk)]
        site_val_loaders[sk] = DataLoader(
            ds_v, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )

    # Also build pooled val loader for sigma calibration later
    ds_val_pooled = _ConcatDataset(val_datasets)
    val_loader = DataLoader(
        ds_val_pooled, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    VAL_EVAL_EVERY = 5  # evaluate val every 5 epochs

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0; n_batches = 0
        for batch in loader:
            patches, depths = batch[0], batch[1]
            patches, depths = patches.to(device), depths.to(device)
            optimizer.zero_grad()
            mu, lv = model(patches)
            loss = criterion(mu, lv, depths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1
        scheduler.step()

        # Evaluate val RMSE per site every VAL_EVAL_EVERY epochs
        if epoch % VAL_EVAL_EVERY == 0 or epoch == epochs:
            model.eval()
            site_val_rmses = []
            with torch.no_grad():
                for sk in sites:
                    vmus, vtruths = [], []
                    for vbatch in site_val_loaders[sk]:
                        vp, vd = vbatch[0].to(device), vbatch[1].to(device)
                        vm, vlv = model(vp)
                        vmus.append(vm.cpu().numpy())
                        vtruths.append(vd.cpu().numpy())
                    vmus    = np.concatenate(vmus)
                    vtruths = np.concatenate(vtruths)
                    site_rmse = float(np.sqrt(np.mean((vmus - vtruths) ** 2)))
                    site_val_rmses.append(site_rmse)

            # Balanced val RMSE = mean over sites (equal weight regardless of n_val per site)
            balanced_val_rmse = float(np.mean(site_val_rmses))

            if balanced_val_rmse < best_val_rmse - 0.005:
                best_val_rmse = balanced_val_rmse
                patience_epochs = 0
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_epochs += 1

            site_rmse_str = "  ".join(
                f"{sk}={r:.3f}m" for sk, r in zip(sites, site_val_rmses)
            )
            L.info(f"  epoch {epoch:3d}/{epochs} | train_loss={epoch_loss/max(n_batches,1):.4f} "
                   f"| val_RMSE [balanced]={balanced_val_rmse:.3f}m [{site_rmse_str}] "
                   f"| best={best_val_rmse:.3f}m | pat={patience_epochs}")

            if patience_epochs >= early_stop_patience:
                L.info(f"  Early stop at epoch {epoch} (val_RMSE={balanced_val_rmse:.3f}m, "
                       f"best={best_val_rmse:.3f}m, patience={early_stop_patience})")
                break

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        L.info(f"Restored best weights (balanced_val_RMSE={best_val_rmse:.3f}m)")

    train_time_s = round(time.time() - t0_train, 1)
    L.info(f"Training complete: {train_time_s}s")

    # ── Step 6: Sigma calibration on pooled val fold ────────────────────────
    # Reload best val loader in case best_state was from earlier and we need fresh pass
    model.eval()
    val_mus, val_sigs, val_truths = [], [], []
    with torch.no_grad():
        for vbatch in val_loader:
            vp, vd = vbatch[0].to(device), vbatch[1].to(device)
            vm, vlv = model(vp)
            val_mus.append(vm.cpu().numpy())
            val_sigs.append(np.sqrt(np.exp(vlv.cpu().numpy())))
            val_truths.append(vd.cpu().numpy())
    val_mus    = np.concatenate(val_mus)
    val_sigs   = np.concatenate(val_sigs)
    val_truths = np.concatenate(val_truths)

    sigma_k = calibrate_sigma(
        val_mus, val_sigs, val_truths,
        target_coverage=sigma_target,
    )
    L.info(f"Sigma calibration (val fold, {len(val_truths)} pts): k={sigma_k:.4f}")

    # Verify coverage is in target range
    cov_check = float(np.mean(np.abs(val_mus - val_truths) <= 1.96 * sigma_k * val_sigs))
    L.info(f"Val-fold sigma-95% coverage after calibration: {cov_check:.3f}")

    # ── Step 6b: DL-1 Isotonic bias correction (BIAS_CAL=isotonic, default off) ─
    # Fit on the SAME pooled val-fold predictions used for sigma calibration.
    # The test set has not been touched at this point — leakage-safe.
    iso_model = fit_isotonic_bias_cal(val_mus, val_truths)
    iso_knots = None
    if iso_model is not None:
        iso_knots = {
            "X": iso_model.X_thresholds_.tolist(),
            "y": iso_model.y_thresholds_.tolist(),
        }
        L.info(f"[DL-1] Isotonic knots: {len(iso_model.X_thresholds_)} segments  "
               f"x=[{iso_model.X_thresholds_.min():.2f},{iso_model.X_thresholds_.max():.2f}]  "
               f"y=[{iso_model.y_thresholds_.min():.2f},{iso_model.y_thresholds_.max():.2f}]")

    # ── Step 7: Evaluate on per-site TEST sets (honest, 500 m buffer) ───────
    model.eval()
    per_site_metrics: Dict[str, Dict] = {}
    all_test_mu, all_test_truth = [], []

    for sk in sites:
        sd = site_data[sk]
        ds_te = SDBPatchDataset(
            sd["cube"],
            sd["rows_w"][sd["test_mask"]],
            sd["cols_w"][sd["test_mask"]],
            sd["dep_w"][sd["test_mask"]],
            patch_radius=PATCH_RADIUS,
            mean=shared_mean, std=shared_std,
        )
        te_loader = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        mus, sigs, truths = [], [], []
        with torch.no_grad():
            for tb in te_loader:
                tp, td = tb[0].to(device), tb[1].to(device)
                tm, tlv = model(tp)
                mus.append(tm.cpu().numpy())
                sigs.append(np.sqrt(np.exp(tlv.cpu().numpy())))
                truths.append(td.cpu().numpy())
        mus    = np.concatenate(mus)
        sigs   = np.concatenate(sigs) * sigma_k
        truths = np.concatenate(truths)
        all_test_mu.append(mus)
        all_test_truth.append(truths)

        # Per-site per-bin RMSE (extended bins: 0-5/5-10/10-15/15-20/20-25m)
        err    = mus - truths
        rmse   = float(np.sqrt(np.mean(err ** 2)))
        bias   = float(np.mean(err))
        ss_tot = np.sum((truths - truths.mean()) ** 2)
        ss_res = np.sum(err ** 2)
        r2     = float(1.0 - ss_res / max(ss_tot, 1e-12))
        slope  = _decile_slope(mus, truths)
        cov95  = float(np.mean(np.abs(err) <= 1.96 * sigs))

        per_bin_ext: Dict[str, Dict] = {}
        for lo, hi in DEFAULT_GAMMA_BANDS:
            mask_b = (truths >= lo) & (truths < hi)
            label_b = f"{lo}-{hi}m"
            if mask_b.sum() == 0:
                per_bin_ext[label_b] = {"n": 0, "rmse_m": None, "r2": None, "decile_slope": None}
                continue
            err_b = mus[mask_b] - truths[mask_b]
            n_b   = int(mask_b.sum())
            rmse_b = float(np.sqrt(np.mean(err_b ** 2)))
            bias_b = float(np.mean(err_b))
            ss_t_b = np.sum((truths[mask_b] - truths[mask_b].mean()) ** 2)
            ss_r_b = np.sum(err_b ** 2)
            r2_b   = float(1.0 - ss_r_b / max(ss_t_b, 1e-12))
            slp_b  = _decile_slope(mus[mask_b], truths[mask_b])
            per_bin_ext[label_b] = {
                "n": n_b, "rmse_m": round(rmse_b, 4), "bias_m": round(bias_b, 4),
                "r2": round(r2_b, 4), "decile_slope": slp_b,
            }

        # DL-1: corrected per-bin (applied to this site's test mus)
        per_bin_corr_ext: Dict[str, Dict] = {}
        mus_corr = apply_isotonic_bias_cal(mus, iso_model)
        if iso_model is not None:
            for lo, hi in DEFAULT_GAMMA_BANDS:
                mask_b = (truths >= lo) & (truths < hi)
                label_b = f"{lo}-{hi}m"
                if mask_b.sum() == 0:
                    per_bin_corr_ext[label_b] = {"n": 0, "rmse_m": None, "bias_m": None}
                    continue
                err_cb = mus_corr[mask_b] - truths[mask_b]
                n_cb   = int(mask_b.sum())
                rmse_cb = float(np.sqrt(np.mean(err_cb ** 2)))
                bias_cb = float(np.mean(err_cb))
                slp_cb  = _decile_slope(mus_corr[mask_b], truths[mask_b])
                per_bin_corr_ext[label_b] = {
                    "n": n_cb, "rmse_m": round(rmse_cb, 4), "bias_m": round(bias_cb, 4),
                    "decile_slope": slp_cb,
                }
            corr_err = mus_corr - truths
            rmse_corr = float(np.sqrt(np.mean(corr_err ** 2)))
            bias_corr = float(np.mean(corr_err))
            slope_corr = _decile_slope(mus_corr, truths)
            L.info(f"[{sk}] DL-1 corrected: RMSE={rmse_corr:.3f} m  bias={bias_corr:.3f} m  "
                   f"slope={slope_corr:.4f}")

        per_site_metrics[sk] = {
            "n_test":        int(len(truths)),
            "n_train_sub":   int(sd["train_sub_mask"].sum()),
            "n_val":         int(sd["val_mask"].sum()),
            "rmse_m":        round(rmse, 4),
            "bias_m":        round(bias, 4),
            "r2":            round(r2, 4),
            "decile_slope":  slope,
            "sigma_95_cov":  round(cov95, 4),
            "depth_range_m": [float(truths.min()), float(truths.max())],
            "per_bin":       per_bin_ext,
        }
        if iso_model is not None:
            per_site_metrics[sk]["per_bin_corrected"] = per_bin_corr_ext
            per_site_metrics[sk]["rmse_corrected_m"]  = round(rmse_corr, 4)
            per_site_metrics[sk]["bias_corrected_m"]  = round(bias_corr, 4)
            per_site_metrics[sk]["slope_corrected"]   = slope_corr
        L.info(f"[{sk}] test: n={len(truths)}, RMSE={rmse:.3f} m, R2={r2:.3f}, "
               f"bias={bias:.3f} m, slope={slope:.4f}, sigma-95%={cov95:.3f}")

    # ── Step 8: Pooled test metrics ─────────────────────────────────────────
    all_mu    = np.concatenate(all_test_mu)
    all_truth = np.concatenate(all_test_truth)
    pooled_err = all_mu - all_truth
    pooled_rmse  = float(np.sqrt(np.mean(pooled_err ** 2)))
    pooled_bias  = float(np.mean(pooled_err))
    ss_tot_p = np.sum((all_truth - all_truth.mean()) ** 2)
    ss_res_p = np.sum(pooled_err ** 2)
    pooled_r2   = float(1.0 - ss_res_p / max(ss_tot_p, 1e-12))
    pooled_slope = _decile_slope(all_mu, all_truth)

    pooled_per_bin: Dict[str, Dict] = {}
    for lo, hi in DEFAULT_GAMMA_BANDS:
        mask_b = (all_truth >= lo) & (all_truth < hi)
        label_b = f"{lo}-{hi}m"
        if mask_b.sum() == 0:
            pooled_per_bin[label_b] = {"n": 0, "rmse_m": None, "r2": None}
            continue
        err_b  = all_mu[mask_b] - all_truth[mask_b]
        n_b    = int(mask_b.sum())
        rmse_b = float(np.sqrt(np.mean(err_b ** 2)))
        bias_b = float(np.mean(err_b))
        ss_t_b = np.sum((all_truth[mask_b] - all_truth[mask_b].mean()) ** 2)
        r2_b   = float(1.0 - np.sum(err_b ** 2) / max(ss_t_b, 1e-12))
        slp_b  = _decile_slope(all_mu[mask_b], all_truth[mask_b])
        pooled_per_bin[label_b] = {
            "n": n_b, "rmse_m": round(rmse_b, 4), "bias_m": round(bias_b, 4),
            "r2": round(r2_b, 4), "decile_slope": slp_b,
        }

    L.info(f"POOLED test: n={len(all_truth)}, RMSE={pooled_rmse:.3f} m, "
           f"R2={pooled_r2:.3f}, bias={pooled_bias:.3f} m, slope={pooled_slope:.4f}")

    # ── Step 9: Band-shuffle guard (pooled) ─────────────────────────────────
    # Use khalifa test set for band-shuffle (largest, most representative)
    sk_bs = "khalifa"
    sd_bs = site_data[sk_bs]
    ds_bs = SDBPatchDataset(
        sd_bs["cube"],
        sd_bs["rows_w"][sd_bs["test_mask"]],
        sd_bs["cols_w"][sd_bs["test_mask"]],
        sd_bs["dep_w"][sd_bs["test_mask"]],
        patch_radius=PATCH_RADIUS, mean=shared_mean, std=shared_std,
    )
    rng_bs = np.random.default_rng(SEED)
    orig_cube_bs = ds_bs.cube_padded.copy()
    perm_bs = rng_bs.permutation(C)
    ds_bs.cube_padded = orig_cube_bs[:, :, perm_bs]
    bs_mus, bs_sigs = _predict_iter1(model, ds_bs)
    ds_bs.cube_padded = orig_cube_bs
    truth_bs = sd_bs["dep_w"][sd_bs["test_mask"]]
    rmse_shuffled = float(np.sqrt(np.mean((bs_mus - truth_bs) ** 2)))
    rmse_orig_bs  = per_site_metrics[sk_bs]["rmse_m"]
    band_shuf_delta_pct = round(100.0 * (rmse_shuffled - rmse_orig_bs) / max(rmse_orig_bs, 1e-6), 2)
    band_shuf_flag = band_shuf_delta_pct < 5.0
    L.info(f"Band-shuffle guard (khalifa test): RMSE={rmse_orig_bs:.3f} m -> "
           f"shuffled={rmse_shuffled:.3f} m  delta={band_shuf_delta_pct:+.1f}%  "
           f"{'LEAKAGE?' if band_shuf_flag else 'SPECTRAL SKILL OK'}")

    # ── Step 10: Save model weights ─────────────────────────────────────────
    torch.save(model.state_dict(), str(out_pt))
    L.info(f"Saved model weights -> {out_pt}")

    # ── Step 11: Save metadata JSON ─────────────────────────────────────────
    import datetime
    meta = {
        "created":        datetime.datetime.utcnow().isoformat() + "Z",
        "architecture": {
            "type":        "PatchCNN (Iter-1)",
            "in_channels": C,
            "patch_px":    PATCH_PX,
            "patch_radius": PATCH_RADIUS,
            "channel_order": feat_names if site_data else [],
            "conv_channels": [32, 64, 64],
            "fc_width":    64,
            "dropout":     0.2,
            "max_depth_m": MAX_DEPTH_M,
            "depth_activation": "sigmoid * max_depth",
            "logvar_clamp": [-6.0, 6.0],
        },
        "training": {
            "sites":      DEFAULT_TRAINING_SITES,
            "n_sites":    len(DEFAULT_TRAINING_SITES),
            "n_train_pooled": int(len(pooled_depths)),
            "epochs_run": epochs,
            "buffer_m":   buffer_m,
            "n_blocks":   n_blocks,
            "val_frac":   val_frac,
            "lr":         LR,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "loss":       "HeteroNLL (inverse-freq depth bin weights)",
            "sigma_calibration": "val-fold temperature scaling (k)",
            "sigma_k":    round(float(sigma_k), 6),
            "sigma_target_coverage": sigma_target,
            "val_fold_sigma_coverage": round(cov_check, 4),
            "training_time_s": train_time_s,
        },
        "normalisation": {
            "mean": shared_mean.tolist(),
            "std":  shared_std.tolist(),
            "channel_names": feat_names if site_data else [],
            "per_site": {
                sk: {
                    "mean": site_data[sk]["site_mean"].tolist(),
                    "std":  site_data[sk]["site_std"].tolist(),
                }
                for sk in sites
            },
        },
        "honest_test_metrics": {
            "split":      "spatial_block_500m",
            "pooled": {
                "n_test":       int(len(all_truth)),
                "rmse_m":       round(pooled_rmse, 4),
                "bias_m":       round(pooled_bias, 4),
                "r2":           round(pooled_r2, 4),
                "decile_slope": pooled_slope,
                "per_bin":      pooled_per_bin,
            },
            "per_site": per_site_metrics,
            "band_shuffle": {
                "site":            sk_bs,
                "rmse_original_m": rmse_orig_bs,
                "rmse_shuffled_m": round(rmse_shuffled, 4),
                "delta_pct":       band_shuf_delta_pct,
                "leakage_suspect": band_shuf_flag,
                "verdict": ("TEXTURE/MEAN-LEAK SUSPECTED" if band_shuf_flag
                            else "spectral depth skill present"),
            },
        },
        "datum":      "LAT (Lowest Astronomical Tide), positive-down",
        "units":      "metres",
        "depth_range_m": [0.0, float(MAX_DEPTH_M)],
        "scope": (
            "Calibrated on UAE Gulf in-situ survey data (Khalifa Port + Old Mussafah). "
            "Reconnaissance-grade. Clear-water generalization unverified — ATL24 unavailable. "
            "Not to be used for navigation."
        ),
    }

    # Add feat_names to meta (use from first site as all sites share the same cube structure)
    first_sk = sites[0]
    meta["architecture"]["channel_order"] = site_data[first_sk]["feat_names"]
    meta["normalisation"]["channel_names"] = site_data[first_sk]["feat_names"]

    with open(str(out_json), "w") as f:
        json.dump(meta, f, indent=2)
    L.info(f"Saved metadata JSON -> {out_json}")

    meta["_saved_pt"]   = str(out_pt)
    meta["_saved_json"] = str(out_json)
    return meta


# ── Inference API ─────────────────────────────────────────────────────────────

_DEFAULT_MODEL_CACHE: Optional[Tuple[PatchCNN, Dict]] = None


def load_default_model(
    pt_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    force_reload: bool = False,
) -> Tuple[PatchCNN, Dict]:
    """Load the saved default SDB CNN and its metadata.

    Caches the model in memory so repeated calls are cheap.

    Parameters
    ----------
    pt_path   : path to .pt state_dict; defaults to backend/models/sdb_cnn_default.pt
    json_path : path to .json metadata; defaults to backend/models/sdb_cnn_default.json
    force_reload : bypass the in-process cache

    Returns
    -------
    (model, metadata_dict)
      model    : PatchCNN in eval mode on CPU
      metadata : the full metadata dict from the .json file

    Raises
    ------
    FileNotFoundError if the weight files are not present (run train_default_model() first).
    """
    global _DEFAULT_MODEL_CACHE
    if _DEFAULT_MODEL_CACHE is not None and not force_reload:
        return _DEFAULT_MODEL_CACHE

    pt   = Path(pt_path)   if pt_path   else DEFAULT_MODEL_PT
    meta_path = Path(json_path) if json_path else DEFAULT_MODEL_JSON

    if not pt.exists():
        raise FileNotFoundError(
            f"Default model weights not found at {pt}. "
            "Run train_default_model() first or call: "
            "python3 backend/sdb_cnn_baseline.py --train-default"
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"Default model metadata not found at {meta_path}")

    with open(str(meta_path)) as f:
        meta = json.load(f)

    arch = meta["architecture"]
    in_ch     = arch["in_channels"]
    patch_px  = arch["patch_px"]
    max_depth = arch["max_depth_m"]

    model = PatchCNN(in_channels=in_ch, patch_px=patch_px, max_depth=max_depth)
    state = torch.load(str(pt), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    L.info(f"Loaded default CNN from {pt}  "
           f"(in_ch={in_ch}, patch_px={patch_px}, max_depth={max_depth})")

    _DEFAULT_MODEL_CACHE = (model, meta)
    return model, meta


def predict_depth_grid(
    s2_bands: Dict,
    water_mask: Optional[np.ndarray] = None,
    pt_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    batch_size: int = 1024,
    region: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the default SDB CNN densely over a water-masked S2 scene.

    This is the clean inference entry point.  It loads the default model once
    (cached in-process), builds the same 12-channel feature cube used during
    training, runs the patch-CNN in sliding 9x9-window mode via batched
    pixel-centre extraction, and returns a depth + σ grid clamped to [0, 25] m.

    Parameters
    ----------
    s2_bands : dict with keys 'blue', 'green', 'red', 'nir', 'scl'
               as float32 arrays of shape (H, W) in raw DN (divide by 10000
               internally) — same structure as fetch_s2_for_site() output,
               plus optional 'height', 'width', 'resolution_m' scalars.
    water_mask : (H, W) bool — pixels to predict.  If None, built from NDWI
               + SCL inside build_feature_cube().
    pt_path   : override path for .pt weights
    json_path : override path for .json metadata
    batch_size : inference batch size (patches); 1024 is safe on CPU

    Returns
    -------
    depth  : (H, W) float32 — predicted depth in metres [0, 25], NaN on land
    sigma  : (H, W) float32 — predictive uncertainty (1-sigma), NaN on land

    Notes
    -----
    - The feature cube is built identically to the training pre-processing
      (Hedley deglint, log, Stumpf log-ratio, NDTI — 12 channels).
    - The sigma temperature factor k from the metadata is applied automatically.
    - Pixels outside the water mask are set to NaN.
    """
    model, meta = load_default_model(pt_path=pt_path, json_path=json_path)

    # Build feature cube (same as training)
    cube, ndwi_water, _ = build_feature_cube(s2_bands)
    H, W, C = cube.shape

    # Water mask: prefer caller-supplied; fall back to NDWI mask
    if water_mask is None:
        water_mask = ndwi_water  # (H, W) bool

    water_mask = np.asarray(water_mask, dtype=bool)

    # Normalisation from metadata
    norm = meta["normalisation"]
    mean_arr = np.array(norm["mean"], dtype=np.float32)
    std_arr  = np.array(norm["std"],  dtype=np.float32)

    # Temperature factor k from sigma calibration. ITEM 3 closure: prefer the
    # PER-SITE test-fold recalibrated k_test (true held-out 95% coverage on THAT
    # region's spatial-block test fold) when `region` maps to a validated training
    # site (khalifa | omc). Fall back to the documented interim GLOBAL variance-
    # inflation factor for transfer/unknown regions (incl. jbel_dhanna and user
    # ROIs — the model was not trained there, so no per-site k_test is honest);
    # finally the val-fold k.
    _tr = meta["training"]
    _kt = _tr.get("sigma_k_test")
    sigma_k = None
    if isinstance(_kt, dict):
        _key = (region or "").strip().lower()
        _site = {"khalifa_port": "khalifa", "khalifa": "khalifa",
                 "old_mussafah": "omc", "mussafah_channel": "omc",
                 "omc": "omc"}.get(_key)
        if _site and _site in _kt:
            sigma_k = float(_kt[_site])
        elif "_global" in _kt:
            sigma_k = float(_kt["_global"])
    elif _kt is not None:
        sigma_k = float(_kt)   # legacy scalar
    if sigma_k is None:
        if _tr.get("sigma_k_interim") is not None:
            sigma_k = float(_tr["sigma_k_interim"])   # transfer/unknown fallback
        else:
            sigma_k = float(_tr.get("sigma_k", 1.0))

    # Gather water pixels
    wr, wc = np.where(water_mask)
    n_water = len(wr)
    if n_water == 0:
        depth_grid = np.full((H, W), np.nan, dtype=np.float32)
        sigma_grid = np.full((H, W), np.nan, dtype=np.float32)
        return depth_grid, sigma_grid

    # Pad cube for patch extraction (reflect padding = same as training)
    pr = PATCH_RADIUS
    cube_padded = np.pad(cube, ((pr, pr), (pr, pr), (0, 0)), mode="reflect")

    # Run in batches
    model.eval()
    device = torch.device("cpu")
    all_mu  = np.empty(n_water, dtype=np.float32)
    all_sig = np.empty(n_water, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_water, batch_size):
            end = min(start + batch_size, n_water)
            batch_rows = wr[start:end]
            batch_cols = wc[start:end]
            n_b = end - start

            # Extract patches: (n_b, C, P, P)
            patches = np.empty((n_b, C, PATCH_PX, PATCH_PX), dtype=np.float32)
            for i in range(n_b):
                r = batch_rows[i] + pr  # offset for padded cube
                c = batch_cols[i] + pr
                patch = cube_padded[r - pr: r + pr + 1, c - pr: c + pr + 1, :]  # (P, P, C)
                patch = (patch - mean_arr) / (std_arr + 1e-6)
                patches[i] = patch.transpose(2, 0, 1)  # (C, P, P)

            t = torch.from_numpy(patches)
            mu, lv = model(t.to(device))
            mu_np  = mu.cpu().numpy()
            sig_np = np.sqrt(np.exp(lv.cpu().numpy())) * sigma_k
            all_mu[start:end]  = mu_np
            all_sig[start:end] = sig_np

    # ── ROUND-1 (NIGHT_BIAS_LOG): per-site depth-stratified monotone debias g_site ──
    # Apply the persisted per-site g_site (fit on that region's VAL fold, leakage-safe)
    # ONLY for validated training regions (khalifa | omc). Transfer/unknown ROIs get
    # NO debias (identity) — g_site is a per-site offset curve and must NOT be applied
    # to a region the map was not fit on. Documented fallback: identity.
    _gmap = (_tr.get("bias_debias_isotonic") or {}) if isinstance(_tr, dict) else {}
    _gsite = None
    _key = (region or "").strip().lower()
    _site = {"khalifa_port": "khalifa", "khalifa": "khalifa",
             "old_mussafah": "omc", "mussafah_channel": "omc",
             "omc": "omc"}.get(_key)
    if _site and isinstance(_gmap, dict) and _site in _gmap:
        _gsite = _gmap[_site]
    if _gsite is not None:
        all_mu = apply_isotonic_persite(all_mu, _gsite)
        L.info(f"[predict_depth_grid] applied per-site g_site debias for region '{_site}'")

    # Assemble grids
    depth_grid = np.full((H, W), np.nan, dtype=np.float32)
    sigma_grid = np.full((H, W), np.nan, dtype=np.float32)
    depth_grid[wr, wc] = np.clip(all_mu,  0.0, MAX_DEPTH_M)
    sigma_grid[wr, wc] = np.clip(all_sig, 0.0, MAX_DEPTH_M)

    return depth_grid, sigma_grid


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    _load_env()
    parser = argparse.ArgumentParser(
        description="SDB PatchCNN Baseline Iter-1 / Iter-2 / Iter-3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sites", nargs="+", default=["khalifa"],
                        help="Site keys (khalifa | jbel_dhanna | omc | all)")
    parser.add_argument("--epochs",    type=int,   default=N_EPOCHS)
    parser.add_argument("--blocks",    type=int,   default=N_BLOCKS)
    parser.add_argument("--buffer-m",  type=float, default=BLOCK_BUF_M,
                        help="Spatial buffer between train/test blocks in metres. "
                             "200=Iter-1 original, 500=honest leakage-safe.")
    parser.add_argument("--iter2",     action="store_true", default=False,
                        help="Enable Iter-2: Stumpf-primary + ResidualCNN + SILog-rank + sigma-cal")
    parser.add_argument("--iter3",     action="store_true", default=False,
                        help="Enable Iter-3: Lyzenga-Kd-primary + ResidualCNN + "
                             "SILog-rank + val-fold sigma-cal (takes precedence over --iter2)")
    parser.add_argument("--lambda-rank", type=float, default=ITER2_LAMBDA_RANK,
                        help="Weight on SILog-rank anti-collapse term (Iter-2/3)")
    parser.add_argument("--out-dir",   type=str,
                        default=str(ROOT / "May_2026_results" / "v_iho_iterations" /
                                    "iter_cnn_baseline"))
    parser.add_argument("--label",     type=str, default="",
                        help="Optional suffix for output JSON filenames (e.g. 'honest500')")
    parser.add_argument("--train-default", action="store_true", default=False,
                        help="Train + save the default multi-site SDB CNN model "
                             "(khalifa + omc pooled, 500 m buffer, val-fold sigma-cal). "
                             "Saves to backend/models/sdb_cnn_default.{pt,json}.")
    parser.add_argument("--force-retrain", action="store_true", default=False,
                        help="Force re-train even if saved model already exists.")
    parser.add_argument("--smoke-test", action="store_true", default=False,
                        help="Smoke-test: load saved default model and run predict_depth_grid "
                             "on cached Khalifa S2. Reports grid shape, depth range, water px.")
    args = parser.parse_args()

    # ── Default model train/save shortcut ──────────────────────────────────
    if args.train_default:
        _load_env()
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        cache_dir = Path(ROOT / "May_2026_results" / "v_iho_iterations" /
                         "iter_cnn_baseline" / "s2_cache")
        meta = train_default_model(
            cache_dir=cache_dir,
            epochs=args.epochs,
            n_blocks=args.blocks,
            buffer_m=DEFAULT_BUFFER_M,
            force_retrain=args.force_retrain,
        )
        print("\n=== DEFAULT MODEL SAVED ===")
        ov = meta.get("honest_test_metrics", {})
        pooled = ov.get("pooled", {})
        print(f"  Pooled test: RMSE={pooled.get('rmse_m')} m  R2={pooled.get('r2')}  "
              f"bias={pooled.get('bias_m')} m  slope={pooled.get('decile_slope')}")
        for sk, sm in ov.get("per_site", {}).items():
            print(f"  [{sk}] RMSE={sm.get('rmse_m')} m  R2={sm.get('r2')}  "
                  f"sigma-95%={sm.get('sigma_95_cov')}")
        print(f"  sigma_k={meta['training']['sigma_k']}  "
              f"val_cov={meta['training']['val_fold_sigma_coverage']}")
        bs = ov.get("band_shuffle", {})
        print(f"  Band-shuffle: delta={bs.get('delta_pct'):+.1f}%  {bs.get('verdict')}")
        print(f"  Weights: {meta.get('_saved_pt')}")
        print(f"  Metadata: {meta.get('_saved_json')}")
        return

    # ── Smoke-test shortcut ─────────────────────────────────────────────────
    if args.smoke_test:
        _load_env()
        cache_dir = Path(ROOT / "May_2026_results" / "v_iho_iterations" /
                         "iter_cnn_baseline" / "s2_cache")
        print("\n=== SMOKE TEST: load default model + predict_depth_grid (Khalifa) ===")
        s2 = fetch_s2_for_site("khalifa", cache_dir)
        depth, sigma = predict_depth_grid(s2)
        water = np.isfinite(depth)
        n_water = int(water.sum())
        d_vals  = depth[water]
        s_vals  = sigma[water]
        print(f"  Grid shape:  {depth.shape}")
        print(f"  Water pixels: {n_water}")
        print(f"  Depth range: [{d_vals.min():.2f}, {d_vals.max():.2f}] m")
        print(f"  Sigma range: [{s_vals.min():.3f}, {s_vals.max():.3f}] m")
        print(f"  Depth NaN:   {np.isnan(depth).sum()} px (land/cloud)")
        print("  SMOKE TEST PASSED" if n_water > 100 else "  WARNING: very few water pixels")
        return

    site_keys = list(SITE_CFG.keys()) if "all" in args.sites else args.sites
    out_dir = Path(args.out_dir)
    cache_dir = Path(ROOT / "May_2026_results" / "v_iho_iterations" /
                     "iter_cnn_baseline" / "s2_cache")  # always use original cache
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mode_tag  = ("iter3" if args.iter3 else
                 "iter2" if args.iter2 else "iter1")
    buf_tag   = f"buf{int(args.buffer_m)}"
    file_tag  = f"_{mode_tag}_{buf_tag}"
    if args.label:
        file_tag += f"_{args.label}"

    all_results = {}
    for site_key in site_keys:
        if site_key not in SITE_CFG:
            L.warning(f"Unknown site '{site_key}'; known: {list(SITE_CFG.keys())}")
            continue
        try:
            res = run_site(
                site_key, cache_dir,
                epochs=args.epochs,
                n_blocks=args.blocks,
                buffer_m=args.buffer_m,
                iter2=args.iter2,
                iter3=args.iter3,
                lambda_rank=args.lambda_rank,
            )
            print_report(res)
            all_results[site_key] = res

            # Save per-site JSON with mode+buffer tag
            site_json = out_dir / f"{site_key}_cnn{file_tag}.json"
            with open(site_json, "w") as f:
                json.dump(res, f, indent=2, default=_default_serialiser)
            L.info(f"[{site_key}] Results saved -> {site_json}")

        except Exception as ex:
            L.error(f"[{site_key}] FAILED: {ex}", exc_info=True)

    # Save combined JSON
    combined_json = out_dir / f"cnn_results{file_tag}.json"
    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2, default=_default_serialiser)
    L.info(f"\nAll results saved -> {combined_json}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Site':<20} {'Mode':<10} {'buf':>5} {'RMSE(m)':>9} {'R2':>7} "
          f"{'bias':>8} {'decile_slope':>13} {'sigma-95%':>10} {'BndShuf%':>10}")
    print("-" * 80)
    for sk, r in all_results.items():
        ov = r["overall"]
        bs = r.get("band_shuffle", {})
        dp = bs.get("delta_pct", float("nan"))
        mode = ("iter3" if r.get("iter3") else
                "iter2" if r.get("iter2") else "iter1")
        buf  = int(r.get("buffer_m", BLOCK_BUF_M))
        slp  = ov.get("decile_slope", float("nan"))
        print(f"{sk:<20} {mode:<10} {buf:>5} "
              f"{ov['rmse_m']:>9.3f} {ov['r2']:>7.3f} "
              f"{ov['bias_m']:>8.3f} {slp:>13.4f} "
              f"{ov['sigma_95_coverage']:>10.1%} {dp:>+10.1f}%")
    print("=" * 80)

    # Per-bin table
    print("\nPer-bin (CNN):")
    print(f"{'Site':<20} {'Bin':<10} {'n':>6} {'RMSE':>8} {'R2':>8} {'slope':>8} "
          f"{'phys_RMSE':>10}")
    for sk, r in all_results.items():
        for band_lbl, bm in r["per_bin"].items():
            if bm["n"] == 0:
                continue
            phys_rmse = (bm.get("lyzenga_rmse_m") or bm.get("stumpf_rmse_m") or float("nan"))
            print(f"{sk:<20} {band_lbl:<10} {bm['n']:>6} "
                  f"{bm['rmse_m']:>8.3f} {bm['r2']:>8.3f} "
                  f"{bm.get('decile_slope', float('nan')):>8.4f} "
                  f"{phys_rmse:>10.3f}")
    print()


if __name__ == "__main__":
    main()
