#!/usr/bin/env python3
"""Train the UAE Cluster + Random Forest SDB model.

Reads the bundled XYZ soundings (KP, KP_EMAL, OMC), fetches a
representative cloud-free Sentinel-2 scene over each region,
samples spectral features at every sounding location, then trains
backend.uae_pretrained.UAEModel and pickles it under
backend/models/uae_clustered_rf.pkl.

Usage:
    source .venv/bin/activate && source .env
    python train_uae_pretrained.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend import uae_pretrained as UAE  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
L = logging.getLogger("train_uae")

# TRAIN-R2: source weights, consistent with very_hr_augment.py's convention
# (W_SLIDERULE=5.0 there). ATL24 is refraction-corrected in-product (higher
# quality than raw SlideRule ATL03), so it outranks in-situ XYZ slightly.
W_ATL24 = 5.5
W_INSITU = 5.0
W_IBOATING = 3.0  # TRAIN-R8 — weakest of the three sources, OCR chart soundings

# iboating_database/ uses different folder names than the REGIONS key.
IBOATING_DIR_FOR_REGION = {
    "khalifa_port": "khalifa",
    "old_mussafah": "old_mussafah",
    "jbel_dhanna": "jbel_dhanna",
}

# TRAIN-R2: 6 new regions folded in from cache/atl24_r2/ (class_ph==40,
# verified bathymetry — see backend/very_hr_augment.py ATL24_CLASS_BATHY).
ATL24_R2_SITES = ["bu_tinah", "yas_lagoon", "marawah", "ras_ghanada",
                  "eastern_mangroves", "lulu_lagoon"]

# Region definitions — bbox padded by 1 km around the sounding extent
REGIONS = [
    {
        "key": "khalifa_port",
        "xyz_files": [("validation/KP Basin Soundings 10m.xyz", "KP_Basin"),
                      ("validation/KP_EMAL_Soundings_10x_New.xyz", "KP_EMAL")],
        "shapefiles": [],
        "s2_window": ("2024-04-01", "2024-09-30"),
    },
    {
        "key": "old_mussafah",
        "xyz_files": [("validation/OMC_Soundings_20m.xyz", "OMC")],
        "shapefiles": [],
        "s2_window": ("2024-04-01", "2024-09-30"),
    },
    {
        "key": "jbel_dhanna",
        "xyz_files": [],
        "shapefiles": [("validation/swot_dhanna/SWOT_Dhanna.shp",
                        "SWOT_Dhanna", 32639)],  # UTM Zone 39N
        "s2_window": ("2024-04-01", "2024-09-30"),
    },
] + [
    {"key": site, "xyz_files": [], "shapefiles": [], "atl24_r2": True,
     "s2_window": ("2024-04-01", "2024-09-30")}
    for site in ATL24_R2_SITES
]


def _utm_to_lonlat(easting: np.ndarray, northing: np.ndarray,
                   epsg: int = 32640) -> Tuple[np.ndarray, np.ndarray]:
    from pyproj import CRS, Transformer
    tr = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326),
                              always_xy=True)
    return tr.transform(easting, northing)


def _load_atl24_r2(region_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TRAIN-R2: load cache/atl24_r2/<region_key>/atl24_tracks.npz (class_ph==40,
    verified bathymetry — see very_hr_augment.ATL24_CLASS_BATHY)."""
    fpath = ROOT / "cache" / "atl24_r2" / region_key / "atl24_tracks.npz"
    if not fpath.exists():
        return np.array([]), np.array([]), np.array([])
    d = np.load(fpath, allow_pickle=True)
    lats, lons, deps = d["lats"], d["lons"], d["depths"]
    valid = (deps > 0.5) & (deps < 25.0) & np.isfinite(lats) & np.isfinite(lons)
    L.info(f"  atl24_r2/{region_key}: {int(valid.sum()):,} usable pts (class_ph==40)")
    return lats[valid], lons[valid], deps[valid]


def _load_iboating(region_key: str, min_confidence: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TRAIN-R8: load iboating_database/<dir>/soundings.csv, confidence >= 0.9
    only (CLAUDE.md gotcha — do not relax)."""
    dirname = IBOATING_DIR_FOR_REGION.get(region_key)
    if not dirname:
        return np.array([]), np.array([]), np.array([])
    fpath = ROOT / "iboating_database" / dirname / "soundings.csv"
    if not fpath.exists():
        return np.array([]), np.array([]), np.array([])
    import csv
    lats, lons, deps = [], [], []
    with open(fpath) as fh:
        for row in csv.DictReader(fh):
            try:
                conf = float(row["confidence"])
                if conf < min_confidence:
                    continue
                d = abs(float(row["depth"]))
                if not (0.5 < d < 25.0):
                    continue
                lats.append(float(row["lat"])); lons.append(float(row["lon"])); deps.append(d)
            except (KeyError, ValueError):
                continue
    L.info(f"  iboating/{dirname}: {len(deps):,} usable pts (confidence>={min_confidence})")
    return np.array(lats), np.array(lons), np.array(deps)


def _load_region_xyz(region: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (lats, lons, deps, weights) — weights encode source (in-situ
    XYZ / ATL24 / i-Boating OCR) per TRAIN-R2/R8's source-weighting scheme."""
    lats_all, lons_all, deps_all, w_all = [], [], [], []
    for fname, _src in region.get("xyz_files", []):
        fpath = ROOT / fname
        if not fpath.exists():
            L.warning(f"  missing {fname}, skipping")
            continue
        data = np.loadtxt(fpath, usecols=(0, 1, 2))
        lons, lats = _utm_to_lonlat(data[:, 0], data[:, 1], epsg=32640)
        deps = np.abs(data[:, 2])
        valid = (deps > 0.5) & (deps < UAE.MAX_DEPTH_M) & np.isfinite(lats) & np.isfinite(lons)
        lats_all.append(lats[valid])
        lons_all.append(lons[valid])
        deps_all.append(deps[valid])
        w_all.append(np.full(int(valid.sum()), W_INSITU))
        L.info(f"  {fname}: {int(valid.sum()):,} usable pts "
               f"(0.5–{UAE.MAX_DEPTH_M}m)")
    # Shapefiles (e.g. SWOT_Dhanna)
    for entry in region.get("shapefiles", []):
        path, src, epsg = entry
        fpath = ROOT / path
        if not fpath.exists():
            L.warning(f"  missing {path}, skipping")
            continue
        try:
            import geopandas as gpd
            g = gpd.read_file(fpath)
            if g.crs is not None and g.crs.to_epsg() != 4326:
                g = g.to_crs("EPSG:4326")
            # Z column or fall back to geometry.z
            if "Z" in g.columns:
                deps = np.abs(g["Z"].astype(float).values)
            elif "depth" in g.columns:
                deps = np.abs(g["depth"].astype(float).values)
            else:
                raise RuntimeError(f"{path}: no Z/depth column found")
            lats = g.geometry.y.values
            lons = g.geometry.x.values
            valid = (deps > 0.5) & (deps < UAE.MAX_DEPTH_M) & np.isfinite(lats) & np.isfinite(lons)
            lats_all.append(lats[valid])
            lons_all.append(lons[valid])
            deps_all.append(deps[valid])
            w_all.append(np.full(int(valid.sum()), W_INSITU))
            L.info(f"  {path}: {int(valid.sum()):,} usable pts "
                   f"(0.5–{UAE.MAX_DEPTH_M}m, {src})")
        except Exception as ex:
            L.warning(f"  {path} load failed: {ex}")
    # TRAIN-R2: ATL24 (class_ph==40), and TRAIN-R8: i-Boating OCR (confidence>=0.9)
    if region.get("atl24_r2"):
        a_lats, a_lons, a_deps = _load_atl24_r2(region["key"])
        if len(a_deps):
            lats_all.append(a_lats); lons_all.append(a_lons); deps_all.append(a_deps)
            w_all.append(np.full(len(a_deps), W_ATL24))
    i_lats, i_lons, i_deps = _load_iboating(region["key"])
    if len(i_deps):
        lats_all.append(i_lats); lons_all.append(i_lons); deps_all.append(i_deps)
        w_all.append(np.full(len(i_deps), W_IBOATING))
    if not deps_all:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.concatenate(lats_all), np.concatenate(lons_all),
            np.concatenate(deps_all), np.concatenate(w_all))


def _bbox_for_pts(lats: np.ndarray, lons: np.ndarray, pad_deg: float = 0.01) -> List[float]:
    return [float(lons.min() - pad_deg), float(lats.min() - pad_deg),
            float(lons.max() + pad_deg), float(lats.max() + pad_deg)]


def _sample_features_at_points(s2: Dict, bbox: List[float],
                               lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature stack and sample at the (lat, lon) of each sounding.

    Returns (features, in_water_mask). features is (n, N_FEATURES);
    in_water_mask is True where the pixel was classified as water.
    """
    feats, water = UAE.build_feature_stack(s2)
    H, W, F = feats.shape
    w_, s_, e_, n_ = bbox
    rows = ((n_ - lats) / (n_ - s_ + 1e-12) * H).astype(int)
    cols = ((lons - w_) / (e_ - w_ + 1e-12) * W).astype(int)
    inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows = np.clip(rows, 0, H - 1)
    cols = np.clip(cols, 0, W - 1)
    sampled = feats[rows, cols, :]
    on_water = water[rows, cols] & inside
    finite = np.all(np.isfinite(sampled), axis=1)
    return sampled, on_water & finite


_GEE_CACHE_DIR = ROOT / "cache" / "s2_gee_trainer"
_GEE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_s2_any(bbox, sd, ed, res=10, cloud=20):
    """Cache-first S2 fetch: disk-cached Sentinel-Hub fetch, falling back to
    GEE (reachable in environments where SH credentials are unavailable —
    not a "live Sentinel-Hub call", a different provider). GEE's
    getDownloadURL caps a single request at 50 MB; on that specific error,
    retry once at 20 m (roughly a quarter the payload). The GEE fallback
    result is cached separately (own disk cache) so repeated calls for the
    same bbox/window don't re-hit the network."""
    from backend.app import _fetch_s2_cached
    try:
        return _fetch_s2_cached(bbox, sd, ed, res=res, cloud=cloud)
    except Exception as ex:
        L.warning(f"  SH fetch failed ({ex}); falling back to GEE")

    import hashlib
    import pickle
    key = hashlib.sha1(f"{tuple(round(float(v), 4) for v in bbox)}|{sd}|{ed}|{res}|{cloud}"
                       .encode()).hexdigest()[:16]
    cpath = _GEE_CACHE_DIR / f"{key}.pkl"
    if cpath.exists():
        with open(cpath, "rb") as fh:
            return pickle.load(fh)

    from backend.app import fetch_s2_gee
    try:
        s2 = fetch_s2_gee(bbox, sd, ed, res=res, cloud=cloud)
    except Exception as ex:
        if "50331648" in str(ex) or "must be less than" in str(ex):
            L.warning(f"  GEE 50MB cap hit at {res}m; retrying at 20m")
            s2 = fetch_s2_gee(bbox, sd, ed, res=20, cloud=cloud)
        else:
            raise
    with open(cpath, "wb") as fh:
        pickle.dump(s2, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return s2


def main():
    all_features: List[np.ndarray] = []
    all_depths: List[np.ndarray] = []
    all_weights: List[np.ndarray] = []
    region_summaries: List[Dict] = []

    for region in REGIONS:
        L.info(f"=== {region['key']} ===")
        lats, lons, deps, wts = _load_region_xyz(region)
        if len(deps) == 0:
            continue
        bbox = _bbox_for_pts(lats, lons)
        L.info(f"  bbox = {bbox}, total {len(deps):,} pts")
        sd, ed = region["s2_window"]
        L.info(f"  fetching S2 {sd}→{ed}…")
        t0 = time.time()
        try:
            s2 = _fetch_s2_any(bbox, sd, ed, res=10, cloud=20)
        except Exception as ex:
            L.warning(f"  {region['key']}: S2 fetch failed ({ex}), skipping region")
            continue
        L.info(f"  S2 fetched in {time.time()-t0:.1f}s "
               f"({s2['width']}×{s2['height']} px)")

        feats, mask = _sample_features_at_points(s2, bbox, lats, lons)
        n_in = int(mask.sum())
        L.info(f"  on-water samples kept: {n_in:,} / {len(deps):,}")
        if n_in < 200:
            L.warning(f"  {region['key']}: too few on-water samples, skipping")
            continue
        all_features.append(feats[mask])
        all_depths.append(deps[mask])
        all_weights.append(wts[mask])
        region_summaries.append({
            "region": region["key"],
            "bbox": bbox, "s2_window": [sd, ed],
            "n_samples": n_in,
            "depth_range_m": [float(deps[mask].min()), float(deps[mask].max())],
            "depth_mean_m": float(deps[mask].mean()),
        })

    if not all_features:
        raise SystemExit("No regions produced training data — aborting")

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.concatenate(all_depths, axis=0).astype(np.float32)
    sw = np.concatenate(all_weights, axis=0).astype(np.float32)
    L.info(f"COMBINED training set: {len(y):,} pts, "
           f"depth range {y.min():.2f}–{y.max():.2f}m, "
           f"regions={[r['region'] for r in region_summaries]}")

    # TRAIN-R2: n_estimators/max_depth trimmed from the pre-expansion defaults
    # (120/14) — with 8x more training points, the same params produced a
    # 103 MB pickle (coordinator-flagged bloat concern). Re-tuned to hold
    # in-sample RMSE within the retrain guard's 10% tolerance.
    model = UAE.fit_uae_model(X, y, n_clusters=6,
                              rf_n_estimators=60, rf_max_depth=12,
                              sample_weight=sw)
    model.meta["regions"] = region_summaries
    out = UAE.save_model(model)
    L.info(f"DONE → {out}")


if __name__ == "__main__":
    main()
