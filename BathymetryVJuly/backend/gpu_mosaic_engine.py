"""
GPU-accelerated mosaic bathymetry engine.

For large regions: splits into sub-tiles, selects best Sentinel-2 scenes
(last 4 weeks, 10m, lowest cloud), runs CNN on GPU, mosaics with
feathered overlap blending.
"""
from __future__ import annotations

import io
import logging
import math
import time as TM
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests
import tifffile
import torch

L = logging.getLogger("bathy.mosaic")
MAX_DEPTH_M = 25.0

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L.info(f"Mosaic engine device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# 1. SENTINEL HUB AUTH (shared with app.py)
# ══════════════════════════════════════════════════════════════
import os

SH_AUTH = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROC = "https://sh.dataspace.copernicus.eu/api/v1/process"
SH_CATALOG = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
_tok = {"t": None, "exp": 0}


def _sh_token():
    if _tok["t"] and TM.time() < _tok["exp"] - 60:
        return _tok["t"]
    cid = os.getenv("SH_CLIENT_ID", "")
    csec = os.getenv("SH_CLIENT_SECRET", "")
    if not cid:
        raise RuntimeError("SH_CLIENT_ID not set")
    r = requests.post(SH_AUTH, data={
        "grant_type": "client_credentials",
        "client_id": cid, "client_secret": csec
    }, timeout=30)
    r.raise_for_status()
    d = r.json()
    _tok["t"] = d["access_token"]
    _tok["exp"] = TM.time() + d["expires_in"]
    return _tok["t"]


# ══════════════════════════════════════════════════════════════
# 2. CATALOG: find best scenes in last 4 weeks
# ══════════════════════════════════════════════════════════════

def search_best_scenes(bbox: list, n_weeks: int = 4, max_cloud: float = 20,
                       top_k: int = 4) -> list[dict]:
    """
    Query CDSE catalog for S2-L2A scenes covering bbox in the last n_weeks.
    Returns up to top_k scenes sorted by cloud cover ascending.
    Each dict: {id, date, cloud_pct, geometry}.
    """
    w, s, e, n = bbox
    ed = datetime.utcnow()
    sd = ed - timedelta(weeks=n_weeks)
    tok = _sh_token()

    body = {
        "bbox": [w, s, e, n],
        "datetime": f"{sd.strftime('%Y-%m-%dT00:00:00Z')}/{ed.strftime('%Y-%m-%dT23:59:59Z')}",
        "collections": ["sentinel-2-l2a"],
        "limit": 50,
        "filter": f"eo:cloud_cover<{max_cloud}",
        "filter-lang": "cql2-text",
        "fields": {
            "include": ["properties.datetime", "properties.eo:cloud_cover", "id"],
        },
    }

    r = requests.post(SH_CATALOG, headers={
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
    }, json=body, timeout=60)

    scenes = []
    if r.ok:
        features = r.json().get("features", [])
        for f in features:
            props = f.get("properties", {})
            scenes.append({
                "id": f.get("id", ""),
                "date": props.get("datetime", "")[:10],
                "cloud_pct": props.get("eo:cloud_cover", 99),
            })
        # Sort by cloud cover, take best
        scenes.sort(key=lambda x: x["cloud_pct"])
        L.info(f"Catalog: {len(features)} scenes found, top {top_k} cloud covers: "
               f"{[s['cloud_pct'] for s in scenes[:top_k]]}")
    else:
        L.warning(f"Catalog search failed {r.status_code}: {r.text[:200]}")

    return scenes[:top_k]


# ══════════════════════════════════════════════════════════════
# 3. EVALSCRIPT — 10m, best-pixel composite (min-cloud)
# ══════════════════════════════════════════════════════════════

EVALSCRIPT_10M = """//VERSION=3
function setup(){
  return {
    input:[{bands:["B02","B03","B04","B08","SCL"],units:"DN",mosaicking:"ORBIT"}],
    output:{bands:5,sampleType:"UINT16"}
  };
}
function isValid(s){
  var c=s.SCL;
  return c!==0&&c!==1&&c!==3&&c!==8&&c!==9&&c!==10&&c!==11;
}
function scoreScene(s){
  // Lower SCL = better; 4,5=vegetation/bare -> good; 6=water -> best
  var c=s.SCL;
  if(c===6) return 100;
  if(c===4||c===5) return 80;
  if(c===2||c===7) return 50;
  return 10;
}
function evaluatePixel(samples){
  // Pick the best valid pixel across orbits (highest score = clearest)
  var bestB02=0,bestB03=0,bestB04=0,bestB08=0,bestScore=-1;
  for(var i=0;i<samples.length;i++){
    if(!isValid(samples[i])) continue;
    var sc=scoreScene(samples[i]);
    if(sc>bestScore){
      bestScore=sc;
      bestB02=samples[i].B02; bestB03=samples[i].B03;
      bestB04=samples[i].B04; bestB08=samples[i].B08;
    }
  }
  if(bestScore<0 && samples.length>0){
    bestB02=samples[0].B02; bestB03=samples[0].B03;
    bestB04=samples[0].B04; bestB08=samples[0].B08;
    bestScore=0;
  }
  return [bestB04,bestB03,bestB02,bestB08,bestScore];
}"""


def fetch_s2_10m(bbox: list, sd: str, ed: str, cloud: int = 20) -> dict:
    """
    Fetch S2 at 10m with best-pixel selection. Returns dict with
    red, green, blue, nir, ndwi, quality, width, height arrays.
    """
    w, s, e, n = bbox
    cl = math.cos(math.radians((n + s) / 2))
    wp = max(32, min(2500, int(abs(e - w) * 111000 * cl / 10)))
    hp = max(32, min(2500, int(abs(n - s) * 111000 / 10)))
    tok = _sh_token()

    body = {
        "input": {
            "bounds": {
                "bbox": [w, s, e, n],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "maxCloudCoverage": cloud,
                    "timeRange": {"from": f"{sd}T00:00:00Z", "to": f"{ed}T23:59:59Z"}
                }
            }]
        },
        "output": {
            "width": wp, "height": hp,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": EVALSCRIPT_10M,
    }

    L.info(f"S2@10m: {wp}x{hp} [{sd} → {ed}]")
    r = requests.post(SH_PROC, headers={
        "Authorization": f"Bearer {tok}", "Content-Type": "application/json"
    }, json=body, timeout=300)

    if not r.ok:
        raise RuntimeError(f"S2@10m {r.status_code}: {r.text[:200]}")

    img = tifffile.imread(io.BytesIO(r.content))
    if img.ndim == 3 and img.shape[0] == 5:
        red, green, blue, nir, quality = img[0], img[1], img[2], img[3], img[4]
    elif img.ndim == 3 and img.shape[2] == 5:
        red, green, blue, nir, quality = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3], img[:,:,4]
    else:
        raise RuntimeError(f"TIFF shape {img.shape}")

    g_f = green.astype(np.float32)
    n_f = nir.astype(np.float32)
    ndwi = (g_f - n_f) / (g_f + n_f + 1e-6)

    return {
        "red": red, "green": green, "blue": blue, "nir": nir,
        "ndwi": ndwi, "quality": quality,
        "width": blue.shape[1], "height": blue.shape[0],
    }


# ══════════════════════════════════════════════════════════════
# 4. GEBCO fetch (same logic as app.py but standalone)
# ══════════════════════════════════════════════════════════════

def _fetch_gebco(bbox: list) -> Optional[dict]:
    w, s, e, n = bbox
    wp = max(16, min(500, int((e - w) * 240)))
    hp = max(16, min(500, int((n - s) * 240)))
    url = (f"https://wms.gebco.net/mapserv?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
           f"&COVERAGE=gebco_latest&CRS=EPSG:4326&FORMAT=GeoTIFF"
           f"&BBOX={s},{w},{n},{e}&WIDTH={wp}&HEIGHT={hp}")
    try:
        r = requests.get(url, timeout=30)
        if r.ok and len(r.content) > 200:
            elev = tifffile.imread(io.BytesIO(r.content)).astype(np.float32)
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
                return {"lats": np.array(lats), "lons": np.array(lons),
                        "depths": np.array(depths)}
    except Exception as ex:
        L.warning(f"GEBCO tile: {ex}")
    return None


# ══════════════════════════════════════════════════════════════
# 5. GPU feature stack + CNN inference
# ══════════════════════════════════════════════════════════════

def _build_feature_stack_gpu(s2: dict) -> torch.Tensor:
    """
    Build (1, 9, H, W) feature tensor directly on GPU.
    Channels: blue, green, red, nir, ndwi, ln(B/G), ln(G/R), B/G, G/R
    """
    eps = 1e-6
    blue = torch.from_numpy(s2["blue"].astype(np.float32) / 10000).to(DEVICE).clamp(min=eps)
    green = torch.from_numpy(s2["green"].astype(np.float32) / 10000).to(DEVICE).clamp(min=eps)
    red = torch.from_numpy(s2["red"].astype(np.float32) / 10000).to(DEVICE).clamp(min=eps)
    nir = torch.from_numpy(s2["nir"].astype(np.float32) / 10000).to(DEVICE).clamp(min=eps)
    ndwi = torch.from_numpy(s2["ndwi"].astype(np.float32)).to(DEVICE)

    lnBG = torch.log(blue + eps) / torch.log(green + eps)
    lnGR = torch.log(green + eps) / torch.log(red + eps)
    BG = blue / (green + eps)
    GR = green / (red + eps)

    stack = torch.stack([blue, green, red, nir, ndwi, lnBG, lnGR, BG, GR], dim=0)  # (9,H,W)

    # Per-channel percentile normalisation on GPU
    for c in range(9):
        ch = stack[c]
        finite = ch[torch.isfinite(ch)]
        if finite.numel() < 10:
            stack[c] = 0.0
            continue
        lo = torch.quantile(finite, 0.02)
        hi = torch.quantile(finite, 0.98)
        if hi - lo < 1e-8:
            hi = lo + 1.0
        stack[c] = ((ch - lo) / (hi - lo)).clamp(0.0, 1.0)

    stack = torch.nan_to_num(stack, nan=0.0)
    return stack.unsqueeze(0)  # (1,9,H,W)


def _gpu_pad(t: torch.Tensor, div: int = 16):
    _, _, h, w = t.shape
    ph = (div - h % div) % div
    pw = (div - w % div) % div
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    return t, h, w


def _rasterise_ref_gpu(ref_pts: dict, bbox: list, H: int, W: int) -> torch.Tensor:
    """Rasterise sparse reference points onto GPU tensor. NaN where no data."""
    w, s, e, n = bbox
    target = torch.full((H, W), float("nan"), device=DEVICE, dtype=torch.float32)
    lats, lons, depths = ref_pts["lats"], ref_pts["lons"], ref_pts["depths"]
    count = 0
    for i in range(len(lats)):
        r = max(0, min(H - 1, int((n - lats[i]) / (n - s + 1e-10) * H)))
        c = max(0, min(W - 1, int((lons[i] - w) / (e - w + 1e-10) * W)))
        d = depths[i]
        if np.isfinite(d) and 0 < d <= MAX_DEPTH_M:
            target[r, c] = min(d, MAX_DEPTH_M)
            count += 1
    return target, count


def gpu_cnn_predict(s2: dict, ref_pts: dict, bbox: list,
                    epochs: int = 60, base_features: int = 32,
                    patch_size: int = 64, lr: float = 3e-4) -> Optional[np.ndarray]:
    """
    Train U-Net on GPU with reference depths, predict full tile.
    Returns depth (H, W) numpy array or None.
    """
    from cnn_engine import BathyUNet

    H, W = s2["red"].shape
    features = _build_feature_stack_gpu(s2)  # (1,9,H,W) on GPU

    water = torch.from_numpy((s2["ndwi"] > 0).astype(np.float32)).to(DEVICE)
    target, n_valid = _rasterise_ref_gpu(ref_pts, bbox, H, W)

    if n_valid < 10:
        L.warning(f"Tile: only {n_valid} ref pts, skipping CNN")
        return None

    # Normalise target
    valid_mask = torch.isfinite(target) & (target > 0)
    valid_depths = target[valid_mask]
    depth_scale = min(float(torch.quantile(valid_depths, 0.98).item()), MAX_DEPTH_M)
    if depth_scale < 1:
        depth_scale = MAX_DEPTH_M

    target_scaled = torch.nan_to_num(target / depth_scale, nan=0.0).unsqueeze(0).unsqueeze(0)
    mask_t = valid_mask.float().unsqueeze(0).unsqueeze(0)

    # Model on GPU
    model = BathyUNet(in_channels=9, base_features=base_features).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    # Extract patches on GPU
    ps = min(patch_size, H, W)
    feat_patches, tgt_patches, mask_patches = [], [], []
    attempts = 0
    while len(feat_patches) < 512 and attempts < 5120:
        y0 = np.random.randint(0, max(H - ps, 1))
        x0 = np.random.randint(0, max(W - ps, 1))
        m = mask_t[:, :, y0:y0+ps, x0:x0+ps]
        if m.sum() < 2:
            attempts += 1
            continue
        feat_patches.append(features[:, :, y0:y0+ps, x0:x0+ps])
        tgt_patches.append(target_scaled[:, :, y0:y0+ps, x0:x0+ps])
        mask_patches.append(m)
        attempts += 1

    if not feat_patches:
        return None

    feat_p = torch.cat(feat_patches)
    tgt_p = torch.cat(tgt_patches)
    mask_p = torch.cat(mask_patches)
    n_patches = feat_p.shape[0]
    batch_size = min(32, n_patches)  # larger batches on GPU

    L.info(f"GPU-CNN: {n_patches} patches, {epochs} epochs, device={DEVICE}")

    # Train
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_patches, device=DEVICE)
        for i in range(0, n_patches, batch_size):
            idx = perm[i:i+batch_size]
            fb = feat_p[idx]
            tb = tgt_p[idx]
            mb = mask_p[idx]

            fb_p, ph, pw = _gpu_pad(fb)
            pred = model(fb_p)[:, :, :ph, :pw]

            diff = (pred - tb) * mb
            loss = torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff))

            if pred.shape[2] > 1 and pred.shape[3] > 1:
                dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
                dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
                loss = loss + 0.003 * (dx + dy)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        scheduler.step()

    # Tiled inference on GPU
    model.eval()
    depth_out = torch.zeros(H, W, device=DEVICE)
    weight_out = torch.zeros(H, W, device=DEVICE)
    tile = 256
    overlap = 32
    step = tile - overlap

    with torch.no_grad():
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1 = min(y0 + tile, H)
                x1 = min(x0 + tile, W)
                inp = features[:, :, y0:y1, x0:x1]
                inp_p, th, tw = _gpu_pad(inp)
                pred = model(inp_p)[0, 0, :th, :tw]
                depth_out[y0:y1, x0:x1] += pred
                weight_out[y0:y1, x0:x1] += 1.0

    weight_out = weight_out.clamp(min=1.0)
    depth_out = (depth_out / weight_out) * depth_scale
    depth_out = depth_out.clamp(0, MAX_DEPTH_M)
    depth_out = torch.where(water > 0.5, depth_out, torch.tensor(float("nan"), device=DEVICE))

    # Gaussian smooth on GPU (3x3 kernel approximation)
    valid = torch.isfinite(depth_out)
    d_clean = torch.nan_to_num(depth_out, nan=0.0).unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32,
                          device=DEVICE).unsqueeze(0).unsqueeze(0) / 16.0
    smoothed = torch.nn.functional.conv2d(d_clean, kernel, padding=1)[0, 0]
    depth_out = torch.where(valid & (water > 0.5), smoothed, torch.tensor(float("nan"), device=DEVICE))

    return depth_out.cpu().numpy()


# ══════════════════════════════════════════════════════════════
# 6. TILING + MOSAIC ENGINE
# ══════════════════════════════════════════════════════════════

def _compute_tiles(bbox: list, tile_deg: float = 0.08,
                   overlap_deg: float = 0.008) -> list[list]:
    """
    Split bbox into overlapping sub-tiles.
    Returns list of [west, south, east, north] bboxes.
    """
    w, s, e, n = bbox
    tiles = []
    lat = s
    while lat < n:
        lat_top = min(lat + tile_deg, n)
        lon = w
        while lon < e:
            lon_right = min(lon + tile_deg, e)
            tiles.append([
                lon - overlap_deg, lat - overlap_deg,
                lon_right + overlap_deg, lat_top + overlap_deg,
            ])
            lon += tile_deg
        lat += tile_deg

    L.info(f"Mosaic: {len(tiles)} tiles @ {tile_deg}° + {overlap_deg}° overlap")
    return tiles


def _feather_weight(H: int, W: int, margin: int) -> np.ndarray:
    """
    Create a feathering weight mask: 1.0 in the centre, linearly fading
    to 0.0 at the edges over `margin` pixels.
    """
    w = np.ones((H, W), dtype=np.float32)
    for i in range(min(margin, H // 2)):
        f = (i + 1) / (margin + 1)
        w[i, :] *= f
        w[H - 1 - i, :] *= f
    for j in range(min(margin, W // 2)):
        f = (j + 1) / (margin + 1)
        w[:, j] *= f
        w[:, W - 1 - j] *= f
    return w


def _process_one_tile(tile_bbox: list, sd: str, ed: str,
                      cloud: int = 20) -> Optional[dict]:
    """
    Process a single sub-tile: fetch S2@10m, GEBCO, run GPU-CNN.
    Returns {bbox, depth, height, width} or None.
    """
    try:
        s2 = fetch_s2_10m(tile_bbox, sd, ed, cloud=cloud)
        gebco = _fetch_gebco(tile_bbox)
        if gebco is None or len(gebco["depths"]) < 10:
            L.warning(f"Tile {tile_bbox}: insufficient GEBCO data, using SDB fallback")
            # Simple SDB fallback
            blue = s2["blue"].astype(np.float64) / 10000
            green = s2["green"].astype(np.float64) / 10000
            water = s2["ndwi"] > 0
            bc = np.clip(blue, 1e-5, None)
            gc = np.clip(green, 1e-5, None)
            ratio = np.log(1000 * bc) / np.log(1000 * gc + 1e-6)
            rv = ratio[water]
            rv = rv[np.isfinite(rv)]
            if len(rv) < 50:
                return None
            p2, p98 = np.percentile(rv, 2), np.percentile(rv, 98)
            depth = np.where(water, MAX_DEPTH_M * (np.clip(ratio, p2, p98) - p2) / (p98 - p2 + 1e-10), np.nan)
            depth[~water] = np.nan
            depth = np.clip(np.where(np.isfinite(depth), depth, np.nan), 0, MAX_DEPTH_M)
        else:
            depth = gpu_cnn_predict(s2, gebco, tile_bbox)
            if depth is None:
                return None

        return {
            "bbox": tile_bbox,
            "depth": depth,
            "height": depth.shape[0],
            "width": depth.shape[1],
        }
    except Exception as ex:
        L.error(f"Tile {tile_bbox} failed: {ex}")
        return None


def mosaic_extract(bbox: list, n_weeks: int = 4, max_cloud: int = 20,
                   tile_deg: float = 0.08, overlap_deg: float = 0.008,
                   ref_source: str = "gebco",
                   progress_cb=None) -> dict:
    """
    Main entry: mosaic bathymetry for a large region.

    Parameters
    ----------
    bbox : [west, south, east, north]
    n_weeks : how many weeks back for S2 scenes
    max_cloud : max cloud %
    tile_deg : sub-tile size in degrees (~8.8 km)
    overlap_deg : overlap between tiles in degrees
    progress_cb : optional callback(pct, msg) for progress updates

    Returns
    -------
    dict with: depth_grid, bbox, stats, tiles_ok, tiles_fail, points, sources
    """
    w, s, e, n = bbox
    area_km2 = abs(e - w) * abs(n - s) * 111 * 111 * math.cos(math.radians((n + s) / 2))
    L.info(f"=== MOSAIC START: {area_km2:.0f} km², device={DEVICE} ===")

    # Determine date range (last n_weeks)
    ed = datetime.utcnow()
    sd = ed - timedelta(weeks=n_weeks)
    sd_str = sd.strftime("%Y-%m-%d")
    ed_str = ed.strftime("%Y-%m-%d")

    # Search catalog for best scenes
    scenes = search_best_scenes(bbox, n_weeks=n_weeks, max_cloud=max_cloud, top_k=6)
    if not scenes:
        L.warning("No scenes found, proceeding with full date range anyway")

    # Narrow date range to the best scenes if possible
    if scenes:
        best_dates = sorted(set(sc["date"] for sc in scenes[:4]))
        if best_dates:
            sd_str = best_dates[0]
            ed_str = best_dates[-1]
            L.info(f"Using best scene dates: {sd_str} → {ed_str}")

    # Split into tiles
    tiles = _compute_tiles(bbox, tile_deg=tile_deg, overlap_deg=overlap_deg)

    # Compute output grid dimensions at 10m
    cl = math.cos(math.radians((n + s) / 2))
    out_w = max(32, int(abs(e - w) * 111000 * cl / 10))
    out_h = max(32, int(abs(n - s) * 111000 / 10))

    # Cap output size to prevent OOM
    MAX_OUT = 8000
    if out_w > MAX_OUT or out_h > MAX_OUT:
        scale = MAX_OUT / max(out_w, out_h)
        out_w = int(out_w * scale)
        out_h = int(out_h * scale)
        L.info(f"Output capped: {out_w}x{out_h}")

    # Allocate output on GPU
    mosaic_depth = torch.zeros(out_h, out_w, device=DEVICE)
    mosaic_weight = torch.zeros(out_h, out_w, device=DEVICE)

    tiles_ok = 0
    tiles_fail = 0
    sources = [f"S2@10m({sd_str}→{ed_str})"]
    if scenes:
        sources.append(f"Catalog({len(scenes)} scenes, best cloud={scenes[0]['cloud_pct']:.0f}%)")

    # Process tiles (sequential to avoid GPU contention)
    for ti, tile_bbox in enumerate(tiles):
        if progress_cb:
            progress_cb(int(100 * ti / len(tiles)),
                        f"Processing tile {ti+1}/{len(tiles)}")
        L.info(f"Tile {ti+1}/{len(tiles)}: {tile_bbox}")

        result = _process_one_tile(tile_bbox, sd_str, ed_str, cloud=max_cloud)
        if result is None:
            tiles_fail += 1
            continue

        tiles_ok += 1
        tdepth = result["depth"]
        th, tw = tdepth.shape
        tw_bb, ts_bb, te_bb, tn_bb = tile_bbox

        # Create feather weight
        margin = max(2, int(min(th, tw) * 0.1))
        fw = _feather_weight(th, tw, margin)

        # Vectorised mapping: tile pixel coords → output grid coords
        row_idx = np.arange(th)
        col_idx = np.arange(tw)
        tile_lats = tn_bb - (row_idx / th) * (tn_bb - ts_bb)   # (th,)
        tile_lons = tw_bb + (col_idx / tw) * (te_bb - tw_bb)    # (tw,)
        out_rows = np.clip(((n - tile_lats) / (n - s + 1e-10) * out_h).astype(int), 0, out_h - 1)
        out_cols = np.clip(((tile_lons - w) / (e - w + 1e-10) * out_w).astype(int), 0, out_w - 1)

        # Valid mask
        valid_tile = np.isfinite(tdepth) & (tdepth > 0)

        # Use meshgrid for full tile → output mapping
        or_grid, oc_grid = np.meshgrid(out_rows, out_cols, indexing='ij')  # (th, tw)

        # Move to GPU tensors for accumulation
        t_depth = torch.from_numpy(np.where(valid_tile, tdepth, 0).astype(np.float32)).to(DEVICE)
        t_fw = torch.from_numpy(np.where(valid_tile, fw, 0).astype(np.float32)).to(DEVICE)
        t_or = torch.from_numpy(or_grid.ravel().astype(np.int64)).to(DEVICE)
        t_oc = torch.from_numpy(oc_grid.ravel().astype(np.int64)).to(DEVICE)
        t_flat_idx = t_or * out_w + t_oc

        mosaic_depth.view(-1).scatter_add_(0, t_flat_idx, (t_depth * t_fw).view(-1))
        mosaic_weight.view(-1).scatter_add_(0, t_flat_idx, t_fw.view(-1))

    # Finalise mosaic
    valid = mosaic_weight > 0
    mosaic_weight = mosaic_weight.clamp(min=1e-6)
    depth_final = torch.where(valid, mosaic_depth / mosaic_weight,
                              torch.tensor(float("nan"), device=DEVICE))
    depth_final = depth_final.clamp(0, MAX_DEPTH_M)
    depth_np = depth_final.cpu().numpy()

    # Gaussian smooth the final mosaic on GPU
    d_clean = torch.nan_to_num(depth_final, nan=0.0).unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32,
                          device=DEVICE).unsqueeze(0).unsqueeze(0) / 16.0
    smoothed = torch.nn.functional.conv2d(d_clean, kernel, padding=1)[0, 0]
    depth_np = torch.where(valid, smoothed, torch.tensor(float("nan"), device=DEVICE)).cpu().numpy()

    # Generate output points
    val = depth_np[np.isfinite(depth_np) & (depth_np > 0)]
    step_pts = max(1, int(math.sqrt(out_h * out_w / 12000)))
    points = []
    for r in range(0, out_h, step_pts):
        lat = n - (r / out_h) * (n - s)
        for c in range(0, out_w, step_pts):
            v = depth_np[r, c]
            if np.isfinite(v) and v > 0.1:
                points.append({
                    "lat": round(lat, 6),
                    "lon": round(w + (c / out_w) * (e - w), 6),
                    "depth": round(min(float(v), MAX_DEPTH_M), 2),
                    "photon_class": "interpolated",
                })

    # Zone counts
    d_t = torch.from_numpy(np.nan_to_num(depth_np, nan=-1)).to(DEVICE)
    water_mask = d_t > 0
    zc = {
        "vs": int((water_mask & (d_t <= 3)).sum().item()),
        "sh": int((water_mask & (d_t > 3) & (d_t <= 8)).sum().item()),
        "md": int((water_mask & (d_t > 8) & (d_t <= 15)).sum().item()),
        "dp": int((water_mask & (d_t > 15)).sum().item()),
    }

    stats = {
        "mean_depth": round(float(np.nanmean(val)), 2) if len(val) else 0,
        "max_depth": round(float(np.nanmax(val)), 2) if len(val) else 0,
        "min_depth": round(float(np.nanmin(val)), 2) if len(val) else 0,
        "std_depth": round(float(np.nanstd(val)), 2) if len(val) else 0,
        "grid_points": len(points),
        "resolution_m": 10,
        "tiles_total": len(tiles),
        "tiles_ok": tiles_ok,
        "tiles_fail": tiles_fail,
        "area_km2": round(area_km2, 1),
        "output_size": f"{out_w}x{out_h}",
        "device": str(DEVICE),
    }
    sources.append(f"Mosaic({tiles_ok}/{len(tiles)} tiles, {out_w}x{out_h})")

    L.info(f"=== MOSAIC DONE: {tiles_ok}/{len(tiles)} tiles, "
           f"{len(points)} pts, device={DEVICE} ===")

    return {
        "points": [],
        "interpolated_points": points,
        "stats": stats,
        "ml_stats": {"method": "GPU Mosaic CNN@10m", "zone_counts": zc},
        "sources_used": sources,
        "bbox": {"west": w, "south": s, "east": e, "north": n},
        "scenes": scenes[:4],
    }
