"""Very HR bathymetry engine — Mapbox VHR + Sentinel-2 + dilated CNN.

Pipeline (v2):
1. Fetch Mapbox VHR mosaic over the ROI (tiled, ≈ 1.5–3 m/px).
2. Fetch Sentinel-2 L2A mosaic (B/G/R/NIR/SCL @ 10 m, cloud ≤ 20 %).
3. Resample S2 onto the VHR pixel grid → unified multi-source raster.
4. Build a robust **water mask** from S2 NDWI (+ SCL where available)
   intersected with VHR brightness sanity checks.
5. Build a 16-channel feature stack:
   - 3  VHR R/G/B (deglinted via dark-pixel correction)
   - 4  S2 R/G/B/NIR (DN /10 000)
   - 1  S2 NDWI
   - 2  Stumpf log-ratios (S2 B/G, S2 G/R)
   - 6  5×5 local mean / std on VHR R/G/B
6. Train a **dilated fully-convolutional CNN** (≈ 200 K params) on the
   labelled water pixels using random crops + masked MSE.
7. Predict the full grid in tiled mode (no boundary seams), cap at 25 m.
8. Compute metrics per depth band (0–5, 5–10, 10–15, 15–20, 20–25 m) and
   plot S-44 1A/1B/Order-2 pass percentages as a grouped bar chart.

Both predictions and labels are capped at MAX_DEPTH_M = 25.0 m.
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

L = logging.getLogger("very_hr")
if not L.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    L.addHandler(h)
    L.setLevel(logging.INFO)

MAX_DEPTH_M = 25.0
DEPTH_BANDS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "Very_HR_Results" / "Updated_Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OSM_CACHE_DIR = RESULTS_DIR / "_osm_cache"
OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# 1. Mapbox VHR tiled mosaic
# ════════════════════════════════════════════════════════════════════════
def fetch_mapbox_vhr(
    bbox: List[float],
    target_res_m: float = 2.0,
    max_tiles_per_side: int = 4,
    style: str = "mapbox/satellite-v9",
    token: Optional[str] = None,
) -> Dict:
    token = token or os.getenv("MAPBOX_TOKEN", "")
    if not token:
        raise RuntimeError("MAPBOX_TOKEN not set")
    w, s, e, n = bbox
    cl = math.cos(math.radians((n + s) / 2))
    bbox_w_m = abs(e - w) * 111000.0 * cl
    bbox_h_m = abs(n - s) * 111000.0
    tile_max_px = 2560
    nx = max(1, min(max_tiles_per_side, math.ceil((bbox_w_m / target_res_m) / tile_max_px)))
    ny = max(1, min(max_tiles_per_side, math.ceil((bbox_h_m / target_res_m) / tile_max_px)))
    px_per_tile_w = min(1280, max(64, int(bbox_w_m / nx / target_res_m / 2)))
    px_per_tile_h = min(1280, max(64, int(bbox_h_m / ny / target_res_m / 2)))
    L.info(
        f"Mapbox: bbox≈{bbox_w_m/1000:.2f}×{bbox_h_m/1000:.2f}km → "
        f"{nx}×{ny} tiles @ {px_per_tile_w}×{px_per_tile_h}@2x"
    )

    def _tile(ix: int, iy: int):
        tw = w + (e - w) * ix / nx
        te = w + (e - w) * (ix + 1) / nx
        ts = s + (n - s) * iy / ny
        tn = s + (n - s) * (iy + 1) / ny
        url = (
            f"https://api.mapbox.com/styles/v1/{style}/static/"
            f"[{tw},{ts},{te},{tn}]/{px_per_tile_w}x{px_per_tile_h}@2x"
            f"?access_token={token}&attribution=false&logo=false"
        )
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=45)
                if r.ok:
                    img = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
                    return ix, iy, img
                L.warning(f"Mapbox tile {ix},{iy} HTTP {r.status_code} attempt {attempt+1}")
            except Exception as ex:
                L.warning(f"Mapbox tile {ix},{iy} error {ex}")
            time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"Mapbox tile {ix},{iy} failed")

    tiles: Dict[Tuple[int, int], np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=min(8, nx * ny)) as pool:
        futs = [pool.submit(_tile, ix, iy) for iy in range(ny) for ix in range(nx)]
        for f in as_completed(futs):
            ix, iy, img = f.result()
            tiles[(ix, iy)] = img
    th, tw_px = tiles[(0, 0)].shape[:2]
    H = th * ny
    W = tw_px * nx
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for (ix, iy), img in tiles.items():
        row0 = (ny - 1 - iy) * th
        col0 = ix * tw_px
        canvas[row0:row0 + th, col0:col0 + tw_px] = img[:th, :tw_px]
    res = ((bbox_w_m / W) + (bbox_h_m / H)) / 2
    L.info(f"Mapbox mosaic: {W}×{H}px ≈ {res:.2f} m/px")
    return {"rgb": canvas, "width": W, "height": H, "bbox": [w, s, e, n],
            "resolution_m": round(res, 3), "tiles": [nx, ny]}


# ════════════════════════════════════════════════════════════════════════
# 2. Sentinel-2 fetch (B/G/R/NIR/SCL via Sentinel-Hub)
# ════════════════════════════════════════════════════════════════════════
_S2_EVALSCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["B02","B03","B04","B08","SCL"],units:"DN",mosaicking:"ORBIT"}],output:{bands:5,sampleType:"UINT16"}};}
function isValid(s){var c=s.SCL;return c!==1&&c!==3&&c!==8&&c!==9&&c!==10&&c!==11;}
function median(a){if(!a.length)return 0;a.sort(function(x,y){return x-y});var m=Math.floor(a.length/2);return a.length%2===0?(a[m-1]+a[m])/2:a[m];}
function evaluatePixel(samples){var b02=[],b03=[],b04=[],b08=[],sclv=[];for(var i=0;i<samples.length;i++){if(isValid(samples[i])){b02.push(samples[i].B02);b03.push(samples[i].B03);b04.push(samples[i].B04);b08.push(samples[i].B08);sclv.push(samples[i].SCL);}}if(!b02.length){for(var j=0;j<samples.length;j++){b02.push(samples[j].B02);b03.push(samples[j].B03);b04.push(samples[j].B04);b08.push(samples[j].B08);sclv.push(samples[j].SCL);}}return[median(b02),median(b03),median(b04),median(b08),sclv.length?sclv[0]:0];}
"""


SH_AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


def _sh_token() -> str:
    cid = os.getenv("SH_CLIENT_ID", "")
    csec = os.getenv("SH_CLIENT_SECRET", "")
    if not (cid and csec):
        raise RuntimeError("SH_CLIENT_ID/SH_CLIENT_SECRET not set")
    r = requests.post(
        SH_AUTH_URL,
        data={"grant_type": "client_credentials", "client_id": cid,
              "client_secret": csec}, timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def fetch_s2(
    bbox: List[float],
    start_date: str = "2024-05-01",
    end_date: str = "2024-09-30",
    res_m: int = 10,
    cloud: int = 20,
) -> Dict:
    """Fetch a multi-temporal least-cloud S2 L2A mosaic over ``bbox``."""
    w, s, e, n = bbox
    cl = math.cos(math.radians((n + s) / 2))
    wp = max(64, min(2500, int(abs(e - w) * 111000 * cl / res_m)))
    hp = max(64, min(2500, int(abs(n - s) * 111000 / res_m)))
    tok = _sh_token()
    body = {
        "input": {
            "bounds": {"bbox": [w, s, e, n], "properties": {
                "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"timeRange": {
                          "from": f"{start_date}T00:00:00Z",
                          "to": f"{end_date}T23:59:59Z"},
                          "maxCloudCoverage": cloud}}],
        },
        "output": {"width": wp, "height": hp,
                   "responses": [{"identifier": "default",
                                  "format": {"type": "image/tiff"}}]},
        "evalscript": _S2_EVALSCRIPT,
    }
    L.info(f"S2: {wp}×{hp} @ {res_m} m  ({start_date} → {end_date}, cloud≤{cloud}%)")
    for attempt in range(4):
        r = requests.post(
            SH_PROCESS_URL,
            headers={"Authorization": f"Bearer {tok}",
                     "Content-Type": "application/json"},
            json=body, timeout=300,
        )
        if r.ok:
            break
        if r.status_code == 429:
            wait = 3 * (2 ** attempt)
            L.warning(f"S2 rate-limited, retry in {wait}s ({attempt+1}/4)")
            time.sleep(wait)
            continue
        if r.status_code == 401 and attempt < 3:
            tok = _sh_token()
            continue
        r.raise_for_status()
    r.raise_for_status()
    import rasterio
    from rasterio.io import MemoryFile
    with MemoryFile(r.content) as mf:
        with mf.open() as src:
            arr = src.read()  # (5, H, W) uint16
    blue, green, red, nir, scl = arr[0], arr[1], arr[2], arr[3], arr[4]
    return {"blue": blue, "green": green, "red": red, "nir": nir,
            "scl": scl, "width": wp, "height": hp, "bbox": [w, s, e, n],
            "resolution_m": res_m,
            "window": {"start": start_date, "end": end_date}}


def _resample(arr: np.ndarray, h: int, w: int,
              resample=Image.BILINEAR) -> np.ndarray:
    pil = Image.fromarray(arr.astype(np.float32))
    return np.array(pil.resize((w, h), resample), dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════
# Google Earth Engine helpers — S2 fallback + ERA-5 + GHRSST SST
# ════════════════════════════════════════════════════════════════════════
_GEE_READY = False


def _resolve_gee_sa_json() -> Optional[str]:
    """Resolve an Earth Engine service-account JSON string from any of the
    common env-var conventions:

      - ``GEE_SERVICE_ACCOUNT_JSON``  : raw JSON or base64-of-JSON
      - ``GEE_CREDENTIALS_JSON``      : path-to-file, raw JSON, or base64
      - ``GEE_SERVICE_ACCOUNT_FILE``  : path-to-JSON-file
      - ``GOOGLE_APPLICATION_CREDENTIALS`` : path-to-JSON-file

    Path values may point to a file containing either the raw JSON or a
    base64-encoded JSON blob.  Returns the decoded JSON text, or None.
    """
    import base64 as _b64, json as _json
    candidates = [
        os.getenv("GEE_SERVICE_ACCOUNT_JSON", "").strip(),
        os.getenv("GEE_CREDENTIALS_JSON", "").strip(),
    ]
    file_paths = [
        os.getenv("GEE_SERVICE_ACCOUNT_FILE", "").strip(),
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip(),
    ]
    for cand in candidates:
        if not cand:
            continue
        # If it's a path, switch to the file branch
        if os.path.exists(cand):
            file_paths.insert(0, cand)
            continue
        # Raw JSON?
        if cand.startswith("{"):
            try:
                _json.loads(cand)
                return cand
            except Exception:
                pass
        # base64?
        try:
            decoded = _b64.b64decode(cand, validate=False).decode("utf-8", "ignore")
            if decoded.strip().startswith("{"):
                _json.loads(decoded)
                return decoded
        except Exception:
            pass

    for p in file_paths:
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                content = f.read().strip()
        except Exception:
            continue
        if content.startswith("{"):
            try:
                _json.loads(content)
                return content
            except Exception:
                pass
        # base64 blob in file?
        try:
            decoded = _b64.b64decode(content, validate=False).decode("utf-8", "ignore")
            if decoded.strip().startswith("{"):
                _json.loads(decoded)
                return decoded
        except Exception:
            continue
    return None


def _init_gee() -> bool:
    """Initialise Earth Engine. Tries service-account auth (any common env-var
    pattern — see ``_resolve_gee_sa_json``), then falls back to default
    ``ee.Initialize(project=…)`` for gcloud-cli developer setups."""
    global _GEE_READY
    if _GEE_READY:
        return True
    try:
        import ee
        import json as _json
        import tempfile
        project = os.getenv("EE_PROJECT", "ee-wassimehtp")

        sa_text = _resolve_gee_sa_json()
        if sa_text:
            try:
                info = _json.loads(sa_text)
                tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
                tf.write(sa_text); tf.close()
                email = info.get("client_email", "")
                creds = ee.ServiceAccountCredentials(email, tf.name)
                ee.Initialize(credentials=creds, project=project)
                _GEE_READY = True
                L.info(f"GEE initialised via service account "
                       f"({email}, project={project})")
                return True
            except Exception as ex:
                L.warning(f"GEE service-account init failed: {ex}")

        ee.Initialize(project=project)
        _GEE_READY = True
        L.info(f"GEE initialised  (project={project})")
        return True
    except Exception as ex:
        L.warning(f"GEE init failed: {ex}")
        return False


def _gee_to_numpy(image, bbox: List[float], res_m: int, bands: List[str]) -> Dict[str, np.ndarray]:
    """Sample a GEE image inside ``bbox`` at ``res_m`` and return numpy arrays."""
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)
    cl = math.cos(math.radians((n + s) / 2))
    wp = max(8, min(2000, int(abs(e - w) * 111000 * cl / res_m)))
    hp = max(8, min(2000, int(abs(n - s) * 111000 / res_m)))
    arr = image.sampleRectangle(region=region, defaultValue=0)
    info = arr.getInfo()
    out: Dict[str, np.ndarray] = {}
    for b in bands:
        if b in info.get("properties", {}):
            out[b] = np.asarray(info["properties"][b], dtype=np.float32)
        else:
            out[b] = np.zeros((hp, wp), dtype=np.float32)
    return {"bands": out, "width": wp, "height": hp, "resolution_m": res_m}


def _gee_download_geotiff(image, bbox: List[float], scale: int,
                           band_order: List[str], tag: str = "vhr"
                           ) -> Dict[str, np.ndarray]:
    """Download a GeoTIFF via ``ee.Image.getDownloadURL`` and key bands by
    the supplied ``band_order`` (index-aligned with the image's selected bands).
    """
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)
    url = image.getDownloadURL({
        "region": region.getInfo()["coordinates"],
        "scale": scale, "crs": "EPSG:4326", "format": "GEO_TIFF",
    })
    r = requests.get(url, timeout=240)
    r.raise_for_status()
    import rasterio
    from rasterio.io import MemoryFile
    with MemoryFile(r.content) as mf:
        with mf.open() as src:
            arr = src.read()                                             # (B, H, W)
            descs = list(src.descriptions) if src.descriptions else []
    if len(descs) == arr.shape[0] and all(descs):
        names_out = [d for d in descs]
    else:
        names_out = list(band_order[:arr.shape[0]])
        if len(names_out) < arr.shape[0]:
            names_out += [f"b{i}" for i in range(len(names_out), arr.shape[0])]
    out = {names_out[i]: arr[i] for i in range(arr.shape[0])}
    return {"bands": out, "width": arr.shape[2], "height": arr.shape[1],
            "resolution_m": scale}


def fetch_s2_gee(bbox: List[float], start_date: str, end_date: str,
                  cloud: int = 30) -> Dict:
    """GEE-backed S2 SR Harmonized fetch — used when CDSE Sentinel-Hub is out."""
    if not _init_gee():
        raise RuntimeError("GEE not available")
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))
    img = (coll.select(["B2", "B3", "B4", "B8", "SCL"])
              .median()
              .rename(["blue", "green", "red", "nir", "scl"])
              .clip(region))
    bundle = _gee_download_geotiff(img, bbox, scale=10,
                                     band_order=["blue", "green", "red", "nir", "scl"],
                                     tag="s2")
    bands = bundle["bands"]
    return {"blue": bands["blue"].astype(np.float32),
            "green": bands["green"].astype(np.float32),
            "red": bands["red"].astype(np.float32),
            "nir": bands["nir"].astype(np.float32),
            "scl": bands["scl"].astype(np.uint8),
            "width": bundle["width"], "height": bundle["height"],
            "resolution_m": bundle["resolution_m"],
            "window": {"start": start_date, "end": end_date}}


def fetch_s2_gee_scenes(
    bbox: List[float],
    start_date: str,
    end_date: str,
    cloud: int = 25,
    max_scenes: int = 10,
) -> List[Dict]:
    """GEE-backed S2 per-scene list — BIAS-R7-1 multi-temporal MLE premise.

    Returns up to `max_scenes` individual Sentinel-2 SR acquisitions sorted by
    a per-scene ROI quality score (low cloud + low NIR glint over the water mask).
    Does NOT call `.median()` — each scene is returned independently so the caller
    can run the inverse-variance MLE composite across K≥2 scenes.

    Keep the existing single-scene `fetch_s2_gee` untouched for back-compat.

    Parameters
    ----------
    bbox        : [west, south, east, north] in EPSG:4326
    start_date  : ISO "YYYY-MM-DD"
    end_date    : ISO "YYYY-MM-DD"
    cloud       : CLOUDY_PIXEL_PERCENTAGE filter (default 25 %)
    max_scenes  : maximum number of scenes to return (default 10)

    Returns
    -------
    scenes : list of dicts, each with blue/green/red/nir/scl + metadata.
             Empty list if GEE unavailable or no qualifying images found.

    References
    ----------
    Caballero & Stumpf 2020, Remote Sensing multi-temporal S2 SDB compositing.
    Li et al. 2021 multi-temporal SDB.
    """
    if not _init_gee():
        L.warning("fetch_s2_gee_scenes: GEE not available; returning empty list")
        return []
    import ee

    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)

    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))

    # Sort by composite quality: lower cloud + lower NIR (glint proxy) → better
    # Use CLOUDY_PIXEL_PERCENTAGE as primary sort; ties resolved by system:time_start
    coll_sorted = coll.sort("CLOUDY_PIXEL_PERCENTAGE")

    # Get image list (limit to 3× max_scenes before download filter)
    prefetch = min(max_scenes * 3, 30)
    img_list = coll_sorted.limit(prefetch).toList(prefetch)
    n_avail = int(coll_sorted.limit(prefetch).size().getInfo())
    L.info(f"fetch_s2_gee_scenes: {n_avail} images available ({start_date}→{end_date}), "
           f"will try up to {n_avail}")

    scenes: List[Dict] = []
    for i in range(n_avail):
        if len(scenes) >= max_scenes:
            break
        try:
            img = ee.Image(img_list.get(i))
            # Per-scene cloud score (for metadata)
            cloud_pct = float(img.get("CLOUDY_PIXEL_PERCENTAGE").getInfo())
            acq_date  = str(img.get("system:time_start").getInfo())

            img_sel = (img.select(["B2", "B3", "B4", "B8", "SCL"])
                          .rename(["blue", "green", "red", "nir", "scl"])
                          .clip(region))
            bundle = _gee_download_geotiff(
                img_sel, bbox, scale=10,
                band_order=["blue", "green", "red", "nir", "scl"],
                tag=f"s2_sc{i:02d}",
            )
            bands = bundle["bands"]
            # Per-scene NIR glint proxy (median over ROI)
            nir_arr = bands.get("nir", np.array([[0.0]])).astype(np.float32)
            nir_refl = nir_arr / 10000.0
            glint_nir = float(np.median(nir_refl))

            # Hard glint reject: NIR median > 0.25 indicates sunglint contamination
            # that corrupts the Lyzenga log-ratio and the MLE composite.
            # This threshold is conservative (saipan scenes were 0.38-0.41).
            GLINT_NIR_HARD_MAX = 0.25
            if glint_nir > GLINT_NIR_HARD_MAX:
                L.info(f"  scene {i}: SKIPPED (glint_NIR={glint_nir:.4f} > {GLINT_NIR_HARD_MAX})")
                continue

            sc = {
                "blue":   bands["blue"].astype(np.float32),
                "green":  bands["green"].astype(np.float32),
                "red":    bands["red"].astype(np.float32),
                "nir":    bands["nir"].astype(np.float32),
                "scl":    bands["scl"].astype(np.float32),
                "width":  bundle["width"],
                "height": bundle["height"],
                "resolution_m": bundle["resolution_m"],
                "cloud_pct": cloud_pct,
                "glint_nir": glint_nir,
                "acq_date":  acq_date,
                "scene_idx": i,
            }
            scenes.append(sc)
            L.info(f"  scene {i}: cloud={cloud_pct:.1f}%, glint_NIR={glint_nir:.4f}, "
                   f"date={acq_date}")
        except Exception as e:
            L.warning(f"  scene {i}: download failed ({e}); skipping")
            continue

    # Secondary sort: within valid scenes sort by NIR glint (ascending) so the
    # best-quality (lowest glint) scene comes first. Cloud was already sorted above.
    scenes.sort(key=lambda sc: (sc.get("cloud_pct", 99), sc.get("glint_nir", 99)))

    L.info(f"fetch_s2_gee_scenes: returned {len(scenes)} scenes")
    return scenes


def fetch_era5_sst_gee(bbox: List[float], start_date: str, end_date: str
                        ) -> Dict[str, Dict[str, np.ndarray]]:
    """ERA-5 hourly (10 m wind, 2 m temperature) + GHRSST MUR SST.

    ERA-5 native resolution ≈ 0.25° (≈ 28 km).  Returned upsampled to ~250 m
    grid via GEE bilinear resampling, then resampled again to the VHR grid in
    the caller.  GHRSST is ≈ 1 km, also upsampled.
    """
    if not _init_gee():
        raise RuntimeError("GEE not available")
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)

    # ERA-5: pick mean of u10, v10, t2m over the date window
    era5 = (ee.ImageCollection("ECMWF/ERA5/HOURLY")
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .select(["u_component_of_wind_10m",
                     "v_component_of_wind_10m",
                     "temperature_2m"])
            .mean()
            .rename(["u10", "v10", "t2m"])
            .clip(region))

    # NOAA OISST V2.1 — daily 0.25° SST (scale = 0.01 °C)
    sst_coll = (ee.ImageCollection("NOAA/CDR/OISST/V2_1")
                .filterDate(start_date, end_date)
                .filterBounds(region)
                .select(["sst"]))
    sst = sst_coll.mean().rename("sst").clip(region)

    bands_out: Dict[str, np.ndarray] = {}
    # Pull ERA-5 at ~500 m so the GEE response stays small (native ≈ 28 km).
    try:
        era5_bundle = _gee_download_geotiff(
            era5, bbox, scale=500,
            band_order=["u10", "v10", "t2m"], tag="era5")
        for k in ("u10", "v10", "t2m"):
            if k in era5_bundle["bands"]:
                bands_out[k] = era5_bundle["bands"][k].astype(np.float32)
        if "u10" in bands_out and "v10" in bands_out:
            u = bands_out["u10"]; v = bands_out["v10"]
            bands_out["wind_speed"] = np.sqrt(u * u + v * v).astype(np.float32)
    except Exception as ex:
        L.warning(f"ERA-5 fetch failed: {ex}")

    # OISST: 0.25° native — fetch at 2500 m
    try:
        sst_bundle = _gee_download_geotiff(
            sst, bbox, scale=2500,
            band_order=["sst"], tag="sst")
        sst_arr = (sst_bundle["bands"].get("sst")
                   if "sst" in sst_bundle["bands"]
                   else list(sst_bundle["bands"].values())[0]).astype(np.float32)
        bands_out["sst"] = (sst_arr * 0.01).astype(np.float32)  # → °C
    except Exception as ex:
        L.warning(f"SST fetch failed: {ex}")

    if not bands_out:
        raise RuntimeError("ERA-5 and GHRSST both empty")
    return {"bands": bands_out,
            "window": {"start": start_date, "end": end_date}}


# ════════════════════════════════════════════════════════════════════════
# OSM land mask  — definitive land/water boundary, cached on disk
# ════════════════════════════════════════════════════════════════════════
def osm_land_mask(bbox: List[float], H: int, W: int,
                  cache_dir: Path = OSM_CACHE_DIR) -> np.ndarray:
    """Return a (H, W) boolean array, ``True`` where OSM says LAND.

    Uses ``osmnx`` to fetch every polygon tagged as land-like (buildings,
    landuse, natural=land/coastline/beach/wood/scrub, place=island,
    man_made=breakwater/pier, …) inside ``bbox`` and rasterises them onto
    the VHR pixel grid. Cached on disk per (bbox, H, W).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    key = hashlib.md5(json.dumps([bbox, H, W], sort_keys=True).encode()).hexdigest()[:12]
    cache_path = cache_dir / f"land_{key}.npz"
    if cache_path.exists():
        try:
            land = np.load(cache_path)["mask"].astype(bool)
            if land.shape == (H, W):
                L.info(f"OSM land mask: cache hit {cache_path.name}  "
                       f"({100*land.mean():.1f}% land)")
                return land
        except Exception as ex:
            L.warning(f"OSM cache read failed: {ex}")

    try:
        import osmnx as ox
        from shapely.geometry import box
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
    except Exception as ex:
        L.warning(f"OSM libs unavailable ({ex}) — returning empty land mask")
        return np.zeros((H, W), dtype=bool)

    w, s, e, n = bbox
    polygon = box(w, s, e, n)
    tags = {
        "natural": ["land", "coastline", "beach", "wood", "scrub",
                     "grassland", "wetland", "bare_rock", "sand"],
        "landuse": True,
        "building": True,
        "man_made": ["breakwater", "pier", "groyne", "embankment",
                     "dyke", "tower", "storage_tank", "wastewater_plant",
                     "works", "silo"],
        "place": ["island", "islet", "archipelago"],
        "amenity": ["parking", "fuel"],
        "leisure": ["pitch", "park"],
    }
    try:
        gdf = ox.features.features_from_polygon(polygon, tags=tags)
    except Exception as ex:
        L.warning(f"OSM fetch failed: {ex} — returning empty land mask")
        return np.zeros((H, W), dtype=bool)

    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(gdf) == 0:
        L.warning("OSM: no land polygons inside bbox")
        return np.zeros((H, W), dtype=bool)

    transform = from_bounds(w, s, e, n, W, H)
    shapes = [(geom, 1) for geom in gdf.geometry if geom.is_valid and not geom.is_empty]
    land = rasterize(shapes, out_shape=(H, W), transform=transform,
                      fill=0, dtype=np.uint8).astype(bool)

    # Dilate slightly to absorb sub-pixel coastline jitter (≈ 1 px = ~2 m)
    try:
        from scipy.ndimage import binary_dilation
        land = binary_dilation(land, iterations=1)
    except Exception:
        pass

    np.savez_compressed(cache_path, mask=land.astype(np.uint8))
    L.info(f"OSM land mask: {len(gdf)} polygons → {100*land.mean():.1f}% land  "
           f"(cached {cache_path.name})")
    return land


# ════════════════════════════════════════════════════════════════════════
# 3. Water mask + per-pixel features
# ════════════════════════════════════════════════════════════════════════
def water_mask_combined(rgb_vhr: np.ndarray,
                         s2_resampled: Optional[Dict],
                         osm_land: Optional[np.ndarray] = None,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Robust water mask = (OSM water polygon ∧ NDWI water ∧ VHR sane).

    OSM is the **definitive** land/water boundary from OpenStreetMap.
    NDWI / RGB checks remove dynamic non-water (wakes, foam, shadows).
    Land pixels later become NaN in the depth output.

    Returns ``(water_mask, ndwi)``.
    """
    r = rgb_vhr[..., 0].astype(np.float32)
    g_v = rgb_vhr[..., 1].astype(np.float32)
    b_v = rgb_vhr[..., 2].astype(np.float32)
    bright = r + g_v + b_v
    not_white = bright < 730
    not_dark = bright > 25

    if s2_resampled is not None:
        g = s2_resampled["green"].astype(np.float32) + 1.0
        nir = s2_resampled["nir"].astype(np.float32) + 1.0
        ndwi = (g - nir) / (g + nir + 1e-6)
        s2_water = ndwi > 0.0
        scl = s2_resampled.get("scl")
        if scl is not None:
            s2_water = s2_water | (scl == 6)
        sat_water = s2_water & not_white & not_dark
    else:
        ndwi = ((b_v - r) / (b_v + r + 1e-6)).astype(np.float32)
        blue_ratio = b_v / (bright + 1.0)
        sat_water = (blue_ratio > 0.34) & (b_v > r) & not_white & not_dark

    # OSM is authoritative for land — anything OSM calls land is NEVER water.
    if osm_land is not None and osm_land.shape == sat_water.shape:
        mask = sat_water & ~osm_land.astype(bool)
    else:
        mask = sat_water

    try:
        from scipy.ndimage import binary_opening, binary_closing
        mask = binary_closing(binary_opening(mask, iterations=1), iterations=2)
    except Exception:
        pass
    return mask.astype(bool), ndwi.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# Linear bias-correction (fitted on training set, applied everywhere)
# ════════════════════════════════════════════════════════════════════════
def fit_linear_calibration(y_pred_train: np.ndarray, y_true_train: np.ndarray
                           ) -> Tuple[float, float]:
    """Fit ``y_true ≈ α + β · y_pred`` via least squares.

    Robust to a few outliers thanks to a single Huber-weighted re-fit.
    Returns (alpha, beta).
    """
    yt = np.asarray(y_true_train, dtype=np.float64)
    yp = np.asarray(y_pred_train, dtype=np.float64)
    if len(yt) < 3:
        return 0.0, 1.0
    A = np.stack([np.ones_like(yp), yp], axis=1)
    sol, *_ = np.linalg.lstsq(A, yt, rcond=None)
    alpha, beta = float(sol[0]), float(sol[1])
    # One Huber-weighted refit
    res = yt - (alpha + beta * yp)
    mad = np.median(np.abs(res - np.median(res))) + 1e-3
    delta = 1.345 * 1.4826 * mad  # Huber threshold
    w = np.where(np.abs(res) <= delta, 1.0, delta / (np.abs(res) + 1e-9))
    Aw = A * np.sqrt(w)[:, None]
    yw = yt * np.sqrt(w)
    sol2, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    alpha, beta = float(sol2[0]), float(sol2[1])
    return alpha, beta


def apply_calibration(arr: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Apply ``α + β · arr``, preserving NaNs and clipping to [0, MAX_DEPTH_M]."""
    out = alpha + beta * arr
    out = np.clip(out, 0.0, MAX_DEPTH_M)
    out = np.where(np.isfinite(arr), out, np.nan)
    return out


def build_features(
    rgb_vhr: np.ndarray,
    s2_resampled: Optional[Dict],
    ndwi: np.ndarray,
    water: np.ndarray,
    era5: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build a 16-channel feature stack (10 channels when S2 is unavailable).

    Channels (S2 path):
      0..2  VHR R/G/B (deglinted)
      3..6  S2 B/G/R/NIR (DN/10000)
      7     NDWI
      8..9  Stumpf log-ratios (S2 B/G  · S2 G/R)
      10..15 VHR 5×5 mean/std for R/G/B
    """
    eps = 1.0
    r_v = rgb_vhr[..., 0].astype(np.float32) / 100.0 + eps / 100.0
    g_v = rgb_vhr[..., 1].astype(np.float32) / 100.0 + eps / 100.0
    b_v = rgb_vhr[..., 2].astype(np.float32) / 100.0 + eps / 100.0

    def _dark_corr(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.any():
            d = float(np.percentile(a[mask], 1.0))
        else:
            d = float(np.percentile(a, 5.0))
        return np.maximum(a - d + 1.0, 1.0)

    r_v_c = _dark_corr(r_v, water)
    g_v_c = _dark_corr(g_v, water)
    b_v_c = _dark_corr(b_v, water)

    have_s2 = s2_resampled is not None
    if have_s2:
        blue = s2_resampled["blue"].astype(np.float32) / 10000.0 + 1e-4
        green = s2_resampled["green"].astype(np.float32) / 10000.0 + 1e-4
        red = s2_resampled["red"].astype(np.float32) / 10000.0 + 1e-4
        nir = s2_resampled["nir"].astype(np.float32) / 10000.0 + 1e-4
        stumpf_bg = np.log(1000.0 * blue) / np.log(1000.0 * green + eps)
        stumpf_gr = np.log(1000.0 * green) / np.log(1000.0 * red + eps)
    else:
        # VHR-only fall-back: synthesise log-ratios from Mapbox B/G/R
        stumpf_bg = (np.log(1000.0 * b_v_c) / np.log(1000.0 * g_v_c + eps)).astype(np.float32)
        stumpf_gr = (np.log(1000.0 * g_v_c) / np.log(1000.0 * (r_v_c + eps))).astype(np.float32)

    # 5×5 texture on VHR
    try:
        from scipy.ndimage import uniform_filter

        def _ms(a, k=5):
            m = uniform_filter(a, size=k, mode="reflect")
            m2 = uniform_filter(a * a, size=k, mode="reflect")
            v = np.clip(m2 - m * m, 0.0, None)
            return m.astype(np.float32), np.sqrt(v).astype(np.float32)
        rm, rs = _ms(r_v_c)
        gm, gs = _ms(g_v_c)
        bm, bs = _ms(b_v_c)
    except Exception:
        rm = rs = gm = gs = bm = bs = np.zeros_like(r_v_c)

    layers: List[np.ndarray] = [r_v_c, g_v_c, b_v_c]
    names: List[str] = ["vhr_R", "vhr_G", "vhr_B"]
    if have_s2:
        layers += [blue, green, red, nir]
        names += ["s2_B", "s2_G", "s2_R", "s2_NIR"]
    layers += [ndwi.astype(np.float32),
               stumpf_bg.astype(np.float32), stumpf_gr.astype(np.float32)]
    names += ["ndwi" if have_s2 else "ndwi_proxy",
              "stumpf_BG" if have_s2 else "stumpf_BG_vhr",
              "stumpf_GR" if have_s2 else "stumpf_GR_vhr"]
    layers += [rm, rs, gm, gs, bm, bs]
    names += ["vhr_R_mean5", "vhr_R_std5",
              "vhr_G_mean5", "vhr_G_std5",
              "vhr_B_mean5", "vhr_B_std5"]

    if era5 is not None:
        for k in ("u10", "v10", "wind_speed", "t2m", "sst"):
            if k in era5 and era5[k] is not None:
                layers.append(era5[k].astype(np.float32))
                names.append(f"era5_{k}")

    feats = np.stack(layers, axis=-1).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, names


# ════════════════════════════════════════════════════════════════════════
# 4. Dilated FCN (PyTorch)
# ════════════════════════════════════════════════════════════════════════
def _build_cnn(c_in: int):
    import torch
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, c: int, dilation: int = 1, p_drop: float = 0.0):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(c), nn.GELU(),
                nn.Dropout2d(p_drop),
                nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(c),
            )
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(x + self.body(x))

    class DilatedFCN(nn.Module):
        """Deeper residual dilated FCN, bounded sigmoid output ∈ [0, 25] m.

        Stem (5×5) → 5 residual dilated blocks (d=1,2,4,8,2) → 3×3 head.
        ≈ 0.7 M parameters, receptive field ≈ 65 px (≈ 130 m at 2 m/px).
        """
        def __init__(self, c_in: int, c_hidden: int = 96, p_drop: float = 0.15):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(c_in, 48, 5, padding=2), nn.BatchNorm2d(48), nn.GELU(),
                nn.Conv2d(48, c_hidden, 3, padding=1), nn.BatchNorm2d(c_hidden), nn.GELU(),
            )
            self.body = nn.Sequential(
                ResBlock(c_hidden, dilation=1, p_drop=0.05),
                ResBlock(c_hidden, dilation=2, p_drop=0.05),
                ResBlock(c_hidden, dilation=4, p_drop=0.10),
                ResBlock(c_hidden, dilation=8, p_drop=0.10),
                ResBlock(c_hidden, dilation=2, p_drop=0.05),
            )
            self.head = nn.Sequential(
                nn.Conv2d(c_hidden, 48, 3, padding=1), nn.BatchNorm2d(48), nn.GELU(),
                nn.Dropout2d(p_drop),
                nn.Conv2d(48, 1, 1),
            )
            self.depth_max = float(MAX_DEPTH_M)

        def forward(self, x):
            x = self.stem(x)
            x = self.body(x)
            x = self.head(x)
            return self.depth_max * torch.sigmoid(x).squeeze(1)
    return DilatedFCN(c_in)


def train_cnn(
    feats: np.ndarray,
    label_grid: np.ndarray,
    label_mask: np.ndarray,
    epochs: int = 30,
    crops_per_epoch: int = 60,
    batch: int = 4,
    crop: int = 192,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Train the FCN with random crops + masked MSE."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    H, W, C = feats.shape

    # Per-channel z-score normalisation (computed on water pixels only)
    feats_z = feats.copy()
    for ci in range(C):
        ch = feats_z[..., ci]
        m = float(ch.mean())
        sd = float(ch.std()) + 1e-6
        feats_z[..., ci] = (ch - m) / sd
    feats_t = torch.from_numpy(feats_z.transpose(2, 0, 1)).float()        # (C, H, W)
    target_t = torch.from_numpy(label_grid.astype(np.float32))            # (H, W)
    mask_t = torch.from_numpy(label_mask.astype(np.float32))              # (H, W)

    label_idx_h, label_idx_w = np.where(label_mask)
    n_lbl = len(label_idx_h)
    if n_lbl < 30:
        raise RuntimeError(f"only {n_lbl} labelled pixels in grid")
    L.info(f"CNN: {n_lbl} labelled px, training {epochs} epochs × "
           f"{crops_per_epoch} crops × batch {batch} (crop={crop})")

    model = _build_cnn(C).float()
    n_params = sum(p.numel() for p in model.parameters())
    L.info(f"CNN: dilated FCN, {n_params:,} params, {C} input channels")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Depth-band weights: equalise rare deep bins so they aren't drowned out
    bins = np.array([0, 5, 10, 15, 20, 25 + 1e-3])
    band_idx = np.digitize(label_grid[label_mask], bins) - 1
    band_counts = np.bincount(band_idx, minlength=len(bins) - 1).astype(np.float32) + 1
    band_w = (band_counts.mean() / band_counts).clip(0.5, 4.0)  # gentle reweighting
    weight_grid = np.zeros_like(label_grid, dtype=np.float32)
    weight_grid[label_mask] = band_w[band_idx]
    weight_t = torch.from_numpy(weight_grid)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        ep_n = 0
        for _ in range(crops_per_epoch):
            xs, ys, ms, ws = [], [], [], []
            for _ in range(batch):
                k = int(rng.integers(0, n_lbl))
                cy = int(label_idx_h[k]); cx = int(label_idx_w[k])
                y0 = int(np.clip(cy - crop // 2 + rng.integers(-30, 30), 0, H - crop))
                x0 = int(np.clip(cx - crop // 2 + rng.integers(-30, 30), 0, W - crop))
                xs.append(feats_t[:, y0:y0 + crop, x0:x0 + crop])
                ys.append(target_t[y0:y0 + crop, x0:x0 + crop])
                ms.append(mask_t[y0:y0 + crop, x0:x0 + crop])
                ws.append(weight_t[y0:y0 + crop, x0:x0 + crop])
            x = torch.stack(xs)
            y = torch.stack(ys)
            m = torch.stack(ms)
            w = torch.stack(ws)
            opt.zero_grad()
            pred = model(x)                                   # (B, H, W)
            # Huber (smooth-L1) — robust to shallow/deep outliers
            diff = pred - y
            abs_diff = diff.abs()
            delta = 2.0
            quad = 0.5 * diff * diff
            lin = delta * (abs_diff - 0.5 * delta)
            elem = torch.where(abs_diff <= delta, quad, lin)
            loss = (elem * m * w).sum() / (m.sum().clamp(min=1.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach()); ep_n += 1
        sched.step()
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            L.info(f"  epoch {ep+1:3d}/{epochs}  loss={ep_loss/ep_n:.4f}  "
                   f"lr={sched.get_last_lr()[0]:.5f}  elapsed={time.time()-t0:.0f}s")
    L.info(f"CNN trained in {time.time()-t0:.0f}s")
    # Stash channel norm so inference uses the same transform
    model.feat_mean = np.array([feats[..., ci].mean() for ci in range(C)], dtype=np.float32)
    model.feat_std = np.array([feats[..., ci].std() + 1e-6 for ci in range(C)], dtype=np.float32)
    return model


def predict_grid(model, feats: np.ndarray, water: np.ndarray,
                 tile: int = 1024, overlap: int = 32) -> np.ndarray:
    import torch
    H, W, C = feats.shape
    feats_z = (feats - model.feat_mean) / model.feat_std
    feats_t = torch.from_numpy(feats_z.transpose(2, 0, 1)).unsqueeze(0).float()
    out = np.zeros((H, W), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        y = 0
        while y < H:
            y_end = min(y + tile, H)
            x = 0
            while x < W:
                x_end = min(x + tile, W)
                yy0 = max(0, y - overlap); yy1 = min(H, y_end + overlap)
                xx0 = max(0, x - overlap); xx1 = min(W, x_end + overlap)
                inp = feats_t[:, :, yy0:yy1, xx0:xx1]
                pred = model(inp).squeeze(0).cpu().numpy()
                py0 = y - yy0; py1 = py0 + (y_end - y)
                px0 = x - xx0; px1 = px0 + (x_end - x)
                out[y:y_end, x:x_end] = pred[py0:py1, px0:px1]
                x = x_end
            y = y_end
    out = np.clip(out, 0, MAX_DEPTH_M)
    out = np.where(water, out, np.nan)
    return out


# ════════════════════════════════════════════════════════════════════════
# 5. Train/test split + metrics
# ════════════════════════════════════════════════════════════════════════
def stratified_split(depths: np.ndarray, train_frac: float = 0.20,
                     seed: int = 42,
                     lats: Optional[np.ndarray] = None,
                     lons: Optional[np.ndarray] = None,
                     mode: str = "stratified",
                     n_spatial_blocks: int = 25,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Split reference samples into train / test.

    mode="stratified" — depth-stratified random per-bin (legacy default).

    mode="spatial_block" — KMeans-cluster the (lat, lon) coordinates into
    ``n_spatial_blocks`` cells; assign whole cells to train/test.  This
    eliminates the spatial-autocorrelation leakage caused by 7×7 label
    patches and S2's natural reflectance autocorrelation length, which
    otherwise inflates train metrics without any generalisation gain.
    Within each train block we still stratify by depth bin so the deep
    end isn't starved.

    The function is backwards compatible: callers that don't pass
    coordinates get the original stratified behaviour.
    """
    n = len(depths)
    rng = np.random.default_rng(seed)
    bins = np.array([0, 2, 5, 10, 15, 20, MAX_DEPTH_M + 1e-3])

    if mode == "spatial_block" and lats is not None and lons is not None and n >= 50:
        try:
            from sklearn.cluster import KMeans
            xy = np.column_stack([np.asarray(lats, dtype=np.float64),
                                   np.asarray(lons, dtype=np.float64)])
            k = max(2, min(n_spatial_blocks, n // 5))
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(xy)
            block = km.labels_
            # Pick blocks for training to roughly hit train_frac, picking
            # in order of block id so the choice is deterministic given seed.
            block_order = list(range(k))
            rng.shuffle(block_order)
            train = np.zeros(n, dtype=bool)
            cum_frac = 0.0
            for bid in block_order:
                in_block = (block == bid)
                if cum_frac >= train_frac:
                    break
                train |= in_block
                cum_frac = float(train.mean())
            # Within each TRAIN block, additionally drop a fraction so the
            # deep-end stays represented (spatial blocks can be depth-biased).
            depth_labels = np.digitize(depths, bins) - 1
            for k_bin in range(len(bins) - 1):
                tr_in_bin = train & (depth_labels == k_bin)
                if tr_in_bin.sum() < 4:
                    # promote a few test-set samples in this bin to train
                    candidates = np.where(~train & (depth_labels == k_bin))[0]
                    take = min(4 - int(tr_in_bin.sum()), len(candidates))
                    if take > 0:
                        sel = rng.choice(candidates, take, replace=False)
                        train[sel] = True
            return train, ~train
        except Exception as ex:
            L.warning(f"spatial_block split failed ({ex}); falling back to stratified")

    # Default: depth-stratified random per-bin
    labels = np.digitize(depths, bins) - 1
    train = np.zeros(n, dtype=bool)
    for k in range(len(bins) - 1):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        ntr = max(1, int(round(len(idx) * train_frac))) if len(idx) >= 5 else len(idx) // 2
        sel = rng.choice(idx, ntr, replace=False)
        train[sel] = True
    return train, ~train


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-9))
    return {
        "rmse_m": round(rmse, 3),
        "mae_m": round(mae, 3),
        "bias_m": round(bias, 3),
        "r2": round(r2, 4),
        "n": int(len(y_true)),
        "depth_min": round(float(y_true.min()), 2),
        "depth_max": round(float(y_true.max()), 2),
    }


def s44_pass_pct(y_true: np.ndarray, y_pred: np.ndarray,
                 a: float = 0.5, b: float = 0.013, k: float = 1.96) -> float:
    """IHO S-44 pass rate at confidence k. TVU = √(a² + (b·d)²)."""
    if len(y_true) == 0:
        return 0.0
    tvu = np.sqrt(a ** 2 + (b * y_true) ** 2)
    return float(100.0 * np.mean(np.abs(y_pred - y_true) <= k * tvu))


def metrics_per_band(y_true: np.ndarray, y_pred: np.ndarray,
                     bands: tuple = DEPTH_BANDS) -> List[Dict]:
    out: List[Dict] = []
    for lo, hi in bands:
        m = (y_true >= lo) & (y_true < hi)
        nb = int(m.sum())
        entry = {"range": [lo, hi], "n": nb}
        if nb >= 1:
            yt, yp = y_true[m], y_pred[m]
            entry.update({
                "rmse_m": round(float(np.sqrt(np.mean((yp - yt) ** 2))), 3),
                "mae_m": round(float(np.mean(np.abs(yp - yt))), 3),
                "bias_m": round(float(np.mean(yp - yt)), 3),
                # IHO 1A and 1B share a common vertical TVU (a=0.5, b=0.013).
                # Two columns are kept so the chart matches user-requested layout.
                "s44_1a_pct": round(s44_pass_pct(yt, yp, a=0.5, b=0.013), 1),
                "s44_1b_pct": round(s44_pass_pct(yt, yp, a=0.5, b=0.013), 1),
                "s44_order2_pct": round(s44_pass_pct(yt, yp, a=1.0, b=0.023), 1),
                "s44_special_pct": round(s44_pass_pct(yt, yp, a=0.25, b=0.0075), 1),
            })
        out.append(entry)
    return out


# ════════════════════════════════════════════════════════════════════════
# 6. Save artefacts (extended)
# ════════════════════════════════════════════════════════════════════════
def _depth_to_rgb(depth: np.ndarray, vmax: float = MAX_DEPTH_M) -> np.ndarray:
    nan = ~np.isfinite(depth)
    safe = np.where(nan, 0.0, depth)
    d = np.clip(safe, 0, vmax) / vmax
    stops = np.array([
        [0.267, 0.005, 0.329], [0.282, 0.140, 0.458], [0.254, 0.265, 0.530],
        [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
        [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
        [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
    ])
    idx_f = d * (len(stops) - 1)
    i0 = np.clip(idx_f.astype(int), 0, len(stops) - 2)
    t = (idx_f - i0).astype(np.float32)[..., None]
    rgb = (1 - t) * stops[i0] + t * stops[i0 + 1]
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    rgb[nan] = 30
    return rgb


def _draw_band_bars(per_band: List[Dict], out_path: Path, title: str):
    """Grouped bar chart: S-44 1A / 1B / Order 2 pass % per depth band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{lo}-{hi}" for lo, hi in DEPTH_BANDS]
    pct_1a = [b.get("s44_1a_pct", 0) for b in per_band]
    pct_1b = [b.get("s44_1b_pct", 0) for b in per_band]
    pct_o2 = [b.get("s44_order2_pct", 0) for b in per_band]
    counts = [b.get("n", 0) for b in per_band]

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)
    b1 = ax.bar(x - w, pct_1a, w, label="S-44 Order 1A", color="#0d9488",
                edgecolor="#0f172a", linewidth=0.6)
    b2 = ax.bar(x,       pct_1b, w, label="S-44 Order 1B", color="#0ea5e9",
                edgecolor="#0f172a", linewidth=0.6)
    b3 = ax.bar(x + w,   pct_o2, w, label="S-44 Order 2", color="#f59e0b",
                edgecolor="#0f172a", linewidth=0.6, alpha=0.85)
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} m\nn={n}" for lab, n in zip(labels, counts)],
                        fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Pass rate (%)", fontsize=10)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.axhline(95, ls="--", lw=0.7, color="#475569", alpha=0.5)
    ax.text(len(labels) - 0.4, 96, "95 % target", fontsize=8, color="#475569")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(axis="y", ls=":", lw=0.4, color="#cbd5e1")
    ax.set_axisbelow(True)
    fig.text(0.99, 0.01,
             "TVU₁ = ±√(0.50² + (0.013·d)²)   TVU₂ = ±√(1.00² + (0.023·d)²)  · 95 % CI",
             ha="right", fontsize=7.5, color="#475569")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _draw_scatter_mpl(y_true, y_pred, out_path, title, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 6), dpi=120)
    ax.scatter(y_true, y_pred, s=2, alpha=0.18, color="#0d9488", edgecolors="none")
    ax.plot([0, MAX_DEPTH_M], [0, MAX_DEPTH_M], color="#dc2626",
            ls="--", lw=1.0, label="1 : 1")
    ax.set_xlim(0, MAX_DEPTH_M); ax.set_ylim(0, MAX_DEPTH_M)
    ax.set_xlabel("Measured depth (m)"); ax.set_ylabel("Predicted depth (m)")
    ax.set_title(title, fontsize=11, weight="bold")
    txt = (f"RMSE = {metrics['rmse_m']} m   MAE = {metrics['mae_m']} m\n"
           f"R² = {metrics['r2']}   bias = {metrics['bias_m']} m\n"
           f"n = {metrics['n']:,}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=9,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1"))
    ax.grid(ls=":", lw=0.4, color="#cbd5e1")
    ax.set_axisbelow(True); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def save_artefacts(
    site_key: str,
    bbox: List[float],
    rgb_image: np.ndarray,
    s2_resampled: Dict,
    water: np.ndarray,
    result: Dict,
    out_dir: Path = RESULTS_DIR,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # VHR mosaic
    vhr_path = out_dir / f"{site_key}_vhr_image.png"
    Image.fromarray(rgb_image).save(vhr_path, optimize=True)
    paths["vhr_image"] = str(vhr_path)

    # S2 RGB (resampled to VHR grid) — only when S2 was actually fetched
    if s2_resampled is not None:
        try:
            s2_rgb = np.stack([
                np.clip(s2_resampled["red"] / 40.0, 0, 255).astype(np.uint8),
                np.clip(s2_resampled["green"] / 40.0, 0, 255).astype(np.uint8),
                np.clip(s2_resampled["blue"] / 40.0, 0, 255).astype(np.uint8),
            ], axis=-1)
            s2_path = out_dir / f"{site_key}_s2_rgb.png"
            Image.fromarray(s2_rgb).save(s2_path, optimize=True)
            paths["s2_rgb"] = str(s2_path)
        except Exception as ex:
            L.warning(f"S2 RGB save skipped: {ex}")

    # Water mask overlay — water = full-colour VHR, LAND = red tint (NaN region)
    overlay_water = rgb_image.copy()
    nl = ~water
    if nl.any():
        red_tint = np.array([200, 80, 80], dtype=np.float32)
        overlay_water[nl] = (overlay_water[nl].astype(np.float32) * 0.35
                              + red_tint * 0.65).astype(np.uint8)
    water_path = out_dir / f"{site_key}_water_mask.png"
    Image.fromarray(overlay_water).save(water_path, optimize=True)
    paths["water_mask"] = str(water_path)

    # OSM land overlay (separate file showing JUST the OSM polygons)
    land = result.get("osm_land")
    if land is not None and land.shape == rgb_image.shape[:2]:
        osm_overlay = rgb_image.copy()
        if land.any():
            yellow = np.array([240, 200, 40], dtype=np.float32)
            osm_overlay[land] = (osm_overlay[land].astype(np.float32) * 0.55
                                  + yellow * 0.45).astype(np.uint8)
        osm_path = out_dir / f"{site_key}_osm_land.png"
        Image.fromarray(osm_overlay).save(osm_path, optimize=True)
        paths["osm_land"] = str(osm_path)

    # Depth viz + overlay
    depth_rgb = _depth_to_rgb(result["depth"])
    depth_png = out_dir / f"{site_key}_depth.png"
    Image.fromarray(depth_rgb).save(depth_png, optimize=True)
    paths["depth_png"] = str(depth_png)

    overlay = (0.45 * rgb_image.astype(np.float32) + 0.55 * depth_rgb.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_png = out_dir / f"{site_key}_overlay.png"
    Image.fromarray(overlay).save(overlay_png, optimize=True)
    paths["overlay_png"] = str(overlay_png)

    # GeoTIFF
    try:
        import rasterio
        from rasterio.transform import from_bounds
        H, W = result["depth"].shape
        w, s, e, n = bbox
        transform = from_bounds(w, s, e, n, W, H)
        tif_path = out_dir / f"{site_key}_depth.tif"
        out = result["depth"].astype(np.float32)
        out = np.where(np.isfinite(out), out, -9999.0)
        with rasterio.open(
            tif_path, "w", driver="GTiff",
            width=W, height=H, count=1, dtype="float32",
            crs="EPSG:4326", transform=transform, nodata=-9999.0,
            compress="deflate",
        ) as dst:
            dst.write(out, 1)
        paths["depth_tif"] = str(tif_path)
    except Exception as ex:
        L.warning(f"GeoTIFF skipped: {ex}")

    # Scatter (matplotlib)
    try:
        sc_path = out_dir / f"{site_key}_scatter.png"
        _draw_scatter_mpl(result["y_true_test"], result["y_pred_test"], sc_path,
                          f"{site_key} — held-out test (CNN, {len(result['y_true_test']):,} pts)",
                          result["metrics"])
        paths["scatter_png"] = str(sc_path)
    except Exception as ex:
        L.warning(f"scatter skipped: {ex}")

    # Per-band bar chart
    try:
        bars_path = out_dir / f"{site_key}_s44_bands.png"
        _draw_band_bars(result["per_band"], bars_path,
                        title=f"S-44 compliance by depth band — {site_key}")
        paths["s44_bands_png"] = str(bars_path)
    except Exception as ex:
        L.warning(f"S-44 bands skipped: {ex}")

    # Metrics JSON
    meta = {
        "site": site_key,
        "bbox": bbox,
        "vhr_resolution_m": float(result.get("resolution_m", 0)),
        "imagery_source": result.get("imagery_source", "vhr"),
        "method": (
            ("Sentinel-2 L2A (10 m) + ERA-5/SST + OSM land mask + dilated FCN CNN "
             "+ linear bias-correction · 20 % train / 80 % held-out test · "
             "capped at 25 m")
            if result.get("imagery_source") == "s2" else
            ("Mapbox VHR + S2 + ERA-5/SST + OSM land mask + dilated FCN CNN "
             "+ linear bias-correction · 20 % train / 80 % held-out test · "
             "capped at 25 m")
        ),
        "s2_status": result.get("s2_status", "unknown"),
        "era5_status": result.get("era5_status", "unknown"),
        "max_depth_cap_m": MAX_DEPTH_M,
        "calibration": result.get("calibration", {}),
        "metrics_test_overall": result["metrics"],
        "metrics_test_per_band": result["per_band"],
        "feature_channels": result.get("feature_names", []),
        "elapsed_s": result.get("elapsed_s", 0),
    }
    metrics_path = out_dir / f"{site_key}_metrics.json"
    metrics_path.write_text(json.dumps(meta, indent=2))
    paths["metrics_json"] = str(metrics_path)

    L.info("Artefacts saved:")
    for k, v in paths.items():
        L.info(f"  {k:14s} → {v}")
    return paths


# ════════════════════════════════════════════════════════════════════════
# 7. End-to-end driver
# ════════════════════════════════════════════════════════════════════════
def _s2_dn_to_rgb_uint8(s2: Dict) -> np.ndarray:
    """Stretch S2 surface-reflectance DN (uint16, ~0–10000) to display uint8.

    Uses a 2nd–98th percentile per-band stretch which matches the look the
    rest of the pipeline expects from the Mapbox VHR mosaic.
    """
    out = np.empty(s2["red"].shape + (3,), dtype=np.uint8)
    for i, k in enumerate(("red", "green", "blue")):
        a = s2[k].astype(np.float32)
        lo, hi = np.percentile(a, (2.0, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        out[..., i] = np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return out


def run_very_hr(
    site_key: str,
    bbox: List[float],
    ref_lats: np.ndarray,
    ref_lons: np.ndarray,
    ref_depths: np.ndarray,
    train_frac: float = 0.20,
    target_res_m: float = 2.0,
    max_tiles_per_side: int = 4,
    s2_start_date: str = "2024-05-01",
    s2_end_date: str = "2024-09-30",
    cnn_epochs: int = 30,
    crop: int = 192,
    batch: int = 4,
    seed: int = 42,
    save: bool = True,
    imagery_source: str = "vhr",
) -> Dict:
    """Run the Very HR pipeline.

    ``imagery_source``:
      * ``"vhr"`` (default) — Mapbox VHR mosaic at ``target_res_m`` is the base
        raster; Sentinel-2 is resampled onto it as auxiliary bands.
      * ``"s2"`` — no Mapbox call.  S2 L2A at native ~10 m is the base raster;
        the synthesised "vhr" RGB is just a stretched S2 R/G/B for visualisation
        and texture features.  Use this for sites with no VHR coverage or to
        keep the pipeline fully on free imagery.
    """
    t0 = time.time()
    imagery_source = (imagery_source or "vhr").lower()
    if imagery_source not in {"vhr", "s2"}:
        raise ValueError(f"imagery_source must be 'vhr' or 's2', got {imagery_source!r}")

    s2_rs: Optional[Dict] = None
    s2_status = "unavailable"

    if imagery_source == "vhr":
        # 1. Mapbox VHR
        vhr = fetch_mapbox_vhr(bbox, target_res_m=target_res_m,
                                max_tiles_per_side=max_tiles_per_side)
        rgb = vhr["rgb"]
        H, W = rgb.shape[:2]

        # 2. Sentinel-2 — Sentinel-Hub first, GEE second; both optional.
        try:
            s2_raw = fetch_s2(bbox, s2_start_date, s2_end_date, res_m=10, cloud=20)
            s2_rs = {
                "blue":  _resample(s2_raw["blue"], H, W),
                "green": _resample(s2_raw["green"], H, W),
                "red":   _resample(s2_raw["red"], H, W),
                "nir":   _resample(s2_raw["nir"], H, W),
                "scl":   _resample(s2_raw["scl"], H, W, resample=Image.NEAREST).astype(np.uint8),
            }
            s2_status = f"SH  {s2_raw['width']}×{s2_raw['height']} @ 10 m"
            L.info(f"S2 attached: {s2_status}")
        except Exception as ex:
            L.warning(f"SH S2 failed ({ex}) — trying GEE fallback")
            try:
                s2_gee = fetch_s2_gee(bbox, s2_start_date, s2_end_date)
                s2_rs = {
                    "blue":  _resample(s2_gee["blue"],  H, W),
                    "green": _resample(s2_gee["green"], H, W),
                    "red":   _resample(s2_gee["red"],   H, W),
                    "nir":   _resample(s2_gee["nir"],   H, W),
                    "scl":   _resample(s2_gee.get("scl", np.zeros_like(s2_gee["blue"])),
                                        H, W, resample=Image.NEAREST).astype(np.uint8),
                }
                s2_status = f"GEE {s2_gee['width']}×{s2_gee['height']} @ {s2_gee.get('resolution_m',10)} m"
                L.info(f"S2 attached via GEE: {s2_status}")
            except Exception as ex2:
                L.warning(f"GEE S2 also failed: {ex2} — VHR-only features")
                s2_status = f"unavailable ({ex2})"
    else:
        # S2-only: native 10 m grid is the base raster, no Mapbox call.
        try:
            s2_raw = fetch_s2(bbox, s2_start_date, s2_end_date, res_m=10, cloud=20)
            s2_status = f"SH  {s2_raw['width']}×{s2_raw['height']} @ 10 m (base raster)"
        except Exception as ex:
            L.warning(f"SH S2 failed ({ex}) — trying GEE fallback for S2-only base")
            try:
                s2_gee = fetch_s2_gee(bbox, s2_start_date, s2_end_date)
                s2_raw = {
                    "blue":  s2_gee["blue"],
                    "green": s2_gee["green"],
                    "red":   s2_gee["red"],
                    "nir":   s2_gee["nir"],
                    "scl":   s2_gee.get("scl",
                                        np.zeros_like(s2_gee["blue"], dtype=np.uint8)),
                    "width": s2_gee["width"],
                    "height": s2_gee["height"],
                    "resolution_m": s2_gee.get("resolution_m", 10),
                }
                s2_status = (f"GEE {s2_raw['width']}×{s2_raw['height']} @ "
                             f"{s2_raw['resolution_m']} m (base raster)")
            except Exception as ex2:
                raise RuntimeError(
                    f"S2-only mode requires S2 imagery but none could be fetched "
                    f"(SH: {ex}; GEE: {ex2})"
                )
        rgb = _s2_dn_to_rgb_uint8(s2_raw)
        H, W = rgb.shape[:2]
        s2_rs = {
            "blue":  s2_raw["blue"].astype(np.uint16),
            "green": s2_raw["green"].astype(np.uint16),
            "red":   s2_raw["red"].astype(np.uint16),
            "nir":   s2_raw["nir"].astype(np.uint16),
            "scl":   np.asarray(s2_raw["scl"], dtype=np.uint8),
        }
        # Native S2 ground sample distance over the bbox (≈ 10 m)
        w_, s_, e_, n_ = bbox
        cl = math.cos(math.radians((n_ + s_) / 2))
        bbox_w_m = abs(e_ - w_) * 111000.0 * cl
        bbox_h_m = abs(n_ - s_) * 111000.0
        res_native = ((bbox_w_m / W) + (bbox_h_m / H)) / 2
        vhr = {"rgb": rgb, "width": W, "height": H,
               "resolution_m": round(res_native, 3),
               "tiles": [1, 1]}
        L.info(f"S2-only base raster: {W}×{H}px ≈ {res_native:.2f} m/px ({s2_status})")

    # 3. ERA-5 reanalysis + GHRSST SST via GEE — coarse but global, free.
    era5_rs: Optional[Dict] = None
    era5_status = "unavailable"
    try:
        era5 = fetch_era5_sst_gee(bbox, s2_start_date, s2_end_date)
        era5_rs = {k: _resample(v, H, W) for k, v in era5["bands"].items()}
        era5_status = (f"GEE  ERA5 + GHRSST  ({len(era5['bands'])} bands)  "
                        f"window {s2_start_date}→{s2_end_date}")
        L.info(f"ERA-5 / SST attached: {era5_status}")
    except Exception as ex:
        L.warning(f"ERA-5 / SST unavailable: {ex}")
        era5_status = f"unavailable ({ex})"

    # 4. OSM land mask (definitive) + composite water mask
    land = osm_land_mask(bbox, H, W)
    water, ndwi = water_mask_combined(rgb, s2_rs, osm_land=land)
    L.info(f"Water mask: {int(water.sum()):,}/{water.size:,} px "
           f"({100*water.mean():.1f}%)  · OSM land = {100*land.mean():.1f}%")

    # 5. Features
    feats, names = build_features(rgb, s2_rs, ndwi, water, era5=era5_rs)
    L.info(f"Features: {feats.shape[2]} channels — {names}")

    # 5. Sample labels onto pixel grid (cap at 25 m on input!)
    w_, s_, e_, n_ = bbox
    cols = ((np.asarray(ref_lons) - w_) / (e_ - w_) * W).astype(int)
    rows = ((n_ - np.asarray(ref_lats)) / (n_ - s_) * H).astype(int)
    deps = np.clip(np.asarray(ref_depths, dtype=np.float64), 0, MAX_DEPTH_M)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & (deps > 0)
    rows, cols, deps = rows[valid], cols[valid], deps[valid]
    on_water = water[rows, cols]
    rows, cols, deps = rows[on_water], cols[on_water], deps[on_water]
    if len(deps) < 30:
        raise RuntimeError(f"only {len(deps)} in-situ on-water points")
    L.info(f"In-situ on water: {len(deps):,} pts  range {deps.min():.2f}–{deps.max():.2f} m")

    # 6. Stratified train / test split
    train_idx, test_idx = stratified_split(deps, train_frac=train_frac, seed=seed)
    L.info(f"Split: {int(train_idx.sum()):,} train ({100*train_frac:.0f}%) / "
           f"{int(test_idx.sum()):,} test")

    # 7. Build train-only label grid + mask
    label_grid = np.zeros((H, W), dtype=np.float32)
    label_mask = np.zeros((H, W), dtype=bool)
    label_grid[rows[train_idx], cols[train_idx]] = deps[train_idx].astype(np.float32)
    label_mask[rows[train_idx], cols[train_idx]] = True

    # 8. Train CNN
    model = train_cnn(
        feats, label_grid, label_mask,
        epochs=cnn_epochs, crops_per_epoch=60, batch=batch, crop=crop,
        lr=1e-3, seed=seed,
    )

    # 9. Predict full grid + extract test predictions
    depth_pred_raw = predict_grid(model, feats, water, tile=1024, overlap=32)
    yp_full_raw = depth_pred_raw[rows, cols]

    # 10. Linear bias correction fitted on TRAINING set only
    yp_train_raw = np.clip(yp_full_raw[train_idx], 0, MAX_DEPTH_M)
    yt_train = deps[train_idx]
    alpha, beta = fit_linear_calibration(yp_train_raw, yt_train)
    L.info(f"Calibration: y_true ≈ {alpha:+.3f} + {beta:.4f} · y_pred  "
           f"(fitted on {int(train_idx.sum())} train pts)")
    depth_pred = apply_calibration(depth_pred_raw, alpha, beta)
    yp_full = alpha + beta * yp_full_raw
    yt_test = deps[test_idx]
    yp_test = np.clip(yp_full[test_idx], 0, MAX_DEPTH_M)

    # Pre-calibration metrics for diagnostics
    metrics_raw = compute_metrics(yt_test, np.clip(yp_full_raw[test_idx], 0, MAX_DEPTH_M))

    metrics = compute_metrics(yt_test, yp_test)
    metrics["calibration_alpha"] = round(alpha, 4)
    metrics["calibration_beta"] = round(beta, 4)
    metrics["bias_before_calibration_m"] = metrics_raw["bias_m"]
    metrics["rmse_before_calibration_m"] = metrics_raw["rmse_m"]
    metrics["s44_1a_pct"] = round(s44_pass_pct(yt_test, yp_test, a=0.5, b=0.013), 1)
    metrics["s44_1b_pct"] = round(s44_pass_pct(yt_test, yp_test, a=0.5, b=0.013), 1)
    metrics["s44_order2_pct"] = round(s44_pass_pct(yt_test, yp_test, a=1.0, b=0.023), 1)
    metrics["s44_special_pct"] = round(s44_pass_pct(yt_test, yp_test, a=0.25, b=0.0075), 1)
    metrics["n_train"] = int(train_idx.sum())
    metrics["n_test"] = int(test_idx.sum())
    metrics["train_fraction"] = float(train_frac)
    per_band = metrics_per_band(yt_test, yp_test)

    elapsed = round(time.time() - t0, 1)
    L.info(f"Done {elapsed}s — RMSE={metrics['rmse_m']}m R²={metrics['r2']} "
           f"S-44 1A={metrics['s44_1a_pct']}% Order2={metrics['s44_order2_pct']}%")

    result = {
        "depth": depth_pred,
        "depth_uncalibrated": depth_pred_raw,
        "water": water,
        "osm_land": land,
        "metrics": metrics,
        "per_band": per_band,
        "y_true_test": yt_test,
        "y_pred_test": yp_test,
        "feature_names": names,
        "resolution_m": vhr["resolution_m"],
        "bbox": bbox,
        "site": site_key,
        "elapsed_s": elapsed,
        "calibration": {"alpha": alpha, "beta": beta},
    }
    result["s2_status"] = s2_status
    result["era5_status"] = era5_status
    result["imagery_source"] = imagery_source
    if save:
        result["paths"] = save_artefacts(site_key, bbox, rgb, s2_rs, water, result)
    return result
