"""Very HR — Clustered SDB.

A different, simpler-than-CNN approach to bathymetry:

1. Fetch Mapbox VHR mosaic + the **full 12-band Sentinel-2 stack** (B1, B2,
   B3, B4, B5, B6, B7, B8, B8A, B11, B12, SCL) via Google Earth Engine.
2. Build a per-pixel feature vector:
   - All 12 S2 bands (deglinted by deep-water dark-pixel correction)
   - 6 Stumpf-style log-band-ratios
   - 3 VHR R/G/B (deglinted)
   - NDWI, NDTI, NDCI (water/turbidity/chlorophyll indices)
3. Mask everything except water (OSM land mask → NaN over land).
4. Run **K-means clustering** (K = 6 by default) on the spectral features
   so each cluster represents a similar bottom-type / water regime.
5. Fit a **HistGradientBoostingRegressor per cluster** on the training
   subset (20 %), weighted by depth-band frequency.
6. Predict full grid via **soft-blend** of the top-3 nearest cluster models.
7. Apply a linear bias-correction (α + β·pred) fit on training residuals.
8. Cap predictions and labels at 25 m. Persist artefacts under
   ``Very_HR_Results/Clustered_Results/``.
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

# Reuse helpers from the engine module
from backend.very_hr_engine import (
    MAX_DEPTH_M, DEPTH_BANDS,
    fetch_mapbox_vhr, osm_land_mask, _resample, _init_gee,
    _gee_download_geotiff, water_mask_combined,
    fit_linear_calibration, apply_calibration,
    compute_metrics, s44_pass_pct, metrics_per_band, stratified_split,
    _draw_band_bars, _draw_scatter_mpl, _depth_to_rgb,
    SH_AUTH_URL, SH_PROCESS_URL, _sh_token,
)

L = logging.getLogger("very_hr_cbr")
if not L.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    L.addHandler(h)
    L.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Very_HR_Results" / "Clustered_Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "Very_HR_Results" / "_model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL"]
S2_BAND_LABELS = {
    "B1":  "coastal",
    "B2":  "blue",
    "B3":  "green",
    "B4":  "red",
    "B5":  "rededge1",
    "B6":  "rededge2",
    "B7":  "rededge3",
    "B8":  "nir",
    "B8A": "nir_narrow",
    "B11": "swir1",
    "B12": "swir2",
}


# ════════════════════════════════════════════════════════════════════════
# 1. Sentinel-2 — full 12-band fetch via GEE
# ════════════════════════════════════════════════════════════════════════
def fetch_s2_full_gee(bbox: List[float], start_date: str, end_date: str,
                      cloud: int = 30) -> Dict:
    """Median composite of all 12 S2 SR bands + SCL via GEE.  ~10 m base scale."""
    if not _init_gee():
        raise RuntimeError("GEE not available")
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))
    n_imgs = coll.size().getInfo()
    L.info(f"S2 full: {n_imgs} scenes in window  (cloud≤{cloud}%)")
    if n_imgs == 0:
        raise RuntimeError("S2 full: 0 scenes in window")
    img = (coll.select(S2_BANDS).median().clip(region))
    bundle = None
    last_err: Optional[Exception] = None
    for scale in (10, 12, 15, 20, 25, 30):
        try:
            bundle = _gee_download_geotiff(img, bbox, scale=scale,
                                             band_order=S2_BANDS, tag="s2full")
            if scale != 10:
                L.info(f"S2 full: succeeded at fallback scale={scale} m")
            break
        except Exception as ex:
            last_err = ex
            msg = str(ex)
            if "Total request size" in msg or "thumbnails" in msg or "exceed" in msg.lower():
                L.warning(f"S2 full @ {scale} m hit thumbnail size cap, retrying coarser")
                continue
            raise
    if bundle is None:
        raise last_err if last_err else RuntimeError("S2 full GEE fetch failed")
    bands = bundle["bands"]
    out: Dict[str, np.ndarray] = {}
    for b in S2_BANDS:
        if b == "SCL":
            out[b] = bands[b].astype(np.uint8)
        else:
            out[b] = bands[b].astype(np.float32)
    out["width"] = bundle["width"]; out["height"] = bundle["height"]
    out["resolution_m"] = bundle["resolution_m"]
    out["window"] = {"start": start_date, "end": end_date}
    return out


# Sentinel-Hub names (with leading zero) → our internal names (no leading zero)
_SH_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                  "B08", "B8A", "B11", "B12", "SCL"]
_SH_TO_INTERNAL = {
    "B01": "B1", "B02": "B2", "B03": "B3", "B04": "B4", "B05": "B5",
    "B06": "B6", "B07": "B7", "B08": "B8", "B8A": "B8A",
    "B11": "B11", "B12": "B12", "SCL": "SCL",
}

_S2_FULL_EVALSCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"],units:"DN",mosaicking:"ORBIT"}],output:{bands:12,sampleType:"UINT16"}};}
function isValid(s){var c=s.SCL;return c!==1&&c!==3&&c!==8&&c!==9&&c!==10&&c!==11;}
function median(a){if(!a.length)return 0;a.sort(function(x,y){return x-y});var m=Math.floor(a.length/2);return a.length%2===0?(a[m-1]+a[m])/2:a[m];}
function evaluatePixel(samples){
  var bands=["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"];
  var pools={};for(var k=0;k<bands.length;k++)pools[bands[k]]=[];
  for(var i=0;i<samples.length;i++){if(isValid(samples[i])){for(var k=0;k<bands.length;k++)pools[bands[k]].push(samples[i][bands[k]]);}}
  if(!pools.B02.length){for(var j=0;j<samples.length;j++){for(var k=0;k<bands.length;k++)pools[bands[k]].push(samples[j][bands[k]]);}}
  var out=new Array(bands.length);
  for(var k=0;k<bands.length;k++){out[k]=(bands[k]==="SCL"&&pools[bands[k]].length)?pools[bands[k]][0]:median(pools[bands[k]]);}
  return out;
}
"""

# ── #REQ1 / PSDB_COMPOSITE=ivar ──────────────────────────────────────────────
# SH evalscript variant that emits per-pixel pSDB statistics (in addition to
# the 12 standard bands) so that the Python side can build the inverse-variance
# weighted composite.  Output order: 12 standard bands + pSDB_BG_mean (×10000)
# + pSDB_BR_mean (×10000) + pSDB_BG_std (×10000) + pSDB_BR_std (×10000) +
# n_valid (uint16).  Total 17 output bands — all UINT16.
_S2_PSDB_IVAR_EVALSCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"],units:"DN",mosaicking:"ORBIT"}],output:{bands:17,sampleType:"UINT16"}};}
function isValid(s){var c=s.SCL;return c!==1&&c!==3&&c!==8&&c!==9&&c!==10&&c!==11;}
function median(a){if(!a.length)return 0;a.sort(function(x,y){return x-y});var m=Math.floor(a.length/2);return a.length%2===0?(a[m-1]+a[m])/2:a[m];}
function clamp(v,lo,hi){return v<lo?lo:(v>hi?hi:v);}
function evaluatePixel(samples){
  var bands=["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"];
  var pools={};for(var k=0;k<bands.length;k++)pools[bands[k]]=[];
  var psBG=[],psBR=[];
  for(var i=0;i<samples.length;i++){
    if(isValid(samples[i])){
      for(var k=0;k<bands.length;k++)pools[bands[k]].push(samples[i][bands[k]]);
      var b2=samples[i].B02+1e-3,b3=samples[i].B03+1e-3,b4=samples[i].B04+1e-3;
      if(b2>0&&b3>0&&b4>0){
        var lbg=Math.log(1000*b2/10000+1e-6)/Math.log(1000*b3/10000+1e-6);
        var lbr=Math.log(1000*b2/10000+1e-6)/Math.log(1000*b4/10000+1e-6);
        psBG.push(lbg);psBR.push(lbr);
      }
    }
  }
  if(!pools.B02.length){for(var j=0;j<samples.length;j++){for(var k=0;k<bands.length;k++)pools[bands[k]].push(samples[j][bands[k]]);}}
  var out=new Array(17);
  for(var k=0;k<bands.length;k++){out[k]=(bands[k]==="SCL"&&pools[bands[k]].length)?pools[bands[k]][0]:median(pools[bands[k]]);}
  var n=psBG.length;
  var bgMean=0,brMean=0,bgStd=0,brStd=0;
  if(n>0){
    for(var i=0;i<n;i++){bgMean+=psBG[i];brMean+=psBR[i];}
    bgMean/=n;brMean/=n;
    if(n>1){for(var i=0;i<n;i++){bgStd+=(psBG[i]-bgMean)*(psBG[i]-bgMean);brStd+=(psBR[i]-brMean)*(psBR[i]-brMean);}bgStd=Math.sqrt(bgStd/(n-1));brStd=Math.sqrt(brStd/(n-1));}
  }
  // scale to uint16: offset 2.0, scale factor 5000 → range [0,4] maps to [0,20000]
  out[12]=clamp(Math.round((bgMean+2.0)*5000),0,65535);
  out[13]=clamp(Math.round((brMean+2.0)*5000),0,65535);
  out[14]=clamp(Math.round(bgStd*5000),0,65535);
  out[15]=clamp(Math.round(brStd*5000),0,65535);
  out[16]=clamp(n,0,65535);
  return out;
}
"""


def _ivar_psdb_composite_gee(bbox: List[float], start_date: str, end_date: str,
                               cloud: int = 30) -> Dict:
    """GEE-path per-pixel inverse-variance pSDB composite (#REQ1).

    Maps pSDB_BG and pSDB_BR per cloud-filtered image, then reduces with
    mean + stdDev across the collection.  Returns a dict with all standard
    S2 band keys PLUS ``psdb_BG_mean``, ``psdb_BG_std``, ``psdb_BR_mean``,
    ``psdb_BR_std``, ``psdb_n``, ``psdb_BG_ivar``, ``psdb_BR_ivar``.
    """
    if not _init_gee():
        raise RuntimeError("GEE not available")
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)
    EPS = 1e-6
    SCALE = 1000.0

    def _add_psdb(img):
        b2 = img.select("B2").toFloat().divide(10000.0).add(EPS)
        b3 = img.select("B3").toFloat().divide(10000.0).add(EPS)
        b4 = img.select("B4").toFloat().divide(10000.0).add(EPS)
        # Stumpf 2003 log-ratio: ln(scale*Rrs_blue) / ln(scale*Rrs_green)
        lb = b2.multiply(SCALE).log()
        lg = b3.multiply(SCALE).log()
        lr = b4.multiply(SCALE).log()
        psdb_BG = lb.divide(lg).rename("psdb_BG")
        psdb_BR = lb.divide(lr).rename("psdb_BR")
        return img.addBands([psdb_BG, psdb_BR])

    # SCL water-only mask (class 6 = water)
    def _water_mask(img):
        scl = img.select("SCL")
        water_px = scl.eq(6)
        # also exclude cloud/shadow/glint classes: 1,3,8,9,10,11
        valid = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8))
                 .And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
        return img.updateMask(valid)

    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud))
            .map(_water_mask)
            .map(_add_psdb))

    n_imgs = coll.size().getInfo()
    L.info(f"PSDB_COMPOSITE(GEE): {n_imgs} scenes → per-pixel IVW pSDB")
    if n_imgs == 0:
        raise RuntimeError("PSDB_COMPOSITE GEE: 0 scenes in window")

    # Standard band composite (robust: use trimmed mean p10-p90 to match IVW spirit)
    # Fallback: median where percentile reducer unavailable
    std_coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))
    img_median = std_coll.select(S2_BANDS).median().clip(region)

    # pSDB stats over the per-scene values
    psdb_mean = coll.select(["psdb_BG", "psdb_BR"]).mean().clip(region)
    psdb_std  = coll.select(["psdb_BG", "psdb_BR"]).reduce(
        ee.Reducer.stdDev()).clip(region)
    # count of valid scenes
    psdb_n    = coll.select(["psdb_BG"]).count().clip(region)

    # Merge into one composite image for a single download call
    composite = (img_median
                 .addBands(psdb_mean.rename(["psdb_BG_mean", "psdb_BR_mean"]))
                 .addBands(psdb_std.rename(["psdb_BG_std", "psdb_BR_std"]))
                 .addBands(psdb_n.rename(["psdb_n"])))

    download_bands = S2_BANDS + ["psdb_BG_mean", "psdb_BR_mean",
                                  "psdb_BG_std", "psdb_BR_std", "psdb_n"]
    bundle = None
    last_err: Optional[Exception] = None
    for scale in (10, 12, 15, 20, 25, 30):
        try:
            bundle = _gee_download_geotiff(composite, bbox, scale=scale,
                                            band_order=download_bands,
                                            tag="s2psdb_ivar")
            if scale != 10:
                L.info(f"PSDB_COMPOSITE GEE: succeeded at fallback scale={scale} m")
            break
        except Exception as ex:
            last_err = ex
            msg = str(ex)
            if "Total request size" in msg or "thumbnails" in msg or "exceed" in msg.lower():
                L.warning(f"PSDB_COMPOSITE GEE @ {scale} m hit cap, retrying coarser")
                continue
            raise
    if bundle is None:
        raise last_err if last_err else RuntimeError("PSDB_COMPOSITE GEE fetch failed")

    bands = bundle["bands"]
    out: Dict[str, np.ndarray] = {}
    for b in S2_BANDS:
        if b == "SCL":
            out[b] = bands[b].astype(np.uint8)
        else:
            out[b] = bands[b].astype(np.float32)
    # pSDB composite bands
    for k in ("psdb_BG_mean", "psdb_BR_mean", "psdb_BG_std", "psdb_BR_std"):
        out[k] = bands[k].astype(np.float32)
    out["psdb_n"] = bands["psdb_n"].astype(np.float32)

    # Compute per-pixel inverse-variance weights
    eps_var = 1e-4
    var_BG = np.square(out["psdb_BG_std"]) + eps_var
    var_BR = np.square(out["psdb_BR_std"]) + eps_var
    out["psdb_BG_ivar"] = (1.0 / var_BG).astype(np.float32)
    out["psdb_BR_ivar"] = (1.0 / var_BR).astype(np.float32)
    n_med = float(np.nanmedian(out["psdb_n"]))
    L.info(f"PSDB_COMPOSITE(GEE): n_valid median={n_med:.1f}  "
           f"σ_BG median={float(np.nanmedian(out['psdb_BG_std'])):.4f}  "
           f"σ_BR median={float(np.nanmedian(out['psdb_BR_std'])):.4f}")
    out["width"] = bundle["width"]; out["height"] = bundle["height"]
    out["resolution_m"] = bundle["resolution_m"]
    out["window"] = {"start": start_date, "end": end_date}
    return out


def _ivar_psdb_composite_sh(bbox: List[float], start_date: str, end_date: str,
                              res_m: int = 10, cloud: int = 30) -> Dict:
    """SH-path per-pixel inverse-variance pSDB composite (#REQ1).

    Uses ``_S2_PSDB_IVAR_EVALSCRIPT`` which emits 17 bands: 12 standard S2 +
    4 pSDB statistics + 1 n_valid count.  Returns the same extended dict as
    ``_ivar_psdb_composite_gee``.
    """
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
                           "to":   f"{end_date}T23:59:59Z"},
                           "maxCloudCoverage": cloud}}],
        },
        "output": {"width": wp, "height": hp,
                    "responses": [{"identifier": "default",
                                    "format": {"type": "image/tiff"}}]},
        "evalscript": _S2_PSDB_IVAR_EVALSCRIPT,
    }
    L.info(f"PSDB_COMPOSITE SH: {wp}×{hp} @ {res_m} m  ({start_date} → {end_date})")
    r = None
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
            L.warning(f"PSDB_COMPOSITE SH rate-limited, retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 401 and attempt < 3:
            tok = _sh_token()
            continue
        r.raise_for_status()
    if r is None or not r.ok:
        raise RuntimeError(f"PSDB_COMPOSITE SH failed: HTTP {getattr(r,'status_code','?')}")

    import rasterio
    from rasterio.io import MemoryFile
    with MemoryFile(r.content) as mf:
        with mf.open() as src:
            arr = src.read()  # (17, H, W) uint16

    if arr.shape[0] < 17:
        raise RuntimeError(f"PSDB_COMPOSITE SH: expected 17 bands, got {arr.shape[0]}")

    out: Dict[str, np.ndarray] = {}
    for i, sh_name in enumerate(_SH_BAND_NAMES):
        internal = _SH_TO_INTERNAL[sh_name]
        if internal == "SCL":
            out[internal] = arr[i].astype(np.uint8)
        else:
            out[internal] = arr[i].astype(np.float32)

    # Decode pSDB statistics from UINT16 (scale factor 5000, offset -2.0)
    PSCALE = 5000.0; POFF = 2.0
    out["psdb_BG_mean"] = (arr[12].astype(np.float32) / PSCALE - POFF)
    out["psdb_BR_mean"] = (arr[13].astype(np.float32) / PSCALE - POFF)
    out["psdb_BG_std"]  = (arr[14].astype(np.float32) / PSCALE)
    out["psdb_BR_std"]  = (arr[15].astype(np.float32) / PSCALE)
    out["psdb_n"]       = arr[16].astype(np.float32)

    eps_var = 1e-4
    var_BG = np.square(out["psdb_BG_std"]) + eps_var
    var_BR = np.square(out["psdb_BR_std"]) + eps_var
    out["psdb_BG_ivar"] = (1.0 / var_BG).astype(np.float32)
    out["psdb_BR_ivar"] = (1.0 / var_BR).astype(np.float32)
    n_med = float(np.nanmedian(out["psdb_n"]))
    L.info(f"PSDB_COMPOSITE(SH): n_valid median={n_med:.1f}  "
           f"σ_BG median={float(np.nanmedian(out['psdb_BG_std'])):.4f}  "
           f"σ_BR median={float(np.nanmedian(out['psdb_BR_std'])):.4f}")
    out["width"] = wp; out["height"] = hp
    out["resolution_m"] = res_m
    out["window"] = {"start": start_date, "end": end_date}
    return out


def fetch_s2_full_sh(bbox: List[float], start_date: str, end_date: str,
                      res_m: int = 10, cloud: int = 30) -> Dict:
    """Median composite of all 12 S2 L2A bands + SCL via Sentinel-Hub (CDSE).

    Returns the same dict shape as :func:`fetch_s2_full_gee` so callers can
    use either backend interchangeably (keys ``B1..B12``, ``B8A``, ``SCL``,
    ``width``, ``height``, ``resolution_m``, ``window``).
    """
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
                           "to":   f"{end_date}T23:59:59Z"},
                           "maxCloudCoverage": cloud}}],
        },
        "output": {"width": wp, "height": hp,
                    "responses": [{"identifier": "default",
                                    "format": {"type": "image/tiff"}}]},
        "evalscript": _S2_FULL_EVALSCRIPT,
    }
    L.info(f"S2 full SH: {wp}×{hp} @ {res_m} m  ({start_date} → {end_date}, cloud≤{cloud}%)")
    r = None
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
            L.warning(f"S2 full SH rate-limited, retry in {wait}s ({attempt+1}/4)")
            time.sleep(wait)
            continue
        if r.status_code == 401 and attempt < 3:
            tok = _sh_token()
            continue
        r.raise_for_status()
    if r is None or not r.ok:
        raise RuntimeError(f"S2 full SH failed: HTTP {getattr(r,'status_code','?')}")

    import rasterio
    from rasterio.io import MemoryFile
    with MemoryFile(r.content) as mf:
        with mf.open() as src:
            arr = src.read()  # (12, H, W) uint16
    out: Dict[str, np.ndarray] = {}
    for i, sh_name in enumerate(_SH_BAND_NAMES):
        internal = _SH_TO_INTERNAL[sh_name]
        if internal == "SCL":
            out[internal] = arr[i].astype(np.uint8)
        else:
            out[internal] = arr[i].astype(np.float32)
    out["width"] = wp
    out["height"] = hp
    out["resolution_m"] = res_m
    out["window"] = {"start": start_date, "end": end_date}
    return out


def fetch_s2_full(bbox: List[float], start_date: str, end_date: str,
                  cloud: int = 30, prefer: str = "auto") -> Tuple[Dict, str]:
    """Fetch the 12-band S2 composite, trying both backends.

    ``prefer``: ``"sh"`` to try Sentinel-Hub first, ``"gee"`` to try Earth
    Engine first, ``"auto"`` to pick whichever is configured (SH first when
    ``SH_CLIENT_ID`` is set, else GEE).  On failure of the primary, the other
    backend is tried.  Returns ``(bundle, source_label)``.

    #REQ1 / PSDB_COMPOSITE=ivar: when this env var is set to ``ivar``, the
    function dispatches to ``_ivar_psdb_composite_gee`` / ``_ivar_psdb_composite_sh``
    which return the standard 12-band composite PLUS per-pixel pSDB mean, std,
    and inverse-variance fields.  The returned bundle is otherwise identical so
    the rest of the pipeline is unchanged.
    """
    psdb_mode = os.environ.get("PSDB_COMPOSITE", "").lower()
    have_sh = bool(os.getenv("SH_CLIENT_ID") and os.getenv("SH_CLIENT_SECRET"))
    if prefer == "auto":
        prefer = "sh" if have_sh else "gee"
    order = ["sh", "gee"] if prefer == "sh" else ["gee", "sh"]

    if psdb_mode == "ivar":
        # Dispatch to the IVW pSDB composite fetchers
        last_err = None
        for backend in order:
            try:
                if backend == "sh":
                    if not have_sh:
                        raise RuntimeError("SH_CLIENT_ID/SH_CLIENT_SECRET not set")
                    bundle = _ivar_psdb_composite_sh(bbox, start_date, end_date, cloud=cloud)
                    return bundle, f"SH PSDB-ivar @ {bundle['resolution_m']} m"
                else:
                    bundle = _ivar_psdb_composite_gee(bbox, start_date, end_date, cloud=cloud)
                    return bundle, f"GEE PSDB-ivar @ {bundle.get('resolution_m', 10)} m"
            except Exception as ex:
                L.warning(f"PSDB_COMPOSITE ivar via {backend.upper()} failed: {ex}")
                last_err = ex
        raise RuntimeError(f"PSDB_COMPOSITE ivar fetch failed on both backends: {last_err}")

    last_err: Optional[Exception] = None
    for backend in order:
        try:
            if backend == "sh":
                if not have_sh:
                    raise RuntimeError("SH_CLIENT_ID/SH_CLIENT_SECRET not set")
                bundle = fetch_s2_full_sh(bbox, start_date, end_date, cloud=cloud)
                return bundle, f"SH 12-band @ {bundle['resolution_m']} m"
            else:
                bundle = fetch_s2_full_gee(bbox, start_date, end_date, cloud=cloud)
                return bundle, f"GEE 12-band @ {bundle.get('resolution_m', 10)} m"
        except Exception as ex:
            L.warning(f"S2 full via {backend.upper()} failed: {ex}")
            last_err = ex
    raise RuntimeError(
        "S2 full fetch failed on both Sentinel-Hub and GEE: "
        f"{last_err}"
    )


# ════════════════════════════════════════════════════════════════════════
# 2. Feature engineering
# ════════════════════════════════════════════════════════════════════════
def build_features_cbr(rgb_vhr: np.ndarray,
                        s2_resampled: Optional[Dict[str, np.ndarray]],
                        water: np.ndarray,
                        era5: Optional[Dict[str, np.ndarray]] = None,
                        ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Return ``(feats[H,W,F], band_names, ndwi)``.

    Falls back to a VHR-only feature stack (3 RGB + 6 texture + 2 Stumpf
    proxies + 1 NDWI proxy = 12 channels) when ``s2_resampled`` is None.
    """
    eps = 1e-4
    r_v = rgb_vhr[..., 0].astype(np.float32) / 100.0
    g_v = rgb_vhr[..., 1].astype(np.float32) / 100.0
    b_v = rgb_vhr[..., 2].astype(np.float32) / 100.0

    def _dark(a: np.ndarray) -> np.ndarray:
        d = float(np.percentile(a[water], 1.0)) if water.any() else float(np.percentile(a, 5.0))
        return np.maximum(a - d + 0.01, 0.01)

    r_v_c, g_v_c, b_v_c = _dark(r_v), _dark(g_v), _dark(b_v)

    # 5×5 VHR texture (cheap, always available)
    try:
        from scipy.ndimage import uniform_filter
        def _ms(a, k=5):
            m = uniform_filter(a, size=k, mode="reflect")
            m2 = uniform_filter(a * a, size=k, mode="reflect")
            v = np.clip(m2 - m * m, 0.0, None)
            return m.astype(np.float32), np.sqrt(v).astype(np.float32)
        rm, rs = _ms(r_v_c); gm, gs = _ms(g_v_c); bm, bs = _ms(b_v_c)
    except Exception:
        rm = rs = gm = gs = bm = bs = np.zeros_like(r_v_c)

    if s2_resampled is None:
        # VHR-only feature stack
        ndwi_proxy = ((b_v_c - r_v_c) / (b_v_c + r_v_c + 1e-6)).astype(np.float32)
        stumpf_BG_v = (np.log(1000.0 * b_v_c + 1e-6) / np.log(1000.0 * g_v_c + 1e-6)).astype(np.float32)
        stumpf_GR_v = (np.log(1000.0 * g_v_c + 1e-6) / np.log(1000.0 * (r_v_c + eps) + 1e-6)).astype(np.float32)
        layers = [r_v_c, g_v_c, b_v_c,
                   stumpf_BG_v, stumpf_GR_v, ndwi_proxy,
                   rm, rs, gm, gs, bm, bs]
        names = [
            "vhr_R", "vhr_G", "vhr_B",
            "stumpf_BG", "stumpf_GR", "ndwi",
            "vhr_R_mean5", "vhr_R_std5",
            "vhr_G_mean5", "vhr_G_std5",
            "vhr_B_mean5", "vhr_B_std5",
        ]
        # Append ERA-5 / SST channels if provided
        if era5:
            for k in ("u10", "v10", "wind_speed", "t2m", "sst"):
                if k in era5 and era5[k] is not None:
                    layers.append(era5[k].astype(np.float32))
                    names.append(f"era5_{k}")
        feats = np.stack(layers, axis=-1).astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats, names, ndwi_proxy

    # Sentinel-2 deglinted
    s2c: Dict[str, np.ndarray] = {}
    for b in S2_BANDS:
        if b == "SCL":
            continue
        a = s2_resampled[b].astype(np.float32) / 10000.0
        s2c[b] = _dark(np.clip(a, eps, None))

    # ── Hedley 2005 NIR sun-glint correction (env HEDLEY_DEGLINT=1) ────────
    # Over calm Gulf water specular sun-glint + bright carbonate sand add a
    # depth-independent additive radiance to the visible bands; the
    # Beer-Lambert bottom term is largest at z->0, so this additive glint is
    # read by the regressor as "even brighter bottom" and the shallowest
    # pixels collapse to the deepest expressible value (observed 0-2 m floor
    # at 5.3-6.2 m).  Hedley regresses each visible band on a NIR band over a
    # homogeneous deep-water sample and subtracts the glint component:
    #   R_corr = R_vis - slope*(R_NIR - min_NIR).
    # Ref: Hedley, Harborne & Mumby 2005 IJRS 26(10):2107; Lyzenga et al. 2006.
    import os as _os
    if _os.environ.get("HEDLEY_DEGLINT", "0") == "1":
        try:
            nir_b = s2c["B8"]
            # Homogeneous deep-water sample: water pixels in the NIR-darkest
            # 40th percentile (open water, away from sand/foam).
            if water.any():
                nir_thr = float(np.percentile(nir_b[water], 40.0))
                deep_sel = water & (nir_b <= nir_thr)
            else:
                deep_sel = nir_b <= float(np.percentile(nir_b, 40.0))
            n_min = float(nir_b[deep_sel].min()) if deep_sel.any() else float(nir_b.min())
            x = nir_b[deep_sel].ravel()
            x_c = x - x.mean()
            denom = float((x_c * x_c).sum()) + 1e-9
            for vb in ("B1", "B2", "B3", "B4"):  # coastal/blue/green/red
                y = s2c[vb][deep_sel].ravel()
                slope = float((x_c * (y - y.mean())).sum() / denom)
                slope = max(slope, 0.0)  # glint is additive, slope >= 0
                s2c[vb] = np.maximum(s2c[vb] - slope * (nir_b - n_min), eps)
        except Exception as _ex:  # pragma: no cover
            L.warning(f"hedley_deglint skipped: {_ex}")

    coastal = s2c["B1"]; blue = s2c["B2"]; green = s2c["B3"]; red = s2c["B4"]
    re1 = s2c["B5"]; re2 = s2c["B6"]; re3 = s2c["B7"]
    nir = s2c["B8"]; nir_n = s2c["B8A"]; swir1 = s2c["B11"]; swir2 = s2c["B12"]

    ndwi = (s2_resampled["B3"].astype(np.float32) - s2_resampled["B8"].astype(np.float32)) / \
           (s2_resampled["B3"].astype(np.float32) + s2_resampled["B8"].astype(np.float32) + 1e-6)
    ndti = (red - green) / (red + green + 1e-6)
    ndci = (re1 - red) / (re1 + red + 1e-6)

    def _lr(a, b_, scale=1000.0):
        return (np.log(scale * a + 1e-6) / np.log(scale * b_ + 1e-6)).astype(np.float32)
    lr_BG = _lr(blue, green); lr_CG = _lr(coastal, green); lr_GR = _lr(green, red)
    lr_BR = _lr(blue, red);   lr_BC = _lr(blue, coastal);  lr_RG = _lr(red, green)

    # ── #REQ1 / PSDB_COMPOSITE=ivar ─────────────────────────────────────────
    # When the pSDB IVW composite was fetched (flag PSDB_COMPOSITE=ivar), the
    # s2_resampled dict carries per-pixel pSDB statistics produced in pSDB space
    # (Stumpf 2003).  Replace the single-composite log-ratios lr_BG / lr_BR with
    # the IVW mean (lower variance = higher trust), and add σ_psdb + w_ivar as
    # extra input channels.  The six standard ratios are retained alongside so
    # no downstream code breaks.
    psdb_ivar_mode = os.environ.get("PSDB_COMPOSITE", "").lower() == "ivar"
    psdb_BG_ivar_ch: Optional[np.ndarray] = None
    psdb_BR_ivar_ch: Optional[np.ndarray] = None
    psdb_BG_sigma: Optional[np.ndarray] = None
    psdb_BR_sigma: Optional[np.ndarray] = None
    if psdb_ivar_mode and "psdb_BG_mean" in s2_resampled:
        # Override the single-composite lr_BG / lr_BR with the IVW composite
        H_f, W_f = lr_BG.shape[:2]
        def _rsz(arr):
            from PIL import Image as _PIL
            if arr.shape[0] == H_f and arr.shape[1] == W_f:
                return arr.astype(np.float32)
            im = _PIL.fromarray(arr)
            return np.array(im.resize((W_f, H_f), _PIL.BILINEAR), dtype=np.float32)
        bg_mean = _rsz(s2_resampled["psdb_BG_mean"])
        br_mean = _rsz(s2_resampled["psdb_BR_mean"])
        bg_std  = _rsz(s2_resampled["psdb_BG_std"])
        br_std  = _rsz(s2_resampled["psdb_BR_std"])
        bg_ivar = _rsz(s2_resampled["psdb_BG_ivar"])
        br_ivar = _rsz(s2_resampled["psdb_BR_ivar"])
        # Replace the per-composite single log-ratio with the IVW mean
        lr_BG = np.nan_to_num(bg_mean, nan=float(np.nanmedian(lr_BG)))
        lr_BR = np.nan_to_num(br_mean, nan=float(np.nanmedian(lr_BR)))
        # Normalise ivar weights (clip to [0,1] for stability)
        bg_ivar_norm = np.clip(bg_ivar / (np.nanpercentile(bg_ivar, 99) + 1e-6), 0, 1)
        br_ivar_norm = np.clip(br_ivar / (np.nanpercentile(br_ivar, 99) + 1e-6), 0, 1)
        psdb_BG_ivar_ch = bg_ivar_norm.astype(np.float32)
        psdb_BR_ivar_ch = br_ivar_norm.astype(np.float32)
        psdb_BG_sigma   = bg_std.astype(np.float32)
        psdb_BR_sigma   = br_std.astype(np.float32)
        L.info(f"PSDB_COMPOSITE=ivar: lr_BG overridden with IVW composite mean  "
               f"(σ_BG median={float(np.nanmedian(bg_std)):.4f})")

    # Wave-texture proxy from the S2 NIR band: water surface roughness on
    # 10 m S2 pixels modulates B8 reflectance — local std captures swell.
    try:
        from scipy.ndimage import uniform_filter as _uf
        nir_arr = s2_resampled["B8"].astype(np.float32)
        n_m = _uf(nir_arr, size=5, mode="reflect")
        n_v = np.clip(_uf(nir_arr * nir_arr, size=5, mode="reflect") - n_m * n_m, 0, None)
        wave_proxy = np.sqrt(n_v).astype(np.float32) / 10000.0
    except Exception:
        wave_proxy = np.zeros_like(rgb_vhr[..., 0], dtype=np.float32)

    layers = [
        r_v_c, g_v_c, b_v_c,
        coastal, blue, green, red, re1, re2, re3, nir, nir_n, swir1, swir2,
        lr_BG, lr_CG, lr_GR, lr_BR, lr_BC, lr_RG,
        ndwi, ndti, ndci,
        wave_proxy,
    ]
    names = [
        "vhr_R", "vhr_G", "vhr_B",
        "s2_coastal", "s2_blue", "s2_green", "s2_red",
        "s2_rededge1", "s2_rededge2", "s2_rededge3",
        "s2_nir", "s2_nir_narrow", "s2_swir1", "s2_swir2",
        "stumpf_BG", "stumpf_CG", "stumpf_GR",
        "stumpf_BR", "stumpf_BC", "stumpf_RG",
        "ndwi", "ndti", "ndci",
        "wave_proxy",
    ]

    # ── #REQ1 extra channels (only when ivar composite present) ─────────────
    if psdb_ivar_mode and psdb_BG_ivar_ch is not None:
        layers.extend([psdb_BG_ivar_ch, psdb_BR_ivar_ch,
                        psdb_BG_sigma, psdb_BR_sigma])
        names.extend(["psdb_BG_ivar", "psdb_BR_ivar",
                       "psdb_BG_sigma", "psdb_BR_sigma"])

    # Append ERA-5 / SST channels if provided
    if era5:
        for k in ("u10", "v10", "wind_speed", "t2m", "sst"):
            if k in era5 and era5[k] is not None:
                layers.append(era5[k].astype(np.float32))
                names.append(f"era5_{k}")

    feats = np.stack(layers, axis=-1).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, names, ndwi.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# 3. K-means clustering of water pixels (sklearn MiniBatchKMeans)
# ════════════════════════════════════════════════════════════════════════
def kmeans_water(
    feats: np.ndarray,
    water: np.ndarray,
    cluster_idx: List[int],
    k: int = 6,
    seed: int = 42,
    sample_cap: int = 200_000,
):
    """Fit MiniBatchKMeans on a sub-sample of water pixels.
    Returns (km, scaler, cluster_feats_train_subset).
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    H, W, F = feats.shape
    flat = feats.reshape(-1, F)
    wflat = water.reshape(-1)
    pool = np.where(wflat)[0]
    rng = np.random.default_rng(seed)
    if len(pool) > sample_cap:
        sel = rng.choice(pool, sample_cap, replace=False)
    else:
        sel = pool
    Xs = flat[sel][:, cluster_idx]
    scaler = StandardScaler().fit(Xs)
    Xs_n = scaler.transform(Xs)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                         batch_size=4096, max_iter=200)
    km.fit(Xs_n)
    L.info(f"K-means: K={k} clusters fit on {len(sel):,} water-pixel samples")
    return km, scaler


def soft_assign(km, scaler, X_sub: np.ndarray, top_k: int = 3,
                temperature: float = 1.0) -> np.ndarray:
    """Return (N, K) soft cluster weights via softmax over -d²/τ², top_k retained."""
    Xn = scaler.transform(X_sub)
    K = km.cluster_centers_.shape[0]
    d2 = np.zeros((Xn.shape[0], K), dtype=np.float64)
    for ki in range(K):
        diff = Xn - km.cluster_centers_[ki][None, :]
        d2[:, ki] = np.sum(diff * diff, axis=1)
    # Softmax with temperature
    z = -d2 / max(1e-6, temperature ** 2)
    z -= z.max(axis=1, keepdims=True)
    w = np.exp(z)
    if top_k < K:
        idx = np.argsort(-w, axis=1)[:, top_k:]
        for r in range(w.shape[0]):
            w[r, idx[r]] = 0.0
    w /= (w.sum(axis=1, keepdims=True) + 1e-9)
    return w.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# 4. Per-cluster regressor + soft blending
# ════════════════════════════════════════════════════════════════════════
def fit_per_cluster(
    X_train: np.ndarray, y_train: np.ndarray,
    soft_train: np.ndarray, n_clusters: int,
    seed: int = 42, min_pts: int = 25,
    max_iter: int = 800,
    learning_rate: float = 0.04,
    max_depth: int = 8,
    n_estimators: int = 3,
    extra_weight: Optional[np.ndarray] = None,
):
    """Fit per-cluster HGB; ``n_estimators`` independent seeds → averaged at predict.

    Returns ``models[k] = list[HGB]`` (one entry per seed).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    models: Dict[int, list] = {}
    for k in range(n_clusters):
        w = soft_train[:, k]
        m = w > 0.05
        if m.sum() < min_pts:
            L.info(f"  cluster {k}: {int(m.sum())} pts (< {min_pts}), using global pool")
            m = np.ones(len(w), dtype=bool)
            sw = None
        else:
            sw = w[m]
        Xk = X_train[m]
        yk = y_train[m]
        bins = np.array([0, 5, 10, 15, 20, 25 + 1e-3])
        bidx = np.digitize(yk, bins) - 1
        cnt = np.bincount(bidx, minlength=5).astype(np.float32) + 1
        bw = (cnt.mean() / cnt).clip(0.5, 4.0)
        bweight = bw[bidx]
        eff_w = bweight if sw is None else sw * bweight
        if extra_weight is not None:
            eff_w = eff_w * extra_weight[m]

        ens: list = []
        for s in range(n_estimators):
            mdl = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_depth=max_depth,
                min_samples_leaf=12,
                l2_regularization=0.10,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=40,
                random_state=seed + 1000 * s,
            )
            try:
                mdl.fit(Xk, yk, sample_weight=eff_w)
            except Exception as ex:
                L.warning(f"  cluster {k} seed {s}: fit failed ({ex}); fallback")
                mdl.fit(Xk, yk)
            ens.append(mdl)
        models[k] = ens
        n_iter_avg = int(np.mean([m_.n_iter_ for m_ in ens]))
        L.info(f"  cluster {k}: {len(ens)}-seed ensemble, n={int(m.sum()):>5}, "
               f"avg iters={n_iter_avg}, depth {yk.min():.1f}–{yk.max():.1f} m")
    return models


def predict_blend(models: Dict[int, list], soft_w: np.ndarray, X: np.ndarray
                  ) -> np.ndarray:
    """Weighted prediction = Σ_k w_k(x) · mean_seed model_{k,s}(x)."""
    out = np.zeros(X.shape[0], dtype=np.float32)
    norm = np.zeros(X.shape[0], dtype=np.float32)
    for k, ens in models.items():
        wk = soft_w[:, k]
        if wk.sum() == 0:
            continue
        # Average across ensemble seeds
        if isinstance(ens, list):
            preds = np.mean([m_.predict(X) for m_ in ens], axis=0)
        else:
            preds = ens.predict(X)
        out += wk * preds.astype(np.float32)
        norm += wk
    valid = norm > 1e-6
    out[valid] = out[valid] / norm[valid]
    out[~valid] = 0.0
    return np.clip(out, 0.0, MAX_DEPTH_M)


# ════════════════════════════════════════════════════════════════════════
# 5. Save (depth, scatter, S-44 bands, cluster map)
# ════════════════════════════════════════════════════════════════════════
def _draw_cluster_map(labels_2d: np.ndarray, water: np.ndarray,
                       k: int, out_path: Path):
    palette = np.array([
        [180,  60,  60], [ 60, 180,  60], [ 60, 100, 220],
        [220, 180,  60], [180,  80, 200], [ 60, 200, 200],
        [240, 130, 100], [100, 240, 200],
    ], dtype=np.uint8)
    H, W = labels_2d.shape
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    for ki in range(k):
        m = (labels_2d == ki) & water
        img[m] = palette[ki % len(palette)]
    Image.fromarray(img).save(out_path, optimize=True)


def save_artefacts(site_key: str, bbox: List[float],
                    rgb_image: np.ndarray, water: np.ndarray,
                    osm_land: np.ndarray,
                    cluster_map: np.ndarray, k_clusters: int,
                    result: Dict, out_dir: Path = RESULTS_DIR) -> Dict[str, str]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    Image.fromarray(rgb_image).save(out_dir / f"{site_key}_vhr_image.png", optimize=True)
    paths["vhr_image"] = str(out_dir / f"{site_key}_vhr_image.png")

    # Water mask: red over land
    overlay_water = rgb_image.copy()
    nl = ~water
    if nl.any():
        red_tint = np.array([200, 80, 80], dtype=np.float32)
        overlay_water[nl] = (overlay_water[nl].astype(np.float32) * 0.35
                              + red_tint * 0.65).astype(np.uint8)
    Image.fromarray(overlay_water).save(out_dir / f"{site_key}_water_mask.png", optimize=True)
    paths["water_mask"] = str(out_dir / f"{site_key}_water_mask.png")

    # OSM land overlay
    if osm_land is not None and osm_land.shape == rgb_image.shape[:2]:
        osm_overlay = rgb_image.copy()
        if osm_land.any():
            yellow = np.array([240, 200, 40], dtype=np.float32)
            osm_overlay[osm_land] = (osm_overlay[osm_land].astype(np.float32) * 0.55
                                      + yellow * 0.45).astype(np.uint8)
        Image.fromarray(osm_overlay).save(out_dir / f"{site_key}_osm_land.png", optimize=True)
        paths["osm_land"] = str(out_dir / f"{site_key}_osm_land.png")

    # Cluster map
    cluster_path = out_dir / f"{site_key}_clusters.png"
    _draw_cluster_map(cluster_map, water, k_clusters, cluster_path)
    paths["cluster_map"] = str(cluster_path)

    # Depth viz
    depth_rgb = _depth_to_rgb(result["depth"])
    Image.fromarray(depth_rgb).save(out_dir / f"{site_key}_depth.png", optimize=True)
    paths["depth_png"] = str(out_dir / f"{site_key}_depth.png")
    overlay = (0.45 * rgb_image.astype(np.float32) + 0.55 * depth_rgb.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / f"{site_key}_overlay.png", optimize=True)
    paths["overlay_png"] = str(out_dir / f"{site_key}_overlay.png")

    # GeoTIFF
    try:
        import rasterio
        from rasterio.transform import from_bounds
        H, W = result["depth"].shape
        w, s, e, n = bbox
        transform = from_bounds(w, s, e, n, W, H)
        out = result["depth"].astype(np.float32)
        out = np.where(np.isfinite(out), out, -9999.0)
        tif_path = out_dir / f"{site_key}_depth.tif"
        with rasterio.open(
            tif_path, "w", driver="GTiff",
            width=W, height=H, count=1, dtype="float32",
            crs="EPSG:4326", transform=transform, nodata=-9999.0,
            compress="deflate",
        ) as dst:
            dst.write(out, 1)
        paths["depth_tif"] = str(tif_path)

        # iter#9.A: persist heteroscedastic σ from the U-Net head as a
        # sidecar GeoTIFF (same affine/CRS).  Closes IHO S-44 §3.4 gap
        # (per-pixel uncertainty at 95 % confidence) — feeds TPU + CATZOC
        # downstream.  σ is None when CNN refinement was skipped or the
        # cluster-only branch fired; in that case no sigma tif is written.
        sigma_grid = result.get("depth_sigma")
        if sigma_grid is not None and getattr(sigma_grid, "shape", None) == (H, W):
            sigma_out = sigma_grid.astype(np.float32)
            sigma_out = np.where(np.isfinite(sigma_out), sigma_out, -9999.0)
            sigma_path = out_dir / f"{site_key}_depth_sigma.tif"
            with rasterio.open(
                sigma_path, "w", driver="GTiff",
                width=W, height=H, count=1, dtype="float32",
                crs="EPSG:4326", transform=transform, nodata=-9999.0,
                compress="deflate",
            ) as dst:
                dst.write(sigma_out, 1)
            paths["depth_sigma_tif"] = str(sigma_path)
    except Exception as ex:
        L.warning(f"GeoTIFF skipped: {ex}")

    # Scatter
    sc_path = out_dir / f"{site_key}_scatter.png"
    _draw_scatter_mpl(
        result["y_true_test"], result["y_pred_test"], sc_path,
        title=f"{site_key} — held-out test (Clustered SDB, "
              f"{len(result['y_true_test']):,} pts)",
        metrics=result["metrics"])
    paths["scatter_png"] = str(sc_path)

    # S-44 bands
    bars_path = out_dir / f"{site_key}_s44_bands.png"
    _draw_band_bars(result["per_band"], bars_path,
                     title=f"S-44 compliance per depth band — {site_key}  (Clustered SDB)")
    paths["s44_bands_png"] = str(bars_path)

    meta = {
        "site": site_key,
        "bbox": bbox,
        "method": "Clustered SDB · 12-band S2 + VHR · K-means + per-cluster HGB · soft blend · linear bias-correction · cap 25 m",
        "vhr_resolution_m": float(result.get("resolution_m", 0)),
        "n_clusters": k_clusters,
        "max_depth_cap_m": MAX_DEPTH_M,
        "calibration": result.get("calibration", {}),
        "augmentation": result.get("augmentation"),
        "metrics_test_overall": result["metrics"],
        "metrics_test_per_band": result["per_band"],
        "feature_channels": result.get("feature_names", []),
        "elapsed_s": result.get("elapsed_s", 0),
        # Raw test arrays so callers can compute custom-band stats
        # (e.g. 2 m bands for the per-band-balanced sampler).
        "y_true_test": result.get("y_true_test", []).tolist()
            if hasattr(result.get("y_true_test", []), "tolist") else [],
        "y_pred_test": result.get("y_pred_test", []).tolist()
            if hasattr(result.get("y_pred_test", []), "tolist") else [],
    }
    metrics_path = out_dir / f"{site_key}_metrics.json"
    metrics_path.write_text(json.dumps(meta, indent=2))
    paths["metrics_json"] = str(metrics_path)

    L.info("Artefacts saved:")
    for k, v in paths.items():
        L.info(f"  {k:14s} → {v}")
    return paths


# ════════════════════════════════════════════════════════════════════════
# 6. End-to-end
# ════════════════════════════════════════════════════════════════════════
# Bump this when the feature stack or pipeline contract changes — it
# invalidates every prior on-disk model bundle.
CBR_PIPELINE_VERSION = "v4-attunet-era5-turbidity-arch_v4-varanchor"


def _model_cache_key(site_key: str, bbox: List[float], n_clusters: int,
                      train_frac: float, n_estimators: int, max_iter: int,
                      max_depth: int, seed: int, augment: bool,
                      min_train_depth_m: float = 0.0,
                      aug_min_confidence: float = 0.5,
                      aug_high_confidence_only: bool = False,
                      use_cnn_refinement: bool = False,
                      cnn_blend_weight: float = 0.5) -> str:
    import hashlib
    payload = json.dumps({
        "v": CBR_PIPELINE_VERSION,
        "site": site_key, "bbox": [round(float(x), 6) for x in bbox],
        "K": n_clusters, "train_frac": train_frac, "ne": n_estimators,
        "max_iter": max_iter, "max_depth": max_depth, "seed": seed,
        "augment": augment, "min_train_depth_m": float(min_train_depth_m),
        "aug_min_confidence": float(aug_min_confidence),
        "aug_high_confidence_only": bool(aug_high_confidence_only),
        "use_cnn_refinement": bool(use_cnn_refinement),
        "cnn_blend_weight": float(cnn_blend_weight),
        # iter#13 (scientist req #3): the deglint / shallow-mask feature-prep
        # knobs and the U-Net loss spec change the trained model, so they MUST
        # be part of the cache key or a stale (non-deglinted, old-loss) bundle
        # gets silently reused.  Default-off knobs leave existing keys intact.
        "hedley_deglint": os.environ.get("HEDLEY_DEGLINT", "0"),
        "shallow_nir_mask": os.environ.get("SHALLOW_NIR_MASK", "0"),
        "loss": os.environ.get("LOSS", "nll"),
        "silog_w": os.environ.get("SILOG_W", "0.5"),
        "range_loss_w": os.environ.get("RANGE_LOSS_W", "0.3"),
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def save_model_bundle(path: Path, bundle: Dict) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_model_bundle(path: Path) -> Optional[Dict]:
    import joblib
    try:
        return joblib.load(path)
    except Exception:
        return None


def run_cbr_vhr(
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
    n_clusters: int = 6,
    cluster_temperature: float = 0.7,
    max_iter: int = 800,
    n_estimators: int = 3,
    learning_rate: float = 0.04,
    max_depth: int = 8,
    out_dir: Optional[Path] = None,
    augment: bool = False,
    aug_min_per_band: int = 80,
    aug_band_w: float = 2.0,
    aug_use_iboating: bool = True,
    aug_use_sliderule: bool = True,
    aug_use_gebco: bool = True,
    aug_sliderule_start: str = "2020-01-01",
    aug_sliderule_end: str = "2025-12-31",
    aug_min_confidence: float = 0.5,
    aug_high_confidence_only: bool = False,
    use_model_cache: bool = True,
    force_retrain: bool = False,
    min_train_depth_m: float = 5.0,   # 0-5 m bin is too noisy: optimise on >=5 m
    max_train_depth_m: float = 0.0,   # 0 disables; >0 caps training depth (v7)
    use_cnn_refinement: bool = True,  # train a small CNN on top of HGB and blend
    cnn_blend_weight: float = 0.5,    # final = (1-w)*HGB + w*CNN
    cnn_epochs: int = 15,
    cnn_crop: int = 128,
    cnn_batch: int = 4,
    cnn_base: int = 16,               # U-Net width — CPU-tractable
    cnn_min_labels: int = 1500,       # below this, skip CNN (override for sparse refs)
    cnn_label_patch: int = 1,         # K×K label dilation around each ref pixel
    cnn_use_cluster_features: bool = False,  # append cluster one-hot to U-Net feats
    seed: int = 42,
    # ── v5 honest-evaluation knobs ────────────────────────────────────────
    split_mode: str = "stratified",   # "stratified" (legacy) | "spatial_block"
    n_spatial_blocks: int = 25,
    # ── v8 per-cluster post-CNN calibration (channels-as-classification) ─
    # When True, after the CNN blend the model fits a separate (α_c, β_c)
    # linear calibration per K-means cluster on training pixels and applies
    # the per-cluster calibration to the final depth grid.  This breaks
    # the global mean-collapse: each spectral cluster gets its own depth
    # regime instead of a single shared calibration line.
    per_cluster_calibration: bool = False,
    save: bool = True,
    imagery_source: str = "vhr",
) -> Dict:
    """Run the clustered Very HR pipeline.

    ``imagery_source``:
      * ``"vhr"`` (default) — Mapbox VHR mosaic at ``target_res_m`` is the base
        raster; 12-band S2 (GEE) is resampled onto it.
      * ``"s2"`` — no Mapbox call; 12-band S2 at native ~10 m is the base
        raster.  Produces a coarser-resolution depth grid but works on free
        imagery only.
    """
    t0 = time.time()
    imagery_source = (imagery_source or "vhr").lower()
    if imagery_source not in {"vhr", "s2"}:
        raise ValueError(f"imagery_source must be 'vhr' or 's2', got {imagery_source!r}")

    s2_rs: Optional[Dict[str, np.ndarray]] = None
    s2_status = "unavailable"

    if imagery_source == "vhr":
        # 1. Mapbox VHR
        vhr = fetch_mapbox_vhr(bbox, target_res_m=target_res_m,
                                max_tiles_per_side=max_tiles_per_side)
        rgb = vhr["rgb"]
        H, W = rgb.shape[:2]

        # 2. S2 full 12-band — try Sentinel-Hub first, GEE second.
        # #REQ1: when PSDB_COMPOSITE=ivar, fetch_s2_full dispatches to the
        # IVW composite fetchers which return the standard 12 bands plus pSDB
        # statistics (psdb_BG_mean, psdb_BG_std, psdb_BG_ivar, etc.).  Those
        # extra keys are passed through to s2_rs so build_features_cbr can use them.
        try:
            s2_full, s2_label = fetch_s2_full(
                bbox, s2_start_date, s2_end_date, cloud=30, prefer="auto",
            )
            s2_rs = {}
            for b in S2_BANDS:
                if b == "SCL":
                    s2_rs[b] = _resample(s2_full[b], H, W, resample=Image.NEAREST).astype(np.uint8)
                else:
                    s2_rs[b] = _resample(s2_full[b], H, W)
            # Propagate any extra pSDB composite fields (present only when
            # PSDB_COMPOSITE=ivar; silently ignored otherwise).
            for _psdb_key in ("psdb_BG_mean", "psdb_BR_mean",
                              "psdb_BG_std",  "psdb_BR_std",
                              "psdb_BG_ivar", "psdb_BR_ivar",
                              "psdb_n"):
                if _psdb_key in s2_full:
                    s2_rs[_psdb_key] = _resample(s2_full[_psdb_key], H, W)
            s2_status = s2_label
        except Exception as ex:
            L.warning(f"S2 12-band fetch failed (SH+GEE): {ex} — falling back to VHR-only features")
            s2_status = f"unavailable ({ex})"
    else:
        # S2-only base raster: native ~10 m grid via SH (preferred) or GEE.
        try:
            s2_full, s2_label = fetch_s2_full(
                bbox, s2_start_date, s2_end_date, cloud=30, prefer="auto",
            )
        except Exception as ex:
            raise RuntimeError(
                f"S2-only mode requires the 12-band S2 composite but it failed "
                f"on both Sentinel-Hub and GEE: {ex}"
            )
        H = int(s2_full["height"]); W = int(s2_full["width"])
        s2_rs = {b: (s2_full[b].astype(np.uint8) if b == "SCL"
                     else s2_full[b].astype(np.float32))
                 for b in S2_BANDS}
        # Synthesise a "vhr" RGB from S2 R/G/B with a 2–98 % stretch so the
        # downstream feature builder, water mask, and visualisations work
        # without modification.
        try:
            from backend.very_hr_engine import _s2_dn_to_rgb_uint8 as _s2rgb
        except ImportError:
            from very_hr_engine import _s2_dn_to_rgb_uint8 as _s2rgb  # type: ignore
        rgb = _s2rgb({"red": s2_rs["B4"], "green": s2_rs["B3"], "blue": s2_rs["B2"]})
        res_m = s2_full.get("resolution_m", 10)
        vhr = {"rgb": rgb, "width": W, "height": H,
               "resolution_m": float(res_m), "tiles": [1, 1]}
        s2_status = f"{s2_label} (base raster)"
        L.info(f"S2-only base raster: {W}×{H}px @ {res_m} m/px ({s2_label})")

    # 3. OSM mask + composite water
    osm_land = osm_land_mask(bbox, H, W)
    s2_for_water = None
    if s2_rs is not None:
        s2_for_water = {
            "green": s2_rs["B3"], "nir": s2_rs["B8"], "scl": s2_rs["SCL"],
        }
    water, ndwi = water_mask_combined(rgb, s2_for_water, osm_land=osm_land)
    L.info(f"Water: {int(water.sum()):,}/{water.size:,} ({100*water.mean():.1f}%) "
           f"· OSM land = {100*osm_land.mean():.1f}% · S2: {s2_status}")

    # ── SMART VHR water/land mask (GMM endmember unmixing) ────────────────────
    # `rgb` here is the genuine Mapbox VHR mosaic already resampled to the ~2 m
    # output grid, so we can unmix water / built-concrete-soil-veg endmembers
    # directly on it and INTERSECT with the composite water mask (both must
    # pass) to cut piers / quays / breakwaters that the 10 m S2 boundary leaks.
    # Only runs on true VHR (res ≤ 5 m); graceful skip otherwise (flagged).
    vhr_mask_meta = {"mask_source": "composite (VHR mask not attempted)",
                     "vhr_applied": False}
    try:
        _vhr_res = float(vhr.get("resolution_m", 99)) if isinstance(vhr, dict) else 99.0
    except Exception:
        _vhr_res = 99.0
    if os.environ.get("VHR_WATER_MASK", "1") == "1" and _vhr_res <= 5.0 \
            and rgb is not None and rgb.shape[:2] == (H, W):
        try:
            try:
                from backend.vhr_water_mask import segment_water_land as _seg
            except ImportError:
                from vhr_water_mask import segment_water_land as _seg  # type: ignore
            _vw, _prob, _sinfo = _seg(rgb, prior_water=water)
            _refined = water & _vw
            _removed = int((water & ~_vw).sum())
            if int(_refined.sum()) >= 50:
                vhr_mask_meta = {
                    "mask_source": "VHR+composite", "vhr_applied": True,
                    "vhr_res_m": _vhr_res,
                    "composite_water_px": int(water.sum()),
                    "refined_water_px": int(_refined.sum()),
                    "infra_px_removed": _removed,
                    "retention_pct": round(100.0 * _refined.sum()
                                           / max(int(water.sum()), 1), 2),
                    "seg": _sinfo,
                }
                water = _refined
                L.info(f"VHR mask: removed {_removed} infra px · retention "
                       f"{vhr_mask_meta['retention_pct']}% "
                       f"({int(water.sum()):,} water)")
            else:
                vhr_mask_meta = {"mask_source": "composite (VHR refined collapsed)",
                                 "vhr_applied": False}
        except Exception as _ex:
            L.warning(f"VHR mask skipped ({_ex})")
            vhr_mask_meta = {"mask_source": "composite fallback", "vhr_applied": False,
                             "error": str(_ex)}
    else:
        vhr_mask_meta = {"mask_source": "composite (S2-only base or VHR disabled)",
                         "vhr_applied": False, "vhr_res_m": _vhr_res}

    # 2b. Stricter very-shallow water gate (env SHALLOW_NIR_MASK=1) ────────
    # Drop wet-sand / exposed carbonate flats / sun-glint-flagged pixels that
    # masquerade as shallow water: NIR-bright AND MNDWI<0 pixels are not water
    # at all (Xu 2006 MNDWI; SCL glint/cloud classes).  These bright shallow
    # pixels are precisely the ones the regressor floors to its deepest value.
    if os.environ.get("SHALLOW_NIR_MASK", "0") == "1" and s2_rs is not None:
        try:
            nir_m = s2_rs["B8"].astype(np.float32) / 10000.0
            green_m = s2_rs["B3"].astype(np.float32) / 10000.0
            swir_m = s2_rs["B11"].astype(np.float32) / 10000.0
            scl_m = s2_rs["SCL"].astype(np.uint8)
            mndwi = (green_m - swir_m) / (green_m + swir_m + 1e-6)
            # NIR-bright threshold: 90th pct of NIR over current water pixels.
            nir_bright = float(np.percentile(nir_m[water], 90.0)) if water.any() \
                else float(np.percentile(nir_m, 90.0))
            wet_sand = (nir_m > nir_bright) & (mndwi < 0.0)
            # SCL glint/cloud/bare-soil classes: 3=cloud-shadow,4=veg,5=bare,
            # 8=cloud-med,9=cloud-high,10=cirrus,11=snow.
            scl_bad = np.isin(scl_m, (3, 5, 8, 9, 10, 11))
            n_before = int(water.sum())
            water = water & ~(wet_sand | scl_bad)
            n_drop = n_before - int(water.sum())
            L.info(f"SHALLOW_NIR_MASK on: dropped {n_drop:,} non-water "
                   f"(wet-sand/glint/SCL) pixels → water now "
                   f"{100*water.mean():.1f}%")
        except Exception as _ex:
            L.warning(f"shallow_nir_mask skipped: {_ex}")

    # 3b. ERA-5 + SST via GEE (optional, free) — wind components, 2 m air
    #     temperature, SST.  Coarse but global; bilinearly upsampled to the
    #     VHR grid in fetch_era5_sst_gee + _resample.
    era5_rs: Optional[Dict[str, np.ndarray]] = None
    era5_status = "unavailable"
    try:
        try:
            from backend.very_hr_engine import fetch_era5_sst_gee as _e5
        except ImportError:
            from very_hr_engine import fetch_era5_sst_gee as _e5  # type: ignore
        e5 = _e5(bbox, s2_start_date, s2_end_date)
        era5_rs = {k: _resample(v, H, W) for k, v in e5["bands"].items()}
        era5_status = f"ERA-5 + SST  ({len(era5_rs)} bands)"
        L.info(f"Ocean state attached: {era5_status}")
    except Exception as ex:
        L.warning(f"ERA-5 / SST fetch failed: {ex}")
        era5_status = f"unavailable ({ex})"

    # 3c. Turbidity mask — pixels with high NDTI are flagged so the model
    #     does not learn from sediment plumes / sun-glint, and the user is
    #     told what fraction of the ROI was excluded.
    turbid_mask = np.zeros((H, W), dtype=bool)
    turbidity_pct = 0.0
    turbid_threshold = 0.10
    if s2_rs is not None:
        red_t = s2_rs["B4"].astype(np.float32) / 10000.0 + 1e-4
        green_t = s2_rs["B3"].astype(np.float32) / 10000.0 + 1e-4
        nir_t = s2_rs["B8"].astype(np.float32) / 10000.0 + 1e-4
        ndti_full = (red_t - green_t) / (red_t + green_t + 1e-6)
        # Adaptive threshold: turbid = NDTI above (median+0.05) of water pixels
        if water.any():
            q = float(np.median(ndti_full[water]))
            adaptive_thresh = max(turbid_threshold, q + 0.05)
        else:
            adaptive_thresh = turbid_threshold
        turbid_mask = water & (ndti_full > adaptive_thresh)
        turbidity_pct = round(100 * float(turbid_mask.sum()) / max(1, water.sum()), 2)
        L.info(f"Turbidity: NDTI threshold={adaptive_thresh:.3f} → "
               f"{int(turbid_mask.sum()):,} px = {turbidity_pct}% of water")
    else:
        # VHR proxy: red/(red+green) — very rough
        r_v = rgb[..., 0].astype(np.float32) + 1
        g_v = rgb[..., 1].astype(np.float32) + 1
        ndti_proxy = (r_v - g_v) / (r_v + g_v + 1e-6)
        adaptive_thresh = max(turbid_threshold, float(np.median(ndti_proxy[water])) + 0.05) if water.any() else turbid_threshold
        turbid_mask = water & (ndti_proxy > adaptive_thresh)
        turbidity_pct = round(100 * float(turbid_mask.sum()) / max(1, water.sum()), 2)
        L.info(f"Turbidity (VHR proxy): {turbidity_pct}% of water flagged")

    # Subtract turbid pixels from the water mask so the model neither trains
    # on them nor predicts them. Track the original mask for reporting.
    water_full = water.copy()
    water = water & ~turbid_mask

    # 4. Features
    feats, names, _ = build_features_cbr(rgb, s2_rs, water, era5=era5_rs)
    L.info(f"Features: {feats.shape[2]} channels — {names}")

    # PAPER_LOOP leakage guard — BAND_SHUFFLE=1 permutes every input channel
    # independently across pixels, destroying the feature↔label spatial pairing
    # while leaving the label grid and water mask untouched. A config that still
    # scores R²>0 under this shuffle is exploiting texture/geometry leakage, not
    # spectral signal (PAPER_LOOP_DL.md band-shuffle guard). Default OFF.
    if os.environ.get("BAND_SHUFFLE", "0") == "1":
        Hbs, Wbs, Cbs = feats.shape
        rng_bs = np.random.default_rng(int(os.environ.get("SEED", "0")) + 7919)
        flat_bs = feats.reshape(-1, Cbs).copy()
        for c in range(Cbs):
            flat_bs[:, c] = flat_bs[rng_bs.permutation(Hbs * Wbs), c]
        feats = flat_bs.reshape(Hbs, Wbs, Cbs)
        L.info(f"BAND_SHUFFLE=1: permuted all {Cbs} input channels across "
               f"{Hbs * Wbs} pixels (leakage guard; labels/water mask intact)")

    # 5. Cluster on a spectral subset.  Use S2 bands when present, otherwise
    #    fall back to VHR-derived features.
    if s2_rs is not None:
        cluster_feat_names = [
            "s2_coastal", "s2_blue", "s2_green", "s2_red",
            "s2_rededge1", "s2_rededge3", "s2_nir", "s2_swir1",
            "stumpf_BG", "stumpf_CG", "ndwi", "ndti", "ndci",
        ]
    else:
        cluster_feat_names = [
            "vhr_R", "vhr_G", "vhr_B",
            "stumpf_BG", "stumpf_GR", "ndwi",
        ]
    # Drop any names not present in this run's feature list
    cluster_feat_names = [n_ for n_ in cluster_feat_names if n_ in names]
    cluster_idx = [names.index(n) for n in cluster_feat_names]
    km, scaler = kmeans_water(feats, water, cluster_idx,
                                k=n_clusters, seed=seed)

    # Hard-cluster map for visualisation (water only)
    flat = feats.reshape(-1, feats.shape[-1])
    wflat = water.reshape(-1)
    hard = np.full(H * W, -1, dtype=np.int32)
    Xc_w = scaler.transform(flat[wflat][:, cluster_idx])
    hard[wflat] = km.predict(Xc_w)
    cluster_map = hard.reshape(H, W)

    # 6. Optional band-wise augmentation (i-Boating → GEBCO)
    src_weights = None
    aug_summary: Optional[Dict] = None
    if augment:
        try:
            from backend.very_hr_augment import augment_in_situ
        except ImportError:
            from very_hr_augment import augment_in_situ  # type: ignore
        aug = augment_in_situ(
            np.asarray(ref_lats), np.asarray(ref_lons),
            np.asarray(ref_depths, dtype=np.float64),
            bbox=bbox, band_w=aug_band_w, max_d=MAX_DEPTH_M + aug_band_w / 2,
            min_per_band=aug_min_per_band,
            use_iboating=aug_use_iboating,
            use_sliderule=aug_use_sliderule,
            use_gebco=aug_use_gebco,
            sliderule_start=aug_sliderule_start,
            sliderule_end=aug_sliderule_end,
            min_aug_confidence=aug_min_confidence,
            high_confidence_only=aug_high_confidence_only,
        )
        ref_lats = aug["lats"]; ref_lons = aug["lons"]; ref_depths = aug["depths"]
        src_weights = aug["weights"]
        aug_summary = {"counts": aug["counts"], "per_band": aug["per_band"]}

    # 7. Sample at reference points (cap labels at 25 m)
    n_input_refs = len(ref_depths)
    w_, s_, e_, n_ = bbox
    cols = ((np.asarray(ref_lons) - w_) / (e_ - w_) * W).astype(int)
    rows = ((n_ - np.asarray(ref_lats)) / (n_ - s_) * H).astype(int)
    deps = np.clip(np.asarray(ref_depths, dtype=np.float64), 0, MAX_DEPTH_M)
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & (deps > 0)
    n_in_bbox = int(valid.sum())
    rows, cols, deps = rows[valid], cols[valid], deps[valid]
    if src_weights is not None:
        src_weights = src_weights[valid]
    on_water = water[rows, cols]
    rows, cols, deps = rows[on_water], cols[on_water], deps[on_water]
    if src_weights is not None:
        src_weights = src_weights[on_water]

    L.info(f"Reference funnel: {n_input_refs} total → {n_in_bbox} in-bbox → "
           f"{len(deps)} on-water  (water mask: {int(water.sum())} px, "
           f"{100*water.mean():.1f}% of grid)")

    if len(deps) < 30:
        aug_counts = (aug_summary or {}).get("counts", {}) if augment else {}
        details = (
            f"only {len(deps)} reference on-water points  "
            f"(input refs={n_input_refs}, after-bbox-filter={n_in_bbox}, "
            f"water-mask={int(water.sum())} px = {100*water.mean():.1f}% of grid"
            + (f", augmentation={aug_counts}" if aug_counts else "")
            + ")"
        )
        # Specific hints for common modes
        if int(water.sum()) == 0:
            details += "  · ROI is entirely land per the OSM + S2 mask — pick a coastal ROI."
        elif n_input_refs == 0:
            details += ("  · No in-situ in ROI and the augmentation cascade "
                        "(SlideRule/i-Boating/GEBCO) returned nothing — "
                        "check that GROQ_API_KEY and GEE service account are "
                        "set, or load a preset (Khalifa, Mussafah).")
        elif n_in_bbox == 0:
            details += "  · References fell outside the ROI — verify the bbox covers the data."
        raise RuntimeError(details)
    L.info(f"Reference on water: {len(deps):,} pts  range {deps.min():.2f}–{deps.max():.2f} m")

    # Train / test split: depth-stratified (legacy) or spatial-block (honest).
    # Spatial-block prevents leakage from the 7×7 label patches + S2 spatial
    # autocorrelation.  Pass the raw lats/lons of the surviving on-water refs.
    keep_lats = np.asarray(ref_lats)[valid][on_water]
    keep_lons = np.asarray(ref_lons)[valid][on_water]
    train_idx, test_idx = stratified_split(
        deps, train_frac=train_frac, seed=seed,
        lats=keep_lats, lons=keep_lons,
        mode=split_mode, n_spatial_blocks=n_spatial_blocks,
    )
    L.info(f"Split [{split_mode}]: {int(train_idx.sum()):,} train "
           f"({100*train_idx.mean():.0f}%) / {int(test_idx.sum()):,} test")

    # 7. Build feature vectors at labelled pixels + soft-cluster weights
    X_all = feats[rows, cols]                                    # (N, F)
    soft_all = soft_assign(km, scaler, X_all[:, cluster_idx],
                            top_k=min(3, n_clusters),
                            temperature=cluster_temperature)
    X_train = X_all[train_idx]; y_train = deps[train_idx].astype(np.float32)
    soft_train = soft_all[train_idx]
    extra_w_train = src_weights[train_idx] if src_weights is not None else None

    # Apply min_train_depth_m filter: drop training points shallower than the
    # threshold (where the 0-5 m bin is noise-dominated by chart digitisation,
    # SlideRule refraction, and water-line ambiguity).
    if min_train_depth_m > 0:
        keep_train = y_train >= float(min_train_depth_m)
        n_drop = int((~keep_train).sum())
        if n_drop > 0:
            L.info(f"min_train_depth_m={min_train_depth_m} m → dropping "
                   f"{n_drop}/{len(y_train)} shallow training points")
            X_train = X_train[keep_train]
            y_train = y_train[keep_train]
            soft_train = soft_train[keep_train]
            if extra_w_train is not None:
                extra_w_train = extra_w_train[keep_train]
    if max_train_depth_m > 0:
        keep_train = y_train <= float(max_train_depth_m)
        n_drop = int((~keep_train).sum())
        if n_drop > 0:
            L.info(f"max_train_depth_m={max_train_depth_m} m → dropping "
                   f"{n_drop}/{len(y_train)} deep training points "
                   f"(v7: avoid extrapolation noise past Beer-Lambert limit)")
            X_train = X_train[keep_train]
            y_train = y_train[keep_train]
            soft_train = soft_train[keep_train]
            if extra_w_train is not None:
                extra_w_train = extra_w_train[keep_train]

    # 8. Per-cluster HGB ensemble — load from cache when available
    cache_key = _model_cache_key(site_key, bbox, n_clusters, train_frac,
                                   n_estimators, max_iter, max_depth, seed,
                                   augment,
                                   min_train_depth_m=min_train_depth_m,
                                   aug_min_confidence=aug_min_confidence,
                                   aug_high_confidence_only=aug_high_confidence_only,
                                   use_cnn_refinement=use_cnn_refinement,
                                   cnn_blend_weight=cnn_blend_weight)
    cache_path = MODEL_CACHE_DIR / f"cbr_{site_key}_{cache_key}.joblib"
    bundle = None
    if use_model_cache and not force_retrain and cache_path.exists():
        bundle = load_model_bundle(cache_path)
        # Reject cache when the feature schema diverges (e.g. ERA-5 added)
        cached_F = bundle.get("n_features") if bundle else None
        if bundle and cached_F is not None and cached_F != feats.shape[2]:
            L.warning(f"Cache feature mismatch ({cached_F} vs {feats.shape[2]}) "
                       "— retraining")
            bundle = None
        if bundle and bundle.get("kmeans") is not None:
            L.info(f"Model cache HIT  {cache_path.name}  — skipping training")
            km = bundle["kmeans"]; scaler = bundle["scaler"]
            models = bundle["models"]
            # Recompute soft assignments + cluster_map under the cached centroids
            soft_all = soft_assign(km, scaler, X_all[:, cluster_idx],
                                    top_k=min(3, n_clusters),
                                    temperature=cluster_temperature)
            soft_train = soft_all[train_idx]
            Xc_w = scaler.transform(flat[wflat][:, cluster_idx])
            hard = np.full(H * W, -1, dtype=np.int32)
            hard[wflat] = km.predict(Xc_w)
            cluster_map = hard.reshape(H, W)
        else:
            bundle = None
    if bundle is None:
        L.info(f"Training {n_clusters} clusters × {n_estimators} seeds  "
               f"(max_iter={max_iter}, lr={learning_rate}, depth={max_depth})")
        models = fit_per_cluster(
            X_train, y_train, soft_train, n_clusters, seed=seed,
            max_iter=max_iter, learning_rate=learning_rate,
            max_depth=max_depth, n_estimators=n_estimators,
            extra_weight=extra_w_train,
        )
        if use_model_cache:
            try:
                save_model_bundle(cache_path, {
                    "kmeans": km, "scaler": scaler, "models": models,
                    "cluster_idx": cluster_idx,
                    "feature_names": names, "n_clusters": n_clusters,
                    "n_features": int(feats.shape[2]),
                    "cluster_temperature": cluster_temperature,
                    "site_key": site_key, "bbox": bbox,
                    "pipeline_version": CBR_PIPELINE_VERSION,
                })
                L.info(f"Model bundle cached → {cache_path.name}")
            except Exception as ex:
                L.warning(f"Model cache save failed: {ex}")

    # 9. Predict at all labelled points (raw)
    yp_full_raw = predict_blend(models, soft_all, X_all)

    # 10. Linear bias correction — fit only on points used for training
    # (>= min_train_depth_m if filter active; else full training set).
    yp_train_for_cal = yp_full_raw[train_idx]
    yt_train_for_cal = deps[train_idx]
    if min_train_depth_m > 0:
        cal_keep = yt_train_for_cal >= float(min_train_depth_m)
        if cal_keep.sum() >= 5:
            yp_train_for_cal = yp_train_for_cal[cal_keep]
            yt_train_for_cal = yt_train_for_cal[cal_keep]
    alpha, beta = fit_linear_calibration(yp_train_for_cal, yt_train_for_cal)
    # Sanity: when the training pool is tiny or the line is implausible,
    # the linear calibration over-corrects and produces ridiculous biases
    # (e.g. α=-18, β=2). Fall back to identity in that regime.
    n_cal = len(yt_train_for_cal)
    # Item 2 (scientist req #2): when ATL24 supplies enough DEEP labels the
    # true fit needs β > 1.4 to un-saturate the mean-collapsed deep
    # predictions — exactly the regime the legacy guard (0.6 ≤ β ≤ 1.4)
    # rejected.  Admit a higher-slope fit (β ≤ 2.5) ONLY when there are
    # ≥ DEEP_CAL_MIN (default 200) deep (>14 m) training labels, else keep
    # the conservative guard so we never over-correct on thin data.
    import os as _os
    n_deep_cal = int((np.asarray(yt_train_for_cal) > 14.0).sum())
    deep_cal_min = int(_os.environ.get("DEEP_CAL_MIN", "200"))
    beta_hi = 2.5 if n_deep_cal >= deep_cal_min else 1.4
    plausible = (n_cal >= 100 and -3.0 <= alpha <= 3.0 and 0.6 <= beta <= beta_hi)
    L.info(f"Calibration guard: n_cal={n_cal}, n_deep(>14m)={n_deep_cal}, "
           f"β allowed ≤ {beta_hi:.1f} (fit β={beta:.3f})")
    if not plausible:
        L.warning(
            f"Calibration deemed unreliable (n={n_cal}, α={alpha:+.2f}, "
            f"β={beta:.3f}) — keeping identity (α=0, β=1)."
        )
        alpha, beta = 0.0, 1.0
    L.info(f"Calibration: y_true ≈ {alpha:+.3f} + {beta:.4f}·y_pred  "
           f"(fitted on {n_cal} train pts ≥ {min_train_depth_m} m)")
    yp_full = np.clip(alpha + beta * yp_full_raw, 0.0, MAX_DEPTH_M)

    yt_test = deps[test_idx]
    yp_test = yp_full[test_idx]

    metrics_raw = compute_metrics(yt_test, np.clip(yp_full_raw[test_idx], 0, MAX_DEPTH_M))
    metrics = compute_metrics(yt_test, yp_test)
    metrics["calibration_alpha"] = round(alpha, 4)
    metrics["calibration_beta"] = round(beta, 4)
    metrics["bias_before_calibration_m"] = metrics_raw["bias_m"]
    metrics["rmse_before_calibration_m"] = metrics_raw["rmse_m"]
    metrics["s44_1a_pct"] = round(s44_pass_pct(yt_test, yp_test, a=0.5,  b=0.013), 1)
    metrics["s44_1b_pct"] = round(s44_pass_pct(yt_test, yp_test, a=0.5,  b=0.013), 1)
    metrics["s44_order2_pct"] = round(s44_pass_pct(yt_test, yp_test, a=1.0,  b=0.023), 1)
    metrics["s44_special_pct"] = round(s44_pass_pct(yt_test, yp_test, a=0.25, b=0.0075), 1)
    metrics["n_train"] = int(len(yt_train_for_cal))
    metrics["n_test"] = int(test_idx.sum())
    metrics["train_fraction"] = float(train_frac)
    metrics["min_train_depth_m"] = float(min_train_depth_m)
    per_band = metrics_per_band(yt_test, yp_test)

    # Report a parallel set of metrics on the test points >= min_train_depth_m
    # (this is the "accuracy without the 0-5 m band" line).
    if min_train_depth_m > 0:
        deep = yt_test >= float(min_train_depth_m)
        if int(deep.sum()) >= 1:
            yt_d, yp_d = yt_test[deep], yp_test[deep]
            md = compute_metrics(yt_d, yp_d)
            md["s44_1a_pct"] = round(s44_pass_pct(yt_d, yp_d, a=0.5, b=0.013), 1)
            md["s44_1b_pct"] = round(s44_pass_pct(yt_d, yp_d, a=0.5, b=0.013), 1)
            md["s44_order2_pct"] = round(s44_pass_pct(yt_d, yp_d, a=1.0, b=0.023), 1)
            md["s44_special_pct"] = round(s44_pass_pct(yt_d, yp_d, a=0.25, b=0.0075), 1)
            md["depth_floor_m"] = float(min_train_depth_m)
            metrics["overall_excluding_shallow"] = md

    # 11. Predict the full grid (chunked, water only)
    L.info("Predicting full grid (chunked)…")
    flat_water_idx = np.where(wflat)[0]
    chunk = 150_000
    pred_water = np.empty(len(flat_water_idx), dtype=np.float32)
    for i in range(0, len(flat_water_idx), chunk):
        sl = flat_water_idx[i:i + chunk]
        Xc = flat[sl]
        sw = soft_assign(km, scaler, Xc[:, cluster_idx],
                         top_k=min(3, n_clusters),
                         temperature=cluster_temperature)
        pk = predict_blend(models, sw, Xc)
        pred_water[i:i + chunk] = pk
    pred_full = np.full(H * W, np.nan, dtype=np.float32)
    pred_full[flat_water_idx] = np.clip(alpha + beta * pred_water, 0, MAX_DEPTH_M)
    depth_grid = pred_full.reshape(H, W)
    depth_grid = np.where(water, depth_grid, np.nan)

    # ────────────────────────────────────────────────────────────────────
    # 12. CNN refinement (Cluster + CNN). The cluster pipeline above
    #     produces a strong per-pixel prior; a small dilated FCN is now
    #     trained on the same labels (>= min_train_depth_m if set) and
    #     blended into the final depth map.  Cached alongside the HGB.
    # ────────────────────────────────────────────────────────────────────
    cnn_grid: Optional[np.ndarray] = None
    cnn_sigma: Optional[np.ndarray] = None
    if use_cnn_refinement:
        try:
            try:
                from backend.unet_sdb import train_unet_sdb, predict_grid_tta
            except ImportError:
                from unet_sdb import train_unet_sdb, predict_grid_tta  # type: ignore
            # Build a (H, W) label grid + mask using the training split
            label_grid_cnn = np.zeros((H, W), dtype=np.float32)
            label_mask_cnn = np.zeros((H, W), dtype=bool)
            tr_rows = rows[train_idx]
            tr_cols = cols[train_idx]
            tr_deps = deps[train_idx]
            if min_train_depth_m > 0:
                keep = tr_deps >= float(min_train_depth_m)
                tr_rows = tr_rows[keep]; tr_cols = tr_cols[keep]; tr_deps = tr_deps[keep]
            if max_train_depth_m > 0:
                keep = tr_deps <= float(max_train_depth_m)
                tr_rows = tr_rows[keep]; tr_cols = tr_cols[keep]; tr_deps = tr_deps[keep]
            label_grid_cnn[tr_rows, tr_cols] = tr_deps.astype(np.float32)
            label_mask_cnn[tr_rows, tr_cols] = True
            # Optional K×K dilation: a chart sounding actually represents the
            # depth in a small neighbourhood (≈5-10 m), so painting a small
            # patch around each ref pixel is physically reasonable AND lifts
            # the label count enough for the Attention U-Net to train when
            # the user is forced to rely on i-Boating-only references.
            k_patch = max(1, int(cnn_label_patch))
            if k_patch > 1:
                rad = k_patch // 2
                from itertools import product as _prod
                for dr, dc in _prod(range(-rad, rad + 1), range(-rad, rad + 1)):
                    rr = tr_rows + dr
                    cc = tr_cols + dc
                    inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                    rr, cc, dd = rr[inb], cc[inb], tr_deps[inb]
                    # Don't overwrite a previously-set label with a different one
                    new = ~label_mask_cnn[rr, cc]
                    label_grid_cnn[rr[new], cc[new]] = dd[new].astype(np.float32)
                    label_mask_cnn[rr[new], cc[new]] = True
            n_lbl = int(label_mask_cnn.sum())
            # Heavy U-Net only when there's a meaningful label pool. The
            # default threshold (1500) holds for typical multi-source runs;
            # an i-Boating-only run with K×K patching can train with as few
            # as ~200 labels if the user dials cnn_min_labels accordingly.
            if n_lbl < int(cnn_min_labels):
                L.info(f"CNN refinement skipped — only {n_lbl} labeled px "
                       f"(need {int(cnn_min_labels)}+ for Attention U-Net)")
                raise RuntimeError("too few labels for CNN refinement")
            # Optionally fuse the K-means cluster output into the U-Net
            # input feature stack (one-hot, K extra channels).  This gives
            # the U-Net the cluster prior the user explicitly asked for:
            # "do clustering first then feed clustering outputs with UNET".
            cnn_feats = feats
            if cnn_use_cluster_features:
                hard_grid = hard.reshape(H, W).astype(np.int32)
                K = int(n_clusters)
                onehot = np.zeros((H, W, K), dtype=np.float32)
                for ck in range(K):
                    onehot[..., ck] = (hard_grid == ck).astype(np.float32)
                # land/non-water rows get all-zeros which is still valid
                cnn_feats = np.concatenate([cnn_feats, onehot], axis=-1)
                L.info(f"CNN feature stack: appended {K} cluster one-hot "
                       f"channels → {cnn_feats.shape[-1]} total channels")
            L.info(f"CNN refinement: training Attention U-Net on {n_lbl} labeled px")
            # ── s2-dl request #1 — leakage-safe crop sampling (default OFF) ──
            # Build a forbidden_mask = every TEST-block label pixel dilated by
            # the §6 buffer (≥500 m) so no training crop ingests a test pixel
            # or its spatially-correlated neighbours.  Only constructed when
            # UNET_BUFFER_TEST=1 AND we are on a spatial-block split.
            forbidden_mask = None
            if os.environ.get("UNET_BUFFER_TEST", "0") == "1":
                if split_mode == "spatial_block":
                    buffer_m = float(os.environ.get("UNET_BUFFER_M", "500"))
                    # metres per pixel from the bbox geometry (lat/lon → m)
                    w_b, s_b, e_b, n_b = bbox
                    lat_mid = 0.5 * (s_b + n_b)
                    mpp_y = (n_b - s_b) * 111000.0 / max(1, H)
                    mpp_x = (e_b - w_b) * 111000.0 * np.cos(np.radians(lat_mid)) / max(1, W)
                    mpp = 0.5 * (mpp_x + mpp_y)
                    rad_px = max(1, int(round(buffer_m / max(1e-6, mpp))))
                    te_rows = rows[test_idx]
                    te_cols = cols[test_idx]
                    seed_mask = np.zeros((H, W), dtype=bool)
                    inb = ((te_rows >= 0) & (te_rows < H)
                           & (te_cols >= 0) & (te_cols < W))
                    seed_mask[te_rows[inb], te_cols[inb]] = True
                    try:
                        from scipy.ndimage import binary_dilation, generate_binary_structure
                        st = generate_binary_structure(2, 1)
                        forbidden_mask = binary_dilation(
                            seed_mask, structure=st, iterations=rad_px)
                    except Exception as _ex:
                        # NumPy fallback: square dilation via max-pool windows
                        fm = seed_mask.copy()
                        ys, xs = np.where(seed_mask)
                        for yy, xx in zip(ys, xs):
                            fm[max(0, yy - rad_px):yy + rad_px + 1,
                               max(0, xx - rad_px):xx + rad_px + 1] = True
                        forbidden_mask = fm
                    L.info(f"UNET_BUFFER_TEST=1: forbidden_mask = {int(test_idx.sum())} "
                           f"test pts dilated by {buffer_m:.0f} m "
                           f"(~{rad_px} px @ {mpp:.1f} m/px) → "
                           f"{int(forbidden_mask.sum())} forbidden px")
                else:
                    L.info("UNET_BUFFER_TEST=1 ignored — split_mode is not spatial_block")
            unet = train_unet_sdb(
                cnn_feats, label_grid_cnn, label_mask_cnn,
                epochs=int(cnn_epochs),
                crops_per_epoch=40,
                batch=int(cnn_batch),
                crop=int(cnn_crop),
                lr=1e-3,
                base=int(cnn_base),    # base=16 -> ~2 M params, CPU-friendly
                seed=seed,
                forbidden_mask=forbidden_mask,
            )
            cnn_grid, cnn_sigma = predict_grid_tta(
                unet, cnn_feats, water, tile=512, overlap=64, use_tta=True)
            # Blend with HGB grid (water cells only; preserve NaN over land)
            wmask = water & np.isfinite(depth_grid) & np.isfinite(cnn_grid)
            blend = depth_grid.copy()
            blend[wmask] = (
                (1.0 - cnn_blend_weight) * depth_grid[wmask]
                + cnn_blend_weight * cnn_grid[wmask]
            )
            depth_grid = np.clip(blend, 0.0, MAX_DEPTH_M)
            depth_grid = np.where(water, depth_grid, np.nan)

            # ── v8: per-cluster post-CNN linear calibration ──
            # Each pixel's depth gets corrected by its CLUSTER-SPECIFIC
            # (α_c, β_c).  This is the "channels classification as first
            # layer" idea applied at the calibration stage: instead of one
            # global α + β·pred line, each spectral cluster gets its own
            # depth regime.  Breaks the global mean-collapse.
            if per_cluster_calibration:
                # Recompute training cluster ids (hard assignment for
                # train rows/cols) and CNN-blended preds for those pixels.
                hard_grid_pc = hard.reshape(H, W)
                pc_alphas = np.zeros(int(n_clusters), dtype=np.float64)
                pc_betas = np.ones(int(n_clusters), dtype=np.float64)
                pc_n = np.zeros(int(n_clusters), dtype=np.int64)
                # Use train_idx samples for fitting
                tr_rows_pc = rows[train_idx]
                tr_cols_pc = cols[train_idx]
                tr_truth_pc = deps[train_idx]
                tr_pred_pc = depth_grid[tr_rows_pc, tr_cols_pc]
                tr_clust_pc = hard_grid_pc[tr_rows_pc, tr_cols_pc]
                # Optionally exclude shallow training points from fitting
                # to match the global calibration's behaviour.
                if min_train_depth_m > 0:
                    keep_pc = tr_truth_pc >= float(min_train_depth_m)
                    tr_truth_pc = tr_truth_pc[keep_pc]
                    tr_pred_pc = tr_pred_pc[keep_pc]
                    tr_clust_pc = tr_clust_pc[keep_pc]
                # Fit per cluster, fall back to identity for clusters with
                # < 8 training samples or implausible β.
                for cid in range(int(n_clusters)):
                    in_c = (tr_clust_pc == cid)
                    n_c = int(in_c.sum())
                    pc_n[cid] = n_c
                    if n_c < 8:
                        continue
                    a_c, b_c = fit_linear_calibration(tr_pred_pc[in_c],
                                                       tr_truth_pc[in_c])
                    # v8: looser bounds — per-cluster cal often needs β > 1.5
                    # to uncompress the mean-collapsed predictions for that
                    # specific spectral cluster.
                    if not (-8.0 <= a_c <= 8.0 and 0.4 <= b_c <= 2.8):
                        continue
                    pc_alphas[cid] = a_c
                    pc_betas[cid] = b_c
                L.info("Per-cluster calibration (α / β / n):")
                for cid in range(int(n_clusters)):
                    L.info(f"  cluster {cid}:  α={pc_alphas[cid]:+.3f}  "
                           f"β={pc_betas[cid]:.3f}  n_train={int(pc_n[cid])}")
                # Apply per-cluster calibration to depth_grid
                cal_grid = depth_grid.copy()
                for cid in range(int(n_clusters)):
                    if pc_n[cid] < 8 or pc_betas[cid] == 1.0 and pc_alphas[cid] == 0.0:
                        continue
                    mask_c = (hard_grid_pc == cid) & water & np.isfinite(depth_grid)
                    cal_grid[mask_c] = np.clip(
                        pc_alphas[cid] + pc_betas[cid] * depth_grid[mask_c],
                        0.0, MAX_DEPTH_M)
                depth_grid = cal_grid
                metrics["per_cluster_calibration"] = {
                    "alphas": pc_alphas.tolist(),
                    "betas": pc_betas.tolist(),
                    "n_per_cluster": pc_n.tolist(),
                }

            # Update test metrics with the blended predictions
            yp_full_blend = depth_grid[rows, cols]
            yp_test_b = np.clip(yp_full_blend[test_idx], 0, MAX_DEPTH_M)
            mb = compute_metrics(yt_test, yp_test_b)
            mb["s44_1a_pct"] = round(s44_pass_pct(yt_test, yp_test_b, a=0.5,  b=0.013), 1)
            mb["s44_1b_pct"] = round(s44_pass_pct(yt_test, yp_test_b, a=0.5,  b=0.013), 1)
            mb["s44_order2_pct"] = round(s44_pass_pct(yt_test, yp_test_b, a=1.0,  b=0.023), 1)
            mb["s44_special_pct"] = round(s44_pass_pct(yt_test, yp_test_b, a=0.25, b=0.0075), 1)
            for k_ in ("rmse_m", "mae_m", "bias_m", "r2",
                        "s44_1a_pct", "s44_1b_pct",
                        "s44_order2_pct", "s44_special_pct"):
                metrics[k_] = mb[k_]
            metrics["cnn_blend_weight"] = float(cnn_blend_weight)
            metrics["used_cnn_refinement"] = True
            metrics["cnn_architecture"] = "AttentionUNet+TTA8"
            yp_test = yp_test_b
            per_band = metrics_per_band(yt_test, yp_test)
            if min_train_depth_m > 0:
                deep = yt_test >= float(min_train_depth_m)
                if int(deep.sum()) >= 1:
                    yt_d, yp_d = yt_test[deep], yp_test[deep]
                    md = compute_metrics(yt_d, yp_d)
                    md["s44_1a_pct"] = round(s44_pass_pct(yt_d, yp_d, a=0.5, b=0.013), 1)
                    md["s44_1b_pct"] = round(s44_pass_pct(yt_d, yp_d, a=0.5, b=0.013), 1)
                    md["s44_order2_pct"] = round(s44_pass_pct(yt_d, yp_d, a=1.0, b=0.023), 1)
                    md["s44_special_pct"] = round(s44_pass_pct(yt_d, yp_d, a=0.25, b=0.0075), 1)
                    md["depth_floor_m"] = float(min_train_depth_m)
                    metrics["overall_excluding_shallow"] = md
            L.info(f"CNN refinement done — blended RMSE={metrics['rmse_m']}m  "
                   f"R²={metrics['r2']}  S-44 1A={metrics['s44_1a_pct']}%")
        except Exception as ex:
            L.warning(f"CNN refinement skipped: {ex}")

    elapsed = round(time.time() - t0, 1)
    L.info(f"Done {elapsed}s — RMSE={metrics['rmse_m']}m  R²={metrics['r2']}  "
           f"bias={metrics['bias_m']}m  S-44 1A={metrics['s44_1a_pct']}%")

    # Decorate the depth grid: turbid pixels become NaN.  The UI consumes
    # ``turbidity_pct`` and the warning string to surface the caveat.
    depth_grid = np.where(turbid_mask, np.nan, depth_grid)

    turbidity_warning = None
    if turbidity_pct >= 25:
        turbidity_warning = (
            f"⚠ {turbidity_pct:.1f}% of the water area is highly turbid "
            "(sediment plume or sun-glint) — those pixels are masked NaN; "
            "treat depths near these regions with caution."
        )
    elif turbidity_pct >= 5:
        turbidity_warning = (
            f"{turbidity_pct:.1f}% of the water area is turbid — masked from "
            "the depth grid."
        )

    result = {
        "depth": depth_grid,
        "depth_sigma": cnn_sigma if 'cnn_sigma' in dir() else None,
        "water": water,
        "water_full": water_full,
        "osm_land": osm_land,
        "turbid_mask": turbid_mask,
        "turbidity_pct": turbidity_pct,
        "turbidity_warning": turbidity_warning,
        "cluster_map": cluster_map,
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
        "augmentation": aug_summary,
        "s2_status": s2_status,
        "era5_status": era5_status,
        "imagery_source": imagery_source,
        "water_mask_meta": vhr_mask_meta,
    }
    if save:
        result["paths"] = save_artefacts(site_key, bbox, rgb, water,
                                           osm_land, cluster_map, n_clusters,
                                           result,
                                           out_dir=Path(out_dir) if out_dir else RESULTS_DIR)
    return result
