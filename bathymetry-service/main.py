"""Abyss Bathymetry Service — real Satellite-Derived Bathymetry.

Two inference paths, same output contract (georeferenced float depth GeoTIFF +
RGB depth map + stats + cross-section + downsampled grids):

* Ingested raster (source_bucket/source_key): the raster's own pixels drive
  `sdb_engine.infer_depth()` — the UAE-calibrated cluster ensemble (RF + MLP)
  when multispectral bands (B/G/R + NIR) are present, or an explicitly
  uncalibrated Stumpf log-ratio pseudo-depth for plain RGB.
* ROI bbox (no source raster): fetch free Sentinel-2 L2A (Planetary Computer)
  and run the DL-Pro v3 model (23-feature turbidity-aware MLP), unchanged.

POST /bathymetry/infer  {raster_id, source_bucket?, source_key?,
                         bbox:{west,south,east,north}|[w,s,e,n],
                         start_date?, end_date?, max_cloud?}
GET  /bathymetry/health
GET  /bathymetry/models
"""
from __future__ import annotations

import io
import os
import math
import logging

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from fastapi import FastAPI, HTTPException
from minio import Minio
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bathymetry")

app = FastAPI(title="Abyss Bathymetry Service", version="2.0.0")

MODEL_NAME = "dl-pro-v3"
MAX_DEPTH_M = 25.0
DEPTH_BUCKET = "depth"


def _minio() -> Minio:
    return Minio(
        os.getenv("MINIO_ENDPOINT", "globe-minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes"),
    )


def _ensure_bucket(client, bucket):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def _clean(o):
    """Recursively convert numpy types → native Python so FastAPI can serialize."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and o != o:  # NaN
        return None
    return o


def _downsample(arr, max_side=512):
    """Downsample a 2D array to <=max_side on its long edge (block mean,
    NaN-aware) and return (rows-of-lists with None for NaN, h, w)."""
    H, W = arr.shape
    step = max(1, int(math.ceil(max(H, W) / float(max_side))))
    gh, gw = H // step, W // step
    gh, gw = max(1, gh), max(1, gw)
    a = arr[: gh * step, : gw * step].reshape(gh, step, gw, step)
    with np.errstate(invalid="ignore"):
        g = np.nanmean(a, axis=(1, 3))
    out = [[None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in g]
    return out, gh, gw


def _colormap(norm):
    from sdb_engine import depth_colormap
    return depth_colormap(norm)


def _put_geotiff(client, key, array, transform, count, dtype, nodata=None,
                 photometric=None, crs="EPSG:4326"):
    profile = dict(driver="GTiff", height=array.shape[-2], width=array.shape[-1],
                   count=count, dtype=dtype, crs=crs, transform=transform,
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    if nodata is not None:
        profile["nodata"] = nodata
    if photometric:
        profile["photometric"] = photometric
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            if count == 1:
                dst.write(array, 1)
            else:
                dst.write(array)
        buf = mem.read()
    client.put_object(DEPTH_BUCKET, key, io.BytesIO(buf), length=len(buf),
                      content_type="image/tiff")


class InferRequest(BaseModel):
    raster_id: str
    bbox: object | None = None          # {west,south,east,north} or [w,s,e,n]
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    max_cloud: int = 40
    # Ingested-raster path: run the SDB engine on this MinIO object's pixels.
    source_bucket: str | None = None
    source_key: str | None = None
    max_depth: float | None = None


def _parse_bbox(bbox):
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        try:
            return [float(bbox["west"]), float(bbox["south"]),
                    float(bbox["east"]), float(bbox["north"])]
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [float(x) for x in bbox]
    return None


@app.get("/bathymetry/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/bathymetry/models")
def models():
    info = {"models": [MODEL_NAME], "default": MODEL_NAME, "units": "meters",
            "max_depth_m": MAX_DEPTH_M}
    try:
        from dl_pro_engine import load_bundle
        b = load_bundle()
        info["meta"] = {
            "name": b["meta"].get("name"),
            "version": b["meta"].get("version"),
            "holdout_rmse": b["meta"]["holdout_metrics_calibrated"]["rmse"],
        }
    except Exception as ex:
        info["meta_error"] = str(ex)
    try:
        from sdb_engine import model_info
        info["raster_engine"] = model_info()
        info["models"].append(info["raster_engine"]["model"])
    except Exception as ex:
        info["raster_engine_error"] = str(ex)
    return info


def _products_payload(client, raster_id, depth, sigma_src, transform, crs,
                      bbox, extra):
    """Write the depth GeoTIFF pair to MinIO and assemble the response payload
    shared by both inference paths. `bbox` is [w,s,e,n] in EPSG:4326."""
    H, W = depth.shape
    valid = np.isfinite(depth)

    depth_f = np.where(valid, depth, -9999.0).astype(np.float32)
    raw_key = f"{raster_id}/depth.tif"
    _put_geotiff(client, raw_key, depth_f, transform, 1, "float32",
                 nodata=-9999.0, crs=crs)

    norm = np.where(valid, np.clip(depth / MAX_DEPTH_M, 0, 1), 0.0)
    rgb = np.transpose(_colormap(norm), (2, 0, 1))      # (3,H,W)
    rgb[:, ~valid] = 0
    rgb_key = f"{raster_id}/depth_rgb.tif"
    _put_geotiff(client, rgb_key, rgb, transform, 3, "uint8", nodata=0,
                 photometric="RGB", crs=crs)

    vals = depth[valid]
    stats = {
        "min_m": round(float(np.min(vals)), 2),
        "max_m": round(float(np.max(vals)), 2),
        "mean_m": round(float(np.mean(vals)), 2),
        "std_m": round(float(np.std(vals)), 2),
        "coverage_pct": round(100.0 * float(valid.sum()) / float(H * W), 1),
    }

    mid = H // 2
    row = depth[mid]
    res_x = abs(transform.a)
    geographic = "4326" in str(crs) or res_x < 0.1
    step_m = res_x * 111320 if geographic else res_x
    idx = np.linspace(0, W - 1, min(64, W)).astype(int)
    profile = {
        "distance_m": [round(float(i * step_m), 1) for i in idx],
        "depth_m": [None if not np.isfinite(row[i]) else round(float(row[i]), 2) for i in idx],
    }

    depth_grid, gh, gw = _downsample(depth)
    sigma_grid = None
    if sigma_src is not None:
        sigma_grid, _, _ = _downsample(np.asarray(sigma_src, dtype=np.float32))

    payload = {
        "raster_id": raster_id,
        "depth_rgb_key": rgb_key,
        "depth_raw_key": raw_key,
        "depth_bucket": DEPTH_BUCKET,
        "crs": str(crs),
        "bbox": {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]},
        "width": W, "height": H,
        "units": "meters",
        "max_depth_m": MAX_DEPTH_M,
        "stats": stats,
        "grid": {
            "depth": depth_grid,
            "sigma": sigma_grid,
            "rows": gh, "cols": gw,
            "bounds": {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]},
            "max_depth_m": MAX_DEPTH_M,
        },
        "profile": profile,
    }
    payload.update(extra)
    return _clean(payload)


def _infer_from_raster(reqp: InferRequest, bbox):
    """Ingested-raster path: read the source raster from MinIO and run the
    vendored SDB engine on its own pixels."""
    import sdb_engine

    client = _minio()
    try:
        obj = client.get_object(reqp.source_bucket, reqp.source_key)
        blob = obj.read()
        obj.close()
        obj.release_conn()
    except Exception as ex:
        raise HTTPException(status_code=404,
                            detail=f"source raster unavailable: {ex}")

    try:
        with MemoryFile(blob) as mem, mem.open() as src:
            bands = src.read(masked=True).astype(np.float32).filled(np.nan)
            names = [d for d in (src.descriptions or ())]
            src_dtype = src.dtypes[0]
            transform = src.transform
            crs = src.crs
            src_bounds = src.bounds
    except Exception as ex:
        raise HTTPException(status_code=422,
                            detail=f"source is not a readable raster: {ex}")

    if crs is None or not transform or transform.is_identity:
        if bbox is None:
            raise HTTPException(status_code=422,
                                detail="source raster has no georeferencing and no bbox given")
        from rasterio.transform import from_bounds
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3],
                                bands.shape[2], bands.shape[1])
        crs = "EPSG:4326"
        bbox4326 = bbox
    else:
        from rasterio.warp import transform_bounds
        bbox4326 = list(transform_bounds(crs, "EPSG:4326", *src_bounds))

    res_x = abs(transform.a)
    res_m = res_x * 111320 if "4326" in str(crs) or res_x < 0.1 else res_x
    max_depth = float(reqp.max_depth or MAX_DEPTH_M)
    try:
        result = sdb_engine.infer(bands, band_names=names,
                                  max_depth=min(max_depth, MAX_DEPTH_M),
                                  resolution_m=res_m, src_dtype=src_dtype)
    except ValueError as ex:
        raise HTTPException(status_code=422, detail=str(ex))
    except Exception as ex:
        log.exception("SDB inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {ex}")

    depth = result["depth"]
    if not np.isfinite(depth).any():
        raise HTTPException(status_code=422,
                            detail="No valid water pixels in the source raster.")

    _ensure_bucket(client, DEPTH_BUCKET)
    extra = {
        "model": result["model"],
        "model_version": result["model_version"],
        "method": result["method"],
        "calibrated": result["calibrated"],
        "confidence": result["confidence"],
        "calibration": result["calibration"],
        "band_mapping": result["band_mapping"],
        "mask": result["mask"],
        "iho_s44_pct": {},
        "holdout_metrics": {},
        "source": {"bucket": reqp.source_bucket, "key": reqp.source_key},
    }
    return _products_payload(client, reqp.raster_id, depth, result.get("sigma"),
                             transform, crs, bbox4326, extra)


@app.post("/bathymetry/infer")
def infer(reqp: InferRequest):
    bbox = _parse_bbox(reqp.bbox)

    # Ingested-raster path: depth from the uploaded scene's own pixels.
    if reqp.source_bucket and reqp.source_key:
        return _infer_from_raster(reqp, bbox)

    if bbox is None:
        raise HTTPException(status_code=400,
                            detail="bbox required: {west,south,east,north} or [w,s,e,n]")

    # 1. Free Sentinel-2 L2A for the ROI.
    from s2_fetch import fetch_s2, S2FetchError
    try:
        s2 = fetch_s2(bbox, reqp.start_date, reqp.end_date, max_cloud=reqp.max_cloud)
    except S2FetchError as ex:
        raise HTTPException(status_code=422, detail=str(ex))
    except Exception as ex:
        log.exception("S2 fetch failed")
        raise HTTPException(status_code=502, detail=f"S2 fetch failed: {ex}")

    # 2. DL-Pro inference (verbatim engine).
    try:
        from dl_pro_engine import predict_dl_pro
        result = predict_dl_pro(s2, bbox=bbox)
    except Exception as ex:
        log.exception("DL-Pro inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {ex}")

    depth = result["depth"]                       # (H,W) float32, NaN on land
    transform = s2["transform"]
    if not np.isfinite(depth).any():
        raise HTTPException(status_code=422,
                            detail="No valid water pixels in the ROI scene.")

    client = _minio()
    _ensure_bucket(client, DEPTH_BUCKET)

    iho = {k: round(float(v), 1) for k, v in (result.get("iho_s44") or {}).items()}
    mm = result.get("model_meta", {})
    holdout = mm.get("holdout_metrics", {})
    sigma_src = result.get("sigma_eff")
    if sigma_src is None:
        sigma_src = result.get("sigma")

    extra = {
        "model": MODEL_NAME,
        "model_version": mm.get("version"),
        "iho_s44_pct": iho,
        "holdout_metrics": {k: round(float(v), 3) for k, v in holdout.items()
                            if isinstance(v, (int, float))},
        "calibration": result.get("calibration"),
        "scene_id": s2.get("scene_id"),
        "cloud_cover": s2.get("cloud_cover"),
        "acquired": s2.get("acquired"),
    }
    return _products_payload(client, reqp.raster_id, depth, sigma_src,
                             transform, "EPSG:4326", bbox, extra)
