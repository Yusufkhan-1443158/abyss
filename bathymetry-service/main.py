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
    # ISO 8601 UTC acquisition time (ingested path) — enables tide correction.
    acquisition_datetime: str | None = None
    # ROI path: number of scenes to composite (1 = single least-cloudy scene).
    n_scenes: int = 1


class ValidateRequest(BaseModel):
    raster_id: str | None = None
    predicted: list | None = None
    observed: list
    max_match_m: float | None = None
    resolution_m: float | None = None
    observed_vertical_datum: str | None = None
    predicted_vertical_datum: str | None = None


def _parse_utc(value):
    """ISO 8601 string -> aware UTC datetime, or None."""
    from datetime import datetime, timezone
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _tide_correct(depth, bbox, acquired_iso):
    """Apply the Open-Meteo tide correction when enabled and the acquisition
    time parses. Returns (depth, tide_block) — depth unchanged unless the API
    delivered a height."""
    if os.getenv("TIDE_CORRECTION", "1") in ("0", "false", "no"):
        return depth, {"applied": False, "applied_m": 0.0, "method": "disabled",
                       "utc": None, "datum": "uncorrected",
                       "reason": "disabled by TIDE_CORRECTION=0"}
    utc_dt = _parse_utc(acquired_iso)
    if utc_dt is None:
        return depth, {"applied": False, "applied_m": 0.0, "method": "none",
                       "utc": None, "datum": "uncorrected",
                       "reason": "acquisition time unknown"}
    from tide import apply_tide_correction
    corr, tide_m, info = apply_tide_correction(depth, bbox, utc_dt,
                                               max_depth_m=MAX_DEPTH_M)
    if info.get("method") == "open-meteo/cmems":
        return corr, {"applied": True, "applied_m": float(tide_m),
                      "method": info["method"], "utc": info.get("utc"),
                      "datum": "MSL",
                      "tide_range_that_day_m": info.get("tide_range_that_day_m")}
    return depth, {"applied": False, "applied_m": 0.0,
                   "method": info.get("method", "unavailable"),
                   "utc": info.get("utc"), "datum": "uncorrected",
                   "reason": info.get("reason", "tide API unavailable")}


def _coastline_cut_grids(bbox, depth, grids=()):
    """NaN out vector-coastline land on `depth` (+ companion grids). Returns
    (depth, grids, info) — unchanged outside the committed regions."""
    from sdb_engine.coastline_mask import coastline_land_for
    land, info = coastline_land_for(bbox, depth.shape)
    if land is not None and land.any():
        depth = np.where(land, np.nan, depth)
        grids = tuple(None if g is None else np.where(land, np.nan, g)
                      for g in grids)
    return depth, grids, info


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
                                  resolution_m=res_m, src_dtype=src_dtype,
                                  bbox4326=bbox4326)
    except ValueError as ex:
        raise HTTPException(status_code=422, detail=str(ex))
    except Exception as ex:
        log.exception("SDB inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {ex}")

    depth = result["depth"]
    if not np.isfinite(depth).any():
        raise HTTPException(status_code=422,
                            detail="No valid water pixels in the source raster.")

    sigma = result.get("sigma")
    if reqp.acquisition_datetime:
        depth, tide = _tide_correct(depth, bbox4326, reqp.acquisition_datetime)
    else:
        tide = {"applied": False, "applied_m": 0.0, "method": "none",
                "utc": None, "datum": "uncorrected",
                "reason": "acquisition time unknown"}

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
        "tide": tide,
    }
    return _products_payload(client, reqp.raster_id, depth, sigma,
                             transform, crs, bbox4326, extra)


def _sample_depth_raster(raster_id: str, max_points: int = 200_000) -> list:
    """Read depth/{raster_id}/depth.tif from MinIO and sample its valid pixels
    to {lat, lon, depth} triples (strided so <= max_points)."""
    client = _minio()
    try:
        obj = client.get_object(DEPTH_BUCKET, f"{raster_id}/depth.tif")
        blob = obj.read()
        obj.close()
        obj.release_conn()
    except Exception as ex:
        raise HTTPException(status_code=404,
                            detail=f"depth product unavailable for raster "
                                   f"{raster_id}: {ex}")
    with MemoryFile(blob) as mem, mem.open() as src:
        depth = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
    valid = np.isfinite(depth) & (depth > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise HTTPException(status_code=422,
                            detail="depth product has no valid pixels")
    from rasterio.transform import xy
    stride = max(1, int(math.ceil(math.sqrt(n_valid / float(max_points)))))
    rows, cols = np.nonzero(valid)
    keep = (rows % stride == 0) & (cols % stride == 0)
    rows, cols = rows[keep], cols[keep]
    xs, ys = xy(transform, rows, cols)
    return [{"lat": float(y), "lon": float(x), "depth": float(depth[r, c])}
            for r, c, x, y in zip(rows, cols, xs, ys)]


@app.post("/bathymetry/validate")
def validate(reqp: ValidateRequest):
    """IHO S-44 validation of a depth product against reference soundings."""
    from validation import validate_points, ValidationError

    predicted = reqp.predicted
    if predicted is None and reqp.raster_id:
        predicted = _sample_depth_raster(reqp.raster_id)
    if predicted is None:
        raise HTTPException(status_code=400,
                            detail="provide 'predicted' points or a 'raster_id' "
                                   "with a stored depth product")
    try:
        result = validate_points(
            predicted, reqp.observed,
            max_match_m=reqp.max_match_m or 50.0,
            resolution_m=reqp.resolution_m,
            observed_vertical_datum=reqp.observed_vertical_datum,
            predicted_vertical_datum=reqp.predicted_vertical_datum or "LAT",
        )
    except ValidationError as ex:
        raise HTTPException(status_code=400, detail=str(ex))
    return _clean(result)


@app.post("/bathymetry/infer")
def infer(reqp: InferRequest):
    bbox = _parse_bbox(reqp.bbox)

    # Ingested-raster path: depth from the uploaded scene's own pixels.
    if reqp.source_bucket and reqp.source_key:
        return _infer_from_raster(reqp, bbox)

    if bbox is None:
        raise HTTPException(status_code=400,
                            detail="bbox required: {west,south,east,north} or [w,s,e,n]")

    n_scenes = max(1, min(int(reqp.n_scenes or 1), 5))

    # 1. Free Sentinel-2 L2A for the ROI.
    from s2_fetch import fetch_s2, fetch_s2_scenes, S2FetchError
    try:
        if n_scenes == 1:
            scenes = [fetch_s2(bbox, reqp.start_date, reqp.end_date,
                               max_cloud=reqp.max_cloud)]
        else:
            scenes = fetch_s2_scenes(bbox, reqp.start_date, reqp.end_date,
                                     max_cloud=reqp.max_cloud, n_scenes=n_scenes)
    except S2FetchError as ex:
        raise HTTPException(status_code=422, detail=str(ex))
    except Exception as ex:
        log.exception("S2 fetch failed")
        raise HTTPException(status_code=502, detail=f"S2 fetch failed: {ex}")

    # 2. DL-Pro inference (verbatim engine), once per scene.
    try:
        from dl_pro_engine import predict_dl_pro
        scene_runs = [(s2, predict_dl_pro(s2, bbox=bbox)) for s2 in scenes]
    except Exception as ex:
        log.exception("DL-Pro inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {ex}")

    composite_block = None
    if n_scenes == 1:
        s2, result = scene_runs[0]
        depth = result["depth"]                   # (H,W) float32, NaN on land
        sigma_src = result.get("sigma_eff")
        if sigma_src is None:
            sigma_src = result.get("sigma")
        depth, tide = _tide_correct(depth, bbox, s2.get("acquired"))
    else:
        # 2b. Per-scene tide correction + inverse-variance composite.
        from composite import compose
        entries, tide_blocks = [], []
        for s2_i, r_i in scene_runs:
            d_i, tb = _tide_correct(r_i["depth"], bbox, s2_i.get("acquired"))
            tide_blocks.append(tb)
            sig = r_i.get("sigma_eff")
            if sig is None:
                sig = r_i.get("sigma")
            if sig is None:
                holdout_rmse = ((r_i.get("model_meta") or {})
                                .get("holdout_metrics") or {}).get("rmse")
                sig = float(holdout_rmse or 1.5)
            glint = None
            nir, wmask = s2_i.get("nir"), s2_i.get("water_mask")
            if nir is not None and wmask is not None:
                wmask = np.asarray(wmask, bool)
                if wmask.any():
                    glint = float(np.nanmean(
                        np.asarray(nir, np.float64)[wmask])) / 10000.0
            entries.append({
                "depth": d_i, "sigma": sig,
                "scene_id": s2_i.get("scene_id"),
                "acquired": s2_i.get("acquired"),
                "cloud_cover": s2_i.get("cloud_cover"),
                "glint": glint,
                "tide_m": tb.get("applied_m") if tb.get("applied") else None,
            })
        comp = compose(entries)
        depth, sigma_src = comp["depth"], comp["sigma"]
        composite_block = comp["agreement"]
        kept_idx = [i for i, p in enumerate(composite_block["per_scene"])
                    if p["kept"]]
        s2, result = scene_runs[kept_idx[0]]
        kept_tides = [tide_blocks[i] for i in kept_idx]
        if all(t["applied"] for t in kept_tides):
            tide = {"applied": True,
                    "applied_m": round(float(np.mean(
                        [t["applied_m"] for t in kept_tides])), 3),
                    "method": "open-meteo/cmems (per scene)",
                    "utc": kept_tides[0].get("utc"), "datum": "MSL"}
        else:
            reasons = {t.get("reason") for t in kept_tides if not t["applied"]}
            tide = {"applied": False, "applied_m": 0.0,
                    "method": "none", "utc": None, "datum": "uncorrected",
                    "reason": "; ".join(sorted(r for r in reasons if r))
                              or "tide unavailable for one or more scenes"}

    transform = s2["transform"]

    # 3. Vector-coastline land cut (authoritative for land only).
    depth, (sigma_src,), coast_info = _coastline_cut_grids(bbox, depth,
                                                           (sigma_src,))
    if not np.isfinite(depth).any():
        raise HTTPException(status_code=422,
                            detail="No valid water pixels in the ROI scene.")

    client = _minio()
    _ensure_bucket(client, DEPTH_BUCKET)

    iho = {k: round(float(v), 1) for k, v in (result.get("iho_s44") or {}).items()}
    mm = result.get("model_meta", {})
    holdout = mm.get("holdout_metrics", {})

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
        "tide": tide,
        "mask_coastline": coast_info,
    }
    if composite_block is not None:
        extra["composite"] = composite_block
    return _products_payload(client, reqp.raster_id, depth, sigma_src,
                             transform, "EPSG:4326", bbox, extra)
