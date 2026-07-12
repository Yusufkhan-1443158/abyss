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

app = FastAPI(title="Abyss Bathymetry Service", version="2.1.0")

DEFAULT_ENGINE = "uae-sdb-ensemble"   # registry-v1 clustered ensemble (VMarch)
FALLBACK_ENGINE = "dl-pro-v3"
ENGINES = (DEFAULT_ENGINE, FALLBACK_ENGINE)
MAX_DEPTH_M = 25.0
MAX_SCENES = 7
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
    # ROI path: depth engine — DEFAULT_ENGINE unless explicitly overridden.
    engine: str | None = None


class CalibrateRequest(BaseModel):
    raster_id: str
    observed: list                     # [{lat, lon, depth}, ...]
    out_raster_id: str | None = None
    holdout_frac: float = 0.2
    max_shift_px: int = 5


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
    return {"status": "ok", "model": DEFAULT_ENGINE, "engines": list(ENGINES)}


@app.get("/bathymetry/models")
def models():
    info = {"models": list(ENGINES), "default": DEFAULT_ENGINE,
            "units": "meters", "max_depth_m": MAX_DEPTH_M}
    try:
        from sdb_engine import model_info
        eng = model_info()
        info["raster_engine"] = eng
        info["meta"] = {
            "name": DEFAULT_ENGINE,
            "version": f"registry-v{eng.get('registry_version', 1)}",
            "in_sample_rmse": (eng.get("calibration") or {}).get("in_sample_rmse_m"),
        }
    except Exception as ex:
        info["raster_engine_error"] = str(ex)
    try:
        from dl_pro_engine import load_bundle
        b = load_bundle()
        info["dl_pro"] = {
            "name": b["meta"].get("name"),
            "version": b["meta"].get("version"),
            "holdout_rmse": b["meta"]["holdout_metrics_calibrated"]["rmse"],
        }
        info.setdefault("meta", {"name": FALLBACK_ENGINE,
                                 "version": b["meta"].get("version")})
    except Exception as ex:
        info["dl_pro_error"] = str(ex)
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


def _run_roi_engine(s2, bbox, engine):
    """Run one fetched S2 scene through the selected depth engine, normalised
    to a shared contract. The UAE ensemble (registry-v1) is the default; if it
    cannot run, the scene falls back to DL-Pro with provenance recorded."""
    if engine == FALLBACK_ENGINE:
        from dl_pro_engine import predict_dl_pro
        r = predict_dl_pro(s2, bbox=bbox)
        mm = r.get("model_meta", {})
        sigma = r.get("sigma_eff")
        if sigma is None:
            sigma = r.get("sigma")
        return {
            "depth": r["depth"], "sigma": sigma,
            "model": FALLBACK_ENGINE, "model_version": mm.get("version"),
            "method": "DL-Pro v3 — 23-feature turbidity-aware MLP",
            "calibration": r.get("calibration"),
            "iho_s44": r.get("iho_s44") or {},
            "holdout_metrics": mm.get("holdout_metrics", {}),
            "engine": FALLBACK_ENGINE, "engine_fallback": None,
        }
    import sdb_engine
    try:
        r = sdb_engine.infer_s2_scene(s2, max_depth=MAX_DEPTH_M, bbox4326=bbox)
        if not np.isfinite(r["depth"]).any():
            raise RuntimeError("ensemble produced no valid water pixels")
    except Exception as ex:
        log.warning("UAE ensemble unavailable (%s) — falling back to DL-Pro", ex)
        out = _run_roi_engine(s2, bbox, FALLBACK_ENGINE)
        out["engine_fallback"] = f"{DEFAULT_ENGINE} unavailable: {ex}"
        return out
    return {
        "depth": r["depth"], "sigma": r.get("sigma"),
        "model": r["model"], "model_version": r["model_version"],
        "method": r["method"], "calibration": r.get("calibration"),
        "iho_s44": {}, "holdout_metrics": {}, "mask": r.get("mask"),
        "engine": DEFAULT_ENGINE, "engine_fallback": None,
    }


@app.post("/bathymetry/calibrate")
def calibrate(reqp: CalibrateRequest):
    """Fit a robust local calibration of a stored depth product to user
    soundings (80/20 honest holdout) and emit the calibrated grid as a NEW
    depth product — the original is never overwritten."""
    import uuid as _uuid
    from calibration import calibrate_local, CalibrationError

    client = _minio()
    try:
        obj = client.get_object(DEPTH_BUCKET, f"{reqp.raster_id}/depth.tif")
        blob = obj.read()
        obj.close()
        obj.release_conn()
    except Exception as ex:
        raise HTTPException(status_code=404,
                            detail=f"depth product unavailable for raster "
                                   f"{reqp.raster_id}: {ex}")
    with MemoryFile(blob) as mem, mem.open() as src:
        depth = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = src.transform
        crs = src.crs or "EPSG:4326"
        bounds = src.bounds
    if "4326" in str(crs):
        bbox4326 = [bounds.left, bounds.bottom, bounds.right, bounds.top]
    else:
        from rasterio.warp import transform_bounds
        bbox4326 = list(transform_bounds(crs, "EPSG:4326", *bounds))

    lats, lons, obs = [], [], []
    for p in reqp.observed or []:
        try:
            lats.append(float(p["lat"]))
            lons.append(float(p["lon"]))
            obs.append(float(p["depth"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not obs:
        raise HTTPException(status_code=400,
                            detail="observed must be [{lat, lon, depth}, ...]")
    xs, ys = lons, lats
    if "4326" not in str(crs):
        from rasterio.warp import transform as _rio_transform
        xs, ys = _rio_transform("EPSG:4326", crs, lons, lats)
    from rasterio.transform import rowcol
    rows, cols = rowcol(transform, xs, ys)

    try:
        depth_cal, info = calibrate_local(
            depth, np.asarray(rows), np.asarray(cols), np.asarray(obs),
            holdout_frac=reqp.holdout_frac, max_shift_px=reqp.max_shift_px,
            max_depth=MAX_DEPTH_M)
    except CalibrationError as ex:
        raise HTTPException(status_code=400, detail=str(ex))

    out_id = reqp.out_raster_id or f"{reqp.raster_id}-cal-{_uuid.uuid4().hex[:8]}"
    _ensure_bucket(client, DEPTH_BUCKET)
    extra = {
        "method": "local calibration (shift + Huber linear + IDW residual field)",
        "calibration_local": info,
        "derived_from_raster_id": reqp.raster_id,
    }
    return _products_payload(client, out_id, depth_cal, None, transform, crs,
                             bbox4326, extra)


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

    n_scenes = max(1, min(int(reqp.n_scenes or 1), MAX_SCENES))
    engine = (reqp.engine or DEFAULT_ENGINE).strip().lower()
    if engine not in ENGINES:
        raise HTTPException(status_code=400,
                            detail=f"unknown engine '{engine}' "
                                   f"(choose from {list(ENGINES)})")

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

    # 2. Depth inference, once per scene (UAE ensemble default; DL-Pro on
    #    request or as automatic fallback).
    try:
        scene_runs = [(s2, _run_roi_engine(s2, bbox, engine)) for s2 in scenes]
    except Exception as ex:
        log.exception("depth inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {ex}")

    composite_block = None
    if n_scenes == 1:
        s2, result = scene_runs[0]
        depth = result["depth"]                   # (H,W) float32, NaN on land
        sigma_src = result.get("sigma")
        depth, tide = _tide_correct(depth, bbox, s2.get("acquired"))
    else:
        # 2b. Per-scene tide correction + inverse-variance composite.
        from composite import compose
        entries, tide_blocks = [], []
        for s2_i, r_i in scene_runs:
            d_i, tb = _tide_correct(r_i["depth"], bbox, s2_i.get("acquired"))
            tide_blocks.append(tb)
            sig = r_i.get("sigma")
            if sig is None:
                holdout_rmse = (r_i.get("holdout_metrics") or {}).get("rmse")
                sig = float(holdout_rmse or 1.5)
            # Glint proxy (production _scene_physical_qc): median NIR
            # reflectance over the DEEP-quartile water pixels — water is
            # NIR-black at depth, so a high value flags glint/haze. An
            # all-water mean is bottom-contaminated over shallow banks.
            glint = None
            nir, wmask = s2_i.get("nir"), s2_i.get("water_mask")
            if nir is not None and wmask is not None:
                wmask = np.asarray(wmask, bool)
                nir_r = np.asarray(nir, np.float64) / 10000.0
                d_arr = np.asarray(r_i["depth"], np.float32)
                wet = wmask & np.isfinite(d_arr) & np.isfinite(nir_r)
                if wet.any():
                    q75 = float(np.nanpercentile(d_arr[wet], 75))
                    deep = wet & (d_arr >= q75)
                    sel = deep if deep.any() else wet
                    glint = float(np.nanmedian(nir_r[sel]))
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
    holdout = result.get("holdout_metrics", {})

    extra = {
        "model": result["model"],
        "model_version": result.get("model_version"),
        "method": result.get("method"),
        "engine": result.get("engine"),
        "engine_requested": engine,
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
    if result.get("engine_fallback"):
        extra["engine_fallback"] = result["engine_fallback"]
    if composite_block is not None:
        extra["composite"] = composite_block
    return _products_payload(client, reqp.raster_id, depth, sigma_src,
                             transform, "EPSG:4326", bbox, extra)
