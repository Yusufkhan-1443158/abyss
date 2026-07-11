"""Free, no-auth Sentinel-2 L2A fetch via Microsoft Planetary Computer.

Returns the `s2` dict that dl_pro_engine.predict_dl_pro expects:
  blue, green, red, coastal, nir  -> raw L2A DN arrays (reflectance*10000), float32 (H,W)
  ndwi        -> (green-nir)/(green+nir), float32 (H,W)
  water_mask  -> bool (H,W) from SCL water + NDWI, clouds removed
  width,height,transform,bounds,crs -> georeferencing for the output GeoTIFF

No credentials required (PC STAC is open; assets are signed with a free,
keyless SAS endpoint via planetary_computer.sign_inplace).
"""
from __future__ import annotations

import io
import os
import json
import math
import hashlib
import logging
import concurrent.futures as _cf

import numpy as np

log = logging.getLogger("s2_fetch")

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
# PC asset keys for L2A (values are reflectance*10000, harmonized baseline).
ASSETS = {"coastal": "B01", "blue": "B02", "green": "B03",
          "red": "B04", "nir": "B08", "scl": "SCL"}

# --- S2 imagery cache (MinIO) -------------------------------------------------
# Once a scene's bands are fetched+warped for an ROI, store them so re-runs (incl.
# re-inference after a model change) never re-pull from Planetary Computer. Keyed
# by area+dates+cloud+resolution only — deliberately model-independent.
S2_CACHE_BUCKET = "s2cache"


class S2FetchError(RuntimeError):
    pass


def _cache_client():
    from minio import Minio
    return Minio(
        os.getenv("MINIO_ENDPOINT", "globe-minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes"),
    )


def _cache_key(bbox, start_date, end_date, max_cloud, max_px, scene_id=None):
    w, s, e, n = bbox
    raw = (f"{round(w, 5)},{round(s, 5)},{round(e, 5)},{round(n, 5)}"
           f"|{start_date}|{end_date}|{max_cloud}|{max_px}")
    if scene_id:
        raw += f"|{scene_id}"
    return "s2_" + hashlib.sha1(raw.encode()).hexdigest()[:24] + ".npz"


def _serialize_s2(s2) -> bytes:
    """Pack the band stack + georef into a compressed .npz (no JSON — avoids any
    numpy-scalar serialization pitfalls)."""
    t = s2["transform"]
    cc = s2.get("cloud_cover")
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        blue=np.asarray(s2["blue"], np.float32), green=np.asarray(s2["green"], np.float32),
        red=np.asarray(s2["red"], np.float32), nir=np.asarray(s2["nir"], np.float32),
        coastal=np.asarray(s2["coastal"], np.float32), ndwi=np.asarray(s2["ndwi"], np.float32),
        water_mask=np.asarray(s2["water_mask"], bool),
        transform=np.array([t.a, t.b, t.c, t.d, t.e, t.f], np.float64),
        bounds=np.array([float(x) for x in s2["bounds"]], np.float64),
        wh=np.array([int(s2["width"]), int(s2["height"])], np.int64),
        cloud=np.array([np.nan if cc is None else float(cc)], np.float64),
        crs=np.array(str(s2.get("crs") or "EPSG:4326")),
        scene_id=np.array(str(s2.get("scene_id") or "")),
        acquired=np.array(str(s2.get("acquired") or "")),
    )
    return buf.getvalue()


def _deserialize_s2(blob: bytes) -> dict:
    from rasterio.transform import Affine
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    wh, t = z["wh"], z["transform"]
    cloud = float(z["cloud"][0])
    sid, acq = str(z["scene_id"]), str(z["acquired"])
    return {
        "blue": z["blue"], "green": z["green"], "red": z["red"], "nir": z["nir"],
        "coastal": z["coastal"], "ndwi": z["ndwi"],
        "water_mask": z["water_mask"].astype(bool),
        "width": int(wh[0]), "height": int(wh[1]),
        "transform": Affine(*[float(x) for x in t]),
        "bounds": [float(x) for x in z["bounds"]], "crs": str(z["crs"]),
        "scene_id": sid or None,
        "cloud_cover": (None if np.isnan(cloud) else cloud),
        "acquired": acq or None,
    }


def _target_grid(bbox, res_m=10.0, max_px=1024):
    """Build an EPSG:4326 target grid (~res_m) for the bbox, capped at max_px."""
    from rasterio.transform import from_bounds
    w, s, e, n = bbox
    lat = (s + n) / 2.0
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * max(math.cos(math.radians(lat)), 0.1))
    width = int(round(abs(e - w) / (res_m * deg_per_m_lon)))
    height = int(round(abs(n - s) / (res_m * deg_per_m_lat)))
    # clamp
    width = max(16, min(max_px, width))
    height = max(16, min(max_px, height))
    transform = from_bounds(w, s, e, n, width, height)
    return width, height, transform


def _read_band(href, dst_crs, dst_transform, width, height, resampling):
    import rasterio
    from rasterio.vrt import WarpedVRT
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=dst_crs, transform=dst_transform,
                       width=width, height=height, resampling=resampling) as vrt:
            return vrt.read(1)


def _warp_item(item, bbox, max_px):
    """Warp one signed STAC item's bands onto the bbox target grid and build
    the s2 dict (bands, NDWI, water mask, georef). Shared by the single-scene
    and multi-scene fetch paths — output identical to the original inline code."""
    from rasterio.enums import Resampling

    w, s, e, n = bbox
    cloud = item.properties.get("eo:cloud_cover")
    width, height, transform = _target_grid(bbox, max_px=max_px)
    dst_crs = "EPSG:4326"

    # Collect the band-read tasks (skip a missing coastal/B01 — engine falls back
    # to blue; error on any other missing asset).
    read_tasks = []
    for key, asset_key in ASSETS.items():
        if asset_key not in item.assets:
            if key == "coastal":  # B01 fallback to blue handled by engine
                continue
            raise S2FetchError(f"Asset {asset_key} missing in scene {item.id}")
        href = item.assets[asset_key].href
        resamp = Resampling.nearest if key == "scl" else Resampling.bilinear
        read_tasks.append((key, href, resamp))

    # Warp the bands CONCURRENTLY. Each is an independent remote-COG read; every
    # thread opens its own dataset (GDAL-safe) and rasterio releases the GIL during
    # I/O, so the network-bound warps overlap instead of running back-to-back —
    # the dominant cost of a fresh fetch. Output is byte-for-byte the same.
    bands = {}
    max_workers = min(len(read_tasks), int(os.getenv("S2_FETCH_WORKERS", "6")))
    with _cf.ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futs = {pool.submit(_read_band, href, dst_crs, transform, width, height,
                            resamp): key
                for key, href, resamp in read_tasks}
        for fut in _cf.as_completed(futs):
            key = futs[fut]
            try:
                bands[key] = fut.result().astype(np.float32)
            except Exception as ex:
                raise S2FetchError(f"band {key} read failed for scene {item.id}: {ex}")

    green = bands["green"]
    nir = bands["nir"]
    eps = 1e-6
    ndwi = ((green - nir) / (green + nir + eps)).astype(np.float32)

    scl = bands.get("scl")
    if scl is not None:
        scl = scl.astype(np.int16)
        # SCL: 6=water, 2=dark, 4=veg, 5=not-vegetated/built, 3=cloud-shadow,
        #      8/9/10=cloud/cirrus, 11=snow. Water-only: SCL water is authoritative;
        #      also accept clearly-water NDWI, but NEVER over vegetation/built/cloud —
        #      this keeps the depth product off land/urban.
        # NB: keep the scalar `cloud` (eo:cloud_cover %) intact — use a distinct
        # name for the per-pixel cloud MASK so it isn't stored as cloud_cover.
        cloud_mask = np.isin(scl, [3, 8, 9, 10, 11])
        land = np.isin(scl, [4, 5])
        water = ((scl == 6) | (ndwi > 0.1)) & (~cloud_mask) & (~land)
    else:
        water = ndwi > 0.1
    water_mask = water.astype(bool)

    return {
        "blue": bands["blue"], "green": green, "red": bands["red"],
        "nir": nir, "coastal": bands.get("coastal", bands["blue"]),
        "ndwi": ndwi, "water_mask": water_mask,
        "width": width, "height": height,
        "transform": transform, "bounds": [w, s, e, n], "crs": dst_crs,
        "scene_id": item.id, "cloud_cover": cloud,
        "acquired": item.properties.get("datetime"),
    }


def fetch_s2(bbox, start_date="2023-01-01", end_date="2024-12-31",
             max_cloud=40, max_px=1024, use_cache=True):
    """Fetch a least-cloudy Sentinel-2 L2A scene for bbox=[W,S,E,N].

    Cached: the warped band stack for a given (area, dates, cloud, resolution)
    is stored in MinIO and reused on subsequent calls — so repeat surveys and
    re-inference after a model change never re-pull from Planetary Computer.
    """
    cache_client = None
    cache_obj = _cache_key(bbox, start_date, end_date, max_cloud, max_px)
    if use_cache:
        try:
            cache_client = _cache_client()
            if not cache_client.bucket_exists(S2_CACHE_BUCKET):
                cache_client.make_bucket(S2_CACHE_BUCKET)
            try:
                resp = cache_client.get_object(S2_CACHE_BUCKET, cache_obj)
                try:
                    blob = resp.read()
                finally:
                    resp.close()
                    resp.release_conn()
                log.info("S2 cache HIT %s (no re-pull)", cache_obj)
                return _deserialize_s2(blob)
            except Exception:
                log.info("S2 cache MISS %s — fetching from Planetary Computer", cache_obj)
        except Exception as ex:
            log.warning("S2 cache unavailable (%s) — fetching live", ex)
            cache_client = None

    try:
        import planetary_computer as pc
        import pystac_client
        from rasterio.enums import Resampling
    except Exception as ex:  # pragma: no cover
        raise S2FetchError(f"S2 deps unavailable: {ex}")

    w, s, e, n = bbox
    cat = pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)
    search = cat.search(
        collections=[COLLECTION],
        bbox=[w, s, e, n],
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=10,
    )
    items = list(search.items())
    if not items:
        raise S2FetchError(
            f"No Sentinel-2 scene with cloud<{max_cloud}% for bbox {bbox} "
            f"in {start_date}..{end_date}. Try a wider date range / cloud limit.")
    item = items[0]
    log.info("S2 scene %s cloud=%.1f%%",
             item.id, item.properties.get("eo:cloud_cover") or -1)

    s2 = _warp_item(item, bbox, max_px)

    # Store the warped band stack so this ROI never re-pulls (model-independent).
    if use_cache and cache_client is not None:
        try:
            blob = _serialize_s2(s2)
            cache_client.put_object(
                S2_CACHE_BUCKET, cache_obj, io.BytesIO(blob), length=len(blob),
                content_type="application/octet-stream")
            log.info("S2 cache STORE %s (%.1f MB)", cache_obj, len(blob) / 1e6)
        except Exception as ex:
            log.warning("S2 cache store failed: %s", ex)

    return s2


def fetch_s2_scenes(bbox, start_date="2023-01-01", end_date="2024-12-31",
                    max_cloud=40, n_scenes=3, max_px=1024, use_cache=True):
    """Fetch up to `n_scenes` distinct-date Sentinel-2 L2A scenes for
    bbox=[W,S,E,N], cloud-ascending. Same STAC search as fetch_s2; each scene's
    warped band stack is cached per (area, dates, cloud, resolution, scene_id).
    Returns a non-empty list of s2 dicts (raises S2FetchError otherwise)."""
    try:
        import planetary_computer as pc
        import pystac_client
    except Exception as ex:  # pragma: no cover
        raise S2FetchError(f"S2 deps unavailable: {ex}")

    n_scenes = max(1, int(n_scenes))
    w, s, e, n = bbox
    cat = pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)
    search = cat.search(
        collections=[COLLECTION],
        bbox=[w, s, e, n],
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=max(10, n_scenes * 5),
    )
    items = list(search.items())
    if not items:
        raise S2FetchError(
            f"No Sentinel-2 scene with cloud<{max_cloud}% for bbox {bbox} "
            f"in {start_date}..{end_date}. Try a wider date range / cloud limit.")

    picked, seen_dates = [], set()
    for item in items:
        day = str(item.properties.get("datetime") or item.id)[:10]
        if day in seen_dates:
            continue
        seen_dates.add(day)
        picked.append(item)
        if len(picked) >= n_scenes:
            break

    cache_client = None
    if use_cache:
        try:
            cache_client = _cache_client()
            if not cache_client.bucket_exists(S2_CACHE_BUCKET):
                cache_client.make_bucket(S2_CACHE_BUCKET)
        except Exception as ex:
            log.warning("S2 cache unavailable (%s) — fetching live", ex)
            cache_client = None

    scenes = []
    for item in picked:
        cache_obj = _cache_key(bbox, start_date, end_date, max_cloud, max_px,
                               scene_id=item.id)
        if cache_client is not None:
            try:
                resp = cache_client.get_object(S2_CACHE_BUCKET, cache_obj)
                try:
                    blob = resp.read()
                finally:
                    resp.close()
                    resp.release_conn()
                log.info("S2 cache HIT %s (%s)", cache_obj, item.id)
                scenes.append(_deserialize_s2(blob))
                continue
            except Exception:
                pass
        try:
            s2 = _warp_item(item, bbox, max_px)
        except S2FetchError as ex:
            log.warning("scene %s skipped: %s", item.id, ex)
            continue
        scenes.append(s2)
        if cache_client is not None:
            try:
                blob = _serialize_s2(s2)
                cache_client.put_object(
                    S2_CACHE_BUCKET, cache_obj, io.BytesIO(blob), length=len(blob),
                    content_type="application/octet-stream")
            except Exception as ex:
                log.warning("S2 cache store failed: %s", ex)

    if not scenes:
        raise S2FetchError("no usable Sentinel-2 scene could be read for the ROI")
    return scenes
