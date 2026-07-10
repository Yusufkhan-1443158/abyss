"""Raster management and HTJ2K streaming endpoints."""

from __future__ import annotations

import asyncio
import io
import math
import os
import uuid
from datetime import datetime, date
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response
from geoalchemy2.functions import ST_AsGeoJSON, ST_Intersects, ST_MakeEnvelope
from osgeo import gdal
from PIL import Image
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.area_check import filter_by_area, verify_area_access
from ..middleware.audit_logger import log_action
from ..middleware.auth_middleware import get_current_user, require_analyst_or_admin
from ..models import (
    RasterCatalog,
    RasterTimeseries,
    RasterTimeseriesEntry,
    User,
)
from ..schemas import (
    RasterListResponse,
    RasterResponse,
    RasterStatusResponse,
    RasterTimeseriesCreate,
    RasterTimeseriesResponse,
    RasterUploadResponse,
)
from ..services import minio_storage, tile_cache
from ..services.tasks import process_raster_upload

# ---------------------------------------------------------------------------
# Tile generation helpers
# ---------------------------------------------------------------------------

# Suppress GDAL error output (use exceptions instead)
gdal.UseExceptions()

TILE_SIZE = 256
RASTER_CACHE_DIR = "/tmp/raster_cache"
os.makedirs(RASTER_CACHE_DIR, exist_ok=True)

# Caps concurrent GDAL renders so a tile burst (e.g. browser hard-refresh on
# the satellite layer) can't drain the worker pool that auth/login/api share.
# 9th request queues at the asyncio layer; no thread consumed.
TILE_RENDER_SEMAPHORE = asyncio.Semaphore(8)

# 1x1 transparent PNG (for empty tiles)
_EMPTY_TILE: bytes | None = None


def _get_empty_tile() -> bytes:
    global _EMPTY_TILE
    if _EMPTY_TILE is None:
        img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        _EMPTY_TILE = buf.getvalue()
    return _EMPTY_TILE


def _tile_response_from_cache(data: bytes) -> Response:
    """Serve a cached tile (migration Step 2 read-through). The 1-byte negative
    sentinel is rendered back as the transparent empty PNG."""
    if data == tile_cache.NEGATIVE_SENTINEL:
        return Response(
            content=_get_empty_tile(), media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600", "X-SW-Cache": "hit-empty"},
        )
    return Response(
        content=data, media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600", "X-SW-Cache": "hit"},
    )


async def _cache_get(key: str) -> bytes | None:
    """L1 inline (memory, no I/O) then L2 via asyncio.to_thread, so a slow/hung
    dedicated redis never blocks the event loop. Skips the thread hop entirely in
    the common L1-only deployment (has_l2() is False)."""
    v = tile_cache.l1_get(key)
    if v is None and tile_cache.has_l2():
        v = await asyncio.to_thread(tile_cache.l2_get, key)
    return v


async def _cache_store(key: str, data: bytes) -> None:
    """L1 store inline; the L2 (network) write runs off the event loop."""
    tile_cache.l1_put(key, data)
    if tile_cache.has_l2():
        await asyncio.to_thread(tile_cache.l2_put, key, data)


def _tile_to_bbox_4326(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert XYZ tile coords to EPSG:4326 (lon_min, lat_min, lon_max, lat_max)."""
    n = 2 ** z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return (lon_min, lat_min, lon_max, lat_max)


_ORIGIN_SHIFT = 20037508.342789244  # half circumference of Earth in meters


def _tile_to_bbox_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert XYZ tile coords to EPSG:3857 Web Mercator (x_min, y_min, x_max, y_max)."""
    resolution = 2 * math.pi * 6378137 / (2 ** z * TILE_SIZE)
    x_min = x * TILE_SIZE * resolution - _ORIGIN_SHIFT
    x_max = (x + 1) * TILE_SIZE * resolution - _ORIGIN_SHIFT
    y_min = _ORIGIN_SHIFT - (y + 1) * TILE_SIZE * resolution
    y_max = _ORIGIN_SHIFT - y * TILE_SIZE * resolution
    return (x_min, y_min, x_max, y_max)


_CACHE_MAX_BYTES = 50 * 1024 * 1024 * 1024  # 50 GB
_CACHE_TARGET_BYTES = 40 * 1024 * 1024 * 1024  # 40 GB (evict down to this)


def _evict_cache() -> None:
    """Delete oldest cached rasters until total size is under _CACHE_TARGET_BYTES."""
    try:
        entries = []
        total_size = 0
        for name in os.listdir(RASTER_CACHE_DIR):
            path = os.path.join(RASTER_CACHE_DIR, name)
            if not os.path.isfile(path) or name.endswith(".tmp"):
                continue
            stat = os.stat(path)
            entries.append((stat.st_atime, stat.st_size, path))
            total_size += stat.st_size

        if total_size <= _CACHE_MAX_BYTES:
            return

        # Sort by access time ascending (oldest first)
        entries.sort()
        for _atime, size, path in entries:
            if total_size <= _CACHE_TARGET_BYTES:
                break
            try:
                os.unlink(path)
                total_size -= size
            except OSError:
                pass
    except OSError:
        pass


def _get_cached_raster(raster_id: str, minio_raw_path: str) -> str:
    """Return a local COG for the raster, fetching/building as cheaply as possible.

    Fast paths (instant-load):
      1. Already in the shared tile cache (ingest Stage A writes it here, so the
         very first tile after a raster goes 'ready' is already warm).
      2. A prebuilt COG persisted to the `cog` bucket (ingest Stage B / other
         workers) — a plain download, no overview build.
    Slow fallback (legacy rows, or COG not yet persisted): build the COG from the
    raw file the old way.
    """
    cached = os.path.join(RASTER_CACHE_DIR, f"{raster_id}.tif")
    if os.path.exists(cached):
        return cached

    # Fast path 2: prebuilt COG from the cog bucket (conventional key).
    cog_key = f"{raster_id}/{raster_id}.cog.tif"
    try:
        cog_data = minio_storage.download_file("cog", cog_key)
        tmp = cached + ".dl.tmp"
        with open(tmp, "wb") as f:
            f.write(cog_data)
        os.replace(tmp, cached)
        _evict_cache()
        return cached
    except Exception:
        pass  # not persisted yet — fall through to build from raw

    raw_data = minio_storage.download_file("raw", minio_raw_path)

    # Write raw file first
    raw_path = cached + ".raw.tmp"
    with open(raw_path, "wb") as f:
        f.write(raw_data)

    # Build COG with internal overviews for fast random tile access
    cog_path = cached + ".cog.tmp"
    try:
        # First build overviews on the raw file
        ds = gdal.Open(raw_path, gdal.GA_Update)
        if ds is not None:
            # Build 2x, 4x, 8x, 16x overviews using average resampling
            overview_levels = []
            min_dim = min(ds.RasterXSize, ds.RasterYSize)
            level = 2
            while min_dim // level >= TILE_SIZE:
                overview_levels.append(level)
                level *= 2
            if overview_levels:
                ds.BuildOverviews("AVERAGE", overview_levels)
            ds = None

        # Convert to COG (Cloud-Optimized GeoTIFF)
        gdal.Translate(
            cog_path,
            raw_path,
            format="COG",
            creationOptions=[
                "COMPRESS=DEFLATE",
                "BLOCKSIZE=256",
                "OVERVIEW_RESAMPLING=AVERAGE",
                "OVERVIEWS=IGNORE_EXISTING",
            ],
        )
        os.replace(cog_path, cached)
    except Exception:
        # Fall back to raw file if COG conversion fails
        os.replace(raw_path, cached)
    finally:
        for tmp in (raw_path, cog_path):
            if os.path.exists(tmp):
                os.unlink(tmp)

    # Evict old cache entries if total size exceeds limit
    _evict_cache()

    return cached


def _render_tile(
    raster_paths: list[tuple[str, list[dict] | None]],
    bbox_3857: tuple[float, float, float, float],
) -> np.ndarray:
    """Render a 256x256 RGBA tile from one or more raster files.

    bbox_3857 is (x_min, y_min, x_max, y_max) in EPSG:3857 Web Mercator meters.
    """
    x_min, y_min, x_max, y_max = bbox_3857
    result = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)

    for path, band_stats in raster_paths:
        try:
            # Read source nodata value for this file
            src_ds = gdal.Open(path)
            src_nodata = None
            if src_ds is not None:
                src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
                src_ds = None

            warp_kwargs: dict = dict(
                format="MEM",
                outputBounds=[x_min, y_min, x_max, y_max],
                outputBoundsSRS="EPSG:3857",
                width=TILE_SIZE,
                height=TILE_SIZE,
                dstSRS="EPSG:3857",
                resampleAlg="bilinear",
                dstAlpha=True,
            )
            if src_nodata is not None:
                warp_kwargs["srcNodata"] = src_nodata

            ds = gdal.Warp("", path, **warp_kwargs)
        except Exception:
            continue

        if ds is None or ds.RasterXSize == 0:
            continue

        data = ds.ReadAsArray()
        if data is None:
            ds = None
            continue

        # Last band is the GDAL-generated alpha band (from dstAlpha=True)
        if data.ndim == 3 and data.shape[0] > 1:
            alpha_band = data[-1]
            data = data[:-1]  # strip the alpha band from data bands
        else:
            alpha_band = None

        bands = data.shape[0] if data.ndim == 3 else 1

        # Build alpha mask using GDAL's alpha band (accurate nodata boundary)
        if alpha_band is not None:
            has_data = alpha_band > 0
        elif data.ndim == 3:
            has_data = np.any(data != 0, axis=0)
        else:
            has_data = data != 0

        # Dynamic range stretch per band
        for b in range(min(bands, 3)):
            band_data = data[b] if data.ndim == 3 else data
            if band_stats and b < len(band_stats):
                lo = float(band_stats[b].get("p2_5", 0))
                hi = float(band_stats[b].get("p97_5", 255))
            else:
                lo = float(band_data.min())
                hi = float(band_data.max())
            if hi <= lo:
                hi = lo + 1.0
            stretched = np.clip(
                (band_data.astype(np.float32) - lo) / (hi - lo) * 255.0,
                0, 255,
            ).astype(np.uint8)
            # Composite: new data overwrites where it has data
            result[:, :, b] = np.where(has_data, stretched, result[:, :, b])

        # If single band, copy to G and B for grayscale
        if bands == 1:
            result[:, :, 1] = np.where(has_data, result[:, :, 0], result[:, :, 1])
            result[:, :, 2] = np.where(has_data, result[:, :, 0], result[:, :, 2])

        result[:, :, 3] = np.where(has_data, 255, result[:, :, 3])
        ds = None

    return result

router = APIRouter()

# ---------------------------------------------------------------------------
# Allowed raster magic bytes (first few bytes per format)
# ---------------------------------------------------------------------------
_RASTER_MAGIC: dict[bytes, str] = {
    b"II\x2a\x00": "geotiff",     # TIFF little-endian
    b"MM\x00\x2a": "geotiff",     # TIFF big-endian
    b"II\x2b\x00": "geotiff",     # BigTIFF little-endian
    b"MM\x00\x2b": "geotiff",     # BigTIFF big-endian
    b"\x00\x00\x00\x0c\x6a\x50": "jp2",  # JPEG2000 signature box
    b"\xff\x4f\xff\x51": "jp2",   # JPEG2000 codestream
    b"\x89HDF\r\n\x1a\n": "hdf5", # HDF5 (also NetCDF4)
    b"\x0e\x03\x13\x01": "hdf4",  # HDF4
    b"CDF\x01": "netcdf",         # NetCDF3 classic
    b"CDF\x02": "netcdf",         # NetCDF3 64-bit offset
    b"NITF02.10": "nitf",         # NITF 2.10 (post-1998)
    b"NITF02.00": "nitf",         # NITF 2.00
    b"NSIF01.00": "nitf",         # NATO Secondary Imagery Format (NITF variant)
}

_ALLOWED_EXTENSIONS = {
    ".tif", ".tiff", ".geotiff",          # GeoTIFF / COG
    ".jp2", ".j2k", ".jph",                # JPEG2000 / HTJ2K
    ".ecw", ".sid",                         # ECW, MrSID
    ".ntf", ".nitf", ".nsif",               # NITF / NSIF
    ".h5", ".hdf5", ".hdf", ".he5",         # HDF5 / HDF4
    ".nc", ".nc4", ".cdf",                  # NetCDF
    ".img",                                 # ERDAS IMAGINE
}


def _render_specs_to_png(
    specs: list[tuple[str, str, list[dict] | None]],
    bbox_3857: tuple[float, float, float, float],
) -> bytes | None:
    """Sync helper that runs the entire blocking render pipeline (MinIO fetch
    + GDAL warp + PIL encode) in one thread. Returns the encoded PNG bytes,
    or None if the rendered tile is fully transparent.

    Called via asyncio.to_thread() so the FastAPI event loop stays free for
    auth/api work even when 50 tiles are rendering in parallel.
    """
    raster_paths: list[tuple[str, list[dict] | None]] = []
    for raster_id, minio_raw_path, band_stats in specs:
        local_path = _get_cached_raster(raster_id, minio_raw_path)
        raster_paths.append((local_path, band_stats))

    tile_array = _render_tile(raster_paths, bbox_3857)
    if not np.any(tile_array[:, :, 3]):
        return None

    img = Image.fromarray(tile_array, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gdal_can_open(header: bytes, filename: str) -> str | None:
    """Last-resort probe: write header to tmpfile and ask GDAL to identify it.

    Returns the GDAL driver short name lowercased if openable, else None.
    """
    import tempfile
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(header)
            tmp_path = tmp.name
        try:
            drv = gdal.IdentifyDriverEx(tmp_path, gdalDriverType=gdal.OF_RASTER)
            if drv is not None:
                return drv.ShortName.lower()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        return None
    return None


def _detect_raster_format(header: bytes, filename: str) -> str:
    """Validate raster format from magic bytes, extension, then GDAL probe."""
    for sig, fmt in _RASTER_MAGIC.items():
        if header[: len(sig)] == sig:
            return fmt

    # ECW signature: "ecs" or ERDAS specific
    if header[:3] == b"ecs" or header[:4] == b"\x00\x00\x40\x03":
        return "ecw"

    # MrSID: starts with "msid"
    if header[:4] == b"msid" or header[:2] == b"\x00\x00":
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext == "sid":
            return "mrsid"

    # Extension allowlist
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in _ALLOWED_EXTENSIONS:
        return ext.lstrip(".")

    # Final fallback: let GDAL sniff the header. Accepts any raster driver GDAL
    # knows (covers vendor variants, ENVI, PCIDSK, BAG, etc.) while still
    # rejecting non-raster files.
    drv = _gdal_can_open(header, filename)
    if drv:
        return drv

    raise ValueError(f"Unsupported raster format: {filename}")


def _raster_to_response(raster: RasterCatalog, db: Session) -> dict[str, Any]:
    """Convert a RasterCatalog ORM row to a response dict with GeoJSON bbox."""
    data = {
        "id": str(raster.id),
        "name": raster.name,
        "description": raster.description,
        "original_filename": raster.original_filename,
        "original_format": raster.original_format,
        "converted_format": raster.converted_format,
        "file_size_bytes": raster.file_size_bytes,
        "width_px": raster.width_px,
        "height_px": raster.height_px,
        "num_bands": raster.num_bands,
        "bit_depth": raster.bit_depth,
        "crs": raster.crs,
        "resolution_m": raster.resolution_m,
        "acquisition_date": raster.acquisition_date,
        "min_zoom": raster.min_zoom,
        "max_zoom": raster.max_zoom,
        "processing_status": raster.processing_status,
        "processing_error": raster.processing_error,
        "uploaded_by": str(raster.uploaded_by) if raster.uploaded_by else None,
        "upload_date": raster.upload_date,
        "metadata": raster.metadata_ or {},
        "tags": raster.tags or [],
        "is_public": raster.is_public,
        "created_at": raster.created_at,
        "updated_at": raster.updated_at,
    }

    # Convert PostGIS geometry to GeoJSON dict
    if raster.bbox is not None:
        import json
        bbox_json = db.scalar(ST_AsGeoJSON(raster.bbox))
        data["bbox"] = json.loads(bbox_json) if bbox_json else None
    else:
        data["bbox"] = None

    if raster.center_point is not None:
        import json
        cp_json = db.scalar(ST_AsGeoJSON(raster.center_point))
        data["center_point"] = json.loads(cp_json) if cp_json else None
    else:
        data["center_point"] = None

    return data


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=RasterUploadResponse, status_code=201)
async def upload_raster(
    file: UploadFile = File(...),
    name: str = Query(None),
    description: str = Query(None),
    acquisition_date: datetime | None = Query(None),
    tags: list[str] = Query(None),
    current_user: User = Depends(require_analyst_or_admin),
    db: Session = Depends(get_db),
):
    """Upload a raster file. Validates format via magic bytes. Dispatches processing."""
    # Read header for format validation
    header = await file.read(32)
    await file.seek(0)

    try:
        fmt = _detect_raster_format(header, file.filename or "unknown")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check file size (10 GB max)
    from ..config import get_settings
    settings = get_settings()
    # We can't know full size from UploadFile easily without reading; rely on
    # content-length header or streaming. For now, proceed and MinIO will handle.

    raster_id = str(uuid.uuid4())
    filename = file.filename or f"{raster_id}.{fmt}"
    minio_raw_path = f"{raster_id}/{filename}"

    # Upload raw to MinIO
    minio_storage.upload_file(
        "raw",
        minio_raw_path,
        file.file,
        content_type=file.content_type or "application/octet-stream",
        length=-1,
    )

    # Get file size from MinIO after upload
    file_size = minio_storage.get_file_size("raw", minio_raw_path)

    # Create catalog row
    raster = RasterCatalog(
        id=raster_id,
        name=name or filename,
        description=description,
        original_filename=filename,
        original_format=fmt,
        file_size_bytes=file_size,
        acquisition_date=acquisition_date,
        tags=tags or [],
        minio_raw_path=minio_raw_path,
        processing_status="pending",
        uploaded_by=current_user.id,
    )
    db.add(raster)
    db.commit()
    db.refresh(raster)

    # Dispatch Celery task
    process_raster_upload.delay(raster_id)

    return RasterUploadResponse(
        id=raster_id,
        name=raster.name,
        processing_status="pending",
    )


_MAIN_RASTER_EXTS = (
    ".tif", ".tiff", ".geotiff",
    ".jp2", ".j2k", ".jph",
    ".ecw", ".sid",
    ".ntf", ".nitf", ".nsif",
    ".h5", ".hdf5", ".hdf", ".he5",
    ".nc", ".nc4", ".cdf",
    ".img",
)


def _safe_relpath(raw: str) -> str:
    """Normalize a relative path and reject traversal/absolute components."""
    rel = (raw or "").replace("\\", "/").lstrip("/")
    rel = os.path.normpath(rel).replace("\\", "/")
    if rel.startswith("..") or rel.startswith("/") or ":" in rel:
        raise HTTPException(status_code=400, detail=f"Invalid path in bundle: {raw!r}")
    return rel


@router.post("/upload-bundle", response_model=RasterUploadResponse, status_code=201)
async def upload_raster_bundle(
    request: Request,
    files: list[UploadFile] = File(default_factory=list),
    name: str = Query(None),
    description: str = Query(None),
    acquisition_date: datetime | None = Query(None),
    tags: list[str] = Query(None),
    current_user: User = Depends(require_analyst_or_admin),
    db: Session = Depends(get_db),
):
    """Upload a raster *bundle*: either a directory (multiple files with
    relative paths via ``webkitdirectory``) or a single ``.zip`` archive.

    Sidecars (``.j2w``, ``.aux.xml``, ``.RPB``, ``.IMD``, ``.XML``, ...) are
    preserved adjacent to the main raster so GDAL reads them natively, and
    vendor-specific parsers extract normalized metadata.
    """
    import tempfile
    import zipfile

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    raster_id = str(uuid.uuid4())
    tmp_root = tempfile.mkdtemp(prefix=f"bundle_{raster_id}_")

    try:
        # 1. Materialize every part under tmp_root. One .zip expands in-place.
        if len(files) == 1 and (files[0].filename or "").lower().endswith(".zip"):
            zip_path = os.path.join(tmp_root, files[0].filename)
            with open(zip_path, "wb") as fh:
                while chunk := await files[0].read(1024 * 1024):
                    fh.write(chunk)
            with zipfile.ZipFile(zip_path) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    rel = _safe_relpath(info.filename)
                    dest = os.path.join(tmp_root, rel)
                    os.makedirs(os.path.dirname(dest) or tmp_root, exist_ok=True)
                    with zf.open(info) as src, open(dest, "wb") as out:
                        while chunk := src.read(1024 * 1024):
                            out.write(chunk)
            os.remove(zip_path)
        else:
            for idx, uf in enumerate(files):
                rel_raw = getattr(uf, "filename", None) or f"file_{idx}"
                rel = _safe_relpath(rel_raw)
                dest = os.path.join(tmp_root, rel)
                os.makedirs(os.path.dirname(dest) or tmp_root, exist_ok=True)
                with open(dest, "wb") as out:
                    while chunk := await uf.read(1024 * 1024):
                        out.write(chunk)

        # 2. Identify the main raster inside the bundle.
        candidates: list[str] = []
        for root, _, fnames in os.walk(tmp_root):
            for f in fnames:
                if f.lower().endswith(_MAIN_RASTER_EXTS):
                    candidates.append(os.path.relpath(os.path.join(root, f), tmp_root))
        if not candidates:
            raise HTTPException(
                status_code=400,
                detail=f"No main raster found. Expected one of {_MAIN_RASTER_EXTS}",
            )
        if len(candidates) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple main rasters found in bundle: {candidates}. "
                        "Upload one raster bundle at a time.",
            )
        main_rel = candidates[0]
        main_path = os.path.join(tmp_root, main_rel)

        # 3. Validate the main raster's magic bytes (reuse single-upload check).
        with open(main_path, "rb") as fh:
            header = fh.read(32)
        try:
            fmt = _detect_raster_format(header, os.path.basename(main_rel))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # 4. Upload every file under {raster_id}/bundle/<relpath>.
        bundle_prefix = f"{raster_id}/bundle/"
        total_size = 0
        for root, _, fnames in os.walk(tmp_root):
            for f in fnames:
                local = os.path.join(root, f)
                rel = os.path.relpath(local, tmp_root).replace(os.sep, "/")
                size = os.path.getsize(local)
                total_size += size
                with open(local, "rb") as fh:
                    minio_storage.upload_file(
                        "raw", bundle_prefix + rel, fh,
                        content_type="application/octet-stream",
                        length=size,
                    )

        main_minio_path = bundle_prefix + main_rel.replace(os.sep, "/")

        # 5. Catalog row.
        raster = RasterCatalog(
            id=raster_id,
            name=name or os.path.basename(main_rel),
            description=description,
            original_filename=os.path.basename(main_rel),
            original_format=fmt,
            file_size_bytes=total_size,
            acquisition_date=acquisition_date,
            tags=tags or [],
            minio_raw_path=main_minio_path,
            processing_status="pending",
            uploaded_by=current_user.id,
        )
        db.add(raster)
        db.commit()
        db.refresh(raster)

        process_raster_upload.delay(raster_id)

        return RasterUploadResponse(
            id=raster_id,
            name=raster.name,
            processing_status="pending",
        )
    finally:
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# List rasters
# ---------------------------------------------------------------------------

@router.get("", response_model=RasterListResponse)
async def list_rasters(
    response: Response,
    bbox: str | None = Query(None, description="minx,miny,maxx,maxy"),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    tags: list[str] | None = Query(None),
    processing_status: str | None = Query(None),
    q: str | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    if_none_match: str | None = Header(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List rasters with filtering, spatial queries, and area access control."""
    query = db.query(RasterCatalog)

    # Spatial filter
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            minx, miny, maxx, maxy = parts
        except (ValueError, IndexError):
            raise HTTPException(status_code=400, detail="bbox must be minx,miny,maxx,maxy")
        envelope = ST_MakeEnvelope(minx, miny, maxx, maxy, 4326)
        query = query.filter(ST_Intersects(RasterCatalog.bbox, envelope))

    # Temporal filter
    if date_from:
        query = query.filter(RasterCatalog.acquisition_date >= date_from)
    if date_to:
        query = query.filter(RasterCatalog.acquisition_date <= date_to)

    # Tag filter (array overlap)
    if tags:
        query = query.filter(RasterCatalog.tags.overlap(tags))

    # Status filter
    if processing_status:
        query = query.filter(RasterCatalog.processing_status == processing_status)

    # Text search on name/description
    if q:
        pattern = f"%{q}%"
        query = query.filter(
            or_(
                RasterCatalog.name.ilike(pattern),
                RasterCatalog.description.ilike(pattern),
            )
        )

    # Area access filtering
    query = filter_by_area(query, current_user, RasterCatalog.bbox, db)

    total = query.count()

    # ETag: collapses (total, latest mtime, page window) into a weak validator.
    # Client (IndexedDB cache) sends If-None-Match on revisits; matching state -> 304.
    max_updated = query.with_entities(func.max(RasterCatalog.updated_at)).scalar()
    max_updated_iso = max_updated.isoformat() if max_updated else ""
    etag = f'W/"{total}-{max_updated_iso}-{offset}-{limit}"'
    if if_none_match == etag:
        return Response(
            status_code=304,
            headers={"ETag": etag, "Cache-Control": "private, max-age=0, must-revalidate"},
        )

    rasters = query.order_by(RasterCatalog.created_at.desc()).offset(offset).limit(limit).all()

    items = [_raster_to_response(r, db) for r in rasters]

    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=0, must-revalidate"
    return RasterListResponse(total=total, offset=offset, limit=limit, items=items)


# ---------------------------------------------------------------------------
# Timeseries endpoints (MUST be before /{raster_id} to avoid route capture)
# ---------------------------------------------------------------------------

@router.get("/timeseries", response_model=list[RasterTimeseriesResponse])
async def list_timeseries(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all time series. Paginated."""
    series = (
        db.query(RasterTimeseries)
        .order_by(RasterTimeseries.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    result = []
    for ts in series:
        result.append(_timeseries_to_response(ts, db))
    return result


@router.post("/timeseries", response_model=RasterTimeseriesResponse, status_code=201)
async def create_timeseries(
    body: RasterTimeseriesCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new time series group."""
    ts = RasterTimeseries(
        name=body.name,
        description=body.description,
        created_by=current_user.id,
    )

    if body.area_bbox:
        from geoalchemy2.elements import WKTElement
        import json
        coords = body.area_bbox.get("coordinates", [[]])[0]
        if coords:
            wkt_ring = ", ".join(f"{c[0]} {c[1]}" for c in coords)
            ts.area_bbox = WKTElement(f"SRID=4326;POLYGON(({wkt_ring}))", srid=4326)

    db.add(ts)
    db.commit()
    db.refresh(ts)

    return _timeseries_to_response(ts, db)


@router.get("/timeseries/{ts_id}", response_model=RasterTimeseriesResponse)
async def get_timeseries(
    ts_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get time series with ordered raster entries."""
    ts = db.query(RasterTimeseries).filter(RasterTimeseries.id == ts_id).first()
    if not ts:
        raise HTTPException(status_code=404, detail="Time series not found")

    return _timeseries_to_response(ts, db)


@router.patch("/timeseries/{ts_id}", response_model=RasterTimeseriesResponse)
async def update_timeseries(
    ts_id: str,
    name: str | None = None,
    description: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update time series name or description."""
    ts = db.query(RasterTimeseries).filter(RasterTimeseries.id == ts_id).first()
    if not ts:
        raise HTTPException(status_code=404, detail="Time series not found")

    if name is not None:
        ts.name = name
    if description is not None:
        ts.description = description

    db.commit()
    db.refresh(ts)
    return _timeseries_to_response(ts, db)


@router.delete("/timeseries/{ts_id}", status_code=204)
async def delete_timeseries(
    ts_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete time series (CASCADE deletes entries, not the rasters themselves)."""
    ts = db.query(RasterTimeseries).filter(RasterTimeseries.id == ts_id).first()
    if not ts:
        raise HTTPException(status_code=404, detail="Time series not found")

    db.delete(ts)
    db.commit()
    return Response(status_code=204)


@router.post("/timeseries/{ts_id}/entries", status_code=201)
async def add_timeseries_entry(
    ts_id: str,
    raster_id: str = Query(...),
    sort_order: int = Query(0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a raster to a time series."""
    ts = db.query(RasterTimeseries).filter(RasterTimeseries.id == ts_id).first()
    if not ts:
        raise HTTPException(status_code=404, detail="Time series not found")

    raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
    if not raster:
        raise HTTPException(status_code=404, detail="Raster not found")

    # Check if entry already exists
    existing = (
        db.query(RasterTimeseriesEntry)
        .filter(
            RasterTimeseriesEntry.timeseries_id == ts_id,
            RasterTimeseriesEntry.raster_id == raster_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="Raster already in time series")

    entry = RasterTimeseriesEntry(
        timeseries_id=ts_id,
        raster_id=raster_id,
        sort_order=sort_order,
    )
    db.add(entry)
    db.commit()

    return {"detail": "Entry added"}


@router.delete("/timeseries/{ts_id}/entries/{raster_id}", status_code=204)
async def remove_timeseries_entry(
    ts_id: str,
    raster_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a raster from a time series."""
    entry = (
        db.query(RasterTimeseriesEntry)
        .filter(
            RasterTimeseriesEntry.timeseries_id == ts_id,
            RasterTimeseriesEntry.raster_id == raster_id,
        )
        .first()
    )
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    db.delete(entry)
    db.commit()
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Metadata (distinct acquisition dates for time slider)
# ---------------------------------------------------------------------------

@router.get("/metadata")
async def raster_metadata(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return distinct acquisition dates for all rasters (for time slider)."""
    rows = (
        db.query(func.distinct(func.date(RasterCatalog.acquisition_date)))
        .filter(RasterCatalog.acquisition_date.isnot(None))
        .order_by(func.date(RasterCatalog.acquisition_date))
        .all()
    )
    dates = [row[0].isoformat() for row in rows if row[0] is not None]
    return {"dates": dates}


# ---------------------------------------------------------------------------
# Cheap viewport probe — drives global-browse satellite gating + footprints.
# Declared before /{raster_id} so the param matcher doesn't capture "in-bbox".
# ---------------------------------------------------------------------------

@router.get("/in-bbox")
async def rasters_in_bbox(
    bbox: str = Query(..., description="minx,miny,maxx,maxy in EPSG:4326"),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List ready rasters whose bbox intersects the given viewport.

    Returns id + acquisition_date + footprint GeoJSON for each (capped at 500).
    Cheap: GIST envelope on idx_raster_bbox, no row hydration. Used by the
    frontend to (a) decide whether to mount the satellite tile source,
    (b) draw footprint outlines so the user knows where imagery exists,
    (c) populate the time scrubber's per-viewport date list.
    """
    import json as _json
    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        minx, miny, maxx, maxy = parts
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="bbox must be minx,miny,maxx,maxy")

    envelope = ST_MakeEnvelope(minx, miny, maxx, maxy, 4326)
    q = db.query(
        RasterCatalog.id,
        RasterCatalog.acquisition_date,
        ST_AsGeoJSON(RasterCatalog.bbox).label("footprint"),
    ).filter(
        RasterCatalog.processing_status == "ready",
        RasterCatalog.bbox.isnot(None),
        ST_Intersects(RasterCatalog.bbox, envelope),
    )
    if date_from:
        q = q.filter(RasterCatalog.acquisition_date >= date_from)
    if date_to:
        q = q.filter(RasterCatalog.acquisition_date <= date_to)
    q = filter_by_area(q, current_user, RasterCatalog.bbox, db)
    rows = q.order_by(RasterCatalog.acquisition_date.desc().nullslast()).limit(500).all()

    return {
        "count": len(rows),
        "ids": [str(r.id) for r in rows],
        "dates": sorted({
            r.acquisition_date.date().isoformat()
            for r in rows if r.acquisition_date
        }),
        "footprints": [_json.loads(r.footprint) for r in rows],
    }


# ---------------------------------------------------------------------------
# Tile hot-layer cache stats (migration Step 2 observability)
# Declared BEFORE /{raster_id} so the param matcher doesn't capture
# "_cache_stats" (same ordering discipline as /in-bbox and /tiles above). The
# leading underscore also can't collide with a raster UUID.
# ---------------------------------------------------------------------------

@router.get("/_cache_stats")
async def tile_cache_stats(
    current_user: User = Depends(require_analyst_or_admin),
) -> dict[str, Any]:
    """Per-worker L1 + shared-L2 counters for the read-through tile cache.

    Analyst-or-admin gated; returns only aggregate counters (no secrets/IDs).
    Useful for platform testing: confirms whether the cache is enabled, whether
    L2 (dedicated redis) is wired, and the live hit-rate. NOTE: L1 is per uvicorn
    worker, so hit-rate reflects this worker only; L2 counters are shared.
    """
    return tile_cache.stats()


# ---------------------------------------------------------------------------
# Super-resolution upscale (DGX Meruem service)
# ---------------------------------------------------------------------------
# Declared BEFORE /tiles and /{raster_id} so the path matcher resolves
# `{raster_id}/upscale` correctly. (FastAPI evaluates literal segments
# before parameterized ones within a route group, but explicit ordering
# in the file is documented practice in this codebase — see the in-bbox
# comment above.)

MERUEM_URL = os.getenv("MERUEM_URL", "http://128.100.100.10:8000")
MERUEM_TIMEOUT_S = int(os.getenv("MERUEM_TIMEOUT_S", "300"))


@router.post("/{raster_id}/upscale", status_code=202)
async def upscale_raster(
    raster_id: uuid.UUID,
    current_user: Annotated[User, Depends(require_analyst_or_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Queue a super-resolution job on the DGX Meruem service.

    Returns 202 + {job_id, external_job_id, status}. The client polls
    /api/jobs/{job_id}/status. When Meruem completes, the jobs router
    promotes the result into a NEW raster row whose parent_raster_id
    points back at this source — that new row appears in collections-browse
    automatically on the next loadApiData() refresh.

    This endpoint is intentionally NON-BLOCKING — we POST to Meruem and
    immediately persist the external job id, returning to the user without
    waiting for the upscale to finish (Meruem jobs run 30–60+ seconds and
    blocking the FastAPI event loop on them would defeat the L2 semaphore
    isolation we ship for tile rendering).
    """
    import httpx
    from sqlalchemy import text as _sql

    # 1. Source raster must exist and be readable by this user
    src = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).one_or_none()
    if not src:
        raise HTTPException(status_code=404, detail="raster not found")
    verify_area_access(src, current_user)

    if not src.minio_key:
        raise HTTPException(status_code=409, detail="source raster has no MinIO object")

    # 2. Pull the source bytes from MinIO. We could stream multipart directly
    #    from MinIO to Meruem with httpx.stream(), but the simpler in-memory
    #    download keeps this endpoint readable; if memory becomes a concern
    #    for large COGs (>500 MB) we switch to streamed multipart later.
    try:
        src_bytes = await asyncio.to_thread(minio_storage.download_bytes, src.minio_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MinIO read failed: {e}")

    # 3. POST to Meruem /process (multipart `file=`). Meruem returns a job id.
    #    On Meruem unreachable, fail fast — the user sees an immediate toast.
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{MERUEM_URL}/process",
                files={"file": (src.original_filename or f"{raster_id}.tif", src_bytes, "image/tiff")},
            )
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Meruem returned {resp.status_code}: {resp.text[:200]}")
        meruem_payload = resp.json()
        external_job_id = meruem_payload.get("job_id") or meruem_payload.get("id")
        if not external_job_id:
            raise HTTPException(status_code=502, detail="Meruem response missing job_id")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Meruem unreachable: {e}")

    # 4. Persist a sr_jobs row so /api/jobs/{job_id}/status can poll
    #    Meruem on the user's behalf and promote the result when done.
    job_id = uuid.uuid4()
    db.execute(
        _sql("""
            INSERT INTO sr_jobs (id, raster_id, external_job_id, status, requested_by, requested_at, metadata)
            VALUES (:id, :raster_id, :external_job_id, 'queued', :requested_by, NOW(),
                    CAST(:metadata AS JSONB))
        """),
        {
            "id": str(job_id),
            "raster_id": str(raster_id),
            "external_job_id": str(external_job_id),
            "requested_by": str(current_user.id) if hasattr(current_user, "id") else None,
            "metadata": '{"service":"meruem","scale":"auto"}',
        },
    )
    db.commit()
    log_action(db, current_user, "sr_upscale_queued", "raster", str(raster_id), {"job_id": str(job_id)})

    return {"job_id": str(job_id), "external_job_id": str(external_job_id), "status": "queued"}


# ---------------------------------------------------------------------------
# XYZ Tile endpoint
# ---------------------------------------------------------------------------

@router.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(
    z: int,
    x: int,
    y: int,
    date: datetime | None = Query(None, description="Only show rasters with acquisition_date <= this datetime"),
    date_exact: date | None = Query(None, description="Only show rasters from this exact date (YYYY-MM-DD)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a 256x256 PNG map tile from stored rasters using GDAL."""
    # Validate zoom range
    if z < 3 or z > 20:
        return Response(
            content=_get_empty_tile(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # Convert tile coords to geographic bbox (for PostGIS spatial query)
    bbox_4326 = _tile_to_bbox_4326(z, x, y)
    lon_min, lat_min, lon_max, lat_max = bbox_4326

    # Convert tile coords to Web Mercator bbox (for GDAL tile rendering)
    bbox_3857 = _tile_to_bbox_3857(z, x, y)

    # Query matching rasters using EPSG:4326 envelope (matches stored geometries)
    envelope = ST_MakeEnvelope(lon_min, lat_min, lon_max, lat_max, 4326)
    query = (
        db.query(RasterCatalog)
        .filter(
            RasterCatalog.processing_status == "ready",
            ST_Intersects(RasterCatalog.bbox, envelope),
        )
    )

    # Date filters
    if date_exact is not None:
        query = query.filter(
            func.date(RasterCatalog.acquisition_date) == date_exact
        )
    elif date is not None:
        query = query.filter(RasterCatalog.acquisition_date <= date)

    # Zoom-level filtering: only enforce min_zoom (show rasters upscaled beyond max_zoom)
    query = query.filter(
        or_(RasterCatalog.min_zoom.is_(None), RasterCatalog.min_zoom <= z),
    )

    # Newest on top (last in list = painted last = on top)
    rasters = query.order_by(RasterCatalog.acquisition_date.asc()).all()

    # --- Migration Step 2: read-through hot-layer cache (flag off => unchanged) --
    # The key folds in the matched raster id-set + each raster's updated_at + the
    # date filter, so any new/changed raster yields a fresh key (correct without
    # invalidation infra). A hit skips the 27-88ms GDAL render entirely.
    cache_key = tile_cache.key_for(z, x, y, date_exact, date, rasters) if tile_cache.enabled() else None
    if cache_key is not None:
        cached = await _cache_get(cache_key)
        if cached is not None:
            return _tile_response_from_cache(cached)

    if not rasters:
        if cache_key is not None:
            await _cache_store(cache_key, tile_cache.NEGATIVE_SENTINEL)
        return Response(
            content=_get_empty_tile(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # Pre-extract everything the sync render needs from ORM objects (so the
    # SQLAlchemy session isn't touched off-thread).
    raster_specs: list[tuple[str, str, list[dict] | None]] = []
    for r in rasters:
        if not r.minio_raw_path:
            continue
        meta = r.metadata_ or {}
        raster_specs.append((str(r.id), r.minio_raw_path, meta.get("band_stats")))

    if not raster_specs:
        if cache_key is not None:
            await _cache_store(cache_key, tile_cache.NEGATIVE_SENTINEL)
        return Response(
            content=_get_empty_tile(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # Heavy block: MinIO fetch + GDAL warp + PNG encode. Run off the event loop,
    # gated by a semaphore so concurrent renders don't starve auth/api workers.
    async with TILE_RENDER_SEMAPHORE:
        png_bytes = await asyncio.to_thread(_render_specs_to_png, raster_specs, bbox_3857)

    if png_bytes is None:
        if cache_key is not None:
            await _cache_store(cache_key, tile_cache.NEGATIVE_SENTINEL)
        return Response(
            content=_get_empty_tile(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    if cache_key is not None:
        await _cache_store(cache_key, png_bytes)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ---------------------------------------------------------------------------
# Get single raster
# ---------------------------------------------------------------------------

@router.get("/{raster_id}", response_model=RasterResponse)
async def get_raster(
    raster_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get raster metadata. Checks area access."""
    raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
    if not raster:
        raise HTTPException(status_code=404, detail="Raster not found")

    if raster.bbox is not None:
        if not verify_area_access(current_user, raster.bbox, "view", db):
            raise HTTPException(status_code=403, detail="Area access denied")

    return _raster_to_response(raster, db)


# ---------------------------------------------------------------------------
# HTJ2K streaming
# ---------------------------------------------------------------------------

@router.get("/{raster_id}/stream")
async def stream_raster(
    raster_id: str,
    request: Request,
    res: int | None = Query(None, description="Resolution level (0=full)"),
    bbox: str | None = Query(None, description="Viewport region: x1,y1,x2,y2"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream HTJ2K bytes with HTTP Range support for browser partial fetches.

    The frontend reads the /info endpoint first to learn the file structure,
    then requests byte ranges of the .jph file for the needed resolution/region.
    """
    raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
    if not raster:
        raise HTTPException(status_code=404, detail="Raster not found")

    if raster.bbox is not None:
        if not verify_area_access(current_user, raster.bbox, "view", db):
            raise HTTPException(status_code=403, detail="Area access denied")

    if not raster.minio_jph_path:
        raise HTTPException(status_code=404, detail="HTJ2K file not yet available")

    jph_path = raster.minio_jph_path
    file_size = minio_storage.get_file_size("jph", jph_path)

    # Handle HTTP Range header for partial content
    range_header = request.headers.get("range")
    if range_header:
        # Parse "bytes=START-END"
        try:
            range_spec = range_header.replace("bytes=", "").strip()
            if "-" in range_spec:
                parts = range_spec.split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else file_size - 1
            else:
                start = int(range_spec)
                end = file_size - 1
        except (ValueError, IndexError):
            raise HTTPException(status_code=416, detail="Invalid range")

        if start >= file_size or end >= file_size:
            raise HTTPException(
                status_code=416,
                detail="Range not satisfiable",
                headers={"Content-Range": f"bytes */{file_size}"},
            )

        length = end - start + 1
        data = minio_storage.get_byte_range("jph", jph_path, start, length)

        return Response(
            content=data,
            status_code=206,
            media_type="application/octet-stream",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
                "Accept-Ranges": "bytes",
            },
        )

    # Full file response (no range)
    data = minio_storage.download_file("jph", jph_path)
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
        },
    )


# ---------------------------------------------------------------------------
# HTJ2K info
# ---------------------------------------------------------------------------

@router.get("/{raster_id}/info")
async def get_raster_info(
    raster_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return HTJ2K file structure info (resolution levels, precinct layout).

    Used by the frontend to plan byte-range requests for streaming.
    """
    raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
    if not raster:
        raise HTTPException(status_code=404, detail="Raster not found")

    if raster.bbox is not None:
        if not verify_area_access(current_user, raster.bbox, "view", db):
            raise HTTPException(status_code=403, detail="Area access denied")

    meta = raster.metadata_ or {}
    htj2k_info = meta.get("htj2k")
    if not htj2k_info:
        raise HTTPException(status_code=404, detail="HTJ2K info not available (still processing?)")

    return htj2k_info


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------

@router.get("/{raster_id}/thumbnail")
async def get_raster_thumbnail(
    raster_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Serve thumbnail JPEG from MinIO."""
    raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
    if not raster:
        raise HTTPException(status_code=404, detail="Raster not found")

    thumb_path = f"{raster_id}/thumbnail.jpg"
    if not minio_storage.file_exists("thumbnails", thumb_path):
        raise HTTPException(status_code=404, detail="Thumbnail not available")

    data = minio_storage.download_file("thumbnails", thumb_path)
    return Response(content=data, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/{raster_id}/status", response_model=RasterStatusResponse)
async def get_raster_status(
    raster_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return processing status and error if any."""
    raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
    if not raster:
        raise HTTPException(status_code=404, detail="Raster not found")

    return RasterStatusResponse(
        id=str(raster.id),
        processing_status=raster.processing_status,
        processing_error=raster.processing_error,
        processing_started_at=raster.processing_started_at,
        processing_completed_at=raster.processing_completed_at,
    )


# ---------------------------------------------------------------------------
# Update raster
# ---------------------------------------------------------------------------

@router.patch("/{raster_id}", response_model=RasterResponse)
async def update_raster(
    raster_id: str,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update raster name, description, tags, or metadata. Owner or admin only."""
    raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
    if not raster:
        raise HTTPException(status_code=404, detail="Raster not found")

    if str(raster.uploaded_by) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner or admin")

    if name is not None:
        raster.name = name
    if description is not None:
        raster.description = description
    if tags is not None:
        raster.tags = tags
    if metadata is not None:
        raster.metadata_ = {**(raster.metadata_ or {}), **metadata}

    db.commit()
    db.refresh(raster)

    changed_fields = [k for k, v in [("name", name), ("description", description), ("tags", tags), ("metadata", metadata)] if v is not None]
    log_action(
        db, current_user, "update_raster",
        resource_type="raster", resource_id=raster_id,
        details={"fields": changed_fields},
        request=request,
    )

    return _raster_to_response(raster, db)


# ---------------------------------------------------------------------------
# Delete raster
# ---------------------------------------------------------------------------

@router.delete("/{raster_id}", status_code=204)
async def delete_raster(
    raster_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete raster, remove from MinIO (raw + jph + thumbnail), delete catalog row."""
    raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
    if not raster:
        raise HTTPException(status_code=404, detail="Raster not found")

    if str(raster.uploaded_by) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner or admin")

    # Remove from MinIO (ignore errors for missing files)
    for bucket, path in [
        ("raw", raster.minio_raw_path),
        ("jph", raster.minio_jph_path),
        ("thumbnails", f"{raster_id}/thumbnail.jpg"),
    ]:
        if path:
            try:
                minio_storage.delete_file(bucket, path)
            except Exception:
                pass

    log_action(
        db, current_user, "delete_raster",
        resource_type="raster", resource_id=raster_id,
        details={"name": raster.name},
        request=request,
    )

    db.delete(raster)
    db.commit()

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timeseries_to_response(ts: RasterTimeseries, db: Session) -> dict[str, Any]:
    """Convert a RasterTimeseries ORM row to a response dict."""
    import json as _json

    entries = (
        db.query(RasterTimeseriesEntry, RasterCatalog)
        .join(RasterCatalog, RasterCatalog.id == RasterTimeseriesEntry.raster_id)
        .filter(RasterTimeseriesEntry.timeseries_id == ts.id)
        .order_by(RasterTimeseriesEntry.sort_order)
        .all()
    )

    area_bbox = None
    if ts.area_bbox is not None:
        raw = db.scalar(ST_AsGeoJSON(ts.area_bbox))
        area_bbox = _json.loads(raw) if raw else None

    return {
        "id": str(ts.id),
        "name": ts.name,
        "description": ts.description,
        "area_bbox": area_bbox,
        "created_by": str(ts.created_by) if ts.created_by else None,
        "created_at": ts.created_at,
        "entries": [_raster_to_response(r, db) for _, r in entries],
    }
