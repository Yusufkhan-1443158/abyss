"""Celery tasks for raster and vector processing pipelines."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..celery_app import celery
from ..database import SessionLocal
from ..models import RasterCatalog, VectorDataset, VectorFeature
from . import minio_storage
from .htj2k_converter import convert_to_htj2k, get_htj2k_info
from .raster_processor import (
    build_cog,
    compute_band_percentiles,
    extract_metadata,
    generate_thumbnail,
    reproject_to_4326,
)
from .vector_processor import process_vector_file
from .vendor_metadata import extract_vendor_metadata


def _get_db() -> Session:
    """Create a standalone DB session for Celery tasks (not FastAPI's)."""
    return SessionLocal()


@celery.task(name="process_raster_upload", bind=True, max_retries=1)
def process_raster_upload(self, raster_id: str) -> dict:
    """Full raster processing pipeline.

    1. Fetch raster_catalog row, get minio_raw_path
    2. Download raw file from MinIO to temp dir
    3. Extract metadata with raster_processor
    4. Reproject if CRS != EPSG:4326
    5. Convert to HTJ2K with htj2k_converter
    6. Upload .jph to MinIO (jph bucket)
    7. Generate thumbnail, upload to MinIO
    8. Get HTJ2K info (resolution levels etc), store in metadata JSONB
    9. Update raster_catalog: status='ready', fill in all metadata fields
    10. On error: status='failed', processing_error=str(e)
    11. Clean up temp files
    """
    db = _get_db()
    tmp_dir = tempfile.mkdtemp(prefix="raster_")

    try:
        # 1. Fetch raster catalog row
        raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
        if not raster:
            raise ValueError(f"Raster {raster_id} not found")

        raster.processing_status = "processing"
        raster.processing_started_at = datetime.now(timezone.utc)
        db.commit()

        raw_path = raster.minio_raw_path
        original_filename = raster.original_filename

        # 2. Download raw file (and sidecars, if this is a bundle upload).
        #    Bundle uploads store every file under "{raster_id}/bundle/<relpath>"
        #    and minio_raw_path points at the main raster within that prefix.
        #    Single-file uploads keep the legacy "{raster_id}/<filename>" layout.
        bundle_prefix = f"{raster_id}/bundle/"
        if raw_path.startswith(bundle_prefix):
            bundle_root = os.path.join(tmp_dir, "bundle")
            os.makedirs(bundle_root, exist_ok=True)
            keys = minio_storage.list_prefix("raw", bundle_prefix)
            for key in keys:
                relpath = key[len(bundle_prefix):]
                dest = os.path.join(bundle_root, relpath)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                data = minio_storage.download_file("raw", key)
                with open(dest, "wb") as fh:
                    fh.write(data)
            main_relpath = raw_path[len(bundle_prefix):]
            local_raw = os.path.join(bundle_root, main_relpath)
            metadata_folder = bundle_root
        else:
            local_raw = os.path.join(tmp_dir, original_filename)
            raw_data = minio_storage.download_file("raw", raw_path)
            with open(local_raw, "wb") as f:
                f.write(raw_data)
            metadata_folder = tmp_dir

        # 2b. Vendor metadata (sidecars). Failure here never blocks ingest.
        try:
            vendor_meta, vendor_warnings = extract_vendor_metadata(metadata_folder)
        except Exception as exc:
            vendor_meta, vendor_warnings = {}, [f"vendor parser crashed: {exc}"]

        # 3. Extract metadata
        meta = extract_metadata(local_raw)

        # 4. Reproject to EPSG:4326 only if needed. (We no longer force non-TIFF
        #    inputs to TIFF here — that was only for ojph_compress, which is now
        #    deferred to Stage B. GDAL's COG driver reads JP2/NITF/ECW natively.)
        input_for_conversion = local_raw
        if meta["crs"] != "EPSG:4326":
            reprojected = os.path.join(tmp_dir, "reprojected.tif")
            reproject_to_4326(local_raw, reprojected)
            input_for_conversion = reprojected
            meta = extract_metadata(reprojected)

        # 5. INSTANT-LOAD: build the COG straight into the shared tile cache so the
        #    very first map tile is already warm (no first-tile download/build
        #    stall). The deep-zoom HTJ2K (.jph) is deferred to Stage B and never
        #    blocks the raster going 'ready'.
        cache_dir = os.environ.get("RASTER_CACHE_DIR", "/tmp/raster_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cog_cache = os.path.join(cache_dir, f"{raster_id}.tif")
        build_cog(input_for_conversion, cog_cache)

        # 6. Generate the 256px thumbnail (used for the instant coarse first-paint).
        thumb_local = os.path.join(tmp_dir, "thumbnail.jpg")
        generate_thumbnail(input_for_conversion, thumb_local, size=256)
        thumb_minio_path = f"{raster_id}/thumbnail.jpg"
        with open(thumb_local, "rb") as f:
            thumb_size = os.path.getsize(thumb_local)
            minio_storage.upload_file(
                "thumbnails", thumb_minio_path, f,
                content_type="image/jpeg",
                length=thumb_size,
            )

        # 8b. Compute band statistics for dynamic range display. 8-bit imagery
        #     (e.g. a display-ready RGB depth product) is already in [0,255] —
        #     use an identity stretch so tiles render its true colours instead
        #     of a per-band percentile re-stretch (which would distort the ramp).
        from osgeo import gdal as _gdal

        _gdal.UseExceptions()
        _ds = _gdal.Open(input_for_conversion)
        if _ds.GetRasterBand(1).DataType == _gdal.GDT_Byte:
            band_stats = [
                {"band": i + 1, "p2_5": 0.0, "p97_5": 255.0}
                for i in range(_ds.RasterCount)
            ]
        else:
            band_stretch = compute_band_percentiles(_ds, 2.5, 97.5)
            band_stats = [
                {"band": i + 1, "p2_5": s[0], "p97_5": s[1]}
                for i, s in enumerate(band_stretch)
            ]
        _ds = None

        # 9. Update raster_catalog with all metadata
        from geoalchemy2.elements import WKTElement

        raster.processing_status = "ready"
        raster.processing_completed_at = datetime.now(timezone.utc)
        raster.crs = "EPSG:4326"
        raster.width_px = meta["width_px"]
        raster.height_px = meta["height_px"]
        raster.num_bands = meta["num_bands"]
        raster.bit_depth = meta["bit_depth"]
        raster.resolution_m = meta["resolution_m"]
        raster.min_zoom = meta["min_zoom"]
        raster.max_zoom = meta["max_zoom"]
        raster.bbox = WKTElement(meta["bbox_wkt"], srid=4326)
        raster.center_point = WKTElement(meta["center_wkt"], srid=4326)
        raster.metadata_ = {
            **(raster.metadata_ or {}),
            "bounds": meta["bounds"],
            "band_stats": band_stats,
        }
        if vendor_meta:
            raster.vendor_metadata = vendor_meta
            if vendor_meta.get("acquired_at") and not raster.acquisition_date:
                try:
                    raster.acquisition_date = datetime.fromisoformat(
                        vendor_meta["acquired_at"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass
        if vendor_warnings:
            raster.metadata_warnings = vendor_warnings

        db.commit()

        # Migration Step 1a: tell the (future) super-worker which bbox changed so
        # it can purge orphaned hot-layer bytes + re-warm. Best-effort; never
        # blocks ingest. Tile-serving CORRECTNESS does not depend on this — the
        # read-through cache key already auto-invalidates on the raster's updated_at.
        try:
            from .tile_cache import publish_invalidation
            publish_invalidation(meta.get("bounds"))
        except Exception:
            pass

        # Stage B (background, low-priority sw_bg queue): persist the COG to object
        # storage (so any worker/restart serves tiles without a rebuild) and
        # generate the deferred HTJ2K. Best-effort — never blocks 'ready'.
        try:
            finalize_raster.apply_async(args=[raster_id], queue="sw_bg")
        except Exception:
            pass

        # Auto-chain bathymetry inference for source imagery (not for derived
        # depth products). Best-effort; the depth layer + template report follow.
        try:
            if (raster.metadata_ or {}).get("source_kind") != "bathymetry":
                from .bathymetry_tasks import run_bathymetry
                run_bathymetry.apply_async(args=[raster_id], queue="sw_bg")
        except Exception:
            pass

        return {"status": "ready", "raster_id": raster_id}

    except Exception as e:
        db.rollback()
        # 10. Mark as failed
        try:
            raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
            if raster:
                raster.processing_status = "failed"
                raster.processing_error = str(e)[:2000]
                db.commit()
        except Exception:
            db.rollback()
        raise

    finally:
        # 11. Clean up temp files
        db.close()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@celery.task(name="finalize_raster", bind=True, max_retries=1)
def finalize_raster(self, raster_id: str) -> dict:
    """Stage B (background): persist the prebuilt COG to object storage and
    generate the deferred HTJ2K deep-zoom product. Runs off the display critical
    path on the low-priority sw_bg queue. Every step is best-effort: a failure
    here degrades only the deep-zoom viewer / multi-worker durability, never the
    map tiles (which already serve from the warm local COG)."""
    import shutil

    db = _get_db()
    tmp_dir = tempfile.mkdtemp(prefix="finalize_")
    try:
        raster = db.query(RasterCatalog).filter(RasterCatalog.id == raster_id).first()
        if not raster:
            return {"status": "skipped", "reason": "raster not found"}

        cache_dir = os.environ.get("RASTER_CACHE_DIR", "/tmp/raster_cache")
        cog_cache = os.path.join(cache_dir, f"{raster_id}.tif")
        if not os.path.exists(cog_cache):
            return {"status": "skipped", "reason": "no local COG to persist"}

        # B1: persist COG → cog bucket (conventional key the tile renderer reads).
        cog_key = f"{raster_id}/{raster_id}.cog.tif"
        try:
            with open(cog_cache, "rb") as f:
                minio_storage.upload_file(
                    "cog", cog_key, f,
                    content_type="image/tiff",
                    length=os.path.getsize(cog_cache),
                )
            raster.minio_cog_path = cog_key
            db.commit()
        except Exception:
            db.rollback()

        # B2: deferred HTJ2K (.jph) for the deep-zoom single-image viewer.
        try:
            from osgeo import gdal as _gdal
            _gdal.UseExceptions()
            plain_tif = os.path.join(tmp_dir, f"{raster_id}.tif")
            # ojph_compress needs an uncompressed TIFF; the COG is DEFLATE-tiled.
            _gdal.Translate(plain_tif, cog_cache, format="GTiff")
            jph_local = os.path.join(tmp_dir, f"{raster_id}.jph")
            convert_to_htj2k(plain_tif, jph_local, lossless=True)
            jph_key = f"{raster_id}/{raster_id}.jph"
            with open(jph_local, "rb") as f:
                minio_storage.upload_file(
                    "jph", jph_key, f,
                    content_type="application/octet-stream",
                    length=os.path.getsize(jph_local),
                )
            htj2k_info = get_htj2k_info(jph_local)
            raster.minio_jph_path = jph_key
            raster.metadata_ = {**(raster.metadata_ or {}), "htj2k": htj2k_info}
            db.commit()
        except Exception:
            db.rollback()  # HTJ2K unavailable (e.g. no ojph_compress) — fine.

        return {"status": "finalized", "raster_id": raster_id}
    finally:
        db.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


@celery.task(name="process_vector_upload", bind=True, max_retries=1)
def process_vector_upload(self, dataset_id: str) -> dict:
    """Full vector processing pipeline.

    1. Fetch vector_datasets row, get minio_raw_path
    2. Download raw file from MinIO to temp dir
    3. Process with vector_processor: detect format, read features, reproject, validate
    4. Batch INSERT into vector_features (1000 per batch)
    5. Update vector_datasets: status='ready', geometry_types, properties_schema, etc.
    6. On error: status='failed', processing_error=str(e)
    7. Clean up temp files
    """
    db = _get_db()
    tmp_dir = tempfile.mkdtemp(prefix="vector_")

    try:
        # 1. Fetch dataset row
        dataset = db.query(VectorDataset).filter(VectorDataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Vector dataset {dataset_id} not found")

        dataset.processing_status = "processing"
        db.commit()

        raw_path = dataset.minio_raw_path
        original_filename = dataset.original_filename

        # 2. Download raw file
        local_raw = os.path.join(tmp_dir, original_filename)
        raw_data = minio_storage.download_file("vectors", raw_path)
        with open(local_raw, "wb") as f:
            f.write(raw_data)

        # 3. Process vector file
        result = process_vector_file(local_raw, original_filename)

        # 4. Batch insert features
        features_data = result["features"]
        batch_size = 1000

        from geoalchemy2.elements import WKTElement

        for i in range(0, len(features_data), batch_size):
            batch = features_data[i : i + batch_size]
            feature_objects = []
            for wkt, geom_type, props in batch:
                vf = VectorFeature(
                    dataset_id=dataset_id,
                    geom=WKTElement(f"SRID=4326;{wkt}", srid=4326),
                    geometry_type=geom_type,
                    properties=props,
                    created_by=dataset.created_by,
                )
                feature_objects.append(vf)
            db.bulk_save_objects(feature_objects)
            db.flush()

        # 5. Update dataset metadata
        dataset.processing_status = "ready"
        dataset.geometry_types = result["geometry_types"]
        dataset.properties_schema = result["properties_schema"]
        dataset.feature_count = result["feature_count"]

        if result["bbox"]:
            minx, miny, maxx, maxy = result["bbox"]
            bbox_wkt = (
                f"SRID=4326;POLYGON(({minx} {miny}, {maxx} {miny}, "
                f"{maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"
            )
            dataset.bbox = WKTElement(bbox_wkt, srid=4326)

        db.commit()

        return {"status": "ready", "dataset_id": dataset_id}

    except Exception as e:
        db.rollback()
        # 6. Mark as failed
        try:
            dataset = db.query(VectorDataset).filter(VectorDataset.id == dataset_id).first()
            if dataset:
                dataset.processing_status = "failed"
                dataset.processing_error = str(e)[:2000]
                db.commit()
        except Exception:
            db.rollback()
        raise

    finally:
        # 7. Clean up
        db.close()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
