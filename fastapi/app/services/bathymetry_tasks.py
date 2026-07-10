"""Bathymetry pipeline orchestration (Celery).

run_bathymetry(source_raster_id):
  1. record a bathymetry_jobs row (running)
  2. call the bathymetry service /infer (reads source from MinIO 'raw')
  3. re-ingest the colour-mapped depth product as a DERIVED raster (so it tiles
     and displays via the normal instant-load path) — tagged source_kind=bathymetry
     so it does NOT recursively re-trigger inference
  4. build a template report from the depth stats + cross-section
  5. mark the job done (depth_raster_id + report_id)

Best-effort and resilient: any failure marks the job 'failed' but never crashes
the worker. Replacing the stub model with the real one requires no change here.
"""

from __future__ import annotations

import os
import uuid

import httpx
from sqlalchemy import text

from ..celery_app import celery
from ..database import SessionLocal
from ..models import RasterCatalog
from . import minio_storage
from . import report_builder
from .tasks import process_raster_upload

BATHYMETRY_SERVICE_URL = os.getenv("BATHYMETRY_SERVICE_URL", "http://bathymetry:8002")


@celery.task(name="run_bathymetry", bind=True, max_retries=0)
def run_bathymetry(self, source_raster_id: str) -> dict:
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    try:
        src = db.query(RasterCatalog).filter(RasterCatalog.id == source_raster_id).first()
        if not src or not src.minio_raw_path:
            return {"status": "skipped", "reason": "source raster not found"}

        db.execute(
            text(
                """INSERT INTO bathymetry_jobs (id, source_raster_id, status, model)
                   VALUES (:id, :src, 'running', 'dl-pro-v3')"""
            ),
            {"id": job_id, "src": source_raster_id},
        )
        db.commit()

        # ROI footprint for the source raster — the SDB model fetches Sentinel-2
        # for this area and runs DL-Pro on it (image -> model -> depth).
        bbox = _source_bbox(db, src)
        if bbox is None:
            _fail(db, job_id, "source raster has no bbox")
            return {"status": "failed", "stage": "bbox", "error": "no bbox"}

        # 1. Inference: ROI -> free Sentinel-2 (Planetary Computer) -> DL-Pro depth.
        try:
            resp = httpx.post(
                f"{BATHYMETRY_SERVICE_URL}/bathymetry/infer",
                json={"raster_id": source_raster_id, "bbox": bbox},
                timeout=httpx.Timeout(900.0, connect=10.0),
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _fail(db, job_id, f"inference failed: {e}")
            return {"status": "failed", "stage": "infer", "error": str(e)}

        # 2. Re-ingest the colour-mapped depth product as a derived raster.
        derived_id = str(uuid.uuid4())
        depth_rgb_key = data["depth_rgb_key"]
        raw_key = f"{derived_id}/depth_rgb.tif"
        try:
            blob = minio_storage.download_file(data.get("depth_bucket", "depth"), depth_rgb_key)
            import io
            minio_storage.upload_file(
                "raw", raw_key, io.BytesIO(blob),
                content_type="image/tiff", length=len(blob),
            )
        except Exception as e:
            _fail(db, job_id, f"depth product transfer failed: {e}")
            return {"status": "failed", "stage": "transfer", "error": str(e)}

        derived = RasterCatalog(
            id=derived_id,
            name=f"Bathymetry — {src.name}",
            description=f"Depth product derived from {src.name}",
            original_filename=f"depth_{source_raster_id}.tif",
            original_format="GTiff",
            minio_raw_path=raw_key,
            processing_status="pending",
            uploaded_by=src.uploaded_by,
            metadata_={
                "source_kind": "bathymetry",
                "parent_raster_id": source_raster_id,
                "bathymetry": {
                    "model": data.get("model"),
                    "stats": data.get("stats"),
                    "max_depth_m": data.get("max_depth_m"),
                },
            },
        )
        db.add(derived)
        db.commit()

        # Tiles it via the normal instant-load path (the source_kind guard in
        # process_raster_upload prevents recursive re-inference).
        process_raster_upload.delay(derived_id)

        # 3. Build the template report.
        try:
            report_id, report_code = report_builder.build_report(
                db, src, derived_id, data, created_by=str(src.uploaded_by) if src.uploaded_by else None
            )
        except Exception as e:
            _fail(db, job_id, f"report build failed: {e}")
            return {"status": "failed", "stage": "report", "error": str(e)}

        db.execute(
            text(
                """UPDATE bathymetry_jobs
                   SET status='done', depth_raster_id=:depth, report_id=:report,
                       updated_at=NOW(), metadata=CAST(:meta AS jsonb)
                   WHERE id=:id"""
            ),
            {
                "id": job_id,
                "depth": derived_id,
                "report": report_id,
                "meta": _json({"report_code": report_code, "stats": data.get("stats")}),
            },
        )
        db.commit()
        return {
            "status": "done",
            "job_id": job_id,
            "depth_raster_id": derived_id,
            "report_id": report_id,
            "report_code": report_code,
        }
    finally:
        db.close()


@celery.task(name="run_bathymetry_roi", bind=True, max_retries=0)
def run_bathymetry_roi(self, job_id, bbox, start_date="2023-01-01",
                       end_date="2024-12-31", name="Survey area", cache_key=None):
    """ROI-first SDB run (no uploaded source): the Studio draws an area; we fetch
    Sentinel-2 for it, run DL-Pro, ingest the depth product, and build a report.
    The job row is pre-inserted (status 'running') by the POST /roi route."""
    import io
    db = SessionLocal()
    try:
        try:
            resp = httpx.post(
                f"{BATHYMETRY_SERVICE_URL}/bathymetry/infer",
                json={"raster_id": job_id, "bbox": bbox,
                      "start_date": start_date, "end_date": end_date},
                timeout=httpx.Timeout(900.0, connect=10.0),
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _fail(db, job_id, f"inference failed: {e}")
            return {"status": "failed", "stage": "infer", "error": str(e)}

        derived_id = str(uuid.uuid4())
        raw_key = f"{derived_id}/depth_rgb.tif"
        try:
            blob = minio_storage.download_file(data.get("depth_bucket", "depth"), data["depth_rgb_key"])
            minio_storage.upload_file("raw", raw_key, io.BytesIO(blob),
                                      content_type="image/tiff", length=len(blob))
        except Exception as e:
            _fail(db, job_id, f"depth product transfer failed: {e}")
            return {"status": "failed", "stage": "transfer", "error": str(e)}

        derived = RasterCatalog(
            id=derived_id, name=f"Bathymetry — {name}",
            description=f"SDB depth product for {name}",
            original_filename=f"depth_{job_id}.tif", original_format="GTiff",
            minio_raw_path=raw_key, processing_status="pending",
            metadata_={"source_kind": "bathymetry", "parent_raster_id": None,
                       "bathymetry": {"model": data.get("model"),
                                      "stats": data.get("stats"),
                                      "max_depth_m": data.get("max_depth_m")}},
        )
        db.add(derived)
        db.commit()
        process_raster_upload.delay(derived_id)

        try:
            report_id, report_code = report_builder.build_report(
                db, None, derived_id, data, site_name=name, cache_key=cache_key)
        except Exception as e:
            _fail(db, job_id, f"report build failed: {e}")
            return {"status": "failed", "stage": "report", "error": str(e)}

        db.execute(
            text("""UPDATE bathymetry_jobs SET status='done', depth_raster_id=:depth,
                    report_id=:report, updated_at=NOW(), metadata=CAST(:meta AS jsonb)
                    WHERE id=:id"""),
            {"id": job_id, "depth": derived_id, "report": report_id,
             "meta": _json({"report_code": report_code, "stats": data.get("stats")})},
        )
        db.commit()
        return {"status": "done", "job_id": job_id, "depth_raster_id": derived_id,
                "report_id": report_id, "report_code": report_code}
    finally:
        db.close()


def _source_bbox(db, src):
    """ROI [west,south,east,north] dict from the raster's stored bounds or its
    PostGIS bbox envelope."""
    b = (src.metadata_ or {}).get("bounds")
    if isinstance(b, dict) and all(k in b for k in ("left", "bottom", "right", "top")):
        return {"west": b["left"], "south": b["bottom"], "east": b["right"], "north": b["top"]}
    try:
        row = db.execute(
            text("SELECT ST_XMin(bbox) w, ST_YMin(bbox) s, ST_XMax(bbox) e, ST_YMax(bbox) n "
                 "FROM raster_catalog WHERE id = :id"),
            {"id": str(src.id)},
        ).first()
        if row and row.w is not None:
            return {"west": row.w, "south": row.s, "east": row.e, "north": row.n}
    except Exception:
        pass
    return None


def _fail(db, job_id: str, msg: str) -> None:
    try:
        db.rollback()
        db.execute(
            text("UPDATE bathymetry_jobs SET status='failed', error_message=:m, updated_at=NOW() WHERE id=:id"),
            {"m": msg[:2000], "id": job_id},
        )
        db.commit()
    except Exception:
        db.rollback()


def _json(obj) -> str:
    import json
    return json.dumps(obj)
