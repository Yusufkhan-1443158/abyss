"""Template-based bathymetry reports API.

Replaces Glyph's freeform `intelligence_reports` (OCR/Airflow) with Abyss
`bathymetry_reports`: each report is an ordered list of typed sections
auto-filled by the pipeline (services/report_builder.py). The NDJSON /stream
protocol (manifest line + slim rows + ETag) is preserved so reports-browse keeps
working; the /{ref} detail returns the full `sections` array for report-view.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import text

from ..database import engine
from ..middleware.auth_middleware import get_current_user
from ..models import User

router = APIRouter()
stats_router = APIRouter()

REPORT_STREAM_SEMAPHORE = asyncio.Semaphore(8)


def _iso(v):
    return v.isoformat() if v is not None and hasattr(v, "isoformat") else v


# ---------------------------------------------------------------------------
# GET /reports  — GeoJSON FeatureCollection (markers)
# ---------------------------------------------------------------------------
@router.get("")
def list_reports(limit: int = Query(1000, le=5000), current_user: User = Depends(get_current_user)):
    sql = text(
        """
        SELECT id, report_code, title, site_name, classification_level, status,
               statistics, generated_at,
               ST_X(geom) AS lon, ST_Y(geom) AS lat
        FROM bathymetry_reports
        ORDER BY generated_at DESC NULLS LAST
        LIMIT :lim
        """
    )
    features = []
    with engine.connect() as conn:
        for r in conn.execute(sql, {"lim": limit}).mappings():
            if r["lon"] is None or r["lat"] is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                "properties": {
                    "id": str(r["id"]),
                    "report_code": r["report_code"],
                    "title": r["title"],
                    "site_name": r["site_name"],
                    "classification_level": r["classification_level"],
                    "status": r["status"],
                    "statistics": r["statistics"],
                    "generated_at": _iso(r["generated_at"]),
                },
            })
    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# GET /reports/stream  — NDJSON (manifest + slim rows), ETag-cached
# ---------------------------------------------------------------------------
@router.get("/stream")
async def stream_reports(
    bbox: str = Query("-180,-90,180,90"),
    if_none_match: Optional[str] = Header(None),
    current_user: User = Depends(get_current_user),
):
    try:
        w, s, e, n = [float(x.strip()) for x in bbox.split(",")]
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="bbox must be minx,miny,maxx,maxy")

    params = dict(w=w, s=s, e=e, n=n)
    etag_sql = text(
        """
        SELECT COUNT(*) AS n, COALESCE(MAX(generated_at)::text, '') AS m
        FROM bathymetry_reports
        WHERE geom IS NULL OR geom && ST_MakeEnvelope(:w, :s, :e, :n, 4326)
        """
    )
    with engine.connect() as conn:
        agg = conn.execute(etag_sql, params).first()
    count = agg.n if agg else 0
    etag = f'W/"{count}-{agg.m if agg else ""}"'
    if if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "private, max-age=0, must-revalidate"})

    select_sql = text(
        """
        SELECT id, report_code, title, site_name, classification_level, status,
               depth_raster_id, statistics, generated_at,
               ST_X(geom) AS lon, ST_Y(geom) AS lat
        FROM bathymetry_reports
        WHERE geom IS NULL OR geom && ST_MakeEnvelope(:w, :s, :e, :n, 4326)
        ORDER BY generated_at DESC NULLS LAST
        """
    )

    async def gen():
        yield (json.dumps({"type": "manifest", "etag": etag, "count": count, "bbox": [w, s, e, n]}) + "\n").encode()
        if count == 0:
            return
        async with REPORT_STREAM_SEMAPHORE:
            with engine.connect().execution_options(stream_results=True, yield_per=500) as conn:
                for row in conn.execute(select_sql, params).mappings():
                    stats = row["statistics"] or {}
                    thumb = f"/api/rasters/{row['depth_raster_id']}/thumbnail" if row["depth_raster_id"] else ""
                    obj = {
                        # field names mirror the old slim-row shape so reports-browse
                        # (_streamRowToReport) renders without changes.
                        "id": str(row["id"]),
                        "report_id": row["report_code"],
                        "country": row["site_name"] or "Bathymetry",
                        "lon": row["lon"],
                        "lat": row["lat"],
                        "report_date": _iso(row["generated_at"]),
                        "processed_at": _iso(row["generated_at"]),
                        "classification_level": row["classification_level"],
                        "document_title": row["title"],
                        "thumbnail_url": thumb,
                        "extraction_confidence": row["status"],
                        "report_type": "bathymetry",
                        "mean_depth_m": stats.get("mean_m") if isinstance(stats, dict) else None,
                    }
                    yield (json.dumps(obj) + "\n").encode()
                    await asyncio.sleep(0)

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"ETag": etag, "Cache-Control": "private, max-age=0, must-revalidate", "X-Report-Count": str(count)},
    )


# ---------------------------------------------------------------------------
# Inbox / batch upload — disabled in Abyss (reports are pipeline-generated)
# ---------------------------------------------------------------------------
@router.get("/inbox/status")
def inbox_status(current_user: User = Depends(get_current_user)):
    return {"enabled": False, "pending": 0, "message": "Reports are generated by the bathymetry pipeline."}


@router.post("/upload-batch")
def upload_batch(current_user: User = Depends(get_current_user)):
    raise HTTPException(status_code=400, detail="Manual report upload is disabled; reports are generated by the bathymetry pipeline.")


# ---------------------------------------------------------------------------
# GET /reports/{ref}  — detail by report_code OR uuid (full sections)
# ---------------------------------------------------------------------------
@router.get("/{ref}")
def get_report(ref: str, current_user: User = Depends(get_current_user)):
    sql = text(
        """
        SELECT id, report_code, source_raster_id, depth_raster_id, template_id,
               title, site_name, classification_level, status, sections, statistics,
               generated_at, created_at,
               ST_X(geom) AS lng, ST_Y(geom) AS lat
        FROM bathymetry_reports
        WHERE report_code = :ref OR CAST(id AS text) = :ref
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"ref": ref}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Report not found")
    r = dict(row)
    r["id"] = str(r["id"])
    r["source_raster_id"] = str(r["source_raster_id"]) if r["source_raster_id"] else None
    r["depth_raster_id"] = str(r["depth_raster_id"]) if r["depth_raster_id"] else None
    r["generated_at"] = _iso(r["generated_at"])
    r["created_at"] = _iso(r["created_at"])
    # compatibility aliases for any old-shape consumer
    r["report_id"] = r["report_code"]
    r["document_title"] = r["title"]
    r["country"] = r["site_name"]
    return r


# ---------------------------------------------------------------------------
# stats_router  — mounted at /api
# ---------------------------------------------------------------------------
@stats_router.get("/stats")
def report_stats(current_user: User = Depends(get_current_user)):
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE status='ready') AS ready FROM bathymetry_reports"
        )).first()
    return {"total": row.total if row else 0, "ready": row.ready if row else 0}


@stats_router.get("/countries")
def report_countries(current_user: User = Depends(get_current_user)):
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT site_name AS name, COUNT(*) AS count FROM bathymetry_reports GROUP BY site_name ORDER BY count DESC"
        )).mappings().all()
    return [{"name": r["name"] or "Bathymetry", "count": r["count"]} for r in rows]
