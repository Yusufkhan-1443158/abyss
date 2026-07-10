"""Unified search across rasters and vectors."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from geoalchemy2.functions import ST_AsGeoJSON, ST_Intersects, ST_MakeEnvelope
from sqlalchemy import or_, union_all
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.area_check import filter_by_area
from ..middleware.auth_middleware import get_current_user
from ..models import RasterCatalog, User, VectorDataset
from ..schemas import SearchResponse, SearchResultItem

router = APIRouter()


@router.get("", response_model=SearchResponse)
async def search(
    q: str | None = Query(None, description="Text search on name/description"),
    bbox: str | None = Query(None, description="minx,miny,maxx,maxy"),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    type: str | None = Query(None, description="raster, vector, or all"),
    tags: list[str] | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Search across rasters and vectors with text, spatial, and temporal filters.

    Returns merged results sorted by creation date (newest first).
    """
    search_type = (type or "all").lower()

    results: list[dict] = []

    # --- Rasters ---
    if search_type in ("raster", "all"):
        rq = db.query(RasterCatalog)

        if q:
            pattern = f"%{q}%"
            rq = rq.filter(
                or_(
                    RasterCatalog.name.ilike(pattern),
                    RasterCatalog.description.ilike(pattern),
                )
            )

        if bbox:
            try:
                parts = [float(x.strip()) for x in bbox.split(",")]
                minx, miny, maxx, maxy = parts
            except (ValueError, IndexError):
                raise HTTPException(status_code=400, detail="bbox must be minx,miny,maxx,maxy")
            envelope = ST_MakeEnvelope(minx, miny, maxx, maxy, 4326)
            rq = rq.filter(ST_Intersects(RasterCatalog.bbox, envelope))

        if date_from:
            rq = rq.filter(RasterCatalog.acquisition_date >= date_from)
        if date_to:
            rq = rq.filter(RasterCatalog.acquisition_date <= date_to)

        if tags:
            rq = rq.filter(RasterCatalog.tags.overlap(tags))

        # Area access filtering
        rq = filter_by_area(rq, current_user, RasterCatalog.bbox, db)

        for r in rq.all():
            bbox_json = None
            if r.bbox is not None:
                raw = db.scalar(ST_AsGeoJSON(r.bbox))
                bbox_json = json.loads(raw) if raw else None

            results.append({
                "result_type": "raster",
                "id": str(r.id),
                "name": r.name,
                "description": r.description,
                "bbox": bbox_json,
                "tags": r.tags or [],
                "created_at": r.created_at,
            })

    # --- Vectors ---
    if search_type in ("vector", "all"):
        vq = db.query(VectorDataset)

        if q:
            pattern = f"%{q}%"
            vq = vq.filter(
                or_(
                    VectorDataset.name.ilike(pattern),
                    VectorDataset.description.ilike(pattern),
                )
            )

        if bbox:
            try:
                parts = [float(x.strip()) for x in bbox.split(",")]
                minx, miny, maxx, maxy = parts
            except (ValueError, IndexError):
                raise HTTPException(status_code=400, detail="bbox must be minx,miny,maxx,maxy")
            envelope = ST_MakeEnvelope(minx, miny, maxx, maxy, 4326)
            vq = vq.filter(ST_Intersects(VectorDataset.bbox, envelope))

        if tags:
            vq = vq.filter(VectorDataset.tags.overlap(tags))

        # Area access filtering
        vq = filter_by_area(vq, current_user, VectorDataset.bbox, db)

        for v in vq.all():
            bbox_json = None
            if v.bbox is not None:
                raw = db.scalar(ST_AsGeoJSON(v.bbox))
                bbox_json = json.loads(raw) if raw else None

            results.append({
                "result_type": "vector",
                "id": str(v.id),
                "name": v.name,
                "description": v.description,
                "bbox": bbox_json,
                "tags": v.tags or [],
                "created_at": v.created_at,
            })

    # Sort by date (newest first)
    results.sort(key=lambda x: x["created_at"], reverse=True)

    total = len(results)
    page = results[offset : offset + limit]

    return SearchResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[SearchResultItem(**item) for item in page],
    )
