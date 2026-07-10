"""Annotation layer management (wraps vector_datasets with source='annotation')."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response
from geoalchemy2.elements import WKTElement
from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.audit_logger import log_action
from ..middleware.auth_middleware import get_current_user, require_analyst_or_admin
from ..models import User, UserGroupMember, VectorDataset, VectorFeature
from ..schemas import (
    AnnotationLayerCreate,
    AnnotationLayerResponse,
    VectorFeatureCreate,
    VectorFeatureResponse,
    VectorFeatureUpdate,
)
from ..services.vector_processor import export_features

router = APIRouter()


def _dataset_to_response(ds: VectorDataset, db: Session) -> dict[str, Any]:
    """Convert annotation layer (VectorDataset) to response dict."""
    data = {
        "id": str(ds.id),
        "name": ds.name,
        "description": ds.description,
        "source": ds.source,
        "original_filename": ds.original_filename,
        "original_format": ds.original_format,
        "file_size_bytes": ds.file_size_bytes,
        "feature_count": ds.feature_count or 0,
        "geometry_types": ds.geometry_types,
        "crs": ds.crs,
        "properties_schema": ds.properties_schema or {},
        "processing_status": ds.processing_status,
        "processing_error": ds.processing_error,
        "created_by": str(ds.created_by) if ds.created_by else None,
        "is_public": ds.is_public,
        "shared_with_groups": [str(g) for g in (ds.shared_with_groups or [])],
        "default_style": ds.default_style,
        "tags": ds.tags or [],
        "metadata": ds.metadata_ or {},
        "created_at": ds.created_at,
        "updated_at": ds.updated_at,
    }

    if ds.bbox is not None:
        raw = db.scalar(ST_AsGeoJSON(ds.bbox))
        data["bbox"] = json.loads(raw) if raw else None
    else:
        data["bbox"] = None

    return data


def _feature_to_response(feat: VectorFeature, db: Session) -> dict[str, Any]:
    """Convert VectorFeature ORM row to response dict."""
    geom_json = None
    if feat.geom is not None:
        raw = db.scalar(ST_AsGeoJSON(feat.geom))
        geom_json = json.loads(raw) if raw else None

    return {
        "id": str(feat.id),
        "dataset_id": str(feat.dataset_id),
        "geom": geom_json,
        "geometry_type": feat.geometry_type,
        "label": feat.label,
        "description": feat.description,
        "properties": feat.properties or {},
        "style": feat.style,
        "created_by": str(feat.created_by) if feat.created_by else None,
        "created_at": feat.created_at,
        "updated_at": feat.updated_at,
    }


# ---------------------------------------------------------------------------
# Create annotation layer
# ---------------------------------------------------------------------------

@router.post("", response_model=AnnotationLayerResponse, status_code=201)
async def create_annotation_layer(
    body: AnnotationLayerCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new annotation layer (vector_datasets with source='annotation')."""
    ds = VectorDataset(
        name=body.name,
        description=body.description,
        source="annotation",
        processing_status="ready",
        default_style=body.default_style,
        created_by=current_user.id,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    return _dataset_to_response(ds, db)


# ---------------------------------------------------------------------------
# List annotation layers
# ---------------------------------------------------------------------------

@router.get("", response_model=list[AnnotationLayerResponse])
async def list_annotation_layers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List current user's annotation layers + layers shared with their groups."""
    # Get user's group IDs
    group_ids = [
        row.group_id
        for row in db.query(UserGroupMember.group_id)
        .filter(UserGroupMember.user_id == current_user.id)
        .all()
    ]

    query = db.query(VectorDataset).filter(VectorDataset.source == "annotation")

    if current_user.role != "admin":
        # Own layers OR shared with any of user's groups
        conditions = [VectorDataset.created_by == current_user.id]
        if group_ids:
            conditions.append(VectorDataset.shared_with_groups.overlap(group_ids))
        query = query.filter(or_(*conditions))

    layers = query.order_by(VectorDataset.created_at.desc()).all()
    return [_dataset_to_response(ds, db) for ds in layers]


# ---------------------------------------------------------------------------
# Get annotation layer
# ---------------------------------------------------------------------------

@router.get("/{layer_id}", response_model=AnnotationLayerResponse)
async def get_annotation_layer(
    layer_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get annotation layer with all features as GeoJSON in response."""
    ds = (
        db.query(VectorDataset)
        .filter(VectorDataset.id == layer_id, VectorDataset.source == "annotation")
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Annotation layer not found")

    return _dataset_to_response(ds, db)


# ---------------------------------------------------------------------------
# Update annotation layer
# ---------------------------------------------------------------------------

@router.patch("/{layer_id}", response_model=AnnotationLayerResponse)
async def update_annotation_layer(
    layer_id: str,
    name: str | None = None,
    default_style: str | None = Query(None, description="JSON string of style object"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update name or default_style. Owner only."""
    ds = (
        db.query(VectorDataset)
        .filter(VectorDataset.id == layer_id, VectorDataset.source == "annotation")
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Annotation layer not found")

    if str(ds.created_by) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not layer owner")

    if name is not None:
        ds.name = name
    if default_style is not None:
        try:
            ds.default_style = json.loads(default_style)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON for default_style")

    db.commit()
    db.refresh(ds)

    return _dataset_to_response(ds, db)


# ---------------------------------------------------------------------------
# Delete annotation layer
# ---------------------------------------------------------------------------

@router.delete("/{layer_id}", status_code=204)
async def delete_annotation_layer(
    layer_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete annotation layer + features. Owner or admin."""
    ds = (
        db.query(VectorDataset)
        .filter(VectorDataset.id == layer_id, VectorDataset.source == "annotation")
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Annotation layer not found")

    if str(ds.created_by) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner or admin")

    log_action(
        db, current_user, "delete_annotation",
        resource_type="annotation", resource_id=layer_id,
        details={"name": ds.name},
        request=request,
    )

    db.delete(ds)
    db.commit()

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Add feature
# ---------------------------------------------------------------------------

@router.post("/{layer_id}/features", response_model=VectorFeatureResponse, status_code=201)
async def add_annotation_feature(
    layer_id: str,
    body: VectorFeatureCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a drawn feature to an annotation layer.

    Accept GeoJSON geometry + properties + style + label. Validates geometry.
    """
    ds = (
        db.query(VectorDataset)
        .filter(VectorDataset.id == layer_id, VectorDataset.source == "annotation")
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Annotation layer not found")

    # Allow any authenticated user to add to their own layers
    if str(ds.created_by) != str(current_user.id) and current_user.role != "admin":
        # Check if layer is shared with user's groups
        group_ids = [
            row.group_id
            for row in db.query(UserGroupMember.group_id)
            .filter(UserGroupMember.user_id == current_user.id)
            .all()
        ]
        if not ds.shared_with_groups or not any(g in (ds.shared_with_groups or []) for g in group_ids):
            raise HTTPException(status_code=403, detail="Not authorized to add features")

    # Validate geometry with shapely
    from shapely.geometry import shape
    from shapely.validation import make_valid

    try:
        shp = shape(body.geom)
        if not shp.is_valid:
            shp = make_valid(shp)
        geom_wkt = shp.wkt
        geom_type = body.geometry_type or shp.geom_type
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid geometry: {e}")

    feat = VectorFeature(
        dataset_id=layer_id,
        geom=WKTElement(f"SRID=4326;{geom_wkt}", srid=4326),
        geometry_type=geom_type,
        label=body.label,
        description=body.description,
        properties=body.properties or {},
        style=body.style,
        created_by=current_user.id,
    )
    db.add(feat)
    db.commit()
    db.refresh(feat)

    return _feature_to_response(feat, db)


# ---------------------------------------------------------------------------
# Update feature
# ---------------------------------------------------------------------------

@router.patch("/{layer_id}/features/{feature_id}", response_model=VectorFeatureResponse)
async def update_annotation_feature(
    layer_id: str,
    feature_id: str,
    body: VectorFeatureUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update feature geometry, properties, style, or label. Owner only."""
    feat = (
        db.query(VectorFeature)
        .filter(VectorFeature.id == feature_id, VectorFeature.dataset_id == layer_id)
        .first()
    )
    if not feat:
        raise HTTPException(status_code=404, detail="Feature not found")

    if str(feat.created_by) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not feature owner")

    if body.geom is not None:
        from shapely.geometry import shape
        from shapely.validation import make_valid

        try:
            shp = shape(body.geom)
            if not shp.is_valid:
                shp = make_valid(shp)
            feat.geom = WKTElement(f"SRID=4326;{shp.wkt}", srid=4326)
            feat.geometry_type = body.geometry_type or shp.geom_type
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid geometry: {e}")

    if body.label is not None:
        feat.label = body.label
    if body.description is not None:
        feat.description = body.description
    if body.properties is not None:
        feat.properties = body.properties
    if body.style is not None:
        feat.style = body.style

    db.commit()
    db.refresh(feat)

    return _feature_to_response(feat, db)


# ---------------------------------------------------------------------------
# Delete feature
# ---------------------------------------------------------------------------

@router.delete("/{layer_id}/features/{feature_id}", status_code=204)
async def delete_annotation_feature(
    layer_id: str,
    feature_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a feature. Owner or admin."""
    feat = (
        db.query(VectorFeature)
        .filter(VectorFeature.id == feature_id, VectorFeature.dataset_id == layer_id)
        .first()
    )
    if not feat:
        raise HTTPException(status_code=404, detail="Feature not found")

    if str(feat.created_by) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not feature owner or admin")

    db.delete(feat)
    db.commit()

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Share
# ---------------------------------------------------------------------------

@router.post("/{layer_id}/share")
async def share_annotation_layer(
    layer_id: str,
    group_ids: list[str] = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Share annotation layer with group IDs. Owner or admin."""
    ds = (
        db.query(VectorDataset)
        .filter(VectorDataset.id == layer_id, VectorDataset.source == "annotation")
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Annotation layer not found")

    if str(ds.created_by) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner or admin")

    ds.shared_with_groups = [uuid.UUID(g) for g in group_ids]
    db.commit()
    db.refresh(ds)

    return _dataset_to_response(ds, db)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@router.post("/{layer_id}/export")
async def export_annotation_layer(
    layer_id: str,
    format: str = Query("geojson", description="geojson or kml"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export annotation layer as GeoJSON or KML."""
    ds = (
        db.query(VectorDataset)
        .filter(VectorDataset.id == layer_id, VectorDataset.source == "annotation")
        .first()
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Annotation layer not found")

    features = db.query(VectorFeature).filter(VectorFeature.dataset_id == layer_id).all()

    feat_dicts = []
    for f in features:
        geom_raw = db.scalar(ST_AsGeoJSON(f.geom))
        geom_json = json.loads(geom_raw) if geom_raw else None
        if geom_json:
            feat_dicts.append({
                "geometry": geom_json,
                "properties": {
                    **(f.properties or {}),
                    "label": f.label,
                },
            })

    if not feat_dicts:
        raise HTTPException(status_code=404, detail="No features to export")

    ext_map = {"geojson": ".geojson", "kml": ".kml"}
    ext = ext_map.get(format, ".geojson")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.close()

    try:
        export_features(feat_dicts, format, tmp.name)

        with open(tmp.name, "rb") as fh:
            content = fh.read()

        ct_map = {
            "geojson": "application/geo+json",
            "kml": "application/vnd.google-earth.kml+xml",
        }

        return Response(
            content=content,
            media_type=ct_map.get(format, "application/octet-stream"),
            headers={
                "Content-Disposition": f'attachment; filename="{ds.name}{ext}"',
            },
        )
    finally:
        os.unlink(tmp.name)
