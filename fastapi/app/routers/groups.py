"""Group and area restriction management router — admin-only."""

from __future__ import annotations

import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from geoalchemy2.functions import ST_AsGeoJSON, ST_GeomFromGeoJSON
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.audit_logger import log_action
from ..middleware.auth_middleware import require_admin
from ..models import GroupAllowedApp, GroupAreaRestriction, RegisteredApp, User, UserGroup, UserGroupMember
from ..schemas import (
    AreaRestrictionCreate,
    AreaRestrictionResponse,
    GroupCreate,
    GroupResponse,
    GroupUpdate,
    GroupWithMembers,
    UserResponse,
)

router = APIRouter()

VALID_ACCESS_LEVELS = {"view", "download", "manage"}


def _get_allowed_apps(group_id, db: Session) -> list[str]:
    """Get the list of app_name strings allowed for a group."""
    rows = db.query(GroupAllowedApp.app_name).filter(GroupAllowedApp.group_id == group_id).all()
    return [r.app_name for r in rows]


def _set_allowed_apps(group_id, app_names: list[str], db: Session):
    """Replace the allowed apps for a group."""
    db.query(GroupAllowedApp).filter(GroupAllowedApp.group_id == group_id).delete()
    for name in app_names:
        db.add(GroupAllowedApp(group_id=group_id, app_name=name))


def _group_to_response(group: UserGroup, member_count: int | None = None, db: Session | None = None) -> dict:
    resp = {
        "id": str(group.id),
        "name": group.name,
        "description": group.description,
        "created_by": str(group.created_by) if group.created_by else None,
        "allowed_apps": _get_allowed_apps(group.id, db) if db else [],
        "created_at": group.created_at,
        "updated_at": group.updated_at,
    }
    if member_count is not None:
        resp["member_count"] = member_count
    return resp


# --------------------------------------------------------------------------
# GET /groups
# --------------------------------------------------------------------------
@router.get("")
def list_groups(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    """List all groups with member counts (admin only)."""
    total = db.query(UserGroup).count()

    rows = (
        db.query(
            UserGroup,
            func.count(UserGroupMember.user_id).label("member_count"),
        )
        .outerjoin(UserGroupMember, UserGroupMember.group_id == UserGroup.id)
        .group_by(UserGroup.id)
        .order_by(UserGroup.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [_group_to_response(g, mc, db) for g, mc in rows],
    }


# --------------------------------------------------------------------------
# POST /groups
# --------------------------------------------------------------------------
@router.post("", status_code=status.HTTP_201_CREATED)
def create_group(
    body: GroupCreate,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Create a new group (admin only)."""
    group = UserGroup(
        name=body.name,
        description=body.description,
        created_by=admin.id,
    )
    db.add(group)
    db.flush()  # get group.id before inserting apps

    if body.allowed_apps:
        _set_allowed_apps(group.id, body.allowed_apps, db)

    db.commit()
    db.refresh(group)

    log_action(
        db, admin, "create_group",
        resource_type="group", resource_id=str(group.id),
        details={"name": group.name, "allowed_apps": body.allowed_apps},
        request=request,
    )

    return _group_to_response(group, 0, db)


# --------------------------------------------------------------------------
# GET /groups/{group_id}
# --------------------------------------------------------------------------
@router.get("/{group_id}")
def get_group(
    group_id: UUID,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Get group details with members and areas (admin only)."""
    group = db.query(UserGroup).filter(UserGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")

    # Members
    members_rows = (
        db.query(User)
        .join(UserGroupMember, UserGroupMember.user_id == User.id)
        .filter(UserGroupMember.group_id == group_id)
        .all()
    )
    members = [
        UserResponse(
            id=str(u.id),
            username=u.username,
            display_name=u.display_name,
            email=u.email,
            role=u.role,
            is_active=u.is_active,
            last_login=u.last_login,
            created_at=u.created_at,
            updated_at=u.updated_at,
        )
        for u in members_rows
    ]

    # Areas
    areas = _get_group_areas(group_id, db)

    allowed_apps = _get_allowed_apps(group_id, db)

    return {
        "id": str(group.id),
        "name": group.name,
        "description": group.description,
        "created_by": str(group.created_by) if group.created_by else None,
        "allowed_apps": allowed_apps,
        "created_at": group.created_at,
        "updated_at": group.updated_at,
        "members": members,
        "areas": areas,
    }


# --------------------------------------------------------------------------
# PATCH /groups/{group_id}
# --------------------------------------------------------------------------
@router.patch("/{group_id}")
def update_group(
    group_id: UUID,
    body: GroupUpdate,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Update group name/description (admin only)."""
    group = db.query(UserGroup).filter(UserGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")

    updates = body.model_dump(exclude_unset=True)
    allowed_apps = updates.pop("allowed_apps", None)

    for field, value in updates.items():
        setattr(group, field, value)

    if allowed_apps is not None:
        _set_allowed_apps(group.id, allowed_apps, db)

    db.commit()
    db.refresh(group)

    log_action(
        db, admin, "update_group",
        resource_type="group", resource_id=str(group.id),
        details={**updates, **({"allowed_apps": allowed_apps} if allowed_apps is not None else {})},
        request=request,
    )

    return _group_to_response(group, db=db)


# --------------------------------------------------------------------------
# DELETE /groups/{group_id}
# --------------------------------------------------------------------------
@router.delete("/{group_id}")
def delete_group(
    group_id: UUID,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Delete a group (admin only). CASCADE handles members and areas."""
    group = db.query(UserGroup).filter(UserGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")

    group_name = group.name
    db.delete(group)
    db.commit()

    log_action(
        db, admin, "delete_group",
        resource_type="group", resource_id=str(group_id),
        details={"name": group_name},
        request=request,
    )

    return {"detail": "Group deleted"}


# ==========================================================================
# Members
# ==========================================================================

# --------------------------------------------------------------------------
# POST /groups/{group_id}/members
# --------------------------------------------------------------------------
@router.post("/{group_id}/members", status_code=status.HTTP_201_CREATED)
def add_member(
    group_id: UUID,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    user_id: UUID = Query(...),
):
    """Add a user to a group (admin only)."""
    group = db.query(UserGroup).filter(UserGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    existing = (
        db.query(UserGroupMember)
        .filter(UserGroupMember.group_id == group_id, UserGroupMember.user_id == user_id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User is already a member")

    membership = UserGroupMember(group_id=group_id, user_id=user_id)
    db.add(membership)
    db.commit()

    log_action(
        db, admin, "add_group_member",
        resource_type="group", resource_id=str(group_id),
        details={"user_id": str(user_id), "username": user.username},
        request=request,
    )

    return {"detail": "Member added"}


# --------------------------------------------------------------------------
# DELETE /groups/{group_id}/members/{user_id}
# --------------------------------------------------------------------------
@router.delete("/{group_id}/members/{user_id}")
def remove_member(
    group_id: UUID,
    user_id: UUID,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Remove a user from a group (admin only)."""
    membership = (
        db.query(UserGroupMember)
        .filter(UserGroupMember.group_id == group_id, UserGroupMember.user_id == user_id)
        .first()
    )
    if not membership:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Membership not found")

    db.delete(membership)
    db.commit()

    log_action(
        db, admin, "remove_group_member",
        resource_type="group", resource_id=str(group_id),
        details={"user_id": str(user_id)},
        request=request,
    )

    return {"detail": "Member removed"}


# ==========================================================================
# Area Restrictions
# ==========================================================================

def _get_group_areas(group_id: UUID, db: Session) -> list[dict]:
    """Fetch area restrictions for a group, with bbox as GeoJSON."""
    rows = (
        db.query(
            GroupAreaRestriction.id,
            GroupAreaRestriction.group_id,
            GroupAreaRestriction.name,
            GroupAreaRestriction.description,
            GroupAreaRestriction.access_level,
            GroupAreaRestriction.created_at,
            ST_AsGeoJSON(GroupAreaRestriction.bbox).label("geojson"),
        )
        .filter(GroupAreaRestriction.group_id == group_id)
        .all()
    )
    return [
        {
            "id": str(r.id),
            "group_id": str(r.group_id),
            "name": r.name,
            "description": r.description,
            "access_level": r.access_level,
            "created_at": r.created_at,
            "bbox": json.loads(r.geojson) if r.geojson else None,
        }
        for r in rows
    ]


# --------------------------------------------------------------------------
# GET /groups/{group_id}/areas
# --------------------------------------------------------------------------
@router.get("/{group_id}/areas")
def list_areas(
    group_id: UUID,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Get a group's area restrictions as GeoJSON (admin only)."""
    group = db.query(UserGroup).filter(UserGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")

    return {"areas": _get_group_areas(group_id, db)}


# --------------------------------------------------------------------------
# POST /groups/{group_id}/areas
# --------------------------------------------------------------------------
@router.post("/{group_id}/areas", status_code=status.HTTP_201_CREATED)
def create_area(
    group_id: UUID,
    body: AreaRestrictionCreate,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Add an area restriction to a group (admin only).

    Accepts bbox as a GeoJSON Polygon object.
    """
    group = db.query(UserGroup).filter(UserGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")

    if body.access_level not in VALID_ACCESS_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid access_level. Must be one of: {', '.join(sorted(VALID_ACCESS_LEVELS))}",
        )

    # Convert GeoJSON dict to PostGIS geometry
    geojson_str = json.dumps(body.bbox)

    area = GroupAreaRestriction(
        group_id=group_id,
        name=body.name,
        description=body.description,
        bbox=ST_GeomFromGeoJSON(geojson_str),
        access_level=body.access_level,
    )
    db.add(area)
    db.commit()
    db.refresh(area)

    log_action(
        db, admin, "create_area_restriction",
        resource_type="group_area", resource_id=str(area.id),
        details={"group_id": str(group_id), "name": body.name, "access_level": body.access_level},
        request=request,
    )

    # Return the area with bbox as GeoJSON
    geojson_result = db.query(ST_AsGeoJSON(area.bbox)).scalar()
    return {
        "id": str(area.id),
        "group_id": str(area.group_id),
        "name": area.name,
        "description": area.description,
        "access_level": area.access_level,
        "created_at": area.created_at,
        "bbox": json.loads(geojson_result) if geojson_result else None,
    }


# --------------------------------------------------------------------------
# PATCH /groups/{group_id}/areas/{area_id}
# --------------------------------------------------------------------------
@router.patch("/{group_id}/areas/{area_id}")
def update_area(
    group_id: UUID,
    area_id: UUID,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    name: str | None = Query(None),
    description: str | None = Query(None),
    access_level: str | None = Query(None),
):
    """Update an area restriction (admin only)."""
    area = (
        db.query(GroupAreaRestriction)
        .filter(GroupAreaRestriction.id == area_id, GroupAreaRestriction.group_id == group_id)
        .first()
    )
    if not area:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Area restriction not found")

    if access_level is not None and access_level not in VALID_ACCESS_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid access_level. Must be one of: {', '.join(sorted(VALID_ACCESS_LEVELS))}",
        )

    changes = {}
    if name is not None:
        area.name = name
        changes["name"] = name
    if description is not None:
        area.description = description
        changes["description"] = description
    if access_level is not None:
        area.access_level = access_level
        changes["access_level"] = access_level

    db.commit()
    db.refresh(area)

    log_action(
        db, admin, "update_area_restriction",
        resource_type="group_area", resource_id=str(area.id),
        details=changes,
        request=request,
    )

    geojson_result = db.query(ST_AsGeoJSON(area.bbox)).scalar()
    return {
        "id": str(area.id),
        "group_id": str(area.group_id),
        "name": area.name,
        "description": area.description,
        "access_level": area.access_level,
        "created_at": area.created_at,
        "bbox": json.loads(geojson_result) if geojson_result else None,
    }


# --------------------------------------------------------------------------
# DELETE /groups/{group_id}/areas/{area_id}
# --------------------------------------------------------------------------
@router.delete("/{group_id}/areas/{area_id}")
def delete_area(
    group_id: UUID,
    area_id: UUID,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Remove an area restriction (admin only)."""
    area = (
        db.query(GroupAreaRestriction)
        .filter(GroupAreaRestriction.id == area_id, GroupAreaRestriction.group_id == group_id)
        .first()
    )
    if not area:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Area restriction not found")

    area_name = area.name
    db.delete(area)
    db.commit()

    log_action(
        db, admin, "delete_area_restriction",
        resource_type="group_area", resource_id=str(area_id),
        details={"group_id": str(group_id), "name": area_name},
        request=request,
    )

    return {"detail": "Area restriction deleted"}
