"""Area-based access control using PostGIS spatial queries."""

from __future__ import annotations

import json

from geoalchemy2 import WKBElement
from geoalchemy2.functions import ST_AsGeoJSON, ST_Intersects, ST_Union
from sqlalchemy.orm import Query, Session

from ..models import GroupAreaRestriction, User, UserGroupMember

# Access level hierarchy: manage > download > view
_LEVEL_HIERARCHY = {"view": 0, "download": 1, "manage": 2}


def verify_area_access(
    user: User,
    bbox: WKBElement,
    required_level: str = "view",
    db: Session | None = None,
) -> bool:
    """Check whether a user has spatial access at the required level.

    Admins always pass. For others, checks if any of the user's group area
    restrictions intersect the given bbox at or above the required level.
    """
    if user.role == "admin":
        return True

    if db is None:
        return False

    required_rank = _LEVEL_HIERARCHY.get(required_level, 0)

    # Find all area restrictions through user's group memberships
    rows = (
        db.query(GroupAreaRestriction.access_level)
        .join(
            UserGroupMember,
            UserGroupMember.group_id == GroupAreaRestriction.group_id,
        )
        .filter(
            UserGroupMember.user_id == user.id,
            ST_Intersects(GroupAreaRestriction.bbox, bbox),
        )
        .all()
    )

    for (level,) in rows:
        if _LEVEL_HIERARCHY.get(level, 0) >= required_rank:
            return True

    return False


def get_user_allowed_areas(user: User, db: Session) -> list[dict]:
    """Return list of allowed area geometries as GeoJSON dicts for the user.

    Admin gets a special indicator instead of explicit areas.
    """
    if user.role == "admin":
        return [{"global_access": True}]

    rows = (
        db.query(
            GroupAreaRestriction.id,
            GroupAreaRestriction.name,
            GroupAreaRestriction.access_level,
            ST_AsGeoJSON(GroupAreaRestriction.bbox).label("geojson"),
        )
        .join(
            UserGroupMember,
            UserGroupMember.group_id == GroupAreaRestriction.group_id,
        )
        .filter(UserGroupMember.user_id == user.id)
        .all()
    )

    return [
        {
            "id": str(r.id),
            "name": r.name,
            "access_level": r.access_level,
            "bbox": json.loads(r.geojson),
        }
        for r in rows
    ]


def filter_by_area(
    query: Query,
    user: User,
    bbox_column,
    db: Session,
) -> Query:
    """Add a WHERE clause restricting results to the user's allowed areas.

    Admin bypasses all area filtering.
    """
    if user.role == "admin":
        return query

    # Subquery: union of all bboxes the user has access to
    allowed_union = (
        db.query(ST_Union(GroupAreaRestriction.bbox).label("allowed"))
        .join(
            UserGroupMember,
            UserGroupMember.group_id == GroupAreaRestriction.group_id,
        )
        .filter(UserGroupMember.user_id == user.id)
        .scalar_subquery()
    )

    return query.filter(ST_Intersects(bbox_column, allowed_union))
