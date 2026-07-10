"""Audit log router — admin-only query, export, and stats."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.auth_middleware import require_admin
from ..models import AuditLog, User
from ..schemas import AuditLogResponse

router = APIRouter()


def _build_audit_query(
    db: Session,
    user_id: str | None = None,
    action: str | None = None,
    resource_type: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
):
    """Build a filtered query on the audit_log table."""
    q = db.query(AuditLog)
    if user_id is not None:
        q = q.filter(AuditLog.user_id == user_id)
    if action is not None:
        q = q.filter(AuditLog.action == action)
    if resource_type is not None:
        q = q.filter(AuditLog.resource_type == resource_type)
    if date_from is not None:
        q = q.filter(AuditLog.timestamp >= date_from)
    if date_to is not None:
        q = q.filter(AuditLog.timestamp <= date_to)
    return q


def _log_to_response(entry: AuditLog) -> dict:
    return {
        "id": entry.id,
        "user_id": str(entry.user_id) if entry.user_id else None,
        "username": entry.username,
        "action": entry.action,
        "resource_type": entry.resource_type,
        "resource_id": entry.resource_id,
        "details": entry.details or {},
        "ip_address": str(entry.ip_address) if entry.ip_address else None,
        "user_agent": entry.user_agent,
        "timestamp": entry.timestamp,
    }


# --------------------------------------------------------------------------
# GET /audit
# --------------------------------------------------------------------------
@router.get("")
def query_audit_log(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    user_id: str | None = Query(None),
    action: str | None = Query(None),
    resource_type: str | None = Query(None),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    """Query audit log entries (admin only). Supports filtering and pagination."""
    q = _build_audit_query(db, user_id, action, resource_type, date_from, date_to)

    total = q.count()
    entries = q.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [_log_to_response(e) for e in entries],
    }


# --------------------------------------------------------------------------
# GET /audit/export
# --------------------------------------------------------------------------
@router.get("/export")
def export_audit_log(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    user_id: str | None = Query(None),
    action: str | None = Query(None),
    resource_type: str | None = Query(None),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
):
    """Export filtered audit log as CSV (admin only)."""
    q = _build_audit_query(db, user_id, action, resource_type, date_from, date_to)
    entries = q.order_by(AuditLog.timestamp.desc()).all()

    def generate_csv():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "user_id", "username", "action",
            "resource_type", "resource_id", "details",
            "ip_address", "user_agent", "timestamp",
        ])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for entry in entries:
            writer.writerow([
                entry.id,
                str(entry.user_id) if entry.user_id else "",
                entry.username or "",
                entry.action,
                entry.resource_type or "",
                entry.resource_id or "",
                str(entry.details) if entry.details else "",
                str(entry.ip_address) if entry.ip_address else "",
                entry.user_agent or "",
                entry.timestamp.isoformat() if entry.timestamp else "",
            ])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    return StreamingResponse(
        generate_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_log.csv"},
    )


# --------------------------------------------------------------------------
# GET /audit/stats
# --------------------------------------------------------------------------
@router.get("/stats")
def audit_stats(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
):
    """Summary statistics for the audit log (admin only)."""
    base = db.query(AuditLog)
    if date_from is not None:
        base = base.filter(AuditLog.timestamp >= date_from)
    if date_to is not None:
        base = base.filter(AuditLog.timestamp <= date_to)

    total_entries = base.count()

    # Action counts
    action_rows = (
        base.with_entities(AuditLog.action, func.count(AuditLog.id))
        .group_by(AuditLog.action)
        .order_by(func.count(AuditLog.id).desc())
        .all()
    )
    action_counts = {action: count for action, count in action_rows}

    # Most active users
    user_rows = (
        base.with_entities(AuditLog.username, func.count(AuditLog.id))
        .filter(AuditLog.username.isnot(None))
        .group_by(AuditLog.username)
        .order_by(func.count(AuditLog.id).desc())
        .limit(10)
        .all()
    )
    most_active_users = [{"username": uname, "count": count} for uname, count in user_rows]

    # Recent activity (last 10 entries)
    recent = (
        base.order_by(AuditLog.timestamp.desc())
        .limit(10)
        .all()
    )

    return {
        "total_entries": total_entries,
        "action_counts": action_counts,
        "most_active_users": most_active_users,
        "recent_activity": [_log_to_response(e) for e in recent],
    }
