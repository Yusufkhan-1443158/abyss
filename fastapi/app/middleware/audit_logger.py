"""Audit logging utility for tracking user actions."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from ..models import AuditLog, User


def log_action(
    db: Session,
    user: User | None,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    details: dict[str, Any] | None = None,
    request: Request | None = None,
) -> None:
    """Insert a row into the audit_log table."""
    ip_address = None
    user_agent = None

    if request is not None:
        # X-Forwarded-For for proxied requests, fall back to client host
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ip_address = forwarded.split(",")[0].strip()
        elif request.client:
            ip_address = request.client.host
        user_agent = request.headers.get("user-agent")

    entry = AuditLog(
        user_id=user.id if user else None,
        username=user.username if user else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details or {},
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(entry)
    db.commit()
