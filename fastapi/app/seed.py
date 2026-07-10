"""Idempotent startup seeding for Abyss demo accounts.

Seeds the admin / analyst / viewer accounts so that (a) the DEV_NO_AUTH bypass
has a real admin row to return (FK targets like ``projects.created_by`` resolve),
and (b) real JWT login + RBAC are demonstrable out of the box.

This is the single source of truth for the demo credentials — migration 002 only
creates the table DDL. When ``SEED_FORCE`` is true (default) each account is
upserted (``ON CONFLICT (username) DO UPDATE``) on every boot so credentials stay
deterministic across restarts; set it false once accounts are admin-managed so
manual password changes survive. Admin keeps the fixed UUID ``0000…0001`` so the
``_dev_user()`` lookup resolves by id.
"""

from __future__ import annotations

import uuid

from passlib.context import CryptContext
from sqlalchemy import text

from .config import get_settings
from .database import SessionLocal

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

_ADMIN_UUID = "00000000-0000-0000-0000-000000000001"
_ANALYST_UUID = "00000000-0000-0000-0000-000000000002"
_VIEWER_UUID = "00000000-0000-0000-0000-000000000003"

# INSERT … ON CONFLICT variants. DO UPDATE makes credentials deterministic
# (SEED_FORCE=true); DO NOTHING preserves admin-managed rows (SEED_FORCE=false).
_UPSERT_FORCE = text(
    """
    INSERT INTO users (id, username, password_hash, display_name, email, role, is_active)
    VALUES (:id, :username, :pw, :display, :email, :role, true)
    ON CONFLICT (username) DO UPDATE
        SET password_hash = EXCLUDED.password_hash,
            role = EXCLUDED.role,
            is_active = true
    """
)
_INSERT_ONLY = text(
    """
    INSERT INTO users (id, username, password_hash, display_name, email, role, is_active)
    VALUES (:id, :username, :pw, :display, :email, :role, true)
    ON CONFLICT (username) DO NOTHING
    """
)


def ensure_seed_users() -> None:
    """Seed (or refresh) the admin / analyst / viewer demo accounts. Best-effort."""
    settings = get_settings()
    if not settings.SEED_ADMIN:
        return

    try:
        admin_uuid = str(uuid.UUID(settings.DEV_USER_ID))
    except (ValueError, TypeError):
        admin_uuid = _ADMIN_UUID

    # (username, password, role, display, email, fixed_uuid)
    accounts = [
        (settings.DEV_USERNAME, settings.seed_admin_password, "admin",
         "Abyss Admin", "admin@abyss.local", admin_uuid),
        ("analyst", settings.seed_analyst_password, "analyst",
         "Abyss Analyst", "analyst@abyss.local", _ANALYST_UUID),
        ("viewer", settings.seed_viewer_password, "viewer",
         "Abyss Viewer", "viewer@abyss.local", _VIEWER_UUID),
    ]

    stmt = _UPSERT_FORCE if settings.SEED_FORCE else _INSERT_ONLY

    db = SessionLocal()
    try:
        for username, password, role, display, email, fixed_uuid in accounts:
            if not password:
                # No password configured for this role (e.g. analyst/viewer in an
                # unconfigured deploy) — skip rather than create a broken account.
                continue
            db.execute(
                stmt,
                {
                    "id": fixed_uuid,
                    "username": username,
                    "pw": _pwd.hash(password),
                    "display": display,
                    "email": email,
                    "role": role,
                },
            )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
