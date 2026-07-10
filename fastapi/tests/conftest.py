"""Pytest fixtures for the Abyss auth suite.

Runs in-container against the live PostgreSQL + Redis (the ORM uses PostGIS/JSONB,
so SQLite is not an option):

    docker compose exec fastapi pytest

Auth must be enforced (DEV_NO_AUTH=false) for these tests to mean anything — in a
dev container they skip (see the module skip in test_auth.py).
"""

from __future__ import annotations

import pytest
import redis as redis_lib
from fastapi.testclient import TestClient
from passlib.context import CryptContext
from sqlalchemy import text

from app.config import get_settings
from app.database import SessionLocal
from app.main import app

settings = get_settings()
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dedicated, deterministic test accounts — kept separate from the demo seed so the
# suite is independent of the seed secrets / SEED_FORCE.
TEST_USERS = [
    ("pytest_admin", "PytestAdmin!123", "admin"),
    ("pytest_analyst", "PytestAnalyst!123", "analyst"),
    ("pytest_viewer", "PytestViewer!123", "viewer"),
]
PASSWORDS = {u: p for u, p, _ in TEST_USERS}


@pytest.fixture(scope="session", autouse=True)
def _seed_test_users():
    """Upsert the test accounts before the suite; best-effort cleanup after."""
    if settings.DEV_NO_AUTH:
        yield
        return
    db = SessionLocal()
    try:
        for username, pw, role in TEST_USERS:
            db.execute(
                text(
                    """
                    INSERT INTO users (username, password_hash, display_name, role, is_active)
                    VALUES (:u, :pw, :d, :r, true)
                    ON CONFLICT (username) DO UPDATE
                        SET password_hash = EXCLUDED.password_hash,
                            role = EXCLUDED.role,
                            is_active = true
                    """
                ),
                {"u": username, "pw": _pwd.hash(pw), "d": username, "r": role},
            )
        db.commit()
    finally:
        db.close()

    yield

    # Best-effort teardown — ignore FK fallout (e.g. audit_log rows). The accounts
    # are upserted each run, so leaving them is harmless.
    db = SessionLocal()
    try:
        db.execute(
            text("DELETE FROM users WHERE username = ANY(:names)"),
            {"names": [u for u, _, _ in TEST_USERS]},
        )
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


@pytest.fixture(scope="session")
def client():
    # The context manager runs the app lifespan (validate_operational + seeding).
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset(client):
    """Fresh rate-limit bucket (slowapi Redis db /4) + cookie jar per test."""
    try:
        redis_lib.from_url(settings.rate_limit_redis_url).flushdb()
    except Exception:
        pass
    client.cookies.clear()
    yield


@pytest.fixture
def login(client):
    """Return a helper: login(username[, password]) -> Response."""

    def _login(username: str, password: str | None = None):
        password = PASSWORDS[username] if password is None else password
        return client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )

    return _login
