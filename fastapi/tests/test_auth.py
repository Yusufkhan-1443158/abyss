"""Auth + RBAC regression suite.

Covers login (success / bad password), the login rate limit (429), refresh-token
rotation + reuse rejection, logout blacklisting the access token, and RBAC
(viewer forbidden from the admin user list). Requires enforcement — skipped when
DEV_NO_AUTH is on.
"""

from __future__ import annotations

import pytest

from app.config import get_settings

pytestmark = pytest.mark.skipif(
    get_settings().DEV_NO_AUTH,
    reason="auth tests require enforcement (DEV_NO_AUTH=false)",
)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_login_success(login):
    r = login("pytest_admin")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["access_token"] and body["refresh_token"]
    # HttpOnly cookies are set on the response.
    assert "access_token" in r.cookies


def test_login_bad_password(login):
    r = login("pytest_admin", password="not-the-password")
    assert r.status_code == 401


def test_login_rate_limit(login):
    # RATE_LIMIT_LOGIN is 5/minute operationally → the 6th rapid attempt is 429.
    statuses = [login("pytest_admin", password="x").status_code for _ in range(6)]
    assert 429 in statuses, statuses


def test_refresh_rotation_and_reuse(login, client):
    r = login("pytest_analyst")
    assert r.status_code == 200, r.text
    refresh1 = r.json()["refresh_token"]

    # First refresh rotates the token and blacklists the old one.
    r2 = client.post("/api/auth/refresh", json={"refresh_token": refresh1})
    assert r2.status_code == 200, r2.text
    refresh2 = r2.json()["refresh_token"]
    assert refresh2 and refresh2 != refresh1

    # Replaying the rotated-out refresh token must be rejected.
    r3 = client.post("/api/auth/refresh", json={"refresh_token": refresh1})
    assert r3.status_code == 401


def test_logout_blacklists_access(login, client):
    token = login("pytest_admin").json()["access_token"]
    hdr = _bearer(token)

    assert client.get("/api/auth/me", headers=hdr).status_code == 200
    assert client.post("/api/auth/logout", headers=hdr).status_code == 200
    # The blacklisted access token is now rejected.
    assert client.get("/api/auth/me", headers=hdr).status_code == 401


def test_rbac_viewer_forbidden(login, client):
    token = login("pytest_viewer").json()["access_token"]
    # GET /api/users requires admin → viewer gets 403.
    assert client.get("/api/users", headers=_bearer(token)).status_code == 403
