"""Authentication and authorization dependencies for FastAPI."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated

import redis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models import User

settings = get_settings()
logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# ---------------------------------------------------------------------------
# Redis connection for token blacklist
# ---------------------------------------------------------------------------

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """Lazy-init a Redis client for token blacklist operations."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


def is_token_blacklisted(jti: str) -> bool:
    """Check if a token JTI is in the Redis blacklist.

    On a Redis error the policy is governed by BLACKLIST_FAIL_OPEN: fail-open
    (default) returns False — treat as not-revoked, favouring availability;
    fail-closed returns True — treat as revoked, favouring strict revocation at
    the cost of locking everyone out while Redis is down.
    """
    try:
        r = get_redis()
        return r.exists(f"bl:{jti}") == 1
    except redis.RedisError as exc:
        logger.warning(
            "Redis blacklist check failed (jti=%s): %s — fail_open=%s",
            jti, exc, settings.BLACKLIST_FAIL_OPEN,
        )
        return not settings.BLACKLIST_FAIL_OPEN


def blacklist_token(jti: str, ttl_seconds: int) -> None:
    """Add a token JTI to the Redis blacklist with TTL."""
    try:
        r = get_redis()
        r.setex(f"bl:{jti}", ttl_seconds, "1")
    except redis.RedisError:
        pass  # Best-effort; log in production


# ---------------------------------------------------------------------------
# Token extraction: Bearer header first, then HttpOnly cookie
# ---------------------------------------------------------------------------

def _extract_token(
    token_from_header: str | None,
    request: Request,
) -> str:
    """Return the JWT access token from Authorization header or cookie."""
    if token_from_header:
        return token_from_header

    # Fallback: read from HttpOnly cookie
    token_from_cookie = request.cookies.get("access_token")
    if token_from_cookie:
        return token_from_cookie

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Current user dependency
# ---------------------------------------------------------------------------

def _dev_user(db: Session) -> User:
    """Return the seeded admin for the DEV_NO_AUTH bypass (real users row)."""
    user = db.query(User).filter(User.id == settings.DEV_USER_ID).first()
    if user is None:
        user = (
            db.query(User)
            .filter(User.role == "admin", User.is_active.is_(True))
            .first()
        )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DEV_NO_AUTH is on but no admin user is seeded yet",
        )
    return user


def get_current_user(
    request: Request,
    token_from_header: Annotated[str | None, Depends(oauth2_scheme)] = None,
    db: Session = Depends(get_db),
) -> User:
    """Decode JWT, validate claims, check blacklist, fetch and return the User ORM object."""
    # Dev bypass: skip JWT entirely and act as the seeded admin.
    if settings.DEV_NO_AUTH:
        return _dev_user(db)

    token = _extract_token(token_from_header, request)

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user_id: str | None = payload.get("sub")
        token_type: str | None = payload.get("type")

        if user_id is None or token_type != "access":
            raise credentials_exception

        # Check expiry (python-jose checks this, but be explicit)
        exp = payload.get("exp")
        if exp is not None and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
            raise credentials_exception

        # HIGH-04: Check Redis blacklist
        jti = payload.get("jti")
        if jti and is_token_blacklisted(jti):
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        raise credentials_exception

    return user


def require_role(required_roles: list[str]):
    """Return a dependency that checks the current user has one of the required roles."""

    def role_checker(
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return role_checker


require_admin = require_role(["admin"])
require_analyst_or_admin = require_role(["admin", "analyst"])
