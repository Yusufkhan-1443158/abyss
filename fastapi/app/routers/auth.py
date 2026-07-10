"""Authentication router — login, refresh, logout, and current-user endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..middleware.area_check import get_user_allowed_areas
from ..middleware.audit_logger import log_action
from ..middleware.auth_middleware import (
    blacklist_token,
    get_current_user,
    is_token_blacklisted,
)
from ..middleware.rate_limit import check_login_rate_limit
from ..models import User
from ..schemas import LoginRequest, Token, UserMeResponse

router = APIRouter()
settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.now(timezone.utc) + expires_delta
    to_encode["jti"] = str(uuid.uuid4())  # unique token ID for blacklisting
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.JWT_ALGORITHM)


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    """Set HttpOnly cookies for both tokens.

    ``Secure`` is driven by COOKIE_SECURE (true operationally over HTTPS, false
    in the HTTP-only dev overlay); SameSite by COOKIE_SAMESITE.
    """
    secure = settings.COOKIE_SECURE
    samesite = settings.COOKIE_SAMESITE
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite=samesite,
        max_age=settings.JWT_ACCESS_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=secure,
        samesite=samesite,
        max_age=settings.JWT_REFRESH_EXPIRE_DAYS * 86400,
        path="/api/auth",  # only sent to auth endpoints
    )


def _clear_auth_cookies(response: Response) -> None:
    """Clear auth cookies by setting max_age=0."""
    response.delete_cookie(key="access_token", path="/")
    response.delete_cookie(key="refresh_token", path="/api/auth")


class RefreshRequest(BaseModel):
    refresh_token: str | None = None  # Optional — can also come from cookie


# --------------------------------------------------------------------------
# POST /login
# --------------------------------------------------------------------------
@router.post("/login", response_model=Token)
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
):
    """Authenticate with username/password and receive JWT tokens."""
    # Strict per-IP brute-force limit (the global default limit also applies).
    check_login_rate_limit(request)

    user = db.query(User).filter(User.username == body.username).first()

    if user is None or not pwd_context.verify(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Build JWT claims
    claims = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
    }

    access_token = _create_token(
        {**claims, "type": "access"},
        timedelta(minutes=settings.JWT_ACCESS_EXPIRE_MINUTES),
    )
    refresh_token = _create_token(
        {**claims, "type": "refresh"},
        timedelta(days=settings.JWT_REFRESH_EXPIRE_DAYS),
    )

    # Update last_login
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    log_action(db, user, "login", request=request)

    # HIGH-03: Set HttpOnly cookies
    _set_auth_cookies(response, access_token, refresh_token)

    # Still return JSON body for backward compatibility
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
    )


# --------------------------------------------------------------------------
# POST /refresh
# --------------------------------------------------------------------------
@router.post("/refresh", response_model=Token)
def refresh(
    body: RefreshRequest,
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
):
    """Exchange a valid refresh token for a new access token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Accept refresh token from body or cookie
    raw_refresh = body.refresh_token or request.cookies.get("refresh_token")
    if not raw_refresh:
        raise credentials_exception

    try:
        payload = jwt.decode(
            raw_refresh,
            settings.jwt_secret_key,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("type") != "refresh":
            raise credentials_exception

        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception

        # HIGH-04: Check if old refresh token is blacklisted
        old_jti = payload.get("jti")
        if old_jti and is_token_blacklisted(old_jti):
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        raise credentials_exception

    claims = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
    }

    access_token = _create_token(
        {**claims, "type": "access"},
        timedelta(minutes=settings.JWT_ACCESS_EXPIRE_MINUTES),
    )
    # Also issue a fresh refresh token (rotation)
    new_refresh = _create_token(
        {**claims, "type": "refresh"},
        timedelta(days=settings.JWT_REFRESH_EXPIRE_DAYS),
    )

    # HIGH-04: Blacklist the old refresh token
    if old_jti:
        exp = payload.get("exp", 0)
        ttl = max(int(exp - datetime.now(timezone.utc).timestamp()), 0)
        blacklist_token(old_jti, ttl)

    # HIGH-03: Set new cookies
    _set_auth_cookies(response, access_token, new_refresh)

    return Token(access_token=access_token, refresh_token=new_refresh)


# --------------------------------------------------------------------------
# POST /logout
# --------------------------------------------------------------------------
@router.post("/logout")
def logout(
    current_user: Annotated[User, Depends(get_current_user)],
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
):
    """Log out — blacklist current tokens and clear cookies."""
    # HIGH-04: Blacklist the access token
    auth_header = request.headers.get("authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        token = request.cookies.get("access_token")

    if token:
        try:
            payload = jwt.decode(
                token, settings.jwt_secret_key,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": False},
            )
            jti = payload.get("jti")
            if jti:
                exp = payload.get("exp", 0)
                ttl = max(int(exp - datetime.now(timezone.utc).timestamp()), 0)
                blacklist_token(jti, ttl if ttl > 0 else 3600)
        except JWTError:
            pass

    # Also blacklist the refresh token from cookie
    refresh_cookie = request.cookies.get("refresh_token")
    if refresh_cookie:
        try:
            payload = jwt.decode(
                refresh_cookie, settings.jwt_secret_key,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": False},
            )
            jti = payload.get("jti")
            if jti:
                exp = payload.get("exp", 0)
                ttl = max(int(exp - datetime.now(timezone.utc).timestamp()), 0)
                blacklist_token(jti, ttl if ttl > 0 else 86400)
        except JWTError:
            pass

    log_action(db, current_user, "logout", request=request)

    # HIGH-03: Clear cookies
    _clear_auth_cookies(response)

    return {"detail": "Logged out successfully"}


# --------------------------------------------------------------------------
# GET /me
# --------------------------------------------------------------------------
@router.get("/me", response_model=UserMeResponse)
def me(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Return the currently authenticated user's profile."""
    areas = get_user_allowed_areas(current_user, db)
    return UserMeResponse(
        id=str(current_user.id),
        username=current_user.username,
        display_name=current_user.display_name,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        last_login=current_user.last_login,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        preferences=current_user.preferences or {},
        areas=areas,
    )


# --------------------------------------------------------------------------
# GET /me/areas
# --------------------------------------------------------------------------
@router.get("/me/areas")
def me_areas(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Return allowed bounding boxes for the current user."""
    areas = get_user_allowed_areas(current_user, db)
    return {"areas": areas}
