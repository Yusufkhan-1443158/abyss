"""User management router — admin-only CRUD for user accounts."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.audit_logger import log_action
from ..middleware.auth_middleware import require_admin
from ..models import User
from ..schemas import UserCreate, UserResponse, UserUpdate

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

VALID_ROLES = {"admin", "analyst", "viewer"}


def _user_to_response(user: User) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        username=user.username,
        display_name=user.display_name,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        last_login=user.last_login,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


# --------------------------------------------------------------------------
# GET /users
# --------------------------------------------------------------------------
@router.get("")
def list_users(
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    role: str | None = Query(None),
    is_active: bool | None = Query(None),
):
    """List all users (admin only). Supports filtering by role and active status."""
    q = db.query(User)
    if role is not None:
        q = q.filter(User.role == role)
    if is_active is not None:
        q = q.filter(User.is_active == is_active)

    total = q.count()
    users = q.order_by(User.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [_user_to_response(u) for u in users],
    }


# --------------------------------------------------------------------------
# POST /users
# --------------------------------------------------------------------------
@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    body: UserCreate,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Create a new user (admin only)."""
    if body.role not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join(sorted(VALID_ROLES))}",
        )

    existing = db.query(User).filter(User.username == body.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )

    user = User(
        username=body.username,
        password_hash=pwd_context.hash(body.password),
        display_name=body.display_name,
        email=body.email,
        role=body.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    log_action(
        db, admin, "create_user",
        resource_type="user", resource_id=str(user.id),
        details={"username": user.username, "role": user.role},
        request=request,
    )

    return _user_to_response(user)


# --------------------------------------------------------------------------
# GET /users/{user_id}
# --------------------------------------------------------------------------
@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: UUID,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Get user details (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return _user_to_response(user)


# --------------------------------------------------------------------------
# PATCH /users/{user_id}
# --------------------------------------------------------------------------
@router.patch("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: UUID,
    body: UserUpdate,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Update a user's profile fields (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    updates = body.model_dump(exclude_unset=True)

    if "role" in updates and updates["role"] not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join(sorted(VALID_ROLES))}",
        )

    # Handle password re-hashing separately
    new_password = updates.pop("password", None)
    if new_password is not None:
        user.password_hash = pwd_context.hash(new_password)

    for field, value in updates.items():
        setattr(user, field, value)

    db.commit()
    db.refresh(user)

    log_action(
        db, admin, "update_user",
        resource_type="user", resource_id=str(user.id),
        details=updates,
        request=request,
    )

    return _user_to_response(user)


# --------------------------------------------------------------------------
# DELETE /users/{user_id}
# --------------------------------------------------------------------------
@router.delete("/{user_id}")
def deactivate_user(
    user_id: UUID,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
):
    """Deactivate a user (admin only). Does not delete the record."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account",
        )

    user.is_active = False
    db.commit()

    log_action(
        db, admin, "deactivate_user",
        resource_type="user", resource_id=str(user.id),
        details={"username": user.username},
        request=request,
    )

    return {"detail": "User deactivated"}
