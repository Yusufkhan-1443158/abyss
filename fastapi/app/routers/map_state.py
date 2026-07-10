"""Saved map state (views) endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.auth_middleware import get_current_user
from ..models import User, UserMapState
from ..schemas import MapStateCreate, MapStateResponse, MapStateUpdate

router = APIRouter()


# ---------------------------------------------------------------------------
# List saved states
# ---------------------------------------------------------------------------

@router.get("", response_model=list[MapStateResponse])
async def list_map_states(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List current user's saved map states."""
    states = (
        db.query(UserMapState)
        .filter(UserMapState.user_id == current_user.id)
        .order_by(UserMapState.updated_at.desc())
        .all()
    )

    return [
        MapStateResponse(
            id=str(s.id),
            user_id=str(s.user_id),
            name=s.name,
            center_lon=s.center_lon,
            center_lat=s.center_lat,
            zoom=s.zoom,
            bearing=s.bearing or 0,
            pitch=s.pitch or 0,
            layer_state=s.layer_state or [],
            is_default=s.is_default,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in states
    ]


# ---------------------------------------------------------------------------
# Save new state
# ---------------------------------------------------------------------------

@router.post("", response_model=MapStateResponse, status_code=201)
async def save_map_state(
    body: MapStateCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a new map state (camera + layer_state JSONB)."""
    # If this state is marked as default, unset other defaults for user
    if body.is_default:
        db.query(UserMapState).filter(
            UserMapState.user_id == current_user.id,
            UserMapState.is_default == True,  # noqa: E712
        ).update({"is_default": False})

    state = UserMapState(
        user_id=current_user.id,
        name=body.name,
        center_lon=body.center_lon,
        center_lat=body.center_lat,
        zoom=body.zoom,
        bearing=body.bearing,
        pitch=body.pitch,
        layer_state=body.layer_state,
        is_default=body.is_default,
    )
    db.add(state)
    db.commit()
    db.refresh(state)

    return MapStateResponse(
        id=str(state.id),
        user_id=str(state.user_id),
        name=state.name,
        center_lon=state.center_lon,
        center_lat=state.center_lat,
        zoom=state.zoom,
        bearing=state.bearing or 0,
        pitch=state.pitch or 0,
        layer_state=state.layer_state or [],
        is_default=state.is_default,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


# ---------------------------------------------------------------------------
# Get saved state
# ---------------------------------------------------------------------------

@router.get("/{state_id}", response_model=MapStateResponse)
async def get_map_state(
    state_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a saved state. Owner only."""
    state = db.query(UserMapState).filter(UserMapState.id == state_id).first()
    if not state:
        raise HTTPException(status_code=404, detail="Map state not found")

    if str(state.user_id) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner")

    return MapStateResponse(
        id=str(state.id),
        user_id=str(state.user_id),
        name=state.name,
        center_lon=state.center_lon,
        center_lat=state.center_lat,
        zoom=state.zoom,
        bearing=state.bearing or 0,
        pitch=state.pitch or 0,
        layer_state=state.layer_state or [],
        is_default=state.is_default,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


# ---------------------------------------------------------------------------
# Update saved state
# ---------------------------------------------------------------------------

@router.patch("/{state_id}", response_model=MapStateResponse)
async def update_map_state(
    state_id: str,
    body: MapStateUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a saved state. Owner only."""
    state = db.query(UserMapState).filter(UserMapState.id == state_id).first()
    if not state:
        raise HTTPException(status_code=404, detail="Map state not found")

    if str(state.user_id) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner")

    if body.name is not None:
        state.name = body.name
    if body.center_lon is not None:
        state.center_lon = body.center_lon
    if body.center_lat is not None:
        state.center_lat = body.center_lat
    if body.zoom is not None:
        state.zoom = body.zoom
    if body.bearing is not None:
        state.bearing = body.bearing
    if body.pitch is not None:
        state.pitch = body.pitch
    if body.layer_state is not None:
        state.layer_state = body.layer_state
    if body.is_default is not None:
        if body.is_default:
            # Unset other defaults for user
            db.query(UserMapState).filter(
                UserMapState.user_id == current_user.id,
                UserMapState.is_default == True,  # noqa: E712
                UserMapState.id != state.id,
            ).update({"is_default": False})
        state.is_default = body.is_default

    db.commit()
    db.refresh(state)

    return MapStateResponse(
        id=str(state.id),
        user_id=str(state.user_id),
        name=state.name,
        center_lon=state.center_lon,
        center_lat=state.center_lat,
        zoom=state.zoom,
        bearing=state.bearing or 0,
        pitch=state.pitch or 0,
        layer_state=state.layer_state or [],
        is_default=state.is_default,
        created_at=state.created_at,
        updated_at=state.updated_at,
    )


# ---------------------------------------------------------------------------
# Delete saved state
# ---------------------------------------------------------------------------

@router.delete("/{state_id}", status_code=204)
async def delete_map_state(
    state_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a saved state. Owner only."""
    state = db.query(UserMapState).filter(UserMapState.id == state_id).first()
    if not state:
        raise HTTPException(status_code=404, detail="Map state not found")

    if str(state.user_id) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner")

    db.delete(state)
    db.commit()

    return Response(status_code=204)
