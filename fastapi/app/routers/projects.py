"""Project workspace endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.auth_middleware import get_current_user
from ..models import Project, User
from ..schemas import ProjectCreate, ProjectResponse, ProjectUpdate

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_to_response(p: Project) -> ProjectResponse:
    return ProjectResponse(
        id=str(p.id),
        name=p.name,
        description=p.description or "",
        owner_id=str(p.owner_id),
        camera_state=p.camera_state or {},
        raster_ids=[str(x) for x in (p.raster_ids or [])],
        vector_ids=[str(x) for x in (p.vector_ids or [])],
        annotation_ids=[str(x) for x in (p.annotation_ids or [])],
        is_default=p.is_default or False,
        project_type=p.project_type or "single",
        auto_ingest=p.auto_ingest or False,
        auto_ingest_config=p.auto_ingest_config or {},
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


# ---------------------------------------------------------------------------
# List projects
# ---------------------------------------------------------------------------

@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List current user's projects."""
    projects = (
        db.query(Project)
        .filter(Project.owner_id == current_user.id)
        .order_by(Project.updated_at.desc())
        .all()
    )
    return [_project_to_response(p) for p in projects]


# ---------------------------------------------------------------------------
# Create project
# ---------------------------------------------------------------------------

@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    body: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new project. Owner is set from JWT."""
    project = Project(
        name=body.name,
        description=body.description,
        owner_id=current_user.id,
        project_type=body.project_type,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _project_to_response(project)


# ---------------------------------------------------------------------------
# Get project detail
# ---------------------------------------------------------------------------

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a project. Owner or admin only."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project.owner_id) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner")

    return _project_to_response(project)


# ---------------------------------------------------------------------------
# Update project
# ---------------------------------------------------------------------------

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    body: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a project. Owner only."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project.owner_id) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner")

    if body.name is not None:
        project.name = body.name
    if body.description is not None:
        project.description = body.description
    if body.camera_state is not None:
        project.camera_state = body.camera_state
    if body.raster_ids is not None:
        project.raster_ids = body.raster_ids
    if body.vector_ids is not None:
        project.vector_ids = body.vector_ids
    if body.annotation_ids is not None:
        project.annotation_ids = body.annotation_ids
    if body.project_type is not None:
        project.project_type = body.project_type
    if body.is_default is not None:
        if body.is_default:
            # Unset other defaults for user
            db.query(Project).filter(
                Project.owner_id == current_user.id,
                Project.is_default == True,  # noqa: E712
                Project.id != project.id,
            ).update({"is_default": False})
        project.is_default = body.is_default

    db.commit()
    db.refresh(project)
    return _project_to_response(project)


# ---------------------------------------------------------------------------
# Save camera view
# ---------------------------------------------------------------------------

@router.put("/{project_id}/save-view", response_model=ProjectResponse)
async def save_project_view(
    project_id: str,
    body: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save current camera state to a project. Accepts {center, zoom, pitch, bearing}."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project.owner_id) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner")

    project.camera_state = {
        "center": body.get("center"),
        "zoom": body.get("zoom"),
        "pitch": body.get("pitch", 0),
        "bearing": body.get("bearing", 0),
    }

    db.commit()
    db.refresh(project)
    return _project_to_response(project)


# ---------------------------------------------------------------------------
# Delete project
# ---------------------------------------------------------------------------

@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a project. Owner only."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if str(project.owner_id) != str(current_user.id) and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not owner")

    db.delete(project)
    db.commit()

    return Response(status_code=204)
