"""
App-level permissions: which apps a user can access based on group membership.
Admin role bypasses — sees all registered apps.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import RegisteredApp, User
from ..middleware.auth_middleware import get_current_user, require_admin
from ..schemas import PermissionsResponse, RegisteredAppResponse

router = APIRouter()


@router.get("/registered-apps", response_model=list[RegisteredAppResponse])
def list_registered_apps(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List all registered apps (admin only). Used by admin UI to populate app checkboxes."""
    apps = db.query(RegisteredApp).order_by(RegisteredApp.display_name).all()
    return [RegisteredAppResponse.model_validate(a) for a in apps]


@router.get("/permissions/{username}", response_model=PermissionsResponse)
def get_user_permissions(
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get the list of apps a user can access.
    Any authenticated user can check their own permissions.
    Admins can check anyone's.
    """
    # Non-admins can only check their own permissions
    if current_user.role != "admin" and current_user.username != username:
        raise HTTPException(status_code=403, detail="Cannot view other users' permissions")

    # Look up target user
    target = db.query(User).filter(User.username == username).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    is_admin = target.role == "admin"

    if is_admin:
        # Admins see all registered apps
        apps = db.query(RegisteredApp).order_by(RegisteredApp.display_name).all()
        allowed = [RegisteredAppResponse.model_validate(a) for a in apps]
    else:
        # Non-admins: get apps from their group memberships
        rows = db.execute(
            text("""
                SELECT DISTINCT ra.app_name, ra.display_name, ra.description, ra.icon, ra.url
                FROM registered_apps ra
                JOIN group_allowed_apps gaa ON gaa.app_name = ra.app_name
                JOIN user_group_members ugm ON ugm.group_id = gaa.group_id
                WHERE ugm.user_id = :uid
                ORDER BY ra.display_name
            """),
            {"uid": str(target.id)},
        ).fetchall()

        allowed = [
            RegisteredAppResponse(
                app_name=r.app_name,
                display_name=r.display_name,
                description=r.description,
                icon=r.icon,
                url=r.url,
            )
            for r in rows
        ]

    return PermissionsResponse(
        username=target.username,
        role=target.role,
        is_admin=is_admin,
        allowed_apps=allowed,
        hide_unauthorized=False,
    )
