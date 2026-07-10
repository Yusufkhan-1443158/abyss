"""Platform settings: logo, broadcast messages, classification defaults."""

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import get_db
from ..middleware.auth_middleware import get_current_user, require_admin
from ..models import User

router = APIRouter()


class BroadcastMessage(BaseModel):
    id: str | None = None
    text: str
    type: str = "info"  # info, warning, success, urgent
    active: bool = True
    created_at: str | None = None


class BroadcastMessageCreate(BaseModel):
    text: str
    type: str = "info"


# --------------------------------------------------------------------------
# GET /api/settings/public — anyone can read (for home page display)
# --------------------------------------------------------------------------
@router.get("/public")
def get_public_settings(db: Session = Depends(get_db)):
    """Public settings visible to all authenticated users (logo, broadcasts)."""
    rows = db.execute(
        text("SELECT key, value FROM platform_settings WHERE key IN ('home_logo_url', 'broadcast_messages', 'default_classification')")
    ).fetchall()
    result = {}
    for r in rows:
        result[r.key] = r.value
    return result


# --------------------------------------------------------------------------
# GET /api/settings — all settings (admin)
# --------------------------------------------------------------------------
@router.get("")
def get_all_settings(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    rows = db.execute(text("SELECT key, value, updated_at FROM platform_settings ORDER BY key")).fetchall()
    return {r.key: {"value": r.value, "updated_at": r.updated_at} for r in rows}


# --------------------------------------------------------------------------
# PUT /api/settings/:key — update a setting (admin)
# --------------------------------------------------------------------------
@router.put("/{key}")
def update_setting(
    key: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
    value: dict | list | str | None = None,
):
    db.execute(
        text("INSERT INTO platform_settings (key, value, updated_at, updated_by) VALUES (:key, :val, NOW(), :uid) ON CONFLICT (key) DO UPDATE SET value = :val, updated_at = NOW(), updated_by = :uid"),
        {"key": key, "val": json.dumps(value), "uid": str(admin.id)},
    )
    db.commit()
    return {"key": key, "value": value}


# --------------------------------------------------------------------------
# POST /api/settings/logo — upload logo image (admin)
# --------------------------------------------------------------------------
@router.post("/logo")
async def upload_logo(
    file: UploadFile = File(...),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Upload platform logo (PNG/JPG, stored in MinIO)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    from ..services.minio_storage import get_s3_client
    s3 = get_s3_client()
    bucket = "thumbnails"
    key = f"platform/logo{_ext(file.filename)}"

    data = await file.read()
    import io
    s3.put_object(Bucket=bucket, Key=key, Body=io.BytesIO(data), ContentLength=len(data), ContentType=file.content_type)

    logo_url = f"/api/rasters/thumbnail/{bucket}/{key}"
    db.execute(
        text("INSERT INTO platform_settings (key, value, updated_at, updated_by) VALUES ('home_logo_url', :val, NOW(), :uid) ON CONFLICT (key) DO UPDATE SET value = :val, updated_at = NOW(), updated_by = :uid"),
        {"val": json.dumps(logo_url), "uid": str(admin.id)},
    )
    db.commit()
    return {"logo_url": logo_url}


# --------------------------------------------------------------------------
# POST /api/settings/broadcast — add broadcast message (admin)
# --------------------------------------------------------------------------
@router.post("/broadcast")
def add_broadcast(
    body: BroadcastMessageCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Add a broadcast message visible on the home page."""
    row = db.execute(text("SELECT value FROM platform_settings WHERE key = 'broadcast_messages'")).fetchone()
    messages = row.value if row and isinstance(row.value, list) else (json.loads(row.value) if row and isinstance(row.value, str) else [])

    msg = {
        "id": str(uuid.uuid4())[:8],
        "text": body.text,
        "type": body.type,
        "active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    messages.append(msg)

    db.execute(
        text("UPDATE platform_settings SET value = :val, updated_at = NOW(), updated_by = :uid WHERE key = 'broadcast_messages'"),
        {"val": json.dumps(messages), "uid": str(admin.id)},
    )
    db.commit()
    return msg


# --------------------------------------------------------------------------
# DELETE /api/settings/broadcast/:msg_id — remove broadcast (admin)
# --------------------------------------------------------------------------
@router.delete("/broadcast/{msg_id}")
def delete_broadcast(
    msg_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    row = db.execute(text("SELECT value FROM platform_settings WHERE key = 'broadcast_messages'")).fetchone()
    messages = row.value if row and isinstance(row.value, list) else (json.loads(row.value) if row and isinstance(row.value, str) else [])
    messages = [m for m in messages if m.get("id") != msg_id]

    db.execute(
        text("UPDATE platform_settings SET value = :val, updated_at = NOW(), updated_by = :uid WHERE key = 'broadcast_messages'"),
        {"val": json.dumps(messages), "uid": str(admin.id)},
    )
    db.commit()
    return {"detail": "Broadcast deleted"}


def _ext(filename: str) -> str:
    if not filename:
        return ".png"
    parts = filename.rsplit(".", 1)
    return "." + parts[1] if len(parts) > 1 else ".png"
