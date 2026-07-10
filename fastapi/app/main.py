from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from .config import get_settings
from .database import engine
from .middleware.rate_limit import limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail-fast: refuse to boot operationally with an insecure/default JWT secret.
    get_settings().validate_operational()
    # Verify DB connection on startup
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    # Ensure MinIO buckets exist (with SSE-S3 encryption where KMS is configured)
    try:
        from .services.minio_storage import ensure_buckets
        ensure_buckets()
    except Exception:
        pass  # MinIO may not be ready yet; buckets created on first use
    # Seed demo accounts (idempotent) so dev-bypass + real login + RBAC work.
    try:
        from .seed import ensure_seed_users
        ensure_seed_users()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Abyss API",
    version="1.0.0",
    lifespan=lifespan,
    # Keep FastAPI's default trailing-slash tolerance (307-redirect). The frontend
    # calls trailing-slash endpoints (/api/rasters/, /api/projects/, …); disabling
    # this 404s them and makes Collections/Reports pages appear empty/broken.
    redirect_slashes=True,
)

# Rate limiting (slowapi + Redis, XFF-aware). The global default limit is applied
# by SlowAPIMiddleware; per-route limits (e.g. login) use @limiter.limit(...).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS — restrict origins in production
# NOTE: allow_origins=["*"] with allow_credentials=True is insecure.
# In production, set CORS_ORIGINS env var to a comma-separated list of allowed origins.
import os as _os
_cors_origins_raw = _os.environ.get("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()] if _cors_origins_raw and _cors_origins_raw != "*" else []
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Development fallback — DO NOT use in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Routers — distilled set for Abyss (image collection → bathymetry → template report).
# Dropped from Glyph: vectors, stories, timelines, slideshows, oob, regions, targets,
# versions, usb, yolo, jobs (SR-upscale).
from .routers import (  # noqa: E402
    annotations,
    audit,
    auth,
    bathymetry,
    chat,
    groups,
    permissions,
    settings,
    map_state,
    projects,
    rasters,
    reports,
    search,
    users,
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(rasters.router, prefix="/api/rasters", tags=["rasters"])
app.include_router(annotations.router, prefix="/api/annotations", tags=["annotations"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(groups.router, prefix="/api/groups", tags=["groups"])
app.include_router(permissions.router, prefix="/api", tags=["permissions"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(audit.router, prefix="/api/audit", tags=["audit"])
app.include_router(map_state.router, prefix="/api/map-state", tags=["map-state"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(reports.stats_router, prefix="/api", tags=["reports"])
app.include_router(bathymetry.router, prefix="/api/bathymetry", tags=["bathymetry"])
app.include_router(chat.router, prefix="/api", tags=["chat"])  # POST /api/chat


@app.get("/api/health")
async def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}
