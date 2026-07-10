from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# GeoJSON primitives
# ---------------------------------------------------------------------------

class Geometry(BaseModel):
    type: str
    coordinates: Any


class Feature(BaseModel):
    type: str = "Feature"
    geometry: Geometry
    properties: dict[str, Any] = {}
    id: str | None = None


class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[Feature] = []


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class PaginationParams(BaseModel):
    offset: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=500)


class PaginatedResponse(BaseModel):
    total: int
    offset: int
    limit: int


# ---------------------------------------------------------------------------
# Auth / Token
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: str | None = None
    username: str | None = None
    role: str | None = None


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    username: str
    password: str
    display_name: str | None = None
    email: str | None = None
    role: str = "viewer"


class UserUpdate(BaseModel):
    display_name: str | None = None
    email: str | None = None
    role: str | None = None
    is_active: bool | None = None
    password: str | None = None
    preferences: dict[str, Any] | None = None


class UserResponse(BaseModel):
    id: str
    username: str
    display_name: str | None = None
    email: str | None = None
    role: str
    is_active: bool
    last_login: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserMeResponse(UserResponse):
    preferences: dict[str, Any] = {}
    areas: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

class GroupCreate(BaseModel):
    name: str
    description: str | None = None
    allowed_apps: list[str] = []


class GroupUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    allowed_apps: list[str] | None = None


class GroupResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_by: str | None = None
    allowed_apps: list[str] = []
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class GroupWithMembers(GroupResponse):
    members: list[UserResponse] = []
    areas: list["AreaRestrictionResponse"] = []


# ---------------------------------------------------------------------------
# App Permissions
# ---------------------------------------------------------------------------

class RegisteredAppResponse(BaseModel):
    app_name: str
    display_name: str | None = None
    description: str | None = None
    icon: str | None = None
    url: str | None = None

    model_config = {"from_attributes": True}


class PermissionsResponse(BaseModel):
    username: str
    role: str
    is_admin: bool
    allowed_apps: list[RegisteredAppResponse] = []
    hide_unauthorized: bool = False


# ---------------------------------------------------------------------------
# Area Restrictions
# ---------------------------------------------------------------------------

class AreaRestrictionCreate(BaseModel):
    name: str
    description: str | None = None
    bbox: dict[str, Any]  # GeoJSON Polygon
    access_level: str = "view"


class AreaRestrictionResponse(BaseModel):
    id: str
    group_id: str
    name: str
    description: str | None = None
    bbox: dict[str, Any] | None = None
    access_level: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------

class RasterResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    original_filename: str
    original_format: str
    converted_format: str | None = None
    file_size_bytes: int | None = None
    width_px: int | None = None
    height_px: int | None = None
    num_bands: int | None = None
    bit_depth: int | None = None
    crs: str | None = None
    resolution_m: float | None = None
    acquisition_date: datetime | None = None
    bbox: dict[str, Any] | None = None
    center_point: dict[str, Any] | None = None
    min_zoom: int | None = None
    max_zoom: int | None = None
    processing_status: str
    processing_error: str | None = None
    uploaded_by: str | None = None
    upload_date: datetime | None = None
    metadata: dict[str, Any] = {}
    tags: list[str] = []
    is_public: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RasterListResponse(PaginatedResponse):
    items: list[RasterResponse] = []


class RasterUploadResponse(BaseModel):
    id: str
    name: str
    processing_status: str


class RasterStatusResponse(BaseModel):
    id: str
    processing_status: str
    processing_error: str | None = None
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Raster Timeseries
# ---------------------------------------------------------------------------

class RasterTimeseriesCreate(BaseModel):
    name: str
    description: str | None = None
    area_bbox: dict[str, Any] | None = None  # GeoJSON Polygon


class RasterTimeseriesResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    area_bbox: dict[str, Any] | None = None
    created_by: str | None = None
    created_at: datetime
    entries: list[RasterResponse] = []

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Vector Datasets
# ---------------------------------------------------------------------------

class VectorDatasetCreate(BaseModel):
    name: str
    description: str | None = None
    source: str = "upload"
    default_style: dict[str, Any] | None = None
    tags: list[str] = []


class VectorDatasetResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    source: str
    original_filename: str | None = None
    original_format: str | None = None
    file_size_bytes: int | None = None
    feature_count: int = 0
    geometry_types: list[str] | None = None
    crs: str | None = None
    bbox: dict[str, Any] | None = None
    properties_schema: dict[str, Any] = {}
    processing_status: str
    processing_error: str | None = None
    created_by: str | None = None
    is_public: bool = False
    shared_with_groups: list[str] = []
    default_style: dict[str, Any] | None = None
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class VectorDatasetListResponse(PaginatedResponse):
    items: list[VectorDatasetResponse] = []


# ---------------------------------------------------------------------------
# Vector Features
# ---------------------------------------------------------------------------

class VectorFeatureCreate(BaseModel):
    geom: dict[str, Any]  # GeoJSON Geometry
    geometry_type: str
    label: str | None = None
    description: str | None = None
    properties: dict[str, Any] = {}
    style: dict[str, Any] | None = None


class VectorFeatureUpdate(BaseModel):
    geom: dict[str, Any] | None = None
    geometry_type: str | None = None
    label: str | None = None
    description: str | None = None
    properties: dict[str, Any] | None = None
    style: dict[str, Any] | None = None


class VectorFeatureResponse(BaseModel):
    id: str
    dataset_id: str
    geom: dict[str, Any] | None = None
    geometry_type: str
    label: str | None = None
    description: str | None = None
    properties: dict[str, Any] = {}
    style: dict[str, Any] | None = None
    created_by: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Annotations (thin wrappers over vector)
# ---------------------------------------------------------------------------

class AnnotationLayerCreate(BaseModel):
    name: str
    description: str | None = None
    default_style: dict[str, Any] | None = None


class AnnotationLayerResponse(VectorDatasetResponse):
    pass


# ---------------------------------------------------------------------------
# Map State
# ---------------------------------------------------------------------------

class MapStateCreate(BaseModel):
    name: str = "default"
    center_lon: float | None = None
    center_lat: float | None = None
    zoom: float | None = None
    bearing: float = 0
    pitch: float = 0
    layer_state: list[dict[str, Any]] = []
    is_default: bool = False


class MapStateUpdate(BaseModel):
    name: str | None = None
    center_lon: float | None = None
    center_lat: float | None = None
    zoom: float | None = None
    bearing: float | None = None
    pitch: float | None = None
    layer_state: list[dict[str, Any]] | None = None
    is_default: bool | None = None


class MapStateResponse(BaseModel):
    id: str
    user_id: str
    name: str
    center_lon: float | None = None
    center_lat: float | None = None
    zoom: float | None = None
    bearing: float = 0
    pitch: float = 0
    layer_state: list[dict[str, Any]] = []
    is_default: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

class AuditLogResponse(BaseModel):
    id: int
    user_id: str | None = None
    username: str | None = None
    action: str
    resource_type: str | None = None
    resource_id: str | None = None
    details: dict[str, Any] = {}
    ip_address: str | None = None
    user_agent: str | None = None
    timestamp: datetime

    model_config = {"from_attributes": True}


class AuditLogQuery(BaseModel):
    user_id: str | None = None
    action: str | None = None
    resource_type: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    offset: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=500)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchQuery(BaseModel):
    q: str | None = None
    bbox: list[float] | None = None  # [x1, y1, x2, y2]
    date_from: datetime | None = None
    date_to: datetime | None = None
    type: str | None = None  # "raster", "vector", "all"
    tags: list[str] | None = None
    offset: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=500)


class SearchResultItem(BaseModel):
    result_type: str  # "raster" or "vector"
    id: str
    name: str
    description: str | None = None
    bbox: dict[str, Any] | None = None
    tags: list[str] = []
    created_at: datetime


class SearchResponse(PaginatedResponse):
    items: list[SearchResultItem] = []


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    project_type: str = "single"


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    camera_state: dict[str, Any] | None = None
    raster_ids: list[str] | None = None
    vector_ids: list[str] | None = None
    annotation_ids: list[str] | None = None
    is_default: bool | None = None
    project_type: str | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str
    camera_state: dict[str, Any]
    raster_ids: list[str]
    vector_ids: list[str]
    annotation_ids: list[str]
    is_default: bool
    project_type: str = "single"
    auto_ingest: bool = False
    auto_ingest_config: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
