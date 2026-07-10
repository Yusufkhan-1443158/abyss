import uuid

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, INET, JSONB, UUID
from sqlalchemy.sql import func
from sqlalchemy.types import DateTime

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    display_name = Column(Text)
    email = Column(Text)
    role = Column(Text, nullable=False, default="viewer")
    is_active = Column(Boolean, default=True)
    preferences = Column(JSONB, default=dict)
    last_login = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'analyst', 'viewer')", name="ck_users_role"),
    )


class UserGroup(Base):
    __tablename__ = "user_groups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserGroupMember(Base):
    __tablename__ = "user_group_members"

    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("user_groups.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    added_at = Column(DateTime(timezone=True), server_default=func.now())


class GroupAreaRestriction(Base):
    __tablename__ = "group_area_restrictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("user_groups.id", ondelete="CASCADE"),
    )
    name = Column(Text, nullable=False)
    description = Column(Text)
    bbox = Column(Geometry("POLYGON", srid=4326), nullable=False)
    access_level = Column(Text, default="view")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "access_level IN ('view', 'download', 'manage')",
            name="ck_area_access_level",
        ),
    )


class RegisteredApp(Base):
    __tablename__ = "registered_apps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    app_name = Column(Text, unique=True, nullable=False)
    display_name = Column(Text)
    description = Column(Text)
    icon = Column(Text)
    url = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GroupAllowedApp(Base):
    __tablename__ = "group_allowed_apps"

    group_id = Column(
        UUID(as_uuid=True),
        ForeignKey("user_groups.id", ondelete="CASCADE"),
        primary_key=True,
    )
    app_name = Column(
        Text,
        ForeignKey("registered_apps.app_name", ondelete="CASCADE"),
        primary_key=True,
    )


class RasterCatalog(Base):
    __tablename__ = "raster_catalog"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text)

    # File info
    original_filename = Column(Text, nullable=False)
    original_format = Column(Text, nullable=False)
    converted_format = Column(Text, default="jph")
    file_size_bytes = Column(BigInteger)

    # Image properties
    width_px = Column(Integer)
    height_px = Column(Integer)
    num_bands = Column(Integer)
    bit_depth = Column(Integer)
    crs = Column(Text, default="EPSG:4326")
    resolution_m = Column(Float)

    # Temporal
    acquisition_date = Column(DateTime(timezone=True))

    # Spatial
    bbox = Column(Geometry("POLYGON", srid=4326))
    center_point = Column(Geometry("POINT", srid=4326))
    min_zoom = Column(Integer)
    max_zoom = Column(Integer)

    # Storage paths
    minio_raw_path = Column(Text)
    minio_jph_path = Column(Text)
    minio_cog_path = Column(Text)  # prebuilt Cloud-Optimized GeoTIFF (instant-load)
    tile_path = Column(Text)

    # Processing
    processing_status = Column(Text, default="pending")
    processing_error = Column(Text)
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))

    # Ownership
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    upload_date = Column(DateTime(timezone=True), server_default=func.now())

    # Metadata
    metadata_ = Column("metadata", JSONB, default=dict)
    vendor_metadata = Column(JSONB, default=dict)
    metadata_warnings = Column(JSONB, default=list)
    tags = Column(ARRAY(Text), default=list)
    is_public = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint(
            "processing_status IN ('pending', 'processing', 'tiling', 'ready', 'failed')",
            name="ck_raster_processing_status",
        ),
    )


class RasterTimeseries(Base):
    __tablename__ = "raster_timeseries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text)
    area_bbox = Column(Geometry("POLYGON", srid=4326))
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RasterTimeseriesEntry(Base):
    __tablename__ = "raster_timeseries_entries"

    timeseries_id = Column(
        UUID(as_uuid=True),
        ForeignKey("raster_timeseries.id", ondelete="CASCADE"),
        primary_key=True,
    )
    raster_id = Column(
        UUID(as_uuid=True),
        ForeignKey("raster_catalog.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sort_order = Column(Integer)


class VectorDataset(Base):
    __tablename__ = "vector_datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text)

    # Source type
    source = Column(Text, nullable=False, default="upload")

    # File info (null for annotations)
    original_filename = Column(Text)
    original_format = Column(Text)
    file_size_bytes = Column(BigInteger)

    # Dataset properties
    feature_count = Column(Integer, default=0)
    geometry_types = Column(ARRAY(Text))
    crs = Column(Text, default="EPSG:4326")
    bbox = Column(Geometry("POLYGON", srid=4326))
    properties_schema = Column(JSONB, default=dict)

    # Storage
    minio_raw_path = Column(Text)

    # Processing
    processing_status = Column(Text, default="ready")
    processing_error = Column(Text)

    # Ownership & sharing
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    is_public = Column(Boolean, default=False)
    shared_with_groups = Column(ARRAY(UUID(as_uuid=True)), default=list)

    # Styling
    default_style = Column(
        JSONB,
        default=lambda: {
            "fill_color": "#3388ff",
            "fill_opacity": 0.3,
            "stroke_color": "#3388ff",
            "stroke_width": 2,
            "stroke_opacity": 1.0,
            "point_radius": 6,
            "point_icon": None,
        },
    )

    # Metadata
    tags = Column(ARRAY(Text), default=list)
    metadata_ = Column("metadata", JSONB, default=dict)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("source IN ('upload', 'annotation')", name="ck_vector_source"),
        CheckConstraint(
            "processing_status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_vector_processing_status",
        ),
    )


class VectorFeature(Base):
    __tablename__ = "vector_features"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("vector_datasets.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Geometry
    geom = Column(Geometry("GEOMETRY", srid=4326), nullable=False)
    geometry_type = Column(Text, nullable=False)

    # Display
    label = Column(Text)
    description = Column(Text)

    # Properties
    properties = Column(JSONB, default=dict)

    # Per-feature style override
    style = Column(JSONB)

    # Ownership (for annotations)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserMapState(Base):
    __tablename__ = "user_map_states"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(Text, nullable=False, default="default")

    # Camera
    center_lon = Column(Float)
    center_lat = Column(Float)
    zoom = Column(Float)
    bearing = Column(Float, default=0)
    pitch = Column(Float, default=0)

    # Layer stack
    layer_state = Column(JSONB, default=list)

    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_user_map_state_name"),
    )


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text, default="")
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    camera_state = Column(JSONB, default=dict)
    raster_ids = Column(ARRAY(UUID(as_uuid=True)), default=list)
    vector_ids = Column(ARRAY(UUID(as_uuid=True)), default=list)
    annotation_ids = Column(ARRAY(UUID(as_uuid=True)), default=list)
    is_default = Column(Boolean, default=False)
    project_type = Column(Text, default="single")
    auto_ingest = Column(Boolean, default=False)
    auto_ingest_config = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    username = Column(Text)
    action = Column(Text, nullable=False)
    resource_type = Column(Text)
    resource_id = Column(Text)
    details = Column(JSONB, default=dict)
    ip_address = Column(INET)
    user_agent = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
