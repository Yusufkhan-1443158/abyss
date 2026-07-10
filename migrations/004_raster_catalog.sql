-- 004_raster_catalog.sql
-- Raster catalog, time series, and time series entries
-- Idempotent: safe to run multiple times

BEGIN;

CREATE TABLE IF NOT EXISTS raster_catalog (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,

    -- File info
    original_filename TEXT NOT NULL,
    original_format TEXT NOT NULL,
    converted_format TEXT DEFAULT 'jph',
    file_size_bytes BIGINT,

    -- Image properties
    width_px INTEGER,
    height_px INTEGER,
    num_bands INTEGER,
    bit_depth INTEGER,
    crs TEXT DEFAULT 'EPSG:4326',
    resolution_m FLOAT,

    -- Temporal
    acquisition_date TIMESTAMP WITH TIME ZONE,

    -- Spatial
    bbox GEOMETRY(Polygon, 4326),
    center_point GEOMETRY(Point, 4326),
    min_zoom INTEGER,
    max_zoom INTEGER,

    -- Storage paths
    minio_raw_path TEXT,
    minio_jph_path TEXT,
    tile_path TEXT,

    -- Processing
    processing_status TEXT DEFAULT 'pending'
        CHECK (processing_status IN (
            'pending', 'processing', 'tiling', 'ready', 'failed'
        )),
    processing_error TEXT,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,

    -- Ownership
    uploaded_by UUID REFERENCES users(id),
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    is_public BOOLEAN DEFAULT false,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raster_bbox ON raster_catalog USING GIST(bbox);
CREATE INDEX IF NOT EXISTS idx_raster_center ON raster_catalog USING GIST(center_point);
CREATE INDEX IF NOT EXISTS idx_raster_date ON raster_catalog(acquisition_date);
CREATE INDEX IF NOT EXISTS idx_raster_status ON raster_catalog(processing_status);
CREATE INDEX IF NOT EXISTS idx_raster_tags ON raster_catalog USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_raster_uploaded_by ON raster_catalog(uploaded_by);

-- Raster time series
CREATE TABLE IF NOT EXISTS raster_timeseries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    area_bbox GEOMETRY(Polygon, 4326),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raster_timeseries_entries (
    timeseries_id UUID REFERENCES raster_timeseries(id) ON DELETE CASCADE,
    raster_id UUID REFERENCES raster_catalog(id) ON DELETE CASCADE,
    sort_order INTEGER,
    PRIMARY KEY (timeseries_id, raster_id)
);

COMMIT;
