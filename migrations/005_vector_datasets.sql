-- 005_vector_datasets.sql
-- Vector datasets (uploads + annotations) and individual vector features
-- Idempotent: safe to run multiple times

BEGIN;

CREATE TABLE IF NOT EXISTS vector_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,

    -- Source type
    source TEXT NOT NULL DEFAULT 'upload'
        CHECK (source IN ('upload', 'annotation')),

    -- File info (null for annotations)
    original_filename TEXT,
    original_format TEXT,
    file_size_bytes BIGINT,

    -- Dataset properties
    feature_count INTEGER DEFAULT 0,
    geometry_types TEXT[],
    crs TEXT DEFAULT 'EPSG:4326',
    bbox GEOMETRY(Polygon, 4326),
    properties_schema JSONB DEFAULT '{}',

    -- Storage
    minio_raw_path TEXT,

    -- Processing
    processing_status TEXT DEFAULT 'ready'
        CHECK (processing_status IN ('pending', 'processing', 'ready', 'failed')),
    processing_error TEXT,

    -- Ownership & sharing
    created_by UUID REFERENCES users(id),
    is_public BOOLEAN DEFAULT false,
    shared_with_groups UUID[] DEFAULT '{}',

    -- Styling
    default_style JSONB DEFAULT '{
        "fill_color": "#3388ff",
        "fill_opacity": 0.3,
        "stroke_color": "#3388ff",
        "stroke_width": 2,
        "stroke_opacity": 1.0,
        "point_radius": 6,
        "point_icon": null
    }',

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vector_bbox ON vector_datasets USING GIST(bbox);
CREATE INDEX IF NOT EXISTS idx_vector_source ON vector_datasets(source);
CREATE INDEX IF NOT EXISTS idx_vector_created_by ON vector_datasets(created_by);
CREATE INDEX IF NOT EXISTS idx_vector_status ON vector_datasets(processing_status);
CREATE INDEX IF NOT EXISTS idx_vector_tags ON vector_datasets USING GIN(tags);

-- Individual vector features
CREATE TABLE IF NOT EXISTS vector_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL
        REFERENCES vector_datasets(id) ON DELETE CASCADE,

    -- Geometry
    geom GEOMETRY(Geometry, 4326) NOT NULL,
    geometry_type TEXT NOT NULL,

    -- Display
    label TEXT,
    description TEXT,

    -- Properties (from source file or user input)
    properties JSONB DEFAULT '{}',

    -- Per-feature style override (null = use dataset default)
    style JSONB,

    -- Ownership (for annotations)
    created_by UUID REFERENCES users(id),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vf_dataset ON vector_features(dataset_id);
CREATE INDEX IF NOT EXISTS idx_vf_geom ON vector_features USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_vf_type ON vector_features(geometry_type);
CREATE INDEX IF NOT EXISTS idx_vf_properties ON vector_features USING GIN(properties);

COMMIT;
