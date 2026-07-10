-- 014_project_versioning.sql
-- Project versioning system: continuous monitoring, version tracking, layer copies

-- =========================================================================
-- 1. Alter projects table — add versioning columns
-- =========================================================================
ALTER TABLE projects ADD COLUMN IF NOT EXISTS project_type VARCHAR(20) DEFAULT 'single';
-- values: 'single', 'continuous'

ALTER TABLE projects ADD COLUMN IF NOT EXISTS auto_ingest BOOLEAN DEFAULT false;

ALTER TABLE projects ADD COLUMN IF NOT EXISTS auto_ingest_config JSONB DEFAULT '{}';
-- auto_ingest_config stores:
-- { sensors: [], min_interval_hours: 24, max_cloud_cover: 30, notify_dashboard: true }

-- =========================================================================
-- 2. Create project_versions table
-- =========================================================================
CREATE TABLE IF NOT EXISTS project_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL DEFAULT 1,
    label VARCHAR(200),  -- e.g. "V1 — Initial Survey", auto-generated if empty
    status VARCHAR(30) DEFAULT 'draft',  -- draft, in_progress, analyzed, reported, closed
    image_date DATE,  -- the satellite image acquisition date
    raster_ids UUID[] DEFAULT '{}',
    annotation_ids UUID[] DEFAULT '{}',
    report_id UUID,  -- link to intelligence_reports if a report was generated
    notes TEXT,
    thumbnail_url VARCHAR(1000),
    change_summary JSONB DEFAULT '[]',  -- array of {type: 'new'|'changed'|'removed', description: '...', geometry: {...}}
    camera_state JSONB DEFAULT '{}',  -- saved map view for this version
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    UNIQUE(project_id, version_number)
);
CREATE INDEX IF NOT EXISTS idx_pv_project ON project_versions(project_id);
CREATE INDEX IF NOT EXISTS idx_pv_date ON project_versions(image_date);

-- =========================================================================
-- 3. Update targets table — support multiple geometry types
-- =========================================================================
ALTER TABLE targets ADD COLUMN IF NOT EXISTS geometry_type VARCHAR(20) DEFAULT 'point';
-- values: 'point', 'polygon', 'multi_polygon'

-- Add general-purpose geometry column (the existing 'location' column stays for Point lookups)
ALTER TABLE targets ADD COLUMN IF NOT EXISTS geom GEOMETRY;

ALTER TABLE targets ADD COLUMN IF NOT EXISTS linked_oob_ids UUID[] DEFAULT '{}';

-- last_observed already exists (as TIMESTAMPTZ) from 010, skip re-adding
ALTER TABLE targets ADD COLUMN IF NOT EXISTS observation_count INTEGER DEFAULT 0;

-- =========================================================================
-- 4. Create version_layer_copies table
-- =========================================================================
CREATE TABLE IF NOT EXISTS version_layer_copies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_version_id UUID NOT NULL REFERENCES project_versions(id),
    target_version_id UUID NOT NULL REFERENCES project_versions(id),
    source_annotation_id UUID NOT NULL,
    copied_annotation_id UUID NOT NULL,
    copied_at TIMESTAMPTZ DEFAULT NOW()
);
