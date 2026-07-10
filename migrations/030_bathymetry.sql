-- Abyss bathymetry pipeline: inference jobs + template-based reports.
-- (Replaces Glyph's Airflow-ingested freeform `intelligence_reports`.)

CREATE TABLE IF NOT EXISTS bathymetry_jobs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_raster_id UUID REFERENCES raster_catalog(id) ON DELETE CASCADE,
    depth_raster_id  UUID REFERENCES raster_catalog(id) ON DELETE SET NULL,
    report_id        UUID,
    model            TEXT,
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','running','done','failed')),
    error_message    TEXT,
    requested_by     UUID REFERENCES users(id),
    metadata         JSONB DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_bathy_jobs_source ON bathymetry_jobs(source_raster_id);
CREATE INDEX IF NOT EXISTS idx_bathy_jobs_status ON bathymetry_jobs(status);

CREATE TABLE IF NOT EXISTS bathymetry_reports (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_code          TEXT UNIQUE NOT NULL,
    source_raster_id     UUID REFERENCES raster_catalog(id) ON DELETE SET NULL,
    depth_raster_id      UUID REFERENCES raster_catalog(id) ON DELETE SET NULL,
    template_id          TEXT NOT NULL DEFAULT 'bathymetry_v1',
    title                TEXT,
    site_name            TEXT,
    classification_level TEXT DEFAULT 'UNCLASSIFIED',
    geom                 geometry(Point, 4326),
    status               TEXT NOT NULL DEFAULT 'draft'
                         CHECK (status IN ('draft','generating','ready','failed')),
    sections             JSONB DEFAULT '[]'::jsonb,
    statistics           JSONB DEFAULT '{}'::jsonb,
    generated_at         TIMESTAMPTZ,
    created_by           UUID REFERENCES users(id),
    created_at           TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_bathy_reports_geom ON bathymetry_reports USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_bathy_reports_status ON bathymetry_reports(status);
