-- Targets table
CREATE TABLE IF NOT EXISTS targets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    target_type VARCHAR(50) DEFAULT 'point',
    priority VARCHAR(20) DEFAULT 'medium',
    classification VARCHAR(50) DEFAULT 'unclassified',
    location GEOGRAPHY(Point, 4326),
    mgrs VARCHAR(50),
    address TEXT,
    country VARCHAR(100),
    status VARCHAR(30) DEFAULT 'active',
    category VARCHAR(100),
    symbol_code VARCHAR(50),
    tags TEXT,
    metadata JSONB,
    assigned_to UUID REFERENCES users(id) ON DELETE SET NULL,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    region_id UUID,
    last_observed TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_targets_project ON targets(project_id);
CREATE INDEX IF NOT EXISTS idx_targets_region ON targets(region_id);

-- Regions table
CREATE TABLE IF NOT EXISTS regions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    geometry GEOMETRY(MultiPolygon, 4326),
    center GEOGRAPHY(Point, 4326),
    country VARCHAR(100),
    region_type VARCHAR(50),
    color VARCHAR(20),
    owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    parent_id UUID REFERENCES regions(id) ON DELETE SET NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_regions_geometry ON regions USING GIST(geometry);

-- Add FK from targets to regions now that regions table exists
ALTER TABLE targets
    ADD CONSTRAINT fk_targets_region FOREIGN KEY (region_id)
    REFERENCES regions(id) ON DELETE SET NULL;
