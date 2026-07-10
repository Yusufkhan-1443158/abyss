CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
    camera_state JSONB DEFAULT '{}',
    raster_ids UUID[] DEFAULT '{}',
    vector_ids UUID[] DEFAULT '{}',
    annotation_ids UUID[] DEFAULT '{}',
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id);
