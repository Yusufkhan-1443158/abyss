-- 003_groups_areas.sql
-- User groups, group membership, and area-based access restrictions
-- Idempotent: safe to run multiple times

BEGIN;

CREATE TABLE IF NOT EXISTS user_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_group_members (
    group_id UUID REFERENCES user_groups(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (group_id, user_id)
);

CREATE TABLE IF NOT EXISTS group_area_restrictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id UUID REFERENCES user_groups(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    bbox GEOMETRY(Polygon, 4326) NOT NULL,
    access_level TEXT DEFAULT 'view'
        CHECK (access_level IN ('view', 'download', 'manage')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_group_area_bbox
    ON group_area_restrictions USING GIST(bbox);

COMMIT;
