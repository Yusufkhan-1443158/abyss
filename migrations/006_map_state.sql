-- 006_map_state.sql
-- User map states (saved views + layer order)
-- Idempotent: safe to run multiple times

BEGIN;

CREATE TABLE IF NOT EXISTS user_map_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL DEFAULT 'default',

    -- Camera
    center_lon FLOAT,
    center_lat FLOAT,
    zoom FLOAT,
    bearing FLOAT DEFAULT 0,
    pitch FLOAT DEFAULT 0,

    -- Layer stack (ordered array, rendered bottom to top)
    layer_state JSONB DEFAULT '[]',

    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_ums_user ON user_map_states(user_id);

COMMIT;
