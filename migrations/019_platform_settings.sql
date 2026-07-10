-- Platform settings (home logo, broadcast banner, default classification).
-- Lived in Glyph's migration 023 alongside the broken rasters/sr_jobs DDL that
-- Abyss drops, so it is re-created cleanly here.
CREATE TABLE IF NOT EXISTS platform_settings (
    key        TEXT PRIMARY KEY,
    value      JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by UUID REFERENCES users(id)
);

INSERT INTO platform_settings (key, value) VALUES
    ('default_classification', '"UNCLASSIFIED"'::jsonb),
    ('broadcast_messages', '[]'::jsonb),
    ('home_logo_url', '""'::jsonb)
ON CONFLICT (key) DO NOTHING;
