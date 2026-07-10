-- Timelines table
CREATE TABLE IF NOT EXISTS timelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    classification VARCHAR(50) DEFAULT 'unclassified',
    owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    date_range_start TIMESTAMPTZ,
    date_range_end TIMESTAMPTZ,
    tags TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_timelines_project ON timelines(project_id);

-- Timeline events table
CREATE TABLE IF NOT EXISTS timeline_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timeline_id UUID NOT NULL REFERENCES timelines(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    event_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ,
    event_type VARCHAR(50),
    location GEOGRAPHY(Point, 4326),
    icon VARCHAR(50),
    color VARCHAR(20),
    report_id UUID,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_timeline_events_timeline ON timeline_events(timeline_id);
CREATE INDEX IF NOT EXISTS idx_timeline_events_date ON timeline_events(event_date);
