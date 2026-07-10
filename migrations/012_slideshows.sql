-- Slideshows table
CREATE TABLE IF NOT EXISTS slideshows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    classification VARCHAR(50) DEFAULT 'unclassified',
    owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    tags TEXT,
    theme VARCHAR(30) DEFAULT 'dark',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Slideshow slides table
CREATE TABLE IF NOT EXISTS slideshow_slides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slideshow_id UUID NOT NULL REFERENCES slideshows(id) ON DELETE CASCADE,
    slide_order INTEGER NOT NULL DEFAULT 0,
    title VARCHAR(255),
    content TEXT,
    layout VARCHAR(30) DEFAULT 'full',
    background_url TEXT,
    media_urls TEXT,
    notes TEXT,
    map_config JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_slideshow_slides_show ON slideshow_slides(slideshow_id);
