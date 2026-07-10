-- 015: App-level permissions for user groups
-- Groups can be assigned specific apps. Admin role bypasses (sees all).

BEGIN;

-- Catalog of all platform apps
CREATE TABLE IF NOT EXISTS registered_apps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    app_name VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200),
    description TEXT,
    icon VARCHAR(50),
    url VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Junction: which apps each group can access
CREATE TABLE IF NOT EXISTS group_allowed_apps (
    group_id UUID REFERENCES user_groups(id) ON DELETE CASCADE,
    app_name VARCHAR(100) REFERENCES registered_apps(app_name) ON DELETE CASCADE,
    PRIMARY KEY (group_id, app_name)
);

CREATE INDEX IF NOT EXISTS idx_group_allowed_apps_group ON group_allowed_apps(group_id);
CREATE INDEX IF NOT EXISTS idx_group_allowed_apps_app ON group_allowed_apps(app_name);

-- Seed all platform apps
INSERT INTO registered_apps (app_name, display_name, description, icon, url) VALUES
    ('global-browse', 'Global Browse', 'Explore satellite imagery, reports & data on the globe', '🌍', '/global-browse.html'),
    ('projects-browse', 'Projects', 'Browse and manage analysis projects', '📁', '/projects-browse.html'),
    ('reports-browse', 'Reports', 'View all intelligence reports', '📋', '/reports-browse.html'),
    ('collections-browse', 'Collections', 'Satellite imagery & raster uploads', '🗂️', '/collections-browse.html'),
    ('oob-browse', 'Order of Battle', 'Military units, equipment & symbology', '⚔️', '/oob-browse.html'),
    ('stories-browse', 'Stories', 'Intelligence narrative stories', '📖', '/stories-browse.html'),
    ('timelines', 'Timelines', 'Interactive event timelines', '🕐', '/timelines.html'),
    ('slideshows', 'Slideshows', 'Map-based presentations', '📊', '/slideshows.html'),
    ('regions', 'Region Profiles', 'Geographic area profiles', '🧠', '/regions.html'),
    ('ai-chat', 'AI Chat', 'AI-assisted analysis chat', '🤖', '/ai-chat'),
    ('map-viewer', 'Map Viewer', 'Interactive 2D map viewer', '🗺️', '/map-viewer.html'),
    ('globe-viewer', 'Globe Viewer', 'Interactive 3D globe viewer', '🌐', '/globe-viewer.html'),
    ('map-annotator', 'Map Annotator', 'Draw and annotate on maps', '✏️', '/map-annotator.html'),
    ('targets', 'Targets', 'Target tracking and management', '🎯', '/targets.html'),
    ('ai-trainer', 'AI Trainer', 'Train & deploy detection models', '🤖', '/ai-trainer.html'),
    ('yolo-annotator', 'YOLO Annotator', 'Annotate training data for detection', '🏷️', '/yolo-annotator.html'),
    ('yolo-training', 'YOLO Training', 'Train YOLO detection models', '⚙️', '/yolo-training.html'),
    ('yolo-inference', 'YOLO Inference', 'Run object detection on imagery', '👁️', '/yolo-inference.html'),
    ('azimuth-plotter', 'Azimuth Plotter', 'Plot bearings and distances on map', '📐', '/azimuth-plotter.html'),
    ('coordinate-converter', 'Coordinate Converter', 'Convert between coordinate systems', '📍', '/coordinate-converter.html'),
    ('timeline-builder', 'Timeline Builder', 'Build event timelines', '📅', '/timeline-builder.html'),
    ('report-slideshow', 'Report Slideshow', 'View reports as slideshows', '🎞️', '/report-slideshow.html'),
    ('report-view', 'Report View', 'Detailed report viewer', '📄', '/report-view.html'),
    ('image-viewer', 'Image Viewer', 'Satellite image viewer', '🖼️', '/image-viewer.html')
ON CONFLICT (app_name) DO NOTHING;

COMMIT;
