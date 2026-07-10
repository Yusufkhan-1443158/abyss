-- Raster vendor metadata: normalized fields extracted from sidecar files
-- (Maxar IMD/RPB, Airbus DIMAP, Planet, Landsat MTL, PAM .aux.xml, etc.)
-- plus raw_by_file preserving original content for future re-parsing.

ALTER TABLE raster_catalog
    ADD COLUMN IF NOT EXISTS vendor_metadata JSONB;

ALTER TABLE raster_catalog
    ADD COLUMN IF NOT EXISTS metadata_warnings JSONB;

-- Indexes on hot search keys
CREATE INDEX IF NOT EXISTS idx_raster_vendor
    ON raster_catalog ((vendor_metadata->>'vendor'));

CREATE INDEX IF NOT EXISTS idx_raster_acquired_at
    ON raster_catalog ((vendor_metadata->>'acquired_at'));

CREATE INDEX IF NOT EXISTS idx_raster_sensor
    ON raster_catalog ((vendor_metadata->>'sensor'));

CREATE INDEX IF NOT EXISTS idx_raster_cloud_cover
    ON raster_catalog (((vendor_metadata->>'cloud_cover_pct')::float));
