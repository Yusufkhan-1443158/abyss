-- Abyss instant-load: prebuilt Cloud-Optimized GeoTIFF path on the raster row.
-- The COG is built once at ingest (Stage A, into the shared tile cache) and
-- persisted to the `cog` bucket by Stage B; the tile renderer fetches it instead
-- of rebuilding overviews on the first tile request.
ALTER TABLE raster_catalog ADD COLUMN IF NOT EXISTS minio_cog_path TEXT;
