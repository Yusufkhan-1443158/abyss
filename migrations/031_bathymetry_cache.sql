-- Abyss bathymetry result cache: dedup identical surveys (same area + date range
-- + model version) so a repeated request returns the stored product instantly
-- instead of re-fetching imagery and re-running inference.
--
-- The S2 imagery cache lives in MinIO (bucket `s2cache`, keyed by area+dates);
-- this column keys the *generated* product. A model-version change yields a new
-- cache_key, which forces re-inference (while the S2 imagery cache is still reused).

ALTER TABLE bathymetry_reports ADD COLUMN IF NOT EXISTS cache_key TEXT;
CREATE INDEX IF NOT EXISTS idx_bathy_reports_cache ON bathymetry_reports(cache_key);
