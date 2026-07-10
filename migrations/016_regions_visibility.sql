-- 016_regions_visibility.sql
-- Adds visibility flag + per-owner name uniqueness for the regions table.
-- Idempotent: safe to run multiple times.

BEGIN;

ALTER TABLE regions
    ADD COLUMN IF NOT EXISTS visibility VARCHAR(20) NOT NULL DEFAULT 'private';

ALTER TABLE regions
    DROP CONSTRAINT IF EXISTS regions_visibility_chk;
ALTER TABLE regions
    ADD CONSTRAINT regions_visibility_chk CHECK (visibility IN ('private', 'public'));

CREATE INDEX IF NOT EXISTS idx_regions_visibility ON regions(visibility);

CREATE UNIQUE INDEX IF NOT EXISTS uq_regions_owner_name
    ON regions(owner_id, LOWER(name));

COMMIT;
