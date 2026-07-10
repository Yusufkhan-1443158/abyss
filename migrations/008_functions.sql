-- 008_functions.sql
-- Helper functions and triggers
-- Idempotent: safe to run multiple times (CREATE OR REPLACE)

BEGIN;

-- Area access check (admins bypass)
CREATE OR REPLACE FUNCTION user_has_area_access(
    p_user_id UUID,
    p_bbox GEOMETRY,
    p_required_level TEXT DEFAULT 'view'
) RETURNS BOOLEAN AS $$
DECLARE
    v_role TEXT;
BEGIN
    SELECT role INTO v_role FROM users WHERE id = p_user_id;
    IF v_role = 'admin' THEN RETURN true; END IF;

    RETURN EXISTS(
        SELECT 1
        FROM user_group_members ugm
        JOIN group_area_restrictions gar ON gar.group_id = ugm.group_id
        WHERE ugm.user_id = p_user_id
          AND ST_Intersects(gar.bbox, p_bbox)
          AND (
              gar.access_level = 'manage'
              OR gar.access_level = p_required_level
              OR (p_required_level = 'view'
                  AND gar.access_level IN ('download', 'manage'))
              OR (p_required_level = 'download'
                  AND gar.access_level = 'manage')
          )
    );
END;
$$ LANGUAGE plpgsql;

-- Auto-update updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- updated_at triggers (DROP IF EXISTS + CREATE for idempotency)
DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_raster_catalog_updated_at ON raster_catalog;
CREATE TRIGGER trg_raster_catalog_updated_at
    BEFORE UPDATE ON raster_catalog
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_user_groups_updated_at ON user_groups;
CREATE TRIGGER trg_user_groups_updated_at
    BEFORE UPDATE ON user_groups
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_vector_datasets_updated_at ON vector_datasets;
CREATE TRIGGER trg_vector_datasets_updated_at
    BEFORE UPDATE ON vector_datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_vector_features_updated_at ON vector_features;
CREATE TRIGGER trg_vector_features_updated_at
    BEFORE UPDATE ON vector_features
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_user_map_states_updated_at ON user_map_states;
CREATE TRIGGER trg_user_map_states_updated_at
    BEFORE UPDATE ON user_map_states
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Auto-update vector_datasets.feature_count on INSERT/DELETE
CREATE OR REPLACE FUNCTION update_vector_feature_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE vector_datasets
        SET feature_count = feature_count + 1
        WHERE id = NEW.dataset_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE vector_datasets
        SET feature_count = feature_count - 1
        WHERE id = OLD.dataset_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_vector_feature_count ON vector_features;
CREATE TRIGGER trg_vector_feature_count
    AFTER INSERT OR DELETE ON vector_features
    FOR EACH ROW EXECUTE FUNCTION update_vector_feature_count();

-- Auto-update vector_datasets.bbox from features
CREATE OR REPLACE FUNCTION update_vector_dataset_bbox()
RETURNS TRIGGER AS $$
DECLARE
    v_env GEOMETRY;
BEGIN
    SELECT ST_Envelope(ST_Collect(geom)) INTO v_env
    FROM vector_features
    WHERE dataset_id = COALESCE(NEW.dataset_id, OLD.dataset_id);

    -- ST_Envelope of a single point returns a Point, not a Polygon.
    -- Buffer it slightly so the bbox column (Polygon) is always valid.
    IF v_env IS NOT NULL AND GeometryType(v_env) IN ('POINT', 'LINESTRING') THEN
        v_env := ST_Envelope(ST_Buffer(v_env::geography, 10)::geometry);
    END IF;

    UPDATE vector_datasets
    SET bbox = v_env
    WHERE id = COALESCE(NEW.dataset_id, OLD.dataset_id);
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_vector_dataset_bbox ON vector_features;
CREATE TRIGGER trg_vector_dataset_bbox
    AFTER INSERT OR UPDATE OR DELETE ON vector_features
    FOR EACH ROW EXECUTE FUNCTION update_vector_dataset_bbox();

COMMIT;
