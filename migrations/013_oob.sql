-- Order of Battle: units
CREATE TABLE IF NOT EXISTS oob_units (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    designation VARCHAR(100),
    unit_type VARCHAR(50) NOT NULL,
    echelon VARCHAR(50),
    parent_id UUID REFERENCES oob_units(id) ON DELETE SET NULL,
    affiliation VARCHAR(30) DEFAULT 'hostile',
    country VARCHAR(100),
    symbol_code VARCHAR(50),
    location GEOGRAPHY(Point, 4326),
    strength_current INTEGER,
    strength_authorized INTEGER,
    commander VARCHAR(255),
    status VARCHAR(30) DEFAULT 'active',
    last_known_activity TEXT,
    last_observed TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_oob_units_parent ON oob_units(parent_id);
CREATE INDEX IF NOT EXISTS idx_oob_units_affiliation ON oob_units(affiliation);

-- Order of Battle: equipment
CREATE TABLE IF NOT EXISTS oob_equipment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    unit_id UUID NOT NULL REFERENCES oob_units(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    equipment_type VARCHAR(100),
    model VARCHAR(100),
    quantity INTEGER DEFAULT 1,
    status VARCHAR(30) DEFAULT 'operational',
    symbol_code VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_oob_equipment_unit ON oob_equipment(unit_id);
