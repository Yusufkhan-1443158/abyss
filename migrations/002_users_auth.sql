-- 002_users_auth.sql
-- Users & authentication table
-- Idempotent: safe to run multiple times

BEGIN;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    email TEXT,
    role TEXT NOT NULL DEFAULT 'viewer'
        CHECK (role IN ('admin', 'analyst', 'viewer')),
    is_active BOOLEAN DEFAULT true,
    preferences JSONB DEFAULT '{}',
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- NOTE: account seeding is owned by the application (fastapi/app/seed.py
-- ensure_seed_users), which upserts admin/analyst/viewer with bcrypt hashes from
-- configured secrets. Keeping a hardcoded INSERT here would resurrect the
-- admin/admin credential drift, so the DDL is intentionally seed-free.

COMMIT;
