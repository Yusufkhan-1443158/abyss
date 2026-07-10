-- 001_extensions.sql
-- Enable required PostgreSQL extensions for Intel Globe v3.0
-- Idempotent: safe to run multiple times

BEGIN;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

COMMIT;
