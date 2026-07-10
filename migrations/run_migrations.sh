#!/bin/bash
# Intel Globe v3.0 — Migration Runner
# Runs all SQL migration files in order against the PostgreSQL database
# Usage: ./run_migrations.sh [host] [port] [database] [user]

set -euo pipefail

HOST="${1:-globe-postgres}"
PORT="${2:-5432}"
DB="${3:-gis}"
USER="${4:-postgres}"

# Resolve the DB password — env var first (matches config.py's env-priority; the
# dev overlay sets POSTGRES_PASSWORD while the inherited secret mount points at an
# empty placeholder), then the Docker secret file (operational). -s skips an empty
# placeholder file so it never yields an empty password.
if [ -n "${POSTGRES_PASSWORD:-}" ]; then
    export PGPASSWORD="$POSTGRES_PASSWORD"
elif [ -s /run/secrets/postgres_password ]; then
    export PGPASSWORD="$(cat /run/secrets/postgres_password)"
else
    echo "ERROR: No password found. Set POSTGRES_PASSWORD or mount the Docker secret."
    exit 1
fi

MIGRATIONS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Intel Globe v3.0 — Database Migration Runner ==="
echo "Host: $HOST:$PORT  Database: $DB  User: $USER"
echo ""

# Create tracking table if not exists
psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -c "
    CREATE TABLE IF NOT EXISTS _migration_history (
        id SERIAL PRIMARY KEY,
        filename TEXT UNIQUE NOT NULL,
        applied_at TIMESTAMPTZ DEFAULT NOW()
    );
" 2>/dev/null

# Run each migration in order
for sql_file in "$MIGRATIONS_DIR"/[0-9]*.sql; do
    filename="$(basename "$sql_file")"

    # Check if already applied
    applied=$(psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -tAc \
        "SELECT COUNT(*) FROM _migration_history WHERE filename = '$filename';")

    if [ "$applied" -gt 0 ]; then
        echo "  SKIP  $filename (already applied)"
        continue
    fi

    echo "  RUN   $filename ..."
    if psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -f "$sql_file" > /dev/null 2>&1; then
        psql -h "$HOST" -p "$PORT" -U "$USER" -d "$DB" -c \
            "INSERT INTO _migration_history (filename) VALUES ('$filename');" > /dev/null
        echo "  OK    $filename"
    else
        echo "  FAIL  $filename"
        exit 1
    fi
done

echo ""
echo "=== All migrations applied ==="
