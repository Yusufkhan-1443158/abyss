#!/usr/bin/env bash
# Abyss single-container entrypoint (operational). Runs all one-time setup
# synchronously, THEN hands off to supervisord for the long-running services.
#   1. generate/refresh Docker-style secrets into /run/secrets
#   2. render redis + minio credentials
#   3. initialise the Postgres data dir (first boot) and run SQL migrations
#   4. exec supervisord (postgres, 2x redis, minio, bathymetry, fastapi,
#      celery worker+beat, nginx)
set -euo pipefail

SECRETS=/run/secrets
KEYS=/app/secure-config/keys
PGDATA=/var/lib/postgresql/data
PGBIN="$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)"
mkdir -p "$SECRETS"

echo "[entrypoint] generating secrets ..."
bash "$KEYS/GENERATE.sh" >/dev/null 2>&1 || true

# Map generated key files → the /run/secrets/* names config.py + services read.
cp -f "$KEYS/postgres-password.txt"     "$SECRETS/postgres_password"
cp -f "$KEYS/redis-password.txt"        "$SECRETS/redis_password"
cp -f "$KEYS/minio-password.txt"        "$SECRETS/minio_password"
cp -f "$KEYS/minio-kms-key.txt"         "$SECRETS/minio_kms_key"
cp -f "$KEYS/jwt-secret.txt"            "$SECRETS/jwt_secret"
cp -f "$KEYS/seed-admin-password.txt"   "$SECRETS/seed_admin_password"
cp -f "$KEYS/seed-analyst-password.txt" "$SECRETS/seed_analyst_password"
cp -f "$KEYS/seed-viewer-password.txt"  "$SECRETS/seed_viewer_password"
# Fixed admin password (override the random one GENERATE.sh just wrote), so the
# admin login is stable across every boot/redeploy instead of regenerating.
# Override without a rebuild by setting SEED_ADMIN_PASSWORD in the pod env.
printf '%s' "${SEED_ADMIN_PASSWORD:-Orbion26}" > "$SECRETS/seed_admin_password"
# Anthropic key: operator-supplied via env; empty ⇒ assistant "not configured".
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  printf '%s' "$ANTHROPIC_API_KEY" > "$SECRETS/anthropic_api_key"
else
  : > "$SECRETS/anthropic_api_key"
fi
chmod 600 "$SECRETS"/* || true

PGPASS="$(cat "$SECRETS/postgres_password")"
REDISPASS="$(cat "$SECRETS/redis_password")"

# ---- Redis (password-protected broker/blacklist/rate-limit/cache) -----------
mkdir -p /etc/redis
cat > /etc/redis/globe.conf <<EOF
port 6379
requirepass "$REDISPASS"
save 60 1
loglevel warning
dir /var/lib/redis
EOF
mkdir -p /var/lib/redis

# ---- MinIO credentials (exported → inherited by the supervisord child) ------
export MINIO_ROOT_USER="admin"
export MINIO_ROOT_PASSWORD="$(cat "$SECRETS/minio_password")"
export MINIO_KMS_SECRET_KEY="$(cat "$SECRETS/minio_kms_key")"
# The bathymetry service reads its MinIO secret from this env var.
export MINIO_SECRET_KEY="$MINIO_ROOT_PASSWORD"
mkdir -p /data

# ---- PostgreSQL: init on first boot, then run migrations --------------------
mkdir -p "$PGDATA"
chown -R postgres:postgres "$PGDATA" /var/lib/postgresql
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo "[entrypoint] initialising Postgres data dir ..."
  su -s /bin/bash postgres -c "$PGBIN/initdb -D $PGDATA --auth-local=trust --auth-host=scram-sha-256 --encoding=UTF8"
  {
    echo "listen_addresses = '127.0.0.1'"
    echo "port = 5432"
    echo "password_encryption = scram-sha-256"
  } >> "$PGDATA/postgresql.conf"
  echo "host all all 127.0.0.1/32 scram-sha-256" >> "$PGDATA/pg_hba.conf"
fi

echo "[entrypoint] starting Postgres (setup phase) ..."
su -s /bin/bash postgres -c "$PGBIN/pg_ctl -D $PGDATA -w -t 60 start"
su -s /bin/bash postgres -c "$PGBIN/psql -v ON_ERROR_STOP=1 --username postgres -c \"ALTER USER postgres PASSWORD '$PGPASS';\""
if ! su -s /bin/bash postgres -c "$PGBIN/psql -tAc \"SELECT 1 FROM pg_database WHERE datname='gis'\"" | grep -q 1; then
  su -s /bin/bash postgres -c "$PGBIN/psql -v ON_ERROR_STOP=1 --username postgres -c \"CREATE DATABASE gis;\""
fi

echo "[entrypoint] running migrations ..."
export POSTGRES_PASSWORD="$PGPASS"
bash /app/migrations/run_migrations.sh 127.0.0.1 5432 gis postgres

echo "[entrypoint] stopping setup Postgres ..."
su -s /bin/bash postgres -c "$PGBIN/pg_ctl -D $PGDATA -w -t 60 stop"

echo "[entrypoint] handing off to supervisord ..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/abyss.conf
