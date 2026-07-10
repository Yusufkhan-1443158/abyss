#!/usr/bin/env bash
# Generates the Docker-secret files required by docker-compose (operational).
# Abyss core: jwt-secret, postgres/redis/minio passwords, minio-kms-key, and the
# three seed-account passwords (admin/analyst/viewer). Legacy extras
# (elasticsearch/grafana/nextcloud/airflow) are kept for the broader stack.
# Run from the package root:  bash secure-config/keys/GENERATE.sh
# Safe to re-run: it will refuse to clobber existing files unless --force.
set -euo pipefail

FORCE=0
if [ "${1:-}" = "--force" ]; then FORCE=1; fi

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

gen() {
  local name="$1" cmd="$2"
  if [ -s "$name" ] && [ "$FORCE" -ne 1 ]; then
    echo "skip  $name (already exists; use --force to overwrite)"
    return
  fi
  eval "$cmd" > "$name"
  chmod 600 "$name"
  echo "wrote $name"
}

gen postgres-password.txt       "openssl rand -base64 32"
gen redis-password.txt          "openssl rand -base64 32"
gen minio-password.txt          "openssl rand -base64 32"
gen jwt-secret.txt              "openssl rand -base64 64"

# MinIO built-in single-key KMS for SSE-S3 bucket encryption (A6).
# Format MUST be <key-name>:<base64-encoded-32-byte-key>.
gen minio-kms-key.txt           'echo "abyss-key-1:$(openssl rand -base64 32)"'

# Seed/demo-account passwords — bcrypt-hashed by fastapi/app/seed.py at boot
# (base64-of-24-bytes stays well under bcrypt's 72-byte input limit).
gen seed-admin-password.txt     "openssl rand -base64 24"
gen seed-analyst-password.txt   "openssl rand -base64 24"
gen seed-viewer-password.txt    "openssl rand -base64 24"

# Anthropic API key for the /api/chat assistant is OPERATOR-SUPPLIED, not minted.
# Touch an empty placeholder so docker-compose can mount the secret; paste your
# real key into it to enable the assistant (empty ⇒ "assistant not configured").
if [ ! -s anthropic-api-key.txt ]; then
  : > anthropic-api-key.txt
  chmod 600 anthropic-api-key.txt
  echo "touched anthropic-api-key.txt — paste your Anthropic API key into it to enable the assistant"
fi

# --- Legacy extras for the broader stack (not used by the Abyss compose) -----
gen elasticsearch-password.txt  "openssl rand -base64 32"
gen grafana-password.txt        "openssl rand -base64 32"
gen nextcloud-db-password.txt   "openssl rand -base64 32"

# Airflow Fernet key MUST be a urlsafe-base64-encoded 32-byte key.
# Use Python's cryptography lib if available, else fall back to openssl.
if command -v python3 >/dev/null 2>&1 && python3 -c "from cryptography.fernet import Fernet" 2>/dev/null; then
  gen airflow-fernet.txt "python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
else
  gen airflow-fernet.txt "python3 -c 'import os, base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'"
fi

echo
echo "Secret files generated in: $DIR"
echo
echo "Next — generate a self-signed TLS cert for the demo (nginx :443), from the package root:"
echo "  mkdir -p secure-config/certs"
echo "  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \\"
echo "    -keyout secure-config/certs/abyss.key -out secure-config/certs/abyss.crt \\"
echo "    -subj \"/CN=abyss.local\""
echo
echo "Then bring up the operational stack (fresh volumes honour the new secrets):"
echo "  docker compose down -v && docker compose up --build      # https://localhost:9443"
echo
echo "Demo logins (password = the file contents above):"
echo "  admin   : $(cat seed-admin-password.txt 2>/dev/null || echo '<seed-admin-password.txt>')"
echo "  analyst : $(cat seed-analyst-password.txt 2>/dev/null || echo '<seed-analyst-password.txt>')"
echo "  viewer  : $(cat seed-viewer-password.txt 2>/dev/null || echo '<seed-viewer-password.txt>')"
