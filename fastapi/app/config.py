from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    PG_HOST: str = "globe-postgres"
    PG_PORT: int = 5432
    PG_DATABASE: str = "gis"
    PG_USER: str = "postgres"
    PG_PASSWORD: str = ""  # Direct env var (takes priority if set)
    POSTGRES_PASSWORD_FILE: str = "/run/secrets/postgres_password"

    # Redis
    REDIS_HOST: str = "globe-redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""  # Direct env var for Redis auth

    # MinIO
    MINIO_ENDPOINT: str = "globe-minio:9000"
    MINIO_ACCESS_KEY: str = "admin"
    MINIO_SECRET_KEY: str = ""  # Direct env var (takes priority if set)
    MINIO_SECRET_KEY_FILE: str = "/run/secrets/minio_password"

    # JWT
    JWT_SECRET: str = ""  # Direct env var (takes priority if set)
    JWT_SECRET_FILE: str = "/run/secrets/jwt_secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_EXPIRE_MINUTES: int = 60  # 1 hour (was 24h — reduced per security audit)
    JWT_REFRESH_EXPIRE_DAYS: int = 7

    # Dev auth bypass — when true, get_current_user short-circuits to the seeded
    # admin so the distilled app is demoable with zero login friction. Keep false
    # in any real deployment (real JWT login still works either way).
    DEV_NO_AUTH: bool = False
    DEV_USER_ID: str = "00000000-0000-0000-0000-000000000001"
    DEV_USERNAME: str = "admin"
    DEV_PASSWORD: str = "abyss"  # admin password fallback (dev real login)
    SEED_ADMIN: bool = True       # seed the demo accounts on startup

    # Deterministic demo-account seeding (admin / analyst / viewer). When
    # SEED_FORCE is true (default) each account is upserted on every boot so
    # credentials stay deterministic across restarts (ideal for the demo). Set
    # false once accounts are admin-managed so manual password changes survive.
    SEED_FORCE: bool = True
    SEED_ADMIN_PASSWORD_FILE: str = "/run/secrets/seed_admin_password"
    SEED_ANALYST_PASSWORD: str = ""   # direct env (takes priority if set)
    SEED_ANALYST_PASSWORD_FILE: str = "/run/secrets/seed_analyst_password"
    SEED_VIEWER_PASSWORD: str = ""    # direct env (takes priority if set)
    SEED_VIEWER_PASSWORD_FILE: str = "/run/secrets/seed_viewer_password"

    # Cookie security — operational sets COOKIE_SECURE=true (HTTPS-only cookies).
    # The dev overlay sets it false so the HTTP-only dev stack still gets cookies.
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"

    # Rate limiting (slowapi + Redis db /4): a permissive global default plus a
    # strict per-IP login limit.
    RATE_LIMIT_DEFAULT: str = "120/minute"
    RATE_LIMIT_LOGIN: str = "5/minute"

    # Token-blacklist behaviour when Redis is unreachable. Fail-open (default)
    # favours availability; fail-closed favours strict revocation. See
    # middleware/auth_middleware.is_token_blacklisted.
    BLACKLIST_FAIL_OPEN: bool = True

    # ---- LLM assistant (Anthropic) -------------------------------------------
    # Phase 1: a basic Claude-backed chat assistant behind /api/chat. Phase 2
    # will ground it on the platform's own data (surveys/reports/depths). The key
    # is operator-supplied (env var, or the /run/secrets/anthropic_api_key file);
    # GENERATE.sh does NOT mint it. When unset, /api/chat returns a friendly
    # "assistant not configured" rather than erroring.
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_API_KEY_FILE: str = "/run/secrets/anthropic_api_key"
    CHAT_MODEL: str = "claude-haiku-4-5"   # fast/cheap default; bump to claude-opus-4-8 for depth
    CHAT_MAX_TOKENS: int = 1024

    # Upload limits
    UPLOAD_MAX_SIZE_BYTES: int = 10 * 1024 * 1024 * 1024  # 10 GB

    # Celery (constructed dynamically in properties below)
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""

    model_config = {"env_prefix": "", "case_sensitive": True}

    def _read_secret(self, path: str, fallback: str = "") -> str:
        p = Path(path)
        if p.is_file():
            return p.read_text().strip()
        return fallback

    @property
    def postgres_password(self) -> str:
        # Direct env var takes priority over file
        if self.PG_PASSWORD:
            return self.PG_PASSWORD
        return self._read_secret(self.POSTGRES_PASSWORD_FILE, "postgres")

    @property
    def redis_password(self) -> str:
        if self.REDIS_PASSWORD:
            return self.REDIS_PASSWORD
        return self._read_secret("/run/secrets/redis_password", "")

    @property
    def minio_secret(self) -> str:
        # Direct env var takes priority over file
        if self.MINIO_SECRET_KEY:
            return self.MINIO_SECRET_KEY
        return self._read_secret(self.MINIO_SECRET_KEY_FILE, "minioadmin")

    @property
    def jwt_secret_key(self) -> str:
        # Direct env var takes priority over file
        if self.JWT_SECRET:
            return self.JWT_SECRET
        secret = self._read_secret(self.JWT_SECRET_FILE, "")
        if not secret:
            import warnings
            warnings.warn(
                "JWT_SECRET is not set and no secret file found at "
                f"{self.JWT_SECRET_FILE}. Using insecure fallback — "
                "DO NOT deploy to production without setting a secret!",
                RuntimeWarning,
                stacklevel=2,
            )
            return "changeme-insecure-dev-secret"
        return secret

    @property
    def seed_admin_password(self) -> str:
        # Admin always exists (the DEV_NO_AUTH bypass + FK targets need it), so it
        # always resolves to *some* password. Operational secret file wins; an
        # absent OR empty file (e.g. the dev placeholder) falls back to
        # DEV_PASSWORD (dev default 'abyss').
        return self._read_secret(self.SEED_ADMIN_PASSWORD_FILE, "") or self.DEV_PASSWORD

    @property
    def seed_analyst_password(self) -> str:
        # Direct env → secret file → empty (empty ⇒ account not seeded).
        if self.SEED_ANALYST_PASSWORD:
            return self.SEED_ANALYST_PASSWORD
        return self._read_secret(self.SEED_ANALYST_PASSWORD_FILE, "")

    @property
    def seed_viewer_password(self) -> str:
        if self.SEED_VIEWER_PASSWORD:
            return self.SEED_VIEWER_PASSWORD
        return self._read_secret(self.SEED_VIEWER_PASSWORD_FILE, "")

    @property
    def anthropic_api_key(self) -> str:
        # Direct env var takes priority over the secret file.
        if self.ANTHROPIC_API_KEY:
            return self.ANTHROPIC_API_KEY
        return self._read_secret(self.ANTHROPIC_API_KEY_FILE, "")

    def _redis_url(self, db: int) -> str:
        """Build redis://[:<pwd>@]host:port/<db>, URL-encoding the password so a
        base64 secret (which can contain +, /, =) doesn't corrupt the URL.
        redis-py/kombu un-quote it back to the raw password on connect."""
        pwd = self.redis_password
        auth = f":{quote(pwd, safe='')}@" if pwd else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{db}"

    @property
    def rate_limit_redis_url(self) -> str:
        """Dedicated Redis db (/4) for slowapi rate-limit counters.

        Kept separate from the token blacklist (/3) and Celery (/1,/2) so
        rate-limit churn never evicts those keys.
        """
        return self._redis_url(4)

    def validate_operational(self) -> None:
        """Fail-fast guard for operational (non-dev) boot.

        When auth is enforced (DEV_NO_AUTH=false), refuse to start with an empty
        or well-known JWT secret — otherwise access tokens would be trivially
        forgeable. No-op in dev (DEV_NO_AUTH=true).
        """
        if self.DEV_NO_AUTH:
            return
        insecure = {"", "changeme-insecure-dev-secret", "abyss-dev-secret-change-me"}
        if self.jwt_secret_key in insecure:
            raise RuntimeError(
                "Refusing to start in operational mode (DEV_NO_AUTH=false) with an "
                "insecure JWT secret. Provide a strong secret via the JWT_SECRET env "
                "var or the /run/secrets/jwt_secret file "
                "(run: bash secure-config/keys/GENERATE.sh)."
            )

    @property
    def redis_url(self) -> str:
        """Redis URL for general-purpose use (token blacklist, etc.)."""
        return self._redis_url(3)

    @property
    def celery_broker(self) -> str:
        return self.CELERY_BROKER_URL or self._redis_url(1)

    @property
    def celery_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND or self._redis_url(2)

    @property
    def DATABASE_URL(self) -> str:
        # URL-encode the password (base64 secrets contain +, /, = which would
        # otherwise corrupt the SQLAlchemy URL).
        return (
            f"postgresql+psycopg2://{self.PG_USER}:{quote(self.postgres_password, safe='')}"
            f"@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DATABASE}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
