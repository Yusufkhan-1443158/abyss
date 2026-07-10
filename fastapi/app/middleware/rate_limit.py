"""Redis-backed, proxy-aware rate limiting (slowapi).

A single shared ``Limiter`` instance:
- keys requests by the real client IP (honouring nginx's ``X-Forwarded-For``,
  mirroring the parse in ``middleware/audit_logger.py``),
- stores counters in a dedicated Redis db (/4) so rate-limit churn never evicts
  the token blacklist (/3) or Celery (/1, /2) keys,
- ``swallow_errors=True`` so a Redis hiccup degrades to *allow* (availability)
  rather than 500-ing every request.

NOTE: uvicorn must run with ``--proxy-headers --forwarded-allow-ips=*`` (see the
Dockerfile) so ``request.client.host`` is the real client when ``X-Forwarded-For``
is absent — otherwise every request behind nginx would share one bucket.
"""

from __future__ import annotations

import redis as _redis
from fastapi import HTTPException, Request, status
from slowapi import Limiter

from ..config import get_settings

settings = get_settings()


def client_ip(request: Request) -> str:
    """Real client IP — X-Forwarded-For first hop, then the socket peer."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


limiter = Limiter(
    key_func=client_ip,
    storage_uri=settings.rate_limit_redis_url,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    swallow_errors=True,
    headers_enabled=True,
)


# ---------------------------------------------------------------------------
# Explicit per-endpoint login limiter.
#
# slowapi's @limiter.limit decorator mangles a FastAPI endpoint's parameter
# annotations (FastAPI then mis-reads the Pydantic body / Depends as query
# params → 422). The global SlowAPIMiddleware default limit is unaffected and
# stays on, but the *strict* login limit is enforced here instead, as a plain
# Redis fixed-window counter on the same dedicated db (/4). Fail-open on a Redis
# error, consistent with the limiter's swallow_errors.
# ---------------------------------------------------------------------------

_PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
_rl_redis: _redis.Redis | None = None


def _get_rl_redis() -> _redis.Redis:
    global _rl_redis
    if _rl_redis is None:
        _rl_redis = _redis.from_url(settings.rate_limit_redis_url, decode_responses=True)
    return _rl_redis


def _parse_limit(spec: str) -> tuple[int, int]:
    """'5/minute' -> (5, 60). Falls back to (5, 60) on a malformed spec."""
    try:
        count, _, period = spec.partition("/")
        return int(count.strip()), _PERIODS.get(period.strip().lower(), 60)
    except (ValueError, AttributeError):
        return 5, 60


def check_login_rate_limit(request: Request) -> None:
    """Raise 429 if the client IP exceeded RATE_LIMIT_LOGIN in the window."""
    count, window = _parse_limit(settings.RATE_LIMIT_LOGIN)
    key = f"login_rl:{client_ip(request)}"
    try:
        r = _get_rl_redis()
        current = r.incr(key)
        if current == 1:
            r.expire(key, window)
    except _redis.RedisError:
        return  # fail-open for availability
    if current > count:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please wait and try again.",
        )
