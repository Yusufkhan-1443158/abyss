"""Read-through hot-layer cache for the XYZ tile endpoint (migration Step 2).

This is the first production drop-in from the super-worker PoC. It sits IN FRONT
of the existing GDAL renderer in rasters.py:get_tile as a read-through cache:

    request -> PostGIS query (cheap) -> cache key -> GET hot layer
        HIT  -> serve PNG from RAM (microseconds), skip the 27-88ms GDAL render
        MISS -> render exactly as today -> store -> serve

DESIGN GUARANTEES (so this is safe to ship behind a flag):

  * REVERSIBLE / OFF BY DEFAULT. Everything is gated on ``enabled()`` (env
    SW_TILE_CACHE). With the flag off, get_tile never calls into this module and
    behaves byte-identically to today.
  * CORRECT BY CONSTRUCTION, no invalidation infra required. The key folds in the
    matched raster id-set AND each raster's updated_at AND the date filter. A new
    upload, an SR-upscale, or an in-place re-process changes the id-set or an
    updated_at -> a different key -> a clean MISS -> a fresh render. The stale
    entry is simply orphaned and reclaimed by TTL / L2 LRU. So correctness never
    depends on a pub/sub subscriber existing.
  * NEVER 5xx ON A CACHE FAULT. Every L2 (Redis) op is wrapped; on any error the
    layer degrades to L1-only (or render-on-miss) and increments a stat. A dead
    cache must never break tile serving.
  * SHARED-REDIS SAFE. L2 uses a DEDICATED redis via SW_REDIS_URL (bounded,
    allkeys-lru in deployment). If SW_REDIS_URL is unset, L2 is DISABLED and we
    run L1-only (per-worker RAM) — we deliberately do NOT fall back to the shared
    globe-redis, which would risk evicting the Celery broker / token-blacklist.

The matching warmer + predictor (Step 3) will reuse the SAME key grammar and the
``sw:invalidate`` channel this module publishes to.
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from datetime import date as _date, datetime
from typing import Any, Iterable

# 1-byte negative sentinel: "no raster intersects this tile" (ocean/edge). Caching
# it stops vast empty viewports from re-running the PostGIS query + empty render.
NEGATIVE_SENTINEL = b"\x00"

_ENABLED = os.getenv("SW_TILE_CACHE", "0").lower() in ("1", "true", "yes", "on")
_REDIS_URL = os.getenv("SW_REDIS_URL") or None          # DEDICATED redis; None => L1-only
_TTL_S = int(os.getenv("SW_TILE_TTL_S", "3600"))
_L1_MAX_BYTES = int(os.getenv("SW_TILE_L1_MB", "128")) * 1024 * 1024
_INVALIDATE_CHANNEL = "sw:invalidate"


def enabled() -> bool:
    """True if the read-through cache is switched on (env SW_TILE_CACHE)."""
    return _ENABLED


# ---------------------------------------------------------------------------
# L1: per-process byte-bounded LRU.  L2: dedicated Redis (optional).
# ---------------------------------------------------------------------------
_lock = threading.RLock()
_l1: "OrderedDict[str, bytes]" = OrderedDict()
_l1_bytes = 0
_stats = {"hits": 0, "misses": 0, "l1_hits": 0, "l2_hits": 0, "l2_down": 0}

_redis: Any = None
_redis_init = False


def _is_shared_redis(url: str) -> bool:
    """Refuse to use the SHARED globe-redis as the tile L2.

    The shared redis holds the Celery broker (db1), results (db2) and the token
    blacklist (db3); filling it with tile bytes could evict them under memory
    pressure. We compare the SW_REDIS_URL host:port against the app's
    REDIS_HOST/REDIS_PORT (a different logical db is NOT enough — maxmemory is
    per-instance). On a match we warn loudly and stay L1-only.
    """
    try:
        from urllib.parse import urlparse
        u = urlparse(url)
        shared_host = os.getenv("REDIS_HOST")
        shared_port = int(os.getenv("REDIS_PORT", "6379"))
        if shared_host and u.hostname == shared_host and (u.port or 6379) == shared_port:
            import warnings
            warnings.warn(
                f"SW_REDIS_URL points at the SHARED redis ({shared_host}:{shared_port}); "
                "refusing to use it as the tile L2 (would risk evicting the Celery "
                "broker/blacklist). Running L1-only — point SW_REDIS_URL at a DEDICATED redis.",
                RuntimeWarning, stacklevel=2,
            )
            return True
    except Exception:
        pass
    return False


def _l2() -> Any:
    """Lazy DEDICATED-Redis client (binary values). None if SW_REDIS_URL is unset
    or points at the shared redis (see _is_shared_redis)."""
    global _redis, _redis_init
    if _redis_init:
        return _redis
    _redis_init = True
    if _REDIS_URL and not _is_shared_redis(_REDIS_URL):
        try:
            import redis  # local import so the module loads even if redis is absent
            _redis = redis.from_url(_REDIS_URL, decode_responses=False)
        except Exception:
            _redis = None
    return _redis


def has_l2() -> bool:
    """True if a dedicated L2 redis is configured & usable (lets the async caller
    skip an asyncio.to_thread hop entirely in the common L1-only deployment)."""
    return _l2() is not None


# ---------------------------------------------------------------------------
# Key grammar (shared with the Step-3 warmer)
# ---------------------------------------------------------------------------
def _ts(v: Any) -> str:
    if isinstance(v, (datetime, _date)):
        return v.isoformat()
    return str(v or "")


def key_for(z: int, x: int, y: int,
            date_exact: _date | None, date_le: datetime | None,
            rasters: Iterable[Any]) -> str:
    """Build the tile key from the matched raster set + date filter.

    rasters are RasterCatalog rows (need .id, .updated_at, .acquisition_date).
    The signature folds in updated_at so an in-place re-process auto-invalidates.
    """
    rs = list(rasters)
    items = sorted(f"{r.id}:{_ts(getattr(r, 'updated_at', None))}" for r in rs)
    sig = hashlib.sha1(",".join(items).encode("utf-8")).hexdigest()[:12] if items else "0" * 12
    if date_exact is not None:
        dk = date_exact.isoformat() if hasattr(date_exact, "isoformat") else str(date_exact)
    elif date_le is not None:
        # date<= is snapshot-fragile: a later backfill with an earlier
        # acquisition_date legitimately changes the result set for the SAME
        # cutoff. max_acq here is DERIVED from the matched rasters (not an input),
        # so folding it in means a backfill yields a new key -> a stale range tile
        # can never be served. (The id-set sig already covers additions/removals;
        # this also covers a same-set re-date.)
        max_acq = max((r.acquisition_date for r in rs if getattr(r, "acquisition_date", None)),
                      default=None)
        dk = f"le:{date_le.isoformat()}:{max_acq.isoformat() if max_acq else 'none'}"
    else:
        dk = "all"
    return f"sw:tile:{z}:{x}:{y}:{dk}:{sig}"


# ---------------------------------------------------------------------------
# get / put  (never raise)
# ---------------------------------------------------------------------------
def l1_get(key: str) -> bytes | None:
    """Memory-only L1 lookup — does NO I/O, safe to call inline on the event loop.
    Returns None on an L1 miss WITHOUT counting it (the miss is counted by l2_get,
    so a composed l1->l2 lookup counts exactly one outcome)."""
    with _lock:
        v = _l1.get(key)
        if v is not None:
            _l1.move_to_end(key)
            _stats["hits"] += 1
            _stats["l1_hits"] += 1
            return v
    return None


def l2_get(key: str) -> bytes | None:
    """L2 (Redis) lookup — does network I/O, so call it via asyncio.to_thread from
    an async handler. Promotes a hit into L1. Counts the terminal hit/miss."""
    r = _l2()
    if r is not None:
        try:
            v = r.get(key)
        except Exception:
            with _lock:
                _stats["l2_down"] += 1
            v = None
        if v is not None:
            _l1_put(key, v)
            with _lock:
                _stats["hits"] += 1
                _stats["l2_hits"] += 1
            return v
    with _lock:
        _stats["misses"] += 1
    return None


def l1_put(key: str, value: bytes) -> None:
    """Memory-only L1 store — safe to call inline (no I/O)."""
    _l1_put(key, value)


def l2_put(key: str, value: bytes, ttl: int | None = None) -> None:
    """L2 (Redis) store — network I/O; call via asyncio.to_thread. Never raises."""
    r = _l2()
    if r is not None:
        try:
            r.set(key, value, ex=ttl or _TTL_S)
        except Exception:
            with _lock:
                _stats["l2_down"] += 1


def get(key: str) -> bytes | None:
    """Full L1->L2 lookup for SYNC callers + tests. Async handlers should instead
    call l1_get inline and l2_get via asyncio.to_thread (see rasters.py) so a slow
    L2 never blocks the event loop."""
    v = l1_get(key)
    return v if v is not None else l2_get(key)


def put(key: str, value: bytes, ttl: int | None = None) -> None:
    """Full store (L1 + L2) for SYNC callers + tests."""
    l1_put(key, value)
    l2_put(key, value, ttl)


def _l1_put(key: str, value: bytes) -> None:
    global _l1_bytes
    with _lock:
        if key in _l1:
            _l1_bytes -= len(_l1[key])
        _l1[key] = value
        _l1.move_to_end(key)
        _l1_bytes += len(value)
        while _l1_bytes > _L1_MAX_BYTES and len(_l1) > 1:
            _k, _v = _l1.popitem(last=False)
            _l1_bytes -= len(_v)


def stats() -> dict:
    with _lock:
        total = _stats["hits"] + _stats["misses"]
        return {
            **_stats,
            "enabled": _ENABLED,
            "l2": "redis" if _l2() is not None else "l1-only",
            "l1_bytes": _l1_bytes,
            "l1_count": len(_l1),
            "hit_rate": round(_stats["hits"] / total, 4) if total else 0.0,
        }


# ---------------------------------------------------------------------------
# Invalidation publish (Step-1 hook; forward-wires the Step-3 warmer/subscriber).
# Correctness does NOT depend on this — the key grammar already auto-invalidates.
# This proactively tells a (future) subscriber which bbox changed so it can purge
# orphaned bytes + re-warm, instead of waiting for TTL.
# ---------------------------------------------------------------------------
def publish_invalidation(bounds: Any) -> None:
    """Best-effort PUBLISH of a changed bbox to ``sw:invalidate``. Never raises.

    ``bounds`` may be a [minx,miny,maxx,maxy] sequence or a dict; we json-encode
    whatever we are given. A no-op when SW_REDIS_URL is unset.
    """
    r = _l2()
    if r is None:
        return
    try:
        import json
        payload = json.dumps(bounds if isinstance(bounds, (list, dict, tuple)) else str(bounds))
        r.publish(_INVALIDATE_CHANNEL, payload)
    except Exception:
        pass
