"""Durable order store for the Very-High-Resolution image *purchase* flow.

Sibling of ``jobs_store.py`` (the R3/R4 results catalogue). Orders persist as a
single JSON index so a purchase survives a backend restart and can be listed in
the Results panel. Intentionally additive — it does not touch the results index.

Each entry::

    {
      "order_id":       "<12-hex>",
      "bbox":           [west, south, east, north],
      "geometry":       <geojson polygon> | null,
      "area_km2":       12.34,
      "price_eur":      61.7,
      "currency":       "EUR",
      "collection":     "DEM... / Airbus Pleiades",
      "date_from":      "2024-01-01" | null,
      "date_to":        "2024-12-31" | null,
      "payment_status": "pending" | "paid" | "failed",
      "session_id":     "<stripe-or-sim session id>" | null,
      "simulated":      true | false,            # stripe stubbed?
      "sh_status":      null | "sandbox" | "created" | "confirmed" | "error",
      "sh_order_id":    "<sh order id>" | null,
      "sh_request":     <the would-be / actual SH request body> | null,
      "result_id":      "<jobs_store result id>" | null,
      "created_at":     1717200000.0,
      "updated_at":     1717200000.0,
      "error":          null
    }

Concurrency: coarse ``fcntl`` advisory lock around read-modify-write.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

try:
    import fcntl  # POSIX only
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

_ROOT = Path(__file__).resolve().parent.parent
# Live alongside the results catalogue, but in their own subdir so neither
# vhr_jobs.list_jobs() nor jobs_store ever globs an order as a job/result.
ORDERS_ROOT = _ROOT / "Very_HR_Results" / "_jobs" / "_orders"
ORDERS_ROOT.mkdir(parents=True, exist_ok=True)
INDEX_PATH = ORDERS_ROOT / "orders_index.json"
_LOCK_PATH = ORDERS_ROOT / ".orders_index.lock"


def _load() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    try:
        data = json.loads(INDEX_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(entries: list[dict[str, Any]]) -> None:
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2, default=str))
    os.replace(tmp, INDEX_PATH)


class _Lock:
    def __enter__(self):
        self._fh = None
        if fcntl is not None:
            try:
                self._fh = open(_LOCK_PATH, "w")
                fcntl.flock(self._fh, fcntl.LOCK_EX)
            except Exception:
                self._fh = None
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass


def create_order(**fields: Any) -> dict[str, Any]:
    oid = fields.pop("order_id", None) or uuid.uuid4().hex[:12]
    now = time.time()
    entry = {
        "order_id": oid,
        "bbox": fields.get("bbox"),
        "geometry": fields.get("geometry"),
        "area_km2": fields.get("area_km2"),
        "price_eur": fields.get("price_eur"),
        "currency": fields.get("currency", "EUR"),
        "collection": fields.get("collection"),
        "resolution_m": fields.get("resolution_m"),
        "provider": fields.get("provider"),
        "date_from": fields.get("date_from"),
        "date_to": fields.get("date_to"),
        "payment_status": fields.get("payment_status", "pending"),
        "session_id": fields.get("session_id"),
        "simulated": bool(fields.get("simulated", False)),
        "sh_status": fields.get("sh_status"),
        "sh_order_id": fields.get("sh_order_id"),
        "sh_request": fields.get("sh_request"),
        "result_id": fields.get("result_id"),
        "created_at": now,
        "updated_at": now,
        "error": fields.get("error"),
    }
    with _Lock():
        entries = _load()
        entries = [e for e in entries if e.get("order_id") != oid]
        entries.append(entry)
        _save(entries)
    return entry


def update_order(order_id: str, **patch: Any) -> dict[str, Any] | None:
    with _Lock():
        entries = _load()
        target = next((e for e in entries if e.get("order_id") == order_id), None)
        if target is None:
            return None
        target.update(patch)
        target["updated_at"] = time.time()
        _save(entries)
        return dict(target)


def list_orders(limit: int = 500) -> list[dict[str, Any]]:
    entries = _load()
    entries.sort(key=lambda e: e.get("created_at", 0), reverse=True)
    return entries[:limit]


def get_order(order_id: str) -> dict[str, Any] | None:
    for e in _load():
        if e.get("order_id") == order_id:
            return e
    return None
