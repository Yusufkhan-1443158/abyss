"""Durable results store for finished bathymetry jobs (R3 / R4).

A single JSON index file under ``Very_HR_Results/_jobs/results_index.json``
records every *finished* compute run so the Results section survives a
backend restart. This is intentionally additive — it does NOT replace the
``vhr_jobs`` running-job queue; it is the persistent catalogue of completed
outputs that the running queue feeds into on completion.

Each entry::

    {
      "id":           "<12-hex>",          # stable result id
      "name":         "Khalifa Port (VHR)",
      "roi_bbox":     [west, south, east, north],
      "resolution":   "vhr" | "10m" | "20m",
      "date":         1717200000.0,        # finished_at epoch seconds
      "status":       "done" | "failed",
      "output_path":  "/abs/path/to/out.tif",
      "output_name":  "out.tif",           # basename for the download route
      "size_bytes":   123456,
      "metrics":      {"rmse_m": .., "r2": .., "coverage_pct": .., "catzoc": ..} | null,
      "job_id":       "<source vhr_jobs id>" | null,
      "error":        "<traceback head>" | null
    }

Concurrency: a coarse file lock (``fcntl``) guards read-modify-write so two
worker threads finishing at once can't clobber the index. Falls back to a
best-effort write if ``fcntl`` is unavailable.
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
# NOTE: the durable index lives in a dedicated `_results` subdir — NOT directly
# under `Very_HR_Results/_jobs/`, because vhr_jobs.list_jobs() globs
# `_jobs/*.json` and would otherwise try to parse this index as a job record.
RESULTS_ROOT = _ROOT / "Very_HR_Results" / "_jobs" / "_results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
# Finished output files (GeoTIFF/JSON/CSV) live here so the download route has
# one well-known, traversal-safe directory to serve from.
OUTPUTS_DIR = RESULTS_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = RESULTS_ROOT / "results_index.json"
_LOCK_PATH = RESULTS_ROOT / ".results_index.lock"


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
    """Best-effort cross-process advisory lock around index mutation."""

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


def _safe_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "._-") or "result.bin"


def register_result(
    *,
    name: str,
    roi_bbox: list[float],
    resolution: str,
    status: str = "done",
    output_bytes: bytes | None = None,
    output_filename: str | None = None,
    output_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    job_id: str | None = None,
    error: str | None = None,
    result_id: str | None = None,
) -> dict[str, Any]:
    """Persist a finished result into the durable index.

    If ``output_bytes`` is provided it is written into ``OUTPUTS_DIR`` under a
    collision-proof ``<result_id>__<safe_filename>`` name. Otherwise an existing
    ``output_path`` on disk is referenced in place (size read from disk).
    Returns the public entry dict.
    """
    rid = result_id or uuid.uuid4().hex[:12]
    out_path = None
    out_name = None
    size_bytes = None

    if output_bytes is not None:
        fname = _safe_name(output_filename or "result.bin")
        stored = OUTPUTS_DIR / f"{rid}__{fname}"
        stored.write_bytes(output_bytes)
        out_path = str(stored)
        out_name = stored.name
        size_bytes = len(output_bytes)
    elif output_path:
        p = Path(output_path)
        out_path = str(p)
        out_name = p.name
        try:
            size_bytes = p.stat().st_size if p.exists() else None
        except Exception:
            size_bytes = None

    entry = {
        "id": rid,
        "name": name,
        "roi_bbox": list(roi_bbox) if roi_bbox else None,
        "resolution": resolution if resolution in ("vhr", "10m", "20m") else "10m",
        "date": time.time(),
        "status": status if status in ("done", "failed") else "done",
        "output_path": out_path,
        "output_name": out_name,
        "size_bytes": size_bytes,
        "metrics": metrics or None,
        "job_id": job_id,
        "error": (str(error)[:1000] if error else None),
    }

    with _Lock():
        entries = _load()
        # De-dupe by id (idempotent re-register).
        entries = [e for e in entries if e.get("id") != rid]
        entries.append(entry)
        _save(entries)
    return entry


def list_results(limit: int = 200) -> list[dict[str, Any]]:
    entries = _load()
    entries.sort(key=lambda e: e.get("date", 0), reverse=True)
    return entries[:limit]


def get_result(result_id: str) -> dict[str, Any] | None:
    for e in _load():
        if e.get("id") == result_id:
            return e
    return None


def delete_result(result_id: str, remove_file: bool = False) -> bool:
    with _Lock():
        entries = _load()
        target = next((e for e in entries if e.get("id") == result_id), None)
        if target is None:
            return False
        if remove_file and target.get("output_path"):
            try:
                p = Path(target["output_path"])
                # Only delete files we own (under OUTPUTS_DIR) to avoid nuking
                # shared engine artefacts.
                if p.exists() and OUTPUTS_DIR in p.parents:
                    p.unlink()
            except Exception:
                pass
        entries = [e for e in entries if e.get("id") != result_id]
        _save(entries)
    return True


def resolve_output(result_id: str) -> Path | None:
    """Traversal-safe resolution of a result's output file for download."""
    e = get_result(result_id)
    if not e or not e.get("output_path"):
        return None
    p = Path(e["output_path"]).resolve()
    if not p.exists():
        return None
    return p
