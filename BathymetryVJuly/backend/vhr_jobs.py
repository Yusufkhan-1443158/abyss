"""Very-HR job queue.

Lightweight JSON-file store for long-running Very-HR bathymetry jobs.

Progress is reported HONESTLY (R1): either a REAL percentage that the
server-side pipeline actually emitted, or nothing (indeterminate). There is
no time-extrapolated / ``estimated_hours`` fake progress bar.

* **Real progress** — server-side pipelines (e.g. ``vhr_mle_pro``) write
  ``real_progress_pct`` / ``stage`` into the job file as they run. The
  frontend polls ``/api/very-hr-job/<id>/status`` and gets the exact
  percentage of the pipeline that has completed.

* **Indeterminate** — when no pipeline progress is available,
  ``progress_pct`` is ``None`` and ``indeterminate`` is ``true``; the UI
  shows a spinner, never a clock-derived bar.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = _ROOT / "Very_HR_Results" / "_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = JOBS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HOURS_PER_KM2 = 0.5  # 1 km² → 0.5 h ⇒ 20 km² → 10 h


def _bbox_area_km2(bbox: list[float] | dict[str, float]) -> float:
    if isinstance(bbox, dict):
        w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    else:
        w, s, e, n = bbox[0], bbox[1], bbox[2], bbox[3]
    lat_mid = math.radians((s + n) / 2.0)
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(lat_mid)
    return max(0.0, (n - s) * km_per_deg_lat * (e - w) * km_per_deg_lon)


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _read(job_id: str) -> dict[str, Any] | None:
    p = _job_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write(job: dict[str, Any]) -> None:
    _job_path(job["id"]).write_text(json.dumps(job, indent=2, default=str))


def create_job(bbox: list[float] | dict[str, float], params: dict[str, Any] | None = None,
               label: str | None = None) -> dict[str, Any]:
    area = _bbox_area_km2(bbox)
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    job = {
        "id": job_id,
        "label": label or "Very HR Bathymetry",
        "bbox": list(bbox) if not isinstance(bbox, dict) else [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
        "area_km2": round(area, 3),
        # R1: no estimated_hours / estimated_seconds — progress is honest
        # (real % from the pipeline, else indeterminate).
        "result_id": None,
        "params": params or {},
        "status": "processing",  # processing | done | cancelled
        "created_at": now,
        "started_at": now,
        "finished_at": None,
        "result_filename": None,
        "result_path": None,
        "result_size_bytes": None,
        "notes": None,
        # Real-progress fields (set by server-side pipelines).
        "real_progress_pct": None,
        "stage": None,
        "stage_index": None,
        "stage_total": None,
        "error": None,
    }
    _write(job)
    return _public(job)


def set_progress(job_id: str, pct: float, stage: str | None = None,
                 stage_index: int | None = None, stage_total: int | None = None) -> None:
    """Update real progress from a server-side pipeline. Idempotent; safe to
    call frequently. ``pct`` is clamped to [0, 99] — only ``save_result`` snaps
    to 100 %."""
    job = _read(job_id)
    if job is None:
        return
    job["real_progress_pct"] = float(max(0.0, min(99.0, pct)))
    if stage is not None:
        job["stage"] = str(stage)[:120]
    if stage_index is not None:
        job["stage_index"] = int(stage_index)
    if stage_total is not None:
        job["stage_total"] = int(stage_total)
    _write(job)


def mark_failed(job_id: str, error: str) -> None:
    job = _read(job_id)
    if job is None:
        return
    job["status"] = "cancelled"
    job["error"] = str(error)[:1000]
    job["finished_at"] = time.time()
    _write(job)


def is_cancelled(job_id: str) -> bool:
    job = _read(job_id)
    return bool(job and job.get("status") == "cancelled")


def status(job_id: str) -> dict[str, Any] | None:
    job = _read(job_id)
    if job is None:
        return None
    return _public(job)


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    jobs = []
    for p in JOBS_DIR.glob("*.json"):
        try:
            j = json.loads(p.read_text())
            # Defensive: only running-job records (dicts with an id) belong
            # here; skip anything else (e.g. a sibling index file).
            if isinstance(j, dict) and "id" in j:
                jobs.append(j)
        except Exception:
            continue
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return [_public(j) for j in jobs[:limit]]


def save_result(job_id: str, filename: str, data: bytes, notes: str | None = None) -> dict[str, Any] | None:
    job = _read(job_id)
    if job is None:
        return None
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._-") or "result.bin"
    out = RESULTS_DIR / f"{job_id}__{safe_name}"
    out.write_bytes(data)
    job["status"] = "done"
    job["finished_at"] = time.time()
    job["result_filename"] = safe_name
    job["result_path"] = str(out)
    job["result_size_bytes"] = len(data)
    if notes:
        job["notes"] = notes
    _write(job)
    return _public(job)


def link_result(job_id: str, result_id: str) -> None:
    """Record the durable results-store id this job produced (R3/R4)."""
    job = _read(job_id)
    if job is None:
        return
    job["result_id"] = result_id
    _write(job)


def cancel(job_id: str) -> dict[str, Any] | None:
    job = _read(job_id)
    if job is None:
        return None
    job["status"] = "cancelled"
    job["finished_at"] = time.time()
    _write(job)
    return _public(job)


def delete(job_id: str) -> bool:
    p = _job_path(job_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def _public(job: dict[str, Any]) -> dict[str, Any]:
    """Honest status payload (R1).

    No ``estimated_hours`` and no time-extrapolated progress are emitted.
    ``progress_pct`` is a REAL percentage only when the running pipeline
    reported one (``real_progress_pct``); otherwise it is ``None`` and the
    UI should show an indeterminate spinner. ``progress_source`` is one of
    ``done | cancelled | real | indeterminate``.
    """
    now = time.time()
    elapsed = max(0.0, now - job.get("started_at", now))
    real = job.get("real_progress_pct")
    if job.get("status") == "done":
        progress = 100.0
        elapsed = (job.get("finished_at") or now) - job.get("started_at", now)
        progress_source = "done"
    elif job.get("status") == "cancelled":
        progress = float(real) if isinstance(real, (int, float)) else None
        progress_source = "cancelled"
    elif isinstance(real, (int, float)):
        progress = float(real)
        progress_source = "real"
    else:
        # No real progress yet → honest indeterminate. Never extrapolate
        # from elapsed time.
        progress = None
        progress_source = "indeterminate"
    return {
        "id": job["id"],
        "label": job.get("label"),
        "bbox": job.get("bbox"),
        "area_km2": job.get("area_km2"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "elapsed_seconds": round(elapsed, 1),
        # Real percentage or None (indeterminate) — never time-derived.
        "progress_pct": (round(progress, 2) if progress is not None else None),
        "progress_source": progress_source,
        "indeterminate": progress is None and job.get("status") == "processing",
        "stage": job.get("stage"),
        "stage_index": job.get("stage_index"),
        "stage_total": job.get("stage_total"),
        "error": job.get("error"),
        "result_filename": job.get("result_filename"),
        "result_size_bytes": job.get("result_size_bytes"),
        "result_id": job.get("result_id"),
        "params": job.get("params") or {},
        "notes": job.get("notes"),
    }
