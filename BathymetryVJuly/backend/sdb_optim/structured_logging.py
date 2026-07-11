"""
Structured JSON logging for SDB pipeline runs.

INTEGRATION:
  - Initialized by `subsystem_interface.run_inference_job` at process start.
  - Also usable standalone: ``from backend.sdb_optim.structured_logging import setup``
  - Replaces/augments the plain ``logging.basicConfig`` in `backend/sdb_cnn_baseline.py`
    and `backend/very_hr_cbr.py` for machine-readable audit trails.

Design goals
-------------
1. Every log record emitted through the standard ``logging`` module is
   re-serialized to a one-record-per-line JSONL file (easy grep / jq / ELK).
2. GDAL/rasterio C-library warnings (printed to C-level stderr, bypassing
   Python's logging) are captured via fd-level redirection and injected as
   structured WARN records.
3. ``stage_timer(name)`` is a context manager that emits a TIMING record with
   wall-clock duration_ms.
4. ``pipeline_health_summary(jsonl_path)`` reads the run log and produces a
   compact JSON health digest (error count, warnings, slow stages, etc.)
   suitable for alerting / CI gating.

JSONL record schema
-------------------
Every line is a JSON object::

    {
        "ts":          "2026-06-16T14:23:01.123456Z",   // ISO-8601 UTC
        "level":       "INFO",                           // DEBUG/INFO/WARNING/ERROR/CRITICAL
        "stage":       "cog_chunker",                    // logger name (dotted)
        "msg":         "chunked_apply: COG written ...",
        "fields":      {"src": "/path/to/file.tif"},     // optional structured fields
        "duration_ms": null                              // filled by stage_timer
    }
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_JSONL_HANDLER: Optional["_JsonlHandler"] = None
_LOG_PATH: Optional[Path] = None
_SETUP_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 1. JSONL handler
# ---------------------------------------------------------------------------

class _JsonlHandler(logging.Handler):
    """Logging handler that writes one JSON object per line to a .jsonl file."""

    def __init__(self, path: Path, level: int = logging.DEBUG):
        super().__init__(level=level)
        self._path = path
        self._lock = threading.Lock()
        # Open in append mode; safe for multi-thread single-process usage.
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8", buffering=1)  # line-buffered

    def emit(self, record: logging.LogRecord):
        try:
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
            obj: Dict[str, Any] = {
                "ts": ts,
                "level": record.levelname,
                "stage": record.name,
                "msg": record.getMessage(),
                "fields": getattr(record, "fields", None),
                "duration_ms": getattr(record, "duration_ms", None),
            }
            if record.exc_info:
                obj["exc"] = self.formatException(record.exc_info)
            line = json.dumps(obj, ensure_ascii=False)
            with self._lock:
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        super().close()


# ---------------------------------------------------------------------------
# 2. setup — attach the JSONL handler to the root logger
# ---------------------------------------------------------------------------

def setup(
    log_path: Union[str, Path],
    level: int = logging.DEBUG,
    also_stderr: bool = True,
) -> Path:
    """Configure structured logging to *log_path* (.jsonl).

    Safe to call multiple times — subsequent calls replace the file handler
    and redirect to the new path.

    Parameters
    ----------
    log_path :
        Destination ``.jsonl`` file.  Parent directories created automatically.
    level :
        Minimum log level.
    also_stderr :
        If True (default) also attach a human-readable StreamHandler on stderr
        so logs are visible in terminal / systemd journal.

    Returns
    -------
    Path of the active log file.
    """
    global _JSONL_HANDLER, _LOG_PATH

    log_path = Path(log_path)

    with _SETUP_LOCK:
        root = logging.getLogger()
        root.setLevel(level)

        # Remove old JSONL handler if re-configuring
        if _JSONL_HANDLER is not None:
            root.removeHandler(_JSONL_HANDLER)
            _JSONL_HANDLER.close()

        handler = _JsonlHandler(log_path, level=level)
        root.addHandler(handler)
        _JSONL_HANDLER = handler
        _LOG_PATH = log_path

        if also_stderr:
            # Only add stderr handler if not already present to avoid duplicates
            has_stderr = any(
                isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
                for h in root.handlers
            )
            if not has_stderr:
                sh = logging.StreamHandler(sys.stderr)
                sh.setLevel(level)
                fmt = logging.Formatter(
                    "%(asctime)s [%(name)s] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S",
                )
                sh.setFormatter(fmt)
                root.addHandler(sh)

    logging.getLogger("sdb_optim.structured_logging").info(
        "Structured logging active",
        extra={"fields": {"log_path": str(log_path)}},
    )
    return log_path


# ---------------------------------------------------------------------------
# 3. stage_timer — context manager emitting TIMING record
# ---------------------------------------------------------------------------

@contextmanager
def stage_timer(name: str, logger: Optional[logging.Logger] = None, **fields):
    """Context manager that emits a structured TIMING record on exit.

    Usage::

        with stage_timer("cog_chunker.apply", src=str(path)):
            result = chunked_apply(...)

    The record has ``level="INFO"``, ``stage=name``, ``duration_ms=<wall_ms>``,
    and any extra *fields* passed as kwargs.

    Parameters
    ----------
    name :
        Stage name used as the logger name and ``stage`` field.
    logger :
        Logger instance to use.  Defaults to ``logging.getLogger(name)``.
    **fields :
        Arbitrary key/value pairs attached to the ``fields`` dict in the JSONL record.
    """
    if logger is None:
        logger = logging.getLogger(name)

    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"Stage '{name}' completed in {dt_ms:.1f} ms",
            extra={"fields": fields, "duration_ms": round(dt_ms, 2)},
        )


# ---------------------------------------------------------------------------
# 4. capture_geospatial_stderr — GDAL/rasterio C-library message capture
# ---------------------------------------------------------------------------

@contextmanager
def capture_geospatial_stderr(
    stage: str = "gdal",
    min_line_length: int = 5,
):
    """Capture GDAL/rasterio C-library warnings and route them to structured logging.

    Python's ``logging`` module only intercepts messages routed through it.
    GDAL and rasterio's C libraries write directly to C-level stderr via
    ``CPLError`` / ``GDALPushErrorHandler``, bypassing Python completely.

    Strategy
    --------
    1. Set ``GDAL_ERROR_FILE`` env var so GDAL writes errors to a temp file
       instead of stderr (GDAL ≥ 3.1 supports this).
    2. Use ``rasterio.Env`` with ``CPL_LOG`` + ``CPL_LOG_ERRORS=ON`` to
       redirect GDAL logs to a temp file path.
    3. On context exit, read the temp file and emit one WARNING record per
       non-empty line.

    This approach avoids fragile fd-dup tricks that break under pytest's
    capsys or when rasterio/GDAL have already opened stderr.

    Parameters
    ----------
    stage :
        ``stage`` field in emitted log records.
    min_line_length :
        Ignore captured lines shorter than this (e.g. stray newlines).

    Yields
    ------
    Nothing.  Side-effect: GDAL messages appear in the structured log.
    """
    log_ = logging.getLogger(f"sdb_optim.{stage}")

    # Create a temp file for GDAL log output
    fd, gdal_log_path = tempfile.mkstemp(suffix=".gdal_log.txt", prefix="sdb_optim_")
    os.close(fd)

    old_cpl_log = os.environ.get("CPL_LOG")
    old_cpl_log_errors = os.environ.get("CPL_LOG_ERRORS")
    old_gdal_error_file = os.environ.get("GDAL_ERROR_FILE")

    os.environ["CPL_LOG"] = gdal_log_path
    os.environ["CPL_LOG_ERRORS"] = "ON"
    os.environ["GDAL_ERROR_FILE"] = gdal_log_path

    try:
        # Try to use rasterio.Env for stricter GDAL config scoping
        try:
            import rasterio
            ctx = rasterio.Env(
                CPL_LOG=gdal_log_path,
                CPL_LOG_ERRORS="ON",
                GDAL_ERROR_FILE=gdal_log_path,
            )
            ctx.__enter__()
            _rasterio_env = ctx
        except Exception:
            _rasterio_env = None

        yield

    finally:
        # Exit rasterio env if used
        if _rasterio_env is not None:
            try:
                _rasterio_env.__exit__(None, None, None)
            except Exception:
                pass

        # Restore env vars
        for k, v in [
            ("CPL_LOG", old_cpl_log),
            ("CPL_LOG_ERRORS", old_cpl_log_errors),
            ("GDAL_ERROR_FILE", old_gdal_error_file),
        ]:
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        # Read and emit captured lines
        try:
            with open(gdal_log_path, "r", errors="replace") as fh:
                lines = fh.readlines()
            for raw_line in lines:
                line = raw_line.rstrip("\n").strip()
                if len(line) >= min_line_length:
                    log_.warning(
                        f"GDAL/rasterio: {line}",
                        extra={"fields": {"source": "gdal_clib", "raw": line}},
                    )
        except OSError:
            pass
        finally:
            try:
                os.unlink(gdal_log_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 5. pipeline_health_summary — aggregate a run's .jsonl into a health digest
# ---------------------------------------------------------------------------

def pipeline_health_summary(jsonl_path: Union[str, Path]) -> Dict[str, Any]:
    """Aggregate a run's ``.jsonl`` log into a compact health digest.

    Reads every line of *jsonl_path* and produces::

        {
            "n_records":    int,
            "n_errors":     int,
            "n_warnings":   int,
            "n_critical":   int,
            "error_msgs":   [str, ...],      // first 20 ERROR/CRITICAL messages
            "warning_msgs": [str, ...],      // first 20 WARNING messages
            "stages_timed": {                // stage → {count, total_ms, max_ms}
                "cog_chunker.apply": {"count": 3, "total_ms": 1240.5, "max_ms": 520.1},
                ...
            },
            "total_duration_ms": float,      // sum of all timed events
            "parse_errors":    int,          // malformed JSONL lines
            "log_path":        str,
        }

    The result is suitable for writing to a CI artifact or posting to a
    monitoring endpoint (``json.dumps(summary)``).

    Parameters
    ----------
    jsonl_path :
        Path to a ``.jsonl`` file produced by this module's ``_JsonlHandler``.

    Returns
    -------
    dict
    """
    jsonl_path = Path(jsonl_path)

    summary: Dict[str, Any] = {
        "n_records": 0,
        "n_errors": 0,
        "n_warnings": 0,
        "n_critical": 0,
        "error_msgs": [],
        "warning_msgs": [],
        "stages_timed": {},
        "total_duration_ms": 0.0,
        "parse_errors": 0,
        "log_path": str(jsonl_path),
    }

    if not jsonl_path.exists():
        summary["parse_errors"] = -1  # file not found sentinel
        return summary

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                rec = json.loads(raw_line)
            except json.JSONDecodeError:
                summary["parse_errors"] += 1
                continue

            summary["n_records"] += 1
            level = rec.get("level", "INFO").upper()

            if level in ("ERROR",):
                summary["n_errors"] += 1
                if len(summary["error_msgs"]) < 20:
                    summary["error_msgs"].append(rec.get("msg", ""))
            elif level == "CRITICAL":
                summary["n_critical"] += 1
                if len(summary["error_msgs"]) < 20:
                    summary["error_msgs"].append(rec.get("msg", ""))
            elif level == "WARNING":
                summary["n_warnings"] += 1
                if len(summary["warning_msgs"]) < 20:
                    summary["warning_msgs"].append(rec.get("msg", ""))

            dur = rec.get("duration_ms")
            if dur is not None:
                stage = rec.get("stage", "unknown")
                entry = summary["stages_timed"].setdefault(
                    stage, {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
                )
                entry["count"] += 1
                entry["total_ms"] = round(entry["total_ms"] + dur, 2)
                entry["max_ms"] = round(max(entry["max_ms"], dur), 2)
                summary["total_duration_ms"] = round(
                    summary["total_duration_ms"] + dur, 2
                )

    return summary
