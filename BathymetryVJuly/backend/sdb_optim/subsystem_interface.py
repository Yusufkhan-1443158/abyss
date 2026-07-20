"""
T4 subsystem interface — ties chunker + parallel_infer + structured_logging together.

INTEGRATION:
  - Primary Python API consumed by `run_sdb_optim.sh` (via argparse CLI at __main__).
  - Wraps `backend/icesat2_bathy.process_granule` (ICESat-2 photon extraction)
    and the CNN predict loop in `backend/sdb_cnn_baseline.py` / `backend/unet_sdb.py`
    by providing the IO scaffolding (windowed reading, parallel dispatch, structured logs).
  - The orchestrator wires this into the live pipeline after all tracks land.

Usage (Python API)
------------------
    from backend.sdb_optim.subsystem_interface import InferenceConfig, run_inference_job

    cfg = InferenceConfig(
        src_path   = "/path/to/input.tif",
        out_path   = "/path/to/output.tif",
        log_path   = "/path/to/run.jsonl",
        block      = 512,
        overlap    = 0,
        n_workers  = "auto",
        use_gpu    = False,
    )
    result = run_inference_job(cfg)
    print(result)  # JobResult dataclass

Usage (CLI)
-----------
    python -m backend.sdb_optim.subsystem_interface \\
        --src /path/to/input.tif \\
        --out /path/to/output.tif \\
        --log /path/to/run.jsonl \\
        --block 512 \\
        --workers auto
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Lazy / guarded imports so partial builds never break ``import backend.sdb_optim``
# ---------------------------------------------------------------------------
try:
    from .cog_chunker import chunked_apply, iter_windows
    _CHUNKER_OK = True
except ImportError as _e:
    _CHUNKER_OK = False
    _CHUNKER_ERR = str(_e)

try:
    from .parallel_infer import worker_pool, resolve_n_workers
    _INFER_OK = True
except ImportError as _e:
    _INFER_OK = False
    _INFER_ERR = str(_e)

try:
    from .structured_logging import (
        setup as setup_logging,
        stage_timer,
        capture_geospatial_stderr,
        pipeline_health_summary,
    )
    _LOG_OK = True
except ImportError as _e:
    _LOG_OK = False
    _LOG_ERR = str(_e)

log = logging.getLogger("sdb_optim.subsystem_interface")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class InferenceConfig:
    """Configuration for a single T4 inference job.

    Attributes
    ----------
    src_path :
        Input raster (GeoTIFF / COG / VRT).  Required.
    out_path :
        Output COG path.  If None, a sidecar ``<src_path>.out.tif`` is created.
    log_path :
        Structured JSONL log path.  If None, defaults to ``<out_path>.jsonl``.
    fn :
        Inference callable ``fn(tile, window, meta) -> np.ndarray``.
        Defaults to the identity function (passthrough — for testing).
    block :
        Tile size in pixels (default 512).
    overlap :
        Overlap pixels per edge (default 0).
    n_workers :
        Worker count or ``"auto"``.
    use_gpu :
        Single-process GPU mode.
    compress :
        Output compression (``"deflate"`` or ``"lzw"``).
    dry_run :
        If True, log the plan but do not execute (useful for CI checks).
    extra :
        Free-form extra fields logged in the run header record.
    """
    src_path: Union[str, Path]
    out_path: Optional[Union[str, Path]] = None
    log_path: Optional[Union[str, Path]] = None
    fn: Optional[Callable] = None
    block: int = 512
    overlap: int = 0
    n_workers: Union[int, str] = "auto"
    use_gpu: bool = False
    compress: str = "deflate"
    dry_run: bool = False
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.src_path = Path(self.src_path)
        if self.out_path is None:
            self.out_path = self.src_path.with_suffix(".out.tif")
        self.out_path = Path(self.out_path)
        if self.log_path is None:
            self.log_path = self.out_path.with_suffix(".jsonl")
        self.log_path = Path(self.log_path)
        if self.fn is None:
            self.fn = _identity_fn


# ---------------------------------------------------------------------------
# Default inference callable (identity — for testing / passthrough)
# ---------------------------------------------------------------------------

def _identity_fn(tile, window, meta):
    """Passthrough callable: returns the input tile unchanged."""
    import numpy as np
    arr = tile if hasattr(tile, "shape") else tile
    if arr.ndim == 2:
        arr = arr[..., None]  # ensure 3-D for consistency
    return arr.squeeze()  # return same ndim as input


# ---------------------------------------------------------------------------
# JobResult dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class JobResult:
    """Summary of a completed inference job."""
    ok: bool
    out_path: Optional[Path]
    log_path: Optional[Path]
    health: Dict[str, Any]
    duration_s: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["out_path"] = str(self.out_path) if self.out_path else None
        d["log_path"] = str(self.log_path) if self.log_path else None
        return d


# ---------------------------------------------------------------------------
# Main API: run_inference_job
# ---------------------------------------------------------------------------

def run_inference_job(config: InferenceConfig) -> JobResult:
    """Execute a windowed COG inference job with structured logging.

    Steps
    -----
    1. Set up structured JSONL logging.
    2. Log job header (config summary).
    3. Open *config.src_path*, iterate windows via :func:`cog_chunker.iter_windows`.
    4. Stream *config.fn* over tiles via :func:`cog_chunker.chunked_apply`
       (single-process) or optionally via :func:`parallel_infer.worker_pool`
       for multi-process dispatch.
    5. Capture any GDAL/rasterio C-library messages via
       :func:`structured_logging.capture_geospatial_stderr`.
    6. Emit a final health record; return :class:`JobResult`.

    Parameters
    ----------
    config : InferenceConfig

    Returns
    -------
    JobResult
    """
    t_start = time.perf_counter()

    # -- 1. Structured logging -----------------------------------------------
    if _LOG_OK:
        setup_logging(config.log_path, also_stderr=True)
    else:
        logging.basicConfig(level=logging.INFO)
        log.warning(f"structured_logging unavailable: {_LOG_ERR}")

    log.info(
        "T4 inference job starting",
        extra={"fields": {
            "src": str(config.src_path),
            "out": str(config.out_path),
            "block": config.block,
            "overlap": config.overlap,
            "n_workers": config.n_workers,
            "use_gpu": config.use_gpu,
            "dry_run": config.dry_run,
            **config.extra,
        }},
    )

    # -- 2. Guard checks -------------------------------------------------------
    if not config.src_path.exists():
        msg = f"src_path not found: {config.src_path}"
        log.error(msg)
        return JobResult(
            ok=False,
            out_path=None,
            log_path=config.log_path,
            health={},
            duration_s=time.perf_counter() - t_start,
            error=msg,
        )

    if not _CHUNKER_OK:
        msg = f"cog_chunker import failed: {_CHUNKER_ERR}"
        log.error(msg)
        return JobResult(
            ok=False,
            out_path=None,
            log_path=config.log_path,
            health={},
            duration_s=time.perf_counter() - t_start,
            error=msg,
        )

    if config.dry_run:
        log.info("dry_run=True — skipping actual processing")
        # Count windows and report plan only
        try:
            import rasterio
            with rasterio.open(str(config.src_path)) as ds:
                windows = list(iter_windows(ds.width, ds.height,
                                            block=config.block,
                                            overlap=config.overlap))
            log.info(
                f"dry_run plan: {len(windows)} windows over "
                f"{ds.width}×{ds.height} px",
                extra={"fields": {"n_windows": len(windows)}},
            )
        except Exception as exc:
            log.warning(f"dry_run window count failed: {exc}")
        duration_s = time.perf_counter() - t_start
        health = pipeline_health_summary(config.log_path) if _LOG_OK else {}
        return JobResult(ok=True, out_path=None, log_path=config.log_path,
                         health=health, duration_s=duration_s)

    # -- 3. Execute with GDAL stderr capture ----------------------------------
    error_msg: Optional[str] = None
    out_path: Optional[Path] = None

    _capture_ctx = capture_geospatial_stderr() if _LOG_OK else _null_ctx()

    try:
        with _capture_ctx:
            with stage_timer("t4.run_inference_job",
                             src=str(config.src_path)) if _LOG_OK else _null_ctx():
                out_path = chunked_apply(
                    src_path=config.src_path,
                    fn=config.fn,
                    out_path=config.out_path,
                    block=config.block,
                    overlap=config.overlap,
                    compress=config.compress,
                )
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        log.error(f"inference job failed: {error_msg}", exc_info=True)

    # -- 4. Health summary ----------------------------------------------------
    health: Dict[str, Any] = {}
    if _LOG_OK:
        try:
            health = pipeline_health_summary(config.log_path)
        except Exception as e:
            log.warning(f"health summary failed: {e}")

    duration_s = time.perf_counter() - t_start
    ok = error_msg is None and out_path is not None

    log.info(
        f"T4 job {'OK' if ok else 'FAILED'} in {duration_s:.2f}s",
        extra={"fields": {
            "ok": ok,
            "n_errors": health.get("n_errors", "?"),
            "n_warnings": health.get("n_warnings", "?"),
        }},
    )

    return JobResult(
        ok=ok,
        out_path=out_path,
        log_path=config.log_path,
        health=health,
        duration_s=duration_s,
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Null context manager (used when structured_logging is unavailable)
# ---------------------------------------------------------------------------

from contextlib import contextmanager as _cm

@_cm
def _null_ctx():
    yield


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sdb_optim",
        description=(
            "T4 HPC/IO subsystem: windowed COG inference with structured logging.\n"
            "Runs fn (identity by default) over all tiles of --src and writes a COG to --out."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src",      required=True,  help="Input raster path (GeoTIFF / COG)")
    p.add_argument("--out",      default=None,   help="Output COG path [<src>.out.tif]")
    p.add_argument("--log",      default=None,   help="JSONL log path [<out>.jsonl]")
    p.add_argument("--block",    type=int, default=512, help="Tile block size in px [512]")
    p.add_argument("--overlap",  type=int, default=0,   help="Overlap pixels [0]")
    p.add_argument("--workers",  default="auto", help="Worker count or 'auto' [auto]")
    p.add_argument("--gpu",      action="store_true",   help="Single-process GPU mode")
    p.add_argument("--compress", default="deflate",     help="Output compression [deflate]")
    p.add_argument("--dry-run",  action="store_true",   help="Plan only, do not process")
    p.add_argument("--health",   default=None,
                   help="Write JSON health summary to this path after the run")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Returns exit code (0=success, 1=failure)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve workers (int or "auto")
    try:
        n_workers: Union[int, str] = int(args.workers)
    except ValueError:
        n_workers = args.workers

    cfg = InferenceConfig(
        src_path=args.src,
        out_path=args.out,
        log_path=args.log,
        block=args.block,
        overlap=args.overlap,
        n_workers=n_workers,
        use_gpu=args.gpu,
        compress=args.compress,
        dry_run=args.dry_run,
    )

    result = run_inference_job(cfg)

    # Optionally write health JSON
    if args.health and result.health:
        health_path = Path(args.health)
        health_path.parent.mkdir(parents=True, exist_ok=True)
        health_path.write_text(
            json.dumps(result.health, indent=2, ensure_ascii=False)
        )
        print(f"Health summary → {health_path}", file=sys.stderr)

    if result.ok:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    else:
        print(f"ERROR: {result.error}", file=sys.stderr)
        print(json.dumps(result.to_dict(), indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
