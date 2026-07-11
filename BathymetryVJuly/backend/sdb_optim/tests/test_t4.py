"""
T4 test battery — HPC / IO / Structured Logging subsystem.

Tests
-----
T4a  Import-clean: all four T4 modules importable without error.
T4b  cog_chunker:
     b1 — iter_windows yields complete coverage (union == full extent, no gaps,
          no duplicates) on a 120×80 px raster with block=32, overlap=0.
     b2 — iter_windows with overlap=8: all pixels covered (overlapping ok).
     b3 — chunked_apply identity round-trip: output pixel values == input
          within nodata mask.
T4c  parallel_infer:
     c1 — worker_pool with n_workers=2 (fork, CPU) returns correct results for
          a trivial doubling callable over 6 tasks.
     c2 — worker_pool propagates a worker exception (does NOT hang).
T4d  structured_logging:
     d1 — setup + emit → every line of JSONL is json.loads-able.
     d2 — capture_geospatial_stderr: emitted message appears in JSONL.
     d3 — pipeline_health_summary counts errors/warnings correctly.
T4e  Shell wrapper:
     e1 — `bash -n run_sdb_optim.sh` passes syntax check.
     e2 — `run_sdb_optim.sh --help` exits 0.

Run::

    /path/to/.venv/bin/python3 -m pytest backend/sdb_optim/tests/test_t4.py -v

or directly::

    /path/to/.venv/bin/python3 backend/sdb_optim/tests/test_t4.py
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Resolve project root so imports work from any cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # Bathymetry_VMarch
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# rasterio guard — some tests are skipped if rasterio is unavailable
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.transform import from_bounds
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False
    rasterio = None  # type: ignore

RASTERIO_SKIP = pytest.mark.skipif(
    not _RASTERIO_OK,
    reason="rasterio not installed — COG tests skipped (flagged honestly)"
)


# ============================================================================
# T4a — Import-clean
# ============================================================================

class TestT4aImports:
    def test_cog_chunker_importable(self):
        from backend.sdb_optim import cog_chunker
        assert hasattr(cog_chunker, "iter_windows")
        assert hasattr(cog_chunker, "read_window")
        assert hasattr(cog_chunker, "chunked_apply")

    def test_parallel_infer_importable(self):
        from backend.sdb_optim import parallel_infer
        assert hasattr(parallel_infer, "worker_pool")
        assert hasattr(parallel_infer, "resolve_n_workers")

    def test_structured_logging_importable(self):
        from backend.sdb_optim import structured_logging
        assert hasattr(structured_logging, "setup")
        assert hasattr(structured_logging, "stage_timer")
        assert hasattr(structured_logging, "capture_geospatial_stderr")
        assert hasattr(structured_logging, "pipeline_health_summary")

    def test_subsystem_interface_importable(self):
        from backend.sdb_optim import subsystem_interface
        assert hasattr(subsystem_interface, "run_inference_job")
        assert hasattr(subsystem_interface, "InferenceConfig")
        assert hasattr(subsystem_interface, "JobResult")


# ============================================================================
# T4b — cog_chunker
# ============================================================================

class TestT4bCogChunker:
    """Tests for iter_windows and chunked_apply."""

    # ---- b1: non-overlapping coverage -------------------------------------

    def test_iter_windows_complete_coverage(self):
        """Union of all non-overlapping windows == full image; no pixel missed/doubled."""
        from backend.sdb_optim.cog_chunker import iter_windows

        W, H = 120, 80
        block = 32

        covered = np.zeros((H, W), dtype=np.int32)
        for col_off, row_off, win_w, win_h in iter_windows(W, H, block=block, overlap=0):
            assert col_off >= 0 and row_off >= 0
            assert col_off + win_w <= W
            assert row_off + win_h <= H
            covered[row_off:row_off + win_h, col_off:col_off + win_w] += 1

        assert covered.min() == 1, f"Some pixels not covered (min={covered.min()})"
        assert covered.max() == 1, f"Some pixels covered twice (max={covered.max()})"
        assert covered.sum() == W * H

    # ---- b2: overlapping coverage -----------------------------------------

    def test_iter_windows_overlap_covers_all(self):
        """With overlap, every pixel is covered at least once."""
        from backend.sdb_optim.cog_chunker import iter_windows

        W, H = 100, 60
        block, overlap = 32, 8

        covered = np.zeros((H, W), dtype=np.int32)
        for col_off, row_off, win_w, win_h in iter_windows(W, H, block=block, overlap=overlap):
            covered[row_off:row_off + win_h, col_off:col_off + win_w] += 1

        assert covered.min() >= 1, "Some pixels not covered with overlap"

    # ---- b3: chunked_apply identity round-trip ----------------------------

    @RASTERIO_SKIP
    def test_chunked_apply_identity_roundtrip(self, tmp_path):
        """chunked_apply with identity fn: output pixel values match input."""
        from backend.sdb_optim.cog_chunker import chunked_apply

        rng = np.random.default_rng(42)
        H, W = 64, 96
        data = rng.random((H, W)).astype(np.float32)

        # Write synthetic input GeoTIFF
        transform = from_bounds(0, 0, W, H, W, H)
        src_path = tmp_path / "src.tif"
        with rasterio.open(
            src_path, "w", driver="GTiff",
            height=H, width=W, count=1, dtype="float32",
            crs="EPSG:4326", transform=transform, nodata=-9999.0,
        ) as ds:
            ds.write(data, 1)

        out_path = tmp_path / "out.tif"

        def identity(tile, window, meta):
            return tile

        result_path = chunked_apply(
            src_path=src_path,
            fn=identity,
            out_path=out_path,
            block=32,
            overlap=0,
            nodata=-9999.0,
        )
        assert result_path.exists()

        with rasterio.open(result_path) as ds:
            out_data = ds.read(1)

        # Values should match within float32 precision
        np.testing.assert_allclose(
            out_data, data, rtol=1e-5, atol=1e-6,
            err_msg="chunked_apply identity round-trip: output != input"
        )

    # ---- additional: bad params raise ----------------------------------------

    def test_iter_windows_bad_block_raises(self):
        from backend.sdb_optim.cog_chunker import iter_windows
        with pytest.raises(ValueError):
            list(iter_windows(100, 100, block=0))

    def test_iter_windows_overlap_ge_block_raises(self):
        from backend.sdb_optim.cog_chunker import iter_windows
        with pytest.raises(ValueError):
            list(iter_windows(100, 100, block=32, overlap=32))


# ============================================================================
# T4c — parallel_infer
# ============================================================================

# Module-level callable (must be picklable)
def _double(arr: np.ndarray, tag: int) -> np.ndarray:
    return arr * 2.0 + tag


def _raise_on_tag(arr: np.ndarray, tag: int) -> np.ndarray:
    if tag == 3:
        raise ValueError(f"Deliberate worker error for tag={tag}")
    return arr


class TestT4cParallelInfer:
    """Correctness + exception propagation tests for worker_pool."""

    def test_worker_pool_correct_results(self):
        """n_workers=2: doubling callable returns arr*2+tag for all 6 tasks."""
        from backend.sdb_optim.parallel_infer import worker_pool

        rng = np.random.default_rng(7)
        n_tasks = 6
        inputs = [rng.random((8, 8)).astype(np.float32) for _ in range(n_tasks)]

        tasks = [((inp, i), {}) for i, inp in enumerate(inputs)]

        results: dict = {}
        for idx, arr in worker_pool(
            _double, tasks, n_workers=2, max_in_flight=4
        ):
            results[idx] = arr

        assert len(results) == n_tasks
        for i, inp in enumerate(inputs):
            expected = inp * 2.0 + i
            np.testing.assert_allclose(
                results[i], expected, rtol=1e-5,
                err_msg=f"Task {i}: result mismatch"
            )

    def test_worker_pool_exception_propagates(self):
        """Worker exception is re-raised in caller, not silently swallowed."""
        from backend.sdb_optim.parallel_infer import worker_pool

        rng = np.random.default_rng(99)
        tasks = [((rng.random((4, 4)).astype(np.float32), i), {})
                 for i in range(6)]

        with pytest.raises(RuntimeError, match="Worker task"):
            list(worker_pool(_raise_on_tag, tasks, n_workers=2, max_in_flight=8))

    def test_resolve_n_workers_auto(self):
        from backend.sdb_optim.parallel_infer import resolve_n_workers
        import multiprocessing
        n = resolve_n_workers("auto", use_gpu=False)
        assert n >= 1
        assert n <= multiprocessing.cpu_count()

    def test_resolve_n_workers_gpu_forces_1(self):
        from backend.sdb_optim.parallel_infer import resolve_n_workers
        assert resolve_n_workers("auto", use_gpu=True) == 1
        assert resolve_n_workers(16, use_gpu=True) == 1

    def test_worker_pool_single_worker_in_process(self):
        """n_workers=1 uses in-process path: no serialization issues."""
        from backend.sdb_optim.parallel_infer import worker_pool

        rng = np.random.default_rng(3)
        tasks = [((rng.random((4, 4)).astype(np.float32), i), {}) for i in range(3)]

        results = dict(worker_pool(_double, tasks, n_workers=1))
        assert len(results) == 3


# ============================================================================
# T4d — structured_logging
# ============================================================================

class TestT4dStructuredLogging:
    """Valid JSONL output and GDAL stderr capture."""

    def test_jsonl_all_lines_parseable(self, tmp_path):
        """Every emitted log record is a valid JSON object."""
        from backend.sdb_optim.structured_logging import setup, pipeline_health_summary

        log_path = tmp_path / "test_run.jsonl"
        setup(log_path, also_stderr=False)

        logger = logging.getLogger("t4d_test")
        logger.info("first record", extra={"fields": {"x": 1}})
        logger.warning("second record")
        logger.error("third record")

        # Flush by closing (via teardown) — just read the file
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 3, f"Expected >= 3 lines, got {len(lines)}"

        for i, line in enumerate(lines):
            obj = json.loads(line)
            assert "ts" in obj,    f"Line {i} missing 'ts'"
            assert "level" in obj, f"Line {i} missing 'level'"
            assert "msg" in obj,   f"Line {i} missing 'msg'"

    def test_jsonl_fields_preserved(self, tmp_path):
        """Structured 'fields' dict is round-tripped in JSONL."""
        from backend.sdb_optim.structured_logging import setup

        log_path = tmp_path / "fields_test.jsonl"
        setup(log_path, also_stderr=False)

        logger = logging.getLogger("t4d_fields")
        logger.info("with fields", extra={"fields": {"rmse": 0.95, "site": "dhanna"}})

        lines = log_path.read_text().strip().splitlines()
        found = False
        for line in lines:
            obj = json.loads(line)
            if obj.get("fields") and obj["fields"].get("site") == "dhanna":
                assert obj["fields"]["rmse"] == pytest.approx(0.95)
                found = True
                break
        assert found, "fields not found in any JSONL record"

    def test_capture_geospatial_stderr_emits_warning(self, tmp_path):
        """capture_geospatial_stderr: an injected GDAL-like message appears in JSONL."""
        from backend.sdb_optim.structured_logging import (
            setup, capture_geospatial_stderr
        )

        log_path = tmp_path / "gdal_cap.jsonl"
        setup(log_path, also_stderr=False)

        # Simulate GDAL writing to the CPL_LOG file by writing directly to the
        # file path set by the context manager after it sets env vars.
        # We do this by using capture_geospatial_stderr and then manually
        # writing to the GDAL log path *inside* the context.
        import os as _os

        # Use the context manager and simulate a GDAL message
        with capture_geospatial_stderr(stage="gdal_test"):
            # Write a test message to CPL_LOG path if it's set
            gdal_log_path = _os.environ.get("CPL_LOG")
            if gdal_log_path:
                try:
                    with open(gdal_log_path, "a") as f:
                        f.write("TIFFReadDirectory: test GDAL warning injected\n")
                except OSError:
                    pass

        # Check the warning was captured
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        found_gdal = any(
            "TIFFReadDirectory" in json.loads(line).get("msg", "")
            for line in lines
            if line.strip()
        )
        # If CPL_LOG was set and writable, this should pass.
        # We assert rather than skip to flag if the mechanism is broken.
        assert found_gdal, (
            "GDAL warning not found in JSONL. "
            "capture_geospatial_stderr may not be routing CPL_LOG correctly. "
            "Lines: " + str(lines[:5])
        )

    def test_pipeline_health_summary_counts(self, tmp_path):
        """pipeline_health_summary correctly counts errors and warnings."""
        from backend.sdb_optim.structured_logging import (
            setup, pipeline_health_summary
        )

        log_path = tmp_path / "health_test.jsonl"
        setup(log_path, also_stderr=False)

        logger = logging.getLogger("t4d_health")
        for _ in range(3):
            logger.warning("a warning")
        for _ in range(2):
            logger.error("an error")
        logger.info("all good")

        health = pipeline_health_summary(log_path)

        assert health["n_warnings"] >= 3, f"Expected >=3 warnings, got {health['n_warnings']}"
        assert health["n_errors"] >= 2,   f"Expected >=2 errors, got {health['n_errors']}"
        assert health["parse_errors"] == 0

    def test_stage_timer_writes_duration_ms(self, tmp_path):
        """stage_timer emits a record with duration_ms populated."""
        from backend.sdb_optim.structured_logging import setup, stage_timer
        import time as _time

        log_path = tmp_path / "timer_test.jsonl"
        setup(log_path, also_stderr=False)

        with stage_timer("test_stage", x=42):
            _time.sleep(0.01)   # 10 ms

        lines = log_path.read_text().strip().splitlines()
        found = False
        for line in lines:
            obj = json.loads(line)
            dur = obj.get("duration_ms")
            if dur is not None and dur >= 1.0:
                found = True
                break
        assert found, "No record with duration_ms >= 1 ms found"


# ============================================================================
# T4e — Shell wrapper
# ============================================================================

SCRIPT_PATH = ROOT / "run_sdb_optim.sh"

class TestT4eShellWrapper:
    def test_bash_syntax_check(self):
        """bash -n run_sdb_optim.sh passes without error."""
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT_PATH)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"bash -n syntax check failed:\n{result.stderr}"
        )

    def test_help_exits_zero(self):
        """run_sdb_optim.sh --help exits 0."""
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), "--help"],
            capture_output=True, text=True,
            cwd=str(ROOT)
        )
        assert result.returncode == 0, (
            f"--help returned non-zero: {result.returncode}\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
        assert len(result.stdout) > 50, "Help output too short"


# ============================================================================
# Runner for direct execution (non-pytest)
# ============================================================================

def _run_all_tests():
    """Simple test runner for direct ``python test_t4.py`` execution."""
    import traceback

    test_classes = [
        TestT4aImports,
        TestT4bCogChunker,
        TestT4cParallelInfer,
        TestT4dStructuredLogging,
        TestT4eShellWrapper,
    ]

    passed = 0
    failed = 0
    skipped = 0
    results: List[Tuple[str, str, str]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for cls in test_classes:
            instance = cls()
            # Collect test methods
            methods = sorted(
                name for name in dir(cls)
                if name.startswith("test_")
            )
            for mname in methods:
                method = getattr(instance, mname)
                # Check skip mark
                skip_mark = getattr(method, "pytestmark", None)
                if skip_mark is None:
                    skip_mark = getattr(cls, "pytestmark", None)

                fqn = f"{cls.__name__}.{mname}"
                # Inject tmp_path for tests that need it
                import inspect
                sig = inspect.signature(method)
                kwargs = {}
                if "tmp_path" in sig.parameters:
                    kwargs["tmp_path"] = tmp_path / mname

                try:
                    method(**kwargs)
                    print(f"  PASS  {fqn}")
                    passed += 1
                    results.append((fqn, "PASS", ""))
                except pytest.skip.Exception as e:
                    print(f"  SKIP  {fqn}  ({e})")
                    skipped += 1
                    results.append((fqn, "SKIP", str(e)))
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"  FAIL  {fqn}  → {e}")
                    failed += 1
                    results.append((fqn, "FAIL", str(e)))

    print(f"\n{'='*60}")
    print(f"T4 test battery: {passed} PASS / {failed} FAIL / {skipped} SKIP")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all_tests()
    sys.exit(0 if ok else 1)
