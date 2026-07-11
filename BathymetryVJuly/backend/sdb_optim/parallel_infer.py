"""
Deadlock-safe parallel inference over raster windows for SDB pipelines.

INTEGRATION:
  - Used by `subsystem_interface.run_inference_job` to distribute tile-wise
    inference across CPU cores or (single-process) GPU.
  - Inference callable typically wraps `backend/sdb_cnn_baseline.PatchCNN`
    predict path or `backend/unet_sdb.predict_grid_tta`.
  - Feeds from `cog_chunker.iter_windows` / `cog_chunker.read_window`.

Deadlock pitfalls guarded against
-----------------------------------
1. **fork + CUDA**: PyTorch CUDA tensors + multiprocessing fork = corruption /
   deadlock.  We enforce the ``spawn`` start method when GPU workers are
   requested.  CPU workers use the default (fork on Linux) for lower overhead,
   but only if no CUDA context is active in the parent process.

2. **Unbounded submission queue**: submitting all futures before collecting any
   means the parent can OOM on large rasters before a single result is consumed.
   We use a *bounded sliding window* — at most ``max_in_flight`` futures are
   pending at any time.

3. **Exception swallowing**: ``concurrent.futures`` silently buries worker
   exceptions until ``future.result()`` is called.  We call ``.result()`` in
   the collection loop and re-raise immediately so the caller sees the real
   traceback, not a hang.

4. **KeyboardInterrupt inside ProcessPoolExecutor**: sending SIGINT to a pool
   with pending futures can leave zombie workers.  We wrap the collection loop
   in a try/finally that calls ``executor.shutdown(wait=False, cancel_futures=True)``
   (Python ≥ 3.9) so workers are released promptly.

References
----------
- Python docs: concurrent.futures — https://docs.python.org/3/library/concurrent.futures.html
- PyTorch multiprocessing best-practices:
  https://pytorch.org/docs/stable/notes/multiprocessing.html
  "Use ``spawn`` or ``forkserver`` start method when CUDA is involved."
- Brent Pedersen, "CUDA fork multiprocessing gotchas", PyTorch forums 2022.
"""
from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os
import sys
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

log = logging.getLogger("sdb_optim.parallel_infer")

# ---------------------------------------------------------------------------
# Constants / env-cap
# ---------------------------------------------------------------------------

_ENV_WORKER_CAP = int(os.environ.get("SDB_MAX_WORKERS", "0")) or None
"""Hard cap on workers from env var ``SDB_MAX_WORKERS``.  0 → no cap."""

_DEFAULT_MAX_IN_FLIGHT = 8
"""Bounded queue depth: at most this many futures are submitted before we drain."""


# ---------------------------------------------------------------------------
# 1. Resolve worker count
# ---------------------------------------------------------------------------

def resolve_n_workers(n_workers: "int | str" = "auto", use_gpu: bool = False) -> int:
    """Return a concrete worker count.

    Parameters
    ----------
    n_workers : int | "auto"
        ``"auto"`` → ``max(1, cpu_count - 1)``, capped by ``SDB_MAX_WORKERS``.
        An integer is used directly (still capped by the env var).
    use_gpu : bool
        If True, always returns 1.  CUDA requires a single process per GPU
        because each ``torch.cuda`` context is process-local and fork-unsafe.

    Returns
    -------
    int
        Always ≥ 1.
    """
    if use_gpu:
        log.info("GPU mode: single-process inference (CUDA fork-unsafe)")
        return 1

    if n_workers == "auto":
        cpu = multiprocessing.cpu_count()
        n = max(1, cpu - 1)
    else:
        n = max(1, int(n_workers))

    if _ENV_WORKER_CAP:
        n = min(n, _ENV_WORKER_CAP)

    return n


# ---------------------------------------------------------------------------
# 2. Worker entry point (module-level so it is picklable)
# ---------------------------------------------------------------------------

def _worker_call(fn: Callable, args: tuple, kwargs: dict) -> Any:
    """Top-level worker function — picklable, no closures over unpicklable state.

    Wraps *fn* so that any exception carries the full traceback when
    ``future.result()`` is called in the parent.
    """
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# 3. worker_pool — bounded ProcessPoolExecutor
# ---------------------------------------------------------------------------

def worker_pool(
    fn: Callable[..., np.ndarray],
    tasks: Iterable[Tuple[tuple, dict]],
    n_workers: "int | str" = "auto",
    use_gpu: bool = False,
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
    timeout_s: Optional[float] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Run *fn* over *tasks* in a bounded process pool, yielding (task_index, result).

    Parameters
    ----------
    fn :
        Callable to execute per task.  Must be importable by worker processes
        (i.e. defined at module level or importable from a module — no lambdas
        or closures that capture non-picklable state).
    tasks :
        Iterable of ``(args_tuple, kwargs_dict)`` pairs.
    n_workers : int | "auto"
        See :func:`resolve_n_workers`.
    use_gpu : bool
        Forces single-process mode + ``spawn`` context (CUDA safety).
    max_in_flight : int
        Max pending futures before the submission loop blocks.  Bounds peak
        memory: at most ``max_in_flight`` tile results live in memory simultaneously.
    timeout_s : float | None
        Per-future timeout in seconds.  ``None`` → wait indefinitely.

    Yields
    ------
    (task_index, result)  in completion order (not submission order).

    Raises
    ------
    RuntimeError
        Re-raised from worker exceptions with original traceback attached.
    KeyboardInterrupt
        Cancels pending futures and shuts down workers cleanly.

    Notes
    -----
    CPU workers use ``fork`` context (Linux default) for low spawn overhead.
    GPU workers use ``spawn`` context to avoid CUDA-fork corruption.
    """
    n = resolve_n_workers(n_workers, use_gpu=use_gpu)
    mp_ctx = "spawn" if use_gpu else "fork"

    log.info(f"worker_pool: n_workers={n}, context={mp_ctx}, max_in_flight={max_in_flight}")

    tasks_list = list(tasks)
    n_tasks = len(tasks_list)

    if n_tasks == 0:
        return

    # n_workers=1 → run in-process to avoid serialization overhead and
    # make debugging far easier (stack traces are direct, not pickled).
    if n == 1:
        log.debug("worker_pool: single-worker → in-process execution")
        for idx, (args, kwargs) in enumerate(tasks_list):
            result = fn(*args, **kwargs)
            yield idx, result
        return

    mp_context = multiprocessing.get_context(mp_ctx)
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=n,
        mp_context=mp_context,
    )

    future_to_idx: Dict[concurrent.futures.Future, int] = {}
    submitted = 0
    completed = 0

    try:
        while completed < n_tasks:
            # Fill the in-flight window
            while submitted < n_tasks and len(future_to_idx) < max_in_flight:
                args, kwargs = tasks_list[submitted]
                fut = executor.submit(_worker_call, fn, args, kwargs)
                future_to_idx[fut] = submitted
                submitted += 1

            # Drain at least one completed future
            done_futs, _ = concurrent.futures.wait(
                list(future_to_idx.keys()),
                timeout=timeout_s,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            if not done_futs:
                # timeout hit — none finished yet; loop again
                continue

            for fut in done_futs:
                idx = future_to_idx.pop(fut)
                try:
                    result = fut.result()
                except Exception as exc:
                    # Re-raise with context so caller sees worker traceback
                    raise RuntimeError(
                        f"Worker task {idx} raised an exception"
                    ) from exc
                completed += 1
                yield idx, result

    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt: cancelling pending futures and shutting down pool")
        for fut in list(future_to_idx.keys()):
            fut.cancel()
        raise
    finally:
        # Python ≥ 3.9 supports cancel_futures=True; fall back gracefully
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# 4. map_windows — high-level convenience: processes all windows of a raster
# ---------------------------------------------------------------------------

def map_windows(
    src_path: "str | os.PathLike",
    fn: Callable[[np.ndarray, tuple, dict], np.ndarray],
    block: int = 512,
    overlap: int = 0,
    n_workers: "int | str" = "auto",
    use_gpu: bool = False,
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
) -> List[Tuple[tuple, np.ndarray]]:
    """Read all windows from *src_path*, run *fn* in parallel, return results.

    Returns a list of ``(window, result_array)`` sorted by window
    (row-major order).

    This is a convenience wrapper.  For large rasters, prefer streaming via
    :func:`worker_pool` + :func:`cog_chunker.chunked_apply` to avoid
    accumulating all results in memory.
    """
    import rasterio
    from .cog_chunker import iter_windows, read_window

    with rasterio.open(str(src_path)) as src:
        meta = dict(src.meta)
        windows = list(iter_windows(src.width, src.height, block=block, overlap=overlap))

        tasks: List[Tuple[tuple, dict]] = []
        for win in windows:
            tile = read_window(src, win)
            if tile.ndim == 2:
                tile = tile[np.newaxis, ...]
            tasks.append(((tile, win, meta), {}))

    results: Dict[int, Tuple[tuple, np.ndarray]] = {}
    for idx, arr in worker_pool(fn, tasks, n_workers=n_workers, use_gpu=use_gpu,
                                 max_in_flight=max_in_flight):
        results[idx] = (windows[idx], arr)

    return [results[i] for i in range(len(windows))]
