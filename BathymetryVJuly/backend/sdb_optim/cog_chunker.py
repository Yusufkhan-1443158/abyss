"""
Memory-safe COG/NetCDF chunked I/O for SDB inference pipelines.

INTEGRATION:
  - Feeds raster tiles to `parallel_infer.worker_pool` (this module).
  - Wraps any rasterio-readable raster produced by `backend/very_hr_cbr.py`
    (depth.tif, depth_sigma.tif) or Sentinel-2 COGs fetched by GEE.
  - `chunked_apply` is the single-function interface used by
    `subsystem_interface.run_inference_job`.

Memory budget
-------------
Peak resident ≈ block_size² × n_bands × itemsize × (1 + n_workers)

    e.g. block=512 px, 4 bands, float32 (4 B), 4 workers:
         512² × 4 × 4 × 5 ≈ 20 MB  ← acceptable for 32-GB nodes

For block=2048 and 8 workers the same math gives ~1.3 GB — still safe but
worth noting when planning GPU workers that keep activations alive.

References
----------
- Cloud-Optimized GeoTIFF specification: https://cogeo.org/
  (internal tiling + overviews → random-access without full download)
- Rasterio windowed I/O: https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html
- xarray chunked reading: https://docs.xarray.dev/en/stable/user-guide/dask.html
"""
from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Callable,
    Generator,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

log = logging.getLogger("sdb_optim.cog_chunker")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PathLike = Union[str, Path]
Window = Tuple[int, int, int, int]   # (col_off, row_off, width, height)
WindowArray = np.ndarray             # shape (bands, height, width) or (height, width)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_block(val: int, block: int) -> int:
    """Round *val* up to the nearest multiple of *block* (for IO alignment)."""
    return math.ceil(val / block) * block


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# 1. iter_windows — pure geometry, no rasterio import required for testing
# ---------------------------------------------------------------------------

def iter_windows(
    width: int,
    height: int,
    block: int = 512,
    overlap: int = 0,
) -> Iterator[Window]:
    """Yield non-overlapping (or overlapping) raster windows covering the full image.

    Parameters
    ----------
    width, height : int
        Full raster dimensions in pixels.
    block : int
        Target tile edge in pixels. Last tiles are clipped to image boundary;
        they may be smaller than *block*.
    overlap : int
        Number of pixels of overlap on each edge (≥ 0).  The caller is
        responsible for trimming the overlap zone before assembling outputs to
        avoid double-counting.

    Yields
    ------
    (col_off, row_off, win_width, win_height) tuples — rasterio Window-compatible.

    Notes
    -----
    With ``overlap=0`` the union of all windows is exactly the full image with
    no gaps and no duplicates (tested in T4b).
    With ``overlap>0`` adjacent windows share *overlap* pixels on each shared
    edge.  The stride becomes ``block - overlap``.
    """
    if block <= 0:
        raise ValueError(f"block must be > 0, got {block}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= block:
        raise ValueError(f"overlap ({overlap}) must be < block ({block})")

    stride = block - overlap

    row = 0
    while row < height:
        col = 0
        while col < width:
            win_w = _clamp(block, 1, width - col)
            win_h = _clamp(block, 1, height - row)
            yield (col, row, win_w, win_h)
            col += stride
        row += stride


# ---------------------------------------------------------------------------
# 2. read_window — rasterio-backed read of a single window
# ---------------------------------------------------------------------------

def read_window(
    src,   # open rasterio.DatasetReader
    window: Window,
    band_indices: Optional[Sequence[int]] = None,
    masked: bool = False,
) -> np.ndarray:
    """Read a spatial window from an open rasterio dataset.

    Parameters
    ----------
    src :
        Open ``rasterio.DatasetReader`` (caller owns the context manager).
    window : (col_off, row_off, width, height)
        Pixel window as returned by :func:`iter_windows`.
    band_indices :
        1-based band indices to read.  ``None`` → all bands.
    masked : bool
        If True, return a masked array (nodata → masked).  Default False.

    Returns
    -------
    np.ndarray  shape (n_bands, height, width) or (height, width) for single band.
    """
    import rasterio
    from rasterio.windows import Window as RioWindow

    col_off, row_off, win_w, win_h = window
    rio_win = RioWindow(col_off, row_off, win_w, win_h)

    if band_indices is None:
        band_indices = list(range(1, src.count + 1))

    arr = src.read(band_indices, window=rio_win, masked=masked)
    if len(band_indices) == 1:
        arr = arr[0]
    return arr


# ---------------------------------------------------------------------------
# 3. read_window_transform — returns the affine transform for a window
# ---------------------------------------------------------------------------

def window_transform(
    src,   # open rasterio.DatasetReader
    window: Window,
):
    """Return the affine transform for a sub-window of a dataset.

    This is the transform that maps pixel (0,0) inside the window to its
    real-world coordinates.
    """
    from rasterio.windows import Window as RioWindow
    from rasterio.transform import guard_transform
    col_off, row_off, win_w, win_h = window
    rio_win = RioWindow(col_off, row_off, win_w, win_h)
    return src.window_transform(rio_win)


# ---------------------------------------------------------------------------
# 4. chunked_apply — streaming window-by-window processing
# ---------------------------------------------------------------------------

def chunked_apply(
    src_path: PathLike,
    fn: Callable[[np.ndarray, Window, object], np.ndarray],
    out_path: PathLike,
    block: int = 512,
    overlap: int = 0,
    nodata: Optional[float] = None,
    dtype: Optional[str] = None,
    n_bands_out: Optional[int] = None,
    compress: str = "deflate",
    predictor: int = 2,
    add_overviews: bool = True,
    overview_levels: Sequence[int] = (2, 4, 8, 16),
) -> Path:
    """Stream *fn* over all windows of *src_path* and write a tiled COG.

    *fn* signature::

        fn(tile: np.ndarray,
           window: Window,
           meta: dict) -> np.ndarray

    where *tile* has shape ``(n_bands, h, w)`` (or ``(h, w)`` for single-band
    source), *window* is ``(col_off, row_off, width, height)``, and *meta* is
    the source dataset's ``meta`` dict augmented with
    ``{"window_transform": affine, "nodata": nodata}``.

    The returned array must have the same spatial footprint (h, w) but may have
    a different number of bands or dtype — controlled by *n_bands_out* and
    *dtype*.

    Parameters
    ----------
    src_path :
        Path to any rasterio-readable raster (GeoTIFF, COG, VRT, etc.).
    fn :
        Callable that processes one tile.  Must be pure / side-effect-free;
        exceptions propagate immediately.
    out_path :
        Destination GeoTIFF path (written as internally-tiled COG).
    block :
        Tile size in pixels.
    overlap :
        Overlap pixels (caller must trim margins before returning from *fn*
        to avoid seam artifacts).
    nodata :
        Output nodata value.  Defaults to the source nodata or ``np.nan``.
    dtype :
        Output dtype (e.g. ``"float32"``).  Defaults to source dtype.
    n_bands_out :
        Number of output bands.  Inferred from the first window's result if None.
    compress :
        DEFLATE compression (change to "lzw" for integer rasters).
    predictor :
        DEFLATE predictor: 2=horizontal differencing (float/int), 3=float predictor.
    add_overviews :
        Build internal overview levels for fast zoom-out rendering.
    overview_levels :
        Overview decimation factors.

    Returns
    -------
    Path to the written output file.

    Notes
    -----
    Memory peak ≈ 2 × block² × n_bands × itemsize (source tile + output tile).
    The output is written as a tiled GeoTIFF with tile size 256×256 (standard
    COG block).  If rasterio.shutil.copy is available, a second pass converts
    to a proper COG layout (overviews at front of file); otherwise the tiled
    GeoTIFF is returned directly — structurally equivalent for windowed reading.
    """
    import rasterio
    from rasterio.transform import from_bounds, guard_transform
    from rasterio.windows import Window as RioWindow
    from rasterio.enums import Resampling
    import tempfile

    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        src_meta = dict(src.meta)
        src_nodata = src.nodata if nodata is None else nodata
        out_nodata = src_nodata if src_nodata is not None else np.nan
        out_dtype = dtype or src.dtypes[0]
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs

        windows = list(iter_windows(width, height, block=block, overlap=overlap))

        # --- first-pass: probe output band count if not given ---
        if n_bands_out is None:
            w0 = windows[0]
            tile0 = read_window(src, w0)
            if tile0.ndim == 2:
                tile0 = tile0[np.newaxis, ...]
            meta0 = dict(src_meta)
            meta0["window_transform"] = window_transform(src, w0)
            meta0["nodata"] = out_nodata
            result0 = fn(tile0 if tile0.shape[0] > 1 else tile0[0], w0, meta0)
            if isinstance(result0, np.ndarray):
                n_bands_out = 1 if result0.ndim == 2 else result0.shape[0]
            else:
                n_bands_out = 1
            probe_result = result0
            probe_window = w0
        else:
            probe_result = None
            probe_window = None

        out_profile = {
            "driver": "GTiff",
            "dtype": out_dtype,
            "width": width,
            "height": height,
            "count": n_bands_out,
            "crs": crs,
            "transform": transform,
            "nodata": out_nodata,
            "tiled": True,
            "blockxsize": min(256, width),
            "blockysize": min(256, height),
            "compress": compress,
            "predictor": predictor,
        }

        # Write to a temp file first (overviews require a seekable file)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif",
                                             dir=out_path.parent,
                                             prefix=".tmp_cog_")
        os.close(tmp_fd)

        try:
            with rasterio.open(tmp_path, "w", **out_profile) as dst:

                def _write_result(result: np.ndarray, win: Window):
                    col_off, row_off, win_w, win_h = win
                    rio_win = RioWindow(col_off, row_off, win_w, win_h)
                    if result.ndim == 2:
                        result = result[np.newaxis, ...]
                    # Ensure shape matches window
                    expected = (n_bands_out, win_h, win_w)
                    if result.shape != expected:
                        raise ValueError(
                            f"fn returned shape {result.shape}, expected {expected} "
                            f"for window {win}"
                        )
                    dst.write(result.astype(out_dtype), window=rio_win)

                # write probe result if we have it
                if probe_result is not None:
                    _write_result(probe_result, probe_window)

                for win in windows:
                    if probe_window is not None and win == probe_window:
                        continue   # already written
                    tile = read_window(src, win)
                    if tile.ndim == 2:
                        tile = tile[np.newaxis, ...]
                    meta = dict(src_meta)
                    meta["window_transform"] = window_transform(src, win)
                    meta["nodata"] = out_nodata
                    result = fn(tile if tile.shape[0] > 1 else tile[0], win, meta)
                    _write_result(result, win)

            # Build overviews in-place on the tmp file
            if add_overviews and any(
                (width // lvl > 1 and height // lvl > 1)
                for lvl in overview_levels
            ):
                with rasterio.open(tmp_path, "r+") as dst:
                    valid_levels = [
                        lvl for lvl in overview_levels
                        if width // lvl > 1 and height // lvl > 1
                    ]
                    dst.build_overviews(valid_levels, Resampling.average)
                    dst.update_tags(ns="rio_overview", resampling="average")

            # Convert to proper COG layout (overviews at front of file)
            try:
                from rasterio.shutil import copy as rio_copy
                cog_profile = dict(out_profile)
                cog_profile["copy_src_overviews"] = True
                rio_copy(tmp_path, str(out_path), driver="GTiff", **{
                    k: v for k, v in cog_profile.items()
                    if k in ("dtype", "compress", "predictor", "tiled",
                             "blockxsize", "blockysize", "copy_src_overviews")
                })
                log.info(f"chunked_apply: COG written → {out_path} "
                         f"({width}×{height} px, block={block})")
            except Exception as cog_err:
                log.warning(f"COG copy failed ({cog_err}); using tiled GeoTIFF")
                import shutil
                shutil.move(tmp_path, str(out_path))
                tmp_path = None

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return out_path


# ---------------------------------------------------------------------------
# 5. read_netcdf_window — xarray-backed chunked reading (no rioxarray needed)
# ---------------------------------------------------------------------------

def iter_netcdf_chunks(
    nc_path: PathLike,
    var: str,
    chunk_size: int = 512,
) -> Iterator[Tuple[np.ndarray, dict]]:
    """Yield numpy chunks from a NetCDF variable without loading the full array.

    Uses xarray's lazy loading (backed by dask if installed, otherwise numpy).
    Each yielded tuple is ``(chunk_array, info_dict)`` where *info_dict* contains
    ``{"i_start", "i_end", "j_start", "j_end"}`` (row/col slice bounds in the
    full 2-D grid).

    Parameters
    ----------
    nc_path :
        Path to a NetCDF-4 / CF-compliant file.
    var :
        Variable name to iterate (must be 2-D or 3-D with leading time dim).
    chunk_size :
        Chunk edge in array index units.

    Notes
    -----
    No rioxarray is required — we rely solely on xarray's native .values accessor
    which triggers a block-read via netCDF4/h5netcdf.  Memory peak ≈
    chunk_size² × itemsize (one chunk at a time).
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required for iter_netcdf_chunks") from exc

    nc_path = Path(nc_path)
    ds = xr.open_dataset(nc_path, chunks={})   # lazy; dask optional
    da = ds[var]

    # Handle optional leading time/depth dimension: use index 0
    if da.ndim == 3:
        da = da.isel({da.dims[0]: 0})
    if da.ndim != 2:
        raise ValueError(f"Variable '{var}' must be 2-D (got {da.ndim}-D after squeezing).")

    nrows, ncols = da.shape

    for r0 in range(0, nrows, chunk_size):
        for c0 in range(0, ncols, chunk_size):
            r1 = min(r0 + chunk_size, nrows)
            c1 = min(c0 + chunk_size, ncols)
            chunk = da.isel(
                {da.dims[0]: slice(r0, r1), da.dims[1]: slice(c0, c1)}
            ).values   # triggers disk read
            info = {"i_start": r0, "i_end": r1, "j_start": c0, "j_end": c1}
            yield chunk, info

    ds.close()
