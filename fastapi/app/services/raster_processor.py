"""GDAL/rasterio-based raster metadata extraction and processing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import rasterio
from osgeo import gdal
from rasterio.warp import calculate_default_transform

gdal.UseExceptions()


def extract_metadata(file_path: str) -> dict[str, Any]:
    """Open a raster with rasterio and extract comprehensive metadata.

    Returns dict with: crs, bbox (WKT polygon), center_point (WKT),
    width_px, height_px, num_bands, bit_depth, resolution_m, min_zoom, max_zoom.
    """
    with rasterio.open(file_path) as ds:
        bounds = ds.bounds
        crs = ds.crs
        width = ds.width
        height = ds.height
        num_bands = ds.count
        dtypes = ds.dtypes

        # Bit depth from first band dtype
        dtype_str = dtypes[0]
        bit_depth_map = {
            "uint8": 8, "int8": 8,
            "uint16": 16, "int16": 16,
            "uint32": 32, "int32": 32,
            "float32": 32, "float64": 64,
        }
        bit_depth = bit_depth_map.get(dtype_str, 8)

        # Calculate ground resolution in meters
        if crs and not crs.is_geographic:
            # Projected CRS -- res is already in map units (meters for most)
            res_x, res_y = ds.res
            resolution_m = (abs(res_x) + abs(res_y)) / 2.0
        else:
            # Geographic CRS -- approximate meters at center latitude
            res_x, res_y = ds.res
            center_lat = (bounds.bottom + bounds.top) / 2.0
            meters_per_deg = 111_320 * math.cos(math.radians(center_lat))
            resolution_m = ((abs(res_x) + abs(res_y)) / 2.0) * meters_per_deg

        # Convert bounds to EPSG:4326 for storage
        if crs and str(crs) != "EPSG:4326":
            transform, w, h = calculate_default_transform(
                crs, "EPSG:4326", width, height,
                left=bounds.left, bottom=bounds.bottom,
                right=bounds.right, top=bounds.top,
            )
            # Get transformed bounds
            from rasterio.transform import array_bounds
            b = array_bounds(h, w, transform)
            left, bottom, right, top = b
        else:
            left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

        # Bbox as WKT polygon
        bbox_wkt = (
            f"SRID=4326;POLYGON(({left} {bottom}, {right} {bottom}, "
            f"{right} {top}, {left} {top}, {left} {bottom}))"
        )

        # Center point
        cx = (left + right) / 2.0
        cy = (bottom + top) / 2.0
        center_wkt = f"SRID=4326;POINT({cx} {cy})"

        # Zoom levels from resolution
        # At zoom 0, one pixel ~156543 m; each zoom halves
        if resolution_m > 0:
            max_zoom = max(0, min(24, int(math.log2(156543.0 / resolution_m))))
            min_zoom = max(0, max_zoom - 8)
        else:
            max_zoom = 18
            min_zoom = 0

    return {
        "crs": str(crs) if crs else "EPSG:4326",
        "bbox_wkt": bbox_wkt,
        "center_wkt": center_wkt,
        "width_px": width,
        "height_px": height,
        "num_bands": num_bands,
        "bit_depth": bit_depth,
        "resolution_m": round(resolution_m, 4),
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "bounds": {"left": left, "bottom": bottom, "right": right, "top": top},
    }


def reproject_to_4326(input_path: str, output_path: str) -> str:
    """Reproject a raster to EPSG:4326 using gdalwarp. Returns output path."""
    opts = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        resampleAlg="bilinear",
        format="GTiff",
    )
    ds = gdal.Warp(output_path, input_path, options=opts)
    ds = None  # noqa: F841 — flush and close
    return output_path


def build_cog(input_path: str, output_path: str) -> str:
    """Build a Cloud-Optimized GeoTIFF (internally tiled + overviews) from any
    GDAL-readable raster, for fast random/overview tile reads.

    This is the work that used to happen lazily on the FIRST map tile request
    (in routers.rasters._get_cached_raster). Doing it once at ingest — and
    writing it straight into the shared tile cache — is what makes the very
    first tile sub-second (it is already warm) instead of paying a 2–10s build.

    The COG driver generates the internal overview pyramid itself; we don't need
    a separate BuildOverviews pass (which also failed on non-GeoTIFF inputs).
    Falls back to a plain GeoTIFF copy if the COG driver rejects the source.
    """
    try:
        gdal.Translate(
            output_path,
            input_path,
            format="COG",
            creationOptions=[
                "COMPRESS=DEFLATE",
                "BLOCKSIZE=256",
                "OVERVIEW_RESAMPLING=AVERAGE",
            ],
        )
    except Exception:
        gdal.Translate(output_path, input_path, format="GTiff",
                       creationOptions=["TILED=YES", "COMPRESS=DEFLATE"])
    return output_path


def compute_band_percentiles(
    ds, percentile_low: float = 2.5, percentile_high: float = 97.5
) -> list[list[float]]:
    """Compute per-band percentile stretch parameters.

    Returns list of [src_min, src_max, 0, 255] for each band,
    suitable for gdal.TranslateOptions scaleParams.
    """
    import numpy as np

    scale_params = []
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b)
        # Read at reduced resolution for speed on large rasters
        data = band.ReadAsArray(
            buf_xsize=min(512, ds.RasterXSize),
            buf_ysize=min(512, ds.RasterYSize),
        )
        nodata = band.GetNoDataValue()
        flat = data.flatten().astype(float)
        if nodata is not None:
            flat = flat[flat != nodata]
        if len(flat) == 0:
            scale_params.append([0, 255, 0, 255])
            continue
        lo = float(np.percentile(flat, percentile_low))
        hi = float(np.percentile(flat, percentile_high))
        if hi <= lo:
            hi = lo + 1
        scale_params.append([lo, hi, 0, 255])
    return scale_params


def generate_thumbnail(input_path: str, output_path: str, size: int = 256) -> str:
    """Generate a JPEG thumbnail. 8-bit imagery (e.g. a display-ready RGB depth
    product) is copied as-is; higher-bit-depth sensor data gets a 95% stretch.
    Re-stretching an already-colour-mapped 8-bit RGB would distort its palette."""
    ds = gdal.Open(input_path)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {input_path}")

    is_byte = ds.GetRasterBand(1).DataType == gdal.GDT_Byte
    kwargs = dict(format="JPEG", width=size, height=size,
                  resampleAlg="average", outputType=gdal.GDT_Byte)
    if not is_byte:
        kwargs["scaleParams"] = compute_band_percentiles(ds)
    gdal.Translate(output_path, ds, options=gdal.TranslateOptions(**kwargs))
    ds = None  # noqa: F841
    return output_path
