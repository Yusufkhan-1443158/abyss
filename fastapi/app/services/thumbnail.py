"""Thumbnail generation wrapper."""

from __future__ import annotations

from .raster_processor import generate_thumbnail as _gdal_thumbnail


def generate_raster_thumbnail(
    input_path: str,
    output_path: str,
    size: int = 256,
) -> str:
    """Generate a JPEG thumbnail from a raster file.

    Wrapper around GDAL-based thumbnail generation.
    """
    return _gdal_thumbnail(input_path, output_path, size=size)
