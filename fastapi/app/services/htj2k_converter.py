"""HTJ2K conversion using OpenJPH (ojph_compress / ojph_expand)."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import rasterio


def convert_to_htj2k(
    input_path: str,
    output_path: str,
    lossless: bool = True,
) -> str:
    """Convert a GeoTIFF to HTJ2K (.jph) using ojph_compress.

    Args:
        input_path: Path to input GeoTIFF (must be EPSG:4326).
        output_path: Desired .jph output path.
        lossless: If True, use reversible (lossless) wavelet transform.

    Returns:
        The output file path.
    """
    # Determine decomposition levels from image dimensions
    with rasterio.open(input_path) as ds:
        width = ds.width
        height = ds.height

    min_dim = min(width, height)
    # Levels so smallest resolution >= 64 px on short side
    num_decomps = max(1, min(12, int(math.log2(max(min_dim, 1) / 64))))

    cmd = [
        "ojph_compress",
        "-i", input_path,
        "-o", output_path,
        "-num_decomps", str(num_decomps),
        "-block_size", "{64,64}",
        "-precincts", "{256,256}",
        "-prog_order", "RPCL",
    ]

    if lossless:
        cmd += ["-reversible", "true"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"ojph_compress failed (exit {result.returncode}): {result.stderr}"
        )

    return output_path


def get_htj2k_info(jph_path: str) -> dict[str, Any]:
    """Read HTJ2K file structure info for the /info endpoint.

    Returns dict with resolution_levels, dimensions at each level,
    precinct layout, etc.
    """
    # Use ojph_expand -info to get codestream info
    cmd = ["ojph_expand", "-i", jph_path, "-info"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    file_size = Path(jph_path).stat().st_size

    info: dict[str, Any] = {
        "file_size_bytes": file_size,
        "resolution_levels": [],
    }

    # Parse output for resolution levels
    # ojph_expand -info outputs structured info about the codestream
    lines = result.stdout.splitlines()
    num_decomps = 0
    width = 0
    height = 0

    for line in lines:
        stripped = line.strip()
        if "num_decomps" in stripped.lower() or "decomposition" in stripped.lower():
            parts = stripped.split()
            for p in parts:
                if p.isdigit():
                    num_decomps = int(p)
                    break
        if "width" in stripped.lower() and "height" in stripped.lower():
            parts = stripped.replace(",", " ").replace("=", " ").split()
            for i, p in enumerate(parts):
                if p.lower() == "width" and i + 1 < len(parts):
                    try:
                        width = int(parts[i + 1])
                    except ValueError:
                        pass
                if p.lower() == "height" and i + 1 < len(parts):
                    try:
                        height = int(parts[i + 1])
                    except ValueError:
                        pass

    # If parsing didn't find dimensions, try rasterio on original
    # (the jph file -- rasterio may not open it, so use what we have)
    if width == 0 or height == 0:
        # Fallback: read from the raw info output
        try:
            import re
            # Try to find dimensions in the output
            dim_match = re.search(r"(\d+)\s*x\s*(\d+)", result.stdout)
            if dim_match:
                width = int(dim_match.group(1))
                height = int(dim_match.group(2))
        except Exception:
            pass

    # Build resolution level info
    if num_decomps == 0:
        num_decomps = 5  # default assumption

    for level in range(num_decomps + 1):
        divisor = 2 ** level
        info["resolution_levels"].append({
            "level": level,
            "width": max(1, width // divisor) if width else None,
            "height": max(1, height // divisor) if height else None,
            "scale": 1.0 / divisor,
        })

    info["num_decompositions"] = num_decomps
    info["precinct_size"] = {"width": 256, "height": 256}
    info["block_size"] = {"width": 64, "height": 64}
    info["progression_order"] = "RPCL"
    info["full_width"] = width or None
    info["full_height"] = height or None

    return info
