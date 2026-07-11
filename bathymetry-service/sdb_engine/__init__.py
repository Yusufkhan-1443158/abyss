"""Satellite-Derived Bathymetry engine (UAE-calibrated cluster ensemble).

Vendored from the Bathymetry_VMarch production backend. Self-contained:
feature engineering, pretrained model bundles (RF + MLP), NDWI land cut,
depth colour ramp and the `infer_depth()` entry point.
"""
from .inference import (  # noqa: F401
    infer,
    infer_depth,
    model_info,
    MAX_DEPTH_M,
)
from .colormap import depth_colormap, DEPTH_RAMP  # noqa: F401
