"""Depth colour ramp (shallow cyan -> abyssal navy), matching the UI tokens."""
from __future__ import annotations

import numpy as np

DEPTH_RAMP = np.array([
    [125, 249, 255], [33, 212, 212], [31, 143, 209],
    [29, 95, 176], [23, 58, 134], [11, 31, 86],
], dtype=np.float64)


def depth_colormap(norm: np.ndarray) -> np.ndarray:
    """Map normalised depth (0..1) to (H, W, 3) uint8 RGB."""
    n = DEPTH_RAMP.shape[0] - 1
    pos = np.clip(norm, 0.0, 1.0) * n
    lo = np.floor(pos).astype(int)
    hi = np.clip(lo + 1, 0, n)
    frac = (pos - lo)[..., None]
    return (DEPTH_RAMP[lo] * (1 - frac) + DEPTH_RAMP[hi] * frac).astype(np.uint8)
