"""TRAIN-R8 — point_fingerprint: content hash of a coordinate/depth array
without ever storing or logging the raw values. Used by the registry
validator (TRAIN-R3, sources[i].content_sha256) and the retrain script
(TRAIN-R5) to detect "same harvest re-submitted" vs "genuinely new data".
"""
from __future__ import annotations

import hashlib

import numpy as np


def point_fingerprint(lats, lons, depths) -> str:
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    depths = np.asarray(depths, dtype=float)
    lines = sorted(f"{la:.6f},{lo:.6f},{d:.3f}"
                   for la, lo, d in zip(lats, lons, depths))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()
