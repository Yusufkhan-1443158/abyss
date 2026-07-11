#!/usr/bin/env python3
"""Train the UAE CNN/MLP variant of the cluster regressor.

Reuses the same XYZ + SWOT_Dhanna training assembly as the RF
variant — only the per-cluster regressor architecture changes.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend import uae_pretrained_cnn as UAECNN  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
L = logging.getLogger("train_uae_cnn")

# Use the exact same region definitions / loader as the RF training
from train_uae_pretrained import REGIONS, _load_region_xyz, _bbox_for_pts, _sample_features_at_points, _fetch_s2_any  # noqa: E402


def main():
    all_features: List[np.ndarray] = []
    all_depths: List[np.ndarray] = []
    all_weights: List[np.ndarray] = []
    region_summaries: List[Dict] = []

    for region in REGIONS:
        L.info(f"=== {region['key']} ===")
        lats, lons, deps, wts = _load_region_xyz(region)
        if len(deps) == 0:
            continue
        bbox = _bbox_for_pts(lats, lons)
        L.info(f"  bbox = {bbox}, total {len(deps):,} pts")
        sd, ed = region["s2_window"]
        L.info(f"  fetching S2 {sd}→{ed}…")
        t0 = time.time()
        try:
            s2 = _fetch_s2_any(bbox, sd, ed, res=10, cloud=20)
        except Exception as ex:
            L.warning(f"  {region['key']}: S2 fetch failed ({ex}), skipping region")
            continue
        L.info(f"  S2 fetched in {time.time() - t0:.1f}s "
               f"({s2['width']}×{s2['height']} px)")

        feats, mask = _sample_features_at_points(s2, bbox, lats, lons)
        n_in = int(mask.sum())
        L.info(f"  on-water samples kept: {n_in:,} / {len(deps):,}")
        if n_in < 200:
            L.warning(f"  {region['key']}: too few on-water samples, skipping")
            continue
        all_features.append(feats[mask])
        all_depths.append(deps[mask])
        all_weights.append(wts[mask])
        region_summaries.append({
            "region": region["key"],
            "bbox": bbox, "s2_window": [sd, ed],
            "n_samples": n_in,
            "depth_range_m": [float(deps[mask].min()), float(deps[mask].max())],
            "depth_mean_m": float(deps[mask].mean()),
        })

    if not all_features:
        raise SystemExit("No regions produced training data")

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.concatenate(all_depths, axis=0).astype(np.float32)
    sw = np.concatenate(all_weights, axis=0).astype(np.float32)
    L.info(f"COMBINED: {len(y):,} pts, depth range {y.min():.2f}-{y.max():.2f}m, "
           f"regions={[r['region'] for r in region_summaries]}")

    model = UAECNN.fit_uae_cnn(X, y, n_clusters=6, source_weights=sw)
    model.meta["regions"] = region_summaries
    out = UAECNN.save_model(model)
    L.info(f"DONE → {out}")


if __name__ == "__main__":
    main()
