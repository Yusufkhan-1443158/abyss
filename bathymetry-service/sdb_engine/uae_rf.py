"""UAE-calibrated cluster + Random Forest SDB model.

StandardScaler -> KMeans (6 optical regimes) -> per-cluster RandomForest,
trained on Sentinel-2 spectra vs. sounding depths across 8 UAE regions.
Inference: features -> scale -> cluster -> matching forest -> depth (m).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .features import build_feature_stack, MAX_DEPTH_M

L = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (Path(__file__).resolve().parent / "models" /
                      "uae_clustered_rf.pkl")


@dataclass
class UAEModel:
    scaler: object
    kmeans: object
    cluster_rfs: Dict[int, object]
    meta: Dict = field(default_factory=dict)

    def predict(self, s2: Dict, water_mask: Optional[np.ndarray] = None,
                clip_max: float = MAX_DEPTH_M) -> Dict:
        feats, water = build_feature_stack(s2)
        if water_mask is not None:
            water = (np.asarray(water_mask, dtype=bool) & water
                     if water_mask.shape == water.shape
                     else np.asarray(water_mask, dtype=bool))
        H, W, _ = feats.shape

        flat = feats.reshape(-1, feats.shape[-1])
        water_flat = water.reshape(-1)
        finite = np.all(np.isfinite(flat), axis=1)
        usable = water_flat & finite
        idx = np.where(usable)[0]
        if len(idx) == 0:
            return {"depth": np.full((H, W), np.nan, dtype=np.float32),
                    "cluster": np.full((H, W), -1, dtype=np.int8),
                    "n_water_predicted": 0}

        Xs = self.scaler.transform(flat[idx])
        clusters = self.kmeans.predict(Xs)

        depth = np.full(flat.shape[0], np.nan, dtype=np.float32)
        cluster_grid = np.full(flat.shape[0], -1, dtype=np.int16)
        cluster_grid[idx] = clusters

        for c, rf in self.cluster_rfs.items():
            mask = clusters == c
            if not mask.any():
                continue
            preds = rf.predict(Xs[mask])
            depth[idx[mask]] = np.clip(preds, 0.0, clip_max).astype(np.float32)

        return {
            "depth": depth.reshape(H, W),
            "cluster": cluster_grid.reshape(H, W).astype(np.int8),
            "n_water_predicted": int(np.sum(np.isfinite(depth))),
        }


_CACHED_MODEL: Optional[UAEModel] = None


def load_model(path: Path = DEFAULT_MODEL_PATH,
               force_reload: bool = False) -> Optional[UAEModel]:
    global _CACHED_MODEL
    path = Path(path)
    if not path.exists():
        return None
    if _CACHED_MODEL is not None and not force_reload:
        return _CACHED_MODEL
    try:
        from .model_io import load_pickle
        model = load_pickle(path)
        if not isinstance(model, UAEModel):
            L.warning("Bundle at %s is not a UAEModel", path)
            return None
        _CACHED_MODEL = model
        L.info("UAE RF model loaded (%d clusters, n_train=%s, R2=%s)",
               len(model.cluster_rfs), model.meta.get("n_train"),
               model.meta.get("in_sample_r2"))
        return model
    except Exception as ex:
        L.warning("UAE RF model load failed: %s", ex)
        return None
