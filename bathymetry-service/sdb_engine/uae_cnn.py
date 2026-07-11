"""UAE-calibrated cluster + MLP SDB model (PyTorch, optional).

Same scaler/kmeans front-end as the RF variant; the per-cluster regressor is
a small MLP (13 -> 64 -> 64 -> 32 -> 1, GELU, dropout 0.2). Loads only when
torch is available; the ensemble degrades gracefully to RF-only otherwise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .features import build_feature_stack, N_FEATURES, MAX_DEPTH_M

L = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (Path(__file__).resolve().parent / "models" /
                      "uae_clustered_cnn.pkl")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    TORCH_AVAILABLE = False


class _DepthMLP(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_dim: int = N_FEATURES, hidden: int = 64,
                 dropout: float = 0.20):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class UAECNNModel:
    scaler: object
    kmeans: object
    cluster_nets: Dict[int, dict]   # cluster_id -> {'state_dict', 'in_dim'}
    meta: Dict = field(default_factory=dict)
    _live: Dict[int, object] = field(default_factory=dict)

    def _rebuild_nets(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        for c, blob in self.cluster_nets.items():
            net = _DepthMLP(in_dim=blob.get("in_dim", N_FEATURES))
            net.load_state_dict(blob["state_dict"])
            net.eval()
            self._live[c] = net

    def predict(self, s2, water_mask=None, clip_max=MAX_DEPTH_M):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available at predict time")
        if not self._live:
            self._rebuild_nets()

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
        depth = np.full(flat.shape[0], np.nan, dtype=np.float32)
        cluster_grid = np.full(flat.shape[0], -1, dtype=np.int16)
        if len(idx) == 0:
            return {"depth": depth.reshape(H, W),
                    "cluster": cluster_grid.reshape(H, W).astype(np.int8),
                    "n_water_predicted": 0}
        Xs = self.scaler.transform(flat[idx])
        clusters = self.kmeans.predict(Xs)
        cluster_grid[idx] = clusters
        with torch.no_grad():
            for c, net in self._live.items():
                mask = clusters == c
                if not mask.any():
                    continue
                Xt = torch.tensor(Xs[mask], dtype=torch.float32)
                preds = net(Xt).cpu().numpy()
                depth[idx[mask]] = np.clip(preds, 0.0, clip_max).astype(np.float32)
        return {
            "depth": depth.reshape(H, W),
            "cluster": cluster_grid.reshape(H, W).astype(np.int8),
            "n_water_predicted": int(np.sum(np.isfinite(depth))),
        }


_CACHED_MODEL: Optional[UAECNNModel] = None


def load_model(path: Path = DEFAULT_MODEL_PATH,
               force_reload: bool = False) -> Optional[UAECNNModel]:
    global _CACHED_MODEL
    if not TORCH_AVAILABLE:
        return None
    path = Path(path)
    if not path.exists():
        return None
    if _CACHED_MODEL is not None and not force_reload:
        return _CACHED_MODEL
    try:
        from .model_io import load_pickle
        model = load_pickle(path)
        if not isinstance(model, UAECNNModel):
            L.warning("Bundle at %s is not a UAECNNModel", path)
            return None
        _CACHED_MODEL = model
        L.info("UAE MLP model loaded (%d clusters, n_train=%s)",
               len(model.cluster_nets), model.meta.get("n_train_total"))
        return model
    except Exception as ex:
        L.warning("UAE MLP model load failed: %s", ex)
        return None
