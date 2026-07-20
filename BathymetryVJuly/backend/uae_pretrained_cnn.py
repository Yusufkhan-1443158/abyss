"""
UAE Cluster + small CNN/MLP pretrained SDB
==========================================

Same architecture as `backend.uae_pretrained` (StandardScaler → KMeans
→ per-cluster regressor) except the per-cluster regressor is a small
PyTorch MLP/CNN-style network instead of a RandomForestRegressor.

Each per-cluster network:

    Input    13 spectral features  (scaled)
    Layer 1  Linear(13 → 64) + GELU + Dropout(0.20)
    Layer 2  Linear(64 → 64) + GELU + Dropout(0.20)
    Layer 3  Linear(64 → 32) + GELU
    Output   Linear(32 → 1)         (depth in metres, clipped to [0, 25])

Trained with weighted MSE on an 80 / 20 split (per-source weights:
in-situ 5.5, SlideRule 5.0, user 6.0, GEBCO 1.0). Adam, lr=1e-3,
ReduceLROnPlateau, early stop after 12 epochs without val improvement.

Inference is the same shape as `UAEModel.predict()`:
    feats → scaler → kmeans cluster → cluster network → depth.

In-situ injection ("darkbox" fine-tune):
    .fine_tune_on_region(features, depths, weights, n_epochs=20)
    Updates only the weights of the per-cluster network whose centroid
    is closest to the supplied features. Used to nudge predictions
    toward the user's calibration set on a predefined region without
    pushing it to disk.
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

L = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (Path(__file__).resolve().parent / "models" /
                      "uae_clustered_cnn.pkl")
MAX_DEPTH_M = 25.0

# Reuse the feature engineering from the RF version
from backend.uae_pretrained import (build_feature_stack, FEATURE_NAMES, N_FEATURES)  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────
# Network architecture
# ────────────────────────────────────────────────────────────────────
class _DepthMLP(nn.Module if TORCH_AVAILABLE else object):
    """Tiny MLP — 13 features → depth in metres."""

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


def _train_one(net, X, y, w, *, epochs=200, batch=256, lr=1e-3,
               val_frac=0.20, patience=12, seed=42, device='cpu'):
    """Train the small MLP on (X, y, w) with an 80/20 split inside."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(y)
    idx = np.arange(n); np.random.shuffle(idx)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    Xtr, ytr, wtr = Xt[tr_idx], yt[tr_idx], wt[tr_idx]
    Xva, yva, wva = Xt[val_idx], yt[val_idx], wt[val_idx]

    net = net.to(device)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    best_val = float('inf'); best_state = None; epochs_since_best = 0
    n_tr = len(tr_idx)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, batch):
            sel = perm[s:s + batch]
            pred = net(Xtr[sel])
            err = (pred - ytr[sel]) ** 2
            loss = (err * wtr[sel]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xva)
            val_loss = (((pv - yva) ** 2) * wva).mean().item()
        sched.step(val_loss)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        train_pred = net(Xtr).cpu().numpy()
        val_pred = net(Xva).cpu().numpy()
    return net, {
        'epochs': ep + 1,
        'val_rmse': float(np.sqrt(np.mean((val_pred - y[val_idx]) ** 2))),
        'val_bias': float(np.mean(val_pred - y[val_idx])),
        'val_r2': float(1 - np.sum((val_pred - y[val_idx]) ** 2) /
                        max(np.sum((y[val_idx] - y[val_idx].mean()) ** 2), 1e-9)),
        'train_rmse': float(np.sqrt(np.mean((train_pred - y[tr_idx]) ** 2))),
    }


# ────────────────────────────────────────────────────────────────────
# Bundle (pickled to disk)
# ────────────────────────────────────────────────────────────────────
@dataclass
class UAECNNModel:
    scaler: object
    kmeans: object
    cluster_nets: Dict[int, dict]   # cluster_id → {'state_dict': ..., 'in_dim': ...}
    meta: Dict = field(default_factory=dict)
    _live: Dict[int, object] = field(default_factory=dict)  # rebuilt at load

    def _rebuild_nets(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        for c, blob in self.cluster_nets.items():
            net = _DepthMLP(in_dim=blob.get('in_dim', N_FEATURES))
            net.load_state_dict(blob['state_dict'])
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

    def fine_tune(self, features, depths, weights=None,
                  n_epochs=20, lr=5e-4):
        """In-situ injection — quick fine-tune of the cluster nets whose
        centroids are touched by `features`. Used per-request when an
        ROI overlaps a calibration region.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        # DETERMINISM (VERYHR_STABLE_LOG ROUND 2, AC-2): a per-request fine-tune
        # must NOT accumulate across calls. `fine_tune` mutates the live nets in
        # place; without a reset, the 2nd call fine-tunes the already-tuned nets
        # of the 1st, so the same ROI returns a different grid each time (~8 m
        # drift). Rebuild the live nets from the FROZEN pickled state_dicts every
        # call so each fine-tune starts from the immutable baseline → idempotent.
        self._rebuild_nets()
        torch.manual_seed(42)  # Adam has no RNG, but pin for any future dropout.
        feats = np.asarray(features, dtype=np.float32)
        deps = np.asarray(depths, dtype=np.float32)
        wts = (np.ones_like(deps) if weights is None
               else np.asarray(weights, dtype=np.float32))
        ok = np.all(np.isfinite(feats), axis=1) & np.isfinite(deps) & (deps > 0)
        feats, deps, wts = feats[ok], deps[ok], wts[ok]
        if len(deps) < 20:
            return {"updated": [], "n_used": int(len(deps))}
        Xs = self.scaler.transform(feats)
        clusters = self.kmeans.predict(Xs)
        updated = []
        for c in np.unique(clusters):
            m = clusters == c
            if int(m.sum()) < 10 or c not in self._live:
                continue
            net = self._live[int(c)]
            Xt = torch.tensor(Xs[m], dtype=torch.float32)
            yt = torch.tensor(deps[m], dtype=torch.float32)
            wtensor = torch.tensor(wts[m], dtype=torch.float32)
            opt = optim.Adam(net.parameters(), lr=lr)
            net.train()
            for _ in range(n_epochs):
                pred = net(Xt)
                err = (pred - yt) ** 2
                loss = (err * wtensor).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            updated.append(int(c))
        return {"updated": updated, "n_used": int(len(deps))}


# ────────────────────────────────────────────────────────────────────
# Training driver
# ────────────────────────────────────────────────────────────────────
def fit_uae_cnn(features: np.ndarray, depths: np.ndarray,
                source_weights: Optional[np.ndarray] = None,
                n_clusters: int = 6, seed: int = 42,
                device: str = 'cpu') -> UAECNNModel:
    """Fit StandardScaler → KMeans → per-cluster MLP.
    Each MLP trains internally with an 80 / 20 split."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    if features.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {features.shape[1]}")
    finite = np.all(np.isfinite(features), axis=1) & np.isfinite(depths) & (depths > 0)
    X = features[finite]
    y = depths[finite].astype(np.float32)
    w = (np.ones_like(y) if source_weights is None
         else np.asarray(source_weights)[finite].astype(np.float32))
    if len(y) < n_clusters * 50:
        raise RuntimeError(f"Too few training points: {len(y)}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    clusters = kmeans.fit_predict(Xs)

    cluster_nets = {}
    per_cluster_metrics = {}
    for c in range(n_clusters):
        m = clusters == c
        if int(m.sum()) < 30:
            L.warning(f"  cluster {c}: only {int(m.sum())} pts — skip")
            continue
        net = _DepthMLP(in_dim=Xs.shape[1])
        net, info = _train_one(net, Xs[m], y[m], w[m], device=device, seed=seed + c)
        cluster_nets[c] = {
            'state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
            'in_dim': Xs.shape[1],
        }
        per_cluster_metrics[c] = {
            "n": int(m.sum()),
            "depth_range_m": [float(y[m].min()), float(y[m].max())],
            "depth_mean_m": float(y[m].mean()),
            "val_rmse_m": round(info['val_rmse'], 3),
            "val_bias_m": round(info['val_bias'], 3),
            "val_r2": round(info['val_r2'], 4),
            "train_rmse_m": round(info['train_rmse'], 3),
            "epochs": info['epochs'],
        }
        L.info(f"  cluster {c}: n={int(m.sum()):,} "
               f"val RMSE={info['val_rmse']:.2f}m  bias={info['val_bias']:+.2f}m  "
               f"R²={info['val_r2']:.3f}  ep={info['epochs']}")

    meta = {
        "version": 1,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_names": list(FEATURE_NAMES),
        "n_clusters": n_clusters,
        "n_train_total": int(len(y)),
        "depth_range_m": [float(y.min()), float(y.max())],
        "per_cluster": per_cluster_metrics,
        "model_kind": "uae_cnn_mlp",
    }
    return UAECNNModel(scaler=scaler, kmeans=kmeans,
                       cluster_nets=cluster_nets, meta=meta)


def save_model(model: UAECNNModel, path: Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model._live = {}  # don't pickle the live nets — only state_dicts
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    L.info(f"UAE CNN model saved to {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


_CACHED_CNN_MODEL: Optional[UAECNNModel] = None


def load_model(path: Path = DEFAULT_MODEL_PATH,
               force_reload: bool = False) -> Optional[UAECNNModel]:
    global _CACHED_CNN_MODEL
    if not TORCH_AVAILABLE:
        return None
    path = Path(path)
    if not path.exists():
        return None
    if _CACHED_CNN_MODEL is not None and not force_reload:
        return _CACHED_CNN_MODEL
    try:
        with open(path, "rb") as fh:
            m = pickle.load(fh)
        if not isinstance(m, UAECNNModel):
            return None
        _CACHED_CNN_MODEL = m
        L.info(f"UAE CNN model loaded "
               f"({len(m.cluster_nets)} clusters, "
               f"trained on {m.meta.get('n_train_total', 0):,} pts)")
        return m
    except Exception as ex:
        L.warning(f"UAE CNN model load failed: {ex}")
        return None
