"""
UAE Cluster + Random Forest pretrained SDB
==========================================

Trains on the bundled Khalifa Port (KP, KP_EMAL) and Old Mussafah (OMC)
XYZ soundings + Sentinel-2 spectra, then exposes a `predict()` function
that runs over a new S2 scene anywhere on Earth (best inside the
calibration envelope: shallow-to-mid Gulf waters).

Architecture
------------
  1.  Spectral features built from 5 S2 bands  (B1, B2, B3, B4, B8)
      plus Stumpf log-ratios (B/G, B/R), Lyzenga log-bands (lnB, lnG,
      lnR, lnC), and NDWI.   13 features total.
  2.  StandardScaler  -- learned during training.
  3.  K-means (default 6 clusters) over scaled features at training
      pixels.   This separates "shallow lagoon", "mid-shelf",
      "deep-channel" type optical regimes.
  4.  Per-cluster RandomForestRegressor (n_estimators=120, max_depth=14)
      trained on (features → depth) restricted to that cluster.
  5.  At inference time: score features → predict cluster per pixel
      → predict depth with the matching forest.

The full bundle (scaler + kmeans + per-cluster RFs + meta) is saved as
a single pickle so the production backend can `load_model()` once and
re-use it across every Sentinel-2 request.

This file deliberately has no Flask / Sentinel-Hub dependency: the
training driver lives in train_uae_pretrained.py (top level).
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
                      "uae_clustered_rf.pkl")
MAX_DEPTH_M = 25.0


# ────────────────────────────────────────────────────────────────────
# Feature engineering — must match between training and inference
# ────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "blue", "green", "red", "coastal", "nir",
    "lnB", "lnG", "lnR", "lnC",
    "stumpf_BG", "stumpf_BR",
    "ndwi", "lyzenga_DII",
]
N_FEATURES = len(FEATURE_NAMES)


def _safe_reflectance(band: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return np.clip(band.astype(np.float64) / 10000.0, eps, None)


def build_feature_stack(s2: Dict, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Build (H, W, N_FEATURES) feature stack + water mask from an S2 dict.

    The input dict is expected to carry keys: blue, green, red, nir,
    coastal, ndwi, water_mask. Bands are uint16 DN (S2 L2A scale).
    """
    blue = _safe_reflectance(s2["blue"], eps)
    green = _safe_reflectance(s2["green"], eps)
    red = _safe_reflectance(s2["red"], eps)
    coastal = _safe_reflectance(s2.get("coastal", s2["blue"]), eps)
    nir = _safe_reflectance(s2.get("nir", s2["red"]), eps)
    ndwi = np.asarray(s2.get("ndwi", (green - nir) / (green + nir + eps)),
                      dtype=np.float64)

    # Stumpf log-ratios (Stumpf 2003)
    n = 1000.0
    stumpf_bg = np.log(n * blue) / np.log(n * green + eps)
    stumpf_br = np.log(n * blue) / np.log(n * red + eps)

    # Lyzenga log-bands (Lyzenga 1985)
    lnB = np.log(blue)
    lnG = np.log(green)
    lnR = np.log(red)
    lnC = np.log(coastal)

    # Lyzenga depth-invariant index DII = lnB - (sigma_B / sigma_G) * lnG
    water = s2.get("water_mask")
    if water is None:
        water = ndwi > 0
    water = np.asarray(water, dtype=bool)
    if water.sum() > 50:
        std_b = float(np.std(lnB[water]))
        std_g = float(np.std(lnG[water]))
        ratio = (std_b / std_g) if std_g > 1e-6 else 1.0
    else:
        ratio = 1.0
    dii = lnB - ratio * lnG

    feats = np.stack([blue, green, red, coastal, nir,
                      lnB, lnG, lnR, lnC,
                      stumpf_bg, stumpf_br,
                      ndwi, dii], axis=-1).astype(np.float32)
    return feats, water


# ────────────────────────────────────────────────────────────────────
# Model bundle
# ────────────────────────────────────────────────────────────────────
@dataclass
class UAEModel:
    scaler: object              # sklearn StandardScaler
    kmeans: object              # sklearn KMeans
    cluster_rfs: Dict[int, object]
    meta: Dict = field(default_factory=dict)

    def predict(self, s2: Dict, water_mask: Optional[np.ndarray] = None,
                clip_max: float = MAX_DEPTH_M) -> Dict:
        """Predict depth grid + per-cluster diagnostics for an S2 scene."""
        feats, water = build_feature_stack(s2)
        if water_mask is not None:
            water = np.asarray(water_mask, dtype=bool) & water if water_mask.shape == water.shape else np.asarray(water_mask, dtype=bool)
        H, W, _ = feats.shape

        # Flatten water pixels only (efficiency)
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
            target_idx = idx[mask]
            depth[target_idx] = np.clip(preds, 0.0, clip_max).astype(np.float32)

        return {
            "depth": depth.reshape(H, W),
            "cluster": cluster_grid.reshape(H, W).astype(np.int8),
            "n_water_predicted": int(np.sum(np.isfinite(depth))),
        }


# ────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────
def fit_uae_model(features: np.ndarray, depths: np.ndarray,
                  n_clusters: int = 6, rf_n_estimators: int = 120,
                  rf_max_depth: int = 14, seed: int = 42,
                  sample_weight: np.ndarray = None) -> UAEModel:
    """Fit StandardScaler → KMeans → per-cluster RandomForest.

    features: (n, N_FEATURES) float32
    depths:   (n,)            float32, in metres, > 0
    sample_weight: (n,) float32, optional — source-quality weighting
    (TRAIN-R2/R8: ATL24 > in-situ XYZ > i-Boating OCR).
    """
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error

    if features.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {features.shape[1]}")
    finite = np.all(np.isfinite(features), axis=1) & np.isfinite(depths) & (depths > 0)
    X = features[finite]
    y = depths[finite].astype(np.float32)
    sw = sample_weight[finite].astype(np.float32) if sample_weight is not None else None
    if len(y) < n_clusters * 50:
        raise RuntimeError(f"Too few training points: {len(y)}")
    L.info(f"UAE training: {len(y):,} pts, {n_clusters} clusters, "
           f"depth range {y.min():.2f}–{y.max():.2f} m")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    clusters = kmeans.fit_predict(Xs)

    cluster_rfs: Dict[int, object] = {}
    per_cluster_metrics: Dict[int, Dict] = {}
    for c in range(n_clusters):
        m = clusters == c
        if m.sum() < 30:
            L.warning(f"  cluster {c}: only {int(m.sum())} pts — skip")
            continue
        Xc, yc = Xs[m], y[m]
        swc = sw[m] if sw is not None else None
        rf = RandomForestRegressor(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth,
            min_samples_leaf=4, n_jobs=-1, random_state=seed,
        )
        rf.fit(Xc, yc, sample_weight=swc)
        pred = rf.predict(Xc)
        r2 = float(r2_score(yc, pred))
        rmse = float(np.sqrt(mean_squared_error(yc, pred)))
        cluster_rfs[c] = rf
        per_cluster_metrics[c] = {
            "n": int(m.sum()),
            "depth_range_m": [float(yc.min()), float(yc.max())],
            "depth_mean_m": float(yc.mean()),
            "in_sample_r2": round(r2, 4),
            "in_sample_rmse_m": round(rmse, 3),
        }
        L.info(f"  cluster {c}: n={int(m.sum()):,} "
               f"depth μ={yc.mean():.2f}m R²={r2:.3f} RMSE={rmse:.2f}m")

    # Whole-model in-sample diagnostics
    full_pred = np.empty_like(y)
    for c, rf in cluster_rfs.items():
        m = clusters == c
        full_pred[m] = rf.predict(Xs[m])
    overall_r2 = float(r2_score(y, full_pred))
    overall_rmse = float(np.sqrt(mean_squared_error(y, full_pred)))
    L.info(f"UAE pretrained: in-sample R²={overall_r2:.4f} "
           f"RMSE={overall_rmse:.3f}m on {len(y):,} pts")

    meta = {
        "version": 1,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_names": list(FEATURE_NAMES),
        "n_clusters": n_clusters,
        "rf_n_estimators": rf_n_estimators,
        "rf_max_depth": rf_max_depth,
        "n_train": int(len(y)),
        "depth_range_m": [float(y.min()), float(y.max())],
        "in_sample_r2": round(overall_r2, 4),
        "in_sample_rmse_m": round(overall_rmse, 3),
        "per_cluster": per_cluster_metrics,
    }
    return UAEModel(scaler=scaler, kmeans=kmeans,
                    cluster_rfs=cluster_rfs, meta=meta)


def save_model(model: UAEModel, path: Path = DEFAULT_MODEL_PATH) -> Path:
    # gzip-compressed pickle — sklearn tree pickles compress ~3-4x (TRAIN-R2
    # pkl bloat concern, coordinator-flagged 49.7 MB baseline).
    import gzip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=6) as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    L.info(f"UAE model saved to {path} ({path.stat().st_size / 1024:.0f} KB, gzip)")
    return path


_CACHED_MODEL: Optional[UAEModel] = None
_CACHED_PATH: Optional[Path] = None


def load_model(path: Path = DEFAULT_MODEL_PATH,
               force_reload: bool = False) -> Optional[UAEModel]:
    """Load the bundled UAE model, caching it across calls."""
    global _CACHED_MODEL, _CACHED_PATH
    path = Path(path)
    if not path.exists():
        return None
    if _CACHED_MODEL is not None and _CACHED_PATH == path and not force_reload:
        return _CACHED_MODEL
    try:
        import gzip
        with open(path, "rb") as fh:
            magic = fh.read(2)
        opener = gzip.open if magic == b"\x1f\x8b" else open
        with opener(path, "rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, UAEModel):
            L.warning(f"UAE model at {path} is not a UAEModel instance")
            return None
        _CACHED_MODEL = model
        _CACHED_PATH = path
        L.info(f"UAE pretrained model loaded ({len(model.cluster_rfs)} clusters, "
               f"trained on {model.meta.get('n_train', 0):,} pts, "
               f"in-sample R²={model.meta.get('in_sample_r2')})")
        return model
    except Exception as ex:
        L.warning(f"UAE model load failed: {ex}")
        return None
