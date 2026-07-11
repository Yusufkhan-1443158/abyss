"""
Inference helper — load a saved BP-NN checkpoint (shallow or deep) and
run it over a fresh Sentinel-2 scene without retraining.

Public API
----------
    list_models()                           -> list[dict]
    load_model(weights_path)                -> (torch.nn.Module, manifest_dict)
    predict(model, manifest, s2, bbox)      -> (depth_grid, info)
    predict_from_path(weights_path, s2, bbox) -> (depth_grid, info)
"""
from __future__ import annotations

import json, os, time, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

L = logging.getLogger("bathymetry.bp_inference")

MODEL_DIR = Path(__file__).parent / "ocean" / "models"
MAX_DEPTH_M = 25.0


def nearest_model(bbox: List[float], prefer_deep: bool = True) -> Optional[Dict[str, Any]]:
    """
    Pick the best saved BP-NN checkpoint for a given ROI.

    Strategy:
      1. Prefer deep models (architecture key contains "Deep").
      2. Of those, prefer models whose manifest bbox *contains* or overlaps the
         requested bbox. Otherwise, pick the one whose centroid is closest.
      3. If no deep model exists, fall back to any available checkpoint with
         a manifest and R² >= 0.3 (meaningful training).
      4. Returns the full model record (same shape as list_models entries).
    """
    models = list_models()
    if not models:
        return None

    def _bbox_centre(b):
        return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))

    def _bbox_contains(outer, inner):
        # outer = [W, S, E, N], inner = same
        return (outer[0] <= inner[0] and outer[1] <= inner[1]
                and outer[2] >= inner[2] and outer[3] >= inner[3])

    def _distance_deg(a, b):
        ax, ay = _bbox_centre(a); bx, by = _bbox_centre(b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    roi_centre = _bbox_centre(bbox)
    candidates = []
    for m in models:
        mf = m.get("manifest") or {}
        bb = mf.get("bbox")
        if not bb or len(bb) != 4:
            continue
        is_deep = "deep" in (mf.get("architecture") or "").lower() or mf.get("deep", False) \
                  or m.get("filename", "").startswith("bp_deep_")
        met = mf.get("metrics") or {}
        r2 = met.get("r2")
        if r2 is None:
            continue
        if r2 < 0.3:
            continue        # skip models that didn't learn anything useful
        contains = _bbox_contains(bb, bbox) or _bbox_contains(bbox, bb)
        d = _distance_deg(bb, bbox)
        candidates.append({
            "record":  m,
            "is_deep": is_deep,
            "r2":      r2,
            "contains": contains,
            "distance_deg": d,
        })

    if not candidates:
        return None
    # Sort: deep first (if preferred), then overlapping, then highest R², then closest
    candidates.sort(key=lambda c: (
        -int(prefer_deep and c["is_deep"]),
        -int(c["contains"]),
        -c["r2"],
        c["distance_deg"],
    ))
    pick = candidates[0]
    rec = pick["record"]
    rec["_selection"] = {
        "is_deep":      pick["is_deep"],
        "contains_roi": pick["contains"],
        "distance_deg": round(pick["distance_deg"], 3),
        "r2":           pick["r2"],
    }
    return rec


def list_models() -> List[Dict[str, Any]]:
    """Discover all trained BP-NN checkpoints under backend/ocean/models/.
    Returns one dict per model with manifest + stats."""
    out = []
    if not MODEL_DIR.exists():
        return out
    for pt in sorted(MODEL_DIR.glob("bp_*.pt")):
        rec = {"weights_file": str(pt), "filename": pt.name}
        mf = pt.with_suffix(".manifest.json")
        if mf.exists():
            try:
                with open(mf) as f:
                    rec["manifest"] = json.load(f)
            except Exception as ex:
                rec["manifest_error"] = str(ex)
        else:
            # No manifest → try reading basic fields from the .pt itself
            try:
                import torch
                ck = torch.load(str(pt), map_location="cpu", weights_only=False)
                if isinstance(ck, dict):
                    rec["architecture"] = ck.get("architecture")
                    rec["deep"]         = bool(ck.get("deep", False))
                    rec["epochs_used"]  = ck.get("epochs_used")
                    rec["saved_at"]     = ck.get("saved_at")
            except Exception as ex:
                rec["load_error"] = str(ex)
        try:
            rec["size_kb"] = round(pt.stat().st_size / 1024, 1)
        except Exception:
            pass
        out.append(rec)
    return out


def _build_features(s2):
    """Identical to bp_network._build_feature_stack — duplicated to avoid
    a circular import."""
    eps = 1e-6
    blue  = np.clip(s2["blue"].astype(np.float64)  / 10000.0, eps, None)
    green = np.clip(s2["green"].astype(np.float64) / 10000.0, eps, None)
    red   = np.clip(s2["red"].astype(np.float64)   / 10000.0, eps, None)
    nir   = np.clip(s2["nir"].astype(np.float64)   / 10000.0, eps, None)
    water = s2.get("water_mask", s2.get("ndwi", None))
    if water is None:
        water = green - nir > 0
    ln_bg = np.log(1000.0 * blue) / (np.log(1000.0 * green) + eps)
    ln_br = np.log(1000.0 * blue) / (np.log(1000.0 * red)   + eps)
    feats = np.stack([blue, green, red, nir, ln_bg, ln_br], axis=-1)
    return feats.astype(np.float32), water.astype(bool)


def load_model(weights_path: str):
    """Reconstruct the right nn.Module (shallow or deep) from the checkpoint
    and load its weights. Returns (model_in_eval_mode, manifest_dict)."""
    import torch
    import torch.nn as nn
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise RuntimeError(f"{weights_path}: not a BP-NN checkpoint (missing state_dict)")

    F = len(ck.get("feature_names", [])) or 6
    deep = bool(ck.get("deep", False))

    if deep:
        class DeepBPNet(nn.Module):
            def __init__(self, fin, p_drop=0.10):
                super().__init__()
                dims = [fin, 128, 64, 32, 16, 8, 1]
                layers = []
                for i in range(len(dims) - 1):
                    layers.append(nn.Linear(dims[i], dims[i + 1]))
                    if i < len(dims) - 2:
                        layers.append(nn.BatchNorm1d(dims[i + 1]))
                        layers.append(nn.ReLU(inplace=True))
                        layers.append(nn.Dropout(p_drop))
                self.net = nn.Sequential(*layers)
                self.final_act = nn.Sigmoid()
            def forward(self, x):
                return self.final_act(self.net(x)).squeeze(-1)
        net = DeepBPNet(F)
    else:
        h1, h2 = ck.get("hidden", [32, 16])[:2]
        class BPNet(nn.Module):
            def __init__(self, fin, h1, h2):
                super().__init__()
                self.fc1 = nn.Linear(fin, h1)
                self.fc2 = nn.Linear(h1, h2)
                self.out = nn.Linear(h2, 1)
                self.act = nn.Sigmoid()
            def forward(self, x):
                x = self.act(self.fc1(x))
                x = self.act(self.fc2(x))
                return self.out(x).squeeze(-1)
        net = BPNet(F, h1, h2)

    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def predict(net, manifest: Dict[str, Any], s2, bbox) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run inference over the S2 scene using the pre-fit feature normalisation
    stats stored in the manifest. Returns depth_grid (m) + info dict."""
    import torch
    feats, water = _build_features(s2)
    H, W, F = feats.shape
    fmin = np.asarray(manifest.get("feature_min"), dtype=np.float32)
    fmax = np.asarray(manifest.get("feature_max"), dtype=np.float32)
    if fmin is None or fmax is None or fmin.shape != (F,):
        raise RuntimeError("manifest missing feature_min/feature_max")
    span = np.maximum(fmax - fmin, 1e-6)
    flat = feats.reshape(-1, F)
    flat_n = np.clip((flat - fmin) / span, 0.0, 1.0).astype(np.float32)

    with torch.no_grad():
        out = net(torch.from_numpy(flat_n)).cpu().numpy()
    scale = float(manifest.get("target_scale_m", MAX_DEPTH_M))
    depth = (out.reshape(H, W) * scale).astype(np.float32)
    depth = np.clip(depth, 0.0, MAX_DEPTH_M)
    depth[~water] = np.nan

    return depth, {
        "method": "BP-NN inference (no retraining)",
        "architecture": manifest.get("architecture"),
        "deep":         bool(manifest.get("deep", False)),
        "epochs_used":  manifest.get("epochs_used"),
        "val_loss":     manifest.get("val_loss"),
        "grid_shape":   [int(H), int(W)],
        "water_px":     int(water.sum()),
        "feature_names": manifest.get("feature_names"),
    }


def predict_from_path(weights_path: str, s2, bbox):
    net, ck = load_model(weights_path)
    return predict(net, ck, s2, bbox)
