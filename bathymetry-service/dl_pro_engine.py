"""DL Pro — production bathymetry engine.

Loads the deployable bundle from `backend/models/dl_pro/` and offers:

    predict_dl_pro(s2, ref_pts=None, fine_tune=False) -> dict

* `s2`        — the dict returned by `backend.app.fetch_s2`
* `ref_pts`   — optional list of `{lat, lon, depth}` from any source
                (CSV upload / SlideRule ATL03 cache / GEBCO raster sample).
                Used to (a) re-fit the IHO calibration line for THIS scene
                and optionally (b) fine-tune the MLP head a few epochs.
* `fine_tune` — when True and ≥ ~30 ref_pts are given, re-fit the MLP
                head on a warm-started weighted mix of (cached pool baseline,
                user refs).  Cheap because we only resume from saved weights.

The function returns a dict with depth grid, σ grid, 95 % bounds, water mask,
calibration coefficients and per-order IHO S-44 compliance — exactly what
the Flask route + frontend need.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

L = logging.getLogger("dl_pro")

# Optional CPU-thread override for the batched MC-Dropout matmuls. PyTorch
# defaults to the physical core count; on a many-core host, set
# DL_PRO_TORCH_THREADS to use more cores for inference. Unset ⇒ torch's default
# (behaviour unchanged). Threads never affect the output, only throughput.
_torch_threads = os.getenv("DL_PRO_TORCH_THREADS")
if _torch_threads:
    try:
        torch.set_num_threads(max(1, int(_torch_threads)))
        L.info("torch num_threads set to %s", torch.get_num_threads())
    except Exception as _ex:  # pragma: no cover
        L.warning("could not set torch threads: %s", _ex)

ROOT = Path(__file__).resolve().parent
BUNDLE_DIR_V1 = ROOT / "models" / "dl_pro"
BUNDLE_DIR_V2 = ROOT / "models" / "dl_pro_v2"
BUNDLE_DIR_V3 = ROOT / "models" / "dl_pro_v3"
# Default: pick the highest version with a saved model.pt.
if (BUNDLE_DIR_V3 / "model.pt").exists():
    BUNDLE_DIR = BUNDLE_DIR_V3
elif (BUNDLE_DIR_V2 / "model.pt").exists():
    BUNDLE_DIR = BUNDLE_DIR_V2
else:
    BUNDLE_DIR = BUNDLE_DIR_V1

MAX_DEPTH_M = 25.0   # global cap (kept in sync with backend.app)


# ════════════════════════════════════════════════════════════════════════
# Model definition (mirrors experiments/khalifa_may_26.SDBMLPHead)
# ════════════════════════════════════════════════════════════════════════
class SDBMLPHead(nn.Module):
    """v1 architecture (16-feature input)."""
    def __init__(self, in_dim: int = 16, hidden: int = 128, p_drop: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SDBMLPv2(nn.Module):
    """v2 architecture (22-feature, turbidity-aware): wider, +1 layer."""
    def __init__(self, in_dim: int = 22, hidden: int = 192, p_drop: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════
# Feature stacks
# ════════════════════════════════════════════════════════════════════════
def build_features_v1(s2: dict) -> np.ndarray:
    """16-dim feature stack (matches v1 bundle)."""
    from scipy.ndimage import uniform_filter
    eps = 1e-6
    blue    = s2["blue"].astype(np.float64) / 10000.0 + eps
    green   = s2["green"].astype(np.float64) / 10000.0 + eps
    red     = s2["red"].astype(np.float64) / 10000.0 + eps
    coastal = s2.get("coastal", s2["blue"]).astype(np.float64) / 10000.0 + eps
    ndwi    = s2["ndwi"].astype(np.float64)
    X_BG = np.log(blue * 1000.0)    / np.log(green * 1000.0 + eps)
    X_CG = np.log(coastal * 1000.0) / np.log(green * 1000.0 + eps)

    def _ms(a, k=5):
        m = uniform_filter(a, size=k, mode="reflect")
        m2 = uniform_filter(a * a, size=k, mode="reflect")
        v = np.clip(m2 - m * m, 0, None)
        return m.astype(np.float32), np.sqrt(v).astype(np.float32)

    bm, bs = _ms(blue);  gm, gs = _ms(green)
    nm, ns = _ms(ndwi);  xm, xs = _ms(X_BG)

    feats = np.stack([
        X_BG.astype(np.float32), X_CG.astype(np.float32),
        ndwi.astype(np.float32),
        np.log(red / green + eps).astype(np.float32),
        (blue / green).astype(np.float32),
        (coastal / green).astype(np.float32),
        np.log10(blue + green + coastal + eps).astype(np.float32),
        (coastal / blue).astype(np.float32),
        bm, bs, gm, gs, nm, ns, xm, xs,
    ], axis=-1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def build_features_v2(s2: dict, k_clusters: int = 5, turb=None) -> np.ndarray:
    """22-dim feature stack: v1 + Nechad SPM + NDTI + 5 bottom-cluster one-hots.

    `turb` — optional pre-computed `turbidity_features(...)` tuple. The turbidity
    stack (incl. a k-means over every water pixel) is otherwise the single most
    expensive non-MLP step and is also needed later for the σ map, so the caller
    computes it once and threads it through here to avoid a duplicate pass.
    """
    f1 = build_features_v1(s2)
    if turb is None:
        try:
            from backend.turbidity import turbidity_features
        except ImportError:
            from turbidity import turbidity_features   # type: ignore
        turb = turbidity_features(s2, k_clusters=k_clusters)
    spm, ndti_arr, _, oh = turb
    spm = np.log1p(spm)                       # local copy; leaves `turb` raw for reuse
    feats = np.concatenate(
        [f1, spm[..., None], ndti_arr[..., None], oh],
        axis=-1).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# Public alias — picks the right stack from the loaded bundle's in_dim.
def build_features(s2: dict, turb=None) -> np.ndarray:
    bundle = load_bundle()
    if bundle["meta"].get("in_dim", 16) >= 22:
        return build_features_v2(s2, turb=turb)
    return build_features_v1(s2)


# ════════════════════════════════════════════════════════════════════════
# Bundle loader (cached) — thread-safe
# ════════════════════════════════════════════════════════════════════════
_BUNDLE_LOCK = threading.Lock()
_BUNDLE: Optional[Dict] = None


def load_bundle() -> Dict:
    global _BUNDLE
    with _BUNDLE_LOCK:
        if _BUNDLE is not None:
            return _BUNDLE
        meta_p = BUNDLE_DIR / "meta.json"
        model_p = BUNDLE_DIR / "model.pt"
        if not (meta_p.exists() and model_p.exists()):
            raise FileNotFoundError(
                f"DL Pro bundle missing — expected {meta_p} and {model_p}. "
                "Run experiments/train_dl_pro_bundle.py first.")
        meta = json.loads(meta_p.read_text())
        ckpt = torch.load(model_p, map_location="cpu", weights_only=False)
        arch = ckpt.get("arch", "SDBMLPHead")
        if arch == "SDBMLPv2" or ckpt.get("in_dim", 16) >= 22:
            model = SDBMLPv2(in_dim=ckpt.get("in_dim", meta["in_dim"]),
                              hidden=ckpt.get("hidden", 192),
                              p_drop=ckpt.get("p_drop", 0.25))
        else:
            model = SDBMLPHead(in_dim=ckpt.get("in_dim", meta["in_dim"]),
                                hidden=ckpt.get("hidden", 128),
                                p_drop=ckpt.get("p_drop", 0.20))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _BUNDLE = {"meta": meta, "model": model,
                    "feat_mean": np.array(meta["feat_mean"], dtype=np.float32),
                    "feat_std":  np.array(meta["feat_std"],  dtype=np.float32),
                    "alpha": float(meta["iho_alpha"]),
                    "beta":  float(meta["iho_beta"])}
        L.info(f"DL Pro bundle loaded — IHO α={_BUNDLE['alpha']:+.3f}, "
                f"β={_BUNDLE['beta']:.3f}, "
                f"hold-out RMSE={meta['holdout_metrics_calibrated']['rmse']:.2f} m")
        return _BUNDLE


# ════════════════════════════════════════════════════════════════════════
# Inference with MC-Dropout uncertainty
# ════════════════════════════════════════════════════════════════════════
# Pixels processed per forward-pass batch. Peak inference RAM scales with this,
# NOT with scene size, so large AOIs no longer OOM the worker. Tunable per
# deployment via env (lower it on tight-memory hosts).
_BATCH_PIXELS = max(1, int(os.getenv("DL_PRO_BATCH_PIXELS", "100000")))

# MC-Dropout stochastic passes for the uncertainty estimate. Default 30 (the
# original value — output is unchanged unless overridden). Lower it via env to
# trade a little σ-map noise for proportionally faster inference on huge AOIs.
_N_MC = max(1, int(os.getenv("DL_PRO_N_MC", "30")))


def _predict(model: SDBMLPHead, X: np.ndarray, mu: np.ndarray, sd: np.ndarray,
             n_mc: int = 30, batch_size: int = _BATCH_PIXELS
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_pred, sigma_pred) over n_mc stochastic forward passes.
    `X` is already feature-normalised by the caller.

    Runs in row-batches so peak memory stays bounded regardless of scene size:
    each hidden layer only ever materialises (batch_size, hidden) activations
    instead of (N, hidden). Statistically equivalent to the un-batched version:
    per-pixel mean/σ are still MC estimates over n_mc stochastic passes (each
    batch just draws its own dropout masks — the estimator distribution is the
    same, results are not bit-for-bit reproducible either way)."""
    # Keep dropout ON, linear OFF (MC-Dropout).
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        elif isinstance(m, nn.Linear):
            m.eval()

    N = len(X)
    mean_out = np.empty(N, dtype=np.float32)
    std_out = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            # Row-slice of a C-contiguous array is a view (no copy); from_numpy
            # shares its buffer, so only the model activations cost memory.
            Xb = torch.from_numpy(X[i:i + batch_size])
            samples = np.empty((n_mc, Xb.shape[0]), dtype=np.float32)
            for s in range(n_mc):
                samples[s] = model(Xb).numpy()
            mean_out[i:i + batch_size] = samples.mean(axis=0)
            std_out[i:i + batch_size] = samples.std(axis=0)
    return mean_out, std_out


def _refit_calibration(pred_at_refs: np.ndarray, ref_depths: np.ndarray
                        ) -> Tuple[float, float, float]:
    """Refit IHO calibration line `measured = α + β · pred` on the supplied
    references.  Robust against tiny/empty inputs by falling back to (0, 1)."""
    valid = np.isfinite(pred_at_refs) & np.isfinite(ref_depths)
    if valid.sum() < 5:
        return 0.0, 1.0, 0.0
    p = pred_at_refs[valid]; t = ref_depths[valid]
    try:
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(p.reshape(-1, 1), t)
        a, b = float(lr.intercept_), float(lr.coef_[0])
        r2 = float(lr.score(p.reshape(-1, 1), t))
    except Exception:
        # Closed-form OLS
        A = np.vstack([np.ones_like(p), p]).T
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        pred_t = a + b * p
        r2 = float(1 - np.sum((pred_t - t) ** 2) / max(ss_tot, 1e-9))
    return a, b, r2


def _fine_tune(model: SDBMLPHead, mu: np.ndarray, sd: np.ndarray,
                X_user: np.ndarray, y_user: np.ndarray, *,
                n_epochs: int = 200, lr: float = 5e-4) -> Dict:
    """Warm-started fine-tune of the MLP head on user references.

    Conservative: small LR, few epochs, weight regularisation.  Returns a
    dict with the inner train-RMSE before/after so the caller can decide.
    """
    Xt = (torch.from_numpy(X_user) - torch.tensor(mu)) / torch.tensor(sd)
    yt = torch.from_numpy(y_user)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    with torch.no_grad():
        rmse0 = float(((model(Xt) - yt) ** 2).mean().sqrt())

    for _ in range(n_epochs):
        opt.zero_grad()
        pred = model(Xt)
        diff = pred - yt
        absd = diff.abs()
        delta = 1.0
        huber = torch.where(absd <= delta, 0.5 * diff * diff,
                              delta * (absd - 0.5 * delta))
        loss = huber.mean()
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        rmse1 = float(((model(Xt) - yt) ** 2).mean().sqrt())
    return {"rmse_before": rmse0, "rmse_after": rmse1, "n_epochs": n_epochs,
             "n_user": int(len(y_user))}


def _refs_to_pixels(ref_pts: List[dict], bbox, H: int, W: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map ref_pts to (rows, cols, depths) inside the AOI.  Pixel-median
    aggregation is applied — multiple refs in the same cell collapse to
    the median of that cell."""
    w, s, e, n = bbox
    rows, cols, deps = [], [], []
    for p in ref_pts:
        try:
            la = float(p["lat"]); lo = float(p["lon"]); d = abs(float(p["depth"]))
        except Exception:
            continue
        if not (s <= la <= n and w <= lo <= e):
            continue
        if not (0.3 <= d <= MAX_DEPTH_M):
            continue
        r = int(round((n - la) / (n - s) * (H - 1)))
        c = int(round((lo - w) / (e - w) * (W - 1)))
        if 0 <= r < H and 0 <= c < W:
            rows.append(r); cols.append(c); deps.append(d)
    if not rows:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32),
                 np.zeros(0, np.float32))
    import pandas as pd
    df = pd.DataFrame({"r": rows, "c": cols, "d": deps})
    g = df.groupby(["r", "c"])["d"].median().reset_index()
    return (g["r"].values.astype(np.int32),
            g["c"].values.astype(np.int32),
            g["d"].values.astype(np.float32))


def _iho_compliance(err_abs: np.ndarray, depth: np.ndarray) -> Dict[str, float]:
    out = {}
    for ord_, (a, b) in {
        "Special": (0.25, 0.0075),
        "Order_1a": (0.50, 0.013),
        "Order_1b": (0.50, 0.013),
        "Order_2":  (1.00, 0.023),
    }.items():
        tvu = np.sqrt(a * a + (b * depth) ** 2)
        out[ord_] = float(100.0 * np.mean(err_abs <= tvu))
    return out


# ════════════════════════════════════════════════════════════════════════
# Public entrypoint
# ════════════════════════════════════════════════════════════════════════
def predict_dl_pro(s2: dict,
                   bbox: Optional[List[float]] = None,
                   ref_pts: Optional[List[dict]] = None,
                   *,
                   fine_tune: bool = False,
                   n_mc: int = _N_MC) -> Dict:
    """Run DL Pro on `s2`. Optional `ref_pts` re-calibrate / fine-tune.

    Returns:
        dict with keys depth, sigma, lower95, upper95, water_mask,
        calibration {alpha, beta, r2, n_refs}, fine_tune {…} or None,
        iho_s44 {Order_1a: %, …} when refs given (else baseline values).
    """
    bundle = load_bundle()
    meta = bundle["meta"]
    base_alpha = bundle["alpha"]; base_beta = bundle["beta"]
    feat_mean  = bundle["feat_mean"]; feat_std = bundle["feat_std"]

    # We always work on a private model copy so concurrent retrains don't
    # poison the cached baseline. Pick the right architecture from meta/in_dim.
    if isinstance(bundle["model"], SDBMLPv2) or meta["in_dim"] >= 22:
        model = SDBMLPv2(in_dim=meta["in_dim"], hidden=192, p_drop=0.25)
    else:
        model = SDBMLPHead(in_dim=meta["in_dim"], hidden=128, p_drop=0.20)
    model.load_state_dict(bundle["model"].state_dict())

    H, W = s2["red"].shape

    # Bottom-type / turbidity stack (Nechad SPM + NDTI + k-means one-hots) is the
    # single priciest non-MLP step and is ALSO needed by the σ-map further down —
    # compute it exactly once here and thread it through both consumers. Fixed
    # random_state ⇒ this is bit-identical to the previous double computation.
    try:
        try:
            from backend.turbidity import turbidity_features
        except ImportError:
            from turbidity import turbidity_features   # type: ignore
        turb = turbidity_features(s2)
    except Exception as ex:
        L.warning(f"turbidity features unavailable ({ex}); continuing")
        turb = None

    feats = build_features(s2, turb=turb)
    F = feats.shape[-1]
    if F != meta["in_dim"]:
        raise RuntimeError(f"Feature dim mismatch — bundle expects "
                            f"{meta['in_dim']}, got {F}")

    # Optional fine-tune on user refs ─────────────────────────────────
    ft_info = None
    if ref_pts and fine_tune and bbox is not None:
        rr, cc, ydep = _refs_to_pixels(ref_pts, bbox, H, W)
        if len(rr) >= 30:
            X_user = feats[rr, cc, :]
            ft_info = _fine_tune(model, feat_mean, feat_std, X_user, ydep)
            L.info(f"DL Pro fine-tune: rmse {ft_info['rmse_before']:.2f}m → "
                    f"{ft_info['rmse_after']:.2f}m on {ft_info['n_user']} pts")

    # Inference w/ MC-Dropout uncertainty — WATER PIXELS ONLY ─────────
    # Every non-water pixel is set to NaN below regardless (land/cloud carry no
    # depth), so running the expensive n_mc× MLP over them is pure waste. We run
    # it only on the water mask and scatter the results back — output is
    # identical to the full-grid version, but cost scales with the water area,
    # not the whole (often mostly-land) scene.
    wm = s2.get("water_mask")
    wm = np.ones((H, W), dtype=bool) if wm is None else np.asarray(wm, dtype=bool)
    wm_flat = wm.reshape(-1)

    flat = feats.reshape(-1, F)
    std_safe = np.where(feat_std < 1e-3, 1e-3, feat_std)
    pred_mean = np.full(H * W, np.nan, dtype=np.float32)
    pred_std = np.full(H * W, np.nan, dtype=np.float32)
    if int(wm_flat.sum()) > 0:
        Xw = np.ascontiguousarray(
            ((flat[wm_flat] - feat_mean) / std_safe).astype(np.float32))
        mw, sw = _predict(model, Xw, feat_mean, feat_std, n_mc=n_mc)
        pred_mean[wm_flat] = mw
        pred_std[wm_flat] = sw
    depth_pred = pred_mean.reshape(H, W)
    sigma      = pred_std.reshape(H, W)

    # Calibration: refit on user refs if we have them, else use baseline.
    cal = {"alpha": base_alpha, "beta": base_beta, "r2": meta.get("iho_R2", 0.0),
            "n_refs": 0, "source": "baseline"}
    iho_compliance = None
    if ref_pts and bbox is not None:
        rr, cc, ydep = _refs_to_pixels(ref_pts, bbox, H, W)
        if len(rr) >= 5:
            p_at_refs = depth_pred[rr, cc]
            a_, b_, r2_ = _refit_calibration(p_at_refs, ydep)
            cal = {"alpha": a_, "beta": b_, "r2": r2_,
                    "n_refs": int(len(rr)), "source": "user_refs"}
            depth_cal = a_ + b_ * depth_pred
            err = np.abs(depth_cal[rr, cc] - ydep)
            iho_compliance = _iho_compliance(err, ydep)

    # Apply calibration
    depth_out = cal["alpha"] + cal["beta"] * depth_pred
    depth_out = np.clip(depth_out, 0.0, MAX_DEPTH_M).astype(np.float32)
    sigma_out = np.clip(sigma + 0.10, 0.10, 10.0).astype(np.float32)
    lower = np.clip(depth_out - 1.96 * sigma_out, 0, MAX_DEPTH_M).astype(np.float32)
    upper = np.clip(depth_out + 1.96 * sigma_out, 0, MAX_DEPTH_M).astype(np.float32)

    # ── Turbidity / probability-of-accuracy maps ─────────────────────
    try:
        try:
            from backend.turbidity import sigma_total, prob_within_iho
        except ImportError:
            from turbidity import sigma_total, prob_within_iho    # type: ignore
        # Reuse the turbidity stack computed once above (no second k-means pass).
        if turb is not None:
            spm, ndti_arr, _, _ = turb
        else:
            try:
                from backend.turbidity import turbidity_features
            except ImportError:
                from turbidity import turbidity_features   # type: ignore
            spm, ndti_arr, _, _ = turbidity_features(s2)
        k_T = float(meta.get("k_T", 0.012))
        k_d = float(meta.get("k_d", 0.04))
        kd_l = float(meta.get("kd_limit", 12.0))
        sigma_eff = sigma_total(sigma_out, spm, depth_out,
                                 k_T=k_T, k_d=k_d, kd_limit=kd_l)
        p_iho_1a  = prob_within_iho(depth_out, sigma_eff, "Order_1a")
        p_iho_1b  = prob_within_iho(depth_out, sigma_eff, "Order_1b")
        p_iho_2a  = prob_within_iho(depth_out, sigma_eff, "Order_2a")
        p_iho_2b  = prob_within_iho(depth_out, sigma_eff, "Order_2b")
        p_iho_2   = p_iho_2a    # alias for back-compat
        p_special = prob_within_iho(depth_out, sigma_eff, "Special")
    except Exception as ex:
        L.warning(f"turbidity / prob-accuracy failed: {ex}")
        spm = ndti_arr = sigma_eff = None
        p_iho_1a = p_iho_1b = p_iho_2a = p_iho_2b = p_iho_2 = p_special = None

    # `wm` (water mask) was already resolved above for the water-only inference.
    for arr in (depth_out, sigma_out, lower, upper, depth_pred):
        arr[~wm] = np.nan
    if sigma_eff is not None:
        sigma_eff = np.where(wm, sigma_eff, np.nan).astype(np.float32)
        for p in (p_iho_1a, p_iho_1b, p_iho_2a, p_iho_2b, p_iho_2, p_special):
            p[~wm] = np.nan
        spm = np.where(wm, spm, np.nan).astype(np.float32)
        ndti_arr = np.where(wm, ndti_arr, np.nan).astype(np.float32)

    return {
        "depth":      depth_out,
        "sigma":      sigma_out,
        "sigma_eff":  sigma_eff,
        "lower95":    lower,
        "upper95":    upper,
        "depth_uncalibrated": depth_pred,
        "turbidity":  spm,
        "ndti":       ndti_arr,
        "p_iho_special": p_special,
        "p_iho_1a":   p_iho_1a,
        "p_iho_1b":   p_iho_1b,
        "p_iho_2a":   p_iho_2a,
        "p_iho_2b":   p_iho_2b,
        "p_iho_2":    p_iho_2,
        "water_mask": wm,
        "calibration": cal,
        "fine_tune":   ft_info,
        "iho_s44":     iho_compliance if iho_compliance is not None
                        else {k: v["pct_within_TVU"] for k, v in
                              meta["iho_s44"].items()},
        "model_meta":  {
            "version": meta.get("version", "1.0"),
            "trained_at": meta.get("trained_at"),
            "training_n": meta["training"].get("n_train"),
            "holdout_metrics": meta["holdout_metrics_calibrated"],
            "k_T": meta.get("k_T", 0.012),
            "k_d": meta.get("k_d", 0.04),
            "kd_limit": meta.get("kd_limit", 12.0),
        },
    }
