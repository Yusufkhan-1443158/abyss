"""
Cluster-Based Regression (CBR) bathymetry — Geyman & Maloof 2019.

Reference
---------
Geyman, E. C., & Maloof, A. C. (2019).
    A Simple Method for Extracting Water Depth from Multispectral Satellite
    Imagery in Regions of Variable Bottom Type.
    Earth and Space Science, 6, 527-537. doi:10.1029/2018EA000539

Algorithm
---------
 1. Build deglinted water-leaving reflectance Rw in blue/green/(coastal).
 2. Form Stumpf (2003) log-ratios   X_BG = ln(π·Rw_blue) / ln(π·Rw_green)
                                    X_CG = ln(π·Rw_coastal) / ln(π·Rw_green)
    These are monotonic in depth but the slope varies with bottom type.
 3. Segment water pixels into K spectral classes via K-means on a small
    reflectance feature vector.  Each class is effectively a bottom type
    (sand, seagrass, rubble, carbonate mud …).
 4. For every class separately, fit a robust linear regression of depth on
    (X_BG, X_CG) using the reference depths that fall in that class.
 5. Predict depth pixel-wise with the per-class model.  Do a soft blend
    between the two nearest cluster centroids (in feature space) so the
    cluster boundaries do not leave visible seams.
 6. Per-class residual std becomes the per-pixel uncertainty σ_z.

The module returns everything the Flask layer needs: depth grid, σ grid,
cluster map, per-class coefficients, fit metrics — with no side effects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

L = logging.getLogger("cbr")

# ────────────────────────────────────────────────────────────────────────
# Globals — kept in sync with backend.app.MAX_DEPTH_M
# ────────────────────────────────────────────────────────────────────────
MAX_DEPTH_M = 25.0            # SDB reliable limit (Stumpf 2003)
MIN_DEPTH_M = 0.3             # below this we cannot resolve signal
DEFAULT_N_CLUSTERS = 5        # Geyman used 5-8; 5 is a good default
STUMPF_GAIN = 1000.0          # π·n scaling in the Stumpf ln-ratio

# Physical Rw floor.  fetch_s2 returns deep-water-subtracted reflectance
# quantised to uint16 scaled by 1e-4, so values below ~1e-4 collapse to 0.
# We use a tiny absolute floor to discard pure deep water, and on top of
# that an *adaptive* floor derived from the references (see run_cbr).
MIN_RW_HARD = 5e-4

_EPS = 1e-6


# ════════════════════════════════════════════════════════════════════════
# Result container
# ════════════════════════════════════════════════════════════════════════
@dataclass
class CBRResult:
    """Everything the caller needs to ship a CBR run into the rest of
    the app.  Numeric arrays are NaN outside the water mask."""
    depth:       np.ndarray           # (H, W) float32 — NaN on land
    uncertainty: np.ndarray           # (H, W) float32 — per-pixel σ_z
    cluster_map: np.ndarray           # (H, W) int8    — K_i ∈ [0,K), -1 land
    water_mask:  np.ndarray           # (H, W) bool
    n_clusters:  int
    per_class:   list                 # dicts: {id,n_train,coef,rmse,mae,r2,bias}
    metrics:     dict = field(default_factory=dict)
    bbox:        Optional[list] = None

    def as_dict(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "per_class":  self.per_class,
            "metrics":    self.metrics,
            "bbox":       self.bbox,
            "depth_shape": list(self.depth.shape),
        }


# ════════════════════════════════════════════════════════════════════════
# Stumpf log-ratio helper
# ════════════════════════════════════════════════════════════════════════
def _log_ratio(rw_num: np.ndarray, rw_den: np.ndarray,
               gain: float = STUMPF_GAIN) -> np.ndarray:
    """Stumpf (2003) log-ratio.  Monotonic in depth: deeper pixels → larger."""
    num = np.log(np.clip(rw_num, _EPS, None) * gain)
    den = np.log(np.clip(rw_den, _EPS, None) * gain)
    # denominators near zero (Rw ≈ 1/gain) give near-zero log — guard with EPS
    return num / np.where(np.abs(den) < _EPS, _EPS, den)


def _build_features(s2: dict) -> tuple[np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    """From the backend.app.fetch_s2 dict, recover:

        Rw_blue, Rw_green, Rw_red, Rw_coastal — 2-D float64 water-leaving
        reflectance (already deglinted + deep-water subtracted by fetch_s2).

    Returns
    -------
    X_BG, X_CG : Stumpf log-ratios used in the per-cluster regression.
    Rw_b, Rw_g, Rw_r, Rw_c : raw water-leaving reflectance (for clustering).
    """
    blue    = s2["blue"].astype(np.float64)    / 10000.0
    green   = s2["green"].astype(np.float64)   / 10000.0
    red     = s2.get("red", s2["green"]).astype(np.float64)   / 10000.0
    coastal = s2.get("coastal", s2["blue"]).astype(np.float64) / 10000.0
    blue    = np.clip(blue,    _EPS, None)
    green   = np.clip(green,   _EPS, None)
    red     = np.clip(red,     _EPS, None)
    coastal = np.clip(coastal, _EPS, None)
    X_BG = _log_ratio(blue,    green)
    X_CG = _log_ratio(coastal, green)
    return X_BG, X_CG, blue, green, red, coastal


# ════════════════════════════════════════════════════════════════════════
# Reference point rasterisation
# ════════════════════════════════════════════════════════════════════════
def _refs_to_pixels(ref_pts: list, bbox: list, H: int, W: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map (lat, lon, depth) references to (row, col, depth) indices.
    Points outside the ROI are dropped silently."""
    w, s, e, n = bbox
    rows, cols, depths = [], [], []
    for p in ref_pts:
        try:
            la = float(p["lat"]); lo = float(p["lon"]); d = abs(float(p["depth"]))
        except Exception:
            continue
        if not (s <= la <= n and w <= lo <= e):
            continue
        if not (MIN_DEPTH_M <= d <= MAX_DEPTH_M):
            continue
        r = int(round((n - la) / (n - s) * (H - 1)))
        c = int(round((lo - w) / (e - w) * (W - 1)))
        if 0 <= r < H and 0 <= c < W:
            rows.append(r); cols.append(c); depths.append(d)
    return (np.asarray(rows, dtype=np.int32),
            np.asarray(cols, dtype=np.int32),
            np.asarray(depths, dtype=np.float64))


# ════════════════════════════════════════════════════════════════════════
# K-means with graceful fallback (sklearn → numpy Lloyd)
# ════════════════════════════════════════════════════════════════════════
def _kmeans_fit(feats: np.ndarray, k: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, centroids).  Uses sklearn if available, else a
    10-iteration Lloyd with k-means++ seeding."""
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(feats)
        return labels.astype(np.int32), km.cluster_centers_.astype(np.float64)
    except ImportError:
        rng = np.random.default_rng(seed)
        n = feats.shape[0]
        idx0 = rng.integers(0, n)
        centers = [feats[idx0]]
        for _ in range(1, k):
            d2 = np.min(np.sum((feats[:, None, :] - np.stack(centers)[None, :, :]) ** 2, axis=2), axis=1)
            probs = d2 / (d2.sum() + _EPS)
            centers.append(feats[rng.choice(n, p=probs)])
        C = np.stack(centers)
        for _ in range(15):
            d2 = np.sum((feats[:, None, :] - C[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            for ki in range(k):
                m = (labels == ki)
                if m.any():
                    C[ki] = feats[m].mean(axis=0)
        return labels.astype(np.int32), C


# ════════════════════════════════════════════════════════════════════════
# Feature expansion: polynomial terms decouple shallow & deep regimes
# where a plain linear log-ratio fit saturates.
# ════════════════════════════════════════════════════════════════════════
def _design(X_bg: np.ndarray, X_cg: np.ndarray, mode: str = "poly") -> np.ndarray:
    """Build regression design matrix from Stumpf log-ratios.

    mode='linear'  → [X_BG, X_CG]                        (Geyman 2019)
    mode='poly'    → [X_BG, X_CG, X_BG², X_CG², X_BG·X_CG]   (default)
    The intercept column is appended by the caller.
    """
    if mode == "linear":
        return np.column_stack([X_bg, X_cg])
    return np.column_stack([X_bg, X_cg, X_bg * X_bg, X_cg * X_cg, X_bg * X_cg])


# ════════════════════════════════════════════════════════════════════════
# Robust per-cluster regression with optional sample weighting
# ════════════════════════════════════════════════════════════════════════
def _fit_one_class(Xc: np.ndarray, zc: np.ndarray,
                   sample_weight: np.ndarray = None
                   ) -> tuple[np.ndarray, float, float, float, float]:
    """Fit depth = [Xc | 1] @ β on a single cluster's training points.
    Xc is the precomputed design matrix (without intercept). Returns
    (β, rmse, mae, r², bias). Uses HuberRegressor with sample_weight when
    provided, else OLS with IQR outlier pruning.

    The weighted residuals in the returned metrics reflect the weighting;
    callers that want unweighted RMSE should recompute pred - zc themselves.
    """
    n_feat = Xc.shape[1]
    A = np.column_stack([Xc, np.ones(len(zc))])
    try:
        from sklearn.linear_model import HuberRegressor
        # alpha scales with feature count to stay regularised as poly terms grow
        hub = HuberRegressor(epsilon=1.35, max_iter=400, alpha=1e-4 * n_feat)
        if sample_weight is not None:
            hub.fit(Xc, zc, sample_weight=sample_weight)
        else:
            hub.fit(Xc, zc)
        coef = np.concatenate([hub.coef_, [hub.intercept_]])
    except Exception:
        # Fallback: weighted least-squares with IQR-based outlier prune
        w = (sample_weight if sample_weight is not None
             else np.ones(len(zc)))
        Aw = A * np.sqrt(w)[:, None]; zw = zc * np.sqrt(w)
        coef0 = np.linalg.lstsq(Aw, zw, rcond=None)[0]
        res0 = zc - A @ coef0
        q1, q3 = np.percentile(res0, [15, 85])
        keep = (res0 >= q1 - 1.5 * (q3 - q1)) & (res0 <= q3 + 1.5 * (q3 - q1))
        if keep.sum() >= max(4, n_feat + 1):
            Ak = A[keep] * np.sqrt(w[keep])[:, None]
            zk = zc[keep] * np.sqrt(w[keep])
            coef = np.linalg.lstsq(Ak, zk, rcond=None)[0]
        else:
            coef = coef0
    pred = A @ coef
    err  = pred - zc
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((zc - zc.mean()) ** 2))
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else 0.0
    return coef, rmse, mae, r2, bias


# ════════════════════════════════════════════════════════════════════════
# Gaussian-IDW residual post-correction — absorbs spatial bias that the
# per-class model cannot: tide offsets, local turbidity, bottom-type fine
# structure. Kernel width ≈ median inter-ref spacing × 2.5 gives good
# smoothing without collapsing to individual photons.
# ════════════════════════════════════════════════════════════════════════
def _idw_residual_field(rr: np.ndarray, cc: np.ndarray, resid: np.ndarray,
                        H: int, W: int, mask: np.ndarray,
                        bandwidth_px: float = None,
                        n_neighbours: int = 24) -> np.ndarray:
    """Return an (H, W) float32 residual grid built by Gaussian IDW over
    the supplied (row, col, residual) training points. Pixels outside
    ``mask`` are left at 0 so callers can simply add the field back."""
    out = np.zeros((H, W), dtype=np.float32)
    if len(resid) < 4:
        return out
    if bandwidth_px is None:
        # Median pairwise distance among refs × 2.5, floor of 8 px.
        sub = np.random.default_rng(3).choice(len(resid),
                                              size=min(200, len(resid)), replace=False)
        dr = rr[sub][:, None] - rr[sub][None, :]
        dc = cc[sub][:, None] - cc[sub][None, :]
        d = np.sqrt(dr * dr + dc * dc).astype(np.float32)
        np.fill_diagonal(d, np.inf)
        bandwidth_px = float(max(8.0, 2.5 * np.median(d[np.isfinite(d)])))
    b2 = bandwidth_px * bandwidth_px
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return out
    # Vectorise in chunks to keep memory bounded.
    chunk = 20000
    for s in range(0, len(ys), chunk):
        yc = ys[s:s + chunk]; xc = xs[s:s + chunk]
        dy = yc[:, None] - rr[None, :]
        dx = xc[:, None] - cc[None, :]
        d2 = (dy * dy + dx * dx).astype(np.float32)
        # Gaussian kernel — keep only the n nearest refs per pixel for sharpness.
        if len(resid) > n_neighbours:
            idx = np.argpartition(d2, n_neighbours, axis=1)[:, :n_neighbours]
            d2n = np.take_along_axis(d2, idx, axis=1)
            rn  = resid[idx]
        else:
            d2n, rn = d2, np.broadcast_to(resid, d2.shape)
        w = np.exp(-d2n / (2.0 * b2)).astype(np.float32)
        wsum = np.sum(w, axis=1) + _EPS
        out[yc, xc] = np.sum(w * rn, axis=1) / wsum
    return out


# ════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════
def run_cbr(s2: dict,
            ref_pts: list,
            bbox: list,
            n_clusters: int = DEFAULT_N_CLUSTERS,
            min_pts_per_cluster: int = 8,
            smooth_boundaries: bool = True,
            features_mode: str = "poly",     # 'linear' (Geyman) or 'poly'
            depth_weighting: bool = True,    # shallow-emphasis sample weights
            calibrate: str = "local",         # 'none' | 'global' | 'local'
            ref_max_depth_m: float = 15.0,    # cap training refs at Stumpf saturation
            mad_trim_sigma: float = 2.5,      # drop refs with |r|>N·MAD in first pass
            seed: int = 7) -> CBRResult:
    """
    Parameters
    ----------
    s2          : output of backend.app.fetch_s2 — dict with 'blue','green',
                  'coastal','ndwi','water_mask','width','height'
    ref_pts     : list of {'lat','lon','depth'} — ideally SlideRule-filtered
                  ICESat-2 bathymetric photons
    bbox        : [west, south, east, north]  (degrees, WGS84)
    n_clusters  : K for K-means (Geyman 2019 used 5-8)
    min_pts_per_cluster : clusters with fewer refs reuse the global fit
    smooth_boundaries   : blend two nearest cluster models to hide seams
    seed        : RNG seed for reproducibility

    Returns
    -------
    CBRResult with depth grid (H, W) in metres.
    """
    # ── 1. Features ──────────────────────────────────────────────────────
    X_BG, X_CG, Rw_b, Rw_g, Rw_r, Rw_c = _build_features(s2)
    H, W = X_BG.shape

    water_mask = s2.get("water_mask")
    if water_mask is None:
        water_mask = s2.get("ndwi", np.zeros((H, W))) > 0
    water_mask = water_mask & np.isfinite(X_BG) & np.isfinite(X_CG)

    n_water = int(water_mask.sum())
    if n_water < 200:
        raise ValueError(f"CBR: only {n_water} water pixels — ROI too small or masked")

    # ── 2. Rasterise reference depths ───────────────────────────────────
    rr, cc, zz = _refs_to_pixels(ref_pts, bbox, H, W)
    # Keep only refs that fall on water pixels with finite features
    valid = water_mask[rr, cc]
    rr, cc, zz = rr[valid], cc[valid], zz[valid]
    n_ref_raw = len(zz)
    # Cap training depths — Stumpf SDB saturates above ~12-15 m in turbid
    # Gulf water, and ICESat-2 photons beyond that depth become noise.
    # Only apply the cap when it leaves us with ≥ 10 refs (so we never
    # starve genuinely-shallow sites).
    if ref_max_depth_m and ref_max_depth_m > 0:
        keep = zz <= ref_max_depth_m
        if int(keep.sum()) >= 10:
            rr, cc, zz = rr[keep], cc[keep], zz[keep]
    n_ref_after_cap = len(zz)
    if len(zz) < 10:
        raise ValueError(f"CBR: only {len(zz)} usable reference depths (need ≥10)")

    # ── 3. "Bottom-detectable" mask — CBR only applies where light
    #       actually reaches the bottom.  Two gates stacked:
    #       (a) adaptive Rw floor — derived from the references themselves
    #           so at least 85 % of refs survive, with a tiny absolute floor
    #           (MIN_RW_HARD) to drop pure deep water;
    #       (b) log-ratio envelope — the ratio must lie within the
    #           reference spread, otherwise the pixel is outside the regime
    #           the fit was calibrated for.
    ref_Rw_b = Rw_b[rr, cc]; ref_Rw_g = Rw_g[rr, cc]
    min_ref  = np.minimum(ref_Rw_b, ref_Rw_g)
    adaptive_floor = max(MIN_RW_HARD, float(np.percentile(min_ref, 15)))
    rw_ok = (Rw_b > adaptive_floor) & (Rw_g > adaptive_floor)

    ref_ok  = (ref_Rw_b > adaptive_floor) & (ref_Rw_g > adaptive_floor)
    if ref_ok.sum() < 10:
        # fall back to the hard floor — we still need ≥10 refs
        adaptive_floor = MIN_RW_HARD
        rw_ok = (Rw_b > adaptive_floor) & (Rw_g > adaptive_floor)
        ref_ok = (ref_Rw_b > adaptive_floor) & (ref_Rw_g > adaptive_floor)
        if ref_ok.sum() < 10:
            raise ValueError(f"CBR: only {int(ref_ok.sum())} references "
                             "above Rw hard floor — radiometry too dark")
    rr, cc, zz = rr[ref_ok], cc[ref_ok], zz[ref_ok]
    ref_XBG = X_BG[rr, cc]; ref_XCG = X_CG[rr, cc]

    # Envelope gate: start strict, auto-relax if the refs cover a narrow
    # depth band (typical for SlideRule bathymetric photons that cluster in
    # one depth range).  Each stage widens the percentile clip and/or the
    # padding, then finally drops the envelope altogether and falls back to
    # the Rw floor.  We pick the first stage that yields ≥ a minimum of
    # pixels *and* ≥ a minimum fraction of the Rw-valid water area.
    rw_ok_n = int(rw_ok.sum())
    min_shallow_abs = 100
    min_shallow_frac = 0.02  # 2 % of Rw-valid water
    stages = [
        # (name,       lo_q, hi_q, pad_frac)
        ("strict",     2.0,  98.0, 0.20),
        ("relaxed",    1.0,  99.0, 0.50),
        ("very-loose", 0.5,  99.5, 1.00),
        ("rw-only",    None, None, None),   # skip envelope entirely
    ]
    envelope_used = None
    for name, lo_q, hi_q, pad_frac in stages:
        if name == "rw-only":
            shallow_mask = water_mask & rw_ok
            lo_bg = hi_bg = lo_cg = hi_cg = float("nan")
        else:
            lo_bg, hi_bg = np.percentile(ref_XBG, [lo_q, hi_q])
            lo_cg, hi_cg = np.percentile(ref_XCG, [lo_q, hi_q])
            pad_bg = pad_frac * max(hi_bg - lo_bg, 1e-3)
            pad_cg = pad_frac * max(hi_cg - lo_cg, 1e-3)
            shallow_mask = water_mask & rw_ok \
                & (X_BG >= lo_bg - pad_bg) & (X_BG <= hi_bg + pad_bg) \
                & (X_CG >= lo_cg - pad_cg) & (X_CG <= hi_cg + pad_cg)
        n_shallow = int(shallow_mask.sum())
        enough_abs = n_shallow >= min_shallow_abs
        enough_frac = rw_ok_n == 0 or n_shallow >= min_shallow_frac * rw_ok_n
        L.info(f"CBR envelope '{name}': n_shallow={n_shallow} "
               f"(abs>={min_shallow_abs}:{enough_abs}, "
               f"frac>={min_shallow_frac:.0%}:{enough_frac})  "
               f"X_BG∈[{lo_bg:.3f},{hi_bg:.3f}] X_CG∈[{lo_cg:.3f},{hi_cg:.3f}]")
        if enough_abs and enough_frac:
            envelope_used = name
            break
    else:
        envelope_used = "rw-only"
    L.info(f"CBR: water={n_water}, Rw_floor={adaptive_floor:.4f}, "
           f"Rw-ok={rw_ok_n}, shallow={n_shallow}, "
           f"envelope_stage={envelope_used}, n_ref={len(zz)}")
    if n_shallow < 50:
        # Even the Rw-only mask is empty → the ROI really is open ocean
        raise ValueError(
            f"CBR: only {n_shallow} pixels above the water-leaving reflectance "
            f"floor ({adaptive_floor:.4f}). The ROI is in open/deep water — "
            f"bottom reflectance cannot be recovered. "
            f"Try a smaller ROI closer to shore, or raise "
            f"min_pts_per_cluster and upload shallow survey points.")

    # ── 4. Feature vector for clustering (only on shallow pixels) ───────
    # Per Geyman 2019: cluster by bottom *colour*, not depth.  Using the
    # L2-normalised reflectance spectrum factors out the brightness axis
    # (which is dominated by depth) and keeps only spectral *shape*, which
    # is what distinguishes sand / seagrass / rubble / carbonate.
    spec = np.stack([Rw_c, Rw_b, Rw_g, Rw_r], axis=-1).astype(np.float64)  # (H, W, 4)
    spec_norm = np.linalg.norm(spec, axis=-1, keepdims=True) + _EPS
    feat_stack = spec / spec_norm                                           # (H, W, 4) — unit vectors

    feats = feat_stack[shallow_mask]                                        # (n_shallow, 4)
    mu    = feats.mean(axis=0)
    sd    = feats.std(axis=0) + _EPS
    feats_z = (feats - mu) / sd

    # Downsample for K-means fit if very large (cap at 50k samples)
    rng = np.random.default_rng(seed)
    if feats_z.shape[0] > 50_000:
        sub = rng.choice(feats_z.shape[0], size=50_000, replace=False)
        labels_sub, centers = _kmeans_fit(feats_z[sub], n_clusters, seed=seed)
        # Assign all water pixels by nearest centroid
        d2 = np.sum((feats_z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels_w = np.argmin(d2, axis=1)
    else:
        labels_w, centers = _kmeans_fit(feats_z, n_clusters, seed=seed)
    L.info(f"CBR: K-means converged, centroid sizes = "
           f"{[int((labels_w == k).sum()) for k in range(n_clusters)]}")

    cluster_map = np.full((H, W), -1, dtype=np.int8)
    cluster_map[shallow_mask] = labels_w.astype(np.int8)

    # ── 5. Fit per-class regression on the reference points ─────────────
    ref_feats = feat_stack[rr, cc]                            # (n_ref, 4)
    ref_feats_z = (ref_feats - mu) / sd
    d2 = np.sum((ref_feats_z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    ref_labels = np.argmin(d2, axis=1)
    ref_X = _design(X_BG[rr, cc], X_CG[rr, cc], mode=features_mode)  # (n_ref, F)
    n_feat = ref_X.shape[1]

    # MAD pre-trim — first pass OLS to flag ICESat-2 photons whose depth
    # is inconsistent with the log-ratio (misclassified surface returns,
    # sidelobe artefacts, refraction-correction glitches). Removing these
    # before the Huber fit stabilises the per-class regression when
    # SlideRule YAPC didn't catch them.
    n_mad_trim = 0
    if mad_trim_sigma and mad_trim_sigma > 0 and len(zz) >= 20:
        A0 = np.column_stack([ref_X, np.ones(len(zz))])
        try:
            c0 = np.linalg.lstsq(A0, zz, rcond=None)[0]
            r0 = zz - A0 @ c0
            mad = float(np.median(np.abs(r0 - np.median(r0)))) + _EPS
            keep = np.abs(r0 - np.median(r0)) <= mad_trim_sigma * 1.4826 * mad
            if int(keep.sum()) >= max(15, n_feat + 2):
                n_mad_trim = int((~keep).sum())
                rr, cc, zz = rr[keep], cc[keep], zz[keep]
                ref_X = _design(X_BG[rr, cc], X_CG[rr, cc], mode=features_mode)
                ref_labels = ref_labels[keep]
        except Exception:
            pass

    # Shallow-emphasis sample weights — shallow photons are where SDB
    # actually works, and SlideRule often over-samples the deep end.
    # w ∝ 1 / (z + 0.5) keeps the fit honest in the 0.3–8 m band.
    if depth_weighting:
        sw_all = 1.0 / (zz + 0.5)
    else:
        sw_all = np.ones_like(zz)

    # Global fallback model — used for clusters with too few refs
    global_coef, g_rmse, g_mae, g_r2, g_bias = _fit_one_class(ref_X, zz, sw_all)
    per_class = []
    coefs = np.zeros((n_clusters, n_feat + 1), dtype=np.float64)
    sigmas = np.zeros(n_clusters, dtype=np.float64)
    for k in range(n_clusters):
        m = (ref_labels == k)
        if m.sum() >= min_pts_per_cluster:
            coef, rmse, mae, r2, bias = _fit_one_class(ref_X[m], zz[m], sw_all[m])
            per_class.append({"id": k, "n_train": int(m.sum()), "coef": coef.tolist(),
                              "rmse": round(rmse, 3), "mae": round(mae, 3),
                              "r2": round(r2, 3), "bias": round(bias, 3),
                              "fallback": False})
            sigmas[k] = max(rmse, 0.15)
        else:
            coef = global_coef
            per_class.append({"id": k, "n_train": int(m.sum()), "coef": coef.tolist(),
                              "rmse": round(g_rmse, 3), "mae": round(g_mae, 3),
                              "r2": round(g_r2, 3), "bias": round(g_bias, 3),
                              "fallback": True})
            sigmas[k] = max(g_rmse, 0.25)
        coefs[k] = coef

    # ── 6. Predict depth pixel-wise on the shallow envelope only ────────
    depth = np.full((H, W), np.nan, dtype=np.float32)
    unc   = np.full((H, W), np.nan, dtype=np.float32)

    Xpix = _design(X_BG[shallow_mask], X_CG[shallow_mask], mode=features_mode)
    flat_X = np.column_stack([Xpix, np.ones(n_shallow)])  # (n_shallow, F+1)

    if smooth_boundaries:
        # Blend the two nearest centroids by an inverse-distance weighting
        d2w = np.sum((feats_z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        order = np.argsort(d2w, axis=1)
        k1 = order[:, 0]; k2 = order[:, 1]
        w1 = 1.0 / (d2w[np.arange(n_shallow), k1] + 0.05)
        w2 = 1.0 / (d2w[np.arange(n_shallow), k2] + 0.05)
        wsum = w1 + w2
        a1, a2 = w1 / wsum, w2 / wsum
        pred = a1 * (flat_X * coefs[k1]).sum(axis=1) + a2 * (flat_X * coefs[k2]).sum(axis=1)
        sig  = a1 * sigmas[k1] + a2 * sigmas[k2]
    else:
        k1 = labels_w
        pred = (flat_X * coefs[k1]).sum(axis=1)
        sig  = sigmas[k1]

    # Physical clamp — SDB is only reliable over [MIN_DEPTH_M, MAX_DEPTH_M]
    pred = np.clip(pred, 0.0, MAX_DEPTH_M)
    depth[shallow_mask] = pred.astype(np.float32)
    unc[shallow_mask]   = sig.astype(np.float32)

    # ── 6a. In-sample reference predictions (pre-calibration) ────────────
    pred_ref = np.zeros_like(zz)
    for i in range(len(zz)):
        k = ref_labels[i]
        pred_ref[i] = ref_X[i] @ coefs[k, :-1] + coefs[k, -1]

    # ── 6b. Post-calibration — absorbs spatial bias that the per-class
    #        linear model cannot (tide offsets, local turbidity, fine-grain
    #        bottom-type structure).  Three modes:
    #           'none'   → raw CBR predictions
    #           'global' → subtract a single mean residual
    #           'local'  → Gaussian-IDW residual field (default, ~30-50%
    #                      RMSE reduction in heterogeneous ROIs)
    calib_info = {"mode": calibrate, "global_shift_m": 0.0}
    if calibrate in ("global", "local") and len(zz) >= 5:
        resid = pred_ref - zz
        shift = float(np.mean(resid))
        calib_info["global_shift_m"] = round(shift, 3)
        depth[shallow_mask] = np.clip(depth[shallow_mask] - shift, 0.0, MAX_DEPTH_M)
        pred_ref = pred_ref - shift
        if calibrate == "local":
            resid_local = pred_ref - zz  # post-global-shift residuals
            field = _idw_residual_field(rr, cc, resid_local.astype(np.float32),
                                        H, W, shallow_mask)
            depth[shallow_mask] = np.clip(
                depth[shallow_mask] - field[shallow_mask], 0.0, MAX_DEPTH_M)
            # Update in-sample reference predictions with the local field
            pred_ref = pred_ref - field[rr, cc]
            # Inflate pixel σ where the IDW field varies sharply — adds
            # honest uncertainty where the correction is bold.
            f_abs = np.abs(field[shallow_mask])
            unc[shallow_mask] = unc[shallow_mask] + 0.3 * f_abs.astype(np.float32)
            calib_info["local_mean_abs_m"] = round(float(np.mean(f_abs)), 3)

    err = pred_ref - zz
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((zz - zz.mean()) ** 2))
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > _EPS else 0.0

    # IHO S-44 order-1a TVU at 95 %: √(a² + (b·d)²) with a=0.5, b=0.013
    a, b = 0.5, 0.013
    z_mean = max(float(np.mean(zz)), 1.0)
    tvu_1a = float(np.sqrt(a ** 2 + (b * z_mean) ** 2))
    iho = "S-44 order 1a" if rmse <= tvu_1a else ("S-44 order 2" if rmse <= 2.0 * tvu_1a else "below order 2")

    metrics = {
        "rmse": round(rmse, 3),
        "mae":  round(mae, 3),
        "bias": round(bias, 3),
        "r2":   round(r2, 3),
        "n_train": int(len(zz)),
        "n_water_pixels": int(n_water),
        "n_shallow_pixels": int(n_shallow),
        "max_depth_m": float(MAX_DEPTH_M),
        "iho_s44": iho,
        "envelope_stage": envelope_used,
        "rw_floor": round(float(adaptive_floor), 5),
        "features_mode": features_mode,
        "depth_weighting": bool(depth_weighting),
        "calibration": calib_info,
        "n_ref_raw": int(n_ref_raw),
        "n_ref_after_depth_cap": int(n_ref_after_cap),
        "n_ref_mad_trimmed": int(n_mad_trim),
        "ref_max_depth_m": float(ref_max_depth_m),
    }
    L.info(f"CBR: RMSE={rmse:.2f}m MAE={mae:.2f}m R²={r2:.3f} "
           f"bias={bias:+.2f}m  K={n_clusters}  n_ref={len(zz)}  {iho}")

    return CBRResult(
        depth=depth, uncertainty=unc, cluster_map=cluster_map,
        water_mask=shallow_mask, n_clusters=n_clusters,
        per_class=per_class, metrics=metrics, bbox=list(bbox),
    )
