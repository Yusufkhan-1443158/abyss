"""Deterministic metric-lattice + fixed-window temporal-composite helpers.

VERYHR_STABLE_LOG ROUND 1 (SPEC-1, SPEC-2, SPEC-3, SPEC-5).

Three reusable building blocks, kept dependency-light so any default product can
route through them without an ``app.py`` import cycle:

  build_geotiff_1m_utm(depth, bbox, ...)  -- SPEC-1: snap depth onto a TRUE 1 m
      (or N m) integer-metre UTM lattice (EPSG:32640 default, UTM 40N), origin
      snapped to whole metres, res = (R, -R). 1 m here is a RENDER resolution,
      NOT an accuracy claim -- the GeoTIFF tags say so explicitly.

  param_hash(roi, params, window)         -- SPEC-2: a stable content key so the
      same (roi, params, window) always names the same artifact (no time.time()).

  stable_composite(...)                   -- SPEC-3: fixed-window, scene-rejected
      (SCL + Hedley glint + Caballero/Stumpf turbidity + n_valid>=3) per-pixel
      inverse-variance composite, reusing vhr_mle_pro._inv_variance_mle.

HONESTY: the composite cuts noise/variance, it does NOT improve bias/RMSE.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

L = logging.getLogger("grid_1m")
if not L.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    L.addHandler(_h)
    L.setLevel(logging.INFO)

MAX_DEPTH_M = 25.0

# Fixed 2-yr standard composite window for the Khalifa-VHR default product.
# Pinned constant -- never date.today()/utcnow() on the deterministic path.
KHALIFA_VHR_WINDOW: Tuple[str, str] = ("2023-01-01", "2024-12-31")

# Per-scene rejection config (SPEC-3). All thresholds are unit-reflectance.
DEFAULT_REJECT_CFG: Dict[str, Any] = {
    "glint_nir_max": 0.03,      # Hedley et al. 2005: NIR>0.03 over water => glint
    "turbidity_ndti_max": 0.05, # Caballero & Stumpf 2020: ROI-median NDTI gate
    "min_valid_per_pixel": 3,   # n_valid>=3 contributing scenes or pixel masked
    "sigma_clamp": (0.5, 5.0),
    # Robust temporal-consistency gate (SPEC-3 tightening): a scene whose
    # ROI-median depth deviates from the cross-scene median by more than
    # consistency_mad_k * MAD is an outlier acquisition (residual cloud/glint/
    # turbidity/tide that slipped the band gates) and is dropped BEFORE the
    # inverse-variance fuse. This is what stabilises leave-one-scene-out when
    # raw-band NDTI is unavailable in-env.
    "consistency_mad_k": 3.0,
    "consistency_min_scenes": 5,  # only apply the gate when >= this many scenes
}


# ──────────────────────────────────────────────────────────────────────────
#  SPEC-2 — deterministic content key
# ──────────────────────────────────────────────────────────────────────────
def param_hash(roi: Sequence[float], params: Dict[str, Any],
               window: Optional[Sequence[str]] = None, n: int = 12) -> str:
    """Stable short hash of (roi bbox, sorted params, window). Replaces
    int(time.time())/job_id in deterministic output filenames."""
    payload = {
        "roi": [round(float(v), 9) for v in roi],
        "params": _canonical(params),
        "window": list(window) if window else None,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:n]


def _canonical(obj: Any) -> Any:
    """JSON-serialisable, order-stable view of params (drops volatile keys)."""
    _VOLATILE = {"job_id", "id", "ts", "timestamp", "created", "now"}
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())
                if k not in _VOLATILE}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 9)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


# ──────────────────────────────────────────────────────────────────────────
#  SPEC-1 — true integer-metre UTM lattice
# ──────────────────────────────────────────────────────────────────────────
def utm_lattice_transform(bbox: Sequence[float], res_m: float = 1.0,
                          epsg: int = 32640):
    """Compute the snapped integer-metre UTM transform + grid shape for a
    lon/lat bbox. Returns (transform, width, height, (x0,y0,x1,y1)).

    Origin snapped to whole metres: x0=floor(xmin), y1=ceil(ymax).
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import transform_bounds

    w, s, e, n = [float(v) for v in bbox]
    # Project the WGS84 bbox to the metric UTM CRS.
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", f"EPSG:{epsg}",
                                              w, s, e, n, densify_pts=21)
    res = float(res_m)
    x0 = math.floor(xmin)
    y1 = math.ceil(ymax)
    width = int(round((math.ceil(xmax) - x0) / res))
    height = int(round((y1 - math.floor(ymin)) / res))
    width = max(1, width)
    height = max(1, height)
    transform = Affine.translation(x0, y1) * Affine.scale(res, -res)
    return transform, width, height, (x0, math.floor(ymin), math.ceil(xmax), y1)


def build_geotiff_1m_utm(depth: np.ndarray, bbox: Sequence[float],
                         target_res_m: float = 1.0, epsg: int = 32640,
                         native_res_m: float = 10.0,
                         extra_tags: Optional[Dict[str, str]] = None,
                         return_array: bool = False):
    """SPEC-1. Resample a depth grid (in 4326 pixel order matching ``bbox``)
    onto a TRUE integer-metre UTM lattice and return base64 GeoTIFF bytes.

    ``depth`` is assumed laid out top-left = (w, n) of ``bbox`` (the layout
    build_geotiff_b64 uses). 1 m is RENDER resolution (bilinear upsample of the
    ``native_res_m`` depth field); tags record that so 1 m is never read as
    accuracy.

    If ``return_array`` is True, also returns (b64, depth_on_lattice, transform).
    """
    import base64
    import io
    import rasterio
    from rasterio.transform import from_bounds as _from_bounds
    from rasterio.warp import reproject, Resampling

    depth = np.asarray(depth, dtype=np.float32)
    H, W = depth.shape
    w, s, e, n = [float(v) for v in bbox]

    dst_transform, dst_w, dst_h, snap = utm_lattice_transform(
        bbox, res_m=target_res_m, epsg=epsg)

    # Source raster in EPSG:4326 spanning bbox at native pixel layout.
    src_transform = _from_bounds(w, s, e, n, W, H)
    src = np.where(np.isfinite(depth), depth, -9999.0).astype(np.float32)

    dst = np.full((dst_h, dst_w), -9999.0, dtype=np.float32)
    reproject(
        source=src, destination=dst,
        src_transform=src_transform, src_crs="EPSG:4326",
        dst_transform=dst_transform, dst_crs=f"EPSG:{epsg}",
        src_nodata=-9999.0, dst_nodata=-9999.0,
        resampling=Resampling.bilinear,
    )

    tags = {
        "AREA_OR_POINT": "Area",
        "VERTICAL_DATUM": "LAT",
        "VERTICAL_UNITS": "metres_positive_down",
        "DATUM_TRANSFORM_APPLIED": "false",
        "DATUM_NOTE": ("LAT assumed (no explicit ellipsoid->geoid->LAT "
                       "transform); u_datum~0.20 m nominal Gulf"),
        "RENDER_RESOLUTION_M": f"{float(target_res_m):.3f}",
        "NATIVE_DEPTH_RESOLUTION_M": f"{float(native_res_m):.3f}",
        "RESOLUTION_NOTE": ("render/grid resolution, NOT accuracy; depth "
                            "computed natively at NATIVE_DEPTH_RESOLUTION_M"),
        "LATTICE": f"integer-metre EPSG:{epsg} snap (origin x0=floor,y1=ceil)",
    }
    if extra_tags:
        tags.update({k: str(v) for k, v in extra_tags.items()})

    buf = io.BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff", height=dst_h, width=dst_w, count=1,
        dtype="float32", crs=f"EPSG:{epsg}", transform=dst_transform,
        nodata=-9999.0, compress="deflate",
    ) as ds:
        ds.write(dst, 1)
        ds.update_tags(**tags)
        ds.set_band_description(1, "depth_m_positive_down_LAT_assumed_RENDER1m")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    L.info(f"GeoTIFF(1m-UTM): {dst_w}x{dst_h}, EPSG:{epsg}, "
           f"res={target_res_m}m, origin=({snap[0]},{snap[3]})")
    if return_array:
        out = np.where(dst <= -9990, np.nan, dst).astype(np.float32)
        return b64, out, dst_transform
    return b64


# ──────────────────────────────────────────────────────────────────────────
#  SPEC-3 — per-scene rejection + fixed-window inverse-variance composite
# ──────────────────────────────────────────────────────────────────────────
def scene_reject(scene: Dict[str, Any], cfg: Dict[str, Any]
                 ) -> Tuple[bool, Dict[str, Any]]:
    """Per-scene accept/reject BEFORE compositing.

    scene must carry: 'depth' (HxW), 'water' (HxW bool). Optional: 'nir'
    (B08 reflectance HxW), 'red' (B04), 'green' (B03) for glint/turbidity.

    Returns (accept, diag). A scene with no NIR/red bands is accepted on SCL
    only (already done upstream by the median evalscript) but a per-pixel glint
    mask is still applied when NIR is present.
    """
    diag: Dict[str, Any] = {"glint_frac": None, "ndti": None}
    water = np.asarray(scene.get("water"))
    nir = scene.get("nir")
    if nir is not None and water is not None and water.any():
        nirw = np.asarray(nir, dtype=np.float32)[water]
        nirw = nirw[np.isfinite(nirw)]
        if nirw.size:
            glint_frac = float(np.mean(nirw > cfg["glint_nir_max"]))
            diag["glint_frac"] = round(glint_frac, 4)
            if glint_frac > 0.5:   # >half the water glinty => drop scene
                return False, {**diag, "reason": "glint"}
    red = scene.get("red"); green = scene.get("green")
    if red is not None and green is not None and water is not None and water.any():
        r = np.asarray(red, dtype=np.float32)[water]
        g = np.asarray(green, dtype=np.float32)[water]
        ok = np.isfinite(r) & np.isfinite(g) & ((r + g) > 1e-6)
        if ok.any():
            ndti = float(np.median((r[ok] - g[ok]) / (r[ok] + g[ok])))
            diag["ndti"] = round(ndti, 4)
            if ndti > cfg["turbidity_ndti_max"]:
                return False, {**diag, "reason": "turbidity"}
    return True, diag


def _glint_pixel_mask(scene: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    """Per-pixel glint reject (Hedley 2005) -> bool mask of pixels to KEEP."""
    water = np.asarray(scene["water"])
    keep = np.asarray(water, dtype=bool).copy()
    nir = scene.get("nir")
    if nir is not None:
        nirf = np.asarray(nir, dtype=np.float32)
        keep &= ~(nirf > cfg["glint_nir_max"])
    return keep


def stable_composite(scenes: List[Dict[str, Any]],
                     cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """SPEC-3. Fixed-window per-pixel inverse-variance composite with
    SCL+glint+turbidity+n_valid>=3 rejection.

    Each scene dict: depth (HxW float, positive-down), water (HxW bool),
    sigma (float, scene RMSE), optional nir/red/green reflectance for rejection.

    Returns {depth, sigma, n_valid, water, accepted, rejected, n_scenes_in,
    n_scenes_used}. depth is NaN where n_valid < min_valid_per_pixel.
    """
    cfg = {**DEFAULT_REJECT_CFG, **(cfg or {})}
    if not scenes:
        raise RuntimeError("stable_composite received zero scenes")

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenes):
        ok, diag = scene_reject(sc, cfg)
        rec = {"i": i, "date": sc.get("date"), **diag, "used": ok}
        if ok:
            sc.setdefault("_diag", diag)
            accepted.append(sc)
        else:
            rejected.append(rec)

    if not accepted:
        raise RuntimeError("stable_composite: all scenes rejected")

    # Robust temporal-consistency (MAD) outlier gate -- drops a single
    # acquisition that swings the whole composite (the leave-one-scene-out
    # failure mode). Applied only with enough scenes to estimate a robust
    # centre. Deterministic (median/MAD), order-independent.
    if len(accepted) >= int(cfg.get("consistency_min_scenes", 5)):
        roi_med = []
        for sc in accepted:
            d = np.asarray(sc["depth"], dtype=np.float32)
            w = np.asarray(sc["water"], dtype=bool)
            v = w & np.isfinite(d) & (d > 0)
            roi_med.append(float(np.median(d[v])) if v.any() else np.nan)
        rm = np.asarray(roi_med, dtype=np.float64)
        fin = np.isfinite(rm)
        if fin.sum() >= int(cfg.get("consistency_min_scenes", 5)):
            centre = float(np.median(rm[fin]))
            mad = float(np.median(np.abs(rm[fin] - centre))) or 1e-6
            k = float(cfg.get("consistency_mad_k", 3.0))
            keep_sc, kept = [], []
            for sc, m in zip(accepted, rm):
                if np.isfinite(m) and abs(m - centre) > k * 1.4826 * mad:
                    rejected.append({"date": sc.get("date"),
                                     "roi_median_m": round(float(m), 3),
                                     "centre_m": round(centre, 3),
                                     "used": False, "reason": "temporal_outlier"})
                else:
                    keep_sc.append(sc)
            if len(keep_sc) >= int(cfg["min_valid_per_pixel"]):
                accepted = keep_sc

    H = min(np.asarray(g["depth"]).shape[0] for g in accepted)
    W = min(np.asarray(g["depth"]).shape[1] for g in accepted)
    lo, hi = cfg["sigma_clamp"]
    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)
    n_valid = np.zeros((H, W), dtype=np.int16)
    union_water = np.zeros((H, W), dtype=bool)

    for sc in accepted:
        d = np.asarray(sc["depth"], dtype=np.float32)[:H, :W]
        w = np.asarray(sc["water"], dtype=bool)[:H, :W]
        keep = _glint_pixel_mask({"water": w,
                                  "nir": (np.asarray(sc["nir"])[:H, :W]
                                          if sc.get("nir") is not None else None)},
                                 cfg)
        sigma = float(np.clip(sc.get("sigma", 1.5), lo, hi))
        inv = 1.0 / (sigma * sigma)
        ok = keep & np.isfinite(d) & (d > 0)
        num[ok] += d[ok] * inv
        den[ok] += inv
        n_valid[ok] += 1
        union_water |= w

    min_v = int(cfg["min_valid_per_pixel"])
    z = np.full((H, W), np.nan, dtype=np.float32)
    s = np.full((H, W), np.nan, dtype=np.float32)
    enough = (den > 0) & (n_valid >= min_v)
    z[enough] = (num[enough] / den[enough]).astype(np.float32)
    s[enough] = (1.0 / np.sqrt(den[enough])).astype(np.float32)
    z = np.clip(z, 0.0, MAX_DEPTH_M)

    return {
        "depth": z, "sigma": s, "n_valid": n_valid, "water": union_water,
        "H": H, "W": W,
        "n_scenes_in": len(scenes),
        "n_scenes_used": len(accepted),
        "rejected": rejected,
    }
