"""vmarch-core adapter — delegates studio method cards to the REAL platform.

The vmarch-core container runs the user's actual Bathymetry_VMarch platform
(BathymetryVJuly/, unmodified pipeline code) behind gunicorn on :8080. Each
"live platform" method card maps to ONE real platform endpoint:

  vmarch-core-standard   → POST /api/sdb-pro            (fast S2 Lyzenga+Stumpf
                                                          +SlideRule+UAE chain)
  vmarch-core-clustered  → POST /api/very-hr-clustered  (Very-HR Clustered SDB)
  vmarch-core-mle        → POST /api/very-hr-mle        (multi-scene MLE
                                                          composite over a year)

The depth product is the platform's own georeferenced GeoTIFF (float32,
EPSG:4326, nodata -9999, positive-down metres) and the method/metrics
provenance is passed through VERBATIM from the platform response — Abyss only
re-packages it into its raster + report pipeline. No re-masking, no second
tide correction, no re-calibration is applied on top.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import requests

log = logging.getLogger("bathymetry.vmarch_core")

VMARCH_CORE_URL = os.getenv("VMARCH_CORE_URL", "http://vmarch-core:8080")

CORE_ENGINE_STANDARD = "vmarch-core-standard"
CORE_ENGINE_CLUSTERED = "vmarch-core-clustered"
CORE_ENGINE_MLE = "vmarch-core-mle"
CORE_ENGINE_WAVE = "vmarch-core-wave"
CORE_ENGINES = (CORE_ENGINE_STANDARD, CORE_ENGINE_CLUSTERED,
                CORE_ENGINE_MLE, CORE_ENGINE_WAVE)

# Read timeouts: the platform fetches its own imagery (Sentinel Hub → GEE
# fallback) and runs the full calibration chain; MLE does it once per scene.
_TIMEOUT_STANDARD = (10, 1200)
_TIMEOUT_MLE = (10, 3600)


class VMarchCoreError(RuntimeError):
    """vmarch-core unreachable or returned an unusable response."""


def core_health(timeout: float = 5.0):
    """GET /api/health — returns the platform's own health dict or None."""
    try:
        r = requests.get(f"{VMARCH_CORE_URL}/api/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as ex:  # noqa: BLE001 — health is best-effort
        log.info("vmarch-core health unavailable: %s", ex)
        return None


def _post(path: str, body: dict, timeout) -> dict:
    try:
        r = requests.post(f"{VMARCH_CORE_URL}{path}", json=body, timeout=timeout)
    except Exception as ex:
        raise VMarchCoreError(f"vmarch-core unreachable ({path}): {ex}") from ex
    if r.status_code != 200:
        raise VMarchCoreError(
            f"vmarch-core {path} HTTP {r.status_code}: {r.text[:300]}")
    try:
        data = r.json()
    except ValueError as ex:
        raise VMarchCoreError(f"vmarch-core {path} returned non-JSON") from ex
    if isinstance(data, dict) and data.get("error"):
        raise VMarchCoreError(f"vmarch-core {path}: {data['error']}")
    return data


def _decode_geotiff(blob: bytes):
    """Platform GeoTIFF bytes → (depth float32 NaN-holed, transform, bbox4326)."""
    from rasterio.io import MemoryFile
    with MemoryFile(blob) as mem, mem.open() as src:
        depth = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        nodata = src.nodata
        transform = src.transform
        b = src.bounds
    if nodata is not None:
        depth = np.where(depth == np.float32(nodata), np.nan, depth)
    depth = np.where(np.isfinite(depth) & (depth > -1000), depth, np.nan)
    return depth, transform, [b.left, b.bottom, b.right, b.top]


def _fetch_download(name: str) -> bytes:
    try:
        r = requests.get(f"{VMARCH_CORE_URL}/downloads/{name}",
                         timeout=_TIMEOUT_STANDARD)
        r.raise_for_status()
        return r.content
    except Exception as ex:
        raise VMarchCoreError(f"vmarch-core download {name} failed: {ex}") from ex


def _holdout_from_metrics(m: dict) -> dict:
    out = {}
    for src_k, dst_k in (("rmse_m", "rmse"), ("mae_m", "mae"),
                         ("bias_m", "bias"), ("r2", "r2"), ("n_test", "n")):
        v = m.get(src_k)
        if isinstance(v, (int, float)):
            out[dst_k] = round(float(v), 3)
    return out


def _tide_block(tc: dict) -> dict:
    """Platform tide_correction disclosure → Abyss report tide block. The
    platform default is disclosure-only (TIDE_REDUCE_TO_MSL off) so `applied`
    is normally False and depths stay exactly as the platform produced them."""
    tc = tc or {}
    applied = bool(tc.get("applied"))
    h = tc.get("height_m")
    return {
        "applied": applied,
        "applied_m": round(float(h), 3) if (applied and h is not None) else 0.0,
        "method": tc.get("source") or "none",
        "utc": tc.get("acquisition_utc"),
        "datum": tc.get("datum_after") or "uncorrected",
        "reason": tc.get("note"),
    }


def _provenance_extra(engine: str, endpoint: str, r: dict,
                      health: dict | None) -> dict:
    """Assemble the Abyss `extra` payload with the platform's own method
    labels/metrics passed through verbatim."""
    metrics = r.get("metrics") or {}
    ml = r.get("ml_stats") or {}
    version = (health or {}).get("v")
    label = r.get("method") or ml.get("method") or "VMarch platform product"
    if version:
        label = f"{label} · Bathymetry-from-Space v{version}"
    extra = {
        "model": engine,
        "model_version": version,
        # VERBATIM platform method label — this is what the report shows.
        "model_label": label,
        "method": ml.get("method") or r.get("method"),
        "engine": engine,
        "resolution_m": r.get("resolution_m")
                        or (r.get("s2shores") or {}).get("resolution_m"),
        "holdout_metrics": _holdout_from_metrics(metrics),
        "iho_s44_pct": {},
        "calibration": {
            "augmentation": r.get("augmentation"),
            "fit": r.get("fit"),
            "water_mask": (r.get("water_mask_meta") or {}).get("mask_source"),
        },
        "tide": _tide_block(r.get("tide_correction")),
        # Full untouched platform blocks for the report JSON.
        "vmarch_core": {
            "endpoint": endpoint,
            "platform_version": version,
            "method": r.get("method"),
            "ml_stats": ml,
            "metrics": metrics,
            "stats": r.get("stats"),
            "tide_correction": r.get("tide_correction"),
            "water_mask_meta": r.get("water_mask_meta"),
            "turbidity_pct": r.get("turbidity_pct"),
            "elapsed_s": r.get("elapsed_s"),
            **({"s2shores": r["s2shores"]} if r.get("s2shores") else {}),
        },
    }
    if (r.get("s2shores") or {}).get("scene_id"):
        extra["scene_id"] = r["s2shores"]["scene_id"]
    return extra


def run_core_engine(engine: str, bbox, start_date: str, end_date: str,
                    max_cloud: int = 20, n_scenes: int = 1,
                    resolution_m: int | None = None) -> dict:
    """Run one method card against the real platform. Returns
    {depth, sigma, transform, bbox, extra} ready for _products_payload.
    Raises VMarchCoreError when the platform is unreachable/failed (the
    caller falls back to the offline "(port)" engine with provenance)."""
    w, s, e, n = bbox
    bd = {"west": w, "south": s, "east": e, "north": n}
    health = core_health()

    if engine == CORE_ENGINE_STANDARD:
        body = {"bbox": bd, "start_date": start_date, "end_date": end_date,
                "max_cloud": int(max_cloud), "include_geotiff": True}
        if resolution_m:
            body["resolution_m"] = int(resolution_m)
        r = _post("/api/sdb-pro", body, _TIMEOUT_STANDARD)
        endpoint = "/api/sdb-pro"
    elif engine == CORE_ENGINE_CLUSTERED:
        body = {"bbox": bd, "s2_start_date": start_date,
                "s2_end_date": end_date, "max_cloud": int(max_cloud),
                "imagery_source": "s2", "include_geotiff": True}
        if resolution_m:
            body["resolution_m"] = int(resolution_m)
        r = _post("/api/very-hr-clustered", body, _TIMEOUT_STANDARD)
        endpoint = "/api/very-hr-clustered"
    elif engine == CORE_ENGINE_MLE:
        year = int((end_date or "2024")[:4])
        ns = int(n_scenes or 5)
        ns = min((3, 5, 7), key=lambda k: abs(k - ns))  # platform accepts 3/5/7
        body = {"bbox": bd, "year": year, "n_scenes": ns,
                "max_cloud": int(max_cloud)}
        r = _post("/api/very-hr-mle", body, _TIMEOUT_MLE)
        endpoint = "/api/very-hr-mle"
    elif engine == CORE_ENGINE_WAVE:
        body = {"bbox": bd, "start_date": start_date, "end_date": end_date,
                "cloud": int(max_cloud), "source": "auto",
                "top_k": max(1, min(int(n_scenes or 1), 7)),
                # wave output grid is 10-30 m — snap coarser requests to 30.
                "resolution_m": min(int(resolution_m or 20), 30)}
        r = _post("/api/s2shores-bathymetry", body, _TIMEOUT_MLE)
        endpoint = "/api/s2shores-bathymetry"
    else:
        raise VMarchCoreError(f"unknown vmarch-core engine '{engine}'")

    # ── Depth product: the platform's own GeoTIFF ────────────────────────────
    if engine == CORE_ENGINE_MLE:
        comp_name = None
        for d in r.get("downloads") or []:
            if str(d.get("name", "")).endswith("_composite.tif"):
                comp_name = d["name"]
                break
        if not comp_name:
            raise VMarchCoreError("MLE response has no composite GeoTIFF")
        blob = _fetch_download(comp_name)
    else:
        b64 = r.get("geotiff_b64")
        if not b64:
            raise VMarchCoreError(f"{endpoint} returned no geotiff_b64")
        import base64
        blob = base64.b64decode(b64)

    depth, transform, bbox4326 = _decode_geotiff(blob)
    if not np.isfinite(depth).any():
        raise VMarchCoreError("platform product has no valid water pixels")

    extra = _provenance_extra(engine, endpoint, r, health)
    if engine == CORE_ENGINE_MLE:
        extra["vmarch_core"]["stability"] = {
            k: (r.get("stability") or {}).get(k)
            for k in ("median_interscene_sigma_m", "scenes")
            if isinstance(r.get("stability"), dict)}
        extra["vmarch_core"]["scenes_kept_label"] = r.get("scenes_kept_label")
        if r.get("scenes_kept") is not None:
            extra["composite"] = {
                "n_scenes_used": r.get("scenes_kept"),
                "n_scenes_requested": r.get("scenes_attempted"),
                "per_scene": (r.get("augmentation") or {}).get("mle_scenes"),
            }
    return {"depth": depth, "sigma": None, "transform": transform,
            "bbox": bbox4326, "extra": extra}
