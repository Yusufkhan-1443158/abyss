"""Bathymetry result-cache helpers.

Keys a generated depth product by (normalized bbox, date range, model, model
version) so a repeated request can be served from the stored report instead of
re-running the pipeline. A model-version change changes the key → re-inference
(the S2 imagery cache in the bathymetry-service is keyed separately, by
area+dates only, so it is still reused across model changes — no re-pull).
"""

from __future__ import annotations

import hashlib


def normalize_bbox(bbox) -> list | None:
    """Coerce {west,south,east,north} or [w,s,e,n] -> [w,s,e,n] floats."""
    if isinstance(bbox, dict):
        try:
            return [float(bbox["west"]), float(bbox["south"]),
                    float(bbox["east"]), float(bbox["north"])]
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return [float(x) for x in bbox]
        except (TypeError, ValueError):
            return None
    return None


def result_key(bbox, start_date: str, end_date: str, model: str,
               model_version) -> str:
    """Stable dedup key. bbox is rounded to ~1 m to absorb float noise."""
    w, s, e, n = bbox
    raw = (f"{round(w, 5)},{round(s, 5)},{round(e, 5)},{round(n, 5)}"
           f"|{start_date}|{end_date}|{model}|{model_version or ''}")
    return hashlib.sha1(raw.encode()).hexdigest()


def current_model(service_url: str, timeout: float = 8.0) -> tuple[str, str | None]:
    """Ask the bathymetry service for the active model + version (cheap, no
    inference). Falls back to ('dl-pro-v3', None) if unreachable so dedup still
    works deterministically."""
    import httpx
    try:
        r = httpx.get(f"{service_url}/bathymetry/models", timeout=timeout)
        r.raise_for_status()
        j = r.json()
        meta = j.get("meta") or {}
        return (j.get("default") or "dl-pro-v3"), meta.get("version")
    except Exception:
        return "dl-pro-v3", None
