"""TRAIN-R3 — schema + validator for backend/models/registry/v<N>/model_card.json.

training_data_summary carries COUNTS/BBOXES/HASHES ONLY — never raw
coordinates. validate() enforces this and the required per-source fields.
"""
from __future__ import annotations

import re

_RAW_COORD_KEYS = {"lat", "lon", "latitude", "longitude", "x", "y",
                   "lats", "lons", "latitudes", "longitudes"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _scan_raw_coordinate_keys(obj, path="training_data_summary") -> list[str]:
    violations = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in _RAW_COORD_KEYS:
                violations.append(f"raw coordinate key '{k}' found at {path}.{k}")
            violations.extend(_scan_raw_coordinate_keys(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            violations.extend(_scan_raw_coordinate_keys(v, f"{path}[{i}]"))
    return violations


def validate(model_card: dict) -> list[str]:
    """Returns a list of violation strings; empty list = valid."""
    violations: list[str] = []

    tds = model_card.get("training_data_summary")
    if not isinstance(tds, dict):
        violations.append("missing or non-dict training_data_summary")
        return violations

    violations.extend(_scan_raw_coordinate_keys(tds))

    sources = tds.get("sources")
    if not isinstance(sources, list) or not sources:
        violations.append("training_data_summary.sources must be a non-empty list")
    else:
        for i, src in enumerate(sources):
            if not isinstance(src, dict):
                violations.append(f"sources[{i}] is not an object")
                continue
            if not isinstance(src.get("n_points"), int) or src["n_points"] <= 0:
                violations.append(f"sources[{i}].n_points missing or not a positive int")
            bbox = src.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4
                    and all(isinstance(v, (int, float)) for v in bbox)):
                violations.append(f"sources[{i}].bbox must be [w, s, e, n] floats")
            h = src.get("content_sha256")
            if not (isinstance(h, str) and _SHA256_RE.match(h)):
                violations.append(f"sources[{i}].content_sha256 must be 64 hex chars")

    metrics = model_card.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("split") != "spatial_block_500m":
        violations.append("metrics.split must equal 'spatial_block_500m'")

    weights = model_card.get("weights")
    if not isinstance(weights, dict) or "sha256" not in weights:
        violations.append("weights.sha256 block missing")

    return violations


if __name__ == "__main__":
    good = {
        "training_data_summary": {"sources": [
            {"n_points": 100, "bbox": [1.0, 2.0, 3.0, 4.0], "content_sha256": "a" * 64}]},
        "metrics": {"split": "spatial_block_500m"},
        "weights": {"sha256": {}},
    }
    bad = dict(good)
    bad["training_data_summary"] = dict(good["training_data_summary"])
    bad["training_data_summary"]["lat"] = [1.0, 2.0]
    print("good:", validate(good))
    print("bad:", validate(bad))
