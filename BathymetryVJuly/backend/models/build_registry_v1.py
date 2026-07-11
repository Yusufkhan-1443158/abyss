#!/usr/bin/env python3
"""TRAIN-R3/R4 — assemble backend/models/registry/v1/model_card.json from the
just-trained uae_clustered_rf.pkl / uae_clustered_cnn.pkl, the TRAIN-R2 region
loaders (for counts/bbox/content_sha256 — never raw coordinates), and the
TRAIN-R7 held-out spatial-block metrics.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.registry_hash import point_fingerprint
from backend.models.registry_schema import validate

REGISTRY_DIR = ROOT / "backend" / "models" / "registry"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _region_source_kind(region_key: str, has_xyz: bool, has_atl24: bool) -> str:
    if has_atl24:
        return "atl24_r2"
    if has_xyz:
        return "in_situ_xyz"
    return "in_situ_xyz"


def build(version: str = "v1", metrics_path: Path = None):
    # Source region counts/bbox from the TRAINED pkl's own meta['regions'] —
    # the on-water, actually-trained-on count, not the theoretical max the
    # raw loaders could produce (a region whose S2 fetch failed and was
    # skipped must not be claimed as trained-on).
    import pickle
    from train_uae_pretrained import REGIONS, _load_region_xyz

    rf_path = ROOT / "backend" / "models" / "uae_clustered_rf.pkl"
    with open(rf_path, "rb") as fh:
        import gzip
        magic = fh.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(rf_path, "rb") as fh:
        trained_model = pickle.load(fh)
    trained_regions = {r["region"]: r for r in trained_model.meta.get("regions", [])}

    sources = []
    n_total = 0
    for region in REGIONS:
        if region["key"] not in trained_regions:
            continue  # this region's S2 fetch failed and was skipped at train time
        lats, lons, deps, wts = _load_region_xyz(region)
        if len(deps) == 0:
            continue
        trained_n = trained_regions[region["key"]]["n_samples"]
        bbox = trained_regions[region["key"]]["bbox"]
        kind = "atl24_r2" if region.get("atl24_r2") else "in_situ_xyz"
        fp = point_fingerprint(lats, lons, deps)
        sources.append({
            "kind": kind, "region": region["key"], "n_points": int(trained_n),
            "bbox": bbox, "content_sha256": fp,
        })
        n_total += trained_n

    rf_path = ROOT / "backend" / "models" / "uae_clustered_rf.pkl"
    cnn_path = ROOT / "backend" / "models" / "uae_clustered_cnn.pkl"
    weights_sha = {}
    if rf_path.exists():
        weights_sha["uae_clustered_rf.pkl"] = _sha256_file(rf_path)
    if cnn_path.exists():
        weights_sha["uae_clustered_cnn.pkl"] = _sha256_file(cnn_path)

    per_site = {}
    if metrics_path and metrics_path.exists():
        with open(metrics_path) as fh:
            per_site = json.load(fh)

    catzoc_mapping = {}
    iho_orders = {}
    for site, m in per_site.items():
        rmse = m.get("rmse_m")
        if rmse is None:
            continue
        if rmse <= 0.5:
            catzoc_mapping[site] = "A1"; iho_orders[site] = "Special"
        elif rmse <= 1.0:
            catzoc_mapping[site] = "B"; iho_orders[site] = "Order 1a"
        elif rmse <= 2.0:
            catzoc_mapping[site] = "C"; iho_orders[site] = "Order 1b"
        else:
            catzoc_mapping[site] = "D"; iho_orders[site] = "Order 2"

    card = {
        "schema_version": 1,
        "semver": "1.0.0",
        "parent_version": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights": {
            "rf_path": "uae_clustered_rf.pkl", "cnn_path": "uae_clustered_cnn.pkl",
            "sha256": weights_sha,
        },
        "training_data_summary": {"sources": sources, "n_total": n_total,
                                  "n_regions": len(sources)},
        "metrics": {"split": "spatial_block_500m", "per_site": per_site,
                   "leakage_guards": {}},
        "catzoc_mapping": catzoc_mapping,
        "iho_orders": iho_orders,
        "lineage": {"data_added_since_parent": [], "metric_delta_vs_parent": {}},
    }

    violations = validate(card)
    if violations:
        raise RuntimeError(f"model_card failed schema validation: {violations}")

    out_dir = REGISTRY_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    if rf_path.exists():
        shutil.copy2(rf_path, out_dir / "uae_clustered_rf.pkl")
    if cnn_path.exists():
        shutil.copy2(cnn_path, out_dir / "uae_clustered_cnn.pkl")
    with open(out_dir / "model_card.json", "w") as fh:
        json.dump(card, fh, indent=2)
    (REGISTRY_DIR / "CURRENT").write_text(version)
    print(f"Registry {version} written -> {out_dir / 'model_card.json'} "
         f"(n_total={n_total}, n_regions={len(sources)})")
    return card


if __name__ == "__main__":
    mp = ROOT / "cache" / "registry_metrics_r7.json"
    build(version="v1", metrics_path=mp if mp.exists() else None)
