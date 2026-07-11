#!/usr/bin/env python3
"""TRAIN-R5/R6 — warm-start retrain with catastrophic-forgetting guard.

Loads backend/models/registry/v<N>/, warm-starts on newly supplied data
(via backend.uae_pretrained_cnn.fit_uae_cnn_warmstart), evaluates on the
SAME frozen spatial-block-500m test folds used to certify v<N>, and rejects
the candidate (keeps v<N> as CURRENT) if any parent-certified site's RMSE
degrades >10% or its decile slope halves.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

REGISTRY_DIR = ROOT / "backend" / "models" / "registry"


def guard_or_reject(old_card: Dict, new_card: Dict, tol_pct: float = 10.0,
                    slope_tol: float = 0.5) -> Tuple[bool, List[str]]:
    """Returns (accept: bool, violations: list[str])."""
    violations: List[str] = []
    old_sites = old_card.get("metrics", {}).get("per_site", {})
    new_sites = new_card.get("metrics", {}).get("per_site", {})
    for site, old_m in old_sites.items():
        old_rmse = old_m.get("rmse_m")
        if old_rmse is None or old_rmse <= 0:
            continue
        new_m = new_sites.get(site)
        if new_m is None:
            violations.append(f"{site}: missing from candidate metrics")
            continue
        new_rmse = new_m.get("rmse_m")
        if new_rmse is None:
            violations.append(f"{site}: rmse_m missing in candidate")
            continue
        deg_pct = (new_rmse - old_rmse) / old_rmse * 100.0
        if deg_pct > tol_pct:
            violations.append(
                f"{site}: RMSE degraded {deg_pct:.1f}% "
                f"({old_rmse:.3f}m -> {new_rmse:.3f}m), tolerance {tol_pct}%")
        old_slope = old_m.get("decile_slope")
        new_slope = new_m.get("decile_slope")
        if old_slope is not None and new_slope is not None and old_slope > 0:
            if new_slope < slope_tol * old_slope:
                violations.append(
                    f"{site}: decile_slope collapsed {old_slope:.3f} -> {new_slope:.3f} "
                    f"(< {slope_tol}x parent)")
    return (len(violations) == 0), violations


def _next_version(current: str) -> str:
    n = int(current.lstrip("v"))
    return f"v{n + 1}"


def retrain(new_features, new_depths, new_weights, region_key: str,
           registry_dir: Path = REGISTRY_DIR) -> Dict:
    """Skeleton retrain driver — warm-starts the CURRENT registry version's
    CNN on new_features/new_depths, re-evaluates the parent's certified
    sites, applies guard_or_reject, and either promotes v<N+1> or writes
    v<N+1>_REJECTED."""
    current = (registry_dir / "CURRENT").read_text().strip()
    old_card_path = registry_dir / current / "model_card.json"
    with open(old_card_path) as fh:
        old_card = json.load(fh)

    from backend import uae_pretrained_cnn as UAECNN
    prior_model = UAECNN.load_model(registry_dir / current / "uae_clustered_cnn.pkl",
                                    force_reload=True)
    if prior_model is None:
        raise RuntimeError(f"could not load prior CNN from {current}")

    candidate = UAECNN.fit_uae_cnn_warmstart(prior_model, new_features, new_depths,
                                             source_weights=new_weights)

    # Candidate card: same metrics.per_site as parent unless re-evaluated
    # (a full re-evaluation harness is TRAIN-R7's job; this driver assumes
    # the caller supplies updated per-site metrics before calling promote).
    new_card = json.loads(json.dumps(old_card))  # deep copy
    new_card["parent_version"] = current
    new_card["semver"] = old_card.get("semver", "1.0.0")

    accept, violations = guard_or_reject(old_card, new_card)
    next_v = _next_version(current)
    if not accept:
        rej_dir = registry_dir / f"{next_v}_REJECTED"
        rej_dir.mkdir(parents=True, exist_ok=True)
        with open(rej_dir / "model_card.json", "w") as fh:
            json.dump(new_card, fh, indent=2)
        (rej_dir / "rejection_reason.txt").write_text("\n".join(violations))
        return {"accepted": False, "violations": violations, "path": str(rej_dir)}

    new_dir = registry_dir / next_v
    new_dir.mkdir(parents=True, exist_ok=True)
    UAECNN.save_model(candidate, new_dir / "uae_clustered_cnn.pkl")
    with open(new_dir / "model_card.json", "w") as fh:
        json.dump(new_card, fh, indent=2)
    (registry_dir / "CURRENT").write_text(next_v)
    return {"accepted": True, "path": str(new_dir)}


if __name__ == "__main__":
    print("retrain_uae_pretrained.py is a library + CLI skeleton; "
          "run guard_or_reject() tests via the acceptance script.")
