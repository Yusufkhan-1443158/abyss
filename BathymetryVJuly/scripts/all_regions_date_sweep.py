#!/usr/bin/env python3
"""Sweep monthly S2 dates over each predefined region and report the
best (lowest |bias|, RMSE) date per region. Writes JSON to
backend/calibration_data/per_region_best_dates.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app import _run_s2_lyzenga_fast  # noqa: E402

REGIONS = {
    "khalifa_port":  [54.636, 24.785, 54.690, 24.840],
    "old_mussafah":  [54.355, 24.412, 54.412, 24.466],
    "abu_al_abyad":  [53.852, 24.213, 53.910, 24.267],
    "jbel_dhanna":   [52.5692, 24.1989, 52.6186, 24.2440],
}

DATES = [f"2024-{mm:02d}-15" for mm in range(1, 13)]


def run_one(date_iso, bbox, days_pad=10):
    centre = datetime.strptime(date_iso, "%Y-%m-%d")
    sd = (centre - timedelta(days=days_pad)).strftime("%Y-%m-%d")
    ed = (centre + timedelta(days=days_pad)).strftime("%Y-%m-%d")
    t0 = time.time()
    try:
        out = _run_s2_lyzenga_fast(bbox, sd, ed, user_pts=[], fetch_sliderule=False)
    except Exception as ex:
        return {"date": date_iso, "error": str(ex)[:120],
                "elapsed_s": round(time.time() - t0, 1)}
    m = out.get("metrics") or {}
    s = out.get("stats") or {}
    return {
        "date": date_iso,
        "elapsed_s": round(time.time() - t0, 1),
        "n_test": m.get("n_test"),
        "rmse_m": m.get("rmse_m"),
        "mae_m":  m.get("mae_m"),
        "bias_m": m.get("bias_m"),
        "r2": m.get("r2"),
        "s44_1a_pct": m.get("s44_1a_pct"),
        "mean_depth": s.get("mean_depth"),
        "max_depth":  s.get("max_depth"),
    }


def best_date(rows, target_bias=1.0):
    """Pick the best date: must have n_test >= 5 and |bias| < target.
    Among those, lowest RMSE wins. If none qualifies, fall back to the
    lowest |bias|."""
    valid = [r for r in rows if r.get("rmse_m") is not None and r.get("n_test", 0) >= 5]
    if not valid:
        return None
    qualified = [r for r in valid if abs(r["bias_m"]) < target_bias]
    if qualified:
        qualified.sort(key=lambda r: r["rmse_m"])
        return qualified[0]
    valid.sort(key=lambda r: abs(r["bias_m"]))
    return valid[0]


def main():
    all_results = {}
    for region, bbox in REGIONS.items():
        print(f"\n████████ {region}  bbox={bbox}  ████████")
        rows = []
        for d in DATES:
            r = run_one(d, bbox)
            if r.get("rmse_m") is None:
                print(f"  {d}  no held-out test pts (water mask too small)")
            else:
                print(f"  {d}  RMSE={r['rmse_m']:.2f}m  bias={r['bias_m']:+.3f}m  "
                      f"R²={r['r2']:.3f}  n_test={r['n_test']:>3}  mean={r['mean_depth']}m")
            rows.append(r)
        chosen = best_date(rows, target_bias=1.0)
        all_results[region] = {
            "rows": rows,
            "best": chosen,
        }
        if chosen is not None:
            print(f"\n  → STANDARD DATE for {region} = {chosen['date']}  "
                  f"(RMSE {chosen['rmse_m']:.2f} m, "
                  f"bias {chosen['bias_m']:+.3f} m, "
                  f"R² {chosen['r2']:.3f})")
        else:
            print(f"\n  → no valid date found for {region}")

    # Persist
    out_path = ROOT / "backend" / "calibration_data" / "per_region_best_dates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({k: v["best"] for k, v in all_results.items() if v["best"]},
                  fh, indent=2)
    print(f"\nWrote → {out_path}")
    print(json.dumps({k: v["best"] for k, v in all_results.items() if v["best"]},
                     indent=2))


if __name__ == "__main__":
    main()
