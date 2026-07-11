#!/usr/bin/env python3
"""Sweep one S2 date per month over the Khalifa Port preset and record
the held-out validation metrics for each.

Pulls the same fast-path pipeline the UI uses (_run_s2_lyzenga_fast),
so the numbers reflect what the user sees.

Usage:
    source .venv/bin/activate && source .env
    python scripts/khalifa_date_sweep.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app import _run_s2_lyzenga_fast  # noqa: E402

KHALIFA_BBOX = [54.636, 24.785, 54.690, 24.840]

DATES = [f"2024-{mm:02d}-15" for mm in range(1, 13)]


def run_one(date_iso: str, bbox=KHALIFA_BBOX, days_pad: int = 10) -> dict:
    centre = datetime.strptime(date_iso, "%Y-%m-%d")
    sd = (centre - timedelta(days=days_pad)).strftime("%Y-%m-%d")
    ed = (centre + timedelta(days=days_pad)).strftime("%Y-%m-%d")
    t0 = time.time()
    try:
        out = _run_s2_lyzenga_fast(
            bbox, sd, ed, user_pts=[], fetch_sliderule=False)
    except Exception as ex:
        return {"date": date_iso, "error": str(ex)[:120],
                "elapsed_s": round(time.time() - t0, 1)}
    m = out.get("metrics") or {}
    s = out.get("stats") or {}
    aug = out.get("augmentation") or {}
    bc = aug.get("bias_correction") or {}
    return {
        "date": date_iso,
        "elapsed_s": round(time.time() - t0, 1),
        "n_test": m.get("n_test"),
        "rmse_m": m.get("rmse_m"),
        "mae_m": m.get("mae_m"),
        "bias_m": m.get("bias_m"),
        "r2": m.get("r2"),
        "s44_1a_pct": m.get("s44_1a_pct"),
        "s44_2_pct": m.get("s44_order2_pct"),
        "mean_depth": s.get("mean_depth"),
        "max_depth": s.get("max_depth"),
        "n_scenes": aug.get("n_scenes"),
        "stage1": bc.get("stage1"),
        "post_rmse": bc.get("post_rmse_m"),
        "post_bias": bc.get("post_bias_m"),
    }


def main():
    rows = []
    for d in DATES:
        print(f"\n──── {d} ────")
        r = run_one(d)
        if "error" in r:
            print(f"  FAILED: {r['error']}")
        else:
            print(f"  elapsed {r['elapsed_s']}s  scenes={r['n_scenes']}  "
                  f"n_test={r['n_test']}  stage1={r['stage1']}")
            if r.get('rmse_m') is not None:
                print(f"  RMSE={r['rmse_m']}m  bias={r['bias_m']:+.3f}m  "
                      f"R²={r['r2']}  S-44 1A={r['s44_1a_pct']}%")
            else:
                print(f"  no held-out test pts (water mask too aggressive)")
            print(f"  mean depth {r['mean_depth']}m  max {r['max_depth']}m")
        rows.append(r)

    # Print summary table sorted by |bias|
    valid = [r for r in rows if r.get("bias_m") is not None]
    valid.sort(key=lambda r: abs(r["bias_m"]))
    print("\n=================== Khalifa date sweep ===================")
    print(f"{'date':<12} {'RMSE':>6} {'bias':>7} {'R²':>7} "
          f"{'S-44_1A':>8} {'n_test':>7} {'mean':>6}")
    print("-" * 64)
    for r in rows:
        if r.get("bias_m") is None:
            print(f"{r['date']:<12} FAILED — {r.get('error','?')[:40]}")
            continue
        print(f"{r['date']:<12} {r['rmse_m']:>6.2f} {r['bias_m']:>+7.3f} "
              f"{r['r2']:>7.3f} {r['s44_1a_pct']:>7.1f}% "
              f"{r['n_test']:>7} {r['mean_depth']:>6.2f}")

    if valid:
        best_bias = valid[0]
        best_rmse = sorted(valid, key=lambda r: r["rmse_m"])[0]
        print(f"\nLOWEST |bias|:  {best_bias['date']}  "
              f"bias={best_bias['bias_m']:+.3f}m  RMSE={best_bias['rmse_m']:.2f}m")
        print(f"LOWEST RMSE:   {best_rmse['date']}  "
              f"RMSE={best_rmse['rmse_m']:.2f}m  bias={best_rmse['bias_m']:+.3f}m")

    return rows


if __name__ == "__main__":
    main()
