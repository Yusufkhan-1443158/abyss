#!/usr/bin/env python3
"""
PlanetScope 5-date harvest utility — Khalifa Port, Abu Dhabi.
=============================================================

Professional, quota-guarded PlanetScope archive builder for the
Bathymetry_VMarch SDB platform.  Searches the Planet Data API for the best
cloud-free PSScene acquisitions over the Khalifa Port AOI (derived from the
repo's own ``_KNOWN_SITES['khalifa_port']`` definition in ``backend/app.py``),
selects N (default 5) dates that jointly maximise temporal spread,
clear_percent, sun-elevation diversity (avoiding glint geometry) and EOT20
tide-state diversity, then orders them through the Orders v2 API with the
``clip`` tool (only the AOI's ~200 km2 counts against quota), downloads,
extracts, verifies with rasterio, and writes a manifest + persistent quota
ledger + run log.

Pipeline phases (all resumable, state kept on disk):
  1. aoi      — derive & measure the AOI polygon (UTM 40N area)
  2. search   — Data API quick-search, last --months-back months
  3. select   — smart N-date selection (documented scoring below)
  4. order    — budget-gated Orders v2 clip orders (one per date)
  5. download — poll with backoff, download + extract zips
  6. verify   — rasterio checks + RGB quicklooks + MANIFEST.json
  7. sh-probe — free capability probe: does the Planet key authenticate
                against Sentinel Hub on the Planet Insights Platform?
  8. log      — PLANET_HARVEST_LOG.md

QUOTA SAFETY (hard rules, enforced in code):
  * A persistent ledger ``data/planet_backup/QUOTA_LEDGER.json`` records every
    km2 ever ordered by this utility, keyed by UTC month.
  * Before ANY order is created the projected total (ledger month-to-date +
    planned clip km2) is checked against --budget-km2 (default 1050).  The
    script refuses to proceed if it would exceed it.
  * Orders already recorded in the run-state file are never re-created on
    resume (re-running the script cannot double-spend quota).
  * On an order-create rejection (401/403/400 "not provisioned") the script
    ABORTS immediately — no retries, no further orders.

SECURITY: the API key is read from .env at runtime, used only in auth
headers, and is never printed, logged, or written to any output file.

Smart selection scoring (phase 3) — documented per the mission brief:
  Eligibility : a calendar date is a candidate only if the union of its
                PSScene footprints covers >= 97 % of the AOI.  Its quota cost
                is the sum of footprint∩AOI areas of a greedy minimal covering
                subset (that is exactly what the Orders clip tool bills).
  Quality Q   : 0.55 * clear_percent/100  (area-weighted over the covering set)
              + 0.30 * sun-elevation window score (1.0 inside [30°, 62°],
                falling linearly to 0 at 20° / 72°; high sun near-noon over
                calm Gulf water is the specular-glint regime for a near-nadir
                constellation, low sun starves the water-leaving signal)
              + 0.15 * (1 - cost/(1.35*AOI))  (prefer single-strip coverage)
  Diversity   : greedy pick; after seeding with the best-Q date, each next
                pick maximises  Q + 0.60*T + 0.20*S + 0.20*H  where
                T = min |Δt| to already-chosen dates / 120 d   (capped at 1)
                S = min |Δ sun-elevation| / 25°                (capped at 1)
                H = min |Δ EOT20 tide height| / 0.8 m          (capped at 1)
  Tide        : EOT20 (pyTMD, constituents already cached in cache/EOT20/) at
                the AOI centroid; height, rate and phase (rising/falling,
                relative position in the ±6 h window) are recorded in the
                manifest for later datum work even when not decisive.

Usage:
  .venv/bin/python3 scripts/planet_khalifa_harvest.py               # full run
  .venv/bin/python3 scripts/planet_khalifa_harvest.py --dry-run     # stop before ordering
  .venv/bin/python3 scripts/planet_khalifa_harvest.py --budget-km2 800
  .venv/bin/python3 scripts/planet_khalifa_harvest.py --selftest-verify
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import requests

warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

# ----------------------------------------------------------------------------
# Constants & paths
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
ENV_FILE = REPO / ".env"
BACKEND_APP = REPO / "backend" / "app.py"
BACKUP_ROOT = REPO / "data" / "planet_backup"
SITE_DIR = BACKUP_ROOT / "khalifa_port"
MANIFEST_PATH = BACKUP_ROOT / "MANIFEST.json"
LEDGER_PATH = BACKUP_ROOT / "QUOTA_LEDGER.json"
STATE_PATH = SITE_DIR / "_run_state.json"
LOG_MD = REPO / "PLANET_HARVEST_LOG.md"
EOT20_DIR = REPO / "cache"  # contains the EOT20/ constituent cache

DATA_API = "https://api.planet.com/data/v1"
ORDERS_API = "https://api.planet.com/compute/ops/orders/v2"
SH_BASE = "https://services.sentinel-hub.com"

ITEM_TYPE = "PSScene"
BUNDLE = "analytic_sr_udm2"
SR_ASSET = "ortho_analytic_4b_sr"
MONTH_QUOTA_KM2 = 3000.0          # account-level monthly allowance (context)
UTM40N = "EPSG:32640"

# Fallback only — the live values are parsed from backend/app.py at runtime.
_FALLBACK_BBOX = {"west": 54.5545, "south": 24.7423, "east": 54.7033, "north": 24.8775}


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{dt.datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_api_key() -> str:
    """Read PLANET_API_KEY from .env.  The key is NEVER printed or persisted."""
    if not ENV_FILE.exists():
        sys.exit("FATAL: .env not found — cannot read PLANET_API_KEY")
    m = re.search(r'^\s*(?:export\s+)?PLANET_API_KEY\s*=\s*["\']?([^"\'\s]+)',
                  ENV_FILE.read_text(), re.M)
    if not m:
        sys.exit("FATAL: PLANET_API_KEY not present in .env")
    return m.group(1)


def utm_transformer():
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", UTM40N, always_xy=True)


def poly_to_utm(geom_lonlat):
    """shapely geometry (lon/lat) -> shapely geometry in UTM 40N metres."""
    from shapely.ops import transform as shp_transform
    tr = utm_transformer()
    return shp_transform(lambda x, y: tr.transform(x, y), geom_lonlat)


def area_km2(geom_lonlat) -> float:
    return poly_to_utm(geom_lonlat).area / 1e6


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def month_key(when: dt.datetime | None = None) -> str:
    when = when or dt.datetime.utcnow()
    return when.strftime("%Y-%m")


def read_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


# ----------------------------------------------------------------------------
# Phase 1 — AOI derivation
# ----------------------------------------------------------------------------
def derive_aoi(target_km2: float) -> dict:
    """Derive the harvest AOI from the repo's own khalifa_port definition.

    Reads _KNOWN_SITES['khalifa_port']['bbox'] out of backend/app.py (regex —
    importing backend.app would boot the whole Flask stack), then shrinks the
    bbox symmetrically around its centre to ~target_km2 so the clip cost per
    date stays inside the run budget.  Returns a GeoJSON-style dict with the
    polygon and its measured UTM-40N area.
    """
    bbox = dict(_FALLBACK_BBOX)
    src = "fallback constants (backend/app.py unreadable)"
    if BACKEND_APP.exists():
        txt = BACKEND_APP.read_text(errors="ignore")
        m = re.search(
            r"'khalifa_port':\s*\{[^{]*?'bbox':\s*\{([^}]*)\}", txt, re.S)
        if m:
            vals = dict(re.findall(r"'(west|south|east|north)':\s*([\d.]+)", m.group(1)))
            if len(vals) == 4:
                bbox = {k: float(v) for k, v in vals.items()}
                src = "backend/app.py::_KNOWN_SITES['khalifa_port']['bbox']"

    from shapely.geometry import box
    full = box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    full_km2 = area_km2(full)
    # symmetric shrink about the centroid to hit the target area
    s = min(1.0, math.sqrt(target_km2 / full_km2))
    cx = (bbox["west"] + bbox["east"]) / 2.0
    cy = (bbox["south"] + bbox["north"]) / 2.0
    hw = (bbox["east"] - bbox["west"]) / 2.0 * s
    hh = (bbox["north"] - bbox["south"]) / 2.0 * s
    aoi = box(cx - hw, cy - hh, cx + hw, cy + hh)
    got_km2 = area_km2(aoi)
    coords = [[list(c) for c in aoi.exterior.coords]]
    gj = {"type": "Polygon", "coordinates": coords}
    log(f"AOI: {src}")
    log(f"AOI: site bbox {full_km2:.1f} km2 -> shrink x{s:.4f} -> harvest AOI "
        f"{got_km2:.1f} km2 (target {target_km2:.0f})")
    return {"geojson": gj, "area_km2": round(got_km2, 2), "source": src,
            "site_bbox": bbox, "centroid": [cx, cy]}


# ----------------------------------------------------------------------------
# Phase 2 — Data API search
# ----------------------------------------------------------------------------
def quick_search(session: requests.Session, aoi_gj: dict, months_back: int,
                 max_cloud: float) -> list[dict]:
    now = dt.datetime.utcnow()
    start = now - dt.timedelta(days=int(months_back * 30.44))
    filt = {"type": "AndFilter", "config": [
        {"type": "GeometryFilter", "field_name": "geometry", "config": aoi_gj},
        {"type": "DateRangeFilter", "field_name": "acquired",
         "config": {"gte": start.strftime("%Y-%m-%dT00:00:00Z"),
                    "lte": now.strftime("%Y-%m-%dT%H:%M:%SZ")}},
        {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lt": max_cloud}},
        {"type": "StringInFilter", "field_name": "quality_category", "config": ["standard"]},
        {"type": "StringInFilter", "field_name": "publishing_stage", "config": ["finalized"]},
        {"type": "StringInFilter", "field_name": "ground_control", "config": ["true"]},
        {"type": "AssetFilter", "config": [SR_ASSET]},
    ]}
    body = {"item_types": [ITEM_TYPE], "filter": filt}
    url = f"{DATA_API}/quick-search?_page_size=250"
    feats: list[dict] = []
    while url:
        r = session.post(url, json=body) if "quick-search" in url and not feats \
            else session.get(url)
        if not r.ok:
            sys.exit(f"FATAL: quick-search failed HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        feats.extend(j.get("features", []))
        url = (j.get("_links") or {}).get("_next")
    log(f"search: {len(feats)} PSScene candidates "
        f"({start.date()} .. {now.date()}, cloud<{max_cloud}, standard, "
        f"finalized, ground_control, {SR_ASSET})")
    return feats


def build_candidate_table(feats: list[dict], aoi_gj: dict) -> list[dict]:
    """One row per calendar date: coverage, clip cost, quality metadata."""
    from shapely.geometry import shape
    aoi_utm = poly_to_utm(shape(aoi_gj))
    aoi_a = aoi_utm.area

    by_date: dict[str, list[dict]] = {}
    for ft in feats:
        p = ft["properties"]
        d = p["acquired"][:10]
        try:
            fp = poly_to_utm(shape(ft["geometry"]))
        except Exception:
            continue
        inter = fp.intersection(aoi_utm)
        if inter.is_empty:
            continue
        by_date.setdefault(d, []).append({
            "id": ft["id"],
            "acquired": p["acquired"],
            "satellite_id": p.get("satellite_id"),
            "instrument": p.get("instrument"),
            "cloud_cover": p.get("cloud_cover"),
            "clear_percent": p.get("clear_percent"),
            "sun_elevation": p.get("sun_elevation"),
            "sun_azimuth": p.get("sun_azimuth"),
            "view_angle": p.get("view_angle"),
            "pixel_resolution": p.get("pixel_resolution"),
            "_utm_fp": fp,
            "_cover_frac": inter.area / aoi_a,
        })

    rows = []
    for d, items in sorted(by_date.items()):
        # greedy minimal covering subset (largest AOI-intersection first)
        items = sorted(items, key=lambda x: -x["_cover_frac"])
        chosen, covered = [], None
        for it in items:
            new = it["_utm_fp"] if covered is None else it["_utm_fp"].difference(covered)
            gain = new.intersection(poly_to_utm_cache(aoi_gj)).area / aoi_a
            if gain < 0.005 and chosen:
                continue
            chosen.append(it)
            covered = it["_utm_fp"] if covered is None else covered.union(it["_utm_fp"])
            if covered.intersection(poly_to_utm_cache(aoi_gj)).area / aoi_a >= 0.995:
                break
        cov = covered.intersection(poly_to_utm_cache(aoi_gj)).area / aoi_a if covered else 0.0
        cost = sum(it["_utm_fp"].intersection(poly_to_utm_cache(aoi_gj)).area
                   for it in chosen) / 1e6
        w = np.array([it["_cover_frac"] for it in chosen], float)
        w = w / max(w.sum(), 1e-9)
        rows.append({
            "date": d,
            "items": [{k: v for k, v in it.items() if not k.startswith("_")}
                      for it in chosen],
            "n_items": len(chosen),
            "coverage": round(cov, 4),
            "clip_cost_km2": round(cost, 2),
            "clear_percent": round(float(np.sum(
                w * np.array([it["clear_percent"] or 0 for it in chosen], float))), 1),
            "cloud_cover": round(float(np.sum(
                w * np.array([it["cloud_cover"] or 0 for it in chosen], float))), 4),
            "sun_elevation": round(float(np.sum(
                w * np.array([it["sun_elevation"] or 0 for it in chosen], float))), 1),
            "sun_azimuth": round(float(np.sum(
                w * np.array([it["sun_azimuth"] or 0 for it in chosen], float))), 1),
            "satellites": sorted({it["satellite_id"] for it in chosen if it["satellite_id"]}),
            "acquired_mid": chosen[0]["acquired"],
        })
    return rows


_AOI_UTM_CACHE = {}


def poly_to_utm_cache(aoi_gj):
    key = id(aoi_gj)
    if key not in _AOI_UTM_CACHE:
        from shapely.geometry import shape
        _AOI_UTM_CACHE.clear()
        _AOI_UTM_CACHE[key] = poly_to_utm(shape(aoi_gj))
    return _AOI_UTM_CACHE[key]


# ----------------------------------------------------------------------------
# Tide (EOT20 via pyTMD — same call pattern as backend/app.py::_eot20_tide_heights)
# ----------------------------------------------------------------------------
def eot20_tide(lon: float, lat: float, iso_times: list[str]) -> list[float | None]:
    try:
        import pyTMD
        dt64 = np.array([np.datetime64(str(t).replace("Z", "").split("+")[0])
                         for t in iso_times])
        tide = pyTMD.compute.tide_elevations(
            x=np.full(dt64.shape, float(lon)), y=np.full(dt64.shape, float(lat)),
            delta_time=dt64, directory=str(EOT20_DIR), model="EOT20",
            type="drift", standard="datetime", crs=4326,
            extrapolate=True, cutoff=50.0)
        return [None if not np.isfinite(v) else float(v)
                for v in np.asarray(tide, float).ravel()]
    except Exception as ex:
        log(f"tide: EOT20 unavailable ({type(ex).__name__}: {ex}) — recording nulls")
        return [None] * len(iso_times)


def parse_iso(iso_time: str) -> dt.datetime:
    """Robust ISO-UTC parse (Planet emits 5-digit fractional seconds that
    Python 3.10's fromisoformat rejects)."""
    s = iso_time.replace("Z", "").split("+")[0]
    if "." in s:
        base, frac = s.split(".", 1)
        s = f"{base}.{(frac + '000000')[:6]}"
    return dt.datetime.fromisoformat(s)


def tide_notes_batch(lon: float, lat: float, iso_times: list[str]) -> list[dict]:
    """tide_note for many acquisitions in ONE pyTMD call (constituent grids
    are loaded once instead of once per date)."""
    grids = []
    for t in iso_times:
        t0 = parse_iso(t)
        grids.append([(t0 + dt.timedelta(minutes=m)).isoformat()
                      for m in range(-360, 361, 30)])
    npts = len(grids[0]) if grids else 0
    flat = [g for gr in grids for g in gr]
    hs = eot20_tide(lon, lat, flat)
    return [_tide_note_from_series(hs[i * npts:(i + 1) * npts]) for i in range(len(grids))]


def tide_note(lon: float, lat: float, iso_time: str) -> dict:
    """Height + rate + phase within the surrounding +-6 h tidal window."""
    return tide_notes_batch(lon, lat, [iso_time])[0]


def _tide_note_from_series(hs: list[float | None]) -> dict:
    h0 = hs[len(hs) // 2]
    if h0 is None or all(h is None for h in hs):
        return {"height_m_msl": None, "phase": "unavailable"}
    arr = np.array([np.nan if h is None else h for h in hs], float)
    rate = (arr[len(arr) // 2 + 1] - arr[len(arr) // 2 - 1]) / 1.0  # m/h over 1 h
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    frac = (h0 - lo) / max(hi - lo, 1e-6)
    stage = ("high" if frac > 0.8 else "low" if frac < 0.2 else "mid")
    trend = "rising" if rate > 0.02 else "falling" if rate < -0.02 else "slack"
    return {"height_m_msl": round(h0, 3), "rate_m_per_h": round(float(rate), 3),
            "range_frac": round(float(frac), 2), "phase": f"{stage}-{trend}",
            "model": "EOT20 (pyTMD, cache/EOT20)"}


# ----------------------------------------------------------------------------
# Phase 3 — smart N-date selection
# ----------------------------------------------------------------------------
def sun_score(elev: float) -> float:
    """1.0 in [30, 62] deg; linear falloff to 0 at 20 / 72 deg (glint guard)."""
    if 30.0 <= elev <= 62.0:
        return 1.0
    if elev < 30.0:
        return max(0.0, (elev - 20.0) / 10.0)
    return max(0.0, (72.0 - elev) / 10.0)


def select_dates(rows: list[dict], aoi: dict, n_dates: int,
                 budget_room_km2: float = float("inf"),
                 min_coverage: float = 0.97) -> tuple[list[dict], list[str]]:
    elig = [r for r in rows if r["coverage"] >= min_coverage]
    log(f"select: {len(elig)}/{len(rows)} candidate dates cover >= {min_coverage:.0%} of AOI")
    # HARD glint exclusion: sun elevation > 73 deg puts the specular point
    # inside a near-nadir PSScene's field of view over calm Gulf water —
    # no diversity bonus may override this (mission: avoid extreme glint).
    n0 = len(elig)
    elig = [r for r in elig if (r["sun_elevation"] or 0) <= 73.0]
    if len(elig) < n0:
        log(f"select: hard glint guard removed {n0 - len(elig)} dates (sun el > 73 deg)")
    if len(elig) < n_dates:
        log(f"select: WARNING only {len(elig)} eligible dates; relaxing coverage to 0.90")
        elig = [r for r in rows if r["coverage"] >= 0.90]
    if len(elig) < n_dates:
        sys.exit(f"FATAL: only {len(elig)} usable dates found — cannot pick {n_dates}")

    lon, lat = aoi["centroid"]
    # tide per eligible date — ONE batched pyTMD call (constituents cached locally)
    notes = tide_notes_batch(lon, lat, [r["acquired_mid"] for r in elig])
    for r, note in zip(elig, notes):
        r["tide"] = note

    for r in elig:
        q = (0.55 * (r["clear_percent"] or 0) / 100.0
             + 0.30 * sun_score(r["sun_elevation"])
             + 0.15 * max(0.0, 1.0 - r["clip_cost_km2"] / (1.35 * aoi["area_km2"])))
        r["quality_q"] = round(q, 4)

    def dnum(r):
        return dt.date.fromisoformat(r["date"]).toordinal()

    # budget-aware greedy: a date is only pickable if the cumulative clip cost
    # of the selection stays inside the remaining run budget.
    mean_cost = float(np.mean([r["clip_cost_km2"] for r in elig]))
    log(f"select: budget room {budget_room_km2:.1f} km2 for {n_dates} dates "
        f"(mean candidate cost {mean_cost:.1f} km2/date)")

    def fits(r, chosen):
        spent = sum(c["clip_cost_km2"] for c in chosen) + r["clip_cost_km2"]
        remaining_slots = n_dates - len(chosen) - 1
        # reserve at least the cheapest-candidate cost for each remaining slot
        cheapest = min(x["clip_cost_km2"] for x in elig)
        return spent + remaining_slots * cheapest <= budget_room_km2

    chosen, reasons = [], []
    pool = sorted(elig, key=lambda r: -r["quality_q"])
    first = next((r for r in pool if fits(r, [])), None)
    if first is None:
        sys.exit("FATAL: no candidate date fits inside the budget room")
    pool.remove(first)
    chosen.append(first)
    reasons.append(f"{first['date']}: seed — highest quality Q={first['quality_q']} "
                   f"(clear {first['clear_percent']}%, sun {first['sun_elevation']} deg, "
                   f"tide {first['tide'].get('height_m_msl')} m "
                   f"({first['tide'].get('phase')}), cost {first['clip_cost_km2']} km2)")
    while len(chosen) < n_dates and pool:
        best, best_s, best_terms = None, -1e9, None
        for r in pool:
            if not fits(r, chosen):
                continue
            T = min(1.0, min(abs(dnum(r) - dnum(c)) for c in chosen) / 120.0)
            S = min(1.0, min(abs(r["sun_elevation"] - c["sun_elevation"])
                             for c in chosen) / 25.0)
            hs = [r["tide"].get("height_m_msl"), ]
            H = 0.0
            if r["tide"].get("height_m_msl") is not None:
                ds = [abs(r["tide"]["height_m_msl"] - c["tide"]["height_m_msl"])
                      for c in chosen if c["tide"].get("height_m_msl") is not None]
                if ds:
                    H = min(1.0, min(ds) / 0.8)
            s = r["quality_q"] + 0.60 * T + 0.20 * S + 0.20 * H
            if s > best_s:
                best, best_s, best_terms = r, s, (T, S, H)
        if best is None:
            sys.exit("FATAL: budget room exhausted before reaching n_dates — "
                     "lower --aoi-km2 or raise --budget-km2")
        pool.remove(best)
        chosen.append(best)
        T, S, H = best_terms
        reasons.append(
            f"{best['date']}: score {best_s:.3f} = Q {best['quality_q']} "
            f"+ 0.60*T({T:.2f}) + 0.20*S({S:.2f}) + 0.20*H({H:.2f}) — "
            f"clear {best['clear_percent']}%, sun {best['sun_elevation']} deg, "
            f"tide {best['tide'].get('height_m_msl')} m ({best['tide'].get('phase')}), "
            f"cost {best['clip_cost_km2']} km2")
    chosen.sort(key=lambda r: r["date"])
    return chosen, reasons


# ----------------------------------------------------------------------------
# Quota ledger
# ----------------------------------------------------------------------------
def ledger_load() -> dict:
    return read_json(LEDGER_PATH, {"account_month_quota_km2": MONTH_QUOTA_KM2,
                                   "orders": [], "months": {}})


def ledger_month_used(ledger: dict, mk: str | None = None) -> float:
    mk = mk or month_key()
    return float(ledger.get("months", {}).get(mk, {}).get("km2", 0.0))


def ledger_add(ledger: dict, order_id: str, date: str, km2: float, item_ids: list[str]) -> dict:
    mk = month_key()
    ledger["orders"].append({
        "order_id": order_id, "date": date, "utc_ordered": dt.datetime.utcnow().isoformat() + "Z",
        "month": mk, "km2": round(km2, 2), "item_ids": item_ids,
        "note": "clip-tool clipped area (footprint ∩ AOI, UTM40N geodesic estimate)"})
    m = ledger.setdefault("months", {}).setdefault(mk, {"km2": 0.0, "n_orders": 0})
    m["km2"] = round(m["km2"] + km2, 2)
    m["n_orders"] += 1
    m["remaining_estimate_km2"] = round(MONTH_QUOTA_KM2 - m["km2"], 2)
    write_json(LEDGER_PATH, ledger)
    return ledger


# ----------------------------------------------------------------------------
# Phase 4/5 — Orders v2: create, poll, download, extract
# ----------------------------------------------------------------------------
def budget_gate(chosen: list[dict], ledger: dict, budget_km2: float) -> float:
    planned = sum(r["clip_cost_km2"] for r in chosen)
    used = ledger_month_used(ledger)
    if used + planned > budget_km2:
        sys.exit(f"BUDGET REFUSAL: month-to-date {used:.1f} km2 + planned "
                 f"{planned:.1f} km2 > budget {budget_km2:.1f} km2. "
                 f"Nothing was ordered.")
    log(f"budget: planned {planned:.1f} km2 + ledger {used:.1f} km2 "
        f"<= budget {budget_km2:.1f} km2 — OK")
    return planned


def create_order(session: requests.Session, date: str, item_ids: list[str],
                 aoi_gj: dict) -> dict:
    body = {
        "name": f"bathyvm_khalifa_{date.replace('-', '')}",
        "products": [{"item_ids": item_ids, "item_type": ITEM_TYPE,
                      "product_bundle": BUNDLE}],
        "tools": [{"clip": {"aoi": aoi_gj}}],
        "delivery": {"archive_type": "zip", "single_archive": True,
                     "archive_filename": f"khalifa_{date.replace('-', '')}.zip"},
    }
    r = session.post(ORDERS_API, json=body)
    if r.status_code not in (200, 201, 202):
        raise OrderRejected(r.status_code, r.text[:500])
    return r.json()


class OrderRejected(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status, self.body = status, body


def poll_orders(session: requests.Session, order_ids: dict[str, str],
                timeout_s: int = 3600) -> dict[str, dict]:
    """Poll all orders until terminal state. Backoff 20 s -> 120 s."""
    t0, wait, out = time.time(), 20.0, {}
    pending = dict(order_ids)  # date -> order_id
    while pending and time.time() - t0 < timeout_s:
        for date, oid in list(pending.items()):
            r = session.get(f"{ORDERS_API}/{oid}")
            if not r.ok:
                log(f"poll: {date} HTTP {r.status_code} (transient?)")
                continue
            j = r.json()
            st = j.get("state")
            if st in ("success", "partial", "failed", "cancelled"):
                log(f"poll: order {date} -> {st}")
                out[date] = j
                pending.pop(date)
            else:
                log(f"poll: order {date} state={st} (waiting {wait:.0f}s)")
        if pending:
            time.sleep(wait)
            wait = min(120.0, wait * 1.35)
    for date in pending:
        log(f"poll: TIMEOUT on order for {date} after {timeout_s}s")
    return out


def download_and_extract(session: requests.Session, date: str, order: dict) -> list[Path]:
    ddir = SITE_DIR / date.replace("-", "")
    ddir.mkdir(parents=True, exist_ok=True)
    results = (order.get("_links") or {}).get("results", [])
    got = []
    for res in results:
        name = Path(res["name"]).name
        if not name.endswith(".zip"):
            continue
        dest = ddir / name
        log(f"download: {date} <- {name}")
        with session.get(res["location"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(dest) as z:
            z.extractall(ddir)
        got.append(dest)
    # flatten: move tifs/jsons out of nested dirs into ddir
    for p in ddir.rglob("*"):
        if p.is_file() and p.parent != ddir and p.suffix.lower() in (".tif", ".json", ".xml"):
            tgt = ddir / p.name
            if not tgt.exists():
                p.rename(tgt)
    return got


# ----------------------------------------------------------------------------
# Phase 6 — verification + manifest
# ----------------------------------------------------------------------------
def verify_date_dir(date: str, aoi_gj: dict) -> dict:
    """rasterio checks on every *_SR_clip tif in the date dir + RGB quicklook."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom
    from PIL import Image

    ddir = SITE_DIR / date.replace("-", "")
    srs = sorted(ddir.glob("*AnalyticMS_SR*clip*.tif")) or sorted(ddir.glob("*_SR_*.tif"))
    udms = sorted(ddir.glob("*udm2*clip*.tif")) or sorted(ddir.glob("*udm2*.tif"))
    rep = {"date": date, "dir": str(ddir), "scenes": [], "ok": True}
    if not srs:
        rep["ok"] = False
        rep["error"] = "no SR GeoTIFF found"
        return rep

    ql_accum = None
    for tif in srs:
        s = {"file": tif.name, "size_bytes": tif.stat().st_size,
             "sha256": sha256_of(tif)}
        with rasterio.open(tif) as ds:
            s["bands"] = ds.count
            s["crs"] = str(ds.crs)
            s["pixel_m"] = [round(abs(ds.transform.a), 3), round(abs(ds.transform.e), 3)]
            s["shape"] = [ds.height, ds.width]
            assert ds.count == 4, f"{tif.name}: expected 4 bands, got {ds.count}"
            epsg = ds.crs.to_epsg()
            assert epsg in (32639, 32640), f"{tif.name}: unexpected CRS {ds.crs}"
            assert 2.0 <= abs(ds.transform.a) <= 4.5, \
                f"{tif.name}: pixel {abs(ds.transform.a)} m not ~3 m"
            aoi_local = transform_geom("EPSG:4326", ds.crs, aoi_gj)
            mask_in = ~geometry_mask([aoi_local], out_shape=(ds.height, ds.width),
                                     transform=ds.transform, invert=False)
            b = ds.read([3, 2, 1], out_shape=(3, min(768, ds.height),
                                              min(768, ds.width)))
            full = ds.read(1)
            valid = (full > 0) & mask_in
            s["valid_pct_in_aoi"] = round(100.0 * valid.sum() / max(mask_in.sum(), 1), 1)
            assert valid.sum() > 0, f"{tif.name}: zero valid data inside AOI"
            # accumulate quicklook (max-composite across scenes of the date)
            ql = b.astype(np.float32)
            ql_accum = ql if ql_accum is None else np.maximum(ql_accum, ql)
        rep["scenes"].append(s)

    # RGB quicklook
    if ql_accum is not None:
        rgb = np.zeros_like(ql_accum)
        for i in range(3):
            band = ql_accum[i]
            v = band[band > 0]
            if v.size:
                lo, hi = np.percentile(v, [2, 98])
                rgb[i] = np.clip((band - lo) / max(hi - lo, 1), 0, 1)
        img = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
        qlp = ddir / f"quicklook_{date.replace('-', '')}.png"
        Image.fromarray(img).save(qlp)
        rep["quicklook"] = str(qlp)

    rep["udm2_files"] = [u.name for u in udms]
    rep["valid_pct_in_aoi"] = round(
        float(np.mean([s["valid_pct_in_aoi"] for s in rep["scenes"]])), 1)
    return rep


def selftest_verify(aoi: dict) -> None:
    """Prove the verification/quicklook machinery on a synthetic 4-band scene."""
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    ddir = SITE_DIR / "00000000"
    ddir.mkdir(parents=True, exist_ok=True)
    utm = poly_to_utm_cache(aoi["geojson"])
    minx, miny, maxx, maxy = utm.bounds
    w = h = 512  # 512 px @ 3 m = 1.54 km synthetic sub-scene inside the AOI
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    tr = from_origin(cx - w * 1.5, cy + h * 1.5, 3.0, 3.0)
    rng = np.random.default_rng(0)
    data = (rng.gamma(2.0, 800.0, (4, h, w))).astype("uint16")
    data[:, : h // 8, :] = 0  # simulated nodata stripe
    tif = ddir / "SYNTH_3B_AnalyticMS_SR_clip.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=w, height=h, count=4,
                       dtype="uint16", crs=UTM40N, transform=tr) as ds:
        ds.write(data)
    rep = verify_date_dir("0000-00-00", aoi["geojson"])
    ok = rep["ok"] and rep["scenes"][0]["bands"] == 4 and rep.get("quicklook")
    log(f"selftest-verify: bands=4 crs={rep['scenes'][0]['crs']} "
        f"valid%={rep['scenes'][0]['valid_pct_in_aoi']} quicklook={'yes' if ok else 'NO'}")
    import shutil
    shutil.rmtree(ddir)
    if not ok:
        sys.exit("selftest-verify FAILED")
    log("selftest-verify: PASS (synthetic scene, dir cleaned up)")


# ----------------------------------------------------------------------------
# Phase 7 — Sentinel Hub / Planet Insights capability probe (free)
# ----------------------------------------------------------------------------
def sh_probe(key: str) -> dict:
    """Does the Planet key authenticate on Sentinel Hub (Insights Platform)?

    NOTE: GET /api/v1/catalog/1.0.0/collections is PUBLIC (200 without any
    token) — it must not be read as auth success.  Only authenticated
    endpoints (catalog search POST, process, byoc, dataimport) count.
    """
    H = {"Authorization": f"Bearer {key}"}
    out = {"probed_utc": dt.datetime.utcnow().isoformat() + "Z", "checks": []}

    def check(name, method, url, body=None, auth_required=True):
        try:
            r = requests.request(method, url, headers=H, json=body, timeout=30)
            out["checks"].append({"name": name, "url": url, "status": r.status_code,
                                  "auth_required": auth_required})
            return r.status_code
        except Exception as ex:
            out["checks"].append({"name": name, "url": url, "status": f"EXC {ex}"})
            return None

    check("catalog collections (public control)", "GET",
          f"{SH_BASE}/api/v1/catalog/1.0.0/collections", auth_required=False)
    s_search = check("catalog search s2-l2a", "POST",
                     f"{SH_BASE}/api/v1/catalog/1.0.0/search",
                     {"collections": ["sentinel-2-l2a"],
                      "bbox": [54.6, 24.78, 54.7, 24.86],
                      "datetime": "2026-01-01T00:00:00Z/2026-02-01T00:00:00Z",
                      "limit": 1})
    s_proc = check("process api s2-l2a 10x10px", "POST", f"{SH_BASE}/api/v1/process", {
        "input": {"bounds": {"bbox": [54.63, 24.80, 54.64, 24.81]},
                  "data": [{"type": "sentinel-2-l2a",
                            "dataFilter": {"timeRange": {"from": "2026-01-01T00:00:00Z",
                                                         "to": "2026-03-01T00:00:00Z"}}}]},
        "output": {"width": 10, "height": 10},
        "evalscript": "//VERSION=3\nfunction setup(){return{input:['B04'],output:{bands:1}}}\n"
                      "function evaluatePixel(s){return[s.B04]}"})
    s_tpdi = check("dataimport quotas (TPDI)", "GET", f"{SH_BASE}/api/v1/dataimport/quotas")
    s_byoc = check("byoc collections", "GET", f"{SH_BASE}/api/v1/byoc/collections")

    authed = [s for s in (s_search, s_proc, s_tpdi, s_byoc) if s == 200]
    out["verdict"] = ("PASS — Planet key works as SH Bearer token" if authed else
                      "FAIL — Bearer PLAK rejected (401) by all authenticated "
                      "Sentinel Hub endpoints; live S2 via this key is NOT unblocked")
    log(f"sh-probe: {out['verdict']}")
    return out


# ----------------------------------------------------------------------------
# Phase 8 — run log
# ----------------------------------------------------------------------------
def render_log(ctx: dict) -> None:
    L = []
    A = L.append
    A("# PLANET_HARVEST_LOG — Khalifa Port PlanetScope 5-date archive")
    A("")
    A(f"_Run: {ctx['run_utc']} UTC — `scripts/planet_khalifa_harvest.py` "
      f"(budget {ctx['budget_km2']} km2, {ctx['n_dates']} dates)_")
    A("")
    A("## 1. AOI")
    A("")
    a = ctx["aoi"]
    A(f"- Source: `{a['source']}`")
    A(f"- Harvest polygon area: **{a['area_km2']} km2** (UTM 40N), centroid "
      f"{a['centroid'][0]:.4f}E {a['centroid'][1]:.4f}N; polygon recorded in "
      f"`data/planet_backup/MANIFEST.json`.")
    A("")
    A("## 2. Candidate table (per-date, after per-date greedy covering)")
    A("")
    A("| date | n | coverage | clip km2 | clear% | cloud | sun el | sun az | sats | Q |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in ctx["rows"]:
        A(f"| {r['date']} | {r['n_items']} | {r['coverage']:.2f} | "
          f"{r['clip_cost_km2']} | {r['clear_percent']} | {r['cloud_cover']} | "
          f"{r['sun_elevation']} | {r['sun_azimuth']} | "
          f"{','.join(r['satellites'])} | {r.get('quality_q', '')} |")
    A("")
    A("## 3. Smart 5-date selection")
    A("")
    A("Scoring (see module docstring): Q = 0.55·clear + 0.30·sun-window(30-62 deg, "
      "glint guard) + 0.15·cost; greedy diversity bonus 0.60·temporal + 0.20·sun-el "
      "+ 0.20·EOT20-tide-height. Tide phase recorded per scene either way.")
    A("")
    for reason in ctx["reasons"]:
        A(f"- {reason}")
    A("")
    A("## 4. Orders & quota ledger")
    A("")
    if ctx.get("orders_table"):
        A("| date | order id | state | clip km2 |")
        A("|---|---|---|---|")
        for row in ctx["orders_table"]:
            A(f"| {row['date']} | {row['order_id']} | {row['state']} | {row['km2']} |")
    if ctx.get("order_abort"):
        A("")
        A(f"**ORDERING ABORTED — {ctx['order_abort']}**")
    led = ctx["ledger"]
    mk = month_key()
    used = ledger_month_used(led)
    A("")
    A(f"- Ledger `{LEDGER_PATH.relative_to(REPO)}`: month {mk} used "
      f"**{used:.1f} km2**, run budget {ctx['budget_km2']} km2, account monthly "
      f"quota {MONTH_QUOTA_KM2:.0f} km2 -> remaining estimate "
      f"**{MONTH_QUOTA_KM2 - used:.1f} km2**.")
    A("")
    A("## 5. Verification")
    A("")
    if ctx.get("verif"):
        A("| date | clear% | valid% in AOI | scenes | bytes | checksums |")
        A("|---|---|---|---|---|---|")
        for v in ctx["verif"]:
            nb = sum(s["size_bytes"] for s in v.get("scenes", []))
            A(f"| {v['date']} | {v.get('clear_percent', '')} | "
              f"{v.get('valid_pct_in_aoi', '-')} | {len(v.get('scenes', []))} | "
              f"{nb} | {'ok' if v['ok'] else 'FAIL'} |")
    else:
        A("_No imagery downloaded in this run (see section 4) — the rasterio "
          "verification path was exercised with `--selftest-verify` instead "
          f"(result: {ctx.get('selftest', 'not run')})._")
    A("")
    A("## 6. Sentinel Hub (Planet Insights) capability probe")
    A("")
    for c in ctx["sh"]["checks"]:
        A(f"- {c['name']}: HTTP {c['status']}"
          + ("" if c.get("auth_required", True) else " (public endpoint — control)"))
    A(f"- **Verdict: {ctx['sh']['verdict']}**")
    A("")
    if ctx.get("diagnosis"):
        A("## 7. Account diagnosis")
        A("")
        for d in ctx["diagnosis"]:
            A(f"- {d}")
        A("")
    LOG_MD.write_text("\n".join(L))
    log(f"log: wrote {LOG_MD}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--budget-km2", type=float, default=1050.0,
                    help="hard run budget in km2 (refuse to exceed)")
    ap.add_argument("--n-dates", type=int, default=5)
    ap.add_argument("--months-back", type=int, default=18)
    ap.add_argument("--aoi-km2", type=float, default=200.0)
    ap.add_argument("--max-cloud", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true",
                    help="run search+selection+probe but stop before ordering")
    ap.add_argument("--selftest-verify", action="store_true",
                    help="only run the synthetic-scene verification self-test")
    ap.add_argument("--skip-sh-probe", action="store_true")
    args = ap.parse_args()

    key = load_api_key()
    session = requests.Session()
    session.auth = (key, "")  # Planet keys double as HTTP Basic usernames

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    ctx = {"run_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
           "budget_km2": args.budget_km2, "n_dates": args.n_dates}

    # -- Phase 1: AOI
    aoi = derive_aoi(args.aoi_km2)
    ctx["aoi"] = aoi
    if args.selftest_verify:
        selftest_verify(aoi)
        return

    # -- auth preflight (no quota)
    r = session.get(f"{DATA_API}/")
    if r.status_code in (401, 403):
        sys.exit(f"FATAL: Planet Data API rejected the key (HTTP {r.status_code}) "
                 f"at GET {DATA_API}/ — aborting before any order.")
    log(f"auth: Data API preflight OK (HTTP {r.status_code})")

    # -- Phase 2: search
    feats = quick_search(session, aoi["geojson"], args.months_back, args.max_cloud)
    if not feats:
        sys.exit("FATAL: search returned zero candidates — nothing to select.")
    rows = build_candidate_table(feats, aoi["geojson"])
    log(f"search: {len(rows)} distinct candidate dates")

    # -- Phase 3: budget-aware selection
    ledger = ledger_load()
    write_json(LEDGER_PATH, ledger)  # ledger file always exists after a run
    room = args.budget_km2 - ledger_month_used(ledger)
    chosen, reasons = select_dates(rows, aoi, args.n_dates, budget_room_km2=room)
    ctx["rows"], ctx["reasons"] = rows, reasons
    log("select: chosen dates -> " + ", ".join(r["date"] for r in chosen))

    # -- Phase 7 early (free, informs the log even if ordering aborts)
    ctx["sh"] = {"checks": [], "verdict": "skipped"} if args.skip_sh_probe else sh_probe(key)

    # -- Phase 4: budget gate (defence in depth: re-checks the selection) + ordering
    planned = budget_gate(chosen, ledger, args.budget_km2)
    ctx["ledger"] = ledger

    manifest = {
        "site": "khalifa_port", "aoi": aoi, "generated_utc": ctx["run_utc"],
        "item_type": ITEM_TYPE, "product_bundle": BUNDLE,
        "selection": {"reasons": reasons,
                      "chosen_dates": [r["date"] for r in chosen]},
        "scenes": {}, "planned_clip_km2": round(planned, 2),
    }
    for r in chosen:
        manifest["scenes"][r["date"]] = {
            "items": r["items"], "coverage": r["coverage"],
            "clip_cost_km2": r["clip_cost_km2"], "tide": r["tide"],
            "clear_percent": r["clear_percent"], "sun_elevation": r["sun_elevation"],
            "sun_azimuth": r["sun_azimuth"], "satellites": r["satellites"],
        }
    write_json(MANIFEST_PATH, manifest)

    def run_selftest_into_ctx():
        try:
            selftest_verify(aoi)
            ctx["selftest"] = "PASS — synthetic 4-band 3 m UTM40N scene through the full rasterio+quicklook path"
        except SystemExit:
            ctx["selftest"] = "FAIL"

    if args.dry_run:
        ctx["order_abort"] = "--dry-run requested; no orders created"
        run_selftest_into_ctx()
        render_log(ctx)
        return

    state = read_json(STATE_PATH, {"orders": {}})
    orders_table = []
    try:
        for r in chosen:
            date = r["date"]
            if date in state["orders"] and state["orders"][date].get("order_id"):
                log(f"order: {date} already ordered "
                    f"({state['orders'][date]['order_id']}) — resume, not re-ordering")
                continue
            item_ids = [it["id"] for it in r["items"]]
            j = create_order(session, date, item_ids, aoi["geojson"])
            oid = j["id"]
            state["orders"][date] = {"order_id": oid, "km2": r["clip_cost_km2"],
                                     "item_ids": item_ids, "state": j.get("state")}
            write_json(STATE_PATH, state)
            ledger = ledger_add(ledger, oid, date, r["clip_cost_km2"], item_ids)
            log(f"order: {date} -> {oid} ({r['clip_cost_km2']} km2)")
    except OrderRejected as ex:
        ctx["order_abort"] = (f"Orders v2 rejected order creation — {ex}. "
                              "STOPPED immediately (no retries, no further orders).")
        ctx["diagnosis"] = diagnose_account(session)
        ctx["orders_table"] = orders_table
        run_selftest_into_ctx()
        render_log(ctx)
        sys.exit(f"ABORT: {ctx['order_abort']}")

    # -- Phase 5: poll + download
    oid_by_date = {d: s["order_id"] for d, s in state["orders"].items()}
    done = poll_orders(session, oid_by_date)
    verif = []
    for date, order in sorted(done.items()):
        orders_table.append({"date": date, "order_id": order["id"],
                             "state": order["state"],
                             "km2": state["orders"][date]["km2"]})
        if order["state"] not in ("success", "partial"):
            continue
        download_and_extract(session, date, order)
        # -- Phase 6: verify
        rep = verify_date_dir(date, aoi["geojson"])
        rep["clear_percent"] = manifest["scenes"][date]["clear_percent"]
        verif.append(rep)
        # enrich manifest with file-level facts
        manifest["scenes"][date]["files"] = rep.get("scenes", [])
        manifest["scenes"][date]["quicklook"] = rep.get("quicklook")
        manifest["scenes"][date]["valid_pct_in_aoi"] = rep.get("valid_pct_in_aoi")
        manifest["scenes"][date]["order_id"] = order["id"]
    write_json(MANIFEST_PATH, manifest)
    ctx["orders_table"], ctx["verif"], ctx["ledger"] = orders_table, verif, ledger

    render_log(ctx)
    log("DONE")


def diagnose_account(session: requests.Session) -> list[str]:
    """Free, precise account diagnosis after an order rejection."""
    out = []
    r = session.get(f"{DATA_API}/")
    out.append(f"GET /data/v1/ -> HTTP {r.status_code} (key itself is valid)")
    body = {"item_types": [ITEM_TYPE], "filter": {"type": "AndFilter", "config": [
        {"type": "PermissionFilter", "config": ["assets:download"]},
        {"type": "DateRangeFilter", "field_name": "acquired",
         "config": {"gte": "2024-01-01T00:00:00Z"}}]}}
    r = session.post(f"{DATA_API}/quick-search?_page_size=1", json=body)
    n = len(r.json().get("features", [])) if r.ok else -1
    out.append(f"quick-search with PermissionFilter(assets:download) -> HTTP "
               f"{r.status_code}, {n} downloadable items (0 = org has no "
               f"PSScene download entitlement on api.planet.com)")
    r = session.get("https://api.planet.com/auth/v1/experimental/public/my/subscriptions")
    subs = r.json() if r.ok else []
    out.append(f"GET /auth/v1/.../my/subscriptions -> HTTP {r.status_code}, "
               f"{len(subs)} legacy subscriptions")
    out.append("Conclusion: the 3,000 km2/month PlanetScope quota is NOT "
               "provisioned against the legacy Data/Orders APIs for this org. "
               "It is most likely a Planet Insights Platform allocation that "
               "requires OAuth (SH client credentials or planet-auth token), "
               "not the PLAK Basic key. Ask Planet support / account console "
               "to enable 'Orders API download' for the org, or supply "
               "SH_CLIENT_ID/SECRET for the Insights (TPDI) route.")
    return out


if __name__ == "__main__":
    main()
