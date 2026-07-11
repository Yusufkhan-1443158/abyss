"""
Persistent SQLite store for i-Boating-digitised soundings.

This is an internal (not client-facing) depth-point cache. Every time we
run the Gemini digitisation pipeline we push the new soundings here,
de-duplicated against the existing rows. Prediction pipelines then RAG
this store to seed the CNN with every sounding we have ever digitised
inside a given bbox — no need to re-query Gemini or re-capture the chart.

The store lives at  backend/ocean/iboating_store.db  (persistent on the
project volume).
"""
from __future__ import annotations
import math
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

_DB_PATH = Path(__file__).parent / "ocean" / "iboating_store.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS soundings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    depth_m       REAL NOT NULL,
    confidence    REAL,
    point_type    TEXT,            -- sounding | contour | colour_sample
    source        TEXT,            -- gemini_z13_north_east, etc.
    tile          TEXT,
    region        TEXT,
    captured_at   REAL,            -- unix epoch
    gemini_model  TEXT
);
CREATE INDEX IF NOT EXISTS idx_soundings_bbox   ON soundings(lat, lon);
CREATE INDEX IF NOT EXISTS idx_soundings_region ON soundings(region);

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    REAL,
    finished_at   REAL,
    region        TEXT,
    bbox          TEXT,
    n_raw         INTEGER,
    n_inserted    INTEGER,
    n_duplicate   INTEGER,
    gemini_model  TEXT,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    region        TEXT,
    bbox          TEXT,
    path          TEXT,
    kind          TEXT,            -- attunet | gbr | patent_mlp
    rmse_val      REAL,
    r2_val        REAL,
    n_train       INTEGER,
    n_val         INTEGER,
    created_at    REAL,
    notes         TEXT
);
"""


def _conn():
    c = sqlite3.connect(str(_DB_PATH))
    c.executescript(_SCHEMA)
    return c


# ── Dedup geo-key ──────────────────────────────────────────────────
def _key(lat: float, lon: float, precision: int = 5) -> str:
    """Round lat/lon to ~1.1 m (5 decimals) so effectively-identical
    points from different pipeline runs dedup cleanly."""
    return f"{round(lat, precision):.{precision}f}_{round(lon, precision):.{precision}f}"


# ── Insert ─────────────────────────────────────────────────────────
def insert_soundings(points: Iterable[dict], region: str,
                      gemini_model: str = "unknown") -> dict:
    """Insert a batch of soundings. Returns {inserted, duplicates, total}."""
    now = time.time()
    inserted = duplicates = total = 0
    seen: set[str] = set()
    with _conn() as c:
        # Preload existing keys for fast dedup
        existing_keys = set()
        for lat, lon in c.execute("SELECT lat, lon FROM soundings WHERE region = ?",
                                    (region,)):
            existing_keys.add(_key(lat, lon))

        rows = []
        for p in points:
            total += 1
            lat = float(p["lat"]); lon = float(p["lon"])
            k = _key(lat, lon)
            if k in existing_keys or k in seen:
                duplicates += 1
                continue
            seen.add(k)
            rows.append((
                lat, lon,
                float(p["depth"]),
                float(p.get("confidence", 0.7)),
                str(p.get("type", "sounding")),
                str(p.get("source", "")),
                str(p.get("tile", "")),
                region,
                now,
                gemini_model,
            ))
            inserted += 1
        if rows:
            c.executemany(
                "INSERT INTO soundings "
                "(lat, lon, depth_m, confidence, point_type, source, tile, "
                " region, captured_at, gemini_model) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
    return {"total": total, "inserted": inserted, "duplicates": duplicates}


# ── RAG retrieval by bbox ──────────────────────────────────────────
def query_bbox(bbox: tuple, region: Optional[str] = None,
                min_confidence: float = 0.5,
                buffer_deg: float = 0.0) -> list[dict]:
    """Return every sounding inside [west, south, east, north] (optionally
    extended by `buffer_deg` on each side). This is the RAG retrieval
    step for the CNN training pipeline."""
    w, s, e, n = bbox
    w -= buffer_deg; e += buffer_deg
    s -= buffer_deg; n += buffer_deg
    q = ("SELECT lat, lon, depth_m, confidence, point_type, source, tile, "
         "       region, captured_at "
         "  FROM soundings "
         " WHERE lat BETWEEN ? AND ? "
         "   AND lon BETWEEN ? AND ? "
         "   AND confidence >= ?")
    args: list = [s, n, w, e, min_confidence]
    if region:
        q += " AND region = ?"
        args.append(region)
    out = []
    with _conn() as c:
        for row in c.execute(q, args):
            out.append({
                "lat": row[0], "lon": row[1], "depth": row[2],
                "confidence": row[3], "type": row[4], "source": row[5],
                "tile": row[6], "region": row[7], "captured_at": row[8],
            })
    return out


def count_points(region: Optional[str] = None) -> int:
    with _conn() as c:
        if region:
            (n,) = c.execute(
                "SELECT COUNT(*) FROM soundings WHERE region = ?", (region,)
            ).fetchone()
        else:
            (n,) = c.execute("SELECT COUNT(*) FROM soundings").fetchone()
    return int(n)


def stats() -> dict:
    with _conn() as c:
        (n_total,) = c.execute("SELECT COUNT(*) FROM soundings").fetchone()
        (n_regions,) = c.execute(
            "SELECT COUNT(DISTINCT region) FROM soundings"
        ).fetchone()
        per_region = [
            {"region": r[0], "count": r[1],
             "depth_min": r[2], "depth_max": r[3], "depth_mean": r[4]}
            for r in c.execute(
                "SELECT region, COUNT(*), MIN(depth_m), MAX(depth_m), AVG(depth_m) "
                "FROM soundings GROUP BY region ORDER BY COUNT(*) DESC"
            )
        ]
    return {"n_total": int(n_total), "n_regions": int(n_regions),
            "per_region": per_region}


# ── Run log ────────────────────────────────────────────────────────
def log_run(region: str, bbox: tuple, n_raw: int, n_inserted: int,
             n_duplicate: int, gemini_model: str = "", notes: str = "") -> int:
    t = time.time()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO runs (started_at, finished_at, region, bbox, n_raw, "
            "                  n_inserted, n_duplicate, gemini_model, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (t, t, region, f"{bbox}", n_raw, n_inserted, n_duplicate,
             gemini_model, notes),
        )
        return int(cur.lastrowid)


# ── Model registry ─────────────────────────────────────────────────
def register_model(region: str, bbox: tuple, path: str, kind: str,
                    rmse_val: float, r2_val: float,
                    n_train: int, n_val: int,
                    notes: str = "") -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO models (region, bbox, path, kind, rmse_val, r2_val, "
            "                     n_train, n_val, created_at, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (region, f"{bbox}", path, kind, rmse_val, r2_val,
             n_train, n_val, time.time(), notes),
        )
        return int(cur.lastrowid)


def latest_model(region: str, kind: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT path, rmse_val, r2_val, n_train, n_val, created_at, notes "
            "  FROM models "
            " WHERE region = ? AND kind = ? "
            " ORDER BY created_at DESC LIMIT 1",
            (region, kind),
        ).fetchone()
    if not row:
        return None
    return {
        "path": row[0], "rmse_val": row[1], "r2_val": row[2],
        "n_train": row[3], "n_val": row[4],
        "created_at": row[5], "notes": row[6],
    }


if __name__ == "__main__":
    print(stats())
