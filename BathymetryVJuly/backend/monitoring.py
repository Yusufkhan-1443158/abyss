"""
Bathymetry Monitoring Engine — Professional Temporal Analysis
══════════════════════════════════════════════════════════════

Provides:
  1. Historical depth storage (SQLite)
  2. Temporal change detection between epochs
  3. Trend analysis (linear regression per grid cell)
  4. Anomaly detection (significant unexpected changes)
  5. Alert system (configurable thresholds)
  6. Summary statistics & reporting
  7. Export monitoring reports

Designed for continuous coastal monitoring:
  - Dredging impact assessment
  - Sediment transport tracking
  - Storm erosion/deposition detection
  - Port/channel depth compliance
"""
from __future__ import annotations
import json, logging, math, os, sqlite3, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

L = logging.getLogger("bathy.monitoring")
MAX_DEPTH_M = 25.0

DB_DIR = Path("/tmp/bathy/monitoring")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "bathymetry_monitoring.db"


# ═══════════════════════════════════════════════════════════
# DATABASE LAYER
# ═══════════════════════════════════════════════════════════

def _get_db():
    """Get or create the monitoring database."""
    db = sqlite3.connect(str(DB_PATH), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS surveys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_id TEXT UNIQUE NOT NULL,
            bbox_json TEXT NOT NULL,
            date TEXT NOT NULL,
            method TEXT DEFAULT 'cnn',
            resolution_m REAL DEFAULT 10,
            n_points INTEGER DEFAULT 0,
            r2 REAL DEFAULT 0,
            rmse REAL DEFAULT 0,
            mean_depth REAL DEFAULT 0,
            max_depth REAL DEFAULT 0,
            min_depth REAL DEFAULT 0,
            std_depth REAL DEFAULT 0,
            sources TEXT DEFAULT '',
            grid_hash TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            metadata_json TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS depth_grids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_id TEXT NOT NULL REFERENCES surveys(survey_id),
            grid_blob BLOB NOT NULL,
            shape_json TEXT NOT NULL,
            bbox_json TEXT NOT NULL,
            uncertainty_blob BLOB
        );

        CREATE TABLE IF NOT EXISTS change_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            survey_a TEXT NOT NULL,
            survey_b TEXT NOT NULL,
            date_a TEXT NOT NULL,
            date_b TEXT NOT NULL,
            bbox_json TEXT NOT NULL,
            mean_change REAL DEFAULT 0,
            max_erosion REAL DEFAULT 0,
            max_deposition REAL DEFAULT 0,
            rmsd REAL DEFAULT 0,
            pct_changed REAL DEFAULT 0,
            n_significant INTEGER DEFAULT 0,
            alert_level TEXT DEFAULT 'none',
            details_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'info',
            message TEXT NOT NULL,
            bbox_json TEXT DEFAULT '{}',
            survey_id TEXT,
            change_event_id INTEGER,
            acknowledged INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS monitoring_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            bbox_json TEXT NOT NULL,
            alert_threshold_m REAL DEFAULT 0.5,
            min_depth_m REAL DEFAULT 0,
            max_depth_m REAL DEFAULT 25,
            check_interval_days INTEGER DEFAULT 30,
            last_checked TEXT,
            active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_surveys_date ON surveys(date);
        CREATE INDEX IF NOT EXISTS idx_surveys_bbox ON surveys(bbox_json);
        CREATE INDEX IF NOT EXISTS idx_change_dates ON change_events(date_a, date_b);
        CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, acknowledged);
    """)
    return db


# ═══════════════════════════════════════════════════════════
# SURVEY STORAGE
# ═══════════════════════════════════════════════════════════

def store_survey(survey_id: str, bbox: list, date: str, depth_grid: np.ndarray,
                 uncertainty: np.ndarray = None, method: str = "cnn",
                 resolution: float = 10, r2: float = 0, rmse: float = 0,
                 sources: str = "", metadata: dict = None):
    """
    Store a bathymetry survey result for temporal tracking.
    """
    db = _get_db()
    valid = depth_grid[np.isfinite(depth_grid) & (depth_grid > 0)]
    stats = {
        "n_points": int(len(valid)),
        "mean_depth": round(float(np.mean(valid)), 2) if len(valid) else 0,
        "max_depth": round(float(np.max(valid)), 2) if len(valid) else 0,
        "min_depth": round(float(np.min(valid)), 2) if len(valid) else 0,
        "std_depth": round(float(np.std(valid)), 2) if len(valid) else 0,
    }

    # Store survey metadata
    db.execute("""
        INSERT OR REPLACE INTO surveys
        (survey_id, bbox_json, date, method, resolution_m, n_points,
         r2, rmse, mean_depth, max_depth, min_depth, std_depth, sources, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        survey_id, json.dumps(bbox), date, method, resolution,
        stats["n_points"], r2, rmse,
        stats["mean_depth"], stats["max_depth"], stats["min_depth"], stats["std_depth"],
        sources, json.dumps(metadata or {}),
    ))

    # Store depth grid as compressed numpy blob
    grid_blob = _compress_grid(depth_grid)
    unc_blob = _compress_grid(uncertainty) if uncertainty is not None else None

    db.execute("""
        INSERT INTO depth_grids (survey_id, grid_blob, shape_json, bbox_json, uncertainty_blob)
        VALUES (?, ?, ?, ?, ?)
    """, (
        survey_id, grid_blob,
        json.dumps(list(depth_grid.shape)), json.dumps(bbox),
        unc_blob,
    ))

    db.commit()
    db.close()
    L.info(f"Monitoring: stored survey '{survey_id}' ({date}), "
           f"{stats['n_points']} pts, mean={stats['mean_depth']}m")
    return stats


def get_survey(survey_id: str):
    """Retrieve a stored survey with its depth grid."""
    db = _get_db()
    row = db.execute("SELECT * FROM surveys WHERE survey_id=?", (survey_id,)).fetchone()
    if not row:
        db.close()
        return None

    grid_row = db.execute(
        "SELECT * FROM depth_grids WHERE survey_id=? ORDER BY id DESC LIMIT 1",
        (survey_id,)
    ).fetchone()
    db.close()

    result = dict(row)
    if grid_row:
        shape = json.loads(grid_row["shape_json"])
        result["depth_grid"] = _decompress_grid(grid_row["grid_blob"], shape)
        if grid_row["uncertainty_blob"]:
            result["uncertainty"] = _decompress_grid(grid_row["uncertainty_blob"], shape)
    return result


def list_surveys(bbox: list = None, limit: int = 50):
    """List stored surveys, optionally filtered by bbox overlap."""
    db = _get_db()
    rows = db.execute(
        "SELECT survey_id, date, method, n_points, r2, rmse, "
        "mean_depth, max_depth, min_depth, std_depth, sources, bbox_json "
        "FROM surveys ORDER BY date DESC LIMIT ?", (limit,)
    ).fetchall()
    db.close()

    surveys = [dict(r) for r in rows]
    if bbox:
        w, s, e, n = bbox
        filtered = []
        for sv in surveys:
            sb = json.loads(sv["bbox_json"])
            if _bbox_overlaps(sb, bbox):
                filtered.append(sv)
        surveys = filtered
    return surveys


# ═══════════════════════════════════════════════════════════
# CHANGE DETECTION
# ═══════════════════════════════════════════════════════════

def compute_change(survey_a_id: str, survey_b_id: str, threshold_m: float = 0.5):
    """
    Compute bathymetric change between two surveys.
    Positive change = deposition (shallower), Negative = erosion (deeper).

    Returns change analysis dict with:
    - diff_grid: B - A (positive = got shallower)
    - statistics: mean, max erosion/deposition, RMSD
    - significant_mask: cells with |change| > threshold
    - alert_level: none/info/warning/critical
    """
    sv_a = get_survey(survey_a_id)
    sv_b = get_survey(survey_b_id)
    if not sv_a or not sv_b:
        return None, "Survey not found"
    if "depth_grid" not in sv_a or "depth_grid" not in sv_b:
        return None, "Depth grid not stored"

    grid_a = sv_a["depth_grid"]
    grid_b = sv_b["depth_grid"]

    # Align grids if different shapes
    if grid_a.shape != grid_b.shape:
        from PIL import Image
        target_h = max(grid_a.shape[0], grid_b.shape[0])
        target_w = max(grid_a.shape[1], grid_b.shape[1])
        grid_a = _resize_grid(grid_a, target_h, target_w)
        grid_b = _resize_grid(grid_b, target_h, target_w)

    # Compute difference: B - A (positive = shallowing/deposition)
    valid = np.isfinite(grid_a) & np.isfinite(grid_b) & (grid_a > 0) & (grid_b > 0)
    diff = np.full_like(grid_a, np.nan)
    diff[valid] = grid_a[valid] - grid_b[valid]  # A deeper than B = erosion at B

    valid_diff = diff[valid]
    if len(valid_diff) < 10:
        return None, "Insufficient overlap between surveys"

    # Statistics
    mean_change = float(np.mean(valid_diff))
    max_erosion = float(np.min(valid_diff))   # most negative = deepest erosion
    max_deposition = float(np.max(valid_diff))  # most positive = most deposition
    rmsd = float(np.sqrt(np.mean(valid_diff ** 2)))
    std_change = float(np.std(valid_diff))

    # Significant changes
    significant = np.abs(valid_diff) > threshold_m
    n_significant = int(np.sum(significant))
    pct_changed = round(n_significant / len(valid_diff) * 100, 1)

    # Classification by type
    erosion_mask = valid_diff < -threshold_m
    deposition_mask = valid_diff > threshold_m
    n_erosion = int(np.sum(erosion_mask))
    n_deposition = int(np.sum(deposition_mask))

    # Alert level
    if abs(max_erosion) > 3.0 or abs(max_deposition) > 3.0 or pct_changed > 30:
        alert = "critical"
    elif abs(max_erosion) > 1.5 or abs(max_deposition) > 1.5 or pct_changed > 15:
        alert = "warning"
    elif pct_changed > 5:
        alert = "info"
    else:
        alert = "none"

    # Depth zone breakdown
    zones = {}
    for zone_name, z_min, z_max in [("shallow_0_5m", 0, 5), ("moderate_5_10m", 5, 10),
                                      ("deep_10_20m", 10, 20), ("vdeep_20_25m", 20, 25)]:
        zone_mask = valid & (grid_a >= z_min) & (grid_a < z_max)
        zone_diff = diff[zone_mask]
        if len(zone_diff) > 0:
            zones[zone_name] = {
                "mean_change": round(float(np.mean(zone_diff)), 3),
                "std_change": round(float(np.std(zone_diff)), 3),
                "n_cells": int(len(zone_diff)),
                "pct_significant": round(float(np.sum(np.abs(zone_diff) > threshold_m) / len(zone_diff) * 100), 1),
            }

    details = {
        "mean_change": round(mean_change, 3),
        "std_change": round(std_change, 3),
        "max_erosion": round(max_erosion, 3),
        "max_deposition": round(max_deposition, 3),
        "rmsd": round(rmsd, 3),
        "n_significant": n_significant,
        "n_erosion": n_erosion,
        "n_deposition": n_deposition,
        "pct_changed": pct_changed,
        "threshold_m": threshold_m,
        "n_valid_cells": int(np.sum(valid)),
        "zones": zones,
        "alert_level": alert,
    }

    # Store change event
    db = _get_db()
    bbox_a = json.loads(sv_a.get("bbox_json", "[]"))
    db.execute("""
        INSERT INTO change_events
        (survey_a, survey_b, date_a, date_b, bbox_json,
         mean_change, max_erosion, max_deposition, rmsd, pct_changed,
         n_significant, alert_level, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        survey_a_id, survey_b_id, sv_a["date"], sv_b["date"],
        sv_a.get("bbox_json", "[]"),
        mean_change, max_erosion, max_deposition, rmsd, pct_changed,
        n_significant, alert, json.dumps(details),
    ))

    # Generate alerts if needed
    if alert in ("warning", "critical"):
        _create_alert(
            alert_type="depth_change",
            severity=alert,
            message=f"Significant depth change detected: {pct_changed}% area changed, "
                    f"max erosion={abs(max_erosion):.1f}m, max deposition={max_deposition:.1f}m",
            bbox=bbox_a,
            survey_id=survey_b_id,
            db=db,
        )

    db.commit()
    db.close()

    L.info(f"Change detection: {survey_a_id} vs {survey_b_id}, "
           f"mean={mean_change:.2f}m, RMSD={rmsd:.2f}m, "
           f"{pct_changed}% changed, alert={alert}")

    return {
        "diff_grid": diff,
        "statistics": details,
        "survey_a": {"id": survey_a_id, "date": sv_a["date"]},
        "survey_b": {"id": survey_b_id, "date": sv_b["date"]},
    }, None


# ═══════════════════════════════════════════════════════════
# TREND ANALYSIS
# ═══════════════════════════════════════════════════════════

def compute_trends(bbox: list, min_surveys: int = 3):
    """
    Compute per-cell linear depth trends across all surveys in a region.
    Returns trend_rate (m/year), significance (p-value proxy), and R².
    """
    surveys = list_surveys(bbox, limit=100)
    if len(surveys) < min_surveys:
        return None, f"Need at least {min_surveys} surveys, found {len(surveys)}"

    # Load all grids
    grids = []
    dates = []
    for sv in sorted(surveys, key=lambda s: s["date"]):
        full = get_survey(sv["survey_id"])
        if full and "depth_grid" in full:
            grids.append(full["depth_grid"])
            dates.append(sv["date"])

    if len(grids) < min_surveys:
        return None, "Insufficient grids with stored data"

    # Align all grids to same shape
    target_h = max(g.shape[0] for g in grids)
    target_w = max(g.shape[1] for g in grids)
    aligned = []
    for g in grids:
        if g.shape != (target_h, target_w):
            g = _resize_grid(g, target_h, target_w)
        aligned.append(g)

    # Time axis (days from first survey)
    t0 = datetime.strptime(dates[0], "%Y-%m-%d")
    time_days = np.array([
        (datetime.strptime(d, "%Y-%m-%d") - t0).days for d in dates
    ], dtype=float)
    time_years = time_days / 365.25

    # Stack into (n_surveys, H, W)
    stack = np.stack(aligned, axis=0)

    # Per-cell linear regression: depth = a * time + b
    trend_rate = np.full((target_h, target_w), np.nan)
    trend_r2 = np.full((target_h, target_w), np.nan)
    trend_intercept = np.full((target_h, target_w), np.nan)

    for r in range(target_h):
        for c in range(target_w):
            series = stack[:, r, c]
            valid = np.isfinite(series)
            if valid.sum() < min_surveys:
                continue
            t = time_years[valid]
            d = series[valid]
            # Linear regression
            n = len(t)
            t_mean = np.mean(t)
            d_mean = np.mean(d)
            ss_tt = np.sum((t - t_mean) ** 2)
            if ss_tt < 1e-10:
                continue
            slope = np.sum((t - t_mean) * (d - d_mean)) / ss_tt
            intercept = d_mean - slope * t_mean
            residuals = d - (slope * t + intercept)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((d - d_mean) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0

            trend_rate[r, c] = round(slope, 4)  # m/year
            trend_r2[r, c] = round(r2, 4)
            trend_intercept[r, c] = round(intercept, 2)

    valid_trends = trend_rate[np.isfinite(trend_rate)]
    if len(valid_trends) == 0:
        return None, "No valid trend data"

    summary = {
        "n_surveys": len(grids),
        "date_range": [dates[0], dates[-1]],
        "span_years": round(float(time_years[-1] - time_years[0]), 2),
        "mean_trend_m_per_year": round(float(np.mean(valid_trends)), 4),
        "max_erosion_rate": round(float(np.min(valid_trends)), 4),
        "max_accretion_rate": round(float(np.max(valid_trends)), 4),
        "pct_eroding": round(float(np.sum(valid_trends < -0.1) / len(valid_trends) * 100), 1),
        "pct_accreting": round(float(np.sum(valid_trends > 0.1) / len(valid_trends) * 100), 1),
        "pct_stable": round(float(np.sum(np.abs(valid_trends) <= 0.1) / len(valid_trends) * 100), 1),
        "mean_r2": round(float(np.nanmean(trend_r2)), 3),
    }

    L.info(f"Trends: {len(grids)} surveys over {summary['span_years']} years, "
           f"mean trend={summary['mean_trend_m_per_year']} m/yr, "
           f"eroding={summary['pct_eroding']}%, stable={summary['pct_stable']}%")

    return {
        "trend_rate": trend_rate,
        "trend_r2": trend_r2,
        "trend_intercept": trend_intercept,
        "summary": summary,
        "dates": dates,
    }, None


# ═══════════════════════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════════════════════

def _create_alert(alert_type, severity, message, bbox=None, survey_id=None, change_event_id=None, db=None):
    """Create a new alert in the database."""
    own_db = db is None
    if own_db:
        db = _get_db()
    db.execute("""
        INSERT INTO alerts (alert_type, severity, message, bbox_json, survey_id, change_event_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (alert_type, severity, message, json.dumps(bbox or []), survey_id, change_event_id))
    if own_db:
        db.commit()
        db.close()
    L.info(f"ALERT [{severity.upper()}]: {message}")


def get_alerts(acknowledged=False, severity=None, limit=50):
    """Get recent alerts."""
    db = _get_db()
    query = "SELECT * FROM alerts WHERE acknowledged=?"
    params = [int(acknowledged)]
    if severity:
        query += " AND severity=?"
        params.append(severity)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    db.close()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: int):
    """Mark an alert as acknowledged."""
    db = _get_db()
    db.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
    db.commit()
    db.close()


# ═══════════════════════════════════════════════════════════
# MONITORING ZONES
# ═══════════════════════════════════════════════════════════

def create_zone(name: str, bbox: list, threshold_m: float = 0.5,
                min_depth: float = 0, max_depth: float = 25,
                check_interval_days: int = 30):
    """Create a monitoring zone for automated tracking."""
    db = _get_db()
    db.execute("""
        INSERT INTO monitoring_zones
        (name, bbox_json, alert_threshold_m, min_depth_m, max_depth_m, check_interval_days)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, json.dumps(bbox), threshold_m, min_depth, max_depth, check_interval_days))
    db.commit()
    zone_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.close()
    L.info(f"Monitoring zone created: '{name}' (threshold={threshold_m}m, interval={check_interval_days}d)")
    return zone_id


def list_zones(active_only=True):
    """List monitoring zones."""
    db = _get_db()
    query = "SELECT * FROM monitoring_zones"
    if active_only:
        query += " WHERE active=1"
    rows = db.execute(query).fetchall()
    db.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════
# DASHBOARD DATA
# ═══════════════════════════════════════════════════════════

def get_dashboard_data(bbox: list = None):
    """
    Get comprehensive monitoring dashboard data:
    - Survey timeline
    - Recent changes
    - Active alerts
    - Zone status
    - Trend summary
    """
    db = _get_db()

    # Survey timeline
    surveys = list_surveys(bbox, limit=100)

    # Recent change events
    change_query = "SELECT * FROM change_events ORDER BY created_at DESC LIMIT 20"
    changes = [dict(r) for r in db.execute(change_query).fetchall()]

    # Active alerts
    alerts = get_alerts(acknowledged=False, limit=20)

    # Alert counts by severity
    alert_counts = {}
    for sev in ["info", "warning", "critical"]:
        row = db.execute(
            "SELECT COUNT(*) as cnt FROM alerts WHERE severity=? AND acknowledged=0",
            (sev,)
        ).fetchone()
        alert_counts[sev] = row["cnt"] if row else 0

    # Monitoring zones
    zones = list_zones()

    # Overall statistics
    total_row = db.execute("SELECT COUNT(*) as cnt FROM surveys").fetchone()
    total_surveys = total_row["cnt"] if total_row else 0

    db.close()

    # Time series for depth tracking
    time_series = []
    for sv in sorted(surveys, key=lambda s: s["date"]):
        time_series.append({
            "date": sv["date"],
            "mean_depth": sv["mean_depth"],
            "max_depth": sv["max_depth"],
            "min_depth": sv["min_depth"],
            "std_depth": sv["std_depth"],
            "r2": sv["r2"],
            "n_points": sv["n_points"],
            "method": sv["method"],
        })

    return {
        "total_surveys": total_surveys,
        "surveys": surveys[:20],
        "time_series": time_series,
        "recent_changes": changes[:10],
        "alerts": alerts,
        "alert_counts": alert_counts,
        "zones": zones,
    }


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _compress_grid(grid: np.ndarray) -> bytes:
    """Compress numpy grid to bytes for SQLite storage."""
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, grid=grid.astype(np.float32))
    return buf.getvalue()


def _decompress_grid(blob: bytes, shape: list) -> np.ndarray:
    """Decompress numpy grid from SQLite blob."""
    import io
    buf = io.BytesIO(blob)
    data = np.load(buf)
    return data["grid"].reshape(shape)


def _resize_grid(grid: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a depth grid to target shape using bilinear interpolation."""
    from PIL import Image
    valid = np.isfinite(grid)
    filled = np.nan_to_num(grid, nan=0).astype(np.float32)
    img = Image.fromarray(filled)
    img = img.resize((target_w, target_h), Image.BILINEAR)
    result = np.array(img)
    # Resize validity mask
    mask_img = Image.fromarray((valid * 255).astype(np.uint8))
    mask_img = mask_img.resize((target_w, target_h), Image.NEAREST)
    result_valid = np.array(mask_img) > 127
    result[~result_valid] = np.nan
    return result


def _bbox_overlaps(bbox1: list, bbox2: list) -> bool:
    """Check if two bboxes overlap."""
    if len(bbox1) < 4 or len(bbox2) < 4:
        return True  # can't filter
    w1, s1, e1, n1 = bbox1
    w2, s2, e2, n2 = bbox2
    return not (e1 < w2 or e2 < w1 or n1 < s2 or n2 < s1)
