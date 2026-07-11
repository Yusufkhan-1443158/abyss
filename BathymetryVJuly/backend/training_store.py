"""
Spatial Training Data Store — RAG for Bathymetry

Persists georeferenced depth observations in SQLite with spatial indexing.
Retrieves nearby training data for any bbox query (spatial RAG).
Maintains pre-trained model weights per region.

Usage:
  store = TrainingStore('/path/to/training.db')
  store.add_points(lats, lons, depths, source='KP_Basin')
  pts = store.query_bbox([west, south, east, north], buffer_km=5)
  store.save_model('region_key', model_bytes, metadata)
  model_bytes, meta = store.load_model('region_key')
"""

import sqlite3
import json
import logging
import os
import time
import numpy as np
from pathlib import Path

L = logging.getLogger('bathy.training')

DB_PATH = os.environ.get('TRAINING_DB', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_store.db'))


class TrainingStore:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS training_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                depth REAL NOT NULL,
                source TEXT DEFAULT 'manual',
                region TEXT DEFAULT '',
                added_ts REAL DEFAULT 0,
                quality TEXT DEFAULT 'medium'
            )''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_tp_lat ON training_points(lat)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_tp_lon ON training_points(lon)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_tp_source ON training_points(source)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_tp_region ON training_points(region)''')

            c.execute('''CREATE TABLE IF NOT EXISTS pretrained_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region_key TEXT UNIQUE NOT NULL,
                model_blob BLOB,
                scaler_blob BLOB,
                metadata TEXT DEFAULT '{}',
                n_train INTEGER DEFAULT 0,
                rmse REAL DEFAULT 0,
                r2 REAL DEFAULT 0,
                created_ts REAL DEFAULT 0,
                updated_ts REAL DEFAULT 0
            )''')

            c.execute('''CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bbox TEXT,
                method TEXT,
                n_points INTEGER,
                rmse REAL,
                r2 REAL,
                source TEXT,
                ts REAL DEFAULT 0
            )''')
            c.commit()
        L.info(f"TrainingStore: {self.db_path}")

    # ── Add training points ──
    def add_points(self, lats, lons, depths, source='manual', region='', quality='medium'):
        """Add georeferenced depth points to the store."""
        ts = time.time()
        rows = [(float(lats[i]), float(lons[i]), float(depths[i]), source, region, ts, quality)
                for i in range(len(lats))]
        with self._conn() as c:
            c.executemany(
                'INSERT INTO training_points (lat, lon, depth, source, region, added_ts, quality) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
            c.commit()
        L.info(f"TrainingStore: added {len(rows)} points (source={source}, region={region})")
        return len(rows)

    def add_manual_point(self, lat, lon, depth, source='manual', region=''):
        """Add a single manually entered depth point."""
        return self.add_points([lat], [lon], [depth], source=source, region=region, quality='manual')

    # ── Spatial query (RAG) ──
    def query_bbox(self, bbox, buffer_km=2.0, max_points=5000):
        """Retrieve training points within bbox + buffer. Core spatial RAG."""
        w, s, e, n = bbox
        # Buffer in degrees (~1km ≈ 0.009°)
        buf = buffer_km * 0.009
        with self._conn() as c:
            rows = c.execute(
                'SELECT lat, lon, depth, source, quality FROM training_points '
                'WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? '
                'ORDER BY quality DESC, added_ts DESC LIMIT ?',
                (s - buf, n + buf, w - buf, e + buf, max_points)
            ).fetchall()
        if rows:
            lats = np.array([r[0] for r in rows])
            lons = np.array([r[1] for r in rows])
            depths = np.array([r[2] for r in rows])
            sources = [r[3] for r in rows]
            L.info(f"TrainingStore RAG: {len(rows)} points for bbox "
                   f"[{w:.3f},{s:.3f},{e:.3f},{n:.3f}] +{buffer_km}km")
            return {'lats': lats, 'lons': lons, 'depths': depths, 'sources': sources, 'count': len(rows)}
        return {'lats': np.array([]), 'lons': np.array([]), 'depths': np.array([]), 'sources': [], 'count': 0}

    def query_region(self, region, max_points=10000):
        """Retrieve all points for a named region."""
        with self._conn() as c:
            rows = c.execute(
                'SELECT lat, lon, depth, source FROM training_points WHERE region=? LIMIT ?',
                (region, max_points)).fetchall()
        if rows:
            return {'lats': np.array([r[0] for r in rows]), 'lons': np.array([r[1] for r in rows]),
                    'depths': np.array([r[2] for r in rows]), 'count': len(rows)}
        return {'lats': np.array([]), 'lons': np.array([]), 'depths': np.array([]), 'count': 0}

    # ── Pre-trained model store ──
    def save_model(self, region_key, model_bytes, scaler_bytes=None, metadata=None,
                   n_train=0, rmse=0, r2=0):
        """Save a pre-trained model for a region."""
        ts = time.time()
        meta_json = json.dumps(metadata or {})
        with self._conn() as c:
            c.execute('''INSERT OR REPLACE INTO pretrained_models
                (region_key, model_blob, scaler_blob, metadata, n_train, rmse, r2, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_ts FROM pretrained_models WHERE region_key=?), ?), ?)''',
                (region_key, model_bytes, scaler_bytes, meta_json, n_train, rmse, r2, region_key, ts, ts))
            c.commit()
        L.info(f"TrainingStore: saved model '{region_key}' ({n_train} pts, R²={r2:.3f}, RMSE={rmse:.2f}m)")

    def load_model(self, region_key):
        """Load a pre-trained model. Returns (model_bytes, scaler_bytes, metadata) or None."""
        with self._conn() as c:
            row = c.execute(
                'SELECT model_blob, scaler_blob, metadata, n_train, rmse, r2 '
                'FROM pretrained_models WHERE region_key=?', (region_key,)).fetchone()
        if row:
            meta = json.loads(row[2]) if row[2] else {}
            meta.update({'n_train': row[3], 'rmse': row[4], 'r2': row[5]})
            return row[0], row[1], meta
        return None, None, None

    def list_models(self):
        """List all pre-trained models."""
        with self._conn() as c:
            rows = c.execute(
                'SELECT region_key, n_train, rmse, r2, updated_ts FROM pretrained_models '
                'ORDER BY updated_ts DESC').fetchall()
        return [{'region': r[0], 'n_train': r[1], 'rmse': r[2], 'r2': r[3], 'updated': r[4]} for r in rows]

    # ── Stats ──
    def stats(self):
        """Get store statistics."""
        with self._conn() as c:
            total = c.execute('SELECT COUNT(*) FROM training_points').fetchone()[0]
            sources = c.execute(
                'SELECT source, COUNT(*) FROM training_points GROUP BY source ORDER BY COUNT(*) DESC'
            ).fetchall()
            regions = c.execute(
                'SELECT region, COUNT(*) FROM training_points WHERE region != "" GROUP BY region'
            ).fetchall()
            models = c.execute('SELECT COUNT(*) FROM pretrained_models').fetchone()[0]
        return {
            'total_points': total,
            'sources': {s[0]: s[1] for s in sources},
            'regions': {r[0]: r[1] for r in regions},
            'pretrained_models': models,
        }

    def log_session(self, bbox, method, n_points, rmse=0, r2=0, source=''):
        """Log a training session for history."""
        with self._conn() as c:
            c.execute('INSERT INTO training_sessions (bbox, method, n_points, rmse, r2, source, ts) '
                      'VALUES (?, ?, ?, ?, ?, ?, ?)',
                      (json.dumps(bbox), method, n_points, rmse, r2, source, time.time()))
            c.commit()


# Global singleton
_store = None

def get_store():
    global _store
    if _store is None:
        _store = TrainingStore()
    return _store
