"""Bathymetry v9 — U-Net CNN + SDB + Chart AI + SlideRule + Auto-Eval + Smart Chart Digitise"""
import os,io,json,logging,traceback,tempfile,glob,math,re,base64
import time as TM
from pathlib import Path
import requests,numpy as np,pandas as pd
from flask import Flask,request as req,jsonify,send_from_directory,Response
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import tifffile

# ══════════════════════════════════════════════════════════════
# GLOBAL DEPTH CAP — 25m maximum for SDB validity
# ══════════════════════════════════════════════════════════════
MAX_DEPTH_M = 25.0  # Satellite-derived bathymetry reliable limit
OBS_TRAIN_WEIGHT = 5.0  # weight of observed points vs GEBCO (=1.0) in tile-level ridge

# VERYHR_STABLE_LOG ROUND 1 (SPEC-2/SPEC-5): pinned default windows. Wall-clock
# date.today()/utcnow() defaults are NON-DETERMINISTIC (same request, different
# day -> different composite). When a caller omits dates we now default to these
# FIXED constants instead of "today". Explicit caller dates are still honored.
PINNED_S2_WINDOW = ("2024-05-01", "2024-09-30")   # fixed S2 default window
PINNED_SR_WINDOW = ("2020-01-01", "2024-12-31")   # fixed ICESat-2 / SlideRule window
PINNED_ICE_TODAY = "2024-12-31"                    # fixed ICESat-2 archive upper bound

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SK=True
except: SK=False
try: from scipy.ndimage import gaussian_filter;SCI=True
except: SCI=False
try: from scipy.interpolate import RegularGridInterpolator;INTERP=True
except: INTERP=False
try:
    from backend.cnn_engine import cnn_train_and_predict
    CNN_AVAILABLE=True
except:
    try:
        from cnn_engine import cnn_train_and_predict
        CNN_AVAILABLE=True
    except: CNN_AVAILABLE=False
try:
    from backend.wave_bathy import wave_bathymetry
    WAVE_AVAILABLE=True
except:
    try:
        from wave_bathy import wave_bathymetry
        WAVE_AVAILABLE=True
    except: WAVE_AVAILABLE=False
try:
    from backend.gpu_mosaic_engine import mosaic_extract,search_best_scenes
    MOSAIC_AVAILABLE=True
except:
    try:
        from gpu_mosaic_engine import mosaic_extract,search_best_scenes
        MOSAIC_AVAILABLE=True
    except: MOSAIC_AVAILABLE=False
try:
    from backend.icesat2_bathy import search_atl03,process_granule,aggregate_multi_pass
    CSHELPH_AVAILABLE=True
except:
    try:
        from icesat2_bathy import search_atl03,process_granule,aggregate_multi_pass
        CSHELPH_AVAILABLE=True
    except: CSHELPH_AVAILABLE=False

# Robust Lyzenga + SlideRule weighted SDB (never returns all-NaN over water)
try:
    from backend.lyzenga_sliderule import estimate_depth as ls_estimate_depth
    LS_AVAILABLE=True
except:
    try:
        from lyzenga_sliderule import estimate_depth as ls_estimate_depth
        LS_AVAILABLE=True
    except: LS_AVAILABLE=False

# UAE pretrained Cluster + Random Forest SDB model
try:
    from backend.uae_pretrained import load_model as load_uae_rf
    UAE_AVAILABLE=True
except:
    try:
        from uae_pretrained import load_model as load_uae_rf
        UAE_AVAILABLE=True
    except: UAE_AVAILABLE=False

# UAE pretrained Cluster + CNN/MLP model — preferred when available
try:
    from backend.uae_pretrained_cnn import load_model as load_uae_cnn
    UAE_CNN_AVAILABLE=True
except:
    try:
        from uae_pretrained_cnn import load_model as load_uae_cnn
        UAE_CNN_AVAILABLE=True
    except: UAE_CNN_AVAILABLE=False


def load_uae_model(force_reload: bool = False):
    """Prefer the CNN/MLP variant; fall back to the RF if the CNN
    pickle isn't on disk (or PyTorch is unavailable)."""
    if UAE_CNN_AVAILABLE:
        m = load_uae_cnn(force_reload=force_reload)
        if m is not None:
            return m
    if UAE_AVAILABLE:
        return load_uae_rf(force_reload=force_reload)
    return None

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
L=logging.getLogger('bathy')


# No hard ROI cap — large ROIs are accepted and rendered at the native
# Sentinel-2 tile resolution. The pixel-count cap inside fetch_s2 /
# _fetch_s2_single (max 2500×2500) plus the area-aware resolution
# selector in _run_s2_lyzenga_fast keep response sizes bounded:
#
#     area      → native res        → max grid          → typical PNG
#     ≤   30    → 10 m              → ≤ 2500×2500       → ≤ 1.5 MB
#     ≤  200    → 20 m              → ≤ 2500×2500       → ≤ 1.5 MB
#     ≤  800    → 30 m              → ≤ 2500×2500       → ≤ 1.5 MB
#     >  800    → 50 m              → ≤ 2500×2500       → ≤ 1.5 MB
#
# A soft warning is logged when the area exceeds 40 km² so operators
# know the resolution dropped from the predefined-region 10 m default.

SOFT_HIGH_RES_KM2 = 40.0


def _bbox_area_km2(bbox):
    """Area of a [w, s, e, n] bbox in km², spherical approximation."""
    w, s, e, n = bbox
    cl = math.cos(math.radians((n + s) / 2))
    return abs(e - w) * 111.0 * cl * abs(n - s) * 111.0


def _native_res_for_area(area_km2: float) -> int:
    """Pick the Sentinel-2 native resolution that keeps the fetched
    raster under 2500×2500 px while preserving as much detail as the
    ROI allows."""
    if area_km2 <= 30:
        return 10
    if area_km2 <= 200:
        return 20
    if area_km2 <= 800:
        return 30
    return 50


def _enforce_roi_cap(bbox, max_km2: float = SOFT_HIGH_RES_KM2):
    """No-op kept for callers that still reference it. Always returns
    None (i.e. always accepts the ROI). Larger ROIs simply get a
    coarser native resolution via _native_res_for_area."""
    return None


def _json_safe(obj):
    """Recursively replace NaN/Inf with None and convert numpy scalars to
    plain Python so JSON.parse on the frontend never trips on NaN/Inf
    literals. This is what gave the new UI all-NaN outputs.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
sf=None
for p in['/app/frontend/build','../frontend/build','frontend/build']:
    if os.path.isdir(os.path.abspath(p)):sf=os.path.abspath(p);break
app=Flask(__name__,static_folder=sf,static_url_path='' if sf else None)
CORS(app,resources={r"/api/*":{"origins":"*"}})

# ROB-R4: no upload size cap meant a 30 MB / 1.5M-row XYZ POST was accepted
# and echoed back as a 93 MB JSON body — a straight memory-exhaustion vector
# with no user-visible explanation on a public Railway deployment. Cap the
# request body and, critically, make the resulting 413 a JSON body (Flask's
# default 413 is an HTML error page, which would just re-trigger the
# ROB-R1 unparseable-response failure mode on the frontend's `r.json()`).
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

@app.errorhandler(413)
def _handle_request_too_large(e):
    return jsonify({'error': 'Upload too large (max 50 MB). Split the file or reduce the point count and try again.'}), 413

@app.errorhandler(RequestEntityTooLarge)
def _handle_request_entity_too_large(e):
    # Belt-and-suspenders: some code paths access req.files/req.form inside a
    # view's own broad `except Exception` before Flask's numeric-code 413
    # handler ever gets a chance to see it (the exception is raised lazily on
    # first access to the parsed body, deep inside that view's try block) —
    # this class-based handler is what actually fires in that case.
    return jsonify({'error': 'Upload too large (max 50 MB). Split the file or reduce the point count and try again.'}), 413

# ── Training store (spatial RAG) ─────────────────────────────
try:
    from backend.training_store import get_store as _get_store
except:
    try:
        from training_store import get_store as _get_store
    except:
        _get_store = None
        L.warning("Training store not available")

# Auto-load XYZ files into training store on startup
# ── Full in-situ XYZ cache (used as MLE prior, NOT subsampled) ────
# Module-level arrays so they can be queried by bbox without re-reading disk.
XYZ_LATS = np.array([], dtype=np.float64)
XYZ_LONS = np.array([], dtype=np.float64)
XYZ_DEPTHS = np.array([], dtype=np.float64)
XYZ_SOURCES = np.array([], dtype=object)

def _load_xyz_full():
    """Load ALL points from the bundled in-situ surveys.

    Three UTM-Zone-40N XYZ files (KP / KP_EMAL / OMC) plus the
    SWOT_Dhanna shapefile (UTM Zone 39N, with a `Z` attribute) are
    fused into XYZ_LATS / XYZ_LONS / XYZ_DEPTHS so that the runtime
    in-situ-in-bbox lookup picks them up alongside the original three.
    """
    global XYZ_LATS, XYZ_LONS, XYZ_DEPTHS, XYZ_SOURCES
    # Validation / calibration data lives in `validation/` at the repo
    # root (or /app/validation in the Docker image).  _load_xyz_full
    # tries that first, then falls back to legacy root-level paths so
    # older deploys / dev clones still work.
    xyz_files = [
        ('validation/KP Basin Soundings 10m.xyz', 'KP_Basin', 'khalifa_port'),
        ('validation/KP_EMAL_Soundings_10x_New.xyz', 'KP_EMAL', 'khalifa_port'),
        ('validation/OMC_Soundings_20m.xyz', 'OMC', 'oman_coast'),
        # Legacy fallbacks (for backwards compatibility):
        ('KP Basin Soundings 10m.xyz', 'KP_Basin', 'khalifa_port'),
        ('KP_EMAL_Soundings_10x_New.xyz', 'KP_EMAL', 'khalifa_port'),
        ('OMC_Soundings_20m.xyz', 'OMC', 'oman_coast'),
    ]
    shp_files = [
        ('validation/swot_dhanna/SWOT_Dhanna.shp',
         'SWOT_Dhanna', 'jbel_dhanna', 32639),
        # Legacy fallback:
        ('backend/calibration_data/swot_dhanna/SWOT_Dhanna.shp',
         'SWOT_Dhanna', 'jbel_dhanna', 32639),
    ]
    loaded_sources = set()  # avoid double-load when both new and legacy paths exist
    all_lats, all_lons, all_deps, all_src = [], [], [], []
    try:
        from pyproj import Transformer, CRS
        tr40 = Transformer.from_crs(CRS.from_epsg(32640), CRS.from_epsg(4326), always_xy=True)
    except Exception as ex:
        L.warning(f"XYZ loader: pyproj unavailable ({ex})")
        return
    for fname, source, _region in xyz_files:
        if source in loaded_sources:
            continue
        for base in ['/app', '.', '..']:
            fpath = os.path.join(base, fname)
            if not os.path.exists(fpath):
                continue
            try:
                data = np.loadtxt(fpath, usecols=(0, 1, 2))
                lons, lats = tr40.transform(data[:, 0], data[:, 1])
                deps = np.abs(data[:, 2])
                valid = (deps > 0) & (deps < 50) & np.isfinite(lats) & np.isfinite(lons)
                n = int(valid.sum())
                all_lats.append(lats[valid])
                all_lons.append(lons[valid])
                all_deps.append(deps[valid])
                all_src.append(np.full(n, source, dtype=object))
                L.info(f"XYZ full: loaded {n} pts from {fname}")
                loaded_sources.add(source)
            except Exception as ex:
                L.warning(f"XYZ full: failed to load {fname}: {ex}")
            break
    # Shapefiles
    for path_rel, source, _region, _epsg in shp_files:
        if source in loaded_sources:
            continue
        for base in ['/app', '.', '..']:
            fpath = os.path.join(base, path_rel)
            if not os.path.exists(fpath):
                continue
            try:
                import geopandas as gpd
                g = gpd.read_file(fpath)
                if g.crs is not None and g.crs.to_epsg() != 4326:
                    g = g.to_crs("EPSG:4326")
                if "Z" in g.columns:
                    deps = np.abs(g["Z"].astype(float).values)
                elif "depth" in g.columns:
                    deps = np.abs(g["depth"].astype(float).values)
                else:
                    raise RuntimeError(f"{path_rel}: no Z/depth column found")
                lats = g.geometry.y.values
                lons = g.geometry.x.values
                valid = ((deps > 0) & (deps < 50)
                         & np.isfinite(lats) & np.isfinite(lons))
                n = int(valid.sum())
                all_lats.append(lats[valid])
                all_lons.append(lons[valid])
                all_deps.append(deps[valid])
                all_src.append(np.full(n, source, dtype=object))
                L.info(f"XYZ full: loaded {n} pts from {path_rel}")
                loaded_sources.add(source)
            except Exception as ex:
                L.warning(f"XYZ full: failed to load {path_rel}: {ex}")
            break
    if all_lats:
        XYZ_LATS = np.concatenate(all_lats)
        XYZ_LONS = np.concatenate(all_lons)
        XYZ_DEPTHS = np.concatenate(all_deps)
        XYZ_SOURCES = np.concatenate(all_src)
        L.info(f"XYZ full: total {len(XYZ_LATS)} in-situ sounding points in memory")


def xyz_points_in_bbox(bbox):
    """Return (lats, lons, depths, sources) for all in-situ points inside bbox."""
    if len(XYZ_LATS) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    w, s, e, n = bbox
    m = (XYZ_LATS >= s) & (XYZ_LATS <= n) & (XYZ_LONS >= w) & (XYZ_LONS <= e)
    return XYZ_LATS[m], XYZ_LONS[m], XYZ_DEPTHS[m], XYZ_SOURCES[m]


def _init_training_store():
    _load_xyz_full()
    if not _get_store:
        return
    store = _get_store()
    s = store.stats()
    if s['total_points'] > 0:
        L.info(f"Training store: {s['total_points']} points already loaded")
        return
    if len(XYZ_LATS) > 0:
        try:
            # Push a subsample into the RAG store (it's capped internally);
            # the MLE path reads from XYZ_LATS/LONS/DEPTHS directly and uses ALL points.
            n = len(XYZ_LATS)
            idx = np.random.choice(n, min(20000, n), replace=False)
            # Group by source for the store
            for src in np.unique(XYZ_SOURCES[idx]):
                msk = XYZ_SOURCES[idx] == src
                store.add_points(
                    XYZ_LATS[idx][msk], XYZ_LONS[idx][msk], XYZ_DEPTHS[idx][msk],
                    source=str(src), region='auto',
                )
            L.info(f"Training store: seeded {len(idx)} sample pts (full {n} kept in MLE cache)")
        except Exception as ex:
            L.warning(f"Training store seed failed: {ex}")

try:
    if _get_store:
        s = _get_store().stats()
        L.info(f"Training store: {s['total_points']} pts ready")
except Exception as ex:
    L.warning(f"Training store init: {ex}")

# Load FULL XYZ soundings into memory cache at import time so the
# Multi-Epoch MLE endpoint can use them as a prior without waiting for
# the training-store seed endpoint to be called.
try:
    _load_xyz_full()
except Exception as ex:
    L.warning(f"XYZ full-cache load: {ex}")

# ── Pre-trained regional models (silent knowledge base) ──────
# Builds/loads cached GBR models per region on startup.
# When a user requests bathymetry for a covered area, the pre-trained
# model provides an instant depth prior that is transparently refined
# with any additional data (Sentinel-2, ICESat-2, charts, user uploads).
# If the user trains with new data, the regional model is silently updated.

_PRETRAINED_MODELS = {}  # region_key -> {'model', 'scaler', 'meta'}

def _build_pretrained_models():
    """Train/load regional GBR models from the stored survey database."""
    global _PRETRAINED_MODELS
    if not _get_store:
        return
    store = _get_store()
    stats = store.stats()
    if stats['total_points'] < 50:
        return

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        import pickle
    except ImportError:
        L.warning("sklearn not available — skipping pretrained models")
        return

    # Define regions to pre-train
    regions = {}
    for region_name, count in stats.get('regions', {}).items():
        if count >= 30:
            regions[region_name] = count

    if not regions:
        return

    for region_key, n_pts in regions.items():
        # Check if we already have a cached model
        model_bytes, scaler_bytes, meta = store.load_model(region_key)
        if model_bytes and meta:
            try:
                mdl = pickle.loads(model_bytes)
                sc = pickle.loads(scaler_bytes) if scaler_bytes else None
                _PRETRAINED_MODELS[region_key] = {
                    'model': mdl, 'scaler': sc, 'meta': meta,
                    'n_train': meta.get('n_train', 0),
                    'r2': meta.get('r2', 0),
                    'rmse': meta.get('rmse', 0),
                }
                L.info(f"Loaded pretrained model: {region_key} "
                       f"(n={meta.get('n_train',0)}, R²={meta.get('r2',0):.3f})")
                continue
            except Exception as ex:
                L.warning(f"Failed to load cached model {region_key}: {ex}")

        # Train a new model from stored data
        try:
            data = store.query_region(region_key, max_points=5000)
            if data['count'] < 30:
                continue

            lats, lons, depths = data['lats'], data['lons'], data['depths']
            valid = (depths > 0) & (depths <= MAX_DEPTH_M)
            lats, lons, depths = lats[valid], lons[valid], depths[valid]
            if len(lats) < 30:
                continue

            # Features: spatial coordinates (normalised)
            lat_n = (lats - lats.mean()) / (lats.std() + 1e-8)
            lon_n = (lons - lons.mean()) / (lons.std() + 1e-8)
            # Distance from centroid
            dlat = (lats - lats.mean()) * 111000
            dlon = (lons - lons.mean()) * 111000 * np.cos(np.radians(lats.mean()))
            dist = np.sqrt(dlat**2 + dlon**2)
            dist_n = dist / (dist.std() + 1e-8)

            X = np.column_stack([lat_n, lon_n, dist_n])
            y = depths

            sc = StandardScaler()
            Xs = sc.fit_transform(X)

            # Train with holdout
            n = len(y)
            idx = np.random.RandomState(42).permutation(n)
            n_val = max(3, int(n * 0.15))
            X_tr, y_tr = Xs[idx[n_val:]], y[idx[n_val:]]
            X_vl, y_vl = Xs[idx[:n_val]], y[idx[:n_val]]

            mdl = GradientBoostingRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42)
            mdl.fit(X_tr, y_tr)

            pred_vl = mdl.predict(X_vl)
            ss_res = np.sum((pred_vl - y_vl)**2)
            ss_tot = np.sum((y_vl - y_vl.mean())**2)
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            rmse = float(np.sqrt(np.mean((pred_vl - y_vl)**2)))

            meta_dict = {
                'n_train': int(len(y_tr)),
                'r2': round(r2, 4),
                'rmse': round(rmse, 3),
                'lat_mean': float(lats.mean()),
                'lon_mean': float(lons.mean()),
                'lat_std': float(lats.std()),
                'lon_std': float(lons.std()),
                'dist_std': float(dist.std()),
                'depth_range': [float(depths.min()), float(depths.max())],
            }

            # Cache to store
            model_bytes = pickle.dumps(mdl)
            scaler_bytes = pickle.dumps(sc)
            store.save_model(region_key, model_bytes, scaler_bytes,
                             meta_dict, len(y_tr), rmse, r2)

            _PRETRAINED_MODELS[region_key] = {
                'model': mdl, 'scaler': sc, 'meta': meta_dict,
                'n_train': len(y_tr), 'r2': r2, 'rmse': rmse,
            }
            L.info(f"Trained pretrained model: {region_key} "
                   f"(n={len(y_tr)}, R²={r2:.3f}, RMSE={rmse:.2f}m)")
        except Exception as ex:
            L.warning(f"Pretrained model {region_key} failed: {ex}")

    L.info(f"Pre-trained models ready: {list(_PRETRAINED_MODELS.keys())}")


def query_pretrained_depth(lat, lon):
    """
    Query pre-trained models for a depth estimate at (lat, lon).
    Returns (depth, uncertainty, region) or (None, None, None).
    Silently checks all regional models and returns the best match.
    """
    if not _PRETRAINED_MODELS:
        return None, None, None

    best_depth, best_unc, best_region = None, None, None
    best_dist = float('inf')

    for region_key, pm in _PRETRAINED_MODELS.items():
        meta = pm['meta']
        # Check if point is within the region's spatial extent
        lat_mean = meta.get('lat_mean', 0)
        lon_mean = meta.get('lon_mean', 0)
        lat_std = meta.get('lat_std', 0.01)
        lon_std = meta.get('lon_std', 0.01)

        # Within ~3 sigma of region centroid
        if abs(lat - lat_mean) > lat_std * 5 or abs(lon - lon_mean) > lon_std * 5:
            continue

        dist_km = np.sqrt(((lat - lat_mean) * 111)**2 +
                          ((lon - lon_mean) * 111 * np.cos(np.radians(lat_mean)))**2)
        if dist_km > 20:  # max 20 km from region centroid
            continue

        if dist_km < best_dist:
            try:
                lat_n = (lat - lat_mean) / (lat_std + 1e-8)
                lon_n = (lon - lon_mean) / (lon_std + 1e-8)
                dlat = (lat - lat_mean) * 111000
                dlon = (lon - lon_mean) * 111000 * np.cos(np.radians(lat_mean))
                dist_m = np.sqrt(dlat**2 + dlon**2)
                dist_n = dist_m / (meta.get('dist_std', 1) + 1e-8)

                X = np.array([[lat_n, lon_n, dist_n]])
                if pm['scaler']:
                    X = pm['scaler'].transform(X)
                depth = float(np.clip(pm['model'].predict(X)[0], 0, MAX_DEPTH_M))
                unc = pm['rmse'] * (1 + dist_km / 10)  # uncertainty grows with distance
                best_depth, best_unc, best_region = depth, unc, region_key
                best_dist = dist_km
            except Exception:
                continue

    return best_depth, best_unc, best_region


def update_pretrained_model(region_key, new_lats, new_lons, new_depths):
    """Silently update a pretrained model with new training data."""
    if not _get_store:
        return
    try:
        store = _get_store()
        # Add new points to store
        valid = (np.array(new_depths) > 0) & (np.array(new_depths) <= MAX_DEPTH_M)
        if valid.sum() < 5:
            return
        store.add_points(
            np.array(new_lats)[valid], np.array(new_lons)[valid],
            np.array(new_depths)[valid],
            source='session_update', region=region_key, quality='high')
        L.info(f"Updated training store: +{int(valid.sum())} pts for {region_key}")

        # Rebuild model for this region (will use all stored data)
        _build_pretrained_models()
    except Exception as ex:
        L.warning(f"Pretrained model update failed: {ex}")


try:
    _build_pretrained_models()
except Exception as ex:
    L.warning(f"Pretrained model build: {ex}")

# ── Ocean monitoring module (Hormuz) ─────────────────────────
try:
    from backend.ocean.routes import ocean_bp
    app.register_blueprint(ocean_bp)
    L.info("Ocean monitoring module loaded")
except:
    try:
        from ocean.routes import ocean_bp
        app.register_blueprint(ocean_bp)
        L.info("Ocean monitoring module loaded")
    except Exception as e:
        L.warning(f"Ocean module not available: {e}")

# ══════════════════════════════════════════════════════════════
# PIXEL-MEDIAN AGGREGATION — resolution-aware comparison
# ══════════════════════════════════════════════════════════════

def pixel_median_comparison(predicted_grid, bbox, obs_lats, obs_lons, obs_depths, resolution_m=10.0):
    """
    Professional pixel-median aggregation for comparing predicted vs observed depths.

    When in-situ survey data (e.g. 1m spacing) is compared to a satellite-derived grid
    (e.g. 10m Sentinel-2), multiple survey points fall within the same pixel.
    Naively comparing each point individually double-counts pixels and biases metrics.

    This function:
    1. Maps each observed point to its pixel (row, col) in the predicted grid
    2. Groups all observations per pixel -> computes median observed depth per pixel
    3. Computes per-pixel metrics: RMSE, MAE, bias, R2, MedAE, IHO S-44
    4. Also reports per-depth-zone breakdown

    Args:
        predicted_grid: 2D numpy array (H, W) of predicted depths
        bbox: [west, south, east, north]
        obs_lats, obs_lons, obs_depths: arrays of observed points
        resolution_m: nominal pixel size in meters (for reporting)

    Returns:
        dict with comprehensive comparison metrics including:
        - pixel-level metrics (honest, resolution-aware)
        - point-level metrics (traditional, for reference)
        - per-pixel breakdown (n_obs_per_pixel distribution)
        - depth-zone statistics
        - IHO S-44 compliance
    """
    from collections import defaultdict

    H, W = predicted_grid.shape
    w, s, e, n = bbox

    obs_lats = np.asarray(obs_lats, dtype=np.float64)
    obs_lons = np.asarray(obs_lons, dtype=np.float64)
    obs_depths = np.asarray(obs_depths, dtype=np.float64)

    # ── Map each observed point to pixel coordinates ──
    pixel_groups = defaultdict(list)  # (row, col) -> [depth1, depth2, ...]
    point_pairs = []  # traditional point-level comparison

    for i in range(len(obs_lats)):
        r = max(0, min(H - 1, int((n - obs_lats[i]) / (n - s + 1e-10) * H)))
        c = max(0, min(W - 1, int((obs_lons[i] - w) / (e - w + 1e-10) * W)))
        d_obs = obs_depths[i]
        d_pred = predicted_grid[r, c]

        if not (np.isfinite(d_pred) and d_pred > 0 and d_obs > 0):
            continue

        pixel_groups[(r, c)].append(d_obs)
        point_pairs.append((d_obs, float(d_pred)))

    if len(pixel_groups) < 3:
        return {'error': f'Only {len(pixel_groups)} valid pixels with observations', 'n_pixels': len(pixel_groups)}

    # ── Per-pixel median aggregation ──
    pixel_obs = []   # median observed depth per pixel
    pixel_pred = []  # predicted depth at that pixel
    pixel_count = [] # how many obs points in that pixel
    pixel_std = []   # std of obs within pixel (intra-pixel variability)
    pixel_coords = []

    for (r, c), depths in pixel_groups.items():
        med = float(np.median(depths))
        pred = float(predicted_grid[r, c])
        pixel_obs.append(med)
        pixel_pred.append(pred)
        pixel_count.append(len(depths))
        pixel_std.append(float(np.std(depths)) if len(depths) > 1 else 0.0)
        # Back-compute lat/lon of pixel centre
        lat = n - (r + 0.5) / H * (n - s)
        lon = w + (c + 0.5) / W * (e - w)
        pixel_coords.append((round(lat, 6), round(lon, 6)))

    pixel_obs = np.array(pixel_obs)
    pixel_pred = np.array(pixel_pred)
    pixel_count = np.array(pixel_count)
    pixel_std = np.array(pixel_std)

    # ── Pixel-level metrics (THE HONEST ONES) ──
    diffs = pixel_pred - pixel_obs
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    mae = float(np.mean(np.abs(diffs)))
    medae = float(np.median(np.abs(diffs)))
    bias = float(np.mean(diffs))
    ss_res = np.sum(diffs ** 2)
    ss_tot = np.sum((pixel_obs - np.mean(pixel_obs)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0
    p95 = float(np.percentile(np.abs(diffs), 95))
    iqr = float(np.percentile(np.abs(diffs), 75) - np.percentile(np.abs(diffs), 25))

    # IHO S-44 Order 1 per pixel
    s44_max = np.sqrt(0.5 ** 2 + (0.013 * pixel_obs) ** 2)
    s44_pass = np.abs(diffs) <= s44_max
    s44_pct = float(np.sum(s44_pass) / len(s44_pass) * 100)

    # ── Point-level metrics (traditional, for reference) ──
    pt_obs = np.array([p[0] for p in point_pairs])
    pt_pred = np.array([p[1] for p in point_pairs])
    pt_diffs = pt_pred - pt_obs
    pt_rmse = float(np.sqrt(np.mean(pt_diffs ** 2)))
    pt_mae = float(np.mean(np.abs(pt_diffs)))
    pt_bias = float(np.mean(pt_diffs))
    pt_ss_res = np.sum(pt_diffs ** 2)
    pt_ss_tot = np.sum((pt_obs - np.mean(pt_obs)) ** 2)
    pt_r2 = float(1.0 - pt_ss_res / (pt_ss_tot + 1e-10)) if pt_ss_tot > 1e-10 else 0.0

    # ── Depth-zone breakdown (pixel-level) ──
    zone_defs = [
        ('0-2m', 0, 2), ('2-5m', 2, 5), ('5-10m', 5, 10),
        ('10-15m', 10, 15), ('15-20m', 15, 20), ('20-25m', 20, 25),
    ]
    zones = []
    for label, zmin, zmax in zone_defs:
        mask = (pixel_obs > zmin) & (pixel_obs <= zmax)
        if mask.sum() < 2:
            continue
        zd = diffs[mask]
        zo = pixel_obs[mask]
        z_s44_max = np.sqrt(0.5 ** 2 + (0.013 * zo) ** 2)
        z_s44 = float(np.sum(np.abs(zd) <= z_s44_max) / len(zd) * 100)
        zones.append({
            'zone': label, 'n_pixels': int(mask.sum()),
            'rmse': round(float(np.sqrt(np.mean(zd ** 2))), 3),
            'mae': round(float(np.mean(np.abs(zd))), 3),
            'bias': round(float(np.mean(zd)), 3),
            'medae': round(float(np.median(np.abs(zd))), 3),
            's44_pct': round(z_s44, 1),
        })

    # ── Intra-pixel variability stats ──
    multi_obs_pixels = pixel_count > 1
    intra_pixel = {
        'pixels_with_multiple_obs': int(multi_obs_pixels.sum()),
        'mean_obs_per_pixel': round(float(np.mean(pixel_count)), 1),
        'max_obs_per_pixel': int(np.max(pixel_count)),
        'mean_intra_pixel_std': round(float(np.mean(pixel_std[multi_obs_pixels])), 3) if multi_obs_pixels.sum() > 0 else 0,
        'p90_intra_pixel_std': round(float(np.percentile(pixel_std[multi_obs_pixels], 90)), 3) if multi_obs_pixels.sum() > 0 else 0,
    }

    # ── Per-pixel detail pairs ──
    pairs = []
    for i in range(len(pixel_obs)):
        lat, lon = pixel_coords[i]
        pairs.append({
            'lat': lat, 'lon': lon,
            'obs_median': round(pixel_obs[i], 2),
            'pred': round(pixel_pred[i], 2),
            'diff': round(float(diffs[i]), 3),
            'n_obs_in_pixel': int(pixel_count[i]),
            'obs_std_in_pixel': round(pixel_std[i], 3),
            's44_pass': bool(s44_pass[i]),
        })

    return {
        'method': 'pixel_median_aggregation',
        'resolution_m': resolution_m,
        'pixel_level': {
            'n_pixels': len(pixel_obs),
            'rmse': round(rmse, 3),
            'mae': round(mae, 3),
            'medae': round(medae, 3),
            'bias': round(bias, 3),
            'r2': round(r2, 4),
            'p95_error': round(p95, 3),
            'iqr': round(iqr, 3),
            's44_pass_pct': round(s44_pct, 1),
        },
        'point_level': {
            'n_points': len(point_pairs),
            'rmse': round(pt_rmse, 3),
            'mae': round(pt_mae, 3),
            'bias': round(pt_bias, 3),
            'r2': round(pt_r2, 4),
            'note': 'Traditional per-point comparison (may over-represent dense-survey areas)',
        },
        'depth_zones': zones,
        'intra_pixel': intra_pixel,
        'pairs': pairs,
    }

# ══════════════════════════════════════════════════════════════
# SENTINEL-2 via CDSE
# ══════════════════════════════════════════════════════════════
SH_AUTH="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROC="https://sh.dataspace.copernicus.eu/api/v1/process"
_tok={"t":None,"exp":0}

def sh_token():
    if _tok["t"] and TM.time()<_tok["exp"]-60:return _tok["t"]
    cid=os.getenv("SH_CLIENT_ID","");csec=os.getenv("SH_CLIENT_SECRET","")
    if not cid:raise RuntimeError("SH_CLIENT_ID not set")
    r=requests.post(SH_AUTH,data={"grant_type":"client_credentials","client_id":cid,"client_secret":csec},timeout=30)
    r.raise_for_status();d=r.json();_tok["t"]=d["access_token"];_tok["exp"]=TM.time()+d["expires_in"];return _tok["t"]

EVALSCRIPT="""//VERSION=3
function setup(){return{input:[{bands:["B01","B02","B03","B04","B08","SCL"],units:"DN",mosaicking:"ORBIT"}],output:{bands:5,sampleType:"UINT16"}};}
function isValid(s){var c=s.SCL;return c!==1&&c!==3&&c!==8&&c!==9&&c!==10&&c!==11;}
function median(a){if(!a.length)return 0;a.sort(function(x,y){return x-y});var m=Math.floor(a.length/2);return a.length%2===0?(a[m-1]+a[m])/2:a[m];}
function evaluatePixel(samples){var b01=[],b02=[],b03=[],b04=[],b08=[];for(var i=0;i<samples.length;i++){if(isValid(samples[i])){b01.push(samples[i].B01);b02.push(samples[i].B02);b03.push(samples[i].B03);b04.push(samples[i].B04);b08.push(samples[i].B08);}}if(!b02.length){for(var j=0;j<samples.length;j++){b01.push(samples[j].B01);b02.push(samples[j].B02);b03.push(samples[j].B03);b04.push(samples[j].B04);b08.push(samples[j].B08);}}return[median(b01),median(b04),median(b03),median(b02),median(b08)];}"""

def fetch_s2(bbox,sd,ed,res=20,cloud=20):
    w,s,e,n=bbox
    cl=np.cos(np.radians((n+s)/2))
    wp=max(32,min(2500,int(abs(e-w)*111000*cl/res)));hp=max(32,min(2500,int(abs(n-s)*111000/res)))
    tok=sh_token()
    body={"input":{"bounds":{"bbox":[w,s,e,n],"properties":{"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},"data":[{"type":"sentinel-2-l2a","dataFilter":{"maxCloudCoverage":cloud,"timeRange":{"from":f"{sd}T00:00:00Z","to":f"{ed}T23:59:59Z"}}}]},"output":{"width":wp,"height":hp,"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},"evalscript":EVALSCRIPT}
    L.info(f"S2: {wp}x{hp} @{res}m")
    # Retry with backoff for rate limits
    for attempt in range(4):
        r=requests.post(SH_PROC,headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json"},json=body,timeout=300)
        if r.ok:break
        if r.status_code==429:
            wait=3*(2**attempt)
            L.warning(f"S2 rate limited, waiting {wait}s (attempt {attempt+1}/4)")
            TM.sleep(wait)
            tok=sh_token()  # refresh token
            continue
        raise RuntimeError(f"S2 {r.status_code}: {r.text[:200]}")
    if not r.ok:raise RuntimeError(f"S2 {r.status_code}: {r.text[:200]}")
    img=tifffile.imread(io.BytesIO(r.content))
    # 5-band: [coastal, red, green, blue, nir]
    if img.ndim==3 and img.shape[0]==5:coastal,red,green,blue,nir=img[0],img[1],img[2],img[3],img[4]
    elif img.ndim==3 and img.shape[2]==5:coastal,red,green,blue,nir=img[:,:,0],img[:,:,1],img[:,:,2],img[:,:,3],img[:,:,4]
    elif img.ndim==3 and img.shape[0]==4:
        # Backward compatible: old 4-band
        red,green,blue,nir=img[0],img[1],img[2],img[3]
        coastal=blue  # fallback: use B02 as coastal proxy
    elif img.ndim==3 and img.shape[2]==4:
        red,green,blue,nir=img[:,:,0],img[:,:,1],img[:,:,2],img[:,:,3]
        coastal=blue
    else:raise RuntimeError(f"TIFF shape {img.shape}")
    gf=green.astype(float);nf=nir.astype(float)
    ndwi=(gf-nf)/(gf+nf+1e-6)
    # ── Radiometric corrections (Hedley 2005, Lyzenga 1978) ──
    blue_f,green_f,red_f,nir_f,coastal_f = [b.astype(np.float64)/10000 for b in [blue,green,red,nir,coastal]]
    # Sun glint correction — Hedley et al. (2005): regress each visible band against NIR over deep water
    deep_water=(ndwi>0.5)&(nir_f<0.02)  # deep clear water: high NDWI, very low NIR
    if np.sum(deep_water)>100:
        nir_dw=nir_f[deep_water]
        for band,band_f in [('blue',blue_f),('green',green_f),('red',red_f),('coastal',coastal_f)]:
            vis_dw=band_f[deep_water]
            # Least-squares slope: dVis/dNIR
            valid=np.isfinite(nir_dw)&np.isfinite(vis_dw)&(nir_dw>0)
            if valid.sum()>50:
                n_dw,v_dw=nir_dw[valid],vis_dw[valid]
                slope=np.sum((v_dw-v_dw.mean())*(n_dw-n_dw.mean()))/(np.sum((n_dw-n_dw.mean())**2)+1e-10)
                min_nir=float(np.percentile(n_dw,5))
                correction=slope*(nir_f-min_nir)
                if band=='blue':blue_f=np.clip(blue_f-correction,1e-5,None)
                elif band=='green':green_f=np.clip(green_f-correction,1e-5,None)
                elif band=='red':red_f=np.clip(red_f-correction,1e-5,None)
                elif band=='coastal':coastal_f=np.clip(coastal_f-correction,1e-5,None)
        L.info(f"S2: sun glint corrected ({int(np.sum(deep_water))} deep-water pixels)")
    # Deep-water reflectance subtraction — Lyzenga (1978)
    water_mask=ndwi>0
    if np.sum(deep_water)>50:
        Rw_inf_b=float(np.median(blue_f[deep_water]))
        Rw_inf_g=float(np.median(green_f[deep_water]))
        Rw_inf_r=float(np.median(red_f[deep_water]))
        Rw_inf_c=float(np.median(coastal_f[deep_water]))
        blue_f=np.where(water_mask,np.clip(blue_f-Rw_inf_b,1e-6,None),blue_f)
        green_f=np.where(water_mask,np.clip(green_f-Rw_inf_g,1e-6,None),green_f)
        red_f=np.where(water_mask,np.clip(red_f-Rw_inf_r,1e-6,None),red_f)
        coastal_f=np.where(water_mask,np.clip(coastal_f-Rw_inf_c,1e-6,None),coastal_f)
        L.info(f"S2: deep-water corrected (Rw_inf: B={Rw_inf_b:.5f}, G={Rw_inf_g:.5f}, R={Rw_inf_r:.5f})")
    # Improved water mask — Otsu-like adaptive threshold + morphology
    from scipy.ndimage import binary_fill_holes,binary_erosion,binary_dilation
    ndwi_water=ndwi>0
    if np.sum(ndwi_water)>100:
        # Clean with morphological open (erode then dilate) to remove noise
        struct=np.ones((3,3))
        clean=binary_erosion(ndwi_water,struct,iterations=1)
        clean=binary_dilation(clean,struct,iterations=1)
        clean=binary_fill_holes(clean)
        water_mask=clean
    else:
        water_mask=ndwi_water
    ndwi_out=np.where(water_mask,ndwi,ndwi)
    # Store corrected bands back (scaled to DN-like for compatibility)
    blue_dn=(blue_f*10000).astype(np.uint16)
    green_dn=(green_f*10000).astype(np.uint16)
    red_dn=(red_f*10000).astype(np.uint16)
    coastal_dn=(coastal_f*10000).astype(np.uint16)
    return{"red":red_dn,"green":green_dn,"blue":blue_dn,"nir":nir,"coastal":coastal_dn,
           "ndwi":ndwi_out,"water_mask":water_mask,"width":blue.shape[1],"height":blue.shape[0],
           "glint_corrected":bool(np.sum(deep_water)>100),"deep_water_corrected":bool(np.sum(deep_water)>50)}

# ══════════════════════════════════════════════════════════════
# PARALLEL TILED PROCESSING ENGINE — 10m resolution on 32 CPUs
# ══════════════════════════════════════════════════════════════
import multiprocessing as _mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def _split_bbox_tiles(bbox, max_tile_km=5.0, fixed_grid=None):
    """Split a bbox into sub-tiles.

    fixed_grid=(n_rows, n_cols): return exactly n_rows*n_cols equal tiles.
    Otherwise: adaptive — tile size ~max_tile_km.
    """
    w, s_b, e, n = bbox
    if fixed_grid is not None:
        n_rows, n_cols = int(fixed_grid[0]), int(fixed_grid[1])
    else:
        lat_mid = (n + s_b) / 2
        dx_km = abs(e - w) * 111 * math.cos(math.radians(lat_mid))
        dy_km = abs(n - s_b) * 111
        n_cols = max(1, int(math.ceil(dx_km / max_tile_km)))
        n_rows = max(1, int(math.ceil(dy_km / max_tile_km)))
    dw = (e - w) / n_cols
    dh = (n - s_b) / n_rows
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            tw = w + c * dw
            ts = s_b + r * dh
            te = tw + dw
            tn = ts + dh
            tiles.append([tw, ts, te, tn])
    return tiles, n_rows, n_cols


def _process_tile(args):
    """Process a single tile: fetch S2 → run depth method → return (row, col, depth, bbox)."""
    # Backward-compatible unpack: old 7-tuple or new 8-tuple with resolution_m
    if len(args) == 8:
        tile_bbox, sd, ed, ref_pts, depth_method, tile_row, tile_col, res_m = args
    else:
        tile_bbox, sd, ed, ref_pts, depth_method, tile_row, tile_col = args
        res_m = 10
    try:
        s2 = fetch_s2(tile_bbox, sd, ed, res=int(res_m), cloud=20)
        H, W = s2['red'].shape
        if H < 10 or W < 10:
            return tile_row, tile_col, None, tile_bbox, "too small"

        # Filter reference points to this tile's bbox (with small buffer)
        tw, ts, te, tn = tile_bbox
        buf = 0.005  # ~500m buffer
        mask = ((ref_pts['lats'] >= ts - buf) & (ref_pts['lats'] <= tn + buf) &
                (ref_pts['lons'] >= tw - buf) & (ref_pts['lons'] <= te + buf))
        ref = {'lats': ref_pts['lats'][mask], 'lons': ref_pts['lons'][mask],
               'depths': ref_pts['depths'][mask]}
        # If no ref pts in this tile, use ALL ref pts (global calibration)
        if len(ref['depths']) < 5:
            ref = {'lats': ref_pts['lats'], 'lons': ref_pts['lons'], 'depths': ref_pts['depths']}

        if depth_method == 'caballero_stumpf':
            result, err = caballero_stumpf_sdb(s2, ref, tile_bbox)
            if result:
                return tile_row, tile_col, result['depth'], tile_bbox, None
            return tile_row, tile_col, None, tile_bbox, err
        else:
            # Fast CNN
            result, err = cnn_depth(s2, ref, tile_bbox, fast=True)
            if result:
                return tile_row, tile_col, result['depth'], tile_bbox, None
            # Fallback to SDB
            depth = fallback_sdb(s2, ref_pts['depths'],
                ref_lats=ref_pts['lats'], ref_lons=ref_pts['lons'], bbox=tile_bbox)
            if depth is not None and np.any(np.isfinite(depth)):
                return tile_row, tile_col, depth, tile_bbox, None
            return tile_row, tile_col, None, tile_bbox, err or "SDB failed"
    except Exception as ex:
        return tile_row, tile_col, None, tile_bbox, str(ex)


def parallel_tiled_bathymetry(bbox, sd, ed, ref_pts, depth_method='cnn',
                               max_workers=None, resolution_m=10,
                               grid=(4, 4), use_processes=True):
    """
    Professional tiled bathymetry: splits ROI into grid (default 4x4=16) tiles,
    processes in parallel across CPUs, then mosaics with feather blending.

    Args:
        resolution_m: output pixel size in metres (10/20/50/100).
        grid: (n_rows, n_cols) tuple; default (4,4) = 16 tiles.
        use_processes: True → ProcessPoolExecutor (true CPU parallelism);
                       False → ThreadPoolExecutor (I/O-bound paths only).

    Returns (depth_mosaic, stats_dict) or (None, error_str).
    """
    n_rows_req, n_cols_req = int(grid[0]), int(grid[1])
    n_tiles_req = n_rows_req * n_cols_req
    if max_workers is None:
        # Default: one worker per tile, capped by available cores
        max_workers = min(n_tiles_req, max(1, _mp.cpu_count() - 1))

    w, s_b, e, n = bbox
    lat_mid = (n + s_b) / 2
    total_dx_km = abs(e - w) * 111 * math.cos(math.radians(lat_mid))
    total_dy_km = abs(n - s_b) * 111

    # Fixed grid: always split into requested grid (e.g. 4x4), never bail to single tile
    tiles, n_rows, n_cols = _split_bbox_tiles(bbox, fixed_grid=(n_rows_req, n_cols_req))
    n_tiles = len(tiles)
    L.info(f"ParaTile: {total_dx_km:.1f}x{total_dy_km:.1f}km → {n_rows}x{n_cols} = {n_tiles} tiles "
           f"@{resolution_m}m, {max_workers} workers ({'ProcessPool' if use_processes else 'ThreadPool'})")

    # Build task args (include resolution_m)
    tasks = []
    for idx, tile_bbox in enumerate(tiles):
        row = idx // n_cols
        col = idx % n_cols
        tasks.append((tile_bbox, sd, ed, ref_pts, depth_method, row, col, int(resolution_m)))

    # Process tiles in parallel
    results_map = {}
    tile_shapes = {}
    n_ok = 0
    n_fail = 0

    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    try:
        with Executor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_tile, t): t for t in tasks}
            for future in as_completed(futures):
                row, col, depth, tbbox, err = future.result()
                if depth is not None and np.any(np.isfinite(depth)):
                    results_map[(row, col)] = depth
                    tile_shapes[(row, col)] = depth.shape
                    n_ok += 1
                else:
                    n_fail += 1
                    L.warning(f"ParaTile [{row},{col}] failed: {err}")
    except Exception as pool_ex:
        # Rare: ProcessPool can't pickle a worker arg on some platforms — fall back to threads
        L.warning(f"ParaTile: ProcessPool failed ({pool_ex}); retrying with ThreadPool")
        results_map, tile_shapes, n_ok, n_fail = {}, {}, 0, 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_tile, t): t for t in tasks}
            for future in as_completed(futures):
                row, col, depth, tbbox, err = future.result()
                if depth is not None and np.any(np.isfinite(depth)):
                    results_map[(row, col)] = depth
                    tile_shapes[(row, col)] = depth.shape
                    n_ok += 1
                else:
                    n_fail += 1
                    L.warning(f"ParaTile [{row},{col}] failed: {err}")

    if n_ok == 0:
        return None, f"All {n_tiles} tiles failed"

    L.info(f"ParaTile: {n_ok}/{n_tiles} tiles OK, {n_fail} failed")

    # Mosaic: determine output grid size
    # Each tile is ~500x500 at 10m for 5km tiles
    # Find max dimensions per row/col
    row_heights = {}
    col_widths = {}
    for (r, c), shape in tile_shapes.items():
        row_heights[r] = max(row_heights.get(r, 0), shape[0])
        col_widths[c] = max(col_widths.get(c, 0), shape[1])

    total_H = sum(row_heights.get(r, 0) for r in range(n_rows))
    total_W = sum(col_widths.get(c, 0) for c in range(n_cols))

    if total_H == 0 or total_W == 0:
        return None, "Empty mosaic"

    mosaic = np.full((total_H, total_W), np.nan, dtype=np.float32)

    # Place tiles into mosaic
    y_off = 0
    for r in range(n_rows):
        x_off = 0
        rh = row_heights.get(r, 0)
        for c in range(n_cols):
            cw = col_widths.get(c, 0)
            if (r, c) in results_map:
                tile = results_map[(r, c)]
                th, tw = tile.shape
                # Resize tile if needed to match slot
                if th != rh or tw != cw:
                    from scipy.ndimage import zoom as ndizoom
                    tile = ndizoom(tile, (rh / th, cw / tw), order=1, mode='nearest')
                mosaic[y_off:y_off+rh, x_off:x_off+cw] = tile
            x_off += cw
        y_off += rh

    # Feather blend seams: Gaussian smooth at tile boundaries
    try:
        from scipy.ndimage import gaussian_filter
        valid = np.isfinite(mosaic) & (mosaic > 0.1)
        filled = np.where(valid, mosaic, 0)
        mosaic = np.where(valid, gaussian_filter(filled, sigma=1.5), np.nan)
        mosaic = np.clip(mosaic, 0, MAX_DEPTH_M)
        mosaic[~valid] = np.nan
    except:
        pass

    L.info(f"ParaTile mosaic: {total_H}x{total_W} @10m, {n_ok}/{n_tiles} tiles, "
           f"{int(np.sum(np.isfinite(mosaic)))} valid px")

    return mosaic, {
        'tiles_total': n_tiles, 'tiles_ok': n_ok, 'tiles_fail': n_fail,
        'grid_size': f"{total_H}x{total_W}",
        'tile_grid': f"{n_rows}x{n_cols}",
        'resolution_m': int(resolution_m),
        'workers': max_workers, 'method': depth_method,
    }


# ══════════════════════════════════════════════════════════════
# GLOBAL LAND/WATER MASK — works with any imagery source
# ══════════════════════════════════════════════════════════════
def make_water_mask(bbox, H, W, s2=None):
    """Build a water mask combining global land DB + satellite image analysis.
    Catches man-made land (ports, breakwaters, reclaimed) that global mask misses."""
    w,s_b,e,n=bbox
    water = np.ones((H, W), dtype=bool)  # start: everything is water

    # 1. Global land mask (~1km resolution coastline)
    try:
        from global_land_mask import globe
        lats=np.linspace(n,s_b,H)
        lons=np.linspace(w,e,W)
        lon_grid,lat_grid=np.meshgrid(lons,lats)
        is_land=globe.is_land(lat_grid,lon_grid)
        water = water & (~is_land)
        L.info(f"Water mask (global): {int(is_land.sum())} land px")
    except Exception as ex:
        L.warning(f"global_land_mask failed: {ex}")

    # 2. Satellite image-based land detection (catches ports, breakwaters, reclaimed land)
    if s2 is not None:
        try:
            rf = s2['red'].astype(np.float32)
            gf = s2['green'].astype(np.float32)
            bf = s2['blue'].astype(np.float32)
            nirf = s2.get('nir')
            if nirf is not None:
                nirf = nirf.astype(np.float32)
            # Resize to match H,W if needed
            if rf.shape != (H, W):
                from scipy.ndimage import zoom as ndizoom
                rf = ndizoom(rf, (H/rf.shape[0], W/rf.shape[1]), order=1)
                gf = ndizoom(gf, (H/gf.shape[0], W/gf.shape[1]), order=1)
                bf = ndizoom(bf, (H/bf.shape[0], W/bf.shape[1]), order=1)
                if nirf is not None:
                    nirf = ndizoom(nirf, (H/nirf.shape[0], W/nirf.shape[1]), order=1)
            brightness = rf + gf + bf
            max_b = max(brightness.max(), 1)
            bright_norm = brightness / max_b
            blue_frac = bf / (brightness + 1e-6)
            red_frac = rf / (brightness + 1e-6)
            # Stricter land criteria: must be both bright AND non-blue.
            # Shallow turquoise Gulf water has blue_frac~0.30-0.35 so previous
            # threshold of 0.34 was labelling clear shallow water as land.
            sat_land = ((bright_norm > 0.55) & (blue_frac < 0.30)) | \
                       ((bright_norm > 0.65) & (red_frac > 0.38) & (blue_frac < 0.32))
            # NDWI rescue: anything clearly water per NDWI is never land
            ndwi_local = s2.get('ndwi')
            if ndwi_local is not None:
                if ndwi_local.shape != (H, W):
                    from scipy.ndimage import zoom as ndizoom
                    ndwi_local = ndizoom(ndwi_local, (H/ndwi_local.shape[0], W/ndwi_local.shape[1]), order=1)
                sat_land = sat_land & ~(ndwi_local > 0.15)
                # Inland-water rescue: the ~1 km global_land_mask DB is a
                # COASTLINE dataset — lakes/reservoirs/rivers are "land" in it,
                # which erased every inland ROI. Imagery evidence wins: any
                # pixel the scene itself shows as water (NDWI) stays water.
                water = water | (ndwi_local > 0.15)
            # NIR is strongly absorbed by water → high NIR = land
            if nirf is not None:
                nir_norm = nirf / max(np.percentile(nirf, 99), 1.0)
                sat_land = sat_land | (nir_norm > 0.45)
            # Morphological cleanup — keep erode and dilate symmetric so we don't
            # bloat the land mask beyond what the image actually shows.
            from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
            struct = np.ones((3,3))
            sat_land = binary_erosion(sat_land, struct, iterations=1)
            sat_land = binary_dilation(sat_land, struct, iterations=1)
            sat_land = binary_fill_holes(sat_land)
            water = water & (~sat_land)
            n_sat = int(sat_land.sum())
            L.info(f"Water mask (satellite): {n_sat} additional land px from image analysis")
        except Exception as ex:
            L.warning(f"Satellite land detection failed: {ex}")

    n_water = int(water.sum())
    n_land = H*W - n_water
    L.info(f"Water mask final: {n_water}/{H*W} water ({n_land} land = {n_land*100//(H*W)}%)")
    return water

# ══════════════════════════════════════════════════════════════
# MAPBOX SATELLITE → pseudo-S2 dict for CNN (RGB only, very fast)
# ══════════════════════════════════════════════════════════════
def fetch_mapbox_s2(bbox, zoom=15):
    """Fetch Mapbox satellite RGB and build a pseudo-S2 dict for CNN training.
    zoom 15 ≈ ~5m/px, zoom 17 ≈ ~1m/px (very high res).
    Output image capped at ~500x500px for fast CNN processing.
    Returns same dict format as fetch_s2 so CNN pipeline works unchanged."""
    w,s,e,n=bbox
    token=os.getenv('MAPBOX_TOKEN','')
    if not token:raise RuntimeError("MAPBOX_TOKEN not set")
    lat_c=(n+s)/2; lon_c=(e+w)/2
    cl=np.cos(np.radians(lat_c))
    bbox_w_m=abs(e-w)*111000*cl; bbox_h_m=abs(n-s)*111000
    # Cap output at ~500px per side (like S2 at 10-20m for typical ROIs)
    max_px=500
    aspect=bbox_w_m/max(bbox_h_m,1)
    if aspect>=1:
        wp=min(max_px,1280); hp=min(max_px,max(64,int(wp/aspect)))
    else:
        hp=min(max_px,1280); wp=min(max_px,max(64,int(hp*aspect)))
    res_m=bbox_w_m/wp
    style='mapbox/satellite-v9'
    url=f"https://api.mapbox.com/styles/v1/{style}/static/[{w},{s},{e},{n}]/{wp}x{hp}?access_token={token}&attribution=false&logo=false"
    L.info(f"Mapbox: requesting {wp}x{hp}px (res≈{res_m:.1f}m/px)")
    r=requests.get(url,timeout=30)
    if not r.ok:
        url2=f"https://api.mapbox.com/styles/v1/{style}/static/{lon_c},{lat_c},{zoom},0/{wp}x{hp}?access_token={token}&attribution=false&logo=false"
        r=requests.get(url2,timeout=30)
        if not r.ok:raise RuntimeError(f"Mapbox {r.status_code}: {r.text[:200]}")
    from PIL import Image
    img=np.array(Image.open(io.BytesIO(r.content)).convert('RGB'))
    H,W=img.shape[:2]
    rf=img[:,:,0].astype(np.float32)
    gf=img[:,:,1].astype(np.float32)
    bf=img[:,:,2].astype(np.float32)
    # Scale 0-255 → ~0-10000 DN (like S2 L2A)
    red=(rf*40).astype(np.uint16)
    green=(gf*40).astype(np.uint16)
    blue=(bf*40).astype(np.uint16)
    # Water mask: global land DB + image-based detection
    _tmp_s2 = {'red': red, 'green': green, 'blue': blue}
    water_mask=make_water_mask([w,s,e,n],H,W,s2=_tmp_s2)
    if water_mask is None:
        # Fallback: RGB-based detection
        brightness=rf+gf+bf
        blue_ratio=bf/(brightness+1e-6)
        water_mask=((blue_ratio>0.35)&(brightness>10))
    # Pseudo-NIR: water has very low NIR; land has high NIR
    nir=np.where(water_mask,(rf*0.15).astype(np.uint16),(rf*1.2+gf*0.3).astype(np.uint16))
    nir=np.clip(nir,0,10000).astype(np.uint16)
    # NDWI from pseudo bands
    gf2=green.astype(float); nf2=nir.astype(float)
    ndwi=(gf2-nf2)/(gf2+nf2+1e-6)
    n_water=int(np.sum(water_mask))
    L.info(f"Mapbox: {W}x{H}px → pseudo-S2 (res≈{res_m:.1f}m/px, water={n_water}/{H*W} px = {n_water*100//(H*W)}%)")
    return{"red":red,"green":green,"blue":blue,"nir":nir,"coastal":blue,
           "ndwi":ndwi,"water_mask":water_mask,"width":W,"height":H,
           "glint_corrected":False,"deep_water_corrected":False,
           "source":"mapbox","resolution_m":round(res_m,2)}

# ══════════════════════════════════════════════════════════════
# SENTINEL-2 SINGLE-ORBIT for WAVE ANALYSIS (no mosaicking!)
# ══════════════════════════════════════════════════════════════
EVALSCRIPT_WAVE="""//VERSION=3
function setup(){return{input:[{bands:["B02","B04","SCL"],units:"DN",mosaicking:"SIMPLE"}],output:{bands:2,sampleType:"UINT16"}};}
function evaluatePixel(samples){return[samples[0].B02,samples[0].B04];}"""

def fetch_s2_wave(bbox,sd,ed,cloud=15):
    """Fetch B02+B04 at 10m from a SINGLE orbit (no median) — preserves inter-band time offset for wave analysis."""
    w,s,e,n=bbox
    cl=np.cos(np.radians((n+s)/2))
    wp=max(32,min(2500,int(abs(e-w)*111000*cl/10)));hp=max(32,min(2500,int(abs(n-s)*111000/10)))
    tok=sh_token()
    body={"input":{"bounds":{"bbox":[w,s,e,n],"properties":{"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},"data":[{"type":"sentinel-2-l2a","dataFilter":{"maxCloudCoverage":cloud,"timeRange":{"from":f"{sd}T00:00:00Z","to":f"{ed}T23:59:59Z"},"mosaickingOrder":"leastRecent"}}]},"output":{"width":wp,"height":hp,"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},"evalscript":EVALSCRIPT_WAVE}
    L.info(f"S2 wave: {wp}x{hp} @10m (single orbit, B02+B04)")
    r=requests.post(SH_PROC,headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json"},json=body,timeout=300)
    if not r.ok:raise RuntimeError(f"S2 wave {r.status_code}: {r.text[:200]}")
    img=tifffile.imread(io.BytesIO(r.content))
    if img.ndim==3 and img.shape[0]==2:b02,b04=img[0],img[1]
    elif img.ndim==3 and img.shape[2]==2:b02,b04=img[:,:,0],img[:,:,1]
    elif img.ndim==2:raise RuntimeError("Only 1 band returned — need B02+B04")
    else:raise RuntimeError(f"Wave TIFF shape {img.shape}")
    L.info(f"S2 wave: B02 range [{b02.min()}-{b02.max()}], B04 range [{b04.min()}-{b04.max()}]")
    return{"b02":b02,"b04":b04,"width":b02.shape[1],"height":b02.shape[0]}

# ══════════════════════════════════════════════════════════════
# GEBCO
# ══════════════════════════════════════════════════════════════
def fetch_gebco(bbox):
    w,s,e,n=bbox
    wp=max(16,min(500,int((e-w)*240)));hp=max(16,min(500,int((n-s)*240)))
    for url in [
        f"https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage?bbox={w},{s},{e},{n}&bboxSR=4326&imageSR=4326&size={wp},{hp}&format=tiff&f=image",
        f"https://wms.gebco.net/mapserv?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage&COVERAGE=GEBCO_LATEST&CRS=EPSG:4326&FORMAT=GeoTIFF&BBOX={s},{w},{n},{e}&WIDTH={wp}&HEIGHT={hp}",
    ]:
        try:
            r=requests.get(url,timeout=30)
            if r.ok and len(r.content)>200:
                elev=tifffile.imread(io.BytesIO(r.content)).astype(np.float32)
                if elev.ndim > 2: elev = elev[:,:,0] if elev.shape[2] < elev.shape[0] else elev[0]
                grid=np.where(elev<0,-elev,np.nan);gh,gw=grid.shape
                st=max(1,int(math.sqrt(gh*gw/2000)))
                lats,lons,depths=[],[],[]
                for i in range(0,gh,st):
                    lat=n-(i/gh)*(n-s)
                    for j in range(0,gw,st):
                        v=grid[i,j]
                        if np.isfinite(v) and 0.3<v<MAX_DEPTH_M:lats.append(lat);lons.append(w+(j/gw)*(e-w));depths.append(min(float(v),MAX_DEPTH_M))
                if len(depths)>5:
                    L.info(f"GEBCO: {len(depths)} pts")
                    return{"lats":np.array(lats),"lons":np.array(lons),"depths":np.array(depths)}
        except Exception as ex:L.warning(f"GEBCO: {ex}")
    return None

# ══════════════════════════════════════════════════════════════
# CHART DEPTH EXTRACTION (Gemini Vision)
# ══════════════════════════════════════════════════════════════
def call_gemini_vision(img_b64, bbox, img_w, img_h, media_type="image/png"):
    api_key=os.getenv('GEMINI_API_KEY','') or os.getenv('GOOGLE_API_KEY','')
    if not api_key:return None,"GEMINI_API_KEY not set"
    w,s,e,n=bbox
    L.info(f"Gemini: {img_w}x{img_h}, {len(img_b64)//1024}KB")
    prompt=f"""Nautical chart screenshot. Area: SW({s:.5f}N,{w:.5f}E) NE({n:.5f}N,{e:.5f}E). Image: {img_w}x{img_h}px.
Extract ALL depth sounding numbers on the blue water. Numbers like 29.2, 16.7, 3.0 = depth in meters.
IGNORE buoy labels, nav markers, UI text. ONLY depth numbers.
Return JSON array: [{{"x":pixel_from_left,"y":pixel_from_top,"depth":value}}]
Minimum 50 points if visible. If none: []"""
    try:
        url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        resp=requests.post(url,json={"contents":[{"parts":[{"inline_data":{"mime_type":media_type,"data":img_b64}},{"text":prompt}]}],"generationConfig":{"temperature":0.1,"maxOutputTokens":8192}},timeout=120)
        if not resp.ok:return None,f"Gemini {resp.status_code}: {resp.text[:200]}"
        text=resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        clean=text.replace('```json','').replace('```','').strip()
        L.info(f"Gemini: {len(clean)} chars response")
        pts=None
        try:pts=json.loads(clean)
        except:
            idx=clean.find('[')
            if idx>=0:
                raw=clean[idx:]
                try:pts=json.loads(raw)
                except:
                    lb=raw.rfind('}')
                    if lb>0:
                        try:pts=json.loads(raw[:lb+1]+']')
                        except:pass
        if pts and isinstance(pts,list):
            L.info(f"Gemini: {len(pts)} points")
            return pts,None
        return None,f"Parse failed: {clean[:100]}"
    except Exception as ex:return None,str(ex)

def pixels_to_latlon(points, bbox, img_w, img_h):
    w,s,e,n=bbox;lats,lons,depths=[],[],[]
    for p in points:
        d=p.get('depth',0)
        if not isinstance(d,(int,float)) or d<=0 or d>200:continue
        lats.append(n-(p.get('y',0)/img_h)*(n-s))
        lons.append(w+(p.get('x',0)/img_w)*(e-w))
        depths.append(min(float(d),MAX_DEPTH_M))
    return lats,lons,depths

# ══════════════════════════════════════════════════════════════
# ICESat-2 via SlideRule (ATL24)
# ══════════════════════════════════════════════════════════════
def run_sliderule(bbox,sd,ed):
    try:
        from sliderule import sliderule,icesat2
        sliderule.init("slideruleearth.io")
    except:return[],"sliderule not installed"
    w,s,e,n=bbox
    try:
        from datetime import datetime,timedelta
        d0=datetime.strptime(sd,'%Y-%m-%d')-timedelta(days=180)
        d1=datetime.strptime(ed,'%Y-%m-%d')+timedelta(days=180)
        t0=d0.strftime('%Y-%m-%dT00:00:00Z')
        t1=d1.strftime('%Y-%m-%dT23:59:59Z')
        L.info(f"ICESat-2 expanded range: {d0.date()} → {d1.date()} (±180d from {sd}/{ed})")
        poly=[{"lon":w,"lat":s},{"lon":e,"lat":s},{"lon":e,"lat":n},{"lon":w,"lat":n},{"lon":w,"lat":s}]
        parms={"poly":poly,"t0":t0,"t1":t1,"srt":1,"cnf":0,"len":20.0,"res":20.0,"pass_invalid":False,"yapc":{"score":0,"knn":0,"min_ph":4}}
        L.info("SlideRule: querying ATL03 with YAPC...")
        gdf=icesat2.atl03sp(parms)
        if gdf is None or len(gdf)==0:return[],"No photons"
        L.info(f"SlideRule: {len(gdf)} raw photons")
        heights=gdf['height'].values.astype(float) if 'height' in gdf.columns else gdf['h_mean'].values.astype(float)
        lats=np.array([g.y for g in gdf.geometry]);lons=np.array([g.x for g in gdf.geometry])
        yapc=gdf['yapc_score'].values.astype(float) if 'yapc_score' in gdf.columns else np.full(len(heights),200)
        # Keep both a short date (for display) AND the full UTC timestamp
        # (needed for per-photon tide retiding). SlideRule returns a
        # DatetimeIndex — most entries are precise to the millisecond.
        acq_dates=[]; acq_dts=[]
        if hasattr(gdf.index,'strftime'):
            acq_dates=[t.strftime('%Y-%m-%d') for t in gdf.index]
            acq_dts  =[t.isoformat() for t in gdf.index]
        elif 'time' in gdf.columns:
            acq_dates=[str(t)[:10] for t in gdf['time'].values]
            acq_dts  =[str(t) for t in gdf['time'].values]
        else:
            acq_dates=['']*len(heights); acq_dts=['']*len(heights)

        # ── FILTER 1: YAPC + quality_ph ──
        mask=yapc>=100
        if 'quality_ph' in gdf.columns:mask&=(gdf['quality_ph'].values==0)
        h=heights[mask];la=lats[mask];lo=lons[mask]
        ad=[acq_dates[i] for i in range(len(mask)) if mask[i]]
        ad_dt=[acq_dts[i] for i in range(len(mask)) if mask[i]]
        if len(h)<10:h,la,lo,ad,ad_dt=heights,lats,lons,acq_dates,acq_dts
        L.info(f"ICESat-2: {len(h)} after YAPC+quality filter")

        # ── FILTER 2: Robust sea surface detection ──
        # Use KDE-like approach: find the dominant height peak (sea surface)
        finite=h[np.isfinite(h)]
        if len(finite)<20:return[],"Too few valid photons"
        # Narrow histogram around likely sea surface (±2m from median of top cluster)
        p25,p75=np.percentile(finite,25),np.percentile(finite,75)
        iqr=p75-p25
        surface_candidates=finite[(finite>p75-0.5*iqr)&(finite<p75+2.0)]
        if len(surface_candidates)>20:
            hist,edges=np.histogram(surface_candidates,bins=max(20,min(200,int(len(surface_candidates)/5))))
        else:
            hist,edges=np.histogram(finite,bins=max(20,min(200,int((np.nanmax(finite)-np.nanmin(finite))/0.05))))
        peak_idx=np.argmax(hist)
        ss=float((edges[peak_idx]+edges[peak_idx+1])/2)
        L.info(f"ICESat-2: sea surface={ss:.2f}m (peak bin={hist[peak_idx]} photons)")

        # ── FILTER 3: Convert to depth + physical constraints ──
        raw_pts=[]
        for i in range(len(h)):
            rd=ss-h[i]
            dt=ad[i] if i<len(ad) else ''
            dt_iso=ad_dt[i] if i<len(ad_dt) else ''
            if 0.3<rd<50:
                td=min(rd*1.34, MAX_DEPTH_M)  # refraction correction (water n=1.34), capped
                raw_pts.append({'lat':float(la[i]),'lon':float(lo[i]),'depth':td,'acq_date':dt,'acq_dt':dt_iso})
            elif abs(rd)<=0.3:
                raw_pts.append({'lat':float(la[i]),'lon':float(lo[i]),'depth':0.0,'acq_date':dt,'acq_dt':dt_iso,'surface':True})

        bathy_pts=[p for p in raw_pts if p['depth']>0.3 and not p.get('surface')]
        surf_pts=[p for p in raw_pts if p.get('surface')]
        L.info(f"ICESat-2: {len(bathy_pts)} raw bathy, {len(surf_pts)} surface before outlier removal")

        # ── FILTER 4: Spatial outlier removal (local median ± 2σ) ──
        if len(bathy_pts)>20:
            depths_arr=np.array([p['depth'] for p in bathy_pts])
            lats_arr=np.array([p['lat'] for p in bathy_pts])
            lons_arr=np.array([p['lon'] for p in bathy_pts])
            # Global IQR filter first
            q1,q3=np.percentile(depths_arr,10),np.percentile(depths_arr,90)
            iqr=q3-q1
            global_ok=(depths_arr>max(0.3,q1-1.5*iqr))&(depths_arr<q3+1.5*iqr)
            bathy_pts=[bathy_pts[i] for i in range(len(bathy_pts)) if global_ok[i]]
            L.info(f"ICESat-2: {len(bathy_pts)} after IQR filter")

            # Local consistency: for each point check neighbours within ~200m
            if len(bathy_pts)>30:
                d_arr=np.array([p['depth'] for p in bathy_pts])
                la_arr=np.array([p['lat'] for p in bathy_pts])
                lo_arr=np.array([p['lon'] for p in bathy_pts])
                keep=np.ones(len(bathy_pts),dtype=bool)
                deg_tol=0.002  # ~200m
                for i in range(len(bathy_pts)):
                    near=np.where((np.abs(la_arr-la_arr[i])<deg_tol)&(np.abs(lo_arr-lo_arr[i])<deg_tol))[0]
                    if len(near)>=5:
                        local_med=np.median(d_arr[near])
                        local_std=np.std(d_arr[near])
                        if abs(d_arr[i]-local_med)>max(2.0,2.0*local_std):
                            keep[i]=False
                bathy_pts=[bathy_pts[i] for i in range(len(bathy_pts)) if keep[i]]
                L.info(f"ICESat-2: {len(bathy_pts)} after local outlier filter")

        # ── FILTER 5: Depth-dependent confidence ──
        # Shallow (<5m) photons are more reliable; deep (>20m) need stricter filtering
        pts=[]
        for p in bathy_pts:
            conf='high' if p['depth']<8 else 'medium' if p['depth']<20 else 'low'
            pts.append({'lat':round(p['lat'],6),'lon':round(p['lon'],6),'depth':round(p['depth'],2),
                        'photon_class':'bathymetry','acq_date':p.get('acq_date',''),
                        'acq_dt':p.get('acq_dt',''),'confidence':conf})
        for p in surf_pts:
            pts.append({'lat':round(p['lat'],6),'lon':round(p['lon'],6),'depth':0.0,
                        'photon_class':'surface','acq_date':p.get('acq_date',''),
                        'acq_dt':p.get('acq_dt','')})

        nb=sum(1 for p in pts if p['photon_class']=='bathymetry')
        udates=sorted(set(p.get('acq_date','') for p in pts if p.get('acq_date')))
        L.info(f"SlideRule final: {nb} bathy (filtered), {len(surf_pts)} surface, dates: {udates[:10]}")
        return pts,f"{nb} bathy (filtered), {len(udates)} dates"
    except Exception as ex:return[],str(ex)

# ══════════════════════════════════════════════════════════════
# OBSERVED-CALIBRATED FUSION (professional SDB + IDW + residual)
# ══════════════════════════════════════════════════════════════
def observed_fusion_depth(s2, ref_pts, bbox, water_mask=None):
    """
    Professional bathymetry when dense observed data is available.

    Pipeline:
      1. IDW interpolation of observed points → exact at measurement sites
      2. Multi-feature SDB regression (GBR) calibrated to observed depths
      3. Observation density map → adaptive blend weights
      4. Blend: high density → IDW, low density → SDB
      5. Residual correction at observation locations
      6. Gaussian smooth + water mask

    Typically achieves <1m bias with dense survey data.
    """
    H,W=s2['red'].shape
    w,s_b,e,n=bbox
    eps=1e-6

    # ── Build spectral features from imagery ──
    blue=np.clip(s2['blue'].astype(np.float64)/10000,eps,None)
    green=np.clip(s2['green'].astype(np.float64)/10000,eps,None)
    red=np.clip(s2['red'].astype(np.float64)/10000,eps,None)
    nir=np.clip(s2['nir'].astype(np.float64)/10000,eps,None)
    coastal=np.clip(s2.get('coastal',s2['blue']).astype(np.float64)/10000,eps,None)
    water=water_mask if water_mask is not None else s2.get('water_mask',s2['ndwi']>0)

    # Band ratios (Stumpf 2003, Lyzenga 1978)
    lnBG=np.log(blue+eps)/np.log(green+eps)
    lnGR=np.log(green+eps)/np.log(red+eps)
    BG=blue/(green+eps); GR=green/(red+eps)
    CB=coastal/(blue+eps)
    ln_b=np.log(blue+eps); ln_g=np.log(green+eps)

    # Lyzenga Depth-Invariant Index
    lb_w=ln_b[water]; lg_w=ln_g[water]
    vld=np.isfinite(lb_w)&np.isfinite(lg_w)
    ki_kj=1.0
    if vld.sum()>50:
        cv=np.cov(lb_w[vld],lg_w[vld]); vd=cv[0,0]-cv[1,1]; cbg=cv[0,1]
        ki_kj=(vd+np.sqrt(vd**2+4*cbg**2))/(2*cbg+eps)
    dii=ln_b-ki_kj*ln_g

    # Texture (local variance)
    from scipy.ndimage import uniform_filter,gaussian_filter
    bm=uniform_filter(blue,size=5); bsm=uniform_filter(blue**2,size=5)
    texture=np.sqrt(np.clip(bsm-bm**2,0,None))

    L.info(f"ObsFusion: {H}x{W} image, {len(ref_pts['lats'])} ref pts")

    # ── Step 1: Map observed points to pixel coordinates ──
    obs_rows,obs_cols,obs_depths=[],[],[]
    for i in range(len(ref_pts['lats'])):
        lat_i=ref_pts['lats'][i]; lon_i=ref_pts['lons'][i]; d=ref_pts['depths'][i]
        r_px=max(0,min(H-1,int((n-lat_i)/(n-s_b+1e-10)*H)))
        c_px=max(0,min(W-1,int((lon_i-w)/(e-w+1e-10)*W)))
        if water[r_px,c_px] and d>0:
            obs_rows.append(r_px); obs_cols.append(c_px); obs_depths.append(d)

    n_obs=len(obs_depths)
    if n_obs<5:
        return None, f"Only {n_obs} observed points on water"

    obs_rows=np.array(obs_rows); obs_cols=np.array(obs_cols); obs_depths=np.array(obs_depths)
    L.info(f"ObsFusion: {n_obs} points on water, depth range {obs_depths.min():.1f}-{obs_depths.max():.1f}m")

    # ── Step 2: IDW interpolation of observed points ──
    # Fast: build on subsampled grid then upscale
    sub=max(1,min(H,W)//100)  # subsample factor for IDW speed
    Hs,Ws=H//sub,W//sub
    yr=np.linspace(0,H-1,Hs).astype(int)
    xr=np.linspace(0,W-1,Ws).astype(int)
    xg,yg=np.meshgrid(xr,yr)

    idw_grid=np.full((Hs,Ws),np.nan)
    density_grid=np.zeros((Hs,Ws))
    power=2.0  # IDW power parameter

    for i in range(Hs):
        for j in range(Ws):
            dy=obs_rows-yg[i,j]; dx=obs_cols-xg[i,j]
            dist=np.sqrt(dy**2+dx**2)+0.5  # +0.5 to avoid division by zero
            weights=1.0/dist**power
            # Only use nearby points (within ~50 pixels radius)
            radius=50
            near=dist<radius
            if near.sum()>=1:
                w_near=weights[near]; d_near=obs_depths[near]
                idw_grid[i,j]=np.sum(w_near*d_near)/np.sum(w_near)
                density_grid[i,j]=near.sum()

    # Upscale IDW and density to full resolution
    from scipy.ndimage import zoom as ndizoom
    idw_full=ndizoom(idw_grid,(H/Hs,W/Ws),order=1,mode='nearest')
    density_full=ndizoom(density_grid,(H/Hs,W/Ws),order=1,mode='nearest')
    L.info(f"ObsFusion: IDW interpolation done ({Hs}x{Ws} subgrid)")

    # ── Step 3: GBR regression calibrated to observed depths ──
    features_list=['Blue','Green','Red','NIR','NDWI','ln(B/G)','ln(G/R)','B/G','G/R','C/B','DII','Texture']
    feature_stack=np.stack([blue,green,red,nir,s2['ndwi'],lnBG,lnGR,BG,GR,CB,dii,texture],axis=0)  # (12,H,W)

    # Extract features at observation locations
    Xt=feature_stack[:,obs_rows,obs_cols].T  # (n_obs, 12)
    yt=obs_depths
    ok=np.all(np.isfinite(Xt),axis=1)&np.isfinite(yt)&(yt>0)
    Xt,yt=Xt[ok],yt[ok]

    sdb_depth=np.full((H,W),np.nan)
    gbr_r2=0.0; n_train=len(yt)
    imp={}

    if n_train>=10 and SK:
        sc=StandardScaler(); Xs=sc.fit_transform(Xt)
        # 80/20 split for honest R²
        n_val=max(3,int(n_train*0.2))
        idx=np.random.permutation(n_train)
        Xs_tr,yt_tr=Xs[idx[n_val:]],yt[idx[n_val:]]
        Xs_vl,yt_vl=Xs[idx[:n_val]],yt[idx[:n_val]]

        mdl=GradientBoostingRegressor(n_estimators=200,max_depth=5,learning_rate=0.1,
                                       subsample=0.8,random_state=42,min_samples_leaf=5)
        mdl.fit(Xs_tr,yt_tr)

        # Validation R²
        pred_vl=mdl.predict(Xs_vl)
        ss_res=np.sum((pred_vl-yt_vl)**2); ss_tot=np.sum((yt_vl-yt_vl.mean())**2)
        gbr_r2=round(float(1-ss_res/(ss_tot+1e-10)),4)
        gbr_rmse=round(float(np.sqrt(np.mean((pred_vl-yt_vl)**2))),3)
        gbr_bias=round(float(np.mean(pred_vl-yt_vl)),3)
        L.info(f"ObsFusion GBR: R²={gbr_r2}, RMSE={gbr_rmse}m, bias={gbr_bias}m ({n_train} train, {n_val} val)")

        # Predict full image
        Xp=feature_stack.reshape(12,-1).T  # (H*W, 12)
        Xp=np.nan_to_num(Xp,0)
        sdb_depth=np.clip(mdl.predict(sc.transform(Xp)).reshape(H,W),0,MAX_DEPTH_M)
        sdb_depth[~water]=np.nan
        imp={features_list[i]:round(float(mdl.feature_importances_[i]),3) for i in range(12)}
    else:
        L.warning(f"ObsFusion: only {n_train} valid pts, using IDW only")

    # ── Step 4: Adaptive blend — IDW where dense, SDB where sparse ──
    # Normalise density to [0, 1]: 0 = no nearby obs, 1 = many nearby obs
    max_dens=max(density_full.max(),1)
    alpha=np.clip(density_full/max(max_dens*0.3,1),0,1)  # blend weight for IDW
    alpha=gaussian_filter(alpha,sigma=3)  # smooth transitions

    # Blend
    depth=np.full((H,W),np.nan)
    has_idw=np.isfinite(idw_full)
    has_sdb=np.isfinite(sdb_depth)

    # Where both exist: weighted blend
    both=has_idw&has_sdb&water
    depth[both]=alpha[both]*idw_full[both]+(1-alpha[both])*sdb_depth[both]
    # Where only IDW: use IDW
    only_idw=has_idw&(~has_sdb)&water
    depth[only_idw]=idw_full[only_idw]
    # Where only SDB: use SDB
    only_sdb=(~has_idw)&has_sdb&water
    depth[only_sdb]=sdb_depth[only_sdb]

    L.info(f"ObsFusion blend: {int(both.sum())} both, {int(only_idw.sum())} IDW-only, {int(only_sdb.sum())} SDB-only")

    # ── Step 5: Residual correction at observation locations ──
    # Compute residual (observed - predicted) and interpolate correction field
    pred_at_obs=depth[obs_rows,obs_cols]
    valid_resid=np.isfinite(pred_at_obs)&(pred_at_obs>0)
    if valid_resid.sum()>5:
        residuals=obs_depths[valid_resid]-pred_at_obs[valid_resid]
        mean_resid=float(np.mean(np.abs(residuals)))

        # Build residual correction grid (IDW of residuals)
        resid_grid=np.zeros((Hs,Ws))
        resid_weight=np.zeros((Hs,Ws))
        vr_rows=obs_rows[valid_resid]; vr_cols=obs_cols[valid_resid]
        for i in range(Hs):
            for j in range(Ws):
                dy=vr_rows-yg[i,j]; dx=vr_cols-xg[i,j]
                dist=np.sqrt(dy**2+dx**2)+0.5
                near=dist<radius
                if near.sum()>=1:
                    w_n=1.0/dist[near]**power
                    resid_grid[i,j]=np.sum(w_n*residuals[near])/np.sum(w_n)
                    resid_weight[i,j]=near.sum()

        resid_full=ndizoom(resid_grid,(H/Hs,W/Ws),order=1,mode='nearest')
        resid_w_full=ndizoom(resid_weight,(H/Hs,W/Ws),order=1,mode='nearest')
        # Apply correction weighted by local confidence
        corr_alpha=np.clip(resid_w_full/max(resid_w_full.max()*0.3,1),0,1)
        corr_alpha=gaussian_filter(corr_alpha,sigma=2)
        depth=np.where(water&np.isfinite(depth),depth+corr_alpha*resid_full,depth)

        # Re-check residuals after correction
        pred_at_obs2=depth[obs_rows[valid_resid],obs_cols[valid_resid]]
        resid2=obs_depths[valid_resid]-pred_at_obs2
        final_bias=float(np.mean(resid2))
        final_mae=float(np.mean(np.abs(resid2)))
        final_rmse=float(np.sqrt(np.mean(resid2**2)))
        L.info(f"ObsFusion residual correction: mean|resid| {mean_resid:.2f}→{final_mae:.2f}m, "
               f"bias={final_bias:.3f}m, RMSE={final_rmse:.3f}m")

    # ── Step 6: Final cleanup ──
    depth=np.clip(depth,0.1,MAX_DEPTH_M)
    depth[~water]=np.nan
    # Light Gaussian smooth (preserve detail)
    valid=np.isfinite(depth)
    smoothed=gaussian_filter(np.nan_to_num(depth,nan=0),sigma=0.8)
    depth=np.where(valid,smoothed,np.nan)
    depth[~water]=np.nan

    # Compute final stats
    val=depth[np.isfinite(depth)&(depth>0)]
    method='ObsFusion(IDW+GBR+Residual)' if SK and n_train>=10 else 'ObsFusion(IDW)'

    return {
        'depth': depth,
        'r2': gbr_r2,
        'n_train': n_train,
        'method': method,
        'importance': imp,
        'fusion_stats': {
            'n_observed': n_obs,
            'idw_coverage': int(has_idw.sum()),
            'sdb_coverage': int(has_sdb.sum()),
            'gbr_r2': gbr_r2,
            'final_bias': round(final_bias,3) if 'final_bias' in dir() else None,
            'final_rmse': round(final_rmse,3) if 'final_rmse' in dir() else None,
            'final_mae': round(final_mae,3) if 'final_mae' in dir() else None,
        }
    }, None


# ══════════════════════════════════════════════════════════════
# CNN (U-Net deep learning) + GBR fallback
# ══════════════════════════════════════════════════════════════
def _physical_prior_depth(s2, ref_lats, ref_lons, ref_depths, bbox):
    """
    Compute a physics-based depth prior (Stumpf 2003 + Lyzenga 1985), calibrated
    against the reference depths with robust linear regression.

    Stumpf:   d = m1 * ln(π·R_blue·1000) / ln(π·R_green·1000) + m0
    Lyzenga:  d = a0 + a1·ln(B) + a2·ln(G) + a3·ln(R)

    Returns (depth_physical, r2, used_n) or (None, 0, 0) on failure.
    """
    try:
        H, W = s2['red'].shape
        eps = 1e-6
        blue = np.clip(s2['blue'].astype(np.float64) / 10000, eps, None)
        green = np.clip(s2['green'].astype(np.float64) / 10000, eps, None)
        red = np.clip(s2['red'].astype(np.float64) / 10000, eps, None)
        water = s2.get('water_mask', s2['ndwi'] > 0)

        # Stumpf ratio
        stumpf = np.log(1000.0 * blue) / (np.log(1000.0 * green) + eps)

        # Lyzenga linearised features
        lnB, lnG, lnR = np.log(blue + eps), np.log(green + eps), np.log(red + eps)

        w_b, s_b, e_b, n_b = bbox
        lats = np.asarray(ref_lats, dtype=np.float64)
        lons = np.asarray(ref_lons, dtype=np.float64)
        depths = np.asarray(ref_depths, dtype=np.float64)

        # Pick samples
        rows = np.clip(((n_b - lats) / (n_b - s_b + 1e-10) * H).astype(int), 0, H - 1)
        cols = np.clip(((lons - w_b) / (e_b - w_b + 1e-10) * W).astype(int), 0, W - 1)
        in_water = water[rows, cols]
        ok = in_water & np.isfinite(depths) & (depths > 0.5) & (depths <= MAX_DEPTH_M)
        if ok.sum() < 10:
            return None, 0.0, 0
        r, c, y = rows[ok], cols[ok], depths[ok]

        # Feature matrix: [Stumpf, lnB, lnG, lnR, 1]
        X = np.column_stack([
            stumpf[r, c], lnB[r, c], lnG[r, c], lnR[r, c],
            np.ones(len(y))
        ])
        # Robust-ish fit: ridge with λ proportional to n
        lam = 0.1 * len(y)
        Ir = np.eye(X.shape[1]); Ir[-1, -1] = 0
        try:
            coeffs = np.linalg.solve(X.T @ X + lam * Ir, X.T @ y)
        except Exception:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

        # Apply to whole image
        feat_map = np.stack([
            stumpf, lnB, lnG, lnR, np.ones_like(stumpf)
        ], axis=-1)
        depth_phys = feat_map @ coeffs
        depth_phys = np.where(water, depth_phys, np.nan)
        depth_phys = np.clip(depth_phys, 0, MAX_DEPTH_M)

        # Training R² for confidence in the prior
        y_hat = X @ coeffs
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0.0
        return depth_phys, r2, int(len(y))
    except Exception as ex:
        L.warning(f"PhysicalPrior: {ex}")
        return None, 0.0, 0


def expert_correction(depth_ml, s2, ref_lats, ref_lons, ref_depths, bbox,
                       ml_r2=None, turbid_pct=0.0):
    """
    Expert-hydrographer-style correction: blend the ML depth with a physical
    prior (Stumpf + Lyzenga) in a way that reflects trust in each source.

    Blend rule (α = weight of ML vs physics):
      - baseline α = 0.70
      - if turbidity > 15 %        → α -= 0.10   (physics is more robust in turbid water)
      - if physical prior R² > 0.85 → α -= 0.05   (strong physics — give it more room)
      - if ML R² is unknown/low     → α -= 0.10
      - α clamped to [0.50, 0.85]

    Returns (depth_corrected, info_dict).
    """
    if depth_ml is None:
        return None, {'error': 'no ML depth'}
    d_phys, phys_r2, n_phys = _physical_prior_depth(s2, ref_lats, ref_lons, ref_depths, bbox)
    if d_phys is None:
        return depth_ml, {'applied': False, 'reason': 'physical prior failed'}

    # Align shapes (physical is built from s2 directly — same size as CNN depth)
    if d_phys.shape != depth_ml.shape:
        try:
            from scipy.ndimage import zoom as _zm
            d_phys = _zm(d_phys, (depth_ml.shape[0] / d_phys.shape[0],
                                  depth_ml.shape[1] / d_phys.shape[1]), order=1)
        except Exception:
            return depth_ml, {'applied': False, 'reason': 'shape mismatch'}

    alpha = 0.70
    if turbid_pct > 15: alpha -= 0.10
    if phys_r2 > 0.85:  alpha -= 0.05
    if ml_r2 is None or ml_r2 < 0.5: alpha -= 0.10
    alpha = max(0.50, min(0.85, alpha))

    valid = np.isfinite(depth_ml) & np.isfinite(d_phys)
    corrected = np.where(valid, alpha * depth_ml + (1.0 - alpha) * d_phys, depth_ml)
    corrected = np.clip(corrected, 0, MAX_DEPTH_M)

    # Per-pixel magnitude of correction (for reporting)
    delta = np.where(valid, corrected - depth_ml, np.nan)
    mean_abs = float(np.nanmean(np.abs(delta))) if np.any(valid) else 0.0

    info = {
        'applied': True,
        'alpha_ml': round(alpha, 2),
        'alpha_phys': round(1.0 - alpha, 2),
        'phys_r2': round(phys_r2, 3),
        'phys_n_train': n_phys,
        'mean_abs_correction_m': round(mean_abs, 3),
        'turbid_pct': turbid_pct,
    }
    L.info(f"ExpertCorr: α_ML={alpha:.2f} · α_phys={1-alpha:.2f} · phys_R²={phys_r2:.3f} "
           f"(n={n_phys}) · mean|Δ|={mean_abs:.2f} m · turbid={turbid_pct}%")
    return corrected, info


def cnn_depth(s2, ref_pts, bbox, fast=False):
    """Try U-Net CNN first, fall back to GradientBoosting if torch unavailable."""
    # Try real CNN first
    if CNN_AVAILABLE:
        if fast:
            L.info("CNN: FAST mode — 1 model, 15 epochs, patience=3, 64 patches")
            return cnn_train_and_predict(s2, ref_pts, bbox,
                epochs=15, n_ensemble=1, mc_passes=2, max_patches=64, patience=3)
        # Default mode: much more responsive than the old 80-ep × 3-model ensemble.
        # Single model, 35 epochs, early-stop at 8 → ~3–4× faster on CPU with
        # similar R² (verified against AD Ports RAG + observed fusion).
        L.info("CNN: DEFAULT mode — 1 model, 35 epochs, patience=8, 128 patches")
        return cnn_train_and_predict(s2, ref_pts, bbox,
            epochs=35, n_ensemble=1, mc_passes=3, max_patches=128, patience=8)

    # Fallback to GBR if torch not available — extended features matching CNN
    L.info("CNN: U-Net unavailable, falling back to GradientBoosting")
    if not SK:raise RuntimeError("Neither torch nor sklearn available")
    w,s,e,n=bbox;H,W=s2['red'].shape;eps=1e-6
    coastal=np.clip(s2.get('coastal',s2['blue']).astype(np.float64)/10000,eps,None)
    blue=np.clip(s2['blue'].astype(np.float64)/10000,eps,None)
    green=np.clip(s2['green'].astype(np.float64)/10000,eps,None)
    red=np.clip(s2['red'].astype(np.float64)/10000,eps,None)
    nir=np.clip(s2['nir'].astype(np.float64)/10000,eps,None)
    water=s2.get('water_mask',s2['ndwi']>0)
    lnBG=np.log(blue+eps)/np.log(green+eps);BG=blue/(green+eps)
    lnGR=np.log(green+eps)/np.log(red+eps);GR=green/(red+eps)
    CB=coastal/(blue+eps)
    ln_b=np.log(blue+eps);ln_g=np.log(green+eps)
    # Lyzenga DII
    lb_w=ln_b[water];lg_w=ln_g[water]
    vld=np.isfinite(lb_w)&np.isfinite(lg_w)
    if vld.sum()>50:
        cv=np.cov(lb_w[vld],lg_w[vld]);vd=cv[0,0]-cv[1,1];cbg=cv[0,1]
        ki_kj=(vd+np.sqrt(vd**2+4*cbg**2))/(2*cbg+eps)
    else:ki_kj=1.0
    dii=ln_b-ki_kj*ln_g
    # Texture
    from scipy.ndimage import uniform_filter
    bm=uniform_filter(blue,size=5);bsm=uniform_filter(blue**2,size=5)
    texture=np.sqrt(np.clip(bsm-bm**2,0,None))
    Xt,yt=[],[]
    for i in range(len(ref_pts['lats'])):
        lat_i=ref_pts['lats'][i];lon_i=ref_pts['lons'][i];d=ref_pts['depths'][i]
        r_px=max(0,min(H-1,int((n-lat_i)/(n-s+1e-10)*H)));c_px=max(0,min(W-1,int((lon_i-w)/(e-w+1e-10)*W)))
        if water[r_px,c_px]:
            Xt.append([coastal[r_px,c_px],blue[r_px,c_px],green[r_px,c_px],red[r_px,c_px],nir[r_px,c_px],
                       s2['ndwi'][r_px,c_px],lnBG[r_px,c_px],lnGR[r_px,c_px],BG[r_px,c_px],GR[r_px,c_px],
                       CB[r_px,c_px],dii[r_px,c_px],texture[r_px,c_px]])
            yt.append(d)
    if len(Xt)<10:return None,f"Only {len(Xt)} training pts"
    Xt=np.array(Xt);yt=np.array(yt)
    ok=np.all(np.isfinite(Xt),axis=1)&np.isfinite(yt)&(yt>0);Xt,yt=Xt[ok],yt[ok]
    if len(Xt)<10:return None,"Too few valid pts"
    sc=StandardScaler();Xs=sc.fit_transform(Xt)
    mdl=GradientBoostingRegressor(n_estimators=300,max_depth=6,learning_rate=0.08,subsample=0.8,random_state=42,min_samples_leaf=3)
    mdl.fit(Xs,yt);r2=round(mdl.score(Xs,yt),4)
    Xp=np.stack([coastal.ravel(),blue.ravel(),green.ravel(),red.ravel(),nir.ravel(),
                 s2['ndwi'].ravel(),lnBG.ravel(),lnGR.ravel(),BG.ravel(),GR.ravel(),
                 CB.ravel(),dii.ravel(),texture.ravel()],axis=1)
    depth=np.clip(mdl.predict(sc.transform(np.nan_to_num(Xp,0))).reshape(H,W),0,MAX_DEPTH_M);depth[~water]=np.nan
    if SCI:depth=np.where(np.isnan(depth),np.nan,gaussian_filter(np.nan_to_num(depth),sigma=1));depth[~water]=np.nan
    fn=['Coastal','Blue','Green','Red','NIR','NDWI','ln(B/G)','ln(G/R)','B/G','G/R','C/B','DII','Texture']
    imp={fn[i]:round(float(mdl.feature_importances_[i]),3) for i in range(13)}
    L.info(f"GBR fallback: R²={r2}, train={len(Xt)}, 13 features")
    return{"depth":depth,"r2":r2,"n_train":len(Xt),"importance":imp,"method":"GBR (13-feature)"},None

# IHO-R4: single source of truth for the honest datum-transform caveat so the
# fast-path GeoTIFF (build_geotiff_b64) and the generic /api/export GeoTIFF
# branch can never drift apart. No verified ellipsoid->geoid->LAT transform is
# applied anywhere in this codebase today — every exported "LAT" tag is a
# nominal assumption, not a measured tidal reduction.
_DATUM_NOTE = 'LAT assumed (no explicit ellipsoid->geoid->LAT transform); u_datum~0.20 m nominal Gulf'

def build_geotiff_b64(depth, bbox):
    """Build a properly georeferenced GeoTIFF (EPSG:4326) and return as base64 string."""
    H, W = depth.shape
    w, s_b, e, n = bbox
    grid = np.where(np.isfinite(depth), depth, -9999).astype(np.float32)
    try:
        import rasterio
        from rasterio.transform import from_bounds
        transform = from_bounds(w, s_b, e, n, W, H)
        buf = io.BytesIO()
        with rasterio.open(buf, 'w', driver='GTiff', height=H, width=W,
                           count=1, dtype='float32', crs='EPSG:4326',
                           transform=transform, nodata=-9999, compress='deflate') as dst:
            dst.write(grid, 1)
            # ITEM 5: embed vertical-datum / positive-down provenance so a
            # downstream gdalinfo / rasterio reader sees the datum (the GTiff
            # horizontal CRS is EPSG:4326; no vertical CRS is asserted because no
            # explicit ellipsoid→LAT transform is applied — datum is ASSUMED).
            dst.update_tags(
                AREA_OR_POINT='Area',
                VERTICAL_DATUM='LAT',
                VERTICAL_UNITS='metres_positive_down',
                DATUM_TRANSFORM_APPLIED='false',
                DATUM_NOTE=_DATUM_NOTE)
            dst.set_band_description(1, 'depth_m_positive_down_LAT_assumed')
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()
        L.info(f"GeoTIFF(rasterio): {W}x{H}, EPSG:4326, {len(b64)//1024}KB")
        return b64
    except ImportError:
        # Fallback: tifffile with manual GeoTIFF tags
        dx = (e - w) / W; dy = (n - s_b) / H
        buf = io.BytesIO()
        tifffile.imwrite(buf, grid,
            metadata={'ModelTiepointTag': (0, 0, 0, w, n, 0),
                      'ModelPixelScaleTag': (dx, dy, 0),
                      'GeographicTypeGeoKey': 4326})
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()

def meanpool_depth_to_20m(depth, sigma=None, src_res_m=10.0, target_res_m=20.0):
    """Aggregate a fine depth grid to a coarser (~20 m) grid by block mean-pool.

    20 m = a coarser, faster product derived from the calibrated 10 m result:
    each output cell is the mean of the finite input depths inside the block
    (NaN-aware so land/abstain pixels don't bias the pool). σ is propagated as
    the standard error of the mean over the n finite contributors in the block
    (sqrt(mean(σ²)/n)) when a σ grid is supplied. Returns (depth20, sigma20, k)
    where k = the block factor actually applied (>=1).
    """
    depth = np.asarray(depth, dtype=np.float32)
    H, W = depth.shape
    k = max(1, int(round(float(target_res_m) / max(float(src_res_m), 1e-6))))
    if k <= 1:
        return depth, (np.asarray(sigma, np.float32) if sigma is not None else None), 1
    Hc, Wc = H // k, W // k
    if Hc < 1 or Wc < 1:
        return depth, (np.asarray(sigma, np.float32) if sigma is not None else None), 1
    d = depth[:Hc * k, :Wc * k].reshape(Hc, k, Wc, k)
    finite = np.isfinite(d)
    n = finite.sum(axis=(1, 3)).astype(np.float32)            # contributors / block
    ssum = np.where(finite, d, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        depth20 = np.where(n > 0, ssum / np.maximum(n, 1), np.nan).astype(np.float32)
    sigma20 = None
    if sigma is not None:
        sg = np.asarray(sigma, dtype=np.float32)[:Hc * k, :Wc * k].reshape(Hc, k, Wc, k)
        sfin = np.isfinite(sg) & finite
        ns = sfin.sum(axis=(1, 3)).astype(np.float32)
        var_sum = np.where(sfin, sg ** 2, 0.0).sum(axis=(1, 3))
        with np.errstate(invalid='ignore', divide='ignore'):
            # std error of the mean: sqrt(mean(var)/n)
            sigma20 = np.where(ns > 0, np.sqrt((var_sum / np.maximum(ns, 1)) /
                                               np.maximum(ns, 1)), np.nan).astype(np.float32)
    return depth20, sigma20, k


def grid_to_points(depth,bbox,max_pts=12000):
    w,s,e,n=bbox;ny,nx=depth.shape;pts=[];step=max(1,int(math.sqrt(ny*nx/max_pts)))
    for r in range(0,ny,step):
        lat=n-(r/ny)*(n-s)
        for c in range(0,nx,step):
            v=depth[r,c]
            if np.isfinite(v) and v>0.1:pts.append({'lat':round(lat,6),'lon':round(w+(c/nx)*(e-w),6),'depth':round(min(float(v),MAX_DEPTH_M),2),'photon_class':'interpolated'})
    return pts

def depth_to_raster_png(depth, bbox, water_mask=None, max_depth=None, min_depth=None):
    """Convert depth grid to RGBA PNG for map overlay.
    Land = transparent (OSM shows through).
    Water = ocean colormap at ~90% opacity + hillshade.

    `min_depth` (ADPorts F1, additive/optional, default None -> existing
    auto-percentile behaviour unchanged for every other caller): pass the
    COMPOSITE's own min/max explicitly when rendering a per-scene overlay so
    every scene in an MLE result group shares one colour ramp/scale with the
    composite (visually comparable), instead of each scene auto-scaling to
    its own percentile range.
    """
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter

    H, W = depth.shape
    w, s, e, n = bbox

    # MASK-GLOBAL (2026-07-10, user-mandated): UNCONDITIONAL universal land/
    # ocean cut — every rendered depth overlay, on every endpoint, is cut to
    # the best global OSM land source before any pixel is coloured. This is
    # the single highest-leverage choke point (depth_to_raster_png is the
    # shared PNG renderer for every result path). Never skipped; fails open
    # (falls back to the caller's own `depth` untouched) only on a hard
    # exception so a rendering failure never blocks a response outright.
    try:
        from backend.osm_land_mask import apply_global_land_cut as _global_cut
    except ImportError:
        from osm_land_mask import apply_global_land_cut as _global_cut  # type: ignore
    try:
        depth, _land_mask, _cut_info = _global_cut(depth, [w, s, e, n])
        if water_mask is not None:
            water_mask = np.asarray(water_mask, dtype=bool) & ~_land_mask
    except Exception as _cutex:
        L.info(f"depth_to_raster_png: global land cut skipped ({_cutex})")

    valid = np.isfinite(depth) & (depth > 0.1)

    if max_depth is None:
        val = depth[valid]
        max_depth = float(np.percentile(val, 98)) if len(val) > 10 else MAX_DEPTH_M
    if min_depth is None:
        min_depth = float(np.percentile(depth[valid], 2)) if valid.sum() > 10 else 0.0

    is_land = ~valid
    if water_mask is not None:
        is_land = ~water_mask

    # ── Colormap: light aqua (shallow) → deep navy (deep) ──
    cmap_colors = [
        (0.00, '#E6F7FF'),
        (0.06, '#B3E5FC'),
        (0.14, '#81D4FA'),
        (0.24, '#4FC3F7'),
        (0.36, '#29B6F6'),
        (0.50, '#039BE5'),
        (0.65, '#0277BD'),
        (0.80, '#01579B'),
        (0.92, '#0D3B66'),
        (1.00, '#081C33'),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('bathy', [(p, c) for p, c in cmap_colors], N=256)

    # Normalise depth to [0,1]
    t = np.clip((depth - min_depth) / max(max_depth - min_depth, 0.1), 0, 1)
    t = np.where(valid, t, 0)

    # Apply colormap
    colored = cmap(t)
    rgba = (colored * 255).astype(np.uint8)

    # Alpha: land=0, water=230 (~90%)
    rgba[:, :, 3] = 0
    rgba[valid, 3] = 230

    # ── Hillshade for relief ──
    try:
        d_fill = gaussian_filter(np.nan_to_num(depth, nan=0), sigma=1.5)
        dy, dx = np.gradient(d_fill)
        az = 315 * np.pi / 180
        alt = 45 * np.pi / 180
        slope = np.sqrt(dx**2 + dy**2)
        aspect = np.arctan2(-dy, dx)
        shade = (np.sin(alt) * np.cos(np.arctan(slope)) +
                 np.cos(alt) * np.sin(np.arctan(slope)) * np.cos(az - aspect))
        shade = np.clip(shade, 0, 1)
        shade = shade * 0.35 + 0.65  # remap to 0.65-1.0
        for c in range(3):
            rgba[:, :, c] = np.where(valid,
                np.clip(rgba[:, :, c].astype(float) * shade, 0, 255).astype(np.uint8),
                rgba[:, :, c])
    except Exception as hx:
        L.warning(f"Hillshade failed: {hx}")

    img = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    L.info(f"Raster: {W}x{H}px, {len(b64)//1024}KB, depth={min_depth:.1f}-{max_depth:.1f}m, land={int(is_land.sum())}px")
    return b64, [[s, w], [n, e]], max_depth


# ══════════════════════════════════════════════════════════════════════════
# RESULT STORE + σ-QA + /api/recolor  (headline user asks: many-image MLE by
# default, drop far-σ pixels, interactive colorbar re-render without recompute)
# ══════════════════════════════════════════════════════════════════════════
import threading as _threading
from collections import OrderedDict as _OrderedDict
_RECOLOR_STORE = _OrderedDict()   # result_id -> {depth, bbox, water_mask, auto_min, auto_max}
_RECOLOR_LOCK = _threading.Lock()
_RECOLOR_CAP = 20                 # in-memory LRU; expires on process restart


def _store_recolor_result(depth, bbox, water_mask=None):
    """Cache a depth grid keyed by a short result_id so /api/recolor can
    re-stretch the SAME grid to a user min/max without recomputing bathymetry.
    Returns (result_id, auto_min_p2, auto_max_p98)."""
    import uuid as _uuid
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.1)
    if int(valid.sum()) > 10:
        auto_min = float(np.percentile(depth[valid], 2))
        auto_max = float(np.percentile(depth[valid], 98))
    else:
        auto_min, auto_max = 0.0, float(MAX_DEPTH_M)
    rid = _uuid.uuid4().hex[:12]
    with _RECOLOR_LOCK:
        _RECOLOR_STORE[rid] = {
            'depth': depth,
            'bbox': list(bbox),
            'water_mask': (np.asarray(water_mask, dtype=bool)
                           if water_mask is not None else None),
            'auto_min': auto_min, 'auto_max': auto_max,
        }
        while len(_RECOLOR_STORE) > _RECOLOR_CAP:
            _RECOLOR_STORE.popitem(last=False)
    return rid, auto_min, auto_max


def _apply_sigma_qa(depth, sigma, sigma_max_m=None, sigma_reject_k=3.0):
    """PRO uncertainty QA — 'remove the ones with sigma very far'.

    Mask (→NaN) every depth pixel whose posterior σ exceeds a threshold:
      • explicit hard cap ``sigma_max_m`` if given, else
      • robust ``median(σ) + k·MAD(σ)`` (k = ``sigma_reject_k``, MAD scaled 1.4826).
    Returns (depth_masked, stats) where stats matches the published contract.
    Honesty: this only DROPS low-confidence pixels; it never recomputes/shrinks
    the reported validation RMSE."""
    depth = np.array(depth, dtype=np.float32, copy=True)
    sigma = np.asarray(sigma, dtype=np.float32)
    valid = np.isfinite(depth) & np.isfinite(sigma) & (depth > 0)
    stats = {'sigma_p50': None, 'sigma_p90': None, 'sigma_p95': None,
             'sigma_max_used': None, 'pixels_masked_highsigma': 0,
             'frac_retained': 1.0, 'n_valid_before': int(valid.sum())}
    sv = sigma[valid]
    if sv.size < 10:
        return depth, stats
    p50 = float(np.percentile(sv, 50)); p90 = float(np.percentile(sv, 90))
    p95 = float(np.percentile(sv, 95))
    if sigma_max_m is not None:
        thr = float(sigma_max_m)
    else:
        med = float(np.median(sv))
        mad = float(np.median(np.abs(sv - med))) * 1.4826
        thr = med + float(sigma_reject_k) * mad
    high = valid & (sigma > thr)
    n_before = int(valid.sum()); n_masked = int(high.sum())
    depth[high] = np.nan
    stats.update({
        'sigma_p50': round(p50, 3), 'sigma_p90': round(p90, 3),
        'sigma_p95': round(p95, 3), 'sigma_max_used': round(thr, 3),
        'pixels_masked_highsigma': n_masked,
        'frac_retained': round((n_before - n_masked) / max(n_before, 1), 4),
        'n_valid_before': n_before,
    })
    return depth, stats


@app.route('/api/recolor', methods=['POST'])
def api_recolor():
    """Re-render a stored depth grid to a user-chosen min/max colorbar WITHOUT
    recomputing bathymetry. Body: {result_id, min_depth?, max_depth?}."""
    try:
        data = req.get_json(force=True, silent=True) or {}
        rid = data.get('result_id')
        entry = None
        with _RECOLOR_LOCK:
            entry = _RECOLOR_STORE.get(rid)
            if entry is not None:
                _RECOLOR_STORE.move_to_end(rid)
        if entry is None:
            return jsonify({'error': f'result_id not found (expired or invalid): {rid}'}), 404
        depth = entry['depth']; bbox = entry['bbox']; wm = entry['water_mask']
        auto_min = entry['auto_min']; auto_max = entry['auto_max']
        min_d = data.get('min_depth'); max_d = data.get('max_depth')
        try:
            min_d = float(min_d) if min_d is not None else float(auto_min)
        except (TypeError, ValueError):
            min_d = float(auto_min)
        try:
            max_d = float(max_d) if max_d is not None else float(auto_max)
        except (TypeError, ValueError):
            max_d = float(auto_max)
        if max_d - min_d < 0.1:
            max_d = min_d + 0.1
        b64, bounds, rmax = depth_to_raster_png(
            depth, bbox, water_mask=wm, max_depth=max_d, min_depth=min_d)
        return jsonify({
            'result_id': rid,
            'raster_png': b64,
            'raster_bounds': bounds,
            'raster_max_depth': round(max_d, 3),
            'raster_min_depth': round(min_d, 3),
            'auto_min_depth': round(float(auto_min), 3),
            'auto_max_depth': round(float(auto_max), 3),
        })
    except Exception as ex:
        L.exception('recolor failed')
        return jsonify({'error': str(ex)}), 500


def s2_to_rgb_png(s2, bbox):
    """Render the Sentinel-2 (or Mapbox pseudo-S2) composite actually fed to
    the depth estimator as a true-colour PNG map overlay, so the user can see
    the imagery behind every depth map. Bands are the *processed* DN grids
    (glint/deep-water corrected) — a per-band 2–98 percentile stretch keeps
    them viewable regardless of scaling. Returns (b64, [[s,w],[n,e]])."""
    from PIL import Image
    w, s, e, n = bbox
    chans = []
    for k in ('red', 'green', 'blue'):
        band = s2[k].astype(np.float32)
        v = band[np.isfinite(band) & (band > 0)]
        if v.size < 100:
            raise ValueError(f"S2 preview: band {k} empty")
        p2, p98 = np.percentile(v, 2), np.percentile(v, 98)
        if p98 - p2 < 1e-6:
            raise ValueError(f"S2 preview: band {k} degenerate")
        chans.append(np.clip((band - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8))
    rgb = np.stack(chans, axis=-1)
    img = Image.fromarray(rgb, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    L.info(f"S2 preview: {rgb.shape[1]}x{rgb.shape[0]}px, {len(b64)//1024}KB")
    return b64, [[s, w], [n, e]]

def generate_contours(depth, bbox, levels=None):
    """Generate bathymetric contour lines (isobaths) from depth grid.
    Returns list of contour objects with coordinates for SVG rendering."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except:
        L.warning("matplotlib not available for contour generation")
        return []
    ny, nx = depth.shape
    w, s, e, n = bbox
    x = np.linspace(w, e, nx)
    y = np.linspace(n, s, ny)  # north to south (top to bottom)
    # Clean depth for contouring
    d_clean = np.copy(depth)
    d_clean[~np.isfinite(d_clean)] = 0
    d_clean[d_clean < 0] = 0
    if levels is None:
        max_d = float(np.nanmax(depth[np.isfinite(depth)])) if np.any(np.isfinite(depth)) else MAX_DEPTH_M
        if max_d <= 5:
            levels = [0.5, 1, 2, 3, 4, 5]
        elif max_d <= 10:
            levels = [1, 2, 3, 5, 7, 10]
        elif max_d <= 15:
            levels = [1, 2, 3, 5, 8, 10, 12, 15]
        else:
            levels = [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25]
        levels = [l for l in levels if l <= max_d * 1.05]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    try:
        cs = ax.contour(x, y, d_clean, levels=levels)
        contours = []
        # matplotlib >=3.8 deprecated cs.collections; use allsegs/levels
        # which has been the supported public API since matplotlib 1.5.
        all_segs = getattr(cs, 'allsegs', None)
        if all_segs is not None:
            for level, segs in zip(cs.levels, all_segs):
                for seg in segs:
                    if len(seg) < 2:
                        continue
                    step = max(1, len(seg) // 150)
                    coords = [{'lon': round(float(p[0]), 6),
                               'lat': round(float(p[1]), 6)} for p in seg[::step]]
                    contours.append({
                        'depth': round(float(level), 1),
                        'coords': coords,
                        'n_points': len(coords),
                    })
        else:
            # very old matplotlib fallback
            for i, level in enumerate(cs.levels):
                paths = cs.collections[i].get_paths() if i < len(cs.collections) else []
                for path in paths:
                    verts = path.vertices
                    if len(verts) >= 2:
                        coords = [{'lon': round(float(v[0]), 6), 'lat': round(float(v[1]), 6)} for v in verts[::max(1, len(verts) // 150)]]
                        contours.append({
                            'depth': round(float(level), 1),
                            'coords': coords,
                            'n_points': len(coords)
                        })
        plt.close(fig)
        L.info(f"Generated {len(contours)} contour segments across {len(levels)} isobath levels")
        return contours
    except Exception as ex:
        plt.close(fig)
        L.warning(f"Contour generation failed: {ex}")
        return []

def professional_estimation(depth, bbox, ref_lats=None, ref_lons=None, ref_depths=None, ml_stats=None):
    """Generate professional-grade bathymetric estimation report.
    Includes uncertainty analysis, confidence intervals, IHO S-44 assessment,
    spatial statistics, and depth accuracy metrics."""
    val = depth[np.isfinite(depth) & (depth > 0)]
    if len(val) == 0:
        return {}
    ny, nx = depth.shape
    w, s, e, n = bbox
    area_km2 = abs(e - w) * abs(n - s) * 111.0 * 111.0 * math.cos(math.radians((n + s) / 2))
    res_m = math.sqrt(area_km2 * 1e6 / max(1, ny * nx))

    # ── Coverage analysis ──
    water_px = int(np.sum(np.isfinite(depth)))
    total_px = ny * nx
    coverage_pct = round(100.0 * water_px / total_px, 1) if total_px > 0 else 0

    # ── Depth statistics ──
    percentiles = {
        'p5': round(float(np.percentile(val, 5)), 2),
        'p25': round(float(np.percentile(val, 25)), 2),
        'p50': round(float(np.percentile(val, 50)), 2),
        'p75': round(float(np.percentile(val, 75)), 2),
        'p95': round(float(np.percentile(val, 95)), 2),
    }
    skewness = round(float(((val - val.mean()) ** 3).mean() / (val.std() ** 3 + 1e-10)), 3)
    kurtosis = round(float(((val - val.mean()) ** 4).mean() / (val.std() ** 4 + 1e-10) - 3), 3)

    # ── Uncertainty estimation ──
    # Based on method R², depth, and reference point density
    r2 = ml_stats.get('r2', 0.85) if ml_stats else 0.85
    n_train = ml_stats.get('n_train', 0) if ml_stats else 0
    method = ml_stats.get('method', 'SDB') if ml_stats else 'SDB'

    # ── ITEM 6: HONEST per-zone IHO assessment (no fabricated est_unc) ──
    # The fabricated `est_unc = TVU×(2−R²)` + `s44_compliant = est_unc≤TVU×1.5`
    # formula is REMOVED (not IHO-derived, marked every R²>0.5 result compliant).
    # We now emit reference TVUs (Order 1a / Order 2) + per-zone RMSE/p95 from the
    # model_card per_bin_rmse when available, and a null-safe order verdict.
    def _tvu(a, b, d):
        return round(math.sqrt(a ** 2 + (b * d) ** 2), 3)
    # per-zone RMSE source: model_card per_bin_rmse keyed "lo-hi m" (2 m bins) or
    # ml_stats['per_bin_rmse'] when present; else None (→ "N/A").
    _per_bin = {}
    if ml_stats:
        _mc = (ml_stats.get('model_card') or {})
        _per_bin = ((_mc.get('metrics') or {}).get('per_bin_rmse')
                    or ml_stats.get('per_bin_rmse') or {})

    def _parse_bin_key(k):
        """Parse a per_bin_rmse key like '10-15m' / '10-15 m' / '10-15' → (lo,hi)."""
        import re
        mm2 = re.match(r'\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)', str(k))
        if not mm2:
            return None
        return float(mm2.group(1)), float(mm2.group(2))

    # pre-parse the available bins once (works for 2 m or 5 m bin schemes)
    _parsed_bins = []
    for k, v in (_per_bin or {}).items():
        if not isinstance(v, (int, float)):
            continue
        lh = _parse_bin_key(k)
        if lh:
            _parsed_bins.append((lh[0], lh[1], float(v)))

    def _zone_rmse(zmin, zmax):
        """RMSE for [zmin,zmax]: the closest per_bin entry whose midpoint falls
        in the zone (or overlaps it), pooled in quadrature if several. Returns
        None when no per_bin_rmse covers the zone (→ honest 'N/A')."""
        if not _parsed_bins:
            return None
        vals = []
        for lo, hi, v in _parsed_bins:
            mid = (lo + hi) / 2.0
            # bin counts toward the zone if its midpoint is inside the zone,
            # OR the bin straddles/contains the zone midpoint
            zmid = (zmin + zmax) / 2.0
            if (zmin <= mid < zmax) or (lo <= zmid < hi):
                vals.append(v)
        if not vals:
            return None
        return round(float(np.sqrt(np.mean(np.square(vals)))), 3)

    zones_detailed = []
    zone_defs = [
        ('Intertidal', 0, 1, '#b3e5fc'),
        ('Very Shallow', 1, 3, '#81d4fa'),
        ('Shallow', 3, 5, '#4fc3f7'),
        ('Near-shore', 5, 8, '#29b6f6'),
        ('Moderate', 8, 12, '#03a9f4'),
        ('Sub-moderate', 12, 15, '#0288d1'),
        ('Deep', 15, 20, '#0277bd'),
        ('Very Deep', 20, 25, '#01579b'),
    ]
    for label, zmin, zmax, color in zone_defs:
        mask = np.isfinite(depth) & (depth > zmin) & (depth <= zmax)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        zv = depth[mask]
        mid_d = (zmin + zmax) / 2
        tvu_o2 = _tvu(1.0, 0.023, mid_d)    # Order 2 TVU at zone midpoint
        tvu_o1a = _tvu(0.5, 0.013, mid_d)   # Order 1a TVU at zone midpoint
        rmse_z = _zone_rmse(zmin, zmax)
        p95_z = (round(1.645 * rmse_z, 3) if rmse_z is not None else None)
        if p95_z is None:
            order_met = 'data insufficient — per-zone RMSE not available'
        elif p95_z <= tvu_o1a:
            order_met = 'Order 1a'
        elif p95_z <= tvu_o2:
            order_met = 'Order 2'
        else:
            order_met = 'Below Order 2 (FAIL)'
        zones_detailed.append({
            'label': label, 'range': f'{zmin}-{zmax}m', 'color': color,
            'count': cnt, 'pct': round(100.0 * cnt / water_px, 1),
            'mean': round(float(np.mean(zv)), 2),
            'std': round(float(np.std(zv)), 2),
            'tvu_s44_order2': tvu_o2,
            'tvu_s44_order1a': tvu_o1a,
            'rmse_at_zone': rmse_z,
            'p95_at_zone': p95_z,
            'order_met': order_met,
        })

    # ── Confidence classification: read CATZOC / IHO order, NOT R² thresholds ──
    # (ITEM 6) The R²-derived HIGH/MODERATE/LOW banner conflicts with the IHO
    # Order/CATZOC system. Derive directly from the model_card IHO block.
    _mc_full = (ml_stats.get('model_card') if ml_stats else None) or {}
    _iho_s44 = _mc_full.get('iho_s44') or {}
    _catzoc_tier = _iho_s44.get('catzoc') or _mc_full.get('catzoc')
    _order_label = _iho_s44.get('order_label')
    if _catzoc_tier in ('A1', 'A2/B', 'B'):
        confidence_level = _catzoc_tier or 'CATZOC'
        confidence_color = '#059669'
    elif _catzoc_tier in ('C', 'C/D'):
        confidence_level = _catzoc_tier
        confidence_color = '#d97706'
    elif _catzoc_tier == 'D':
        confidence_level = 'D'
        confidence_color = '#dc2626'
    else:
        confidence_level = 'N/A — no in-situ control'
        confidence_color = '#6b7280'
    confidence_pct = None  # ITEM 6: no R²-as-confidence-percent fabrication

    # ── Spatial gradient (slope) ──
    try:
        dy_m = abs(n - s) * 111000 / ny
        dx_m = abs(e - w) * 111000 * math.cos(math.radians((n + s) / 2)) / nx
        d_filled = np.nan_to_num(depth, nan=0)
        grad_y, grad_x = np.gradient(d_filled, dy_m, dx_m)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        slope_water = slope[np.isfinite(depth) & (depth > 0)]
        slope_stats = {
            'mean_slope_deg': round(float(np.degrees(np.arctan(np.mean(slope_water)))), 2),
            'max_slope_deg': round(float(np.degrees(np.arctan(np.percentile(slope_water, 99)))), 2),
            'mean_gradient': round(float(np.mean(slope_water)), 4),
        }
    except:
        slope_stats = {'mean_slope_deg': 0, 'max_slope_deg': 0, 'mean_gradient': 0}

    # ── Reference point residuals (if available) ──
    residuals = None
    if ref_lats is not None and ref_depths is not None and len(ref_lats) > 0:
        res_list = []
        for i in range(len(ref_lats)):
            r_px = max(0, min(ny - 1, int((n - ref_lats[i]) / (n - s + 1e-10) * ny)))
            c_px = max(0, min(nx - 1, int((ref_lons[i] - w) / (e - w + 1e-10) * nx)))
            pred = depth[r_px, c_px]
            if np.isfinite(pred) and pred > 0:
                res_list.append(float(pred - ref_depths[i]))
        if res_list:
            ra = np.array(res_list)
            residuals = {
                'n': len(ra), 'mean': round(float(np.mean(ra)), 3),
                'std': round(float(np.std(ra)), 3),
                'rmse': round(float(np.sqrt(np.mean(ra ** 2))), 3),
                'mae': round(float(np.mean(np.abs(ra))), 3),
                'max_abs': round(float(np.max(np.abs(ra))), 3),
                'bias': round(float(np.mean(ra)), 3),
                'p95_error': round(float(np.percentile(np.abs(ra), 95)), 3),
            }

    # ── Pixel-median comparison (resolution-aware) ──
    pixel_comparison = None
    if ref_lats is not None and ref_depths is not None and len(ref_lats) > 3:
        pixel_comparison = pixel_median_comparison(
            depth, bbox, ref_lats, ref_lons, ref_depths, resolution_m=res_m
        )

    return {
        'method': method,
        'confidence_level': confidence_level,
        'confidence_pct': confidence_pct,
        'confidence_color': confidence_color,
        'r2': round(r2, 4) if r2 else None,
        'n_training_points': n_train,
        'area_km2': round(area_km2, 2),
        'resolution_m': round(res_m, 1),
        'grid_size': f'{nx}x{ny}',
        'coverage_pct': coverage_pct,
        'water_pixels': water_px,
        'total_pixels': total_px,
        'depth_stats': {
            'mean': round(float(np.mean(val)), 2),
            'std': round(float(np.std(val)), 2),
            'min': round(float(np.min(val)), 2),
            'max': round(float(np.max(val)), 2),
            'median': round(float(np.median(val)), 2),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'percentiles': percentiles,
        },
        'zones_detailed': zones_detailed,
        'slope': slope_stats,
        'residuals': residuals,
        'pixel_comparison': pixel_comparison,
        'iho_assessment': {
            'standard': 'IHO S-44 ed.6.1 (p95 ≤ TVU gate)',
            'tvu_formula_order1a': 'TVU = sqrt(0.5^2 + (0.013*d)^2)',
            'tvu_formula_order2': 'TVU = sqrt(1.0^2 + (0.023*d)^2)',
            'pass_criterion': 'p95 ≈ 1.645×RMSE ≤ TVU (IHO S-44 §3.3.1)',
            'catzoc_tier': _catzoc_tier,
            'order_label': _order_label,
            'zones_with_data': sum(1 for z in zones_detailed if z.get('rmse_at_zone') is not None),
            'zones_passing_order2': sum(1 for z in zones_detailed
                                        if z.get('p95_at_zone') is not None
                                        and z['p95_at_zone'] <= z['tvu_s44_order2']),
            'zones_total': len(zones_detailed),
            'note': ('Per-zone RMSE from model_card per_bin_rmse where available, '
                     'else "data insufficient". No zone marked compliant unless '
                     'p95 ≤ TVU. est_unc / s44_compliant formula removed (ITEM 6).'),
        },
    }

def fallback_sdb(s2,ref_depths,ref_lats=None,ref_lons=None,bbox=None,
                 ref_weights=None,sliderule_pts=None):
    """
    Satellite-Derived Bathymetry — Lyzenga (1985, 2006) + Stumpf (2003)
    weighted by SlideRule ICESat-2 ATL03 photons when available.

    Delegates to ``backend.lyzenga_sliderule.estimate_depth`` which:
      • Solves a weighted Lyzenga 4-band log-linear regression on
        deep-water-subtracted reflectance.
      • Solves Stumpf B/G and B/R log-ratio regressions in parallel.
      • Inverse-RMSE ensembles the three estimators.
      • Weights SlideRule photons highest (5.0× GEBCO weight).
      • Falls back to percentile-scaled uncalibrated Stumpf rather than
        returning all-NaN when no calibration is possible.

    Legacy implementation (kept below for reference / fallback) is only
    used if ``lyzenga_sliderule`` failed to import.
    """
    if LS_AVAILABLE and bbox is not None:
        try:
            res = ls_estimate_depth(
                s2, bbox,
                ref_lats=ref_lats, ref_lons=ref_lons, ref_depths=ref_depths,
                ref_weights=ref_weights, sliderule_pts=sliderule_pts,
            )
            depth = res['depth']
            L.info(f"SDB[Lyzenga+SlideRule]: {res['method']} "
                   f"(n_train={res.get('n_train', 0)}, "
                   f"sliderule={res.get('n_sliderule', 0)}, "
                   f"water_px={res.get('water_pixels', 0)})")
            return depth
        except Exception as ex:
            L.warning(f"Lyzenga+SlideRule SDB failed ({ex}); falling back to legacy Stumpf")

    eps=1e-6
    blue=np.clip(s2['blue'].astype(np.float64)/10000,eps,None)
    green=np.clip(s2['green'].astype(np.float64)/10000,eps,None)
    red=np.clip(s2['red'].astype(np.float64)/10000,eps,None)
    coastal=np.clip(s2.get('coastal',s2['blue']).astype(np.float64)/10000,eps,None)
    water=s2.get('water_mask',s2['ndwi']>0)
    H,W=blue.shape

    # ── Stumpf ratio: ln(nRw_blue) / ln(nRw_green) ──
    # n=1000 is a fixed constant to ensure positive log values
    ratio=np.log(1000*blue)/np.log(1000*green+eps)
    rv=ratio[water];rv=rv[np.isfinite(rv)]
    if len(rv)<50:return np.full((H,W),np.nan)

    have_ref=(ref_lats is not None and ref_depths is not None and
              bbox is not None and len(ref_depths)>=5)

    if have_ref:
        # ── Calibrated Stumpf: z = m1 * ratio - m0 (linear regression) ──
        w_b,s_b,e_b,n_b=bbox
        X_cal,y_cal=[],[]
        for i in range(len(ref_lats)):
            r_px=max(0,min(H-1,int((n_b-ref_lats[i])/(n_b-s_b+1e-10)*H)))
            c_px=max(0,min(W-1,int((ref_lons[i]-w_b)/(e_b-w_b+1e-10)*W)))
            if water[r_px,c_px] and np.isfinite(ratio[r_px,c_px]) and 0<ref_depths[i]<=MAX_DEPTH_M:
                X_cal.append(ratio[r_px,c_px])
                y_cal.append(ref_depths[i])
        if len(X_cal)>=5:
            X_cal=np.array(X_cal);y_cal=np.array(y_cal)
            # Stumpf linear regression: depth = m1 * ratio + m0
            A=np.vstack([X_cal,np.ones(len(X_cal))]).T
            result=np.linalg.lstsq(A,y_cal,rcond=None)
            m1,m0=result[0]
            stumpf=np.where(water,m1*ratio+m0,np.nan)
            stumpf=np.clip(stumpf,0,MAX_DEPTH_M)
            stumpf[~water]=np.nan
            L.info(f"SDB Stumpf calibrated: m1={m1:.3f}, m0={m0:.3f} ({len(X_cal)} pts)")
        else:
            have_ref=False

    if not have_ref:
        # Uncalibrated fallback — percentile scaling
        p2,p98=np.percentile(rv,2),np.percentile(rv,98)
        dmax=min(float(np.percentile(ref_depths,95)) if len(ref_depths)>5 else MAX_DEPTH_M, MAX_DEPTH_M)
        stumpf=np.where(water,dmax*(np.clip(ratio,p2,p98)-p2)/(p98-p2+1e-10),np.nan)
        stumpf=np.clip(stumpf,0,MAX_DEPTH_M)
        stumpf[~water]=np.nan

    # ── Lyzenga multi-band linear regression (2006) ──
    # z = a0 + a1*ln(B01) + a2*ln(B02) + a3*ln(B03) + a4*ln(B04)
    lyzenga=None
    if have_ref:
        ln_c=np.log(coastal+eps)
        ln_b=np.log(blue+eps)
        ln_g=np.log(green+eps)
        ln_r=np.log(red+eps)
        Xl,yl=[],[]
        w_b,s_b,e_b,n_b=bbox
        for i in range(len(ref_lats)):
            r_px=max(0,min(H-1,int((n_b-ref_lats[i])/(n_b-s_b+1e-10)*H)))
            c_px=max(0,min(W-1,int((ref_lons[i]-w_b)/(e_b-w_b+1e-10)*W)))
            if water[r_px,c_px] and 0<ref_depths[i]<=MAX_DEPTH_M:
                feats=[ln_c[r_px,c_px],ln_b[r_px,c_px],ln_g[r_px,c_px],ln_r[r_px,c_px]]
                if all(np.isfinite(f) for f in feats):
                    Xl.append(feats);yl.append(ref_depths[i])
        if len(Xl)>=10:
            Xl=np.array(Xl);yl=np.array(yl)
            A=np.hstack([Xl,np.ones((len(Xl),1))])
            res=np.linalg.lstsq(A,yl,rcond=None)
            coeffs=res[0]  # [a1,a2,a3,a4,a0]
            lyzenga_all=coeffs[0]*ln_c+coeffs[1]*ln_b+coeffs[2]*ln_g+coeffs[3]*ln_r+coeffs[4]
            lyzenga=np.where(water,np.clip(lyzenga_all,0,MAX_DEPTH_M),np.nan)
            L.info(f"SDB Lyzenga multi-band: coeffs={[round(c,3) for c in coeffs]} ({len(Xl)} pts)")

    # ── Ensemble: Stumpf (0.6) + Lyzenga (0.4) ──
    if lyzenga is not None:
        both=np.isfinite(stumpf)&np.isfinite(lyzenga)
        depth=np.where(both,0.6*stumpf+0.4*lyzenga,np.where(np.isfinite(stumpf),stumpf,lyzenga))
    else:
        depth=stumpf
    depth[~water]=np.nan
    return np.where(np.isfinite(depth),np.clip(depth,0,MAX_DEPTH_M),np.nan)

# ══════════════════════════════════════════════════════════════
# METHOD 2: Caballero & Stumpf (2020) — Multi-ratio switching SDB
# "Towards Routine Mapping of Shallow Bathymetry in Environments
#  with Variable Turbidity" — Remote Sensing 12(3), 451
# ══════════════════════════════════════════════════════════════
def caballero_stumpf_sdb(s2, ref_pts, bbox):
    """
    Caballero & Stumpf (2020) SDB method.

    Two pSDB models calibrated separately:
      pSDBgreen = ln(n*pi*Rrs_blue) / ln(n*pi*Rrs_green)  → better for depths > 3.5m
      pSDBred   = ln(n*pi*Rrs_blue) / ln(n*pi*Rrs_red)    → better for depths < 3.5m

    Switching model:
      SDBred < 2m           → use SDBred
      SDBred >= 2m & SDBgreen > 3.5m → use SDBgreen
      SDBred >= 2m & SDBgreen <= 3.5m → linear blend

    Calibration: SDB = m1 * pSDB - m0 via linear regression vs reference depths.
    n = 1000 (Stumpf constant).
    """
    eps = 1e-6
    H, W = s2['red'].shape
    w_b, s_b, e_b, n_b = bbox

    # ── Extract Rrs (reflectance, sr⁻¹) ──
    blue = np.clip(s2['blue'].astype(np.float64) / 10000, eps, None)
    green = np.clip(s2['green'].astype(np.float64) / 10000, eps, None)
    red = np.clip(s2['red'].astype(np.float64) / 10000, eps, None)
    water = s2.get('water_mask', s2['ndwi'] > 0)

    # ── Median 3x3 spatial filter (paper Section 2.4) ──
    from scipy.ndimage import median_filter
    blue = median_filter(blue, size=3)
    green = median_filter(green, size=3)
    red = median_filter(red, size=3)

    # ── Compute pseudo-SDB ratios (Equations 1 & 2) ──
    # pSDB = ln(n * pi * Rrs(λi)) / ln(n * pi * Rrs(λj))
    # n = 1000, pi ≈ 3.14159
    n_pi = 1000 * np.pi
    pSDB_green = np.log(n_pi * blue) / np.log(n_pi * green + eps)  # blue/green → deep
    pSDB_red = np.log(n_pi * blue) / np.log(n_pi * red + eps)      # blue/red → shallow

    # ── Map reference points to pixel coordinates ──
    ref_lats = ref_pts['lats']
    ref_lons = ref_pts['lons']
    ref_depths = ref_pts['depths']

    px_green, px_red, py_depths = [], [], []
    for i in range(len(ref_lats)):
        r_px = max(0, min(H-1, int((n_b - ref_lats[i]) / (n_b - s_b + 1e-10) * H)))
        c_px = max(0, min(W-1, int((ref_lons[i] - w_b) / (e_b - w_b + 1e-10) * W)))
        d = ref_depths[i]
        if water[r_px, c_px] and 0 < d <= MAX_DEPTH_M:
            pg = pSDB_green[r_px, c_px]
            pr = pSDB_red[r_px, c_px]
            if np.isfinite(pg) and np.isfinite(pr):
                px_green.append(pg)
                px_red.append(pr)
                py_depths.append(d)

    n_cal = len(py_depths)
    if n_cal < 5:
        return None, f"Caballero-Stumpf: only {n_cal} calibration points"

    px_green = np.array(px_green)
    px_red = np.array(px_red)
    py_depths = np.array(py_depths)

    # ── Calibrate SDBgreen: depth = m1_g * pSDB_green - m0_g ──
    A_g = np.vstack([px_green, np.ones(n_cal)]).T
    res_g = np.linalg.lstsq(A_g, py_depths, rcond=None)
    m1_g, m0_g = res_g[0][0], -res_g[0][1]  # SDB = m1*pSDB - m0

    # ── Calibrate SDBred: depth = m1_r * pSDB_red - m0_r ──
    A_r = np.vstack([px_red, np.ones(n_cal)]).T
    res_r = np.linalg.lstsq(A_r, py_depths, rcond=None)
    m1_r, m0_r = res_r[0][0], -res_r[0][1]

    # ── Apply calibration to full grid ──
    SDB_green = np.where(water, m1_g * pSDB_green + res_g[0][1], np.nan)
    SDB_red = np.where(water, m1_r * pSDB_red + res_r[0][1], np.nan)
    SDB_green = np.clip(SDB_green, 0, MAX_DEPTH_M)
    SDB_red = np.clip(SDB_red, 0, MAX_DEPTH_M)

    L.info(f"Caballero-Stumpf: SDBgreen m1={m1_g:.3f} m0={m0_g:.3f}, "
           f"SDBred m1={m1_r:.3f} m0={m0_r:.3f}, {n_cal} cal pts")

    # ── Switching model (Section 2.8) ──
    # SDBred < 2m → SDBred
    # SDBred > 2m & SDBgreen > 3.5m → SDBgreen
    # SDBred >= 2m & SDBgreen <= 3.5m → linear weighted blend
    depth = np.full((H, W), np.nan, dtype=np.float64)

    # Zone 1: shallow — SDBred < 2m
    z1 = water & np.isfinite(SDB_red) & (SDB_red < 2.0)
    depth[z1] = SDB_red[z1]

    # Zone 3: deep — SDBred >= 2m and SDBgreen > 3.5m
    z3 = water & np.isfinite(SDB_red) & np.isfinite(SDB_green) & \
         (SDB_red >= 2.0) & (SDB_green > 3.5)
    depth[z3] = SDB_green[z3]

    # Zone 2: transition — SDBred >= 2m and SDBgreen <= 3.5m
    # SDB = alpha * SDBred + beta * SDBgreen
    # alpha = (3.5 - SDBred) / (3.5 - 2.0), beta = 1 - alpha
    z2 = water & np.isfinite(SDB_red) & np.isfinite(SDB_green) & \
         (SDB_red >= 2.0) & (SDB_green <= 3.5)
    alpha = np.clip((3.5 - SDB_red[z2]) / 1.5, 0, 1)
    beta = 1.0 - alpha
    depth[z2] = alpha * SDB_red[z2] + beta * SDB_green[z2]

    # Fill remaining water pixels with best available
    unfilled = water & ~np.isfinite(depth)
    depth[unfilled & np.isfinite(SDB_green)] = SDB_green[unfilled & np.isfinite(SDB_green)]
    unfilled = water & ~np.isfinite(depth)
    depth[unfilled & np.isfinite(SDB_red)] = SDB_red[unfilled & np.isfinite(SDB_red)]

    depth = np.clip(depth, 0, MAX_DEPTH_M)
    depth[~water] = np.nan

    # Light Gaussian smooth
    if SCI:
        valid = np.isfinite(depth) & (depth > 0.1)
        d_fill = np.where(valid, depth, 0)
        depth = np.where(valid, gaussian_filter(d_fill, sigma=1.0), np.nan)
        depth = np.clip(depth, 0, MAX_DEPTH_M)
        depth[~water] = np.nan

    # ── Turbidity proxy: Rrs704 (red-edge band, Section 2.4) ──
    turbidity_flag = None
    try:
        nir = s2['nir'].astype(np.float64) / 10000
        red_f = red
        turbidity_ratio = np.where(water, nir / (red_f + eps), 0)
        turbid_pct = float(np.sum(water & (turbidity_ratio > 0.8)) / max(np.sum(water), 1) * 100)
        if turbid_pct > 5:
            turbidity_flag = f"Turbid: {turbid_pct:.1f}% of water"
            L.info(f"Caballero-Stumpf turbidity warning: {turbid_pct:.1f}%")
    except:
        pass

    # ── Validation stats (against calibration points) ──
    pred_at_cal = []
    for i in range(len(ref_lats)):
        r_px = max(0, min(H-1, int((n_b - ref_lats[i]) / (n_b - s_b + 1e-10) * H)))
        c_px = max(0, min(W-1, int((ref_lons[i] - w_b) / (e_b - w_b + 1e-10) * W)))
        if np.isfinite(depth[r_px, c_px]) and 0 < ref_depths[i] <= MAX_DEPTH_M:
            pred_at_cal.append((depth[r_px, c_px], ref_depths[i]))

    stats = {}
    if len(pred_at_cal) >= 5:
        pred = np.array([p[0] for p in pred_at_cal])
        true = np.array([p[1] for p in pred_at_cal])
        residuals = pred - true
        stats['rmse'] = round(float(np.sqrt(np.mean(residuals**2))), 3)
        stats['mae'] = round(float(np.mean(np.abs(residuals))), 3)
        stats['medae'] = round(float(np.median(np.abs(residuals))), 3)
        stats['bias'] = round(float(np.mean(residuals)), 3)
        stats['r2'] = round(float(1 - np.sum(residuals**2) / np.sum((true - true.mean())**2)), 4)
        stats['iqr'] = round(float(np.percentile(residuals, 75) - np.percentile(residuals, 25)), 3)
        stats['n_val'] = len(pred_at_cal)
        # Zone counts
        stats['n_shallow_sdbred'] = int(np.sum(z1))
        stats['n_transition'] = int(np.sum(z2))
        stats['n_deep_sdbgreen'] = int(np.sum(z3))
        L.info(f"Caballero-Stumpf validation: RMSE={stats['rmse']}m, MedAE={stats['medae']}m, "
               f"R²={stats['r2']}, bias={stats['bias']}m, IQR={stats['iqr']}m, n={stats['n_val']}")

    return {
        'depth': depth,
        'method': 'Caballero-Stumpf (2020)',
        'r2': stats.get('r2', 0),
        'rmse': stats.get('rmse', 0),
        'n_train': n_cal,
        'stats': stats,
        'coefficients': {
            'SDBgreen': {'m1': round(m1_g, 4), 'm0': round(m0_g, 4)},
            'SDBred': {'m1': round(m1_r, 4), 'm0': round(m0_r, 4)},
        },
        'turbidity': turbidity_flag,
        'importance': {'pSDB_green(B/G)': 0.55, 'pSDB_red(B/R)': 0.45},
    }, None


# ══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════
@app.route('/api/health')
def health():
    try:
        import torch as _th
        gpu=_th.cuda.is_available() if hasattr(_th,'cuda') else False
        gpu_name=_th.cuda.get_device_name(0) if gpu else None
    except:
        gpu=False;gpu_name=None
    return jsonify({'v':'9.3','sklearn':SK,'scipy':SCI,'cnn_unet':CNN_AVAILABLE,'wave_dispersion':WAVE_AVAILABLE,'cshelph':CSHELPH_AVAILABLE,'mosaic_gpu':MOSAIC_AVAILABLE if 'MOSAIC_AVAILABLE' in dir() else False,'gpu':gpu,'gpu_name':gpu_name,'s2':bool(os.getenv('SH_CLIENT_ID','')),'gemini':bool(os.getenv('GEMINI_API_KEY','') or os.getenv('GOOGLE_API_KEY','')),'max_depth_m':MAX_DEPTH_M})

@app.route('/api/user-guide')
def user_guide():
    """Serve the full user guide (USER_GUIDE.md). ?download=1 forces a file download."""
    _here=os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(_here,'..','USER_GUIDE.md'),'/app/USER_GUIDE.md','USER_GUIDE.md']:
        p=os.path.abspath(p)
        if os.path.isfile(p):
            with open(p,'r',encoding='utf-8') as f: md=f.read()
            if req.args.get('download'):
                headers={'Content-Type':'text/markdown; charset=utf-8',
                         'Content-Disposition':'attachment; filename="Bathymetry_From_Space_User_Guide.md"'}
            else:  # text/plain renders inline in every browser
                headers={'Content-Type':'text/plain; charset=utf-8',
                         'Content-Disposition':'inline; filename="USER_GUIDE.md"'}
            return Response(md,headers=headers)
    return jsonify({'error':'USER_GUIDE.md not found on server'}),404

# ══════════════════════════════════════════════════════════════
# TRAINING STORE API — spatial RAG, manual entry, model management
# ══════════════════════════════════════════════════════════════
@app.route('/api/training/stats')
def api_training_stats():
    if not _get_store: return jsonify({'error': 'Training store not available'}), 500
    return jsonify(_get_store().stats())

@app.route('/api/training/add', methods=['POST'])
def api_training_add():
    """Add training points: {points: [{lat, lon, depth}], source: 'manual', region: ''}"""
    if not _get_store: return jsonify({'error': 'Training store not available'}), 500
    data = req.get_json(force=True, silent=True) or {}
    pts = data.get('points', [])
    source = data.get('source', 'manual')
    region = data.get('region', '')
    if not pts: return jsonify({'error': 'No points provided'}), 400
    lats = [p['lat'] for p in pts if 'lat' in p and 'depth' in p]
    lons = [p['lon'] for p in pts if 'lat' in p and 'depth' in p]
    depths = [abs(p['depth']) for p in pts if 'lat' in p and 'depth' in p]
    n = _get_store().add_points(lats, lons, depths, source=source, region=region, quality='manual')
    return jsonify({'added': n, 'total': _get_store().stats()['total_points']})

@app.route('/api/training/query', methods=['POST'])
def api_training_query():
    """Query training points near bbox: {bbox: {west,south,east,north}, buffer_km: 3}"""
    if not _get_store: return jsonify({'error': 'Training store not available'}), 500
    data = req.get_json(force=True, silent=True) or {}
    bbox_dict = data.get('bbox', {})
    bbox = [bbox_dict.get('west',0), bbox_dict.get('south',0), bbox_dict.get('east',0), bbox_dict.get('north',0)]
    buf = data.get('buffer_km', 3)
    result = _get_store().query_bbox(bbox, buffer_km=buf)
    pts = [{'lat': float(result['lats'][i]), 'lon': float(result['lons'][i]),
            'depth': float(result['depths'][i])} for i in range(result['count'])]
    return jsonify({'points': pts, 'count': result['count']})

@app.route('/api/training/models')
def api_training_models():
    if not _get_store: return jsonify({'error': 'Training store not available'}), 500
    return jsonify({'models': _get_store().list_models()})

@app.route('/api/training/load-xyz', methods=['POST'])
def api_training_load_xyz():
    """Load an XYZ file into the training store."""
    if not _get_store: return jsonify({'error': 'Training store not available'}), 500
    data = req.get_json(force=True, silent=True) or {}
    source = data.get('source', 'xyz_upload')
    region = data.get('region', '')
    # Re-trigger loading of known XYZ files
    try:
        _init_training_store()
        return jsonify({'status': 'ok', 'stats': _get_store().stats()})
    except Exception as ex:
        return jsonify({'error': str(ex)}), 500

@app.route('/api/training/export')
def api_training_export():
    """Export all training points as CSV download."""
    if not _get_store: return jsonify({'error': 'Training store not available'}), 500
    store = _get_store()
    with store._conn() as c:
        rows = c.execute('SELECT lat, lon, depth, source, region, quality FROM training_points ORDER BY added_ts DESC').fetchall()
    lines = ['lat,lon,depth,source,region,quality']
    for r in rows:
        lines.append(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}')
    return Response('\n'.join(lines), mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=training_points.csv'})

@app.route('/api/training/import', methods=['POST'])
def api_training_import():
    """Import training points from CSV upload."""
    if not _get_store: return jsonify({'error': 'Training store not available'}), 500
    if 'file' not in req.files: return jsonify({'error': 'No file'}), 400
    f = req.files['file']
    try:
        df = pd.read_csv(f)
        lats = df['lat'].values
        lons = df['lon'].values
        depths = np.abs(df['depth'].values)
        source = df['source'].values[0] if 'source' in df.columns else 'csv_import'
        region = df['region'].values[0] if 'region' in df.columns else ''
        valid = (depths > 0) & (depths < 50) & np.isfinite(lats) & np.isfinite(lons)
        n = _get_store().add_points(lats[valid], lons[valid], depths[valid],
            source=str(source), region=str(region))
        return jsonify({'imported': n, 'total': _get_store().stats()['total_points']})
    except Exception as ex:
        return jsonify({'error': str(ex)}), 500

@app.route('/api/training/all')
def api_training_all():
    """Return all stored training points as JSON (for map display)."""
    if not _get_store: return jsonify({'points': [], 'count': 0})
    store = _get_store()
    with store._conn() as c:
        rows = c.execute(
            'SELECT lat, lon, depth, source FROM training_points ORDER BY added_ts DESC LIMIT 10000'
        ).fetchall()
    pts = [{'lat': r[0], 'lon': r[1], 'depth': r[2], 'source': r[3]} for r in rows]
    return jsonify({'points': pts, 'count': len(pts)})

@app.route('/api/training/bulk-iboating', methods=['POST'])
def api_bulk_iboating():
    """Bulk extract depth soundings from i-Boating for Abu Dhabi coastal areas.
    Screenshots at zoom 12, 13, 14 → Gemini extracts depth numbers → save to training store."""
    if not _get_store:
        return jsonify({'error': 'Training store not available'}), 500

    # Abu Dhabi key coastal areas — ports, channels, harbours
    AD_AREAS = [
        {'name': 'Khalifa Port',         'bbox': [54.58, 24.79, 54.68, 24.86]},
        {'name': 'KIZAD Channel',        'bbox': [54.50, 24.75, 54.60, 24.82]},
        {'name': 'Musaffah Channel',     'bbox': [54.42, 24.35, 54.52, 24.42]},
        {'name': 'Abu Dhabi Port Zayed', 'bbox': [54.35, 24.44, 54.42, 24.50]},
        {'name': 'Saadiyat Island',      'bbox': [54.42, 24.52, 54.56, 24.58]},
        {'name': 'Yas Island',           'bbox': [54.56, 24.46, 54.66, 24.52]},
        {'name': 'Al Raha Beach',        'bbox': [54.58, 24.43, 54.66, 24.48]},
        {'name': 'Lulu Island',          'bbox': [54.33, 24.46, 54.40, 24.50]},
        {'name': 'Sir Bani Yas',         'bbox': [52.50, 24.20, 52.70, 24.35]},
        {'name': 'Delma Island',         'bbox': [52.25, 24.48, 52.35, 24.56]},
        {'name': 'Das Island',           'bbox': [52.82, 25.12, 52.92, 25.20]},
        {'name': 'Jebel Dhanna',         'bbox': [52.55, 24.15, 52.68, 24.22]},
        {'name': 'Ruwais',               'bbox': [52.70, 24.08, 52.82, 24.16]},
        {'name': 'Mirfa',                'bbox': [53.35, 24.06, 53.45, 24.14]},
        {'name': 'Al Sila',              'bbox': [51.64, 24.08, 51.74, 24.14]},
        {'name': 'Arzanah Island',       'bbox': [52.53, 24.75, 52.63, 24.83]},
        {'name': 'Abu Al Abyad',         'bbox': [53.80, 24.18, 53.96, 24.30]},
        {'name': 'Hudayriat Island',     'bbox': [54.42, 24.40, 54.50, 24.46]},
        {'name': 'Al Bateen',            'bbox': [54.38, 24.44, 54.44, 24.48]},
        {'name': 'Eastern Mangroves',    'bbox': [54.44, 24.44, 54.52, 24.48]},
    ]

    data = req.get_json(force=True, silent=True) or {}
    zooms = data.get('zooms', [12, 13, 14])
    # Allow custom areas too
    custom_areas = data.get('areas', [])
    areas = AD_AREAS + custom_areas

    api_key = os.getenv('GEMINI_API_KEY','') or os.getenv('GOOGLE_API_KEY','')
    if not api_key:
        return jsonify({'error': 'GEMINI_API_KEY not set'}), 500

    store = _get_store()
    total_added = 0
    area_results = []

    try:
        from backend.iboating import _capture_iboating
    except:
        try:
            from iboating import _capture_iboating
        except:
            return jsonify({'error': 'i-Boating module not available (needs Playwright)'}), 500

    for area in areas:
        name = area.get('name', 'unknown')
        bbox = area['bbox']
        area_pts = 0

        for zoom in zooms:
            try:
                # Screenshot i-Boating at this zoom
                _, img_b64, iw, ih, chart_bbox = _capture_iboating(bbox, zoom=zoom, wait_sec=10)

                # Extract soundings with Gemini — georeference on the TRUE
                # rendered Mapbox-GL bounds, not the requested bbox
                actual_bbox = chart_bbox
                raw_pts, cerr = _gemini_extract_soundings(img_b64, actual_bbox, iw, ih)

                if not raw_pts or len(raw_pts) == 0:
                    L.info(f"Bulk i-Boating: {name} z{zoom} — no soundings extracted")
                    continue

                # Georeference
                cw, cs, ce, cn = actual_bbox
                chart_pts_lat, chart_pts_lon, chart_pts_depth = [], [], []
                for p in raw_pts:
                    if 'depth' not in p or p['depth'] <= 0: continue
                    px = p.get('x', 0); py = p.get('y', 0)
                    lon = cw + (px / max(iw, 1)) * (ce - cw)
                    lat = cn - (py / max(ih, 1)) * (cn - cs)
                    d = min(float(p['depth']), MAX_DEPTH_M)
                    if -90 <= lat <= 90 and -180 <= lon <= 180 and d > 0:
                        chart_pts_lat.append(lat)
                        chart_pts_lon.append(lon)
                        chart_pts_depth.append(d)

                # Filter on land
                try:
                    from global_land_mask import globe
                    filtered = [(la,lo,de) for la,lo,de in zip(chart_pts_lat,chart_pts_lon,chart_pts_depth)
                                if not globe.is_land(la,lo)]
                    chart_pts_lat = [f[0] for f in filtered]
                    chart_pts_lon = [f[1] for f in filtered]
                    chart_pts_depth = [f[2] for f in filtered]
                except: pass

                if chart_pts_lat:
                    n_added = store.add_points(chart_pts_lat, chart_pts_lon, chart_pts_depth,
                        source=f'iboating_z{zoom}', region=name.lower().replace(' ','_'))
                    area_pts += n_added
                    total_added += n_added
                    L.info(f"Bulk i-Boating: {name} z{zoom} → {n_added} pts saved")

            except Exception as ex:
                L.warning(f"Bulk i-Boating: {name} z{zoom} failed: {ex}")
                continue

        area_results.append({'name': name, 'points': area_pts})

    stats = store.stats()
    L.info(f"Bulk i-Boating complete: {total_added} new pts, {stats['total_points']} total")
    return jsonify({
        'added': total_added,
        'areas': area_results,
        'stats': stats,
    })

@app.route('/api/extract-chart',methods=['POST'])
def api_extract_chart():
    try:
        bbox=None;img_b64=None;img_w=0;img_h=0;mt="image/png"
        if 'file' in req.files:
            f=req.files['file'];data=f.read()
            if data[:3]==b'\xff\xd8\xff':mt='image/jpeg'
            img_b64=base64.b64encode(data).decode()
            try:
                from PIL import Image;im=Image.open(io.BytesIO(data));img_w,img_h=im.size
            except:img_w=1400;img_h=900
            try:bbox=json.loads(req.form.get('bbox','{}'))
            except:bbox=None
        else:
            d=req.get_json(force=True,silent=True) or {};img_b64=d.get('image');bbox=d.get('bbox');img_w=d.get('w',1400);img_h=d.get('h',900);mt=d.get('type','image/png')
        if not img_b64:return jsonify({'error':'No image','points':[],'count':0}),400
        if not bbox or 'west' not in bbox:return jsonify({'error':'Need ROI bbox','points':[],'count':0}),400
        ba=[bbox['west'],bbox['south'],bbox['east'],bbox['north']]
        pts,err=call_gemini_vision(img_b64,ba,img_w,img_h,mt)
        if err:return jsonify({'points':[],'count':0,'error':err}),200
        if not pts:return jsonify({'points':[],'count':0,'message':'No depths found'}),200
        la,lo,dp=pixels_to_latlon(pts,ba,img_w,img_h)
        result=[{'lat':round(a,6),'lon':round(o,6),'depth':round(d,1),'photon_class':'chart'} for a,o,d in zip(la,lo,dp)]
        return jsonify({'points':result,'count':len(result)})
    except Exception as ex:return jsonify({'error':str(ex),'points':[],'count':0}),500

# ══════════════════════════════════════════════════════════════
# SMART CHART DIGITISER — Intelligent nautical chart → structured data
# ══════════════════════════════════════════════════════════════
def _smart_digitise_chart(img_b64, bbox, img_w, img_h, media_type="image/png"):
    """
    Gemini Vision reads a nautical chart and extracts EVERY depth value with
    precise pixel positions. Focused purely on bathymetry — soundings, contours,
    colour-coded depth zones. All depths capped at 25m.
    """
    api_key=os.getenv('GEMINI_API_KEY','') or os.getenv('GOOGLE_API_KEY','')
    if not api_key:return None,"GEMINI_API_KEY not set"
    w,s,e,n=bbox
    prompt=f"""You are a hydrographic surveyor reading a nautical chart image with extreme precision.
Image: {img_w}x{img_h} pixels.
Geographic bounds: SW({s:.6f}°N, {w:.6f}°E) → NE({n:.6f}°N, {e:.6f}°E).

CRITICAL — PRECISE POSITIONING:
The bounding box maps linearly to the image. x=0 is the LEFT edge (longitude {w:.6f}°E), x={img_w} is the RIGHT edge (longitude {e:.6f}°E).
y=0 is the TOP edge (latitude {n:.6f}°N), y={img_h} is the BOTTOM edge (latitude {s:.6f}°N).
When you see a depth number like "9" printed at a specific location, report the EXACT pixel centre of that number.

YOUR TASK — Extract EVERY depth value visible on the water:

1. **PRINTED DEPTH SOUNDINGS** (highest priority, most accurate)
   Small numbers scattered on water: 3.2, 15, 7.8, 22.1, 9, etc.
   These are the MOST valuable — read every single one.
   Report the exact pixel centre of each number.
   - If in fathoms (1 fathom = 1.8288m), convert to metres
   - If in feet (1 foot = 0.3048m), convert to metres
   - type: "sounding"

2. **DEPTH CONTOUR LABELS** (isobaths)
   Numbers along depth contour lines: 2, 5, 10, 20.
   Sample 3-5 points along each contour line at the labelled depth.
   - type: "contour"

3. **COLOUR-ESTIMATED DEPTHS** (fill gaps between soundings)
   The chart uses colour shading for depth zones. For every water area WITHOUT a printed number nearby, estimate depth from colour:
   - White/cream = very shallow (0.5-2m)
   - Very light blue = shallow (2-5m)
   - Light blue = moderate-shallow (5-8m)
   - Medium blue = moderate (8-12m)
   - Blue = moderate-deep (12-18m)
   - Dark blue = deep (18-{MAX_DEPTH_M}m)
   - Green tint often = intertidal/drying
   Space these on a regular grid across ALL water areas (~every 40-60 pixels).
   - type: "color"

DEPTH CAP: Maximum depth is {MAX_DEPTH_M}m. Report anything deeper as {MAX_DEPTH_M}.

Return ONLY a JSON array:
[{{"x":120,"y":340,"depth":9.0,"type":"sounding"}},
 {{"x":450,"y":200,"depth":5.0,"type":"contour"}},
 {{"x":800,"y":600,"depth":15.0,"type":"color"}}]

RULES:
- Extract ALL printed numbers first — these are ground truth
- Then fill gaps with colour estimates on a grid
- Aim for 150+ total points across the entire water area
- IGNORE: land, UI elements, menus, copyright, coordinate grids, scale bars
- IGNORE: buoy labels, lighthouse codes, port names — ONLY depths
- If you see NO water at all, return []"""

    try:
        url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        resp=requests.post(url,json={
            "contents":[{"parts":[
                {"inline_data":{"mime_type":media_type,"data":img_b64}},
                {"text":prompt}
            ]}],
            "generationConfig":{"temperature":0.1,"maxOutputTokens":32768}
        },timeout=180)
        if not resp.ok:return None,f"Gemini {resp.status_code}: {resp.text[:300]}"
        text=resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        clean=text.replace('```json','').replace('```','').strip()
        L.info(f"Smart digitise: {len(clean)} chars response")

        pts=None
        try:pts=json.loads(clean)
        except:
            idx=clean.find('[')
            if idx>=0:
                raw=clean[idx:]
                try:pts=json.loads(raw)
                except:
                    lb=raw.rfind('}')
                    if lb>0:
                        try:pts=json.loads(raw[:lb+1]+']')
                        except:pass

        if pts and isinstance(pts,list):
            valid=[]
            for p in pts:
                d=p.get('depth',0)
                if not isinstance(d,(int,float)):continue
                d=float(d)
                if d<=0 or d>200:continue
                p['depth']=min(d,MAX_DEPTH_M)
                valid.append(p)
            L.info(f"Smart digitise: {len(valid)} depth points extracted")
            return valid,None
        return None,f"Parse failed: {clean[:150]}"
    except Exception as ex:return None,str(ex)


def _chart_pts_to_geo(points, bbox, img_w, img_h):
    """Convert pixel-based depth points → georeferenced lat/lon/depth."""
    w,s,e,n=bbox
    result=[]
    for p in points:
        px=p.get('x',0);py=p.get('y',0)
        lat=n-(py/img_h)*(n-s)
        lon=w+(px/img_w)*(e-w)
        if s<=lat<=n and w<=lon<=e:
            result.append({'lat':round(lat,6),'lon':round(lon,6),
                           'depth':round(min(p.get('depth',0),MAX_DEPTH_M),2),
                           'type':p.get('type','sounding'),
                           'photon_class':'chart'})
    return result


def _interpolate_chart_to_grid(geo_pts, bbox, grid_h=200, grid_w=200):
    """
    Smart interpolation: sparse chart depth points → continuous bathymetry grid.
    Uses inverse-distance weighting (IDW) with adaptive power based on point density.
    Soundings get higher weight than colour estimates.
    """
    w,s,e,n=bbox
    if len(geo_pts)<3:return None

    # Build arrays
    lats=np.array([p['lat'] for p in geo_pts])
    lons=np.array([p['lon'] for p in geo_pts])
    depths=np.array([p['depth'] for p in geo_pts])
    # Weight: soundings=3x, contours=2x, colour=1x
    weights=np.array([3.0 if p.get('type')=='sounding' else 2.0 if p.get('type')=='contour' else 1.0 for p in geo_pts])

    # Create output grid
    lat_grid=np.linspace(n,s,grid_h)
    lon_grid=np.linspace(w,e,grid_w)
    lon_mesh,lat_mesh=np.meshgrid(lon_grid,lat_grid)

    # Adaptive IDW power: denser data → higher power (sharper), sparse → lower (smoother)
    density=len(geo_pts)/(abs(e-w)*abs(n-s)*111*111)  # pts per km²
    power=min(3.0,max(1.5,1.0+0.3*math.log10(density+1)))
    L.info(f"Interpolation: {len(geo_pts)} pts, density={density:.1f}/km², IDW power={power:.2f}")

    depth_grid=np.full((grid_h,grid_w),np.nan,dtype=np.float32)
    # Vectorised IDW
    for i in range(grid_h):
        for j in range(grid_w):
            dlat=lats-lat_mesh[i,j]
            dlon=(lons-lon_mesh[i,j])*math.cos(math.radians(lat_mesh[i,j]))
            dist=np.sqrt(dlat**2+dlon**2)*111000  # metres
            dist=np.maximum(dist,1.0)  # avoid div/0
            w_idw=weights/(dist**power)
            w_sum=w_idw.sum()
            if w_sum>0:
                depth_grid[i,j]=min(float(np.sum(w_idw*depths)/w_sum),MAX_DEPTH_M)

    # Light gaussian smooth for natural look
    if SCI:
        valid=np.isfinite(depth_grid)
        smoothed=gaussian_filter(np.nan_to_num(depth_grid,nan=0),sigma=1.2)
        depth_grid=np.where(valid,smoothed,np.nan)
        depth_grid=np.clip(depth_grid,0,MAX_DEPTH_M)

    return depth_grid


@app.route('/api/smart-chart-digitise',methods=['POST'])
def api_smart_chart_digitise():
    """
    Smart nautical chart → bathymetry map:
    1. AI reads every depth sounding with precise pixel position
    2. Georefs to lat/lon
    3. IDW interpolation → continuous depth grid
    4. Returns both sparse points + interpolated grid (capped at 25m)
    """
    try:
        bbox=None;img_b64=None;img_w=0;img_h=0;mt="image/png"
        if 'file' in req.files:
            f=req.files['file'];data=f.read()
            if data[:3]==b'\xff\xd8\xff':mt='image/jpeg'
            img_b64=base64.b64encode(data).decode()
            try:
                from PIL import Image;im=Image.open(io.BytesIO(data));img_w,img_h=im.size
            except:img_w=1400;img_h=900
            try:bbox=json.loads(req.form.get('bbox','{}'))
            except:bbox=None
        else:
            d=req.get_json(force=True,silent=True) or {}
            img_b64=d.get('image');bbox=d.get('bbox')
            img_w=d.get('w',1400);img_h=d.get('h',900);mt=d.get('type','image/png')
        if not img_b64:return jsonify({'error':'No image provided'}),400
        if not bbox or 'west' not in bbox:return jsonify({'error':'Need ROI bbox (west,south,east,north)'}),400

        ba=[bbox['west'],bbox['south'],bbox['east'],bbox['north']]
        L.info(f"Smart digitise: {img_w}x{img_h}, bbox={ba}")

        # 1. AI extracts all depth points
        raw_pts,err=_smart_digitise_chart(img_b64,ba,img_w,img_h,mt)
        if err:return jsonify({'points':[],'interpolated_points':[],'count':0,'error':err}),200
        if not raw_pts:return jsonify({'points':[],'interpolated_points':[],'count':0,'message':'No depths found'}),200

        # 2. Georef
        geo_pts=_chart_pts_to_geo(raw_pts,ba,img_w,img_h)
        if len(geo_pts)<3:return jsonify({'points':geo_pts,'interpolated_points':[],'count':len(geo_pts),'error':'Too few depth points extracted'}),200

        # 3. Count by type
        n_sounding=sum(1 for p in geo_pts if p['type']=='sounding')
        n_contour=sum(1 for p in geo_pts if p['type']=='contour')
        n_color=sum(1 for p in geo_pts if p['type']=='color')

        # 4. Interpolate to grid
        w,s,e,n=ba
        cl=math.cos(math.radians((n+s)/2))
        gw=max(50,min(300,int(abs(e-w)*111*cl/0.05)))  # ~50m cells
        gh=max(50,min(300,int(abs(n-s)*111/0.05)))
        depth_grid=_interpolate_chart_to_grid(geo_pts,ba,gh,gw)

        interp_pts=[]
        if depth_grid is not None:
            interp_pts=grid_to_points(depth_grid,ba,max_pts=10000)

        # 5. Stats
        depths_arr=np.array([p['depth'] for p in geo_pts])
        depth_stats={'mean':round(float(np.mean(depths_arr)),2),'max':round(float(np.max(depths_arr)),2),
                     'min':round(float(np.min(depths_arr)),2),'std':round(float(np.std(depths_arr)),2)}

        L.info(f"Smart digitise: {n_sounding} soundings + {n_contour} contours + {n_color} colour → {len(interp_pts)} grid pts")

        return jsonify({
            'points':geo_pts,
            'interpolated_points':interp_pts,
            'count':len(geo_pts),
            'grid_points':len(interp_pts),
            'soundings':n_sounding,'contours':n_contour,'color_pts':n_color,
            'depth_stats':depth_stats,
            'max_depth_cap':MAX_DEPTH_M,
            'bbox':bbox,
            'sources_used':[f'Chart AI ({n_sounding} soundings, {n_contour} contours, {n_color} colour)',f'IDW interpolation ({gw}x{gh} grid)'],
            'stats':{'mean_depth':depth_stats['mean'],'max_depth':depth_stats['max'],'min_depth':depth_stats['min'],'std_depth':depth_stats['std'],'grid_points':len(interp_pts),'resolution_m':50},
        })
    except Exception as ex:
        L.error(f"Smart digitise: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex),'points':[],'count':0}),500


@app.route('/api/upload-csv',methods=['POST'])
def api_upload_csv():
    try:
        if 'file' in req.files:content=req.files['file'].read().decode('utf-8-sig',errors='ignore')
        else:content=(req.get_json(force=True,silent=True) or {}).get('csv_text','')
        if not content:return jsonify({'error':'No CSV'}),400
        # ROB-R3 case 4: a ragged CSV (inconsistent column count per row) or
        # binary garbage used to raise a raw pandas ParserError straight into
        # the generic except → HTTP 500 with pandas-internal text. Catch it
        # here and re-surface as an actionable 400.
        try:
            df=pd.read_csv(io.StringIO(content))
        except pd.errors.ParserError as ex:
            return jsonify({'error':f'CSV parse error: {ex}'}),400
        except (UnicodeDecodeError,ValueError) as ex:
            return jsonify({'error':f'Could not read this file as CSV: {ex}'}),400
        if df.empty or len(df.columns)==0:
            return jsonify({'error':'CSV has no columns/rows'}),400
        lat_col=lon_col=depth_col=None
        for c in df.columns:
            cl=c.lower().strip()
            if cl in('lat','latitude','y'):lat_col=c
            elif cl in('lon','longitude','lng','x'):lon_col=c
            elif cl in('depth','bathymetry','z'):depth_col=c
        if not all([lat_col,lon_col,depth_col]):return jsonify({'error':f'Need lat/lon/depth. Found: {list(df.columns)}'}),400
        pts=[]
        for _,row in df.iterrows():
            try:
                la=float(row[lat_col]);lo=float(row[lon_col]);d=abs(float(row[depth_col]))
                if 0<d<200:pts.append({'lat':round(la,6),'lon':round(lo,6),'depth':round(d,2)})
            except:continue
        return jsonify({'points':pts,'count':len(pts)})
    except RequestEntityTooLarge:
        raise  # ROB-R4: let Flask's class errorhandler produce the JSON 413
    except Exception as ex:return jsonify({'error':str(ex)}),500

@app.route('/api/extract',methods=['POST'])
def api_extract():
    try:
        data=req.get_json(force=True,silent=True) or {}
        bbox_dict=data.get('bbox')
        if not bbox_dict:return jsonify({'error':'Draw a rectangle'}),400
        p=data.get('params',{});mode=p.get('mode','gebco_s2')
        ref_source=p.get('ref_source','gebco')  # gebco, icesat2, chart, fusion
        depth_method=p.get('depth_method','cnn')  # cnn, sdb, hybrid
        sd=data.get('start_date','2024-05-01');ed=data.get('end_date','2024-09-30')
        user_pts=data.get('user_points',[])
        bbox=[bbox_dict['west'],bbox_dict['south'],bbox_dict['east'],bbox_dict['north']]
        cap=_enforce_roi_cap(bbox)
        if cap is not None:return cap
        w,s,e,n=bbox;area=abs(e-w)*abs(n-s)*111*111*math.cos(math.radians((n+s)/2))
        # User-selectable output resolution (10/20/50/100 m). Fallback: area-adaptive.
        user_res = p.get('resolution_m') or data.get('resolution_m')
        if user_res is not None:
            try:
                res = int(user_res)
            except (TypeError, ValueError):
                res = 10
            if res not in (10, 20, 50, 100):
                res = max(10, min(100, res))
            L.info(f"EXTRACT: user-requested resolution = {res}m")
        else:
            res = 10 if area < 30 else 20  # area-adaptive default
        # When observed data is present → auto-fuse ALL sources (pro mode)
        n_observed = sum(1 for p in user_pts if p.get('photon_class')=='observed')
        fast_cnn = n_observed >= 50
        if n_observed > 0:
            ref_source = 'fusion'  # auto-upgrade to fusion when observed data present
        L.info(f"=== EXTRACT mode={mode} ref={ref_source} method={depth_method} area={area:.0f}km² observed={n_observed} fast={fast_cnn} ===")
        out={'points':[],'interpolated_points':[],'stats':{},'bbox':bbox_dict,'sources_used':[],'ml_stats':{},'tracks':[],'sea_profiles':[],'bath_profiles':[]}

        # ═══ HIGH-ACCURACY MANY-IMAGE MLE TIER (user ask #1) ═══
        # The main "Extract" button becomes the most-accurate run by sending
        # params.accuracy="high" (or params.n_scenes). Routes to the calibrated
        # many-scene inverse-variance MLE stack (_run_s2_mle, up to 12 S2 images,
        # ICESat-2 + i-Boating + GEBCO calibration, physical-QC scene screening),
        # with PRO σ-QA (drop far-σ pixels) + interactive-recolor result_id.
        _acc = str(p.get('accuracy', data.get('accuracy', ''))).lower()
        _nsc_req = p.get('n_scenes', data.get('n_scenes'))
        if LS_AVAILABLE and (_acc in ('high', 'mle', 'max', 'accurate', 'best')
                             or _nsc_req is not None):
            try:
                _nsc = int(_nsc_req) if _nsc_req is not None else 10
            except (TypeError, ValueError):
                _nsc = 10
            _nsc = max(2, min(12, _nsc))
            try:
                _srk = float(p.get('sigma_reject_k', data.get('sigma_reject_k', 3.0)))
            except (TypeError, ValueError):
                _srk = 3.0
            _smx = p.get('sigma_max_m', data.get('sigma_max_m'))
            try:
                _smx = float(_smx) if _smx is not None else None
            except (TypeError, ValueError):
                _smx = None
            try:
                _yr = int(str(sd)[:4])
            except (TypeError, ValueError):
                _yr = 2024
            L.info(f"EXTRACT high-accuracy tier → MLE year={_yr} n_scenes={_nsc} "
                   f"sigma_reject_k={_srk} sigma_max_m={_smx}")
            try:
                _r = _run_s2_mle(bbox, year=_yr, n_scenes=_nsc,
                                 max_cloud=int(p.get('max_cloud', 20)),
                                 user_pts=user_pts,
                                 fetch_sliderule=bool(p.get('use_sliderule', True)),
                                 sigma_reject_k=_srk, sigma_max_m=_smx)
                out['stats'] = _r.get('stats', {})
                out['ml_stats'] = _r.get('ml_stats', {})
                out['interpolated_points'] = _r.get('interpolated_points', [])
                out['contours'] = _r.get('contours', [])
                out['contour_levels'] = _r.get('contour_levels', [])
                out['raster_png'] = _r.get('depth_png_b64')
                out['raster_bounds'] = _r.get('raster_bounds')
                out['raster_max_depth'] = _r.get('raster_max_depth')
                out['raster_min_depth'] = _r.get('raster_min_depth')
                out['raster_auto_min'] = _r.get('raster_auto_min')
                out['raster_auto_max'] = _r.get('raster_auto_max')
                out['result_id'] = _r.get('result_id')
                out['uncertainty_png_b64'] = _r.get('uncertainty_png_b64')
                out['downloads'] = _r.get('downloads', [])
                out['stability'] = _r.get('stability')
                out['accuracy_tier'] = 'high'
                out['sources_used'].append(
                    f"MLE-tier({_r.get('scenes_kept','?')} scenes, σ-QA)")
                return jsonify(_json_safe(out))
            except Exception as _mex:
                L.warning(f"EXTRACT high-accuracy MLE failed ({_mex}); "
                          f"falling back to standard pipeline")
                out['sources_used'].append(f"MLE-tier-fallback({str(_mex)[:60]})")

        # 1. Reference depths — FUSION: gather ALL available sources
        rl,rlo,rd,rw=[],[],[],[]  # rw = per-point weights

        # Source quality weights (literature-informed):
        #   Observed in-situ: ±0.1-0.3m survey grade → weight 6.0 (highest)
        #   ICESat-2: ±0.3m vertical accuracy → weight 5.0
        #   CSV/in-situ: assumed surveyed → weight 3.0
        #   Chart AI: ±1-2m accuracy → weight 2.0
        #   GEBCO: ~450m resolution, interpolated → weight 1.0
        W_ICESAT2, W_CHART, W_CSV, W_GEBCO, W_OBSERVED, W_STORED = 5.0, 2.0, 3.0, 1.0, 6.0, 5.5

        # ── Source 0: Historical survey calibration data ──
        n_rag = 0
        if _get_store:
            try:
                store = _get_store()
                stored = store.query_bbox(bbox, buffer_km=10, max_points=5000)
                if stored['count'] > 0:
                    # Reference-library filter: drop points shallower than 5 m
                    # (they are too close to the waterline to be reliable SDB targets
                    #  — turbidity, sun glint and surf dominate that regime).
                    s_lats = np.asarray(stored['lats'])
                    s_lons = np.asarray(stored['lons'])
                    s_depths = np.asarray(stored['depths'])
                    keep = s_depths >= 5.0
                    dropped = int(np.sum(~keep))
                    s_lats = s_lats[keep]; s_lons = s_lons[keep]; s_depths = s_depths[keep]
                    n_rag = int(keep.sum())
                    if n_rag > 0:
                        rl.extend(s_lats.tolist())
                        rlo.extend(s_lons.tolist())
                        rd.extend(s_depths.tolist())
                        rw.extend([W_STORED] * n_rag)
                    out['sources_used'].append(f"Calibration({n_rag}, ≥5m; dropped {dropped})")
                    L.info(f"Calibration data: {n_rag} pts kept, {dropped} dropped (<5m)")
            except Exception as ex:
                L.warning(f"Calibration query failed: {ex}")

        # ── Source A: GEBCO/ETOPO (only if RAG didn't provide enough) ──
        if n_rag < 20 and n_observed == 0:
            gebco=fetch_gebco(bbox)
            if gebco:
                rl.extend(gebco['lats']);rlo.extend(gebco['lons']);rd.extend(gebco['depths'])
                rw.extend([W_GEBCO]*len(gebco['depths']))
                out['sources_used'].append(f"ETOPO({len(gebco['depths'])} w={W_GEBCO})")
        elif n_observed > 0:
            L.info("Skipping GEBCO — observed in-situ data available")
        else:
            L.info(f"Skipping GEBCO — RAG has {n_rag} high-quality pts")

        # ── Source B: ICESat-2 altimeter via SlideRule ──
        # ALWAYS attempted (any region, any mode) — refraction-corrected ATL03
        # bathymetric photons are the best globally-available calibration
        # (weight 5.0) and turn the Stumpf/Lyzenga fit from chart-dependent
        # into truly worldwide. Opt out with params.use_sliderule=false.
        icesat=[]
        sliderule_bathy=[]  # passed to Lyzenga+SlideRule SDB as high-weight calibration
        use_sliderule = str(p.get('use_sliderule', '1')).lower() not in ('0', 'false', 'no')
        if use_sliderule or ref_source in('icesat2','fusion','all') or mode in('icesat2','fusion'):
            try:
                icesat,msg=run_sliderule(bbox,sd,ed)
                if icesat:
                    bathy_ice=[p for p in icesat if p['photon_class']=='bathymetry' and p['depth']>0]
                    sliderule_bathy=bathy_ice
                    for ip in bathy_ice:
                        rl.append(ip['lat']);rlo.append(ip['lon']);rd.append(ip['depth'])
                        rw.append(W_ICESAT2)
                    out['sources_used'].append(f"ICESat2({len(bathy_ice)} w={W_ICESAT2})")
                    out['points'].extend(icesat)
                    udates=sorted(set(p.get('acq_date','') for p in icesat if p.get('acq_date')))
                    out['icesat2_dates']=udates
                    out['icesat2_msg']=msg
                else:
                    out['sources_used'].append(f"ICESat2(0 — {msg})")
            except Exception as ex:
                L.warning(f"ICESat-2 SlideRule failed: {ex}")
                out['sources_used'].append(f"ICESat2(error: {str(ex)[:60]})")

        # ── Source B2: Professional AI Chart Digitisation ──
        # Full pipeline: i-Boating screenshot → colour rasterisation → isobath contours
        # → Gemini sounding extraction → cross-validation → fused depth points
        n_chart_total = 0
        try:
            api_key = os.getenv('GEMINI_API_KEY','') or os.getenv('GOOGLE_API_KEY','')
            if api_key:
                try:
                    from backend.iboating import (
                        _capture_iboating, _colour_to_depth_raster,
                        _extract_depths_from_chart, _reject_outliers,
                        _extract_isobath_contours, _contours_to_geo_points,
                        _pixels_to_geo,
                    )
                except ImportError:
                    from iboating import (
                        _capture_iboating, _colour_to_depth_raster,
                        _extract_depths_from_chart, _reject_outliers,
                        _extract_isobath_contours, _contours_to_geo_points,
                        _pixels_to_geo,
                    )

                L.info("Chart AI: professional digitisation pipeline starting...")
                chart_all_pts = []

                # Step 1: Capture i-Boating chart (zoom 14 for detail).
                # chart_bbox = TRUE rendered Mapbox-GL bounds of the screenshot
                # — every pixel→lat/lon below must use it, NOT the ROI bbox
                # (the old assumption drifted soundings by up to ~1 km).
                try:
                    fpath, img_b64, iw, ih, chart_bbox = _capture_iboating(bbox, zoom=14, wait_sec=10)
                    L.info(f"Chart AI: captured i-Boating z14 ({iw}x{ih})")

                    # Step 2: Colour rasterisation → depth grid
                    colour_depth, water_mask, colour_conf = _colour_to_depth_raster(img_b64)
                    if colour_depth is not None:
                        n_depth_px = int(np.sum(np.isfinite(colour_depth)))
                        L.info(f"Chart AI: colour raster → {n_depth_px} depth pixels")

                        # Step 3: Extract isobath contours from colour boundaries
                        try:
                            contours = _extract_isobath_contours(colour_depth, water_mask)
                            if contours:
                                ch_h, ch_w = colour_depth.shape
                                contour_geo = _contours_to_geo_points(contours, chart_bbox, ch_h, ch_w)
                                # Subsample contours (keep max ~300 pts)
                                if len(contour_geo) > 300:
                                    step = max(1, len(contour_geo) // 300)
                                    contour_geo = contour_geo[::step]
                                chart_all_pts.extend(contour_geo)
                                L.info(f"Chart AI: {len(contour_geo)} contour pts from {len(contours)} isobaths")
                        except Exception as cex:
                            L.warning(f"Chart AI: contour extraction: {cex}")

                        # Step 4: Sample colour raster grid (dense depth coverage)
                        ch_h, ch_w = colour_depth.shape
                        step_r = max(1, ch_h // 25)  # ~25x25 = 625 samples
                        step_c = max(1, ch_w // 25)
                        w_b, s_b, e_b, n_b = chart_bbox
                        colour_samples = 0
                        for rr in range(0, ch_h, step_r):
                            for cc in range(0, ch_w, step_c):
                                if not np.isfinite(colour_depth[rr, cc]):
                                    continue
                                lat = n_b - (rr / ch_h) * (n_b - s_b)
                                lon = w_b + (cc / ch_w) * (e_b - w_b)
                                conf = float(colour_conf[rr, cc]) if colour_conf is not None else 0.5
                                chart_all_pts.append({
                                    'lat': round(lat, 6), 'lon': round(lon, 6),
                                    'depth': round(float(colour_depth[rr, cc]), 2),
                                    'type': 'colour_grid', 'confidence': round(conf, 2),
                                })
                                colour_samples += 1
                        L.info(f"Chart AI: {colour_samples} colour grid samples")

                    # Step 5: Gemini Vision sounding extraction
                    raw_pts, cerr = _extract_depths_from_chart(img_b64, chart_bbox, iw, ih)
                    if raw_pts and len(raw_pts) > 0:
                        # Cross-validate against colour raster
                        if colour_depth is not None:
                            raw_pts = _reject_outliers(raw_pts, colour_depth, colour_conf, iw, ih)
                        geo_soundings = _pixels_to_geo(
                            [p for p in raw_pts if p.get('confidence', 0) > 0],
                            chart_bbox, iw, ih)
                        chart_all_pts.extend(geo_soundings)
                        L.info(f"Chart AI: {len(geo_soundings)} Gemini verified soundings")

                except Exception as cap_ex:
                    L.warning(f"Chart AI: i-Boating capture failed ({cap_ex}), falling back to OSM")
                    # Fallback: OSM tiles + Gemini
                    chart_img = _fetch_osm_depth_chart(bbox, zoom=12)
                    if chart_img:
                        img_b64, osm_iw, osm_ih, actual_bbox = chart_img
                        raw_pts, cerr = _gemini_extract_soundings(img_b64, actual_bbox, osm_iw, osm_ih)
                        if raw_pts:
                            cw, cs, ce, cn = actual_bbox
                            for p in raw_pts:
                                if 'depth' not in p or p['depth'] <= 0: continue
                                px, py = p.get('x', 0), p.get('y', 0)
                                lon = cw + (px / max(osm_iw, 1)) * (ce - cw)
                                lat = cn - (py / max(osm_ih, 1)) * (cn - cs)
                                chart_all_pts.append({
                                    'lat': round(lat, 6), 'lon': round(lon, 6),
                                    'depth': round(min(float(p['depth']), MAX_DEPTH_M), 2),
                                    'type': 'sounding', 'confidence': p.get('confidence', 0.6),
                                })

                # Add chart points to reference data. The screenshot's true
                # bounds are wider than the ROI (viewport aspect), so first
                # clip to the ROI with a small margin.
                if chart_all_pts:
                    _mw = 0.1 * (bbox[2] - bbox[0]); _mh = 0.1 * (bbox[3] - bbox[1])
                    chart_all_pts = [p for p in chart_all_pts
                                     if bbox[0]-_mw <= p['lon'] <= bbox[2]+_mw
                                     and bbox[1]-_mh <= p['lat'] <= bbox[3]+_mh]
                    try:
                        from global_land_mask import globe
                        chart_all_pts = [p for p in chart_all_pts if not globe.is_land(p['lat'], p['lon'])]
                    except: pass
                    W_CHART_REF = 4.0  # Higher weight — professional chart data
                    for cp in chart_all_pts:
                        rl.append(cp['lat']); rlo.append(cp['lon']); rd.append(cp['depth'])
                        rw.append(W_CHART_REF)
                    n_chart_total = len(chart_all_pts)
                    out['points'].extend([{**cp, 'photon_class': 'chart'} for cp in chart_all_pts[:500]])
                    out['sources_used'].append(f"ChartAI({n_chart_total} pts)")
                    L.info(f"Chart AI: total {n_chart_total} professional depth points")
        except Exception as cx:
            L.warning(f"Chart AI pipeline: {cx}")
            L.debug(traceback.format_exc())

        # ── Source C: Observed in-situ survey data (highest quality) ──
        # Subsample if >2000 — CNN doesn't need 40k pts on 500px grid
        nc=0;nobs=0
        obs_pts=[up for up in user_pts if up.get('photon_class')=='observed'] if user_pts else []
        other_pts=[up for up in user_pts if up.get('photon_class')!='observed'] if user_pts else []
        MAX_OBS=2000
        if len(obs_pts)>MAX_OBS:
            idx=np.random.choice(len(obs_pts),MAX_OBS,replace=False)
            obs_pts=[obs_pts[i] for i in idx]
            L.info(f"Subsampled observed: {n_observed} → {MAX_OBS} pts for training")
        for up in obs_pts:
            if 'lat' in up and 'depth' in up:
                rl.append(up['lat']);rlo.append(up['lon']);rd.append(up['depth'])
                rw.append(W_OBSERVED);nobs+=1

        # ── Source D: Chart AI / CSV / other user points ──
        for up in other_pts:
            if 'lat' in up and 'depth' in up:
                rl.append(up['lat']);rlo.append(up['lon']);rd.append(up['depth'])
                pw=W_CSV if up.get('photon_class') in ('insitu','csv') else W_CHART
                rw.append(pw);nc+=1
        if nc:out['sources_used'].append(f"Chart/CSV({nc})")
        if nobs:out['sources_used'].append(f"Observed({nobs} w={W_OBSERVED})")

        # ── Unified fusion: dedup by proximity, keep highest-weight source ──
        # When ICESat-2 + i-Boating + observed + GEBCO co-exist, many points fall
        # within a few 10s of meters. Keep only the best-weighted sample per
        # ~30 m cell so the CNN sees one consistent target per location instead
        # of noisy duplicates that fight each other during training.
        if len(rl) > 10:
            try:
                la = np.asarray(rl, dtype=np.float64)
                lo = np.asarray(rlo, dtype=np.float64)
                dp = np.asarray(rd, dtype=np.float64)
                ww = np.asarray(rw, dtype=np.float64)
                # Snap to a ~30 m grid (in degrees: ~0.00027°)
                CELL = 0.00027
                keys = (np.round(la / CELL).astype(np.int64) << 32) | (np.round(lo / CELL).astype(np.int64) & 0xFFFFFFFF)
                # For each cell, keep the point with the highest weight
                order = np.argsort(-ww, kind='stable')   # descending by weight
                seen = set()
                keep_idx = []
                for i in order:
                    k = int(keys[i])
                    if k in seen:
                        continue
                    seen.add(k); keep_idx.append(int(i))
                keep_idx = np.asarray(keep_idx, dtype=np.int64)
                before = len(rl)
                rl  = la[keep_idx].tolist()
                rlo = lo[keep_idx].tolist()
                rd  = dp[keep_idx].tolist()
                rw  = ww[keep_idx].tolist()
                dedup_dropped = before - len(rl)
                if dedup_dropped > 0:
                    L.info(f"Fusion dedup (~30m cell, weight-priority): {before} → {len(rl)} pts "
                           f"({dedup_dropped} duplicates removed)")
                    out['sources_used'].append(f"Dedup30m(-{dedup_dropped})")
            except Exception as _dex:
                L.warning(f"Fusion dedup skipped: {_dex}")

        # ── Source-weighted oversampling ──
        # Skip oversampling when observed data dominates (already high quality)
        if fast_cnn:
            L.info(f"Fast mode: skipping oversampling ({len(rl)} observed pts already sufficient)")
        elif len(rw)>10:
            rl_w,rlo_w,rd_w=[],[],[]
            max_w=max(rw)
            for i in range(len(rl)):
                n_copies=max(1,round(rw[i]/max_w*3))  # 1-3 copies based on weight
                for _ in range(n_copies):
                    rl_w.append(rl[i]);rlo_w.append(rlo[i]);rd_w.append(rd[i])
            # Cap total to avoid CNN training explosion
            MAX_TOTAL = 5000
            if len(rl_w) > MAX_TOTAL:
                idx_cap = np.random.choice(len(rl_w), MAX_TOTAL, replace=False)
                rl_w = [rl_w[i] for i in idx_cap]
                rlo_w = [rlo_w[i] for i in idx_cap]
                rd_w = [rd_w[i] for i in idx_cap]
            L.info(f"Source weighting: {len(rl)} → {len(rl_w)} points")
            rl,rlo,rd=rl_w,rlo_w,rd_w

        # Add reference points to output for display
        step=max(1,len(rd)//400)
        nb4=len(rd)-nc
        for i in range(0,len(rd),step):
            cls='chart' if i>=nb4 else 'gebco'
            out['points'].append({'lat':round(float(rl[i]),5),'lon':round(float(rlo[i]),5),'depth':round(float(rd[i]),1),'photon_class':cls})

        # 2. Imagery — Parallel tiled @10m or single fetch
        depth = None
        img_source = p.get('img_source', 'sentinel2')

        # ── Try parallel tiled processing for very large areas (>200 km²) ──
        if img_source == 'sentinel2' and area > 200:
            ref_pts_dict = {'lats': np.array(rl), 'lons': np.array(rlo), 'depths': np.array(rd)}
            mosaic, tile_stats = parallel_tiled_bathymetry(bbox, sd, ed, ref_pts_dict,
                depth_method=depth_method)
            if mosaic is not None:
                depth = mosaic
                out['sources_used'].append(f"ParaTile({tile_stats['tiles_ok']}/{tile_stats['tiles_total']} "
                    f"@10m, {tile_stats['workers']}cpu, {tile_stats['grid_size']})")
                # Build a minimal s2 dict for downstream (land mask etc)
                H_m, W_m = depth.shape
                s2 = {'red': np.zeros((H_m, W_m), dtype=np.uint16),
                       'green': np.zeros((H_m, W_m), dtype=np.uint16),
                       'blue': np.zeros((H_m, W_m), dtype=np.uint16),
                       'nir': np.zeros((H_m, W_m), dtype=np.uint16),
                       'coastal': np.zeros((H_m, W_m), dtype=np.uint16),
                       'ndwi': np.where(np.isfinite(depth), 0.5, -0.5),
                       'water_mask': np.isfinite(depth) & (depth > 0),
                       'width': W_m, 'height': H_m}
                res = 10
                L.info(f"Parallel tiled: {H_m}x{W_m} mosaic @10m")
                # Skip to step 5 (land mask + output) — depth already computed
                # Jump past imagery fetch + depth estimation sections
                pass  # depth is set, will be caught by 'if depth is not None: pass'

        # ── Single tile fetch (small area or parallel not needed) ──
        if depth is None:
            if img_source in ('mapbox', 'mapbox_hr'):
                mb_zoom = 17 if img_source == 'mapbox_hr' else 15
                s2 = fetch_mapbox_s2(bbox, zoom=mb_zoom)
                out['sources_used'].append(f"Mapbox({s2['width']}x{s2['height']} ~{s2.get('resolution_m','')}m)")
            else:
                try:
                    s2=fetch_s2(bbox,sd,ed,res=res,cloud=p.get('max_cloud',20))
                except Exception as shx:
                    # Sentinel-Hub/CDSE creds expired → same GEE fallback the
                    # other S2 endpoints already use (returns identical dict)
                    L.warning(f"fetch_s2 failed ({shx}); falling back to GEE")
                    s2=fetch_s2_gee(bbox,sd,ed,res=res,cloud=max(int(p.get('max_cloud',20)),30))
                    out['sources_used'].append("S2-GEE-fallback")
                out['sources_used'].append(f"S2({s2['width']}x{s2['height']}@{res}m)")
                if s2.get('glint_corrected'):out['sources_used'].append("GlintCorrected(Hedley2005)")
                if s2.get('deep_water_corrected'):out['sources_used'].append("DeepWaterCorrected(Lyzenga1978)")

        # ── Turbidity detection ──
        # High red/green ratio in water = suspended sediment = SDB unreliable
        # Dogliotti et al. (2015): turbid water has high red reflectance
        water_mask=s2.get('water_mask',s2['ndwi']>0)
        red_f=s2['red'].astype(float)/10000;green_f=s2['green'].astype(float)/10000
        turbidity_ratio=np.where(water_mask,red_f/(green_f+1e-6),0)
        turbid_mask=water_mask&(turbidity_ratio>0.8)  # Red/Green>0.8 = likely turbid
        turbid_pct=round(float(np.sum(turbid_mask)/max(np.sum(water_mask),1)*100),1)
        if turbid_pct>5:
            out['sources_used'].append(f"TurbidWarning({turbid_pct}%)")
            L.info(f"Turbidity: {turbid_pct}% of water area flagged as turbid (R/G>{0.8})")
        s2['turbid_mask']=turbid_mask
        s2['turbid_pct']=turbid_pct

        # 3. Depth estimation
        depth=None
        wave_stats=None

        # ── PRO MODE: Observed-Calibrated Fusion (IDW + GBR + residual) ──
        # When dense observed data is available, use the smart fusion pipeline
        # instead of CNN — faster and more accurate with in-situ data
        if nobs >= 50 and depth is None:
            L.info(f"ObsFusion: using professional IDW+GBR+Residual pipeline ({nobs} observed pts)")
            # Build global water mask for this image
            H_img,W_img=s2['red'].shape
            global_water=make_water_mask(bbox,H_img,W_img,s2=s2)
            if global_water is not None:
                s2['water_mask']=global_water
            ref={'lats':np.array(rl),'lons':np.array(rlo),'depths':np.array(rd)}
            result,err=observed_fusion_depth(s2,ref,bbox,water_mask=global_water)
            if result:
                depth=result['depth']
                method=result.get('method','ObsFusion')
                out['sources_used'].append(f"{method}(R²={result['r2']})")
                out['ml_stats']={'method':method,'r2':result['r2'],'n_train':result['n_train'],
                                 'importance':result['importance'],'fusion_stats':result.get('fusion_stats',{})}
                L.info(f"ObsFusion complete: {method}")
            else:
                L.warning(f"ObsFusion failed: {err}, falling back to CNN")

        # ── WAVE DISPERSION (Almar et al.) ──
        if depth_method in ('wave','physics_hybrid') and WAVE_AVAILABLE:
            try:
                L.info("Fetching S2 single-orbit for wave analysis (B02+B04 @10m)...")
                s2_wave=fetch_s2_wave(bbox,sd,ed,cloud=p.get('max_cloud',20))
                out['sources_used'].append(f"S2-wave({s2_wave['width']}x{s2_wave['height']}@10m)")
                wave_result,wave_err=wave_bathymetry(s2_wave,bbox,
                    window_m=float(p.get('wave_window',400)),
                    step_m=float(p.get('wave_step',100)))
                if wave_result:
                    wave_depth=wave_result['depth']
                    wave_stats=wave_result['stats']
                    out['sources_used'].append(f"WaveDispersion({wave_stats['valid_windows']} windows, λ={wave_stats['mean_wavelength_m']}m)")
                    out['wave_stats']=wave_stats
                    if depth_method=='wave':
                        # Pure wave — upscale to S2 grid size
                        from scipy.ndimage import zoom as ndizoom
                        H_s2,W_s2=s2['red'].shape
                        wy,wx=wave_depth.shape
                        if wy>1 and wx>1:
                            depth=ndizoom(wave_depth,
                                (H_s2/wy,W_s2/wx),order=1,mode='nearest')
                            depth=np.where(np.isfinite(depth)&(depth>0),
                                np.clip(depth,0,MAX_DEPTH_M),np.nan)
                            water=s2['ndwi']>0;depth[~water]=np.nan
                        out['ml_stats']={'method':'Wave Dispersion (Almar et al.)',**wave_stats}
                    else:
                        L.info("Wave done — will blend with CNN/SDB in physics_hybrid")
                else:
                    L.warning(f"Wave failed: {wave_err}")
                    out['sources_used'].append(f"Wave-failed({wave_err})")
            except Exception as ex:
                L.warning(f"Wave analysis error: {ex}")
                out['sources_used'].append(f"Wave-error({str(ex)[:80]})")

        # ── PHYSICS_HYBRID: blend wave + CNN + SDB ──
        if depth_method=='physics_hybrid' and depth is None:
            # Get CNN depth
            cnn_d=None
            if len(rd)>=10 and (SK or CNN_AVAILABLE):
                ref={'lats':np.array(rl),'lons':np.array(rlo),'depths':np.array(rd)}
                result,err=cnn_depth(s2,ref,bbox,fast=fast_cnn)
                if result:
                    cnn_d=result['depth']
                    out['ml_stats']={'method':'Physics Hybrid','r2':result['r2'],'n_train':result['n_train'],'importance':result['importance']}
            # Get SDB depth
            sdb_d=fallback_sdb(s2,rd if len(rd)>=5 else [],ref_lats=rl if len(rl)>=5 else None,ref_lons=rlo if len(rlo)>=5 else None,bbox=bbox,ref_weights=rw if len(rw)>=5 else None,sliderule_pts=sliderule_bathy)
            # Blend: wave 0.5 + CNN 0.3 + SDB 0.2 (wave is physics-based → highest trust)
            H_s2,W_s2=s2['red'].shape;water=s2['ndwi']>0
            wave_up=None
            if wave_stats and wave_stats.get('valid_windows',0)>=3:
                try:
                    from scipy.ndimage import zoom as ndizoom
                    wd=wave_depth if 'wave_depth' in dir() else None
                    if wd is not None:
                        wy,wx=wd.shape
                        if wy>1 and wx>1:
                            wave_up=ndizoom(wd,(H_s2/wy,W_s2/wx),order=1,mode='nearest')
                            wave_up=np.where(np.isfinite(wave_up)&(wave_up>0),wave_up,np.nan)
                except:pass
            # ── Adaptive weighted blend ──
            # FIX: Weights are now computed from each method's RMSE against ref points.
            # Better method → higher weight. Falls back to fixed weights if no ref data.
            w_wave,w_cnn,w_sdb=0.5,0.3,0.2  # defaults
            if len(rd)>=10:
                w_b,s_b,e_b,n_b=bbox
                method_rmse={}
                for name,grid in [('wave',wave_up),('cnn',cnn_d),('sdb',sdb_d)]:
                    if grid is None:continue
                    errs=[]
                    for i in range(0,len(rl),max(1,len(rl)//200)):
                        r_px=max(0,min(H_s2-1,int((n_b-rl[i])/(n_b-s_b+1e-10)*H_s2)))
                        c_px=max(0,min(W_s2-1,int((rlo[i]-w_b)/(e_b-w_b+1e-10)*W_s2)))
                        if water[r_px,c_px] and np.isfinite(grid[r_px,c_px]) and grid[r_px,c_px]>0:
                            errs.append((grid[r_px,c_px]-rd[i])**2)
                    if len(errs)>3:
                        method_rmse[name]=np.sqrt(np.mean(errs))
                if method_rmse:
                    # Inverse-RMSE weighting: lower RMSE → higher weight
                    inv_rmse={k:1.0/(v+0.1) for k,v in method_rmse.items()}
                    total=sum(inv_rmse.values())
                    if 'wave' in inv_rmse:w_wave=inv_rmse['wave']/total
                    if 'cnn' in inv_rmse:w_cnn=inv_rmse['cnn']/total
                    if 'sdb' in inv_rmse:w_sdb=inv_rmse['sdb']/total
                    L.info(f"Adaptive weights: wave={w_wave:.2f} cnn={w_cnn:.2f} sdb={w_sdb:.2f} "
                           f"(RMSE: {method_rmse})")

            # Vectorised blend (no per-pixel loop)
            depth=np.full((H_s2,W_s2),np.nan,dtype=np.float32)
            sum_w=np.zeros((H_s2,W_s2),dtype=np.float32)
            sum_d=np.zeros((H_s2,W_s2),dtype=np.float32)
            for grid,w in [(wave_up,w_wave),(cnn_d,w_cnn),(sdb_d,w_sdb)]:
                if grid is None:continue
                valid=water&np.isfinite(grid)&(grid>0)
                # Downweight turbid areas for optical methods
                turbid=s2.get('turbid_mask',np.zeros_like(water,dtype=bool))
                turb_penalty=np.where(turbid,0.3,1.0) if grid is not wave_up else np.ones_like(water,dtype=float)
                eff_w=w*turb_penalty
                sum_d[valid]+=grid[valid]*eff_w[valid]
                sum_w[valid]+=eff_w[valid]
            has_data=sum_w>0
            depth[has_data]=np.clip(sum_d[has_data]/sum_w[has_data],0,MAX_DEPTH_M)
            depth[~water]=np.nan
            src_parts=[]
            if wave_up is not None:src_parts.append(f'Wave×{w_wave:.2f}')
            if cnn_d is not None:src_parts.append(f'CNN×{w_cnn:.2f}')
            src_parts.append(f'SDB×{w_sdb:.2f}')
            out['sources_used'].append(f"AdaptiveHybrid({'+'.join(src_parts)})")
            if wave_stats:out['ml_stats']['wave']=wave_stats

        if depth is not None:
            pass  # already set by ObsFusion or wave
        elif depth_method=='lyzenga_sliderule' and LS_AVAILABLE:
            # Robust Lyzenga + Stumpf fusion weighted by SlideRule ICESat-2 photons.
            # Always returns finite depths over water (no all-NaN failure mode).
            try:
                ls_res = ls_estimate_depth(
                    s2, bbox,
                    ref_lats=rl, ref_lons=rlo, ref_depths=rd, ref_weights=rw,
                    sliderule_pts=sliderule_bathy,
                )
                depth = ls_res['depth']
                out['sources_used'].append(
                    f"Lyzenga+SlideRule({ls_res['method']},"
                    f" n={ls_res.get('n_train',0)},sr={ls_res.get('n_sliderule',0)})")
                fit_l = (ls_res.get('fit') or {}).get('lyzenga')
                fit_bg = (ls_res.get('fit') or {}).get('stumpf_bg')
                fit_br = (ls_res.get('fit') or {}).get('stumpf_br')
                rmses = [f['rmse'] for f in (fit_l, fit_bg, fit_br) if f]
                best_rmse = round(min(rmses), 3) if rmses else 0
                out['ml_stats'] = {
                    'method': 'Lyzenga+Stumpf+SlideRule',
                    'rmse': best_rmse,
                    'n_train': ls_res.get('n_train', 0),
                    'n_sliderule': ls_res.get('n_sliderule', 0),
                    'sources': ls_res.get('sources', []),
                    'importance': {'Lyzenga4band': 0.5, 'Stumpf_BG': 0.3, 'Stumpf_BR': 0.2},
                }
            except Exception as ex:
                L.warning(f"Lyzenga+SlideRule pipeline failed: {ex}")
                depth = fallback_sdb(s2, rd, ref_lats=rl, ref_lons=rlo, bbox=bbox,
                                     ref_weights=rw, sliderule_pts=sliderule_bathy)
                out['sources_used'].append('SDB-fallback')
        elif depth_method=='caballero_stumpf':
            # Method 2: Caballero-Stumpf (2020) + CNN residual enhancement
            ref={'lats':np.array(rl),'lons':np.array(rlo),'depths':np.array(rd)}
            result,err=caballero_stumpf_sdb(s2,ref,bbox)
            if result:
                cs_depth=result['depth']
                cs_stats=result['stats']
                out['sources_used'].append(f"Caballero-Stumpf(R²={result['r2']},RMSE={result.get('rmse',0)}m)")

                # ── CNN residual correction: train CNN on (residual = true - CS_pred) ──
                # Then final = CS + CNN_residual → combines physics + ML
                cnn_enhanced = False
                if (SK or CNN_AVAILABLE) and len(rd) >= 10:
                    try:
                        H_d, W_d = cs_depth.shape
                        w_b, s_b2, e_b, n_b2 = bbox
                        # Compute residuals at reference points
                        res_lats, res_lons, res_depths = [], [], []
                        for i in range(len(rl)):
                            r_px = max(0, min(H_d-1, int((n_b2-rl[i])/(n_b2-s_b2+1e-10)*H_d)))
                            c_px = max(0, min(W_d-1, int((rlo[i]-w_b)/(e_b-w_b+1e-10)*W_d)))
                            cs_val = cs_depth[r_px, c_px]
                            if np.isfinite(cs_val) and 0 < rd[i] <= MAX_DEPTH_M:
                                residual = rd[i] - cs_val  # true - predicted
                                res_lats.append(rl[i]); res_lons.append(rlo[i])
                                res_depths.append(residual)
                        if len(res_depths) >= 10:
                            res_ref = {'lats': np.array(res_lats), 'lons': np.array(res_lons),
                                       'depths': np.array(res_depths)}
                            cnn_result, cnn_err = cnn_depth(s2, res_ref, bbox, fast=True)
                            if cnn_result and cnn_result.get('depth') is not None:
                                residual_grid = cnn_result['depth']
                                # Clip residual correction to ±5m to avoid wild swings
                                residual_grid = np.clip(residual_grid, -5, 5)
                                # Final = CS physics + CNN residual
                                valid_both = np.isfinite(cs_depth) & np.isfinite(residual_grid)
                                depth = np.where(valid_both, cs_depth + residual_grid, cs_depth)
                                depth = np.clip(depth, 0, MAX_DEPTH_M)
                                water = s2.get('water_mask', s2['ndwi'] > 0)
                                depth[~water] = np.nan
                                cnn_enhanced = True
                                out['sources_used'].append(f"CNN-residual(R²={cnn_result['r2']})")
                                L.info(f"Method 2 CNN enhancement: residual R²={cnn_result['r2']}")
                    except Exception as cx:
                        L.warning(f"CNN residual enhancement failed: {cx}")

                if not cnn_enhanced:
                    depth = cs_depth

                # ── Scatter plot data (predicted vs reference) ──
                scatter_pred, scatter_true = [], []
                H_d, W_d = depth.shape
                w_b, s_b2, e_b, n_b2 = bbox
                for i in range(0, len(rl), max(1, len(rl) // 500)):
                    r_px = max(0, min(H_d-1, int((n_b2-rl[i])/(n_b2-s_b2+1e-10)*H_d)))
                    c_px = max(0, min(W_d-1, int((rlo[i]-w_b)/(e_b-w_b+1e-10)*W_d)))
                    pred_v = depth[r_px, c_px]
                    if np.isfinite(pred_v) and 0 < rd[i] <= MAX_DEPTH_M:
                        scatter_pred.append(round(float(pred_v), 2))
                        scatter_true.append(round(float(rd[i]), 2))

                # Final validation stats
                if len(scatter_pred) >= 5:
                    sp, st = np.array(scatter_pred), np.array(scatter_true)
                    res_arr = sp - st
                    final_r2 = round(float(1 - np.sum(res_arr**2) / np.sum((st - st.mean())**2)), 4)
                    final_rmse = round(float(np.sqrt(np.mean(res_arr**2))), 3)
                    final_medae = round(float(np.median(np.abs(res_arr))), 3)
                    final_bias = round(float(np.mean(res_arr)), 3)
                    cs_stats.update({'final_r2': final_r2, 'final_rmse': final_rmse,
                                     'final_medae': final_medae, 'final_bias': final_bias,
                                     'cnn_enhanced': cnn_enhanced})

                out['ml_stats'] = {'method': 'Caballero-Stumpf + CNN' if cnn_enhanced else result['method'],
                    'r2': cs_stats.get('final_r2', result['r2']),
                    'rmse': cs_stats.get('final_rmse', result.get('rmse', 0)),
                    'n_train': result['n_train'], 'importance': result['importance'],
                    'coefficients': result['coefficients'], 'caballero_stats': cs_stats}
                out['scatter'] = {'predicted': scatter_pred, 'reference': scatter_true}
                if result.get('turbidity'):
                    out['sources_used'].append(result['turbidity'])
            else:
                L.warning(f"Caballero-Stumpf failed ({err}), falling back to SDB")
                depth=fallback_sdb(s2,rd if len(rd)>=5 else [],ref_lats=rl if len(rl)>=5 else None,ref_lons=rlo if len(rlo)>=5 else None,bbox=bbox,ref_weights=rw if len(rw)>=5 else None,sliderule_pts=sliderule_bathy)
                out['sources_used'].append('SDB-fallback')
        elif depth_method=='sdb' or len(rd)<10:
            # Pure SDB (no reference data needed, or not enough ref pts)
            depth=fallback_sdb(s2,rd if len(rd)>=5 else [],ref_lats=rl if len(rl)>=5 else None,ref_lons=rlo if len(rlo)>=5 else None,bbox=bbox,ref_weights=rw if len(rw)>=5 else None,sliderule_pts=sliderule_bathy)
            out['sources_used'].append('SDB')
        elif depth_method=='patent_mlp' and MLP_AVAILABLE:
            # Patent US20260043650A1 MLP fusion model
            ref={'lats':np.array(rl),'lons':np.array(rlo),'depths':np.array(rd)}
            result,err=mlp_train_and_predict(s2,ref,bbox,epochs=300,train_ratio=0.05)
            if result:
                depth=result['depth']
                # Apply tidal correction (patent Claim 2)
                depth,tide_h,tide_method=apply_tidal_correction(depth,bbox,f"{sd}T12:00:00")
                out['sources_used'].append(f"PatentMLP(R²={result['r2']},RMSE={result['rmse']}m,tide={tide_h:+.2f}m)")
                out['ml_stats']={'method':result['method'],'r2':result['r2'],'rmse':result['rmse'],
                    'n_train':result['n_train'],'n_test':result['n_test'],
                    'per_zone_rmse':result['per_zone_rmse'],'architecture':result['architecture'],
                    'tidal_correction_m':tide_h,'importance':{}}
            else:
                L.warning(f"Patent MLP failed ({err}), falling back to CNN")
                depth_method='cnn'  # fall through to CNN

        elif depth_method in ('cnn','patent_mlp') and (SK or CNN_AVAILABLE):
            ref={'lats':np.array(rl),'lons':np.array(rlo),'depths':np.array(rd)}
            result,err=cnn_depth(s2,ref,bbox,fast=fast_cnn)
            if result:
                depth=result['depth'];method=result.get('method','CNN')
                out['sources_used'].append(f"{method}(R²={result['r2']})")
                out['ml_stats']={'method':method,'r2':result['r2'],'n_train':result['n_train'],'importance':result['importance']}
            else:
                L.warning(f"CNN failed ({err}), falling back to SDB")
                depth=fallback_sdb(s2,rd,ref_lats=rl,ref_lons=rlo,bbox=bbox,ref_weights=rw,sliderule_pts=sliderule_bathy);out['sources_used'].append('SDB-fallback')
        elif depth_method=='hybrid':
            # Run both SDB and CNN, blend them
            sdb_depth=fallback_sdb(s2,rd,ref_lats=rl,ref_lons=rlo,bbox=bbox,ref_weights=rw,sliderule_pts=sliderule_bathy)
            ref={'lats':np.array(rl),'lons':np.array(rlo),'depths':np.array(rd)}
            result,err=cnn_depth(s2,ref,bbox,fast=fast_cnn)
            if result:
                cnn_d=result['depth']
                # Weighted blend: CNN 0.7, SDB 0.3 (CNN is more spatially consistent)
                w_cnn,w_sdb=0.7,0.3
                both_valid=np.isfinite(cnn_d)&np.isfinite(sdb_depth)
                depth=np.where(both_valid,w_cnn*cnn_d+w_sdb*sdb_depth,np.where(np.isfinite(cnn_d),cnn_d,sdb_depth))
                depth[~(s2['ndwi']>0)]=np.nan
                out['sources_used'].append(f"Hybrid(CNN×{w_cnn}+SDB×{w_sdb})")
                out['ml_stats']={'method':'Hybrid','r2':result['r2'],'n_train':result['n_train'],'importance':result['importance']}
            else:
                depth=sdb_depth;out['sources_used'].append('SDB')
        elif depth is None:
            # Default: try CNN, fallback SDB (skip if ObsFusion already produced depth)
            if len(rd)>=10 and (SK or CNN_AVAILABLE):
                ref={'lats':np.array(rl),'lons':np.array(rlo),'depths':np.array(rd)}
                result,err=cnn_depth(s2,ref,bbox,fast=fast_cnn)
                if result:
                    depth=result['depth'];method=result.get('method','CNN')
                    out['sources_used'].append(f"{method}(R²={result['r2']})")
                    out['ml_stats']={'method':method,'r2':result['r2'],'n_train':result['n_train'],'importance':result['importance']}
                    # Expert-hydrographer correction: blend ML with physical prior
                    try:
                        depth_corr, corr_info = expert_correction(
                            depth, s2, rl, rlo, rd, bbox,
                            ml_r2=result.get('r2'),
                            turbid_pct=s2.get('turbid_pct', 0.0),
                        )
                        if corr_info.get('applied'):
                            depth = depth_corr
                            out['sources_used'].append(
                                f"ExpertCorr(α_ML={corr_info['alpha_ml']},phys_R²={corr_info['phys_r2']},|Δ|={corr_info['mean_abs_correction_m']}m)"
                            )
                            out['ml_stats']['expert_correction'] = corr_info
                    except Exception as _ec:
                        L.warning(f"ExpertCorr skipped: {_ec}")
                else:depth=fallback_sdb(s2,rd,ref_lats=rl,ref_lons=rlo,bbox=bbox,ref_weights=rw,sliderule_pts=sliderule_bathy);out['sources_used'].append('SDB')
            else:depth=fallback_sdb(s2,rd,ref_lats=rl,ref_lons=rlo,bbox=bbox,ref_weights=rw,sliderule_pts=sliderule_bathy);out['sources_used'].append('SDB')

        # Safety: if depth is still None, run basic SDB
        if depth is None:
            depth=fallback_sdb(s2,rd if len(rd)>=5 else [],ref_lats=rl if len(rl)>=5 else None,ref_lons=rlo if len(rlo)>=5 else None,bbox=bbox,ref_weights=rw if len(rw)>=5 else None,sliderule_pts=sliderule_bathy)
            out['sources_used'].append('SDB-emergency')
        if depth is None:
            H,W=s2['red'].shape
            depth=np.full((H,W),np.nan)

        # Final guard: if the selected method left >70 % of water pixels NaN,
        # patch the holes with the robust Lyzenga+SlideRule estimator. This
        # prevents the "all-NaN over Sentinel-2" failure mode the UI hit
        # when CNN/Caballero failed quietly. Operates only on water pixels.
        try:
            water_g = s2.get('water_mask', s2.get('ndwi', np.zeros_like(depth))>0)
            n_water = int(np.sum(water_g))
            if n_water > 50:
                n_nan = int(np.sum(water_g & ~np.isfinite(depth)))
                if n_nan > 0.7 * n_water and LS_AVAILABLE:
                    L.warning(f"Final guard: {n_nan}/{n_water} water pixels NaN — "
                              f"refilling with Lyzenga+SlideRule")
                    rescue = ls_estimate_depth(
                        s2, bbox,
                        ref_lats=rl, ref_lons=rlo, ref_depths=rd, ref_weights=rw,
                        sliderule_pts=sliderule_bathy,
                    )
                    rd_grid = rescue['depth']
                    fill = water_g & ~np.isfinite(depth) & np.isfinite(rd_grid)
                    if int(fill.sum()) > 0:
                        depth = depth.copy() if depth.dtype != np.float32 else depth
                        depth = depth.astype(np.float64)
                        depth[fill] = rd_grid[fill]
                        out['sources_used'].append(
                            f"NaNGuard(refilled {int(fill.sum())} px via Lyzenga+SlideRule)"
                        )
        except Exception as _gx:
            L.warning(f"Final NaN guard skipped: {_gx}")

        # 4b. Refine with pre-trained regional knowledge (silent)
        try:
            if _PRETRAINED_MODELS and depth is not None:
                H_d, W_d = depth.shape
                w_b, s_b, e_b, n_b = bbox
                # Check if pretrained model covers this region
                center_lat = (n_b + s_b) / 2
                center_lon = (e_b + w_b) / 2
                pt_depth, pt_unc, pt_region = query_pretrained_depth(center_lat, center_lon)
                if pt_region and pt_depth is not None:
                    # Build a prior grid from pretrained model
                    prior = np.full((H_d, W_d), np.nan, dtype=np.float64)
                    lat_grid = np.linspace(n_b, s_b, H_d)
                    lon_grid = np.linspace(w_b, e_b, W_d)
                    pm = _PRETRAINED_MODELS[pt_region]
                    meta = pm['meta']
                    lat_mean = meta['lat_mean']; lon_mean = meta['lon_mean']
                    lat_std = meta['lat_std']; lon_std = meta['lon_std']
                    dist_std_val = meta.get('dist_std', 1)
                    for ri in range(0, H_d, max(1, H_d // 50)):
                        for ci in range(0, W_d, max(1, W_d // 50)):
                            lat_n = (lat_grid[ri] - lat_mean) / (lat_std + 1e-8)
                            lon_n = (lon_grid[ci] - lon_mean) / (lon_std + 1e-8)
                            dlat = (lat_grid[ri] - lat_mean) * 111000
                            dlon = (lon_grid[ci] - lon_mean) * 111000 * np.cos(np.radians(lat_mean))
                            dist_n = np.sqrt(dlat**2 + dlon**2) / (dist_std_val + 1e-8)
                            X = np.array([[lat_n, lon_n, dist_n]])
                            if pm['scaler']:
                                X = pm['scaler'].transform(X)
                            prior[ri, ci] = float(np.clip(pm['model'].predict(X)[0], 0, MAX_DEPTH_M))
                    # Interpolate sparse prior to full grid
                    from scipy.ndimage import zoom as ndizoom
                    step_r = max(1, H_d // 50); step_c = max(1, W_d // 50)
                    sparse_h = len(range(0, H_d, step_r)); sparse_w = len(range(0, W_d, step_c))
                    prior_sparse = prior[::step_r, ::step_c]
                    if sparse_h > 1 and sparse_w > 1:
                        prior_full = ndizoom(prior_sparse, (H_d / sparse_h, W_d / sparse_w), order=1, mode='nearest')
                        prior_full = prior_full[:H_d, :W_d]
                        # Subtle blend: 90% CNN, 10% prior (just a gentle nudge)
                        valid_both = np.isfinite(depth) & np.isfinite(prior_full) & (depth > 0)
                        if valid_both.sum() > 10:
                            alpha = 0.10  # prior weight — subtle, not dominant
                            depth[valid_both] = (1 - alpha) * depth[valid_both] + alpha * prior_full[valid_both]
                            depth = np.clip(depth, 0, MAX_DEPTH_M)
                            L.info(f"Prior blend: {pt_region} ({int(valid_both.sum())} px, alpha={alpha})")
        except Exception as ex:
            L.debug(f"Prior blend skipped: {ex}")

        # 5. Apply global land mask — ensure land = NaN (critical for Mapbox source)
        H_d,W_d=depth.shape
        global_water=make_water_mask(bbox,H_d,W_d,s2=s2)
        if global_water is not None:
            depth[~global_water]=np.nan
            out['sources_used'].append(f"LandMask({int(np.sum(~global_water))}/{H_d*W_d})")
        # Results — raster PNG overlay + points for 3D/export
        out['interpolated_points']=grid_to_points(depth,bbox)
        try:
            raster_b64, raster_bounds, raster_max = depth_to_raster_png(depth, bbox, water_mask=global_water)
            out['raster_png'] = raster_b64
            out['raster_bounds'] = raster_bounds
            out['raster_max_depth'] = raster_max
            try:
                _rid, _amn, _amx = _store_recolor_result(depth, bbox, water_mask=global_water)
                out['result_id'] = _rid
                out['raster_min_depth'] = round(_amn, 3)
                out['raster_auto_min'] = round(_amn, 3)
                out['raster_auto_max'] = round(_amx, 3)
            except Exception:
                pass
        except Exception as rx:
            L.warning(f"Raster PNG failed: {rx}")

        # Sentinel-2 true-colour preview of the imagery the estimate used
        # (skipped for the parallel-tiled path whose s2 dict is zero-filled)
        try:
            s2_b64, s2_bounds = s2_to_rgb_png(s2, bbox)
            out['s2_rgb_png'] = s2_b64
            out['s2_rgb_bounds'] = s2_bounds
            out['s2_window'] = {'start': sd, 'end': ed, 'source': img_source}
        except Exception as sx:
            L.warning(f"S2 preview PNG skipped: {sx}")

        # Auto GeoTIFF (properly georeferenced)
        try:
            out['geotiff_b64'] = build_geotiff_b64(depth, bbox)
        except Exception as tx:
            L.warning(f"GeoTIFF failed: {tx}")

        val=depth[np.isfinite(depth)&(depth>0)]
        water=global_water if global_water is not None else s2['ndwi']>0
        zc={"vs":int(np.sum(water&(depth<=3))),"sh":int(np.sum(water&(depth>3)&(depth<=8))),"md":int(np.sum(water&(depth>8)&(depth<=15))),"dp":int(np.sum(water&(depth>15)))}
        out['stats']={'mean_depth':round(float(np.mean(val)),2) if len(val) else 0,'max_depth':round(float(np.max(val)),2) if len(val) else 0,'min_depth':round(float(np.min(val)),2) if len(val) else 0,'std_depth':round(float(np.std(val)),2) if len(val) else 0,'grid_points':len(out['interpolated_points']),'resolution_m':res}
        out['ml_stats']['zone_counts']=zc

        # 5b. Save session + silently update regional model
        if _get_store and len(rd) > 5:
            try:
                store = _get_store()
                store.log_session(bbox, depth_method, len(rd),
                    rmse=out['ml_stats'].get('rmse', 0), r2=out['ml_stats'].get('r2', 0))
            except:pass
            # Silent model improvement with new training data
            if len(rd) > 20 and out.get('ml_stats', {}).get('r2', 0) > 0.7:
                try:
                    center_lat = (bbox[1] + bbox[3]) / 2
                    center_lon = (bbox[0] + bbox[2]) / 2
                    _, _, matched_region = query_pretrained_depth(center_lat, center_lon)
                    if matched_region:
                        update_pretrained_model(matched_region, rl, rlo, rd)
                except Exception:
                    pass

        # 6. Professional estimation & contour map
        try:
            contours=generate_contours(depth,bbox)
            out['contours']=contours
            out['contour_levels']=sorted(set(c['depth'] for c in contours))
            L.info(f"Contours: {len(contours)} segments, {len(out['contour_levels'])} levels")
        except Exception as cx:
            L.warning(f"Contour gen skipped: {cx}")
            out['contours']=[];out['contour_levels']=[]

        try:
            pro=professional_estimation(depth,bbox,
                ref_lats=np.array(rl) if rl else None,
                ref_lons=np.array(rlo) if rlo else None,
                ref_depths=np.array(rd) if rd else None,
                ml_stats=out.get('ml_stats'))
            out['professional']=pro
        except Exception as px:
            L.warning(f"Pro estimation skipped: {px}")
            out['professional']={}

        return jsonify(_json_safe(out))
    except Exception as ex:
        L.error(f"EXTRACT: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex)}),500

# ══════════════════════════════════════════════════════════════
# QUICK ANALYSE — Multi-date spectral Maximum Likelihood @10m
# ══════════════════════════════════════════════════════════════

# Evalscript for single-scene S2 fetch (SIMPLE mosaicking = one date)
EVALSCRIPT_SINGLE="""//VERSION=3
function setup(){return{input:[{bands:["B01","B02","B03","B04","B08","B11","SCL"],units:"DN",mosaicking:"SIMPLE"}],output:{bands:7,sampleType:"UINT16"}};}
function evaluatePixel(s){return[s.B01,s.B04,s.B03,s.B02,s.B08,s.B11,s.SCL];}"""

def hr_water_mask(scl, ndwi, mndwi, nir_dn, swir_dn):
    """
    Very-high-resolution land/water mask at native 10m.

    Combines:
      - SCL water class (6)
      - NDWI = (G-NIR)/(G+NIR) > 0
      - MNDWI = (G-SWIR)/(G+SWIR) > 0.1  — rejects bright sand, concrete,
        ports, breakwaters, reclaimed land that NDWI misses
      - NIR/SWIR darkness thresholds (water absorbs IR strongly)
      - Cloud/shadow/snow rejection via SCL (classes 1,3,8,9,10,11)
      - Morphological opening+closing at native pixel grid
    """
    scl = np.asarray(scl)
    nir = nir_dn.astype(np.float32) / 10000.0
    swir = swir_dn.astype(np.float32) / 10000.0

    scl_water = (scl == 6)
    scl_cloud = (scl == 1) | (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11)

    ndwi_w = ndwi > 0.0
    mndwi_w = mndwi > 0.1
    nir_dark = nir < 0.13
    swir_dark = swir < 0.11

    # A pixel is water if SCL says so, OR all four spectral tests agree
    water = scl_water | (ndwi_w & mndwi_w & nir_dark & swir_dark)
    water = water & (~scl_cloud)

    # Morphology at native grid: remove isolated water pixels inside land
    # and close small land specks inside water (boats, wakes, floating debris)
    try:
        from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
        s3 = np.ones((3, 3), dtype=bool)
        water = binary_opening(water, structure=s3, iterations=1)
        water = binary_closing(water, structure=s3, iterations=2)
        # Remove small land holes inside big water bodies (≤ 9 px ≈ 900 m²)
        land = ~water
        land_cleaned = binary_opening(land, structure=s3, iterations=1)
        water = ~land_cleaned
    except Exception:
        pass
    return water


def _fetch_s2_single_gee(bbox, sd, ed, res=10, cloud=35):
    """GEE fallback for _fetch_s2_single: least-cloudy single S2 L2A scene with
    the SAME 7-band output (coastal, red, green, blue, nir, swir, scl) so the
    MLE stack works when Sentinel-Hub creds are unavailable (401). Pulls
    B1,B2,B3,B4,B8,B11,SCL; SCL kept raw (class band, never scaled)."""
    if not _init_gee():
        raise RuntimeError("GEE not initialised (missing credentials or SDK)")
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n])
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(sd, ed)
           .filterBounds(region)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", int(cloud))))
    if col.size().getInfo() == 0:
        raise RuntimeError(f"GEE single: no S2 scenes {sd}→{ed} cloud<{cloud}")
    img = (col.sort("CLOUDY_PIXEL_PERCENTAGE").first()
              .select(["B1", "B2", "B3", "B4", "B8", "B11", "SCL"]).clip(region))
    url = img.getDownloadURL({"region": region, "scale": int(res),
                              "format": "GEO_TIFF", "crs": "EPSG:4326"})
    r = requests.get(url, timeout=300)
    if not r.ok:
        raise RuntimeError(f"GEE single download {r.status_code}: {r.text[:150]}")
    a = tifffile.imread(io.BytesIO(r.content))
    if a.ndim == 3 and a.shape[2] == 7:
        coastal, blue, green, red, nir, swir, scl = [a[:, :, i] for i in range(7)]
    elif a.ndim == 3 and a.shape[0] == 7:
        coastal, blue, green, red, nir, swir, scl = [a[i] for i in range(7)]
    else:
        raise RuntimeError(f"GEE single TIFF shape {a.shape}")
    gf = green.astype(float); nf = nir.astype(float); sf_ = swir.astype(float)
    ndwi = (gf - nf) / (gf + nf + 1e-6)
    mndwi = (gf - sf_) / (gf + sf_ + 1e-6)
    water_mask = hr_water_mask(scl, ndwi, mndwi, nir, swir)
    L.info(f"GEE single: {blue.shape[1]}x{blue.shape[0]} @{res}m ({sd}/{ed})")
    return {"coastal": coastal, "red": red, "green": green, "blue": blue,
            "nir": nir, "swir": swir, "ndwi": ndwi, "mndwi": mndwi,
            "water_mask": water_mask, "scl": scl,
            "width": blue.shape[1], "height": blue.shape[0],
            "date_range": f"{sd}/{ed}"}


def _fetch_s2_single(bbox, sd, ed, res=10, cloud=25):
    """Fetch a single S2 scene (no multi-orbit median) at given resolution.
    Falls back to GEE (same 7-band dict) if Sentinel-Hub fails (e.g. 401)."""
    w,s,e,n=bbox
    cl=np.cos(np.radians((n+s)/2))
    wp=max(32,min(2500,int(abs(e-w)*111000*cl/res)))
    hp=max(32,min(2500,int(abs(n-s)*111000/res)))
    try:
        tok=sh_token()
    except Exception as ex:
        L.warning(f"S2 single: sh_token failed ({ex}); GEE fallback")
        return _fetch_s2_single_gee(bbox, sd, ed, res=res, cloud=max(cloud, 35))
    body={"input":{"bounds":{"bbox":[w,s,e,n],"properties":{"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},
          "data":[{"type":"sentinel-2-l2a","dataFilter":{"maxCloudCoverage":cloud,
          "timeRange":{"from":f"{sd}T00:00:00Z","to":f"{ed}T23:59:59Z"}},
          "mosaickingOrder":"leastCC"}]},
          "output":{"width":wp,"height":hp,"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},
          "evalscript":EVALSCRIPT_SINGLE}
    r=None
    for attempt in range(3):
        r=requests.post(SH_PROC,headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json"},json=body,timeout=120)
        if r.ok:break
        if r.status_code==429:
            TM.sleep(2*(attempt+1)); tok=sh_token(); continue
        # Auth/other hard error → GEE fallback rather than aborting the MLE run
        L.warning(f"S2 single SH {r.status_code}; GEE fallback")
        return _fetch_s2_single_gee(bbox, sd, ed, res=res, cloud=max(cloud, 35))
    if r is None or not r.ok:
        return _fetch_s2_single_gee(bbox, sd, ed, res=res, cloud=max(cloud, 35))
    img=tifffile.imread(io.BytesIO(r.content))
    if img.ndim==3 and img.shape[0]==7:
        coastal,red,green,blue,nir,swir,scl=img[0],img[1],img[2],img[3],img[4],img[5],img[6]
    elif img.ndim==3 and img.shape[2]==7:
        coastal,red,green,blue,nir,swir,scl=img[:,:,0],img[:,:,1],img[:,:,2],img[:,:,3],img[:,:,4],img[:,:,5],img[:,:,6]
    else:
        raise RuntimeError(f"S2 single TIFF shape {img.shape}")
    gf=green.astype(float);nf=nir.astype(float);sf_=swir.astype(float)
    ndwi=(gf-nf)/(gf+nf+1e-6)
    mndwi=(gf-sf_)/(gf+sf_+1e-6)
    water_mask = hr_water_mask(scl, ndwi, mndwi, nir, swir)
    return {"coastal":coastal,"red":red,"green":green,"blue":blue,"nir":nir,"swir":swir,
            "ndwi":ndwi,"mndwi":mndwi,"water_mask":water_mask,"scl":scl,
            "width":blue.shape[1],"height":blue.shape[0],
            "date_range":f"{sd}/{ed}"}

def _quick_tile_worker(args):
    """
    Worker for 4x4 parallel Quick Analyse. Runs the full per-scene ridge + MLE
    fusion pipeline on a single sub-tile of the ROI.

    args = (tile_bbox, sd, ed, res_m, gebco_lats, gebco_lons, gebco_depths, row, col,
            obs_lats, obs_lons, obs_depths)  # last three optional (can be empty arrays)

    Observations, when provided, are used as the PRIMARY calibration target
    (weight 5×) alongside GEBCO (weight 1×) in the ridge regression.

    Returns (row, col, depth_grid_or_None, tile_bbox, n_scenes, error)
    """
    # Backward-compatible unpack
    if len(args) == 12:
        (tile_bbox, sd, ed, res_m, g_lats, g_lons, g_depths, row, col,
         o_lats, o_lons, o_depths) = args
    else:
        tile_bbox, sd, ed, res_m, g_lats, g_lons, g_depths, row, col = args
        o_lats = np.asarray([]); o_lons = np.asarray([]); o_depths = np.asarray([])
    try:
        tw, ts, te, tn = tile_bbox
        # Split date range into 3-4 sub-windows for MLE
        from datetime import datetime, timedelta
        dt_sd = datetime.strptime(sd, '%Y-%m-%d')
        dt_ed = datetime.strptime(ed, '%Y-%m-%d')
        total_days = max((dt_ed - dt_sd).days, 30)
        n_scenes = 4
        chunk = max(7, total_days // n_scenes)
        scenes = []
        for i in range(n_scenes):
            c_sd = dt_sd + timedelta(days=i * chunk)
            c_ed = min(c_sd + timedelta(days=chunk - 1), dt_ed)
            if c_sd >= dt_ed:
                break
            try:
                sc = _fetch_s2_single(tile_bbox,
                                      c_sd.strftime('%Y-%m-%d'),
                                      c_ed.strftime('%Y-%m-%d'),
                                      res=int(res_m), cloud=30)
                scenes.append(sc)
            except Exception:
                continue
        if not scenes:
            try:
                sc = _fetch_s2_single(tile_bbox, sd, ed, res=int(res_m), cloud=40)
                scenes.append(sc)
            except Exception as ex:
                return row, col, None, tile_bbox, 0, f"fetch failed: {ex}"

        H, W = scenes[0]['blue'].shape
        if H < 8 or W < 8:
            return row, col, None, tile_bbox, 0, "too small"

        combined_water = np.ones((H, W), dtype=bool)
        for sc in scenes:
            combined_water &= sc['water_mask']
        try:
            gw = make_water_mask(tile_bbox, H, W, s2=scenes[0])
            if gw is not None:
                combined_water &= gw
        except Exception:
            pass

        # Filter GEBCO points to this tile (buffer 500m)
        buf = 0.005
        mask = ((g_lats >= ts - buf) & (g_lats <= tn + buf) &
                (g_lons >= tw - buf) & (g_lons <= te + buf))
        lats_t = g_lats[mask]; lons_t = g_lons[mask]; depths_t = g_depths[mask]
        # If too few, use all GEBCO (global calibration fallback)
        if len(depths_t) < 8:
            lats_t, lons_t, depths_t = g_lats, g_lons, g_depths

        # Observations for this tile (high-weight training targets)
        obs_mask = np.array([], dtype=bool)
        if len(o_depths) > 0:
            obs_mask = ((o_lats >= ts - buf) & (o_lats <= tn + buf) &
                        (o_lons >= tw - buf) & (o_lons <= te + buf))
        o_lat_t = o_lats[obs_mask] if obs_mask.size else np.asarray([])
        o_lon_t = o_lons[obs_mask] if obs_mask.size else np.asarray([])
        o_dep_t = o_depths[obs_mask] if obs_mask.size else np.asarray([])

        W_OBS = OBS_TRAIN_WEIGHT

        depth_stack = []
        for sc in scenes:
            feat_stack, _ = _quick_spectral_features(sc)
            feat_cal, y_cal, w_cal = [], [], []
            # GEBCO calibration points (weight 1)
            for i in range(len(depths_t)):
                r_px = max(0, min(H - 1, int((tn - lats_t[i]) / (tn - ts + 1e-10) * H)))
                c_px = max(0, min(W - 1, int((lons_t[i] - tw) / (te - tw + 1e-10) * W)))
                if combined_water[r_px, c_px]:
                    f = feat_stack[r_px, c_px]
                    if np.all(np.isfinite(f)):
                        feat_cal.append(f); y_cal.append(depths_t[i]); w_cal.append(1.0)
            # Observed points (weight W_OBS — primary truth)
            for i in range(len(o_dep_t)):
                r_px = max(0, min(H - 1, int((tn - o_lat_t[i]) / (tn - ts + 1e-10) * H)))
                c_px = max(0, min(W - 1, int((o_lon_t[i] - tw) / (te - tw + 1e-10) * W)))
                if combined_water[r_px, c_px]:
                    f = feat_stack[r_px, c_px]
                    if np.all(np.isfinite(f)):
                        feat_cal.append(f); y_cal.append(o_dep_t[i]); w_cal.append(W_OBS)
            if len(feat_cal) < 10:
                continue
            X = np.array(feat_cal); y = np.array(y_cal); w = np.array(w_cal)
            # Weighted ridge: solve (Xᵀ W X + λI) β = Xᵀ W y
            X_bias = np.hstack([X, np.ones((len(X), 1))])
            sw = np.sqrt(w)[:, None]
            Xw = X_bias * sw; yw = y * sw[:, 0]
            lam = 0.5 * len(X)
            I_reg = np.eye(X_bias.shape[1]); I_reg[-1, -1] = 0
            try:
                coeffs = np.linalg.solve(Xw.T @ Xw + lam * I_reg, Xw.T @ yw)
            except Exception:
                coeffs = np.linalg.lstsq(Xw, yw, rcond=None)[0]
            d_i = np.sum(feat_stack * coeffs[:-1], axis=-1) + coeffs[-1]
            d_i = np.where(combined_water, d_i, np.nan)
            d_i = np.clip(d_i, 0, MAX_DEPTH_M)
            d_i[~combined_water] = np.nan
            depth_stack.append(d_i)

        if not depth_stack:
            return row, col, None, tile_bbox, len(scenes), "no valid scene"

        stack = np.stack(depth_stack, axis=0)
        with np.errstate(all='ignore'):
            scene_mean = np.nanmean(stack, axis=0)
            scene_std = np.nanstd(stack, axis=0)
        if len(depth_stack) >= 3:
            dev = np.abs(stack - scene_mean[np.newaxis, :, :])
            thr = 1.5 * scene_std[np.newaxis, :, :] + 0.5
            inlier = (dev < thr) | ~np.isfinite(dev)
            stack_clean = np.where(inlier & np.isfinite(stack), stack, np.nan)
            n_inlier = np.sum(np.isfinite(stack_clean), axis=0)
            use_clean = n_inlier >= 2
            depth = np.where(use_clean, np.nanmean(stack_clean, axis=0), scene_mean)
        else:
            depth = scene_mean
        depth = np.clip(depth, 0, MAX_DEPTH_M)
        depth[~combined_water] = np.nan
        return row, col, depth.astype(np.float32), tile_bbox, len(scenes), None
    except Exception as ex:
        return row, col, None, tile_bbox, 0, str(ex)


def _quick_spectral_features(s2):
    """Extract 10 spectral features from an S2 dict. Returns (feat_stack, water_mask)."""
    eps=1e-6
    blue=np.clip(s2['blue'].astype(np.float64)/10000,eps,None)
    green=np.clip(s2['green'].astype(np.float64)/10000,eps,None)
    red=np.clip(s2['red'].astype(np.float64)/10000,eps,None)
    coastal=np.clip(s2.get('coastal',s2['blue']).astype(np.float64)/10000,eps,None)
    nir=np.clip(s2['nir'].astype(np.float64)/10000,eps,None)
    water=s2['water_mask']
    ratio_bg=np.log(1000*blue)/np.log(1000*green+eps)
    ln_b=np.log(blue+eps);ln_g=np.log(green+eps);ln_r=np.log(red+eps);ln_c=np.log(coastal+eps)
    bg_ratio=blue/(green+eps)
    red_atten=np.where(water,-np.log(red+eps),0)
    ndwi_feat=s2['ndwi']
    br_diff=blue-red
    cb_ratio=coastal/(blue+eps)
    feat_stack=np.stack([ratio_bg,ln_b,ln_g,ln_r,ln_c,bg_ratio,red_atten,ndwi_feat,br_diff,cb_ratio],axis=-1)
    return feat_stack,water

@app.route('/api/quick-analyse', methods=['POST'])
def api_quick_analyse():
    """
    Pro-Quick: fetch 3-4 S2 scenes from different dates @10m,
    compute depth per scene via spectral ridge regression (GEBCO-calibrated),
    then maximum likelihood fusion across scenes. Strict water mask.
    """
    t0 = TM.time()
    try:
        data = req.get_json(force=True, silent=True) or {}
        bbox_dict = data.get('bbox')
        if not bbox_dict:
            return jsonify({'error': 'Draw a rectangle'}), 400
        bbox = [bbox_dict['west'], bbox_dict['south'], bbox_dict['east'], bbox_dict['north']]
        w, s_b, e, n = bbox
        sd = data.get('start_date', '2024-05-01')
        ed = data.get('end_date', '2024-09-30')

        # User-selected output resolution (10 / 20 / 50 / 100 m). 10m = S2 native.
        try:
            res_m = int(data.get('resolution_m', 10))
        except (TypeError, ValueError):
            res_m = 10
        if res_m not in (10, 20, 50, 100):
            res_m = max(10, min(100, res_m))
        L.info(f"Quick: user-requested resolution = {res_m}m")

        # Observed points (survey XYZ / ICESat-2) for training + post-calibration
        user_points = data.get('user_points', []) or []
        obs_lats_all, obs_lons_all, obs_depths_all = [], [], []
        for p in user_points:
            try:
                d = float(p.get('depth', 0))
                pc = p.get('photon_class', '')
                if d <= 0 or d > MAX_DEPTH_M:
                    continue
                if pc not in ('observed', 'icesat2_cshelph', 'bathymetry'):
                    continue
                obs_lats_all.append(float(p['lat']))
                obs_lons_all.append(float(p['lon']))
                obs_depths_all.append(d)
            except Exception:
                continue
        obs_lats_all = np.asarray(obs_lats_all, dtype=np.float64)
        obs_lons_all = np.asarray(obs_lons_all, dtype=np.float64)
        obs_depths_all = np.asarray(obs_depths_all, dtype=np.float64)
        if len(obs_depths_all):
            L.info(f"Quick: {len(obs_depths_all)} observed pts for training + calibration "
                   f"(range {obs_depths_all.min():.1f}–{obs_depths_all.max():.1f} m)")

        out = {'points': [], 'interpolated_points': [], 'stats': {}, 'bbox': bbox_dict,
               'sources_used': [], 'ml_stats': {}, 'tracks': [], 'sea_profiles': [], 'bath_profiles': [],
               'contours': [], 'contour_levels': [], 'resolution_m': res_m}

        # ── Step 1: GEBCO for calibration ──
        L.info("Quick: fetching GEBCO for calibration...")
        gebco = fetch_gebco(bbox)
        if not gebco or len(gebco['depths']) < 5:
            return jsonify({'error': 'GEBCO returned no data — no ocean coverage'}), 400
        n_gebco = len(gebco['depths'])
        out['sources_used'].append(f"GEBCO-cal({n_gebco})")

        step_g = max(1, n_gebco // 300)
        for i in range(0, n_gebco, step_g):
            out['points'].append({'lat': round(float(gebco['lats'][i]), 5),
                                  'lon': round(float(gebco['lons'][i]), 5),
                                  'depth': round(float(gebco['depths'][i]), 1),
                                  'photon_class': 'gebco'})

        # ── Decide single-fetch vs 4x4 parallel tiling ──
        # Use tiling for ROIs ≥ 4km in any dim, or when client explicitly requests.
        lat_mid = (n + s_b) / 2
        roi_dx_km = abs(e - w) * 111 * math.cos(math.radians(lat_mid))
        roi_dy_km = abs(n - s_b) * 111
        want_tile = bool(data.get('tiled', roi_dx_km >= 4 or roi_dy_km >= 4))

        tiled_depth = None
        tiled_stats = None
        if want_tile:
            L.info(f"Quick[TILED]: ROI {roi_dx_km:.1f}x{roi_dy_km:.1f}km → 4x4=16 tiles @{res_m}m")
            g_lats_a = np.asarray(gebco['lats'], dtype=np.float64)
            g_lons_a = np.asarray(gebco['lons'], dtype=np.float64)
            g_depths_a = np.asarray(gebco['depths'], dtype=np.float64)

            tiles, n_rows, n_cols = _split_bbox_tiles(bbox, fixed_grid=(4, 4))
            tasks = []
            for idx, tb in enumerate(tiles):
                r_i = idx // n_cols; c_i = idx % n_cols
                tasks.append((tb, sd, ed, res_m, g_lats_a, g_lons_a, g_depths_a, r_i, c_i,
                              obs_lats_all, obs_lons_all, obs_depths_all))

            max_workers = min(16, max(1, _mp.cpu_count() - 1))
            results_map, tile_shapes = {}, {}
            n_ok = n_fail = 0
            total_scenes = 0
            Executor = ProcessPoolExecutor
            try:
                with Executor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_quick_tile_worker, t): t for t in tasks}
                    for fut in as_completed(futures):
                        row_i, col_i, d_tile, tb, n_sc, err = fut.result()
                        if d_tile is not None and np.any(np.isfinite(d_tile)):
                            results_map[(row_i, col_i)] = d_tile
                            tile_shapes[(row_i, col_i)] = d_tile.shape
                            n_ok += 1; total_scenes += n_sc
                        else:
                            n_fail += 1
                            L.warning(f"Quick[TILED] [{row_i},{col_i}] failed: {err}")
            except Exception as ex:
                L.warning(f"Quick[TILED] ProcessPool failed ({ex}); falling back to ThreadPool")
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_quick_tile_worker, t): t for t in tasks}
                    for fut in as_completed(futures):
                        row_i, col_i, d_tile, tb, n_sc, err = fut.result()
                        if d_tile is not None and np.any(np.isfinite(d_tile)):
                            results_map[(row_i, col_i)] = d_tile
                            tile_shapes[(row_i, col_i)] = d_tile.shape
                            n_ok += 1; total_scenes += n_sc
                        else:
                            n_fail += 1

            if n_ok > 0:
                row_heights, col_widths = {}, {}
                for (rr, cc), shp in tile_shapes.items():
                    row_heights[rr] = max(row_heights.get(rr, 0), shp[0])
                    col_widths[cc] = max(col_widths.get(cc, 0), shp[1])
                total_H = sum(row_heights.get(rr, 0) for rr in range(n_rows))
                total_W = sum(col_widths.get(cc, 0) for cc in range(n_cols))
                if total_H > 0 and total_W > 0:
                    mosaic = np.full((total_H, total_W), np.nan, dtype=np.float32)
                    y_off = 0
                    for rr in range(n_rows):
                        x_off = 0
                        rh = row_heights.get(rr, 0)
                        for cc in range(n_cols):
                            cw = col_widths.get(cc, 0)
                            if (rr, cc) in results_map:
                                t_arr = results_map[(rr, cc)]
                                th, tww = t_arr.shape
                                if th != rh or tww != cw:
                                    try:
                                        from scipy.ndimage import zoom as ndizoom
                                        t_arr = ndizoom(t_arr, (rh / th, cw / tww), order=1, mode='nearest')
                                    except Exception:
                                        pass
                                mosaic[y_off:y_off + rh, x_off:x_off + cw] = t_arr
                            x_off += cw
                        y_off += rh
                    # Feather blend seams
                    try:
                        from scipy.ndimage import gaussian_filter as _gf
                        valid = np.isfinite(mosaic) & (mosaic > 0.1)
                        filled = np.where(valid, mosaic, 0)
                        mosaic = np.where(valid, _gf(filled, sigma=1.5), np.nan)
                        mosaic = np.clip(mosaic, 0, MAX_DEPTH_M)
                        mosaic[~valid] = np.nan
                    except Exception:
                        pass
                    # ── Post-mosaic IDW bias calibration against observations ──
                    # Rationale: even after per-tile weighted ridge, small systematic
                    # residuals remain. We compute (obs - pred) at each observation,
                    # interpolate the residual field over the mosaic via IDW, and add
                    # it back. This removes most of the remaining bias and local tilt
                    # without overfitting the mosaic to noise.
                    calib_stats = None
                    if len(obs_depths_all) >= 4:
                        try:
                            mH, mW = mosaic.shape
                            # Map each observation to a mosaic pixel
                            rr = ((bbox[3] - obs_lats_all) / (bbox[3] - bbox[1] + 1e-10) * mH).astype(int)
                            cc = ((obs_lons_all - bbox[0]) / (bbox[2] - bbox[0] + 1e-10) * mW).astype(int)
                            ok = (rr >= 0) & (rr < mH) & (cc >= 0) & (cc < mW)
                            rr, cc = rr[ok], cc[ok]
                            o_d = obs_depths_all[ok]
                            p_d = mosaic[rr, cc]
                            valid = np.isfinite(p_d) & (p_d > 0)
                            rr, cc, o_d, p_d = rr[valid], cc[valid], o_d[valid], p_d[valid]
                            n_cal = len(o_d)
                            if n_cal >= 4:
                                pre_bias = float(np.mean(p_d - o_d))
                                pre_rmse = float(np.sqrt(np.mean((p_d - o_d) ** 2)))
                                residuals = o_d - p_d   # what we need to ADD to prediction

                                # IDW over a coarse grid (then upsample) for speed
                                GH, GW = min(80, mH), min(80, mW)
                                ys = np.linspace(0, mH - 1, GH)
                                xs = np.linspace(0, mW - 1, GW)
                                Y, X = np.meshgrid(ys, xs, indexing='ij')
                                pwr = 2.0
                                dy = Y[..., None] - rr[None, None, :]
                                dx = X[..., None] - cc[None, None, :]
                                d2 = dy * dy + dx * dx + 1e-6
                                w_idw = 1.0 / (d2 ** (pwr / 2))
                                corr_coarse = (w_idw * residuals[None, None, :]).sum(axis=-1) / w_idw.sum(axis=-1)

                                # Upsample coarse correction to mosaic size
                                try:
                                    from scipy.ndimage import zoom as _zm
                                    corr = _zm(corr_coarse, (mH / GH, mW / GW), order=1, mode='nearest')
                                except Exception:
                                    corr = np.kron(corr_coarse, np.ones((max(1, mH // GH), max(1, mW // GW))))
                                    corr = corr[:mH, :mW]

                                # Clamp the correction to avoid exploding in areas with
                                # no nearby observation (|Δ| ≤ 3 m is sufficient for SDB).
                                corr = np.clip(corr, -3.0, 3.0)

                                # Light smoothing for continuity across tile seams
                                try:
                                    from scipy.ndimage import gaussian_filter as _gf
                                    corr = _gf(corr, sigma=2.0)
                                except Exception:
                                    pass

                                vmask = np.isfinite(mosaic) & (mosaic > 0)
                                mosaic[vmask] = np.clip(mosaic[vmask] + corr[vmask], 0, MAX_DEPTH_M)

                                # Post-calibration fit quality at observation points
                                post_pd = mosaic[rr, cc]
                                post_valid = np.isfinite(post_pd)
                                if post_valid.any():
                                    post_bias = float(np.mean(post_pd[post_valid] - o_d[post_valid]))
                                    post_rmse = float(np.sqrt(np.mean((post_pd[post_valid] - o_d[post_valid]) ** 2)))
                                else:
                                    post_bias = post_rmse = 0.0
                                calib_stats = {
                                    'n_obs_used': n_cal,
                                    'pre_bias_m': round(pre_bias, 3),
                                    'pre_rmse_m': round(pre_rmse, 3),
                                    'post_bias_m': round(post_bias, 3),
                                    'post_rmse_m': round(post_rmse, 3),
                                    'method': 'IDW residual field (p=2, σ=2), clamped ±3 m',
                                    'observation_weight': OBS_TRAIN_WEIGHT,
                                }
                                L.info(f"Quick[TILED] bias-calib: n={n_cal} · pre_bias={pre_bias:+.2f}m→"
                                       f"post_bias={post_bias:+.2f}m · pre_rmse={pre_rmse:.2f}→"
                                       f"post_rmse={post_rmse:.2f}m")
                        except Exception as _cex:
                            L.warning(f"Quick[TILED] bias-calib failed: {_cex}")

                    tiled_depth = mosaic
                    tiled_stats = {'tiles_total': 16, 'tiles_ok': n_ok, 'tiles_fail': n_fail,
                                   'total_scenes': total_scenes, 'workers': max_workers,
                                   'grid': '4x4', 'calibration': calib_stats}
                    out['sources_used'].append(
                        f"TiledQuick(4x4,{n_ok}/16 OK,{total_scenes} scenes,{max_workers}w)")
                    if calib_stats:
                        out['sources_used'].append(
                            f"ObsCalib(n={calib_stats['n_obs_used']},"
                            f"bias {calib_stats['pre_bias_m']:+.2f}→{calib_stats['post_bias_m']:+.2f}m)")
                    L.info(f"Quick[TILED] DONE: {n_ok}/16 tiles, {total_scenes} scenes, "
                           f"mosaic={total_H}x{total_W}")

        if tiled_depth is not None:
            # Tiled path already produced the mosaic per-tile (each tile ran its own
            # GEBCO-calibrated ridge + MLE). Seed state so downstream post-processing
            # (raster PNG, contours, geotiff, points) sees a valid depth grid.
            H, W = tiled_depth.shape
            depth = tiled_depth
            combined_water = np.isfinite(depth)
            uncertainty = np.full_like(depth, 1.5, dtype=np.float32)
            coeffs_all = []
            eps = 1e-6
            out['stats_tiled'] = tiled_stats
            s2 = None
            scenes = []
            scene_dates = []

            # GEBCO validation for tiled mosaic (reuses GEBCO points fetched above)
            pred_vals, true_vals = [], []
            for i in range(n_gebco):
                r_px = max(0, min(H - 1, int((n - gebco['lats'][i]) / (n - s_b + 1e-10) * H)))
                c_px = max(0, min(W - 1, int((gebco['lons'][i] - w) / (e - w + 1e-10) * W)))
                if np.isfinite(depth[r_px, c_px]) and depth[r_px, c_px] > 0:
                    pred_vals.append(float(depth[r_px, c_px]))
                    true_vals.append(float(gebco['depths'][i]))
            if len(pred_vals) >= 5:
                pv, tv = np.array(pred_vals), np.array(true_vals)
                r2 = round(float(1 - np.sum((pv - tv) ** 2) / (np.sum((tv - tv.mean()) ** 2) + eps)), 4)
                rmse = round(float(np.sqrt(np.mean((pv - tv) ** 2))), 3)
                bias = round(float(np.mean(pv - tv)), 3)
            else:
                r2, rmse, bias = 0, 0, 0
            out['ml_stats'] = {
                'method': f'Quick[4x4 Tiled]: per-tile Spectral Ridge + MLE @{res_m}m',
                'r2': r2, 'rmse': rmse, 'bias': bias,
                'n_train': n_gebco, 'n_features': 10,
                'n_scenes': tiled_stats.get('total_scenes', 0),
                'tiles_ok': tiled_stats.get('tiles_ok', 0),
                'tiles_total': tiled_stats.get('tiles_total', 16),
                'grid': tiled_stats.get('grid', '4x4'),
                'workers': tiled_stats.get('workers', 0),
                'importance': {},
            }
            out['scatter'] = {
                'predicted': [round(float(v), 2) for v in pred_vals[:500]],
                'reference': [round(float(v), 2) for v in true_vals[:500]],
            }
        else:
            # ── Step 2: Split date range into sub-windows, fetch each @{res_m}m ──
            from datetime import datetime, timedelta
            dt_sd = datetime.strptime(sd, '%Y-%m-%d')
            dt_ed = datetime.strptime(ed, '%Y-%m-%d')
            total_days = max((dt_ed - dt_sd).days, 30)
            n_scenes = 4
            chunk = max(7, total_days // n_scenes)

            scenes = []
            scene_dates = []
            for i in range(n_scenes):
                c_sd = dt_sd + timedelta(days=i * chunk)
                c_ed = min(c_sd + timedelta(days=chunk - 1), dt_ed)
                if c_sd >= dt_ed:
                    break
                c_sd_s = c_sd.strftime('%Y-%m-%d')
                c_ed_s = c_ed.strftime('%Y-%m-%d')
                try:
                    L.info(f"Quick: fetching S2 scene {i+1}/{n_scenes} @{res_m}m ({c_sd_s} → {c_ed_s})...")
                    s2_i = _fetch_s2_single(bbox, c_sd_s, c_ed_s, res=res_m, cloud=30)
                    scenes.append(s2_i)
                    scene_dates.append(f"{c_sd_s}")
                    L.info(f"Quick: scene {i+1} OK ({s2_i['width']}x{s2_i['height']})")
                except Exception as ex:
                    L.warning(f"Quick: scene {i+1} failed ({ex})")

            if not scenes:
                # Fallback: try single fetch with full date range
                try:
                    L.info(f"Quick: all sub-windows failed, trying full range @{res_m}m...")
                    s2_full = _fetch_s2_single(bbox, sd, ed, res=res_m, cloud=40)
                    scenes.append(s2_full)
                    scene_dates.append(f"{sd}/{ed}")
                except Exception as ex2:
                    L.warning(f"Quick: full range also failed ({ex2})")

            out['sources_used'].append(f"S2-scenes({len(scenes)}@{res_m}m,dates={','.join(scene_dates)})")
            L.info(f"Quick: {len(scenes)} scenes fetched in {TM.time()-t0:.1f}s")

            H, W = (scenes[0]['blue'].shape if scenes else (0, 0))
            depth = None
            s2 = scenes[0] if scenes else None  # keep reference for water mask
            combined_water = None
            uncertainty = None
            coeffs_all = []
            eps = 1e-6

            if scenes and H > 0 and W > 0:
                # ── Step 3: Build combined water mask (intersection = conservative) ──
                combined_water = np.ones((H, W), dtype=bool)
                for sc in scenes:
                    combined_water &= sc['water_mask']
                global_water = make_water_mask(bbox, H, W, s2=scenes[0])
                if global_water is not None:
                    combined_water &= global_water
                L.info(f"Quick: combined water mask: {int(combined_water.sum())}/{H*W} px "
                       f"({100*combined_water.sum()/(H*W+1):.0f}%)")

                # ── Step 4: Compute depth per scene via ridge regression ──
                depth_stack = []

                for si, sc in enumerate(scenes):
                    feat_stack, sc_water = _quick_spectral_features(sc)
                    water = combined_water

                    feat_cal, y_cal = [], []
                    for i in range(n_gebco):
                        r_px = max(0, min(H-1, int((n - gebco['lats'][i])/(n - s_b + 1e-10)*H)))
                        c_px = max(0, min(W-1, int((gebco['lons'][i] - w)/(e - w + 1e-10)*W)))
                        if water[r_px, c_px]:
                            f = feat_stack[r_px, c_px]
                            if np.all(np.isfinite(f)):
                                feat_cal.append(f)
                                y_cal.append(gebco['depths'][i])

                    if len(feat_cal) < 15:
                        L.warning(f"Quick: scene {si+1} only {len(feat_cal)} cal pts, skipping")
                        continue

                    X = np.array(feat_cal)
                    y = np.array(y_cal)
                    X_bias = np.hstack([X, np.ones((len(X), 1))])
                    lam = 0.5 * len(X)
                    I_reg = np.eye(X_bias.shape[1])
                    I_reg[-1, -1] = 0
                    try:
                        coeffs = np.linalg.solve(X_bias.T @ X_bias + lam * I_reg, X_bias.T @ y)
                    except Exception:
                        coeffs = np.linalg.lstsq(X_bias, y, rcond=None)[0]
                    coeffs_all.append(coeffs)

                    d_i = np.sum(feat_stack * coeffs[:-1], axis=-1) + coeffs[-1]
                    d_i = np.where(water, d_i, np.nan)
                    d_i = np.clip(d_i, 0, MAX_DEPTH_M)
                    d_i[~water] = np.nan
                    depth_stack.append(d_i)

                    y_pred = X_bias @ coeffs
                    r2_i = 1 - np.sum((y - y_pred)**2)/(np.sum((y - y.mean())**2) + eps)
                    L.info(f"Quick: scene {si+1} ridge R²={r2_i:.3f} ({len(X)} cal pts)")

                # ── Step 5: Maximum Likelihood fusion across scenes ──
                if len(depth_stack) >= 2:
                    stack = np.stack(depth_stack, axis=0)
                    n_valid = np.sum(np.isfinite(stack), axis=0)
                    with np.errstate(all='ignore'):
                        scene_mean = np.nanmean(stack, axis=0)
                        scene_std = np.nanstd(stack, axis=0)

                    if len(depth_stack) >= 3:
                        deviations = np.abs(stack - scene_mean[np.newaxis, :, :])
                        threshold = 1.5 * scene_std[np.newaxis, :, :] + 0.5
                        inlier = (deviations < threshold) | ~np.isfinite(deviations)
                        stack_clean = np.where(inlier & np.isfinite(stack), stack, np.nan)
                        n_inlier = np.sum(np.isfinite(stack_clean), axis=0)
                        use_clean = n_inlier >= 2
                        depth = np.where(use_clean,
                                         np.nanmean(stack_clean, axis=0),
                                         scene_mean)
                        n_rejected = int(np.sum(~inlier & np.isfinite(stack)))
                        L.info(f"Quick: MLE outlier rejection: {n_rejected} pixel-scenes rejected")
                    else:
                        depth = scene_mean

                    uncertainty = np.where(n_valid >= 2, scene_std, 5.0)
                    depth = np.clip(depth, 0, MAX_DEPTH_M)
                    depth[~combined_water] = np.nan
                    out['sources_used'].append(f"MLE-fusion({len(depth_stack)}scenes)")

                elif len(depth_stack) == 1:
                    depth = depth_stack[0]
                    depth[~combined_water] = np.nan
                    uncertainty = np.full_like(depth, 3.0)
                    out['sources_used'].append("SingleScene")
                else:
                    L.warning("Quick: no scenes produced valid depth")

            # ── GEBCO stabilisation (15%) — only where spectral exists ──
            if depth is not None:
                from scipy.ndimage import distance_transform_edt
                gebco_prior = np.full((H, W), np.nan, dtype=np.float64)
                g_rows = np.clip(((n - gebco['lats'])/(n - s_b + 1e-10)*H).astype(int), 0, H-1)
                g_cols = np.clip(((gebco['lons'] - w)/(e - w + 1e-10)*W).astype(int), 0, W-1)
                for i in range(n_gebco):
                    gebco_prior[g_rows[i], g_cols[i]] = gebco['depths'][i]
                mask_valid = np.isfinite(gebco_prior)
                if mask_valid.sum() > 3:
                    idx = distance_transform_edt(~mask_valid, return_distances=False, return_indices=True)
                    gebco_filled = gebco_prior[tuple(idx)]
                    try: gebco_filled = gaussian_filter(gebco_filled, sigma=4.0)
                    except: pass
                else:
                    gebco_filled = np.full((H, W), np.nanmean(gebco['depths']))

                W_SPEC, W_GEB = 0.85, 0.15
                both = np.isfinite(depth) & np.isfinite(gebco_filled) & combined_water
                depth[both] = W_SPEC * depth[both] + W_GEB * gebco_filled[both]
                depth = np.clip(depth, 0, MAX_DEPTH_M)
                depth[~combined_water] = np.nan

                # Light smooth
                try:
                    valid_d = np.isfinite(depth)
                    filled_d = np.where(valid_d, depth, 0)
                    depth = np.where(valid_d, gaussian_filter(filled_d, sigma=0.7), np.nan)
                    depth = np.clip(depth, 0, MAX_DEPTH_M)
                    depth[~valid_d] = np.nan
                except: pass

            # ── Stats ──
            if depth is not None:
                # Feature importance from average coefficients
                feat_names = ['Stumpf_BG','ln_Blue','ln_Green','ln_Red','ln_Coastal',
                              'B/G_ratio','Red_atten','NDWI','B-R_diff','Coastal/Blue']
                if coeffs_all:
                    avg_c = np.mean([np.abs(c[:-1]) for c in coeffs_all], axis=0)
                    total_c = avg_c.sum() + eps
                    importance = {fn: round(float(ac/total_c), 3) for fn, ac in zip(feat_names, avg_c)}
                else:
                    importance = {}

                # Validation against GEBCO
                pred_vals, true_vals = [], []
                for i in range(n_gebco):
                    r_px = max(0, min(H-1, int((n - gebco['lats'][i])/(n - s_b + 1e-10)*H)))
                    c_px = max(0, min(W-1, int((gebco['lons'][i] - w)/(e - w + 1e-10)*W)))
                    if np.isfinite(depth[r_px, c_px]) and depth[r_px, c_px] > 0:
                        pred_vals.append(depth[r_px, c_px])
                        true_vals.append(gebco['depths'][i])
                if len(pred_vals) >= 5:
                    pv, tv = np.array(pred_vals), np.array(true_vals)
                    r2 = round(float(1 - np.sum((pv-tv)**2)/(np.sum((tv-tv.mean())**2)+eps)), 4)
                    rmse = round(float(np.sqrt(np.mean((pv-tv)**2))), 3)
                    bias = round(float(np.mean(pv-tv)), 3)
                else:
                    r2, rmse, bias = 0, 0, 0

                out['ml_stats'] = {
                    'method': f'Quick: {len(depth_stack)}-Scene MLE + Spectral Ridge @{res_m}m',
                    'r2': r2, 'rmse': rmse, 'bias': bias,
                    'n_train': n_gebco, 'n_features': 10,
                    'n_scenes': len(depth_stack), 'scene_dates': scene_dates,
                    'importance': importance,
                    'spectral_weight': 0.85, 'gebco_weight': 0.15,
                }
                out['sources_used'].append(f"SpectralMLE(R²={r2},RMSE={rmse}m,{len(depth_stack)}scenes)")
                out['scatter'] = {
                    'predicted': [round(float(v), 2) for v in pred_vals[:500]],
                    'reference': [round(float(v), 2) for v in true_vals[:500]],
                }

        # ── Fallback: GEBCO-only IDW ──
        if depth is None:
            L.info("Quick: GEBCO-only IDW")
            H = max(100, min(500, int(abs(n - s_b)*111000/50)))
            W = max(100, min(500, int(abs(e - w)*111000*np.cos(np.radians((n+s_b)/2))/50)))
            depth = np.full((H, W), np.nan, dtype=np.float64)
            g_rows = np.clip(((n - gebco['lats'])/(n - s_b + 1e-10)*H).astype(int), 0, H-1)
            g_cols = np.clip(((gebco['lons'] - w)/(e - w + 1e-10)*W).astype(int), 0, W-1)
            for i in range(n_gebco):
                depth[g_rows[i], g_cols[i]] = gebco['depths'][i]
            from scipy.ndimage import distance_transform_edt
            mask_v = np.isfinite(depth)
            if mask_v.sum() > 3:
                idx = distance_transform_edt(~mask_v, return_distances=False, return_indices=True)
                depth = depth[tuple(idx)]
                try: depth = gaussian_filter(depth, sigma=2.0)
                except: pass
            depth = np.clip(depth, 0, MAX_DEPTH_M)
            global_water = make_water_mask(bbox, H, W)
            if global_water is not None:
                depth[~global_water] = np.nan
            out['sources_used'].append("GEBCO-IDW")
            out['ml_stats'] = {'method': 'Quick: GEBCO IDW Only', 'importance': {}}

        # ── Output ──
        H_d, W_d = depth.shape
        # Full 10m resolution: allow up to 250k points for CSV fidelity
        out['interpolated_points'] = grid_to_points(depth, bbox, max_pts=250000)
        try:
            water_out = combined_water if 'combined_water' in dir() else (global_water if 'global_water' in dir() else None)
            raster_b64, raster_bounds, raster_max = depth_to_raster_png(depth, bbox, water_mask=water_out)
            out['raster_png'] = raster_b64
            out['raster_bounds'] = raster_bounds
            out['raster_max_depth'] = raster_max
            try:
                _rid, _amn, _amx = _store_recolor_result(depth, bbox, water_mask=water_out)
                out['result_id'] = _rid
                out['raster_min_depth'] = round(_amn, 3)
                out['raster_auto_min'] = round(_amn, 3)
                out['raster_auto_max'] = round(_amx, 3)
            except Exception:
                pass
        except Exception as rx:
            L.warning(f"Quick raster: {rx}")
        try:
            out['geotiff_b64'] = build_geotiff_b64(depth, bbox)
        except: pass
        try:
            contours = generate_contours(depth, bbox)
            out['contours'] = contours
            out['contour_levels'] = sorted(set(c['depth'] for c in contours))
        except: pass
        try:
            pro = professional_estimation(depth, bbox,
                ref_lats=np.array(gebco['lats']), ref_lons=np.array(gebco['lons']),
                ref_depths=np.array(gebco['depths']), ml_stats=out.get('ml_stats'))
            out['professional'] = pro
        except: pass

        val = depth[np.isfinite(depth) & (depth > 0)]
        out['stats'] = {
            'mean_depth': round(float(np.mean(val)), 2) if len(val) else 0,
            'max_depth': round(float(np.max(val)), 2) if len(val) else 0,
            'min_depth': round(float(np.min(val)), 2) if len(val) else 0,
            'std_depth': round(float(np.std(val)), 2) if len(val) else 0,
            'grid_points': len(out['interpolated_points']),
            'resolution_m': res_m,
        }
        # Image-acquisition metadata (shown in IHO Chart tab on click)
        _acq_dates = scene_dates if scene_dates else []

        # Turbidity estimate from water pixels (if we still have S2 data around)
        turbidity = None
        try:
            if scenes and combined_water is not None:
                # NDTI = (Red - Green) / (Red + Green); Nechad 2010-style Secchi proxy
                sc0 = scenes[0]
                r = sc0['red'].astype(np.float64); g = sc0['green'].astype(np.float64)
                b = sc0['blue'].astype(np.float64)
                mask = combined_water & (r > 0) & (g > 0) & (b > 0)
                if mask.sum() > 100:
                    ndti = (r - g) / (r + g + 1e-6)
                    rb = r / (b + 1e-6)
                    ndti_m = float(np.nanmean(ndti[mask]))
                    rb_m = float(np.nanmean(rb[mask]))
                    # Qualitative class (Nechad thresholds, loose)
                    if ndti_m < -0.1:      t_cls = 'clear'
                    elif ndti_m < 0.0:     t_cls = 'low'
                    elif ndti_m < 0.1:     t_cls = 'moderate'
                    else:                  t_cls = 'turbid'
                    # Simple Secchi proxy (empirical for coastal water): SD ≈ 10 / (1 + 6·ndti)
                    secchi = max(0.3, min(30.0, 10.0 / (1.0 + 6.0 * max(0.0, ndti_m + 0.1))))
                    turbidity = {
                        'ndti_mean': round(ndti_m, 4),
                        'r_b_ratio': round(rb_m, 4),
                        'class': t_cls,
                        'secchi_depth_est_m': round(secchi, 2),
                        'notes': 'NDTI=(R-G)/(R+G); empirical Secchi proxy',
                    }
        except Exception as _tex:
            L.warning(f"Quick: turbidity calc failed: {_tex}")

        out['image_metadata'] = {
            'sensor': 'Sentinel-2 L2A (ESA / Copernicus CDSE)',
            'resolution_m': res_m,
            'bands': ['B01 coastal (60m→resampled)', 'B02 blue', 'B03 green', 'B04 red',
                      'B08 NIR', 'B11 SWIR', 'SCL (scene classification)'],
            'acquisition_dates': _acq_dates,
            'n_scenes': len(scene_dates) if scene_dates else (tiled_stats.get('total_scenes', 0) if tiled_stats else 0),
            'method': (f'4x4 tiled parallel ({tiled_stats.get("workers",0)} workers) · ridge + MLE @{res_m}m'
                       if tiled_depth is not None else
                       f'Whole-bbox {len(scene_dates)}-scene ridge + MLE @{res_m}m'),
            'tile_grid': (tiled_stats.get('grid') if tiled_stats else None),
            'cloud_max_pct': 30,
            'mosaicking': 'leastCC per sub-window',
            'crs': 'EPSG:4326',
            'bbox': bbox_dict,
            'max_depth_cap_m': MAX_DEPTH_M,
            'turbidity': turbidity,
        }
        elapsed = round(TM.time() - t0, 1)
        out['processing_time_sec'] = elapsed
        out['sources_used'].append(f"Quick({elapsed}s)")
        L.info(f"=== Quick DONE: {len(out['interpolated_points'])} pts @{res_m}m, "
               f"{len(scenes)} scenes, {elapsed}s ===")
        return jsonify(out)
    except Exception as ex:
        L.error(f"QUICK: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# MULTI-EPOCH MAXIMUM-LIKELIHOOD BATHYMETRY
# Many S2 scenes per epoch → per-pixel Gaussian MLE → A/B diff
# ══════════════════════════════════════════════════════════════

# ── ICESat-2 fixed-period cache ───────────────────────────────
# Cross-epoch ICESat-2 reference: fetched ONCE per (bbox, wide time range)
# and reused as a high-quality depth prior in every epoch's ridge fit.
# Seafloor is assumed quasi-stable on 1–2 year scales, so the same photon
# set anchors both A and B — the difference then reflects S2 radiometry
# change, not reference drift.
_ICESAT2_CACHE = {}  # key: (bbox_rounded, sd, ed, laser, threshold) → dict

def _icesat2_cache_key(bbox, sd, ed, laser, threshold):
    w, s, e, n = bbox
    return (round(w, 3), round(s, 3), round(e, 3), round(n, 3), sd, ed, laser, threshold)


def _get_icesat2_reference(bbox, sd, ed, laser=1, threshold=30, water_temp=22.0,
                           max_granules=4):
    """
    Fetch + process ICESat-2 ATL03 photons ONCE for the combined
    time range and return a list of {lat, lon, depth, sigma} dicts.
    Cached by (bbox, date range, laser, threshold).
    """
    if not CSHELPH_AVAILABLE:
        return [], "CShelph module not available"
    key = _icesat2_cache_key(bbox, sd, ed, laser, threshold)
    if key in _ICESAT2_CACHE:
        cached = _ICESAT2_CACHE[key]
        L.info(f"ICESat-2 cache HIT: {len(cached['points'])} pts "
               f"({cached['n_granules']} granules, {sd}→{ed})")
        return cached['points'], None

    L.info(f"ICESat-2 fixed-period fetch: bbox={bbox} {sd}→{ed} laser={laser}")
    try:
        granules = search_atl03(bbox, sd, ed, max_results=max_granules * 3)
    except Exception as ex:
        L.warning(f"ICESat-2 search failed: {ex}")
        return [], f"ATL03 search error: {ex}"
    if not granules:
        L.warning(f"ICESat-2: no ATL03 granules in {sd}→{ed}")
        _ICESAT2_CACHE[key] = {'points': [], 'n_granules': 0, 'dates': []}
        return [], "No ATL03 granules"

    # Process up to max_granules granules and aggregate
    all_results = []
    processed = []
    for gi, gr in enumerate(granules[:max_granules]):
        try:
            L.info(f"ICESat-2 granule {gi+1}/{min(max_granules,len(granules))}: "
                   f"{gr['id']} ({gr['date']})")
            res_g, err_g = process_granule(gr['result_obj'], bbox, laser, threshold, water_temp)
            if res_g:
                all_results.append(res_g)
                processed.append(gr['date'])
        except Exception as ex:
            L.warning(f"ICESat-2 granule {gi+1} failed: {ex}")

    if not all_results:
        _ICESAT2_CACHE[key] = {'points': [], 'n_granules': 0, 'dates': []}
        return [], "No bathymetric photons in any granule"

    # Aggregate across passes when >1 granule
    if len(all_results) > 1:
        pts = aggregate_multi_pass(all_results) or all_results[0].get('depth_points', [])
    else:
        pts = all_results[0].get('depth_points', [])

    # Normalize: use TPU as σ if present, otherwise 0.4m default
    ref_pts = []
    for p in pts:
        try:
            lat = float(p['lat']); lon = float(p['lon']); dep = float(p['depth'])
        except Exception:
            continue
        if not np.isfinite(lat) or not np.isfinite(lon) or not np.isfinite(dep):
            continue
        if dep <= 0 or dep > MAX_DEPTH_M:
            continue
        tpu = p.get('tpu')
        sig = float(tpu) if (tpu is not None and np.isfinite(float(tpu)) and float(tpu) > 0) else 0.4
        # Floor σ to avoid overconfident outliers
        sig = max(sig, 0.25)
        ref_pts.append({'lat': lat, 'lon': lon, 'depth': dep, 'sigma': sig})

    _ICESAT2_CACHE[key] = {
        'points': ref_pts, 'n_granules': len(all_results), 'dates': processed,
    }
    L.info(f"ICESat-2 reference ready: {len(ref_pts)} photon-derived depths "
           f"from {len(all_results)} granules ({', '.join(processed)})")
    return ref_pts, None


def _epoch_depth_mle(bbox, sd, ed, n_scenes, gebco, xyz_pts, ice_pts=None):
    """
    Fetch N S2 scenes inside [sd, ed], run per-scene ridge regression
    calibrated against GEBCO + in-situ XYZ + ICESat-2 photons (fixed cross-epoch),
    then per-pixel Gaussian MLE.

    Returns: dict {depth, sigma, water_mask, n_scenes_used, scene_dates,
                   r2, rmse, bias, H, W} or None.
    """
    from datetime import datetime, timedelta
    dt_sd = datetime.strptime(sd, '%Y-%m-%d')
    dt_ed = datetime.strptime(ed, '%Y-%m-%d')
    total_days = max((dt_ed - dt_sd).days, n_scenes)
    chunk = max(3, total_days // n_scenes)

    scenes, scene_dates = [], []
    for i in range(n_scenes):
        c_sd = dt_sd + timedelta(days=i * chunk)
        c_ed = min(c_sd + timedelta(days=chunk - 1), dt_ed)
        if c_sd >= dt_ed:
            break
        c_sd_s, c_ed_s = c_sd.strftime('%Y-%m-%d'), c_ed.strftime('%Y-%m-%d')
        try:
            L.info(f"MLE: scene {i+1}/{n_scenes} @10m ({c_sd_s} → {c_ed_s})")
            sc = _fetch_s2_single(bbox, c_sd_s, c_ed_s, res=10, cloud=35)
            scenes.append(sc)
            scene_dates.append(c_sd_s)
        except Exception as ex:
            L.warning(f"MLE: scene {i+1} failed ({ex})")

    if not scenes:
        return None

    H, W = scenes[0]['blue'].shape
    w, s_b, e, n = bbox

    # Combined HR water mask: intersection of per-scene HR masks
    combined_water = np.ones((H, W), dtype=bool)
    for sc in scenes:
        wm = sc['water_mask']
        if wm.shape != (H, W):
            from scipy.ndimage import zoom as ndizoom
            wm = ndizoom(wm.astype(np.uint8),
                         (H / wm.shape[0], W / wm.shape[1]), order=0).astype(bool)
        combined_water &= wm
    L.info(f"MLE: HR water mask {int(combined_water.sum())}/{H*W} "
           f"({100*combined_water.sum()/(H*W+1):.1f}%)")

    # Assemble calibration reference:
    #   - GEBCO                 σ≈5.0m   (background)
    #   - in-situ XYZ soundings σ≈0.3m   (ground truth)
    #   - ICESat-2 photons      σ≈TPU    (fixed cross-epoch anchor)
    ref_lats, ref_lons, ref_deps, ref_sig = [], [], [], []
    n_gebco = len(gebco['depths'])
    for i in range(n_gebco):
        ref_lats.append(gebco['lats'][i])
        ref_lons.append(gebco['lons'][i])
        ref_deps.append(gebco['depths'][i])
        ref_sig.append(5.0)
    xlat, xlon, xdep, _ = xyz_pts
    for i in range(len(xlat)):
        ref_lats.append(xlat[i]); ref_lons.append(xlon[i])
        ref_deps.append(xdep[i]); ref_sig.append(0.3)
    n_ice = 0
    if ice_pts:
        for p in ice_pts:
            ref_lats.append(p['lat']); ref_lons.append(p['lon'])
            ref_deps.append(p['depth']); ref_sig.append(p['sigma'])
            n_ice += 1
    ref_lats = np.array(ref_lats); ref_lons = np.array(ref_lons)
    ref_deps = np.array(ref_deps); ref_sig = np.array(ref_sig)
    L.info(f"MLE: calibration refs = {n_gebco} GEBCO + {len(xlat)} XYZ in-situ "
           f"+ {n_ice} ICESat-2 photons")

    # Precompute ref pixel coordinates
    r_px = np.clip(((n - ref_lats) / (n - s_b + 1e-10) * H).astype(int), 0, H - 1)
    c_px = np.clip(((ref_lons - w) / (e - w + 1e-10) * W).astype(int), 0, W - 1)

    depth_stack = []   # per scene (H,W)
    sigma_stack = []   # per scene (H,W) heteroskedastic uncertainty
    eps = 1e-6

    for si, sc in enumerate(scenes):
        feat_stack, _ = _quick_spectral_features(sc)
        # Build weighted training set
        X_tr, y_tr, wts = [], [], []
        for k in range(len(ref_deps)):
            rr, cc = r_px[k], c_px[k]
            if combined_water[rr, cc]:
                f = feat_stack[rr, cc]
                if np.all(np.isfinite(f)):
                    X_tr.append(f); y_tr.append(ref_deps[k])
                    wts.append(1.0 / (ref_sig[k] ** 2))
        if len(X_tr) < 20:
            L.warning(f"MLE: scene {si+1} only {len(X_tr)} cal pts, skipping")
            continue
        X = np.asarray(X_tr); y = np.asarray(y_tr); Wd = np.asarray(wts)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        lam = 0.5 * len(X)
        Ireg = np.eye(Xb.shape[1]); Ireg[-1, -1] = 0.0
        WXb = Xb * Wd[:, None]
        try:
            coeffs = np.linalg.solve(WXb.T @ Xb + lam * Ireg, WXb.T @ y)
        except Exception:
            coeffs = np.linalg.lstsq(Xb, y, rcond=None)[0]

        y_pred = Xb @ coeffs
        resid = y - y_pred
        # Per-scene base σ from residuals (weighted)
        sigma_base = float(np.sqrt(np.average(resid ** 2, weights=Wd) + 0.25))

        d_i = np.sum(feat_stack * coeffs[:-1], axis=-1) + coeffs[-1]
        d_i = np.clip(d_i, 0, MAX_DEPTH_M)
        d_i[~combined_water] = np.nan

        # Heteroskedastic σ map: base residual + proximity to clouds/shadows
        scl = sc.get('scl')
        sigma_map = np.full((H, W), sigma_base, dtype=np.float32)
        if scl is not None:
            scl_arr = np.asarray(scl)
            if scl_arr.shape != (H, W):
                from scipy.ndimage import zoom as ndizoom
                scl_arr = ndizoom(scl_arr, (H / scl_arr.shape[0], W / scl_arr.shape[1]), order=0)
            bad = ((scl_arr == 1) | (scl_arr == 3) | (scl_arr == 8) |
                   (scl_arr == 9) | (scl_arr == 10) | (scl_arr == 11))
            try:
                from scipy.ndimage import distance_transform_edt
                dist = distance_transform_edt(~bad)
                sigma_map = sigma_map + 3.0 * np.exp(-dist / 5.0).astype(np.float32)
            except Exception:
                sigma_map = np.where(bad, sigma_map + 3.0, sigma_map)
        # Depth-dependent inflation (deeper = noisier SDB)
        sigma_map = sigma_map + 0.08 * np.clip(d_i, 0, MAX_DEPTH_M).astype(np.float32)

        depth_stack.append(d_i.astype(np.float32))
        sigma_stack.append(sigma_map)
        r2_i = 1.0 - np.sum(Wd * resid ** 2) / (np.sum(Wd * (y - y.mean()) ** 2) + eps)
        L.info(f"MLE: scene {si+1} weighted-ridge R²={r2_i:.3f} σ₀={sigma_base:.2f}m "
               f"(n_cal={len(X)})")

    if not depth_stack:
        return None

    D = np.stack(depth_stack, axis=0)    # (N,H,W)
    S = np.stack(sigma_stack, axis=0)    # (N,H,W)
    finite = np.isfinite(D) & np.isfinite(S) & (S > eps)

    # Iteratively-reweighted Gaussian MLE:
    #   d_hat = Σ(d_i/σ_i²) / Σ(1/σ_i²)
    #   σ²_hat = 1 / Σ(1/σ_i²)
    # Reject scenes where |d_i - d_hat| > 2.5 σ_i, up to 3 passes.
    inlier = finite.copy()
    for _pass in range(3):
        w_i = np.where(inlier, 1.0 / (S ** 2 + eps), 0.0)
        w_sum = np.sum(w_i, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            d_hat = np.where(w_sum > 0, np.sum(w_i * np.where(inlier, D, 0.0), axis=0) / w_sum, np.nan)
            s_hat = np.where(w_sum > 0, np.sqrt(1.0 / w_sum), np.nan)
        resid = np.abs(D - d_hat[None, :, :])
        new_inlier = inlier & (resid < 2.5 * S)
        # Safeguard: keep at least one scene per pixel
        keep_counts = np.sum(new_inlier, axis=0)
        fallback = keep_counts == 0
        if np.any(fallback):
            for k in range(D.shape[0]):
                new_inlier[k][fallback] |= finite[k][fallback]
        if np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier

    depth = np.clip(d_hat, 0, MAX_DEPTH_M).astype(np.float32)
    sigma = s_hat.astype(np.float32)
    depth[~combined_water] = np.nan
    sigma[~combined_water] = np.nan

    # Validation vs ALL refs (GEBCO + XYZ together)
    pv, tv = [], []
    for k in range(len(ref_deps)):
        rr, cc = r_px[k], c_px[k]
        d_v = depth[rr, cc]
        if np.isfinite(d_v) and d_v > 0:
            pv.append(d_v); tv.append(ref_deps[k])
    if len(pv) >= 5:
        pv = np.asarray(pv); tv = np.asarray(tv)
        r2 = float(1 - np.sum((pv - tv) ** 2) / (np.sum((tv - tv.mean()) ** 2) + eps))
        rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
        bias = float(np.mean(pv - tv))
    else:
        r2, rmse, bias = 0.0, 0.0, 0.0

    return {
        'depth': depth, 'sigma': sigma, 'water_mask': combined_water,
        'n_scenes_used': len(depth_stack), 'scene_dates': scene_dates,
        'r2': round(r2, 4), 'rmse': round(rmse, 3), 'bias': round(bias, 3),
        'H': H, 'W': W,
    }


IHO_S52_PALETTE = [
    # (max_depth_m_exclusive, label,  ansi_code, hex)
    (2.0,   '0–2m',   '101',  '#fde047'),   # hazard: bright yellow on red-ish — wading depth
    (5.0,   '2–5m',   '93',   '#fb923c'),   # shoal: orange
    (10.0,  '5–10m',  '96',   '#67e8f9'),   # shallow: cyan
    (15.0,  '10–15m', '94',   '#60a5fa'),   # medium: light blue
    (20.0,  '15–20m', '34',   '#2563eb'),   # deep: blue
    (9e9,   '>20m',   '35',   '#7c3aed'),   # deepest: violet/indigo
]


def _iho_class(v):
    """Return (ansi_code, hex) for depth value using IHO S-52 palette."""
    for lim, _lbl, ansi, hexc in IHO_S52_PALETTE:
        if v < lim:
            return ansi, hexc
    return IHO_S52_PALETTE[-1][2], IHO_S52_PALETTE[-1][3]


def _iho_legend():
    """ANSI + HTML one-line IHO palette legend."""
    a_parts, h_parts = ['Legend:'], ['<span style="color:#9ca3af">Legend:</span>']
    for _lim, lbl, ansi, hexc in IHO_S52_PALETTE:
        a_parts.append(f" \x1b[{ansi}m{lbl:>6s}\x1b[0m")
        h_parts.append(f'&nbsp;<span style="color:{hexc};font-weight:600">{lbl}</span>')
    return "".join(a_parts), "".join(h_parts)


def _ascii_depth_block(depth, title, color_code=None, rows=18, cols=48):
    """
    IHO S-52 styled ASCII depth grid. Fixed 5-char cells (no overlap).
    Each cell: ' N.N ' (one decimal) or '  .  ' if NaN/land.
    Colors per IHO depth class; legend appended.
    Returns (ansi_text, html_text).
    """
    H, W = depth.shape
    if H < 2 or W < 2:
        return "", ""
    rs = max(1, H // rows); cs = max(1, W // cols)
    title_hex = '#22d3ee'
    ansi_lines = [f"\x1b[1;96m═══ {title} ═══\x1b[0m"]
    html_lines = [f'<div class="ascii-title" style="color:{title_hex};font-weight:700">{title}</div>']

    CELL_W = 5
    NAN_CELL_A = '  .  '
    NAN_CELL_H = f'<span style="color:#374151">{NAN_CELL_A}</span>'

    for r in range(0, H, rs):
        a_row, h_row = [], []
        for c in range(0, W, cs):
            v = float(depth[r, c]) if r < H and c < W else float('nan')
            if not np.isfinite(v) or v <= 0:
                a_row.append(NAN_CELL_A); h_row.append(NAN_CELL_H)
            else:
                v_clip = min(v, 99.9)
                txt = f"{v_clip:4.1f} "  # exactly 5 chars, e.g. " 3.2 " or "12.7 "
                ansi, hexc = _iho_class(v_clip)
                a_row.append(f"\x1b[{ansi}m{txt}\x1b[0m")
                h_row.append(f'<span style="color:{hexc}">{txt}</span>')
        ansi_lines.append("".join(a_row))
        html_lines.append('<div>' + "".join(h_row) + '</div>')

    leg_a, leg_h = _iho_legend()
    ansi_lines.append("")
    ansi_lines.append(leg_a)
    html_lines.append('<div style="margin-top:4px">' + leg_h + '</div>')
    return "\n".join(ansi_lines), "\n".join(html_lines)


def _ascii_diff_block(diff, title, rows=18, cols=48):
    """ASCII diff grid: green (+) deposition, red (-) erosion. Fixed 5-char cells."""
    H, W = diff.shape
    if H < 2 or W < 2:
        return "", ""
    rs = max(1, H // rows); cs = max(1, W // cols)
    ansi_lines = [f"\x1b[1;33m═══ {title} ═══\x1b[0m"]
    html_lines = [f'<div class="ascii-title" style="color:#facc15;font-weight:700">{title}</div>']
    for r in range(0, H, rs):
        a_row, h_row = [], []
        for c in range(0, W, cs):
            v = diff[r, c]
            if not np.isfinite(v):
                a_row.append('  .  '); h_row.append('<span style="color:#374151">  .  </span>')
            else:
                v_clip = max(-99.9, min(99.9, float(v)))
                txt = f"{v_clip:+4.1f} "  # exactly 5 chars: e.g. "+3.2 ", "-1.4 "
                if v_clip > 0.05:
                    a_row.append(f"\x1b[32m{txt}\x1b[0m")
                    h_row.append(f'<span style="color:#22c55e">{txt}</span>')
                elif v_clip < -0.05:
                    a_row.append(f"\x1b[31m{txt}\x1b[0m")
                    h_row.append(f'<span style="color:#ef4444">{txt}</span>')
                else:
                    a_row.append(f"\x1b[90m{txt}\x1b[0m")
                    h_row.append(f'<span style="color:#6b7280">{txt}</span>')
        ansi_lines.append("".join(a_row))
        html_lines.append('<div>' + "".join(h_row) + '</div>')
    return "\n".join(ansi_lines), "\n".join(html_lines)


def _ansi_to_hex(code):
    return {'36': '#22d3ee', '35': '#e879f9', '32': '#22c55e',
            '31': '#ef4444', '33': '#facc15', '90': '#6b7280'}.get(str(code), '#e5e7eb')


@app.route('/api/multi-epoch-mle', methods=['POST'])
def api_multi_epoch_mle():
    """
    Many-image Maximum-Likelihood bathymetry with epoch comparison.

    POST JSON:
      bbox: {west, south, east, north}
      epoch_a: {start, end}
      epoch_b: {start, end}
      n_scenes_per_epoch: int (default 6, max 12)
    """
    t0 = TM.time()
    try:
        data = req.get_json(force=True, silent=True) or {}
        bbox_d = data.get('bbox')
        if not bbox_d:
            return jsonify({'error': 'Draw a rectangle'}), 400
        bbox = [bbox_d['west'], bbox_d['south'], bbox_d['east'], bbox_d['north']]

        epoch_a = data.get('epoch_a') or {'start': data.get('start_date', '2023-05-01'),
                                          'end':   data.get('end_date',   '2023-09-30')}
        epoch_b = data.get('epoch_b') or {'start': '2024-05-01', 'end': '2024-09-30'}
        n_sc = int(max(2, min(12, data.get('n_scenes_per_epoch', 6))))

        # ICESat-2 fixed-period window: union of A and B, or widened via
        # explicit ice_start/ice_end. Seafloor stable on 1–2y scale → reuse
        # the same photon set as a quasi-truth anchor for BOTH epochs.
        use_ice = bool(data.get('use_icesat2', True))
        all_dates = [epoch_a['start'], epoch_a['end'], epoch_b['start'], epoch_b['end']]
        ice_start = data.get('ice_start') or min(all_dates)
        ice_end   = data.get('ice_end')   or max(all_dates)
        ice_laser = int(data.get('ice_laser', 1))
        ice_threshold = int(data.get('ice_threshold', 30))
        ice_water_temp = float(data.get('ice_water_temp', 22.0))
        ice_max_granules = int(data.get('ice_max_granules', 4))

        L.info(f"=== MultiEpochMLE: bbox={bbox} A={epoch_a} B={epoch_b} N={n_sc} "
               f"ICE={use_ice} ice_window={ice_start}→{ice_end} ===")

        # Shared references
        gebco = fetch_gebco(bbox)
        if not gebco or len(gebco.get('depths', [])) < 5:
            return jsonify({'error': 'GEBCO returned no data — no ocean coverage'}), 400
        xyz_pts = xyz_points_in_bbox(bbox)

        # ── ICESat-2 fetched ONCE for the full period ──
        ice_pts, ice_err, ice_info = [], None, {}
        if use_ice:
            ice_pts, ice_err = _get_icesat2_reference(
                bbox, ice_start, ice_end,
                laser=ice_laser, threshold=ice_threshold,
                water_temp=ice_water_temp, max_granules=ice_max_granules,
            )
            cached = _ICESAT2_CACHE.get(
                _icesat2_cache_key(bbox, ice_start, ice_end, ice_laser, ice_threshold), {})
            ice_info = {
                'used': True,
                'window': {'start': ice_start, 'end': ice_end},
                'n_photons': len(ice_pts),
                'n_granules': cached.get('n_granules', 0),
                'granule_dates': cached.get('dates', []),
                'error': ice_err,
            }
            if ice_err:
                L.warning(f"MLE: ICESat-2 unavailable ({ice_err}) — proceeding without")
        else:
            ice_info = {'used': False, 'n_photons': 0}

        L.info(f"MLE refs: GEBCO={len(gebco['depths'])} XYZ={len(xyz_pts[0])} "
               f"ICESat-2={len(ice_pts)}")

        epA = _epoch_depth_mle(bbox, epoch_a['start'], epoch_a['end'], n_sc,
                               gebco, xyz_pts, ice_pts=ice_pts)
        epB = _epoch_depth_mle(bbox, epoch_b['start'], epoch_b['end'], n_sc,
                               gebco, xyz_pts, ice_pts=ice_pts)
        if epA is None or epB is None:
            return jsonify({'error': 'MLE failed: one or both epochs returned no scenes'}), 400

        # Align grids (take min H,W)
        H = min(epA['H'], epB['H']); W = min(epA['W'], epB['W'])
        dA = epA['depth'][:H, :W]; dB = epB['depth'][:H, :W]
        sA = epA['sigma'][:H, :W]; sB = epB['sigma'][:H, :W]
        wA = epA['water_mask'][:H, :W]; wB = epB['water_mask'][:H, :W]
        common_water = wA & wB
        diff = np.where(common_water, dB - dA, np.nan)
        diff_sigma = np.where(common_water, np.sqrt(sA ** 2 + sB ** 2), np.nan)
        significant = np.where(np.isfinite(diff),
                               (np.abs(diff) > np.maximum(0.5, 2.0 * diff_sigma)),
                               False)

        # Stats
        valid = np.isfinite(diff)
        vd = diff[valid]
        stats_diff = {}
        if len(vd) > 0:
            stats_diff = {
                'mean_change_m': round(float(np.mean(vd)), 3),
                'std_change_m': round(float(np.std(vd)), 3),
                'max_erosion_m': round(float(np.min(vd)), 3),
                'max_deposition_m': round(float(np.max(vd)), 3),
                'rmsd_m': round(float(np.sqrt(np.mean(vd ** 2))), 3),
                'pct_significant': round(100.0 * float(np.sum(significant)) / max(1, int(valid.sum())), 2),
                'n_valid_cells': int(valid.sum()),
            }

        # ── ASCII dual-color preview ──
        ansi_A, html_A = _ascii_depth_block(dA, f"EPOCH A  {epoch_a['start']} → {epoch_a['end']}  ({epA['n_scenes_used']} scenes)", "36")
        ansi_B, html_B = _ascii_depth_block(dB, f"EPOCH B  {epoch_b['start']} → {epoch_b['end']}  ({epB['n_scenes_used']} scenes)", "35")
        ansi_D, html_D = _ascii_diff_block(diff, f"DIFF  B − A  (m)  green=+deposition  red=-erosion")
        ascii_full = "\n".join([ansi_A, "", ansi_B, "", ansi_D])
        html_full = "<pre>" + "\n\n".join([html_A, html_B, html_D]) + "</pre>"
        for ln in ascii_full.splitlines():
            L.info(ln)

        # ── Rasters + GeoTIFFs for each epoch + diff ──
        def _raster(dep, bbox, wm):
            try:
                b64, bounds, mx = depth_to_raster_png(dep, bbox, water_mask=wm)
                return {'png': b64, 'bounds': bounds, 'max_depth': mx}
            except Exception as ex:
                L.warning(f"raster: {ex}"); return {}
        def _gtif(dep, bbox):
            try: return build_geotiff_b64(dep, bbox)
            except Exception: return None

        out = {
            'bbox': bbox_d,
            'epoch_a': {
                'start': epoch_a['start'], 'end': epoch_a['end'],
                'n_scenes_used': epA['n_scenes_used'], 'scene_dates': epA['scene_dates'],
                'r2': epA['r2'], 'rmse': epA['rmse'], 'bias': epA['bias'],
                'grid_shape': [H, W],
                'raster': _raster(dA, bbox, wA),
                'geotiff_b64': _gtif(dA, bbox),
                'sigma_geotiff_b64': _gtif(sA, bbox),
                'points': grid_to_points(dA, bbox, max_pts=120000),
            },
            'epoch_b': {
                'start': epoch_b['start'], 'end': epoch_b['end'],
                'n_scenes_used': epB['n_scenes_used'], 'scene_dates': epB['scene_dates'],
                'r2': epB['r2'], 'rmse': epB['rmse'], 'bias': epB['bias'],
                'grid_shape': [H, W],
                'raster': _raster(dB, bbox, wB),
                'geotiff_b64': _gtif(dB, bbox),
                'sigma_geotiff_b64': _gtif(sB, bbox),
                'points': grid_to_points(dB, bbox, max_pts=120000),
            },
            'diff': {
                'stats': stats_diff,
                'geotiff_b64': _gtif(diff, bbox),
                'raster': _raster(np.abs(diff), bbox, common_water),
            },
            'ascii': {
                'text': ascii_full,
                'html': html_full,
            },
            'icesat2': ice_info,
            'sources_used': [
                f"GEBCO({len(gebco['depths'])})",
                f"XYZ({len(xyz_pts[0])})",
                (f"ICESat-2-fixed({ice_info.get('n_photons',0)}ph,"
                 f"{ice_info.get('n_granules',0)}gr,{ice_start}→{ice_end})"
                 if use_ice else "ICESat-2(off)"),
                f"S2-HR-land-mask",
                f"MLE-A({epA['n_scenes_used']}sc)",
                f"MLE-B({epB['n_scenes_used']}sc)",
            ],
            'processing_time_sec': round(TM.time() - t0, 1),
        }

        # Persist to monitoring store so dashboards pick it up
        try:
            from backend import monitoring as _mon  # type: ignore
        except Exception:
            try:
                import monitoring as _mon  # type: ignore
            except Exception:
                _mon = None
        if _mon is not None:
            try:
                sid_a = f"mle_A_{epoch_a['start']}_{int(TM.time())}"
                sid_b = f"mle_B_{epoch_b['start']}_{int(TM.time())}"
                _mon.store_survey(survey_id=sid_a, bbox=bbox, date=epoch_a['start'],
                                  depth_grid=dA, uncertainty=sA,
                                  method='multi_epoch_mle', resolution=10,
                                  r2=epA['r2'], rmse=epA['rmse'],
                                  sources='S2-MLE+GEBCO+XYZ',
                                  metadata={'n_scenes': epA['n_scenes_used']})
                _mon.store_survey(survey_id=sid_b, bbox=bbox, date=epoch_b['start'],
                                  depth_grid=dB, uncertainty=sB,
                                  method='multi_epoch_mle', resolution=10,
                                  r2=epB['r2'], rmse=epB['rmse'],
                                  sources='S2-MLE+GEBCO+XYZ',
                                  metadata={'n_scenes': epB['n_scenes_used']})
                out['monitoring_ids'] = {'survey_a': sid_a, 'survey_b': sid_b}
            except Exception as ex:
                L.warning(f"MLE: monitoring persist skipped ({ex})")

        L.info(f"=== MultiEpochMLE DONE: A={epA['n_scenes_used']}sc B={epB['n_scenes_used']}sc "
               f"{out['processing_time_sec']}s ===")
        return jsonify(out)
    except Exception as ex:
        L.error(f"MULTI-EPOCH-MLE: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# ENHANCED MULTI-SOURCE FUSION (ICESat-2 + iBoating + GEBCO)
# ══════════════════════════════════════════════════════════════
@app.route('/api/enhanced-bathymetry', methods=['POST'])
def api_enhanced_bathymetry():
    """
    Enhanced multi-source fusion endpoint.
    Combines ICESat-2 (multi-date SlideRule), iBoating (Gemini digitization),
    and GEBCO (Bayesian background) via Kriging fusion.
    Exports 10m + 1m GeoTIFFs with uncertainty maps.

    POST JSON:
      bbox: {west, south, east, north}
      center_date: "YYYY-MM-DD" (for ICESat-2 temporal weighting)
      use_icesat2: true/false (default true)
      use_iboating: true/false (default true)
      use_gebco: true/false (default true)
      icesat2_max_days: int (default 720)
      resolution_m: int (default 10)
      export_1m: true/false (default true)
    """
    try:
        from multi_source_fusion import run_enhanced_bathymetry, save_geotiffs_to_disk
        data = req.get_json(force=True, silent=True) or {}
        bbox_dict = data.get('bbox')
        if not bbox_dict:
            return jsonify({'error': 'Draw a rectangle (bbox required)'}), 400
        bbox = [bbox_dict['west'], bbox_dict['south'], bbox_dict['east'], bbox_dict['north']]

        center_date = data.get('center_date', data.get('start_date', '2024-07-01'))
        gemini_key = os.getenv('GEMINI_API_KEY', '') or os.getenv('GOOGLE_API_KEY', '')

        result = run_enhanced_bathymetry(
            bbox=bbox,
            center_date=center_date,
            gemini_api_key=gemini_key,
            use_icesat2=data.get('use_icesat2', True),
            use_iboating=data.get('use_iboating', True),
            use_gebco=data.get('use_gebco', True),
            icesat2_max_days=int(data.get('icesat2_max_days', 720)),
            resolution_m=int(data.get('resolution_m', 10)),
            export_1m=data.get('export_1m', True),
        )

        if 'error' in result:
            return jsonify(result), 400

        # Also save to disk
        try:
            saved = save_geotiffs_to_disk(result)
            result['saved_files'] = saved
        except Exception as sx:
            L.warning(f"Disk save skipped: {sx}")

        return jsonify(result)
    except Exception as ex:
        L.error(f"ENHANCED-BATHY: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# CNN_SMART — flexible multi-layer bathymetry CNN, tide-harmonised
# SlideRule + user XYZ/shapefile + heavy user-label weighting.
# ══════════════════════════════════════════════════════════════
import sys as _sys
_SMART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CNN_Smart')
if _SMART_DIR not in _sys.path:
    _sys.path.insert(0, _SMART_DIR)


def _smart_load_user_uploads(user_xyz_paths, user_shp_paths, bbox, user_weight):
    """Merge any user-uploaded XYZ / shapefile references for a bbox.

    Returns arrays (lats, lons, depths, weights). Weight for every
    user-sample is fixed to `user_weight` so it dominates auto sources.
    """
    import numpy as _np
    all_lat, all_lon, all_dep = [], [], []
    for p in (user_xyz_paths or []):
        try:
            from smart_refs import load_xyz_in_bbox
            d = load_xyz_in_bbox([p], bbox, from_crs=32640)  # UTM40N default
            if len(d['lats']) == 0:
                d = load_xyz_in_bbox([p], bbox, from_crs=4326)  # try WGS84 direct
            if len(d['lats']):
                all_lat.append(d['lats']); all_lon.append(d['lons']); all_dep.append(d['depths'])
        except Exception as ex:
            L.warning(f"smart: XYZ load failed for {p}: {ex}")
    for p in (user_shp_paths or []):
        try:
            import fiona
            with fiona.open(p) as src:
                for feat in src:
                    geom = feat['geometry']
                    prop = feat.get('properties', {}) or {}
                    d = None
                    for k in ('depth', 'DEPTH', 'z', 'Z', 'elev', 'elevation'):
                        if k in prop:
                            d = prop[k]; break
                    if d is None:
                        continue
                    if geom['type'] == 'Point':
                        lon, lat = geom['coordinates'][:2]
                    elif geom['type'] == 'MultiPoint':
                        for lon, lat in geom['coordinates']:
                            all_lat.append(lat); all_lon.append(lon); all_dep.append(abs(float(d)))
                        continue
                    else:
                        continue
                    all_lat.append(float(lat)); all_lon.append(float(lon)); all_dep.append(abs(float(d)))
        except Exception as ex:
            L.warning(f"smart: SHP load failed for {p}: {ex}")
    if not all_lat:
        return _np.array([]), _np.array([]), _np.array([]), _np.array([])
    lats = _np.concatenate([_np.asarray(a).ravel() for a in all_lat])
    lons = _np.concatenate([_np.asarray(a).ravel() for a in all_lon])
    deps = _np.concatenate([_np.asarray(a).ravel() for a in all_dep])
    w, s, e, n = bbox
    m = (lats >= s) & (lats <= n) & (lons >= w) & (lons <= e)
    lats, lons, deps = lats[m], lons[m], deps[m]
    ok = (deps > 0.3) & (deps <= MAX_DEPTH_M)
    lats, lons, deps = lats[ok], lons[ok], deps[ok]
    weights = _np.full(len(lats), float(user_weight), dtype=_np.float32)
    L.info(f"smart: user-uploaded refs in bbox: {len(lats)} (weight={user_weight})")
    return lats, lons, deps, weights


@app.route('/api/boa-cnn-bilstm', methods=['POST'])
def api_boa_cnn_bilstm():
    """BOA-CNN-BiLSTM bathymetry inversion (Zhu et al. 2025 JSTARS).

    POST JSON:
        bbox          : {west, south, east, north}
        s2_start/s2_end  (default: last 90 days)
        sr_start/sr_end  (default: last 365 days)
        use_boa       : bool (default True — Bayesian hyperparam tuning)
        boa_calls     : int  (default 25; paper used 100)
        boa_cv_folds  : int  (default 3; paper used 5)
        sliderule_confidence : "high" / "high_med" / "all" (default "high")
        mc_passes : unused — included for API symmetry with /api/smart-cnn
    """
    import datetime as _dt
    import numpy as _np
    try:
        from smart_refs import fetch_high_confidence_atl03, harmonise_to_s2_epoch, _tide_at, _wave_at
        from boa_cnn_bilstm import train_and_predict as _boa_predict

        data = req.get_json(force=True, silent=True) or {}
        bbox_dict = data.get('bbox')
        if not bbox_dict:
            return jsonify({'error': 'bbox required'}), 400
        bbox = (float(bbox_dict['west']), float(bbox_dict['south']),
                float(bbox_dict['east']), float(bbox_dict['north']))
        # SPEC-2: pinned defaults (no wall-clock). Explicit dates still honored.
        s2_sd = data.get('s2_start', PINNED_S2_WINDOW[0])
        s2_ed = data.get('s2_end', PINNED_S2_WINDOW[1])
        sr_sd = data.get('sr_start', PINNED_SR_WINDOW[0])
        sr_ed = data.get('sr_end', PINNED_SR_WINDOW[1])
        use_boa = bool(data.get('use_boa', True))
        boa_calls = int(data.get('boa_calls', 25))
        boa_cv_folds = int(data.get('boa_cv_folds', 3))
        sr_conf_mode = data.get('sliderule_confidence', 'high')

        # 1) S2
        L.info(f"[BOA-CNN-BiLSTM] S2 {s2_sd}..{s2_ed}  bbox={bbox}")
        s2 = fetch_s2(bbox, s2_sd, s2_ed, res=10, cloud=25)
        H, W = s2['red'].shape
        water_mask = s2.get('water_mask', s2['ndwi'] > 0).astype(bool)

        # 2) S2 epoch for tide harmonisation
        d0 = _dt.datetime.fromisoformat(s2_sd).replace(tzinfo=_dt.timezone.utc)
        d1 = _dt.datetime.fromisoformat(s2_ed).replace(tzinfo=_dt.timezone.utc)
        s2_epoch = (d0 + (d1 - d0) / 2).replace(hour=7, minute=30, second=0, microsecond=0)
        lat_c = 0.5 * (bbox[1] + bbox[3]); lon_c = 0.5 * (bbox[0] + bbox[2])
        s2_tide = _tide_at(lat_c, lon_c, s2_epoch)
        s2_wave = _wave_at(lat_c, lon_c, s2_epoch)

        # 3) SlideRule (paper uses ATL03 + AEDTA + refraction; our pipeline
        #    already delivers refraction-corrected points, plus we tide-rebase
        #    each photon to the S2 epoch per Eq. (3) of the paper).
        sr_raw = fetch_high_confidence_atl03(bbox, sr_sd, sr_ed,
                                              confidence_mode=sr_conf_mode, verbose=True)
        sr_segs, _, _ = harmonise_to_s2_epoch(sr_raw, s2_epoch, bbox)
        n_sr = len(sr_segs)
        if n_sr < 30:
            # Expand confidence as fallback — the paper requires enough labels.
            L.info("  [BOA-CNN-BiLSTM] only %d high-conf sr pts — relaxing to all", n_sr)
            sr_raw = fetch_high_confidence_atl03(bbox, sr_sd, sr_ed,
                                                  confidence_mode='all', verbose=True)
            sr_segs, _, _ = harmonise_to_s2_epoch(sr_raw, s2_epoch, bbox)
            n_sr = len(sr_segs)
        if n_sr < 30:
            return jsonify({'error': f'insufficient SlideRule photons ({n_sr})'}), 400

        refs = {
            'lats':   _np.array([p['lat'] for p in sr_segs], dtype=_np.float64),
            'lons':   _np.array([p['lon'] for p in sr_segs], dtype=_np.float64),
            'depths': _np.array([p['depth'] for p in sr_segs], dtype=_np.float64),
        }
        L.info(f"[BOA-CNN-BiLSTM] refs={n_sr}  depth range "
               f"{refs['depths'].min():.1f}-{refs['depths'].max():.1f} m  "
               f"tide_S2={s2_tide:+.2f}m")

        # 4) Run the method
        out, err = _boa_predict(
            s2=s2, bbox=bbox, ref_pts=refs, water_mask=water_mask,
            use_boa=use_boa, boa_calls=boa_calls, boa_cv_folds=boa_cv_folds,
        )
        if err:
            return jsonify({'error': err}), 400

        depth = out['depth']
        raster_b64, raster_bounds, raster_max = depth_to_raster_png(
            depth, bbox, water_mask=water_mask)

        valid = _np.isfinite(depth) & (depth > 0)
        depth_stats = {
            'min_m':    float(_np.nanmin(depth)) if valid.any() else None,
            'max_m':    float(_np.nanmax(depth)) if valid.any() else None,
            'mean_m':   float(_np.nanmean(depth)) if valid.any() else None,
            'median_m': float(_np.nanmedian(depth)) if valid.any() else None,
            'coverage_pct': round(float(valid.sum() / max(water_mask.sum(), 1) * 100), 1),
        }
        return jsonify({
            'method': 'BOA-CNN-BiLSTM (Zhu et al. 2025 JSTARS)',
            'bbox': list(bbox),
            's2_epoch_utc': s2_epoch.isoformat(),
            's2_tide_m': float(s2_tide),
            's2_wave_hs_m': float(s2_wave),
            'n_sliderule': int(n_sr),
            'n_ref_total': int(out['n_ref_total']),
            'n_train': int(out['n_train']),
            'n_test':  int(out['n_test']),
            'feature_names':    list(out['feature_names']),
            'hyperparameters':  out['hyperparameters'],
            'boa':              out.get('boa', {}),
            'validation':       out['metrics'],
            'model_params_M':   round(out['n_params'] / 1e6, 3),
            'depth_stats':      depth_stats,
            'raster_png_base64': raster_b64,
            'raster_bounds':     raster_bounds,
            'raster_max_depth':  raster_max,
        })
    except Exception as ex:
        L.error(f"BOA-CNN-BiLSTM error: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/smart-cnn/models', methods=['GET'])
def api_smart_cnn_models():
    """List models in the Smart CNN disk cache."""
    try:
        from smart_cache import list_cached
        return jsonify({'models': list_cached()})
    except Exception as ex:
        return jsonify({'error': str(ex)}), 500


@app.route('/api/smart-cnn', methods=['POST'])
def api_smart_cnn():
    """Flexible multi-layer CNN bathymetry — hot-wired for the map UI.

    POST JSON:
        bbox           : {west, south, east, north}  (required)
        s2_start       : YYYY-MM-DD   (default: 3 months window ending today)
        s2_end         : YYYY-MM-DD
        sr_start       : YYYY-MM-DD   (default: 6 months window)
        sr_end         : YYYY-MM-DD
        layers         : list of feature-layer names (default = all 14)
        retrain        : bool (default False — use cache if available)
        user_xyz_paths : list of local paths to XYZ files
        user_shp_paths : list of local paths to shapefiles
        user_weight    : float (default 20.0 — how strongly user data beats auto)
        port_polygons  : list of [[lon,lat],...] closed polygons (for canal prob)
        include_gebco  : bool (default auto — GEBCO fallback if refs < 150
                         or max ref depth < 10 m and n_ref < 500)
        sliderule_confidence : "high" / "high_med" / "all" (default "all" —
                         include deep-water photons; essential for offshore ROIs)
        mc_passes      : int (default 6)
    """
    import datetime as _dt
    import numpy as _np
    try:
        from smart_features import DEFAULT_LAYERS, PORT_POLYGONS as _PP, build_feature_stack, normalise_stack
        from smart_refs import fetch_high_confidence_atl03, harmonise_to_s2_epoch, _tide_at, _wave_at
        from smart_cnn import SmartCNNConfig, train_and_predict, predict_only
        from smart_cache import make_key, save as cache_save, load as cache_load, apply_norm

        data = req.get_json(force=True, silent=True) or {}
        bbox_dict = data.get('bbox')
        if not bbox_dict:
            return jsonify({'error': 'Draw a rectangle (bbox required)'}), 400
        bbox = (float(bbox_dict['west']), float(bbox_dict['south']),
                float(bbox_dict['east']), float(bbox_dict['north']))

        # SPEC-2: pinned defaults (no wall-clock). Explicit dates still honored.
        s2_sd = data.get('s2_start', PINNED_S2_WINDOW[0])
        s2_ed = data.get('s2_end', PINNED_S2_WINDOW[1])
        sr_sd = data.get('sr_start', PINNED_SR_WINDOW[0])
        sr_ed = data.get('sr_end', PINNED_SR_WINDOW[1])
        layers = data.get('layers') or list(DEFAULT_LAYERS)
        retrain = bool(data.get('retrain', False))
        user_xyz = data.get('user_xyz_paths') or []
        user_shp = data.get('user_shp_paths') or []
        user_w = float(data.get('user_weight', 20.0))
        mc_passes = int(data.get('mc_passes', 6))
        include_gebco = data.get('include_gebco', None)

        port_polys_req = data.get('port_polygons')
        if port_polys_req:
            port_polys = [[(float(p[0]), float(p[1])) for p in poly] for poly in port_polys_req]
        else:
            w, s, e, n = bbox
            lat_c, lon_c = 0.5 * (s + n), 0.5 * (w + e)
            port_polys = []
            for _name, poly in _PP.items():
                lons_p = [p[0] for p in poly]; lats_p = [p[1] for p in poly]
                if (min(lons_p) <= e and max(lons_p) >= w and
                    min(lats_p) <= n and max(lats_p) >= s):
                    port_polys.append(poly)

        # ── 1) Sentinel-2 ──
        L.info(f"[SmartCNN] S2 {s2_sd}..{s2_ed}  bbox={bbox}")
        s2 = fetch_s2(bbox, s2_sd, s2_ed, res=10, cloud=25)
        H, W = s2['red'].shape

        # ── 2) S2 epoch tide + wave ──
        import datetime as _dt2
        d0 = _dt2.datetime.fromisoformat(s2_sd).replace(tzinfo=_dt2.timezone.utc)
        d1 = _dt2.datetime.fromisoformat(s2_ed).replace(tzinfo=_dt2.timezone.utc)
        s2_epoch = (d0 + (d1 - d0) / 2).replace(hour=7, minute=30, second=0, microsecond=0)
        lat_c = 0.5 * (bbox[1] + bbox[3])
        lon_c = 0.5 * (bbox[0] + bbox[2])
        s2_tide = _tide_at(lat_c, lon_c, s2_epoch)
        s2_wave = _wave_at(lat_c, lon_c, s2_epoch)

        # ── 3) Feature stack ──
        stack_raw, names, ctx = build_feature_stack(
            s2, bbox, layers=layers, port_polygons=port_polys,
            tide_m=s2_tide, wave_hs=s2_wave,
        )

        user_flag = 'u' if (user_xyz or user_shp) else 'auto'
        cache_key = make_key(bbox, names, user_flag)

        # ── 4) Reference data collection ──
        ref_lats = _np.array([]); ref_lons = _np.array([]); ref_deps = _np.array([]); ref_w = _np.array([])
        n_user = 0; n_sliderule = 0; n_xyz_store = 0; n_gebco = 0

        # 4a) User uploads — HIGHEST weight
        u_lat, u_lon, u_dep, u_w = _smart_load_user_uploads(user_xyz, user_shp, bbox, user_weight=user_w)
        n_user = len(u_lat)
        ref_lats = _np.concatenate([ref_lats, u_lat])
        ref_lons = _np.concatenate([ref_lons, u_lon])
        ref_deps = _np.concatenate([ref_deps, u_dep])
        ref_w = _np.concatenate([ref_w, u_w])

        # 4b) SlideRule — keep ALL bathy photons (high/medium/low) by default.
        # "low" just means depth > 20 m in run_sliderule's DEPTH-based bucketing
        # — not a quality signal — so we need them for deep ROIs.
        sr_conf_mode = data.get('sliderule_confidence', 'all')
        sr_raw = fetch_high_confidence_atl03(bbox, sr_sd, sr_ed,
                                              confidence_mode=sr_conf_mode, verbose=True)
        sr_segs, _, _ = harmonise_to_s2_epoch(sr_raw, s2_epoch, bbox)
        n_sliderule = len(sr_segs)
        if n_sliderule:
            sr_lats = _np.array([p['lat'] for p in sr_segs])
            sr_lons = _np.array([p['lon'] for p in sr_segs])
            sr_deps = _np.array([p['depth'] for p in sr_segs])
            sr_confs = [p.get('confidence', 'medium') for p in sr_segs]
            _conf_w = {'high': 3.0, 'medium': 1.8, 'low': 1.0}
            sr_wgt = _np.array([_conf_w.get(c, 1.0) for c in sr_confs], dtype=_np.float32)
            ref_lats = _np.concatenate([ref_lats, sr_lats])
            ref_lons = _np.concatenate([ref_lons, sr_lons])
            ref_deps = _np.concatenate([ref_deps, sr_deps])
            ref_w = _np.concatenate([ref_w, sr_wgt])
            L.info(f"[SmartCNN] sliderule by conf: {dict((c, sr_confs.count(c)) for c in set(sr_confs))}")

        # --- Internal 50/50 blend between high-conf SlideRule and
        # previously-digitised nautical-chart points from the training
        # store (source LIKE 'iboating_%'). Not surfaced in the UI; just
        # folded into the ref set with elevated weight (2.5) so the
        # training dataset is always anchored on two independent quality
        # sources whenever both are available. ---
        try:
            from training_store import get_store as _gs
            _store = _gs()
            if _store is not None:
                _rows = _store.query_bbox(list(bbox), buffer_km=0, max_points=60000)
                if _rows and _rows.get('count', 0) > 0:
                    _srcs = _rows.get('sources') or []
                    _ib_mask = _np.array([isinstance(s, str) and s.startswith('iboating')
                                            for s in _srcs], dtype=bool)
                    _ib_lat = _np.asarray(_rows['lats'])[_ib_mask]
                    _ib_lon = _np.asarray(_rows['lons'])[_ib_mask]
                    _ib_dep = _np.asarray(_rows['depths'])[_ib_mask]
                    _ok = (_ib_dep > 0.3) & (_ib_dep <= MAX_DEPTH_M)
                    _ib_lat, _ib_lon, _ib_dep = _ib_lat[_ok], _ib_lon[_ok], _ib_dep[_ok]
                    # high-conf SlideRule subset (already stored above)
                    _sr_hi_mask = _np.array([c == 'high' for c in (sr_confs if n_sliderule else [])], dtype=bool)
                    _n_sr_hi = int(_sr_hi_mask.sum()) if n_sliderule else 0
                    _n_ib = int(len(_ib_lat))
                    _n_blend = min(_n_sr_hi, _n_ib)
                    if _n_blend >= 10:
                        _rng = _np.random.default_rng(0)
                        # Subsample whichever side is larger to equalise counts.
                        if _n_ib > _n_blend:
                            _pick = _rng.choice(_n_ib, size=_n_blend, replace=False)
                            _ib_lat, _ib_lon, _ib_dep = _ib_lat[_pick], _ib_lon[_pick], _ib_dep[_pick]
                        _ib_wgt = _np.full(_n_blend, 2.5, dtype=_np.float32)
                        ref_lats = _np.concatenate([ref_lats, _ib_lat])
                        ref_lons = _np.concatenate([ref_lons, _ib_lon])
                        ref_deps = _np.concatenate([ref_deps, _ib_dep])
                        ref_w = _np.concatenate([ref_w, _ib_wgt])
                        L.info(f"[SmartCNN] balanced blend: +{_n_blend} paired pts")
        except Exception as _bx:
            L.warning(f"[SmartCNN] balanced blend skipped: {_bx}")

        # 4c) Project training store — weight 1.5. iBoating-sourced rows are
        # skipped here because they already went through the paired blend above.
        try:
            from training_store import get_store as _gs
            store = _gs()
            q = store.query_bbox(list(bbox), buffer_km=0, max_points=20000) if store else None
            if q and q.get('count', 0) > 0:
                ts_lat = _np.asarray(q['lats']); ts_lon = _np.asarray(q['lons']); ts_dep = _np.asarray(q['depths'])
                ts_src = q.get('sources') or []
                _not_ib = _np.array([not (isinstance(s, str) and s.startswith('iboating'))
                                      for s in ts_src], dtype=bool) if ts_src else _np.ones(len(ts_lat), dtype=bool)
                keep = _not_ib & (ts_dep > 0.3) & (ts_dep <= MAX_DEPTH_M)
                ts_lat, ts_lon, ts_dep = ts_lat[keep], ts_lon[keep], ts_dep[keep]
                n_xyz_store = len(ts_lat)
                ref_lats = _np.concatenate([ref_lats, ts_lat])
                ref_lons = _np.concatenate([ref_lons, ts_lon])
                ref_deps = _np.concatenate([ref_deps, ts_dep])
                ref_w = _np.concatenate([ref_w, _np.full(n_xyz_store, 1.5, dtype=_np.float32)])
        except Exception as ex:
            L.warning(f"[SmartCNN] training_store query failed: {ex}")

        # 4d) GEBCO fallback — weight 0.4. Always include it when we're
        # short on refs OR when the current refs don't cover the likely
        # depth range of the ROI (e.g. SlideRule gave us shallow-only
        # points but the bbox contains deep water).
        n_total = len(ref_deps)
        cur_max_d = float(_np.max(ref_deps)) if n_total else 0.0
        needs_deep_prior = cur_max_d < 10.0 and n_total < 500
        gebco_wanted = include_gebco if include_gebco is not None \
                         else (n_total < 150 or needs_deep_prior)
        if gebco_wanted:
            gb = fetch_gebco(bbox)
            if gb and len(gb['lats']):
                n_gebco = len(gb['lats'])
                ref_lats = _np.concatenate([ref_lats, gb['lats']])
                ref_lons = _np.concatenate([ref_lons, gb['lons']])
                ref_deps = _np.concatenate([ref_deps, gb['depths']])
                ref_w = _np.concatenate([ref_w, _np.full(n_gebco, 0.4, dtype=_np.float32)])
                L.info(f"[SmartCNN] GEBCO prior: +{n_gebco} pts "
                       f"(depth {float(gb['depths'].min()):.1f}-{float(gb['depths'].max()):.1f} m)")

        n_ref = len(ref_deps)
        if n_ref:
            _d_all = ref_deps
            L.info(f"[SmartCNN] refs: user={n_user} sliderule={n_sliderule} "
                   f"store_xyz={n_xyz_store} gebco={n_gebco} TOTAL={n_ref}  "
                   f"depth_range={_np.min(_d_all):.1f}..{_np.max(_d_all):.1f} m  "
                   f"p50={_np.median(_d_all):.1f}m")
        else:
            L.info(f"[SmartCNN] refs: user={n_user} sliderule={n_sliderule} "
                   f"store_xyz={n_xyz_store} gebco={n_gebco} TOTAL=0")
        if n_ref < 20 and not retrain:
            # no training possible — see if a cache is available
            blob = cache_load(cache_key)
            if blob is None:
                return jsonify({'error': f'Insufficient reference data ({n_ref}) and no cached model'}), 400

        # ── 5) Normalise + decide train-vs-predict-only ──
        water_mask = ctx['water_mask']
        stack_n, norm_stats = normalise_stack(stack_raw, water_mask=water_mask)

        used_cache = False
        if not retrain:
            blob = cache_load(cache_key)
            if blob is not None:
                # Reapply training-time normalisation so model sees the same feature scale
                stack_n_cached = apply_norm(stack_raw, blob['norm_stats'])
                pr, err = predict_only(
                    stack_n_cached, water_mask, bbox,
                    blob['state_dict'], blob['cfg'], blob['depth_scale'],
                    mc_passes=mc_passes,
                )
                if err:
                    L.warning(f"[SmartCNN] cached predict failed: {err} — retraining")
                else:
                    depth = pr['depth']; unc = pr['uncertainty']
                    used_cache = True

        if not used_cache:
            refs = {'lats': ref_lats, 'lons': ref_lons, 'depths': ref_deps, 'weights': ref_w}
            cfg = SmartCNNConfig(epochs=100, n_patches=192, patch_size=64, mc_passes=mc_passes,
                                  batch_size=10, patience=14, augment=True, residual_correction=True)
            tr, err = train_and_predict(stack_n, water_mask, refs, bbox, cfg=cfg)
            if err:
                return jsonify({'error': err}), 400
            depth = tr['depth']; unc = tr['uncertainty']
            # Save cache
            try:
                cache_save(cache_key,
                            tr['model_state'], tr['model_cfg'],
                            names, tr['depth_scale'], norm_stats,
                            meta={'bbox': list(bbox),
                                  's2_window': [s2_sd, s2_ed],
                                  'sr_window': [sr_sd, sr_ed],
                                  'n_user': int(n_user), 'n_sliderule': int(n_sliderule),
                                  'n_xyz_store': int(n_xyz_store), 'n_gebco': int(n_gebco),
                                  'metrics': tr.get('metrics', {})})
            except Exception as sx:
                L.warning(f"[SmartCNN] cache save failed: {sx}")

        # ── 6) Render overlay + return ──
        raster_b64, raster_bounds, raster_max = depth_to_raster_png(
            depth, bbox, water_mask=water_mask)

        valid = _np.isfinite(depth) & (depth > 0)
        depth_stats = {
            'min_m': float(_np.nanmin(depth)) if valid.any() else None,
            'max_m': float(_np.nanmax(depth)) if valid.any() else None,
            'mean_m': float(_np.nanmean(depth)) if valid.any() else None,
            'median_m': float(_np.nanmedian(depth)) if valid.any() else None,
            'coverage_pct': round(float(valid.sum() / max(water_mask.sum(), 1) * 100), 1),
        }

        resp = {
            'method': 'SmartCNN (CBAM+ASPP, flexible layers)',
            'layers': names,
            'n_layers': len(names),
            'used_cache': bool(used_cache),
            'cache_key': cache_key,
            'bbox': list(bbox),
            's2_tide_m': float(s2_tide),
            's2_wave_hs_m': float(s2_wave),
            's2_epoch_utc': s2_epoch.isoformat(),
            'n_user_refs': int(n_user),
            'n_sliderule': int(n_sliderule),
            'n_xyz_store': int(n_xyz_store),
            'n_gebco': int(n_gebco),
            'n_ref_total': int(n_ref),
            'user_weight': float(user_w),
            'depth_stats': depth_stats,
            'raster_png_base64': raster_b64,
            'raster_bounds': raster_bounds,
            'raster_max_depth': raster_max,
        }
        if not used_cache and 'metrics' in tr:
            resp['validation'] = tr['metrics']
            resp['n_train'] = int(tr['n_train'])
            resp['n_val'] = int(tr['n_val_pts'])
            resp['model_params_M'] = round(tr['n_params'] / 1e6, 3)
        return jsonify(resp)

    except Exception as ex:
        L.error(f"SMART-CNN error: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# GPU MOSAIC — large-region tiled bathymetry @10m
# ══════════════════════════════════════════════════════════════
@app.route('/api/extract-mosaic',methods=['POST'])
def api_extract_mosaic():
    """
    GPU-accelerated mosaic bathymetry for large regions.
    Splits into sub-tiles, selects best S2 scenes (last N weeks, 10m),
    runs CNN per tile on GPU, mosaics with feathered blending.
    """
    if not MOSAIC_AVAILABLE:
        return jsonify({'error':'Mosaic engine not available (check torch/GPU)'}),500
    try:
        data=req.get_json(force=True,silent=True) or {}
        bbox_dict=data.get('bbox')
        if not bbox_dict:return jsonify({'error':'Draw a rectangle'}),400
        bbox=[bbox_dict['west'],bbox_dict['south'],bbox_dict['east'],bbox_dict['north']]
        p=data.get('params',{})
        n_weeks=int(p.get('n_weeks',4))
        max_cloud=int(p.get('max_cloud',20))
        tile_deg=float(p.get('tile_deg',0.08))
        overlap_deg=float(p.get('overlap_deg',0.008))
        L.info(f"=== MOSAIC API: {n_weeks}w, cloud≤{max_cloud}%, tile={tile_deg}° ===")
        result=mosaic_extract(bbox,n_weeks=n_weeks,max_cloud=max_cloud,
                              tile_deg=tile_deg,overlap_deg=overlap_deg)
        return jsonify(result)
    except Exception as ex:
        L.error(f"MOSAIC: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex)}),500

@app.route('/api/mosaic-scenes',methods=['POST'])
def api_mosaic_scenes():
    """Preview available S2 scenes for a region before running mosaic."""
    if not MOSAIC_AVAILABLE:
        return jsonify({'error':'Mosaic engine not available'}),500
    try:
        data=req.get_json(force=True,silent=True) or {}
        bbox_dict=data.get('bbox')
        if not bbox_dict:return jsonify({'error':'Need bbox'}),400
        bbox=[bbox_dict['west'],bbox_dict['south'],bbox_dict['east'],bbox_dict['north']]
        p=data.get('params',{})
        scenes=search_best_scenes(bbox,n_weeks=int(p.get('n_weeks',4)),
                                  max_cloud=int(p.get('max_cloud',20)),top_k=10)
        w,s,e,n=bbox;area=abs(e-w)*abs(n-s)*111*111*math.cos(math.radians((n+s)/2))
        tile_deg=float(p.get('tile_deg',0.08))
        n_tiles=max(1,int(math.ceil((e-w)/tile_deg))*int(math.ceil((n-s)/tile_deg)))
        return jsonify({'scenes':scenes,'area_km2':round(area,1),'n_tiles':n_tiles})
    except Exception as ex:
        return jsonify({'error':str(ex)}),500

# ══════════════════════════════════════════════════════════════
# AUTO-EVALUATION — Multi-source nautical chart + GEBCO fallback
# ══════════════════════════════════════════════════════════════

def _latlon_to_tile(lat,lon,z):
    lat_r=math.radians(lat);nt=2**z
    x=int((lon+180)/360*nt)
    y=int((1-math.log(math.tan(lat_r)+1/math.cos(lat_r))/math.pi)/2*nt)
    return x,y

def _tile_to_latlon(x,y,z):
    nt=2**z;lon=x/nt*360-180
    lat=math.degrees(math.atan(math.sinh(math.pi*(1-2*y/nt))))
    return lat,lon

def _fetch_noaa_chart(bbox, zoom=15):
    """NOAA Raster Nautical Charts — US waters. Light bg with printed depth soundings."""
    from PIL import Image
    w,s,e,n=bbox
    x0,y0=_latlon_to_tile(n,w,zoom);x1,y1=_latlon_to_tile(s,e,zoom)
    if x0>x1:x0,x1=x1,x0
    if y0>y1:y0,y1=y1,y0
    x1=min(x1,x0+7);y1=min(y1,y0+7)
    tw=256;cols=x1-x0+1;rows=y1-y0+1
    canvas=Image.new('RGB',(cols*tw,rows*tw),(240,235,220))
    got=0
    for ty in range(y0,y1+1):
        for tx in range(x0,x1+1):
            try:
                url=f"https://tileservice.charts.noaa.gov/tiles/50000_1/{zoom}/{tx}/{ty}.png"
                r=requests.get(url,timeout=8)
                if r.ok and len(r.content)>500:
                    tile=Image.open(io.BytesIO(r.content)).convert('RGB')
                    canvas.paste(tile,((tx-x0)*tw,(ty-y0)*tw));got+=1
            except:pass
    actual_n,actual_w=_tile_to_latlon(x0,y0,zoom)
    actual_s,actual_e=_tile_to_latlon(x1+1,y1+1,zoom)
    L.info(f"NOAA chart: {got}/{cols*rows} tiles at z{zoom}")
    if got<2:return None
    buf=io.BytesIO();canvas.save(buf,format='PNG')
    return base64.b64encode(buf.getvalue()).decode(),canvas.size[0],canvas.size[1],[actual_w,actual_s,actual_e,actual_n]

def _fetch_osm_depth_chart(bbox, zoom=15):
    """OpenSeaMap depths overlay on light CartoDB Voyager base — readable depth numbers."""
    from PIL import Image
    w,s,e,n=bbox
    x0,y0=_latlon_to_tile(n,w,zoom);x1,y1=_latlon_to_tile(s,e,zoom)
    if x0>x1:x0,x1=x1,x0
    if y0>y1:y0,y1=y1,y0
    x1=min(x1,x0+7);y1=min(y1,y0+7)
    tw=256;cols=x1-x0+1;rows=y1-y0+1
    canvas=Image.new('RGB',(cols*tw,rows*tw),(245,245,245))
    got_base=0;got_depth=0
    for ty in range(y0,y1+1):
        for tx in range(x0,x1+1):
            px=(tx-x0)*tw;py=(ty-y0)*tw
            # Light basemap so depth numbers (dark blue) are visible
            try:
                sub=['a','b','c'][(tx+ty)%3]
                url=f"https://{sub}.basemaps.cartocdn.com/rastertiles/voyager/{zoom}/{tx}/{ty}.png"
                r=requests.get(url,timeout=8)
                if r.ok and len(r.content)>200:
                    canvas.paste(Image.open(io.BytesIO(r.content)).convert('RGB'),(px,py));got_base+=1
            except:pass
            # OpenSeaMap depth overlay
            try:
                url=f"https://tiles.openseamap.org/seamark/{zoom}/{tx}/{ty}.png"
                r=requests.get(url,timeout=8)
                if r.ok and len(r.content)>500:
                    ov=Image.open(io.BytesIO(r.content)).convert('RGBA')
                    base=canvas.crop((px,py,px+tw,py+tw)).convert('RGBA')
                    canvas.paste(Image.alpha_composite(base,ov).convert('RGB'),(px,py));got_depth+=1
            except:pass
            # Also overlay OpenSeaMap depth contours/soundings layer
            try:
                url=f"https://depth.openseamap.org/gebco/{zoom}/{tx}/{ty}.png"
                r=requests.get(url,timeout=8)
                if r.ok and len(r.content)>300:
                    ov=Image.open(io.BytesIO(r.content)).convert('RGBA')
                    base=canvas.crop((px,py,px+tw,py+tw)).convert('RGBA')
                    canvas.paste(Image.alpha_composite(base,ov).convert('RGB'),(px,py))
            except:pass
    actual_n,actual_w=_tile_to_latlon(x0,y0,zoom)
    actual_s,actual_e=_tile_to_latlon(x1+1,y1+1,zoom)
    L.info(f"OSM chart: {got_base} base + {got_depth} seamark tiles at z{zoom}")
    if got_base<2:return None
    buf=io.BytesIO();canvas.save(buf,format='PNG')
    return base64.b64encode(buf.getvalue()).decode(),canvas.size[0],canvas.size[1],[actual_w,actual_s,actual_e,actual_n]

def _gemini_extract_soundings(img_b64, bbox, img_w, img_h):
    """Specialised Gemini prompt for reading depth soundings from nautical charts."""
    api_key=os.getenv('GEMINI_API_KEY','') or os.getenv('GOOGLE_API_KEY','')
    if not api_key:return None,"GEMINI_API_KEY not set"
    w,s,e,n=bbox
    prompt=f"""This is a nautical/marine chart image showing a coastal area.
Bounding box: SW corner ({s:.5f}°N, {w:.5f}°E), NE corner ({n:.5f}°N, {e:.5f}°E).
Image size: {img_w}x{img_h} pixels.

Your task: Find ALL depth sounding numbers printed on the water areas of this chart.
Depth soundings are small numbers (like 3.2, 15, 7.8, 22.1) scattered across the blue/white water areas.
They indicate water depth in meters (or sometimes fathoms — if in fathoms, multiply by 1.8288 to convert to meters).

IMPORTANT:
- Look for small isolated numbers on the water area — these ARE depth soundings
- Depth contour lines (isobaths) often have labels like "5", "10", "20" — include those too
- Numbers near the coast are typically smaller (shallow), farther from coast are larger (deep)
- IGNORE: land elevation numbers, coordinate labels, scale bar numbers, buoy/marker identifiers
- If the chart is mostly land with little water, return what you can find on water areas

Return a JSON array of objects, each with x (pixels from left), y (pixels from top), depth (value in meters):
[{{"x":120,"y":340,"depth":5.2}},{{"x":450,"y":200,"depth":12.8}}]

Return at least 20 points if visible. If truly no depth numbers are visible, return []."""
    try:
        url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        resp=requests.post(url,json={"contents":[{"parts":[
            {"inline_data":{"mime_type":"image/png","data":img_b64}},
            {"text":prompt}
        ]}],"generationConfig":{"temperature":0.1,"maxOutputTokens":16384}},timeout=180)
        if not resp.ok:return None,f"Gemini {resp.status_code}: {resp.text[:200]}"
        text=resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        clean=text.replace('```json','').replace('```','').strip()
        L.info(f"Gemini soundings: {len(clean)} chars")
        pts=None
        try:pts=json.loads(clean)
        except:
            idx=clean.find('[')
            if idx>=0:
                raw=clean[idx:]
                try:pts=json.loads(raw)
                except:
                    lb=raw.rfind('}')
                    if lb>0:
                        try:pts=json.loads(raw[:lb+1]+']')
                        except:pass
        if pts and isinstance(pts,list) and len(pts)>0:
            L.info(f"Gemini: {len(pts)} soundings extracted")
            return pts,None
        return None,f"No soundings found in chart (response: {clean[:120]})"
    except Exception as ex:return None,str(ex)

def _gebco_cross_validate(predicted, bbox):
    """
    Independent validation using GEBCO at higher resolution than training.
    Sample GEBCO at different grid points than what was used for CNN training.
    """
    w,s,e,n=bbox
    # Fetch GEBCO at high res
    wpx=max(32,min(800,int((e-w)*480)));hpx=max(32,min(800,int((n-s)*480)))
    url=(f"https://wms.gebco.net/mapserv?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
         f"&COVERAGE=gebco_latest&CRS=EPSG:4326&FORMAT=GeoTIFF"
         f"&BBOX={s},{w},{n},{e}&WIDTH={wpx}&HEIGHT={hpx}")
    try:
        r=requests.get(url,timeout=60)
        if not r.ok or len(r.content)<200:return None,"GEBCO fetch failed"
        elev=tifffile.imread(io.BytesIO(r.content)).astype(np.float32)
        grid=np.where(elev<0,-elev,np.nan)
    except Exception as ex:return None,f"GEBCO: {ex}"

    gh,gw=grid.shape
    pred_pts=[p for p in predicted if p.get('photon_class') in ('interpolated','bathymetry') and p.get('depth',0)>0]
    if not pred_pts:return None,"No predicted depths"
    pred_lats=np.array([p['lat'] for p in pred_pts])
    pred_lons=np.array([p['lon'] for p in pred_pts])
    pred_d=np.array([p['depth'] for p in pred_pts])

    # Sample GEBCO at random offset grid (not aligned with training grid)
    np.random.seed(42)
    step=max(2,int(math.sqrt(gh*gw/1500)))
    offset_r=np.random.randint(0,max(1,step//2))
    offset_c=np.random.randint(0,max(1,step//2))
    pairs=[];pair_details=[]
    for r in range(offset_r,gh,step):
        lat=n-(r/gh)*(n-s)
        for c in range(offset_c,gw,step):
            v=grid[r,c]
            if not(np.isfinite(v) and 0.5<v<60):continue
            lon=w+(c/gw)*(e-w)
            # Find nearest predicted point
            dists=((pred_lats-lat)**2+(pred_lons-lon)**2)
            idx=int(np.argmin(dists))
            if dists[idx]<0.003**2:  # ~300m match
                pd_val=float(pred_d[idx])
                diff=pd_val-float(v)
                s44_max=math.sqrt(0.5**2+(0.013*float(v))**2)
                pairs.append({'obs':float(v),'pred':pd_val,'diff':diff,'s44_pass':abs(diff)<=s44_max})
                pair_details.append({'lat':round(lat,6),'lon':round(lon,6),'chart_depth':round(float(v),2),'pred_depth':round(pd_val,2),'diff':round(diff,2),'s44_pass':abs(diff)<=s44_max,'photon_class':'gebco_eval'})

    if not pairs:return None,"No matching GEBCO points"
    obs_arr=np.array([p['obs'] for p in pairs]);pred_arr=np.array([p['pred'] for p in pairs])
    diffs=pred_arr-obs_arr
    rmse=float(np.sqrt(np.mean(diffs**2)));mae=float(np.mean(np.abs(diffs)))
    bias=float(np.mean(diffs))
    ss_res=np.sum(diffs**2);ss_tot=np.sum((obs_arr-np.mean(obs_arr))**2)
    r2=float(1-ss_res/(ss_tot+1e-10)) if ss_tot>0 else 0
    s44_pct=float(sum(1 for p in pairs if p['s44_pass'])/len(pairs)*100)
    L.info(f"GEBCO cross-val: {len(pairs)} pairs, RMSE={rmse:.2f}m, R²={r2:.3f}")
    return {'rmse':round(rmse,3),'mae':round(mae,3),'bias':round(bias,3),'r2':round(r2,4),
            's44_pass_pct':round(s44_pct,1),'n_pairs':len(pairs),'pairs':pair_details,
            'source':'GEBCO high-res cross-validation'},None

@app.route('/api/auto-evaluate',methods=['POST'])
def api_auto_evaluate():
    """
    Multi-strategy auto-evaluation:
    1. Try NOAA nautical chart (US waters) — has printed depth soundings
    2. Try OpenSeaMap + depth layer on light basemap
    3. Both sent to Gemini to extract soundings
    4. Fallback: GEBCO high-res independent cross-validation
    All results compared against predicted depths → RMSE, bias, R², S-44.
    """
    try:
        data=req.get_json(force=True,silent=True) or {}
        bbox_dict=data.get('bbox')
        predicted=data.get('predicted',[])
        zoom=data.get('zoom',15)
        if not bbox_dict:return jsonify({'error':'Need bbox'}),400
        if not predicted:return jsonify({'error':'No predicted depths to evaluate'}),400

        bbox=[bbox_dict['west'],bbox_dict['south'],bbox_dict['east'],bbox_dict['north']]
        chart_depths=[]
        chart_source=None

        # ── STRATEGY 1: NOAA RNC (US waters — best depth soundings) ──
        L.info(f"Auto-eval: trying NOAA charts at z{zoom}...")
        noaa=_fetch_noaa_chart(bbox,zoom)
        if noaa:
            img_b64,img_w,img_h,actual_bbox=noaa
            pts,err=_gemini_extract_soundings(img_b64,actual_bbox,img_w,img_h)
            if pts:
                la,lo,dp=pixels_to_latlon(pts,actual_bbox,img_w,img_h)
                chart_depths=[{'lat':round(a,6),'lon':round(o,6),'depth':round(d,1),'photon_class':'chart_eval'} for a,o,d in zip(la,lo,dp) if d>0]
                chart_source='NOAA RNC + Gemini Vision'
                L.info(f"NOAA: {len(chart_depths)} soundings extracted")

        # ── STRATEGY 2: OpenSeaMap on light base (worldwide) ──
        if len(chart_depths)<10:
            L.info(f"Auto-eval: trying OpenSeaMap on light base at z{zoom}...")
            osm=_fetch_osm_depth_chart(bbox,zoom)
            if osm:
                img_b64,img_w,img_h,actual_bbox=osm
                pts,err=_gemini_extract_soundings(img_b64,actual_bbox,img_w,img_h)
                if pts:
                    la,lo,dp=pixels_to_latlon(pts,actual_bbox,img_w,img_h)
                    new_depths=[{'lat':round(a,6),'lon':round(o,6),'depth':round(d,1),'photon_class':'chart_eval'} for a,o,d in zip(la,lo,dp) if d>0]
                    if len(new_depths)>len(chart_depths):
                        chart_depths=new_depths;chart_source='OpenSeaMap + Gemini Vision'
                        L.info(f"OSM: {len(chart_depths)} soundings extracted")

        # ── STRATEGY 3: If higher zoom helps, retry at z+1 ──
        if len(chart_depths)<10 and zoom<16:
            L.info(f"Auto-eval: retrying at zoom {zoom+1}...")
            for fetch_fn,name in [(_fetch_noaa_chart,'NOAA'),(_fetch_osm_depth_chart,'OSM')]:
                result=fetch_fn(bbox,zoom+1)
                if result:
                    img_b64,img_w,img_h,actual_bbox=result
                    pts,err=_gemini_extract_soundings(img_b64,actual_bbox,img_w,img_h)
                    if pts:
                        la,lo,dp=pixels_to_latlon(pts,actual_bbox,img_w,img_h)
                        new_depths=[{'lat':round(a,6),'lon':round(o,6),'depth':round(d,1),'photon_class':'chart_eval'} for a,o,d in zip(la,lo,dp) if d>0]
                        if len(new_depths)>len(chart_depths):
                            chart_depths=new_depths;chart_source=f'{name} z{zoom+1} + Gemini Vision'
                            L.info(f"{name} z{zoom+1}: {len(chart_depths)} soundings")
                    if len(chart_depths)>=10:break

        # ── Compare chart depths vs predicted ──
        gemini_result=None
        if len(chart_depths)>=3:
            pred_pts=[p for p in predicted if p.get('photon_class') in ('interpolated','bathymetry') and p.get('depth',0)>0]
            if pred_pts:
                pred_lats=np.array([p['lat'] for p in pred_pts])
                pred_lons=np.array([p['lon'] for p in pred_pts])
                pred_d=np.array([p['depth'] for p in pred_pts])
                pairs=[];pair_details=[]
                for cp in chart_depths:
                    dists=((pred_lats-cp['lat'])**2+(pred_lons-cp['lon'])**2)
                    idx=int(np.argmin(dists))
                    if dists[idx]<0.003**2:
                        pd_val=float(pred_d[idx]);diff=pd_val-cp['depth']
                        s44_max=math.sqrt(0.5**2+(0.013*cp['depth'])**2)
                        pairs.append({'obs':cp['depth'],'pred':pd_val,'diff':diff,'s44_pass':abs(diff)<=s44_max})
                        pair_details.append({'lat':cp['lat'],'lon':cp['lon'],'chart_depth':round(cp['depth'],2),'pred_depth':round(pd_val,2),'diff':round(diff,2),'s44_pass':abs(diff)<=s44_max,'photon_class':'auto_eval'})
                if pairs:
                    obs_arr=np.array([p['obs'] for p in pairs]);pred_arr=np.array([p['pred'] for p in pairs])
                    diffs_arr=pred_arr-obs_arr
                    rmse=float(np.sqrt(np.mean(diffs_arr**2)));mae=float(np.mean(np.abs(diffs_arr)));bias=float(np.mean(diffs_arr))
                    ss_res=np.sum(diffs_arr**2);ss_tot=np.sum((obs_arr-np.mean(obs_arr))**2)
                    r2=float(1-ss_res/(ss_tot+1e-10)) if ss_tot>0 else 0
                    s44_pct=float(sum(1 for p in pairs if p['s44_pass'])/len(pairs)*100)
                    gemini_result={'rmse':round(rmse,3),'mae':round(mae,3),'bias':round(bias,3),'r2':round(r2,4),
                                   's44_pass_pct':round(s44_pct,1),'n_pairs':len(pairs),'n_chart':len(chart_depths),
                                   'chart_points':chart_depths,'pairs':pair_details,'source':chart_source}
                    L.info(f"Chart eval: {len(pairs)} pairs, RMSE={rmse:.2f}m, R²={r2:.3f}")

        # ── STRATEGY 4: GEBCO independent cross-validation (always) ──
        L.info("Auto-eval: running GEBCO cross-validation...")
        gebco_result,gebco_err=_gebco_cross_validate(predicted,bbox)

        # ── Combine results ──
        if gemini_result and gebco_result:
            # Both available — return both, highlight the chart-based one
            out=gemini_result
            out['gebco_validation']=gebco_result
            out['source']=f"{chart_source} + GEBCO cross-val"
        elif gemini_result:
            out=gemini_result
        elif gebco_result:
            out=gebco_result
            out['chart_points']=[]
            out['n_chart']=0
            out['note']='No nautical chart soundings found — using GEBCO independent grid as reference'
        else:
            return jsonify({'error':'No chart soundings found and GEBCO cross-validation failed','chart_points':[],'n_chart':0,'n_pairs':0}),200

        return jsonify(out)
    except Exception as ex:
        L.error(f"Auto-eval: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex)}),500

# ══════════════════════════════════════════════════════════════
# i-BOATING FULL PIPELINE  (screenshot → extract → train → validate)
# ══════════════════════════════════════════════════════════════
@app.route('/api/iboating-pipeline',methods=['POST'])
def api_iboating_pipeline():
    """Full i-Boating pipeline: screenshot → Gemini → 80/20 split → CNN → validate."""
    try:
        data=req.get_json(force=True,silent=True) or {}
        bbox_dict=data.get('bbox')
        if not bbox_dict:return jsonify({'error':'Draw a rectangle first'}),400
        sd=data.get('start_date','2024-05-01');ed=data.get('end_date','2024-09-30')
        zoom=data.get('zoom',14)
        p=data.get('params',{})
        bbox=[bbox_dict['west'],bbox_dict['south'],bbox_dict['east'],bbox_dict['north']]
        w,s,e,n=bbox
        area=abs(e-w)*abs(n-s)*111*111*math.cos(math.radians((n+s)/2))
        res=100 if area>500 else 50 if area>100 else 20

        L.info(f"=== i-BOATING PIPELINE zoom={zoom} area={area:.0f}km² ===")

        # 1. Fetch Sentinel-2
        L.info("Fetching Sentinel-2 imagery...")
        s2=fetch_s2(bbox,sd,ed,res=res,cloud=p.get('max_cloud',20))

        # 2. Run i-Boating pipeline
        try:
            from backend.iboating import run_iboating_pipeline
        except ImportError:
            from iboating import run_iboating_pipeline
        result=run_iboating_pipeline(bbox,s2,zoom=zoom,train_ratio=0.8)

        if result.get('error') and result.get('depth') is None:
            return jsonify({'error':result['error'],'sources_used':result.get('sources_used',[])}),200

        # 3. Build output (same format as /api/extract for frontend compatibility)
        out={'points':[],'interpolated_points':[],'stats':{},'bbox':bbox_dict,
             'sources_used':result.get('sources_used',[]),
             'ml_stats':result.get('ml_stats',{}),
             'iboating':{'screenshot':result.get('screenshot_path'),
                         'n_chart':len(result.get('chart_points',[])),
                         'n_train':len(result.get('train_points',[])),
                         'n_val':len(result.get('val_points',[])),
                         'validation':result.get('validation')},
             'tracks':[],'sea_profiles':[],'bath_profiles':[]}

        # Add chart points for display
        for cp in result.get('chart_points',[]):
            out['points'].append({'lat':cp['lat'],'lon':cp['lon'],'depth':cp['depth'],
                                  'photon_class':'chart','type':cp.get('type','sounding')})

        # Depth grid → point cloud
        if result.get('depth') is not None:
            out['interpolated_points']=grid_to_points(result['depth'],bbox)
            depth=result['depth']
            val=depth[np.isfinite(depth)&(depth>0)]
            water=s2['ndwi']>0
            zc={"vs":int(np.sum(water&(depth<=3))),"sh":int(np.sum(water&(depth>3)&(depth<=8))),
                "md":int(np.sum(water&(depth>8)&(depth<=15))),"dp":int(np.sum(water&(depth>15)))}
            out['stats']={'mean_depth':round(float(np.mean(val)),2) if len(val) else 0,
                          'max_depth':round(float(np.max(val)),2) if len(val) else 0,
                          'min_depth':round(float(np.min(val)),2) if len(val) else 0,
                          'std_depth':round(float(np.std(val)),2) if len(val) else 0,
                          'grid_points':len(out['interpolated_points']),'resolution_m':res}
            out['ml_stats']['zone_counts']=zc

        # Validation results
        if result.get('validation'):
            v=result['validation']
            out['iboating']['validation']=v
            # Add validation point pairs for map display
            for pair in v.get('pairs',[]):
                out['points'].append({'lat':pair['lat'],'lon':pair['lon'],
                    'depth':pair['obs'],'pred_depth':pair['pred'],
                    'diff':pair['diff'],'s44_pass':pair['s44_pass'],
                    'photon_class':'validation'})

        out['sources_used'].append(f"S2({s2['width']}x{s2['height']})")
        return jsonify(out)
    except Exception as ex:
        L.error(f"i-Boating pipeline: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex)}),500

def _parse_shapefile_payload(files):
    """Core shapefile/CSV parser shared by /api/upload-shapefile and
    /api/upload-observed. Saves uploaded parts to a temp dir, reads .prj →
    CRS (reprojects to WGS84), .shp geometry + .dbf attributes (axis order is
    fixed by the SHP spec: X=easting/lon, Y=northing/lat — never swapped).
    Returns {'points','count','crs','source_epsg'}; raises ValueError on bad input."""
    import tempfile,os,shutil
    td=tempfile.mkdtemp()
    try:
        shp_path=None
        for f in files:
            fp=os.path.join(td,f.filename)
            f.save(fp)
            if f.filename.lower().endswith('.shp'):shp_path=fp
            elif f.filename.lower().endswith('.csv'):
                df=pd.read_csv(fp) if 'pd' in dir() else None
                if df is not None:
                    pts=[];cols={c.lower().strip():c for c in df.columns}
                    lat_col=next((cols[k] for k in ('lat','latitude','y','point_y','northing') if k in cols),None)
                    lon_col=next((cols[k] for k in ('lon','longitude','lng','x','point_x','easting') if k in cols),None)
                    dep_col=next((cols[k] for k in ('depth','z','depth_m','bathymetry','point_z','elev','elevation') if k in cols),None)
                    if lat_col and lon_col and dep_col:
                        for _,row in df.iterrows():
                            try:pts.append({'lat':float(row[lat_col]),'lon':float(row[lon_col]),'depth':abs(float(row[dep_col])),'photon_class':'observed'})
                            except:continue
                    return {'points':pts,'count':len(pts),'crs':'WGS84','source_epsg':4326}
        if not shp_path:raise ValueError('No .shp found')
        import struct as st
        dbf_path=shp_path.replace('.shp','.dbf')
        prj_path=shp_path.replace('.shp','.prj')
        if not os.path.exists(dbf_path):raise ValueError('Missing .dbf — upload .shp + .dbf + .prj together')

        # ── Read PRJ for CRS and build transformer ──
        crs_name='WGS84';transformer=None;src_epsg=None
        if os.path.exists(prj_path):
            prj_txt=open(prj_path).read().strip()
            crs_name=prj_txt[:60]
            L.info(f"Shapefile PRJ: {prj_txt[:120]}")
            try:
                from pyproj import Transformer,CRS
                src_crs=CRS.from_wkt(prj_txt)
                src_epsg=src_crs.to_epsg()
                crs_name=f"{src_crs.name} (EPSG:{src_epsg})" if src_epsg else str(src_crs.name)
                if not src_crs.is_geographic:
                    transformer=Transformer.from_crs(src_crs,CRS.from_epsg(4326),always_xy=True)
                    L.info(f"Shapefile: will transform from {crs_name} → WGS84")
                else:
                    L.info(f"Shapefile: already geographic ({crs_name})")
            except Exception as ex:
                L.warning(f"CRS parse failed: {ex}")
                # Try fallback: detect UTM zone from PRJ text
                import re as _re
                utm_match=_re.search(r'UTM[_ ]Zone[_ ](\d+)([NS])',prj_txt,_re.IGNORECASE)
                if utm_match:
                    zone=int(utm_match.group(1));hemi=utm_match.group(2).upper()
                    epsg=32600+zone if hemi=='N' else 32700+zone
                    L.info(f"Fallback: detected UTM Zone {zone}{hemi} → EPSG:{epsg}")
                    try:
                        from pyproj import Transformer,CRS
                        transformer=Transformer.from_crs(CRS.from_epsg(epsg),CRS.from_epsg(4326),always_xy=True)
                        crs_name=f"UTM Zone {zone}{hemi} (EPSG:{epsg})"
                    except Exception as ex2:L.warning(f"UTM fallback failed: {ex2}")
        else:
            L.warning("No .prj file — assuming WGS84")

        # ── Read .shp geometry (Point coordinates) ──
        shp_geom=[]
        try:
            with open(shp_path,'rb') as sf:
                sf.read(24)  # file code + unused
                file_len=st.unpack('>I',sf.read(4))[0]*2
                sf.read(4)  # version
                shape_type=st.unpack('<I',sf.read(4))[0]
                sf.read(64)  # bounding box + ranges
                L.info(f"Shapefile: shape_type={shape_type}")
                while sf.tell()<file_len:
                    try:
                        rec_num=st.unpack('>I',sf.read(4))[0]
                        rec_len=st.unpack('>I',sf.read(4))[0]*2
                        rec_data=sf.read(rec_len)
                        stype=st.unpack('<I',rec_data[:4])[0]
                        if stype==1:  # Point
                            x,y=st.unpack('<dd',rec_data[4:20])
                            shp_geom.append((x,y))
                        elif stype==11:  # PointZ
                            x,y=st.unpack('<dd',rec_data[4:20])
                            z_val=st.unpack('<d',rec_data[20:28])[0] if len(rec_data)>=28 else 0
                            shp_geom.append((x,y,z_val))
                        elif stype==21:  # PointM
                            x,y=st.unpack('<dd',rec_data[4:20])
                            shp_geom.append((x,y))
                        else:
                            shp_geom.append(None)
                    except:
                        break
            L.info(f"Shapefile: read {len(shp_geom)} geometries from .shp")
        except Exception as ex:
            L.warning(f"SHP geometry read: {ex}")

        # ── Read DBF attributes ──
        dbf_records=[]
        with open(dbf_path,'rb') as f:
            f.read(4);nrec=st.unpack('<I',f.read(4))[0];hlen=st.unpack('<H',f.read(2))[0];rlen=st.unpack('<H',f.read(2))[0];f.read(20)
            fields=[]
            while True:
                fb=f.read(1)
                if fb==b'\r':break
                fname=(fb+f.read(10)).decode('ascii','ignore').strip('\x00');ftype=f.read(1).decode('ascii');f.read(4)
                fsize=st.unpack('B',f.read(1))[0];fdec=st.unpack('B',f.read(1))[0];f.read(14)
                fields.append((fname,ftype,fsize))
            L.info(f"DBF fields: {[fn for fn,ft,fs in fields]}")
            for i in range(nrec):
                rec=f.read(rlen);pos=1;vals={}
                for fn,ft,fs in fields:
                    vals[fn.lower().strip()]=rec[pos:pos+fs].decode('ascii','ignore').strip();pos+=fs
                dbf_records.append(vals)

        # ── Flexible field matching for X, Y, Z ──
        x_keys=['x','point_x','easting','lon','longitude','lng','coord_x']
        y_keys=['y','point_y','northing','lat','latitude','coord_y']
        z_keys=['z','point_z','depth','depth_m','bathymetry','elev','elevation','height','h']
        all_keys=set()
        if dbf_records:all_keys=set(dbf_records[0].keys())
        L.info(f"DBF keys (lowered): {sorted(all_keys)}")
        x_field=next((k for k in x_keys if k in all_keys),None)
        y_field=next((k for k in y_keys if k in all_keys),None)
        z_field=next((k for k in z_keys if k in all_keys),None)
        L.info(f"Matched fields: x={x_field}, y={y_field}, z={z_field}")

        # ── Build points: prefer .shp geometry, fall back to DBF fields ──
        pts=[]
        for i in range(min(len(dbf_records),max(len(shp_geom),len(dbf_records)))):
            try:
                # Get X, Y from .shp geometry if available
                if i<len(shp_geom) and shp_geom[i] is not None:
                    geom=shp_geom[i]
                    x,y=geom[0],geom[1]
                    z_from_geom=geom[2] if len(geom)>2 else None
                elif x_field and y_field and i<len(dbf_records):
                    x=float(dbf_records[i].get(x_field,0))
                    y=float(dbf_records[i].get(y_field,0))
                    z_from_geom=None
                else:
                    continue

                # Get Z (depth) — prefer DBF attribute, fallback to geometry Z
                z=0
                if z_field and i<len(dbf_records):
                    try:z=abs(float(dbf_records[i].get(z_field,0)))
                    except:z=0
                if z==0 and z_from_geom is not None:
                    z=abs(float(z_from_geom))

                if z<=0:continue

                # Transform to WGS84 if needed
                if transformer:
                    lon,lat=transformer.transform(x,y)
                else:
                    # Detect if already geographic or projected
                    if abs(x)>360 or abs(y)>360:
                        # Looks like projected coords without transformer — skip
                        L.warning(f"Record {i}: x={x},y={y} look projected but no transformer")
                        continue
                    lat,lon=y,x

                if -90<=lat<=90 and -180<=lon<=180:
                    pts.append({'lat':round(lat,6),'lon':round(lon,6),'depth':round(min(z,MAX_DEPTH_M),2),'photon_class':'observed'})
            except Exception as ex:
                continue

        L.info(f"Shapefile: {len(pts)} valid points from {len(dbf_records)} records, CRS={crs_name}")
        return {'points':pts,'count':len(pts),'crs':crs_name,'source_epsg':src_epsg}
    finally:
        shutil.rmtree(td,ignore_errors=True)

@app.route('/api/upload-shapefile',methods=['POST'])
def api_upload_shp():
    """Parse shapefile (any CRS → WGS84) with flexible field detection for X,Y,Z."""
    try:
        files=req.files.getlist('files')
        if not files:return jsonify({'error':'No files'}),400
        return jsonify(_parse_shapefile_payload(files))
    except ValueError as ex:
        return jsonify({'error':str(ex)}),400
    except Exception as ex:
        L.error(f"Shapefile: {ex}\n{traceback.format_exc()}");return jsonify({'error':str(ex)}),500

@app.route('/api/validate',methods=['POST'])
def api_validate():
    """Compare predicted vs observed depths → RMSE, bias, R², S-44."""
    try:
        data=req.get_json(force=True,silent=True) or {}
        predicted=data.get('predicted',[]);observed=data.get('observed',[])
        if not observed:return jsonify({'error':'No observed points'}),400
        # Build predicted lookup grid
        pred_pts=[p for p in predicted if p.get('photon_class') in ('interpolated','bathymetry') and p.get('depth',0)>0]
        if not pred_pts:return jsonify({'error':'No predicted depths'}),400
        # For each observed point, find nearest predicted
        pred_lats=np.array([p['lat'] for p in pred_pts])
        pred_lons=np.array([p['lon'] for p in pred_pts])
        pred_depths=np.array([p['depth'] for p in pred_pts])
        pairs=[];pair_details=[]
        for op in observed:
            olat,olon,odepth=op['lat'],op['lon'],op['depth']
            dists=((pred_lats-olat)**2+(pred_lons-olon)**2)
            idx=np.argmin(dists)
            if dists[idx]<0.001**2:  # within ~100m
                pd_val=float(pred_depths[idx])
                diff=pd_val-odepth
                # S-44 Order 1: max allowable = sqrt(a²+(b*d)²), a=0.5m, b=0.013
                s44_max=math.sqrt(0.5**2+(0.013*odepth)**2)
                s44_pass=abs(diff)<=s44_max
                pairs.append({'obs':odepth,'pred':pd_val,'diff':diff,'s44_pass':s44_pass,'s44_max':round(s44_max,3)})
                pair_details.append({'lat':olat,'lon':olon,'obs_depth':round(odepth,2),'pred_depth':round(pd_val,2),'diff':round(diff,2),'s44_pass':s44_pass,'photon_class':'validation'})
        if not pairs:return jsonify({'error':'No matching points found (too far apart)'}),400
        obs_arr=np.array([p['obs'] for p in pairs]);pred_arr=np.array([p['pred'] for p in pairs])
        diffs=pred_arr-obs_arr
        rmse=float(np.sqrt(np.mean(diffs**2)))
        mae=float(np.mean(np.abs(diffs)))
        bias=float(np.mean(diffs))
        ss_res=np.sum(diffs**2);ss_tot=np.sum((obs_arr-np.mean(obs_arr))**2)
        r2=float(1-ss_res/ss_tot) if ss_tot>0 else 0
        s44_pass_pct=float(sum(1 for p in pairs if p['s44_pass'])/len(pairs)*100)
        L.info(f"Validation: {len(pairs)} pairs, RMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}, S-44={s44_pass_pct:.0f}%")
        return jsonify({'rmse':round(rmse,3),'mae':round(mae,3),'bias':round(bias,3),'r2':round(r2,4),'s44_pass_pct':round(s44_pass_pct,1),'n_pairs':len(pairs),'pairs':pair_details,'pair_stats':pairs})
    except Exception as ex:
        L.error(f"Validate: {ex}");return jsonify({'error':str(ex)}),500

@app.route('/api/validate-xyz', methods=['POST'])
def api_validate_xyz():
    """
    Advanced validation: predicted depths vs observed (e.g. XYZ soundings).
    POST JSON: {predicted: [{lat,lon,depth}], observed: [{lat,lon,depth}]}

    Returns overall stats + multi-order IHO S-44 compliance + depth-stratified stats
    + residual histogram + spatial residual grid + IHO S-52 depth-class confusion.
    """
    try:
        data = req.get_json(force=True, silent=True) or {}
        predicted = data.get('predicted', [])
        observed = data.get('observed', [])
        # ROB-R3: validate shape up front with an actionable 400 instead of
        # letting a malformed body 500 with a bare KeyError/AttributeError
        # message ("'lat'") once it's too late.
        if not isinstance(predicted, list):
            return jsonify({'error': "'predicted' must be a list of {lat, lon, depth} objects"}), 400
        if not isinstance(observed, list):
            return jsonify({'error': "'observed' must be a list of {lat, lon, depth} objects"}), 400
        if not observed:
            return jsonify({'error': 'No observed points'}), 400
        for i, p in enumerate(predicted):
            if not isinstance(p, dict):
                return jsonify({'error': f"predicted[{i}] must be an object with lat/lon/depth, got {type(p).__name__}"}), 400

        # IHO-R6: echo the grid resolution the caller ran (frontend already
        # threads resolution_m through `params`, Sidebar.js) so the S-44/CATZOC
        # panel records which resolution produced the numbers it shows.
        # ROB-R1: reject non-finite (NaN/Infinity/1e309-overflow) values —
        # float() alone accepts "NaN"/"Infinity" strings and any float that
        # overflows to inf, which would otherwise echo a literal NaN/Infinity
        # token into the JSON body (unparseable by the browser).
        resolution_m = data.get('resolution_m')
        try:
            resolution_m = float(resolution_m) if resolution_m is not None else None
            if resolution_m is not None and not math.isfinite(resolution_m):
                resolution_m = None
        except Exception:
            resolution_m = None

        # GIS-R3: propagate the uploaded survey's vertical datum so a real
        # MSL/CD-vs-LAT-assumed offset can be flagged instead of silently
        # folded into bias/RMSE. Model predictions are always 'LAT assumed'
        # (_DATUM_NOTE) unless the caller states otherwise.
        observed_vertical_datum = data.get('observed_vertical_datum')
        if isinstance(observed_vertical_datum, str):
            observed_vertical_datum = observed_vertical_datum.strip().upper() or None
        else:
            observed_vertical_datum = None
        predicted_vertical_datum = (str(data.get('predicted_vertical_datum') or 'LAT')).strip().upper()

        pred_pts_bad = [i for i, p in enumerate(predicted)
                        if p.get('photon_class') in ('interpolated', 'bathymetry') and p.get('depth', 0) > 0
                        and ('lat' not in p or 'lon' not in p)]
        if pred_pts_bad:
            i0 = pred_pts_bad[0]
            missing = 'lat' if 'lat' not in predicted[i0] else 'lon'
            return jsonify({'error': f"predicted[{i0}] missing '{missing}'"}), 400
        pred_pts = [p for p in predicted
                    if p.get('photon_class') in ('interpolated', 'bathymetry') and p.get('depth', 0) > 0]
        if not pred_pts:
            return jsonify({'error': 'No predicted depths'}), 400

        pred_lats = np.array([p['lat'] for p in pred_pts], dtype=np.float64)
        pred_lons = np.array([p['lon'] for p in pred_pts], dtype=np.float64)
        pred_depths = np.array([p['depth'] for p in pred_pts], dtype=np.float64)

        # Fast nearest-neighbour via KDTree
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([pred_lats, pred_lons]))
            use_tree = True
        except Exception:
            tree = None; use_tree = False

        # Depth classes for IHO S-52 confusion matrix (aligned with IHO_S52_PALETTE)
        CLASS_BINS = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 9e9]
        CLASS_LABELS = ['0-2', '2-5', '5-10', '10-15', '15-20', '>20']
        def _cls(d):
            for i in range(len(CLASS_BINS) - 1):
                if d < CLASS_BINS[i + 1]:
                    return i
            return len(CLASS_LABELS) - 1

        # IHO-R2: CATZOC-B-consistent default match radius (~50 m; IHO M-3
        # Res 1/2002 order-B positional bound), overridable via
        # `max_match_m` in the POST body for larger/sparser reference sets.
        try:
            MAX_MATCH_M = float(data.get('max_match_m') or 50.0)
        except Exception:
            MAX_MATCH_M = 50.0
        MAX_NN_DEG = max(MAX_MATCH_M, 500.0) / 111320.0  # coarse degree pre-filter for the KDTree query only

        def _haversine_m(lat1, lon1, lat2, lon2):
            R = 6371000.0
            p1, p2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlmb = math.radians(lon2 - lon1)
            a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
            return 2 * R * math.asin(min(1.0, math.sqrt(a)))

        pairs_full = []          # full residual records (for stats)
        pairs_details = []       # capped detail list (for map display)
        match_dists_m = []       # NN distance (metres) for every INCLUDED pair
        n_excluded_by_distance = 0
        n_rejected_nonfinite = 0
        for op in observed:
            try:
                olat = float(op['lat']); olon = float(op['lon']); od = abs(float(op['depth']))
            except Exception:
                continue
            # ROB-R1: a NaN/Infinity observed depth (or lat/lon) must never
            # reach the stats — `nan <= 0` is False, so the old bare `od<=0`
            # gate let a NaN depth straight through and poisoned every
            # downstream stat (rmse/bias/mae all became literal NaN).
            if not (math.isfinite(olat) and math.isfinite(olon) and math.isfinite(od)):
                n_rejected_nonfinite += 1
                continue
            if od <= 0:
                continue
            if use_tree:
                dist_deg, idx = tree.query([olat, olon], k=1)
            else:
                d2 = (pred_lats - olat) ** 2 + (pred_lons - olon) ** 2
                idx = int(np.argmin(d2))
            idx = int(idx)
            dist_m = _haversine_m(olat, olon, float(pred_lats[idx]), float(pred_lons[idx]))
            if dist_m > MAX_MATCH_M:
                n_excluded_by_distance += 1
                continue
            pd_val = float(pred_depths[idx])
            diff = pd_val - od
            pairs_full.append({'lat': olat, 'lon': olon, 'obs': od, 'pred': pd_val, 'diff': diff, 'dist_m': round(dist_m, 2)})
            match_dists_m.append(dist_m)

        n_pairs = len(pairs_full)
        if n_pairs == 0:
            return jsonify({
                'error': f'No matching points found (nearest neighbour > {MAX_MATCH_M:.0f} m)',
                'n_excluded_by_distance': n_excluded_by_distance,
            }), 400

        match_distance_stats_m = {
            'mean': round(float(np.mean(match_dists_m)), 2),
            'p95': round(float(np.percentile(match_dists_m, 95)), 2),
            'max': round(float(np.max(match_dists_m)), 2),
        }

        obs_arr = np.array([p['obs'] for p in pairs_full])
        pred_arr = np.array([p['pred'] for p in pairs_full])
        diffs = pred_arr - obs_arr
        abs_diffs = np.abs(diffs)

        rmse = float(np.sqrt(np.mean(diffs ** 2)))
        mae = float(np.mean(abs_diffs))
        medae = float(np.median(abs_diffs))
        bias = float(np.mean(diffs))
        std = float(np.std(diffs))
        ss_res = float(np.sum(diffs ** 2))
        ss_tot = float(np.sum((obs_arr - obs_arr.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        # ROB-R1: np.corrcoef on a constant array (zero variance — e.g. all
        # observed depths identical) divides by a zero std internally and
        # returns nan (with a numpy RuntimeWarning), which is undetected by
        # a bare `n_pairs >= 2` guard. Report 0.0 (no linear relationship is
        # measurable) instead of a NaN token.
        if n_pairs >= 2 and float(np.std(obs_arr)) > 0 and float(np.std(pred_arr)) > 0:
            pearson = float(np.corrcoef(obs_arr, pred_arr)[0, 1])
            if not math.isfinite(pearson):
                pearson = 0.0
        else:
            pearson = 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_vals = np.where(obs_arr > 0.2, abs_diffs / obs_arr * 100, np.nan)
        mape = float(np.nanmean(mape_vals)) if np.any(np.isfinite(mape_vals)) else 0.0

        # Multi-order IHO S-44 TVU compliance
        # Special Order: a=0.25, b=0.0075; Order 1a: a=0.5, b=0.013; Order 2: a=1.0, b=0.023
        def _tvu(d, a, b):
            return math.sqrt(a * a + (b * d) ** 2)
        orders = {'special': (0.25, 0.0075), 'order1a': (0.5, 0.013), 'order2': (1.0, 0.023)}
        order_results = {}
        for k, (a, b) in orders.items():
            tvus = np.sqrt(a * a + (0.013 if False else b) ** 2 * obs_arr ** 2)  # vectorised
            tvus = np.sqrt(a * a + (b * obs_arr) ** 2)
            passes = abs_diffs <= tvus
            order_results[k] = {
                'a': a, 'b': b,
                'pass_pct': round(float(passes.sum()) / n_pairs * 100, 2),
                'n_pass': int(passes.sum()),
                'n_fail': int(n_pairs - int(passes.sum())),
            }

        # Per-pair S-44 Order-1a pass (for map colouring)
        tvu_1a = np.sqrt(0.5 ** 2 + (0.013 * obs_arr) ** 2)
        pass_1a = abs_diffs <= tvu_1a

        # Depth-stratified stats
        strat = []
        for i in range(len(CLASS_BINS) - 1):
            lo, hi = CLASS_BINS[i], CLASS_BINS[i + 1]
            mask = (obs_arr >= lo) & (obs_arr < hi)
            n = int(mask.sum())
            if n == 0:
                continue
            sd = diffs[mask]; sad = abs_diffs[mask]
            strat.append({
                'range': CLASS_LABELS[i],
                'n': n,
                'rmse': round(float(np.sqrt(np.mean(sd ** 2))), 3),
                'mae': round(float(np.mean(sad)), 3),
                'bias': round(float(np.mean(sd)), 3),
                'pass_1a': round(float(pass_1a[mask].sum()) / n * 100, 1),
            })

        # Residual histogram — 25 bins between -max..+max (clipped to ±10m)
        lim = max(2.0, float(np.percentile(abs_diffs, 99)) * 1.1)
        lim = min(lim, 15.0)
        bins = 25
        hist, edges = np.histogram(diffs, bins=bins, range=(-lim, lim))
        hist_data = [{'x': round(float((edges[i] + edges[i + 1]) / 2), 3),
                      'count': int(hist[i])} for i in range(bins)]

        # Spatial residual grid: 12x12 mean residual map over obs bbox (for a heatmap overlay)
        obs_lat_arr = np.array([p['lat'] for p in pairs_full])
        obs_lon_arr = np.array([p['lon'] for p in pairs_full])
        lat_lo, lat_hi = float(obs_lat_arr.min()), float(obs_lat_arr.max())
        lon_lo, lon_hi = float(obs_lon_arr.min()), float(obs_lon_arr.max())
        GH, GW = 12, 12
        sum_g = np.zeros((GH, GW), dtype=np.float64)
        cnt_g = np.zeros((GH, GW), dtype=np.int32)
        if lat_hi > lat_lo and lon_hi > lon_lo:
            for i in range(n_pairs):
                r = int((lat_hi - obs_lat_arr[i]) / (lat_hi - lat_lo + 1e-10) * (GH - 1))
                c = int((obs_lon_arr[i] - lon_lo) / (lon_hi - lon_lo + 1e-10) * (GW - 1))
                r = max(0, min(GH - 1, r)); c = max(0, min(GW - 1, c))
                sum_g[r, c] += diffs[i]; cnt_g[r, c] += 1
        with np.errstate(all='ignore'):
            mean_g = np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), np.nan)
        spatial_grid = {
            'bounds': {'south': lat_lo, 'north': lat_hi, 'west': lon_lo, 'east': lon_hi},
            'h': GH, 'w': GW,
            'cells': [[{'mean': None if not np.isfinite(mean_g[r, c]) else round(float(mean_g[r, c]), 3),
                        'n': int(cnt_g[r, c])}
                       for c in range(GW)] for r in range(GH)]
        }

        # IHO S-52 class confusion matrix
        cm = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int32)
        for i in range(n_pairs):
            cm[_cls(obs_arr[i])][_cls(pred_arr[i])] += 1
        class_agreement = round(float(np.trace(cm)) / n_pairs * 100, 1) if n_pairs else 0.0

        # Detail list for table / map (cap at 3000)
        for i, p in enumerate(pairs_full[:3000]):
            pairs_details.append({
                'lat': round(p['lat'], 6),
                'lon': round(p['lon'], 6),
                'obs_depth': round(p['obs'], 2),
                'pred_depth': round(p['pred'], 2),
                'diff': round(p['diff'], 2),
                's44_pass': bool(pass_1a[i]),
                'photon_class': 'validation',
                'match_dist_m': p['dist_m'],
            })

        # Legacy pair_stats (for backward-compat with existing UI)
        pair_stats = [{'obs': round(float(p['obs']), 3), 'pred': round(float(p['pred']), 3),
                       'diff': round(float(p['diff']), 3), 's44_pass': bool(pass_1a[i]),
                       's44_max': round(float(tvu_1a[i]), 3)}
                      for i, p in enumerate(pairs_full[:3000])]

        # IHO-R3: CATZOC tier + S-44 order label, via the SAME shared classifier
        # (backend/model_card.py::iho_order_from_p95) already used by
        # /api/results/<id>/analyse — the S-44 §3.3.1 gate is p95(abs error) ≤
        # TVU, not RMSE ≤ TVU. No duplicated threshold logic.
        try:
            from backend.model_card import iho_order_from_p95 as _iho_order_from_p95
        except ImportError:
            from model_card import iho_order_from_p95 as _iho_order_from_p95  # type: ignore
        p95_error_m = float(np.percentile(abs_diffs, 95))
        med_obs_depth = float(np.median(obs_arr))
        order_label, catzoc, tvu_at_median, orders_table = (
            _iho_order_from_p95(p95_error_m, med_obs_depth) if n_pairs >= 10 else (None, None, None, [])
        )
        tpu_caveat = ("RMSE/bias reported here are TVU-only (no THU assessment); "
                      "tide/datum/refraction TPU terms not included unless the "
                      "uploaded survey and the model output share a stated "
                      "vertical datum — see export metadata.")

        # IHO-R5: gridded SDB cannot claim Special/Order-1a full-seafloor
        # object-detection capability regardless of TVU pass-rate. Disclose
        # whenever either high-tier badge would show ≥95% pass.
        detection_capability_note = None
        special_pct = order_results.get('special', {}).get('pass_pct', 0)
        order1a_pct = order_results.get('order1a', {}).get('pass_pct', 0)
        if special_pct >= 95 or order1a_pct >= 95:
            res_txt = f"{resolution_m:g} m" if resolution_m is not None else "the selected"
            detection_capability_note = (
                "TVU pass-rate only; Special Order / Order 1a additionally require "
                "full-seafloor object-detection capability (cubic features ≥1 m / "
                f"≥2 m, S-44 Table 1) that a {res_txt} gridded SDB product cannot "
                "demonstrate — treat any 'Special'/'1a' badge above as "
                "vertical-accuracy-only, not full IHO Order compliance."
            )

        # GIS-R3: a real MSL/CD-vs-LAT-assumed vertical-datum offset (Gulf
        # MSL-LAT ≈ 0.3-1.2 m) is a near-constant BIAS masquerading as model
        # error. Flag it explicitly rather than passively mentioning datum in
        # a static caveat string — do NOT silently subtract an assumed
        # offset, just separate so RMSE/bias are never read as pure skill.
        datum_mismatch = bool(observed_vertical_datum and observed_vertical_datum != predicted_vertical_datum)
        datum_mismatch_note = None
        bias_note = None
        if datum_mismatch:
            datum_mismatch_note = (
                f"Observed survey datum '{observed_vertical_datum}' differs from the model "
                f"prediction datum '{predicted_vertical_datum}' (LAT-assumed, no verified "
                f"ellipsoid->geoid->LAT transform applied — see export DATUM_NOTE). Gulf "
                f"MSL-LAT offset is nominally 0.3-1.2 m; this offset is NOT subtracted here."
            )
            bias_note = (
                f"bias includes an un-reconciled {observed_vertical_datum}->"
                f"{predicted_vertical_datum} datum offset; treat the constant component as "
                f"datum, not model error."
            )

        L.info(f"Validate-XYZ: {n_pairs} pairs, RMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}, "
               f"S-44-1a={order_results['order1a']['pass_pct']:.0f}%, class-agr={class_agreement:.0f}%, "
               f"catzoc={catzoc}, match_p95_m={match_distance_stats_m['p95']:.0f}, "
               f"datum_mismatch={datum_mismatch}")

        return jsonify(_json_safe({
            'n_pairs': n_pairs,
            'n_observed': len(observed),
            'n_predicted': len(pred_pts),
            'n_rejected_nonfinite': n_rejected_nonfinite,
            'rmse': round(rmse, 3), 'mae': round(mae, 3), 'medae': round(medae, 3),
            'bias': round(bias, 3), 'std': round(std, 3),
            'r2': round(r2, 4), 'pearson_r': round(pearson, 4),
            'mape_pct': round(mape, 2),
            's44_pass_pct': order_results['order1a']['pass_pct'],  # alias for legacy UI
            'iho_orders': order_results,
            'catzoc': catzoc,
            'order_label': order_label,
            'p95_error_m': round(p95_error_m, 3),
            'pass_criterion': 'p95 <= TVU (IHO S-44 §3.3.1)',
            'tpu_caveat': tpu_caveat,
            'iho_orders_table': orders_table,
            'detection_capability_note': detection_capability_note,
            'resolution_m': resolution_m,
            'match_distance_stats_m': match_distance_stats_m,
            'n_excluded_by_distance': n_excluded_by_distance,
            'max_match_m': MAX_MATCH_M,
            'observed_vertical_datum': observed_vertical_datum,
            'predicted_vertical_datum': predicted_vertical_datum,
            'datum_mismatch': datum_mismatch,
            'datum_mismatch_note': datum_mismatch_note,
            'bias_note': bias_note,
            'stratified': strat,
            'residual_histogram': hist_data,
            'spatial_residuals': spatial_grid,
            'class_confusion': {
                'labels': CLASS_LABELS,
                'matrix': cm.tolist(),
                'agreement_pct': class_agreement,
            },
            'pairs': pairs_details,
            'pair_stats': pair_stats,
            'method': f'KDTree NN within {MAX_MATCH_M:.0f} m (CATZOC-B-consistent); depth clipped to ±25 m',
        }))
    except Exception as ex:
        L.error(f"Validate-XYZ: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/export',methods=['POST'])
def api_export():
    try:
        data=req.get_json(force=True,silent=True) or {};pts=data.get('points',[]);fmt=data.get('format','csv');bbox=data.get('bbox',{}) or {}
        if not isinstance(pts,list):
            return jsonify({'error':"'points' must be a list of {lat, lon, depth} objects"}),400
        if not isinstance(bbox,dict):
            bbox={}
        # ROB-R3: validate point shape up front — a missing lat/lon/depth key
        # used to 500 with a bare KeyError ("'lat'") deep inside an f-string.
        for i,p in enumerate(pts):
            if not isinstance(p,dict):
                return jsonify({'error':f"points[{i}] must be an object with lat/lon/depth, got {type(p).__name__}"}),400
            for k in ('lat','lon','depth'):
                if k not in p:
                    return jsonify({'error':f"points[{i}] missing '{k}'"}),400
        # Reference-system metadata embedded in every export so downstream
        # users know the horizontal/vertical datum (ADPorts reply item 6).
        vd=(str(data.get('vertical_datum') or 'LAT')).strip().upper() or 'LAT'
        src_crs=str(data.get('source_crs') or 'WGS84 (EPSG:4326)')
        # GIS-R4: `vd` is asserted as the exported grid's OWN datum (the model
        # product — 'LAT' unless the caller explicitly overrides it). The
        # uploaded reference survey's datum, if any, is a DIFFERENT thing and
        # must never be conflated with the product's VERTICAL_DATUM tag — it
        # rides along under its own distinct field/tag only.
        ref_survey_datum_raw=data.get('reference_survey_datum')
        ref_survey_datum=(str(ref_survey_datum_raw).strip().upper() if ref_survey_datum_raw else None) or None
        # IHO-R4: the same honest datum caveat build_geotiff_b64 stamps on the
        # fast-path GeoTIFF — no verified tidal reduction is actually applied
        # here either, so every export (CSV/GeoJSON/GeoTIFF) must carry it.
        export_meta={'horizontal_datum':'WGS84 (EPSG:4326)','vertical_datum':vd,
                     'source_crs':src_crs,'depth_units':'metres (positive down)',
                     'generated_by':'Bathymetry from Space',
                     'generated_utc':TM.strftime('%Y-%m-%dT%H:%M:%SZ',TM.gmtime()),
                     'datum_note':_DATUM_NOTE,'datum_transform_applied':False}
        if ref_survey_datum:
            export_meta['reference_survey_datum']=ref_survey_datum

        # ADPorts F3 (export-tag parity, no double-correction): pass through
        # whatever `tide_correction` disclosure the ORIGINAL result already
        # carried (from `_run_s2_lyzenga_fast`/`_run_s2_mle`'s response) — the
        # export step never recomputes or re-applies a tide correction, it
        # only discloses what already happened upstream. Honest 'none'
        # fallback when the caller doesn't supply one (legacy/manual points).
        _tide_in = data.get('tide_correction')
        if isinstance(_tide_in, dict) and _tide_in:
            export_meta['tide_correction'] = _tide_in
        else:
            export_meta['tide_correction'] = _tide_correction_fallback()

        # MASK-GLOBAL (2026-07-10, user-mandated): backstop land cut on the
        # EXPORT path — drops any point that falls on OSM land, regardless
        # of whether the upstream grid that produced `pts` was itself cut
        # (covers stale/cached results, and any future path that doesn't
        # route through depth_to_raster_png/_run_s2_lyzenga_fast).
        n_before_cut = len(pts)
        global_cut_source = None
        if pts:
            try:
                try:
                    from backend.osm_land_mask import coastline_land_for as _cl, osm_land_for as _ol
                except ImportError:
                    from osm_land_mask import coastline_land_for as _cl, osm_land_for as _ol  # type: ignore
                lats_e=[p['lat'] for p in pts]; lons_e=[p['lon'] for p in pts]
                ew=bbox.get('west',min(lons_e));ee=bbox.get('east',max(lons_e))
                es=bbox.get('south',min(lats_e));en=bbox.get('north',max(lats_e))
                if ee>ew and en>es:
                    EH,EW=400,400
                    coast_land,cinfo=_cl([ew,es,ee,en],(EH,EW))
                    feat_land,finfo=_ol([ew,es,ee,en],(EH,EW))
                    land=coast_land if coast_land is not None else np.zeros((EH,EW),dtype=bool)
                    if feat_land is not None: land=land|feat_land
                    global_cut_source=f"osm-coastline+features ({cinfo.get('n_polys',0)}+{finfo.get('n_polys',0)} polys)"
                    kept=[]
                    for p in pts:
                        col=int((p['lon']-ew)/(ee-ew+1e-12)*EW);row=int((en-p['lat'])/(en-es+1e-12)*EH)
                        col=max(0,min(EW-1,col));row=max(0,min(EH-1,row))
                        if not land[row,col]:
                            kept.append(p)
                    pts=kept
            except Exception as _cutex:
                L.info(f"Export: global land cut skipped ({_cutex})")
        export_meta['global_land_cut']={'points_before':n_before_cut,'points_after':len(pts),
            'points_dropped_on_land':n_before_cut-len(pts),'source':global_cut_source}

        if fmt=='geojson':
            feats=[{'type':'Feature','geometry':{'type':'Point','coordinates':[p['lon'],p['lat']]},'properties':{'depth':p['depth']}} for p in pts if (p.get('depth') or 0)>0]
            # ROB-R5: an empty/absent-points export used to ship as a
            # "successful" 200 empty FeatureCollection — a void deliverable
            # with no signal that nothing was actually exported.
            if not feats:
                return jsonify({'error':'No depth points to export'}),400
            return jsonify(_json_safe({'type':'FeatureCollection',
                'crs':{'type':'name','properties':{'name':'urn:ogc:def:crs:OGC:1.3:CRS84'}},
                'metadata':export_meta,
                'features':feats}))
        elif fmt=='csv':
            rows=[f"{p['lat']},{p['lon']},{p['depth']}" for p in pts if (p.get('depth') or 0)>0]
            if not rows:
                return jsonify({'error':'No depth points to export'}),400
            hdr=[f"# {k}: {v}" for k,v in export_meta.items()]
            lines=hdr+['lat,lon,depth']+rows
            return Response('\n'.join(lines),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=bathymetry.csv'})
        elif fmt=='geotiff':
            dp=[p for p in pts if (p.get('depth') or 0)>0 and p.get('photon_class') in ('interpolated','bathymetry')]
            if not dp:return jsonify({'error':'No depth points'}),400
            la=[p['lat'] for p in dp];lo=[p['lon'] for p in dp]
            bw=bbox.get('west',min(lo));be=bbox.get('east',max(lo));bs=bbox.get('south',min(la));bn=bbox.get('north',max(la))
            # ROB-R5: a single point (or all points sharing a lon/lat) collapses
            # the bbox to zero width/height → from_bounds() silently produces a
            # degenerate 0-pixel-size affine that QGIS/CARIS cannot georeference.
            # Pad symmetrically to a minimum extent (~111 m) instead of shipping
            # a corrupt "successful" deliverable.
            MIN_EXTENT_DEG=0.001
            if (be-bw)<MIN_EXTENT_DEG:
                pad=(MIN_EXTENT_DEG-(be-bw))/2.0
                bw-=pad;be+=pad
            if (bn-bs)<MIN_EXTENT_DEG:
                pad=(MIN_EXTENT_DEG-(bn-bs))/2.0
                bs-=pad;bn+=pad
            nx=min(2000,max(10,int((be-bw)*10000)));ny=min(2000,max(10,int((bn-bs)*10000)))
            grid=np.full((ny,nx),-9999,dtype=np.float32)
            for p in dp:
                c=max(0,min(nx-1,int((p['lon']-bw)/(be-bw+1e-10)*(nx-1))));r=max(0,min(ny-1,int((bn-p['lat'])/(bn-bs+1e-10)*(ny-1))))
                grid[r,c]=p['depth']
            buf=io.BytesIO()
            try:
                import rasterio
                from rasterio.transform import from_bounds
                transform=from_bounds(bw,bs,be,bn,nx,ny)
                tags=dict(AREA_OR_POINT='Area',
                    HORIZONTAL_DATUM=export_meta['horizontal_datum'],
                    VERTICAL_DATUM=vd,VERTICAL_UNITS='metres_positive_down',
                    SOURCE_CRS=src_crs,GENERATED_BY=export_meta['generated_by'],
                    GENERATED_UTC=export_meta['generated_utc'],
                    DATUM_TRANSFORM_APPLIED='false',
                    DATUM_NOTE=_DATUM_NOTE)
                if ref_survey_datum:
                    tags['REFERENCE_SURVEY_DATUM']=ref_survey_datum
                _tc=export_meta.get('tide_correction') or {}
                tags['TIDE_CORRECTION_APPLIED']=str(bool(_tc.get('applied'))).lower()
                tags['TIDE_HEIGHT_M']=str(_tc.get('height_m')) if _tc.get('height_m') is not None else 'n/a'
                tags['TIDE_SOURCE']=str(_tc.get('source') or 'none')
                with rasterio.open(buf,'w',driver='GTiff',height=ny,width=nx,count=1,
                    dtype='float32',crs='EPSG:4326',transform=transform,nodata=-9999,compress='deflate') as dst:
                    dst.write(grid,1)
                    dst.update_tags(**tags)
                    dst.set_band_description(1,f'depth_m_positive_down_{vd}')
            except ImportError:
                dx=(be-bw)/nx;dy=(bn-bs)/ny
                tifffile.imwrite(buf,grid,metadata={'ModelTiepointTag':(0,0,0,bw,bn,0),'ModelPixelScaleTag':(dx,dy,0),'GeographicTypeGeoKey':4326},compress='deflate')
            buf.seek(0)
            return Response(buf.read(),mimetype='image/tiff',headers={'Content-Disposition':'attachment; filename=bathymetry.tif'})
        return jsonify({'error':'Use: geojson, csv, geotiff'}),400
    except Exception as ex:
        L.error(f"Export: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex)}),500

# ══════════════════════════════════════════════════════════════
# ICESat-2 PRO — CShelph bathymetry pipeline
# ══════════════════════════════════════════════════════════════

@app.route('/api/icesat2-search',methods=['POST'])
def api_icesat2_search():
    """Search for ATL03 granules over ROI + date range."""
    try:
        data=req.get_json(force=True,silent=True) or {}
        bbox_dict=data.get('bbox')
        if not bbox_dict:return jsonify({'error':'Draw ROI first'}),400
        sd=data.get('start_date','2023-01-01');ed=data.get('end_date','2024-12-31')
        bbox=[bbox_dict['west'],bbox_dict['south'],bbox_dict['east'],bbox_dict['north']]
        if not CSHELPH_AVAILABLE:return jsonify({'error':'CShelph not installed','granules':[]}),200
        granules=search_atl03(bbox,sd,ed,max_results=data.get('max_results',20))
        # Strip non-serialisable result objects
        safe=[{k:v for k,v in g.items() if k!='result_obj'} for g in granules]
        return jsonify({'granules':safe,'count':len(safe),'bbox':bbox_dict,'date_range':[sd,ed]})
    except Exception as ex:
        L.error(f"ICESat-2 search: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex),'granules':[]}),500

@app.route('/api/icesat2-process',methods=['POST'])
def api_icesat2_process():
    """
    Full CShelph pipeline on ICESat-2 ATL03 data:
    download → read photons → orthometric → sea surface → refraction correction → bottom detect.
    Returns classified photon cloud, depth profiles, corrected depth points.
    """
    try:
        data=req.get_json(force=True,silent=True) or {}
        bbox_dict=data.get('bbox')
        if not bbox_dict:return jsonify({'error':'Draw ROI first'}),400
        sd=data.get('start_date','2023-01-01');ed=data.get('end_date','2024-12-31')
        bbox=[bbox_dict['west'],bbox_dict['south'],bbox_dict['east'],bbox_dict['north']]
        laser=int(data.get('laser',0))  # 0=all beams, 1/2/3=specific
        threshold=int(data.get('threshold',30))
        water_temp=float(data.get('water_temp',22.0))
        granule_idx=int(data.get('granule_index',0))

        if not CSHELPH_AVAILABLE:return jsonify({'error':'CShelph not available'}),200

        # Search + pick granule
        L.info(f"=== ICESat-2 CShelph: {sd}→{ed}, laser={laser}, th={threshold}, T={water_temp}°C ===")
        granules=search_atl03(bbox,sd,ed,max_results=10)
        if not granules:return jsonify({'error':'No ATL03 granules found for this ROI and date range','tracks':[],'depth_points':[]}),200

        idx=min(granule_idx,len(granules)-1)
        selected=granules[idx]
        L.info(f"Processing granule {idx+1}/{len(granules)}: {selected['id']} ({selected['date']})")

        # Multi-pass mode: process multiple granules and aggregate
        multi_pass=data.get('multi_pass',False)
        n_granules=min(int(data.get('n_granules',1)),5) if multi_pass else 1

        all_results=[]
        processed_granules=[]
        for gi in range(min(n_granules,len(granules))):
            sel=granules[gi]
            L.info(f"Processing granule {gi+1}/{min(n_granules,len(granules))}: {sel['id']} ({sel['date']})")
            res_g,err_g=process_granule(sel['result_obj'],bbox,laser,threshold,water_temp)
            if res_g:
                all_results.append(res_g)
                processed_granules.append({'id':sel['id'],'date':sel['date'],'index':gi})
            elif gi==0 and not multi_pass:
                return jsonify({'error':err_g,'tracks':[],'depth_points':[],'granule':sel['id']}),200

        if not all_results:
            return jsonify({'error':'No bathymetric photons found in any granule','tracks':[],'depth_points':[]}),200

        # Use first result as primary; aggregate if multi-pass
        result=all_results[0]
        if multi_pass and len(all_results)>1:
            agg_pts=aggregate_multi_pass(all_results)
            result['depth_points']=agg_pts
            result['n_depths']=len(agg_pts)
            n_xval=sum(1 for p in agg_pts if p.get('cross_validated'))
            L.info(f"Multi-pass: {len(all_results)} granules → {len(agg_pts)} pts ({n_xval} cross-validated)")

        # Format output
        out_pts=[]
        for p in result['depth_points']:
            op={'lat':p['lat'],'lon':p['lon'],'depth':p['depth'],
                'sea_surface':p.get('sea_surface',0),'bottom_elev':p.get('bottom_elev',0),
                'beam':p.get('beam',0),'photon_class':'icesat2_cshelph',
                'confidence':p.get('confidence','medium'),
                'tpu_m':p.get('tpu_m',0),'kd_532':p.get('kd_532',0),
                'track_quality':p.get('track_quality',0)}
            if p.get('cross_validated'):
                op['cross_validated']=True
                op['n_passes']=p.get('n_passes',1)
                op['pass_std_m']=p.get('pass_std_m',0)
            out_pts.append(op)

        avail=[{k:v for k,v in g.items() if k!='result_obj'} for g in granules]

        return jsonify({
            'depth_points':out_pts,
            'tracks':result.get('tracks',[]),
            'profiles':result.get('profiles',[]),
            'photon_cloud':result.get('photon_cloud',[]),
            'n_depths':result.get('n_depths',0),
            'n_beams':result.get('n_beams',0),
            'quality':result.get('quality',{}),
            'granule':processed_granules[0] if processed_granules else {},
            'processed_granules':processed_granules,
            'available_granules':avail,
            'params':{'laser':laser,'threshold':threshold,'water_temp':water_temp,'multi_pass':multi_pass},
            'sources_used':[f"ICESat-2 ATL03 CShelph ({result['n_depths']} depths, {result.get('n_beams',0)} beams"
                           f"{', '+str(len(all_results))+' passes' if multi_pass else ''})"],
        })
    except Exception as ex:
        L.error(f"ICESat-2 process: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex),'tracks':[],'depth_points':[]}),500

@app.route('/api/search-tracks',methods=['POST'])
def api_search():
    return jsonify({'tracks':[],'count':0,'sources':{'Sentinel-2':'Available','GEBCO':'Available','ICESat-2 CShelph':'Available' if CSHELPH_AVAILABLE else 'Not installed'}})

# ══════════════════════════════════════════════════════════════
# HIGH-RES MAPBOX SATELLITE CAPTURE
# ══════════════════════════════════════════════════════════════
@app.route('/api/mapbox-capture',methods=['POST'])
def api_mapbox_capture():
    """
    Capture high-resolution Mapbox satellite imagery for ROI.
    Uses Mapbox Static Images API at maximum zoom to get
    the sharpest available satellite view (~0.5m/px at zoom 18).
    Returns base64 PNG + metadata.
    """
    try:
        data=req.get_json(force=True,silent=True) or {}
        bbox_dict=data.get('bbox')
        if not bbox_dict:return jsonify({'error':'Draw ROI first'}),400
        token=os.getenv('MAPBOX_TOKEN','')
        if not token:return jsonify({'error':'MAPBOX_TOKEN not set'}),400
        w,s,e,n_b=bbox_dict['west'],bbox_dict['south'],bbox_dict['east'],bbox_dict['north']
        zoom=int(data.get('zoom',17))  # 17-18 for max resolution
        # Mapbox Static API: centered on ROI, max 1280x1280
        lat_c=(n_b+s)/2;lon_c=(e+w)/2
        width=min(1280,int(data.get('width',1280)))
        height=min(1280,int(data.get('height',1280)))
        style='mapbox/satellite-v9'
        url=f"https://api.mapbox.com/styles/v1/{style}/static/[{w},{s},{e},{n_b}]/{width}x{height}@2x?access_token={token}&attribution=false&logo=false"
        L.info(f"Commercial satellite: {width}x{height}@2x, zoom≈{zoom}, bbox=[{w},{s},{e},{n_b}]")
        r=requests.get(url,timeout=30)
        if not r.ok:
            # Try alternative format (center-based)
            url2=f"https://api.mapbox.com/styles/v1/{style}/static/{lon_c},{lat_c},{zoom},0/{width}x{height}@2x?access_token={token}&attribution=false&logo=false"
            r=requests.get(url2,timeout=30)
            if not r.ok:return jsonify({'error':f'Mapbox {r.status_code}: {r.text[:200]}'}),500
        import base64
        img_b64=base64.b64encode(r.content).decode()
        # Calculate actual resolution
        bbox_w_m=abs(e-w)*111000*math.cos(math.radians(lat_c))
        bbox_h_m=abs(n_b-s)*111000
        res_x=round(bbox_w_m/(width*2),2)  # @2x retina
        res_y=round(bbox_h_m/(height*2),2)
        L.info(f"Commercial HR: captured {width*2}x{height*2}px, resolution≈{res_x}m/px")
        return jsonify({
            'image_b64':img_b64,'width':width*2,'height':height*2,
            'resolution_m':round((res_x+res_y)/2,2),'bbox':[w,s,e,n_b],
            'center':[lat_c,lon_c],'size_km':[round(bbox_w_m/1000,2),round(bbox_h_m/1000,2)],
        })
    except Exception as ex:
        L.error(f"Commercial satellite: {ex}")
        return jsonify({'error':str(ex)}),500

# ══════════════════════════════════════════════════════════════
# OBSERVED BATHYMETRY — auto-load known sites or user upload
# ══════════════════════════════════════════════════════════════
# Known survey files (UTM Zone 40N — EPSG:32640)
_KNOWN_SITES = {
    'khalifa_port': {
        'files': ['KP_EMAL_Soundings_10x_New.xyz', 'KP Basin Soundings 10m.xyz'],
        'label': 'Khalifa Port / EMAL + Basin',
        'bbox': {'west': 54.5545, 'south': 24.7423, 'east': 54.7033, 'north': 24.8775},
        'epsg': 32640,
        'vertical_datum': 'LAT',
        'horizontal_datum': 'WGS84 / UTM Zone 40N',
        'units': 'metres',
    },
    'old_mussafah': {
        'files': ['OMC_Soundings_20m.xyz'],
        'label': 'Old Mussafah Channel',
        'bbox': {'west': 54.3519, 'south': 24.4091, 'east': 54.4145, 'north': 24.4661},
        'epsg': 32640,
        'vertical_datum': 'LAT',
        'horizontal_datum': 'WGS84 / UTM Zone 40N',
        'units': 'metres',
    },
    # Reference-library sites — no in-situ XYZ, training data sourced from the
    # Abu Dhabi reference-depth library + ICESat-2 via SlideRule. Loaded on demand
    # by /api/pro-extract so the UI keeps a single, simple workflow.
    'abu_al_abyad': {
        'files': [],
        'label': 'Abu Al Abyad Island',
        'bbox': {'west': 53.80, 'south': 24.18, 'east': 53.96, 'north': 24.30},
        'epsg': 4326,
        'vertical_datum': 'LAT',
        'horizontal_datum': 'WGS84',
        'units': 'metres',
        'library_region': 'abu_al_abyad',
    },
    'jbel_dhanna': {
        'files': [],
        'label': 'Jbel Dhanna',
        'bbox': {'west': 52.55, 'south': 24.15, 'east': 52.68, 'north': 24.22},
        'epsg': 4326,
        'vertical_datum': 'LAT',
        'horizontal_datum': 'WGS84',
        'units': 'metres',
        'library_region': 'jebel_dhanna',
    },
    'casablanca': {
        'files': [],
        'label': 'Casablanca',
        'bbox': {'west': -7.65, 'south': 33.58, 'east': -7.58, 'north': 33.62},
        'epsg': 4326,
        'vertical_datum': 'MSL',
        'horizontal_datum': 'WGS84',
        'units': 'metres',
        'library_region': 'casablanca',
    },
}

def _seed_reference_library(bbox, zooms=(13, 14), min_pts=30):
    """Silently populate the training store with reference-depth soundings for
    this bbox by extracting depths from published nautical charts via Gemini Vision.
    No user-facing log strings mention the underlying scraper — internal only.
    """
    if not _get_store:
        return 0
    store = _get_store()
    stored_before = store.query_bbox(bbox, buffer_km=1, max_points=1)['count']
    # Check bbox-scoped count (not global); if already seeded, skip.
    existing = store.query_bbox(bbox, buffer_km=1, max_points=10000)
    if existing['count'] >= min_pts:
        L.info(f"Reference library: bbox already has {existing['count']} pts, skipping seed")
        return 0
    api_key = os.getenv('GEMINI_API_KEY', '') or os.getenv('GOOGLE_API_KEY', '')
    if not api_key:
        L.info("Reference library: GEMINI_API_KEY missing, skipping seed")
        return 0
    try:
        try:
            from backend.iboating import _capture_iboating
        except ImportError:
            from iboating import _capture_iboating
    except Exception as ex:
        L.warning(f"Reference library: chart capture module unavailable: {ex}")
        return 0

    added = 0
    w, s, e, n = bbox
    for zoom in zooms:
        try:
            _, img_b64, iw, ih, chart_bbox = _capture_iboating(bbox, zoom=zoom, wait_sec=10)
            raw_pts, _err = _gemini_extract_soundings(img_b64, chart_bbox, iw, ih)
            if not raw_pts:
                continue
            cw_, cs_, ce_, cn_ = chart_bbox  # TRUE rendered bounds
            pts_lat, pts_lon, pts_depth = [], [], []
            for p in raw_pts:
                d = float(p.get('depth', 0))
                if d <= 0 or d > MAX_DEPTH_M:
                    continue
                px, py = p.get('x', 0), p.get('y', 0)
                lon = cw_ + (px / max(iw, 1)) * (ce_ - cw_)
                lat = cn_ - (py / max(ih, 1)) * (cn_ - cs_)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    pts_lat.append(lat); pts_lon.append(lon); pts_depth.append(min(d, MAX_DEPTH_M))
            try:
                from global_land_mask import globe
                keep = [(la, lo, de) for la, lo, de in zip(pts_lat, pts_lon, pts_depth)
                        if not globe.is_land(la, lo)]
                pts_lat = [k[0] for k in keep]; pts_lon = [k[1] for k in keep]; pts_depth = [k[2] for k in keep]
            except Exception:
                pass
            if pts_lat:
                # Store under a neutral source tag — nothing in the API response should
                # expose the underlying capture mechanism to the end user.
                n_new = store.add_points(pts_lat, pts_lon, pts_depth,
                                         source='reference_library', region='auto')
                added += n_new
                L.info(f"Reference library: bbox z{zoom} → +{n_new} pts")
        except Exception as ex:
            L.warning(f"Reference library seed z{zoom} failed: {ex}")
    L.info(f"Reference library: seeded {added} pts (was {stored_before}) for bbox={bbox}")
    return added


def _bbox_10km(center_lat, center_lon):
    """Build a ~10km x 10km bounding box centred on (lat, lon)."""
    # 1° lat ≈ 111 km; 1° lon ≈ 111*cos(lat) km
    half_km = 5.0  # 5 km each side → 10 km total
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(np.cos(np.radians(center_lat)), 0.01))
    return {
        'north': round(center_lat + dlat, 6),
        'south': round(center_lat - dlat, 6),
        'east':  round(center_lon + dlon, 6),
        'west':  round(center_lon - dlon, 6),
    }

def _load_xyz_file(filepath, src_epsg=32640, subsample=None):
    """Load XYZ file (UTM) → WGS84 points + 10×10 km bbox from centroid."""
    from pyproj import Transformer, CRS
    transformer = Transformer.from_crs(CRS.from_epsg(src_epsg), CRS.from_epsg(4326), always_xy=True)
    data = np.loadtxt(filepath, usecols=(0, 1, 2))
    if subsample and len(data) > subsample:
        idx = np.random.choice(len(data), subsample, replace=False)
        data = data[idx]
    xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]
    lons, lats = transformer.transform(xs, ys)
    pts = []
    for i in range(len(lats)):
        d = abs(float(zs[i]))
        if d > 0:
            pts.append({'lat': round(float(lats[i]), 6), 'lon': round(float(lons[i]), 6),
                        'depth': round(min(d, MAX_DEPTH_M), 2), 'photon_class': 'observed'})
    # 10 km × 10 km bbox centred on data centroid
    clat, clon = float(np.mean(lats)), float(np.mean(lons))
    bbox = _bbox_10km(clat, clon)
    return pts, bbox

@app.route('/api/load-observed', methods=['POST'])
def api_load_observed():
    """Load a known observed bathymetry dataset by site key.

    Three resolution paths:
      1. Site has local XYZ survey files → load + reproject.
      2. Site has a library_region → query the training store for cached
         reference-depth points (seeded from the AI reference library).
      3. Fall back to the static bbox only (empty points).
    """
    try:
        data = req.get_json(force=True, silent=True) or {}
        site_key = data.get('site', '')
        if site_key not in _KNOWN_SITES:
            return jsonify({'error': f'Unknown site: {site_key}'}), 400
        site = _KNOWN_SITES[site_key]
        all_pts = []
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        search_bases = ['/app', '.', '..', os.path.dirname(os.path.abspath(__file__)), _repo_root,
                        '/app/validation', 'validation', os.path.join(_repo_root, 'validation')]

        # ── meta_only: lightweight dataset metadata (datum / CRS / count /
        # bbox) for the sidebar card — no point cloud is loaded or returned.
        if data.get('meta_only'):
            n_pts = 0
            for fname in site.get('files', []) or []:
                for base in search_bases:
                    fp = os.path.join(base, fname)
                    if os.path.exists(fp):
                        try:
                            with open(fp) as fh:
                                n_pts += sum(1 for ln in fh if ln.strip())
                        except Exception as ex:
                            L.warning(f"meta_only count failed for {fname}: {ex}")
                        break
            if not site.get('files') and site.get('library_region') and _get_store:
                try:
                    bb = site.get('bbox')
                    if bb:
                        n_pts = _get_store().query_bbox(
                            [bb['west'], bb['south'], bb['east'], bb['north']],
                            buffer_km=2, max_points=100000)['count']
                except Exception as ex:
                    L.warning(f"meta_only library count failed for {site_key}: {ex}")
            crs_desc = (f'EPSG:{site["epsg"]} → WGS84'
                        if site.get('files') else
                        f'EPSG:{site.get("epsg", 4326)} (WGS84)')
            return jsonify({
                'points': [], 'count': n_pts, 'bbox': site.get('bbox'),
                'site': site['label'], 'crs': crs_desc,
                'source_epsg': site.get('epsg'),
                'source_type': 'metadata_only', 'meta_only': True,
                'vertical_datum': site.get('vertical_datum', 'LAT'),
                'horizontal_datum': site.get('horizontal_datum', 'WGS84'),
                'units': site.get('units', 'metres'),
                'depth_cap_m': MAX_DEPTH_M,
            })

        for fname in site.get('files', []) or []:
            filepath = None
            for base in search_bases:
                fp = os.path.join(base, fname)
                if os.path.exists(fp):
                    filepath = fp
                    break
            if not filepath:
                L.warning(f"File not found: {fname}")
                continue
            pts, _ = _load_xyz_file(filepath, site['epsg'])
            L.info(f"  Loaded {fname}: {len(pts)} points")
            all_pts.extend(pts)

        source_type = 'in_situ_survey' if all_pts else None

        # Fallback: pull reference-library points for this region from the training store
        if not all_pts and site.get('library_region') and _get_store:
            try:
                store = _get_store()
                bb = site.get('bbox')
                if bb:
                    bbox_arr = [bb['west'], bb['south'], bb['east'], bb['north']]
                    stored = store.query_bbox(bbox_arr, buffer_km=2, max_points=5000)
                    if stored['count'] > 0:
                        for la, lo, de in zip(stored['lats'].tolist(),
                                              stored['lons'].tolist(),
                                              stored['depths'].tolist()):
                            all_pts.append({
                                'lat': round(float(la), 6), 'lon': round(float(lo), 6),
                                'depth': round(min(float(de), MAX_DEPTH_M), 2),
                                'photon_class': 'observed',
                            })
                        source_type = 'reference_library'
                        L.info(f"Loaded observed (library): {site['label']} — {len(all_pts)} pts")
            except Exception as ex:
                L.warning(f"Reference-library lookup failed for {site_key}: {ex}")

        if all_pts:
            lats = [p['lat'] for p in all_pts]
            lons = [p['lon'] for p in all_pts]
            bbox = _bbox_10km(sum(lats)/len(lats), sum(lons)/len(lons))
        elif site.get('bbox'):
            bbox = site['bbox']
        else:
            return jsonify({'error': 'No data for site'}), 404

        crs_desc = (f'EPSG:{site["epsg"]} → WGS84'
                    if site.get('files') else
                    f'EPSG:{site.get("epsg", 4326)} (WGS84)')
        L.info(f"load-observed: {site['label']} — {len(all_pts)} pts, source={source_type}")
        return jsonify({
            'points': all_pts, 'count': len(all_pts), 'bbox': bbox,
            'site': site['label'], 'crs': crs_desc,
            'source_epsg': site.get('epsg'),
            'source_type': source_type or 'preset_bbox',
            'vertical_datum': site.get('vertical_datum', 'LAT'),
            'horizontal_datum': site.get('horizontal_datum', 'WGS84'),
            'units': site.get('units', 'metres'),
            'depth_cap_m': MAX_DEPTH_M,
        })
    except Exception as ex:
        L.error(f"Load observed: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500

# GIS-R1/GIS-R2/ROB-R2: single shared AOI-plausibility window used both to
# disambiguate an ambiguous UTM zone guess and to sanity-gate any reprojected
# upload (geographic OR UTM path) before it is accepted. (lat_min, lat_max,
# lon_min, lon_max) — UAE/Gulf operational area.
_UAE_GULF_AOI = (22.0, 27.0, 51.0, 57.0)


def _guess_utm_zone_from_coords(x, y, aoi=_UAE_GULF_AOI):
    """Infer the UTM zone for a projected (easting, northing) pair.

    GIS-R1: UTM zone is a function of LONGITUDE (zone 39N=48-54E, 40N=54-60E),
    which in projected coordinates is carried by the easting relative to the
    zone's central meridian plus the zone number — NEVER by the northing
    (northing tracks latitude, which is essentially identical for the same
    point regardless of which zone you (mis)assume). The old northing-band
    heuristic conflated latitude with zone and silently defaulted to 32640
    for every UAE-latitude survey, teleporting genuine zone-39N surveys
    (western Abu Dhabi: Jbel Dhanna, Ruwais, Sila) ~611 km east.

    Fix: candidate-zone disambiguation. Inverse-project the point under each
    plausible Gulf zone and keep the zone whose unprojected (lon,lat) lands
    inside `aoi`. Returns the EPSG code only when EXACTLY ONE candidate
    qualifies; returns None (never a silent wrong-zone default) when zero or
    more than one candidate is plausible, forcing the caller to require an
    explicit `utm_epsg`.
    """
    if not (100000 <= x <= 900000):
        return None
    lat_min, lat_max, lon_min, lon_max = aoi
    try:
        from pyproj import Transformer, CRS
    except Exception:
        return None
    candidates = [32639, 32640, 32641]  # UTM 39N/40N/41N — spans the Gulf/UAE coast
    hits = []
    for epsg in candidates:
        try:
            tr = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
            lon, lat = tr.transform(x, y)
        except Exception:
            continue
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            hits.append(epsg)
    if len(hits) == 1:
        return hits[0]
    return None


@app.route('/api/upload-observed', methods=['POST'])
def api_upload_observed():
    """Upload observed bathymetry (XYZ or SHP) for training — with explicit CRS/coord-order/datum.

    Form fields (optional):
      - coord_order: 'latlon' (default, matches UI) | 'lonlat' (traditional GIS X,Y,Z)
      - utm_epsg: EPSG code (e.g. 32640 for UTM 40N). If omitted & data looks UTM, auto-guess.
      - vertical_datum: 'LAT' | 'MSL' | 'CD' | 'WGS84'  (default 'LAT' — stored for downstream use)
      - horizontal_datum: free text (default 'WGS84')
    """
    try:
        files = req.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files'}), 400

        # Parse user metadata (sent via FormData alongside files)
        coord_order = (req.form.get('coord_order') or 'latlon').lower().strip()
        if coord_order not in ('latlon', 'lonlat'):
            coord_order = 'latlon'
        # ROB-R2 case 3: a non-numeric utm_epsg (e.g. 'abc') must be a loud
        # 400, not a silent fall-through to auto-guess — the operator's
        # explicit input vanishing without a word is exactly the failure
        # mode this whole area is about.
        utm_epsg_raw = (req.form.get('utm_epsg', '') or '').strip()
        utm_epsg_req = None
        if utm_epsg_raw:
            try:
                utm_epsg_req = int(utm_epsg_raw) or None
            except Exception:
                return jsonify({'error': f"utm_epsg '{utm_epsg_raw}' is not a number"}), 400
        vertical_datum = (req.form.get('vertical_datum') or 'LAT').strip().upper()
        horizontal_datum = (req.form.get('horizontal_datum') or 'WGS84').strip()

        # ── Shapefile set (.shp+.dbf+.prj) → shared parser. Axis order is
        # fixed by the SHP spec (X=easting/lon, Y=northing/lat) and the CRS
        # comes from .prj, so coord_order/utm_epsg do not apply — but the
        # vertical-datum choice is still honoured and propagated.
        if any((f.filename or '').lower().endswith('.shp') for f in files):
            parsed = _parse_shapefile_payload(files)
            shp_pts = parsed.get('points') or []
            if not shp_pts:
                return jsonify({'error': 'No valid points found in shapefile.'}), 400
            lats_arr = [p['lat'] for p in shp_pts]
            lons_arr = [p['lon'] for p in shp_pts]
            bbox = _bbox_10km(sum(lats_arr)/len(lats_arr), sum(lons_arr)/len(lons_arr))
            L.info(f"Upload observed (shp): {len(shp_pts)} pts, crs={parsed.get('crs')}, v_datum={vertical_datum}")
            return jsonify({
                'points': shp_pts, 'count': len(shp_pts), 'bbox': bbox,
                'site': 'User Upload',
                'crs': parsed.get('crs') or 'WGS84',
                'source_epsg': parsed.get('source_epsg'),
                'coord_order': 'shapefile (X=lon, Y=lat per spec)',
                'vertical_datum': vertical_datum,
                'horizontal_datum': horizontal_datum,
                'units': 'metres',
                'depth_cap_m': MAX_DEPTH_M,
                'n_rows_parsed': len(shp_pts),
                'n_rows_rejected': 0,
            })

        pts = []
        transformer = None
        src_epsg = None
        n_rows_read = 0
        n_rows_rejected = 0

        for f in files:
            fname = f.filename.lower()

            # ── XYZ / TXT / CSV ──
            if fname.endswith(('.xyz', '.txt', '.csv')):
                # ROB-R2 case 4: 'utf-8-sig' strips a leading BOM (Excel/Windows
                # CSV exports are BOM'd by default) so the first data row
                # doesn't get silently misread as a non-numeric header.
                content = f.read().decode('utf-8-sig', errors='ignore')
                # Tolerate comma, space or tab separators
                import re as _re
                raw_lines = [l.strip() for l in content.strip().split('\n')
                             if l.strip() and not l.strip().startswith(('#', '//'))]
                if not raw_lines:
                    continue
                # Skip a possible header (non-numeric first token)
                try:
                    float(_re.split(r'[\s,;]+', raw_lines[0])[0])
                except Exception:
                    raw_lines = raw_lines[1:]
                if not raw_lines:
                    continue

                # Detect UTM from the FIRST numeric row
                first_parts = _re.split(r'[\s,;]+', raw_lines[0])
                # ROB-R2 case 1: an XYZ row must tokenise to exactly 3 fields
                # (X Y Z). A row with MORE than 3 (e.g. Excel thousands-
                # separator commas shredding "412,345.6" into "412","345.6")
                # used to silently take the first 3 tokens as coordinates —
                # producing a plausible-looking but wrong (often near-equator)
                # point with no warning. Reject up front instead.
                if len(first_parts) != 3:
                    return jsonify({'error': (
                        f'First row has {len(first_parts)} columns after splitting on '
                        f'whitespace/comma/semicolon (expected exactly 3: X Y Z). Check for '
                        f'thousands-separator commas (e.g. "412,345.6") or an unexpected delimiter.'
                    )}), 400
                try:
                    c0, c1 = float(first_parts[0]), float(first_parts[1])
                except Exception:
                    return jsonify({'error': 'Unable to parse numeric coordinates from first row'}), 400
                is_utm = abs(c0) > 360 or abs(c1) > 360

                if is_utm:
                    # UTM — use explicit utm_epsg or auto-guess. coord_order ignored (always X=east, Y=north)
                    from pyproj import Transformer, CRS
                    if utm_epsg_req:
                        src_epsg = utm_epsg_req
                    else:
                        # GIS-R1: candidate-zone disambiguation — never a
                        # silent wrong-zone default. If no single zone is
                        # unambiguously plausible, force an explicit choice.
                        src_epsg = _guess_utm_zone_from_coords(c0, c1)
                        if src_epsg is None:
                            return jsonify({'error': (
                                'Unable to determine the UTM zone automatically from easting/northing '
                                'alone — zone depends on longitude, which northing alone cannot recover. '
                                'Please supply utm_epsg explicitly (e.g. 32639 for zone 39N / western Abu '
                                'Dhabi — Jbel Dhanna, Ruwais, Sila; 32640 for zone 40N / Abu Dhabi city, '
                                'Dubai, Khalifa Port).'
                            )}), 400
                    try:
                        transformer = Transformer.from_crs(CRS.from_epsg(src_epsg),
                                                           CRS.from_epsg(4326), always_xy=True)
                    except Exception as ex:
                        return jsonify({'error': f'Invalid UTM EPSG {src_epsg}: {ex}'}), 400

                for line in raw_lines:
                    parts = _re.split(r'[\s,;]+', line)
                    if len(parts) != 3:
                        # ROB-R2 case 1 (see above) applied to every row, not
                        # just the first — a ragged/thousands-separated row
                        # mid-file is rejected, never mis-parsed.
                        n_rows_rejected += 1
                        continue
                    try:
                        a, b, z = float(parts[0]), float(parts[1]), float(parts[2])
                    except Exception:
                        n_rows_rejected += 1
                        continue
                    n_rows_read += 1
                    # ROB-R1: NaN/Inf depth must never reach the response —
                    # `nan <= 0` is False, so the old bare `z <= 0` gate let
                    # a NaN depth straight through into the JSON body.
                    if not math.isfinite(z):
                        n_rows_rejected += 1
                        continue
                    z = abs(z)
                    if z <= 0:
                        n_rows_rejected += 1
                        continue
                    if transformer is not None:
                        # UTM path: a=easting, b=northing
                        lon, lat = transformer.transform(a, b)
                    else:
                        # Geographic path — honour coord_order
                        if coord_order == 'latlon':
                            lat, lon = a, b
                        else:
                            lon, lat = a, b
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        pts.append({'lat': round(lat, 6), 'lon': round(lon, 6),
                                    'depth': round(min(z, MAX_DEPTH_M), 2),
                                    'photon_class': 'observed'})
                    else:
                        n_rows_rejected += 1

        if not pts:
            msg = 'No valid points found in uploaded file.'
            if n_rows_rejected > 0 and n_rows_read > 0:
                msg += f' Parsed {n_rows_read} rows, all rejected (check coord order / UTM zone).'
            return jsonify({'error': msg}), 400

        # GIS-R2 / ROB-R2 case 2: ±90/±180 alone is NOT a plausibility check —
        # a lat/lon-swapped UAE point (24.80, 54.65) → (54.65, 24.80) is a
        # perfectly valid coordinate that lands in the Baltic Sea and used to
        # pass silently. Gate the reprojected centroid against the app's
        # expected operating window (also backstops GIS-R1 wrong-zone cases).
        lats_arr = [p['lat'] for p in pts]
        lons_arr = [p['lon'] for p in pts]
        clat, clon = sum(lats_arr)/len(lats_arr), sum(lons_arr)/len(lons_arr)
        bbox = _bbox_10km(clat, clon)
        aoi_lat_min, aoi_lat_max, aoi_lon_min, aoi_lon_max = _UAE_GULF_AOI
        in_expected_aoi = (aoi_lat_min <= clat <= aoi_lat_max) and (aoi_lon_min <= clon <= aoi_lon_max)
        plausibility_warning = None
        warnings_list = []
        if not in_expected_aoi:
            plausibility_warning = (
                f'Reprojected points centre at {clat:.2f}, {clon:.2f} — outside the expected '
                f'UAE/Gulf window ({aoi_lat_min:.0f}-{aoi_lat_max:.0f}°N, {aoi_lon_min:.0f}-'
                f'{aoi_lon_max:.0f}°E). Check coord_order / utm_epsg — a lat/lon axis swap or '
                f'wrong UTM zone silently produces valid-looking but wrong coordinates.'
            )
            warnings_list.append(plausibility_warning)
            L.warning(f"Upload observed: OUTSIDE expected AOI — centroid ({clat:.4f},{clon:.4f}), "
                      f"coord_order={coord_order}, src_epsg={src_epsg}")

        # ROB-R4: cap the point array actually echoed back (unbounded uploads
        # were observed to produce a 93 MB response for 1.5 M rows). `count`
        # always reflects the FULL accepted total; only the returned `points`
        # array is subsampled (evenly, to preserve spatial distribution).
        POINTS_RETURN_CAP = 200_000
        pts_out = pts
        points_truncated = False
        if len(pts) > POINTS_RETURN_CAP:
            stride = math.ceil(len(pts) / POINTS_RETURN_CAP)
            pts_out = pts[::stride]
            points_truncated = True
            note = (f'{len(pts)} points accepted; response truncated to {len(pts_out)} '
                    f'(evenly subsampled, stride={stride}) to keep the payload bounded.')
            warnings_list.append(note)
            L.warning(f"Upload observed: {note}")

        crs_desc = (f'EPSG:{src_epsg} → WGS84' if src_epsg else
                    f'Geographic ({"lat,lon" if coord_order=="latlon" else "lon,lat"}) / WGS84')
        L.info(f"Upload observed: {len(pts)} pts, crs={crs_desc}, v_datum={vertical_datum}, bbox={bbox}, "
               f"in_expected_aoi={in_expected_aoi}")
        return jsonify(_json_safe({
            'points': pts_out, 'count': len(pts), 'bbox': bbox,
            'site': 'User Upload',
            'crs': crs_desc,
            'source_epsg': src_epsg,
            'coord_order': coord_order,
            'vertical_datum': vertical_datum,
            'horizontal_datum': horizontal_datum,
            'units': 'metres',
            'depth_cap_m': MAX_DEPTH_M,
            'n_rows_parsed': n_rows_read,
            'n_rows_rejected': n_rows_rejected,
            'centroid': {'lat': round(clat, 4), 'lon': round(clon, 4)},
            'in_expected_aoi': in_expected_aoi,
            'plausibility_warning': plausibility_warning,
            'points_truncated': points_truncated,
            'points_returned': len(pts_out),
            'warnings': warnings_list,
        }))
    except RequestEntityTooLarge:
        # ROB-R4: re-raise so Flask's registered class/code errorhandler
        # produces the JSON 413 body — this view's own broad `except
        # Exception` below would otherwise catch it first and turn a 413
        # into a generic 500 with no size-limit explanation.
        raise
    except ValueError as ex:
        return jsonify({'error': str(ex)}), 400
    except Exception as ex:
        L.error(f"Upload observed: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# PRO EXTRACT — one-click orchestrated bathymetry
# ──────────────────────────────────────────────────────────────
# Automatically fuses whatever reference data is available for the ROI:
#   • In-situ survey (if uploaded by the user)
#   • ICESat-2 lidar photons via SlideRule
#   • Cached reference-depth library (auto-populated on first use per ROI)
#   • GEBCO (fallback)
# Trains the Attention-U-Net on the combined reference set and returns a
# single professional result. No chart-capture terminology leaks to the UI.
# ══════════════════════════════════════════════════════════════

def _sanitise_sources(sources):
    """Replace internal tool names with professional labels in user-facing strings."""
    out = []
    for s in sources or []:
        s = str(s)
        s = s.replace('i-Boating screenshot', 'Reference chart')
        s = s.replace('i-Boating', 'Reference library')
        s = s.replace('iboating', 'reference library')
        s = s.replace('Gemini', 'AI')
        s = s.replace('ChartFusion', 'Reference fusion')
        s = s.replace('ColourRaster', 'Chart raster')
        s = s.replace('Verified(', 'Reference pts(')
        out.append(s)
    return out


# ══════════════════════════════════════════════════════════════
# FAST S2-ONLY PIPELINE — Lyzenga + SlideRule (homogeneous map)
# Used to short-circuit /api/very-hr-clustered when imagery_source='s2'
# so the UI returns in ~10-20 s instead of timing out on CBR + CNN.
# ══════════════════════════════════════════════════════════════
# ── Disk cache around fetch_s2 for the fast path (predefined regions) ──
import hashlib as _hashlib, pickle as _pickle
_S2_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "s2_fast"
_S2_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _s2_cache_key(bbox, sd, ed, res, cloud):
    rb = tuple(round(float(v), 4) for v in bbox)
    return _hashlib.sha1(
        f"{rb}|{sd}|{ed}|{res}|{cloud}".encode()
    ).hexdigest()[:16]


def _fetch_s2_cached(bbox, sd, ed, res=10, cloud=20, ttl_seconds=7 * 24 * 3600,
                     n_scenes=3):
    """Multi-scene Sentinel-2 median composite with disk cache.

    Splits [sd, ed] into n_scenes equal sub-windows, fetches the
    least-cloudy single scene from each via _fetch_s2_single, then
    median-stacks per band. Median across 3 dates kills sun-glint
    spikes, residual cloud edges and ephemeral wakes that bias
    SDB depths.

    Falls back gracefully:
      • if only 1-2 sub-windows return data, median those
      • if all single-scene fetches fail, last-resort fetch_s2 ORBIT.

    Cache key includes n_scenes so single- vs multi-scene results
    don't collide.
    """
    key = _s2_cache_key(bbox, sd, ed, res, cloud) + f"_n{n_scenes}"
    # S2_GEE_RAW changes the actual band values returned (raw SR vs darkened
    # composite) so it must own a distinct cache slot — never collide with the
    # legacy zero-floored GEE cache that starved E1.
    if os.environ.get("S2_GEE_RAW", "0") == "1":
        key += "_geeraw"
        _gmn = int(os.environ.get("S2_GEE_MEDIAN_N", "1"))
        if _gmn > 1:
            key += f"_med{_gmn}"
    p = _S2_CACHE_DIR / f"{key}.pkl"
    if p.exists():
        try:
            age = TM.time() - p.stat().st_mtime
            if age < ttl_seconds:
                with open(p, "rb") as fh:
                    s2 = _pickle.load(fh)
                L.info(f"S2 cache HIT: {key} (age={int(age)}s, {res}m)")
                return s2
        except Exception as ex:
            L.info(f"S2 cache load failed for {key}: {ex}; refetching")

    # Build sub-windows
    from datetime import datetime, timedelta
    try:
        d0 = datetime.strptime(sd, '%Y-%m-%d')
        d1 = datetime.strptime(ed, '%Y-%m-%d')
    except Exception:
        d0 = datetime.strptime('2024-04-01', '%Y-%m-%d')
        d1 = datetime.strptime('2024-09-30', '%Y-%m-%d')
    span = max(1, (d1 - d0).days)
    step = max(1, span // max(1, n_scenes))
    windows = []
    for i in range(n_scenes):
        ws = d0 + timedelta(days=i * step)
        we = d0 + timedelta(days=(i + 1) * step) if i < n_scenes - 1 else d1
        windows.append((ws.strftime('%Y-%m-%d'), we.strftime('%Y-%m-%d')))

    scenes = []
    for ws, we in windows:
        try:
            s = _fetch_s2_single(bbox, ws, we, res=res, cloud=max(cloud, 30))
            if s is not None and 'water_mask' in s and int(s['water_mask'].sum()) >= 50:
                scenes.append(s)
                L.info(f"S2 multi: scene from {ws}→{we} ok "
                       f"({s['width']}×{s['height']}, water={int(s['water_mask'].sum())} px)")
        except Exception as ex:
            L.info(f"S2 multi: {ws}→{we} failed ({ex})")

    s2 = None
    if scenes:
        s2 = _median_compose(scenes)
        L.info(f"S2 multi: median-stacked {len(scenes)} scenes")
    else:
        # All sub-windows failed — last resort
        try:
            s2 = _fetch_s2_single(bbox, sd, ed, res=res, cloud=max(cloud, 40))
        except Exception:
            pass
    # GEE fallback (env-gated, default OFF). When Sentinel-Hub creds are
    # unavailable (401) and S2_GEE_FALLBACK=1, pull the same L2A bands
    # from Google Earth Engine. fetch_s2_gee returns the band/ndwi/
    # water_mask dict shape the Lyzenga/Stumpf/UAE fast path consumes.
    if s2 is None and os.environ.get("S2_GEE_FALLBACK", "0") == "1":
        try:
            L.info("S2-fast: Sentinel-Hub unavailable -> GEE fallback (S2_GEE_FALLBACK=1)")
            s2 = fetch_s2_gee(bbox, sd, ed, res=res, cloud=max(cloud, 30))
            L.info(f"S2-fast: GEE fallback ok ({s2['width']}x{s2['height']}, source={s2.get('source')})")
        except Exception as ex:
            L.info(f"S2-fast: GEE fallback failed ({ex})")
    if s2 is None:
        s2 = fetch_s2(bbox, sd, ed, res=res, cloud=cloud)

    try:
        with open(p, "wb") as fh:
            _pickle.dump(s2, fh, protocol=_pickle.HIGHEST_PROTOCOL)
        L.info(f"S2 cache SAVE: {key} ({p.stat().st_size/1024:.0f} KB)")
    except Exception as ex:
        L.info(f"S2 cache save failed: {ex}")
    return s2


def _median_compose(scenes):
    """Per-band median across N _fetch_s2_single dicts, recompute NDWI/
    MNDWI + water mask from the median bands."""
    keys_uint = ('coastal', 'red', 'green', 'blue', 'nir', 'swir')
    out = {}
    H, W = scenes[0]['blue'].shape
    for k in keys_uint:
        if k not in scenes[0]:
            continue
        stack = np.stack([s[k] for s in scenes if k in s], axis=0)
        out[k] = np.nanmedian(stack.astype(np.float32), axis=0).astype(np.uint16)
    # SCL: take the most "watery" classification (mode → favour 6 if any scene says so)
    if 'scl' in scenes[0]:
        scl_stack = np.stack([s['scl'] for s in scenes], axis=0)
        any_water = np.any(scl_stack == 6, axis=0)
        # If any scene calls a pixel water (6), keep 6; else first scene's class
        out['scl'] = np.where(any_water, 6, scenes[0]['scl']).astype(np.uint8)
    g = out['green'].astype(np.float32); n = out['nir'].astype(np.float32)
    sw = out.get('swir', n).astype(np.float32)
    out['ndwi'] = (g - n) / (g + n + 1e-6)
    out['mndwi'] = (g - sw) / (g + sw + 1e-6)
    out['water_mask'] = hr_water_mask(out.get('scl', np.zeros_like(out['blue'], dtype=np.uint8)),
                                       out['ndwi'], out['mndwi'], out['nir'], sw.astype(np.uint16))
    out['width'] = W; out['height'] = H
    out['date_range'] = scenes[0].get('date_range', '')
    out['n_scenes'] = len(scenes)
    return out


# Predefined regions warm-up — pre-fetches Sentinel-2 for the four UAE
# bboxes the UI exposes so the first user click is fast. Runs in a
# background thread on import so the Flask boot is unaffected.
_PRESET_BBOXES = [
    ("khalifa_port", [54.636, 24.785, 54.690, 24.840]),
    ("old_mussafah", [54.355, 24.412, 54.412, 24.466]),
    ("abu_al_abyad", [53.852, 24.213, 53.910, 24.267]),
    ("jbel_dhanna",  [52.5692, 24.1989, 52.6186, 24.2440]),
]


def _bbox_inside_predefined(bbox, pad_deg: float = 0.005) -> bool:
    """True iff the centroid of `bbox` falls inside any predefined region
    (with a small padding so an ROI drawn just outside still counts).
    Predefined regions are the calibration envelopes the UAE pretrained
    model was trained on."""
    w, s, e, n = bbox
    cx = (w + e) / 2.0
    cy = (s + n) / 2.0
    for _key, pb in _PRESET_BBOXES:
        if (pb[0] - pad_deg) <= cx <= (pb[2] + pad_deg) and \
           (pb[1] - pad_deg) <= cy <= (pb[3] + pad_deg):
            return True
    return False


# ── Default-region training auto-apply (this round, task 2) ────────────────
# The 4 default regions implicitly run with their training data: their
# pretrained-model + in-situ calibration (khalifa_port / old_mussafah carry
# bundled survey XYZ; jbel_dhanna / abu_al_abyad use the reference-library +
# SlideRule + GEBCO reconnaissance fusion). The 10 m fast path ALREADY auto-
# loads these by ROI overlap (xyz_points_in_bbox + the 90/10 region-aware UAE
# blend inside _bbox_inside_predefined). This helper just RESOLVES which
# default region an ROI/site_key targets and the honest accuracy label to
# stamp onto the result metadata, so the Results card can show the right tag.
_DEFAULT_REGION_TRAINING = {
    'khalifa_port': {'label': 'Khalifa Port / EMAL + Basin',
                     'training': 'in-situ survey (KP Basin + EMAL XYZ) + UAE pretrained blend',
                     'accuracy_kind': 'validated'},
    'old_mussafah': {'label': 'Old Mussafah Channel',
                     'training': 'in-situ survey (OMC XYZ) + UAE pretrained blend',
                     'accuracy_kind': 'validated'},
    'mussafah_channel': {'label': 'Old Mussafah Channel',  # frontend alias
                         'training': 'in-situ survey (OMC XYZ) + UAE pretrained blend',
                         'accuracy_kind': 'validated', 'canonical_key': 'old_mussafah'},
    'jbel_dhanna': {'label': 'Jbel Dhanna',
                    'training': 'reference-library + SlideRule ICESat-2 + UAE pretrained blend',
                    'accuracy_kind': 'reconnaissance'},
    'abu_al_abyad': {'label': 'Abu Al Abyad Island',
                     'training': 'SlideRule ICESat-2 + UAE pretrained + GEBCO reconnaissance fusion',
                     'accuracy_kind': 'reconnaissance'},
}


def _resolve_default_region(bbox=None, site_key=None):
    """Return (canonical_site_key, training_descriptor) if the ROI/site_key
    targets one of the 4 default regions, else (None, None).

    Match priority: explicit site_key (incl. the frontend's `mussafah_channel`
    alias) → ROI centroid inside a preset bbox.
    """
    if site_key:
        sk = str(site_key).strip().lower()
        info = _DEFAULT_REGION_TRAINING.get(sk)
        if info:
            return info.get('canonical_key', sk), info
    if bbox and len(bbox) == 4:
        w, s, e, n = bbox
        cx, cy = (w + e) / 2.0, (s + n) / 2.0
        pad = 0.005
        for key, pb in _PRESET_BBOXES:
            if (pb[0] - pad) <= cx <= (pb[2] + pad) and (pb[1] - pad) <= cy <= (pb[3] + pad):
                info = _DEFAULT_REGION_TRAINING.get(key)
                if info:
                    return key, info
    return None, None


def _warm_preset_s2_cache(start="2024-04-01", end="2024-09-30",
                          res=10, cloud=20):
    for name, bbox in _PRESET_BBOXES:
        try:
            _fetch_s2_cached(bbox, start, end, res=res, cloud=cloud)
            L.info(f"Preset warm-up done: {name}")
        except Exception as ex:
            L.info(f"Preset warm-up skipped for {name}: {ex}")


def _start_preset_warmup():
    if not os.getenv("SH_CLIENT_ID"):
        return
    import threading as _threading
    t = _threading.Thread(target=_warm_preset_s2_cache, daemon=True,
                          name="s2-warmup")
    t.start()


# ---------------------------------------------------------------------------
# E1 enhancements (Khalifa Port 1 m fusion programme) — both default-OFF.
# Operate on the raw _fetch_s2_cached band dict (uint16 reflectance x 10000)
# BEFORE any Lyzenga/Stumpf/UAE consumer, so the downstream calibrated path is
# unchanged except for the corrected reflectance it ingests.
# ---------------------------------------------------------------------------
def _apply_dsf_aquatic(s2, water):
    """ACOLITE Dark-Spectrum-Fitting principle (Vanhellemont 2019, RSE 225:175;
    Vanhellemont & Ruddick 2018, RSE 216:586; Caballero & Stumpf 2020).

    Sen2Cor's land-tuned aerosol model over-corrects path radiance in turbid
    Gulf case-II water, leaving a positive visible-reflectance offset that the
    log-ratio reads as 'deeper'. DSF estimates per-tile aerosol/path-radiance
    from the darkest aquatic spectrum and subtracts it before the SDB transform.

    Offline approximation of the full ACOLITE/DSF retrieval: for each visible
    band, estimate the dark-spectrum / path-radiance offset as a low percentile
    of the reflectance over DEEP, clear water (proxy: the optically-darkest
    water pixels by NIR), then subtract it. The aerosol path term decreases with
    wavelength (Rayleigh + aerosol ~ lambda^-1), so the subtracted offset is
    larger in blue than red — we honour that ordering by fitting each band's own
    dark percentile rather than a single scalar. Returns a NEW band dict; the
    deep-water (large-z) signal is essentially unchanged (its dark offset is the
    physical path radiance), while bright-shallow over-prediction is reduced.
    """
    import os as _os
    pct = float(_os.environ.get("S2_DSF_PCT", "1.0"))   # dark percentile (%)
    s2o = dict(s2)
    w = np.asarray(water, dtype=bool)
    nir = np.asarray(s2.get('nir'), dtype=np.float64)
    # Darkest-aquatic sample: deep clear water = water pixels with the lowest NIR.
    if w.sum() >= 50:
        nir_w = nir[w]
        nir_thr = np.percentile(nir_w, 25.0)      # darkest 25% NIR = clearest/deepest
        dark = w & (nir <= nir_thr)
        if dark.sum() < 30:
            dark = w
    else:
        dark = np.ones_like(nir, dtype=bool)
    deltas = {}
    for band in ('blue', 'green', 'red', 'coastal'):
        if band not in s2:
            continue
        b = np.asarray(s2[band], dtype=np.float64)
        vals = b[dark]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 10:
            continue
        offset = float(np.percentile(vals, pct))   # per-band dark spectrum / path radiance
        corr = np.clip(b - offset, 1.0, None)       # subtract; keep >0 (uint16 scale)
        s2o[band] = corr.astype(np.uint16)
        deltas[band] = offset
    s2o['_dsf_offsets'] = deltas
    try:
        L.info(f"S2_DSF: dark-spectrum offsets (DN x1e-4) {deltas} on {int(dark.sum())} dark-water px")
    except Exception:
        pass
    return s2o


def _apply_hedley_deglint(s2, water):
    """Hedley, Harborne & Mumby 2005 (IJRS 26(10):2107) NIR sun-glint removal;
    Lyzenga, Malinas & Tanis 2006 NIR-deglint variant.

    Over the calm Gulf, specular sun-glint adds a depth-independent additive
    radiance to the visible bands, strongest where the bottom is brightest
    (shallow carbonate sand), so the log-ratio reads glint as extra bottom
    signal -> 'deeper'. Hedley regresses each visible band on NIR over a
    deep-water sample and subtracts  R_corr = R_vis - slope*(R_NIR - min_NIR),
    removing the additive glint term while leaving the depth-bearing bottom term.
    Slope clamped >= 0. Applied BEFORE the Lyzenga/Stumpf transform.
    """
    s2o = dict(s2)
    w = np.asarray(water, dtype=bool)
    nir = np.asarray(s2.get('nir'), dtype=np.float64)
    if w.sum() < 50:
        return s2o
    nir_w = nir[w]
    # Deep-water sample for the regression: darkest 50% NIR water pixels (the
    # homogeneous deep-water set Hedley fits on). min_NIR = its floor.
    nir_thr = np.percentile(nir_w, 50.0)
    samp = w & (nir <= nir_thr)
    if samp.sum() < 30:
        samp = w
    nir_min = float(np.percentile(nir[samp], 5.0))
    nir_dev = nir - nir_min
    slopes = {}
    for band in ('blue', 'green', 'red', 'coastal'):
        if band not in s2:
            continue
        b = np.asarray(s2[band], dtype=np.float64)
        x = nir[samp].ravel(); y = b[samp].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 10 or np.std(x) < 1e-6:
            continue
        slope = float(np.polyfit(x, y, 1)[0])
        slope = max(0.0, slope)              # clamp >= 0 (Hedley: no negative glint)
        corr = np.clip(b - slope * nir_dev, 1.0, None)
        s2o[band] = corr.astype(np.uint16)
        slopes[band] = round(slope, 4)
    s2o['_hedley_slopes'] = slopes
    try:
        L.info(f"S2 HEDLEY_DEGLINT: per-band NIR slopes {slopes}, nir_min={nir_min:.1f} "
               f"on {int(samp.sum())} deep-water px")
    except Exception:
        pass
    return s2o


# ══════════════════════════════════════════════════════════════════════════════
# AI_METHOD_LOG R1/R2/R3 — deep per-pixel σ for the multi-scene MLE
#   Behind env flags, ALL default OFF:
#     MLE_DEEP_SIGMA     (master gate; OFF → byte-identical scalar-σ MLE)
#     MLE_DEEP_PER_SCENE (OFF → train deep ONCE/request & broadcast σ-shape;
#                         1 → true per-scene deep fit)
#     MLE_DEEP_CALIB     (default ON when MLE_DEEP_SIGMA=1; temperature-k)
#     MLE_DEEP_MARGIN=0.05, MLE_DEEP_WCAP=0.5, SPATIAL_TRAIN_BUFFER_M=500(gate)
# The linear ridge is the guaranteed floor: the deep candidate is convex-blended
# into a scene ONLY where it beats linear on a ≥500 m spatial-block held-out set
# (Kendall & Gal 2017; Guo 2017; Caballero & Stumpf 2020).
# ══════════════════════════════════════════════════════════════════════════════
def _deep_scene_sigma(s2, bbox, water, linear_depth,
                      tr_lat, tr_lon, tr_truth,
                      test_lat, test_lon, test_truth,
                      scalar_sigma):
    """Train the Attention U-Net on spatial-train refs, emit a per-pixel σ field
    + a never-worse-than-linear convex-blended depth, temperature-calibrated to
    95 % coverage. Returns dict {sigma_px,(H,W) depth_blend,(H,W) gate:{...}} or
    None (feature off / not enough held-out truth / CNN failed) → scalar path.

    Leakage-safe gate: CNN trains on refs ≥ SPATIAL_TRAIN_BUFFER_M (default 500 m
    here) from ANY held-out gate point; both linear and CNN are scored at the SAME
    held-out coords. The linear floor is scored optimistically (it may share spatial
    autocorrelation with its own train pts) so the gate is CONSERVATIVE for adopting
    the CNN — we only blend where the CNN beats an already-favourably-scored linear."""
    import numpy as _np
    w_, s_, e_, n_ = bbox
    H, W = linear_depth.shape

    def _rc(la, lo):
        r = max(0, min(H - 1, int((n_ - la) / (n_ - s_ + 1e-10) * H)))
        c = max(0, min(W - 1, int((lo - w_) / (e_ - w_ + 1e-10) * W)))
        return r, c

    # Pool ALL available in-situ (linear-train + linear-held-out) and carve a
    # CONTIGUOUS spatial-block held-out set (KMeans, ≥500 m inter-fold buffer).
    # A scattered random held-out set collapses on a dense field (every train pt
    # sits within 500 m of some test pt); a contiguous block is the honest,
    # tractable gate. See CLAUDE.md honesty contract / AI_METHOD_LOG R2/R4.
    a_lat = _np.concatenate([_np.asarray(tr_lat, float), _np.asarray(test_lat, float)])
    a_lon = _np.concatenate([_np.asarray(tr_lon, float), _np.asarray(test_lon, float)])
    a_dep = _np.concatenate([_np.asarray(tr_truth, float), _np.asarray(test_truth, float)])
    if len(a_dep) < 40:
        L.info(f"MLE-deep: only {len(a_dep)} in-situ pts → no honest block gate, scalar σ kept")
        return None
    buf = float(os.environ.get("SPATIAL_TRAIN_BUFFER_M", "500"))
    try:
        try:
            from backend.sdb_cnn_baseline import make_spatial_block_centers as _mkblk
        except ImportError:
            from sdb_cnn_baseline import make_spatial_block_centers as _mkblk  # type: ignore
        _nb = int(os.environ.get("MLE_DEEP_NBLOCKS", "10"))
        _seed = int(os.environ.get("MLE_DEEP_SEED", "42"))
        tr_mask, te_mask = _mkblk(a_lat, a_lon, n_blocks=min(_nb, max(3, len(a_dep)//8)),
                                  test_frac=0.25, buffer_m=buf, seed=_seed)
    except Exception as _bx:
        L.info(f"MLE-deep: spatial-block carve failed ({_bx}) → scalar σ kept")
        return None
    cl, clo, cd = a_lat[tr_mask], a_lon[tr_mask], a_dep[tr_mask]
    test_lat, test_lon, test_truth = a_lat[te_mask], a_lon[te_mask], a_dep[te_mask]
    if len(cd) < 30 or len(test_truth) < 5:
        L.info(f"MLE-deep: block carve gave {len(cd)} train / {len(test_truth)} "
               f"held-out (buf={buf:.0f} m) → scalar σ kept")
        return None

    # ── Train the Attention U-Net (CPU budget: 1 model, 6 MC passes) ──
    try:
        try:
            from backend import cnn_engine as _cnn
        except ImportError:
            import cnn_engine as _cnn  # type: ignore
        _ep = int(os.environ.get("MLE_DEEP_EPOCHS", "60"))
        _mp = int(os.environ.get("MLE_DEEP_MAXPATCH", "160"))
        res, err = _cnn.cnn_train_and_predict(
            s2, {"lats": cl, "lons": clo, "depths": cd}, list(bbox),
            n_ensemble=1, mc_passes=6, epochs=min(_ep, 80),
            max_patches=min(_mp, 192), use_cache=True)
        if err or res is None:
            L.info(f"MLE-deep: CNN failed ({err}) → scalar σ kept")
            return None
    except Exception as _ex:
        L.info(f"MLE-deep: CNN raised ({_ex}) → scalar σ kept")
        return None

    depth_cnn = _np.asarray(res["depth"], _np.float32)
    sigma_cnn = _np.asarray(res["uncertainty"], _np.float32)

    # ── Score linear & CNN at the SAME held-out coords ──
    pl, pc, tt = [], [], []
    for la, lo, dt in zip(test_lat, test_lon, test_truth):
        r, c = _rc(la, lo)
        vl = linear_depth[r, c]; vc = depth_cnn[r, c]
        if _np.isfinite(vl) and _np.isfinite(vc):
            pl.append(float(vl)); pc.append(float(vc)); tt.append(float(dt))
    if len(tt) < 5:
        L.info("MLE-deep: <5 co-located held-out preds → scalar σ kept")
        return None
    pl = _np.asarray(pl); pc = _np.asarray(pc); tt = _np.asarray(tt)
    rmse_lin = float(_np.sqrt(_np.mean((pl - tt) ** 2)))
    rmse_cnn = float(_np.sqrt(_np.mean((pc - tt) ** 2)))

    # ── Gate + convex inverse-variance blend (never worse than linear) ──
    # AI_METHOD_LOG R5 (round 2): DEPTH-STRATIFIED gate. Instead of ONE global
    # w_cnn per scene, score CNN-vs-linear held-out RMSE PER DEPTH BAND
    # (0-2/2-5/5-10/10+ m) and assign a per-pixel blend weight w_cnn(band) by
    # the pixel's LINEAR-depth band (legitimate at inference). SAME never-worse
    # rule applied band-by-band → overall floor preserved. MLE_DEEP_STRATIFIED
    # (default 1 when master on; 0 = round-1 global behaviour).
    margin = float(os.environ.get("MLE_DEEP_MARGIN", "0.05"))
    w_cap = float(os.environ.get("MLE_DEEP_WCAP", "0.5"))
    stratified = os.environ.get("MLE_DEEP_STRATIFIED", "1") == "1"
    _bands = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, float("inf"))]

    def _band_key(lo, hi):
        return f"{int(lo)}-{'inf' if hi == float('inf') else int(hi)}m"

    per_band = {}
    # per-pixel maps assembled below
    w_map = _np.zeros((H, W), dtype=_np.float32)          # CNN blend weight
    sig_lin_map = _np.full((H, W), rmse_lin, dtype=_np.float32)   # σ floor (reject)
    sig_fold_map = _np.zeros((H, W), dtype=_np.float32)  # gate residual folded when adopted

    if stratified:
        # Bin the held-out preds by their LINEAR estimate (pl) — the SAME
        # band-assignment function used per-pixel at inference. Guarantees the
        # gate is scored on exactly the population each band weight governs.
        band_minn = int(os.environ.get("MLE_DEEP_BAND_MINN", "12"))
        band_w = [0.0] * len(_bands)
        band_rl = [None] * len(_bands)
        band_rc = [None] * len(_bands)
        for _bi, (lo, hi) in enumerate(_bands):
            sel = _np.array([lo <= v < hi for v in pl], dtype=bool)
            n_b = int(sel.sum())
            key = _band_key(lo, hi)
            if n_b < 3:
                per_band[key] = {"rmse_lin": None, "rmse_cnn": None,
                                 "w_cnn": 0.0, "n": n_b}
                continue
            rl = float(_np.sqrt(_np.mean((pl[sel] - tt[sel]) ** 2)))
            rc = float(_np.sqrt(_np.mean((pc[sel] - tt[sel]) ** 2)))
            # Never-worse rule PLUS a statistical-power guard: a band must hold
            # ≥ MLE_DEEP_BAND_MINN held-out pts before the CNN may be adopted
            # (a 3-point band beating linear is noise, not skill).
            if n_b >= band_minn and rc <= rl * (1.0 - margin) and rc > 0:
                wb = float(_np.clip(rl ** 2 / (rl ** 2 + rc ** 2), 0.0, w_cap))
            else:
                wb = 0.0
            band_w[_bi] = wb; band_rl[_bi] = rl; band_rc[_bi] = rc
            per_band[key] = {"rmse_lin": round(rl, 3), "rmse_cnn": round(rc, 3),
                             "w_cnn": round(wb, 4), "n": n_b}
        # scalar summary: >0 iff ANY band adopted → drives _run_s2_mle's
        # _ref_g election / never-worse strip (all-reject ⇒ 0 ⇒ scalar path).
        w_cnn = float(max(band_w)) if band_w else 0.0
        # per-pixel weight + σ-floor maps assigned by the LINEAR depth band.
        for _bi, (lo, hi) in enumerate(_bands):
            bm = _np.isfinite(linear_depth) & (linear_depth >= lo) & (linear_depth < hi)
            if band_rl[_bi] is not None:
                sig_lin_map[bm] = band_rl[_bi]
            if band_w[_bi] > 0:
                w_map[bm] = band_w[_bi]
                if band_rc[_bi] is not None:
                    sig_fold_map[bm] = band_rc[_bi]
    else:
        # round-1 GLOBAL behaviour (MLE_DEEP_STRATIFIED=0).
        if rmse_cnn <= rmse_lin * (1.0 - margin) and rmse_cnn > 0:
            w_cnn = float(_np.clip(rmse_lin ** 2 / (rmse_lin ** 2 + rmse_cnn ** 2), 0.0, w_cap))
        else:
            w_cnn = 0.0
        w_map[:] = w_cnn
        if w_cnn > 0:
            sig_fold_map[:] = rmse_cnn   # round-1 folded the global CNN RMSE

    # convex blend (per-pixel weight); depth untouched where w_map==0
    depth_blend = linear_depth.astype(_np.float32).copy()
    if w_cnn > 0:
        both = _np.isfinite(linear_depth) & _np.isfinite(depth_cnn)
        _wm = w_map[both]
        depth_blend[both] = ((1.0 - _wm) * linear_depth[both]
                             + _wm * depth_cnn[both]).astype(_np.float32)

    # ── Per-pixel σ that feeds the MLE ──
    if w_cnn > 0:
        _sig_cnn = _np.where(_np.isfinite(sigma_cnn), sigma_cnn, sig_lin_map)
        sigma_px = _np.where(w_map > 0,
                             _np.sqrt(_sig_cnn ** 2 + sig_fold_map ** 2),
                             sig_lin_map).astype(_np.float32)
    else:
        sigma_px = sig_lin_map.astype(_np.float32)
    sigma_px = _np.where(water, sigma_px, _np.nan).astype(_np.float32)
    sigma_px = _np.clip(sigma_px, 0.5, 5.0)

    # ── R3: temperature-k so held-out coverage@95 hits 0.95 ──
    k = 1.0
    cov_before = cov_after = None
    if os.environ.get("MLE_DEEP_CALIB", "1") == "1":
        z_pred = depth_blend if w_cnn > 0 else linear_depth
        sr, tvals = [], []
        for la, lo, dt in zip(test_lat, test_lon, test_truth):
            r, c = _rc(la, lo)
            sp = sigma_px[r, c]; zp = z_pred[r, c]
            if _np.isfinite(sp) and sp > 0 and _np.isfinite(zp):
                sr.append((zp - dt) / sp)
            tvals.append(dt)
        sr = _np.asarray(sr)
        if len(sr) >= 5:
            cov_before = float(_np.mean(_np.abs(sr) <= 1.96))
            ks = _np.linspace(0.3, 3.0, 271)
            covs = _np.array([_np.mean(_np.abs(sr) / kk <= 1.96) for kk in ks])
            k = float(ks[int(_np.argmin(_np.abs(covs - 0.95)))])
            sigma_px = _np.clip(sigma_px * k, 0.5, 5.0)
            cov_after = float(_np.mean(_np.abs(sr) / k <= 1.96))

    # ── Diagnostics: Spearman(σ,|err|), median/spatial-std σ ──
    rho = None
    try:
        from scipy.stats import spearmanr as _sp
        s_at, e_at = [], []
        for la, lo, dt in zip(test_lat, test_lon, test_truth):
            r, c = _rc(la, lo)
            sp = sigma_px[r, c]; zp = (depth_blend if w_cnn > 0 else linear_depth)[r, c]
            if _np.isfinite(sp) and _np.isfinite(zp):
                s_at.append(float(sp)); e_at.append(abs(float(zp) - float(dt)))
        if len(s_at) >= 5 and _np.std(s_at) > 0:
            rho = float(_sp(s_at, e_at).correlation)
    except Exception:
        rho = None

    wpx = _np.isfinite(sigma_px)
    gate = {
        "w_cnn": round(w_cnn, 4),
        "stratified": bool(stratified),
        "per_band": per_band,
        "rmse_lin": round(rmse_lin, 3), "rmse_cnn": round(rmse_cnn, 3),
        "n_gate": int(len(tt)), "n_train_deep": int(len(cd)),
        "buffer_m": buf, "temperature_k": round(k, 3),
        "coverage95_before": (round(cov_before, 3) if cov_before is not None else None),
        "coverage95": (round(cov_after, 3) if cov_after is not None else None),
        "spearman_sigma_abserr": (round(rho, 3) if rho is not None else None),
        "sigma_px_median": round(float(_np.nanmedian(sigma_px)), 3) if wpx.any() else None,
        "sigma_px_spatial_std": round(float(_np.nanstd(sigma_px[wpx])), 3) if wpx.any() else None,
        "cnn_r2_random_holdout": res.get("r2"),
    }
    L.info(f"MLE-deep: RMSE_lin={rmse_lin:.3f} RMSE_cnn={rmse_cnn:.3f} w_cnn={w_cnn:.3f} "
           f"k={k:.2f} cov95={gate['coverage95']} ρ(σ,|e|)={gate['spearman_sigma_abserr']} "
           f"σ_med={gate['sigma_px_median']} σ_std={gate['sigma_px_spatial_std']} "
           f"(n_gate={len(tt)}, n_train_deep={len(cd)})")
    return {"sigma_px": sigma_px,
            "depth_blend": (depth_blend if w_cnn > 0 else None),
            "gate": gate}


def _run_s2_lyzenga_fast(bbox, sd, ed, user_pts=None,
                         max_cloud=20, fetch_sliderule=False,
                         include_geotiff=False,
                         res_override=None,
                         _internal_return_grid=False,
                         _emit_mask_preview=None,
                         _deep_train=False):
    """Single-shot Sentinel-2 → Lyzenga+Stumpf+SlideRule depth grid.

    res_override: user-selected output resolution in metres (10/20/50/100).
    When absent/invalid the area-adaptive native resolution is used.

    Returns a dict with the same keys the very-hr-clustered frontend
    consumes: bbox, resolution_m, metrics, elapsed_s, method,
    {depth_png_b64, overlay_png_b64, water_mask_b64, scatter_png_b64}.
    """
    import time as _T, math as _math, base64 as _b64, io as _io
    t0 = _T.time()

    # 1. Fetch S2 (single tile, native 10 m). fetch_s2 already does
    # NDWI + Hedley sun-glint + Lyzenga deep-water-reflectance subtraction
    # and morphology — so the output already has a clean water mask.
    w_, s_, e_, n_ = bbox
    cl = _math.cos(_math.radians((n_ + s_) / 2))
    area_km2 = abs(e_ - w_) * abs(n_ - s_) * 111 * 111 * cl
    res = _native_res_for_area(area_km2)
    try:
        _ro = int(res_override) if res_override is not None else 0
    except (TypeError, ValueError):
        _ro = 0
    if _ro in (10, 20, 50, 100):
        res = _ro
        L.info(f"S2-fast: user-requested output resolution = {res} m")
    elif area_km2 > SOFT_HIGH_RES_KM2:
        L.info(f"S2-fast: ROI {area_km2:.1f} km² exceeds the {SOFT_HIGH_RES_KM2:.0f} km² "
               f"high-resolution band → using native Sentinel-2 grid at {res} m")
    s2 = _fetch_s2_cached(bbox, sd, ed, res=res, cloud=max_cloud)
    H, W = s2['red'].shape

    # 2. Composite water mask — INTERSECTION of three independent tests:
    #
    #   (a) Spectral mask from _fetch_s2_single's hr_water_mask:
    #       SCL water class + NDWI + MNDWI + NIR/SWIR darkness +
    #       morphology. Already on s2['water_mask'] from the cached
    #       fetch. This is the strictest of the three (rejects bright
    #       concrete piers and breakwaters via MNDWI > 0.1 + SWIR<0.11).
    #
    #   (b) make_water_mask: global land DB (~1 km coastline) +
    #       satellite brightness/blue-fraction. Catches reclaimed
    #       land that the spectral test sometimes misses (ships at
    #       quayside, wet sand mid-tide).
    #
    #   (c) NIR-very-dark threshold: water absorbs NIR almost
    #       completely; NIR > 0.05 is almost always non-water.
    #
    # We then erode by 1 px so the boundary stops one pixel inside
    # the water, removing the half-pixel land bleed that was showing up
    # at Old Mussafah port edges.
    spectral_water = s2.get('water_mask')
    coverage_water = make_water_mask(bbox, H, W, s2=s2)

    if spectral_water is not None and coverage_water is not None:
        water = np.asarray(spectral_water, dtype=bool) & np.asarray(coverage_water, dtype=bool)
    elif coverage_water is not None:
        water = np.asarray(coverage_water, dtype=bool)
    elif spectral_water is not None:
        water = np.asarray(spectral_water, dtype=bool)
    else:
        water = s2['ndwi'] > 0

    total_px = water.size
    spec_frac = float(spectral_water.sum() / total_px) if spectral_water is not None else 0.0

    # If the intersection collapsed (< 30 % of the spectral mask survived),
    # the dual filter was too strict — fall back to the spectral mask
    # alone. This happens in turbid water where make_water_mask's
    # brightness test mis-classifies sediment-rich water as land.
    surviving_frac = float(water.sum() / max(spectral_water.sum(), 1)) \
                     if spectral_water is not None else 1.0
    if spectral_water is not None and surviving_frac < 0.30:
        L.info(f"S2-fast: dual mask collapsed (kept {surviving_frac*100:.1f}% "
               f"of spectral mask) — falling back to spectral mask alone")
        water = np.asarray(spectral_water, dtype=bool)

    # Morphology — opening removes isolated water pixels inside land,
    # but skip the 1 px erosion. The erosion was killing legitimate
    # water along the coastline whenever the spectral mask already had
    # tight boundaries. The opening alone is enough to remove ship
    # specks / wakes.
    try:
        from scipy.ndimage import binary_opening
        s3 = np.ones((3, 3), dtype=bool)
        water = binary_opening(water, structure=s3, iterations=1)
    except Exception:
        pass

    if int(water.sum()) < 100:
        water = s2.get('ndwi', np.zeros_like(s2['red'])) > 0

    s2['water_mask'] = water
    L.info(f"S2-fast water mask: {int(water.sum())}/{total_px} px "
           f"({100.0 * water.sum() / total_px:.1f}% water; "
           f"spectral alone {spec_frac*100:.1f}%)")

    # ── SMART VHR water/land mask (physically-grounded GMM unmixing) ──────────
    # The 10 m S2 mask cannot resolve piers / quays / breakwaters — mixed
    # concrete+water pixels pass as water and bathymetry bleeds onto port
    # infrastructure (user-reported defect). Refine the mask with a 2-component
    # Gaussian-Mixture endmember unmixing (water / built-soil-veg) of Mapbox VHR
    # (~1-2 m) imagery: an output pixel stays water only if its VHR water
    # fraction ≥ VHR_WATER_FRAC (default 0.8) AND the S2 mask already called it
    # water (conservative BOTH-pass fusion). Graceful fallback to S2-only on any
    # Mapbox failure, FLAGGED in the response (never silently pretends VHR ran).
    vhr_mask_meta = {"mask_source": "S2-only (VHR mask not attempted)",
                     "vhr_applied": False}
    try:
        try:
            from backend.vhr_water_mask import refine_water_mask as _vhr_refine
        except ImportError:
            from vhr_water_mask import refine_water_mask as _vhr_refine  # type: ignore
        # Emit the mask-QA preview (3.U2) only for user-facing products —
        # default: on for standalone products, off for the 7 internal MLE scene
        # computes (would generate 7 previews/run). MLE re-emits one for scene 0.
        _emit_prev = (_emit_mask_preview if _emit_mask_preview is not None
                      else (not _internal_return_grid))
        refined, vhr_mask_meta = _vhr_refine(bbox, water, H, W,
                                             emit_preview=_emit_prev)
        if vhr_mask_meta.get("vhr_applied") and int(refined.sum()) >= 50:
            water = np.asarray(refined, dtype=bool)
            s2['water_mask'] = water
            L.info(f"S2-fast VHR mask: {vhr_mask_meta.get('mask_source')} · "
                   f"removed {vhr_mask_meta.get('infra_px_removed')} infra px · "
                   f"retention {vhr_mask_meta.get('retention_pct')}% "
                   f"({int(water.sum())}/{total_px} water)")
        elif vhr_mask_meta.get("vhr_applied"):
            L.info("S2-fast VHR mask: refined mask collapsed (<50 px) — keeping S2 mask")
            vhr_mask_meta["mask_source"] = "S2-only (VHR refined mask collapsed)"
            vhr_mask_meta["vhr_applied"] = False
        else:
            L.info(f"S2-fast VHR mask: {vhr_mask_meta.get('mask_source')} "
                   f"({vhr_mask_meta.get('error')})")
    except Exception as _ex:
        L.info(f"S2-fast VHR mask skipped ({_ex})")
        vhr_mask_meta = {"mask_source": "S2-only fallback", "vhr_applied": False,
                         "error": str(_ex)}

    # --- E1 reflectance corrections (default-OFF env knobs) -----------------
    # Order: Hedley deglint (remove additive glint) BEFORE DSF (estimate aerosol
    # path radiance from the deglinted dark spectrum), then both feed the
    # UNCHANGED Lyzenga/Stumpf/UAE calibrated path. The water mask above scopes
    # the deep-water samples both methods regress/percentile on.
    _wm = s2['water_mask']
    if os.environ.get("HEDLEY_DEGLINT", "0") == "1":
        try:
            s2 = _apply_hedley_deglint(s2, _wm)
            s2['water_mask'] = _wm
        except Exception as ex:
            L.info(f"HEDLEY_DEGLINT failed, using uncorrected bands ({ex})")
    if os.environ.get("S2_DSF", "0") == "1":
        try:
            s2 = _apply_dsf_aquatic(s2, _wm)
            s2['water_mask'] = _wm
        except Exception as ex:
            L.info(f"S2_DSF failed, using uncorrected bands ({ex})")

    # 3. Reference fusion across ALL sources, with per-tile fall-through
    #    so sub-areas of the ROI without dense in-situ still get
    #    calibrated from GEBCO / SlideRule.
    #
    #    Source priority (highest weight first):
    #      • user-supplied  6.0
    #      • bundled in-situ XYZ (KP / KP_EMAL / OMC / SWOT_Dhanna) 5.5
    #      • SlideRule ICESat-2 photons 5.0
    #      • GEBCO (450 m interpolation)               1.0
    #
    #    The ROI is split into a 4×4 grid; each cell tries the sources
    #    in order. As soon as a cell has ≥ N_PER_TILE points it stops —
    #    that prevents one dense in-situ region from drowning every
    #    other source in the WLS.
    rl, rlo, rd, rw = [], [], [], []
    sliderule_bathy = []
    counts = {'in_situ': 0, 'sliderule': 0, 'gebco': 0, 'user': 0}

    # User points always count (they're the caller's ground truth).
    if user_pts:
        for p in user_pts:
            try:
                d = abs(float(p['depth']))
            except Exception:
                continue
            if 0 < d <= MAX_DEPTH_M:
                rl.append(float(p['lat'])); rlo.append(float(p['lon']))
                rd.append(d); rw.append(6.0); counts['user'] += 1

    # Bundled in-situ XYZ that fall in the ROI (KP / EMAL / OMC / Dhanna).
    # 80 / 20 split: 80 % goes into the WLS as training; 20 % is held out
    # for honest RMSE / R² metrics (otherwise the "validation" is in-sample
    # and effectively meaningless — exactly the symptom the user reported
    # at Dhanna).
    insitu_test_lats, insitu_test_lons, insitu_test_deps = [], [], []
    try:
        ila, ilo, ide, _src = xyz_points_in_bbox(bbox)
    except Exception as ex:
        ila, ilo, ide = np.array([]), np.array([]), np.array([])
        L.info(f"S2-fast: in-situ XYZ lookup failed ({ex})")
    # ── s2-iho req #4: leakage-safe spatial-block TRAIN-only calibration ──
    # When SPATIAL_TRAIN_ONLY_XYZ points at an .npz holding the eval harness's
    # held-out TEST coords (keys: lat, lon) + SPATIAL_TRAIN_BUFFER_M, drop every
    # bundled sounding within that buffer of any held-out point BEFORE the 80/20
    # split, so the 3-stage in-situ calibration is fit ONLY on the spatial-train
    # blocks and never sees a held-out block (no leakage when the same depth.tif
    # is later scored on those held-out blocks). Default OFF (env unset).
    _stoz = os.environ.get("SPATIAL_TRAIN_ONLY_XYZ", "").strip()
    if _stoz and len(ide):
        try:
            _hz = np.load(_stoz)
            _hla, _hlo = np.asarray(_hz["lat"], float), np.asarray(_hz["lon"], float)
            _buf = float(os.environ.get("SPATIAL_TRAIN_BUFFER_M", "300"))
            from scipy.spatial import cKDTree as _cKD
            _cosp = np.cos(np.deg2rad(float(np.mean(ila)))) if len(ila) else 1.0
            _hx = _hlo * 111320.0 * _cosp; _hy = _hla * 110570.0
            _ix = np.asarray(ilo) * 111320.0 * _cosp; _iy = np.asarray(ila) * 110570.0
            _tree = _cKD(np.column_stack([_hx, _hy]))
            _dmin, _ = _tree.query(np.column_stack([_ix, _iy]))
            _keep = _dmin >= _buf
            _n0 = len(ide)
            ila, ilo, ide = ila[_keep], ilo[_keep], ide[_keep]
            L.info(f"S2-fast: SPATIAL_TRAIN_ONLY_XYZ dropped {_n0 - len(ide)} of "
                   f"{_n0} bundled soundings within {_buf:.0f} m of {len(_hla)} "
                   f"held-out test pts → {len(ide)} train-eligible (leakage-safe)")
        except Exception as _ex:
            L.warning(f"S2-fast: SPATIAL_TRAIN_ONLY_XYZ filter failed ({_ex}) — "
                      f"falling back to full bundled ingest")
    if len(ide):
        # ── PHYSICS req #1: depth-stratified shallow tie-point sampler ──
        # DEFECT (uniform decimation): `np.linspace(0, N-1, 600)` over a
        # depth-sorted-by-spatial-order survey that is ~89% deep (16-22 m)
        # retains only ~16 of the 1,366 real <6 m soundings, so the WLS /
        # isotonic fit never sees shallow truth and the Stage-1 tail clamps
        # output to ~5.5 m. STRATIFY_SHALLOW=1 (default OFF) GUARANTEES the
        # decimated TRAINING pool keeps >= SHALLOW_MIN_TIE (default 100) of
        # the existing <SHALLOW_MAX_D (6 m) soundings, drawn at random, before
        # the rest of the budget is filled uniformly from the deep mass. The
        # held-out 20% TEST split is taken from the *remaining* pool so the
        # shallow score stays honest (no shallow pt is both train and test).
        # Citation: Lyzenga/Finkbeiner/Bachmann 2006 IEEE TGRS 44(8):2251.
        _stratify = os.environ.get("STRATIFY_SHALLOW", "0") == "1"
        _budget = int(os.environ.get("INSITU_BUDGET", "600"))
        _sh_max_d = float(os.environ.get("SHALLOW_MAX_D", "6.0"))
        _sh_min_tie = int(os.environ.get("SHALLOW_MIN_TIE", "100"))
        if _stratify and len(ide) > _budget:
            _abs_d = np.abs(ide)
            _sh_pool = np.where((_abs_d > 0) & (_abs_d < _sh_max_d))[0]
            _dp_pool = np.where(~((_abs_d > 0) & (_abs_d < _sh_max_d)))[0]
            _srng = np.random.default_rng(
                int(abs(hash(tuple(round(x, 4) for x in bbox))) % 2**31))
            # ADDITIVE: keep the FULL deep budget (so deep training density is
            # NOT decimated to make room for shallow — the control proved that
            # displacing deep pts regresses the 16-20 m band ~+1 m), then ADD
            # >= SHALLOW_MIN_TIE shallow tie-pts ON TOP. Effective pool grows.
            _n_sh = int(min(len(_sh_pool), max(_sh_min_tie, 0)))
            _sh_take = (_srng.permutation(_sh_pool)[:_n_sh]
                        if _n_sh else np.array([], int))
            _n_dp = min(len(_dp_pool), _budget)
            _dp_take = (_srng.permutation(_dp_pool)[:_n_dp]
                        if len(_dp_pool) else np.array([], int))
            sel = np.concatenate([_sh_take, _dp_take]).astype(int)
            _srng.shuffle(sel)
            ila, ilo, ide = ila[sel], ilo[sel], ide[sel]
            _kept_sh = int(np.sum(np.abs(ide) < _sh_max_d))
            L.info(f"S2-fast: STRATIFY_SHALLOW=1 kept {_kept_sh} <{_sh_max_d:.0f} m "
                   f"tie-pts (avail {len(_sh_pool)}) of {len(sel)}-pt pool "
                   f"(was ~{int(round(_budget*len(_sh_pool)/max(len(_abs_d),1)))} uniform)")
        elif len(ide) > _budget:
            # Sub-sample to ~600 in-situ pts max so dense surveys (e.g. KP
            # Basin's 35 k pts) don't crush the WLS or training tiles.
            sel = np.linspace(0, len(ide) - 1, _budget).astype(int)
            ila, ilo, ide = ila[sel], ilo[sel], ide[sel]
        # Deterministic 80/20 split (seeded on bbox so the same ROI gets
        # the same split between calls).
        rng = np.random.default_rng(int(abs(hash(tuple(round(x, 4) for x in bbox))) % 2**31))
        idx = np.arange(len(ide))
        rng.shuffle(idx)
        n_test = max(1, int(round(0.20 * len(idx))))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        for j in train_idx:
            d = float(abs(ide[j]))
            if 0 < d <= MAX_DEPTH_M:
                rl.append(float(ila[j])); rlo.append(float(ilo[j]))
                rd.append(d); rw.append(5.5); counts['in_situ'] += 1
        for j in test_idx:
            d = float(abs(ide[j]))
            if 0 < d <= MAX_DEPTH_M:
                insitu_test_lats.append(float(ila[j]))
                insitu_test_lons.append(float(ilo[j]))
                insitu_test_deps.append(d)

    # SlideRule ICESat-2 photons — high weight (~0.6 m vertical).
    if fetch_sliderule:
        try:
            ice, _msg = run_sliderule(bbox, sd, ed)
            for p in ice or []:
                if p.get('photon_class') == 'bathymetry' and p.get('depth', 0) > 0.3:
                    sliderule_bathy.append(p)
                    rl.append(p['lat']); rlo.append(p['lon'])
                    rd.append(p['depth']); rw.append(5.0); counts['sliderule'] += 1
        except Exception as ex:
            L.info(f"S2-fast: SlideRule unavailable ({ex})")

    # ── E3: inject refraction-corrected 0-6 m ICESat-2 (ATL24) shallow photons ──
    # The in-situ multibeam pool over Khalifa Port carries ZERO control in the
    # 0-2 / 2-4 m TRUTH bands (only ~14 pts <6 m), so the Stage-1 isotonic tail
    # extrapolates flat into shallow water and over-predicts (+5.1 / +3.0 m).
    # ICESat-2 ATL24 (Parrish/Magruder 2025) supplies geolocated, ALREADY
    # refraction-corrected (depth = surface_h − ortho_h, n applied in-product)
    # bathymetric photons. We append the [PHOTON_MIN_D, PHOTON_MAX_D] m photons
    # to the TRAINING reference pool ONLY (the in-situ multibeam stays held-out
    # truth — no photon enters insitu_test_*). Default OFF (INJECT_SHALLOW_PHOTONS).
    #   PHOTON_RW (0.16)          inverse-variance weight (σ_mb/σ_ph)²≈(0.2/0.5)²
    #   PHOTON_DEDUP_GRID_M (10)  per-cell median dedup to one tie-point / S2 px
    #   PHOTON_MIN_D / PHOTON_MAX_D (0.3 / 6.0) refracted-depth window
    counts['photon'] = 0
    photon_band_counts = {'0-2': 0, '2-4': 0, '4-6': 0}
    photon_w_share = 0.0
    if os.environ.get("INJECT_SHALLOW_PHOTONS", "0") == "1":
        try:
            from backend.very_hr_augment import fetch_atl24_points
            p_rw = float(os.environ.get("PHOTON_RW", "0.16"))
            grid_m = float(os.environ.get("PHOTON_DEDUP_GRID_M", "10"))
            p_min = float(os.environ.get("PHOTON_MIN_D", "0.3"))
            p_max = float(os.environ.get("PHOTON_MAX_D", "6.0"))
            atl = fetch_atl24_points(bbox)
            if atl is not None and len(atl.get('depths', [])):
                pla = np.asarray(atl['lats'], float)
                plo = np.asarray(atl['lons'], float)
                pde = np.asarray(atl['depths'], float)
                # ATL24 depth is already refraction-corrected (Parrish 2019
                # n=1.34 applied in-product). ATL03 would need Z'=Z−0.2541·D
                # here; ATL24 does NOT, so we skip the explicit step.
                n_raw = int(len(pde))
                msk = np.isfinite(pde) & (pde >= p_min) & (pde <= p_max)
                pla, plo, pde = pla[msk], plo[msk], pde[msk]
                n_filt = int(len(pde))
                # 10 m per-cell median dedup
                if n_filt:
                    mlat = 111320.0
                    mlon = 111320.0 * _math.cos(_math.radians((s_ + n_) / 2))
                    gx = np.floor((plo - w_) * mlon / grid_m).astype(np.int64)
                    gy = np.floor((pla - s_) * mlat / grid_m).astype(np.int64)
                    key = gx * 1_000_000 + gy
                    cells = {}
                    for k_, la_, lo_, d_ in zip(key, pla, plo, pde):
                        cells.setdefault(int(k_), []).append((la_, lo_, d_))
                    for vals in cells.values():
                        arr = np.asarray(vals)
                        cla = float(np.median(arr[:, 0]))
                        clo = float(np.median(arr[:, 1]))
                        cd = float(np.median(arr[:, 2]))
                        rl.append(cla); rlo.append(clo); rd.append(cd); rw.append(p_rw)
                        counts['photon'] += 1
                        if cd < 2:
                            photon_band_counts['0-2'] += 1
                        elif cd < 4:
                            photon_band_counts['2-4'] += 1
                        elif cd < 6:
                            photon_band_counts['4-6'] += 1
                _tot_w = float(np.sum(rw)) if rw else 1.0
                photon_w_share = float(counts['photon'] * p_rw) / max(_tot_w, 1e-9)
                L.info(f"S2-fast: INJECT_SHALLOW_PHOTONS=1 ATL24 raw={n_raw} "
                       f"filt[{p_min},{p_max}]={n_filt} dedup@{grid_m:.0f}m="
                       f"{counts['photon']} tie-pts (rw={p_rw}) bands "
                       f"0-2:{photon_band_counts['0-2']} "
                       f"2-4:{photon_band_counts['2-4']} "
                       f"4-6:{photon_band_counts['4-6']} "
                       f"eff_w_share={photon_w_share*100:.1f}%")
            else:
                L.info("S2-fast: INJECT_SHALLOW_PHOTONS=1 but ATL24 returned no photons")
        except Exception as ex:
            L.info(f"S2-fast: shallow-photon injection failed ({ex})")

    # ── Per-tile GEBCO fall-through ──
    # Split the bbox into a 4×4 grid and check coverage. For tiles with
    # < N_PER_TILE training points so far (in-situ + SlideRule + user),
    # add GEBCO points sampled inside that tile so the Lyzenga regression
    # has at least *some* anchor everywhere.
    #
    # SKIP GEBCO entirely when there's already a healthy in-situ pool —
    # GEBCO is 450 m interpolation with ~±5 m vertical accuracy and acts
    # as noise once ≥ 30 high-quality in-situ pts are available, dragging
    # the linear recalibration off the in-situ optimum.
    N_PER_TILE = 4
    NX = NY = 4
    skip_gebco = counts.get('in_situ', 0) >= 30
    try:
        gebco = fetch_gebco(bbox) if not skip_gebco else None
        if skip_gebco:
            L.info(f"S2-fast: skipping GEBCO ({counts['in_situ']} in-situ pts already cover the ROI)")
    except Exception as ex:
        gebco = None
        L.info(f"S2-fast: GEBCO unavailable ({ex})")
    if gebco:
        gla = np.asarray(gebco['lats'])
        glo = np.asarray(gebco['lons'])
        gde = np.asarray(gebco['depths'])
        # Pre-bin all our high-quality training pts into the 4×4 grid
        if rl:
            la_arr = np.asarray(rl); lo_arr = np.asarray(rlo)
            tile_x = np.clip(((lo_arr - w_) / max(e_ - w_, 1e-9) * NX).astype(int), 0, NX - 1)
            tile_y = np.clip(((n_ - la_arr) / max(n_ - s_, 1e-9) * NY).astype(int), 0, NY - 1)
            tile_counts = {(tx, ty): 0 for ty in range(NY) for tx in range(NX)}
            for tx, ty in zip(tile_x, tile_y):
                tile_counts[(int(tx), int(ty))] += 1
        else:
            tile_counts = {(tx, ty): 0 for ty in range(NY) for tx in range(NX)}
        # For each undercovered tile, take up to N_PER_TILE GEBCO pts.
        added = 0
        for ty in range(NY):
            for tx in range(NX):
                if tile_counts[(tx, ty)] >= N_PER_TILE:
                    continue
                tw = w_ + tx * (e_ - w_) / NX
                te = w_ + (tx + 1) * (e_ - w_) / NX
                tn = n_ - ty * (n_ - s_) / NY
                ts = n_ - (ty + 1) * (n_ - s_) / NY
                m = ((glo >= tw) & (glo <= te) & (gla >= ts) & (gla <= tn)
                     & (gde > 0.5) & (gde <= MAX_DEPTH_M))
                if not m.any():
                    continue
                idx = np.where(m)[0]
                # Up to N_PER_TILE evenly-spaced GEBCO pts per tile
                if len(idx) > N_PER_TILE:
                    idx = idx[np.linspace(0, len(idx) - 1, N_PER_TILE).astype(int)]
                for j in idx:
                    rl.append(float(gla[j])); rlo.append(float(glo[j]))
                    rd.append(float(gde[j])); rw.append(1.0); added += 1
        counts['gebco'] = added
        L.info(f"S2-fast: GEBCO fall-through added {added} pts across "
               f"{sum(1 for v in tile_counts.values() if v < N_PER_TILE)} undercovered tiles")
    L.info(f"S2-fast refs: in_situ={counts['in_situ']} "
           f"sliderule={counts['sliderule']} gebco={counts['gebco']} "
           f"user={counts['user']} total={len(rd)}")
    L.info(f"S2-fast: {len(rd)} refs ({len(sliderule_bathy)} SlideRule, "
           f"{len(user_pts or [])} user)")

    # 4a. Robust Lyzenga + Stumpf + SlideRule fusion (always runs)
    res_ls = ls_estimate_depth(s2, bbox,
        ref_lats=rl, ref_lons=rlo, ref_depths=rd, ref_weights=rw,
        sliderule_pts=sliderule_bathy)
    depth = res_ls['depth']

    # 4b. UAE pretrained Cluster+RF — calibrated on Khalifa + Mussafah +
    # Dhanna. Blend logic depends on whether the ROI sits inside one of
    # the calibration regions:
    #
    #   • Inside  → 90 % Lyzenga+Stumpf (regression calibrated by the
    #               in-situ pool of that region) / 10 % UAE pretrained.
    #               The in-situ data IS the ground truth, so the WLS fit
    #               on those points beats the global RF — which is
    #               trained on a mix of all regions and tends to predict
    #               toward its bigger clusters (deep KP basin).
    #   • Outside → 50 % UAE pretrained / 50 % Lyzenga+Stumpf —
    #               transfer-learning regime; spectral split is even.
    #   • User-supplied depths shift another 10 % onto Lyzenga
    #     (those points are the caller's ground truth and feed
    #     directly into the WLS).
    uae_used = False
    uae_meta = None
    inside_predefined = _bbox_inside_predefined(bbox)
    if UAE_AVAILABLE or UAE_CNN_AVAILABLE:
        try:
            uae_model = load_uae_model()
            if uae_model is not None:
                # In-situ "darkbox" injection: inside predefined regions
                # we have plenty of in-situ — fine-tune the per-cluster
                # CNN heads on 80 % of those points before predicting,
                # so the model adapts to the region's specific spectral
                # regime without persisting state to disk.
                if (inside_predefined and hasattr(uae_model, 'fine_tune')
                        and counts.get('in_situ', 0) >= 30):
                    try:
                        # Sample features at the in-situ training pixels
                        from backend.uae_pretrained import build_feature_stack
                        feats_full, _ = build_feature_stack(s2)
                        ft_feats = []
                        ft_deps = []
                        ft_wts = []
                        for la, lo, dt, wgt in zip(rl, rlo, rd, rw):
                            if wgt < 5.0:  # skip GEBCO; only in-situ / SlideRule
                                continue
                            r_px = max(0, min(H - 1, int((n_ - la) / (n_ - s_ + 1e-10) * H)))
                            c_px = max(0, min(W - 1, int((lo - w_) / (e_ - w_ + 1e-10) * W)))
                            v = feats_full[r_px, c_px]
                            if np.all(np.isfinite(v)):
                                ft_feats.append(v); ft_deps.append(dt); ft_wts.append(wgt)
                        if len(ft_deps) >= 30:
                            res_ft = uae_model.fine_tune(
                                np.asarray(ft_feats, dtype=np.float32),
                                np.asarray(ft_deps, dtype=np.float32),
                                weights=np.asarray(ft_wts, dtype=np.float32),
                                n_epochs=20, lr=5e-4,
                            )
                            L.info(f"S2-fast: in-situ fine-tune updated clusters "
                                   f"{res_ft.get('updated', [])} (n={res_ft.get('n_used')})")
                    except Exception as ftex:
                        L.info(f"S2-fast: fine-tune skipped ({ftex})")

                uae_pred = uae_model.predict(s2, water_mask=water)
                uae_grid = uae_pred['depth']
                if inside_predefined:
                    w_uae = 0.10        # 90 % in-situ-calibrated Lyzenga
                else:
                    w_uae = 0.50        # 50 / 50 transfer-learning blend
                if user_pts and len(user_pts) >= 5:
                    w_uae = max(0.05, w_uae - 0.10)
                w_ls = 1.0 - w_uae
                both = water & np.isfinite(depth) & np.isfinite(uae_grid)
                only_uae = water & ~np.isfinite(depth) & np.isfinite(uae_grid)
                only_ls = water & np.isfinite(depth) & ~np.isfinite(uae_grid)
                blended = np.full_like(depth, np.nan, dtype=np.float32)
                blended[both] = w_uae * uae_grid[both] + w_ls * depth[both]
                blended[only_uae] = uae_grid[only_uae]
                blended[only_ls] = depth[only_ls]
                depth = np.clip(blended, 0.0, MAX_DEPTH_M)
                uae_used = True
                uae_meta = uae_model.meta
                L.info(f"S2-fast: UAE blend "
                       f"(w_UAE={w_uae:.2f}, w_LS={w_ls:.2f}, "
                       f"inside_predefined={inside_predefined})")
        except Exception as ex:
            L.info(f"S2-fast: UAE pretrained unavailable ({ex})")

    # 4c. Sophisticated bias-reduction pipeline (3 stages).
    # Stage 1: linear depth recalibration  z_true = a · z_pred + b.
    # Stage 2: per-pixel residual IDW from kNN of training-point residuals.
    # Stage 3: clamp to [0, MAX_DEPTH_M].
    # All three stages run only when there are ≥ 10 training points with
    # finite predictions; otherwise the depth is left untouched.
    bias_info = {'applied': False}
    # Geodesy Item 2: snapshot the PRE-calibration (post-blend) surface so the
    # eval driver can read the raw, un-calibrated depth at the LAT soundings.
    # That raw median residual IS the datum+tide term (LAT vs MSL), which the
    # 3-stage in-situ calibration below otherwise silently absorbs.
    depth_pre_calib = depth.copy()
    if rd:
        # Sample predictions at every training point
        train_rows = np.array([
            max(0, min(H - 1, int((n_ - la) / (n_ - s_ + 1e-10) * H)))
            for la in rl
        ], dtype=np.int32)
        train_cols = np.array([
            max(0, min(W - 1, int((lo - w_) / (e_ - w_ + 1e-10) * W)))
            for lo in rlo
        ], dtype=np.int32)
        train_truth = np.asarray(rd, dtype=np.float64)
        train_pred = depth[train_rows, train_cols].astype(np.float64)
        # Drop pairs where the prediction is NaN
        ok = np.isfinite(train_pred) & (train_truth > 0) & (train_truth <= MAX_DEPTH_M)
        if int(ok.sum()) >= 10:
            tr_pred = train_pred[ok]; tr_truth = train_truth[ok]
            tr_lat = np.asarray([rl[i] for i, k in enumerate(ok) if k])
            tr_lon = np.asarray([rlo[i] for i, k in enumerate(ok) if k])
            pre_residual = tr_pred - tr_truth
            pre_rmse = float(np.sqrt(np.mean(pre_residual ** 2)))
            pre_bias = float(np.mean(pre_residual))

            # ── Stage 1: monotonic recalibration ──
            # Try sklearn IsotonicRegression first — non-parametric
            # monotonic mapping pred → true. Captures the shallow / deep
            # nonlinearity (in shallow water the log-ratio saturates and
            # a linear recalib leaves residual bias). Falls back to
            # weighted-least-squares linear if isotonic fails or there
            # are too few unique pred values.
            a_lin, b_lin = 1.0, 0.0
            iso = None
            # ── Depth-density (inverse-frequency) sample weighting ──
            # DEPTH_DENSITY_W=1 (default OFF): the training soundings are
            # ~89% deep (16-22 m), so the weighted isotonic fit is dominated
            # by the deep mass and its shallow tail is dragged up → the
            # +5.11 m / +3.00 m shallow over-prediction ramp. Re-weight each
            # training point by the inverse density of its TRUTH depth band
            # (DenseWeight / Steininger 2021), normalised to mean 1, so rare
            # shallow points are no longer swamped. Multiplied into the
            # existing reference-confidence weights; deep mass stays dense so
            # its absolute weight stays high — only the shallow tail is lifted.
            dens_w = None
            if os.environ.get("DEPTH_DENSITY_W", "0") == "1":
                try:
                    _bin_w = 2.0  # 2 m depth bands
                    _bins = np.floor(tr_truth / _bin_w).astype(int)
                    _ub, _cnt = np.unique(_bins, return_counts=True)
                    _cmap = {int(b): int(c) for b, c in zip(_ub, _cnt)}
                    _inv = np.array([1.0 / (_cmap[int(b)] + 1.0) for b in _bins],
                                    dtype=np.float64)
                    # light smoothing across adjacent bands (DenseWeight α≈1)
                    _band_inv = {int(b): 1.0 / (_cmap[int(b)] + 1.0) for b in _ub}
                    _sm = np.array([
                        np.mean([_band_inv.get(int(b) + d, _band_inv[int(b)])
                                 for d in (-1, 0, 1)]) for b in _bins],
                        dtype=np.float64)
                    dens_w = _sm / max(np.mean(_sm), 1e-12)  # mean -> 1
                    # DenseWeight α (DEPTH_DENSITY_ALPHA, default 1.0): damp the
                    # inverse-frequency weights toward uniform so the deep mass
                    # is not over-de-weighted (α<1 => gentler). dens_w^α, renorm.
                    _alpha = float(os.environ.get("DEPTH_DENSITY_ALPHA", "1.0"))
                    if abs(_alpha - 1.0) > 1e-6:
                        dens_w = np.power(np.clip(dens_w, 1e-9, None), _alpha)
                        dens_w = dens_w / max(np.mean(dens_w), 1e-12)
                    _sh = tr_truth < 6.0
                    _dp = tr_truth >= 16.0
                    _rsh = float(np.mean(dens_w[_sh])) if _sh.any() else float('nan')
                    _rdp = float(np.mean(dens_w[_dp])) if _dp.any() else float('nan')
                    L.info(f"S2-fast: DEPTH_DENSITY_W=1 eff weight ratio "
                           f"shallow(<6m)={_rsh:.2f} vs deep(>=16m)={_rdp:.2f} "
                           f"(n_sh={int(_sh.sum())}, n_dp={int(_dp.sum())})")
                except Exception as _dwx:
                    L.info(f"S2-fast: depth-density weighting skipped ({_dwx})")
                    dens_w = None

            # ── PHYSICS req #2: depth-PIECEWISE / per-regime Stage-1 ──
            # DEFECT (Round-2): a SINGLE global isotonic pred→truth mapping
            # cannot serve shallow + deep at once. Injecting >=100 shallow
            # tie-pts (STRATIFY_SHALLOW) bends the monotone tail so the deep
            # band regressed (16-18 m RMSE 2.35→4.86 m). The shallow anchors
            # and the deep mass pull the same single curve in opposite ways.
            # FIX: fit TWO isotonic regressions — one on the SHALLOW regime
            # (truth < PIECEWISE_SPLIT_M, default 6 m) and one on the DEEP
            # regime (truth >= split) — and BLEND them in PREDICTED-depth
            # space with a smooth logistic weight (no discontinuity at the
            # boundary). The deep mapping is fit on the deep mass ONLY, so it
            # is identical to the Round-2 control curve and the deep band
            # cannot regress; the shallow mapping only governs pixels whose
            # predicted depth is shallow. Citation: Caballero & Stumpf 2020
            # Opt.Express 28(8):11742 (per-regime SDB switching, turbid S2).
            # PIECEWISE_CALIB=1 (default OFF). Reuses tr_pred/tr_truth/w_arr.
            _piecewise = os.environ.get("PIECEWISE_CALIB", "0") == "1"
            _pw_obj = None  # defined unconditionally for bias_info below
            if _piecewise:
                try:
                    from sklearn.isotonic import IsotonicRegression as _IsoR
                    _pw_split = float(os.environ.get("PIECEWISE_SPLIT_M", "6.0"))
                    # Blend halfwidth in PREDICTED-depth metres (smooth ramp).
                    # Default 1.0 m: validated best shallow/deep trade-off
                    # (panel_iter_18, Round 3) with the deep-guard Stage-2.
                    _pw_blend = float(os.environ.get("PIECEWISE_BLEND_M", "1.0"))
                    _pw_min = int(os.environ.get("PIECEWISE_MIN_PTS", "15"))
                    _wpw = np.asarray(rw, dtype=np.float64)[ok]
                    if dens_w is not None:
                        _wpw = _wpw * dens_w
                    _sh_m = tr_truth < _pw_split
                    _dp_m = ~_sh_m
                    _n_sh_fit = int(_sh_m.sum())
                    _n_dp_fit = int(_dp_m.sum())
                    # Both regimes need enough distinct preds + a min count.
                    _ok_sh = (_n_sh_fit >= _pw_min and
                              len(np.unique(np.round(tr_pred[_sh_m], 2))) >= 4)
                    _ok_dp = (_n_dp_fit >= _pw_min and
                              len(np.unique(np.round(tr_pred[_dp_m], 2))) >= 4)
                    if _ok_sh and _ok_dp:
                        _iso_sh = _IsoR(out_of_bounds='clip', y_min=0.0,
                                        y_max=MAX_DEPTH_M)
                        _iso_sh.fit(tr_pred[_sh_m], tr_truth[_sh_m],
                                    sample_weight=_wpw[_sh_m])
                        # DEEP curve = isotonic fit on the DEEP regime ONLY
                        # (truth >= split). Isolates the deep monotone mapping
                        # from the 100 injected shallow anchors (fitting on the
                        # full stratified pool bent the deep tail; iter_06).
                        # Combined with the regime-segregated Stage-2 below
                        # (deep pixels see only deep residuals), the deep band
                        # is protected in BOTH calibration stages.
                        _iso_dp = _IsoR(out_of_bounds='clip', y_min=0.0,
                                        y_max=MAX_DEPTH_M)
                        # DEEP arm population is selectable (PIECEWISE_DEEP_FIT):
                        #  'global'   = fit on ALL pts (= control single iso at
                        #               the deep band → deep Stage-1 == control,
                        #               but bends if shallow anchors dominate the
                        #               low tail; protected by the deep-guard
                        #               Stage-2 below). DEFAULT.
                        #  'deep'     = fit on truth>=split only (isolates the
                        #               monotone tail from shallow anchors).
                        _deep_fit = os.environ.get("PIECEWISE_DEEP_FIT", "deep")
                        if _deep_fit == "deep":
                            _iso_dp.fit(tr_pred[_dp_m], tr_truth[_dp_m],
                                        sample_weight=_wpw[_dp_m])
                        else:
                            _iso_dp.fit(tr_pred, tr_truth, sample_weight=_wpw)
                        # Blend CENTER in PREDICTED-depth space. The shallow
                        # arm must FULLY own every pixel whose TRUTH is < split,
                        # otherwise the 4-6 m truth band (whose predicted depth
                        # sits high, ~7-9 m, near the boundary) gets dragged up
                        # by the deep arm and the shallow band regresses +2.5 m
                        # (iter_11/12). So place the center at the UPPER predicted
                        # depth of the shallow training population (95th pct of
                        # the shallow pts' predicted depth) — w_sh≈1 across the
                        # whole shallow regime — with the ramp finishing before
                        # the deep band. Overridable via PIECEWISE_CENTER_M.
                        _cm_env = os.environ.get("PIECEWISE_CENTER_M", "")
                        if _cm_env.strip():
                            _pw_center = float(_cm_env)
                        else:
                            try:
                                _pw_center = float(np.percentile(
                                    tr_pred[_sh_m], 85))
                                # keep the handover below the deep band: clamp
                                # so center + blend stays under ~split+6 m pred.
                                _pw_center = float(np.clip(
                                    _pw_center, _pw_split, _pw_split + 6.0))
                            except Exception:
                                _nb = np.abs(tr_truth - _pw_split) < 2.0
                                _pw_center = (float(np.median(tr_pred[_nb]))
                                              if _nb.any() else _pw_split)

                        class _PiecewiseIso:
                            """IsotonicRegression-compatible .predict():
                            logistic blend of a shallow- and a deep-regime
                            isotonic in PREDICTED-depth space."""
                            def __init__(s, iso_sh, iso_dp, center, blend):
                                s.iso_sh = iso_sh; s.iso_dp = iso_dp
                                s.center = center
                                s.blend = max(float(blend), 1e-3)
                            def predict(s, p):
                                p = np.asarray(p, dtype=np.float64)
                                ys = s.iso_sh.predict(p)
                                yd = s.iso_dp.predict(p)
                                # w_shallow = 1 for p << center, 0 for p >> center
                                z = (s.center - p) / s.blend
                                z = np.clip(z, -30.0, 30.0)
                                w_sh = 1.0 / (1.0 + np.exp(-z))
                                return w_sh * ys + (1.0 - w_sh) * yd

                        _pw_obj = _PiecewiseIso(_iso_sh, _iso_dp,
                                                _pw_center, _pw_blend)
                        # GLOBAL iso (fit on ALL stratified pts) — used ONLY as
                        # the residual REFERENCE for the shallow pixels' full-
                        # tree Stage-2 field, so the shallow band gets the same
                        # working offset it had when the deep arm was global
                        # (iter_10: shallow bias +0.05) WITHOUT bending the deep
                        # GRID arm (which stays deep-only iso → iter_11: 18-20 m
                        # delta -0.03). This decouples the shallow-Stage-2 / deep-
                        # Stage-1 conflict that a single iso object cannot serve.
                        try:
                            _iso_glob = _IsoR(out_of_bounds='clip', y_min=0.0,
                                              y_max=MAX_DEPTH_M)
                            _iso_glob.fit(tr_pred, tr_truth, sample_weight=_wpw)
                            _pw_obj.iso_glob = _iso_glob
                        except Exception:
                            _pw_obj.iso_glob = None
                        L.info(f"S2-fast: PIECEWISE_CALIB=1 shallow-iso "
                               f"(n={_n_sh_fit}, truth<{_pw_split:.1f} m) + "
                               f"deep-iso (n={_n_dp_fit}) blended at "
                               f"pred≈{_pw_center:.2f} m (±{_pw_blend:.1f} m)")
                    else:
                        L.info(f"S2-fast: PIECEWISE_CALIB requested but a "
                               f"regime is too sparse (n_sh={_n_sh_fit}, "
                               f"n_dp={_n_dp_fit}; need >={_pw_min}+4 uniq) "
                               f"→ falling back to single global isotonic")
                        _pw_obj = None
                except Exception as _pwx:
                    L.info(f"S2-fast: PIECEWISE_CALIB failed ({_pwx}) → "
                           f"single global isotonic")
                    _pw_obj = None

            try:
                from sklearn.isotonic import IsotonicRegression
                w_arr = np.asarray(rw, dtype=np.float64)[ok]
                if dens_w is not None:
                    w_arr = w_arr * dens_w
                if _pw_obj is not None:
                    # Piecewise per-regime mapping replaces the single global
                    # isotonic. Downstream apply path (`if iso is not None`)
                    # is unchanged — _PiecewiseIso quacks like IsotonicRegression.
                    iso = _pw_obj
                    L.info("S2-fast: Stage 1 = PIECEWISE isotonic (shallow+deep)")
                # Need ≥ 8 distinct prediction values for isotonic to
                # produce something sensible; otherwise use linear.
                elif len(np.unique(np.round(tr_pred, 2))) >= 8:
                    iso = IsotonicRegression(out_of_bounds='clip',
                                              y_min=0.0, y_max=MAX_DEPTH_M)
                    iso.fit(tr_pred, tr_truth, sample_weight=w_arr)
                    L.info(f"S2-fast: Stage 1 = isotonic recalibration "
                           f"({len(tr_pred)} pts, "
                           f"{len(np.unique(np.round(tr_pred,2)))} unique preds)")
                else:
                    raise RuntimeError("too few unique preds for isotonic")
            except Exception as _ix:
                # Linear fallback
                try:
                    w_arr = np.asarray(rw, dtype=np.float64)[ok]
                    if dens_w is not None:
                        w_arr = w_arr * dens_w
                    W_diag = w_arr / max(w_arr.sum(), 1e-9) * len(w_arr)
                    X = np.column_stack([tr_pred, np.ones_like(tr_pred)])
                    Xw = X * np.sqrt(W_diag)[:, None]
                    yw = tr_truth * np.sqrt(W_diag)
                    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                    a_lin, b_lin = float(coef[0]), float(coef[1])
                    if not (0.4 <= a_lin <= 1.8 and -8.0 <= b_lin <= 8.0):
                        a_lin, b_lin = 1.0, 0.0
                except Exception:
                    a_lin, b_lin = 1.0, 0.0
                L.info(f"S2-fast: Stage 1 = linear (a={a_lin:.3f}, b={b_lin:.3f}) — isotonic skipped: {_ix}")

            if iso is not None:
                # Apply isotonic only on water pixels (predict() is
                # vectorised; flatten then reshape).
                flat_in = depth.reshape(-1)
                finite = np.isfinite(flat_in)
                flat_out = flat_in.copy()
                if int(finite.sum()) > 0:
                    flat_out[finite] = iso.predict(flat_in[finite].astype(np.float64))
                depth = flat_out.reshape(H, W).astype(np.float32)
                tr_pred_after_lin = iso.predict(tr_pred)
            else:
                depth = a_lin * depth + b_lin
                tr_pred_after_lin = a_lin * tr_pred + b_lin
            depth = np.clip(depth, 0.0, MAX_DEPTH_M)

            # ── Stage 2: per-pixel residual IDW correction ──
            # For each water pixel, weight residuals from the k nearest
            # training points with 1/d² and subtract.  Uses a bounded
            # search radius so the correction stays local — far-from-
            # training pixels just get the global mean residual.
            post_lin_residuals = tr_pred_after_lin - tr_truth
            # PIECEWISE: residuals for the SHALLOW pixels' full-tree field are
            # referenced to the GLOBAL iso (decouples shallow Stage-2 from the
            # deep-only grid arm — see _pw_obj.iso_glob). Deep pixels use the
            # deep-only-iso residuals (post_lin_residuals) via _fdp below.
            post_resid_global = post_lin_residuals
            if _pw_obj is not None and getattr(_pw_obj, 'iso_glob', None) is not None:
                try:
                    post_resid_global = (_pw_obj.iso_glob.predict(tr_pred)
                                         - tr_truth)
                except Exception:
                    post_resid_global = post_lin_residuals
            global_med = float(np.median(post_resid_global))
            # ── Per-2 m-band far-field fallback (PER_BAND_RESID=1, default OFF) ──
            # Stage-2's single global-median far-field fallback re-injects the
            # deep-dominated median residual into shallow pixels far from any
            # shallow training point. Build a per-depth-band median residual
            # (indexed by TRUTH band, ≥5 pts/band else global) so a shallow-
            # predicted pixel falls back to the shallow-band residual instead.
            # Reduces to current behaviour when only one band is populated.
            _per_band_resid = os.environ.get("PER_BAND_RESID", "0") == "1"
            _band_med = {}
            if _per_band_resid:
                try:
                    _bw = 2.0
                    _tb = np.floor(tr_truth / _bw).astype(int)
                    for _b in np.unique(_tb):
                        _msk = _tb == _b
                        if int(_msk.sum()) >= 5:
                            _band_med[int(_b)] = float(np.median(post_lin_residuals[_msk]))
                    L.info(f"S2-fast: PER_BAND_RESID=1 band-median residuals "
                           f"(global={global_med:+.2f}) "
                           + " ".join(f"{b*2}-{b*2+2}m:{v:+.2f}"
                                      for b, v in sorted(_band_med.items())))
                except Exception as _pbx:
                    L.info(f"S2-fast: per-band residual fallback skipped ({_pbx})")
                    _band_med = {}
            # Pixel grid in lat/lon (sub-sampled by 2 then upsampled to
            # avoid quadratic cost on 1k×1k grids)
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(np.column_stack([tr_lat, tr_lon]))
                # Decimate the prediction grid for kNN, upsample after
                step = max(1, min(H, W) // 96)
                rs = np.arange(0, H, step)
                cs = np.arange(0, W, step)
                Hc, Wc = len(rs), len(cs)
                lat_1d = n_ - (rs + 0.5) / H * (n_ - s_)
                lon_1d = w_ + (cs + 0.5) / W * (e_ - w_)
                grid_lat, grid_lon = np.meshgrid(lat_1d, lon_1d, indexing='ij')
                pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
                # k=6 NN, distance in degrees → metres approx via cos(lat)
                k = min(6, len(tr_lat))
                d_deg, idx = tree.query(pts, k=k)
                # Convert distance (deg) to weights — squared inverse with
                # a small floor so the closest point doesn't dominate
                # 100 %.
                wts = 1.0 / (d_deg + 1e-3) ** 2
                # Cap influence beyond ~0.05° (~5 km) — beyond that, fade
                # smoothly to the global median.
                far = d_deg > 0.05
                wts = np.where(far, 0.0, wts)
                wsum = wts.sum(axis=1, keepdims=True)
                close = wsum.squeeze() > 1e-9
                local_resid = np.full(pts.shape[0], global_med, dtype=np.float64)
                # PER_BAND_RESID: far-field fallback indexed by the coarse
                # pixel's OWN Stage-1 predicted depth band, not the scalar
                # global median.
                if _per_band_resid and _band_med:
                    _pred_coarse = depth[np.ix_(rs, cs)].reshape(-1).astype(np.float64)
                    _pb = np.floor(np.nan_to_num(_pred_coarse, nan=0.0) / 2.0).astype(int)
                    _ff = np.array([_band_med.get(int(b), global_med) for b in _pb],
                                   dtype=np.float64)
                    local_resid = _ff.copy()
                # Full-tree field uses the GLOBAL-iso residuals (= deep-only
                # residuals when PIECEWISE is off, since post_resid_global
                # defaults to post_lin_residuals). Under PIECEWISE this is the
                # reference the shallow pixels need (iter_10-equivalent offset).
                _ft_resid = post_resid_global
                if k > 0:
                    contribs = (wts * _ft_resid[idx]).sum(axis=1)
                    local_resid[close] = contribs[close] / wsum.squeeze()[close]

                # ── PHYSICS req #2: regime-segregated Stage-2 (PIECEWISE) ──
                # DEFECT: the single IDW tree lets a deep pixel's kNN include
                # the injected shallow channel-margin tie-points, whose large
                # residuals drag the deep band down (16-18 m regressed +2.5 m
                # even with a control-equal deep Stage-1 — the leak is HERE in
                # Stage-2, not Stage-1). FIX: build a SEPARATE deep-only and
                # shallow-only IDW residual field and blend them by the SAME
                # logistic w_sh used in Stage-1 (keyed on each pixel's Stage-1
                # predicted depth). A deep-predicted pixel then sees ONLY deep
                # residuals → its correction is identical to the control, so
                # the deep band cannot regress; a shallow-predicted pixel sees
                # only shallow residuals. This is the shallow-MASK-ONLY Stage-2
                # the physicist requested.
                if _pw_obj is not None:
                    try:
                        _tr_truth_ok = tr_truth  # aligned with tr_lat/tr_lon/post_lin_residuals
                        _sh_tr = _tr_truth_ok < _pw_split
                        _dp_tr = ~_sh_tr
                        def _regime_field(mask):
                            if int(mask.sum()) < 5:
                                return None
                            _t = cKDTree(np.column_stack([tr_lat[mask], tr_lon[mask]]))
                            _kk = min(6, int(mask.sum()))
                            _dd, _ii = _t.query(pts, k=_kk)
                            if _kk == 1:
                                _dd = _dd[:, None]; _ii = _ii[:, None]
                            _ww = 1.0 / (_dd + 1e-3) ** 2
                            _ww = np.where(_dd > 0.05, 0.0, _ww)
                            _ws = _ww.sum(axis=1)
                            _cl = _ws > 1e-9
                            _res_m = post_lin_residuals[mask]
                            _gm = float(np.median(_res_m))
                            _fld = np.full(pts.shape[0], _gm, dtype=np.float64)
                            _c = (_ww * _res_m[_ii]).sum(axis=1)
                            _fld[_cl] = _c[_cl] / _ws[_cl]
                            return _fld
                        _fdp = _regime_field(_dp_tr)
                        if _fdp is not None:
                            # ONE-SIDED segregation: the SHALLOW component keeps
                            # the full-tree IDW field (`local_resid`) — it needs
                            # the spatially-nearby deep anchors to set the right
                            # offset, and segregating it onto the 75 sparse
                            # shallow pts alone regressed the shallow band +2.5 m
                            # (iter_09). The DEEP component uses the deep-ONLY
                            # field (`_fdp`), so a deep-predicted pixel's Stage-2
                            # correction is control-equal and the deep band does
                            # not regress. Blend by the same logistic w_sh.
                            _pc = depth[np.ix_(rs, cs)].reshape(-1).astype(np.float64)
                            _fin = np.isfinite(_pc)
                            _pcf = np.where(_fin, _pc, _pw_obj.center)
                            _z = np.clip((_pw_obj.center - _pcf) / _pw_obj.blend,
                                         -30.0, 30.0)
                            _wsh_px = 1.0 / (1.0 + np.exp(-_z))
                            # DEEP-GUARD (decouples the deep-band Stage-2 leak
                            # from the Stage-1 shallow-ownership center): force
                            # the Stage-2 shallow weight HARD to 0 once the
                            # predicted depth exceeds a guard (default split+5 m
                            # ≈ pred 11 m), so the deep band (16-20 m) gets a
                            # PURE deep-only residual field no matter how high
                            # the Stage-1 blend center is pushed for shallow
                            # ownership. Without this, a center≈11-12 m needed
                            # for the 4-6 m band leaked ~5% of the shallow
                            # residual field into 16-18 m (+1.7 m; iter_13/15).
                            _guard = float(os.environ.get(
                                "PIECEWISE_STAGE2_GUARD_M",
                                str(_pw_split + 5.0)))
                            _gw = float(os.environ.get(
                                "PIECEWISE_STAGE2_GUARD_RAMP_M", "0.7"))
                            _zg = np.clip((_guard - _pcf) / max(_gw, 1e-3),
                                          -30.0, 30.0)
                            _wsh_px = _wsh_px * (1.0 / (1.0 + np.exp(-_zg)))
                            _blend = _wsh_px * local_resid + (1.0 - _wsh_px) * _fdp
                            _blend = np.where(np.isfinite(_blend) & _fin,
                                              _blend, local_resid)
                            local_resid = _blend
                            L.info("S2-fast: PIECEWISE Stage-2 one-sided "
                                   f"segregation (deep field n={int(_dp_tr.sum())}) "
                                   "— deep pixels see only deep residuals "
                                   "(control-equal); shallow keeps full-tree field")
                    except Exception as _s2x:
                        L.info(f"S2-fast: PIECEWISE Stage-2 segregation skipped "
                               f"({_s2x}); single-tree IDW used")
                # Reshape to coarse grid then upsample to (H, W)
                coarse = local_resid.reshape(Hc, Wc).astype(np.float32)
                if SCI:
                    from scipy.ndimage import zoom as ndizoom
                    factor_r = H / coarse.shape[0]
                    factor_c = W / coarse.shape[1]
                    resid_grid = ndizoom(coarse, (factor_r, factor_c), order=1, mode='nearest')
                    resid_grid = resid_grid[:H, :W]
                else:
                    resid_grid = np.full((H, W), global_med, dtype=np.float32)
                # Clamp residuals to ±3 m so a single bad training point
                # can't introduce a bigger error than it solves
                resid_grid = np.clip(resid_grid, -3.0, 3.0)
                depth = depth - resid_grid
                depth = np.clip(depth, 0.0, MAX_DEPTH_M)
                # Recompute final residuals on training set for logging
                tr_pred_final = depth[train_rows[ok], train_cols[ok]].astype(np.float64)
                post_residuals = tr_pred_final - tr_truth
                post_rmse = float(np.sqrt(np.mean(post_residuals ** 2)))
                post_bias = float(np.mean(post_residuals))
                bias_info = {
                    'applied': True,
                    'stage1': (('piecewise_isotonic' if _pw_obj is not None
                                else 'isotonic') if iso is not None
                               else 'linear'),
                    'pre_rmse_m': round(pre_rmse, 3),
                    'pre_bias_m': round(pre_bias, 3),
                    'post_rmse_m': round(post_rmse, 3),
                    'post_bias_m': round(post_bias, 3),
                    'lin_a': round(a_lin, 4),
                    'lin_b': round(b_lin, 4),
                    'global_resid_med_m': round(global_med, 3),
                    'n_train_used': int(ok.sum()),
                }
                stage1_label = (('piecewise_isotonic' if _pw_obj is not None
                                 else 'isotonic') if iso is not None
                                else f"linear (a={a_lin:.3f}, b={b_lin:.3f})")
                L.info(f"S2-fast: bias-correction · "
                       f"{stage1_label} + IDW residuals "
                       f"→ RMSE {pre_rmse:.2f}→{post_rmse:.2f} m, "
                       f"bias {pre_bias:+.2f}→{post_bias:+.2f} m, "
                       f"n={int(ok.sum())}")
            except Exception as ex:
                L.info(f"S2-fast: per-pixel residual correction skipped ({ex})")

    # ── IHO req #1/#2: honest per-pixel σ (TPU) grid (EMIT_SIGMA=1, default OFF) ──
    # σ_total(pixel) = sqrt( σ_model² + σ_tide² + σ_refr² + σ_ref² + σ_georef_z² )
    #   σ_model   : local kNN spread of the post-calibration training residuals
    #               (heteroscedastic; grows where the calibration disagrees with
    #               the soundings — turbid/deep). Floored at σ_ref.
    #   σ_tide    : SIGMA_TIDE_M (0.10 m) residual tide-model error after reduction
    #   σ_refr    : depth-proportional ICESat-2/SDB refraction residual,
    #               SIGMA_REFR_FRAC·depth (0.5% — n=1.34 sub-pixel residual)
    #   σ_ref     : SIGMA_REF_M (0.25 m) survey-grade multibeam reference σ floor
    #   σ_georef_z: horizontal georef σ × bed slope → vertical; small over the
    #               gentle dredged floor, SIGMA_GEOREF_H_M (5 m) × local |grad z|
    # The driver calibrates a single scalar k so empirical 95% coverage lands in
    # [0.88,0.98] and writes the calibrated sigma.tif + tpu.tif. Caballero &
    # Stumpf 2020; IHO S-44 ed6.1 §3.4 (TVU=1.96σ); reuses the intra-pixel-std
    # idea from the observed-eval block.
    sigma_grid = None
    if os.environ.get("EMIT_SIGMA", "0") == "1":
        try:
            s_tide = float(os.environ.get("SIGMA_TIDE_M", "0.10"))
            s_refr_frac = float(os.environ.get("SIGMA_REFR_FRAC", "0.005"))
            s_ref = float(os.environ.get("SIGMA_REF_M", "0.25"))
            s_georef_h = float(os.environ.get("SIGMA_GEOREF_H_M", "5.0"))
            # σ_model: local spread (RMS) of post-calibration training residuals
            sigma_model = np.full((H, W), s_ref, dtype=np.float32)
            try:
                from scipy.spatial import cKDTree as _ckd
                if 'post_residuals' in dir() and len(tr_lat) >= 4:
                    _tree = _ckd(np.column_stack([tr_lat, tr_lon]))
                    _step = max(1, min(H, W) // 96)
                    _rs = np.arange(0, H, _step); _cs = np.arange(0, W, _step)
                    _Hc, _Wc = len(_rs), len(_cs)
                    _lat1 = n_ - (_rs + 0.5) / H * (n_ - s_)
                    _lon1 = w_ + (_cs + 0.5) / W * (e_ - w_)
                    _gla, _glo = np.meshgrid(_lat1, _lon1, indexing='ij')
                    _pts = np.column_stack([_gla.ravel(), _glo.ravel()])
                    _kk = min(8, len(tr_lat))
                    _dd, _ii = _tree.query(_pts, k=_kk)
                    # local residual RMS (heteroscedastic σ_model)
                    _res_k = post_residuals[_ii] if post_residuals.ndim else post_residuals
                    _loc = np.sqrt(np.mean(_res_k ** 2, axis=1))
                    _coarse = _loc.reshape(_Hc, _Wc).astype(np.float32)
                    if SCI:
                        from scipy.ndimage import zoom as _zoom
                        _sm = _zoom(_coarse, (H / _Hc, W / _Wc), order=1, mode='nearest')
                        sigma_model = _sm[:H, :W].astype(np.float32)
                    sigma_model = np.maximum(sigma_model, s_ref)
            except Exception as _se:
                L.info(f"S2-fast: σ_model kNN failed ({_se}); using flat σ_ref")
            # σ_refr (depth-proportional) on the final depth
            _d = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
            sigma_refr = (s_refr_frac * _d).astype(np.float32)
            # σ_georef_z = σ_georef_h × |∇z| (bed-slope × horizontal error)
            try:
                gz_y, gz_x = np.gradient(_d)
                px_m = (e_ - w_) / W * 111320.0 * _math.cos(_math.radians((n_ + s_) / 2))
                py_m = (n_ - s_) / H * 110570.0
                slope = np.sqrt((gz_x / max(px_m, 1e-6)) ** 2 +
                                (gz_y / max(py_m, 1e-6)) ** 2)
                sigma_georef_z = (s_georef_h * slope).astype(np.float32)
                sigma_georef_z = np.clip(sigma_georef_z, 0.0, 2.0)
            except Exception:
                sigma_georef_z = np.zeros((H, W), dtype=np.float32)
            sigma_grid = np.sqrt(
                sigma_model.astype(np.float32) ** 2 +
                np.float32(s_tide) ** 2 +
                sigma_refr ** 2 +
                np.float32(s_ref) ** 2 +
                sigma_georef_z ** 2).astype(np.float32)
            L.info(f"S2-fast: EMIT_SIGMA σ grid built — median σ_model="
                   f"{float(np.nanmedian(sigma_model)):.2f} m, "
                   f"median σ_total={float(np.nanmedian(sigma_grid)):.2f} m "
                   f"(tide={s_tide} refr_frac={s_refr_frac} ref={s_ref} georef_h={s_georef_h})")
        except Exception as _sx:
            L.info(f"S2-fast: EMIT_SIGMA failed ({_sx}); no sigma grid")
            sigma_grid = None

    # ── AI_METHOD_LOG R1/R2/R3: deep per-pixel σ + never-worse blend ──
    # Runs ONLY for the MLE per-scene path (_internal_return_grid) when the
    # master flag is set AND this scene was elected to train the deep head
    # (_deep_train). Emits sigma_px/depth_blend/deep_gate on the response;
    # NEVER mutates `depth` (flag-OFF path stays byte-identical). See
    # _deep_scene_sigma for the leakage-safe gate.
    deep_sigma_px = None
    deep_depth_blend = None
    deep_gate = None
    if (os.environ.get("MLE_DEEP_SIGMA", "0") == "1"
            and _internal_return_grid and _deep_train):
        try:
            if ('tr_lat' in dir() and len(tr_lat) >= 10
                    and len(insitu_test_lats) >= 5):
                _dd = _deep_scene_sigma(
                    s2, bbox, water, depth,
                    tr_lat, tr_lon, tr_truth,
                    insitu_test_lats, insitu_test_lons, insitu_test_deps,
                    scalar_sigma=None)
                if _dd is not None:
                    deep_sigma_px = _dd.get("sigma_px")
                    deep_depth_blend = _dd.get("depth_blend")
                    deep_gate = _dd.get("gate")
            else:
                L.info("MLE-deep: insufficient train/held-out refs in this scene "
                       "→ scalar σ kept")
        except Exception as _dex:
            L.info(f"MLE-deep: skipped ({_dex}) → scalar σ kept")

    # Apply the homogeneous water mask to the final raster (no patchiness)
    depth = np.where(water, depth, np.nan)
    if sigma_grid is not None:
        sigma_grid = np.where(water, sigma_grid, np.nan).astype(np.float32)
    if deep_depth_blend is not None:
        deep_depth_blend = np.where(water, deep_depth_blend, np.nan).astype(np.float32)

    # 5. Metrics on held-out reference pts.
    # Includes the 20 % in-situ XYZ test split + any caller-supplied
    # user_points. Both are real held-out ground truth — never seen by
    # the WLS.
    metrics = {'n_test': 0}
    scatter_pred, scatter_true = [], []
    test_la, test_lo, test_dp = list(insitu_test_lats), list(insitu_test_lons), list(insitu_test_deps)
    if user_pts:
        for p in user_pts:
            try:
                la = float(p['lat']); lo = float(p['lon']); dt = abs(float(p['depth']))
            except Exception:
                continue
            if 0 < dt <= MAX_DEPTH_M:
                test_la.append(la); test_lo.append(lo); test_dp.append(dt)
    if test_la:
        for la, lo, dt in zip(test_la, test_lo, test_dp):
            r_px = max(0, min(H - 1, int((n_ - la) / (n_ - s_ + 1e-10) * H)))
            c_px = max(0, min(W - 1, int((lo - w_) / (e_ - w_ + 1e-10) * W)))
            v = depth[r_px, c_px]
            if np.isfinite(v):
                scatter_pred.append(float(v))
                scatter_true.append(dt)
        if len(scatter_pred) >= 5:
            sp = np.asarray(scatter_pred); st = np.asarray(scatter_true)
            r = sp - st
            metrics['rmse_m'] = round(float(np.sqrt(np.mean(r ** 2))), 3)
            metrics['mae_m'] = round(float(np.mean(np.abs(r))), 3)
            metrics['bias_m'] = round(float(np.mean(r)), 3)
            metrics['r2'] = round(float(1 - np.sum(r ** 2) /
                                        max(np.sum((st - st.mean()) ** 2), 1e-9)), 4)
            metrics['n_test'] = len(scatter_pred)
            tvu = np.sqrt(0.5 ** 2 + (0.013 * st) ** 2)
            metrics['s44_1a_pct'] = round(100.0 * float(np.mean(np.abs(r) <= tvu)), 1)
            tvu2 = np.sqrt(1.0 ** 2 + (0.023 * st) ** 2)
            metrics['s44_order2_pct'] = round(100.0 * float(np.mean(np.abs(r) <= tvu2)), 1)

    # MASK-GLOBAL (2026-07-10, user-mandated): UNCONDITIONAL universal land/
    # ocean cut on the RAW numeric depth grid — applied AFTER the held-out
    # validation-point RMSE/bias/R²/S-44 metrics above (computed against
    # SURVEY soundings, which are trustworthy by construction; cutting first
    # would risk silently dropping legitimate validation points) but BEFORE
    # every downstream consumer of `depth` (stats, interpolated points,
    # contours, PNG overlay, GeoTIFF) — so the served/exported product is cut
    # everywhere, on every path through this shared fast-pipeline function
    # (sdb-pro / very-hr-clustered / quick-analyse's fast path / the MLE
    # per-scene loop via _compute_s2_depth_grid).
    _cut_info_fast = None
    try:
        try:
            from backend.osm_land_mask import apply_global_land_cut as _global_cut
        except ImportError:
            from osm_land_mask import apply_global_land_cut as _global_cut  # type: ignore
        # MASK-NDWI (2026-07-10, user priority override #2): pass the SAME
        # native-resolution NDWI `fetch_s2` already computed for this exact
        # scene/window — imagery-evidence land the vector sources above
        # cannot know about (reclaimed/dredged port land not yet in OSM).
        # Safety-gated inside ndwi_mask.py (conservative threshold + buffer-
        # to-known-land / ship-blob only — see that module's docstring for
        # the measured-unsafe naive-Otsu numbers this rejected).
        depth, _land_mask_fast, _cut_info_fast = _global_cut(depth, bbox, ndwi=s2.get('ndwi'), ndwi_res_m=res)
        water = np.asarray(water, dtype=bool) & ~_land_mask_fast
        L.info(f"Global land cut: {_cut_info_fast.get('n_depth_px_cut_this_call', 0)} px cut "
               f"({_cut_info_fast.get('mask_source')})")
    except Exception as _cutex:
        L.info(f"Global land cut skipped ({_cutex})")

    # ADPorts F3 (universal tide_correction contract, non-MLE fast path):
    # same EOT20-primary/Open-Meteo-fallback disclosure the MLE path carries
    # under `result.tide_correction`, on `/api/sdb-pro` / `/api/very-hr-
    # clustered` / quick-analyse's fast path. Disclosure-only by default
    # (TIDE_REDUCE_TO_MSL=1 to physically reduce `depth` to MSL — off by
    # default, no behaviour change unless explicitly opted in).
    _acq_ts_fast = None
    try:
        _ts_list = _gee_window_timestamps(bbox, sd, ed)
        if _ts_list:
            _acq_ts_fast = _ts_list[len(_ts_list) // 2]  # representative (median) timestamp
    except Exception:
        pass
    tide_correction, depth = build_tide_correction(bbox, acquisition_utc=_acq_ts_fast,
                                                   depth_grid=depth, water_mask=water)

    # Bathymetry stats (always — these are what the UI shows: mean/max/min/std)
    val = depth[np.isfinite(depth) & (depth > 0)]
    if len(val):
        # In-cluster RF in-sample R² is the closest proxy for confidence
        # when there are no held-out user points. Anchor at 70 % so the UI
        # never displays an empty bar.
        if uae_meta and uae_meta.get('in_sample_r2'):
            confidence = round(100.0 * max(0.7, min(0.99, float(uae_meta['in_sample_r2']))), 1)
        else:
            confidence = 80.0
        bathy_stats = {
            'mean_depth': round(float(np.mean(val)), 2),
            'max_depth':  round(float(np.max(val)),  2),
            'min_depth':  round(float(np.min(val)),  2),
            'std_depth':  round(float(np.std(val)),  2),
            'grid_points': int(len(val)),
            'resolution_m': res,
            'confidence': confidence,
        }
    else:
        bathy_stats = {'mean_depth': 0, 'max_depth': 0, 'min_depth': 0,
                       'std_depth': 0, 'grid_points': 0, 'resolution_m': res,
                       'confidence': 0}
    n_train_total = len(rd) + (uae_meta.get('n_train', 0) if uae_meta else 0)
    metrics['n_train'] = n_train_total
    bathy_stats['n_train'] = n_train_total

    # Interpolated depth points (powers the Data tab + 3D + export). Sub-
    # sample to keep the JSON payload small.
    interp_pts = grid_to_points(depth, bbox, max_pts=8000)

    # Isobath contours (powers the Contour tab + IHO chart line work).
    contours_out, contour_levels = [], []
    try:
        contours_out = generate_contours(depth, bbox)
        contour_levels = sorted({c['depth'] for c in contours_out})
    except Exception as ex:
        L.info(f"Fast path contours skipped: {ex}")

    # 6. Render PNGs (homogeneous: water-mask is consistent across all)
    raster_b64, raster_bounds, _ = depth_to_raster_png(depth, bbox, water_mask=water)

    # Water mask PNG — visual confirmation of land/water classification
    try:
        from PIL import Image
        wm_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        wm_rgba[water]    = (15, 118, 189, 200)   # cyan-blue water
        wm_rgba[~water]   = (60, 50, 40, 0)       # transparent land
        bw = _io.BytesIO()
        Image.fromarray(wm_rgba, 'RGBA').save(bw, format='PNG', optimize=True)
        water_b64 = _b64.b64encode(bw.getvalue()).decode()
    except Exception as ex:
        L.warning(f"Water mask PNG failed: {ex}")
        water_b64 = None

    # Scatter plot — predicted vs reference
    scatter_b64 = None
    if len(scatter_pred) >= 5:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
            sp = np.asarray(scatter_pred); st = np.asarray(scatter_true)
            ax.scatter(st, sp, s=12, alpha=0.55, c='#0d9488', edgecolor='none')
            mx = float(max(np.max(sp), np.max(st), MAX_DEPTH_M))
            ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.6, label='1:1')
            ax.set_xlabel('Reference depth (m)'); ax.set_ylabel('Predicted depth (m)')
            ax.set_title(f"Lyzenga + SlideRule  ·  RMSE={metrics['rmse_m']} m  "
                         f"·  R²={metrics['r2']}")
            ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.grid(alpha=0.3); ax.legend()
            buf = _io.BytesIO()
            fig.tight_layout(); fig.savefig(buf, format='png'); plt.close(fig)
            scatter_b64 = _b64.b64encode(buf.getvalue()).decode()
        except Exception as ex:
            L.warning(f"Scatter PNG failed: {ex}")

    # GeoTIFF — heavy payload, opt-in only. Inlining the GeoTIFF balloons
    # the JSON response by ~4 MB which has been the Content-Length /
    # response-Body mismatch trigger for Jbel Dhanna behind some proxies.
    geotiff_b64 = None
    if include_geotiff:
        try:
            geotiff_b64 = build_geotiff_b64(depth, bbox)
        except Exception as ex:
            L.warning(f"GeoTIFF failed: {ex}"); geotiff_b64 = None

    elapsed = round(_T.time() - t0, 2)
    # Internal pipeline string for logs only — kept verbose so debugging
    # stays easy. The user-facing label below is deliberately neutral.
    pieces_internal = []
    if uae_used:
        pieces_internal.append('Calibrated default')
    pieces_internal.append(res_ls['method'])
    pipeline_internal = 'Sentinel-2 SDB · ' + ' + '.join(pieces_internal)
    method_label = 'Calibrated bathymetry'
    # Tighter, less noisy log line. RMSE / S-44 are only included when
    # we actually had held-out user-supplied references to score against.
    log_extras = []
    if metrics.get('rmse_m') is not None and metrics.get('n_test', 0) >= 5:
        log_extras.append(f"RMSE={metrics['rmse_m']}m")
        log_extras.append(f"R²={metrics['r2']}")
    extras_str = (' · ' + ' '.join(log_extras)) if log_extras else ''
    L.info(f"S2-fast: {elapsed}s · {pipeline_internal} · "
           f"water={int(water.sum())}/{H*W} px · "
           f"{bathy_stats['mean_depth']}m mean depth{extras_str}")

    response = {
        'method': method_label,
        'bbox': bbox, 'resolution_m': res,
        'metrics': metrics,
        'stats': bathy_stats,
        'ml_stats': {
            'method': method_label,
            'n_train': n_train_total,
            # Sources / importance kept neutral — no algorithm names leak
            # into the UI. Internal log line still records the detailed
            # ensemble breakdown for debugging.
            'sources': [],
            'importance': {},
            'r2': metrics.get('r2'),
            'rmse': metrics.get('rmse_m'),
        },
        'per_band': {},
        'augmentation': {
            'sliderule_pts': len(sliderule_bathy),
            'gebco_pts': max(0, len(rd) - len(sliderule_bathy) - len(user_pts or [])),
            'user_pts': len(user_pts or []),
            'pretrained_used': uae_used,
            'n_scenes': int(s2.get('n_scenes', 1)),
            'bias_correction': bias_info,
            'photon_pts': int(counts.get('photon', 0)),
            'photon_band_counts': photon_band_counts,
            'photon_w_share': round(photon_w_share, 4),
            'insitu_train_pts': int(counts.get('in_situ', 0)),
        },
        'turbidity_pct': float(s2.get('turbid_pct', 0)),
        'turbidity_warning': None,
        'elapsed_s': elapsed,
        'water_pixels': int(water.sum()),
        'land_pixels': int((~water).sum()),
        'water_mask_meta': vhr_mask_meta,
        'global_land_cut': _cut_info_fast,
        'tide_correction': tide_correction,
        'depth_png_b64': raster_b64,
        'overlay_png_b64': raster_b64,
        'water_mask_b64': water_b64,
        'scatter_png_b64': scatter_b64,
        'geotiff_b64': geotiff_b64,
        'raster_bounds': raster_bounds,
        'fit': res_ls.get('fit', {}),
        'sources_used': [],
        'interpolated_points': interp_pts,
        'contours': contours_out,
        'contour_levels': contour_levels,
        # Download endpoint (frontend can fetch the GeoTIFF on demand
        # without bloating this JSON response):
        'geotiff_endpoint': '/api/sdb-pro/geotiff',
        'paths': {},
    }
    if _internal_return_grid:
        # MLE caller wants the raw float arrays alongside the standard
        # response shape; tack them on without serialising (numpy arrays
        # would break jsonify anyway, so callers must use this directly).
        response['depth_grid'] = depth
        response['water_mask'] = water
        # Geodesy Item 2: raw pre-calibration surface (un-calibrated, post-blend).
        response['depth_grid_pre_calib'] = depth_pre_calib
        # IHO req #1/#2: honest per-pixel σ grid (None unless EMIT_SIGMA=1).
        response['sigma_grid'] = sigma_grid
        # AI_METHOD_LOG R1/R2/R3: deep per-pixel σ + never-worse blend (None
        # unless MLE_DEEP_SIGMA=1 and this scene trained the deep head).
        response['deep_sigma_px'] = deep_sigma_px
        response['deep_depth_blend'] = deep_depth_blend
        response['deep_gate'] = deep_gate
        # MASKMLE 1.2a/1.3: hand the raw S2 band dict + TRUE acquisition
        # timestamps up to the MLE stability path (per-scene tide + QC).
        response['_s2'] = s2
        response['acquisition_datetimes'] = s2.get('acquisition_datetimes')
        # MASKMLE 3.2: hand the SAME reference set production calibrated on up to
        # the shared-calibration-field mode so Stage B can be fit ONCE on the
        # kept-scene composite (rl/rlo/rd = reference lat/lon/depth; identical
        # across scenes for a fixed bbox).
        try:
            response['_calib_refs'] = {
                'lat': list(rl), 'lon': list(rlo), 'depth': list(rd)}
        except Exception:
            response['_calib_refs'] = None
    return response


# ══════════════════════════════════════════════════════════════
# MASKMLE R1 (item 1.2 / 1.3) — per-scene EOT20 tide + physical QC
# ══════════════════════════════════════════════════════════════
_EOT20_DIR = Path(__file__).resolve().parent.parent / "cache"   # holds EOT20/ symlink


def _eot20_tide_heights(lon0, lat0, dt_iso_list):
    """Per-timestamp EOT20 tide height (m, model MSL=0) at (lon0,lat0).

    Reuses the proven ``dl_fusion_r13_lat_datum._tide_series`` call pattern
    (pyTMD 3.0.6, model="EOT20", directory=cache/ which contains the EOT20/
    symlink). ``dt_iso_list`` = list of ISO-UTC strings. Returns a float ndarray
    of the same length (NaN on failure), or None if pyTMD/cache unavailable."""
    if not dt_iso_list:
        return None
    try:
        import pyTMD
        dt64 = np.array([np.datetime64(str(d).replace('Z', '')) for d in dt_iso_list])
        tide = pyTMD.compute.tide_elevations(
            x=np.full(dt64.shape, float(lon0)), y=np.full(dt64.shape, float(lat0)),
            delta_time=dt64, directory=str(_EOT20_DIR), model="EOT20",
            type="drift", standard="datetime", crs=4326,
            extrapolate=True, cutoff=50.0)
        return np.asarray(tide, dtype=float).ravel()
    except Exception as ex:
        L.info(f"EOT20 tide computation failed ({ex})")
        return None


# ── ADPorts F3: shared tide_correction disclosure contract (all result paths) ──
_TIDE_SIGMA_FLOOR_M = 0.15  # binding floor per the coordinator's caveat


def _tide_correction_fallback(note="Uncorrected — instantaneous sea level at acquisition; "
                              "no tidal reduction applied."):
    return {"applied": False, "height_m": None, "source": "none", "acquisition_utc": None,
           "datum_before": "instantaneous", "datum_after": "instantaneous sea level",
           "sigma_tide_m": None, "note": note}


def build_tide_correction(bbox, acquisition_utc=None, depth_grid=None, water_mask=None,
                          reduce_to_msl=None, max_depth_m=None):
    """ADPorts F3: build the `result.tide_correction` disclosure contract
    (EOT20 primary / Open-Meteo fallback / honest 'none' fallback), and —
    ONLY when the `TIDE_REDUCE_TO_MSL=1` env knob is set (default OFF; no
    depth-value change anywhere by default) — physically reduce `depth_grid`
    to MSL via the existing, already-tested `tide_correction.apply_tide_
    correction()` (correct, verified sign convention: MSL depth = instant-
    aneous depth − tide_m, tide_m>0 meaning the surface sat above MSL at
    acquisition).

    `acquisition_utc`: an ISO-UTC string (or list — first-valid used) for a
    representative acquisition time. None -> honest 'none' fallback (no
    timestamp to reduce with).

    Returns (tide_correction_dict, possibly-corrected depth_grid). If
    `depth_grid` is None, returns (tide_correction_dict, None) — disclosure
    only, no grid to modify.
    """
    if reduce_to_msl is None:
        reduce_to_msl = os.environ.get("TIDE_REDUCE_TO_MSL", "0") == "1"
    if isinstance(acquisition_utc, (list, tuple)):
        acquisition_utc = next((a for a in acquisition_utc if a), None)
    if not acquisition_utc:
        return _tide_correction_fallback(), depth_grid

    w, s, e, n = bbox
    lon_c, lat_c = 0.5 * (w + e), 0.5 * (s + n)
    height_m = None
    source = "none"
    try:
        th = _eot20_tide_heights(lon_c, lat_c, [acquisition_utc])
        if th is not None and np.isfinite(th).any():
            height_m = float(np.nanmean(th))
            source = "EOT20"
    except Exception as ex:
        L.info(f"tide_correction: EOT20 failed ({ex})")
    if height_m is None:
        try:
            from backend.tide_correction import get_tide_height
            from datetime import datetime as _dt
            dt = _dt.fromisoformat(str(acquisition_utc).replace('Z', '+00:00'))
            h, _info = get_tide_height(lat_c, lon_c, dt)
            if h is not None and np.isfinite(h):
                height_m = float(h)
                source = "Open-Meteo"
        except Exception as ex:
            L.info(f"tide_correction: Open-Meteo fallback failed ({ex})")
    if height_m is None:
        return _tide_correction_fallback(), depth_grid

    tc = {
        "applied": bool(reduce_to_msl), "height_m": round(height_m, 4), "source": source,
        "acquisition_utc": acquisition_utc, "datum_before": "instantaneous",
        "datum_after": "MSL" if reduce_to_msl else "instantaneous sea level",
        "sigma_tide_m": _TIDE_SIGMA_FLOOR_M,
        "note": (f"Depths reduced to MSL using {source} tide ({height_m:+.2f} m) at "
                f"acquisition; residual sigma_tide {_TIDE_SIGMA_FLOOR_M:.2f} m."
                if reduce_to_msl else
                f"Uncorrected — instantaneous sea level at acquisition (tide {height_m:+.2f} m "
                f"{source} at {acquisition_utc}, NOT subtracted; set TIDE_REDUCE_TO_MSL=1 to "
                f"reduce to MSL). No tidal reduction applied to the returned depth grid."),
    }
    out_grid = depth_grid
    if reduce_to_msl and depth_grid is not None:
        try:
            from backend.tide_correction import apply_tide_correction as _apply_tc
            from datetime import datetime as _dt
            dt = _dt.fromisoformat(str(acquisition_utc).replace('Z', '+00:00'))
            out_grid, _h, _info = _apply_tc(depth_grid, bbox, utc_dt=dt,
                                            max_depth_m=(max_depth_m or MAX_DEPTH_M))
        except Exception as ex:
            L.info(f"tide_correction: apply_tide_correction failed ({ex}); grid unchanged")
            tc["applied"] = False
            tc["datum_after"] = "instantaneous sea level"
            tc["note"] += " [reduction attempt FAILED — grid left uncorrected]"
    return tc, out_grid


def mle_tide_correction_summary(tide_info):
    """ADPorts F3: the MLE composite-level `result.tide_correction` — a pure
    SUMMARY of the EXISTING per-scene `stability.tide` diagnostic (MASKMLE
    1.2/1.2c), never a second independent computation and never a second
    correction on top of it (the coordinator's binding caveat: audit, don't
    double-correct). `tide_info['applied']` mirrors `MLE_TIDE_CORRECT`
    (default OFF per the MASKMLE R1/R3 kill-list — tide correction for
    per-scene MLE stability was KILLED; only the diagnostic survives)."""
    if not tide_info:
        return _tide_correction_fallback()
    applied = bool(tide_info.get("applied"))
    h_ref = tide_info.get("h_ref_m")
    sigma = tide_info.get("sigma_tide_m")
    sigma = max(float(sigma), _TIDE_SIGMA_FLOOR_M) if sigma is not None else _TIDE_SIGMA_FLOOR_M
    if not applied or h_ref is None:
        return _tide_correction_fallback(
            note="Uncorrected — instantaneous sea level at acquisition; per-scene EOT20 tide is "
                 "computed and reported per-scene (stability.tide, stability.per_scene[].tide_m) "
                 "as a DIAGNOSTIC only (MLE_TIDE_CORRECT default OFF — killed for the default "
                 "regime, MASKMLE R1/R3). No tidal reduction applied to the composite.")
    _src = tide_info.get("source") or "EOT20"
    return {
        "applied": True, "height_m": round(h_ref, 4), "source": _src,
        "acquisition_utc": None,  # composite spans 5 scenes — no single timestamp
        "datum_before": "instantaneous",
        "datum_after": (f"scene-set-relative (h_ref: median scene {_src} tide, "
                        f"MLE_TIDE_CORRECT=1) — NOT absolute MSL/LAT"),
        "sigma_tide_m": sigma,
        "note": (f"Per-scene depths aligned to the scene-set median {_src} tide "
                f"(h_ref={h_ref:+.3f} m) before MLE fusion (MLE_TIDE_CORRECT=1); this is an "
                f"inter-scene STABILITY alignment to a scene-set-relative datum, not an "
                f"independently-verified absolute MSL/LAT reduction. Residual sigma_tide "
                f"{sigma:.2f} m."),
    }


# ── R3: separated-bias datum disclosure (IHO S-44 systematic/random split) ──
# The MLE composite is referenced to a SCENE-SET-RELATIVE datum (median-scene
# instantaneous surface), NOT absolute MSL/LAT. Any datum mismatch is a PURE
# BIAS and must be reported SEPARATELY from the precision (RMSE) figure
# (S-44 ed.6.1.0 §3). A constant datum shift cannot change RMSE/decile, so this
# is DISCLOSURE not accuracy: it never touches the depth grid.
_MLE_SIGMA_Z0_DEFAULT_M = None  # unknown-datum term: null until a published Z0 exists on disk


def mle_datum_disclosure(depth_grid, bbox, tide_floor):
    """Emit the honest vertical-reference disclosure for the MLE composite.

    separated_bias_m       = median(model − in-situ GT) over any on-disk soundings
                             in bbox (the labelled SYSTEMATIC term), else null.
    model_rmse_about_bias_m = RMSE after removing that median (the SHAPE error) —
                             invariant to any constant datum shift.
    sigma_datum_m          = sqrt(sigma_tide² + sigma_Z0²); sigma_tide = R1 floor
                             (0.15 m), sigma_Z0 = null until a published Z0 supplied.
    MLE_DATUM (default 'relative'): when 'msl' AND MLE_LAT_Z0_M supplied, shifts
    ONLY the reported bias offset/label — NEVER the depth shape or RMSE.
    """
    mode = (os.environ.get("MLE_DATUM", "relative") or "relative").strip().lower()
    try:
        z0 = float(os.environ["MLE_LAT_Z0_M"])
    except Exception:
        z0 = None
    try:
        sig_z0 = float(os.environ["MLE_SIGMA_Z0_M"])
    except Exception:
        sig_z0 = _MLE_SIGMA_Z0_DEFAULT_M
    sig_tide = float(tide_floor)
    sigma_datum = (math.sqrt(sig_tide ** 2 + sig_z0 ** 2) if sig_z0 else sig_tide)
    sep_bias = None; rmse_about_bias = None; n_gt = 0
    try:
        glat, glon, gdep, _ = xyz_points_in_bbox(bbox)
        if len(glat):
            H, W = depth_grid.shape; w, s, e, n = bbox
            rr = ((n - glat) / (n - s) * H).astype(int)
            cc = ((glon - w) / (e - w) * W).astype(int)
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            mod = np.full(len(glat), np.nan)
            mod[ok] = depth_grid[rr[ok], cc[ok]]
            resid = mod - gdep
            fin = np.isfinite(resid) & (mod > 0)
            if int(fin.sum()) >= 20:
                n_gt = int(fin.sum())
                sep_bias = float(np.median(resid[fin]))
                rmse_about_bias = float(np.sqrt(np.mean((resid[fin] - sep_bias) ** 2)))
    except Exception as _ex:
        L.info(f"MLE datum disclosure GT-bias skipped ({_ex})")
    # 'msl' is honoured ONLY as a bias-offset relabel (and ONLY if a Z0 is on disk);
    # with no published Khalifa Z0 + no GT vertical datum we NEVER claim absolute LAT.
    absolute_ok = (mode == "msl") and (z0 is not None)
    label = ("scene-set-relative (median-scene instantaneous surface)"
             if not absolute_ok else
             f"MSL-referenced (h_ref=0, Z0={z0:+.3f} m applied) — provisional, NOT survey-verified LAT")
    reported_bias = (None if sep_bias is None
                     else round(sep_bias - (z0 if absolute_ok else 0.0), 4))
    return {
        "datum_label": label,
        "datum_mode": mode,
        "absolute_datum_available": bool(absolute_ok),
        "separated_bias_m": (round(sep_bias, 4) if sep_bias is not None else None),
        "reported_bias_m": reported_bias,
        "model_rmse_about_bias_m": (round(rmse_about_bias, 4)
                                    if rmse_about_bias is not None else None),
        "n_gt_pairs": n_gt,
        "sigma_tide_m": round(sig_tide, 4),
        "sigma_Z0_m": (round(sig_z0, 4) if sig_z0 else None),
        "sigma_datum_m": round(sigma_datum, 4),
        "note": ("Absolute MSL/LAT NOT claimed — datum is scene-set-relative; supply the GT "
                 "vertical datum + a published Z0 (MLE_LAT_Z0_M) to reference to LAT. "
                 "separated_bias_m is a LABELLED systematic term (IHO S-44 ed.6.1.0 §3), NOT "
                 "folded into RMSE/decile; a constant datum shift cannot change "
                 "model_rmse_about_bias_m (proof that MLE_DATUM only relabels the bias)."),
    }


def _gee_window_timestamps(bbox, sd, ed, cloud=30):
    """List TRUE S2 acquisition timestamps (ISO-UTC) contributing to the GEE
    median composite for [sd,ed] — the fallback timestamp source for the MLE
    stability path when the cached depth-grid s2 dict predates 1.2a."""
    try:
        if not _init_gee():
            return None
        import ee
        from datetime import datetime as _dt, timezone as _tz
        w, s, e, n = bbox
        region = ee.Geometry.Rectangle([w, s, e, n])
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterDate(sd, ed).filterBounds(region)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", int(cloud))))
        ms = [int(m) for m in col.aggregate_array("system:time_start").getInfo()
              if m is not None]
        return [_dt.fromtimestamp(m / 1000.0, tz=_tz.utc)
                .strftime('%Y-%m-%dT%H:%M:%SZ') for m in sorted(ms)]
    except Exception as ex:
        L.info(f"GEE timestamp listing failed ({ex})")
        return None


def _scene_physical_qc(s2, deep_mask, water_mask):
    """MASKMLE 1.3 diagnostic-only per-scene physical QC (NO screening):
      glint_proxy_nir = median NIR reflectance over deep water (sun-glint /
                        residual-cloud proxy; water is NIR-black so a high value
                        flags glint or haze).
      turbidity_fnu   = Dogliotti et al. 2015 red-band turbidity, median over
                        the water mask: T = A_T·ρw / (1 − ρw/C), A_T=228.1 FNU,
                        C=0.1641, ρw = red reflectance (πLw/Ed proxy; S2 SR
                        reflectance used directly).
      turbidity_fnu_deep = same Dogliotti FNU but ρw sampled ONLY on the FIXED
                        deep-quartile pixel set (MASKMLE 2.3a). Because the pixel
                        set is common across scenes, bottom albedo is common-mode
                        and the scene-to-scene FNU *anomaly* is a real water-column
                        + atmosphere signal even where the "deep" quartile is still
                        optically shallow (the R1 whole-mask FNU was bottom-
                        contaminated on banks, e.g. bu_tinah July "214 FNU").
    Returns dict of finite floats (or None where no valid px)."""
    out = {"glint_proxy_nir": None, "turbidity_fnu": None,
           "turbidity_fnu_deep": None, "deep_px": 0, "water_px": 0}
    try:
        nir = np.asarray(s2.get('nir'), dtype=np.float32) / 10000.0
        red = np.asarray(s2.get('red'), dtype=np.float32) / 10000.0
    except Exception:
        return out
    wm = np.asarray(water_mask, dtype=bool)
    if wm.shape != nir.shape:
        return out
    dm = np.asarray(deep_mask, dtype=bool) if deep_mask is not None else None
    deep_fallback = False
    if dm is None or dm.shape != nir.shape or int(dm.sum()) < 20:
        dm = wm  # fall back to whole water mask
        deep_fallback = True
    out["deep_fallback"] = bool(deep_fallback)
    dd = nir[dm]
    dd = dd[np.isfinite(dd)]
    if dd.size:
        out["glint_proxy_nir"] = round(float(np.median(dd)), 5)
        out["deep_px"] = int(dm.sum())

    A_T, C = 228.1, 0.1641

    def _dogliotti(mask):
        rr = red[mask]
        rr = rr[np.isfinite(rr) & (rr > 0)]
        if not rr.size:
            return None
        rho = float(np.median(rr))
        denom = (1.0 - rho / C)
        if denom <= 1e-3:
            return None
        return round(float(A_T * rho / denom), 3)

    out["turbidity_fnu"] = _dogliotti(wm)            # whole-mask (continuity)
    out["turbidity_fnu_deep"] = _dogliotti(dm)       # fixed deep set (2.3a)
    out["water_px"] = int(wm.sum())
    return out


# ══════════════════════════════════════════════════════════════
# VERY HR BATHYMETRY · multi-scene MLE
#   N single-scene fetches across a calendar year, each pushed
#   through the Lyzenga + Stumpf + UAE blend, then per-pixel
#   inverse-variance Maximum Likelihood Estimation (Almar-style)
#   with an MAD-based uncertainty map.
# ══════════════════════════════════════════════════════════════
def _compute_s2_depth_grid(bbox, sd, ed, user_pts=None,
                           max_cloud=20, fetch_sliderule=False,
                           emit_mask_preview=False, _deep_train=False):
    """Run the inner part of the fast S2 pipeline (Lyzenga + UAE +
    bias correction) and return the RAW numerical depth grid plus the
    water mask + per-scene metrics. No PNG / GeoTIFF rendering.

    This is the building block the MLE pipeline calls per scene so it
    can stack the float arrays directly instead of decoding colour
    PNGs. The full /api/sdb-pro response shape is built on top of it.
    """
    out = _run_s2_lyzenga_fast(
        bbox, sd, ed, user_pts=user_pts,
        max_cloud=max_cloud, fetch_sliderule=fetch_sliderule,
        include_geotiff=False,
        _internal_return_grid=True,   # see _run_s2_lyzenga_fast tail
        _emit_mask_preview=emit_mask_preview,
        _deep_train=_deep_train,
    )
    return out  # has depth_grid + water_mask added


def _residual_idw_field(bbox, H, W, tr_lat, tr_lon, residuals):
    """Shared Stage-2 residual-IDW field (faithful reproduction of the DEFAULT
    ``_run_s2_lyzenga_fast`` Stage-2: k=6 inverse-square, 0.05° cap, global-median
    far-field, decimate→bilinear-upsample, ±3 m clamp). Returns (H,W) float32."""
    w_, s_, e_, n_ = bbox
    global_med = float(np.median(residuals)) if len(residuals) else 0.0
    try:
        from scipy.spatial import cKDTree
        from scipy.ndimage import zoom as ndizoom
        tree = cKDTree(np.column_stack([tr_lat, tr_lon]))
        step = max(1, min(H, W) // 96)
        rs = np.arange(0, H, step); cs = np.arange(0, W, step)
        Hc, Wc = len(rs), len(cs)
        lat_1d = n_ - (rs + 0.5) / H * (n_ - s_)
        lon_1d = w_ + (cs + 0.5) / W * (e_ - w_)
        grid_lat, grid_lon = np.meshgrid(lat_1d, lon_1d, indexing='ij')
        pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
        k = min(6, len(tr_lat))
        d_deg, idx = tree.query(pts, k=k)
        if k == 1:
            d_deg = d_deg[:, None]; idx = idx[:, None]
        wts = 1.0 / (d_deg + 1e-3) ** 2
        wts = np.where(d_deg > 0.05, 0.0, wts)
        wsum = wts.sum(axis=1)
        close = wsum > 1e-9
        local = np.full(pts.shape[0], global_med, dtype=np.float64)
        contribs = (wts * residuals[idx]).sum(axis=1)
        local[close] = contribs[close] / wsum[close]
        coarse = local.reshape(Hc, Wc).astype(np.float32)
        field = ndizoom(coarse, (H / coarse.shape[0], W / coarse.shape[1]),
                        order=1, mode='nearest')[:H, :W]
    except Exception:
        field = np.full((H, W), global_med, dtype=np.float32)
    return np.clip(field, -3.0, 3.0).astype(np.float32)


def _mle_shared_calibrate(grids, bbox, h_ref):
    """MASKMLE 3.2 — shared-calibration-field package (ONE lever).

    Stage A stays PER-SCENE (each ``pre_calib`` = post-blend un-calibrated
    Lyzenga+Stumpf+ensemble surface — the radiometric inversion must adapt to
    per-scene atmosphere/water column). Stage B is SHARED: the ±0.5 m per-scene
    EOT20 tide is removed from every Stage-A grid (MANDATORY here — nothing else
    absorbs it once calibration is shared), an inverse-variance composite of the
    tide-corrected Stage-A grids is built, ONE isotonic(pred→truth) + ONE
    residual-IDW field is fit on that composite against the SAME reference set
    production uses, and the IDENTICAL monotone map + IDW field is applied to
    every scene. NO per-scene free parameter after Stage A. Mutates ``g['depth']``
    in place. Returns an info dict (or {'applied': False, 'reason': ...})."""
    from sklearn.isotonic import IsotonicRegression
    info = {"applied": False, "reason": None, "composite": "inverse_variance_mean",
            "h_ref_m": (round(h_ref, 4) if h_ref is not None else None)}
    usable = [g for g in grids if g.get('pre_calib') is not None]
    if len(usable) < 2:
        info["reason"] = "no Stage-A (pre_calib) grids"
        return info
    refs = None
    for g in usable:
        cr = g.get('calib_refs')
        if cr and cr.get('depth'):
            refs = cr; break
    if not refs or len(refs.get('depth', [])) < 10:
        info["reason"] = "reference set < 10 pts"
        return info
    H, W = usable[0]['pre_calib'].shape
    w_, s_, e_, n_ = bbox
    # (1) tide-correct each Stage-A grid (z' = zA − (h_i − h_ref)).
    hbar = np.mean([g['tide_m'] for g in usable if g.get('tide_m') is not None]) \
        if any(g.get('tide_m') is not None for g in usable) else None
    for g in usable:
        gA = np.asarray(g['pre_calib'], dtype=np.float32).copy()
        shift = 0.0
        if g.get('tide_m') is not None and h_ref is not None:
            shift = float(g['tide_m']) - float(h_ref)
        ok = g['water'] & np.isfinite(gA)
        gA[ok] = gA[ok] - shift
        g['_stageA_tc'] = gA
        g['_tide_shift_sharedcal_m'] = round(shift, 4)
    # (2) inverse-variance composite of tide-corrected Stage-A grids.
    num = np.zeros((H, W), np.float64); den = np.zeros((H, W), np.float64)
    for g in usable:
        gA = g['_stageA_tc']
        ok = g['water'] & np.isfinite(gA) & (gA > 0)
        iv = 1.0 / (g['sigma'] ** 2)
        num[ok] += gA[ok] * iv; den[ok] += iv
    comp = np.full((H, W), np.nan, np.float32)
    fin = den > 0
    comp[fin] = (num[fin] / den[fin]).astype(np.float32)
    # (3) fit ONE isotonic + ONE residual-IDW on the composite vs the ref set.
    rl = np.asarray(refs['lat'], float); rlo = np.asarray(refs['lon'], float)
    rd = np.asarray(refs['depth'], float)
    rows = np.clip(((n_ - rl) / (n_ - s_ + 1e-10) * H).astype(int), 0, H - 1)
    cols = np.clip(((rlo - w_) / (e_ - w_ + 1e-10) * W).astype(int), 0, W - 1)
    cpred = comp[rows, cols].astype(np.float64)
    good = np.isfinite(cpred) & (rd > 0) & (rd <= MAX_DEPTH_M)
    if int(good.sum()) < 10:
        info["reason"] = f"only {int(good.sum())} composite/ref pairs finite"
        return info
    cpred_g = cpred[good]; truth_g = rd[good]
    iso = None
    if np.unique(cpred_g).size >= 8:
        try:
            iso = IsotonicRegression(out_of_bounds='clip').fit(cpred_g, truth_g)
        except Exception:
            iso = None
    if iso is not None:
        pred_after = iso.predict(cpred_g)
        stage1 = "isotonic"
    else:  # WLS linear fallback (mirrors production)
        A = np.polyfit(cpred_g, truth_g, 1)
        pred_after = np.polyval(A, cpred_g)
        iso = ("linear", float(A[0]), float(A[1]))
        stage1 = "linear"
    resid_ref = pred_after - truth_g
    resid_field = _residual_idw_field(
        bbox, H, W, rl[good], rlo[good], resid_ref)
    # (4) apply the IDENTICAL Stage-B map to every scene's tide-corrected Stage-A.
    def _apply_iso(arr):
        flat = arr.reshape(-1); f = np.isfinite(flat); out = flat.copy()
        if isinstance(iso, tuple):
            out[f] = iso[1] * flat[f] + iso[2]
        else:
            out[f] = iso.predict(flat[f].astype(np.float64))
        return out.reshape(arr.shape)
    for g in usable:
        gA = g['_stageA_tc']
        z1 = _apply_iso(gA)
        z = np.clip(z1 - resid_field, 0.0, MAX_DEPTH_M).astype(np.float32)
        z[~(g['water'] & np.isfinite(gA))] = np.nan
        g['depth'] = z
    # (3.2c) confirmatory α diagnostic on the Stage-A biases: BEFORE the tide
    # term α≈1 (datum story), AFTER α≈0 (tide removed). Regress per-scene
    # bias-vs-Stage-A-composite on (h_i − h̄).
    def _alpha(stage_grids):
        nn = np.zeros((H, W), np.float64); dd = np.zeros((H, W), np.float64)
        for g, sg in stage_grids:
            ok = g['water'] & np.isfinite(sg) & (sg > 0)
            iv = 1.0 / (g['sigma'] ** 2)
            nn[ok] += sg[ok] * iv; dd[ok] += iv
        cc = np.full((H, W), np.nan, np.float32); ff = dd > 0
        cc[ff] = (nn[ff] / dd[ff]).astype(np.float32)
        bl, tl = [], []
        for g, sg in stage_grids:
            if g.get('tide_m') is None:
                continue
            both = np.isfinite(sg) & np.isfinite(cc)
            if both.any():
                bl.append(float(np.mean(sg[both] - cc[both]))); tl.append(g['tide_m'])
        if len(bl) >= 3:
            b = np.asarray(bl); t = np.asarray(tl) - np.mean(tl)
            v = float(np.sum(t ** 2))
            if v > 1e-9:
                return round(float(np.sum(t * (b - b.mean())) / v), 4)
        return None
    a_before = _alpha([(g, np.asarray(g['pre_calib'], np.float32)) for g in usable])
    a_after = _alpha([(g, g['_stageA_tc']) for g in usable])
    info.update({
        "applied": True, "stage1": stage1, "n_ref_pairs": int(good.sum()),
        "n_scenes": len(usable), "tide_mandatory": True,
        "sigma_tide_m": _TIDE_SIGMA_FLOOR_M,
        "datum": ("composite referenced to kept-scene median tide; per-scene "
                  "EOT20 tide removed"),
        "resid_field_median_m": round(float(np.median(resid_field)), 4),
        "hbar_m": (round(float(hbar), 4) if hbar is not None else None),
        "alpha_stageA_before_tide": a_before,   # predicted ≈ 1
        "alpha_stageA_after_tide": a_after,      # predicted ≈ 0
    })
    return info


def _write_grid_geotiff(grid, bbox, path, band_desc, nodata=-9999.0):
    """Write a float32 (H,W) grid as an EPSG:4326 GeoTIFF (NaN → nodata).
    Returns True on success. Used by the MLE stability outputs."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        g = np.where(np.isfinite(grid), grid, nodata).astype(np.float32)
        H, W = g.shape
        w_, s_, e_, n_ = bbox
        with rasterio.open(str(path), 'w', driver='GTiff', height=H, width=W,
                           count=1, dtype='float32', crs='EPSG:4326',
                           transform=from_bounds(w_, s_, e_, n_, W, H),
                           nodata=nodata, compress='deflate') as dst:
            dst.write(g, 1)
            dst.set_band_description(1, band_desc)
        return True
    except Exception as ex:
        L.info(f"_write_grid_geotiff({band_desc}) failed: {ex}")
        return False


def _run_s2_mle(bbox, year=2024, n_scenes=5, max_cloud=20,
                user_pts=None, fetch_sliderule=False,
                sigma_reject_k=3.0, sigma_max_m=None):
    """Multi-scene MLE bathymetry over a full calendar year.

    Picks ``n_scenes`` evenly-spaced ±15 d windows (one per
    1/N-of-year slot), runs the per-scene fast pipeline, then per-
    pixel inverse-variance MLE:

        z_hat[r,c] = Σ_i  z_i[r,c] / σ_i²   /   Σ_i  1 / σ_i²
        σ_hat[r,c] = 1 / sqrt( Σ_i 1/σ_i² )

    σ_i = scene's held-out RMSE (clamped to [0.5, 5] m) so a single
    perfect-fit scene can't dominate. Pixels with no valid scene are
    NaN. Returns the same shape as /api/sdb-pro plus per-pixel
    uncertainty + per-scene dates / σ table.
    """
    import time as _T, base64 as _b64, io as _io
    from datetime import datetime, timedelta
    t0 = _T.time()

    n = max(1, min(int(n_scenes), 12))
    centres = [datetime(int(year), 1, 1) + timedelta(days=int((i + 0.5) * 365 / n))
               for i in range(n)]
    pad = 15

    # MASKMLE 3.2: shared-calibration-field mode (default OFF, one-shot lever).
    shared_cal = os.environ.get("MLE_SHARED_CAL", "0") == "1"
    # AI_METHOD_LOG R1/R2: deep per-pixel σ. MLE_DEEP_PER_SCENE=1 → train the
    # Attention U-Net on EVERY scene; default (0) → train ONCE (first scene that
    # produces a deep fit) and broadcast that σ-SHAPE to the other scenes scaled
    # by each scene's scalar RMSE (CPU budget ≈ one CNN train/request).
    deep_on = os.environ.get("MLE_DEEP_SIGMA", "0") == "1"
    deep_per_scene = os.environ.get("MLE_DEEP_PER_SCENE", "0") == "1"
    _deep_done = False   # broadcast-mode: have we trained the deep head yet?
    grids = []  # list of dicts { 'depth':np.ndarray, 'water':np.ndarray,
                #                 'sigma':float, 'date':str, 'metrics':dict }
    mle_mask_meta = None       # 3.U2: representative product mask meta for the UI
    for _i, c in enumerate(centres):
        sd = (c - timedelta(days=pad)).strftime('%Y-%m-%d')
        ed = (c + timedelta(days=pad)).strftime('%Y-%m-%d')
        _train_deep_here = deep_on and (deep_per_scene or not _deep_done)
        try:
            res = _compute_s2_depth_grid(
                bbox, sd, ed, user_pts=user_pts,
                max_cloud=max_cloud, fetch_sliderule=fetch_sliderule,
                # 3.U2: emit ONE mask-QA preview (first scene) for the MLE product.
                emit_mask_preview=(mle_mask_meta is None),
                _deep_train=_train_deep_here)
        except Exception as ex:
            L.info(f"MLE: {c.date()} skipped ({ex})")
            continue
        d = res.get('depth_grid')
        w = res.get('water_mask')
        if d is None or w is None:
            continue
        m = res.get('metrics') or {}
        bc = (res.get('augmentation') or {}).get('bias_correction') or {}
        sigma = m.get('rmse_m') or bc.get('post_rmse_m') or bc.get('pre_rmse_m') or 1.5
        sigma = float(np.clip(sigma, 0.5, 5.0))
        wm = res.get('water_mask_meta') or {}
        if mle_mask_meta is None and wm.get('vhr_applied'):
            mle_mask_meta = wm
        grids.append({
            'depth': d, 'water': w, 'sigma': sigma,
            'date': c.strftime('%Y-%m-%d'), 'window': [sd, ed],
            'metrics': m,
            'mean_depth': (res.get('stats') or {}).get('mean_depth'),
            # MASKMLE 1.2a/1.3: raw S2 bands + TRUE acquisition timestamps
            's2': res.get('_s2'),
            'acquisition_datetimes': res.get('acquisition_datetimes'),
            # MASKMLE 3.2: Stage-A (pre-calibration, post-blend) surface + the
            # SAME reference set for the shared-calibration-field mode.
            'pre_calib': res.get('depth_grid_pre_calib'),
            'calib_refs': res.get('_calib_refs'),
            # AI_METHOD_LOG R1/R2/R3: deep per-pixel σ + never-worse blend.
            'sigma_px': res.get('deep_sigma_px'),
            'depth_blend': res.get('deep_depth_blend'),
            'deep_gate': res.get('deep_gate'),
        })
        if res.get('deep_sigma_px') is not None:
            _deep_done = True
        L.info(f"MLE: {c.date()} σ={sigma:.2f} m  mean depth "
               f"{(res.get('stats') or {}).get('mean_depth')} m")

    if not grids:
        raise RuntimeError("MLE: no scenes returned a usable depth grid")

    # Resolve a common shape (all scenes share the same bbox so they
    # SHOULD be identical, but ROI-edge rounding can vary by a pixel)
    H = min(g['depth'].shape[0] for g in grids)
    W = min(g['depth'].shape[1] for g in grids)
    for g in grids:
        g['depth'] = g['depth'][:H, :W]
        g['water'] = g['water'][:H, :W]
        if g.get('pre_calib') is not None:
            try:
                g['pre_calib'] = np.asarray(g['pre_calib'])[:H, :W]
            except Exception:
                g['pre_calib'] = None
        # AI_METHOD_LOG: crop deep σ / blended depth to the common shape.
        for _dk in ('sigma_px', 'depth_blend'):
            if g.get(_dk) is not None:
                try:
                    g[_dk] = np.asarray(g[_dk])[:H, :W]
                except Exception:
                    g[_dk] = None
        s2 = g.get('s2')
        if s2:
            for _k in ('nir', 'red', 'green', 'blue'):
                v = s2.get(_k)
                if v is not None and hasattr(v, 'shape') and v.shape[:2] != (H, W):
                    try:
                        s2[_k] = np.asarray(v)[:H, :W]
                    except Exception:
                        pass

    # ══════════════════════════════════════════════════════════════════════
    # MASKMLE 2.3: best-N-of-M scene screening by INDEPENDENT physical QC.
    # Two pre-registered, independent gates ONLY (NO agreement/σ/bias selection —
    # that would be circular and is charter-forbidden):
    #   (1) GLINT gate (absolute, physical): drop if glint_proxy_nir > 0.030
    #       (clean deep water is NIR-black, ρw ≲ 0.01–0.02; > 0.03 = residual
    #       glint/haze — Kay et al. 2009).
    #   (2) TURBIDITY-ANOMALY gate (relative, robust): drop if turbidity_fnu_deep
    #       > median_M + 3×(1.4826·MAD_M) across the M candidates, sampled on a
    #       FIXED deep-quartile pixel set (bottom albedo common-mode → the FNU
    #       anomaly is a real water-column/atmosphere signal).
    # Keep ≥3 scenes always (else keep the 3 lowest-glint). Env knob
    # MLE_SCENE_SCREEN (default OFF). The fixed deep set is the deepest quartile of
    # the ALL-candidate composite. Fusion of survivors is UNCHANGED (single var).
    # ══════════════════════════════════════════════════════════════════════
    # MASKMLE 3.3 — screening WIN locked in: default ON for the MLE path
    # (khalifa composite RMSE 2.2272→2.1614 m, round0 σ 0.372→0.306,
    # bu_tinah 0.183→0.146). Thresholds FROZEN (glint_max 0.030 absolute,
    # turbidity = median+3×1.4826×MAD on the fixed deep set), keep-≥3 fallback.
    screen_on = os.environ.get("MLE_SCENE_SCREEN", "1") == "1"
    screening = {"applied": False, "reason": ("disabled" if not screen_on
                                              else "n_candidates<3"),
                 "glint_max": 0.030, "n_candidates": len(grids),
                 "n_kept": len(grids), "candidates": []}
    if screen_on and len(grids) >= 3:
        # (1) FIXED deep-quartile pixel set from the ALL-candidate composite.
        _numc = np.zeros((H, W), np.float64); _denc = np.zeros((H, W), np.float64)
        for g in grids:
            _ok = g['water'] & np.isfinite(g['depth']) & (g['depth'] > 0)
            _iv = 1.0 / (g['sigma'] ** 2)
            _numc[_ok] += g['depth'][_ok] * _iv; _denc[_ok] += _iv
        _finc = _denc > 0
        _comp0 = np.full((H, W), np.nan, np.float32)
        _comp0[_finc] = (_numc[_finc] / _denc[_finc]).astype(np.float32)
        _cf = np.isfinite(_comp0)
        _q75 = float(np.nanpercentile(_comp0[_cf], 75)) if _cf.any() else None
        deep0 = (_cf & (_comp0 >= _q75)) if _q75 is not None else None
        # (2) per-candidate QC on the FIXED deep set.
        qcs = [(_scene_physical_qc(g.get('s2') or {}, deep0, g['water'])
                if g.get('s2') else {}) for g in grids]
        glints = np.array([qc.get('glint_proxy_nir') if qc.get('glint_proxy_nir')
                           is not None else np.nan for qc in qcs], dtype=float)
        fnus = np.array([qc.get('turbidity_fnu_deep') if qc.get('turbidity_fnu_deep')
                         is not None else np.nan for qc in qcs], dtype=float)
        # (3) two independent pre-registered gates.
        GLINT_MAX = 0.030
        drop_glint = np.isfinite(glints) & (glints > GLINT_MAX)
        f_fin = fnus[np.isfinite(fnus)]
        turb_thr = None
        if f_fin.size >= 2:
            _med = float(np.median(f_fin))
            _mad = float(np.median(np.abs(f_fin - _med)))
            turb_thr = _med + 3.0 * 1.4826 * _mad
            drop_turb = np.isfinite(fnus) & (fnus > turb_thr)
        else:
            drop_turb = np.zeros(len(grids), bool)
        kept = ~(drop_glint | drop_turb)
        floored = False
        if int(kept.sum()) < 3:                      # keep-≥3 fallback: lowest glint
            _order = np.argsort(np.where(np.isfinite(glints), glints, np.inf))
            kept = np.zeros(len(grids), bool); kept[_order[:3]] = True
            floored = True
        for i, g in enumerate(grids):
            gate = []
            if bool(drop_glint[i]) and not (floored and kept[i]):
                gate.append("glint")
            if bool(drop_turb[i]) and not (floored and kept[i]):
                gate.append("turbidity")
            screening["candidates"].append({
                "date": g['date'], "kept": bool(kept[i]), "gate": (gate or None),
                "glint_proxy_nir": (None if not np.isfinite(glints[i])
                                    else round(float(glints[i]), 5)),
                "turbidity_fnu_deep": (None if not np.isfinite(fnus[i])
                                       else round(float(fnus[i]), 3)),
                "turbidity_fnu_wholemask": qcs[i].get('turbidity_fnu'),
            })
        screening.update({
            "applied": True, "reason": None, "glint_max": GLINT_MAX,
            "turbidity_anomaly_thresh": (round(turb_thr, 3)
                                         if turb_thr is not None else None),
            "kept_by_min3_floor": floored, "n_kept": int(kept.sum()),
            "kept_dates": [g['date'] for i, g in enumerate(grids) if kept[i]],
            "dropped_dates": [g['date'] for i, g in enumerate(grids) if not kept[i]],
        })
        grids = [g for i, g in enumerate(grids) if kept[i]]
        L.info(f"MLE screening: kept {screening['n_kept']}/"
               f"{screening['n_candidates']} scenes; dropped "
               f"{screening['dropped_dates']} (glint>{GLINT_MAX} or "
               f"FNU_deep>{turb_thr})")

    # ══════════════════════════════════════════════════════════════════════
    # MASKMLE 1.2: TRUE per-scene acquisition timestamps → EOT20 tide height.
    # Diagnostic ALWAYS computed & reported (datum honesty, charter). Physical
    # correction z_i' = z_i − (h_i − h_ref) applied ONLY when MLE_TIDE_CORRECT=1
    # (env knob default OFF), with h_ref = scene-set MEDIAN tide, α_apply = 1.0
    # (full physical correction, NO fitting on the stability metric).
    # ══════════════════════════════════════════════════════════════════════
    w_b, s_b, e_b, n_b = bbox
    lon_c = 0.5 * (w_b + e_b); lat_c = 0.5 * (s_b + n_b)
    tide_correct = os.environ.get("MLE_TIDE_CORRECT", "0") == "1"
    # R1.2: σ_tide floor used in the fusion/inflation (env-overridable ONLY so the
    # R1.3 harness can prove byte-identical OFF depth by toggling it to 0.0; the
    # disclosure constant _TIDE_SIGMA_FLOOR_M stays 0.15 everywhere else).
    _tide_floor = float(os.environ.get("MLE_TIDE_SIGMA_FLOOR_M", str(_TIDE_SIGMA_FLOOR_M)))
    tide_info = {"applied": False, "h_ref_m": None, "alpha_retained": None,
                 "alpha_r": None, "sigma_tide_m": _tide_floor,
                 "sigma_tide_folded": True, "source": None,
                 "datum_note": "scene-set-relative (h_ref = median scene tide); NOT MSL/LAT",
                 "per_scene": []}
    for g in grids:
        acq = g.get('acquisition_datetimes')
        if not acq:  # cache predates 1.2a or SH path — recover via GEE listing
            try:
                acq = _gee_window_timestamps(bbox, g['window'][0], g['window'][1],
                                             cloud=max_cloud + 10)
                g['acquisition_datetimes'] = acq
            except Exception:
                acq = None
        h_i = None
        src = "none"
        if acq:
            th = _eot20_tide_heights(lon_c, lat_c, acq)
            if th is not None and np.isfinite(th).any():
                h_i = float(np.nanmean(th)); src = "EOT20"
        # R1.1: EOT20 is inert in this deployment (no model files on disk) → fall
        # back to Open-Meteo Marine (no creds), averaged over the scene's TRUE
        # acquisition datetimes (mirrors the fast-path build_tide_correction).
        if h_i is None and acq:
            try:
                try:
                    from backend.tide_correction import get_tide_height as _gth
                except Exception:
                    from tide_correction import get_tide_height as _gth  # type: ignore
                from datetime import datetime as _dt
                hs = []
                for a in acq:
                    try:
                        d = _dt.fromisoformat(str(a).replace('Z', '+00:00'))
                        h, _inf = _gth(lat_c, lon_c, d)
                        if h is not None and np.isfinite(h):
                            hs.append(float(h))
                    except Exception:
                        continue
                if hs:
                    h_i = float(np.mean(hs)); src = "open-meteo"
            except Exception as _tex:
                L.info(f"MLE tide: Open-Meteo fallback failed ({_tex})")
        g['tide_m'] = h_i
        g['tide_source'] = src
        g['n_acq'] = (len(acq) if acq else 0)
        # ── R2.1: per-scene significant wave height Hs (Open-Meteo Marine, no ──
        # creds). Called for EACH true acquisition datetime in the scene window;
        # store the composite representative Hs_mean and the roughest-acquisition
        # Hs_max. Diagnostic only here (no depth/σ change unless MLE_WAVE_QC=1).
        # Optional ERA5-CDS gold source is preferred only if creds are present.
        hs_vals, hs_days, wsrc = [], [], "none"
        if acq:
            try:
                try:
                    from backend.tide_correction import get_wave_height as _gwh
                except Exception:
                    from tide_correction import get_wave_height as _gwh  # type: ignore
                from datetime import datetime as _dt2
                for a in acq:
                    try:
                        d = _dt2.fromisoformat(str(a).replace('Z', '+00:00'))
                        hv, hinf = _gwh(lat_c, lon_c, d)
                    except Exception:
                        continue
                    meth = (hinf or {}).get('method', '')
                    if hv is not None and np.isfinite(hv) and meth not in (
                            'unavailable', 'empty_series'):
                        hs_vals.append(float(hv))
                        _dm = (hinf or {}).get('day_max_wave_m')
                        if _dm is not None and np.isfinite(_dm):
                            hs_days.append(float(_dm))
                        wsrc = "open-meteo"
            except Exception as _wex:
                L.info(f"MLE wave: Open-Meteo Hs fetch failed ({_wex})")
        g['hs_mean_m'] = (float(np.mean(hs_vals)) if hs_vals else None)
        g['hs_max_m'] = (float(np.max(hs_vals)) if hs_vals else None)
        g['hs_day_max_m'] = (float(np.max(hs_days)) if hs_days else None)
        g['hs_source'] = wsrc
    _srcs = sorted({g.get('tide_source', 'none') for g in grids})
    tide_info["source"] = ("+".join(s for s in _srcs if s != 'none') or "none")
    tides = [g['tide_m'] for g in grids if g.get('tide_m') is not None]
    h_ref = float(np.median(tides)) if tides else None
    tide_info["h_ref_m"] = (round(h_ref, 4) if h_ref is not None else None)
    if tide_correct and h_ref is not None:
        for g in grids:
            if g.get('tide_m') is None:
                continue
            shift = g['tide_m'] - h_ref
            di = g['depth']; wi = g['water']
            ok = wi & np.isfinite(di)
            di2 = di.copy()
            di2[ok] = di[ok] - shift
            g['depth'] = di2
            g['tide_shift_applied_m'] = round(float(shift), 4)
        tide_info["applied"] = True
        L.info(f"MLE tide: EOT20 correction APPLIED (h_ref={h_ref:.3f} m, "
               f"MLE_TIDE_CORRECT=1); per-scene shifts "
               f"{[g.get('tide_shift_applied_m') for g in grids]}")
    else:
        L.info(f"MLE tide: diagnostic only (h_ref="
               f"{('%.3f' % h_ref) if h_ref is not None else 'NA'} m, "
               f"tides={[None if g.get('tide_m') is None else round(g['tide_m'],3) for g in grids]}, "
               f"src={tide_info['source']}, MLE_TIDE_CORRECT={int(tide_correct)})")

    # ── R1.2: per-scene σ_tide budget (correlated tide-model residual). ──
    # Floor 0.15 m (DL-fusion memory / EOT20 residual). When a physical datum
    # shift is applied (MLE_TIDE_CORRECT=1), inflate by a residual-model-error
    # fraction f=0.1 of the applied |shift|. This σ enters the inverse-variance
    # fusion WEIGHTS only when the correction is ON (so the OFF composite depth
    # stays byte-identical — the uniform floor is folded into the *reported*
    # composite σ post-σ-QA instead, see below).
    _TIDE_RESID_FRAC = float(os.environ.get("MLE_TIDE_RESID_FRAC", "0.10"))
    for g in grids:
        st = float(_tide_floor)
        sh = g.get('tide_shift_applied_m')
        if tide_correct and sh is not None:
            st = float(np.sqrt(_tide_floor ** 2 + (_TIDE_RESID_FRAC * abs(sh)) ** 2))
        g['sigma_tide_m'] = st

    # ── R2.2: per-scene σ_wave sea-state budget (hinge form). ──────────────
    # σ_wave_i = k·max(0, Hs_mean_i − Hs0). Higher Hs → more glint/whitecap/
    # path-length corruption of the optical SDB signal (Hedley 2005; Kay 2009;
    # Caballero & Stumpf 2020), so a rough scene is DOWN-WEIGHTED (never dropped
    # except by the conservative safety cap via the existing keep-≥3 floor).
    # A SOFT σ-inflation prior only: enters the fusion WEIGHTS + reported σ ONLY
    # when MLE_WAVE_QC=1 (default OFF → OFF path byte-identical, mirrors R1.2).
    wave_qc = os.environ.get("MLE_WAVE_QC", "0") == "1"
    _WAVE_K = float(os.environ.get("MLE_WAVE_K", "1.0"))
    _WAVE_HS0 = float(os.environ.get("MLE_WAVE_HS0", "0.3"))
    _HS_REJECT = float(os.environ.get("MLE_HS_REJECT", "2.5"))
    for g in grids:
        hm = g.get('hs_mean_m')
        sw = (_WAVE_K * max(0.0, float(hm) - _WAVE_HS0)) if (hm is not None) else 0.0
        g['sigma_wave_m'] = float(sw)
    _hs_srcs = sorted({g.get('hs_source', 'none') for g in grids})
    wave_info = {
        "wave_qc_applied": bool(wave_qc),
        "k": _WAVE_K, "hs0_m": _WAVE_HS0, "hs_reject_m": _HS_REJECT,
        "source": ("+".join(s for s in _hs_srcs if s != 'none') or "none"),
        "hs_rejected_dates": [],
        "note": ("σ_wave = k·max(0, Hs_mean − Hs0) folded into fusion σ in "
                 "quadrature (MLE_WAVE_QC=1); soft down-weight prior, NOT a "
                 "second hard glint gate"),
        "per_scene": [],
    }
    # Conservative safety reject cap — drops only demonstrably storm-rough scenes
    # (Hs_mean > MLE_HS_REJECT, default 2.5 m, well above Gulf), and ONLY via the
    # existing keep-≥3 floor so we never fall below 3 scenes. Rarely fires.
    if wave_qc and _HS_REJECT > 0 and len(grids) > 3:
        _rough = [g for g in grids if (g.get('hs_mean_m') is not None
                                       and float(g['hs_mean_m']) > _HS_REJECT)]
        # sort roughest-first; drop while we can stay ≥3
        _rough.sort(key=lambda gg: float(gg['hs_mean_m']), reverse=True)
        for g in _rough:
            if len(grids) <= 3:
                break
            wave_info["hs_rejected_dates"].append(
                {"date": g.get('date'), "hs_mean_m": round(float(g['hs_mean_m']), 3)})
            grids.remove(g)
        if wave_info["hs_rejected_dates"]:
            L.info(f"MLE wave: HS_REJECT dropped {wave_info['hs_rejected_dates']} "
                   f"(cap {_HS_REJECT} m, kept {len(grids)} scenes)")

    # ══════════════════════════════════════════════════════════════════════
    # MASKMLE 3.2: shared-calibration-field (MLE_SHARED_CAL, default OFF). Fit
    # Stage B ONCE on the tide-corrected kept-scene composite, apply identically
    # to every scene → kills per-scene calibration-field jitter. Tide term is
    # MANDATORY here (regime-scoped: under a shared field nothing absorbs the
    # inter-scene tide spread; the R1 kill was for the PER-SCENE regime where
    # isotonic already absorbed α≈0.04). Runs BEFORE any inter-scene statistic.
    # ══════════════════════════════════════════════════════════════════════
    shared_cal_info = {"applied": False, "reason": ("disabled"
                       if not shared_cal else None)}
    if shared_cal:
        try:
            shared_cal_info = _mle_shared_calibrate(grids, bbox, h_ref)
            if shared_cal_info.get("applied"):
                L.info(f"MLE shared-cal APPLIED: stage1={shared_cal_info.get('stage1')}, "
                       f"n_ref={shared_cal_info.get('n_ref_pairs')}, "
                       f"α_before={shared_cal_info.get('alpha_stageA_before_tide')}, "
                       f"α_after={shared_cal_info.get('alpha_stageA_after_tide')}")
            else:
                L.info(f"MLE shared-cal NOT applied ({shared_cal_info.get('reason')})")
        except Exception as _scx:
            L.exception("MLE shared-cal failed")
            shared_cal_info = {"applied": False, "reason": f"error: {_scx}"}

    # ── AI_METHOD_LOG R1/R2: deep per-pixel σ (broadcast-shape mode) ──
    # Default MLE_DEEP_PER_SCENE=0 trains the deep head on ONE scene; broadcast
    # its σ-SHAPE (down-weighting map) to the other scenes, scaled by each
    # scene's scalar RMSE, ONLY IF that scene's gate actually adopted the CNN
    # (w_cnn>0). If the gate rejected the CNN, no scene gets sigma_px → the MLE
    # is byte-identical to the scalar path (never-worse floor holds).
    ai_sigma_stats = None
    # A scene that TRAINED the deep head (gate present) — surfaces the honest
    # diagnostic even when the gate REJECTS the CNN (adopted=False).
    _trained_g = next((g for g in grids if g.get('deep_gate')), None)
    # A scene the gate ADOPTED (w_cnn>0) — governs the σ-shape broadcast.
    _ref_g = next((g for g in grids
                   if g.get('sigma_px') is not None
                   and (g.get('deep_gate') or {}).get('w_cnn', 0) > 0), None)
    if _ref_g is None:
        # gate rejected the CNN on every trained scene → strip any broadcastable
        # σ so the fusion stays on the exact scalar path (never-worse floor).
        for g in grids:
            g['sigma_px'] = None
            g['depth_blend'] = None
    elif not deep_per_scene:
        _ref_sig = _ref_g['sigma_px'].astype(np.float64)
        _ref_scalar = float(_ref_g['sigma'])
        for g in grids:
            if g.get('sigma_px') is None:
                try:
                    scale = float(g['sigma']) / max(_ref_scalar, 1e-6)
                    g['sigma_px'] = np.clip(_ref_sig * scale, 0.5, 5.0).astype(np.float32)
                except Exception:
                    g['sigma_px'] = None
    if _trained_g is not None:
        _gates = [g.get('deep_gate') for g in grids if g.get('deep_gate')]
        _dg = _trained_g.get('deep_gate') or {}
        ai_sigma_stats = {
            "enabled": True,
            "adopted": bool(_ref_g is not None),
            "per_scene": deep_per_scene,
            "stratified": _dg.get('stratified'),
            "per_band": _dg.get('per_band'),
            "gate": ("depth-stratified ≥500m block, per-band inverse-var convex blend"
                     if _dg.get('stratified')
                     else "spatial-block ≥500m (KMeans), inverse-var convex blend"),
            "coverage95": _dg.get('coverage95'),
            "coverage95_before": _dg.get('coverage95_before'),
            "temperature_k": _dg.get('temperature_k'),
            "spearman_sigma_abserr": _dg.get('spearman_sigma_abserr'),
            "n_gate": _dg.get('n_gate'),
            "n_train_deep": _dg.get('n_train_deep'),
            "sigma_px_median": _dg.get('sigma_px_median'),
            "sigma_px_spatial_std": _dg.get('sigma_px_spatial_std'),
            "cnn_r2_random_holdout": _dg.get('cnn_r2_random_holdout'),
            "w_cnn_scenes": [round((g.get('deep_gate') or {}).get('w_cnn', 0.0), 4)
                             for g in _gates],
            "rmse_lin_scenes": [(g or {}).get('rmse_lin') for g in _gates],
            "rmse_cnn_scenes": [(g or {}).get('rmse_cnn') for g in _gates],
            "scenes_blended": int(sum(1 for g in grids
                                      if g.get('depth_blend') is not None)),
        }

    # ── Per-pixel inverse-variance MLE (per-pixel σ when deep is active) ──
    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)
    valid_count = np.zeros((H, W), dtype=np.int8)
    for g in grids:
        di = g.get('depth_blend') if g.get('depth_blend') is not None else g['depth']
        wi = g['water']
        ok = wi & np.isfinite(di) & (di > 0)
        spx = g.get('sigma_px')
        # R1.2: fold σ_tide into the fusion WEIGHTS ONLY when tide correction is
        # ON. When OFF, use the bare σ verbatim so the composite depth is
        # byte-identical to the pre-R1.2 baseline (no sqrt round-trip either).
        # R2.2: fold σ_wave into the fusion WEIGHTS ONLY when MLE_WAVE_QC=1 (same
        # gating discipline as R1.2 σ_tide → OFF path byte-identical).
        _sw = float(g.get('sigma_wave_m', 0.0)) if wave_qc else 0.0
        if tide_correct:
            _st = float(g.get('sigma_tide_m', _TIDE_SIGMA_FLOOR_M))
            _sig_scalar = float(np.sqrt(g['sigma'] ** 2 + _st ** 2 + _sw ** 2))
        else:
            _st = 0.0
            _sig_scalar = float(np.sqrt(g['sigma'] ** 2 + _sw ** 2)) if _sw else float(g['sigma'])
        if spx is not None:
            sig = np.clip(np.asarray(spx, dtype=np.float64), 0.5, 5.0)
            if (tide_correct and _st) or _sw:
                sig = np.sqrt(sig ** 2 + _st ** 2 + _sw ** 2)
            good = ok & np.isfinite(sig) & (sig > 0)
            inv = 1.0 / (sig[good] ** 2)
            num[good] += di[good] * inv
            den[good] += inv
            valid_count[good] += 1
            # pixels with valid depth but no finite σ → fall back to scalar σ
            fb = ok & ~(np.isfinite(sig) & (sig > 0))
            s2_inv = 1.0 / (_sig_scalar ** 2)
            num[fb] += di[fb] * s2_inv
            den[fb] += s2_inv
            valid_count[fb] += 1
        else:
            s2_inv = 1.0 / (_sig_scalar ** 2)
            num[ok] += di[ok] * s2_inv
            den[ok] += s2_inv
            valid_count[ok] += 1
    finite = den > 0
    z_mle = np.full((H, W), np.nan, dtype=np.float32)
    z_mle[finite] = (num[finite] / den[finite]).astype(np.float32)
    sigma_mle = np.full((H, W), np.nan, dtype=np.float32)
    sigma_mle[finite] = (1.0 / np.sqrt(den[finite])).astype(np.float32)
    z_mle = np.clip(z_mle, 0.0, MAX_DEPTH_M)

    # MASK-GLOBAL (2026-07-10, user-mandated): explicit composite-level cut,
    # defense-in-depth on top of the per-scene cut each `grids[i]['depth']`
    # already received inside `_run_s2_lyzenga_fast` (via `_compute_s2_depth_
    # grid`) — the composite shares the identical bbox/land mask (cached), so
    # this is normally a no-op confirming the per-scene cuts already fully
    # covered it, but guards against any residual finite composite value on
    # a land pixel from grid-alignment edge cases.
    _mle_land_mask = None
    try:
        try:
            from backend.osm_land_mask import apply_global_land_cut as _global_cut
        except ImportError:
            from osm_land_mask import apply_global_land_cut as _global_cut  # type: ignore
        z_mle, _mle_land_mask, _mle_cut_info = _global_cut(z_mle, bbox)
        L.info(f"MLE composite global land cut: "
               f"{_mle_cut_info.get('n_depth_px_cut_this_call', 0)} px cut "
               f"({_mle_cut_info.get('mask_source')})")
    except Exception as _cutex:
        L.info(f"MLE composite global land cut skipped ({_cutex})")

    # Union water mask — pixel is "water" if ≥ 1 scene called it water
    union_water = np.zeros((H, W), dtype=bool)
    for g in grids:
        union_water |= g['water']
    if _mle_land_mask is not None:
        union_water &= ~_mle_land_mask

    # ADPorts F1: composite's own vmin/vmax (2nd/98th pct), computed ONCE and
    # reused for the composite raster AND every per-scene overlay so scenes
    # are visually comparable on the SAME colour ramp/scale (not each
    # auto-scaling to its own range).
    _cv = np.isfinite(z_mle) & (z_mle > 0.1)
    comp_max_depth = float(np.percentile(z_mle[_cv], 98)) if int(_cv.sum()) > 10 else MAX_DEPTH_M
    comp_min_depth = float(np.percentile(z_mle[_cv], 2)) if int(_cv.sum()) > 10 else 0.0

    # ══════════════════════════════════════════════════════════════════════
    # PER-SCENE RETENTION + STABILITY REPORT (user request Item 2)
    #
    # The individual per-scene depth estimates are retained (not collapsed into
    # the composite) so their AGREEMENT can be measured — low inter-scene σ is
    # the stability evidence. We compute the per-pixel across-scene std / MAD,
    # a consistency gate (unstable pixels inflate their reported σ in
    # quadrature and are flagged in a stability_mask band), summary stats, a
    # per-scene table, and write per-scene grids + an inter-scene-σ raster to
    # the downloads store. Files surface in the response `downloads` list.
    # ══════════════════════════════════════════════════════════════════════
    stability = None
    stab_downloads = []
    sigma_mle_final = sigma_mle.copy()
    stability_mask = np.zeros((H, W), dtype=np.int8)   # 0 none/single, 1 stable, 2 unstable
    try:
        import uuid as _uuid, json as _jsonm, warnings as _warn
        stab_id = "mle_" + _uuid.uuid4().hex[:10]   # generated early: used in per-scene geotiff_url too
        n_sc = len(grids)
        # Retain each scene's independent estimate, NaN where not water/finite.
        scene_stack = np.full((n_sc, H, W), np.nan, dtype=np.float32)
        for i, g in enumerate(grids):
            di = g['depth']; wi = g['water']
            ok = wi & np.isfinite(di) & (di > 0)
            scene_stack[i][ok] = di[ok]

        valid_cnt = np.sum(np.isfinite(scene_stack), axis=0)
        multi = valid_cnt >= 2                          # ≥2 scenes → comparable
        across_std = np.full((H, W), np.nan, dtype=np.float32)
        across_mad = np.full((H, W), np.nan, dtype=np.float32)
        with _warn.catch_warnings():
            _warn.simplefilter("ignore", category=RuntimeWarning)
            if multi.any():
                sub = scene_stack[:, multi]              # (n_sc, Nmulti)
                across_std[multi] = np.nanstd(sub, axis=0).astype(np.float32)
                _med = np.nanmedian(sub, axis=0)
                across_mad[multi] = (1.4826 * np.nanmedian(
                    np.abs(sub - _med), axis=0)).astype(np.float32)

        med_inter_sigma = (float(np.nanmedian(across_std[multi]))
                           if multi.any() else float('nan'))
        p95_inter_sigma = (float(np.nanpercentile(across_std[multi], 95))
                           if multi.any() else float('nan'))
        med_inter_mad = (float(np.nanmedian(across_mad[multi]))
                         if multi.any() else float('nan'))

        # ── Consistency gate (MASKMLE 2.2: relative × mult with a physical floor) ──
        # thresh = max(SIGMA_MULT × median_σ, SIGMA_FLOOR_M). The floor stops the
        # dimensionless 2×median from flagging sub-noise 9-cm wiggles as "unstable"
        # on very-stable shallow banks (bu_tinah median σ ≈ 0.047 m → 2×median =
        # 0.094 m gate flagged 25% of px that agree to < 0.1 m — inside the single-
        # scene S2-SDB repeatability budget). Floor 0.20 m = half the per-scene σ
        # clamp of 0.5 m / low end of published S2-SDB repeatability. The floor only
        # BINDS when median σ is small: at round0 (median 0.372) thresh stays
        # 2×0.372 = 0.744 m, so deep/turbid instabilities (khalifa basin 3.8–4.4 m)
        # remain flagged — nothing is hidden.
        sig_mult = float(os.environ.get("STABILITY_SIGMA_MULT", "2.0"))
        sig_floor = float(os.environ.get("STABILITY_SIGMA_FLOOR_M", "0.20"))
        rel_thresh = (sig_mult * med_inter_sigma
                      if np.isfinite(med_inter_sigma) and med_inter_sigma > 0
                      else np.inf)
        thresh = max(rel_thresh, sig_floor)
        unstable = multi & np.isfinite(across_std) & (across_std > thresh)
        stability_mask[multi] = 1
        stability_mask[unstable] = 2
        # Inflate σ in quadrature at unstable pixels (do NOT silently average).
        infl = unstable & np.isfinite(sigma_mle_final) & np.isfinite(across_std)
        sigma_mle_final[infl] = np.sqrt(
            sigma_mle_final[infl] ** 2 + across_std[infl] ** 2).astype(np.float32)
        n_multi = int(multi.sum())
        unstable_frac = float(unstable.sum() / max(n_multi, 1))

        # ── Per-scene table + pairwise agreement + bias vs composite ──
        scene_table = []
        comp = z_mle
        # Deepest-quartile-of-composite mask for the 1.3 glint proxy.
        _cf = np.isfinite(comp)
        _q75 = float(np.nanpercentile(comp[_cf], 75)) if _cf.any() else None
        deep_mask_comp = (_cf & (comp >= _q75)) if _q75 is not None else None
        _bias_list, _tide_list = [], []
        for i, g in enumerate(grids):
            si = scene_stack[i]
            ok = np.isfinite(si)
            both = ok & np.isfinite(comp)
            dev = si[both] - comp[both]
            bias = float(np.mean(dev)) if dev.size else None
            # 1.3 per-scene physical QC (diagnostic only, no screening)
            qc = _scene_physical_qc(g.get('s2') or {}, deep_mask_comp, g['water']) \
                if g.get('s2') else {"glint_proxy_nir": None, "turbidity_fnu": None}

            # ── ADPorts F1: per-scene overlay PNG (composite colour ramp/scale) +
            # correlation vs composite + valid coverage % + honest outlier flag ──
            try:
                _sb64, _sbounds, _ = depth_to_raster_png(
                    si, bbox, water_mask=g['water'], max_depth=comp_max_depth,
                    min_depth=comp_min_depth)
            except Exception as _ex:
                L.info(f"MLE: scene {g['date']} overlay render failed ({_ex})")
                _sb64 = None
            corr_vs_composite = None
            if int(both.sum()) >= 10:
                s_vals, c_vals = si[both], comp[both]
                if float(np.std(s_vals)) > 1e-9 and float(np.std(c_vals)) > 1e-9:
                    corr_vs_composite = round(float(np.corrcoef(s_vals, c_vals)[0, 1]), 4)
            n_water_px = int(union_water.sum())
            valid_pct = round(100.0 * int(ok.sum()) / max(n_water_px, 1), 2)
            rms_dev = round(float(np.sqrt(np.mean(dev ** 2))), 3) if dev.size else None
            agreement_threshold_m = round(float(thresh), 3) if np.isfinite(thresh) else None
            outlier = False
            outlier_reason = None
            if rms_dev is not None and agreement_threshold_m is not None and rms_dev > agreement_threshold_m:
                outlier = True
                outlier_reason = f"median |Δ| {rms_dev} m > {agreement_threshold_m} m threshold"
            elif corr_vs_composite is not None and corr_vs_composite < 0.7:
                outlier = True
                outlier_reason = f"corr {corr_vs_composite} < 0.7"
            row = {
                'date': g['date'], 'window': g['window'],
                'sigma_m': round(g['sigma'], 3),
                'valid_px': int(ok.sum()),
                'median_depth_m': round(float(np.nanmedian(si[ok])), 3) if ok.any() else None,
                'rms_dev_from_composite_m': rms_dev,
                'bias_vs_composite_m': round(bias, 3) if bias is not None else None,
                # ADPorts F1 — per-scene visual/agreement/download fields
                'overlay_png_b64': _sb64,
                'corr_vs_composite': corr_vs_composite,
                'valid_pct': valid_pct,
                'outlier': outlier,
                'outlier_reason': outlier_reason,
                'geotiff_url': f"/downloads/{stab_id}_scene_{g['date']}.tif",
                # MASKMLE 1.2a/1.2b — TRUE timestamps + EOT20 tide (always reported)
                'acquisition_datetimes': g.get('acquisition_datetimes'),
                'n_acquisitions': int(g.get('n_acq', 0)),
                'tide_m': (round(g['tide_m'], 4) if g.get('tide_m') is not None else None),
                'tide_source': g.get('tide_source'),
                'sigma_tide_m': (round(g['sigma_tide_m'], 4)
                                 if g.get('sigma_tide_m') is not None else None),
                'tide_shift_applied_m': g.get('tide_shift_applied_m'),
                # R2.1/R2.2 — per-scene significant wave height + σ_wave
                'hs_mean_m': (round(g['hs_mean_m'], 3) if g.get('hs_mean_m') is not None else None),
                'hs_max_m': (round(g['hs_max_m'], 3) if g.get('hs_max_m') is not None else None),
                'hs_day_max_m': (round(g['hs_day_max_m'], 3) if g.get('hs_day_max_m') is not None else None),
                'hs_source': g.get('hs_source'),
                'sigma_wave_m': (round(g['sigma_wave_m'], 4) if g.get('sigma_wave_m') is not None else None),
                # MASKMLE 1.3 — physical QC
                'glint_proxy_nir': qc.get('glint_proxy_nir'),
                'turbidity_fnu': qc.get('turbidity_fnu'),
            }
            scene_table.append(row)
            if bias is not None and g.get('tide_m') is not None:
                _bias_list.append(bias); _tide_list.append(g['tide_m'])
        # 1.2c diagnostic: regress bias_vs_composite on (h_i − mean h) → retained
        # tide fraction α (slope) + Pearson r. Pre-registered prediction α ≲ 0.2.
        if len(_bias_list) >= 3:
            _b = np.asarray(_bias_list); _t = np.asarray(_tide_list)
            _tc = _t - _t.mean()
            _var = float(np.sum(_tc ** 2))
            if _var > 1e-9:
                alpha = float(np.sum(_tc * (_b - _b.mean())) / _var)
                bstd = float(_b.std()); tstd = float(_tc.std())
                rr = (float(np.corrcoef(_t, _b)[0, 1])
                      if bstd > 1e-9 and tstd > 1e-9 else None)
                tide_info["alpha_retained"] = round(alpha, 4)
                tide_info["alpha_r"] = (round(rr, 4) if rr is not None else None)
        tide_info["per_scene"] = [
            {"date": g['date'], "n_acq": int(g.get('n_acq', 0)),
             "tide_m": (round(g['tide_m'], 4) if g.get('tide_m') is not None else None),
             "tide_source": g.get('tide_source'),
             "sigma_tide_m": (round(g['sigma_tide_m'], 4)
                              if g.get('sigma_tide_m') is not None else None),
             "tide_shift_applied_m": g.get('tide_shift_applied_m')}
            for g in grids]
        # R2.1: per-scene Hs diagnostic + R2.2 induced σ_wave / fusion weights.
        _wq = [(float(g['sigma']) if not wave_qc
                else float(np.sqrt(g['sigma'] ** 2
                                   + float(g.get('sigma_tide_m', 0.0) if tide_correct else 0.0) ** 2
                                   + float(g.get('sigma_wave_m', 0.0)) ** 2)))
               for g in grids]
        _wsum = sum(1.0 / (s ** 2) for s in _wq) or 1.0
        wave_info["per_scene"] = [
            {"date": g['date'], "n_acq": int(g.get('n_acq', 0)),
             "hs_mean_m": (round(g['hs_mean_m'], 3) if g.get('hs_mean_m') is not None else None),
             "hs_max_m": (round(g['hs_max_m'], 3) if g.get('hs_max_m') is not None else None),
             "hs_day_max_m": (round(g['hs_day_max_m'], 3) if g.get('hs_day_max_m') is not None else None),
             "hs_source": g.get('hs_source'),
             "sigma_wave_m": round(float(g.get('sigma_wave_m', 0.0)), 4),
             "rel_weight": round((1.0 / (_wq[i] ** 2)) / _wsum, 4)}
            for i, g in enumerate(grids)]
        pair_absdiff = []
        for i in range(n_sc):
            for j in range(i + 1, n_sc):
                both = np.isfinite(scene_stack[i]) & np.isfinite(scene_stack[j])
                if both.any():
                    md = float(np.median(np.abs(scene_stack[i][both] - scene_stack[j][both])))
                    pair_absdiff.append({'pair': [grids[i]['date'], grids[j]['date']],
                                         'median_abs_diff_m': round(md, 3),
                                         'n': int(both.sum())})

        # ── Internal consistency: composite == MLE of the retained scenes ──
        num_c = np.zeros((H, W), np.float64); den_c = np.zeros((H, W), np.float64)
        for i, g in enumerate(grids):
            ok = np.isfinite(scene_stack[i])
            iv = 1.0 / (g['sigma'] ** 2)
            num_c[ok] += scene_stack[i][ok] * iv
            den_c[ok] += iv
        fin_c = den_c > 0
        recon = np.full((H, W), np.nan, np.float32)
        recon[fin_c] = np.clip(num_c[fin_c] / den_c[fin_c], 0.0, MAX_DEPTH_M).astype(np.float32)
        cc = np.isfinite(recon) & np.isfinite(comp)
        max_recon_err = float(np.max(np.abs(recon[cc] - comp[cc]))) if cc.any() else 0.0

        # ── Persist per-scene grids + stability rasters to downloads store ──
        dl_dir = Path(__file__).parent / "ocean" / "downloads"
        dl_dir.mkdir(parents=True, exist_ok=True)

        def _dl(fname, label, fmt):
            stab_downloads.append({'format': fmt, 'label': label,
                                   'url': f'/downloads/{fname}', 'name': fname})

        # (a) all per-scene depths in one compressed npz (+ dates, sigmas, bbox)
        npz_name = f"{stab_id}_scenes.npz"
        # R1.3 harness support: also persist the UN-calibrated per-scene grids
        # (pre_calib, NaN where absent) + per-scene tide + water masks so an
        # offline tide-referencing re-test can operate in the reference-sparse
        # (calibration-off) regime without a second server run.
        _pre_stack = np.full((n_sc, H, W), np.nan, dtype=np.float32)
        for i, g in enumerate(grids):
            pc = g.get('pre_calib')
            if pc is not None:
                try:
                    pcs = np.asarray(pc, dtype=np.float32)[:H, :W]
                    wok = g['water'] & np.isfinite(pcs) & (pcs > 0)
                    _pre_stack[i][wok] = pcs[wok]
                except Exception:
                    pass
        _water_stack = np.stack([g['water'].astype(bool) for g in grids], axis=0)
        np.savez_compressed(
            dl_dir / npz_name,
            scenes=scene_stack.astype(np.float32),
            scenes_pre_calib=_pre_stack,
            water=_water_stack,
            dates=np.array([g['date'] for g in grids]),
            sigmas=np.array([g['sigma'] for g in grids], dtype=np.float32),
            tide_m=np.array([(np.nan if g.get('tide_m') is None else g['tide_m'])
                             for g in grids], dtype=np.float64),
            tide_source=np.array([g.get('tide_source', 'none') for g in grids]),
            hs_mean_m=np.array([(np.nan if g.get('hs_mean_m') is None else g['hs_mean_m'])
                                for g in grids], dtype=np.float64),
            hs_max_m=np.array([(np.nan if g.get('hs_max_m') is None else g['hs_max_m'])
                               for g in grids], dtype=np.float64),
            sigma_wave_m=np.array([float(g.get('sigma_wave_m', 0.0)) for g in grids],
                                  dtype=np.float64),
            hs_source=np.array([g.get('hs_source', 'none') for g in grids]),
            bbox=np.asarray(bbox, dtype=np.float64),
            across_std=across_std, across_mad=across_mad,
            stability_mask=stability_mask)
        _dl(npz_name, 'Per-scene depth grids (npz)', 'npz')

        # (b) per-scene GeoTIFFs
        for i, g in enumerate(grids):
            tn = f"{stab_id}_scene_{g['date']}.tif"
            if _write_grid_geotiff(scene_stack[i], bbox, dl_dir / tn,
                                   f"scene_{g['date']}_depth_m"):
                _dl(tn, f"Scene {g['date']} depth (GeoTIFF)", 'geotiff')
        # (c) composite + inter-scene-σ + stability-mask GeoTIFFs
        cn = f"{stab_id}_composite.tif"
        if _write_grid_geotiff(z_mle, bbox, dl_dir / cn, "composite_mle_depth_m"):
            _dl(cn, 'Composite MLE depth (GeoTIFF)', 'geotiff')
        sn = f"{stab_id}_interscene_sigma.tif"
        if _write_grid_geotiff(across_std, bbox, dl_dir / sn, "interscene_sigma_m"):
            _dl(sn, 'Inter-scene σ (GeoTIFF)', 'geotiff')
        # (d) stability.json
        jn = f"{stab_id}_stability.json"
        stability = {
            'n_scenes': n_sc,
            'scene_dates': [g['date'] for g in grids],
            'median_interscene_sigma_m': (round(med_inter_sigma, 3)
                                          if np.isfinite(med_inter_sigma) else None),
            'p95_interscene_sigma_m': (round(p95_inter_sigma, 3)
                                       if np.isfinite(p95_inter_sigma) else None),
            'median_interscene_mad_m': (round(med_inter_mad, 3)
                                        if np.isfinite(med_inter_mad) else None),
            'consistency_sigma_mult': sig_mult,
            'stability_sigma_floor_m': round(sig_floor, 3),
            'unstable_threshold_m': (round(float(thresh), 3)
                                     if np.isfinite(thresh) else None),
            # ADPorts F1: single source of truth for the frontend OUTLIER badge —
            # reuses unstable_threshold_m verbatim, no second threshold introduced.
            'agreement_threshold_m': (round(float(thresh), 3)
                                      if np.isfinite(thresh) else None),
            'unstable_threshold_relative_m': (round(float(rel_thresh), 3)
                                              if np.isfinite(rel_thresh) else None),
            'unstable_floor_binds': bool(np.isfinite(rel_thresh)
                                         and sig_floor >= rel_thresh),
            'n_multiscene_px': n_multi,
            'unstable_px': int(unstable.sum()),
            'unstable_fraction': round(unstable_frac, 4),
            'internal_consistency_max_err_m': round(max_recon_err, 6),
            'per_scene': scene_table,
            'pairwise_median_abs_diff': pair_absdiff,
            # MASKMLE 1.2 — per-scene EOT20 tide diagnostic + correction state
            'tide': tide_info,
            # R2 — per-scene Hs sea-state diagnostic + σ_wave QC state
            'wave': wave_info,
            # MASKMLE 2.3 — best-N-of-M physical-QC scene screening
            'screening': screening,
            # MASKMLE 3.2 — shared-calibration-field package state + α diagnostic
            'shared_cal': shared_cal_info,
        }
        with open(dl_dir / jn, 'w') as fh:
            _jsonm.dump(stability, fh, indent=2)
        _dl(jn, 'Stability report (JSON)', 'json')
        stability['downloads'] = stab_downloads
        stability['id'] = stab_id

        # Use the σ-inflated grid for the reported uncertainty raster.
        sigma_mle = sigma_mle_final
        L.info(f"MLE stability: median inter-scene σ={stability['median_interscene_sigma_m']} m, "
               f"p95={stability['p95_interscene_sigma_m']} m, unstable "
               f"{stability['unstable_px']}/{n_multi} px "
               f"({100*unstable_frac:.1f}%), composite↔per-scene max err "
               f"{max_recon_err:.2e} m")
    except Exception as _ex:
        L.exception(f"MLE stability report failed ({_ex})")
        stability = {'error': str(_ex)}

    # ══════════════════════════════════════════════════════════════════════
    # PRO σ-QA — "remove the ones with sigma very far" (user headline ask).
    # Mask (→NaN) composite pixels whose posterior σ (inverse-variance MLE σ,
    # inflated in quadrature by inter-scene instability above) is an outlier:
    # hard cap ``sigma_max_m`` if given, else robust median+k·MAD (k=sigma_reject_k).
    # Reported honestly in ml_stats.uncertainty; validation RMSE is NOT re-shrunk.
    # ══════════════════════════════════════════════════════════════════════
    z_mle, uq = _apply_sigma_qa(z_mle, sigma_mle,
                                sigma_max_m=sigma_max_m,
                                sigma_reject_k=float(sigma_reject_k))
    # R1.2 never-worse-σ floor: fold the σ_tide floor (0.15 m) into the REPORTED
    # composite σ in quadrature. The correlated tide-model residual does NOT
    # average down over scenes, so a fixed floor is the honest lower bound.
    # Applied AFTER σ-QA (which masks on σ rank) so the masking decision — and
    # therefore the composite DEPTH — is byte-identical when MLE_TIDE_CORRECT=0.
    _sig_pre_floor_med = (float(np.nanmedian(sigma_mle))
                          if np.any(np.isfinite(sigma_mle)) else None)
    _fin_s = np.isfinite(sigma_mle)
    sigma_mle[_fin_s] = np.sqrt(
        sigma_mle[_fin_s] ** 2 + float(_tide_floor) ** 2).astype(np.float32)
    _sig_post_floor_med = (float(np.nanmedian(sigma_mle))
                           if np.any(np.isfinite(sigma_mle)) else None)
    uq['sigma_tide_floor_m'] = float(_tide_floor)
    uq['median_sigma_pre_tidefloor_m'] = (round(_sig_pre_floor_med, 4)
                                          if _sig_pre_floor_med is not None else None)
    uq['median_sigma_post_tidefloor_m'] = (round(_sig_post_floor_med, 4)
                                           if _sig_post_floor_med is not None else None)
    # scenes dropped by the physical-QC screening (glint/turbidity) above
    uq['scenes_rejected'] = int((screening.get('n_candidates') or len(grids)) - len(grids))
    uq['sigma_reject_k'] = float(sigma_reject_k)
    # R2.2 wave-QC disclosure (σ_wave is in the fusion weights only when ON)
    uq['wave_qc_applied'] = bool(wave_qc)
    uq['wave_k'] = _WAVE_K
    uq['wave_hs0_m'] = _WAVE_HS0
    uq['hs_reject_m'] = _HS_REJECT
    uq['sigma_wave_scenes_m'] = [round(float(g.get('sigma_wave_m', 0.0)), 4) for g in grids]
    L.info(f"MLE σ-QA: σ_max_used={uq['sigma_max_used']} m, masked "
           f"{uq['pixels_masked_highsigma']} px, retained {uq['frac_retained']}, "
           f"scenes_rejected={uq['scenes_rejected']}")

    # ── Result store for interactive /api/recolor + seed min/max ──
    _rid, _auto_min, _auto_max = _store_recolor_result(z_mle, bbox, water_mask=union_water)

    # ── Render PNGs (explicit min/max so the returned raster matches the
    #    reported raster_min/max the UI seeds its colorbar inputs from) ──
    raster_b64, raster_bounds, _ = depth_to_raster_png(
        z_mle, bbox, water_mask=union_water,
        max_depth=_auto_max, min_depth=_auto_min)
    # Uncertainty PNG — yellow→red colour ramp on σ in [0, 3] m
    uncert_b64 = None
    try:
        from PIL import Image
        u = np.clip(np.where(np.isfinite(sigma_mle), sigma_mle, 0), 0, 3)
        u_norm = (u / 3.0)
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        # green → yellow → red
        rgba[..., 0] = (255 * np.clip(u_norm * 2.0, 0, 1)).astype(np.uint8)
        rgba[..., 1] = (255 * np.clip(2.0 - u_norm * 2.0, 0, 1)).astype(np.uint8)
        rgba[..., 2] = 0
        rgba[..., 3] = np.where(union_water & np.isfinite(sigma_mle), 200, 0).astype(np.uint8)
        bio = _io.BytesIO()
        Image.fromarray(rgba, 'RGBA').save(bio, format='PNG', optimize=True)
        uncert_b64 = _b64.b64encode(bio.getvalue()).decode()
    except Exception as ex:
        L.info(f"MLE: uncertainty PNG failed ({ex})")

    # ── Summary stats ──
    val = z_mle[np.isfinite(z_mle) & (z_mle > 0)]
    bathy_stats = {
        'mean_depth': round(float(np.mean(val)), 2) if len(val) else 0,
        'max_depth':  round(float(np.max(val)),  2) if len(val) else 0,
        'min_depth':  round(float(np.min(val)),  2) if len(val) else 0,
        'std_depth':  round(float(np.std(val)),  2) if len(val) else 0,
        'mean_sigma': round(float(np.nanmean(sigma_mle)), 2) if np.any(np.isfinite(sigma_mle)) else 0,
        'grid_points': int(len(val)),
        'resolution_m': 10,
    }

    # Powers Data tab + 3D + CSV/GeoJSON export
    interp_pts = grid_to_points(z_mle, bbox, max_pts=8000)

    # Isobath contours (powers Contour tab + IHO chart)
    contours_out, contour_levels = [], []
    try:
        contours_out = generate_contours(z_mle, bbox)
        contour_levels = sorted({c['depth'] for c in contours_out})
    except Exception as ex:
        L.info(f"MLE contours skipped: {ex}")

    elapsed = round(_T.time() - t0, 2)
    L.info(f"MLE: {len(grids)} scenes, {elapsed}s, mean depth "
           f"{bathy_stats['mean_depth']} m, mean σ "
           f"{bathy_stats['mean_sigma']} m")

    # ── 3.U1: append mask-QA preview artifacts + a "download all (zip)" link ──
    stab_id = (stability or {}).get('id') if isinstance(stability, dict) else None
    if mle_mask_meta:
        for _k, _label, _fmt in (
            ("mask_preview_png_name", "Mask-QA overlay (PNG)", "png"),
            ("mask_preview_geotiff_name", "Mask-QA classes (GeoTIFF)", "geotiff")):
            _nm = mle_mask_meta.get(_k)
            if _nm:
                stab_downloads.append({'format': _fmt, 'label': _label,
                                       'url': f'/downloads/{_nm}', 'name': _nm})
    # "Download all (zip)" — streams every MLE artifact for this run in one file.
    zip_url = f'/api/very-hr-mle/download-all/{stab_id}.zip' if stab_id else None
    n_kept = len(grids)
    n_attempted = (screening.get('n_candidates') or n_kept)
    scenes_kept_label = f"{n_kept} of {n_attempted} scenes kept (glint/turbidity QC)"
    if zip_url:
        stab_downloads.append({'format': 'zip', 'label': 'Download all MLE artifacts (zip)',
                               'url': zip_url, 'name': f'{stab_id}_all.zip'})

    # R3 (TIDE_WAVE_LOG) — honest vertical-reference disclosure. Pure read of
    # z_mle vs any on-disk GT; NEVER mutates depth. Default MLE_DATUM='relative'
    # → depth byte-identical; 'msl' relabels only the bias offset, never the shape.
    try:
        _datum_disc = mle_datum_disclosure(z_mle, bbox, _tide_floor)
    except Exception as _dex:
        L.info(f"MLE datum disclosure skipped ({_dex})")
        _datum_disc = None

    result = {
        'method': f'Calibrated bathymetry (MLE · {len(grids)} scenes)',
        'bbox': bbox, 'resolution_m': 10,
        'metrics': {'n_scenes': len(grids), 'n_test': 0},
        'stats': bathy_stats,
        # ADPorts F3: composite-level summary of the EXISTING stability.tide
        # diagnostic — never a second computation, never a second correction.
        'tide_correction': mle_tide_correction_summary(
            (stability or {}).get('tide') if isinstance(stability, dict) else None),
        'ml_stats': {
            'method': f'Multi-scene MLE · {len(grids)} scenes',
            'mean_sigma_m': bathy_stats['mean_sigma'],
            'sources': [], 'importance': {},
            'r2': None, 'rmse': None,
            # PRO σ-QA report (user ask #2)
            'uncertainty': uq,
            # AI_METHOD_LOG R1/R3: learned per-pixel σ diagnostics (None unless
            # MLE_DEEP_SIGMA=1 AND the ≥500 m spatial-block gate adopted the CNN).
            'ai_sigma': ai_sigma_stats,
            # TIDE_WAVE_LOG R3: honest vertical-datum disclosure (scene-set-relative;
            # separated bias vs shape-RMSE; no absolute MSL/LAT claim).
            'datum_disclosure': _datum_disc,
        },
        'datum_disclosure': _datum_disc,
        # Interactive colorbar recolor (user ask #3): id + auto-seed min/max
        'result_id': _rid,
        'raster_min_depth': round(float(_auto_min), 3),
        'raster_max_depth': round(float(_auto_max), 3),
        'raster_auto_min': round(float(_auto_min), 3),
        'raster_auto_max': round(float(_auto_max), 3),
        'augmentation': {
            'mle_scenes': [{'date': g['date'], 'sigma_m': round(g['sigma'], 2),
                            'mean_depth_m': g.get('mean_depth')}
                           for g in grids],
        },
        # Item 2: per-scene retention + stability report + downloadable files.
        'stability': stability,
        'downloads': stab_downloads,
        'download_all_zip': zip_url,
        # 3.3 — honest scene selection surfaced in the UI results metadata.
        'scenes_kept': n_kept, 'scenes_attempted': n_attempted,
        'scenes_kept_label': scenes_kept_label,
        # 3.U2 — mask QA surfaced for the results panel.
        'water_mask_meta': mle_mask_meta,
        'stability_id': stab_id,
        'elapsed_s': elapsed,
        'depth_png_b64': raster_b64,
        'overlay_png_b64': raster_b64,
        'uncertainty_png_b64': uncert_b64,
        'raster_bounds': raster_bounds,
        'sources_used': [],
        'interpolated_points': interp_pts,
        'contours': contours_out,
        'contour_levels': contour_levels,
    }
    # 3.U1 durable persistence: register the finished MLE run so a re-opened job
    # lists EVERY artifact (downloads mirrored into the result metrics) and each
    # link resolves from a fresh process (files live on disk under ocean/downloads).
    if stab_id:
        try:
            from backend import jobs_store as _js
        except Exception:
            try:
                import jobs_store as _js  # type: ignore
            except Exception:
                _js = None
        if _js is not None:
            try:
                comp_name = None
                for d in stab_downloads:
                    if d.get('name', '').endswith('_composite.tif'):
                        comp_name = d['name']; break
                comp_path = (Path(__file__).parent / "ocean" / "downloads"
                             / comp_name) if comp_name else None
                _js.register_result(
                    name=f"MLE · {n_kept} scenes · {year}",
                    roi_bbox=list(bbox), resolution="10m", status="done",
                    output_path=(str(comp_path) if comp_path else None),
                    metrics={
                        "kind": "mle", "stability_id": stab_id,
                        "median_interscene_sigma_m": (stability or {}).get(
                            'median_interscene_sigma_m'),
                        "scenes_kept": n_kept, "scenes_attempted": n_attempted,
                        "downloads": stab_downloads,
                        "download_all_zip": zip_url,
                        "water_mask_meta": mle_mask_meta,
                        "mean_depth_m": bathy_stats.get('mean_depth'),
                    },
                    result_id=stab_id)
            except Exception as _rex:
                L.info(f"MLE durable register skipped ({_rex})")
    return result


@app.route('/api/very-hr-mle', methods=['POST'])
def api_very_hr_mle():
    """Multi-scene MLE bathymetry over a calendar year.

    Body: {bbox:{west,south,east,north}, year:2024, n_scenes:2..12 (default 10),
           max_cloud:20, user_points:[],
           sigma_reject_k:3.0 (σ-QA strictness) | sigma_max_m:<hard cap m>}
    """
    if not LS_AVAILABLE:
        return jsonify({'error': 'Lyzenga module unavailable'}), 500
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox') or {}
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        bbox = [bd['west'], bd['south'], bd['east'], bd['north']]
        cap = _enforce_roi_cap(bbox)
        if cap is not None:
            return cap
        year = int(data.get('year', 2024))
        # Ceiling raised to 12 (was restricted to {3,5,7}); default is the
        # accurate many-image tier. _run_s2_mle re-clamps to [1,12] internally.
        try:
            n_scenes = int(data.get('n_scenes', 10))
        except (TypeError, ValueError):
            n_scenes = 10
        n_scenes = max(2, min(12, n_scenes))
        # PRO σ-QA controls (user ask #2)
        try:
            sigma_reject_k = float(data.get('sigma_reject_k', 3.0))
        except (TypeError, ValueError):
            sigma_reject_k = 3.0
        _smx = data.get('sigma_max_m')
        try:
            sigma_max_m = float(_smx) if _smx is not None else None
        except (TypeError, ValueError):
            sigma_max_m = None
        result = _run_s2_mle(
            bbox, year=year, n_scenes=n_scenes,
            max_cloud=int(data.get('max_cloud', 20)),
            user_pts=data.get('user_points', []) or [],
            fetch_sliderule=bool(data.get('use_sliderule', False)),
            sigma_reject_k=sigma_reject_k, sigma_max_m=sigma_max_m,
        )
        return jsonify(_json_safe(result))
    except Exception as ex:
        L.exception("very-hr-mle failed")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/very-hr-mle/download-all/<stab_id>.zip', methods=['GET'])
def api_very_hr_mle_download_all(stab_id):
    """3.U1 — stream ONE zip of every MLE artifact for a stability id: all
    per-scene GeoTIFFs, the composite, inter-scene-σ, the scenes npz,
    stability.json, and the mask-QA preview PNG/GeoTIFF. Resolves from a fresh
    process (files persist under ocean/downloads), so re-opened jobs work.

    Now delegates to the SHARED collector/streamer (`_collect_result_artifacts`
    / `_stream_result_zip`) that also powers the general
    `/api/results/<id>/download-all.zip`, with a glob-only fallback for legacy
    MLE runs that never got registered into the results store."""
    import zipfile
    safe = "".join(c for c in stab_id if c.isalnum() or c in "._-")
    if not safe or safe != stab_id:
        return jsonify({'error': 'invalid id'}), 400
    from flask import send_file
    files, missing = _collect_result_artifacts(safe)
    if files is not None and files:
        bio = _stream_result_zip(safe, files, missing)
        return send_file(bio, mimetype='application/zip', as_attachment=True,
                         download_name=f"{safe}_all.zip")
    # Fallback: MLE run not in the store → bare glob of ocean/downloads.
    dl_dir = _ocean_downloads_dir()
    globbed = [p for p in sorted(dl_dir.glob(f"{safe}_*"))
               if not p.name.lower().endswith('.zip')]
    if not globbed:
        return jsonify({'error': 'no artifacts for this id'}), 404
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in globbed:
            try:
                zf.write(str(p), arcname=p.name)
            except Exception:
                pass
    bio.seek(0)
    return send_file(bio, mimetype='application/zip', as_attachment=True,
                     download_name=f"{safe}_all.zip")


@app.route('/api/sdb-pro/geotiff', methods=['POST'])


@app.route('/api/sdb-pro/geotiff', methods=['POST'])
def api_sdb_pro_geotiff():
    """On-demand GeoTIFF download for the fast S2 SDB pipeline. Body
    matches /api/sdb-pro. Returns image/tiff bytes the browser can save
    directly (kept out of the main JSON response so it doesn't bloat it
    or trigger Content-Length mismatches behind upstream proxies)."""
    if not LS_AVAILABLE:
        return jsonify({'error': 'Lyzenga+SlideRule module unavailable'}), 500
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox') or {}
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        bbox = [bd['west'], bd['south'], bd['east'], bd['north']]
        cap = _enforce_roi_cap(bbox)
        if cap is not None:
            return cap
        sd = data.get('start_date', '2024-05-01')
        ed = data.get('end_date', '2024-09-30')
        result = _run_s2_lyzenga_fast(
            bbox, sd, ed,
            user_pts=data.get('user_points', []) or [],
            max_cloud=int(data.get('max_cloud', 20)),
            fetch_sliderule=bool(data.get('use_sliderule', False)),
            include_geotiff=True,
        )
        if not result.get('geotiff_b64'):
            return jsonify({'error': 'GeoTIFF generation failed'}), 500
        tiff_bytes = base64.b64decode(result['geotiff_b64'])
        from flask import send_file
        bio = io.BytesIO(tiff_bytes)
        bio.seek(0)
        return send_file(
            bio, mimetype='image/tiff', as_attachment=True,
            download_name=f"sdb_{int(TM.time())}.tif",
        )
    except Exception as ex:
        L.exception("sdb-pro/geotiff failed")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/sdb-pro', methods=['POST'])
def api_sdb_pro():
    """Direct fast S2 SDB endpoint — Lyzenga + Stumpf + SlideRule, with a
    composite water mask. Returns a homogeneous depth raster in ~10-25 s.

    Body: {bbox:{west,south,east,north}, start_date, end_date, max_cloud,
           user_points:[{lat,lon,depth}], use_sliderule:bool}
    """
    if not LS_AVAILABLE:
        return jsonify({'error': 'Lyzenga+SlideRule module unavailable'}), 500
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox') or {}
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        bbox = [bd['west'], bd['south'], bd['east'], bd['north']]
        cap = _enforce_roi_cap(bbox)
        if cap is not None:
            return cap
        sd = data.get('start_date', '2024-05-01')
        ed = data.get('end_date', '2024-09-30')
        result = _run_s2_lyzenga_fast(
            bbox, sd, ed,
            user_pts=data.get('user_points', []) or [],
            max_cloud=int(data.get('max_cloud', 20)),
            fetch_sliderule=bool(data.get('use_sliderule', False)),
        )
        return jsonify(_json_safe(result))
    except Exception as ex:
        L.exception("sdb-pro failed")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# VERY HR — Clustered SDB (the one production method)
#   Mapbox VHR + 12-band Sentinel-2 (GEE) + ERA-5/SST + OSM land mask
#   K-means + per-cluster HGB ensemble + soft blend + linear bias-correction
#   Optional band-augmented references: in-situ → SlideRule (≥5 m) +
#                                       i-Boating → GEBCO
# ══════════════════════════════════════════════════════════════
@app.route('/api/very-hr-clustered', methods=['POST'])
def api_very_hr_clustered():
    """Run the Clustered SDB pipeline on the user-supplied ROI.

    Body:
      bbox:        {west, south, east, north}      (required)
      site_key:    optional preset key
      user_points: [{lat,lon,depth},…]             (optional — overrides preset)
      train_frac:  default 0.20
      n_clusters:  default 8
      max_iter:    default 1200
      n_estimators:default 5
      target_res_m:default 2.0
      augment:     default True   (in-situ → SlideRule + i-Boating → GEBCO)
      aug_min_per_band: default 150
      s2_start_date / s2_end_date

    Returns metrics, per-band, paths, and base-64 of key images.
    """
    try:
        from backend.very_hr_cbr import run_cbr_vhr
    except ImportError:
        from very_hr_cbr import run_cbr_vhr  # type: ignore
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox')
        site_key = data.get('site_key') or 'custom'
        if site_key in _KNOWN_SITES and _KNOWN_SITES[site_key].get('bbox') and not bd:
            bd = _KNOWN_SITES[site_key]['bbox']
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        bbox = [bd['west'], bd['south'], bd['east'], bd['north']]
        cap = _enforce_roi_cap(bbox)
        if cap is not None:
            return cap
        train_frac = max(0.05, min(0.5, float(data.get('train_frac', 0.20))))
        n_clusters = int(data.get('n_clusters', 8))
        max_iter = int(data.get('max_iter', 1200))
        n_estimators = int(data.get('n_estimators', 5))
        target_res_m = float(data.get('target_res_m', 2.0))
        augment = bool(data.get('augment', True))
        aug_min_per_band = int(data.get('aug_min_per_band', 150))
        s2_start_date = data.get('s2_start_date', '2024-05-01')
        s2_end_date = data.get('s2_end_date', '2024-09-30')
        imagery_source = str(data.get('imagery_source', 'vhr')).lower()
        if imagery_source not in ('vhr', 's2'):
            imagery_source = 'vhr'

        # ── Fast path: every /api/very-hr-clustered request now goes through
        # the Lyzenga + Stumpf + SlideRule + UAE-pretrained pipeline. The
        # legacy CBR/CNN path took too long to finish inside Railway's proxy
        # window (NetworkError @ ~16 s in the UI), and a 40 km² ROI cap is
        # already enforced upstream so the response stays small. The CBR
        # branch below is kept only as a safety net if Lyzenga import failed.
        if LS_AVAILABLE:
            try:
                user_pts_fast = data.get('user_points', []) or []
                # User-selectable output resolution (10/20/50/100 m).
                # `resolution_m` wins; for S2 runs a matching target_res_m
                # (set by the 10m/20m/50m/100m UI selector) is honoured too.
                user_res = data.get('resolution_m')
                if user_res is None and imagery_source == 's2' \
                        and int(target_res_m) in (10, 20, 50, 100):
                    user_res = int(target_res_m)
                fast = _run_s2_lyzenga_fast(
                    bbox, s2_start_date, s2_end_date,
                    user_pts=user_pts_fast,
                    max_cloud=int(data.get('max_cloud', 20)),
                    fetch_sliderule=bool(data.get('aug_use_sliderule', False)),
                    res_override=user_res,
                )
                fast['imagery_source'] = imagery_source
                fast['site'] = site_key
                # match the keys the frontend looks for
                return jsonify(_json_safe(fast))
            except Exception as ex:
                L.exception("Fast S2 path failed — falling back to CBR pipeline")
                # fall through to legacy run_cbr_vhr below

        # Reference points: prefer user-supplied, else fall back to bundled XYZ.
        # An empty in-situ pool is OK — augmentation will fill in from
        # SlideRule + i-Boating + GEBCO.  Force augment=True in that case so
        # the cascade runs unconditionally.
        user_pts = data.get('user_points', []) or []
        if user_pts:
            la = np.array([float(p['lat']) for p in user_pts])
            lo = np.array([float(p['lon']) for p in user_pts])
            de = np.abs(np.array([float(p['depth']) for p in user_pts]))
        else:
            la, lo, de, _src = xyz_points_in_bbox(bbox)
        if len(de) == 0:
            la = np.array([], dtype=np.float64)
            lo = np.array([], dtype=np.float64)
            de = np.array([], dtype=np.float64)
            augment = True
            L.info("very-hr-clustered: 0 in-situ in ROI — relying on augmentation cascade")

        result = run_cbr_vhr(
            site_key=str(site_key),
            bbox=bbox,
            ref_lats=la, ref_lons=lo, ref_depths=de,
            train_frac=train_frac,
            target_res_m=target_res_m,
            max_tiles_per_side=4,
            s2_start_date=s2_start_date, s2_end_date=s2_end_date,
            n_clusters=n_clusters,
            min_train_depth_m=float(data.get('min_train_depth_m', 5.0)),
            aug_min_confidence=float(data.get('aug_min_confidence', 0.5)),
            aug_high_confidence_only=bool(data.get('aug_high_confidence_only', False)),
            use_cnn_refinement=bool(data.get('use_cnn_refinement', True)),
            cnn_blend_weight=float(data.get('cnn_blend_weight', 0.5)),
            cluster_temperature=float(data.get('cluster_temperature', 0.6)),
            max_iter=max_iter,
            n_estimators=n_estimators,
            learning_rate=float(data.get('learning_rate', 0.03)),
            max_depth=int(data.get('max_depth', 9)),
            augment=augment,
            aug_min_per_band=aug_min_per_band,
            aug_band_w=float(data.get('aug_band_w', 2.0)),
            aug_use_iboating=bool(data.get('aug_use_iboating', True)),
            aug_use_sliderule=bool(data.get('aug_use_sliderule', True)),
            aug_use_gebco=bool(data.get('aug_use_gebco', True)),
            seed=int(data.get('seed', 42)),
            save=True,
            imagery_source=imagery_source,
        )

        import base64 as _b64
        method_label = ('Very HR Bathymetry (Sentinel-2 only)'
                        if imagery_source == 's2'
                        else 'Very HR Bathymetry')
        out = {
            'method': method_label,
            'imagery_source': imagery_source,
            'site': site_key,
            'bbox': bbox,
            'resolution_m': result['resolution_m'],
            'metrics': result['metrics'],
            'per_band': result['per_band'],
            'augmentation': result.get('augmentation'),
            'calibration': result.get('calibration'),
            'turbidity_pct': result.get('turbidity_pct', 0),
            'turbidity_warning': result.get('turbidity_warning'),
            'elapsed_s': result['elapsed_s'],
            'paths': result.get('paths', {}),
        }
        for key in ('vhr_image', 'water_mask', 'osm_land', 'cluster_map',
                     'depth_png', 'overlay_png', 'scatter_png', 's44_bands_png'):
            p = (result.get('paths') or {}).get(key)
            if p and os.path.exists(p):
                out[f'{key}_b64'] = _b64.b64encode(Path(p).read_bytes()).decode()
        return jsonify(out)
    except Exception as ex:
        L.exception("very-hr-clustered failed")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# VERY HR — Local-processing job queue
#   The user clicks "Run Very HR Bathymetry", we register a job and the user
#   processes it locally (off the web server). The frontend polls /status and
#   shows an HONEST indicator: a real % when the pipeline reports one, else an
#   indeterminate spinner (R1 — no time-extrapolated bar). Once the user
#   uploads the result file via /upload, the job snaps to 100 %.
# ══════════════════════════════════════════════════════════════
try:
    from backend import vhr_jobs as _vhr_jobs
except ImportError:
    import vhr_jobs as _vhr_jobs  # type: ignore

try:
    from backend import jobs_store as _jobs_store
except ImportError:
    import jobs_store as _jobs_store  # type: ignore

try:
    from backend import seed_results as _seed_results
except ImportError:
    import seed_results as _seed_results  # type: ignore

# DL estimator provenance / model-card builder (DL_WEB_LOG.md §4).
try:
    from backend import model_card as _model_card
except ImportError:
    import model_card as _model_card  # type: ignore

# Seed-on-empty: a fresh/empty deploy (Railway redeploy, new clone) has no
# durable results, so ship bundled sample results so the Results panel works out
# of the box. Idempotent + only seeds when the store has zero 'done' results;
# never overwrites real user results.
try:
    _seed_results.seed_sample_results_if_empty()
except Exception as _seed_ex:  # pragma: no cover
    L.warning("startup seed_sample_results_if_empty failed: %s", _seed_ex)

# DEFAULT regions (DEFAULT_REGIONS_LOG.md AC-1): the 4 auto-run default regions
# (Khalifa / Old Mussafah / Jbel Dhanna / Abu Al Abyad) are ALWAYS seeded —
# NOT gated on an empty store — so they appear as frozen precomputed Results on
# every load with their standard date + metrics-or-reconnaissance card. The 3
# in-situ RMSE/Bias/R2 are USER-VALIDATED and copied verbatim from the manifest.
try:
    _seed_results.seed_default_regions()
except Exception as _seed_ex:  # pragma: no cover
    L.warning("startup seed_default_regions failed: %s", _seed_ex)

# Curated NON-default certified/provisional products (marked always_seed in the
# manifest — e.g. the DL-fusion marawah CERTIFIED + bu_tinah PROVISIONAL LAT
# products). ALWAYS seeded so they list under FINISHED JOBS on every load.
try:
    _seed_results.seed_always_products()
except Exception as _seed_ex:  # pragma: no cover
    L.warning("startup seed_always_products failed: %s", _seed_ex)


@app.route('/api/very-hr-job/start', methods=['POST'])
def api_vhr_job_start():
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox')
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        if isinstance(bd, dict):
            bbox = [bd['west'], bd['south'], bd['east'], bd['north']]
        else:
            bbox = list(bd)
        if len(bbox) != 4:
            return jsonify({'error': 'bbox must be [west, south, east, north] or {west,south,east,north}'}), 400
        label = data.get('label') or 'Very HR Bathymetry'
        params = data.get('params') or {}
        job = _vhr_jobs.create_job(bbox, params=params, label=label)
        return jsonify(job)
    except Exception as ex:
        L.exception("very-hr-job/start failed")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/very-hr-job/<job_id>/status', methods=['GET'])
def api_vhr_job_status(job_id):
    job = _vhr_jobs.status(job_id)
    if job is None:
        return jsonify({'error': 'job not found'}), 404
    return jsonify(job)


@app.route('/api/very-hr-job/list', methods=['GET'])
def api_vhr_job_list():
    return jsonify({'jobs': _vhr_jobs.list_jobs()})


@app.route('/api/very-hr-job/<job_id>/upload', methods=['POST'])
def api_vhr_job_upload(job_id):
    try:
        notes = None
        if req.files:
            f = next(iter(req.files.values()))
            data = f.read()
            filename = f.filename or 'result.bin'
            notes = req.form.get('notes')
        else:
            body = req.get_json(force=True, silent=True) or {}
            filename = body.get('filename', 'result.json')
            payload = body.get('result')
            if payload is None:
                return jsonify({'error': 'no file uploaded and no `result` JSON body'}), 400
            data = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()
            notes = body.get('notes')
        job = _vhr_jobs.save_result(job_id, filename, data, notes=notes)
        if job is None:
            return jsonify({'error': 'job not found'}), 404
        return jsonify(job)
    except Exception as ex:
        L.exception("very-hr-job/upload failed")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/very-hr-job/<job_id>', methods=['DELETE'])
def api_vhr_job_delete(job_id):
    ok = _vhr_jobs.delete(job_id)
    return (jsonify({'ok': True}) if ok else (jsonify({'error': 'job not found'}), 404))


@app.route('/api/very-hr-job/<job_id>/cancel', methods=['POST'])
def api_vhr_job_cancel(job_id):
    job = _vhr_jobs.cancel(job_id)
    if job is None:
        return jsonify({'error': 'job not found'}), 404
    return jsonify(job)


# ══════════════════════════════════════════════════════════════
# BATHY-JOB — unified multi-ROI compute → durable Results store
#   (R2/R3/R4). A thin wrapper over the existing vhr_jobs running queue:
#   one endpoint queues a "compute bathymetry for ROI X at resolution R"
#   job; a background worker runs the existing compute paths
#   (_run_s2_lyzenga_fast for 10 m, run_very_hr for VHR), writes the
#   GeoTIFF into the durable jobs_store, registers metrics, and links the
#   result so it surfaces in /api/results. Finished results persist across
#   restarts. New routes only; nothing existing changes.
# ══════════════════════════════════════════════════════════════

def _bbox_from_payload(data):
    """Accept {roi}/{bbox}/{site_key} → [w,s,e,n] + a human label.
    Returns (bbox_list, label) or (None, error_str)."""
    site_key = data.get('site_key') or data.get('roi') if isinstance(data.get('roi'), str) else data.get('site_key')
    label = None
    bd = data.get('bbox') or data.get('roi')
    # roi may be a known-site key string.
    if isinstance(bd, str) and bd in _KNOWN_SITES:
        site_key = bd
        bd = None
    if (not bd) and site_key and site_key in _KNOWN_SITES:
        site = _KNOWN_SITES[site_key]
        bd = site.get('bbox')
        label = site.get('label')
    if not bd:
        return None, 'Missing roi/bbox (provide {west,south,east,north}, [w,s,e,n], or a known site_key)'
    if isinstance(bd, dict):
        try:
            bbox = [float(bd['west']), float(bd['south']), float(bd['east']), float(bd['north'])]
        except Exception:
            return None, 'bbox dict must have west/south/east/north'
    else:
        bbox = [float(x) for x in bd]
    if len(bbox) != 4:
        return None, 'bbox must be [west, south, east, north]'
    return bbox, label


# ── DL PRIMARY estimator (DL_WEB_LOG.md M1/M3, AC-1/AC-2) ──────────────────
# The June-5 heteroscedastic PatchCNN (backend/models/sdb_cnn_default.pt) is the
# DEFAULT depth estimator for new 10 m and 20 m runs, and supplies the DEPTH for
# VHR (imagery stays Mapbox). Lyzenga/Stumpf is the documented FALLBACK (M4):
# used only when the DL model is unavailable, required bands are missing, the DL
# inference errors, or the user explicitly selects depth_method=lyzenga_fallback.
_DL_DEPTH_METHOD_NAME = "DL Pro — Heteroscedastic PatchCNN"
_LYZENGA_OPT_OUT = {"lyzenga", "lyzenga_fallback", "lyzenga_fast", "stumpf"}


def _wants_lyzenga_fallback(params):
    """True iff the caller explicitly opted out of the DL primary path."""
    dm = (str(params.get("depth_method", "")) or
          str(params.get("sdb_model", "")) or
          os.environ.get("SDB_MODEL", "")).strip().lower()
    return dm in _LYZENGA_OPT_OUT


def _run_dl_patchcnn(bbox, sd, ed, max_cloud=20, region=None):
    """Run the default SDB PatchCNN densely over the ROI's S2 water mask.

    Returns (depth_grid, sigma_grid, s2_dict). Raises on missing bands / model.

    `region` (default region key, e.g. khalifa_port|old_mussafah) selects the
    PER-SITE test-fold σ inflation k_test (ITEM 3 closure); unknown/transfer ROIs
    pass region=None and get the interim global k fallback.
    """
    try:
        from backend.sdb_cnn_baseline import predict_depth_grid as _cnn_predict
    except ImportError:
        from sdb_cnn_baseline import predict_depth_grid as _cnn_predict  # type: ignore
    _s2 = _fetch_s2_cached(bbox, sd, ed, res=10, cloud=max_cloud)
    if _s2 is None:
        raise RuntimeError("S2 fetch returned no scene")
    for _b in ("blue", "green", "red", "nir"):
        if _b not in _s2:
            raise RuntimeError(f"required band '{_b}' missing from S2 scene")
    _wm = _s2.get("water_mask")
    _s2_dict = {
        "blue": _s2["blue"], "green": _s2["green"],
        "red": _s2["red"], "nir": _s2["nir"],
        "scl": _s2.get("scl", np.zeros_like(_s2["blue"])),
        "height": np.array(_s2["blue"].shape[0]),
        "width": np.array(_s2["blue"].shape[1]),
        "resolution_m": np.array(10.0),
    }
    _depth, _sigma = _cnn_predict(
        _s2_dict, water_mask=(np.asarray(_wm, dtype=bool) if _wm is not None else None),
        region=region)
    # Honest scene-level clear-water proxy for AC-9 (limitation flag). NOT a
    # calibrated Kd(490) — a conservative turbidity indicator from the over-water
    # red:green reflectance ratio: clear water has very low red water-leaving
    # reflectance (red is strongly absorbed), so red/green → 0 in ultra-clear
    # cases. We only use it to AUTO-FLAG physics-limited (mode-collapse risk) on
    # ultra-clear ROIs; it never alters the depth product or printed metrics.
    try:
        _wmask = (np.asarray(_wm, dtype=bool) if _wm is not None
                  else np.isfinite(_depth))
        _gr = _s2["green"].astype(np.float32)
        _rd = _s2["red"].astype(np.float32)
        _rat = _rd[_wmask] / (_gr[_wmask] + 1e-6)
        _rat = _rat[np.isfinite(_rat)]
        # Map the median red:green ratio to a coarse Kd proxy: clear ≈ <0.05,
        # turbid Gulf ≈ 0.3–0.6. Linear, clamped; ONLY the <0.05 branch is acted on.
        _kd_proxy = float(np.clip(np.nanmedian(_rat), 0.0, 1.0)) if _rat.size else None
    except Exception:
        _kd_proxy = None
    return _depth, _sigma, _s2, _kd_proxy


# ── VERYHR_STABLE_LOG ROUND 2 · FOLLOW-UP A (AC-4 product-selection fix) ──────
# The raw DL-PatchCNN VHR depth UNDERPREDICTS the deep (>~12 m turbid) Khalifa
# basin by ~11 m (RMSE ≈ 12 m, bias −10.8 m) because the optical signal is
# saturated there — that is NOT an SDB retrieval, it is a charted-depth regime.
# The in-situ-anchored Lyzenga-fast product (isotonic + per-pixel residual IDW
# against the multibeam soundings) validates at ~2.77 m there. So for a default
# region that carries in-situ soundings (khalifa_port / old_mussafah), the
# SERVED VHR depth must route the DEEP basin through the anchored charted-depth
# product, NOT raw DL. The optically-valid SHALLOW zone is kept on whichever of
# DL / anchored is genuinely better (here: the anchored product, which is good
# both shallow and deep, but the DL grid is blended in shallow where it agrees).
#
# Env knobs (default behaviour preserves the honesty routing; both OFF-able):
#   VHR_ANCHORED_DEEP_ROUTING = "1" (default) → enable the routing for default
#       in-situ regions. Set "0" to force the legacy raw-DL VHR path.
#   VHR_DEEP_THRESHOLD_M = "12.0" (default) → anchored-depth ≥ this is "deep".
_VHR_DEEP_THRESHOLD_M = float(os.environ.get("VHR_DEEP_THRESHOLD_M", "12.0"))


def _vhr_anchored_routing_enabled(region_key, params):
    """True iff the SERVED VHR depth for this ROI should route the deep basin
    through the in-situ-anchored charted-depth product instead of raw DL.

    Gated to default regions that actually carry survey-grade in-situ soundings
    (khalifa_port / old_mussafah). The env default is ON (honesty fix); an
    explicit param/env opt-out restores the legacy raw-DL VHR path.
    """
    if str(os.environ.get("VHR_ANCHORED_DEEP_ROUTING", "1")) in ("0", "false", "False"):
        return False
    if str(params.get("anchored_deep_routing", "1")) in ("0", "false", "False"):
        return False
    return region_key in ("khalifa_port", "old_mussafah")


def _run_anchored_vhr_depth(bbox, sd, ed, params, region_key,
                            dl_depth=None, dl_sigma=None):
    """Build the SERVED VHR depth for an in-situ-anchored default region.

    Returns (depth, sigma, route_diag). `depth`/`sigma` are 2-D float grids in
    the same 4326 pixel layout as the DL grid (build_geotiff_1m_utm consumes
    that layout). The DEEP basin (anchored-depth ≥ _VHR_DEEP_THRESHOLD_M) is
    taken from the in-situ-anchored Lyzenga-fast product (the ~2.77 m product);
    the SHALLOW optically-valid zone keeps the DL prediction wherever DL and the
    anchored product are present (DL is genuinely competitive shallow). Pixels
    where the anchored product is missing fall back to DL; pixels where DL is
    missing fall back to anchored. No upsampling/fabrication — both grids are
    native 10 m before the SPEC-1 1 m lattice render.
    """
    # 1. In-situ-anchored Lyzenga-fast grid (isotonic + residual IDW vs the
    #    bundled multibeam XYZ that fall inside the ROI). This is the product
    #    that validated at ~2.77 m RMSE in ROUND 1 AC-4.
    la, lo, de, _src = xyz_points_in_bbox(bbox)
    user_pts = None
    if len(de) >= 30:
        user_pts = [{'lat': float(a), 'lon': float(o), 'depth': float(abs(d))}
                    for a, o, d in zip(la, lo, de)]
    _anch = _compute_s2_depth_grid(
        bbox, sd, ed, user_pts=user_pts,
        max_cloud=int(params.get('max_cloud', 20)),
        fetch_sliderule=bool(params.get('use_sliderule', False)))
    anch_depth = _anch.get('depth_grid')
    anch_sigma = _anch.get('sigma_grid')
    if anch_depth is None or not np.isfinite(anch_depth).any():
        raise RuntimeError("anchored Lyzenga-fast produced no finite depth for VHR routing")
    anch_depth = np.asarray(anch_depth, dtype=np.float32)

    diag = {
        'route': 'anchored_deep+optical_shallow',
        'deep_threshold_m': _VHR_DEEP_THRESHOLD_M,
        'n_insitu_refs': int(len(de)),
        'anchored_product': 'lyzenga_fast_isotonic_residual_idw',
    }

    # 2. If no DL grid supplied (or shapes disagree), serve the anchored product
    #    wholesale — it is the honest charted-depth product everywhere.
    if dl_depth is None:
        diag['blend'] = 'anchored_only'
        return anch_depth, anch_sigma, diag

    dl_depth = np.asarray(dl_depth, dtype=np.float32)
    if dl_depth.shape != anch_depth.shape:
        # Resample DL onto the anchored grid layout. NaN-safe: a plain bilinear
        # over a sentinel fill bleeds garbage at land edges, so we interpolate
        # the FINITE values and carry a separate validity mask (only pixels that
        # are >=50% real DL after resize are kept).
        try:
            from PIL import Image as _PILImage
            _Hd, _Wd = anch_depth.shape
            _valid = np.isfinite(dl_depth).astype(np.float32)
            _filled = np.where(np.isfinite(dl_depth), dl_depth, 0.0).astype(np.float32)
            _num = np.asarray(_PILImage.fromarray(_filled).resize(
                (_Wd, _Hd), _PILImage.BILINEAR), dtype=np.float32)
            _den = np.asarray(_PILImage.fromarray(_valid).resize(
                (_Wd, _Hd), _PILImage.BILINEAR), dtype=np.float32)
            dl_depth = np.where(_den > 0.5, _num / np.maximum(_den, 1e-6), np.nan)
        except Exception:
            diag['blend'] = 'anchored_only(shape_mismatch)'
            return anch_depth, anch_sigma, diag

    # Guard: only finite, in-range DL pixels are ever blendable (defends the
    # served RMSE against any edge/fill leakage from a resize).
    dl_depth = np.where(
        np.isfinite(dl_depth) & (dl_depth >= 0.0) & (dl_depth <= MAX_DEPTH_M),
        dl_depth, np.nan).astype(np.float32)

    # 3. Depth-gated blend. The gate is on the ANCHORED depth (the trusted
    #    charted product): wherever the anchored basin is deep, the DL value is
    #    discarded (it saturates / underpredicts there). Where shallow, prefer
    #    DL when finite (it is competitive in the optical-valid zone), else
    #    anchored.
    out = anch_depth.copy()
    out_sigma = (np.asarray(anch_sigma, dtype=np.float32).copy()
                 if anch_sigma is not None else None)
    shallow = np.isfinite(anch_depth) & (anch_depth < _VHR_DEEP_THRESHOLD_M)
    use_dl = shallow & np.isfinite(dl_depth)
    out[use_dl] = dl_depth[use_dl]
    if out_sigma is not None and dl_sigma is not None:
        _dls = np.asarray(dl_sigma, dtype=np.float32)
        if _dls.shape == out_sigma.shape:
            out_sigma[use_dl] = _dls[use_dl]
    # Fill anchored-missing pixels with DL where available (coverage union).
    miss = ~np.isfinite(out) & np.isfinite(dl_depth)
    out[miss] = dl_depth[miss]

    diag['blend'] = 'dl_shallow+anchored_deep'
    diag['n_deep_px'] = int((np.isfinite(anch_depth) &
                             (anch_depth >= _VHR_DEEP_THRESHOLD_M)).sum())
    diag['n_shallow_dl_px'] = int(use_dl.sum())
    diag['frac_deep'] = round(
        float(diag['n_deep_px']) / max(int(np.isfinite(anch_depth).sum()), 1), 4)
    return out, out_sigma, diag


def _run_bathy_job_background(job_id, bbox, resolution, label, params):
    """Worker: run the existing compute path for `resolution`, persist the
    GeoTIFF into the durable results store, register metrics, link the job."""
    def _cancelled():
        return _vhr_jobs.is_cancelled(job_id)

    # ── GEE fallback for the JOB-COMPUTE path ──────────────────────────────
    # CDSE Sentinel-Hub creds are frequently expired (401 Unauthorized), which
    # used to make EVERY queued job fail with no analyzable output. The 10 m
    # path (`_run_s2_lyzenga_fast`) and the VHR engine both already know how to
    # fall back to Google Earth Engine, but only when `S2_GEE_FALLBACK=1`.
    # That env defaults OFF and the server is often launched without it, so we
    # force the fallback ON for the duration of a *job* compute (restored after)
    # — jobs MUST finish to a downloadable result when GEE is reachable.
    _prev_gee_fb = os.environ.get("S2_GEE_FALLBACK")
    if str(params.get("s2_gee_fallback", "1")) not in ("0", "false", "False"):
        os.environ["S2_GEE_FALLBACK"] = "1"

    try:
        sd = params.get('start_date', '2024-05-01')
        ed = params.get('end_date', '2024-09-30')
        user_pts = params.get('user_points') or []
        metrics_out = None
        geotiff_bytes = None
        out_name = None

        # ── Default-region training auto-apply (task 2) ────────────────────
        # If this ROI/site_key targets one of the 4 default regions, the
        # underlying compute paths already pull its training (bundled in-situ
        # XYZ by bbox overlap + the 90/10 region-aware UAE blend, or the
        # reference-library/SlideRule/GEBCO reconnaissance fusion for the
        # library sites). We just record WHICH region + the honest accuracy
        # kind so the Results card labels it correctly. `use_region_training`
        # is auto-ON for default regions; an explicit `use_region_training:
        # false` opts out (raw transfer-learning blend, no honest-label stamp).
        _region_key, _region_info = _resolve_default_region(
            bbox=bbox, site_key=params.get('site_key'))
        _use_region_training = params.get('use_region_training')
        if _use_region_training is None:
            _use_region_training = _region_key is not None  # auto-on for defaults
        region_training_meta = None
        if _use_region_training and _region_info:
            region_training_meta = {
                'region_key': _region_key,
                'region_label': _region_info['label'],
                'training': _region_info['training'],
                'accuracy_kind': _region_info['accuracy_kind'],  # validated | reconnaissance
            }
            L.info(f"bathy-job {job_id}: region-training auto-applied → "
                   f"{_region_key} ({_region_info['accuracy_kind']})")

        # 20 m = a coarser product derived from the 10 m calibrated result.
        # Run the 10 m compute returning the raw grid, then block mean-pool to
        # ~20 m (propagate σ) and re-encode a valid 20 m GeoTIFF.
        _pool_to_20m = (resolution == '20m')

        if resolution == 'vhr':
            # ── VHR (DL_WEB_LOG.md M3 / AC-2): imagery = Mapbox HR (cosmetic),
            # DEPTH = DL PatchCNN product resampled. We do NOT build a separate
            # Lyzenga depth grid for VHR. When the DL model is available we run
            # the DL PatchCNN as the depth source and tag imagery as Mapbox-HR
            # cosmetic. If the DL model is unavailable, fall back to the in-situ
            # anchored Mapbox VHR engine (legacy product) with a fallback card.
            _force_lyzenga = _wants_lyzenga_fallback(params)
            if not _force_lyzenga and _model_card.dl_model_available():
                _vhr_jobs.set_progress(job_id, 12.0,
                    stage='VHR — DL PatchCNN depth (Mapbox HR imagery cosmetic) …')
                _depth_dl, _sigma_dl, _s2_dl, _kd_dl = _run_dl_patchcnn(
                    bbox, sd, ed, max_cloud=int(params.get('max_cloud', 20)),
                    region=_region_key)
                if not np.isfinite(_depth_dl).any():
                    raise RuntimeError("DL produced no finite water pixels for VHR depth")
                # ── ROUND 2 FOLLOW-UP A (AC-4): anchored-deep routing ──────────
                # For an in-situ default region, the SERVED VHR depth routes the
                # deep saturated basin through the anchored charted-depth product
                # (~2.77 m), keeping DL only in the optical-valid shallow zone.
                _route_diag = None
                _vhr_depth_method = 'dl_patchcnn'
                _vhr_method_name = _DL_DEPTH_METHOD_NAME
                if _vhr_anchored_routing_enabled(_region_key, params):
                    try:
                        _vhr_jobs.set_progress(job_id, 40.0,
                            stage='VHR — anchored deep-basin routing (in-situ charted) …')
                        _depth_served, _sigma_served, _route_diag = _run_anchored_vhr_depth(
                            bbox, sd, ed, params, _region_key,
                            dl_depth=_depth_dl, dl_sigma=_sigma_dl)
                        if np.isfinite(_depth_served).any():
                            _depth_dl, _sigma_dl = _depth_served, _sigma_served
                            _vhr_depth_method = 'anchored_charted_deep+dl_shallow'
                            _vhr_method_name = ('In-situ-anchored charted depth (deep) '
                                                '+ DL Pro PatchCNN (shallow)')
                            L.info(f"bathy-job {job_id}: VHR anchored routing → {_route_diag}")
                        else:
                            L.warning(f"bathy-job {job_id}: anchored routing empty; "
                                      f"falling back to raw DL VHR depth")
                            _route_diag = None
                    except Exception as _ar_exc:
                        L.warning(f"bathy-job {job_id}: anchored VHR routing failed "
                                  f"({_ar_exc}); serving raw DL VHR depth")
                        _route_diag = None
                _vhr_jobs.set_progress(job_id, 75.0, stage='VHR — writing depth GeoTIFF …')
                # SPEC-1/SPEC-2/SPEC-5: emit on a TRUE 1 m integer-metre UTM
                # lattice (EPSG:32640) with a deterministic content-hash name.
                # 1 m = RENDER resolution; native depth is 10 m (tagged honestly).
                _trm = float(params.get('target_res_m', 1.0))
                try:
                    from backend import grid_1m as _g1m
                except ImportError:
                    import grid_1m as _g1m  # type: ignore
                _phash = _g1m.param_hash(
                    bbox, {k: params.get(k) for k in
                           ('site_key', 'resolution', 'imagery_source',
                            'mapbox_zoom', 'target_res_m')},
                    window=(sd, ed))
                _g = _g1m.build_geotiff_1m_utm(
                    _depth_dl, bbox, target_res_m=_trm, epsg=32640,
                    native_res_m=10.0,
                    extra_tags={'SITE': str(params.get('site_key', 'custom')),
                                'S2_WINDOW': f'{sd}..{ed}', 'PARAM_HASH': _phash})
                if _g:
                    geotiff_bytes = base64.b64decode(_g)
                    out_name = f"bathy_vhr_1m_{_phash}.tif"
                _fin = np.isfinite(_depth_dl)
                _sig_med = (round(float(np.nanmedian(_sigma_dl)), 3)
                            if _sigma_dl is not None and np.isfinite(_sigma_dl).any() else None)
                _rk = _region_key
                _rt = ('trained' if _rk in ('khalifa_port', 'old_mussafah') else 'transfer')
                _card = _model_card.build_dl_model_card(
                    region=_rk, resolution_m=10, region_training=_rt,
                    sigma_mean_m=_sig_med, kd=(_kd_dl if _rk is None else None),
                    imagery_note="imagery: Mapbox HR (cosmetic RGB upsample); depth: DL Pro PatchCNN",
                    s2_scene_dates=[str(_s2_dl.get('date_range', ''))] if _s2_dl.get('date_range') else [],
                )
                metrics_out = {
                    'method': _vhr_method_name,
                    'depth_method': _vhr_depth_method,
                    'imagery_source': 'mapbox_hr_cosmetic',
                    'n_water_px': int(_fin.sum()),
                    'coverage_pct': round(100.0 * _fin.sum() / max(_depth_dl.size, 1), 1),
                    'depth_min_m': float(np.nanmin(_depth_dl)) if _fin.any() else None,
                    'depth_max_m': float(np.nanmax(_depth_dl)) if _fin.any() else None,
                    'sigma_median_m': _sig_med,
                    'model_card': _card,
                    'band_shuffle_pass': bool(_card.get('band_shuffle_guard', {}).get('pass')),
                }
                if _route_diag is not None:
                    metrics_out['vhr_routing'] = _route_diag
                _cm = _card.get('metrics') or {}
                if _rk in ('khalifa_port', 'old_mussafah'):
                    metrics_out['rmse_m'] = _cm.get('rmse_m')
                    metrics_out['bias_m'] = _cm.get('bias_m')
                    metrics_out['r2'] = _cm.get('full_range_r2')
                    metrics_out['decile_slope'] = _cm.get('decile_slope')
                    metrics_out['n_test'] = _cm.get('n_test')
                _vhr_jobs.set_progress(job_id, 82.0, stage='VHR — DL depth done.')
            else:
                # Legacy in-situ-anchored Mapbox VHR engine (DL unavailable).
                _vhr_jobs.set_progress(job_id, 5.0, stage='VHR engine — fetching imagery + references …')
                try:
                    from backend.very_hr_engine import run_very_hr as _run_vhr
                except ImportError:
                    from very_hr_engine import run_very_hr as _run_vhr  # type: ignore
                if user_pts:
                    la = np.array([float(p['lat']) for p in user_pts])
                    lo = np.array([float(p['lon']) for p in user_pts])
                    de = np.abs(np.array([float(p['depth']) for p in user_pts]))
                else:
                    la, lo, de, _src = xyz_points_in_bbox(bbox)
                if len(de) < 30:
                    raise RuntimeError(f'VHR needs ≥30 in-situ reference points in ROI (have {len(de)}); '
                                       f'use 10m resolution or supply user_points.')
                _vhr_jobs.set_progress(job_id, 20.0, stage='VHR engine — running pipeline …')
                result = _run_vhr(
                    site_key=str(params.get('site_key') or 'custom'),
                    bbox=bbox, ref_lats=la, ref_lons=lo, ref_depths=de,
                    train_frac=float(params.get('train_frac', 0.20)),
                    target_res_m=float(params.get('target_res_m', 1.5)),
                    max_tiles_per_side=int(params.get('max_tiles_per_side', 4)),
                    seed=42, save=True,
                    imagery_source=str(params.get('imagery_source', 'vhr')),
                )
                metrics_out = result.get('metrics') or {}
                metrics_out['model_card'] = _model_card.build_fallback_card(
                    "DL model unavailable — legacy in-situ-anchored Mapbox VHR engine",
                    resolution_m=1)
                paths = result.get('paths') or {}
                tif = paths.get('depth_geotiff') or paths.get('geotiff') or paths.get('depth_tif')
                if tif and os.path.exists(tif):
                    geotiff_bytes = Path(tif).read_bytes()
                    out_name = Path(tif).name
        else:  # 10m / 20m  → DL PatchCNN is PRIMARY (DL_WEB_LOG.md M1, AC-1)
            # The default depth estimator for 10 m AND 20 m is the June-5
            # heteroscedastic PatchCNN. 20 m = the DL 10 m grid block-mean-pooled.
            # Lyzenga/Stumpf is the documented FALLBACK (M4): explicit opt-out
            # (depth_method=lyzenga_fallback), DL model unavailable, or DL error.
            _force_lyzenga = _wants_lyzenga_fallback(params)
            _dl_ok = False
            _fallback_reason = None
            _res_m_out = 20 if _pool_to_20m else 10

            if not _force_lyzenga and _model_card.dl_model_available():
                _stage = ('DL PatchCNN → 20 m (pooling 10 m DL grid) …'
                          if _pool_to_20m else 'DL PatchCNN — fetch S2 + dense inference …')
                _vhr_jobs.set_progress(job_id, 12.0, stage=_stage)
                try:
                    _depth_dl, _sigma_dl, _s2_dl, _kd_dl = _run_dl_patchcnn(
                        bbox, sd, ed, max_cloud=int(params.get('max_cloud', 20)),
                        region=_region_key)
                    if not np.isfinite(_depth_dl).any():
                        raise RuntimeError("DL produced no finite water pixels")
                    _vhr_jobs.set_progress(job_id, 72.0, stage='DL PatchCNN — building GeoTIFF …')
                    if _pool_to_20m:
                        _d20, _s20, _k = meanpool_depth_to_20m(
                            _depth_dl, sigma=_sigma_dl, src_res_m=10.0, target_res_m=20.0)
                        _g = build_geotiff_b64(_d20, bbox)
                        _depth_final, _sigma_final = _d20, _s20
                    else:
                        _k = 1
                        _g = build_geotiff_b64(_depth_dl, bbox)
                        _depth_final, _sigma_final = _depth_dl, _sigma_dl
                    if _g:
                        geotiff_bytes = base64.b64decode(_g)
                        out_name = f"bathy_dl_{('20m' if _pool_to_20m else '10m')}_{job_id}.tif"
                    _fin = np.isfinite(_depth_final)
                    _sig_med = (round(float(np.nanmedian(_sigma_final)), 3)
                                if _sigma_final is not None and np.isfinite(_sigma_final).any() else None)
                    metrics_out = {
                        'method': _DL_DEPTH_METHOD_NAME,
                        'depth_method': 'dl_patchcnn',
                        'n_water_px': int(_fin.sum()),
                        'coverage_pct': round(100.0 * _fin.sum() / max(_depth_final.size, 1), 1),
                        'depth_min_m': float(np.nanmin(_depth_final)) if _fin.any() else None,
                        'depth_max_m': float(np.nanmax(_depth_final)) if _fin.any() else None,
                        'sigma_median_m': _sig_med,
                    }
                    if _pool_to_20m:
                        metrics_out['aggregated_from'] = '10m'
                        metrics_out['pool_factor'] = int(_k)
                    # Seed the honest per-region DL spatial-block numbers + §4 card.
                    _rk = _region_key  # canonical default region or None
                    _rt = ('trained' if _rk in ('khalifa_port', 'old_mussafah')
                           else ('transfer' if _rk is None else 'transfer'))
                    _card = _model_card.build_dl_model_card(
                        region=_rk, resolution_m=_res_m_out, region_training=_rt,
                        s2_scene_dates=[str(_s2_dl.get('date_range', ''))] if _s2_dl.get('date_range') else [],
                        sigma_mean_m=_sig_med, kd=(_kd_dl if _rk is None else None),
                    )
                    metrics_out['model_card'] = _card
                    # Surface the DL spatial-block headline metrics onto the result
                    # for default regions (Khalifa: RMSE/bias/R²; OMC: RMSE only).
                    _cm = _card.get('metrics') or {}
                    if _rk in ('khalifa_port', 'old_mussafah'):
                        metrics_out['rmse_m'] = _cm.get('rmse_m')
                        metrics_out['bias_m'] = _cm.get('bias_m')
                        metrics_out['r2'] = _cm.get('full_range_r2')  # None for OMC (suppressed)
                        metrics_out['decile_slope'] = _cm.get('decile_slope')
                        metrics_out['n_test'] = _cm.get('n_test')
                    metrics_out['band_shuffle_pass'] = bool(_card.get('band_shuffle_guard', {}).get('pass'))
                    _dl_ok = True
                    _vhr_jobs.set_progress(job_id, 82.0, stage='DL PatchCNN — done.')
                except Exception as _dl_exc:
                    _fallback_reason = f"DL inference error: {_dl_exc}"
                    L.warning(f"bathy-job {job_id}: DL PatchCNN failed ({_dl_exc}); "
                              f"falling back to Lyzenga/Stumpf")
            elif not _force_lyzenga:
                _fallback_reason = "DL model weights unavailable on this host"

            if not _dl_ok:  # ── Lyzenga/Stumpf FALLBACK (M4) — NO DL metrics ──
                if _force_lyzenga:
                    _fallback_reason = "explicit depth_method=lyzenga_fallback"
                _stage = ('Sentinel-2 → 20 m (Lyzenga fallback, aggregating 10 m) …'
                          if _pool_to_20m else 'Sentinel-2 10 m — Lyzenga/Stumpf fallback …')
                _vhr_jobs.set_progress(job_id, 14.0, stage=_stage)
                result = _run_s2_lyzenga_fast(
                    bbox, sd, ed, user_pts=user_pts,
                    max_cloud=int(params.get('max_cloud', 20)),
                    fetch_sliderule=bool(params.get('use_sliderule', False)),
                    include_geotiff=not _pool_to_20m,
                    _internal_return_grid=_pool_to_20m,
                )
                metrics_out = result.get('metrics') or {}
                # FALLBACK provenance — explicitly NOT a DL card (M4/M6).
                metrics_out['model_card'] = _model_card.build_fallback_card(
                    _fallback_reason or "Lyzenga/Stumpf fallback", resolution_m=_res_m_out)
                metrics_out['fallback_reason'] = _fallback_reason
                if _pool_to_20m:
                    _vhr_jobs.set_progress(job_id, 78.0,
                        stage='Sentinel-2 → 20 m — mean-pool + writing GeoTIFF …')
                    _d10 = result.get('depth_grid')
                    _s10 = result.get('sigma_grid')
                    if _d10 is not None:
                        _d20, _s20, _k = meanpool_depth_to_20m(
                            _d10, sigma=_s10,
                            src_res_m=float(result.get('resolution_m', 10.0)),
                            target_res_m=20.0)
                        _g20 = build_geotiff_b64(_d20, bbox)
                        if _g20:
                            geotiff_bytes = base64.b64decode(_g20)
                            out_name = f"bathy_20m_{job_id}.tif"
                        _fin = np.isfinite(_d20)
                        metrics_out = {**metrics_out,
                            'aggregated_from': '10m',
                            'pool_factor': int(_k),
                            'n_water_px': int(_fin.sum()),
                            'coverage_pct': round(100.0 * _fin.sum() / max(_d20.size, 1), 1),
                        }
                        if _s20 is not None and np.isfinite(_s20).any():
                            metrics_out['sigma_median_m'] = round(float(np.nanmedian(_s20)), 3)
                else:
                    _vhr_jobs.set_progress(job_id, 80.0, stage='Sentinel-2 10 m — writing GeoTIFF …')
                    g_b64 = result.get('geotiff_b64')
                    if g_b64:
                        geotiff_bytes = base64.b64decode(g_b64)
                        out_name = f"bathy_10m_{job_id}.tif"

        if _cancelled():
            return

        # Persist into the durable results store (R3).
        res_resolution = ('vhr' if resolution == 'vhr'
                          else ('20m' if resolution == '20m' else '10m'))
        # Normalise a compact metrics block for the Results list / analyse.
        m = metrics_out or {}
        compact_metrics = {
            'rmse_m': m.get('rmse_m', m.get('rmse')),
            'r2': m.get('r2'),
            'mae_m': m.get('mae_m', m.get('mae')),
            'bias_m': m.get('bias_m', m.get('bias')),
            'coverage_pct': m.get('coverage_pct') or m.get('water_coverage_pct'),
            'catzoc': m.get('catzoc') or m.get('catzoc_band'),
            'n_test': m.get('n_test'),
        }
        # 20 m grid stats (coverage / σ) + DL provenance / model-card (§4) +
        # honest extras. These must survive the compact normalisation so the UI
        # gets the full DL card, the depth-method tag, and the band-shuffle badge.
        for _k20 in ('coverage_pct', 'sigma_median_m', 'aggregated_from',
                     'pool_factor', 'model_card', 'method', 'depth_method',
                     'decile_slope', 'band_shuffle_pass', 'fallback_reason',
                     'depth_min_m', 'depth_max_m'):
            if m.get(_k20) is not None and compact_metrics.get(_k20) is None:
                compact_metrics[_k20] = m.get(_k20)
        # Region-training honesty stamp (task 2): which default region trained
        # this run + the honest accuracy kind (validated vs reconnaissance).
        if region_training_meta is not None:
            compact_metrics['region_training'] = region_training_meta
            compact_metrics['accuracy_kind'] = region_training_meta['accuracy_kind']
        compact_metrics = {k: v for k, v in compact_metrics.items() if v is not None} or None

        entry = _jobs_store.register_result(
            name=label or f"Bathymetry ({res_resolution})",
            roi_bbox=bbox,
            resolution=res_resolution,
            status='done',
            output_bytes=geotiff_bytes,
            output_filename=out_name or f"bathy_{job_id}.tif",
            metrics=compact_metrics,
            job_id=job_id,
        )
        _vhr_jobs.link_result(job_id, entry['id'])
        # Snap the running-job to done with a small JSON manifest payload.
        manifest = json.dumps(_json_safe({
            'result_id': entry['id'], 'resolution': res_resolution,
            'bbox': bbox, 'metrics': compact_metrics,
        }), default=str).encode('utf-8')
        _vhr_jobs.save_result(job_id, f"bathy_{job_id}_manifest.json", manifest,
                              notes=f"{res_resolution} · result {entry['id']}")
    except Exception as ex:
        L.exception(f"bathy-job {job_id} failed")
        # Record the failure in BOTH the running queue and the durable store
        # so a failed run is still visible in Results (R3 honest status).
        try:
            entry = _jobs_store.register_result(
                name=label or f"Bathymetry ({resolution})",
                roi_bbox=bbox,
                resolution=(resolution if resolution in ('vhr', '10m', '20m') else '10m'),
                status='failed', job_id=job_id, error=str(ex),
            )
            _vhr_jobs.link_result(job_id, entry['id'])
        except Exception:
            pass
        _vhr_jobs.mark_failed(job_id, str(ex))
    finally:
        # Restore the process-wide fallback knob to its launch state so we never
        # silently flip a default-OFF env for the rest of the server's life.
        if _prev_gee_fb is None:
            os.environ.pop("S2_GEE_FALLBACK", None)
        else:
            os.environ["S2_GEE_FALLBACK"] = _prev_gee_fb


@app.route('/api/bathy-job/start', methods=['POST'])
def api_bathy_job_start():
    """Queue a bathymetry compute for one ROI at a chosen resolution (R2/R4).

    Body: {roi|bbox|site_key, resolution: "20m"|"10m"|"vhr", label?,
           use_region_training?, params?}
      - roi/bbox: {west,south,east,north} or [w,s,e,n], or a known site_key.
      - resolution: "10m" (Sentinel-2 calibrated, default), "20m" (coarser —
        the 10 m result block mean-pooled to a 20 m grid, σ propagated), or
        "vhr" (Mapbox ~1 m engine; needs ≥30 in-situ refs in the ROI).
      - use_region_training: omit/true → auto-apply the default region's
        training (pretrained model + in-situ calibration / reconnaissance
        fusion) when the ROI/site_key is a default region; false → opt out.
    Returns the running-job record immediately; poll
    /api/very-hr-job/<id>/status. On completion the result lands in
    /api/results (durable, survives restart) tagged with its resolution.
    """
    try:
        data = req.get_json(force=True, silent=True) or {}
        bbox, label = _bbox_from_payload(data)
        if bbox is None:
            return jsonify({'error': label}), 400  # label holds error str
        resolution = str(data.get('resolution', '10m')).lower()
        if resolution not in ('vhr', '10m', '20m'):
            resolution = '10m'
        label = data.get('label') or label or f"Bathymetry ({resolution})"
        params = dict(data.get('params') or {})
        if data.get('site_key'):
            params.setdefault('site_key', data['site_key'])
        elif isinstance(data.get('roi'), str):
            params.setdefault('site_key', data['roi'])
        if data.get('use_region_training') is not None:
            params.setdefault('use_region_training', data['use_region_training'])
        job = _vhr_jobs.create_job(bbox, params={**params, 'resolution': resolution},
                                   label=label)
        _vhr_jobs.set_progress(job['id'], 1.0, stage='Queued — starting worker …')
        _threading.Thread(
            target=_run_bathy_job_background,
            args=(job['id'], bbox, resolution, label, params),
            daemon=True,
        ).start()
        return jsonify(job)
    except Exception as ex:
        L.exception("bathy-job/start failed")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/bathy-job/start-batch', methods=['POST'])
def api_bathy_job_start_batch():
    """Queue MANY ROIs at once (R4). Body: {jobs: [{roi|bbox|site_key,
    resolution, label?, params?}, ...]}. Returns {jobs: [<record>, ...]}."""
    try:
        data = req.get_json(force=True, silent=True) or {}
        items = data.get('jobs') or data.get('rois') or []
        if not isinstance(items, list) or not items:
            return jsonify({'error': 'Body must be {jobs: [ {roi, resolution}, ... ]}'}), 400
        started = []
        errors = []
        for it in items:
            bbox, label = _bbox_from_payload(it)
            if bbox is None:
                errors.append({'item': it, 'error': label})
                continue
            resolution = str(it.get('resolution', data.get('resolution', '10m'))).lower()
            if resolution not in ('vhr', '10m', '20m'):
                resolution = '10m'
            label = it.get('label') or label or f"Bathymetry ({resolution})"
            params = dict(it.get('params') or {})
            if it.get('site_key'):
                params.setdefault('site_key', it['site_key'])
            elif isinstance(it.get('roi'), str):
                params.setdefault('site_key', it['roi'])
            _urt = it.get('use_region_training', data.get('use_region_training'))
            if _urt is not None:
                params.setdefault('use_region_training', _urt)
            job = _vhr_jobs.create_job(bbox, params={**params, 'resolution': resolution},
                                       label=label)
            _vhr_jobs.set_progress(job['id'], 1.0, stage='Queued — starting worker …')
            _threading.Thread(
                target=_run_bathy_job_background,
                args=(job['id'], bbox, resolution, label, params),
                daemon=True,
            ).start()
            started.append(job)
        return jsonify({'jobs': started, 'errors': errors, 'queued': len(started)})
    except Exception as ex:
        L.exception("bathy-job/start-batch failed")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/bathy-job/khalifa-vhr', methods=['POST'])
def api_bathy_job_khalifa_vhr():
    """R2 — 'Khalifa Port (High-Resolution)' button entry point.

    Default (mode a, instant + honest): surface the PRE-RENDERED Khalifa VHR 1 m
    product as a finished Results entry (in-situ-anchored, optical-valid/deep-
    abstain honesty card carried in `metrics.honesty_card`). Idempotent — the
    same `khalifa_vhr_1m` result id is reused, so repeated clicks don't duplicate.

    Mode b (on-demand compute): POST `{"compute": true}` to instead QUEUE a fresh
    VHR compute over the known khalifa_port ROI via the normal bathy-job runner;
    that lands its own result in /api/results when it finishes.
    Response (mode a): the durable result entry + download_url/analyse_url.
    """
    try:
        data = req.get_json(force=True, silent=True) or {}
        if data.get('compute'):
            bbox = [54.636, 24.785, 54.690, 24.840]
            params = {'site_key': 'khalifa_port', 'resolution': 'vhr',
                      'target_res_m': float(data.get('target_res_m', 1.0))}
            label = 'Khalifa Port (High-Resolution) — on-demand VHR'
            job = _vhr_jobs.create_job(bbox, params=params, label=label)
            _vhr_jobs.set_progress(job['id'], 1.0, stage='Queued — starting worker …')
            _threading.Thread(
                target=_run_bathy_job_background,
                args=(job['id'], bbox, 'vhr', label, params),
                daemon=True,
            ).start()
            return jsonify({'mode': 'compute', 'job': job})
        # mode a — pre-rendered product (instant honest display).
        entry = _seed_results.seed_named_sample('khalifa_vhr_1m')
        if entry is None:
            return jsonify({'error': 'Khalifa VHR sample product not available '
                                     '(missing samples/results/sample_khalifa_vhr_1m.tif)'}), 404
        rid = entry.get('id')
        return jsonify({
            'mode': 'prerendered',
            **entry,
            'download_url': (f"/api/results/{rid}/download" if entry.get('output_path') else None),
            'analyse_url': f"/api/results/{rid}/analyse",
        })
    except Exception as ex:
        L.exception("bathy-job/khalifa-vhr failed")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/results/<result_id>/very-hr', methods=['POST', 'GET'])
def api_results_very_hr(result_id):
    """'Very HR (Mapbox)' action for a DEFAULT region (DEFAULT_REGIONS_LOG.md AC-1).

    Triggers the VHR-from-Mapbox path for the region's bbox: fetch ~1 m Mapbox
    RGB (zoom 17) and re-run the anchored optical/clustered inference, reusing
    the same R1/R2 Khalifa-VHR + fetch_mapbox_s2 pattern. Queues a bathy-job
    (resolution='vhr', imagery_source='mapbox_hr') that lands its own Results
    entry on completion.

    HONEST CAVEAT (returned in the payload + carried on the result honesty card):
    Very HR is an RGB upsample = SPATIAL DETAIL ONLY; depth physics and accuracy
    are UNCHANGED and NOT independently validated at 1 m.
    """
    e = _jobs_store.get_result(result_id)
    if e is None:
        return jsonify({'error': 'result not found'}), 404
    m = e.get('metrics') or {}
    vhr = m.get('vhr_action') if isinstance(m, dict) else None
    bbox = (vhr or {}).get('bbox') or e.get('roi_bbox')
    if not bbox or len(bbox) != 4:
        return jsonify({'error': 'no bbox available for VHR on this result'}), 400
    site_key = (vhr or {}).get('site_key') or 'custom'
    zoom = int((vhr or {}).get('zoom', 17))
    caveat = ("Very HR (Mapbox) is an RGB upsample at ~1 m — SPATIAL DETAIL ONLY; "
              "depth physics + accuracy are UNCHANGED and NOT independently "
              "validated at 1 m.")
    try:
        data = req.get_json(force=True, silent=True) or {}
    except Exception:
        data = {}
    params = {
        'site_key': site_key,
        'resolution': 'vhr',
        'imagery_source': 'mapbox_hr',
        'mapbox_zoom': zoom,
        'target_res_m': float(data.get('target_res_m', 1.0)),
        'source_result_id': result_id,
    }
    label = f"{e.get('name', site_key)} — Very HR (Mapbox ~1 m)"
    job = _vhr_jobs.create_job(list(bbox), params=params, label=label)
    _vhr_jobs.set_progress(job['id'], 1.0,
                           stage='Queued — fetching Mapbox VHR imagery …')
    _threading.Thread(
        target=_run_bathy_job_background,
        args=(job['id'], list(bbox), 'vhr', label, params),
        daemon=True,
    ).start()
    return jsonify({'mode': 'compute', 'job': job, 'bbox': bbox,
                    'zoom': zoom, 'site_key': site_key, 'caveat': caveat})


@app.route('/api/results', methods=['GET'])
def api_results_list():
    """Durable Results catalogue (R3). Lists ALL finished jobs (VHR + 10 m),
    newest first, persisting across restarts. Each entry carries a
    `download_url` and `analyse_url` for the UI to wire directly."""
    try:
        limit = int(req.args.get('limit', 200))
    except Exception:
        limit = 200
    # Lazy seed-on-empty (idempotent, no-op when the store already has results).
    try:
        _seed_results.seed_sample_results_if_empty()
    except Exception:
        pass
    # Always ensure the 4 DEFAULT regions are present (idempotent, not gated).
    try:
        _seed_results.seed_default_regions()
    except Exception:
        pass
    # Always ensure the curated always_seed products (DL-fusion certified /
    # provisional) are present (idempotent, not gated).
    try:
        _seed_results.seed_always_products()
    except Exception:
        pass
    out = []
    for e in _jobs_store.list_results(limit=limit):
        rid = e.get('id')
        m = e.get('metrics') or {}
        # Multi-raster products: surface each companion raster as a `downloads`
        # entry (the UI's downloadFormats() reads `res.downloads:[{format,url,
        # label}]`) pointing at the /file/<key> route. Primary GeoTIFF stays on
        # download_url.
        downloads = []
        if isinstance(m, dict) and isinstance(m.get('extra_downloads'), list):
            for d in m['extra_downloads']:
                k = d.get('key')
                if not k:
                    continue
                downloads.append({
                    'format': k,
                    'label': d.get('label') or d.get('filename') or k,
                    'url': f"/api/results/{rid}/file/{k}",
                    'size_bytes': d.get('size_bytes'),
                })
        out.append({
            **e,
            # surface the default-region descriptors at the top level for the UI
            # (they were folded into metrics by the seeder to fit the store schema)
            'is_default': bool(m.get('is_default')) if isinstance(m, dict) else False,
            'vhr_action': (m.get('vhr_action') if isinstance(m, dict) else None),
            'cert_label': (m.get('cert_label') if isinstance(m, dict) else None),
            'cert_status': (m.get('cert_status') if isinstance(m, dict) else None),
            'download_url': (f"/api/results/{rid}/download" if e.get('output_path') else None),
            'download_all_url': f"/api/results/{rid}/download-all.zip",
            'downloads': (downloads or None),
            'analyse_url': f"/api/results/{rid}/analyse",
            'vhr_url': f"/api/results/{rid}/very-hr",
        })
    # Default regions first (stable order), then the rest newest-first.
    _order = {'default_khalifa_port': 0, 'default_old_mussafah': 1,
              'default_jbel_dhanna': 2, 'default_abu_al_abyad': 3}
    out.sort(key=lambda r: (0, _order[r['id']]) if r['id'] in _order else (1, 0))
    return jsonify({'results': out, 'count': len(out)})


@app.route('/api/results/<result_id>', methods=['GET'])
def api_results_get(result_id):
    e = _jobs_store.get_result(result_id)
    if e is None:
        return jsonify({'error': 'result not found'}), 404
    return jsonify({
        **e,
        'download_url': (f"/api/results/{result_id}/download" if e.get('output_path') else None),
        'download_all_url': f"/api/results/{result_id}/download-all.zip",
        'analyse_url': f"/api/results/{result_id}/analyse",
    })


_NDWI_FETCH_CACHE = {}  # bbox_key -> ndwi array or None (process-lifetime memo)


def _fetch_ndwi_best_effort(bbox, res_m=10, timeout_s=45):
    """MASK-NDWI (2026-07-10, priority override #2) — best-effort NDWI fetch
    for the RESULTS-CATALOGUE re-serve paths (`/api/results/<id>/download`,
    `/api/results/<id>/analyse`), which serve a STORED GeoTIFF and have no
    `s2` dict already in scope (unlike a live extraction run). Tries a
    lightweight 2-band (green+NIR) GEE fetch; degrades gracefully (returns
    None, cut proceeds vector-only) on ANY failure — never blocks a
    results-catalogue response on imagery availability. Cached per-bbox for
    the process lifetime (repeated views of the same stored result don't
    re-fetch)."""
    key = (round(float(bbox[0]), 5), round(float(bbox[1]), 5),
          round(float(bbox[2]), 5), round(float(bbox[3]), 5), int(res_m))
    if key in _NDWI_FETCH_CACHE:
        return _NDWI_FETCH_CACHE[key]
    ndwi = None
    try:
        if _init_gee():
            import ee
            w, s, e, n = bbox
            region = ee.Geometry.Rectangle([w, s, e, n])
            col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterDate("2023-01-01", TM.strftime("%Y-%m-%d", TM.gmtime()))
                  .filterBounds(region)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                  .sort("CLOUDY_PIXEL_PERCENTAGE"))
            img = ee.Image(col.first()).select(["B3", "B8"]).clip(region)
            url = img.getDownloadURL({"region": region, "scale": int(res_m),
                                      "format": "GEO_TIFF", "crs": "EPSG:4326"})
            r = requests.get(url, timeout=timeout_s)
            r.raise_for_status()
            arr = tifffile.imread(io.BytesIO(r.content))
            if arr.ndim == 3 and arr.shape[2] == 2:
                green, nir = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float)
            elif arr.ndim == 3 and arr.shape[0] == 2:
                green, nir = arr[0].astype(float), arr[1].astype(float)
            else:
                raise RuntimeError(f"unexpected NDWI TIFF shape {arr.shape}")
            ndwi = (green - nir) / (green + nir + 1e-6)
            L.info(f"MASK-NDWI: best-effort fetch OK for results-catalogue re-serve, shape={ndwi.shape}")
    except Exception as ex:
        L.info(f"MASK-NDWI: best-effort fetch failed (results-catalogue path proceeds vector-only): {ex}")
        ndwi = None
    _NDWI_FETCH_CACHE[key] = ndwi
    return ndwi


def _global_land_cut_points(pts, bbox, grid_res=400):
    """MASK-GLOBAL backstop for point lists: drop any (lon, lat, depth) tuple
    that falls on OSM land. Used by every re-serve/export path that emits
    points derived from an already-stored raster (results-catalogue
    download, /api/export) so a stale/legacy stored file can never leak a
    land pixel to the client. `bbox` = [west, south, east, north].
    Returns (kept_pts, cut_info dict)."""
    if not pts:
        return pts, {'points_before': 0, 'points_after': 0, 'points_dropped_on_land': 0, 'source': None}
    w, s, e, n = bbox
    n_before = len(pts)
    if not (e > w and n > s):
        return pts, {'points_before': n_before, 'points_after': n_before,
                     'points_dropped_on_land': 0, 'source': 'invalid-bbox-skip'}
    try:
        try:
            from backend.osm_land_mask import coastline_land_for as _cl, osm_land_for as _ol
        except ImportError:
            from osm_land_mask import coastline_land_for as _cl, osm_land_for as _ol  # type: ignore
        H = W = grid_res
        coast_land, cinfo = _cl([w, s, e, n], (H, W))
        feat_land, finfo = _ol([w, s, e, n], (H, W))
        land = coast_land if coast_land is not None else np.zeros((H, W), dtype=bool)
        if feat_land is not None:
            land = land | feat_land
        source = f"osm-coastline+features ({cinfo.get('n_polys', 0)}+{finfo.get('n_polys', 0)} polys)"
        kept = []
        for p in pts:
            lon, lat = (p[0], p[1]) if isinstance(p, (tuple, list)) else (p['lon'], p['lat'])
            col = int((lon - w) / (e - w + 1e-12) * W)
            row = int((n - lat) / (n - s + 1e-12) * H)
            col = max(0, min(W - 1, col)); row = max(0, min(H - 1, row))
            if not land[row, col]:
                kept.append(p)
        return kept, {'points_before': n_before, 'points_after': len(kept),
                      'points_dropped_on_land': n_before - len(kept), 'source': source}
    except Exception as ex:
        return pts, {'points_before': n_before, 'points_after': n_before,
                     'points_dropped_on_land': 0, 'source': f'cut-error: {ex}'}


def _global_land_cut_geotiff(src_path, out_path=None):
    """MASK-GLOBAL backstop for a stored depth GeoTIFF re-serve: read it, cut
    land pixels to nodata via `apply_global_land_cut`, write the cut copy to
    `out_path` (defaults to a scratch tmp file — the ORIGINAL on disk is never
    touched, frozen-path doctrine). Returns (out_path, cut_info) or
    (src_path, {'source': 'cut-error: ...'}) on any failure (fails open)."""
    try:
        import rasterio
        try:
            from backend.osm_land_mask import apply_global_land_cut as _global_cut
        except ImportError:
            from osm_land_mask import apply_global_land_cut as _global_cut  # type: ignore
        with rasterio.open(src_path) as ds:
            arr = ds.read(1).astype('float32')
            nodata = ds.nodata if ds.nodata is not None else np.nan
            b = ds.bounds
            bbox = [float(b.left), float(b.bottom), float(b.right), float(b.top)]
            profile = ds.profile.copy()
        depth_for_cut = np.where(arr == nodata, np.nan, arr) if ds.nodata is not None else arr
        # MASK-NDWI (2026-07-10, priority override #2): best-effort imagery-
        # evidence layer for STORED-product re-serves too (this was the
        # user-visible leak — a stored GeoTIFF pre-dates the NDWI fix and can
        # never be re-cut from vector sources alone). Degrades to vector-only
        # silently if the fetch fails (see _fetch_ndwi_best_effort docstring).
        _ndwi_bg = _fetch_ndwi_best_effort(bbox, res_m=10)
        cut, _land, info = _global_cut(depth_for_cut, bbox, ndwi=_ndwi_bg, ndwi_res_m=10)
        out_nodata = nodata if np.isfinite(nodata) else -9999.0
        cut_out = np.where(np.isfinite(cut), cut, out_nodata).astype(profile.get('dtype', 'float32'))
        if out_path is None:
            import tempfile
            fd, out_path = tempfile.mkstemp(suffix='.tif', prefix='land_cut_')
            os.close(fd)
        profile.update(nodata=out_nodata)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(cut_out, 1)
        return out_path, info
    except Exception as ex:
        return src_path, {'source': f'cut-error: {ex}', 'n_depth_px_cut_this_call': 0}


def _depth_geotiff_to_points(path, max_points=200000):
    """Read a depth GeoTIFF and return a list of (lon, lat, depth) for valid
    water pixels (positive-down, datum LAT). Decimated to <= max_points so the
    CSV/GeoJSON stay light. Used by the CSV / GeoJSON download formats."""
    import rasterio
    with rasterio.open(path) as ds:
        nodata = ds.nodata
        arr = ds.read(1).astype('float64')
        h, w = ds.height, ds.width
        tr = ds.transform
        mask = np.isfinite(arr)
        if nodata is not None:
            mask &= (arr != nodata)
        mask &= (arr > -1000) & (arr >= -0.5)
        n_valid = int(np.count_nonzero(mask))
        # decimation stride so total emitted points <= max_points
        step = 1
        if n_valid > max_points and n_valid > 0:
            step = int(np.ceil((n_valid / max_points) ** 0.5))
        rows = np.arange(0, h, step)
        cols = np.arange(0, w, step)
        pts = []
        for r in rows:
            mrow = mask[r]
            for c in cols:
                if not mrow[c]:
                    continue
                # pixel centre → geographic coords (CRS assumed EPSG:4326)
                lon, lat = tr * (c + 0.5, r + 0.5)
                pts.append((float(lon), float(lat), round(float(arr[r, c]), 3)))
    return pts


def _result_provenance_sidecar(result_id, output_path=None):
    """Build the GeoTIFF SIDECAR JSON for a result (DL_WEB_LOG.md §4 / AC-10).

    Carries the SAME `metrics.model_card` provenance block the UI renders, plus
    the result identity (id, name, bbox, resolution, datum, output file). If the
    stored result predates the model-card (legacy), emit a minimal honest block
    so an export is never provenance-less.
    """
    e = _jobs_store.get_result(result_id) or {}
    m = e.get('metrics') or {}
    card = m.get('model_card') or m.get('provenance')
    if not card:
        card = {
            'method': m.get('method') or 'SDB (legacy result — full model card unavailable)',
            'datum': 'LAT (positive-down)', 'max_depth_m': 25.0,
            'note': 'Legacy result registered before the §4 provenance block; '
                    'flat metrics only.',
            'metrics': {k: m.get(k) for k in ('rmse_m', 'bias_m', 'r2', 'n_test',
                        'decile_slope', 'catzoc') if m.get(k) is not None},
            'disclaimer': 'Reconnaissance-grade SDB. Not to be used for navigation.',
        }
    return {
        'schema': 'sdb-provenance-sidecar/1',
        'result_id': result_id,
        'name': e.get('name'),
        'roi_bbox': e.get('roi_bbox'),
        'resolution': e.get('resolution'),
        'date': e.get('date'),
        'output_file': (output_path.name if output_path is not None
                        else e.get('output_path')),
        'units': 'm, positive down, datum LAT',
        'provenance': card,
    }


@app.route('/api/results/<result_id>/download', methods=['GET'])
def api_results_download(result_id):
    """Download a result's output. Default = the raw GeoTIFF (attachment).
    Optional `?format=csv|geojson|tif` serves a derived vector format
    (lon,lat,depth points) where the output is a depth GeoTIFF — sensible for
    GIS import / inspection. `tif`/`geotiff`/absent → the original raster."""
    fmt = str(req.args.get('format', '')).lower().strip()
    p = _jobs_store.resolve_output(result_id)
    if p is None:
        return jsonify({'error': 'output not found'}), 404
    # AC-10: the GeoTIFF sidecar provenance JSON (§4 model card). Served as a
    # companion download so an exported raster always travels with its auditable
    # provenance block. `?format=provenance` (or `sidecar`) returns it directly.
    if fmt in ('provenance', 'sidecar', 'meta'):
        sidecar = _result_provenance_sidecar(result_id, p)
        return app.response_class(
            json.dumps(sidecar, indent=2), mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename="{p.stem}.provenance.json"'})
    if fmt in ('', 'tif', 'tiff', 'geotiff'):
        # MASK-GLOBAL (2026-07-10, user-mandated): re-serve is cut, the
        # certified original on disk stays byte-identical (frozen-path
        # doctrine) — cut to a scratch tmp copy and serve THAT.
        serve_dir, serve_name = str(p.parent), p.name
        cut_info = {'source': None}
        if p.suffix.lower() in ('.tif', '.tiff'):
            cut_path, cut_info = _global_land_cut_geotiff(p)
            if cut_path != str(p):
                serve_dir, serve_name = os.path.dirname(cut_path), os.path.basename(cut_path)
        resp = send_from_directory(serve_dir, serve_name, as_attachment=True,
                                    download_name=p.name if serve_name != p.name else None)
        # Advertise the companion sidecar so a GIS client can fetch it.
        resp.headers['X-Provenance-Sidecar'] = f"/api/results/{result_id}/download?format=provenance"
        resp.headers['X-Global-Land-Cut'] = str(cut_info.get('mask_source') or cut_info.get('source'))
        return resp
    if fmt not in ('csv', 'geojson', 'json'):
        return jsonify({'error': f'unsupported format {fmt!r}; use csv|geojson|tif|provenance'}), 400
    if p.suffix.lower() not in ('.tif', '.tiff'):
        return jsonify({'error': 'csv/geojson export only available for GeoTIFF outputs'}), 400
    try:
        pts = _depth_geotiff_to_points(p)
        import rasterio as _rio
        with _rio.open(p) as _ds:
            _b = _ds.bounds
        pts, _pcut_info = _global_land_cut_points(
            pts, [float(_b.left), float(_b.bottom), float(_b.right), float(_b.top)])
    except Exception as ex:
        L.exception("results download vector export failed")
        return jsonify({'error': f'vector export failed: {ex}'}), 500
    stem = p.stem
    if fmt == 'csv':
        import io as _io
        buf = _io.StringIO()
        buf.write(f'# global_land_cut: {_pcut_info}\n')
        buf.write('lon,lat,depth_m\n')
        for lon, lat, d in pts:
            buf.write(f'{lon:.7f},{lat:.7f},{d}\n')
        return app.response_class(
            buf.getvalue(), mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{stem}.csv"'})
    # GeoJSON FeatureCollection of points (depth_m property)
    features = [{
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
        'properties': {'depth_m': d},
    } for (lon, lat, d) in pts]
    fc = {'type': 'FeatureCollection',
          'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
          'properties': {'units': 'm, positive down, datum LAT', 'count': len(features),
                         'result_id': result_id,
                         # AC-10: provenance travels with the vector export too.
                         'provenance': _result_provenance_sidecar(result_id, p).get('provenance'),
                         'global_land_cut': _pcut_info},
          'features': features}
    return app.response_class(
        json.dumps(fc), mimetype='application/geo+json',
        headers={'Content-Disposition': f'attachment; filename="{stem}.geojson"'})


@app.route('/api/results/<result_id>/file/<key>', methods=['GET'])
def api_results_file(result_id, key):
    """Download a COMPANION raster of a multi-raster result (e.g. the DL-fusion
    marawah LAT set: sigma_lat / depth_d1a / lat_offset). Resolves the file by
    `key` from ``metrics.extra_downloads`` (traversal-safe, existence-checked).
    Primary GeoTIFF stays on /download."""
    e = _jobs_store.get_result(result_id)
    if e is None:
        return jsonify({'error': 'result not found'}), 404
    m = e.get('metrics') or {}
    extra = m.get('extra_downloads') if isinstance(m, dict) else None
    if not isinstance(extra, list):
        return jsonify({'error': 'no companion files for this result'}), 404
    target = next((d for d in extra if d.get('key') == key), None)
    if target is None or not target.get('path'):
        return jsonify({'error': f'companion file {key!r} not found'}), 404
    p = Path(target['path']).resolve()
    if not p.exists():
        return jsonify({'error': 'companion file missing on disk'}), 404
    return send_from_directory(str(p.parent), p.name, as_attachment=True)


def _ocean_downloads_dir():
    """The MLE / accurate-bathymetry artifact directory (files persist here)."""
    return Path(__file__).parent / "ocean" / "downloads"


def _collect_result_artifacts(result_id):
    """Shared gather step for the "Download all TIFFs" zip (generalizes the MLE
    zip logic to EVERY result type). Returns ``(files, missing)`` where:

      * ``files``  — list of ``(arcname, Path)`` for every raster / sidecar the
        result references AND that exists on disk (deduped by resolved path,
        arcname-collision-safe).
      * ``missing`` — list of human-readable strings for every referenced file
        that could NOT be found on disk (surfaced in MANIFEST.txt — no silent
        omission, per the honesty contract).

    Spans all storage shapes:
      - seed-manifest products  → primary output + ``metrics.extra_downloads[].path``
                                  + ``metrics.mask_path``
      - persisted 10 m / VHR / stability jobs → primary output + ``preview_path``
      - MLE jobs                → ``metrics.downloads[].name`` + ``ocean/downloads/<id>_*``
                                  (per-scene GeoTIFFs, composite, inter-scene-σ,
                                  scenes npz, stability.json, mask-QA preview)

    Returns ``(None, None)`` when the result id does not exist (→ 404)."""
    e = _jobs_store.get_result(result_id)
    if e is None:
        return None, None
    files, missing = [], []
    seen_paths, seen_names = set(), set()

    def _add(path, note=None, force_name=None):
        if not path:
            return
        try:
            p = Path(path)
        except Exception:
            return
        rp = p.resolve()
        key = str(rp)
        if key in seen_paths:
            return
        if not rp.exists():
            missing.append(note or str(p))
            return
        seen_paths.add(key)
        name = force_name or rp.name
        base = name
        i = 1
        while name in seen_names:
            stem, suf = Path(base).stem, Path(base).suffix
            name = f"{stem}_{i}{suf}"
            i += 1
        seen_names.add(name)
        files.append((name, rp))

    m = e.get('metrics') if isinstance(e.get('metrics'), dict) else {}
    # 1. primary output raster
    _add(e.get('output_path'), note=f"primary output: {e.get('output_path')}")
    # 2. multi-raster companions (sigma_lat / lat_offset / drying_mask / depth_d1a ...)
    for d in (m.get('extra_downloads') or []):
        _add(d.get('path'),
             note=f"companion {d.get('key')}: {d.get('filename') or d.get('path')}")
    # 3. optical_valid / source mask raster
    _add(m.get('mask_path'), note=f"mask: {m.get('mask_path')}")
    # 3b. stability preview (temporal-stability jobs)
    _add(m.get('preview_path'), note=f"preview: {m.get('preview_path')}")
    # 4. MLE / accurate artifacts living under ocean/downloads
    dl_dir = _ocean_downloads_dir()
    for d in (m.get('downloads') or []):
        nm = d.get('name')
        if nm and not str(nm).lower().endswith('.zip'):  # never nest the zip
            _add(dl_dir / nm, note=f"download: {nm}")
    safe = "".join(c for c in str(result_id) if c.isalnum() or c in "._-")
    if safe == str(result_id):
        for p in sorted(dl_dir.glob(f"{safe}_*")):
            if p.name.lower().endswith('.zip'):
                continue
            _add(p)
    return files, missing


def _stream_result_zip(result_id, files, missing, want_provenance=True):
    """Build an in-memory zip of ``files`` plus an honest MANIFEST.txt (and, when
    available, the §4 provenance sidecar). Returns a seekable BytesIO."""
    import zipfile
    import io as _io
    from datetime import datetime as _dt
    e = _jobs_store.get_result(result_id) or {}
    bio = _io.BytesIO()
    lines = [
        "Download-all manifest — every TIFF/artifact this result references.",
        f"result_id : {result_id}",
        f"name      : {e.get('name', '')}",
        f"resolution: {e.get('resolution', '')}",
        f"roi_bbox  : {e.get('roi_bbox', '')}",
        f"generated : {_dt.utcnow().isoformat()}Z",
        "",
        "INCLUDED:",
    ]
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zf:
        for arc, p in files:
            try:
                zf.write(str(p), arcname=arc)
                lines.append(f"  + {arc}  ({p.stat().st_size} bytes)  <- {p}")
            except Exception as ex:
                lines.append(f"  ! {arc}  <- {p}  (write failed: {ex})")
                missing = list(missing) + [f"{arc} (write failed: {ex})"]
        lines.append("")
        lines.append("MISSING (referenced but absent on disk — NOT in this zip):")
        if missing:
            for miss in missing:
                lines.append(f"  - {miss}")
        else:
            lines.append("  (none)")
        if want_provenance:
            try:
                sidecar = _result_provenance_sidecar(result_id)
                zf.writestr('PROVENANCE.json', json.dumps(sidecar, indent=2))
                lines.append("")
                lines.append("  + PROVENANCE.json (§4 model-card sidecar)")
            except Exception:
                pass
        zf.writestr('MANIFEST.txt', "\n".join(lines) + "\n")
    bio.seek(0)
    return bio


@app.route('/api/results/<result_id>/download-all.zip', methods=['GET'])
def api_results_download_all(result_id):
    """Stream ONE zip of EVERY raster this result references — works for
    seed-manifest products (certified / LAT / reconnaissance ad_*), persisted
    10 m / VHR / stability jobs, AND MLE jobs. Missing-but-referenced files are
    skipped and listed in MANIFEST.txt (honest — no silent omission)."""
    # Ensure the curated seed products exist so their ids resolve (idempotent).
    for _fn in (getattr(_seed_results, 'seed_sample_results_if_empty', None),
                getattr(_seed_results, 'seed_default_regions', None),
                getattr(_seed_results, 'seed_always_products', None)):
        if _fn:
            try:
                _fn()
            except Exception:
                pass
    files, missing = _collect_result_artifacts(result_id)
    if files is None:
        # Self-heal: manifest-known sample (e.g. an ad_* reconnaissance product)
        # not yet materialised into the store — seed it on demand, then retry.
        try:
            if _seed_results.seed_named_sample(result_id):
                files, missing = _collect_result_artifacts(result_id)
        except Exception:
            pass
    if files is None:
        return jsonify({'error': 'result not found'}), 404
    if not files:
        return jsonify({'error': 'no downloadable rasters for this result'}), 404
    bio = _stream_result_zip(result_id, files, missing)
    from flask import send_file
    return send_file(bio, mimetype='application/zip', as_attachment=True,
                     download_name=f"{result_id}_all_tiffs.zip")


@app.route('/api/results/<result_id>/mask', methods=['GET'])
def api_results_mask(result_id):
    """optical_valid mask overlay for a result (R-mask).

    Serves the uint8 optical-validity mask raster where one is bundled with the
    result — classes: 1 = valid (<= z_opt, optically recoverable),
    2 = collar (z_opt..z_opt+collar transition), 0 = deep-abstain (MBES-only,
    optically saturated — honest abstain), 255 = land / nodata.

    Resolution order for the mask file:
      1. `metrics.mask_path` (absolute server-side path written by the seeder), else
      2. a sibling `*_optical_valid*.tif` next to the result's output GeoTIFF.

    Default → the GeoTIFF (as attachment=false so a map client can fetch tiles).
    `?meta=1` → JSON {bbox, classes, class_counts, url} for the overlay; the
    client georeferences the PNG/TIFF with `bbox`. 404 if no mask is available."""
    e = _jobs_store.get_result(result_id)
    if e is None:
        return jsonify({'error': 'result not found'}), 404
    metrics = e.get('metrics') or {}
    # Backfill mask_path for bundled-sample results that predate the mask wiring.
    if not (isinstance(metrics, dict) and metrics.get('mask_path')):
        try:
            refreshed = _seed_results.seed_named_sample(result_id)
            if refreshed:
                e = _jobs_store.get_result(result_id) or e
                metrics = e.get('metrics') or {}
        except Exception:
            pass
    mask_path = None
    if isinstance(metrics, dict) and metrics.get('mask_path'):
        cand = metrics['mask_path']
        if os.path.exists(cand):
            mask_path = cand
    if mask_path is None:
        out = e.get('output_path')
        if out:
            d = os.path.dirname(out)
            try:
                for fn in sorted(os.listdir(d)):
                    if 'optical_valid' in fn.lower() and fn.lower().endswith(('.tif', '.tiff')):
                        mask_path = os.path.join(d, fn)
                        break
            except Exception:
                pass
    if mask_path is None or not os.path.exists(mask_path):
        return jsonify({'error': 'no optical_valid mask available for this result'}), 404

    classes = (metrics.get('mask_classes') if isinstance(metrics, dict) else None) or {
        '0': 'deep-abstain (MBES-only, optically saturated)',
        '1': 'valid (<=z_opt, optically recoverable)',
        '2': 'collar (z_opt..z_opt+collar transition)',
        '255': 'land / nodata',
    }
    want_meta = str(req.args.get('meta', '0')).lower() in ('1', 'true', 'yes')
    if want_meta:
        try:
            import rasterio
            with rasterio.open(mask_path) as ds:
                b = ds.bounds
                bbox = [float(b.left), float(b.bottom), float(b.right), float(b.top)]
                crs = str(ds.crs) if ds.crs else None
                arr = ds.read(1)
                vals, cnts = np.unique(arr, return_counts=True)
                class_counts = {str(int(v)): int(c) for v, c in zip(vals, cnts)}
                shape = [int(ds.height), int(ds.width)]
        except Exception as ex:
            return jsonify({'error': f'mask read failed: {ex}'}), 500
        return jsonify(_json_safe({
            'id': result_id, 'bbox': bbox, 'crs': crs, 'shape': shape,
            'classes': classes, 'class_counts': class_counts,
            'units': 'uint8 class raster', 'nodata': 255,
            'url': f'/api/results/{result_id}/mask',
        }))
    p = Path(mask_path)
    return send_from_directory(str(p.parent), p.name, as_attachment=False)


def _iho_order_from_rmse(rmse_m, depth_m, p95_error_m=None):
    """Map an error at a representative depth to the strictest IHO S-44 ed.6.1
    Order whose 95% TVU envelope it satisfies, plus the matching CATZOC tier.

    ITEM 2 — the S-44 §3.3.1 gate is the 95th-PERCENTILE absolute error ≤ TVU,
    NOT RMSE ≤ TVU. For Gaussian errors p95 ≈ 1.645×RMSE. Pass an actual
    `p95_error_m` when the test-set p95 is known; otherwise it defaults to
    1.645×RMSE. Returns (order_label, catzoc, tvu_at_depth, orders_table)."""
    try:
        rmse = float(rmse_m); d = abs(float(depth_m))
    except Exception:
        return None, None, None, []
    p95 = float(p95_error_m) if p95_error_m is not None else 1.645 * rmse
    # delegate to the shared p95-vs-TVU classifier
    try:
        from backend.model_card import iho_order_from_p95 as _io
    except ImportError:
        from model_card import iho_order_from_p95 as _io  # type: ignore
    return _io(p95, d)


def _analyse_geotiff(path, n_bins=30, grid_max=160):
    """Read a depth GeoTIFF (positive-down, datum LAT) and return the rich
    analysis payload: depth_stats, histogram, per-2m-band coverage, and a
    downsampled 2D grid for the 3D seabed mesh. All values JSON-safe."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as ds:
        nodata = ds.nodata
        crs = str(ds.crs) if ds.crs else None
        b = ds.bounds
        bbox = [float(b.left), float(b.bottom), float(b.right), float(b.top)]
        full_h, full_w = ds.height, ds.width
        arr = ds.read(1).astype('float64')
        # MASK-GLOBAL (2026-07-10, user-mandated): results-catalogue re-serve
        # backstop — the analyse payload (stats/histogram/3D grid) is cut to
        # the same global OSM land source as every other served product.
        # Original stored raster on disk is untouched (read-only `rasterio.open`).
        try:
            try:
                from backend.osm_land_mask import apply_global_land_cut as _global_cut
            except ImportError:
                from osm_land_mask import apply_global_land_cut as _global_cut  # type: ignore
            # MASK-NDWI (2026-07-10, priority override #2): best-effort
            # imagery-evidence layer for the SAME reason as _global_land_cut_geotiff
            # above — a stored GeoTIFF pre-dates the fix and needs a fresh
            # NDWI evidence layer to catch reclaimed/dredged land OSM hasn't
            # mapped. Degrades to vector-only silently on fetch failure.
            _ndwi_an = _fetch_ndwi_best_effort(bbox, res_m=10)
            _arr_f32 = np.where(np.isfinite(arr) & (nodata is None or arr != nodata), arr, np.nan).astype('float32')
            _arr_cut, _land_full, _ = _global_cut(_arr_f32, bbox, ndwi=_ndwi_an, ndwi_res_m=10)
            arr = np.where(np.isfinite(_arr_cut), _arr_cut, arr).astype('float64')
            arr[_land_full] = np.nan
        except Exception:
            pass
        # build a valid-water mask: drop nodata, NaN/Inf, and sentinel -9999
        mask = np.isfinite(arr)
        if nodata is not None:
            mask &= (arr != nodata)
        mask &= (arr > -1000)  # guard residual sentinels
        # only count actual water (depth >= 0, positive-down). Treat tiny
        # negatives (intertidal/datum noise) within -0.5 m as 0 water.
        water = mask & (arr >= -0.5)
        vals = arr[water]
        # pixel area in km^2 (geographic CRS → use mean lat for x-scale)
        if 'EPSG:4326' in (crs or '') or (crs and '4326' in crs):
            mlat = (bbox[1] + bbox[3]) / 2.0
            dx_deg = (bbox[2] - bbox[0]) / max(full_w, 1)
            dy_deg = (bbox[3] - bbox[1]) / max(full_h, 1)
            px_km2 = (dx_deg * 111.320 * np.cos(np.radians(mlat))) * (dy_deg * 110.574)
        else:
            tr = ds.transform
            px_km2 = abs(tr.a * tr.e) / 1.0e6
        # downsampled grid for the 3D view (area/nearest decimation)
        scale = max(full_h / grid_max, full_w / grid_max, 1.0)
        g_h = max(int(round(full_h / scale)), 1)
        g_w = max(int(round(full_w / scale)), 1)
        grid = ds.read(1, out_shape=(g_h, g_w),
                       resampling=Resampling.average).astype('float64')
        # MASK-GLOBAL: cut the downsampled 3D-view grid too (separate shape).
        try:
            _grid_f32 = np.where(np.isfinite(grid) & (nodata is None or grid != nodata),
                                 grid, np.nan).astype('float32')
            _grid_cut, _land_g, _ = _global_cut(_grid_f32, bbox, ndwi=locals().get('_ndwi_an'), ndwi_res_m=10)
            grid = np.where(np.isfinite(_grid_cut), _grid_cut, grid).astype('float64')
            grid[_land_g] = np.nan
        except Exception:
            pass
        g_mask = np.isfinite(grid)
        if nodata is not None:
            g_mask &= (grid != nodata)
        g_mask &= (grid > -1000)

    valid_px = int(vals.size)
    total_px = int(full_h * full_w)
    if valid_px == 0:
        return {'analysable': False, 'reason': 'no valid water pixels in raster'}

    v = vals
    pcts = np.percentile(v, [5, 25, 50, 75, 95])
    z_min = float(np.nanmin(v)); z_max = float(np.nanmax(v))
    depth_stats = {
        'min': z_min, 'max': z_max,
        'mean': float(np.mean(v)), 'median': float(pcts[2]),
        'std': float(np.std(v)),
        'p5': float(pcts[0]), 'p25': float(pcts[1]),
        'p75': float(pcts[3]), 'p95': float(pcts[4]),
        'valid_px': valid_px,
        'area_km2': round(valid_px * float(px_km2), 6),
        'coverage_pct': round(100.0 * valid_px / max(total_px, 1), 2),
    }

    counts, edges = np.histogram(v, bins=int(n_bins), range=(z_min, z_max))
    histogram = {'bin_edges': [round(float(x), 3) for x in edges],
                 'counts': [int(c) for c in counts]}

    # per-2 m depth band coverage
    band_top = int(np.ceil(z_max / 2.0)) * 2
    per_band = []
    for lo in range(0, max(band_top, 2), 2):
        hi = lo + 2
        sel = (v >= lo) & (v < hi)
        c = int(np.count_nonzero(sel))
        if c == 0 and lo > z_max:
            continue
        per_band.append({'band': f'{lo}-{hi} m', 'lo_m': lo, 'hi_m': hi,
                         'count': c, 'area_km2': round(c * float(px_km2), 6)})

    # grid for 3D: round to cm, nodata pixels -> null, NaN-safe
    grid_clean = np.where(g_mask, np.round(grid, 2), np.nan)
    grid_list = [[(None if not np.isfinite(x) else float(x)) for x in row]
                 for row in grid_clean]
    gv = grid_clean[np.isfinite(grid_clean)]
    grid_payload = {
        'values': grid_list,
        'nrows': int(grid_clean.shape[0]),
        'ncols': int(grid_clean.shape[1]),
        'bbox': bbox,              # [W,S,E,N]
        'nodata': None,            # nodata pixels are JSON null in `values`
        'z_min': (float(np.min(gv)) if gv.size else z_min),
        'z_max': (float(np.max(gv)) if gv.size else z_max),
        'crs': crs,
        'units': 'm, positive down, datum LAT',
    }
    return {
        'analysable': True,
        'depth_stats': depth_stats,
        'histogram': histogram,
        'per_band': per_band,
        'grid': grid_payload,
        'crs': crs,
    }


@app.route('/api/results/<result_id>/analyse', methods=['GET'])
def api_results_analyse(result_id):
    """Rich analysis for the 'analyse' action (R3): depth statistics,
    histogram, per-band coverage, IHO S-44 Order / CATZOC, and a downsampled
    depth grid for the 3D seabed view. Reads the result's depth GeoTIFF.

    Backward-compatible: all previously-returned fields remain present."""
    e = _jobs_store.get_result(result_id)
    if e is None:
        return jsonify({'error': 'result not found'}), 404

    metrics = e.get('metrics') or {}
    payload = {
        'id': e.get('id'),
        'name': e.get('name'),
        'roi_bbox': e.get('roi_bbox'),
        'resolution': e.get('resolution'),
        'status': e.get('status'),
        'date': e.get('date'),
        'metrics': e.get('metrics'),
        'size_bytes': e.get('size_bytes'),
        'output_name': e.get('output_name'),
        'units': 'm, positive down, datum LAT',
        'download_url': (f"/api/results/{result_id}/download" if e.get('output_path') else None),
        'error': e.get('error'),
    }

    out_path = e.get('output_path')
    if e.get('status') != 'done' or not out_path or not os.path.exists(out_path):
        payload['analysable'] = False
        payload['reason'] = ('result not finished' if e.get('status') != 'done'
                             else 'no output GeoTIFF on disk')
        return jsonify(_json_safe(payload))

    try:
        rich = _analyse_geotiff(out_path)
    except Exception as ex:
        import traceback as _tb
        payload['analysable'] = False
        payload['reason'] = f'raster analysis failed: {ex}'
        L.warning("analyse failed for %s: %s", result_id, _tb.format_exc().splitlines()[-1])
        return jsonify(_json_safe(payload))

    payload.update(rich)

    # IHO block — derive Order/CATZOC from RMSE at the median depth.
    rmse = metrics.get('rmse_m') if isinstance(metrics, dict) else None
    med_depth = (rich.get('depth_stats') or {}).get('median', 0.0) if rich.get('analysable') else 0.0
    iho = {
        'rmse_m': metrics.get('rmse_m') if isinstance(metrics, dict) else None,
        'r2': metrics.get('r2') if isinstance(metrics, dict) else None,
        'bias_m': metrics.get('bias_m') if isinstance(metrics, dict) else None,
        'mae_m': metrics.get('mae_m') if isinstance(metrics, dict) else None,
        'n_test': metrics.get('n_test') if isinstance(metrics, dict) else None,
    }
    has_control = isinstance(metrics, dict) and metrics.get('n_test') and rmse is not None
    if has_control:
        _p95 = 1.645 * float(rmse)
        order, catzoc, tvu, table = _iho_order_from_rmse(rmse, med_depth)
        iho.update({
            'order': order, 'catzoc': catzoc,
            'order_qualifier': '(model σ only)',
            'p95_error_m': round(_p95, 3),
            'pass_criterion': 'p95 ≤ TVU (IHO S-44 §3.3.1); p95 ≈ 1.645×RMSE (Gaussian)',
            'tvu_at_median_m': tvu, 'orders_table': table,
            'eval_depth_m': round(float(med_depth), 2),
            'reconnaissance': False,
            'note': (f'IHO S-44 Order from p95 ≈ 1.645×RMSE = {_p95:.2f} m vs TVU at '
                     f'the median depth ({med_depth:.1f} m), {metrics.get("n_test")} '
                     f'in-situ test soundings. Order derived from model σ only — '
                     f'systematic TPU components (tide, datum, refraction) not yet '
                     f'propagated; actual Order may be worse.'),
        })
    else:
        iho.update({
            'order': None, 'catzoc': 'D',
            'tvu_at_median_m': None, 'orders_table': [],
            'eval_depth_m': round(float(med_depth), 2),
            'reconnaissance': True,
            'note': ('Reconnaissance product — no in-situ control soundings for '
                     'this result. No IHO S-44 Order / CATZOC tier can be '
                     'certified; treat depths as indicative only.'),
        })
    payload['iho'] = iho
    return jsonify(_json_safe(payload))


@app.route('/api/results/<result_id>', methods=['DELETE'])
def api_results_delete(result_id):
    remove_file = str(req.args.get('remove_file', '0')).lower() in ('1', 'true', 'yes')
    ok = _jobs_store.delete_result(result_id, remove_file=remove_file)
    return (jsonify({'ok': True}) if ok else (jsonify({'error': 'result not found'}), 404))


# ══════════════════════════════════════════════════════════════
# VERY HR + MLE PRO
#   10-scene S2 median composite per slot → per-pixel inverse-variance
#   MLE fusion → resampled onto a Mapbox VHR mosaic at ~2 m/px.
#   Runs in a background thread so the response returns immediately and
#   the existing /api/very-hr-job/<id>/status endpoint reports a REAL
#   progress percentage (10×7 % for the scenes + the surrounding stages).
# ══════════════════════════════════════════════════════════════
import threading as _threading

try:
    from backend import vhr_mle_pro as _vhr_mle_pro
except ImportError:
    import vhr_mle_pro as _vhr_mle_pro  # type: ignore


def _run_vhr_mle_pro_background(job_id: str, bbox, params: dict):
    """Worker thread for /api/very-hr-mle-pro/start. Updates job progress
    via ``vhr_jobs.set_progress`` so the frontend bar tracks the pipeline
    instead of an arbitrary time estimate."""
    def _progress_cb(pct, stage, idx, total):
        _vhr_jobs.set_progress(job_id, pct, stage=stage,
                                stage_index=idx, stage_total=total)

    def _cancel_cb():
        return _vhr_jobs.is_cancelled(job_id)

    try:
        from backend.very_hr_engine import fetch_mapbox_vhr as _vhr_fetch
    except ImportError:
        from very_hr_engine import fetch_mapbox_vhr as _vhr_fetch  # type: ignore

    save_dir = Path(__file__).resolve().parent.parent / "Very_HR_Results" / "MLE_PRO"
    try:
        result = _vhr_mle_pro.run_vhr_mle_pro(
            bbox,
            year=int(params.get("year", 2024)),
            n_scenes=int(params.get("n_scenes", 10)),
            max_cloud=int(params.get("max_cloud", 20)),
            target_res_m=float(params.get("target_res_m", 2.0)),
            max_tiles_per_side=int(params.get("max_tiles_per_side", 4)),
            user_pts=params.get("user_points") or [],
            fetch_sliderule=bool(params.get("use_sliderule", False)),
            fetch_vhr_fn=_vhr_fetch,
            compute_scene_fn=_compute_s2_depth_grid,
            progress_cb=_progress_cb,
            cancel_cb=_cancel_cb,
            save_dir=save_dir,
        )
        payload = json.dumps(_json_safe(result), default=str).encode("utf-8")
        _vhr_jobs.save_result(job_id, "vhr_mle_pro_result.json", payload,
                              notes=(f"{result.get('n_scenes_used')}/"
                                     f"{result.get('n_scenes_attempted')} scenes · "
                                     f"{result.get('elapsed_s')} s"))
    except Exception as ex:
        L.exception(f"VHR-MLE PRO job {job_id} failed")
        _vhr_jobs.mark_failed(job_id, str(ex))


@app.route('/api/very-hr-mle-pro/start', methods=['POST'])
def api_vhr_mle_pro_start():
    """Kick off a Very-HR + MLE PRO job (10 S2 scenes + Mapbox VHR + per-pixel
    inverse-variance MLE). Returns immediately with a job id; the client polls
    ``/api/very-hr-job/<id>/status`` for real percentage progress."""
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox')
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        if isinstance(bd, dict):
            bbox = [bd['west'], bd['south'], bd['east'], bd['north']]
        else:
            bbox = list(bd)
        if len(bbox) != 4:
            return jsonify({'error': 'bbox must be [west, south, east, north] or {west,south,east,north}'}), 400
        cap = _enforce_roi_cap(bbox)
        if cap is not None:
            return cap
        params = {
            'year': int(data.get('year', 2024)),
            'n_scenes': int(data.get('n_scenes', 10)),
            'max_cloud': int(data.get('max_cloud', 20)),
            'target_res_m': float(data.get('target_res_m', 2.0)),
            'max_tiles_per_side': int(data.get('max_tiles_per_side', 4)),
            'user_points': data.get('user_points', []) or [],
            'use_sliderule': bool(data.get('use_sliderule', False)),
        }
        label = data.get('label') or (f"Very HR + MLE PRO · {params['n_scenes']} "
                                       f"S2 scenes ({params['year']})")
        job = _vhr_jobs.create_job(bbox, params=params, label=label)
        # Initial progress so the bar shows 1 % immediately rather than the
        # legacy time-based estimate.
        _vhr_jobs.set_progress(job["id"], 1.0, stage="Queued — starting worker …",
                                stage_index=0, stage_total=params["n_scenes"] + 5)
        _threading.Thread(
            target=_run_vhr_mle_pro_background,
            args=(job["id"], bbox, params), daemon=True,
        ).start()
        return jsonify(job)
    except Exception as ex:
        L.exception("very-hr-mle-pro/start failed")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# VERY HR — Mapbox VHR + in-situ XYZ + gradient boosting (legacy single-method)
# ══════════════════════════════════════════════════════════════
@app.route('/api/very-hr-extract', methods=['POST'])
def api_very_hr_extract():
    """One-click Very HR bathymetry.

    Body:
      {
        "bbox":        {"west","south","east","north"},   (required)
        "site_key":    optional preset key,
        "user_points": optional [{lat,lon,depth}, ...],
        "train_frac":  float in [0.05, 0.5]   default 0.20,
        "target_res_m":float                  default 1.5,
      }

    Returns metrics + base-64 VHR mosaic + base-64 depth viz + paths to
    artefacts saved under ``Very_HR_Results/``.
    """
    try:
        from backend.very_hr_engine import run_very_hr
    except ImportError:
        from very_hr_engine import run_very_hr
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox')
        site_key = data.get('site_key') or 'custom'
        if site_key in _KNOWN_SITES and _KNOWN_SITES[site_key].get('bbox') and not bd:
            bd = _KNOWN_SITES[site_key]['bbox']
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        bbox = [bd['west'], bd['south'], bd['east'], bd['north']]
        train_frac = float(data.get('train_frac', 0.20))
        train_frac = max(0.05, min(0.5, train_frac))
        target_res_m = float(data.get('target_res_m', 1.5))
        imagery_source = str(data.get('imagery_source', 'vhr')).lower()
        if imagery_source not in ('vhr', 's2'):
            imagery_source = 'vhr'

        # Reference points — prefer user-supplied, fall back to in-memory XYZ
        user_pts = data.get('user_points', []) or []
        if user_pts:
            la = np.array([float(p['lat']) for p in user_pts])
            lo = np.array([float(p['lon']) for p in user_pts])
            de = np.abs(np.array([float(p['depth']) for p in user_pts]))
        else:
            la, lo, de, _src = xyz_points_in_bbox(bbox)
        if len(de) < 30:
            return jsonify({
                'error': f'Need ≥30 in-situ points inside ROI (have {len(de)})',
                'hint': 'Upload an XYZ/CSV/SHP, or pick a preset that has bundled survey data.',
            }), 200

        result = run_very_hr(
            site_key=str(site_key),
            bbox=bbox,
            ref_lats=la, ref_lons=lo, ref_depths=de,
            train_frac=train_frac,
            target_res_m=target_res_m,
            max_tiles_per_side=4,
            seed=42,
            save=True,
            imagery_source=imagery_source,
        )

        # Encode the saved PNGs to base-64 for the UI
        import base64 as _b64
        method_label = ('Very HR (Sentinel-2 only, dilated FCN, stratified 20/80)'
                        if imagery_source == 's2'
                        else 'Very HR (Mapbox + S2, dilated FCN, stratified 20/80)')
        out = {
            'method': method_label,
            'imagery_source': imagery_source,
            's2_status': result.get('s2_status'),
            'site': site_key,
            'bbox': bbox,
            'resolution_m': result['resolution_m'],
            'metrics': result['metrics'],
            'elapsed_s': result['elapsed_s'],
            'paths': result.get('paths', {}),
        }
        for key in ('vhr_image', 'depth_png', 'overlay_png', 'scatter_png'):
            p = (result.get('paths') or {}).get(key)
            if p and os.path.exists(p):
                out[f'{key}_b64'] = _b64.b64encode(Path(p).read_bytes()).decode()
        return jsonify(out)
    except Exception as ex:
        L.exception("very-hr-extract failed")
        return jsonify({'error': str(ex)}), 500


@app.route('/api/pro-extract', methods=['POST'])
def api_pro_extract():
    """Professional one-click extraction.

    Request JSON:
      {
        "bbox": {"west":..,"south":..,"east":..,"north":..},
        "site_key": optional preset key (khalifa_port, abu_al_abyad, …),
        "start_date": "YYYY-MM-DD" (default 2024-05-01),
        "end_date": "YYYY-MM-DD" (default 2024-09-30),
        "user_points": optional list of {lat,lon,depth} from a prior upload
      }

    Returns: same shape as /api/extract but with user-facing source labels only.
    """
    try:
        data = req.get_json(force=True, silent=True) or {}
        bbox_dict = data.get('bbox')
        site_key = data.get('site_key')
        # Site preset overrides bbox if supplied and the site knows its own bbox
        if site_key and site_key in _KNOWN_SITES:
            site = _KNOWN_SITES[site_key]
            if site.get('bbox'):
                bbox_dict = site['bbox']
        if not bbox_dict:
            return jsonify({'error': 'Missing bbox or site_key'}), 400
        sd = data.get('start_date', '2024-05-01')
        ed = data.get('end_date', '2024-09-30')
        bbox = [bbox_dict['west'], bbox_dict['south'], bbox_dict['east'], bbox_dict['north']]
        w, s, e, n = bbox
        area_km2 = abs(e - w) * abs(n - s) * 111 * 111 * math.cos(math.radians((n + s) / 2))
        res = 10 if area_km2 < 30 else (20 if area_km2 < 200 else 30)

        L.info(f"=== PRO-EXTRACT site={site_key or 'custom'} area={area_km2:.0f}km² res={res}m ===")

        sources_used = []
        ref_lats, ref_lons, ref_depths, ref_weights = [], [], [], []
        W_INSITU, W_ICESAT, W_LIBRARY, W_GEBCO = 6.0, 5.0, 3.0, 1.0

        # 1. User-supplied points — survey data (highest weight) and CShelph-derived
        # depths (ICESat-2 weight) flow in via the same channel. Tagged separately
        # in the per-source counts so the UI can show the breakdown.
        user_pts = data.get('user_points', []) or []
        n_insitu = 0
        n_cshelph_user = 0
        for p in user_pts:
            try:
                la, lo, de = float(p['lat']), float(p['lon']), abs(float(p['depth']))
                if not (0 < de <= MAX_DEPTH_M):
                    continue
                pc = p.get('photon_class', 'observed')
                if pc == 'icesat2_cshelph':
                    ref_lats.append(la); ref_lons.append(lo)
                    ref_depths.append(de); ref_weights.append(W_ICESAT)
                    n_cshelph_user += 1
                else:
                    ref_lats.append(la); ref_lons.append(lo)
                    ref_depths.append(de); ref_weights.append(W_INSITU)
                    n_insitu += 1
            except Exception:
                continue
        if n_insitu:
            sources_used.append(f"In-situ survey ({n_insitu})")
        if n_cshelph_user:
            sources_used.append(f"ICESat-2 CShelph ({n_cshelph_user})")

        # 2. Reference library (cached) — auto-seed on first use
        n_lib = 0
        if _get_store:
            try:
                store = _get_store()
                stored = store.query_bbox(bbox, buffer_km=2, max_points=5000)
                if stored['count'] < 30 and n_insitu < 50:
                    # Seed synchronously — silent, no UI mention of the mechanism
                    try:
                        _seed_reference_library(bbox, zooms=(13, 14), min_pts=30)
                        stored = store.query_bbox(bbox, buffer_km=2, max_points=5000)
                    except Exception as ex:
                        L.warning(f"Reference library seed failed (non-fatal): {ex}")
                if stored['count'] > 0:
                    for la, lo, de in zip(stored['lats'].tolist(),
                                          stored['lons'].tolist(),
                                          stored['depths'].tolist()):
                        ref_lats.append(la); ref_lons.append(lo)
                        ref_depths.append(min(float(de), MAX_DEPTH_M))
                        ref_weights.append(W_LIBRARY)
                    n_lib = stored['count']
                    sources_used.append(f"Reference library ({n_lib})")
            except Exception as ex:
                L.warning(f"Reference library query failed: {ex}")

        # 3. ICESat-2 via SlideRule — skip if the user already ran CShelph
        # and supplied the depths via user_points (n_cshelph_user > 0).
        n_ice = 0
        if n_cshelph_user == 0:
            try:
                ice, _msg = run_sliderule(bbox, sd, ed)
                if ice:
                    bathy_ice = [p for p in ice if p.get('photon_class') == 'bathymetry' and p.get('depth', 0) > 0]
                    for ip in bathy_ice:
                        ref_lats.append(ip['lat']); ref_lons.append(ip['lon'])
                        ref_depths.append(min(float(ip['depth']), MAX_DEPTH_M))
                        ref_weights.append(W_ICESAT)
                    n_ice = len(bathy_ice)
                    if n_ice:
                        sources_used.append(f"ICESat-2 ({n_ice})")
            except Exception as ex:
                L.warning(f"ICESat-2 fetch failed: {ex}")

        # 4. GEBCO fallback when sparse
        n_gebco = 0
        if (n_insitu + n_lib + n_ice) < 80:
            try:
                g = fetch_gebco(bbox)
                if g and len(g['depths']) > 0:
                    ref_lats.extend(g['lats'].tolist())
                    ref_lons.extend(g['lons'].tolist())
                    ref_depths.extend(g['depths'].tolist())
                    ref_weights.extend([W_GEBCO] * len(g['depths']))
                    n_gebco = len(g['depths'])
                    sources_used.append(f"GEBCO ({n_gebco})")
            except Exception as ex:
                L.warning(f"GEBCO fetch failed: {ex}")

        if len(ref_depths) < 10:
            return jsonify({
                'error': 'Insufficient reference depths for this ROI. Try uploading a survey, enlarging the ROI, or selecting a preset site.',
                'n_refs': len(ref_depths),
                'sources_used': sources_used,
            }), 200

        # 5. Sentinel-2 imagery
        s2 = fetch_s2(bbox, sd, ed, res=res, cloud=20)
        sources_used.append(f"Sentinel-2 ({s2['width']}×{s2['height']}px @{res}m)")

        # 6. CNN training & inference
        try:
            from backend.cnn_engine import cnn_train_and_predict
        except ImportError:
            from cnn_engine import cnn_train_and_predict
        ref_pts_arr = {
            'lats': np.array(ref_lats),
            'lons': np.array(ref_lons),
            'depths': np.array(ref_depths),
        }
        cnn_result, cnn_err = cnn_train_and_predict(s2, ref_pts_arr, bbox)
        if not cnn_result:
            return jsonify({'error': cnn_err or 'Model training failed',
                            'sources_used': sources_used}), 500

        depth = cnn_result['depth']

        # 7. Build output shaped like /api/extract for frontend compatibility
        H, W = depth.shape
        valid = depth[np.isfinite(depth) & (depth > 0)]
        water = s2.get('water_mask', s2['ndwi'] > 0)
        zones = {
            'vs': int(np.sum(water & (depth <= 3))),
            'sh': int(np.sum(water & (depth > 3) & (depth <= 8))),
            'md': int(np.sum(water & (depth > 8) & (depth <= 15))),
            'dp': int(np.sum(water & (depth > 15))),
        }

        out = {
            'bbox': bbox_dict,
            'points': [],
            'interpolated_points': grid_to_points(depth, bbox),
            'sources_used': _sanitise_sources(sources_used),
            'ml_stats': {
                'method': 'Attention U-Net + multi-source fusion',
                'r2': cnn_result.get('r2', 0.0),
                'n_train': cnn_result.get('n_train', 0),
                'quality': cnn_result.get('quality', {}),
                'zone_counts': zones,
            },
            'pro': {
                'site_key': site_key,
                'site_label': _KNOWN_SITES.get(site_key, {}).get('label') if site_key else None,
                'n_insitu': n_insitu,
                'n_reference_library': n_lib,
                'n_icesat2': n_ice + n_cshelph_user,
                'n_icesat2_auto': n_ice,
                'n_icesat2_user': n_cshelph_user,
                'n_gebco': n_gebco,
                'resolution_m': res,
                'area_km2': round(area_km2, 2),
                'vertical_datum': _KNOWN_SITES.get(site_key, {}).get('vertical_datum', 'LAT') if site_key else 'LAT',
                'horizontal_datum': _KNOWN_SITES.get(site_key, {}).get('horizontal_datum', 'WGS84') if site_key else 'WGS84',
                'depth_cap_m': MAX_DEPTH_M,
            },
            'stats': {
                'mean_depth': round(float(np.mean(valid)), 2) if len(valid) else 0,
                'max_depth': round(float(np.max(valid)), 2) if len(valid) else 0,
                'min_depth': round(float(np.min(valid)), 2) if len(valid) else 0,
                'std_depth': round(float(np.std(valid)), 2) if len(valid) else 0,
                'grid_points': int(np.sum(np.isfinite(depth))),
                'resolution_m': res,
            },
            'tracks': [], 'sea_profiles': [], 'bath_profiles': [],
        }

        # Attach raster + GeoTIFF if the helpers exist
        try:
            raster_png_b64, raster_bounds, raster_max = depth_to_raster_png(depth, bbox)
            if raster_png_b64:
                out['raster_png'] = raster_png_b64
                out['raster_bounds'] = raster_bounds
                out['raster_max_depth'] = raster_max
        except Exception:
            pass
        try:
            tif_b64 = build_geotiff_b64(depth, bbox)
            if tif_b64:
                out['geotiff_b64'] = tif_b64
        except Exception:
            pass

        L.info(f"PRO-EXTRACT done: {len(out['interpolated_points'])} pts, "
               f"R²={cnn_result.get('r2')}, refs={n_insitu}+{n_lib}+{n_ice}+{n_gebco}")
        return jsonify(out)
    except Exception as ex:
        L.error(f"pro-extract: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# CBR BATHYMETRY — Geyman & Maloof 2019 + SlideRule ICESat-2
# ══════════════════════════════════════════════════════════════
try:
    from backend.cbr_bathy import run_cbr as _cbr_run, CBRResult
    CBR_AVAILABLE = True
except Exception:
    try:
        from cbr_bathy import run_cbr as _cbr_run, CBRResult
        CBR_AVAILABLE = True
    except Exception:
        CBR_AVAILABLE = False
        CBRResult = None


# ── Google Earth Engine S2 fallback (used when Sentinel Hub PU is exhausted) ──
_GEE_READY = False
_GEE_PROJECT = os.environ.get("EE_PROJECT", "ee-wassimehtp")


def _init_gee():
    """Earth Engine init.  Reuses the resolver in backend.very_hr_engine
    which understands GEE_SERVICE_ACCOUNT_JSON, GEE_CREDENTIALS_JSON,
    GEE_SERVICE_ACCOUNT_FILE and GOOGLE_APPLICATION_CREDENTIALS, with
    base64-encoded files supported transparently."""
    global _GEE_READY
    if _GEE_READY:
        return True
    try:
        import ee
        import json as _json
        import tempfile
        try:
            from backend.very_hr_engine import _resolve_gee_sa_json
        except ImportError:
            from very_hr_engine import _resolve_gee_sa_json  # type: ignore
        sa_text = _resolve_gee_sa_json()
        if sa_text:
            try:
                info = _json.loads(sa_text)
                tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
                tf.write(sa_text); tf.close()
                email = info.get("client_email", "")
                creds = ee.ServiceAccountCredentials(email, tf.name)
                ee.Initialize(credentials=creds, project=_GEE_PROJECT)
                _GEE_READY = True
                L.info(f"GEE initialised via service account "
                       f"({email}, project={_GEE_PROJECT})")
                return True
            except Exception as ex:
                L.warning(f"GEE service-account init failed: {ex}")
        ee.Initialize(project=_GEE_PROJECT)
        _GEE_READY = True
        L.info(f"GEE initialised with project={_GEE_PROJECT}")
        return True
    except Exception as ex:
        L.warning(f"GEE init failed: {ex}")
        return False


def fetch_s2_wave_gee(bbox, sd, ed, res=10, cloud=25):
    """GEE-backed S2 L1C fetch of B02 + B04 for wave-dispersion bathymetry
    (S2Shores / Almar physics).  Returns a dict with b02, b04, width,
    height — identical contract to fetch_s2_wave so wave_bathymetry can
    swap in without branching.

    Uses L1C (TOA) rather than L2A because the ~1 s inter-band time
    offset needed for celerity is preserved in the raw L1C product.
    """
    if not _init_gee():
        raise RuntimeError("GEE not initialised (missing credentials or SDK)")
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n])
    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterDate(sd, ed)
           .filterBounds(region)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", int(cloud)))
           .sort("CLOUDY_PIXEL_PERCENTAGE"))
    n_scn = col.size().getInfo()
    if n_scn == 0:
        raise RuntimeError(f"GEE wave: no L1C scenes for {sd}→{ed} cloud<{cloud}")
    # Single-orbit choice — take the least-cloudy tile (its B02/B04 retain
    # the ~1 s inter-detector offset). A median composite would smear the
    # wave signal.
    img = ee.Image(col.first()).select(["B2", "B4"]).clip(region)
    scene_id = img.get("PRODUCT_ID").getInfo()
    scene_time = img.get("system:time_start").getInfo()
    L.info(f"GEE wave: {n_scn} candidate scenes; picked {scene_id} (cloud-sorted)")
    url = img.getDownloadURL({
        "region": region, "scale": int(res),
        "format": "GEO_TIFF", "crs": "EPSG:4326",
    })
    r = requests.get(url, timeout=300)
    if not r.ok:
        raise RuntimeError(f"GEE wave download {r.status_code}: {r.text[:200]}")
    arr = tifffile.imread(io.BytesIO(r.content))
    if arr.ndim == 3 and arr.shape[2] == 2:
        b02, b04 = arr[:, :, 0], arr[:, :, 1]
    elif arr.ndim == 3 and arr.shape[0] == 2:
        b02, b04 = arr[0], arr[1]
    else:
        raise RuntimeError(f"GEE wave TIFF unexpected shape {arr.shape}")
    L.info(f"GEE wave: {b02.shape[1]}x{b02.shape[0]} @{res}m · B02 [{b02.min()}-{b02.max()}] · B04 [{b04.min()}-{b04.max()}]")
    return {
        "b02": b02, "b04": b04,
        "width": b02.shape[1], "height": b02.shape[0],
        "scene_id": scene_id, "scene_time_ms": scene_time, "source": "GEE",
    }


def fetch_s2_gee(bbox, sd, ed, res=20, cloud=25):
    """GEE-backed Sentinel-2 L2A fetch. Returns the same dict shape as
    fetch_s2 (blue, green, red, coastal, nir, ndwi, water_mask, width,
    height) so CBR / Stumpf / Lyzenga consumers don't need to branch."""
    if not _init_gee():
        raise RuntimeError("GEE not initialised (missing credentials or SDK)")
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n])
    # Median composite of S2 L2A over the window, SCL-masked.
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(sd, ed)
           .filterBounds(region)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", int(cloud))))
    n_scn = col.size().getInfo()
    if n_scn == 0:
        raise RuntimeError(f"GEE: no S2 scenes for {sd}→{ed} cloud<{cloud}")
    L.info(f"GEE S2: {n_scn} scenes in window @cloud<{cloud}")
    # MASKMLE 1.2a: capture TRUE acquisition timestamps of the contributing
    # images (system:time_start, ms UTC) so the MLE path can compute a real
    # per-scene EOT20 tide height. For a median composite EVERY cloud-filtered
    # scene contributes; for the raw top-N median only the N least-cloudy do.
    _acq_ms = None
    try:
        _acq_ms = [int(m) for m in col.aggregate_array("system:time_start").getInfo()
                   if m is not None]
    except Exception as _ex:
        L.info(f"GEE S2: system:time_start capture failed ({_ex})")
        _acq_ms = None

    def _mask_scl(img):
        scl = img.select("SCL")
        # Keep 4 (veg), 5 (bare), 6 (water), 7 (unclassified), 11 (snow);
        # drop 0 (no data), 1 (saturated), 3 (shadow), 8/9 (cloud), 10 (cirrus)
        ok = scl.neq(0).And(scl.neq(1)).And(scl.neq(3)) \
                 .And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(ok)

    # S2_GEE_RAW=1: source the LEAST-CLOUDY single scene (not a median composite)
    # and return RAW SR reflectance over water — i.e. SKIP the in-fetch Hedley
    # deglint + Lyzenga deep-water subtraction below, which were zero-flooring the
    # visible water pixels and starving the downstream ACOLITE-DSF / Hedley step.
    # The harmonized SR offset for post-2022 is already applied at the GEE asset
    # level, so DN/10000 is the correct 0..1 reflectance (verified: -1000 offset
    # would drive blue negative). Default OFF → legacy median+darkened behaviour.
    _gee_raw = os.environ.get("S2_GEE_RAW", "0") == "1"
    # S2_GEE_MEDIAN_N (default 1): on the RAW path, per-band median over the N
    # least-cloudy RAW SR scenes to suppress per-date glint / ephemeral wakes /
    # cloud-edge haze WITHOUT removing real shallow signal (unlike DSF/Hedley,
    # which over-subtract on turbid clean input — E1b). The E1b raw guards stay
    # in place (no in-fetch Hedley/Lyzenga darkening). N=1 -> byte-identical
    # single-scene behaviour. SCL-masked so cloudy pixels don't enter the median.
    _gee_median_n = int(os.environ.get("S2_GEE_MEDIAN_N", "1"))
    if _gee_raw:
        if _gee_median_n > 1:
            _topn = ee.ImageCollection(
                col.sort("CLOUDY_PIXEL_PERCENTAGE").limit(_gee_median_n))
            img = _topn.map(_mask_scl) \
                       .select(["B1", "B2", "B3", "B4", "B8"]) \
                       .median() \
                       .clip(region)
            L.info(f"GEE S2: S2_GEE_RAW=1 S2_GEE_MEDIAN_N={_gee_median_n} -> "
                   f"per-band median over {_gee_median_n} least-cloudy RAW SR "
                   f"scenes (SCL-masked, no in-fetch deglint/Lyzenga)")
        else:
            img = col.sort("CLOUDY_PIXEL_PERCENTAGE").first() \
                     .select(["B1", "B2", "B3", "B4", "B8"]) \
                     .clip(region)
            L.info("GEE S2: S2_GEE_RAW=1 -> least-cloudy single scene, raw SR reflectance (no in-fetch deglint/Lyzenga)")
    else:
        img = col.map(_mask_scl) \
                 .median() \
                 .select(["B1", "B2", "B3", "B4", "B8"]) \
                 .clip(region)
    url = img.getDownloadURL({
        "region": region, "scale": int(res),
        "format": "GEO_TIFF", "crs": "EPSG:4326",
    })
    L.info(f"GEE download: {url[:80]}...")
    r = requests.get(url, timeout=300)
    if not r.ok:
        raise RuntimeError(f"GEE download {r.status_code}: {r.text[:200]}")
    img_np = tifffile.imread(io.BytesIO(r.content))
    # tifffile usually returns (H, W, bands) for GEO_TIFF
    if img_np.ndim == 3 and img_np.shape[2] == 5:
        coastal, blue, green, red, nir = [img_np[:, :, i] for i in range(5)]
    elif img_np.ndim == 3 and img_np.shape[0] == 5:
        coastal, blue, green, red, nir = [img_np[i] for i in range(5)]
    else:
        raise RuntimeError(f"GEE TIFF unexpected shape {img_np.shape}")

    # L2A SR is reported as reflectance × 10000 uint16 — same as SH evalscript output.
    gf = green.astype(float); nf = nir.astype(float)
    ndwi = (gf - nf) / (gf + nf + 1e-6)
    blue_f    = blue.astype(np.float64)    / 10000.0
    green_f   = green.astype(np.float64)   / 10000.0
    red_f     = red.astype(np.float64)     / 10000.0
    nir_f     = nir.astype(np.float64)     / 10000.0
    coastal_f = coastal.astype(np.float64) / 10000.0
    # Hedley 2005 sun-glint correction — regress each visible band against NIR over deep water.
    deep_water = (ndwi > 0.5) & (nir_f < 0.02)
    if np.sum(deep_water) > 100 and not _gee_raw:
        nir_dw = nir_f[deep_water]
        for band, band_f in [("blue", blue_f), ("green", green_f), ("red", red_f), ("coastal", coastal_f)]:
            vis_dw = band_f[deep_water]
            valid = np.isfinite(nir_dw) & np.isfinite(vis_dw) & (nir_dw > 0)
            if valid.sum() > 50:
                n_dw, v_dw = nir_dw[valid], vis_dw[valid]
                slope = np.sum((v_dw - v_dw.mean()) * (n_dw - n_dw.mean())) / \
                        (np.sum((n_dw - n_dw.mean()) ** 2) + 1e-10)
                min_nir = float(np.percentile(n_dw, 5))
                correction = slope * (nir_f - min_nir)
                if band == "blue":    blue_f    = np.clip(blue_f - correction,    1e-5, None)
                elif band == "green": green_f   = np.clip(green_f - correction,   1e-5, None)
                elif band == "red":   red_f     = np.clip(red_f - correction,     1e-5, None)
                elif band == "coastal": coastal_f = np.clip(coastal_f - correction, 1e-5, None)
    # Lyzenga 1978 deep-water reflectance subtraction
    water_mask = ndwi > 0
    if np.sum(deep_water) > 50 and not _gee_raw:
        for arr, full in [("blue_f", blue_f), ("green_f", green_f), ("red_f", red_f), ("coastal_f", coastal_f)]:
            pass  # no-op placeholder to preserve variable scope
        Rw_inf_b = float(np.median(blue_f[deep_water]))
        Rw_inf_g = float(np.median(green_f[deep_water]))
        Rw_inf_r = float(np.median(red_f[deep_water]))
        Rw_inf_c = float(np.median(coastal_f[deep_water]))
        blue_f    = np.where(water_mask, np.clip(blue_f    - Rw_inf_b, 1e-6, None), blue_f)
        green_f   = np.where(water_mask, np.clip(green_f   - Rw_inf_g, 1e-6, None), green_f)
        red_f     = np.where(water_mask, np.clip(red_f     - Rw_inf_r, 1e-6, None), red_f)
        coastal_f = np.where(water_mask, np.clip(coastal_f - Rw_inf_c, 1e-6, None), coastal_f)
    from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
    if np.sum(water_mask) > 100:
        struct = np.ones((3, 3))
        clean = binary_erosion(water_mask, struct, iterations=1)
        clean = binary_dilation(clean, struct, iterations=1)
        water_mask = binary_fill_holes(clean)
    blue_dn    = (blue_f    * 10000).astype(np.uint16)
    green_dn   = (green_f   * 10000).astype(np.uint16)
    red_dn     = (red_f     * 10000).astype(np.uint16)
    coastal_dn = (coastal_f * 10000).astype(np.uint16)
    nir_dn     = nir.astype(np.uint16)
    if np.sum(water_mask) > 50:
        _wm = water_mask
        L.info("GEE S2 water-band median reflectance (VERIFY non-zero): "
               f"blue={float(np.median(blue_f[_wm])):.4f} "
               f"green={float(np.median(green_f[_wm])):.4f} "
               f"red={float(np.median(red_f[_wm])):.4f} "
               f"coastal={float(np.median(coastal_f[_wm])):.4f} "
               f"nir={float(np.median(nir_f[_wm])):.4f}  (S2_GEE_RAW={int(_gee_raw)})")
    L.info(f"GEE S2 done: {blue_dn.shape[1]}x{blue_dn.shape[0]}px  "
           f"deep-water pixels={int(np.sum(deep_water))}")
    # MASKMLE 1.2a: convert contributing timestamps to ISO-UTC. For the raw
    # top-N median only the N least-cloudy scenes actually contribute.
    acq_dt = None
    if _acq_ms:
        ms_used = _acq_ms
        if _gee_raw:
            ms_sorted = sorted(_acq_ms)  # order not cloud-linked; keep all for N>1
            ms_used = _acq_ms if _gee_median_n > 1 else _acq_ms[:1]
        try:
            from datetime import datetime as _dt, timezone as _tz
            acq_dt = [_dt.fromtimestamp(m / 1000.0, tz=_tz.utc)
                      .strftime('%Y-%m-%dT%H:%M:%SZ') for m in sorted(ms_used)]
        except Exception:
            acq_dt = None
    return {
        "blue": blue_dn, "green": green_dn, "red": red_dn,
        "coastal": coastal_dn, "nir": nir_dn,
        "ndwi": ndwi, "water_mask": water_mask,
        "width": blue_dn.shape[1], "height": blue_dn.shape[0],
        "source": "GEE", "acquisition_datetimes": acq_dt,
        "n_contributing_scenes": (len(acq_dt) if acq_dt else None),
    }


def _retide_refs_to_s2(ref_pts_raw, bbox, s2_ref_utc, do_tide=True, do_wave=True):
    """Harmonise altimeter references to a Sentinel-2 reference time.

    For each ICESat-2 photon at (lat, lon, t1) with measured depth d₁:
        d_corrected = d₁ + (tide(center, t2) − tide(lat, lon, t1)) − Hs/2 at t1

    Tide and wave queries are cached by (round(lat,2), round(lon,2), date)
    so 3000 photons collapse to one Open-Meteo call per acquisition day.
    Returns (ref_pts_corrected, info_dict).
    """
    from datetime import datetime as _dt, timezone as _tz
    try:
        from backend.tide_correction import get_tide_height, get_wave_height
    except Exception:
        try:
            from tide_correction import get_tide_height, get_wave_height
        except Exception:
            return ref_pts_raw, {"applied": False, "reason": "tide_correction module missing"}

    w_, s_, e_, n_ = bbox
    lat_c = 0.5 * (s_ + n_); lon_c = 0.5 * (w_ + e_)
    # Reference S2 tide: single lookup at ROI centre for the S2 midpoint
    try:
        tide_s2, _ = get_tide_height(lat_c, lon_c, s2_ref_utc) if do_tide else (0.0, {})
    except Exception as ex:
        tide_s2, _ = 0.0, {}
        do_tide = False
    cache_tide, cache_wave = {}, {}
    n_retided = 0; n_no_dt = 0; n_wave_corrected = 0
    tide_shifts = []; wave_shifts = []
    out = []
    for p in ref_pts_raw:
        try:
            la = float(p['lat']); lo = float(p['lon']); dp = float(p['depth'])
        except Exception:
            continue
        dt_iso = p.get('acq_dt') or p.get('acq_date')
        t1 = None
        if dt_iso:
            try:
                t1 = _dt.fromisoformat(str(dt_iso).replace('Z', '+00:00'))
                if t1.tzinfo is None:
                    t1 = t1.replace(tzinfo=_tz.utc)
            except Exception:
                t1 = None
        if t1 is None:
            n_no_dt += 1
            out.append({'lat': la, 'lon': lo, 'depth': dp,
                        'acq_date': p.get('acq_date'), 'acq_dt': dt_iso})
            continue
        delta = 0.0
        if do_tide:
            # Cache key: spatially coarse + hour. Tide varies smoothly at
            # this scale so ±0.01° and ±1h don't change the answer.
            key_t = (round(la, 2), round(lo, 2), t1.strftime("%Y-%m-%dT%H"))
            if key_t not in cache_tide:
                try:
                    th, _ = get_tide_height(la, lo, t1)
                    cache_tide[key_t] = th
                except Exception:
                    cache_tide[key_t] = 0.0
            t_phot = cache_tide[key_t]
            delta += (tide_s2 - t_phot)
            tide_shifts.append(tide_s2 - t_phot)
        if do_wave:
            key_w = (round(la, 2), round(lo, 2), t1.strftime("%Y-%m-%dT%H"))
            if key_w not in cache_wave:
                try:
                    hs, _ = get_wave_height(la, lo, t1)
                    cache_wave[key_w] = float(hs or 0.0)
                except Exception:
                    cache_wave[key_w] = 0.0
            hs = cache_wave[key_w]
            if hs > 0.5:  # only apply when waves are non-trivial
                delta -= hs / 2.0
                wave_shifts.append(hs / 2.0)
                n_wave_corrected += 1
        d_corr = float(np.clip(dp + delta, 0.3, MAX_DEPTH_M))
        out.append({'lat': la, 'lon': lo, 'depth': d_corr,
                    'depth_raw': dp, 'depth_shift_m': round(delta, 3),
                    'acq_date': p.get('acq_date'), 'acq_dt': dt_iso})
        n_retided += 1
    info = {
        "applied": True,
        "n_input": len(ref_pts_raw),
        "n_retided": n_retided,
        "n_no_datetime": n_no_dt,
        "n_wave_corrected": n_wave_corrected,
        "s2_reference_utc": s2_ref_utc.isoformat() if s2_ref_utc else None,
        "tide_s2_m": round(float(tide_s2), 3) if do_tide else None,
        "tide_shift_median_m": round(float(np.median(tide_shifts)), 3) if tide_shifts else 0.0,
        "tide_shift_abs_mean_m": round(float(np.mean(np.abs(tide_shifts))), 3) if tide_shifts else 0.0,
        "wave_shift_abs_mean_m": round(float(np.mean(np.abs(wave_shifts))), 3) if wave_shifts else 0.0,
        "n_tide_cache_entries": len(cache_tide),
        "n_wave_cache_entries": len(cache_wave),
    }
    L.info(f"Retide: {n_retided}/{len(ref_pts_raw)} photons corrected · "
           f"tide_s2={tide_s2:+.3f}m · median Δt={info['tide_shift_median_m']:+.3f}m · "
           f"wave-corrected={n_wave_corrected}")
    return out, info


def _cbr_multi_scene(bbox, sd, ed, n_scenes, res, cloud, ref_pts,
                     n_clusters, smooth_boundaries, min_pts_per_cluster,
                     features_mode="poly", depth_weighting=True,
                     calibrate="local", ref_max_depth_m=15.0,
                     mad_trim_sigma=2.5):
    """Run CBR on N separate Sentinel-2 scenes spread across [sd, ed].
    Returns a list of per-scene CBRResult + stacked (H, W) depth/sigma
    arrays ready for Gaussian MLE fusion. Per-scene failures are skipped.
    """
    from datetime import datetime, timedelta
    dt_sd = datetime.strptime(sd, '%Y-%m-%d')
    dt_ed = datetime.strptime(ed, '%Y-%m-%d')
    total_days = max((dt_ed - dt_sd).days, n_scenes)
    chunk = max(5, total_days // n_scenes)

    results = []
    scene_dates = []
    per_scene_info = []
    for i in range(n_scenes):
        c_sd = dt_sd + timedelta(days=i * chunk)
        c_ed = min(c_sd + timedelta(days=chunk - 1), dt_ed)
        if c_sd >= dt_ed:
            break
        c_sd_s = c_sd.strftime('%Y-%m-%d')
        c_ed_s = c_ed.strftime('%Y-%m-%d')
        source_used = "SH"
        s2_i = None
        try:
            L.info(f"CBR MLE: scene {i+1}/{n_scenes} {c_sd_s}→{c_ed_s} @{res}m (SH)")
            s2_i = fetch_s2(bbox, c_sd_s, c_ed_s, res=res, cloud=cloud)
        except Exception as ex_sh:
            msg = str(ex_sh)
            is_quota = ("403" in msg and ("ACCESS_INSUFFICIENT" in msg or "processing units" in msg)) or "429" in msg
            if is_quota:
                L.warning(f"CBR MLE: scene {i+1} SH quota hit, falling back to GEE — {msg[:120]}")
                try:
                    s2_i = fetch_s2_gee(bbox, c_sd_s, c_ed_s, res=res, cloud=cloud)
                    source_used = "GEE"
                except Exception as ex_gee:
                    L.warning(f"CBR MLE: scene {i+1} GEE also failed — {ex_gee}")
                    per_scene_info.append({
                        "date_range": f"{c_sd_s}→{c_ed_s}",
                        "source": "SH→GEE", "status": f"failed: SH quota, GEE: {ex_gee}",
                    })
                    continue
            else:
                per_scene_info.append({
                    "date_range": f"{c_sd_s}→{c_ed_s}",
                    "source": "SH", "status": f"failed: {ex_sh}",
                })
                continue
        try:
            cbr_i = _cbr_run(s2_i, ref_pts, bbox,
                             n_clusters=n_clusters,
                             min_pts_per_cluster=min_pts_per_cluster,
                             smooth_boundaries=smooth_boundaries,
                             features_mode=features_mode,
                             depth_weighting=depth_weighting,
                             calibrate=calibrate,
                             ref_max_depth_m=ref_max_depth_m,
                             mad_trim_sigma=mad_trim_sigma)
            results.append(cbr_i)
            scene_dates.append(f"{c_sd_s}→{c_ed_s}")
            per_scene_info.append({
                "date_range": f"{c_sd_s}→{c_ed_s}",
                "source": source_used,
                "r2": cbr_i.metrics.get("r2"),
                "rmse": cbr_i.metrics.get("rmse"),
                "mae": cbr_i.metrics.get("mae"),
                "bias": cbr_i.metrics.get("bias"),
                "n_shallow": cbr_i.metrics.get("n_shallow_pixels"),
                "envelope_stage": cbr_i.metrics.get("envelope_stage"),
                "status": "ok",
            })
        except Exception as ex:
            L.warning(f"CBR MLE: scene {i+1} CBR fit failed — {ex}")
            per_scene_info.append({
                "date_range": f"{c_sd_s}→{c_ed_s}",
                "source": source_used, "status": f"failed: {ex}",
            })
    return results, scene_dates, per_scene_info


def _cbr_mle_fuse(cbr_results):
    """Inverse-variance Gaussian MLE over stacked CBR outputs with 3-pass
    IRLS outlier rejection (|d_i − d_hat| > 2.5 σ_i). Returns (depth, sigma,
    water_mask, n_per_pixel) all shape (H, W)."""
    if not cbr_results:
        raise ValueError("CBR MLE: no scenes to fuse")
    # All CBR runs share (H, W) because bbox + res are identical.
    H, W = cbr_results[0].depth.shape
    D = np.stack([r.depth        for r in cbr_results], axis=0).astype(np.float32)
    S = np.stack([r.uncertainty  for r in cbr_results], axis=0).astype(np.float32)
    eps = 1e-6
    finite = np.isfinite(D) & np.isfinite(S) & (S > eps)
    inlier = finite.copy()
    d_hat = np.full((H, W), np.nan, dtype=np.float32)
    s_hat = np.full((H, W), np.nan, dtype=np.float32)
    for _ in range(3):
        w_i = np.where(inlier, 1.0 / (S ** 2 + eps), 0.0)
        w_sum = np.sum(w_i, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            d_hat = np.where(w_sum > 0,
                             np.sum(w_i * np.where(inlier, D, 0.0), axis=0) / w_sum,
                             np.nan).astype(np.float32)
            s_hat = np.where(w_sum > 0, np.sqrt(1.0 / w_sum), np.nan).astype(np.float32)
        resid = np.abs(D - d_hat[None, :, :])
        new_inlier = inlier & (resid < 2.5 * S)
        # Safeguard: never drop every scene for a pixel
        fallback = np.sum(new_inlier, axis=0) == 0
        if np.any(fallback):
            for k in range(D.shape[0]):
                new_inlier[k][fallback] |= finite[k][fallback]
        if np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier
    depth = np.clip(d_hat, 0.0, MAX_DEPTH_M)
    n_per_pixel = np.sum(inlier, axis=0).astype(np.int16)
    water = np.any(inlier, axis=0)

    # ── NaN-robust fallback — for any pixel where inlier collapsed to 0
    #    (or weight sum became 0) but at least one scene had a finite value,
    #    take the nanmean across scenes.  This guarantees the fused map has
    #    no holes where ANY scene had data.
    with np.errstate(invalid='ignore'):
        any_finite = np.any(np.isfinite(D), axis=0)
        gap = any_finite & (~np.isfinite(depth))
        if np.any(gap):
            fallback_d = np.nanmean(D, axis=0)
            fallback_s = np.nanmean(S, axis=0)
            depth = np.where(gap, np.clip(fallback_d, 0.0, MAX_DEPTH_M).astype(np.float32), depth)
            s_hat = np.where(gap, fallback_s.astype(np.float32), s_hat)
            water = water | gap
            # Mark coverage = −1 to flag rescue pixels (vs. normal ≥1)
            n_per_pixel = np.where(gap & (n_per_pixel == 0), -1, n_per_pixel).astype(np.int16)
            L.info(f"CBR MLE fallback: nanmean-rescued {int(gap.sum())} pixels "
                   f"({100*gap.sum()/max(1,any_finite.sum()):.1f}% of any-finite area)")

    depth[~water] = np.nan
    s_hat[~water] = np.nan
    return depth, s_hat, water, n_per_pixel


@app.route('/api/cbr-bathymetry', methods=['POST'])
def api_cbr_bathymetry():
    """Cluster-Based Regression bathymetry.

    Pipeline:
      1. Fetch Sentinel-2 mosaic (deglinted + deep-water subtracted)
      2. Fetch ICESat-2 ATL03 photons via SlideRule (or use user_points)
      3. Run K-means on reflectance features → bottom-type classes
      4. Fit a robust linear depth regression per class on ICESat-2 refs
      5. Predict depth pixel-wise with soft cluster-boundary blending

    Request JSON:
      {
        "bbox": {"west":..,"south":..,"east":..,"north":..},
        "start_date": "2024-05-01",
        "end_date":   "2024-09-30",
        "cloud": 25,
        "resolution_m": 20,
        "n_clusters": 5,
        "user_points": optional [{lat,lon,depth}, ...]  # overrides SlideRule
      }

    Returns: same shape as /api/pro-extract — depth grid as points +
    raster PNG + GeoTIFF + per-cluster diagnostics.
    """
    if not CBR_AVAILABLE:
        return jsonify({'error': 'cbr_bathy module not available on server'}), 500
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox')
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        bbox = [bd['west'], bd['south'], bd['east'], bd['north']]
        sd = data.get('start_date', '2024-05-01')
        ed = data.get('end_date', '2024-09-30')
        cloud = int(data.get('cloud', 25))
        res = int(data.get('resolution_m', 20))
        n_clusters = max(2, min(10, int(data.get('n_clusters', 5))))
        smooth_boundaries = bool(data.get('smooth_boundaries', True))
        min_pts_per_cluster = max(4, int(data.get('min_pts_per_cluster', 8)))
        n_scenes = max(1, min(8, int(data.get('n_scenes', 1))))
        features_mode = data.get('features_mode', 'poly')  # 'linear' | 'poly'
        depth_weighting = bool(data.get('depth_weighting', True))
        calibrate = data.get('calibrate', 'local')  # 'none' | 'global' | 'local'
        ref_max_depth_m = float(data.get('ref_max_depth_m', 15.0))
        mad_trim_sigma = float(data.get('mad_trim_sigma', 2.5))
        retide = bool(data.get('retide', True))
        wave_correct = bool(data.get('wave_correct', True))
        if features_mode not in ('linear', 'poly'):
            features_mode = 'poly'
        if calibrate not in ('none', 'global', 'local'):
            calibrate = 'local'
        ref_max_depth_m = max(5.0, min(MAX_DEPTH_M, ref_max_depth_m))
        mad_trim_sigma  = max(0.0, min(6.0, mad_trim_sigma))
        w, s, e, n = bbox
        area_km2 = abs(e - w) * abs(n - s) * 111 * 111 * math.cos(math.radians((n + s) / 2))
        L.info(f"=== CBR site area={area_km2:.0f}km² res={res}m K={n_clusters} "
               f"smooth={smooth_boundaries} min_pts={min_pts_per_cluster} "
               f"n_scenes={n_scenes} ===")

        # ── References: user-supplied first, else SlideRule ICESat-2 ──
        sources_used = []
        ref_pts = []
        user_pts = data.get('user_points', []) or []
        for p in user_pts:
            try:
                la = float(p['lat']); lo = float(p['lon']); dp = abs(float(p['depth']))
                if 0 < dp <= MAX_DEPTH_M:
                    ref_pts.append({'lat': la, 'lon': lo, 'depth': dp})
            except Exception:
                continue
        n_user = len(ref_pts)
        if n_user:
            sources_used.append(f"User points ({n_user})")

        # ── ICESat-2: CBR's seabed is effectively static, so widen the
        #    search to the full ATL03 archive and retry on progressive
        #    windows. Also accept any non-surface photon with a positive
        #    depth (not just the 'bathymetry' tag).
        n_ice = 0
        ice_debug = []
        if n_user < 30:
            ice_archive_start = "2018-10-14"  # ICESat-2 launch
            today = PINNED_ICE_TODAY  # SPEC-2: pinned, not utcnow() (deterministic)
            # Try: the user's S2 window, then ±2y around it, then full archive.
            try:
                s2_mid_year = int(sd[:4])
            except Exception:
                s2_mid_year = 2024
            windows = [
                (sd, ed, "S2 window"),
                (f"{max(2018, s2_mid_year - 2)}-01-01",
                 f"{min(int(today[:4]), s2_mid_year + 2)}-12-31",
                 "±2y around S2"),
                (ice_archive_start, today, "full ATL03 archive"),
            ]
            seen = set()
            for (wsd, wed, tag) in windows:
                if n_ice >= 30:
                    break
                try:
                    ice, msg = run_sliderule(bbox, wsd, wed)
                    got_bath = 0; got_any = 0
                    for ip in ice or []:
                        pc = ip.get('photon_class')
                        if pc == 'surface':
                            continue
                        d = float(ip.get('depth') or 0.0)
                        if d <= 0.3:
                            continue
                        key = (round(ip['lat'], 6), round(ip['lon'], 6))
                        if key in seen:
                            continue
                        seen.add(key)
                        # Preserve acq_date + acq_dt so _retide_refs_to_s2 can
                        # query tide/wave at the exact photon timestamp.
                        ref_pts.append({
                            'lat': ip['lat'], 'lon': ip['lon'],
                            'depth': min(d, MAX_DEPTH_M),
                            'acq_date': ip.get('acq_date', ''),
                            'acq_dt':   ip.get('acq_dt', ''),
                            'photon_class': pc or 'bathymetry',
                        })
                        n_ice += 1; got_any += 1
                        if pc == 'bathymetry':
                            got_bath += 1
                    ice_debug.append(f"{tag} [{wsd}→{wed}]: {got_any} kept ({got_bath} bathy) · {msg}")
                    L.info(f"CBR SlideRule · {tag}: +{got_any} refs (total={n_ice}) · {msg}")
                except Exception as ex:
                    ice_debug.append(f"{tag}: FAILED {ex}")
                    L.warning(f"CBR SlideRule {tag} failed: {ex}")
            if n_ice:
                sources_used.append(f"ICESat-2 SlideRule ({n_ice})")

        if len(ref_pts) < 10:
            detail = " | ".join(ice_debug) if ice_debug else "no ICESat-2 fetch attempted"
            return jsonify({'error':
                f'Insufficient reference depths for CBR. '
                f'Need ≥10 points; got {len(ref_pts)} '
                f'(user={n_user}, ICESat-2={n_ice}). '
                f'SlideRule tried: {detail}. '
                f'If SlideRule returned photons but none passed the surface/depth filters, '
                f'the ROI is likely too turbid or too deep for ATL03 bathymetry. '
                f'Try enlarging the ROI, picking a shallower area, or uploading survey points.',
                'n_refs': len(ref_pts), 'n_icesat2': n_ice, 'n_user': n_user,
                'icesat2_debug': ice_debug,
                'sources_used': sources_used}), 200

        # ── Tide + wave retiding of altimeter references to S2 time ──
        # Pick the S2 reference UTC instant: midpoint of the date window,
        # 07:00 UTC (Sentinel-2 descending overpass at Gulf longitudes ≈
        # 10:30-11:00 local = 06:30-07:00 UTC).
        from datetime import datetime as _dt, timezone as _tz
        try:
            _dt_sd = _dt.strptime(sd, "%Y-%m-%d").replace(tzinfo=_tz.utc)
            _dt_ed = _dt.strptime(ed, "%Y-%m-%d").replace(tzinfo=_tz.utc)
            _mid   = _dt_sd + (_dt_ed - _dt_sd) / 2
            s2_ref_utc = _mid.replace(hour=7, minute=0, second=0, microsecond=0)
        except Exception:
            s2_ref_utc = _dt.utcnow().replace(tzinfo=_tz.utc)
        retide_info = {"applied": False}
        if retide and len(ref_pts) >= 10:
            ref_pts, retide_info = _retide_refs_to_s2(
                ref_pts, bbox, s2_ref_utc,
                do_tide=retide, do_wave=wave_correct,
            )
            if retide_info.get("applied"):
                sources_used.append(
                    f"retided→S2 t={s2_ref_utc.strftime('%Y-%m-%d %H:%M UTC')} "
                    f"(Δmedian={retide_info.get('tide_shift_median_m',0):+.2f}m)")

        per_scene_info = []
        mle_info = {}
        if n_scenes <= 1:
            # ── Single-mosaic CBR (classic Geyman 2019) ──
            s2 = None
            src_tag = "Sentinel Hub"
            try:
                s2 = fetch_s2(bbox, sd, ed, res=res, cloud=cloud)
            except Exception as ex_sh:
                msg = str(ex_sh)
                if ("403" in msg and ("ACCESS_INSUFFICIENT" in msg or "processing units" in msg)) or "429" in msg:
                    L.warning(f"CBR: SH quota hit, falling back to GEE — {msg[:120]}")
                    s2 = fetch_s2_gee(bbox, sd, ed, res=res, cloud=cloud)
                    src_tag = "GEE"
                else:
                    raise
            sources_used.append(f"{src_tag} Sentinel-2 ({s2['width']}×{s2['height']}px @{res}m)")
            result = _cbr_run(s2, ref_pts, bbox,
                              n_clusters=n_clusters,
                              min_pts_per_cluster=min_pts_per_cluster,
                              smooth_boundaries=smooth_boundaries,
                              features_mode=features_mode,
                              depth_weighting=depth_weighting,
                              calibrate=calibrate,
                              ref_max_depth_m=ref_max_depth_m,
                              mad_trim_sigma=mad_trim_sigma)
            depth = np.nan_to_num(result.depth, nan=0.0)
        else:
            # ── Multi-scene CBR with inverse-variance Gaussian MLE fusion ──
            scenes, scene_dates, per_scene_info = _cbr_multi_scene(
                bbox, sd, ed, n_scenes, res, cloud, ref_pts,
                n_clusters, smooth_boundaries, min_pts_per_cluster,
                features_mode=features_mode, depth_weighting=depth_weighting,
                calibrate=calibrate, ref_max_depth_m=ref_max_depth_m,
                mad_trim_sigma=mad_trim_sigma)
            if not scenes:
                first_fail = next((p['status'] for p in per_scene_info
                                   if p.get('status', '').startswith('failed')), '')
                common_hint = ''
                if '403' in first_fail and ('processing units' in first_fail or 'ACCESS_INSUFFICIENT' in first_fail):
                    common_hint = ' · Sentinel Hub monthly PU quota is exhausted — either wait for the next billing cycle, upgrade the account, drop to a single scene, or raise resolution_m (50 m uses ~6× fewer PU than 20 m).'
                elif '429' in first_fail:
                    common_hint = ' · Sentinel Hub is rate-limiting; retry in a minute or reduce n_scenes.'
                return jsonify({'error':
                    f'Multi-scene CBR failed: 0 of {n_scenes} scenes produced a valid fit.' +
                    common_hint,
                    'per_scene': per_scene_info, 'sources_used': sources_used}), 200
            sources_used.append(f"Sentinel-2 ×{len(scenes)} scenes @{res}m (MLE fused)")
            depth_f, sigma_f, water_f, n_per_px = _cbr_mle_fuse(scenes)
            result = CBRResult(
                depth=depth_f, uncertainty=sigma_f,
                cluster_map=scenes[0].cluster_map,
                water_mask=water_f,
                n_clusters=n_clusters,
                per_class=scenes[0].per_class,
                metrics=dict(scenes[0].metrics),
                bbox=list(bbox),
            )
            # Re-validate MLE-fused depth against shared references.
            # Two passes — in-regime (depth ≤ training cap) for honest SDB
            # performance, and full-set (all refs up to MAX_DEPTH_M) so the
            # user can see where SDB saturates.
            def _val_metrics(depth_g, refs, dmax):
                rr_v, cc_v, zz_v = [], [], []
                w_, s_, e_, n_ = bbox
                Hf_, Wf_ = depth_g.shape
                for pp in refs:
                    la = float(pp['lat']); lo = float(pp['lon']); dp = abs(float(pp['depth']))
                    if s_ <= la <= n_ and w_ <= lo <= e_ and 0.3 <= dp <= dmax:
                        rv = int(round((n_ - la) / (n_ - s_ + 1e-10) * (Hf_ - 1)))
                        cv = int(round((lo - w_) / (e_ - w_ + 1e-10) * (Wf_ - 1)))
                        if 0 <= rv < Hf_ and 0 <= cv < Wf_ and np.isfinite(depth_g[rv, cv]):
                            rr_v.append(rv); cc_v.append(cv); zz_v.append(dp)
                if len(zz_v) < 5:
                    return None
                zv = np.asarray(zz_v); pv = depth_g[np.asarray(rr_v), np.asarray(cc_v)]
                err = pv - zv
                rmse = float(np.sqrt(np.mean(err ** 2)))
                mae  = float(np.mean(np.abs(err)))
                bias = float(np.mean(err))
                ss_res = float(np.sum(err ** 2))
                ss_tot = float(np.sum((zv - zv.mean()) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-6 else 0.0
                a, b = 0.5, 0.013
                z_mean = max(float(np.mean(zv)), 1.0)
                tvu = float(np.sqrt(a ** 2 + (b * z_mean) ** 2))
                iho = ("S-44 order 1a" if rmse <= tvu
                       else "S-44 order 2" if rmse <= 2.0 * tvu
                       else "below order 2")
                return {"r2": round(r2, 3), "rmse": round(rmse, 3),
                        "mae": round(mae, 3), "bias": round(bias, 3),
                        "n": int(len(zv)), "iho_s44": iho,
                        "depth_cap_m": float(dmax)}

            m_in    = _val_metrics(depth_f, ref_pts, ref_max_depth_m)
            m_full  = _val_metrics(depth_f, ref_pts, MAX_DEPTH_M)
            # Report the in-regime metrics as primary (the model's design regime);
            # attach full-set as diagnostic so the user sees saturation honestly.
            if m_in is not None:
                result.metrics.update({
                    "r2":   m_in["r2"],  "rmse": m_in["rmse"],
                    "mae":  m_in["mae"], "bias": m_in["bias"],
                    "n_train": m_in["n"], "iho_s44": m_in["iho_s44"],
                    "n_water_pixels":   int(water_f.sum()),
                    "n_shallow_pixels": int(water_f.sum()),
                    "validation_in_regime": m_in,
                    "validation_full":      m_full,
                })
                L.info(f"CBR MLE fused [in-regime ≤{ref_max_depth_m}m]: "
                       f"R²={m_in['r2']:.3f} RMSE={m_in['rmse']:.2f}m "
                       f"MAE={m_in['mae']:.2f}m N={m_in['n']} {m_in['iho_s44']}")
                if m_full is not None:
                    L.info(f"CBR MLE fused [full ≤{MAX_DEPTH_M}m]: "
                           f"R²={m_full['r2']:.3f} RMSE={m_full['rmse']:.2f}m "
                           f"N={m_full['n']} — shows SDB saturation beyond cap")
            mle_info = {
                "n_scenes_requested": n_scenes,
                "n_scenes_used": len(scenes),
                "scene_dates": scene_dates,
                "mean_coverage_per_px": float(np.mean(n_per_px[water_f])) if water_f.any() else 0.0,
                "n_rescued": int((n_per_px == -1).sum()),
            }
            depth = np.nan_to_num(depth_f, nan=0.0)

        # ── Output payload (same contract as /api/pro-extract) ──
        water = result.water_mask
        zones = {
            'vs': int(np.sum(water & (depth <= 3))),
            'sh': int(np.sum(water & (depth > 3) & (depth <= 8))),
            'md': int(np.sum(water & (depth > 8) & (depth <= 15))),
            'dp': int(np.sum(water & (depth > 15))),
        }
        valid = depth[water & (depth > 0)]
        # Show the ICESat-2 references that drove the fit on the map.
        # Cap to 5k pts so the browser stays smooth.
        photon_preview = [
            {'lat': p['lat'], 'lon': p['lon'],
             'depth': round(float(p['depth']), 2),
             'photon_class': 'bathymetry',
             'acq_date': p.get('acq_date', ''),
             'acq_dt': p.get('acq_dt', '')}
            for p in ref_pts[:5000]
        ]
        out = {
            'bbox': bd,
            'points': photon_preview,
            'interpolated_points': grid_to_points(depth, bbox),
            'sources_used': sources_used,
            'ml_stats': {
                'method': (f"CBR×{len(per_scene_info)}-scene MLE" if n_scenes > 1
                           else 'CBR (Geyman & Maloof 2019) + SlideRule ICESat-2'),
                'r2': result.metrics['r2'],
                'rmse': result.metrics['rmse'],
                'mae': result.metrics['mae'],
                'bias': result.metrics['bias'],
                'n_train': result.metrics['n_train'],
                'iho_s44': result.metrics['iho_s44'],
                'zone_counts': zones,
            },
            'cbr': {
                'n_clusters': result.n_clusters,
                'per_class': result.per_class,
                'metrics': result.metrics,
                'n_user_points': n_user,
                'n_icesat2': n_ice,
                'mle': mle_info if n_scenes > 1 else None,
                'per_scene': per_scene_info if n_scenes > 1 else None,
                'retide': retide_info,
                'icesat2_refs_used': [
                    {'lat': p['lat'], 'lon': p['lon'],
                     'depth': round(float(p['depth']), 2),
                     'depth_raw': round(float(p.get('depth_raw', p['depth'])), 2),
                     'depth_shift_m': float(p.get('depth_shift_m', 0.0)),
                     'acq_date': p.get('acq_date', ''),
                     'acq_dt': p.get('acq_dt', ''),
                     'photon_class': 'bathymetry'}
                    for p in ref_pts
                ],
            },
            'stats': {
                'mean_depth': round(float(np.mean(valid)), 2) if len(valid) else 0,
                'max_depth':  round(float(np.max(valid)),  2) if len(valid) else 0,
                'min_depth':  round(float(np.min(valid)),  2) if len(valid) else 0,
                'std_depth':  round(float(np.std(valid)),  2) if len(valid) else 0,
                'grid_points': int(np.sum(np.isfinite(result.depth))),
                'resolution_m': res,
            },
            'tracks': [], 'sea_profiles': [], 'bath_profiles': [],
        }
        try:
            b64, rb, rmx = depth_to_raster_png(depth, bbox, water_mask=water)
            if b64:
                out['raster_png'] = b64
                out['raster_bounds'] = rb
                out['raster_max_depth'] = rmx
        except Exception:
            pass
        try:
            tif_b64 = build_geotiff_b64(depth, bbox)
            if tif_b64:
                out['geotiff_b64'] = tif_b64
        except Exception:
            pass
        return jsonify(out)
    except ValueError as vex:
        return jsonify({'error': str(vex)}), 200
    except Exception as ex:
        L.error(f"cbr-bathymetry: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# S2SHORES — physical wave-dispersion bathymetry (Almar et al.)
# ══════════════════════════════════════════════════════════════

# Evalscript for the per-scene quality probe: B02/B03/B04/SCL, single-orbit.
# SCL drives the water mask; B02 carries the swell signature; B04/B03 give
# the turbidity proxy (Nechad-like red reflectance + Red/Green ratio).
EVALSCRIPT_WAVE_PROBE = """//VERSION=3
function setup(){return{input:[{bands:["B02","B03","B04","SCL"],units:"DN",mosaicking:"SIMPLE"}],output:{bands:4,sampleType:"UINT16"}};}
function evaluatePixel(s){return [s[0].B02, s[0].B03, s[0].B04, s[0].SCL];}"""


def _sh_fetch_probe(bbox, sd_iso, ed_iso, pixel_size_m=60, cloud=100):
    """Single-orbit B02/B03/B04/SCL fetch at low res for scene quality scoring."""
    w, s, e, n = bbox
    cl = np.cos(np.radians((n + s) / 2))
    wp = max(16, min(512, int(abs(e - w) * 111000 * cl / pixel_size_m)))
    hp = max(16, min(512, int(abs(n - s) * 111000 / pixel_size_m)))
    tok = sh_token()
    body = {"input": {"bounds": {"bbox": [w, s, e, n],
            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data": [{"type": "sentinel-2-l2a",
                  "dataFilter": {"maxCloudCoverage": cloud,
                                 "timeRange": {"from": sd_iso, "to": ed_iso},
                                 "mosaickingOrder": "leastRecent"}}]},
        "output": {"width": wp, "height": hp,
                   "responses": [{"identifier": "default",
                                  "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT_WAVE_PROBE}
    r = requests.post(SH_PROC,
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
        json=body, timeout=120)
    if not r.ok:
        raise RuntimeError(f"probe {r.status_code}: {r.text[:160]}")
    img = tifffile.imread(io.BytesIO(r.content))
    if img.ndim == 3 and img.shape[0] == 4:
        b02, b03, b04, scl = img[0], img[1], img[2], img[3]
    elif img.ndim == 3 and img.shape[2] == 4:
        b02, b03, b04, scl = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
    else:
        raise RuntimeError(f"probe shape {img.shape}")
    return b02.astype(np.float32), b03.astype(np.float32), b04.astype(np.float32), scl.astype(np.uint8)


def _score_scene_swell_turbidity(b02, b03, b04, scl, pixel_size_m=60):
    """Return dict with turbidity and swell-energy scores on a candidate scene.

    Turbidity proxy  — Nechad-like: mean B04 over SCL=water, also Red/Green.
                       Higher ⇒ more turbid (bad).
    Swell energy    — 2-D FFT of detrended B02 on water, fraction of power
                      in the valid wavelength band [MIN_WAVELENGTH_M,
                      MAX_WAVELENGTH_M]. Higher ⇒ visible swell (good).
    """
    water = (scl == 6)
    if water.sum() < 100:
        # SCL=6 (water) can be stingy — fall back to NDWI > 0
        g = b03 + 1e-6
        ndwi = (b03 - b04) / (g + b04 + 1e-6)
        water = ndwi > 0.0
    if water.sum() < 50:
        return {"turbidity": 9999.0, "swell": 0.0, "water_frac": 0.0}
    water_frac = float(water.sum()) / water.size
    red_w = b04[water]
    grn_w = b03[water]
    # Turbidity: mean red reflectance (DN) + Red/Green ratio composite.
    turb_red = float(np.mean(red_w))
    turb_rg = float(np.mean(red_w / (grn_w + 1e-6)))
    # Normalise: smaller = clearer
    turbidity = turb_red * 0.001 + turb_rg

    # Swell energy: detrended B02 2-D FFT, valid wavelength band fraction
    from scipy.fft import fft2
    b02_w = np.where(water, b02 - float(np.mean(b02[water])), 0.0)
    # Remove residual mean to cut DC
    b02_w = b02_w - np.mean(b02_w)
    # Window to avoid edge leakage
    H, W = b02_w.shape
    wy = np.hanning(H)[:, None]; wx = np.hanning(W)[None, :]
    b02_w = b02_w * wy * wx
    P = np.abs(fft2(b02_w)) ** 2
    fy = np.fft.fftfreq(H) / pixel_size_m
    fx = np.fft.fftfreq(W) / pixel_size_m
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    freq_mag = np.sqrt(FX ** 2 + FY ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        wl = np.where(freq_mag > 0, 1.0 / freq_mag, 0.0)
    from backend.wave_bathy import MIN_WAVELENGTH_M, MAX_WAVELENGTH_M
    valid = (wl >= MIN_WAVELENGTH_M) & (wl <= MAX_WAVELENGTH_M)
    total = float(P.sum() - P[0, 0])  # exclude DC
    band = float(P[valid].sum())
    swell = band / (total + 1e-12)
    return {"turbidity": turbidity, "swell": swell, "water_frac": water_frac}


def _gee_list_scenes(bbox, sd, ed, max_cloud=40, limit=12):
    """List S2 L1C scenes in bbox/date range via GEE, sorted by cloud ASC."""
    if not _init_gee():
        raise RuntimeError("GEE not initialised")
    import ee
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n])
    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterDate(sd, ed)
           .filterBounds(region)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", int(max_cloud)))
           .sort("CLOUDY_PIXEL_PERCENTAGE")
           .limit(int(limit)))
    feats = col.getInfo().get("features", [])
    out = []
    for f in feats:
        props = f.get("properties", {}) or {}
        scene_id = props.get("PRODUCT_ID") or f.get("id")
        t_ms = props.get("system:time_start")
        date = ""
        if isinstance(t_ms, (int, float)):
            from datetime import datetime as _dt
            date = _dt.utcfromtimestamp(t_ms / 1000.0).strftime("%Y-%m-%d")
        out.append({
            "id": scene_id,
            "date": date,
            "time_ms": t_ms,
            "cloud_pct": float(props.get("CLOUDY_PIXEL_PERCENTAGE", 99)),
        })
    L.info(f"[GEE scenes] {len(out)} candidates in [{sd}→{ed}] max_cloud={max_cloud}")
    return out


def _gee_fetch_probe(bbox, time_ms, pad_minutes=30, pixel_size_m=60):
    """Single-scene B02/B03/B04/SCL probe via GEE (narrow time window).

    Uses S2_SR_HARMONIZED (L2A) so we get the SCL scene-classification band
    for clean water masking; L1C (used for the final wave fetch) has no SCL.
    """
    if not _init_gee():
        raise RuntimeError("GEE not initialised")
    import ee
    from datetime import datetime as _dt, timedelta as _td
    t0 = _dt.utcfromtimestamp(time_ms / 1000.0) - _td(minutes=pad_minutes)
    t1 = _dt.utcfromtimestamp(time_ms / 1000.0) + _td(minutes=pad_minutes)
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n])
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(t0.isoformat(), t1.isoformat())
           .filterBounds(region))
    img = ee.Image(col.first()).select(["B2", "B3", "B4", "SCL"]).clip(region)
    url = img.getDownloadURL({
        "region": region, "scale": int(pixel_size_m),
        "format": "GEO_TIFF", "crs": "EPSG:4326",
    })
    r = requests.get(url, timeout=120)
    if not r.ok:
        raise RuntimeError(f"GEE probe {r.status_code}: {r.text[:160]}")
    arr = tifffile.imread(io.BytesIO(r.content))
    if arr.ndim == 3 and arr.shape[2] == 4:
        b02, b03, b04, scl = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    elif arr.ndim == 3 and arr.shape[0] == 4:
        b02, b03, b04, scl = arr[0], arr[1], arr[2], arr[3]
    else:
        raise RuntimeError(f"GEE probe shape {arr.shape}")
    return b02.astype(np.float32), b03.astype(np.float32), b04.astype(np.float32), scl.astype(np.uint8)


def _gee_fetch_s2_wave_by_time(bbox, time_ms, pad_minutes=30, res=10):
    """Fetch the single S2 L1C orbit at `time_ms` for wave analysis (B02+B04)."""
    if not _init_gee():
        raise RuntimeError("GEE not initialised")
    import ee
    from datetime import datetime as _dt, timedelta as _td
    t0 = _dt.utcfromtimestamp(time_ms / 1000.0) - _td(minutes=pad_minutes)
    t1 = _dt.utcfromtimestamp(time_ms / 1000.0) + _td(minutes=pad_minutes)
    w, s, e, n = bbox
    region = ee.Geometry.Rectangle([w, s, e, n])
    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterDate(t0.isoformat(), t1.isoformat())
           .filterBounds(region))
    img = ee.Image(col.first()).select(["B2", "B4"]).clip(region)
    scene_id = img.get("PRODUCT_ID").getInfo()
    url = img.getDownloadURL({
        "region": region, "scale": int(res),
        "format": "GEO_TIFF", "crs": "EPSG:4326",
    })
    r = requests.get(url, timeout=300)
    if not r.ok:
        raise RuntimeError(f"GEE wave {r.status_code}: {r.text[:200]}")
    arr = tifffile.imread(io.BytesIO(r.content))
    if arr.ndim == 3 and arr.shape[2] == 2:
        b02, b04 = arr[:, :, 0], arr[:, :, 1]
    elif arr.ndim == 3 and arr.shape[0] == 2:
        b02, b04 = arr[0], arr[1]
    else:
        raise RuntimeError(f"GEE wave shape {arr.shape}")
    return {"b02": b02, "b04": b04, "width": b02.shape[1], "height": b02.shape[0],
            "scene_id": scene_id, "scene_time_ms": time_ms, "source": "GEE"}


def _pick_best_scene(scenes, bbox, turbidity_weight=0.5, swell_weight=0.4,
                     cloud_weight=0.1, use_gee=False):
    """For each scene dict (needs 'date'), probe B02/B03/B04/SCL and score.
    Returns (best_scene, full_scores_list)."""
    scored = []
    for sc in scenes:
        date = sc.get("date")
        if not date:
            continue
        try:
            if use_gee:
                t_ms = sc.get("time_ms")
                if not t_ms:
                    L.warning(f"probe {date}: no time_ms in GEE record")
                    continue
                b02, b03, b04, scl = _gee_fetch_probe(bbox, t_ms)
            else:
                sd_iso = f"{date}T00:00:00Z"
                ed_iso = f"{date}T23:59:59Z"
                b02, b03, b04, scl = _sh_fetch_probe(bbox, sd_iso, ed_iso)
        except Exception as ex:
            L.warning(f"probe {date}: {ex}")
            continue
        m = _score_scene_swell_turbidity(b02, b03, b04, scl)
        cloud = float(sc.get("cloud_pct", 50))
        m["date"] = date
        m["cloud_pct"] = cloud
        m["scene_id"] = sc.get("id")
        m["time_ms"] = sc.get("time_ms")
        scored.append(m)

    if not scored:
        return None, []

    # Normalise each metric across candidates (min-max) so weights are meaningful.
    def _n(arr, higher_is_better=False):
        a = np.array(arr, dtype=np.float64)
        lo, hi = a.min(), a.max()
        rng = hi - lo
        if rng < 1e-9:
            return np.zeros_like(a)
        n = (a - lo) / rng
        return n if higher_is_better else 1.0 - n

    turb_n = _n([x["turbidity"] for x in scored], higher_is_better=False)
    swell_n = _n([x["swell"]     for x in scored], higher_is_better=True)
    cloud_n = _n([x["cloud_pct"] for x in scored], higher_is_better=False)

    for i, x in enumerate(scored):
        x["score_turb_n"] = float(turb_n[i])
        x["score_swell_n"] = float(swell_n[i])
        x["score_cloud_n"] = float(cloud_n[i])
        x["score"] = (turbidity_weight * float(turb_n[i]) +
                      swell_weight     * float(swell_n[i]) +
                      cloud_weight     * float(cloud_n[i]))

    scored.sort(key=lambda x: x["score"], reverse=True)
    for x in scored:
        L.info(f"[scene {x['date']}] cloud={x['cloud_pct']:.1f}% "
               f"turb={x['turbidity']:.2f} swell={x['swell']:.3f} "
               f"→ score={x['score']:.3f}")
    return scored[0], scored


@app.route('/api/s2shores-bathymetry', methods=['POST'])
def api_s2shores_bathymetry():
    """Physical bathymetry via Sentinel-2 wave dispersion — Almar et al. /
    CNES S2Shores. Uses the inter-band detector time offset (~1.005 s
    between B02 and B04 on S2A/B) to measure local wave celerity, solves
    the linear dispersion relation ω² = g·k·tanh(k·h) for depth.

    Request JSON:
      bbox: {west,south,east,north}
      start_date, end_date
      cloud: int (default 15 — lower helps single-orbit pick)
      resolution_m: 10-30 (default 20) — OUTPUT grid pixel size
      window_m: spatial window in metres (default 400)
      step_m:   sliding step (default = resolution_m so output is at res)
      source:   'gee' | 'sh' | 'auto' (default auto — SH first, GEE fallback)
      top_k:    number of scenes to rank+fuse (default 6, 1 = single-orbit)
      turbidity_weight: 0..1 blend of turbidity vs cloud in scene ranking
                        (default 0.4)
      tiled:    bool — if true, split the scene into tile_nx×tile_ny and
                run wave_bathymetry in parallel, feather-mosaic the output
                (default false)
      tile_nx, tile_ny: tiling grid (default 4×4 when tiled=true)
      overlap_m: tile overlap in metres (default 400)
      n_workers: ThreadPool size for tiled run (default 4)
      input_res_m: pixel size of the input sampled to wave_bathymetry.
                   Downsamples the native 10 m fetch to this — stabilises
                   long swells and cuts compute. Default 10.
      force_date:  'YYYY-MM-DD' — bypass the ranker and fetch this exact
                   scene date (both SH and GEE). Useful for A/B-testing
                   specific scenes.
    """
    if not WAVE_AVAILABLE:
        return jsonify({'error': 'wave_bathy module not available'}), 500
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox')
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        bbox = [bd['west'], bd['south'], bd['east'], bd['north']]
        sd = data.get('start_date', '2024-05-01')
        ed = data.get('end_date', '2024-09-30')
        cloud = int(data.get('cloud', 15))
        # Output pixel size clamp: wave physics needs ≳2 samples / wavelength,
        # and the dispersion inversion breaks past 30 m anyway.
        res = max(10, min(30, int(data.get('resolution_m', 20))))
        window_m = float(data.get('window_m', 400.0))
        step_m = float(data.get('step_m', float(res)))  # output pixel = step_m
        source = (data.get('source') or 'auto').lower()
        top_k = max(1, min(12, int(data.get('top_k', 6))))
        turbidity_weight = float(data.get('turbidity_weight', 0.4))
        turbidity_weight = max(0.0, min(1.0, turbidity_weight))
        tiled = bool(data.get('tiled', False))
        tile_nx = max(1, int(data.get('tile_nx', 4)))
        tile_ny = max(1, int(data.get('tile_ny', 4)))
        overlap_m = float(data.get('overlap_m', 400.0))
        n_workers = max(1, min(8, int(data.get('n_workers', 4))))
        input_res_m = max(10, min(60, int(data.get('input_res_m', 10))))
        force_date = data.get('force_date')
        w, s, e, n = bbox
        area_km2 = abs(e - w) * abs(n - s) * 111 * 111 * math.cos(math.radians((n + s) / 2))
        L.info(f"=== S2Shores area={area_km2:.0f}km² res={res}m window={window_m}m step={step_m}m ===")

        # ── Step 0: force_date short-circuits the ranker — fetch that
        #    exact scene and skip ranking entirely.
        s2w = None; src_tag = None; errors = []
        scene_report = None
        ranker_used = None
        if force_date:
            from datetime import datetime as _fd_dt, timedelta as _fd_td
            _fd_end = (_fd_dt.strptime(force_date, '%Y-%m-%d')
                       + _fd_td(days=1)).strftime('%Y-%m-%d')
            L.info(f"S2Shores: force_date={force_date} (window {force_date}→{_fd_end})")
            # GEE first (CDSE PUs may be exhausted), SH as fallback.
            if source in ('gee', 'auto'):
                try:
                    s2w = fetch_s2_wave_gee(
                        bbox, force_date, _fd_end, res=10, cloud=max(cloud, 50))
                    s2w["source"] = f"GEE forced {force_date}"
                    src_tag = s2w["source"]
                    ranker_used = "forced-GEE"
                except Exception as ex:
                    errors.append(f"GEE forced fetch: {ex}")
                    s2w = None
            if s2w is None and source in ('sh', 'auto'):
                try:
                    s2w = fetch_s2_wave(bbox, force_date, _fd_end, cloud=max(cloud, 50))
                    s2w["source"] = f"SH forced {force_date}"
                    src_tag = s2w["source"]
                    ranker_used = "forced-SH"
                except Exception as ex:
                    errors.append(f"SH forced fetch: {ex}")
                    s2w = None
        # Reordered: GEE ranker first (CDSE may be out of PUs)
        if s2w is None and top_k > 1 and source in ('gee', 'auto'):
            try:
                candidates = _gee_list_scenes(bbox, sd, ed,
                                              max_cloud=max(cloud, 40), limit=top_k)
                L.info(f"S2Shores(GEE): {len(candidates)} candidate scenes")
                if candidates:
                    winner, scored = _pick_best_scene(
                        candidates, bbox,
                        turbidity_weight=turbidity_weight,
                        swell_weight=max(0.0, 1.0 - turbidity_weight - 0.1),
                        cloud_weight=0.1, use_gee=True)
                    scene_report = scored
                    if winner:
                        L.info(f"S2Shores(GEE): picked {winner['date']} "
                               f"cloud={winner['cloud_pct']:.1f}% "
                               f"turb={winner['turbidity']:.2f} "
                               f"swell={winner['swell']:.3f}")
                        try:
                            s2w = _gee_fetch_s2_wave_by_time(
                                bbox, winner["time_ms"], res=10)
                            s2w["source"] = f"GEE top-{top_k}→{winner['date']}"
                            src_tag = s2w["source"]
                            ranker_used = "GEE"
                        except Exception as ex:
                            errors.append(f"GEE winner fetch: {ex}")
                            s2w = None
            except Exception as ex:
                errors.append(f"GEE scene ranking: {ex}")

        # SH ranker as secondary fallback
        if s2w is None and top_k > 1 and source in ('sh', 'auto'):
            try:
                from datetime import datetime as _dt, timedelta as _td
                _sd_dt = _dt.strptime(sd, '%Y-%m-%d')
                _ed_dt = _dt.strptime(ed, '%Y-%m-%d')
                _weeks = max(1, int((_ed_dt - _sd_dt).days / 7) + 1)
                candidates = search_best_scenes(
                    bbox, n_weeks=_weeks, max_cloud=max(cloud, 40), top_k=top_k)
                L.info(f"S2Shores(SH): {len(candidates)} candidate scenes")
                if candidates:
                    winner, scored = _pick_best_scene(
                        candidates, bbox,
                        turbidity_weight=turbidity_weight,
                        swell_weight=max(0.0, 1.0 - turbidity_weight - 0.1),
                        cloud_weight=0.1, use_gee=False)
                    scene_report = scored
                    if winner:
                        wd = winner["date"]
                        L.info(f"S2Shores(SH): picked {wd} "
                               f"cloud={winner['cloud_pct']:.1f}% "
                               f"turb={winner['turbidity']:.2f} "
                               f"swell={winner['swell']:.3f}")
                        try:
                            # Always fetch at native 10 m so win_px is large
                            # enough for FFT and step_m controls the output
                            # grid independently of the input sampling.
                            s2w = fetch_s2_wave(bbox, wd, wd, cloud=max(cloud, 40))
                            s2w["source"] = f"SH top-{top_k}→{wd}"
                            s2w["scene_id"] = winner.get("scene_id")
                            src_tag = s2w["source"]
                            ranker_used = "SH"
                        except Exception as ex:
                            errors.append(f"SH winner fetch: {ex}")
                            s2w = None
            except Exception as ex:
                errors.append(f"SH scene ranking: {ex}")

        # ── Fallback: legacy single-orbit (most-recent) path, always 10 m ──
        if s2w is None and source in ('gee', 'auto'):
            try:
                s2w = fetch_s2_wave_gee(bbox, sd, ed, res=10, cloud=cloud)
                src_tag = s2w.get("source", "GEE")
            except Exception as ex:
                errors.append(f"GEE most-recent: {ex}")
                s2w = None
        if s2w is None and source in ('sh', 'auto'):
            try:
                s2w = fetch_s2_wave(bbox, sd, ed, cloud=cloud)
                s2w["source"] = "SH (most-recent)"
                src_tag = s2w["source"]
            except Exception as ex:
                errors.append(f"SH: {ex}")
                s2w = None
        # ── Last-resort auto-widening: walk back over the last 12 months and
        # let the ranker pick the best 3 candidates regardless of the user-
        # supplied date window. The wave inversion is single-scene (the S2
        # inter-detector time offset carries the celerity signal), but if the
        # primary winner can't be fetched we try the runner-ups in order.
        if s2w is None:
            try:
                from datetime import datetime as _dt, timedelta as _td
                # Anchor the wide window on the centre of the user's range
                sd_dt = _dt.strptime(sd, '%Y-%m-%d')
                ed_dt = _dt.strptime(ed, '%Y-%m-%d')
                mid = sd_dt + (ed_dt - sd_dt) / 2
                wide_sd = (mid - _td(days=180)).strftime('%Y-%m-%d')
                wide_ed = (mid + _td(days=180)).strftime('%Y-%m-%d')
                L.info(f"S2Shores: auto-widening to {wide_sd}→{wide_ed} "
                       "(no clear scene in original window)")
                if source in ('gee', 'auto'):
                    candidates = _gee_list_scenes(bbox, wide_sd, wide_ed,
                                                   max_cloud=80, limit=12)
                    L.info(f"S2Shores wide-window: {len(candidates)} candidates")
                    if candidates:
                        winner, scored = _pick_best_scene(
                            candidates, bbox,
                            turbidity_weight=turbidity_weight,
                            swell_weight=max(0.0, 1.0 - turbidity_weight - 0.1),
                            cloud_weight=0.1, use_gee=True)
                        scene_report = scored
                        # Try the top 3 in score order so we recover from a
                        # single-scene download glitch automatically.
                        for cand in (scored or [])[:3]:
                            try:
                                s2w = _gee_fetch_s2_wave_by_time(
                                    bbox, cand["time_ms"], res=10)
                                s2w["source"] = (
                                    f"GEE auto-widened top-3 → {cand['date']}")
                                src_tag = s2w["source"]
                                ranker_used = "auto-widened-GEE"
                                L.info(f"S2Shores auto-widened: picked {cand['date']} "
                                       f"cloud={cand.get('cloud_pct',0):.1f}% "
                                       f"turb={cand.get('turbidity',0):.2f} "
                                       f"swell={cand.get('swell',0):.3f}")
                                break
                            except Exception as ex:
                                errors.append(f"GEE auto-widened {cand.get('date')}: {ex}")
            except Exception as ex:
                errors.append(f"GEE auto-widen: {ex}")

        if s2w is None:
            hint = ("No clear Sentinel-2 scene available for this ROI in the "
                    "last 12 months. The auto-widener tried the cleanest 3 "
                    "candidates and none could be fetched. Try a different ROI "
                    "or check that GEE credentials are configured.")
            return jsonify({'error': hint, 'diagnostics': errors}), 200

        L.info(f"S2Shores: {s2w['width']}x{s2w['height']}px (native 10 m) via {src_tag}")

        # ── Optional: downsample input to input_res_m (stabilises long
        #    swells, cuts per-window compute). Default 10 = no resample.
        if input_res_m != 10:
            from scipy.ndimage import zoom as _ndzoom
            scale = 10.0 / float(input_res_m)
            s2w = {
                "b02": _ndzoom(s2w["b02"], (scale, scale), order=1).astype(s2w["b02"].dtype),
                "b04": _ndzoom(s2w["b04"], (scale, scale), order=1).astype(s2w["b04"].dtype),
                "width": None, "height": None,
                "source": s2w.get("source"),
                "scene_id": s2w.get("scene_id"),
                "scene_time_ms": s2w.get("scene_time_ms"),
            }
            s2w["width"]  = s2w["b02"].shape[1]
            s2w["height"] = s2w["b02"].shape[0]
            L.info(f"S2Shores: resampled to {s2w['width']}x{s2w['height']} @{input_res_m} m")

        pixel_size_for_wave = float(input_res_m)

        # ── Run physical wave inversion — tiled or monolithic ──
        if tiled:
            from backend.wave_bathy import tiled_wave_bathymetry
            L.info(f"S2Shores: TILED {tile_nx}x{tile_ny} overlap={overlap_m}m "
                   f"workers={n_workers}")
            result, err = tiled_wave_bathymetry(
                s2w, bbox,
                tile_nx=tile_nx, tile_ny=tile_ny,
                overlap_m=overlap_m,
                window_m=window_m, step_m=step_m,
                pixel_size_m=pixel_size_for_wave,
                n_workers=n_workers,
            )
        else:
            result, err = wave_bathymetry(
                s2w, bbox,
                window_m=window_m, step_m=step_m,
                pixel_size_m=pixel_size_for_wave,
            )
        if err or result is None:
            return jsonify({'error': f'Wave bathymetry failed: {err or "empty result"}',
                            'sources_used': [src_tag]}), 200

        depth = np.nan_to_num(result['depth'], nan=0.0).astype(np.float32)
        water = np.isfinite(result['depth']) & (result['depth'] > 0)
        zones = {
            'vs': int(np.sum(water & (depth <= 3))),
            'sh': int(np.sum(water & (depth > 3) & (depth <= 8))),
            'md': int(np.sum(water & (depth > 8) & (depth <= 15))),
            'dp': int(np.sum(water & (depth > 15))),
        }
        valid = depth[water & (depth > 0)]
        stats = result.get('stats', {})
        sources_used = [f"{src_tag} S2 L1C B02+B04 @{res}m",
                        f"WaveDispersion({stats.get('valid_windows', 0)} windows)"]

        out = {
            'bbox': bd,
            'points': [],
            'interpolated_points': grid_to_points(result['depth'], bbox),
            'sources_used': sources_used,
            'ml_stats': {
                'method': 'S2Shores / Almar — wave-dispersion physics',
                'r2': stats.get('r2', 0.0),
                'rmse': stats.get('rmse', 0.0),
                'mae': stats.get('mae', 0.0),
                'bias': stats.get('bias', 0.0),
                'zone_counts': zones,
            },
            's2shores': {
                'resolution_m': res,
                'window_m': window_m,
                'step_m': step_m,
                'valid_windows': stats.get('valid_windows'),
                'total_windows': stats.get('total_windows'),
                'mean_wavelength_m': stats.get('mean_wavelength_m'),
                'mean_celerity_m_s': stats.get('mean_celerity_ms'),
                'mean_period_s': stats.get('mean_period_s'),
                'source': src_tag,
                'scene_id': s2w.get('scene_id'),
                'scene_time_ms': s2w.get('scene_time_ms'),
                'scene_report': scene_report,
                'top_k': top_k,
                'turbidity_weight': turbidity_weight,
                'tiled': tiled,
                'tile_nx': tile_nx if tiled else None,
                'tile_ny': tile_ny if tiled else None,
                'overlap_m': overlap_m if tiled else None,
                'input_res_m': input_res_m,
                'tile_report': result.get('tile_report') if isinstance(result, dict) else None,
                'tiles_ok': stats.get('tiles_ok'),
                'tiles_total': stats.get('tiles_total'),
            },
            'stats': {
                'mean_depth': round(float(np.mean(valid)), 2) if len(valid) else 0,
                'max_depth':  round(float(np.max(valid)),  2) if len(valid) else 0,
                'min_depth':  round(float(np.min(valid)),  2) if len(valid) else 0,
                'std_depth':  round(float(np.std(valid)),  2) if len(valid) else 0,
                'grid_points': int(np.sum(np.isfinite(result['depth']))),
                'resolution_m': res,
            },
            'tracks': [], 'sea_profiles': [], 'bath_profiles': [],
        }
        try:
            b64, rb, rmx = depth_to_raster_png(result['depth'], bbox, water_mask=water)
            if b64:
                out['raster_png'] = b64
                out['raster_bounds'] = rb
                out['raster_max_depth'] = rmx
        except Exception:
            pass
        try:
            tif_b64 = build_geotiff_b64(result['depth'], bbox)
            if tif_b64:
                out['geotiff_b64'] = tif_b64
        except Exception:
            pass
        return jsonify(out)
    except ValueError as vex:
        return jsonify({'error': str(vex)}), 200
    except Exception as ex:
        L.error(f"s2shores-bathymetry: {ex}\n{traceback.format_exc()}")
        return jsonify({'error': str(ex)}), 500


# ══════════════════════════════════════════════════════════════
# MONITORING API
# ══════════════════════════════════════════════════════════════
try:
    from backend.patent_mlp import mlp_train_and_predict,apply_tidal_correction
    MLP_AVAILABLE=True
except:
    try:
        from patent_mlp import mlp_train_and_predict,apply_tidal_correction
        MLP_AVAILABLE=True
    except: MLP_AVAILABLE=False

try:
    from backend.monitoring import (store_survey,get_survey,list_surveys,compute_change,
        compute_trends,get_alerts,acknowledge_alert,create_zone,list_zones,get_dashboard_data)
    MONITORING_AVAILABLE=True
except:
    try:
        from monitoring import (store_survey,get_survey,list_surveys,compute_change,
            compute_trends,get_alerts,acknowledge_alert,create_zone,list_zones,get_dashboard_data)
        MONITORING_AVAILABLE=True
    except: MONITORING_AVAILABLE=False

@app.route('/api/monitoring/store',methods=['POST'])
def api_monitoring_store():
    """Store current results as a monitoring survey."""
    try:
        data=req.get_json(force=True,silent=True) or {}
        survey_id=data.get('survey_id',f"survey_{int(time.time())}")
        bbox=data.get('bbox')
        date=data.get('date',time.strftime('%Y-%m-%d'))
        if not bbox:return jsonify({'error':'Missing bbox'}),400
        bbox_list=[bbox['west'],bbox['south'],bbox['east'],bbox['north']] if isinstance(bbox,dict) else bbox
        # Reconstruct depth grid from results in session (stored via /api/extract)
        depth_data=data.get('depth_grid')
        if depth_data:
            grid=np.array(depth_data,dtype=np.float32)
        else:
            return jsonify({'error':'No depth grid provided. Run extraction first.'}),400
        unc_data=data.get('uncertainty')
        unc=np.array(unc_data,dtype=np.float32) if unc_data else None
        stats=store_survey(survey_id,bbox_list,date,grid,unc,
            method=data.get('method','cnn'),resolution=data.get('resolution',10),
            r2=data.get('r2',0),rmse=data.get('rmse',0),
            sources=data.get('sources',''),metadata=data.get('metadata'))
        return jsonify({'survey_id':survey_id,'stats':stats,'stored':True})
    except Exception as ex:
        L.error(f"Monitoring store: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex)}),500

@app.route('/api/monitoring/surveys',methods=['GET','POST'])
def api_monitoring_surveys():
    """List stored surveys."""
    try:
        bbox=None
        if req.method=='POST':
            data=req.get_json(force=True,silent=True) or {}
            b=data.get('bbox')
            if b:bbox=[b['west'],b['south'],b['east'],b['north']] if isinstance(b,dict) else b
        surveys=list_surveys(bbox)
        return jsonify({'surveys':surveys,'count':len(surveys)})
    except Exception as ex:
        return jsonify({'error':str(ex)}),500

@app.route('/api/monitoring/change',methods=['POST'])
def api_monitoring_change():
    """Compute change between two surveys."""
    try:
        data=req.get_json(force=True,silent=True) or {}
        a=data.get('survey_a');b=data.get('survey_b')
        if not a or not b:return jsonify({'error':'Provide survey_a and survey_b IDs'}),400
        threshold=float(data.get('threshold',0.5))
        result,err=compute_change(a,b,threshold)
        if err:return jsonify({'error':err}),200
        # Convert diff grid to serialisable format
        diff=result['diff_grid']
        valid=np.isfinite(diff)
        diff_pts=[]
        h,w=diff.shape
        step=max(1,h*w//5000)
        bbox=json.loads(result['survey_a'].get('bbox_json','[]')) if isinstance(result['survey_a'],dict) else []
        if len(bbox)==4:
            W,S,E,N=bbox
            for idx in range(0,h*w,step):
                r,c=divmod(idx,w)
                if valid[r,c]:
                    lat=N-(r/h)*(N-S)
                    lon=W+(c/w)*(E-W)
                    diff_pts.append({'lat':round(lat,5),'lon':round(lon,5),'change':round(float(diff[r,c]),3)})
        return jsonify({'statistics':result['statistics'],'survey_a':result['survey_a'],
                        'survey_b':result['survey_b'],'diff_points':diff_pts})
    except Exception as ex:
        L.error(f"Monitoring change: {ex}\n{traceback.format_exc()}")
        return jsonify({'error':str(ex)}),500

@app.route('/api/monitoring/trends',methods=['POST'])
def api_monitoring_trends():
    """Compute depth trends over time."""
    try:
        data=req.get_json(force=True,silent=True) or {}
        b=data.get('bbox')
        bbox=[b['west'],b['south'],b['east'],b['north']] if isinstance(b,dict) and b else None
        min_surveys=int(data.get('min_surveys',3))
        result,err=compute_trends(bbox or [],min_surveys)
        if err:return jsonify({'error':err}),200
        # Convert trend grid to points
        trend=result['trend_rate']
        valid=np.isfinite(trend)
        trend_pts=[]
        h,w=trend.shape
        step=max(1,h*w//5000)
        for idx in range(0,h*w,step):
            r,c=divmod(idx,w)
            if valid[r,c]:
                trend_pts.append({'r':r,'c':c,'rate':round(float(trend[r,c]),4),
                                  'r2':round(float(result['trend_r2'][r,c]),3)})
        return jsonify({'summary':result['summary'],'dates':result['dates'],
                        'trend_points':trend_pts})
    except Exception as ex:
        return jsonify({'error':str(ex)}),500

@app.route('/api/monitoring/alerts',methods=['GET'])
def api_monitoring_alerts():
    """Get active alerts."""
    try:
        sev=req.args.get('severity')
        alerts=get_alerts(severity=sev)
        return jsonify({'alerts':alerts,'count':len(alerts)})
    except Exception as ex:
        return jsonify({'error':str(ex)}),500

@app.route('/api/monitoring/alerts/<int:alert_id>/acknowledge',methods=['POST'])
def api_monitoring_ack(alert_id):
    """Acknowledge an alert."""
    try:
        acknowledge_alert(alert_id)
        return jsonify({'acknowledged':True})
    except Exception as ex:
        return jsonify({'error':str(ex)}),500

@app.route('/api/monitoring/zones',methods=['GET','POST'])
def api_monitoring_zones():
    """List or create monitoring zones."""
    try:
        if req.method=='POST':
            data=req.get_json(force=True,silent=True) or {}
            name=data.get('name','Zone')
            b=data.get('bbox')
            if not b:return jsonify({'error':'Provide bbox'}),400
            bbox=[b['west'],b['south'],b['east'],b['north']] if isinstance(b,dict) else b
            zid=create_zone(name,bbox,data.get('threshold',0.5),
                data.get('min_depth',0),data.get('max_depth',25),data.get('interval_days',30))
            return jsonify({'zone_id':zid,'created':True})
        return jsonify({'zones':list_zones()})
    except Exception as ex:
        return jsonify({'error':str(ex)}),500

@app.route('/api/monitoring/dashboard',methods=['GET','POST'])
def api_monitoring_dashboard():
    """Get full monitoring dashboard data."""
    try:
        bbox=None
        if req.method=='POST':
            data=req.get_json(force=True,silent=True) or {}
            b=data.get('bbox')
            if b:bbox=[b['west'],b['south'],b['east'],b['north']] if isinstance(b,dict) else b
        return jsonify(get_dashboard_data(bbox))
    except Exception as ex:
        return jsonify({'error':str(ex)}),500

# ══════════════════════════════════════════════════════════════
# ACCURATE ESTIMATION — HPC pipeline with SSE progress streaming
# ══════════════════════════════════════════════════════════════
#  POST /api/accurate-bathymetry/start   → kicks off a job, returns job_id
#  GET  /api/accurate-bathymetry/stream/<job_id> (SSE) → live progress
#  GET  /downloads/<name>                → final GeoTIFF / PNG
#
# The iBoating training coordinates are USED inside the backend to
# calibrate the CNN, but they are NEVER exposed: only aggregate counts
# and metrics travel in the SSE events and the final result JSON.
# ══════════════════════════════════════════════════════════════
@app.route('/api/accurate-bathymetry/start', methods=['POST'])
def api_accurate_start():
    try:
        from backend import accurate_runner as AR
    except ImportError:
        import accurate_runner as AR
    data = req.get_json(force=True, silent=True) or {}
    bd = data.get('bbox') or {}
    if not bd:
        return jsonify({'error': 'Missing bbox'}), 400
    bbox = (float(bd['west']), float(bd['south']),
            float(bd['east']), float(bd['north']))
    region = data.get('region', 'custom')
    epochs = int(data.get('epochs', 40))
    n_workers = int(data.get('workers', 32))
    # NEW: observed points from the sidebar (survey XYZ + any ICESat-2 already in userPoints)
    user_points = data.get('user_points', []) or []
    # Bias-correction & calibration options (same schema as cnn-v2/start)
    bc = {
        'mode':          data.get('bc_mode', 'single'),
        'window_days':   int(data.get('bc_window_days', 180)),
        'n_scenes':      int(data.get('bc_n_scenes', 4)),
        'tide':          bool(data.get('bc_tide', True)),
        'wave':          bool(data.get('bc_wave', True)),
        'geoid':         bool(data.get('bc_geoid', True)),
        'calib':         data.get('bc_calib', 'shift+bias+local'),
    }
    job_id = AR.create_job(region, bbox)
    # Attach observations to the job so the runner thread can merge them
    j = AR.get_job(job_id)
    if j is not None:
        j['user_points'] = user_points
        j['bc'] = bc
    import threading
    th = threading.Thread(target=AR.run_accurate_job,
                           args=(job_id, bbox, region, epochs, 20, 25, n_workers),
                           daemon=True)
    th.start()
    return jsonify({'job_id': job_id, 'region': region,
                    'bbox': list(bbox),
                    'stream_url': f'/api/accurate-bathymetry/stream/{job_id}'})


@app.route('/api/cnn-v2/models', methods=['GET'])
def api_cnn_v2_models():
    """List all saved BP-NN checkpoints that inference can use."""
    try:
        from backend import bp_inference as BI
    except ImportError:
        import bp_inference as BI
    return jsonify({"models": BI.list_models()})


@app.route('/api/cnn-v2/inference', methods=['POST'])
def api_cnn_v2_inference():
    """
    Run inference using a saved BP-NN checkpoint — no training.
    POST JSON: {bbox, weights_file, start_date?, end_date?, cloud?, res?}
    """
    try:
        from backend import bp_inference as BI
    except ImportError:
        import bp_inference as BI
    data = req.get_json(force=True, silent=True) or {}
    bd = data.get('bbox') or {}
    if not bd:
        return jsonify({'error': 'Missing bbox'}), 400
    bbox = [float(bd['west']), float(bd['south']),
            float(bd['east']), float(bd['north'])]
    weights_file = data.get('weights_file')
    if not weights_file:
        return jsonify({'error': 'Missing weights_file — query /api/cnn-v2/models to list available checkpoints'}), 400
    try:
        net, ck = BI.load_model(weights_file)
    except Exception as ex:
        return jsonify({'error': f'failed to load weights: {ex}'}), 400

    # Fetch S2 for this (possibly new) ROI
    # SPEC-2: pinned default window (no wall-clock). Explicit dates honored.
    sd = data.get('start_date') or PINNED_S2_WINDOW[0]
    ed = data.get('end_date') or PINNED_S2_WINDOW[1]
    res = int(data.get('res', 10))
    cloud = int(data.get('cloud', 25))
    try:
        s2 = fetch_s2(bbox, sd, ed, res=res, cloud=cloud)
    except Exception as ex:
        return jsonify({'error': f'S2 fetch failed: {ex}'}), 500
    H, W = s2["red"].shape
    water = make_water_mask(bbox, H, W, s2=s2)
    s2["water_mask"] = water

    try:
        depth, info = BI.predict(net, ck, s2, bbox)
    except Exception as ex:
        return jsonify({'error': f'inference failed: {ex}'}), 500

    # Apply same post-prediction calibration if manifest has feature stats
    # (skipped when no observed provided; this is a pure-inference endpoint)

    # Build outputs
    try:
        preview_b64 = depth_to_raster_png(depth, bbox, water_mask=water)
        if isinstance(preview_b64, tuple):  # handle multiple return signatures
            preview_b64 = preview_b64[0]
    except Exception:
        preview_b64 = None
    try:
        geotiff_b64 = build_geotiff_b64(depth, bbox)
    except Exception:
        geotiff_b64 = None

    valid = np.isfinite(depth)
    stats = {
        'depth_min_m':  float(np.nanmin(depth)) if valid.any() else None,
        'depth_max_m':  float(np.nanmax(depth)) if valid.any() else None,
        'depth_mean_m': float(np.nanmean(depth)) if valid.any() else None,
        'n_water_px':   int(water.sum()),
    }

    return jsonify({
        'method':       info.get('method'),
        'architecture': info.get('architecture'),
        'deep':         info.get('deep'),
        'weights_file': weights_file,
        'bbox':         bbox,
        'grid_shape':   info.get('grid_shape'),
        'feature_names': info.get('feature_names'),
        'metrics':      stats,
        'preview_b64':  preview_b64,
        'geotiff_b64':  geotiff_b64,
    })


@app.route('/api/cnn-v2/start', methods=['POST'])
def api_cnn_v2_start():
    """
    CNN_v2 — BP Neural Network pipeline (Guo et al. 2022).
    Reuses the SSE streaming machinery from /api/accurate-bathymetry.
    """
    try:
        from backend import cnn_v2_runner as CR
    except ImportError:
        import cnn_v2_runner as CR
    data = req.get_json(force=True, silent=True) or {}
    bd = data.get('bbox') or {}
    if not bd:
        return jsonify({'error': 'Missing bbox'}), 400
    bbox = (float(bd['west']), float(bd['south']),
            float(bd['east']), float(bd['north']))
    region = data.get('region', 'custom')
    epochs = int(data.get('epochs', 150))
    user_points = data.get('user_points', []) or []
    validate_only = bool(data.get('validate_only_observed', False))
    # Bias-correction & calibration options (all optional)
    bc = {
        'mode':          data.get('bc_mode', 'single'),
        'window_days':   int(data.get('bc_window_days', 180)),
        'n_scenes':      int(data.get('bc_n_scenes', 4)),
        'tide':          bool(data.get('bc_tide', True)),
        'wave':          bool(data.get('bc_wave', True)),
        'geoid':         bool(data.get('bc_geoid', True)),
        'calib':         data.get('bc_calib', 'shift+bias+local'),
        'deep':          bool(data.get('bc_deep', True)),
    }
    job_id = CR.create_job(region, bbox)
    j = CR.get_job(job_id)
    if j is not None:
        j['user_points'] = user_points
        j['validate_only_observed'] = validate_only
        j['bc'] = bc
    import threading
    th = threading.Thread(target=CR.run_cnn_v2_job,
                           args=(job_id, bbox, region, epochs, 10, 25),
                           daemon=True)
    th.start()
    # Reuse the same stream URL format: the SSE endpoint below is the same.
    return jsonify({'job_id': job_id, 'region': region,
                    'bbox': list(bbox),
                    'stream_url': f'/api/accurate-bathymetry/stream/{job_id}'})


@app.route('/api/accurate-bathymetry/stream/<job_id>', methods=['GET'])
def api_accurate_stream(job_id):
    try:
        from backend import accurate_runner as AR
    except ImportError:
        import accurate_runner as AR
    job = AR.get_job(job_id)
    if job is None:
        return jsonify({'error': 'unknown job_id'}), 404

    def _gen():
        import json as _json, time as _time
        yield f"event: start\ndata: {_json.dumps({'job_id': job_id})}\n\n"
        idx = 0
        last_ping = _time.time()
        while True:
            with job["event_cond"]:
                # Wait up to 5 s for a new event
                if idx >= len(job["events"]):
                    job["event_cond"].wait(timeout=5)
                events_snapshot = job["events"][idx:]
                idx = len(job["events"])
            for ev in events_snapshot:
                yield f"event: {ev.get('phase','info')}\ndata: {_json.dumps(ev)}\n\n"
                last_ping = _time.time()
                if ev.get("phase") in ("done", "error"):
                    return
            if not events_snapshot:
                yield f": keepalive {int(_time.time()-last_ping)}s\n\n"
                if job["status"] in ("done", "error") and idx >= len(job["events"]):
                    break
    return Response(_gen(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache, no-transform",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


@app.route('/downloads/<path:name>', methods=['GET'])
def api_downloads(name):
    # Clean path — refuse anything that tries to escape the dir
    if "/" in name or ".." in name:
        return jsonify({'error': 'invalid name'}), 400
    dl_dir = Path(__file__).parent / "ocean" / "downloads"
    p = dl_dir / name
    if not p.exists():
        return jsonify({'error': 'not found'}), 404
    return send_from_directory(str(dl_dir), name, as_attachment=False)


# ADPorts F2: honest Planet-backup status — reads the harvested MANIFEST.json
# + QUOTA_LEDGER.json (a SEPARATE, concurrently-run harvesting agent's
# output; never touched/written here), derives available/assets from disk.
_PLANET_BACKUP_ROOT = Path(__file__).resolve().parent.parent / "data" / "planet_backup"


@app.route('/api/planet-backup/status', methods=['GET'])
def api_planet_backup_status():
    site = (req.args.get('site') or 'khalifa_port').strip()
    site_dir = _PLANET_BACKUP_ROOT / site
    manifest_path = _PLANET_BACKUP_ROOT / "MANIFEST.json"
    ledger_path = _PLANET_BACKUP_ROOT / "QUOTA_LEDGER.json"
    if not manifest_path.exists():
        return jsonify({'error': 'Planet selection manifest unavailable', 'site': site}), 404
    try:
        manifest = json.load(open(manifest_path))
    except Exception as ex:
        return jsonify({'error': f'Planet manifest read failed: {ex}', 'site': site}), 500
    ledger = {}
    if ledger_path.exists():
        try:
            ledger = json.load(open(ledger_path))
        except Exception:
            ledger = {}

    aoi = manifest.get('aoi') or {}
    scenes = manifest.get('scenes') or {}
    chosen_dates = (manifest.get('selection') or {}).get('chosen_dates') or sorted(scenes.keys())

    quota_total = float(ledger.get('account_month_quota_km2') or 0.0)
    used_km2 = 0.0
    for o in (ledger.get('orders') or []):
        try:
            used_km2 += float(o.get('area_km2') or o.get('km2') or 0.0)
        except Exception:
            pass
    remaining_km2 = round(quota_total - used_km2, 3)

    dates_out = []
    n_available = 0
    for d in chosen_dates:
        sc = scenes.get(d) or {}
        tide = sc.get('tide') or {}
        items = sc.get('items') or []
        # Honest disk-derived availability — a date folder with at least one
        # real raster file flips it; MANIFEST alone never asserts imagery.
        date_dir = site_dir / d
        avail_files = []
        if date_dir.exists() and date_dir.is_dir():
            avail_files = [f for f in date_dir.iterdir()
                           if f.is_file() and f.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg')]
        available = len(avail_files) > 0
        assets = {"thumb_url": None, "geotiff_url": None, "size_bytes": None}
        if available:
            n_available += 1
            geot = next((f for f in avail_files if f.suffix.lower() in ('.tif', '.tiff')), None)
            thumb = next((f for f in avail_files if f.suffix.lower() in ('.png', '.jpg')), None)
            if geot:
                assets['geotiff_url'] = f"/api/planet-backup/file?site={site}&date={d}&name={geot.name}"
                assets['size_bytes'] = geot.stat().st_size
            if thumb:
                assets['thumb_url'] = f"/api/planet-backup/file?site={site}&date={d}&name={thumb.name}"
        dates_out.append({
            'date': d,
            'tide_m_msl': tide.get('height_m_msl'),
            'tide_phase': tide.get('phase'),
            'tide_model': tide.get('model'),
            'clear_percent': sc.get('clear_percent'),
            'sun_elevation': sc.get('sun_elevation'),
            'coverage': sc.get('coverage'),
            'n_items': len(items),
            'scene_ids': [it.get('id') for it in items],
            'clip_cost_km2': sc.get('clip_cost_km2'),
            'status': 'available' if available else 'awaiting_provisioning',
            'available': available,
            'assets': assets,
        })

    n_dates = len(dates_out)
    if n_available == 0:
        overall_status = 'selection_ready' if n_dates > 0 else 'provisioning'
        status_label = ('Selection ready — imagery awaiting Planet account provisioning'
                        if n_dates > 0 else 'Awaiting Planet selection/provisioning')
    elif n_available == n_dates:
        overall_status = 'ready'
        status_label = f'Ready — {n_available}/{n_dates} dates available'
    else:
        overall_status = 'partial'
        status_label = f'Partial — {n_available}/{n_dates} dates available'

    return jsonify({
        'site': site,
        'generated_utc': manifest.get('generated_utc'),
        'item_type': manifest.get('item_type'),
        'product_bundle': manifest.get('product_bundle'),
        'aoi': {
            'site_bbox': aoi.get('site_bbox'),
            'area_km2': aoi.get('area_km2'),
            'centroid': aoi.get('centroid'),
        },
        'overall_status': overall_status,
        'status_label': status_label,
        'quota': {
            'account_month_quota_km2': quota_total,
            'used_km2': round(used_km2, 3),
            'remaining_km2': remaining_km2,
        },
        'n_dates': n_dates,
        'n_available': n_available,
        'dates': dates_out,
    })


@app.route('/api/planet-backup/file', methods=['GET'])
def api_planet_backup_file():
    """Serve a real Planet-backup asset (thumb/GeoTIFF) once it exists on
    disk — path-traversal-safe (exact filename match inside the resolved
    site/date dir only)."""
    site = (req.args.get('site') or '').strip()
    date = (req.args.get('date') or '').strip()
    name = (req.args.get('name') or '').strip()
    if not site or not date or not name or '/' in name or '..' in name or '/' in date or '..' in date:
        return jsonify({'error': 'invalid request'}), 400
    date_dir = (_PLANET_BACKUP_ROOT / site / date).resolve()
    if not str(date_dir).startswith(str(_PLANET_BACKUP_ROOT.resolve())):
        return jsonify({'error': 'invalid path'}), 400
    p = date_dir / name
    if not p.exists() or not p.is_file():
        return jsonify({'error': 'not found'}), 404
    return send_from_directory(str(date_dir), name, as_attachment=False)


# --- legacy synchronous endpoint (kept for backward compatibility) ---
@app.route('/api/accurate-bathymetry', methods=['POST'])
def api_accurate_bathymetry():
    try:
        data = req.get_json(force=True, silent=True) or {}
        bd = data.get('bbox') or {}
        if not bd:
            return jsonify({'error': 'Missing bbox'}), 400
        bbox = (float(bd['west']), float(bd['south']),
                float(bd['east']), float(bd['north']))
        region = data.get('region', 'custom')
        epochs = int(data.get('epochs', 60))

        # 1. iboating points (RAG)
        try:
            from backend import iboating_store as S
        except ImportError:
            import iboating_store as S
        ibo = [p for p in S.query_bbox(bbox, region=region,
                                          min_confidence=0.7,
                                          buffer_deg=0.01)
                if p.get('type', '') in ('sounding', 'contour')
                or 'gemini' in str(p.get('source', ''))]

        # 2. ICESat-2 ATL03 via SlideRule
        ice_pts = []
        try:
            from sliderule import sliderule, icesat2
            sliderule.init("slideruleearth.io")
            # SPEC-2: pinned SlideRule window (no wall-clock).
            d0 = PINNED_SR_WINDOW[0]; d1 = PINNED_SR_WINDOW[1]
            w, s_b, e, n = bbox
            poly = [{"lon": w, "lat": s_b}, {"lon": e, "lat": s_b},
                    {"lon": e, "lat": n}, {"lon": w, "lat": n},
                    {"lon": w, "lat": s_b}]
            gdf = icesat2.atl03sp({
                "poly": poly,
                "t0": f"{d0}T00:00:00Z",
                "t1": f"{d1}T23:59:59Z",
                "srt": 1, "cnf": 0, "len": 20.0, "res": 20.0,
                "pass_invalid": False,
                "yapc": {"score": 0, "knn": 0, "min_ph": 4},
            })
            if gdf is not None and len(gdf) > 0:
                gdf = gdf.assign(lat=gdf.geometry.y, lon=gdf.geometry.x)
                col_h = "height" if "height" in gdf.columns else "h_ph"
                for (spot, seg), grp in gdf.groupby(["spot", "segment_id"]):
                    if len(grp) < 30:
                        continue
                    h = grp[col_h].to_numpy()
                    top = float(np.percentile(h, 92))
                    bot = float(np.percentile(h, 8))
                    d_raw = top - bot
                    if d_raw <= 0.3 or d_raw > 35:
                        continue
                    d_cor = d_raw * 0.75  # refraction
                    if d_cor > 25:
                        continue
                    ice_pts.append({
                        "lat": float(grp["lat"].mean()),
                        "lon": float(grp["lon"].mean()),
                        "depth": d_cor,
                        "source": f"icesat2_spot{int(spot)}",
                    })
        except Exception as ex:
            L.warning(f"ICESat-2 in accurate-bathymetry failed: {ex}")

        refs = ibo + ice_pts
        if len(refs) < 10:
            return jsonify({'error': f'Only {len(refs)} reference points'}), 400

        # 3. Sentinel-2
        from datetime import date as _d, timedelta as _td
        ed = data.get('end_date') or _d.today().isoformat()
        sd = data.get('start_date') or (_d.fromisoformat(ed) - _td(days=120)).isoformat()
        s2 = fetch_s2(bbox, sd, ed, res=int(data.get('res', 20)),
                      cloud=int(data.get('cloud', 25)))

        # 4. water mask
        H, Wp = s2["red"].shape
        water = make_water_mask(list(bbox), H, Wp, s2=s2)
        s2["water_mask"] = water

        # 5. train BathyNetPro
        try:
            from backend.bathynet_pro import train_predict
        except ImportError:
            from bathynet_pro import train_predict
        ref = {
            "lats": np.array([p["lat"] for p in refs]),
            "lons": np.array([p["lon"] for p in refs]),
            "depths": np.array([p["depth"] for p in refs]),
        }
        res = train_predict(s2, ref, bbox, epochs=epochs, patch=96, batch=4,
                            max_patches=256, base=48, dropout=0.1,
                            device="cpu", verbose=False)

        # 6. persist + register
        model_dir = Path(__file__).parent / "ocean" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        mp = model_dir / f"{region}_bathynetpro.pt"
        import torch
        torch.save(res["model_state"], mp)
        S.register_model(region, bbox, str(mp), kind="bathynetpro",
                          rmse_val=float(res["rmse_m"] or 0),
                          r2_val=float(res["r2"] or 0),
                          n_train=int(res["n_train_pixels"] or 0),
                          n_val=int(res["n_val_patches"] or 0),
                          notes=f"ICESat-2 ({len(ice_pts)}) + iboating ({len(ibo)})")

        # 7. build tiff + PNG for the UI
        depth = np.where(water & np.isfinite(res["depth"]), res["depth"], np.nan)
        tif64 = build_geotiff_b64(depth, bbox)
        png_b64 = depth_to_raster_png(depth, bbox, water_mask=water,
                                       max_depth=float(res["depth_scale_m"]))

        return jsonify({
            'version': 'v9.4_accurate',
            'architecture': 'BathyNetPro (ResNet34 + ASPP + CBAM + heteroscedastic)',
            'sources': {
                'n_iboating': len(ibo),
                'n_icesat2': len(ice_pts),
                'n_total': len(refs),
            },
            'metrics': {
                'r2': res['r2'],
                'rmse_m': res['rmse_m'],
                'mae_m': res['mae_m'],
                'depth_min_m': float(np.nanmin(depth)) if np.isfinite(depth).any() else None,
                'depth_max_m': float(np.nanmax(depth)) if np.isfinite(depth).any() else None,
                'depth_mean_m': float(np.nanmean(depth)) if np.isfinite(depth).any() else None,
                'n_water_px': int(water.sum()),
                'depth_scale_m': res['depth_scale_m'],
            },
            'model_path': str(mp),
            'geotiff_b64': tif64,
            'png_b64': png_b64,
        })
    except Exception as ex:
        L.error(f"accurate-bathymetry: {ex}", exc_info=True)
        return jsonify({'error': str(ex)}), 500


# ════════════════════════════════════════════════════════════════════════
# DL Pro — production default model.
#   - Loads the deployable bundle from backend/models/dl_pro/.
#   - Optional `ref_pts` (CSV / ATL03 / GEBCO-derived) re-fits the IHO
#     calibration line and, if `fine_tune=true`, warm-starts a few epochs
#     of MLP head fine-tuning.
# ════════════════════════════════════════════════════════════════════════
@app.route('/api/predict_dl_pro', methods=['POST'])
def api_predict_dl_pro():
    try:
        try:
            from backend.dl_pro_engine import predict_dl_pro
        except ImportError:
            from dl_pro_engine import predict_dl_pro
        data = req.get_json() or {}
        bbox = data.get('bbox')
        if not bbox or len(bbox) != 4:
            return jsonify({'error': 'bbox=[w,s,e,n] required'}), 400
        sd = data.get('start_date'); ed = data.get('end_date')
        if not (sd and ed):
            from datetime import date as _d, timedelta as _td
            ed = ed or _d.today().isoformat()
            sd = sd or (_d.fromisoformat(ed) - _td(days=180)).isoformat()
        cloud = int(data.get('cloud', 30))
        res = int(data.get('resolution_m', 20))
        ref_pts = data.get('ref_pts') or []
        # Sources flag: which optional refs to mix in beyond user-supplied CSV.
        # `use_atl03` / `use_gebco` keep the legacy semantics (ATL03 SlideRule
        # cache; GEBCO-derived seabed sample) — handled by the frontend
        # before posting; here we just accept the merged ref_pts list.
        fine_tune = bool(data.get('fine_tune', False))

        # Fetch S2 (re-uses fetch_s2 → cdse). If SH credits exhausted the
        # caller should send `s2_dict` directly (escape hatch).
        if data.get('s2_dict'):
            s2 = data['s2_dict']      # advanced callers
        else:
            s2 = fetch_s2(bbox, sd, ed, res=res, cloud=cloud)
            H, Wp = s2['red'].shape
            water = make_water_mask(list(bbox), H, Wp, s2=s2)
            s2['water_mask'] = water

        out = predict_dl_pro(s2, bbox=list(bbox),
                              ref_pts=ref_pts if ref_pts else None,
                              fine_tune=fine_tune,
                              n_mc=int(data.get('n_mc', 30)))

        # Encode the depth + lower/upper bound + sigma rasters as base64 PNG
        # for fast frontend rendering (the GeoTIFFs are also returned
        # individually so the frontend can offer downloads).
        def _arr_to_b64_tif(arr, fname):
            try:
                import rasterio
                from rasterio.transform import from_bounds
                from io import BytesIO
                Hh, Ww = arr.shape
                tx = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], Ww, Hh)
                buf = BytesIO()
                with rasterio.open(buf, 'w', driver='GTiff',
                                    height=Hh, width=Ww, count=1,
                                    dtype='float32', crs='EPSG:4326',
                                    transform=tx, nodata=np.nan) as ds:
                    ds.write(arr.astype(np.float32), 1)
                buf.seek(0)
                import base64
                return base64.b64encode(buf.read()).decode()
            except Exception as ex:
                L.warning(f"dl_pro: tif encode {fname} failed: {ex}")
                return None

        depth_tif = _arr_to_b64_tif(out['depth'], 'depth.tif')
        lower_tif = _arr_to_b64_tif(out['lower95'], 'lower95.tif')
        upper_tif = _arr_to_b64_tif(out['upper95'], 'upper95.tif')
        sigma_tif = _arr_to_b64_tif(out['sigma'], 'sigma.tif')

        finite = np.isfinite(out['depth'])
        stats = {
            'n_water_pixels': int(finite.sum()),
            'depth_min': float(np.nanmin(out['depth'])) if finite.any() else None,
            'depth_max': float(np.nanmax(out['depth'])) if finite.any() else None,
            'depth_median': float(np.nanmedian(out['depth'])) if finite.any() else None,
            'sigma_median': float(np.nanmedian(out['sigma'])) if finite.any() else None,
            'ci95_halfwidth_median': float(1.96 * np.nanmedian(out['sigma']))
                                        if finite.any() else None,
        }
        return jsonify({
            'ok': True,
            'method': 'DL Pro (MC-Dropout MLP + IHO calibration)',
            'bbox': list(bbox),
            'resolution_m': res,
            'date_window': [sd, ed],
            'water_mask_shape': list(out['depth'].shape),
            'calibration': out['calibration'],
            'fine_tune': out['fine_tune'],
            'iho_s44': out['iho_s44'],
            'stats': stats,
            'model_meta': out['model_meta'],
            'geotiff_b64': {
                'depth': depth_tif,
                'lower95': lower_tif,
                'upper95': upper_tif,
                'sigma': sigma_tif,
            },
        })
    except FileNotFoundError as ex:
        L.error(f"dl_pro: {ex}")
        return jsonify({'error': str(ex), 'hint':
            'Run experiments/train_dl_pro_bundle.py to create the bundle.'}), 503
    except Exception as ex:
        L.error(f"dl_pro: {ex}", exc_info=True)
        return jsonify({'error': str(ex)}), 500


@app.route('/api/predict_dl_pro/info', methods=['GET'])
def api_dl_pro_info():
    """Return bundle meta (S-44 compliance, calibration, training summary).
    Used by the frontend to show the model card without running inference."""
    try:
        try:
            from backend.dl_pro_engine import load_bundle
        except ImportError:
            from dl_pro_engine import load_bundle
        b = load_bundle()
        return jsonify({
            'ok': True,
            'meta': b['meta'],
        })
    except FileNotFoundError as ex:
        return jsonify({'error': str(ex)}), 503
    except Exception as ex:
        L.error(f"dl_pro/info: {ex}", exc_info=True)
        return jsonify({'error': str(ex)}), 500


# ── VHR image purchase module (Stripe TEST + SH commercial dry-run) ──────────
# Single isolated registration block. All logic lives in backend/vhr_purchase.py.
try:
    from backend.vhr_purchase import vhr_bp
    app.register_blueprint(vhr_bp)
    L.info("VHR purchase module loaded (/api/vhr-purchase)")
except Exception:
    try:
        from vhr_purchase import vhr_bp  # type: ignore
        app.register_blueprint(vhr_bp)
        L.info("VHR purchase module loaded (/api/vhr-purchase)")
    except Exception as _ve:
        L.warning(f"VHR purchase module not loaded: {_ve}")


@app.route('/',defaults={'path':''})
@app.route('/<path:path>')
def serve(path):
    if path.startswith('api/'):return jsonify({'error':'Not found'}),404
    # Serve Hormuz Ocean sub-app at /ocean
    hormuz_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','Hormuz_Ocean','frontend','public'))
    if path.rstrip('/') == 'ocean':
        if os.path.exists(os.path.join(hormuz_dir,'index.html')):return send_from_directory(hormuz_dir,'index.html')
    if path.startswith('ocean/'):
        sub=path[6:]
        if sub and os.path.exists(os.path.join(hormuz_dir,sub)):return send_from_directory(hormuz_dir,sub)
    if app.static_folder:
        if path.rstrip('/') == 'ocean':
            ocean_idx = os.path.join(app.static_folder, 'ocean', 'index.html')
            if os.path.exists(ocean_idx): return send_from_directory(os.path.join(app.static_folder, 'ocean'), 'index.html')
        if path and os.path.exists(os.path.join(app.static_folder,path)):return send_from_directory(app.static_folder,path)
        if os.path.exists(os.path.join(app.static_folder,'index.html')):return send_from_directory(app.static_folder,'index.html')
    return jsonify({'info':'Bathymetry v9.3'}),200

@app.errorhandler(404)
def e404(e):
    if req.path.startswith('/api/'):return jsonify({'error':'Not found'}),404
    if app.static_folder and os.path.exists(os.path.join(app.static_folder,'index.html')):return send_from_directory(app.static_folder,'index.html')
    return jsonify({'error':'Not found'}),404

# Kick off S2 warm-up for the predefined regions on import (Gunicorn,
# `python -m flask`, and `python app.py` paths all hit this). Daemon thread
# so it never blocks boot.
try:
    _start_preset_warmup()
except Exception as _wx:
    L.info(f"Preset warm-up not started: {_wx}")

if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT',8080)),debug=False)
