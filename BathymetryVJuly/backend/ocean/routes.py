"""
Ocean monitoring Blueprint — Hormuz currents, SST, mine detection, SAR vessel detection.
Registers all /api/ocean/* routes.
"""

import json
import os
import datetime
import numpy as np
from flask import Blueprint, jsonify, request, send_from_directory

# Load Sentinel Hub credentials from .env
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from .data_fetcher import get_cached_or_fetch
from .land_mask import get_land_mask_json
from .anomaly_detector import detect_anomalies
from .sentinel_detector import detect_vessels, get_default_bbox, get_default_dates, _roi_bbox
from .optical_detector import detect_optical

ocean_bp = Blueprint("ocean", __name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocean")
os.makedirs(DATA_DIR, exist_ok=True)

# Strait of Hormuz bounding box
HORMUZ_BBOX = {
    "lat_min": 24.5,
    "lat_max": 27.5,
    "lon_min": 54.0,
    "lon_max": 58.0,
}


@ocean_bp.route("/api/currents")
def get_currents():
    date_str = request.args.get("date", datetime.date.today().isoformat())
    depth = float(request.args.get("depth", 0))
    try:
        data = get_cached_or_fetch("currents", date_str, depth, HORMUZ_BBOX, DATA_DIR)
        return jsonify(data)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ocean_bp.route("/api/sst")
def get_sst():
    date_str = request.args.get("date", datetime.date.today().isoformat())
    try:
        data = get_cached_or_fetch("sst", date_str, 0, HORMUZ_BBOX, DATA_DIR)
        return jsonify(data)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ocean_bp.route("/api/sst_anomaly")
def get_sst_anomaly():
    date_str = request.args.get("date", datetime.date.today().isoformat())
    try:
        data = get_cached_or_fetch("sst_anomaly", date_str, 0, HORMUZ_BBOX, DATA_DIR)
        return jsonify(data)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ocean_bp.route("/api/detect")
def run_detection():
    n_days = int(request.args.get("days", 30))
    cache_file = os.path.join(DATA_DIR, f"detection_{n_days}d.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return jsonify(json.load(f))
    try:
        result = detect_anomalies(HORMUZ_BBOX, n_days=n_days)
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ocean_bp.route("/api/landmask")
def get_landmask():
    try:
        data = get_land_mask_json(HORMUZ_BBOX)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ocean_bp.route("/api/bbox")
def get_bbox():
    return jsonify(HORMUZ_BBOX)


@ocean_bp.route("/api/sentinel")
def run_sentinel_detection():
    default_start, default_end = get_default_dates()
    start_date = request.args.get("start_date", default_start)
    end_date = request.args.get("end_date", default_end)
    lat = float(request.args.get("lat", 24.81))
    lon = float(request.args.get("lon", 54.60))
    roi_km = float(request.args.get("roi_km", 25))

    cache_key = f"sentinel_{start_date}_{end_date}_{lat}_{lon}_{roi_km}"
    cache_file = os.path.join(DATA_DIR, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return jsonify(json.load(f))

    try:
        bbox = _roi_bbox(lat, lon, roi_km)
        result = detect_vessels(bbox, start_date, end_date)
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ocean_bp.route("/api/optical")
def run_optical_detection():
    """Optical vessel detection + wave coherence anomaly (Sentinel-2)."""
    default_start, default_end = get_default_dates()
    start_date = request.args.get("start_date", default_start)
    end_date = request.args.get("end_date", default_end)
    lat = float(request.args.get("lat", 24.81))
    lon = float(request.args.get("lon", 54.60))
    roi_km = float(request.args.get("roi_km", 25))
    wave_sigma = float(request.args.get("wave_sigma", 2.0))
    wave_min_coh = float(request.args.get("wave_min_coherence", 1.5))

    cache_key = f"optical_{start_date}_{end_date}_{lat}_{lon}_{roi_km}_ws{wave_sigma}_wc{wave_min_coh}"
    cache_file = os.path.join(DATA_DIR, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return jsonify(json.load(f))

    try:
        bbox = _roi_bbox(lat, lon, roi_km)
        result = detect_optical(bbox, start_date, end_date,
                                wave_sigma=wave_sigma, wave_min_coherence=wave_min_coh)
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ocean_bp.route("/api/optical-scan")
def run_optical_scan():
    """
    Full Strait of Hormuz tiled optical scan.
    Splits the strait into sub-images (tiles), processes each independently,
    merges all vessels + wave anomalies into a single result set.
    GPU-parallel where available.
    """
    default_start, default_end = get_default_dates()
    start_date = request.args.get("start_date", default_start)
    end_date = request.args.get("end_date", default_end)
    tile_km = float(request.args.get("tile_km", 40))
    wave_sigma = float(request.args.get("wave_sigma", 2.0))
    wave_min_coh = float(request.args.get("wave_min_coherence", 1.5))

    # Custom bbox or default full Hormuz
    lat_min = float(request.args.get("lat_min", HORMUZ_BBOX["lat_min"]))
    lat_max = float(request.args.get("lat_max", HORMUZ_BBOX["lat_max"]))
    lon_min = float(request.args.get("lon_min", HORMUZ_BBOX["lon_min"]))
    lon_max = float(request.args.get("lon_max", HORMUZ_BBOX["lon_max"]))

    cache_key = f"scan_{start_date}_{end_date}_{lat_min}_{lat_max}_{lon_min}_{lon_max}_t{tile_km}_ws{wave_sigma}"
    cache_file = os.path.join(DATA_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return jsonify(json.load(f))

    import math
    try:
        # Build tile grid
        dlat = tile_km / 111.0
        dlon = tile_km / (111.0 * np.cos(np.radians((lat_min + lat_max) / 2)))
        lat_steps = np.arange(lat_min, lat_max, dlat)
        lon_steps = np.arange(lon_min, lon_max, dlon)
        n_tiles = len(lat_steps) * len(lon_steps)

        all_vessels = []
        all_wave_anomalies = []
        tile_results = []
        processed = 0
        failed = 0

        print(f"[SCAN] Full Hormuz scan: {n_tiles} tiles ({len(lat_steps)}x{len(lon_steps)}) @ {tile_km}km")

        for i, lat0 in enumerate(lat_steps):
            for j, lon0 in enumerate(lon_steps):
                lat1 = min(lat0 + dlat, lat_max)
                lon1 = min(lon0 + dlon, lon_max)
                bbox = [round(lon0, 5), round(lat0, 5), round(lon1, 5), round(lat1, 5)]
                tile_id = f"T{i:02d}{j:02d}"

                try:
                    print(f"[SCAN] Tile {tile_id} ({processed+1}/{n_tiles}): {bbox}")
                    result = detect_optical(bbox, start_date, end_date,
                                            wave_sigma=wave_sigma,
                                            wave_min_coherence=wave_min_coh)

                    # Collect vessels with tile ID
                    for v in result.get("vessels", []):
                        v["tile_id"] = tile_id
                        v["tile_bbox"] = bbox
                        all_vessels.append(v)

                    # Collect wave anomalies
                    for wa in result.get("wave_anomalies", []):
                        wa["tile_id"] = tile_id
                        wa["tile_bbox"] = bbox
                        all_wave_anomalies.append(wa)

                    tile_results.append({
                        "tile_id": tile_id,
                        "bbox": bbox,
                        "n_vessels": len(result.get("vessels", [])),
                        "n_wave_anomalies": len(result.get("wave_anomalies", [])),
                        "ocean_pct": result.get("metadata", {}).get("ocean_pct", 0),
                        "status": "ok",
                    })
                    processed += 1
                except Exception as te:
                    print(f"[SCAN] Tile {tile_id} failed: {te}")
                    tile_results.append({
                        "tile_id": tile_id, "bbox": bbox,
                        "n_vessels": 0, "n_wave_anomalies": 0,
                        "status": "failed", "error": str(te)[:100],
                    })
                    failed += 1

        # Re-index merged results
        all_vessels.sort(key=lambda v: v.get("peak_nir", 0), reverse=True)
        for i, v in enumerate(all_vessels): v["id"] = i + 1
        all_wave_anomalies.sort(key=lambda a: a.get("coherence", 99))
        for i, a in enumerate(all_wave_anomalies): a["id"] = i + 1

        # Deduplicate nearby detections (within 200m)
        def _dedup(items, dist_m=200):
            kept = []
            for item in items:
                duplicate = False
                for k in kept:
                    dlat = abs(item["lat"] - k["lat"]) * 111000
                    dlon = abs(item["lon"] - k["lon"]) * 111000 * np.cos(np.radians(item["lat"]))
                    if math.sqrt(dlat**2 + dlon**2) < dist_m:
                        duplicate = True; break
                if not duplicate:
                    kept.append(item)
            return kept

        all_vessels = _dedup(all_vessels, 200)
        all_wave_anomalies = _dedup(all_wave_anomalies, 500)
        for i, v in enumerate(all_vessels): v["id"] = i + 1
        for i, a in enumerate(all_wave_anomalies): a["id"] = i + 1

        scan_result = {
            "vessels": all_vessels,
            "wave_anomalies": all_wave_anomalies,
            "tiles": tile_results,
            "metadata": {
                "scan_type": "full_hormuz_tiled",
                "start_date": start_date, "end_date": end_date,
                "bbox": [lon_min, lat_min, lon_max, lat_max],
                "tile_km": tile_km,
                "n_tiles": n_tiles, "processed": processed, "failed": failed,
                "n_vessels": len(all_vessels),
                "n_wave_anomalies": len(all_wave_anomalies),
                "wave_sigma": wave_sigma,
                "wave_min_coherence": wave_min_coh,
            },
        }

        print(f"[SCAN] Done: {len(all_vessels)} vessels, {len(all_wave_anomalies)} wave anomalies across {processed}/{n_tiles} tiles")

        with open(cache_file, "w") as f:
            json.dump(scan_result, f)
        return jsonify(scan_result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
