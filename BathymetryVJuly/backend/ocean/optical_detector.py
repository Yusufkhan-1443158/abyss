"""
Sentinel-2 Optical Vessel Detection + Wave Anomaly Analysis
════════════════════════════════════════════════════════════
1. Optical boat detection: bright objects on dark water (NIR band CFAR)
2. Wave coherence analysis: Radon-transform spectral method (Almar-inspired)
   to find regions where wave patterns are disrupted — indicates
   underwater objects, submarine wakes, or submerged infrastructure.

Physics:
  - Ocean waves are coherent plane waves (dominant swell + wind sea)
  - A submerged object disrupts the surface wave field locally
  - The Radon transform projects 2D→1D along each angle
  - In coherent wave regions, one angle dominates (wave direction)
  - In disrupted regions, energy is spread across angles = low coherence
  - Radon coherence ratio = max_angle_energy / mean_energy
  - Low ratio = anomalous wave field = possible underwater object
"""

import numpy as np
import os
import io
import base64
import time as TM
import requests
from datetime import datetime, timedelta
from scipy.ndimage import label, median_filter, gaussian_filter
from .world_land_mask import is_land_global as is_land, make_ocean_mask

# Sentinel Hub
SH_AUTH = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROC = "https://sh.dataspace.copernicus.eu/api/v1/process"
_tok = {"t": None, "exp": 0}


def _sh_token():
    if _tok["t"] and TM.time() < _tok["exp"] - 60: return _tok["t"]
    cid = os.getenv("SH_CLIENT_ID", ""); csec = os.getenv("SH_CLIENT_SECRET", "")
    if not cid: raise RuntimeError("SH_CLIENT_ID not set")
    r = requests.post(SH_AUTH, data={"grant_type": "client_credentials", "client_id": cid, "client_secret": csec}, timeout=30)
    r.raise_for_status(); d = r.json(); _tok["t"] = d["access_token"]; _tok["exp"] = TM.time() + d["expires_in"]; return _tok["t"]


def _roi_bbox(lat, lon, km=25):
    dlat = (km / 2) / 111.0; dlon = (km / 2) / (111.0 * np.cos(np.radians(lat)))
    return [round(lon - dlon, 6), round(lat - dlat, 6), round(lon + dlon, 6), round(lat + dlat, 6)]


# ═══════════════════════════════════════════════════════
# Sentinel-2 fetch (B02 blue, B03 green, B04 red, B08 NIR)
# ═══════════════════════════════════════════════════════
EVALSCRIPT_S2 = """//VERSION=3
function setup(){
  return{
    input:[{bands:["B02","B03","B04","B08","B11","SCL"],units:"DN",mosaicking:"ORBIT"}],
    output:{bands:6,sampleType:"UINT16"}
  };
}
function isValid(s){
  var c=s.SCL;
  return c!==0&&c!==1&&c!==3&&c!==8&&c!==9&&c!==10&&c!==11;
}
function evaluatePixel(samples){
  // Pick the best valid pixel across orbits (clearest water preferred)
  var best=null, bestScore=-1;
  for(var i=0;i<samples.length;i++){
    if(!isValid(samples[i])) continue;
    var sc=samples[i].SCL===6?100:samples[i].SCL===2?50:10;
    if(sc>bestScore){bestScore=sc;best=samples[i];}
  }
  if(!best){best=samples[0]||{B02:0,B03:0,B04:0,B08:0,B11:0,SCL:0};}
  return[best.B02,best.B03,best.B04,best.B08,best.B11,best.SCL];
}"""


def fetch_s2(bbox, start_date, end_date, resolution=10, cloud=10):
    import tifffile
    w, s, e, n = bbox
    cl = np.cos(np.radians((n + s) / 2))
    wp = max(32, min(2500, int(abs(e - w) * 111000 * cl / resolution)))
    hp = max(32, min(2500, int(abs(n - s) * 111000 / resolution)))
    tok = _sh_token()
    body = {"input": {"bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data": [{"type": "sentinel-2-l2a", "dataFilter": {"maxCloudCoverage": cloud,
            "timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"}},
            "processing": {"upsampling": "BILINEAR"}}]},
        "output": {"width": wp, "height": hp, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT_S2}
    print(f"[OPT] Fetch S2 {wp}x{hp} @{resolution}m {start_date}→{end_date}")
    r = requests.post(SH_PROC, headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}, json=body, timeout=300)
    if not r.ok: raise RuntimeError(f"S2 {r.status_code}: {r.text[:300]}")
    img = tifffile.imread(io.BytesIO(r.content))
    if img.ndim == 3 and img.shape[0] == 6:
        b02, b03, b04, b08, b11, scl = img[0], img[1], img[2], img[3], img[4], img[5]
    elif img.ndim == 3 and img.shape[2] == 6:
        b02, b03, b04, b08, b11, scl = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3], img[:,:,4], img[:,:,5]
    else:
        raise RuntimeError(f"S2 shape {img.shape}")
    print(f"[OPT] B02:[{b02.min()},{b02.max()}] B08:[{b08.min()},{b08.max()}] B11:[{b11.min()},{b11.max()}]")
    return {"b02": b02.astype(np.float32), "b03": b03.astype(np.float32),
            "b04": b04.astype(np.float32), "b08": b08.astype(np.float32),
            "b11": b11.astype(np.float32), "scl": scl.astype(np.uint8),
            "width": wp, "height": hp, "bbox": bbox}


# ═══════════════════════════════════════════════════════
# Optical vessel detection (NIR-based CFAR on water)
# ═══════════════════════════════════════════════════════
def _detect_optical_vessels(b08, b11, b03, scl, ocean_mask, resolution=10):
    """
    Strict multi-band vessel detection on clear ocean pixels.

    Strategy:
    1. SCL strict filter: only keep class 6 (water) — reject cloud, shadow, land, cirrus
    2. NIR CFAR: bright objects on dark water (6σ threshold)
    3. SWIR confirmation: real vessels reflect in SWIR too, foam/glint don't
    4. NDWI cross-check: vessel pixels have NDWI < 0 (more NIR than green)
    5. Minimum cluster size: ≥ 3 pixels (30m @ 10m res = real object)
    """
    h, w = b08.shape
    nir = b08.astype(np.float32)
    swir = b11.astype(np.float32)
    green = b03.astype(np.float32)

    # Strict SCL: ONLY water (6) — reject everything else
    # SCL classes: 0=nodata, 1=saturated, 2=dark, 3=shadow, 4=veg, 5=bare,
    #              6=water, 7=unclass, 8=cloud_med, 9=cloud_high, 10=cirrus, 11=snow
    scl_water = (scl == 6)
    clean_ocean = ocean_mask & scl_water & (nir > 0) & (swir > 0)
    clean_pct = float(clean_ocean.sum() / max(ocean_mask.sum(), 1) * 100)
    print(f"[OPT] Clean water (SCL=6): {clean_ocean.sum()} px ({clean_pct:.0f}% of ocean)")

    # Fallback: if SCL is too strict (< 30% clean), relax to include class 2,7
    if clean_pct < 30:
        clean_ocean = ocean_mask & np.isin(scl, [2, 6, 7]) & (nir > 0)
        clean_pct = float(clean_ocean.sum() / max(ocean_mask.sum(), 1) * 100)
        print(f"[OPT] Relaxed SCL (2,6,7): {clean_ocean.sum()} px ({clean_pct:.0f}%)")

    nir[~clean_ocean] = np.nan
    swir[~clean_ocean] = np.nan

    # NIR stats on clean water — use robust estimators (MAD not std)
    nir_ocean = nir[clean_ocean]
    if len(nir_ocean) < 100:
        print("[OPT] Too few clean water pixels, skipping vessel detection")
        return np.zeros_like(nir), np.zeros(nir.shape, dtype=bool), clean_ocean

    ocean_med = float(np.nanmedian(nir_ocean))
    ocean_mad = float(np.nanmedian(np.abs(nir_ocean - ocean_med)))  # MAD
    ocean_robust_std = ocean_mad * 1.4826  # scale MAD to equivalent σ
    print(f"[OPT] NIR water: median={ocean_med:.0f}, MAD={ocean_mad:.0f}, robust_σ={ocean_robust_std:.0f}")

    # Remove sun glint: glint is specular reflection → NIR high but SWIR even higher
    # Glint ratio: B11/B08 > 0.8 for glint, < 0.5 for vessels
    swir_filled = swir.copy(); swir_filled[np.isnan(swir_filled)] = float(np.nanmedian(swir[clean_ocean]) if clean_ocean.any() else 0)
    nir_filled = nir.copy(); nir_filled[np.isnan(nir_filled)] = ocean_med
    glint_ratio = swir_filled / (nir_filled + 1e-6)
    not_glint = glint_ratio < 0.7  # vessels reflect more in NIR than SWIR

    # CFAR on NIR: local background subtraction + strict threshold
    bg = median_filter(nir_filled, size=51)
    tcr = nir_filled - bg
    tcr[~clean_ocean] = 0

    # Use robust σ for threshold — immune to outliers/glint
    thresh = ocean_robust_std * 10  # 10× robust sigma above background
    abs_nir_thresh = ocean_med + ocean_robust_std * 15  # absolute floor
    nir_bright = (tcr > thresh) & (nir_filled > abs_nir_thresh) & clean_ocean

    # SWIR confirmation (robust)
    swir_ocean = swir[clean_ocean]
    swir_med = float(np.nanmedian(swir_ocean)) if len(swir_ocean) > 0 else 0
    swir_mad = float(np.nanmedian(np.abs(swir_ocean - swir_med)))
    swir_robust_std = swir_mad * 1.4826
    swir_bright = (swir_filled > swir_med + 8 * swir_robust_std) & clean_ocean

    # NDWI: vessels have NDWI < -0.1 (clearly more NIR than green)
    ndwi = (green - nir_filled) / (green + nir_filled + 1e-6)
    vessel_ndwi = ndwi < -0.1

    # Combine: ALL must agree + not glint
    detected = nir_bright & swir_bright & vessel_ndwi & not_glint & clean_ocean

    det_px = int(detected.sum())
    print(f"[OPT] Detection: NIR 6σ thresh={thresh:.0f}, "
          f"NIR bright={nir_bright.sum()}, SWIR confirm={swir_bright.sum()}, "
          f"final={det_px} px")

    return tcr, detected, clean_ocean


# ═══════════════════════════════════════════════════════
# Wave coherence analysis (Radon transform, Almar-inspired)
# ═══════════════════════════════════════════════════════
def _wave_coherence(b02, ocean_mask, tile_size=128, step=64):
    """
    Radon-transform wave coherence map.

    For each tile:
    1. Detrend (remove mean) → surface texture only
    2. 2D FFT → power spectrum
    3. Radon-like angular integration: sum power along each direction
    4. Coherence = max_direction / mean_all_directions
       - High coherence = clean swell (one dominant direction)
       - Low coherence = disrupted wave field → anomaly

    Returns: coherence map (same size as input, interpolated from tiles)
    """
    h, w = b02.shape
    blue = b02.astype(np.float32)
    blue[~ocean_mask] = np.nan

    n_angles = 36  # every 5 degrees
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    # Tile grid
    tile_rows = list(range(0, h - tile_size + 1, step))
    tile_cols = list(range(0, w - tile_size + 1, step))
    coh_map = np.full((len(tile_rows), len(tile_cols)), np.nan)
    dom_dir = np.full((len(tile_rows), len(tile_cols)), np.nan)
    dom_wl = np.full((len(tile_rows), len(tile_cols)), np.nan)

    # Hanning window for FFT
    win = np.outer(np.hanning(tile_size), np.hanning(tile_size))

    for ti, r0 in enumerate(tile_rows):
        for tj, c0 in enumerate(tile_cols):
            tile = blue[r0:r0+tile_size, c0:c0+tile_size].copy()
            mask_tile = ocean_mask[r0:r0+tile_size, c0:c0+tile_size]

            # Skip tiles with > 30% land/cloud
            if mask_tile.sum() < 0.7 * tile_size * tile_size:
                continue

            # Fill NaN for FFT
            tile_med = np.nanmedian(tile)
            tile[np.isnan(tile)] = tile_med if not np.isnan(tile_med) else 0
            tile = tile - tile.mean()  # detrend

            # 2D FFT
            fft2 = np.fft.fft2(tile * win)
            power = np.abs(np.fft.fftshift(fft2)) ** 2
            cy, cx = tile_size // 2, tile_size // 2

            # Angular energy distribution
            Y, X = np.mgrid[-cy:tile_size-cy, -cx:tile_size-cx]
            R = np.sqrt(X**2 + Y**2)
            Theta = np.arctan2(Y, X) % np.pi  # 0 to pi

            # Only consider wavelengths 20-200m (frequencies in FFT space)
            min_freq = tile_size / (200 / 10)  # 200m wavelength at 10m resolution
            max_freq = tile_size / (20 / 10)   # 20m wavelength
            freq_mask = (R >= min_freq) & (R <= max_freq)

            angular_energy = np.zeros(n_angles)
            for ai, angle in enumerate(angles):
                # Wedge: ±5 degrees around this angle
                angle_diff = np.abs(Theta - angle)
                angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
                wedge = (angle_diff < np.pi / n_angles) & freq_mask
                if wedge.sum() > 0:
                    angular_energy[ai] = np.mean(power[wedge])

            if angular_energy.max() > 0:
                coherence = angular_energy.max() / (np.mean(angular_energy) + 1e-10)
                coh_map[ti, tj] = float(coherence)
                dom_dir[ti, tj] = float(np.degrees(angles[np.argmax(angular_energy)]))
                # Dominant wavelength
                best_angle = angles[np.argmax(angular_energy)]
                angle_diff = np.abs(Theta - best_angle)
                angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
                wedge = (angle_diff < np.pi / n_angles) & freq_mask
                if wedge.sum() > 0:
                    radial_power = np.zeros(tile_size // 2)
                    for ri in range(1, tile_size // 2):
                        ring = wedge & (np.abs(R - ri) < 1)
                        if ring.sum() > 0:
                            radial_power[ri] = np.mean(power[ring])
                    peak_freq = np.argmax(radial_power[1:]) + 1
                    if peak_freq > 0:
                        dom_wl[ti, tj] = float(tile_size * 10 / peak_freq)  # wavelength in meters

    print(f"[WAVE] Coherence: {(~np.isnan(coh_map)).sum()} tiles computed")
    return coh_map, dom_dir, dom_wl, tile_rows, tile_cols, tile_size, step


def _find_wave_anomalies(coh_map, dom_dir, dom_wl, tile_rows, tile_cols,
                          tile_size, step, lats, lons, ocean_mask, resolution,
                          sigma_factor=2.0, min_coherence=1.5):
    """Find tiles with anomalously low wave coherence.
    sigma_factor: how many std below mean = anomaly (default 2.0, lower = more sensitive)
    min_coherence: absolute floor for anomaly threshold (default 1.5)
    """
    valid = ~np.isnan(coh_map)
    if valid.sum() < 5:
        return []

    vals = coh_map[valid]
    mean_coh = float(np.mean(vals))
    std_coh = float(np.std(vals))
    thresh = max(min_coherence, mean_coh - sigma_factor * std_coh)

    anomalies = []
    for ti in range(coh_map.shape[0]):
        for tj in range(coh_map.shape[1]):
            if np.isnan(coh_map[ti, tj]):
                continue
            if coh_map[ti, tj] < thresh:
                r0, c0 = tile_rows[ti], tile_cols[tj]
                cr, cc = r0 + tile_size // 2, c0 + tile_size // 2
                if cr < len(lats) and cc < len(lons):
                    lat, lon = float(lats[cr]), float(lons[cc])
                    if is_land(lon, lat):
                        continue
                    anomalies.append({
                        "lat": round(lat, 5), "lon": round(lon, 5),
                        "coherence": round(float(coh_map[ti, tj]), 2),
                        "mean_coherence": round(mean_coh, 2),
                        "dominant_direction_deg": round(float(dom_dir[ti, tj]), 1) if not np.isnan(dom_dir[ti, tj]) else None,
                        "dominant_wavelength_m": round(float(dom_wl[ti, tj]), 1) if not np.isnan(dom_wl[ti, tj]) else None,
                        "anomaly_strength": round(float((mean_coh - coh_map[ti, tj]) / max(std_coh, 0.1)), 2),
                        "type": "wave_disruption",
                    })

    anomalies.sort(key=lambda a: a["coherence"])
    for i, a in enumerate(anomalies): a["id"] = i + 1
    print(f"[WAVE] {len(anomalies)} wave anomalies (thresh={thresh:.2f}, mean={mean_coh:.2f})")
    return anomalies


# ═══════════════════════════════════════════════════════
# Main optical pipeline
# ═══════════════════════════════════════════════════════
def detect_optical(bbox, start_date, end_date, resolution=10,
                    wave_sigma=2.0, wave_min_coherence=1.5):
    """
    Full optical detection: vessel CFAR + wave coherence anomaly.
    wave_sigma: std factor for anomaly detection (lower = more sensitive)
    wave_min_coherence: absolute coherence floor for anomalies
    """
    s2 = fetch_s2(bbox, start_date, end_date, resolution)
    b02, b03, b04, b08, b11, scl = s2["b02"], s2["b03"], s2["b04"], s2["b08"], s2["b11"], s2["scl"]
    h, w = b02.shape
    west, south, east, north = bbox
    lats = np.linspace(north, south, h)
    lons = np.linspace(west, east, w)

    # Ocean mask (land only — cloud filtering is done in detector)
    print("[OPT] Ocean mask (Natural Earth 10m)...")
    ocean_geo = make_ocean_mask(lons, lats)
    ocean_mask = ocean_geo & (b08 > 0)
    ocean_pct = float(ocean_mask.sum() / (h * w) * 100)
    print(f"[OPT] Ocean (geo): {ocean_mask.sum()}/{h*w} ({ocean_pct:.0f}%)")

    # ─── Optical vessel detection (strict) ───────────
    print("[OPT] Strict multi-band vessel detection...")
    tcr, detected, clean_ocean = _detect_optical_vessels(b08, b11, b03, scl, ocean_mask, resolution)

    labeled, n_clusters = label(detected)
    vessels = []
    for c in range(1, n_clusters + 1):
        cluster = labeled == c
        n_pix = int(cluster.sum())
        if n_pix < 3: continue  # min 3 pixels = 30m real object
        rows, cols = np.where(cluster)
        nir_vals = b08[cluster].astype(float)
        pi = np.argmax(nir_vals)
        pr, pc = int(rows[pi]), int(cols[pi])
        lat, lon = float(lats[pr]), float(lons[pc])
        if is_land(lon, lat): continue

        length_m = float(int(rows.max() - rows.min() + 1) * resolution)
        width_m = float(int(cols.max() - cols.min() + 1) * resolution)
        # NDWI at target (Green-NIR)/(Green+NIR) — should be negative for vessels
        g = float(b03[pr, pc]); n = float(b08[pr, pc])
        ndwi = (g - n) / (g + n + 1e-6)

        # Make RGB thumbnail
        thumb = _make_rgb_thumbnail(b04, b03, b02, pr, pc, clean_ocean, crop=60)

        vessels.append({
            "id": len(vessels) + 1,
            "lat": round(lat, 5), "lon": round(lon, 5),
            "type": "optical_vessel",
            "length_m": round(length_m, 1), "width_m": round(width_m, 1),
            "n_pixels": n_pix,
            "peak_nir": round(float(nir_vals.max()), 0),
            "ndwi": round(ndwi, 3),
            "thumbnail": thumb,
        })

    vessels.sort(key=lambda v: v["peak_nir"], reverse=True)
    for i, v in enumerate(vessels): v["id"] = i + 1
    print(f"[OPT] {len(vessels)} optical vessels")

    # ─── Wave coherence analysis ─────────────────────
    print("[OPT] Wave coherence (Radon-FFT)...")
    coh_map, dom_dir, dom_wl, tile_rows, tile_cols, ts, step = _wave_coherence(
        b02, clean_ocean, tile_size=128, step=64)
    wave_anomalies = _find_wave_anomalies(
        coh_map, dom_dir, dom_wl, tile_rows, tile_cols, ts, step,
        lats, lons, clean_ocean, resolution,
        sigma_factor=wave_sigma, min_coherence=wave_min_coherence)

    # ─── Build display layers ────────────────────────
    max_disp = 500
    dh, dw = min(h, max_disp), min(w, max_disp)
    from scipy.ndimage import zoom as szoom

    def ds(arr):
        if arr.shape[0] <= dh and arr.shape[1] <= dw: return arr.copy()
        m = np.isnan(arr); filled = arr.copy(); filled[m] = 0
        out = szoom(filled, (dh / arr.shape[0], dw / arr.shape[1]), order=1)
        mds = szoom(m.astype(float), (dh / arr.shape[0], dw / arr.shape[1]), order=0) > 0.5
        out[mds] = np.nan; return out

    # RGB composite (true color)
    rgb_r = b04.astype(float); rgb_g = b03.astype(float); rgb_b = b02.astype(float)
    # Normalize to 0-1 (typical S2 L2A reflectance * 10000)
    for ch in [rgb_r, rgb_g, rgb_b]:
        ch[~clean_ocean] = np.nan
    rgb_brightness = np.nanpercentile(np.stack([rgb_r, rgb_g, rgb_b]), 98)
    if rgb_brightness > 0:
        rgb_r = np.clip(rgb_r / rgb_brightness, 0, 1)
        rgb_g = np.clip(rgb_g / rgb_brightness, 0, 1)
        rgb_b = np.clip(rgb_b / rgb_brightness, 0, 1)

    rgb_r_ds, rgb_g_ds, rgb_b_ds = ds(rgb_r), ds(rgb_g), ds(rgb_b)
    dha, dwa = rgb_r_ds.shape
    lats_out = np.linspace(south, north, dha).tolist()
    lons_out = np.linspace(west, east, dwa).tolist()

    def to_mat_rgb(r, g, b):
        """Encode RGB as single matrix: R*65536 + G*256 + B (decoded in frontend)."""
        ny, nx = r.shape; out = []
        for i in range(ny):
            ri = ny - 1 - i; row = []
            for j in range(nx):
                rv, gv, bv = r[ri, j], g[ri, j], b[ri, j]
                if np.isnan(rv): row.append(None)
                else: row.append(int(min(255, rv*255)) * 65536 + int(min(255, gv*255)) * 256 + int(min(255, bv*255)))
            out.append(row)
        return out

    # Wave coherence as display layer
    coh_full = np.full((h, w), np.nan)
    for ti in range(coh_map.shape[0]):
        for tj in range(coh_map.shape[1]):
            if not np.isnan(coh_map[ti, tj]):
                r0, c0 = tile_rows[ti], tile_cols[tj]
                coh_full[r0:r0+ts, c0:c0+ts] = coh_map[ti, tj]
    coh_full[~clean_ocean] = np.nan
    valid_coh = coh_full[~np.isnan(coh_full)]
    coh_min = float(np.percentile(valid_coh, 2)) if len(valid_coh) > 0 else 0
    coh_max = float(np.percentile(valid_coh, 98)) if len(valid_coh) > 0 else 1
    coh_norm = np.clip((coh_full - coh_min) / max(coh_max - coh_min, 0.1), 0, 1)
    coh_ds = ds(coh_norm)

    def to_mat(arr):
        ny, nx = arr.shape; out = []
        for i in range(ny):
            ri = ny - 1 - i; row = []
            for j in range(nx):
                v = float(arr[ri, j]); row.append(None if np.isnan(v) else round(v, 4))
            out.append(row)
        return out

    # Sanitize
    import math
    def _san(obj):
        if isinstance(obj, dict): return {k: _san(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [_san(v) for v in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
        return obj

    result = {
        "layers": {
            "rgb": {"lats": lats_out, "lons": lons_out, "matrix": to_mat_rgb(rgb_r_ds, rgb_g_ds, rgb_b_ds),
                    "ny": dha, "nx": dwa, "min_val": 0, "max_val": 1, "encoding": "rgb_packed"},
            "wave_coherence": {"lats": lats_out, "lons": lons_out, "matrix": to_mat(coh_ds),
                               "ny": dha, "nx": dwa, "min_val": 0, "max_val": 1},
        },
        "vessels": _san(vessels),
        "wave_anomalies": _san(wave_anomalies),
        "metadata": {
            "start_date": start_date, "end_date": end_date,
            "resolution_m": resolution, "grid": f"{h}x{w}", "display_grid": f"{dha}x{dwa}",
            "bbox": [float(x) for x in bbox], "ocean_pct": round(ocean_pct, 1),
            "n_vessels": len(vessels), "n_wave_anomalies": len(wave_anomalies),
            "wave_coherence_range": [round(coh_min, 2), round(coh_max, 2)],
        },
    }
    print(f"[OPT] Done: {len(vessels)} vessels, {len(wave_anomalies)} wave anomalies")
    return result


def _make_rgb_thumbnail(r, g, b, pr, pc, ocean_mask, crop=60):
    """RGB thumbnail around target."""
    from PIL import Image
    h, w = r.shape
    r0 = max(0, pr-crop); r1 = min(h, pr+crop)
    c0 = max(0, pc-crop); c1 = min(w, pc+crop)
    rp, gp, bp = r[r0:r1,c0:c1].astype(float), g[r0:r1,c0:c1].astype(float), b[r0:r1,c0:c1].astype(float)
    mp = ocean_mask[r0:r1,c0:c1]
    mx = max(np.nanpercentile(rp[mp], 98) if mp.any() else 1, 1)
    rn = np.clip(rp/mx*255, 0, 255).astype(np.uint8)
    gn = np.clip(gp/mx*255, 0, 255).astype(np.uint8)
    bn = np.clip(bp/mx*255, 0, 255).astype(np.uint8)
    rn[~mp] = 0; gn[~mp] = 0; bn[~mp] = 0
    rgb = np.stack([rn, gn, bn], axis=-1)
    img = Image.fromarray(rgb)
    img = img.resize((img.width*2, img.height*2), Image.NEAREST)
    buf = io.BytesIO(); img.save(buf, format='PNG', optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
