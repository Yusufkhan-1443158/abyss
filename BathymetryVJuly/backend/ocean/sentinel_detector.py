"""
Sentinel-1 SAR Maritime Object Detection — Multi-scale CFAR + NMS
═════════════════════════════════════════════════════════════════
GPU-accelerated (PyTorch) multi-scale CFAR with Non-Maximum Suppression.
10m native Sentinel-1 GRD IW resolution.
Per-target SAR thumbnail crops (base64 PNG) for zoom popups.
Results cached as Zarr arrays.
"""

import numpy as np
import requests
import os
import io
import base64
import time as TM
from datetime import datetime, timedelta
from scipy.ndimage import label, median_filter, gaussian_filter
from .world_land_mask import is_land_global as is_land, make_ocean_mask

try:
    import torch
    import torch.nn.functional as F
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU = torch.cuda.is_available()
    print(f"[SAR] PyTorch: {DEVICE} ({'GPU' if GPU else 'CPU'})")
except ImportError:
    torch = None; F = None; DEVICE = None; GPU = False

try:
    import zarr; ZARR_OK = True
except ImportError:
    ZARR_OK = False

SH_AUTH = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROC = "https://sh.dataspace.copernicus.eu/api/v1/process"
_tok = {"t": None, "exp": 0}
KHALIFA_PORT = {"lat": 24.81, "lon": 54.60}
ROI_KM = 25


def _sh_token():
    if _tok["t"] and TM.time() < _tok["exp"] - 60: return _tok["t"]
    cid = os.getenv("SH_CLIENT_ID", ""); csec = os.getenv("SH_CLIENT_SECRET", "")
    if not cid: raise RuntimeError("SH_CLIENT_ID not set")
    r = requests.post(SH_AUTH, data={"grant_type": "client_credentials", "client_id": cid, "client_secret": csec}, timeout=30)
    r.raise_for_status(); d = r.json(); _tok["t"] = d["access_token"]; _tok["exp"] = TM.time() + d["expires_in"]; return _tok["t"]


def _roi_bbox(center_lat, center_lon, km=25):
    dlat = (km / 2) / 111.0; dlon = (km / 2) / (111.0 * np.cos(np.radians(center_lat)))
    return [round(center_lon - dlon, 6), round(center_lat - dlat, 6), round(center_lon + dlon, 6), round(center_lat + dlat, 6)]


EVALSCRIPT_SAR = """//VERSION=3
function setup(){return{input:[{bands:["VV","VH"],units:"LINEAR_POWER"}],output:{bands:2,sampleType:"FLOAT32"}};}
function evaluatePixel(s){return[s.VV,s.VH];}"""


def fetch_sar(bbox, start_date, end_date, resolution=10):
    import tifffile
    w, s, e, n = bbox
    cl = np.cos(np.radians((n + s) / 2))
    wp = max(32, min(2500, int(abs(e - w) * 111000 * cl / resolution)))
    hp = max(32, min(2500, int(abs(n - s) * 111000 / resolution)))
    tok = _sh_token()
    body = {"input": {"bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data": [{"type": "sentinel-1-grd", "dataFilter": {"timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"},
            "acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH"},
            "processing": {"orthorectify": True, "backCoeff": "SIGMA0_ELLIPSOID"}}]},
        "output": {"width": wp, "height": hp, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT_SAR}
    print(f"[SAR] Fetch {wp}x{hp} @{resolution}m {start_date}→{end_date}")
    r = requests.post(SH_PROC, headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}, json=body, timeout=300)
    if not r.ok: raise RuntimeError(f"SAR {r.status_code}: {r.text[:300]}")
    img = tifffile.imread(io.BytesIO(r.content))
    if img.ndim == 3 and img.shape[0] == 2: vv_lin, vh_lin = img[0], img[1]
    elif img.ndim == 3 and img.shape[2] == 2: vv_lin, vh_lin = img[:, :, 0], img[:, :, 1]
    else: raise RuntimeError(f"SAR shape {img.shape}")
    vv_db = (10 * np.log10(np.maximum(vv_lin, 1e-10))).astype(np.float32)
    vh_db = (10 * np.log10(np.maximum(vh_lin, 1e-10))).astype(np.float32)
    print(f"[SAR] VV:[{vv_db.min():.1f},{vv_db.max():.1f}]dB VH:[{vh_db.min():.1f},{vh_db.max():.1f}]dB")
    return {"vv_db": vv_db, "vh_db": vh_db, "width": wp, "height": hp, "bbox": bbox}


# ══════════════════════════════════════════════════════
# Multi-scale CFAR + NMS (YOLO-inspired)
# ══════════════════════════════════════════════════════
def _cfar_multiscale(vv, ocean_mask):
    """
    Multi-scale CFAR — run at 3 scales like YOLO's multi-scale detection.
    Small guard/bg catches small boats, large catches big vessels.
    Merge with Non-Maximum Suppression.
    """
    scales = [
        {"guard": 3, "bg": 15, "pfa": 5e-4, "label": "fine"},    # small boats
        {"guard": 5, "bg": 25, "pfa": 1e-4, "label": "medium"},   # medium vessels
        {"guard": 8, "bg": 40, "pfa": 5e-5, "label": "coarse"},   # large ships
    ]

    ocean_med = float(np.nanmedian(vv[ocean_mask])) if ocean_mask.any() else -20.0

    if torch is not None:
        return _cfar_multiscale_torch(vv, ocean_mask, ocean_med, scales)
    else:
        return _cfar_multiscale_scipy(vv, ocean_mask, ocean_med, scales)


def _cfar_multiscale_torch(vv, ocean_mask, ocean_med, scales):
    from scipy.stats import norm
    h, w = vv.shape
    t_vv = torch.from_numpy(vv.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    t_mask = torch.from_numpy(ocean_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    t_vv_filled = t_vv * t_mask + ocean_med * (1 - t_mask)

    def box_sum(x, k):
        pad = k // 2
        return F.avg_pool2d(F.pad(x, [pad]*4, mode='reflect'), k, stride=1) * (k * k)

    combined_tcr = torch.zeros(1, 1, h, w, device=DEVICE)
    combined_det = torch.zeros(1, 1, h, w, dtype=torch.bool, device=DEVICE)
    confidence = torch.zeros(1, 1, h, w, device=DEVICE)

    for sc in scales:
        g, b = sc["guard"], sc["bg"]
        alpha = float(norm.ppf(1 - sc["pfa"]))
        bg_sum = box_sum(t_vv_filled, 2*b+1) - box_sum(t_vv_filled, 2*g+1)
        bg_cnt = torch.clamp(box_sum(t_mask, 2*b+1) - box_sum(t_mask, 2*g+1), min=1)
        bg_mean = bg_sum / bg_cnt
        bg_var = box_sum(t_vv_filled**2, 2*b+1)/torch.clamp(box_sum(t_mask, 2*b+1), min=1) - (box_sum(t_vv_filled, 2*b+1)/torch.clamp(box_sum(t_mask, 2*b+1), min=1))**2
        bg_std = torch.sqrt(torch.clamp(bg_var, min=0.01))
        thresh = bg_mean + alpha * bg_std
        tcr = t_vv_filled - bg_mean
        det = (t_vv_filled > thresh) & (t_mask > 0.5)
        # Accumulate: union of detections, max TCR, sum confidence
        combined_det |= det
        combined_tcr = torch.max(combined_tcr, tcr)
        confidence += det.float()
        print(f"[SAR] CFAR {sc['label']}: g={g} b={b} α={alpha:.2f} → {det.sum().item()} px")

    # Confidence = how many scales detected this pixel (0-3)
    conf_np = confidence.squeeze().cpu().numpy()
    tcr_np = combined_tcr.squeeze().cpu().numpy()
    det_np = combined_det.squeeze().cpu().numpy() & ocean_mask
    return tcr_np, det_np, conf_np


def _cfar_multiscale_scipy(vv, ocean_mask, ocean_med, scales):
    vv_pad = vv.copy(); vv_pad[~ocean_mask] = ocean_med
    combined_tcr = np.zeros_like(vv)
    combined_det = np.zeros_like(vv, dtype=bool)
    confidence = np.zeros_like(vv, dtype=np.float32)

    for sc in scales:
        bg_size = 2 * sc["bg"] + 1
        local_bg = median_filter(vv_pad, size=bg_size)
        tcr = vv - local_bg; tcr[~ocean_mask] = 0
        from scipy.stats import norm
        alpha = float(norm.ppf(1 - sc["pfa"]))
        local_std = np.sqrt(np.maximum(median_filter(vv_pad**2, size=bg_size) - local_bg**2, 0.01))
        thresh = local_bg + alpha * local_std
        det = (vv > thresh) & ocean_mask
        combined_det |= det
        combined_tcr = np.maximum(combined_tcr, tcr)
        confidence += det.astype(np.float32)
        print(f"[SAR] CFAR {sc['label']}: bg={bg_size} → {det.sum()} px")

    return combined_tcr, combined_det, confidence


def _nms_clusters(labeled, n_clusters, vv, min_dist_px=5):
    """Non-Maximum Suppression on cluster peaks — merge nearby detections."""
    peaks = []
    for c in range(1, n_clusters + 1):
        cluster = labeled == c
        if cluster.sum() < 2: continue
        rows, cols = np.where(cluster)
        cluster_vv = vv[cluster]
        pi = np.argmax(cluster_vv)
        peaks.append({"cluster_id": c, "row": rows[pi], "col": cols[pi], "vv": float(cluster_vv[pi]), "npix": int(cluster.sum())})

    # Sort by brightness
    peaks.sort(key=lambda p: p["vv"], reverse=True)

    # NMS: suppress weaker peaks within min_dist_px of a stronger one
    keep = []
    for p in peaks:
        suppressed = False
        for k in keep:
            dist = np.sqrt((p["row"] - k["row"])**2 + (p["col"] - k["col"])**2)
            if dist < min_dist_px:
                # Merge pixel count to the stronger detection
                k["npix"] += p["npix"]
                suppressed = True; break
        if not suppressed:
            keep.append(p)
    return keep


# ══════════════════════════════════════════════════════
# Classification
# ══════════════════════════════════════════════════════
def _classify(length_m, peak_vv, mean_tcr, pol_ratio, n_pix, aspect, confidence):
    scores = {"small_vessel": 0.0, "large_vessel": 0.0, "submarine_uuv": 0.0}

    # Size
    if length_m > 200: scores["large_vessel"] += 3.5
    elif length_m > 80: scores["large_vessel"] += 2.0; scores["small_vessel"] += 0.5
    elif length_m > 30: scores["small_vessel"] += 2.0; scores["submarine_uuv"] += 0.5
    else: scores["small_vessel"] += 1.0; scores["submarine_uuv"] += 1.5

    # Backscatter
    if peak_vv > 0: scores["large_vessel"] += 2.5; scores["small_vessel"] += 1.0
    elif peak_vv > -5: scores["large_vessel"] += 1.5; scores["small_vessel"] += 1.5
    elif peak_vv > -10: scores["small_vessel"] += 2.0; scores["submarine_uuv"] += 0.5
    else: scores["submarine_uuv"] += 2.5

    # TCR
    if mean_tcr > 12: scores["large_vessel"] += 2.0
    elif mean_tcr > 8: scores["large_vessel"] += 1.0; scores["small_vessel"] += 1.0
    elif mean_tcr > 4: scores["small_vessel"] += 1.5
    else: scores["submarine_uuv"] += 2.0

    # Cross-pol
    if pol_ratio > 10: scores["large_vessel"] += 1.5
    elif pol_ratio > 6: scores["small_vessel"] += 1.0
    else: scores["submarine_uuv"] += 1.0

    # Aspect & multi-scale confidence
    if aspect > 3: scores["large_vessel"] += 1.0
    elif aspect > 1.5: scores["small_vessel"] += 0.5
    else: scores["submarine_uuv"] += 0.5

    if confidence >= 3: scores["large_vessel"] += 1.0  # detected at all scales
    elif confidence <= 1: scores["submarine_uuv"] += 0.8  # only faint scale

    total = sum(np.exp(v) for v in scores.values())
    probs = {k: float(np.exp(v) / total) for k, v in scores.items()}
    return max(probs, key=probs.get), probs


# ══════════════════════════════════════════════════════
# Target thumbnail (base64 PNG)
# ══════════════════════════════════════════════════════
def _make_thumbnail(vv, pr, pc, ocean_mask, crop=100):
    """Crop a SAR thumbnail around target, return base64 PNG."""
    from PIL import Image
    h, w = vv.shape
    r0 = max(0, pr - crop); r1 = min(h, pr + crop)
    c0 = max(0, pc - crop); c1 = min(w, pc + crop)
    patch = vv[r0:r1, c0:c1].copy()
    mask_patch = ocean_mask[r0:r1, c0:c1]

    # Normalize to 0-255 grayscale
    vmin, vmax = -22, 2
    norm = np.clip((patch - vmin) / (vmax - vmin), 0, 1)
    norm[~mask_patch] = 0
    gray = (norm * 255).astype(np.uint8)

    # Draw a crosshair at target center
    tr, tc = pr - r0, pc - c0
    ch_len = 8
    for i in range(-ch_len, ch_len + 1):
        if 0 <= tr + i < gray.shape[0]: gray[tr + i, tc] = min(255, gray[tr + i, tc] + 120)
        if 0 <= tc + i < gray.shape[1]: gray[tr, tc + i] = min(255, gray[tr, tc + i] + 120)

    img = Image.fromarray(gray, mode='L')
    # Upscale 2x for visibility
    img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════
def detect_vessels(bbox, start_date, end_date, resolution=10):
    sar = fetch_sar(bbox, start_date, end_date, resolution)
    vv, vh = sar["vv_db"], sar["vh_db"]
    h, w = vv.shape
    west, south, east, north = bbox
    lats = np.linspace(north, south, h)
    lons = np.linspace(west, east, w)

    # Ocean mask — global Natural Earth 10m coastline
    print("[SAR] Building ocean mask (Natural Earth 10m)...")
    ocean_geo = make_ocean_mask(lons, lats)  # True = ocean
    invalid = np.isnan(vv) | (vv == 0) | (vv > 5)
    ocean_mask = ocean_geo & ~invalid
    ocean_pct = float(ocean_mask.sum() / (h * w) * 100)
    land_pct = 100 - ocean_pct
    print(f"[SAR] Ocean: {ocean_mask.sum()}/{h*w} ({ocean_pct:.0f}%) — Land masked: {land_pct:.0f}%")

    # ZERO OUT ALL LAND — never compute on land pixels
    vv[~ocean_mask] = np.nan
    vh[~ocean_mask] = np.nan

    # Multi-scale CFAR
    tcr, detected, confidence = _cfar_multiscale(vv, ocean_mask)
    cross_pol = vv - vh

    # Cluster + NMS
    print("[SAR] Clustering + NMS...")
    labeled, n_clusters = label(detected)
    nms_peaks = _nms_clusters(labeled, n_clusters, vv, min_dist_px=8)
    print(f"[SAR] {n_clusters} raw clusters → {len(nms_peaks)} after NMS")

    # Build detections with thumbnails
    print("[SAR] Classifying & generating thumbnails...")
    detections = []
    for peak in nms_peaks:
        pr, pc = peak["row"], peak["col"]
        lat, lon = float(lats[pr]), float(lons[pc])
        if is_land(lon, lat): continue

        # Reconstruct cluster from labeled map
        cid = labeled[pr, pc]
        if cid == 0: continue
        cluster = labeled == cid
        rows, cols = np.where(cluster)

        row_span = int(rows.max() - rows.min() + 1) * resolution
        col_span = int(cols.max() - cols.min() + 1) * resolution
        length_m = float(max(row_span, col_span))
        width_m = float(min(row_span, col_span))
        aspect = float(length_m / max(width_m, resolution))
        peak_vv = float(vv[pr, pc]) if not np.isnan(vv[pr, pc]) else -20.0
        peak_vh = float(vh[pr, pc]) if not np.isnan(vh[pr, pc]) else -25.0
        _tcr_vals = tcr[cluster]; _tcr_vals = _tcr_vals[~np.isnan(_tcr_vals)]
        mean_tcr = float(np.mean(_tcr_vals)) if len(_tcr_vals) > 0 else 0.0
        _cp_vals = cross_pol[cluster]; _cp_vals = _cp_vals[~np.isnan(_cp_vals)]
        pol_ratio = float(np.mean(_cp_vals)) if len(_cp_vals) > 0 else 0.0
        conf = float(confidence[pr, pc])

        best_class, probs = _classify(length_m, peak_vv, mean_tcr, pol_ratio, peak["npix"], aspect, conf)

        # Heading via PCA
        heading = None
        if len(rows) >= 3:
            try:
                coords = np.column_stack([cols - cols.mean(), rows - rows.mean()])
                _, _, Vt = np.linalg.svd(coords, full_matrices=False)
                heading = float(np.degrees(np.arctan2(Vt[0, 0], -Vt[0, 1])) % 360)
            except: pass

        # Thumbnail
        thumb = _make_thumbnail(vv, pr, pc, ocean_mask, crop=80)

        detections.append({
            "id": len(detections) + 1,
            "lat": round(lat, 5), "lon": round(lon, 5),
            "class": best_class,
            "prob_small_vessel": round(probs["small_vessel"], 3),
            "prob_large_vessel": round(probs["large_vessel"], 3),
            "prob_submarine": round(probs["submarine_uuv"], 3),
            "length_m": round(length_m, 1), "width_m": round(width_m, 1),
            "aspect_ratio": round(aspect, 2),
            "heading_deg": round(heading, 1) if heading else None,
            "n_pixels": peak["npix"],
            "peak_vv_db": round(peak_vv, 2), "peak_vh_db": round(peak_vh, 2),
            "mean_tcr_db": round(mean_tcr, 2), "cross_pol_db": round(pol_ratio, 2),
            "confidence": round(conf, 1),
            "thumbnail": thumb,
        })

    detections.sort(key=lambda d: d["peak_vv_db"], reverse=True)
    for i, d in enumerate(detections): d["id"] = i + 1

    # Sanitize: replace any NaN/inf with None (JSON-safe)
    import math
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    detections = _sanitize(detections)

    print(f"[SAR] {len(detections)} targets final")

    # Zarr
    zarr_path = None
    if ZARR_OK:
        try:
            zdir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocean", "zarr")
            os.makedirs(zdir, exist_ok=True)
            zpath = os.path.join(zdir, f"sar_{start_date}_{end_date}.zarr")
            root = zarr.open(zpath, mode="w")
            root.create_dataset("vv_db", data=vv, chunks=(256, 256), compressor=zarr.Blosc())
            root.create_dataset("vh_db", data=vh, chunks=(256, 256), compressor=zarr.Blosc())
            root.create_dataset("tcr", data=tcr, chunks=(256, 256), compressor=zarr.Blosc())
            root.attrs["bbox"] = [float(x) for x in bbox]
            zarr_path = zpath; print(f"[SAR] Zarr: {zpath}")
        except Exception as e: print(f"[SAR] Zarr err: {e}")

    # Display layers (downsampled)
    max_disp = 500
    dh, dw = min(h, max_disp), min(w, max_disp)
    from scipy.ndimage import zoom as szoom

    def ds(arr):
        if arr.shape[0] <= dh and arr.shape[1] <= dw: return arr.copy()
        m = np.isnan(arr); filled = arr.copy(); filled[m] = 0
        out = szoom(filled, (dh/arr.shape[0], dw/arr.shape[1]), order=1)
        mds = szoom(m.astype(float), (dh/arr.shape[0], dw/arr.shape[1]), order=0) > 0.5
        out[mds] = np.nan; return out

    vv_vis = vv.copy(); vv_vis[~ocean_mask] = np.nan
    vv_norm = np.clip((vv_vis - (-25)) / 25.0, 0, 1)
    tcr_vis = tcr.copy(); tcr_vis[~ocean_mask] = np.nan
    tcr_norm = np.clip(tcr_vis / 15.0, 0, 1)
    vv_ds, tcr_ds = ds(vv_norm), ds(tcr_norm)
    dha, dwa = vv_ds.shape
    lats_out = np.linspace(south, north, dha).tolist()
    lons_out = np.linspace(west, east, dwa).tolist()
    print(f"[SAR] Display: {dha}x{dwa}")

    def to_mat(arr):
        ny, nx = arr.shape; out = []
        for i in range(ny):
            ri = ny - 1 - i; row = []
            for j in range(nx):
                v = float(arr[ri, j]); row.append(None if np.isnan(v) else round(v, 4))
            out.append(row)
        return out

    counts = {"small_vessel": 0, "large_vessel": 0, "submarine_uuv": 0}
    for d in detections: counts[d["class"]] += 1

    return {
        "layers": {
            "sar_vv": {"lats": lats_out, "lons": lons_out, "matrix": to_mat(vv_ds), "ny": dha, "nx": dwa, "min_val": 0, "max_val": 1},
            "tcr": {"lats": lats_out, "lons": lons_out, "matrix": to_mat(tcr_ds), "ny": dha, "nx": dwa, "min_val": 0, "max_val": 1},
        },
        "detections": detections,
        "metadata": {
            "start_date": start_date, "end_date": end_date,
            "resolution_m": resolution, "grid": f"{h}x{w}", "display_grid": f"{dha}x{dwa}",
            "bbox": [float(x) for x in bbox], "ocean_pct": round(ocean_pct, 1),
            "n_detections": len(detections), "counts": counts,
            "engine": "multiscale-CFAR+NMS", "device": str(DEVICE) if DEVICE else "cpu",
            "scales": 3, "zarr_path": zarr_path,
        },
    }


def get_default_bbox(): return _roi_bbox(KHALIFA_PORT["lat"], KHALIFA_PORT["lon"], ROI_KM)
def get_default_dates():
    end = datetime.utcnow().date(); start = end - timedelta(days=30)
    return start.isoformat(), end.isoformat()
