"""
i-Boating Nautical Chart Pipeline — Professional Edition
─────────────────────────────────────────────────────────
1. Headless Chromium captures i-Boating chart for the ROI
2. PIL-based colour rasterization → depth grid from chart colours
3. Gemini Vision extracts VERIFIED depth soundings (not buoy IDs, markers, etc.)
4. Multi-source fusion: colour raster (base) + contours + soundings → final grid
5. Outlier rejection: cross-validate soundings against colour context
6. 80/20 split → train CNN → validate on held-out set
"""
from __future__ import annotations
import os, io, json, math, time, base64, logging, tempfile
from pathlib import Path
import numpy as np, requests
from PIL import Image
from scipy import ndimage
from scipy.interpolate import griddata

L = logging.getLogger("bathy.iboating")
MAX_DEPTH_M = 25.0  # SDB reliable limit — cap all depths

SCREENSHOT_DIR = Path("/tmp/bathy/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. HEADLESS SCREENSHOT of i-Boating
# ══════════════════════════════════════════════════════════════

# i-Boating runs Mapbox GL JS (512-px tiles, NOT Leaflet). The Map instance is
# closed-over by the app, so we hook mapboxgl.Map.prototype.fire from an init
# script to capture it — the ONLY reliable way to read the true rendered
# bounds. (Proven in run_iboating_subgrid.py::capture_tile; the old
# viewport-math / "centered is close enough" georeferencing drifted ~1 km.)
_MAPBOXGL_HOOK = """
(() => {
    const tryHook = () => {
        if (!window.mapboxgl || !window.mapboxgl.Map) return false;
        const proto = window.mapboxgl.Map.prototype;
        if (proto.__hooked) return true;
        const origFire = proto.fire;
        proto.fire = function() {
            if (!window.__mapboxMap) window.__mapboxMap = this;
            return origFire.apply(this, arguments);
        };
        proto.__hooked = true;
        return true;
    };
    if (!tryHook()) {
        const t = setInterval(() => { if (tryHook()) clearInterval(t); }, 30);
        setTimeout(() => clearInterval(t), 30000);
    }
})();
"""


def _capture_iboating(bbox, zoom=14, wait_sec=12, viewport=(1920, 1080)):
    """
    Open i-Boating in headless Chromium, jump to the ROI centre, screenshot,
    and read the TRUE rendered bounds from Mapbox GL itself.

    Returns (screenshot_path, img_b64, width, height, geo_bbox) where
    geo_bbox = [w, s, e, n] of the pixels actually in the screenshot — use it
    (never the requested bbox) for every pixel→lat/lon conversion. Falls back
    to the requested bbox (logged) only if the Map instance can't be captured.
    """
    w, s, e, n = bbox
    lat_c = (n + s) / 2
    lon_c = (e + w) / 2
    url = (
        f"https://fishing-app.gpsnauticalcharts.com/i-boating-fishing-web-app/"
        f"fishing-marine-charts-navigation.html#{zoom}/{lat_c:.5f}/{lon_c:.5f}"
    )
    L.info(f"i-Boating: opening {url} ({viewport[0]}x{viewport[1]})")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        chromium_path = os.environ.get("CHROMIUM_PATH")
        launch_args = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-web-security",
                "--window-size={},{}".format(*viewport),
            ],
        }
        if chromium_path and os.path.exists(chromium_path):
            launch_args["executable_path"] = chromium_path

        browser = pw.chromium.launch(**launch_args)
        ctx = browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]},
            device_scale_factor=2,  # 2x for crisp depth numbers
        )
        ctx.add_init_script(_MAPBOXGL_HOOK)
        page = ctx.new_page()
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        time.sleep(3)

        # Dismiss popups
        for sel in ["button:has-text('Accept')", "button:has-text('Close')",
                     "button:has-text('OK')", ".modal-close", "#close-btn",
                     "button:has-text('Got it')", ".cookie-close"]:
            try:
                page.click(sel, timeout=2000)
            except Exception:
                pass

        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # Wait for the hook to capture the Map, then force-jump to the ROI
        # centre (the URL fragment only sets the initial view).
        try:
            page.wait_for_function(
                "() => window.__mapboxMap "
                "&& typeof window.__mapboxMap.getBounds === 'function'",
                timeout=20000,
            )
        except Exception:
            pass
        try:
            page.evaluate(f"""() => {{
                const m = window.__mapboxMap;
                if (m && typeof m.jumpTo === 'function') {{
                    m.jumpTo({{center: [{lon_c}, {lat_c}], zoom: {zoom},
                               bearing: 0, pitch: 0}});
                }}
            }}""")
        except Exception:
            pass

        time.sleep(max(4, wait_sec - 8))
        try:
            page.wait_for_function(
                "() => { const m = window.__mapboxMap; "
                "return m && typeof m.areTilesLoaded === 'function' "
                "&& m.areTilesLoaded() && !m.isMoving() && !m.isZooming(); }",
                timeout=15000,
            )
        except Exception:
            pass

        # ── Single source of truth: Mapbox GL .getBounds() ──
        mb = None
        for _attempt in range(3):
            try:
                mb = page.evaluate("""() => {
                    const m = window.__mapboxMap;
                    if (!m || typeof m.getBounds !== 'function') return null;
                    let b; try { b = m.getBounds(); } catch (e) { return null; }
                    if (!b) return null;
                    return {sw_lat: b.getSouth(), sw_lon: b.getWest(),
                            ne_lat: b.getNorth(), ne_lon: b.getEast(),
                            z: m.getZoom()};
                }""")
            except Exception:
                mb = None
            if mb:
                break
            time.sleep(1.0)

        ts = int(time.time())
        fname = f"iboating_{lat_c:.4f}_{lon_c:.4f}_z{zoom}_{ts}.png"
        fpath = SCREENSHOT_DIR / fname

        # Screenshot the Mapbox canvas only (clips UI overlays) — its pixels
        # correspond 1:1 to getBounds(). Fall back to the full viewport.
        buf = None
        try:
            buf = page.locator("canvas.mapboxgl-canvas").first.screenshot(
                path=str(fpath), timeout=30000)
        except Exception:
            buf = None
        if buf is None:
            page.screenshot(path=str(fpath), full_page=False)
            buf = page.screenshot(full_page=False)

        img_b64 = base64.b64encode(buf).decode()
        browser.close()

    # True pixel dimensions of what we actually captured
    try:
        _im = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        actual_w, actual_h = _im.size
    except Exception:
        actual_w, actual_h = viewport[0] * 2, viewport[1] * 2

    if mb:
        geo_bbox = [mb["sw_lon"], mb["sw_lat"], mb["ne_lon"], mb["ne_lat"]]
        L.info(f"i-Boating: {fpath} ({actual_w}x{actual_h}) "
               f"bbox[mapbox.getBounds]={[round(v,5) for v in geo_bbox]} z={mb.get('z')}")
    else:
        geo_bbox = [w, s, e, n]
        L.warning(f"i-Boating: {fpath} ({actual_w}x{actual_h}) "
                  f"bbox[requested-FALLBACK] — Map instance not captured, "
                  f"georeferencing may drift")
    return str(fpath), img_b64, actual_w, actual_h, geo_bbox


def _multi_zoom_capture(bbox, zooms=(12, 14), wait_sec=10):
    """
    Professional multi-zoom chart capture.
    Lower zoom (12) gives overview of deeper areas and major contours.
    Higher zoom (14) gives detailed soundings in port areas.
    Returns list of (screenshot_path, img_b64, w, h, zoom).
    """
    captures = []
    for z in zooms:
        try:
            vp = (1920, 1080) if z <= 13 else (1920, 1080)
            path, b64, w, h, geo_bbox = _capture_iboating(bbox, zoom=z, wait_sec=wait_sec, viewport=vp)
            captures.append((path, b64, w, h, z, geo_bbox))
            L.info(f"Multi-zoom: captured z{z} ({w}x{h})")
        except Exception as ex:
            L.warning(f"Multi-zoom: z{z} failed: {ex}")
    return captures


# ══════════════════════════════════════════════════════════════
# 2. COLOUR RASTERIZATION — PIL-based chart-to-depth conversion
# ══════════════════════════════════════════════════════════════

# i-Boating colour palette (HSV ranges) → depth bands
# Calibrated from typical i-Boating chart renders:
#   - Land: greens/browns/grays (H outside blue range, or very low S)
#   - Very shallow (0-2m): very light cyan / near-white with blue tint
#   - Shallow (2-5m): light blue / pale cyan
#   - Moderate (5-10m): medium blue
#   - Deep (10-20m): darker blue
#   - Very deep (20-25m+): navy / dark blue
_DEPTH_COLOUR_TABLE = [
    # UAE-calibrated HSV ranges for i-Boating charts
    # Finer bands in shallow water for better gradient resolution
    # (H_min, H_max, S_min, S_max, V_min, V_max, depth_min, depth_max)
    # Intertidal / very shallow — near-white, faintest blue tint
    # Intertidal colour band: we assign a floor of 0.1m so the CNN
    # still sees a usable training signal for tidal flats (0m readings from
    # the raster would be rejected downstream as missing data).
    (140, 210,   3,  25, 220, 255,   0.1,  0.5),
    (140, 215,   5,  35, 210, 255,   0.5,  1.0),
    # Very shallow — pale cyan
    (145, 220,  15,  50, 200, 250,   1.0,  2.0),
    (150, 220,  25,  60, 190, 245,   2.0,  3.0),
    # Shallow — light blue
    (155, 225,  35,  75, 175, 240,   3.0,  5.0),
    # Moderate shallow — medium-light blue
    (160, 225,  50,  95, 155, 225,   5.0,  7.0),
    # Moderate — medium blue
    (165, 230,  60, 120, 130, 210,   7.0, 10.0),
    # Moderately deep — blue
    (175, 240,  75, 160, 100, 180,  10.0, 13.0),
    # Deep — darker blue
    (180, 245,  90, 190,  65, 150,  13.0, 17.0),
    # Deep — dark blue
    (185, 250, 100, 210,  40, 120,  17.0, 20.0),
    # Very deep — navy
    (190, 260, 110, 255,  15,  90,  20.0, 25.0),
]


def _classify_water_mask(hsv_arr):
    """Identify water pixels vs land/UI from HSV array. Returns bool mask."""
    h, s, v = hsv_arr[:, :, 0], hsv_arr[:, :, 1], hsv_arr[:, :, 2]
    # Water: blue-ish hue (140-260°), some saturation, not pure black/white
    water = (h >= 140) & (h <= 260) & (s >= 5) & (v >= 10) & (v <= 255)
    # Exclude very dark pixels (UI chrome, borders)
    water &= ~((v < 15) & (s < 20))
    # Exclude near-pure white (clouds, text backgrounds, UI)
    water &= ~((s < 5) & (v > 240))
    return water


def _colour_to_depth_raster(img_b64, target_h=None, target_w=None):
    """
    Convert chart screenshot to a depth raster using colour classification.
    Returns (depth_grid, water_mask, confidence_grid) at target resolution,
    or at image resolution if target not specified.
    """
    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    rgb = np.array(img)
    orig_h, orig_w = rgb.shape[:2]

    # Convert RGB → HSV (0-360, 0-255, 0-255)
    # PIL's HSV uses H=0-255 so we do manual conversion for 0-360° range
    r, g, b = rgb[:,:,0].astype(float), rgb[:,:,1].astype(float), rgb[:,:,2].astype(float)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn + 1e-10

    # Hue (0-360)
    hue = np.zeros_like(r)
    mask_r = (mx == r)
    mask_g = (mx == g) & ~mask_r
    mask_b = ~mask_r & ~mask_g
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
    # Saturation (0-255)
    sat = np.where(mx > 0, (diff / (mx + 1e-10)) * 255, 0)
    # Value (0-255)
    val = mx

    hsv = np.stack([hue, sat, val], axis=2)

    water = _classify_water_mask(hsv)
    depth = np.full((orig_h, orig_w), np.nan, dtype=np.float64)
    confidence = np.zeros((orig_h, orig_w), dtype=np.float64)

    # Classify each water pixel into depth bands
    for h_min, h_max, s_min, s_max, v_min, v_max, d_min, d_max in _DEPTH_COLOUR_TABLE:
        band_mask = (
            water &
            (hue >= h_min) & (hue <= h_max) &
            (sat >= s_min) & (sat <= s_max) &
            (val >= v_min) & (val <= v_max)
        )
        if not np.any(band_mask):
            continue

        # Interpolate depth within the band based on darkness
        # Darker = deeper within band
        v_norm = np.clip((val[band_mask] - v_min) / max(v_max - v_min, 1), 0, 1)
        # Invert: lower value (darker) → deeper
        band_depth = d_min + (1.0 - v_norm) * (d_max - d_min)
        depth[band_mask] = band_depth
        # Confidence: how well the pixel matches the centre of the HSV range
        h_centre = (h_min + h_max) / 2
        s_centre = (s_min + s_max) / 2
        h_dist = np.abs(hue[band_mask] - h_centre) / max(h_max - h_min, 1)
        s_dist = np.abs(sat[band_mask] - s_centre) / max(s_max - s_min, 1)
        conf = np.clip(1.0 - (h_dist + s_dist) / 2, 0.1, 1.0)
        # Keep highest confidence if pixel matched multiple bands
        better = conf > confidence[band_mask]
        where_better = np.where(band_mask)
        rows_b = where_better[0][better]
        cols_b = where_better[1][better]
        confidence[rows_b, cols_b] = conf[better]

    # Fill small gaps in water areas with median filter
    valid_depth = np.where(np.isfinite(depth), depth, 0)
    filled = ndimage.median_filter(valid_depth, size=5)
    gap_mask = water & ~np.isfinite(depth)
    depth[gap_mask] = filled[gap_mask]
    confidence[gap_mask] = 0.2  # low confidence for gap-filled pixels

    # Smooth the raster to remove pixel-level noise
    kernel = np.ones((7, 7)) / 49
    smooth = ndimage.convolve(np.where(np.isfinite(depth), depth, 0), kernel)
    smooth_w = ndimage.convolve(np.where(np.isfinite(depth), 1.0, 0.0), kernel)
    smooth_valid = smooth_w > 0.3
    depth[water & smooth_valid] = smooth[water & smooth_valid] / (smooth_w[water & smooth_valid] + 1e-10)
    depth[~water] = np.nan

    # Clamp to MAX_DEPTH_M
    depth = np.clip(depth, 0, MAX_DEPTH_M)
    depth[~water] = np.nan

    # Downsample to target resolution if needed
    if target_h and target_w and (target_h != orig_h or target_w != orig_w):
        from PIL import Image as PILImage
        depth_img = PILImage.fromarray(np.nan_to_num(depth, nan=0).astype(np.float32))
        depth_img = depth_img.resize((target_w, target_h), PILImage.BILINEAR)
        depth_rs = np.array(depth_img, dtype=np.float64)

        conf_img = PILImage.fromarray(confidence.astype(np.float32))
        conf_img = conf_img.resize((target_w, target_h), PILImage.BILINEAR)
        confidence_rs = np.array(conf_img, dtype=np.float64)

        water_img = PILImage.fromarray(water.astype(np.uint8) * 255)
        water_img = water_img.resize((target_w, target_h), PILImage.NEAREST)
        water_rs = np.array(water_img) > 127

        depth_rs[~water_rs] = np.nan
        return depth_rs, water_rs, confidence_rs

    n_water = int(np.sum(water))
    n_depth = int(np.sum(np.isfinite(depth)))
    L.info(f"Colour raster: {n_water} water pixels, {n_depth} with depth, "
           f"range {np.nanmin(depth):.1f}-{np.nanmax(depth):.1f}m")
    return depth, water, confidence


def _extract_isobath_contours(depth_grid, water_mask, isobaths=None):
    """
    Extract isobath contour lines from the colour-classified depth raster.
    These provide precise depth constraints at colour-band boundaries.

    Professional hydrographic approach:
    - Detect transitions between depth bands (gradient peaks)
    - Extract contour lines at standard isobath levels
    - These become high-confidence training points along the contour

    Returns list of dicts: [{depth, points: [(row, col), ...], confidence}, ...]
    """
    if isobaths is None:
        # Standard hydrographic isobaths for shallow water
        isobaths = [0.5, 1, 2, 3, 5, 7, 10, 13, 15, 17, 20, 25]

    contours = []

    filled = np.where(np.isfinite(depth_grid), depth_grid, -999)

    for iso_depth in isobaths:
        # Binary mask: water shallower than iso_depth
        shallow_mask = water_mask & (filled > 0) & (filled <= iso_depth)
        deep_mask = water_mask & (filled > iso_depth)

        if shallow_mask.sum() < 10 or deep_mask.sum() < 10:
            continue

        # Find boundary pixels (edge between shallow and deep)
        # Dilate shallow mask and intersect with deep mask → boundary
        dilated = ndimage.binary_dilation(shallow_mask, iterations=1)
        boundary = dilated & deep_mask

        if boundary.sum() < 5:
            continue

        # Extract boundary pixel coordinates
        rows, cols = np.where(boundary)

        # Subsample if too many points (keep every Nth)
        if len(rows) > 200:
            step = max(1, len(rows) // 200)
            rows, cols = rows[::step], cols[::step]

        # Confidence: higher for standard isobaths, lower for interpolated
        conf = 0.85 if iso_depth in [2, 5, 10, 15, 20] else 0.7

        contours.append({
            'depth': float(iso_depth),
            'points': list(zip(rows.tolist(), cols.tolist())),
            'n_points': len(rows),
            'confidence': conf,
        })

    return contours


def _contours_to_geo_points(contours, bbox, grid_h, grid_w):
    """Convert extracted isobath contour points to georeferenced depth points."""
    w, s, e, n = bbox
    geo_pts = []
    for c in contours:
        for (r, col) in c['points']:
            lat = n - (r / grid_h) * (n - s)
            lon = w + (col / grid_w) * (e - w)
            if s <= lat <= n and w <= lon <= e:
                geo_pts.append({
                    'lat': round(lat, 6),
                    'lon': round(lon, 6),
                    'depth': round(c['depth'], 2),
                    'type': 'contour_extracted',
                    'confidence': round(c['confidence'], 2),
                    'validation': 'isobath_boundary',
                })
    return geo_pts


# ══════════════════════════════════════════════════════════════
# 3. GEMINI VISION — PROFESSIONAL sounding extraction
# ══════════════════════════════════════════════════════════════

def _extract_depths_from_chart(img_b64, bbox, img_w, img_h):
    """
    Gemini Vision extracts ONLY verified depth soundings from i-Boating.
    Explicitly rejects non-depth numbers (buoy IDs, markers, routes, etc.).
    """
    api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return None, "GEMINI_API_KEY not set"

    w, s, e, n = bbox
    prompt = f"""You are a professional hydrographic analyst examining a high-resolution screenshot
of the i-Boating nautical chart application.

Image: {img_w} x {img_h} pixels.
Bounds: SW({s:.5f}°N, {w:.5f}°E) to NE({n:.5f}°N, {e:.5f}°E).

TASK: Extract ONLY genuine depth soundings from the water area of this chart.

─── WHAT IS A DEPTH SOUNDING ───
Depth soundings are small standalone numbers printed ON the water (blue area) indicating
the water depth at that exact position. They are typically:
• Plain numbers like 3.2, 15, 7.8, 22.1 — often with one decimal
• Printed in a small, consistent font directly on the water
• NOT enclosed in any special symbol, circle, diamond, or box
• NOT underlined (underlined = drying height, not depth)
• Positioned AWAY from any chart symbol or feature marker
• Units: meters (default), feet (÷ 3.281), or fathoms (× 1.8288)

─── WHAT IS NOT A DEPTH SOUNDING — MUST IGNORE ───
❌ Buoy/marker numbers: numbers inside circles, diamonds, triangles, or near ▲/● symbols
❌ Route/channel numbers: numbers along shipping lanes or traffic separation schemes
❌ Light characteristics: text like "Fl 5s", "Fl(2)R 10s", "Iso 4s", "Q(6)+LFl 15s"
❌ Bridge clearances: numbers near bridge symbols (usually with a line above/below)
❌ Height numbers: numbers on land (green/brown areas) = elevation, NOT depth
❌ Distance markers: numbers along measured distance lines
❌ Chart reference numbers: large numbers in chart corners or borders
❌ Coordinates/graticule labels: lat/lon numbers along the edges
❌ UI elements: zoom level, scale bar, copyright text, menu items
❌ Anchorage/port identifiers: alphanumeric codes near anchor symbols
❌ Cable/pipeline numbers: numbers along dashed lines
❌ Radio frequency numbers: VHF channel numbers near port areas

─── DEPTH CONTOUR LINES (ISOBATHS) ───
Depth contour lines are curved lines in the water connecting points of equal depth.
They sometimes have a depth label along them (2, 5, 10, 20, etc.).
Extract these labels WITH their position — mark type as "contour".

─── OUTPUT FORMAT ───
Return ONLY a JSON array. For each verified sounding:
{{"x": pixel_from_left, "y": pixel_from_top, "depth": meters, "type": "sounding", "confidence": 0.0-1.0}}

For contour labels:
{{"x": pixel_x, "y": pixel_y, "depth": meters, "type": "contour", "confidence": 0.0-1.0}}

Confidence guide:
• 0.9-1.0: clearly a standalone depth number on open water, easy to read
• 0.7-0.9: likely a sounding but slightly ambiguous (near a symbol, partially obscured)
• 0.5-0.7: possible sounding but uncertain — could be something else
• Below 0.5: do NOT include — too uncertain

IMPORTANT:
• x: 0 (left) to {img_w} (right)
• y: 0 (top) to {img_h} (bottom)
• Depth always in METERS, positive
• QUALITY over QUANTITY — 30 verified soundings > 200 dubious ones
• When in doubt, EXCLUDE the number
• If you see NO verified soundings, return []
• Do NOT guess or fabricate soundings — only report what you actually see"""

    try:
        _model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model}:generateContent?key={api_key}"
        resp = requests.post(url, json={
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                {"text": prompt}
            ]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 32768}
        }, timeout=180)
        if not resp.ok:
            return None, f"Gemini {resp.status_code}: {resp.text[:300]}"

        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        clean = text.replace("```json", "").replace("```", "").strip()
        L.info(f"Gemini i-Boating: {len(clean)} chars response")

        pts = None
        try:
            pts = json.loads(clean)
        except Exception:
            idx = clean.find("[")
            if idx >= 0:
                raw = clean[idx:]
                try:
                    pts = json.loads(raw)
                except Exception:
                    lb = raw.rfind("}")
                    if lb > 0:
                        try:
                            pts = json.loads(raw[: lb + 1] + "]")
                        except Exception:
                            pass

        if pts and isinstance(pts, list) and len(pts) > 0:
            valid = []
            for p in pts:
                d = p.get("depth", 0)
                conf = p.get("confidence", 0.5)
                if not isinstance(d, (int, float)) or d <= 0 or d > MAX_DEPTH_M:
                    continue
                if conf < 0.5:
                    continue  # reject low-confidence
                p["depth"] = min(float(d), MAX_DEPTH_M)
                p["confidence"] = float(conf)
                valid.append(p)

            n_snd = len([p for p in valid if p.get("type") == "sounding"])
            n_cnt = len([p for p in valid if p.get("type") == "contour"])
            L.info(f"Gemini: {len(valid)} verified points "
                   f"({n_snd} soundings, {n_cnt} contours) — "
                   f"avg confidence {np.mean([p['confidence'] for p in valid]):.2f}")
            return valid, None
        return None, f"No depths extracted (response: {clean[:150]})"
    except Exception as ex:
        return None, str(ex)


# ══════════════════════════════════════════════════════════════
# 4. OUTLIER REJECTION — cross-validate soundings vs colour
# ══════════════════════════════════════════════════════════════

def _reject_outliers(soundings, colour_depth, colour_conf, img_w, img_h, tolerance=5.0):
    """
    Reject sounding points that conflict with surrounding colour context.
    A sounding reading 20m in a clearly light-blue (2-5m) area is likely
    a misread buoy number, not a depth.

    tolerance: max allowed deviation (meters) between sounding and colour depth.
    Points outside tolerance are downweighted, not immediately removed.
    """
    if colour_depth is None or len(soundings) == 0:
        return soundings

    ch, cw = colour_depth.shape
    verified = []

    for p in soundings:
        px = p.get("x", 0)
        py = p.get("y", 0)
        sounding_depth = p["depth"]
        conf = p.get("confidence", 0.7)

        # Map pixel coords to colour raster coords
        cr = max(0, min(ch - 1, int(py / img_h * ch)))
        cc = max(0, min(cw - 1, int(px / img_w * cw)))

        # Sample neighbourhood (5x5) in colour raster
        r_min, r_max = max(0, cr - 2), min(ch, cr + 3)
        c_min, c_max = max(0, cc - 2), min(cw, cc + 3)
        patch = colour_depth[r_min:r_max, c_min:c_max]
        patch_conf = colour_conf[r_min:r_max, c_min:c_max]
        valid_patch = patch[np.isfinite(patch)]

        if len(valid_patch) < 3:
            # No colour context → keep the sounding but lower confidence
            p["confidence"] = conf * 0.8
            p["validation"] = "no_colour_context"
            verified.append(p)
            continue

        colour_median = float(np.median(valid_patch))
        colour_mean_conf = float(np.mean(patch_conf[np.isfinite(patch)]))
        deviation = abs(sounding_depth - colour_median)

        if deviation <= tolerance:
            # Good agreement — boost confidence
            p["confidence"] = min(1.0, conf * 1.1)
            p["validation"] = "colour_confirmed"
            verified.append(p)
        elif deviation <= tolerance * 2:
            # Moderate disagreement — keep but downweight
            penalty = 1.0 - (deviation - tolerance) / tolerance
            p["confidence"] = max(0.3, conf * penalty)
            p["validation"] = "colour_uncertain"
            verified.append(p)
        else:
            # Strong disagreement with high-confidence colour context
            if colour_mean_conf > 0.5:
                L.info(f"  REJECTED sounding {sounding_depth:.1f}m at ({px},{py}) — "
                       f"colour says {colour_median:.1f}m (dev={deviation:.1f}m)")
                p["confidence"] = 0
                p["validation"] = "colour_rejected"
                # Still include but mark as rejected so it shows in diagnostics
                verified.append(p)
            else:
                # Colour context is also low-confidence → keep sounding
                p["confidence"] = conf * 0.5
                p["validation"] = "both_uncertain"
                verified.append(p)

    accepted = [p for p in verified if p["confidence"] > 0]
    rejected = [p for p in verified if p["confidence"] == 0]
    L.info(f"Outlier filter: {len(accepted)} accepted, {len(rejected)} rejected "
           f"out of {len(soundings)} soundings")
    return accepted


# ══════════════════════════════════════════════════════════════
# 5. MULTI-SOURCE FUSION — colour + soundings → final depth
# ══════════════════════════════════════════════════════════════

def _fuse_depth_sources(colour_depth, colour_conf, water_mask,
                        soundings, bbox, img_w, img_h):
    """
    Fuse the colour-derived raster with verified sounding points:
      - Soundings override colour locally (high weight)
      - Contour lines refine transitions (medium weight)
      - Colour raster fills everywhere else (base layer)
    Uses inverse-distance weighted blending in a neighbourhood around each sounding.
    """
    h, c_w = colour_depth.shape
    fused = np.copy(colour_depth)
    fused_conf = np.copy(colour_conf)
    w, s, e, n = bbox

    if not soundings:
        return fused

    # Separate by type and filter out rejected
    snd_points = [p for p in soundings if p.get("confidence", 0) > 0]
    if not snd_points:
        return fused

    # Weight by type
    TYPE_WEIGHT = {"sounding": 3.0, "contour": 2.0}

    for p in snd_points:
        px = p.get("x", 0)
        py = p.get("y", 0)
        depth = p["depth"]
        conf = p.get("confidence", 0.7)
        type_w = TYPE_WEIGHT.get(p.get("type", "sounding"), 1.0)
        total_w = conf * type_w

        # Map to raster coords
        cr = max(0, min(h - 1, int(py / img_h * h)))
        cc = max(0, min(c_w - 1, int(px / img_w * c_w)))

        # Influence radius: proportional to confidence (high conf → larger radius)
        radius = max(3, int(15 * conf))

        r_min, r_max = max(0, cr - radius), min(h, cr + radius + 1)
        c_min, c_max = max(0, cc - radius), min(c_w, cc + radius + 1)

        for rr in range(r_min, r_max):
            for cc2 in range(c_min, c_max):
                if not water_mask[rr, cc2]:
                    continue
                dist = math.sqrt((rr - cr) ** 2 + (cc2 - cc) ** 2) + 0.5
                idw = total_w / (dist ** 1.5)

                existing_w = fused_conf[rr, cc2]
                existing_d = fused[rr, cc2] if np.isfinite(fused[rr, cc2]) else 0
                new_d = (existing_d * existing_w + depth * idw) / (existing_w + idw + 1e-10)
                fused[rr, cc2] = np.clip(new_d, 0, MAX_DEPTH_M)
                fused_conf[rr, cc2] = existing_w + idw

    # Final smooth pass to blend sounding influence zones
    kernel = np.ones((3, 3)) / 9
    smooth = ndimage.convolve(np.nan_to_num(fused, nan=0), kernel)
    smooth_w = ndimage.convolve(np.where(np.isfinite(fused), 1.0, 0.0), kernel)
    blend_mask = water_mask & (smooth_w > 0.3)
    # Light blend: 70% fused, 30% smoothed to remove harsh transitions
    fused[blend_mask] = 0.7 * fused[blend_mask] + 0.3 * (smooth[blend_mask] / (smooth_w[blend_mask] + 1e-10))

    fused[~water_mask] = np.nan
    fused = np.clip(fused, 0, MAX_DEPTH_M)
    fused[~water_mask] = np.nan

    L.info(f"Fusion: {len(snd_points)} points applied to {h}x{c_w} raster")
    return fused


def _pixels_to_geo(points, bbox, img_w, img_h):
    """Convert pixel coords → lat/lon/depth dicts."""
    w, s, e, n = bbox
    result = []
    for p in points:
        d = p.get("depth", 0)
        if not isinstance(d, (int, float)) or d <= 0 or d > MAX_DEPTH_M:
            continue
        d = min(d, MAX_DEPTH_M)
        px = p.get("x", 0)
        py = p.get("y", 0)
        lat = n - (py / img_h) * (n - s)
        lon = w + (px / img_w) * (e - w)
        if s <= lat <= n and w <= lon <= e:
            result.append({
                "lat": round(lat, 6), "lon": round(lon, 6),
                "depth": round(float(d), 2),
                "type": p.get("type", "sounding"),
                "confidence": round(p.get("confidence", 0.7), 2),
                "validation": p.get("validation", "unverified"),
            })
    return result


# ══════════════════════════════════════════════════════════════
# 6. FULL PIPELINE: screenshot → rasterize → extract → fuse → train → validate
# ══════════════════════════════════════════════════════════════

def run_iboating_pipeline(bbox, s2_data, zoom=14, train_ratio=0.8):
    """
    Professional i-Boating pipeline:
    1. Screenshot i-Boating chart
    2. Colour rasterization → base depth grid from chart colours
    3. Gemini extracts verified soundings (rejects buoy IDs, markers, etc.)
    4. Cross-validate soundings against colour context → reject outliers
    5. Multi-source fusion: colour raster + soundings → refined depth grid
    6. 80/20 train/val split on verified sounding points
    7. Train CNN on chart-derived depths + S2 bands
    8. Validate on held-out 20%
    """
    w, s, e, n = bbox
    result = {
        "screenshot_path": None,
        "chart_points": [],
        "train_points": [],
        "val_points": [],
        "depth": None,
        "colour_raster_depth": None,
        "validation": None,
        "ml_stats": {},
        "sources_used": [],
        "diagnostics": {},
        "error": None,
    }

    # ── Step 1: Screenshot ──
    L.info("=== i-Boating Pipeline START (Professional) ===")
    try:
        fpath, img_b64, img_w, img_h, chart_bbox = _capture_iboating(bbox, zoom)
        result["screenshot_path"] = fpath
        result["sources_used"].append(f"i-Boating screenshot z{zoom}")
    except Exception as ex:
        result["error"] = f"Screenshot failed: {ex}"
        L.error(f"Screenshot: {ex}")
        return result

    # ── Step 2: Colour rasterization ──
    L.info("Converting chart colours to depth raster...")
    try:
        colour_depth, water_mask, colour_conf = _colour_to_depth_raster(img_b64)
        result["colour_raster_depth"] = colour_depth
        n_water = int(np.sum(water_mask))
        n_valid = int(np.sum(np.isfinite(colour_depth)))
        result["sources_used"].append(f"ColourRaster({n_valid} px)")
        result["diagnostics"]["colour_raster"] = {
            "water_pixels": n_water,
            "depth_pixels": n_valid,
            "depth_range": [
                round(float(np.nanmin(colour_depth)), 1) if n_valid else 0,
                round(float(np.nanmax(colour_depth)), 1) if n_valid else 0,
            ],
            "mean_confidence": round(float(np.mean(colour_conf[water_mask])), 3) if n_water else 0,
        }
        L.info(f"Colour raster: {n_valid}/{n_water} water pixels classified")
    except Exception as ex:
        L.warning(f"Colour rasterization failed: {ex} — continuing with Gemini only")
        colour_depth, water_mask, colour_conf = None, None, None

    # ── Step 2b: Extract isobath contours ──
    contour_pts_geo = []
    if colour_depth is not None:
        L.info("Extracting isobath contours from colour raster...")
        try:
            contours = _extract_isobath_contours(colour_depth, water_mask)
            ch, cw = colour_depth.shape
            contour_pts_geo = _contours_to_geo_points(contours, chart_bbox, ch, cw)
            n_contour = len(contour_pts_geo)
            if n_contour > 0:
                result["sources_used"].append(f"Isobaths({n_contour} pts from {len(contours)} contours)")
                result["diagnostics"]["isobath_contours"] = {
                    "n_contours": len(contours),
                    "n_points": n_contour,
                    "depths": [c['depth'] for c in contours],
                }
            L.info(f"Extracted {n_contour} contour points from {len(contours)} isobaths")
        except Exception as ex:
            L.warning(f"Contour extraction failed: {ex}")

    # ── Step 3: Gemini extracts verified soundings ──
    L.info("Extracting verified depth soundings via Gemini...")
    raw_pts, err = _extract_depths_from_chart(img_b64, chart_bbox, img_w, img_h)
    n_raw = len(raw_pts) if raw_pts else 0

    # ── Step 4: Cross-validate soundings against colour ──
    if raw_pts and colour_depth is not None:
        L.info("Cross-validating soundings against colour context...")
        raw_pts = _reject_outliers(raw_pts, colour_depth, colour_conf, img_w, img_h)
        n_after = len([p for p in raw_pts if p.get("confidence", 0) > 0])
        result["diagnostics"]["sounding_validation"] = {
            "raw_from_gemini": n_raw,
            "after_colour_filter": n_after,
            "rejected": n_raw - n_after,
        }

    # ── Step 5: Fuse colour raster + soundings ──
    fused_depth = None
    if colour_depth is not None:
        if raw_pts:
            L.info("Fusing colour raster with verified soundings...")
            fused_depth = _fuse_depth_sources(
                colour_depth, colour_conf, water_mask,
                raw_pts, bbox, img_w, img_h
            )
        else:
            fused_depth = colour_depth
        result["sources_used"].append("ChartFusion")

    # Convert sounding points to geo for training
    geo_pts = []
    if raw_pts:
        geo_pts = _pixels_to_geo(
            [p for p in raw_pts if p.get("confidence", 0) > 0],
            chart_bbox, img_w, img_h
        )

    # If we have too few sounding points, supplement with colour-raster samples
    if len(geo_pts) < 20 and fused_depth is not None:
        L.info("Supplementing with colour-raster sample points...")
        fh, fw = fused_depth.shape
        step_r = max(1, fh // 15)
        step_c = max(1, fw // 15)
        cb_w, cb_s, cb_e, cb_n = chart_bbox  # pixels ↔ TRUE rendered bounds
        for rr in range(0, fh, step_r):
            for cc in range(0, fw, step_c):
                if not np.isfinite(fused_depth[rr, cc]):
                    continue
                lat = cb_n - (rr / fh) * (cb_n - cb_s)
                lon = cb_w + (cc / fw) * (cb_e - cb_w)
                geo_pts.append({
                    "lat": round(lat, 6), "lon": round(lon, 6),
                    "depth": round(float(fused_depth[rr, cc]), 2),
                    "type": "colour_sample",
                    "confidence": round(float(colour_conf[rr, cc]), 2) if colour_conf is not None else 0.3,
                    "validation": "colour_derived",
                })
        L.info(f"Supplemented to {len(geo_pts)} total points")

    # Merge contour-derived points with sounding points
    if contour_pts_geo:
        geo_pts.extend(contour_pts_geo)
        L.info(f"Merged {len(contour_pts_geo)} contour points → total {len(geo_pts)}")

    if len(geo_pts) < 5:
        if err:
            result["error"] = f"Insufficient depth data: {err}"
        else:
            result["error"] = f"Only {len(geo_pts)} valid depth points extracted"
        return result

    result["chart_points"] = geo_pts
    result["sources_used"].append(f"Verified({len(geo_pts)} depths)")
    L.info(f"Total georeferenced depth points: {len(geo_pts)}")

    # ── Step 6: Spatially-stratified 80/20 split ──
    # Ensure validation points are spread across the spatial extent, not clustered
    np.random.seed(42)
    if len(geo_pts) >= 10:
        # Grid-based stratification: divide bbox into cells, sample from each
        lats_arr = np.array([p['lat'] for p in geo_pts])
        lons_arr = np.array([p['lon'] for p in geo_pts])
        n_cells = min(5, int(np.sqrt(len(geo_pts) / 4)))  # 2-5 cells per axis
        lat_bins = np.linspace(lats_arr.min(), lats_arr.max() + 1e-8, n_cells + 1)
        lon_bins = np.linspace(lons_arr.min(), lons_arr.max() + 1e-8, n_cells + 1)
        val_idx_set = set()
        for li in range(n_cells):
            for lo in range(n_cells):
                cell_mask = (
                    (lats_arr >= lat_bins[li]) & (lats_arr < lat_bins[li + 1]) &
                    (lons_arr >= lon_bins[lo]) & (lons_arr < lon_bins[lo + 1])
                )
                cell_indices = np.where(cell_mask)[0]
                if len(cell_indices) >= 2:
                    n_val_cell = max(1, int(len(cell_indices) * (1 - train_ratio)))
                    chosen = np.random.choice(cell_indices, n_val_cell, replace=False)
                    val_idx_set.update(chosen.tolist())
        val_idx = sorted(val_idx_set)
        train_idx = [i for i in range(len(geo_pts)) if i not in val_idx_set]
    else:
        idx = np.random.permutation(len(geo_pts))
        n_train = max(5, int(len(geo_pts) * train_ratio))
        train_idx = idx[:n_train].tolist()
        val_idx = idx[n_train:].tolist()

    train_pts = [geo_pts[i] for i in train_idx]
    val_pts = [geo_pts[i] for i in val_idx] if val_idx else []
    result["train_points"] = train_pts
    result["val_points"] = val_pts
    L.info(f"Split: {len(train_pts)} train, {len(val_pts)} validation")

    # ── Step 7: Train CNN ──
    if s2_data is None:
        result["error"] = "No Sentinel-2 data provided"
        return result

    ref = {
        "lats": np.array([p["lat"] for p in train_pts]),
        "lons": np.array([p["lon"] for p in train_pts]),
        "depths": np.array([p["depth"] for p in train_pts]),
    }

    depth = None
    try:
        try:
            from backend.cnn_engine import cnn_train_and_predict
        except ImportError:
            from cnn_engine import cnn_train_and_predict
        cnn_result, cnn_err = cnn_train_and_predict(s2_data, ref, list(bbox))
        if cnn_result:
            depth = cnn_result["depth"]
            result["ml_stats"] = {
                "method": "U-Net CNN",
                "r2": cnn_result["r2"],
                "n_train": cnn_result["n_train"],
                "importance": cnn_result["importance"],
            }
            result["sources_used"].append(f"U-Net(R²={cnn_result['r2']})")
            L.info(f"CNN trained: R²={cnn_result['r2']}")
    except Exception as ex:
        L.warning(f"CNN failed: {ex}")

    # GBR fallback
    if depth is None:
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            H, W = s2_data["red"].shape
            eps = 1e-6
            blue = np.clip(s2_data["blue"].astype(np.float64) / 10000, eps, None)
            green = np.clip(s2_data["green"].astype(np.float64) / 10000, eps, None)
            red = np.clip(s2_data["red"].astype(np.float64) / 10000, eps, None)
            nir = np.clip(s2_data["nir"].astype(np.float64) / 10000, eps, None)
            water = s2_data["ndwi"] > 0
            lnBG = np.log(blue + eps) / np.log(green + eps)
            lnGR = np.log(green + eps) / np.log(red + eps)
            BG = blue / (green + eps)
            GR = green / (red + eps)

            Xt, yt = [], []
            for p in train_pts:
                r_px = max(0, min(H - 1, int((n - p["lat"]) / (n - s + 1e-10) * H)))
                c_px = max(0, min(W - 1, int((p["lon"] - w) / (e - w + 1e-10) * W)))
                if water[r_px, c_px]:
                    Xt.append([blue[r_px, c_px], green[r_px, c_px], red[r_px, c_px],
                               nir[r_px, c_px], lnBG[r_px, c_px], lnGR[r_px, c_px],
                               BG[r_px, c_px], GR[r_px, c_px], s2_data["ndwi"][r_px, c_px]])
                    yt.append(p["depth"])

            if len(Xt) >= 5:
                Xt, yt = np.array(Xt), np.array(yt)
                sc = StandardScaler()
                Xs = sc.fit_transform(Xt)
                mdl = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                subsample=0.8, random_state=42)
                mdl.fit(Xs, yt)
                r2 = round(mdl.score(Xs, yt), 4)
                Xp = np.stack([blue.ravel(), green.ravel(), red.ravel(), nir.ravel(),
                               lnBG.ravel(), lnGR.ravel(), BG.ravel(), GR.ravel(),
                               s2_data["ndwi"].ravel()], axis=1)
                depth = np.clip(mdl.predict(sc.transform(np.nan_to_num(Xp, 0))).reshape(H, W), 0, None)
                depth[~water] = np.nan
                result["ml_stats"] = {"method": "GBR", "r2": r2, "n_train": len(Xt)}
                result["sources_used"].append(f"GBR(R²={r2})")
        except Exception as ex:
            L.error(f"GBR fallback: {ex}")

    if depth is None:
        result["error"] = "Model training failed"
        return result

    result["depth"] = depth

    # ── Step 8: Validate on held-out 20% ──
    if len(val_pts) >= 3:
        H, W = depth.shape
        pairs = []
        for p in val_pts:
            r_px = max(0, min(H - 1, int((n - p["lat"]) / (n - s + 1e-10) * H)))
            c_px = max(0, min(W - 1, int((p["lon"] - w) / (e - w + 1e-10) * W)))
            pred_val = depth[r_px, c_px]
            if np.isfinite(pred_val) and pred_val > 0:
                diff = float(pred_val) - p["depth"]
                s44_max = math.sqrt(0.5 ** 2 + (0.013 * p["depth"]) ** 2)
                pairs.append({
                    "lat": p["lat"], "lon": p["lon"],
                    "obs": p["depth"], "pred": round(float(pred_val), 2),
                    "diff": round(diff, 2),
                    "s44_pass": abs(diff) <= s44_max,
                    "type": p.get("type", "sounding"),
                    "confidence": p.get("confidence", 0.7),
                })

        if pairs:
            obs = np.array([p["obs"] for p in pairs])
            pred = np.array([p["pred"] for p in pairs])
            diffs = pred - obs
            rmse = float(np.sqrt(np.mean(diffs ** 2)))
            mae = float(np.mean(np.abs(diffs)))
            bias = float(np.mean(diffs))
            ss_res = np.sum(diffs ** 2)
            ss_tot = np.sum((obs - np.mean(obs)) ** 2)
            r2_val = float(1 - ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0
            s44_pct = float(sum(1 for p in pairs if p["s44_pass"]) / len(pairs) * 100)

            result["validation"] = {
                "rmse": round(rmse, 3), "mae": round(mae, 3), "bias": round(bias, 3),
                "r2": round(r2_val, 4), "s44_pass_pct": round(s44_pct, 1),
                "n_pairs": len(pairs), "n_val": len(val_pts),
                "pairs": pairs,
            }
            L.info(f"Validation: RMSE={rmse:.2f}m, R²={r2_val:.3f}, "
                   f"S-44={s44_pct:.0f}% ({len(pairs)} pairs)")

            # ── Pixel-median aggregation (resolution-aware metrics) ──
            _pixel_median_fn = None
            try:
                from app import pixel_median_comparison as _pixel_median_fn
            except ImportError:
                try:
                    from backend.app import pixel_median_comparison as _pixel_median_fn
                except ImportError:
                    pass
            if _pixel_median_fn is not None and len(val_pts) > 3:
                try:
                    val_lats = np.array([p["lat"] for p in val_pts])
                    val_lons = np.array([p["lon"] for p in val_pts])
                    val_depths_arr = np.array([p["depth"] for p in val_pts])
                    cl = math.cos(math.radians((n + s) / 2))
                    res_m = math.sqrt(abs(e - w) * 111000 * cl * abs(n - s) * 111000 / max(1, H * W))
                    pm = _pixel_median_fn(depth, [w, s, e, n], val_lats, val_lons, val_depths_arr, resolution_m=res_m)
                    result["validation"]["pixel_comparison"] = pm
                    if 'pixel_level' in pm:
                        L.info(f"Pixel-median: RMSE={pm['pixel_level']['rmse']:.3f}m, "
                               f"R\u00b2={pm['pixel_level']['r2']:.4f}, "
                               f"S-44={pm['pixel_level']['s44_pass_pct']:.1f}% "
                               f"({pm['pixel_level']['n_pixels']} pixels)")
                except Exception as ex:
                    L.warning(f"Pixel-median comparison failed: {ex}")

    L.info("=== i-Boating Pipeline DONE (Professional) ===")
    return result
