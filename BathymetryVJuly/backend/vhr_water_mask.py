"""
vhr_water_mask.py — physically-grounded VHR water/land segmentation for the
production SDB depth endpoints.

Motivation
----------
The production depth paths (`/api/sdb-pro`, `/api/very-hr-clustered`,
`/api/very-hr-mle` → ``_run_s2_lyzenga_fast`` / ``run_cbr_vhr``) mask depth with a
coarse 10 m Sentinel-2 MNDWI/NDWI water mask.  Over a PORT (Khalifa Port: quays,
breakwaters, caissons, moored ships) a 10 m mixed pixel that is half concrete / half
water passes the S2 test as "water", so depth bleeds onto hard structures.

This module segments Mapbox VHR satellite RGB (~1–2 m) into water vs land with an
unsupervised, physically-grounded **2-component Gaussian-Mixture endmember unmixing**
(water / built-concrete-soil-vegetation) on water-discriminating colour + texture
features — NOT a hand-tuned threshold.  The per-VHR-pixel water probability is then
aggregated to the coarse output grid (area-average water FRACTION); an output pixel is
declared WATER only if its VHR water fraction ≥ ``water_frac_thresh`` (default 0.8).

The result is FUSED CONSERVATIVELY with the existing S2 mask (a pixel must pass BOTH).
Morphological cleanup removes specks and closes pinholes but **never grows water across
a pier** (opening only; hole-fill is bounded to sub-pier sizes).

Graceful degradation: if the Mapbox fetch fails, callers fall back to the S2 mask and
FLAG ``mask_source = "S2-only fallback"`` in the response metadata — the module never
silently pretends VHR masking ran.

Ported / consolidated from ``vhr_water_mask.py`` + ``apply_vhr_water_mask.py``
(commit d002b1f, branch vhr-shallow-technique).
"""

from __future__ import annotations
import os
import math
import hashlib
from io import BytesIO
from pathlib import Path
import numpy as np


# ── env knobs (defaults chosen so the fix is active but tunable) ──────────────
def _fenv(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _ienv(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


# NOTE: these are read LIVE at call time (see functions below) so tests /
# callers can toggle the env between calls without re-importing the module.
def _mask_on() -> bool:
    return os.environ.get("VHR_WATER_MASK", "1") == "1"   # master switch, default ON


def _osm_on() -> bool:
    # OSM-hybrid land override + symmetric restore (item 1.1). Default ON; graceful
    # AI-only fallback if OSM/Overpass is unavailable (never crashes the pipeline).
    return os.environ.get("VHR_OSM_MASK", "1") == "1"


def _blue_reclaim_on() -> bool:
    # Blue-water-index reclaim of GMM-"land" px (MASKMLE item 2.1). Default ON;
    # monotone (can only ADD water), OSM_land veto stays authoritative after it.
    return os.environ.get("VHR_BLUE_RECLAIM", "1") == "1"


def _coastline_on() -> bool:
    # MASKMLE R6 — globally-complete OSMCoastline land-polygons as PRIMARY OSM_land
    # source (crisp coast EVERYWHERE). Default ON; graceful fallback to feature-land
    # (osmnx) if the cached split file / regional GPKG is absent.
    return os.environ.get("VHR_OSM_COASTLINE", "1") == "1"


def _port_veto_on() -> bool:
    # MASK1M addendum — connected-component port-land veto (closes the "new/
    # reclaimed pad not yet in OSM" gap, additive-only). Default ON; graceful
    # no-op if no port tags exist in the bbox at all.
    return os.environ.get("VHR_PORT_LAND_VETO", "1") == "1"


def _refined_coastline_on() -> bool:
    # MASK-7 — CoastBA+ refined coastline (sub-pixel precision on ENGINEERED
    # segments, additive OSM_land input). Default ON; graceful no-op if no
    # refined GPKG exists for this bbox (cold start / site never processed).
    return os.environ.get("VHR_REFINED_COASTLINE", "1") == "1"


def _union_land(a, b):
    """Boolean OR of two (H,W) land masks, tolerating None on either side."""
    if a is None:
        return b
    if b is None:
        return a
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        return a if a.shape[0] * a.shape[1] >= b.shape[0] * b.shape[1] else b
    return a | b


def _blue_reclaim_thresh() -> float:
    # FROZEN at 0.3 (pre-registered; field-audited at ras_ghanada — real water
    # blobs 0.35–0.71, confirmed land < 0.3). Env override exists only for the
    # diagnostic histogram, NOT for tuning.
    return _fenv("VHR_BLUE_RECLAIM_THRESH", 0.3)


def _osm_recall_floor() -> float:
    # MASKMLE R5 item 5.1 — FROZEN OSM-land-recall floor for the coverage confidence gate.
    # Picked from the visible histogram GAP across the 13-ROI probe (see RESPONSE 5): the
    # developed/mapped coasts (khalifa, hudayriyat) sit high; the under-mapped coasts
    # (oualidia, dakhla) sit low, with a clean empty band between. FROZEN at 0.55 — this is
    # a NEW diagnostic metric, NOT a sweep of an existing frozen threshold. The env override
    # exists only for reproducing the histogram, not for tuning.
    return _fenv("VHR_OSM_RECALL_FLOOR", 0.55)


DEFAULT_WATER_FRAC = _fenv("VHR_WATER_FRAC", 0.8)                  # ≥0.8 VHR water → water
DEFAULT_ZOOM = _ienv("VHR_MASK_ZOOM", 16)                         # ~1-2 m/px resolves quays
DEFAULT_MAX_TILES = _ienv("VHR_MASK_MAX_TILES", 160)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "vhr_mask"


def log(m):
    print(m, flush=True)


def _json_dumps(obj) -> str:
    import json as _json
    try:
        return _json.dumps(obj)
    except Exception:
        return "{}"


# ── features ─────────────────────────────────────────────────────────────────
def physical_features(rgb: np.ndarray):
    """rgb (H,W,3) uint8 Mapbox tile → (feat (H*W,F) float32, valid (H*W,) bool)."""
    from scipy.ndimage import uniform_filter
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    eps = 1e-6
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    blue_water = (b - r) / (b + r + eps)          # >0 water, <0 beige/urban land
    blue_frac = b / (r + g + b + eps)
    green_frac = g / (r + g + b + eps)
    m1 = uniform_filter(lum, size=7, mode="nearest")
    m2 = uniform_filter(lum * lum, size=7, mode="nearest")
    texture = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    feat = np.stack([lum, blue_water, blue_frac, green_frac, texture], axis=-1)
    feat = feat.reshape(-1, feat.shape[-1]).astype(np.float32)
    flat = rgb.reshape(-1, 3)
    valid = ~((flat.sum(axis=1) < 0.02) | (flat.min(axis=1) > 0.98))
    return feat, valid


# ── segmentation core ────────────────────────────────────────────────────────
def segment_water_land(rgb, prior_water=None, min_object_frac=2e-4,
                       fill_hole_frac=2e-4, max_fit_px=40000, random_state=42):
    """Segment Mapbox RGB into water (True) / land (False) at full resolution.
    Returns (water bool (H,W), prob float (H,W), info dict)."""
    from sklearn.mixture import GaussianMixture
    from scipy.ndimage import (label as cc_label, binary_opening,
                               generate_binary_structure)

    H, W = rgb.shape[:2]
    feat, valid = physical_features(rgb)
    mu = feat[valid].mean(axis=0)
    sd = feat[valid].std(axis=0) + 1e-6
    z = (feat - mu) / sd

    rng = np.random.default_rng(random_state)
    idx_valid = np.flatnonzero(valid)
    if idx_valid.size == 0:
        return (np.zeros((H, W), bool), np.zeros((H, W), np.float32),
                {"method": "gmm2_physical_unmixing", "error": "no valid px",
                 "water_frac": 0.0, "valid_frac": 0.0},
                np.zeros((H, W), bool))
    fit_idx = rng.choice(idx_valid, size=min(max_fit_px, idx_valid.size), replace=False)
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          random_state=random_state, n_init=2, reg_covar=1e-5)
    gmm.fit(z[fit_idx])

    proba = np.empty((z.shape[0], 2), dtype=np.float32)
    step = 1_000_000
    for a in range(0, z.shape[0], step):
        proba[a:a + step] = gmm.predict_proba(z[a:a + step]).astype(np.float32)

    cmeans = gmm.means_
    # physical water score: dark(-lum) + blue(+blue_water) + high blue_frac + smooth(-texture)
    phys_score = (-cmeans[:, 0] + cmeans[:, 1] + cmeans[:, 2] + 0.5 * cmeans[:, 3]
                  - cmeans[:, 4])

    comp_sel = "physical_score"
    if prior_water is not None:
        pw = prior_water.reshape(-1).astype(bool) & valid
        if pw.sum() > 50:
            hard = proba.argmax(axis=1)
            overlap = [np.mean(hard[pw] == k) for k in (0, 1)]
            water_comp = int(np.argmax(overlap)); comp_sel = "prior_overlap"
        else:
            water_comp = int(np.argmax(phys_score))
    else:
        water_comp = int(np.argmax(phys_score))

    prob = proba[:, water_comp]
    prob[~valid] = 0.0
    water_flat = (prob >= 0.5)

    # ── Blue-water-index reclaim (MASKMLE item 2.1) ───────────────────────────
    # The 2-component GMM over-masks bright shallow banks: forced into 2 endmembers
    # on a genuinely 1-class (all-blue) crop, it splits water into bright-bank vs
    # dark-channel and mislabels one. Reclaim GMM-"land" px that ARE physically
    # blue: (B-R)/(B+R) ≥ 0.3. Physics — two-way red absorption by the water column
    # (a_w(665)≈0.42 vs a_w(490)≈0.015 m⁻¹) keeps even ~0.5 m of water over bright
    # carbonate sand ≥ 0.3, while dry sand / concrete / ship hulls are spectrally
    # flat (index ≈ 0). Monotone (can only ADD water). Threshold 0.3 FROZEN. The
    # OSM_land veto is applied AFTER this (in vhr_water_grid) so mapped quays/
    # breakwaters can NEVER be re-admitted; the product mask stays s2_water ∧
    # vhr_water so reclaim cannot add water where S2/NDWI says land. Opening below
    # runs AFTER the reclaim to remove any speckle it introduces.
    reclaimed_px = 0
    thr_bw = _blue_reclaim_thresh()
    if _blue_reclaim_on():
        blue_idx = feat[:, 1]                       # (B-R)/(B+R) channel
        reclaim = (~water_flat) & valid & (blue_idx >= thr_bw)
        reclaimed_px = int(reclaim.sum())
        if reclaimed_px:
            water_flat = water_flat | reclaim
            prob[reclaim] = np.maximum(prob[reclaim], 0.5)   # keep field coherent
    reclaim_frac = float(reclaimed_px) / float(max(int(valid.sum()), 1))
    water = water_flat.reshape(H, W)

    # morphological cleanup — opening removes specks (never grows water);
    # bounded hole-fill closes sub-pier pinholes only.
    struct = generate_binary_structure(2, 2)
    water = binary_opening(water, structure=struct, iterations=1)
    min_px = max(16, int(min_object_frac * H * W))
    hole_px = max(16, int(fill_hole_frac * H * W))
    lab, n = cc_label(water, structure=struct)
    if n > 0:
        sizes = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, dtype=bool); keep[1:] = sizes[1:] >= min_px
        water = keep[lab]
    land = ~water
    lab_l, n_l = cc_label(land, structure=struct)
    if n_l > 0:
        sizes_l = np.bincount(lab_l.ravel())
        small_land = np.zeros(n_l + 1, dtype=bool); small_land[1:] = sizes_l[1:] < hole_px
        water = water | small_land[lab_l]

    prob = prob.reshape(H, W)
    prob[~water & (prob >= 0.5)] = 0.49

    # ── MASKMLE 3.1: unambiguous-subtidal VHR mask (blue-water index ≥ 0.3) ────
    # The frozen R2/R3 physical criterion: (B-R)/(B+R) ≥ 0.3 ⇔ ≳0.3-0.5 m of
    # standing water over carbonate sand (two-way red absorption). Aggregated to
    # the product grid this is the "subtidal_cell" denominator for
    # retention_subtidal (Goal-1 acceptance) — cells below it are the drying /
    # intertidal fringe that a LAT chart product correctly masks. Threshold 0.3
    # FROZEN (same as the reclaim / restore guard). Returned separately so the
    # caller can aggregate it exactly like the water fraction.
    subtidal_vhr = ((feat[:, 1] >= _blue_reclaim_thresh()) & valid).reshape(H, W)

    info = {
        "method": "gmm2_physical_unmixing",
        "component_selection": comp_sel,
        "water_component": int(water_comp),
        "phys_score": [float(x) for x in phys_score],
        "water_frac": float(water.mean()),
        "valid_frac": float(valid.mean()),
        "min_object_px": int(min_px), "fill_hole_px": int(hole_px),
        "n_fit_px": int(fit_idx.size),
        # MASKMLE 2.1 blue-water reclaim audit trail
        "blue_reclaim_on": _blue_reclaim_on(),
        "blue_reclaim_thresh": thr_bw,
        "reclaimed_px": int(reclaimed_px),
        "reclaim_frac": round(reclaim_frac, 5),
        # MASKMLE 3.1: fraction of valid VHR px that are unambiguous-subtidal water
        "subtidal_frac_vhr": round(float(subtidal_vhr[valid.reshape(H, W)].mean())
                                   if valid.any() else 0.0, 5),
    }
    return water, prob, info, subtidal_vhr


# ── Mapbox fetch (native resolution, returns rgb + exact crop bbox) ───────────
def fetch_mapbox_native(bbox, token, zoom=DEFAULT_ZOOM, max_tiles=DEFAULT_MAX_TILES):
    """Fetch Mapbox satellite tiles covering bbox, crop to bbox at native res.
    Returns (rgb uint8 (H,W,3), crop_bbox [w,s,e,n]) or (None, None)."""
    import requests
    from PIL import Image as PILImage
    w_, s_, e_, n_ = bbox

    def lonlat_to_tile(lon, lat, z):
        x = int((lon + 180) / 360 * 2 ** z)
        y = int((1 - math.log(math.tan(math.radians(lat)) +
                              1 / math.cos(math.radians(lat))) / math.pi) / 2 * 2 ** z)
        return x, y

    def tile_bounds(x, y, z):
        nn = 2 ** z
        lon_w = x / nn * 360 - 180
        lon_e = (x + 1) / nn * 360 - 180
        lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / nn))))
        lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / nn))))
        return lon_w, lat_s, lon_e, lat_n

    x0, y0 = lonlat_to_tile(w_, n_, zoom)
    x1, y1 = lonlat_to_tile(e_, s_, zoom)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
    if n_tiles > max_tiles:
        if zoom <= 13:
            log(f"  VHR mask: {n_tiles} tiles > max at zoom {zoom}; giving up")
            return None, None
        log(f"  VHR mask: {n_tiles} tiles > max_tiles={max_tiles}; drop zoom to {zoom-1}")
        return fetch_mapbox_native(bbox, token, zoom=zoom - 1, max_tiles=max_tiles)

    ts = 512
    mosaic = np.zeros(((y1 - y0 + 1) * ts, (x1 - x0 + 1) * ts, 3), dtype=np.uint8)
    fetched = 0
    for xi in range(x0, x1 + 1):
        for yi in range(y0, y1 + 1):
            url = (f"https://api.mapbox.com/v4/mapbox.satellite/"
                   f"{zoom}/{xi}/{yi}@2x.png?access_token={token}")
            try:
                r = requests.get(url, timeout=20)
                if r.status_code != 200:
                    continue
                img = np.array(PILImage.open(BytesIO(r.content)).convert("RGB"))
                if img.shape[0] != ts:
                    img = np.array(PILImage.fromarray(img).resize((ts, ts), PILImage.BILINEAR))
                mosaic[(yi - y0) * ts:(yi - y0 + 1) * ts,
                       (xi - x0) * ts:(xi - x0 + 1) * ts] = img
                fetched += 1
            except Exception as ex:
                log(f"  VHR mask tile {xi}/{yi} failed: {ex}")
    if fetched == 0:
        return None, None

    full_w = tile_bounds(x0, y0, zoom)[0]
    full_e = tile_bounds(x1, y0, zoom)[2]
    full_n = tile_bounds(x0, y0, zoom)[3]
    full_s = tile_bounds(x0, y1, zoom)[1]
    mh, mw = mosaic.shape[:2]
    c0 = max(0, int((w_ - full_w) / (full_e - full_w) * mw))
    c1 = min(mw, int((e_ - full_w) / (full_e - full_w) * mw))
    r0 = max(0, int((full_n - n_) / (full_n - full_s) * mh))
    r1 = min(mh, int((full_n - s_) / (full_n - full_s) * mh))
    crop = mosaic[r0:r1, c0:c1]
    if crop.size == 0:
        return None, None
    cb_w = full_w + c0 / mw * (full_e - full_w)
    cb_e = full_w + c1 / mw * (full_e - full_w)
    cb_n = full_n - r0 / mh * (full_n - full_s)
    cb_s = full_n - r1 / mh * (full_n - full_s)
    mpp = 111000 * (cb_e - cb_w) * math.cos(math.radians((cb_n + cb_s) / 2)) / crop.shape[1]
    log(f"  VHR mask z{zoom}: {fetched} tiles -> crop {crop.shape[:2]} (~{mpp:.1f} m/px)")
    return crop, [cb_w, cb_s, cb_e, cb_n]


# ── aggregation VHR → output grid ────────────────────────────────────────────
def _frac_to_grid(vhr_bool, out_H, out_W):
    """Area-average a full-res VHR bool mask to (out_H,out_W) → water fraction.
    Legacy CO-REGISTERED path (assumes crop_bbox == product bbox). Kept only as a
    fallback when the crop/product bboxes are unavailable — prefer
    ``_frac_to_grid_geo`` (item 1.1c) which is coordinate-correct."""
    from PIL import Image as PILImage
    im = PILImage.fromarray((vhr_bool.astype(np.uint8) * 255))
    resample = getattr(PILImage, "BOX", PILImage.BILINEAR)   # area-average on downsample
    im = im.resize((out_W, out_H), resample)
    return np.asarray(im, dtype=np.float32) / 255.0


def _frac_to_grid_geo(vhr_bool, out_H, out_W, src_bbox, dst_bbox):
    """Georeferenced VHR-bool → output-grid FRACTION (item 1.1c, ported from
    ``apply_vhr_water_mask.downsample_majority``).

    The VHR mask spans ``src_bbox`` (the Mapbox tile-mosaic crop, snapped to tile
    pixel edges) while the product grid spans ``dst_bbox`` — up to ~2 m offset if
    stamped by a bare co-registered resize. This does a ``rasterio.warp.reproject``
    from the mask's own affine (src_bbox) to the product affine (dst_bbox) with
    ``Resampling.average`` (proper anti-aliased block-average for the ~4x down-
    sample), returning the per-output-cell fraction covered by True VHR pixels.
    Falls back to the legacy co-registered BOX resize if bboxes are missing or
    rasterio is unavailable."""
    if src_bbox is None or dst_bbox is None:
        return _frac_to_grid(vhr_bool, out_H, out_W)
    try:
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds
    except Exception:
        return _frac_to_grid(vhr_bool, out_H, out_W)
    Hs, Ws = vhr_bool.shape
    sw, ss, se, sn = [float(x) for x in src_bbox]
    dw, ds_, de, dn = [float(x) for x in dst_bbox]
    src_transform = from_bounds(sw, ss, se, sn, Ws, Hs)
    dst_transform = from_bounds(dw, ds_, de, dn, out_W, out_H)
    frac = np.zeros((out_H, out_W), dtype=np.float32)
    reproject(
        source=vhr_bool.astype(np.float32), destination=frac,
        src_transform=src_transform, src_crs="EPSG:4326",
        dst_transform=dst_transform, dst_crs="EPSG:4326",
        resampling=Resampling.average, src_nodata=None, dst_nodata=None,
    )
    return frac


def _osm_coverage(ai_water, osm_land, osm_water, crop_bbox, land_info, water_info,
                  coastline_land=None, coastline_info=None):
    """MASKMLE R5 item 5.1 / R6 item 6.2 — OSM-coverage confidence gate.

    ``ai_water`` is the GMM+blue-reclaim water mask at VHR resolution BEFORE the OSM
    override (so AI-land = ~ai_water is what the imagery independently found).

    ``osm_land_recall`` = of the SOLID AI coastal land in a ~50 m inward near-shore band
    (dilate the AI-water boundary inward), the fraction that OSM actually mapped as land —
    i.e. |AI_coastal_land ∩ OSM_land| / |AI_coastal_land| (AI-land sitting inside OSM_water
    is excluded from the denominator). A well-mapped developed coast has high recall (OSM
    caught the quays); where OSM under-covers a developed edge the GMM finds coastal land
    OSM missed → low recall.

    Class:
      absent — both land AND water fetch returned None/empty (no OSM evidence at all).
      sparse — recall below the FROZEN floor, OR (water side None/empty while AI finds an
               enclosed, non-open-sea water body OSM failed to map).
      rich   — otherwise.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure
    try:
        from backend.osm_land_mask import open_sea_component
    except ImportError:
        from osm_land_mask import open_sea_component  # type: ignore

    ai_water = np.asarray(ai_water, dtype=bool)
    H, W = ai_water.shape
    land_info = land_info or {}
    water_info = water_info or {}
    land_absent = (osm_land is None) or (not np.asarray(osm_land, bool).any())
    water_absent = (osm_water is None) or (not np.asarray(osm_water, bool).any())

    # metres per pixel from the crop bbox (for the ~50 m near-shore band width)
    try:
        w_, s_, e_, n_ = [float(x) for x in crop_bbox]
        mpp = 111000.0 * (e_ - w_) * math.cos(math.radians((n_ + s_) / 2)) / max(W, 1)
    except Exception:
        mpp = 2.0
    mpp = max(mpp, 0.5)
    iters = max(1, min(60, int(round(50.0 / mpp))))

    struct = generate_binary_structure(2, 2)
    water_dil = binary_dilation(ai_water, structure=struct, iterations=iters)
    band = water_dil & ~ai_water                       # AI-land ring within ~50 m of water

    ol = np.asarray(osm_land, bool) if osm_land is not None else np.zeros((H, W), bool)
    ow = np.asarray(osm_water, bool) if osm_water is not None else np.zeros((H, W), bool)
    coastal_land = band & ~ow                          # AI coastal land NOT inside OSM-water
    n_coastal = int(coastal_land.sum())
    covered = int((coastal_land & ol).sum())
    recall = (float(covered) / float(n_coastal)) if n_coastal > 0 else None

    # enclosed (non-open-sea) AI water body — the "OSM missed the lagoon" signal
    open_sea = open_sea_component(ai_water, structure=struct)
    enclosed = ai_water & ~open_sea
    enclosed_px = int(enclosed.sum())
    enclosed_frac = float(enclosed_px) / float(max(ai_water.size, 1))
    enclosed_present = (enclosed_frac >= 0.005) and (enclosed_px >= 200)

    floor = _osm_recall_floor()

    # ── MASKMLE R6 item 6.2: coastline-based coverage ────────────────────────
    # With the globally-complete OSMCoastline land-polygons unioned into `osm_land`,
    # the coast is authoritatively resolved EVERYWHERE the split file has a tile.
    # `coastline_available` = the cached file was read (source ok), even if 0 polys
    # (a genuinely open-water tile — no coast in view). `coastline_present` = the
    # bbox actually contains coastline land. Classification precedence:
    #   - coastline present & recall≥floor (coastline covers the AI-found coast) → rich (coastline)
    #   - coastline available but NO coast in view (open water, n_coastal==0)     → rich (open-water)
    #   - coastline present but recall<floor (a genuine positional/tile mismatch) → sparse (coastline gap)
    #   - coastline file ABSENT → fall back to the R5 feature-recall gate (below).
    coastline_info = coastline_info or {}
    csrc = coastline_info.get("source", "unavailable")
    coastline_available = isinstance(csrc, str) and csrc == "osm-coastline-landpoly"
    coastline_present = (coastline_land is not None
                         and bool(np.asarray(coastline_land, bool).any()))
    coverage_basis = "feature"  # R5 default

    possible_coastline_gap = False
    if coastline_available:
        coverage_basis = "coastline"
        # The globally-complete OSMCoastline is AUTHORITATIVE wherever it has a tile.
        #   present  → a crisp ≈MHW coast is defined here → rich (coastline-based).
        #   !present → the coastline finds NO permanent (≈MHW) land in this bbox →
        #              open-water / offshore-bank / intertidal tile; no permanent coast
        #              to misplace, so still HIGH confidence (AI "land" = drying flats).
        # NOTE: recall<floor WITH the coastline present is EXPECTED, not a failure — the
        # AI near-shore "land" ring is largely intertidal sabkha/bank that lies SEAWARD
        # of the MHW coastline the polygon correctly excludes (datum honesty). So recall
        # no longer downgrades a coastline-covered ROI; it stays a reported diagnostic.
        cls = "rich"
        # Soft diagnostic only (does NOT downgrade the class): coastline reports no land
        # yet the AI finds a large solid near-shore land ring → *might* be a true tile
        # gap rather than an offshore bank. Surfaced for audit, never fabricated.
        if (not coastline_present) and n_coastal > 0 and enclosed_present:
            possible_coastline_gap = True
    else:
        # R5 feature-only fallback (coastline file unavailable this run).
        if (osm_land is None and osm_water is None) or (land_absent and water_absent):
            cls = "absent"
        elif recall is not None and recall < floor:
            cls = "sparse"
        elif water_absent and enclosed_present:
            cls = "sparse"
        else:
            cls = "rich"

    return {
        "osm_coverage_class": cls,
        "coverage_basis": coverage_basis,
        "coastline_available": bool(coastline_available),
        "coastline_present": bool(coastline_present),
        "coastline_source": csrc,
        "coastline_vintage": coastline_info.get("vintage"),
        "coastline_n_polys": coastline_info.get("n_polys"),
        "coastline_region": coastline_info.get("region"),
        "coastline_datum": coastline_info.get("datum"),
        "possible_coastline_gap": bool(possible_coastline_gap),
        "osm_land_recall": round(recall, 4) if recall is not None else None,
        "osm_recall_floor": floor,
        "near_shore_land_px": n_coastal,
        "near_shore_covered_px": covered,
        "enclosed_water_present": bool(enclosed_present),
        "enclosed_water_frac": round(enclosed_frac, 5),
        "band_iters": int(iters), "mpp": round(mpp, 2),
        "osm_land_empty": bool(land_absent),
        "osm_water_empty": bool(water_absent),
        "osm_land_source": land_info.get("source"),
        "osm_water_source": water_info.get("source"),
        "osm_land_live": bool(land_info.get("live", False)),
        "osm_land_age_days": land_info.get("age_days"),
        "osm_land_n_features": land_info.get("n_features"),
    }


def _cache_key(bbox, out_H, out_W, zoom, thresh):
    # v2: OSM-hybrid symmetric rule + georeferenced transfer (item 1.1) changes the
    # produced mask, so the tag busts pre-OSM cached water_frac npz files.
    osm_tag = "osm1" if _osm_on() else "osm0"
    # v3 (MASKMLE 2.1): blue-water reclaim changes the produced mask → bust cache.
    bw_tag = "bw1" if _blue_reclaim_on() else "bw0"
    # v4 (MASKMLE 3.1): subtidal-cell + osm_land_frac now persisted in the npz.
    # v5 (MASKMLE 5.1): osm_coverage_class/recall now persisted in osm_json → bump so
    # pre-R5 caches (no coverage fields) recompute rather than emit a stale bare label.
    # v6 (MASKMLE R6): OSMCoastline global land-polygons unioned into OSM_land →
    # changes the produced mask + coverage class → bust pre-R6 caches.
    # v7 (MASK1M): connected-component port-land veto + MASK-7 refined-coastline
    # union both change the produced mask (Khalifa port-land leak fix) → bust
    # pre-MASK1M caches (coast2 -> coast3).
    coast_tag = "coast3" if _coastline_on() else "coast0"
    port_tag = "port1" if _port_veto_on() else "port0"
    refined_tag = "ref1" if _refined_coastline_on() else "ref0"
    s = (f"{[round(x,6) for x in bbox]}_{out_H}x{out_W}_z{zoom}_t{thresh:.3f}_"
         f"{osm_tag}_{bw_tag}_st1_cov1_{coast_tag}_{port_tag}_{refined_tag}")
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def vhr_water_grid(bbox, out_H, out_W, prior_water=None,
                   water_frac_thresh=None, zoom=None, token=None,
                   use_cache=True):
    """Fetch Mapbox VHR, unmix water/land, aggregate to (out_H,out_W).

    Returns dict:
      ok (bool)            — True if VHR segmentation ran; False → caller falls back
      water (H,W bool)     — output-grid water (VHR fraction ≥ thresh), None if not ok
      water_frac (H,W)     — per-cell VHR water fraction (float32), None if not ok
      land_frac (H,W)      — 1 - water_frac (built/soil/veg footprint), None if not ok
      meta (dict)          — method, thresh, zoom, crop bbox, seg info, error
    """
    thresh = DEFAULT_WATER_FRAC if water_frac_thresh is None else float(water_frac_thresh)
    zoom = DEFAULT_ZOOM if zoom is None else int(zoom)
    token = token or os.environ.get("MAPBOX_TOKEN", "")
    meta = {"method": "gmm2_physical_unmixing", "water_frac_thresh": thresh,
            "zoom": zoom, "error": None}

    if not token:
        meta["error"] = "MAPBOX_TOKEN not set"
        return {"ok": False, "water": None, "water_frac": None, "land_frac": None, "meta": meta}

    ckey = _cache_key(bbox, out_H, out_W, zoom, thresh)
    cpath = _CACHE_DIR / f"{ckey}.npz"
    if use_cache and cpath.exists():
        try:
            z = np.load(cpath, allow_pickle=False)
            wf = z["water_frac"].astype(np.float32)
            meta.update({"cached": True, "crop_bbox": z["crop_bbox"].tolist()})
            if "osm_json" in z.files:
                try:
                    import json as _json
                    meta["osm"] = _json.loads(str(z["osm_json"]))
                except Exception:
                    pass
            if "reclaimed_px" in z.files:
                meta["seg"] = {"reclaimed_px": int(z["reclaimed_px"]),
                               "reclaim_frac": float(z["reclaim_frac"])}
            water = wf >= thresh
            # MASKMLE 3.1: subtidal cell mask (cached blue≥0.3 fraction, 0.5 majority)
            subtidal_cell = None
            if "subtidal_frac" in z.files:
                subtidal_cell = z["subtidal_frac"].astype(np.float32) >= 0.5
            olg = None
            if "osm_land_frac" in z.files:
                olg = z["osm_land_frac"].astype(np.float32) >= 0.5
            return {"ok": True, "water": water, "water_frac": wf,
                    "land_frac": (1.0 - wf).astype(np.float32),
                    "subtidal_cell": subtidal_cell, "osm_land_grid": olg,
                    "meta": meta}
        except Exception:
            pass

    try:
        rgb, crop_bbox = fetch_mapbox_native(bbox, token, zoom=zoom)
    except Exception as ex:
        meta["error"] = f"mapbox fetch failed: {ex}"
        return {"ok": False, "water": None, "water_frac": None, "land_frac": None, "meta": meta}
    if rgb is None:
        meta["error"] = "mapbox returned no tiles"
        return {"ok": False, "water": None, "water_frac": None, "land_frac": None, "meta": meta}

    # coarse prior resampled to VHR res (used ONLY to pick the water GMM component)
    prior_vhr = None
    if prior_water is not None:
        try:
            from PIL import Image as PILImage
            pim = PILImage.fromarray((np.asarray(prior_water).astype(np.uint8) * 255))
            pim = pim.resize((rgb.shape[1], rgb.shape[0]), PILImage.NEAREST)
            prior_vhr = np.asarray(pim) > 127
        except Exception:
            prior_vhr = None

    try:
        vhr_water, _prob, seg_info, subtidal_vhr = segment_water_land(
            rgb, prior_water=prior_vhr)
    except Exception as ex:
        meta["error"] = f"segmentation failed: {ex}"
        return {"ok": False, "water": None, "water_frac": None, "land_frac": None, "meta": meta}

    # AI (GMM+blue-reclaim) water BEFORE the OSM override — the imagery's independent
    # coastline, used by the OSM-coverage confidence gate (item 5.1).
    ai_water_pre_osm = vhr_water.copy()

    # ── OSM-hybrid symmetric rule at VHR resolution (item 1.1a/1.1b) ──────────
    # land = OSM_land ∪ (AI_land ∩ ¬OSM_water ∩ ¬open_sea); enclosed GMM-land
    # blobs restored to water iff (i) touch no OSM_land, (ii) enclosed by
    # open_sea ∪ OSM_water, (iii) mean blue-water index (B-R)/(B+R) ≥ 0.3.
    # OSM_land is strictly authoritative and kills the quay-rim leak the pure GMM
    # mask leaves. Graceful AI-only fallback if OSM/Overpass unavailable.
    osm_meta = {"osm_applied": False, "osm_source": "disabled",
                "osm_land_frac": None, "n_blobs_restored": 0, "restored_px": 0,
                "osm_land_removed_px": 0}
    osm_land_vhr = None
    if _osm_on():
        try:
            try:
                from backend.osm_land_mask import (osm_land_for, osm_water_for,
                                                   apply_symmetric_osm_rule, coastline_land_for,
                                                   port_buffer_land_for)
                try:
                    from backend.osm_land_mask import refined_coastline_land_for
                except ImportError:
                    refined_coastline_land_for = None
            except ImportError:
                from osm_land_mask import (osm_land_for, osm_water_for,  # type: ignore
                                           apply_symmetric_osm_rule, coastline_land_for,
                                           port_buffer_land_for)
                try:
                    from osm_land_mask import refined_coastline_land_for  # type: ignore
                except ImportError:
                    refined_coastline_land_for = None
            vhr_shape = vhr_water.shape
            feature_land, land_info = osm_land_for(crop_bbox, vhr_shape)
            osm_water, water_info = osm_water_for(crop_bbox, vhr_shape)
            # ── MASKMLE R6 item 6.1/6.2: globally-complete OSMCoastline land-polygons
            # as the PRIMARY authoritative land source (crisp coast EVERYWHERE, coverage-
            # independent). OSM_land = coastline_land ∪ feature_land — coastline snaps the
            # coarse outer edge; the feature polygons + ~1 m VHR GMM add sub-coastline
            # detail (individual quays/pontoons/vessels inside a mapped port polygon).
            if _coastline_on():
                coast_land, coast_info = coastline_land_for(crop_bbox, vhr_shape)
            else:
                coast_land, coast_info = None, {"source": "disabled"}
            osm_land = _union_land(coast_land, feature_land)
            # ── MASK1M addendum (Khalifa port-land leak): OSM coverage of newly
            # reclaimed/under-construction port pads lags real imagery — a user-
            # reported defect (depth rendered over quays/terminals). Additive-only,
            # connected-component anchored to EXISTING mapped port infrastructure
            # (never fabricates land disconnected from any OSM port evidence).
            # Measured on Khalifa Port (real corridor imagery): closes 93.9% of
            # the VHR-dry-vs-OSM-water gap (0.94% -> 0.058% of corridor pixels).
            port_meta = {"applied": False}
            if _port_veto_on():
                try:
                    port_land, port_info = port_buffer_land_for(crop_bbox, vhr_shape, rgb)
                    if port_land is not None:
                        osm_land = _union_land(osm_land, port_land)
                        port_meta = {"applied": True, **port_info}
                except Exception as ex:
                    log(f"  VHR mask: port-buffer veto skipped ({ex})")
                    port_meta = {"applied": False, "error": str(ex)}
            # MASK-7: refined CoastBA+ coastline (sub-pixel precision on
            # ENGINEERED segments where computed) — additive, corridor-limited,
            # frozen-path preservation: substituted ONLY as extra OSM_land input,
            # never touches apply_symmetric_osm_rule/segment_water_land.
            refined_meta = {"applied": False}
            if _refined_coastline_on() and refined_coastline_land_for is not None:
                try:
                    refined_land, refined_info = refined_coastline_land_for(crop_bbox, vhr_shape)
                    if refined_land is not None:
                        osm_land = _union_land(osm_land, refined_land)
                        refined_meta = {"applied": True, **refined_info}
                except Exception as ex:
                    log(f"  VHR mask: refined-coastline step skipped ({ex})")
                    refined_meta = {"applied": False, "error": str(ex)}
            osm_land_vhr = osm_land
            # item 5.1/6.2: coverage confidence gate on the AI-vs-OSM agreement (pre-override),
            # now with the global coastline unioned in → coastal ROIs classify rich (coastline).
            cov = _osm_coverage(ai_water_pre_osm, osm_land, osm_water, crop_bbox,
                                land_info, water_info,
                                coastline_land=coast_land, coastline_info=coast_info)
            # item 5.2/5.3/5.4: surface land/water provenance (live/empty/error/stale).
            prov_common = {
                "osm_land_polys": land_info.get("n_polys"),
                "osm_land_lines": land_info.get("n_lines"),
                "osm_land_frac": land_info.get("osm_land_frac"),
                "osm_water_polys": water_info.get("n_polys"),
                "osm_water_frac": water_info.get("osm_water_frac"),
                "osm_land_source": land_info.get("source"),
                "osm_water_source": water_info.get("source"),
                "osm_land_live": bool(land_info.get("live", False)),
                "osm_water_live": bool(water_info.get("live", False)),
                "osm_land_age_days": land_info.get("age_days"),
                # MASKMLE R6 — coastline provenance (datum ≈ MHW, few-metre, not LAT)
                "coastline_source": coast_info.get("source"),
                "coastline_vintage": coast_info.get("vintage"),
                "coastline_n_polys": coast_info.get("n_polys"),
                "coastline_land_frac": coast_info.get("osm_land_frac"),
                "coastline_region": coast_info.get("region"),
                "coastline_datum": coast_info.get("datum"),
            }
            if osm_land is not None or osm_water is not None:
                vhr_water, sym_info = apply_symmetric_osm_rule(
                    vhr_water, _prob, osm_land=osm_land, osm_water=osm_water, rgb=rgb)
                osm_meta = {
                    "osm_applied": True,
                    "osm_source": (land_info.get("source") if osm_land is not None
                                   else water_info.get("source")),
                    "osm_land_removed_px": sym_info.get("osm_land_removed_px"),
                    "n_blobs_restored": sym_info.get("n_blobs_restored"),
                    "restored_px": sym_info.get("restored_px"),
                    "restored_blobs": sym_info.get("blobs"),
                }
            else:
                osm_meta = {"osm_applied": False, "osm_source": "unavailable",
                            "n_blobs_restored": 0, "restored_px": 0,
                            "osm_land_removed_px": 0}
            osm_meta.update(prov_common)
            osm_meta.update(cov)   # osm_coverage_class, osm_land_recall, etc.
            osm_meta["port_land_veto"] = port_meta
            osm_meta["refined_coastline"] = refined_meta
        except Exception as ex:
            log(f"  VHR mask: OSM step skipped ({ex})")
            osm_meta = {"osm_applied": False, "osm_source": f"error: {ex}",
                        "osm_land_frac": None, "n_blobs_restored": 0,
                        "restored_px": 0, "osm_land_removed_px": 0,
                        "osm_coverage_class": None, "osm_land_recall": None}

    # Georeferenced VHR→grid transfer (item 1.1c): reproject the VHR water bool
    # from its own crop_bbox affine to the product bbox affine (average), NOT a
    # co-registered BOX resize that assumes crop_bbox == product bbox.
    water_frac = _frac_to_grid_geo(vhr_water, out_H, out_W, crop_bbox, bbox)
    # MASKMLE 3.1: subtidal (blue≥0.3) VHR mask → product grid, same aggregation
    # (Resampling.average) → 0.5 majority = subtidal_cell. This is the
    # retention_subtidal denominator component (S2-water ∧ subtidal_cell).
    subtidal_frac = _frac_to_grid_geo(subtidal_vhr, out_H, out_W, crop_bbox, bbox)
    subtidal_cell = subtidal_frac >= 0.5
    # OSM-land aggregated to the product grid (≥0.5 majority) — lets the caller
    # report infrastructure pixels remaining as water (acceptance metric 1.1).
    osm_land_grid = None
    osm_land_frac_grid = None
    if osm_land_vhr is not None:
        try:
            osm_land_frac_grid = _frac_to_grid_geo(osm_land_vhr, out_H, out_W,
                                                   crop_bbox, bbox)
            osm_land_grid = osm_land_frac_grid >= 0.5
        except Exception:
            osm_land_grid = None
    meta.update({"crop_bbox": crop_bbox, "seg": seg_info, "osm": osm_meta,
                 "vhr_shape": list(vhr_water.shape), "cached": False})
    if use_cache:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _extra = {}
            if osm_land_frac_grid is not None:
                _extra["osm_land_frac"] = osm_land_frac_grid.astype(np.float16)
            np.savez_compressed(cpath, water_frac=water_frac.astype(np.float16),
                                subtidal_frac=subtidal_frac.astype(np.float16),
                                crop_bbox=np.asarray(crop_bbox, float),
                                osm_json=np.asarray(_json_dumps(osm_meta)),
                                reclaimed_px=np.asarray(seg_info.get("reclaimed_px", 0)),
                                reclaim_frac=np.asarray(seg_info.get("reclaim_frac", 0.0)),
                                **_extra)
        except Exception:
            pass
    water = water_frac >= thresh
    return {"ok": True, "water": water, "water_frac": water_frac,
            "land_frac": (1.0 - water_frac).astype(np.float32),
            "subtidal_cell": subtidal_cell,
            "osm_land_grid": osm_land_grid, "meta": meta}


def refine_water_mask(bbox, s2_water, out_H, out_W, water_frac_thresh=None,
                      zoom=None, token=None, use_cache=True, emit_preview=True):
    """Conservative fusion: refined = s2_water AND (VHR water fraction ≥ thresh).

    Returns (refined_water (H,W bool), meta dict). meta['mask_source'] is
    'VHR+S2' on success or 'S2-only fallback' when the VHR fetch/segmentation
    could not run (graceful degradation — never silently pretends VHR ran).
    On success meta also carries land_frac (VHR built/soil/veg footprint) so the
    caller can report which S2-water pixels the VHR test removed as land.
    """
    s2_water = np.asarray(s2_water, dtype=bool)
    if not _mask_on():
        return s2_water, {"mask_source": "S2-only (VHR mask disabled)",
                          "vhr_applied": False}
    thr = DEFAULT_WATER_FRAC if water_frac_thresh is None \
        else float(water_frac_thresh)
    thr = _fenv("VHR_WATER_FRAC", thr)   # live env override
    res = vhr_water_grid(bbox, out_H, out_W, prior_water=s2_water,
                         water_frac_thresh=thr, zoom=zoom,
                         token=token, use_cache=use_cache)
    if not res["ok"]:
        return s2_water, {"mask_source": "S2-only fallback", "vhr_applied": False,
                          "error": res["meta"].get("error"), "meta": res["meta"]}
    vhr_water = res["water"]
    if vhr_water.shape != s2_water.shape:
        # align by nearest (should match; guard against 1-px rounding)
        from PIL import Image as PILImage
        vim = PILImage.fromarray((vhr_water.astype(np.uint8) * 255))
        vim = vim.resize((s2_water.shape[1], s2_water.shape[0]), PILImage.NEAREST)
        vhr_water = np.asarray(vim) > 127
    refined = s2_water & vhr_water
    removed_mask = s2_water & ~vhr_water                   # S2-water cut as VHR-land
    removed = int(removed_mask.sum())
    osm = (res["meta"].get("osm") or {})
    seg = (res["meta"].get("seg") or {})
    osm_applied = bool(osm.get("osm_applied"))
    osm_source = osm.get("osm_source", "unavailable")
    # ── MASKMLE R5 item 5.2: coverage-aware HONEST mask_source ────────────────
    # Never emit a bare, clean "VHR+OSM+S2" when the OSM evidence is thin. The
    # coverage class (5.1) drives the label + a low_confidence flag for the UI.
    cov_class = osm.get("osm_coverage_class")
    recall = osm.get("osm_land_recall")
    wfrac = osm.get("osm_water_frac")
    water_empty = osm.get("osm_water_empty")
    coverage_basis = osm.get("coverage_basis")
    coastline_present = osm.get("coastline_present")
    coast_vintage = osm.get("coastline_vintage")
    low_confidence = False
    rtxt = f"{recall:.0%}" if isinstance(recall, (int, float)) else "n/a"
    # MASK-10: mask_source names the refined-coastline contribution explicitly
    # whenever it was actually applied (spec exact string
    # "VHR+OSM-coast(refined)+S2") — never claimed when refined_coastline_land_for
    # returned None (no product for this site yet).
    refined_applied = bool((osm.get("refined_coastline") or {}).get("applied"))
    coast_label = "OSM-coast(refined)" if refined_applied else "OSM-coast"
    if cov_class == "rich" and coverage_basis == "coastline":
        # MASKMLE R6 — the crisp-coast claim is now TRUE (globally-complete coastline).
        vtxt = f", vintage {coast_vintage}" if coast_vintage else ""
        if coastline_present:
            mask_source = f"VHR+{coast_label}+S2 (datum≈MHW{vtxt})"
        else:
            # coastline file present but bbox is open water (no coast in view)
            mask_source = f"VHR+{coast_label}+S2 (open water, no coast in view{vtxt})"
    elif cov_class == "rich":
        mask_source = "VHR+OSM+S2"
    elif cov_class == "sparse":
        low_confidence = True
        wtxt = ("water absent" if (water_empty or wfrac in (None, 0, 0.0))
                else f"water {wfrac:.1%}")
        mask_source = (f"VHR+S2 (OSM-sparse: land recall {rtxt}, {wtxt}) — "
                       f"GMM-authoritative at coastline")
    elif cov_class == "absent":
        low_confidence = True
        mask_source = "VHR+S2 (OSM-absent) — GMM-authoritative"
    elif osm_source in (None, "disabled"):
        mask_source = "VHR+S2 (OSM disabled)"
    elif osm_applied:
        # coverage class missing (legacy cache path) but OSM ran — least-surprise fallback
        mask_source = "VHR+OSM+S2"
    else:
        low_confidence = True
        mask_source = f"VHR+S2 (OSM unavailable: {osm_source})"
    # Infrastructure pixels (OSM-mapped land) still carrying water after masking —
    # the acceptance metric: should be 0 over a port ROI.
    infra_total = infra_remaining = None
    olg = res.get("osm_land_grid")
    if olg is not None and olg.shape == s2_water.shape:
        infra_total = int((olg & s2_water).sum())          # OSM-land px S2 called water
        infra_remaining = int((olg & refined).sum())       # ... still water after mask

    # ── MASKMLE 3.1: retention_subtidal + intertidal_excluded_frac ────────────
    # retention_subtidal uses ONLY the frozen 0.3 blue-index criterion: the
    # denominator is UNAMBIGUOUS-SUBTIDAL water (S2-water ∧ subtidal_cell) so the
    # drying/intertidal fringe (blue-index 0.2-0.3 ⇔ ≲0.3-0.5 m standing water,
    # correctly masked on a LAT chart) is not counted as "lost". intertidal_
    # excluded_frac surfaces the conservative exclusion for app honesty.
    sub_cell = res.get("subtidal_cell")
    retention_subtidal = None
    subtidal_denom = subtidal_retained = None
    intertidal_excluded_frac = None
    if sub_cell is not None and sub_cell.shape == s2_water.shape:
        denom_mask = s2_water & sub_cell
        subtidal_denom = int(denom_mask.sum())
        subtidal_retained = int((refined & sub_cell).sum())
        retention_subtidal = round(
            100.0 * subtidal_retained / max(subtidal_denom, 1), 3)
        intertidal_excluded = removed_mask & ~sub_cell     # masked drying fringe
        intertidal_excluded_frac = round(
            float(intertidal_excluded.sum()) / max(int(s2_water.sum()), 1), 5)

    meta = {
        "mask_source": mask_source, "vhr_applied": True,
        "low_confidence": low_confidence,
        # MASKMLE R5 5.1/5.2 — OSM coverage confidence + provenance surfaced for the UI
        "osm_coverage_class": cov_class,
        "coverage_basis": coverage_basis,
        "coastline_source": osm.get("coastline_source"),
        "coastline_vintage": coast_vintage,
        "coastline_present": coastline_present,
        "coastline_n_polys": osm.get("coastline_n_polys"),
        "coastline_region": osm.get("coastline_region"),
        "coastline_datum": osm.get("coastline_datum"),
        "osm_land_recall": recall,
        "osm_recall_floor": osm.get("osm_recall_floor"),
        "osm_land_polys": osm.get("osm_land_polys"),
        "osm_land_lines": osm.get("osm_land_lines"),
        "osm_water_polys": osm.get("osm_water_polys"),
        "osm_water_frac": osm.get("osm_water_frac"),
        "osm_land_source": osm.get("osm_land_source"),
        "osm_water_source": osm.get("osm_water_source"),
        "osm_land_live": osm.get("osm_land_live"),
        "osm_land_age_days": osm.get("osm_land_age_days"),
        "osm_applied": osm_applied, "osm_source": osm_source,
        "osm_land_frac": osm.get("osm_land_frac"),
        "osm_land_removed_px": osm.get("osm_land_removed_px"),
        "n_blobs_restored": osm.get("n_blobs_restored"),
        "restored_px": osm.get("restored_px"),
        "restored_blobs": osm.get("restored_blobs"),
        "reclaimed_px": seg.get("reclaimed_px"),
        "reclaim_frac": seg.get("reclaim_frac"),
        "water_frac_thresh": res["meta"]["water_frac_thresh"],
        "zoom": res["meta"]["zoom"],
        "s2_water_px": int(s2_water.sum()),
        "refined_water_px": int(refined.sum()),
        "infra_px_removed": removed,
        "infra_px_total": infra_total,
        "infra_px_remaining": infra_remaining,
        "retention_pct": round(100.0 * refined.sum() / max(int(s2_water.sum()), 1), 2),
        # MASKMLE 3.1 — Goal-1 acceptance metric + conservative-exclusion honesty
        "retention_subtidal_pct": retention_subtidal,
        "subtidal_denom_px": subtidal_denom,
        "subtidal_retained_px": subtidal_retained,
        "intertidal_excluded_frac": intertidal_excluded_frac,
        "osm_veto_px": osm.get("osm_land_removed_px"),
        "cached": res["meta"].get("cached", False),
    }
    # ── MASK-10: honesty surface (binding — see MASK1M_SPEC.md Section 3) ──────
    refined_site_quality = ((osm.get("refined_coastline") or {}).get("site_quality")) or {}
    meta.update({
        "coastline_datum": osm.get("coastline_datum") or "MHW (approx; not LAT)",
        "claim_withheld": refined_site_quality.get("claim_withheld"),
        "possible_coastline_gap": refined_site_quality.get("possible_coastline_gap"),
        "cross_ref_independent": refined_site_quality.get("cross_ref_independent"),
        "confidence_class_summary": {
            "CRISP-ENGINEERED": refined_site_quality.get("n_crisp_engineered"),
            "INSTANTANEOUS-SLOPING": refined_site_quality.get("n_instantaneous_sloping"),
            "INHERITED-PRIOR": refined_site_quality.get("n_inherited_prior"),
        } if refined_site_quality else None,
        "port_land_veto": osm.get("port_land_veto"),
        "refined_coastline": osm.get("refined_coastline"),
        # Two-class datum sentence + the exact permissible claim scope (Section 3)
        # — a static, spec-verbatim string; NEVER a blanket "1 m accurate" claim,
        # NEVER a horizontal LAT claim. Callers must not alter this string.
        "datum_honesty_sentence": (
            "Refined MHW coastline. On accepted ENGINEERED-VERTICAL segments "
            "(quays, caissons, breakwaters): ~1 m (1sigma) RELATIVE horizontal "
            "precision on the product/Mapbox grid where CRISP-ENGINEERED "
            "(dual-source independent, spread <=1.5 px). ABSOLUTE geodetic "
            "accuracy remains few-metre. On SLOPING-NATURAL segments: a "
            "conservative landward MHW envelope, no metre-level claim. "
            "INHERITED-PRIOR segments: MHW (approx; not LAT), few-metre "
            "positional. Horizontal boundary datum is MHW/structure-edge — "
            "never LAT."
        ),
    })
    # ── MASKMLE 3.U2: mask-QA preview overlay (PNG + small GeoTIFF) ────────────
    # Colour-code water-kept / GMM-land / OSM-veto / blue-reclaimed / intertidal-
    # excluded so the user can visually QA the mask per ROI. Written into the
    # downloads store; paths surfaced in water_mask_meta for the UI.
    if emit_preview:
        try:
            preview = _build_mask_preview(bbox, s2_water, refined, removed_mask,
                                          sub_cell, olg, res)
            if preview:
                meta.update(preview)
        except Exception as _px:
            log(f"  VHR mask: preview overlay skipped ({_px})")
    return refined, meta


def _mask_preview_classes(s2_water, refined, removed_mask, sub_cell, olg):
    """Return an (H,W) uint8 class raster for the mask-QA preview.
    0=not-water(transparent) 1=water-kept 2=GMM/AI-land-removed
    3=OSM-veto(infrastructure) 4=intertidal-excluded(drying fringe)."""
    H, W = s2_water.shape
    cls = np.zeros((H, W), dtype=np.uint8)
    cls[refined] = 1                                        # kept water
    rem = removed_mask & ~refined
    cls[rem] = 2                                            # GMM/AI land removed
    if sub_cell is not None and sub_cell.shape == (H, W):
        cls[rem & ~sub_cell] = 4                            # drying/intertidal fringe
    if olg is not None and olg.shape == (H, W):
        cls[olg & s2_water & ~refined] = 3                 # OSM-veto infrastructure
    return cls


# class → RGBA colour (water=teal, land=grey, osm=red, intertidal=orange)
_MASK_PREVIEW_COLOURS = {
    0: (0, 0, 0, 0),
    1: (0, 150, 200, 170),
    2: (120, 120, 120, 190),
    3: (220, 40, 40, 220),
    4: (245, 160, 40, 210),
}


def _build_mask_preview(bbox, s2_water, refined, removed_mask, sub_cell, olg, res):
    """Persist a PNG + small GeoTIFF mask-QA overlay to the downloads store.
    Returns dict of {mask_preview_png, mask_preview_geotiff, ...} URLs/paths."""
    from PIL import Image as PILImage
    import uuid as _uuid
    cls = _mask_preview_classes(s2_water, refined, removed_mask, sub_cell, olg)
    H, W = cls.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for k, col in _MASK_PREVIEW_COLOURS.items():
        rgba[cls == k] = col
    dl_dir = Path(__file__).resolve().parent / "ocean" / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    pid = "maskqa_" + _uuid.uuid4().hex[:10]
    png_name = f"{pid}.png"
    PILImage.fromarray(rgba, "RGBA").save(dl_dir / png_name, format="PNG",
                                          optimize=True)
    out = {
        "mask_preview_png": f"/downloads/{png_name}",
        "mask_preview_png_name": png_name,
        "mask_preview_bounds": [bbox[1], bbox[0], bbox[3], bbox[2]],  # S,W,N,E
        "mask_preview_legend": {
            "1": "water kept", "2": "GMM/AI land removed",
            "3": "OSM veto (infrastructure)", "4": "intertidal / drying (excluded)"},
        "mask_preview_class_px": {str(k): int((cls == k).sum())
                                  for k in _MASK_PREVIEW_COLOURS},
    }
    # Small single-band class GeoTIFF (georeferenced) for GIS QA.
    try:
        import rasterio
        from rasterio.transform import from_bounds
        tif_name = f"{pid}.tif"
        w_, s_, e_, n_ = bbox
        with rasterio.open(
                str(dl_dir / tif_name), "w", driver="GTiff", height=H, width=W,
                count=1, dtype="uint8", crs="EPSG:4326",
                transform=from_bounds(w_, s_, e_, n_, W, H),
                compress="deflate") as dst:
            dst.write(cls, 1)
            dst.set_band_description(
                1, "mask_qa 1=water 2=land 3=osm_veto 4=intertidal")
        out["mask_preview_geotiff"] = f"/downloads/{tif_name}"
        out["mask_preview_geotiff_name"] = tif_name
    except Exception as ex:
        log(f"  VHR mask: preview GeoTIFF skipped ({ex})")
    return out
