"""
osm_land_mask.py — OSM-authoritative LAND mask for VHR water/land segmentation.

Motivation
----------
The pure-AI (GMM-on-RGB) water/land segmentation in `vhr_water_mask.py` occasionally
leaks land into water (a bright wet quay, a light-coloured breakwater, a beached vessel
can all look "water-ish" on colour+texture alone).  OpenStreetMap already has semantically
exact polygons for ports, quays, buildings, breakwaters etc. — a mapped structure can
never be water regardless of pixel colour.  This module fetches those features (cached,
since the network here has transient DNS failures and Overpass calls take 1-2 min) and
rasterizes them to a boolean LAND mask at the resolution of the VHR imagery.

Design (binding, hybrid, OSM-authoritative — SYMMETRIC as of item 10.1):
    open_sea    = the connected component(s) of the AI water mask that touch the image
                  border (flood-filled "we are looking at open sea here" proxy — cheaper
                  and more robust than closing OSM natural=coastline into a sea polygon,
                  since it is derived from the imagery actually being masked).
    final_land  = OSM_land ∪ (AI_land ∩ ¬OSM_water ∩ ¬open_sea)
    final_water = ¬final_land
OSM_land is strictly authoritative for LAND and is NEVER overridden (a mapped structure
can never be water regardless of pixel colour or enclosure).  A GMM-land blob is flipped
back to water only if it (i) touches no OSM_land pixel, (ii) is enclosed by open_sea/
OSM_water (its water-adjacent ring lies inside open_sea ∪ OSM_water, not open coastline),
and (iii) its mean blue-water colour index (B-R)/(B+R) is >= 0.3 — a 7-site batch audit
found the GMM's own posterior probability does NOT separate turbid water from real
unmapped land (sand islet/mudflat/mangrove score in the same low-probability range as
genuine turbid water), but the raw colour index does cleanly (see `apply_symmetric_osm_rule`
docstring for the audited numbers). This restores turbid dredge-plume / suspended-sediment
patches the luminance+texture GMM misreads as land, without ever touching a ship hulled up
against a mapped quay (which stays part of the same connected "land" component as the quay,
so guard (i) excludes it for free) and without flipping real unmapped land (guard iii).

Line features (breakwater/pier/quay/groyne/jetty/dyke/embankment/coastline) are 1-D in
OSM and are buffered 10 m in the local UTM zone before rasterizing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # backend/ -> repo root (cache/ lives at repo root)
DEFAULT_CACHE_DIR = ROOT / "cache" / "osm_land"
DEFAULT_WATER_CACHE_DIR = ROOT / "cache" / "osm_water"

# ── OSMCoastline global LAND-POLYGONS ──────────────────────────────
# Globally-complete, side-of-line-resolved (OSM left-hand-land already applied) land
# polygons regenerated ~weekly by the OSMCoastline tool from the current OSM planet.
# Coverage-INDEPENDENT: a crisp coast exists for EVERY coastline on Earth, not just
# mapped cities — this is the "exact like OSM" land source (datum ≈ MHW, few-metre
# positional accuracy — NOT LAT, NOT sub-metre).
COASTLINE_URL = "https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip"
COASTLINE_CACHE_DIR = ROOT / "cache" / "osm_coastline"
COASTLINE_ZIP = COASTLINE_CACHE_DIR / "land-polygons-split-4326.zip"
COASTLINE_SHP = COASTLINE_CACHE_DIR / "land-polygons-split-4326" / "land_polygons.shp"
COASTLINE_VINTAGE_JSON = COASTLINE_CACHE_DIR / "vintage.json"
# Pre-clipped regional GPKG subsets for fast runtime clipping (battery + Railway); the
# full split shapefile stays for arbitrary user ROIs. Boxes are (w, s, e, n) in EPSG:4326.
COASTLINE_REGIONS = {
    "uae":     (50.0, 22.0, 58.0, 27.0),
    "morocco": (-17.5, 20.5, -8.0, 36.0),
}
COASTLINE_REGION_GPKG = {
    "uae":     COASTLINE_CACHE_DIR / "coastline_land_uae.gpkg",
    "morocco": COASTLINE_CACHE_DIR / "coastline_land_morocco.gpkg",
}
# Committed fallback (small, <10 MB each) so the UAE + Morocco coast works on Railway
# WITHOUT the 923 MB split archive (which stays gitignored under cache/). The cache/
# copies (built by prepare_coastline_cache) take precedence when present.
COASTLINE_COMMITTED_DIR = ROOT / "backend" / "data" / "coastline"
COASTLINE_COMMITTED_GPKG = {
    "uae":     COASTLINE_COMMITTED_DIR / "coastline_land_uae.gpkg",
    "morocco": COASTLINE_COMMITTED_DIR / "coastline_land_morocco.gpkg",
}

# Tag set verified over Khalifa Port (34 land polys + 4 line features -> exact port outline).
# MASK-12 fix: `landuse=port/harbour/industrial`, bare `harbour=yes`, and bare
# `industrial=yes` are OSM AREA polygons that ENCLOSE water (a port's operational
# boundary, not its dry land) — rasterizing them as solid land painted the entire
# 16-20 m dredged Khalifa basin as "land" (1,031 soundings >=0.5 m, median 17.74 m,
# destroyed — the root cause of the Section-2 sounding-envelope safety-gate failure).
# REMOVED from LAND_TAGS. Verified on real data: Khalifa Port's own polygon
# ("ميناء خليفة") is tagged landuse=industrial + industrial=port — i.e. real-world
# OSM mappers use `landuse=industrial` interchangeably with port tagging for a whole
# operational port complex (NOT a dry factory yard), so `industrial` cannot be
# selectively kept in the landuse list either (confirmed by direct osmnx query: it
# was the single largest polygon in the bbox, ~4x the next-largest). Port land is
# instead confirmed via `port_buffer_land_for`'s spectrally-gated connected-component
# mechanism (it can only ADD land where the VHR imagery independently agrees the
# pixel is dry AND it touches already-mapped genuine structure — never fills on the
# landuse tag alone).
# MASK-12 (same root-cause class, found during the acceptance re-check):
# `natural=coastline` RAW LineStrings, buffered a flat 10 m and rasterized with no
# side-of-line resolution, were the DOMINANT contributor (791 of 1,025 violations,
# 77%) — bigger than the port-tag bug itself. These lines are the OLD, unresolved OSM
# mainland-coastline linework (13-24 km long in the Khalifa bbox, tracing the
# PRE-dredging shoreline through what is now open, sounded channel water) and are
# REDUNDANT with `coastline_land_for()` (R6's authoritative, globally-complete,
# side-of-line-RESOLVED land-polygon product, already unioned in separately and
# already responsible for the much smaller 234-violation baseline). Removed from the
# feature-tag fetch — the module's own docstring already names
# `coastline_land_for` as "the exact OSM coastline" source; this raw-line duplicate
# only ever made it worse, never better.
LAND_TAGS = {
    "building": True,
    "man_made": ["pier", "breakwater", "quay", "groyne", "jetty", "dyke", "embankment"],
    "landuse": ["railway", "commercial", "construction",
                "retail", "residential", "brownfield", "depot"],
    "natural": ["beach", "sand", "bare_rock"],
    "place": ["island", "islet"],
    "aeroway": True,
}

# Water tags for the symmetric restore side (item 10.1). natural=coastline is deliberately
# NOT included here — closing an open coastline LineString into a sea polygon needs a
# side-of-line decision that OSM does not encode; we get the same "is this open sea"
# answer more robustly from the imagery itself via `open_sea_component` below.
WATER_TAGS = {
    "natural": ["water", "bay", "strait"],
    "waterway": ["dock", "canal", "riverbank"],
    "leisure": ["marina"],
}

LINE_BUFFER_M = 10.0


def log(m):
    print(m, flush=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def _bbox_cache_key(bbox, precision=5):
    """Round the bbox to `precision` decimals (~1 m at UAE latitudes) and hash it so
    repeated calls on the (numerically identical, floating-point-jittered) bbox hit the
    same cache file."""
    w, s, e, n = [round(float(x), precision) for x in bbox]
    key = f"{w},{s},{e},{n}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return h, key


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


# ── fetch (cached) ───────────────────────────────────────────────────────────

# osmnx raises this when the Overpass query is REACHABLE but returns 0 features
# (a trustworthy "nothing mapped here" — distinct from a transport/DNS/timeout error).
try:  # osmnx >= 1.x
    from osmnx._errors import InsufficientResponseError as _OSMEmptyError  # type: ignore
except Exception:  # pragma: no cover — very old / missing osmnx
    class _OSMEmptyError(Exception):
        pass


def fetch_osm_land(bbox, cache_dir=None, tags=None, timeout=180):
    """
    Fetch OSM land features (buildings, port/industrial landuse, breakwaters/piers/quays,
    natural=beach/coastline, islands) covering `bbox` = [west, south, east, north] (EPSG:4326).

    Returns ``(gdf_or_None, prov)`` where ``prov`` is a provenance dict
    carrying, always:
      source      — one of "osmnx" (fresh live non-empty), "osmnx-empty" (reachable, nothing
                    mapped — TRUSTWORTHY), "fetch-error" (transport/DNS/timeout — UNTRUSTWORTHY),
                    "stale-cache ({age}d, {n} feat)" (a re-served GOOD live cache), or
                    "unavailable" (no osmnx + no usable cache).
      live        — True ONLY for a fresh non-empty osmnx fetch.
      empty       — True for a reachable-but-nothing-mapped result.
      fetched_utc — ISO-8601 UTC of the underlying fetch (None if unknown).
      n_features  — feature count of the returned/served set.
      age_days    — age of a re-served cache (None for fresh).

    Cache trust (5.4): a cache is re-served on a transport error ONLY if its meta records
    ``live=True, empty=False`` (a fresh non-empty fetch). An empty/failed result is NEVER
    written as an authoritative .gpkg, and a re-served cache is always stamped "stale-cache".
    """
    import geopandas as gpd
    from datetime import datetime, timezone

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_hash, key_str = _bbox_cache_key(bbox)
    gpkg_path = cache_dir / f"{key_hash}.gpkg"
    meta_path = cache_dir / f"{key_hash}.json"
    tags = tags or LAND_TAGS

    def _now():
        return datetime.now(timezone.utc)

    def _read_meta():
        if meta_path.exists():
            try:
                return json.load(open(meta_path))
            except Exception:
                return {}
        return {}

    def _good_cache_exists():
        m = _read_meta()
        return gpkg_path.exists() and int(m.get("n_features", 0)) > 0 \
            and m.get("live", int(m.get("n_features", 0)) > 0) \
            and not m.get("empty", False)

    def _empty_prov():
        return {"source": "osmnx-empty", "live": False, "empty": True,
                "fetched_utc": _now().isoformat(), "n_features": 0, "age_days": 0.0}

    def _write_empty_meta():
        # Record the trustworthy-empty result — but NEVER clobber a good historical cache.
        if _good_cache_exists():
            return
        try:
            json.dump({"bbox": key_str, "n_features": 0, "live": False, "empty": True,
                       "fetched_utc": _now().isoformat()}, open(meta_path, "w"), indent=2)
        except Exception:
            pass

    def _serve_good_cache_on_error(err_note):
        """5.4: on a transport error, re-serve a cache ONLY if it was a fresh non-empty fetch."""
        if not gpkg_path.exists():
            return None, {"source": "fetch-error", "live": False, "empty": False,
                          "fetched_utc": None, "n_features": 0, "age_days": None, "note": err_note}
        m = _read_meta()
        n_feat = int(m.get("n_features", 0))
        live = bool(m.get("live", n_feat > 0))       # back-compat: pre-R5 caches were live non-empty
        empty = bool(m.get("empty", n_feat == 0))
        if (not live) or empty or n_feat == 0:
            return None, {"source": "fetch-error (untrusted cache)", "live": False,
                          "empty": empty, "fetched_utc": m.get("fetched_utc"),
                          "n_features": n_feat, "age_days": None, "note": err_note}
        try:
            g = gpd.read_file(str(gpkg_path))
            if g.crs is None:
                g = g.set_crs(4326)
        except Exception as ex:
            log(f"  [osm_land] cache read failed ({gpkg_path}): {ex}")
            return None, {"source": "fetch-error", "live": False, "empty": False,
                          "fetched_utc": m.get("fetched_utc"), "n_features": n_feat,
                          "age_days": None, "note": str(ex)}
        age_days = None
        fu = m.get("fetched_utc")
        if fu:
            try:
                age_days = round((_now() - datetime.fromisoformat(fu)).total_seconds()
                                 / 86400.0, 1)
            except Exception:
                age_days = None
        agestr = f"{age_days}d" if age_days is not None else "unknown-age"
        prov = {"source": f"stale-cache ({agestr}, {n_feat} feat)", "live": True,
                "empty": False, "fetched_utc": fu, "n_features": n_feat, "age_days": age_days}
        g.attrs["osm_source"] = prov["source"]
        log(f"  [osm_land] transport error ({err_note}); re-serving GOOD cache "
            f"({n_feat} feat, {agestr})")
        return g, prov

    try:
        import osmnx as ox
        ox.settings.requests_timeout = timeout
        ox.settings.overpass_rate_limit = True

        w, s, e, n = [float(x) for x in bbox]
        try:
            g = ox.features_from_bbox((w, s, e, n), tags)
        except _OSMEmptyError:
            g = None
        if g is None or len(g) == 0:
            # Reachable, nothing mapped -> TRUSTWORTHY empty. Prefer a good historical cache
            # if one exists (a suspicious live-empty should not erase known features).
            if _good_cache_exists():
                return _serve_good_cache_on_error("live fetch returned empty; using good cache")
            log(f"  [osm_land] osmnx returned EMPTY (reachable, nothing mapped) for bbox={key_str}")
            _write_empty_meta()
            return None, _empty_prov()
        g = g[["geometry"]].reset_index(drop=True)
        g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
        g.to_file(str(gpkg_path), driver="GPKG")
        fetched_utc = _now().isoformat()
        json.dump({"bbox": key_str, "n_features": int(len(g)),
                   "geom_types": g.geometry.type.value_counts().to_dict(),
                   "live": True, "empty": False, "fetched_utc": fetched_utc},
                  open(meta_path, "w"), indent=2)
        g.attrs["osm_source"] = "osmnx"
        log(f"  [osm_land] fetched {len(g)} features for bbox={key_str} -> cached {gpkg_path.name}")
        return g, {"source": "osmnx", "live": True, "empty": False,
                   "fetched_utc": fetched_utc, "n_features": int(len(g)), "age_days": 0.0}
    except _OSMEmptyError:
        if _good_cache_exists():
            return _serve_good_cache_on_error("live fetch returned empty; using good cache")
        _write_empty_meta()
        return None, _empty_prov()
    except Exception as ex:
        # Transport / DNS / timeout / parse error -> UNTRUSTWORTHY (5.3 fetch-error).
        log(f"  [osm_land] osmnx fetch failed ({ex}); trying trusted cache...")
        return _serve_good_cache_on_error(str(ex))


# ── rasterize ────────────────────────────────────────────────────────────────

def rasterize_osm_land(gdf, out_shape, bbox, line_buffer_m: float = LINE_BUFFER_M,
                       include_lines: bool = True):
    """
    Rasterize OSM land features to a boolean (H, W) LAND mask over `bbox`.

    Polygons are rasterized directly.  Line features (breakwater/pier/quay/coastline etc.
    come through OSM as LineString/MultiLineString, not polygons) are buffered
    `line_buffer_m` metres in the local UTM zone before rasterizing, since they are 1-D
    features in OSM but represent a physical structure with real width.

    Returns bool (H, W) array, or an all-False array if gdf is empty/None.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    H, W = out_shape
    if gdf is None or len(gdf) == 0:
        return np.zeros((H, W), dtype=bool)

    w, s, e, n = [float(x) for x in bbox]
    geoms = []

    poly_mask = gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    line_mask = gdf.geometry.type.isin(["LineString", "MultiLineString"])

    polys = gdf.loc[poly_mask, "geometry"]
    for gm in polys:
        if gm is not None and not gm.is_empty:
            geoms.append(gm)

    lines = gdf.loc[line_mask, "geometry"]
    if len(lines) > 0 and include_lines:  # 5.6: include_lines=False measures polygon-only land
        lon_c = (w + e) / 2.0
        lat_c = (s + n) / 2.0
        epsg = _utm_epsg(lon_c, lat_c)
        try:
            import geopandas as gpd
            lines_gs = gpd.GeoSeries(lines.values, crs=gdf.crs or 4326)
            buffered = lines_gs.to_crs(epsg).buffer(line_buffer_m).to_crs(4326)
            for gm in buffered:
                if gm is not None and not gm.is_empty:
                    geoms.append(gm)
        except Exception as ex:
            log(f"  [osm_land] line buffering failed ({ex}); skipping line features")

    if not geoms:
        return np.zeros((H, W), dtype=bool)

    transform = from_bounds(w, s, e, n, W, H)
    land = rasterize([(gm, 1) for gm in geoms], out_shape=(H, W), transform=transform,
                      fill=0, dtype="uint8")
    return land.astype(bool)


# ── convenience ──────────────────────────────────────────────────────────────

def _prov_info(prov):
    """Common provenance fields surfaced from fetch_osm_land into the *_for() info dicts."""
    prov = prov or {}
    return {
        "source": prov.get("source", "unavailable"),
        "live": bool(prov.get("live", False)),
        "empty": bool(prov.get("empty", False)),
        "fetched_utc": prov.get("fetched_utc"),
        "age_days": prov.get("age_days"),
        "n_features": prov.get("n_features", 0),
    }


def osm_land_for(bbox, out_shape, cache_dir=None, tags=None, line_buffer_m: float = LINE_BUFFER_M,
                 include_lines: bool = True):
    """
    Fetch (cached) + rasterize OSM land for `bbox` at `out_shape` = (H, W).
    Returns (land bool (H, W), info dict) or (None, info) if OSM is unavailable.
    info always has: n_polys, n_lines, osm_land_frac, source, live, empty, fetched_utc,
    age_days, n_features. `include_lines=False` (5.6 diagnostic) rasterizes polygons only.
    """
    gdf, prov = fetch_osm_land(bbox, cache_dir=cache_dir, tags=tags)
    if gdf is None:
        info = {"n_polys": 0, "n_lines": 0, "osm_land_frac": None}
        info.update(_prov_info(prov))
        return None, info

    n_polys = int(gdf.geometry.type.isin(["Polygon", "MultiPolygon"]).sum())
    n_lines = int(gdf.geometry.type.isin(["LineString", "MultiLineString"]).sum())
    land = rasterize_osm_land(gdf, out_shape, bbox, line_buffer_m=line_buffer_m,
                              include_lines=include_lines)
    info = {
        "n_polys": n_polys,
        "n_lines": n_lines,
        "osm_land_frac": float(land.mean()),
    }
    info.update(_prov_info(prov))
    return land, info


def osm_water_for(bbox, out_shape, cache_dir=None, tags=None, line_buffer_m: float = LINE_BUFFER_M):
    """
    Fetch (cached, separate cache dir from osm_land) + rasterize OSM WATER for `bbox` at
    `out_shape` = (H, W) — natural=water/bay/strait, waterway=dock/canal/riverbank,
    leisure=marina. Used as the "restore" side of the symmetric OSM rule (item 10.1):
    these polygons are authoritative water and are never overridden by the GMM mask.
    Returns (water bool (H, W), info dict) or (None, info) if OSM is unavailable.
    info always has: n_polys, n_lines, osm_water_frac, source.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_WATER_CACHE_DIR
    gdf, prov = fetch_osm_land(bbox, cache_dir=cache_dir, tags=tags or WATER_TAGS)
    if gdf is None:
        info = {"n_polys": 0, "n_lines": 0, "osm_water_frac": None}
        info.update(_prov_info(prov))
        return None, info

    n_polys = int(gdf.geometry.type.isin(["Polygon", "MultiPolygon"]).sum())
    n_lines = int(gdf.geometry.type.isin(["LineString", "MultiLineString"]).sum())
    water = rasterize_osm_land(gdf, out_shape, bbox, line_buffer_m=line_buffer_m)
    info = {
        "n_polys": n_polys,
        "n_lines": n_lines,
        "osm_water_frac": float(water.mean()),
    }
    info.update(_prov_info(prov))
    return water, info


# ── OSMCoastline global land-polygons ──────────────────────────

def _bbox_in_region(bbox):
    """Return the region key whose box fully contains `bbox`, else None."""
    w, s, e, n = [float(x) for x in bbox]
    for key, (rw, rs, re_, rn) in COASTLINE_REGIONS.items():
        if w >= rw and s >= rs and e <= re_ and n <= rn:
            return key
    return None


def coastline_vintage():
    """Generation/vintage date of the cached land-polygons (ISO date str) or None."""
    if COASTLINE_VINTAGE_JSON.exists():
        try:
            return json.load(open(COASTLINE_VINTAGE_JSON)).get("vintage")
        except Exception:
            pass
    for p in (COASTLINE_SHP, COASTLINE_ZIP):
        if p.exists():
            try:
                from datetime import datetime, timezone
                return datetime.fromtimestamp(p.stat().st_mtime,
                                              timezone.utc).date().isoformat()
            except Exception:
                pass
    return None


def _coastline_source_path(bbox):
    """Pick the fastest cached source for `bbox`: a pre-clipped regional GPKG if the
    bbox sits inside one and the GPKG exists, else the full split shapefile. Returns
    (path_or_None, region_key_or_'full'_or_None)."""
    region = _bbox_in_region(bbox)
    if region is not None:
        # cache copy first (freshest), then the committed fallback (Railway)
        for gpkg in (COASTLINE_REGION_GPKG.get(region),
                     COASTLINE_COMMITTED_GPKG.get(region)):
            if gpkg is not None and gpkg.exists():
                return gpkg, region
    if COASTLINE_SHP.exists():
        return COASTLINE_SHP, "full"
    return None, None


def coastline_land_for(bbox, out_shape, source_path=None):
    """Rasterize the globally-complete OSMCoastline land polygons over
    `bbox` = [w, s, e, n] (EPSG:4326) to a boolean (H, W) LAND mask at `out_shape`.

    Reads the cached split shapefile (or a pre-clipped regional GPKG) with a `bbox=`
    filtered read (pyogrio/geopandas), so a ROI clips in <1 s even off the 900 MB file.

    Returns ``(land bool (H, W) or None, info)`` where info always carries:
      source   — "osm-coastline-landpoly" (success), "osm-coastline-unavailable"
                 (no cached file), or "osm-coastline-error: …".
      vintage  — generation date of the land-polygons (ISO, from the zip/shapefile).
      n_polys  — clipped polygon count.
      datum    — "MHW (approx; not LAT)" honesty note.
      region   — which cached source served the clip ("uae"/"morocco"/"full").
      osm_land_frac — rasterized land fraction.

    Datum honesty (binding): the OSM natural=coastline convention ≈ Mean High Water,
    a FEW-METRE-accurate HIGH-water line at the LANDWARD edge of the intertidal zone —
    NOT LAT, NOT sub-metre. It is used ONLY as authoritative land / crisp outer
    envelope; because the intertidal flat lies SEAWARD of it, it only ADDS land and
    never removes intertidal/subtidal water.
    """
    H, W = out_shape
    info = {"source": "osm-coastline-unavailable", "vintage": coastline_vintage(),
            "n_polys": 0, "datum": "MHW (approx; not LAT)", "region": None,
            "osm_land_frac": None}
    path = source_path if source_path is not None else _coastline_source_path(bbox)[0]
    region = _coastline_source_path(bbox)[1] if source_path is None else "explicit"
    if path is None:
        return None, info
    try:
        import geopandas as gpd
        w_, s_, e_, n_ = [float(x) for x in bbox]
        gdf = gpd.read_file(str(path), bbox=(w_, s_, e_, n_))
        info["region"] = region
        if gdf is None or len(gdf) == 0:
            # A genuine tile gap OR the bbox is fully open ocean (no land here). Honest empty.
            info["source"] = "osm-coastline-landpoly"
            info["n_polys"] = 0
            info["osm_land_frac"] = 0.0
            return np.zeros((H, W), dtype=bool), info
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        elif str(gdf.crs).upper() not in ("EPSG:4326",):
            gdf = gdf.to_crs(4326)
        land = rasterize_osm_land(gdf, out_shape, bbox, include_lines=False)
        info["source"] = "osm-coastline-landpoly"
        info["n_polys"] = int(len(gdf))
        info["osm_land_frac"] = float(land.mean())
        return land, info
    except Exception as ex:
        log(f"  [coastline] read/rasterize failed ({ex})")
        info["source"] = f"osm-coastline-error: {ex}"
        return None, info


def prepare_coastline_cache(regions=None, force=False):
    """One-time setup: unzip the downloaded land-polygons split shapefile, build the
    GDAL .qix spatial index, stamp the vintage, and pre-clip the small regional GPKGs.

    Run after `COASTLINE_ZIP` is downloaded:
        curl -o cache/osm_coastline/land-polygons-split-4326.zip \
            https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip
        .venv/bin/python3 -m backend.osm_land_mask   # → unzip + .qix + regional GPKGs
    """
    import subprocess
    from datetime import datetime, timezone
    COASTLINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # 1) unzip
    if force or not COASTLINE_SHP.exists():
        if not COASTLINE_ZIP.exists():
            raise FileNotFoundError(f"{COASTLINE_ZIP} not present — download it first:\n"
                                    f"  curl -o {COASTLINE_ZIP} {COASTLINE_URL}")
        import zipfile
        log(f"  [coastline] unzipping {COASTLINE_ZIP.name} …")
        with zipfile.ZipFile(COASTLINE_ZIP) as zf:
            zf.extractall(COASTLINE_CACHE_DIR)
    if not COASTLINE_SHP.exists():
        raise FileNotFoundError(f"unzip did not produce {COASTLINE_SHP}")
    # 2) vintage stamp (zip mtime = OSM planet generation)
    try:
        vintage = datetime.fromtimestamp(COASTLINE_ZIP.stat().st_mtime,
                                         timezone.utc).date().isoformat()
    except Exception:
        vintage = datetime.now(timezone.utc).date().isoformat()
    json.dump({"vintage": vintage, "url": COASTLINE_URL,
               "prepared_utc": datetime.now(timezone.utc).isoformat()},
              open(COASTLINE_VINTAGE_JSON, "w"), indent=2)
    # 3) GDAL spatial index (.qix) for <1 s bbox reads off the full file
    qix = COASTLINE_SHP.with_suffix(".qix")
    if force or not qix.exists():
        try:
            layer = COASTLINE_SHP.stem
            subprocess.run(["ogrinfo", "-sql",
                            f'CREATE SPATIAL INDEX ON "{layer}"', str(COASTLINE_SHP)],
                           check=True, capture_output=True, timeout=1800)
            log(f"  [coastline] built spatial index {qix.name}")
        except Exception as ex:
            log(f"  [coastline] .qix build via ogrinfo failed ({ex}); "
                f"pyogrio still bbox-reads (slower). continuing.")
    # 4) pre-clip regional GPKGs
    import geopandas as gpd
    for key in (regions or list(COASTLINE_REGIONS)):
        box = COASTLINE_REGIONS[key]
        out = COASTLINE_REGION_GPKG[key]
        if out.exists() and not force:
            log(f"  [coastline] regional GPKG exists: {out.name}")
            continue
        log(f"  [coastline] clipping {key} {box} → {out.name} …")
        g = gpd.read_file(str(COASTLINE_SHP), bbox=box)
        if g.crs is None:
            g = g.set_crs(4326)
        g.to_file(str(out), driver="GPKG")
        log(f"  [coastline]   {len(g)} polys, {out.stat().st_size/1e6:.2f} MB")
    log(f"  [coastline] cache ready (vintage {vintage}).")
    return {"vintage": vintage, "shp": str(COASTLINE_SHP),
            "regions": {k: str(v) for k, v in COASTLINE_REGION_GPKG.items()}}


# ── symmetric OSM rule (item 10.1) ──────────────────────────────────────────

def open_sea_component(water, structure=None):
    """
    Border-connected component(s) of a boolean `water` array — a cheap, imagery-derived
    proxy for "open sea" (rather than trying to close OSM natural=coastline into a
    polygon). Any connected water component touching row 0/H-1 or col 0/W-1 is open_sea;
    interior water components that never reach the frame edge (a fully OSM-water-enclosed
    lagoon, say) are NOT open_sea by this definition, but the caller still gets to restore
    them via OSM_water.
    """
    from scipy.ndimage import label as cc_label

    if structure is None:
        structure = np.ones((3, 3), dtype=int)
    water = np.asarray(water, dtype=bool)
    H, W = water.shape
    lab, n = cc_label(water, structure=structure)
    if n == 0:
        return np.zeros((H, W), dtype=bool)
    border_labels = set(np.unique(lab[0, :]).tolist()) | set(np.unique(lab[-1, :]).tolist()) \
        | set(np.unique(lab[:, 0]).tolist()) | set(np.unique(lab[:, -1]).tolist())
    border_labels.discard(0)
    if not border_labels:
        return np.zeros((H, W), dtype=bool)
    keep = np.zeros(n + 1, dtype=bool)
    for bl in border_labels:
        keep[bl] = True
    return keep[lab]


def apply_symmetric_osm_rule(water, prob, osm_land=None, osm_water=None, rgb=None,
                              min_water_prob: float = 0.0, min_blue_water: float = 0.3,
                              enclose_tol: float = 0.05, min_ring_px: int = 4):
    """
    Symmetric OSM land/water combination (item 10.1):
        open_sea    = open_sea_component(water & ~OSM_land)
        final_land  = OSM_land ∪ (AI_land ∩ ¬OSM_water ∩ ¬open_sea)
        final_water = ¬final_land

    OSM_land is applied first and stays strictly authoritative (never restored). Every
    remaining GMM-land connected component is then a candidate for restoring to water iff:
      (i)   it touches NO OSM_land pixel (a ship hulled against a mapped quay merges into
            the SAME connected component as the quay once OSM_land is subtracted, so this
            guard excludes berthed ships automatically — no separate check needed);
      (ii)  it does not touch the array border (a border-touching blob may be real
            coastline continuing off-frame, not an enclosed patch);
      (iii) it is enclosed by open_sea ∪ OSM_water: dilate the blob by 2 px, take the ring
            pixels that are water — at most `enclose_tol` of them may lie OUTSIDE
            open_sea ∪ OSM_water (small tolerance for GMM boundary speckle); and
      (iv)  its mean blue-water colour index (B-R)/(B+R) from `rgb` is >= `min_blue_water`
            (see NOTE below — this, not the GMM posterior, is the real land/turbid-water
            discriminator); and its mean GMM water-probability is >= `min_water_prob`.

    NOTE on the guard, revised after a 7-site batch audit: the GMM posterior (`prob`) is
    UNRELIABLE here — at Khalifa the two audited turbid dredge-plume blobs score mean
    water-prob 0.083/0.012 (the SAME GMM that misclassifies them as land also scores them
    confidently non-water), but real unmapped land (a sand islet, a mudflat, mangrove
    canopy — found flipped at yas_lagoon/eastern_mangroves when this was first tried with
    prob as the only guard) scores in the SAME 0.04-0.23 range, so no `min_water_prob`
    threshold separates the two classes. The physical blue-water colour index does:
    every visually-confirmed genuine turbid-water blob across Khalifa/bu_tinah/marawah/
    ras_ghanada scored blue_water in [0.35, 0.71]; every visually-confirmed real-land blob
    (sand islet, causeway, mangrove canopy at yas_lagoon/eastern_mangroves) scored in
    [-0.13, 0.29] — clean separation at 0.3, matching the module's own physical convention
    (`vhr_water_mask.physical_features`: blue_water > 0 water, < 0 beige/urban land).
    `min_water_prob` is kept as a secondary knob (default 0.0, effectively off) since it adds
    no discrimination once blue_water gates; `min_blue_water` is the operative guard.

    water : bool (H, W) AI water mask (pre-OSM).
    prob  : float (H, W) GMM water-probability field returned by segment_water_land
            (same shape as `water`), or None.
    osm_land / osm_water : bool (H, W) or None.
    rgb   : uint8 (H, W, 3) or None — the SAME Mapbox image `water`/`prob` were segmented
            from. If None, the blue-water guard is skipped (legacy/no-image callers) and
            only `min_water_prob` gates — NOT recommended given the finding above.

    Returns (final_water bool (H, W), info dict) with osm_land_removed_px, restored_px,
    n_blobs_restored, and a per-blob list (px, row_c, col_c, mean_water_prob,
    mean_blue_water, lon/lat left to the caller since this module has no bbox here beyond
    what's needed for pixel geometry).
    """
    from scipy.ndimage import label as cc_label, binary_dilation, generate_binary_structure

    water = np.asarray(water, dtype=bool)
    H, W = water.shape
    struct = generate_binary_structure(2, 2)
    osm_land = np.asarray(osm_land, dtype=bool) if osm_land is not None else np.zeros((H, W), dtype=bool)
    osm_water = np.asarray(osm_water, dtype=bool) if osm_water is not None else np.zeros((H, W), dtype=bool)

    blue_water = None
    if rgb is not None:
        rgb_f = np.asarray(rgb, dtype=np.float32) / 255.0
        r_, b_ = rgb_f[..., 0], rgb_f[..., 2]
        blue_water = (b_ - r_) / (b_ + r_ + 1e-6)

    n_before = int(water.sum())
    water0 = water & ~osm_land
    osm_land_removed_px = n_before - int(water0.sum())

    open_sea = open_sea_component(water0, structure=struct)
    sea_domain = open_sea | osm_water

    land = ~water0
    lab, n = cc_label(land, structure=struct)
    restored_px = 0
    blob_report = []
    for i in range(1, n + 1):
        blob = lab == i
        if blob[0, :].any() or blob[-1, :].any() or blob[:, 0].any() or blob[:, -1].any():
            continue  # touches border: possible real coastline continuing off-frame
        if (blob & osm_land).any():
            continue  # guard: never flip anything touching a mapped land feature
        ring = binary_dilation(blob, structure=struct, iterations=2) & ~blob
        ring_water = ring & water0
        if int(ring_water.sum()) < min_ring_px:
            continue  # not adjacent to any water at all (interior of a bigger land mass)
        leak = ring_water & ~sea_domain
        leak_frac = float(leak.sum()) / float(ring_water.sum())
        if leak_frac > enclose_tol:
            continue  # touches open coastline / non-sea water, not enclosed
        mean_prob = float(prob[blob].mean()) if prob is not None else 1.0
        if mean_prob < min_water_prob:
            continue  # confident-land pixel (dark rock/islet) — keep as land
        mean_bw = float(blue_water[blob].mean()) if blue_water is not None else None
        if mean_bw is not None and mean_bw < min_blue_water:
            continue  # NOT physically blue -> real unmapped land (sand/mudflat/mangrove)
        water0[blob] = True
        restored_px += int(blob.sum())
        ys, xs = np.nonzero(blob)
        blob_report.append({"px": int(blob.sum()), "row_c": float(ys.mean()),
                             "col_c": float(xs.mean()), "mean_water_prob": mean_prob,
                             "mean_blue_water": mean_bw, "leak_frac": leak_frac})

    info = {
        "osm_land_removed_px": int(osm_land_removed_px),
        "restored_px": int(restored_px),
        "n_blobs_restored": int(len(blob_report)),
        "blobs": blob_report,
    }
    return water0, info


# ── MASK1M port-land veto (Khalifa port-land leak addendum) ────────────────

PORT_TAGS = {
    "landuse": ["industrial", "port", "harbour"],
    "man_made": ["pier", "breakwater", "quay", "groyne", "jetty"],
    "harbour": True,
    "industrial": True,
}
PORT_BUFFER_M = 30.0            # how far a new/reclaimed pad can extend beyond mapped port land
PORT_BLUE_INDEX_MAX = 0.3       # frozen threshold — reused, not redefined (dry side)
PORT_GLINT_MAX = 0.030          # frozen threshold — reused, not redefined


def port_buffer_land_for(bbox, out_shape, rgb, cache_dir=None, buffer_m=PORT_BUFFER_M,
                         blue_index_max=PORT_BLUE_INDEX_MAX, touch_dilate_px=2):
    """MASK1M addendum: close the "new/reclaimed port pad not yet in OSM"
    gap WITHOUT depending on CoastBA+'s sub-pixel corridor pipeline (which
    cannot discover land beyond its own corridor by design). Evidence-based,
    additive-only, CONNECTED-COMPONENT anchored (not a blanket VHR-land
    rule, and empirically far more complete than a flat distance buffer —
    see MASK1M_LOG.md: flat 30 m buffer closed 52% of the measured Khalifa
    gap, 60 m closed 73%, growing the buffer further hits diminishing
    returns and starts risking disconnected false positives; the connected-
    component version below closed 96%):

    1. Fetch existing OSM port-related polygons/lines (`PORT_TAGS`:
       landuse=port/harbour/industrial, man_made=quay/breakwater/pier/
       groyne/jetty — already spatially anchors the REAL port).
    2. Classify the VHR RGB spectrally DRY (blue-water index < 0.3, the SAME
       frozen threshold `vhr_water_mask` uses — reused, not redefined) AND
       not glinting/saturated (reused frozen 0.030) -> a candidate dry mask.
    3. Connected-component label the candidate dry mask; keep ONLY the
       components that TOUCH (within `touch_dilate_px` px of) the existing
       OSM port land — i.e. a new/reclaimed pad is recognised because it is
       PHYSICALLY CONTIGUOUS with already-mapped infrastructure (a quay
       "grows" a new berth), never because of a flat distance threshold.
       A disconnected dry patch (a boat, a sandbar, glare) is NEVER added,
       regardless of proximity — this is the key safety property a flat
       buffer does not have.

    Returns (land bool (H,W) or None, info dict) — None/all-False if no
    port tags found in `bbox` (never fabricates a port where OSM has no
    port evidence at all).
    """
    from scipy.ndimage import label as cc_label, binary_dilation, generate_binary_structure

    H, W = out_shape
    info = {"n_port_features": 0, "touch_dilate_px": touch_dilate_px, "confirmed_px": 0,
           "n_components_kept": 0, "source": "port-buffer-unavailable"}
    gdf, prov = fetch_osm_land(bbox, cache_dir=cache_dir or (ROOT / "cache" / "osm_port"),
                               tags=PORT_TAGS)
    if gdf is None or len(gdf) == 0:
        info["source"] = "no-port-tags-in-bbox"
        return None, info
    info["n_port_features"] = int(len(gdf))
    info["source"] = "port-buffer-connected-component"

    core_land = rasterize_osm_land(gdf, out_shape, bbox, include_lines=True)

    confirmed = np.zeros((H, W), dtype=bool)
    if rgb is not None:
        rgb_arr = np.asarray(rgb, dtype=np.float32)
        if rgb_arr.shape[:2] != (H, W):
            from PIL import Image as PILImage
            rgb_arr = np.asarray(PILImage.fromarray(rgb_arr.astype(np.uint8)).resize((W, H)),
                                 dtype=np.float32)
        r_, g_, b_ = rgb_arr[..., 0] / 255.0, rgb_arr[..., 1] / 255.0, rgb_arr[..., 2] / 255.0
        blue_water = (b_ - r_) / (b_ + r_ + 1e-6)
        near_sat = np.any(rgb_arr / 255.0 >= (1.0 - PORT_GLINT_MAX), axis=-1)
        fetched = rgb_arr.sum(axis=-1) > 0  # exclude sparse-corridor zero-fill
        dry = (blue_water < blue_index_max) & ~near_sat & fetched

        struct = generate_binary_structure(2, 2)
        lab, n_comp = cc_label(dry, structure=struct)
        if n_comp > 0:
            touch_zone = binary_dilation(core_land, structure=struct,
                                         iterations=max(1, touch_dilate_px))
            touching = set(np.unique(lab[touch_zone & (lab > 0)]).tolist())
            touching.discard(0)
            if touching:
                confirmed = np.isin(lab, list(touching))
            info["n_components_kept"] = len(touching)
            info["n_components_total"] = int(n_comp)

    info["confirmed_px"] = int(confirmed.sum())
    land = core_land | confirmed
    return land, info


# ── MASK-7: serve-time refined-coastline rasterization ─────────────────────

REFINED_COAST_CACHE_DIR = ROOT / "cache" / "refined_coast"
REFINED_COAST_COMMITTED_DIR = ROOT / "backend" / "data" / "coastline_refined"
# vhr_water_mask.py's cache-key version bumps coast2 -> coast3 to reflect this
# module's new additive inputs (port veto + refined coastline).


def _refined_gpkg_candidates():
    """Committed refined-coast GPKGs on disk — the small, accept=True-only
    copies (MASK-6 `write_committed_refined_gpkg`), deliberately NOT the
    large full per-site cache/ copy (which also carries every unmoved
    INHERITED-PRIOR vertex — by definition "prior unchanged", so unioning
    those adds zero new land beyond what `coastline_land_for` already
    contributes; reading the full ~17k-point file per request was also
    measured to be too slow for the <1 s serve-time rasterize budget on a
    dense real site). Listing is cheap and always current — no separate
    index to go stale."""
    if REFINED_COAST_COMMITTED_DIR.exists():
        return sorted(REFINED_COAST_COMMITTED_DIR.glob("refined_coast_*.gpkg"))
    return []


REFINED_POINT_BUFFER_M = 2.5   # matches MASK-1's 2 m transect spacing — no gaps along a run


def refined_coastline_land_for(bbox, out_shape):
    """MASK-7: rasterize the refined CoastBA+ coastline (MASK-6 committed
    GPKG, accept=True vertices only) to a boolean LAND mask over `bbox`,
    for use as an ADDITIONAL, purely-additive OSM_land input (never a
    replacement for `coastline_land_for`/`osm_land_for` — see
    `vhr_water_mask.vhr_water_grid`, which unions this in alongside them,
    never in place of them).

    Design note (safety-motivated, found during implementation): the
    committed GPKG carries only the SPARSE accept=True vertices (a real
    site typically accepts a small fraction of its transects — see
    MASK1M_LOG.md). Naively closing those sparse points into a filled
    Polygon(pts) (connecting far-apart accepted vertices directly, skipping
    every unaccepted one in between) produces a self-intersecting,
    geometrically WRONG shape that can claim a large false interior as
    land. Instead, each accepted vertex is rasterized as a small buffered
    disc (`REFINED_POINT_BUFFER_M`, matching the 2 m transect spacing so a
    contiguous accepted RUN paints an unbroken thin strip along the true
    refined line) — additive-only and geometrically safe by construction:
    it can only ever assert land within a couple of metres of an ACTUAL
    accepted measurement, never fabricate an interior.

    Returns (land bool (H,W) or None, info dict). None (graceful, honest
    fallback — NOT an error) when no refined product exists for this bbox
    yet (cold start / site never processed) — caller falls through to the
    unchanged R6/R7 coastline chain, per MASK-7's frozen-path guarantee.
    """
    import geopandas as gpd

    H, W = out_shape
    info = {"source": "refined-coastline-unavailable", "n_sites": 0, "n_points": 0,
           "coast_refine_version": None}
    candidates = _refined_gpkg_candidates()
    if not candidates:
        return None, info

    w, s, e, n = [float(x) for x in bbox]
    all_pts_utm = []   # (epsg, x, y) — buffer in each file's own UTM before reprojecting
    sites_seen = set()
    for gpkg in candidates:
        try:
            # NOTE: these GPKGs are written in local UTM (per-site EPSG), not
            # 4326 — read whole (files are small, <=2 MB per MASK-6's cap)
            # and reproject, rather than a `bbox=` filtered read in the
            # WRONG (caller's lon/lat) CRS, which silently returns empty.
            gdf = gpd.read_file(str(gpkg))
        except Exception:
            continue
        if gdf is None or len(gdf) == 0 or gdf.crs is None:
            continue
        gdf4326 = gdf.to_crs(4326)
        gdf4326 = gdf4326.cx[w:e, s:n]
        if len(gdf4326) == 0:
            continue
        sites_seen.update(gdf4326["site"].unique().tolist())
        epsg = gdf.crs.to_epsg()
        idx_in_bbox = gdf4326.index
        gdf_utm_subset = gdf.loc[idx_in_bbox]
        for geom in gdf_utm_subset.geometry:
            all_pts_utm.append((epsg, float(geom.x), float(geom.y)))

    if not all_pts_utm:
        return None, info

    from shapely.geometry import Point
    from shapely.ops import unary_union, transform as shp_transform
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    by_epsg = {}
    for epsg, x, y in all_pts_utm:
        by_epsg.setdefault(epsg, []).append((x, y))

    geoms_4326 = []
    for epsg, pts in by_epsg.items():
        to_wgs = Transformer.from_crs(epsg, 4326, always_xy=True)
        buffered_utm = [Point(x, y).buffer(REFINED_POINT_BUFFER_M) for x, y in pts]
        merged_utm = unary_union(buffered_utm)
        merged_4326 = shp_transform(lambda xx, yy: to_wgs.transform(xx, yy), merged_utm)
        geoms_4326.append(merged_4326)

    transform = from_bounds(w, s, e, n, W, H)
    land = rasterize([(gm, 1) for gm in geoms_4326 if gm is not None and not gm.is_empty],
                     out_shape=(H, W), transform=transform, fill=0, dtype="uint8").astype(bool)
    info.update({"source": "refined-coastline", "n_sites": len(sites_seen),
                "n_points": len(all_pts_utm), "coast_refine_version": COAST_REFINE_VERSION_TAG,
                "refined_land_frac": float(land.mean())})
    # MASK-10 honesty surface: attach the matching site's quality summary
    # (claim_withheld / possible_coastline_gap / cross_ref_independent /
    # confidence-class counts) from vintage.json, so a live response can
    # surface it without re-deriving anything.
    info["site_quality"] = _lookup_vintage_site_quality(sites_seen)
    return land, info


def _lookup_vintage_site_quality(site_names):
    """Best-effort read of MASK-6's vintage.json for the FIRST matching
    site's quality summary (claim_withheld, possible_coastline_gap,
    cross_ref_independent, confidence-class counts). Returns None if the
    vintage file is absent or no site matches — never raises."""
    vintage_path = REFINED_COAST_COMMITTED_DIR / "vintage.json"  # MASK-13 fix (was REFINED_COMMITTED_DIR, NameError)
    if not vintage_path.exists():
        return None
    try:
        v = json.load(open(vintage_path))
        sites = v.get("sites", {})
        for name in site_names:
            if name in sites:
                sm = sites[name]
                return {
                    "site": name,
                    "claim_withheld": sm.get("claim_withheld"),
                    "possible_coastline_gap": sm.get("possible_coastline_gap"),
                    "cross_ref_independent": sm.get("cross_ref_independent"),
                    "n_crisp_engineered": sm.get("n_crisp_engineered"),
                    "n_instantaneous_sloping": sm.get("n_instantaneous_sloping"),
                    "n_inherited_prior": sm.get("n_inherited_prior"),
                    "accept_frac": sm.get("accept_frac"),
                }
    except Exception:
        pass
    return None


try:
    from backend.coast_refine import COAST_REFINE_VERSION as COAST_REFINE_VERSION_TAG
except Exception:
    try:
        from coast_refine import COAST_REFINE_VERSION as COAST_REFINE_VERSION_TAG  # type: ignore
    except Exception:
        COAST_REFINE_VERSION_TAG = "mask1m-v1"


# ── MASK-GLOBAL: universal OSM land/ocean cut on every served depth product ──

_GLOBAL_CUT_CACHE = {}  # bbox_key -> (land_mask, out_shape, mask_source_info) — process-lifetime memo


def apply_global_land_cut(depth_grid, bbox, water_mask=None, rgb=None,
                          include_port_veto=True, include_refined=True, cache=True,
                          ndwi=None, ndwi_res_m=10.0):
    """User-mandated (2026-07-10) universal enforcement point: set depth to
    NoData (NaN) on every pixel the BEST AVAILABLE global land/ocean vector
    source calls LAND. Applied unconditionally — every path that returns or
    exports a depth product must route its grid through this before it is
    rendered/exported/stacked.

    Source priority (all additive, unioned — never a replacement for any one
    of them; each degrades gracefully to "unavailable" if its cache/fetch
    fails, and `mask_source` always discloses which contributed):
      1. `coastline_land_for` — OSMCoastline global land-polygons (PRIMARY;
         committed regional GPKG under backend/data/coastline, crisp,
         side-of-line-resolved, MHW-datum, coverage-independent).
      2. `osm_land_for` — OSM feature tags (buildings, structures, MASK-12-
         fixed: port/harbour/industrial AREA polygons and raw natural=
         coastline LINES excluded — they enclosed/duplicated real water).
      3. `port_buffer_land_for` (if `rgb` supplied) — connected-component
         spectrally-gated port-pad veto (MASK1M addendum).
      4. `refined_coastline_land_for` (if a committed CoastBA+ product
         exists for the ROI) — sub-pixel-precision additive coastline.
      5. `ndwi_mask.ndwi_land_mask` (MASK-NDWI, 2026-07-10, if `ndwi` supplied)
         — imagery-evidence land from the SAME Sentinel-2 scene the depth run
         used, native resolution. Vector sources (1-4) can only know about
         land OSM has mapped; reclaimed/dredged port land often is NOT in OSM
         yet — only the imagery can catch that gap. Safety-gated (conservative
         fixed NDWI threshold, buffer-to-known-land OR ship-sized-isolated-
         blob only — see `ndwi_mask.py` docstring for the measured-unsafe
         alternatives this rejected). NOT cached (imagery is per-call/per-scene).

    Returns (cut_grid, land_mask bool (H,W), mask_source dict).
    """
    depth_grid = np.asarray(depth_grid, dtype=np.float32)
    H, W = depth_grid.shape
    key = (round(float(bbox[0]), 6), round(float(bbox[1]), 6), round(float(bbox[2]), 6),
          round(float(bbox[3]), 6), H, W, bool(include_port_veto and rgb is not None),
          bool(include_refined))
    if cache and key in _GLOBAL_CUT_CACHE:
        land, info = _GLOBAL_CUT_CACHE[key]
    else:
        coast_land, cinfo = coastline_land_for(bbox, (H, W))
        feat_land, finfo = osm_land_for(bbox, (H, W))
        land = coast_land if coast_land is not None else np.zeros((H, W), dtype=bool)
        if feat_land is not None:
            land = land | feat_land
        sources = []
        if coast_land is not None:
            sources.append(f"osm-coastline({cinfo.get('n_polys', 0)} polys, vintage {cinfo.get('vintage')})")
        else:
            sources.append("osm-coastline-unavailable")
        if feat_land is not None:
            sources.append(f"osm-features({finfo.get('n_polys', 0)} polys)")
        port_info = None
        if include_port_veto and rgb is not None:
            try:
                port_land, port_info = port_buffer_land_for(bbox, (H, W), rgb)
                if port_land is not None:
                    land = land | port_land
                    sources.append(f"port-buffer({port_info.get('confirmed_px', 0)} px)")
            except Exception as ex:
                sources.append(f"port-buffer-error: {ex}")
        refined_info = None
        if include_refined:
            try:
                refined_land, refined_info = refined_coastline_land_for(bbox, (H, W))
                if refined_land is not None:
                    land = land | refined_land
                    sources.append(f"refined-coastline({refined_info.get('n_points', 0)} pts)")
            except Exception as ex:
                sources.append(f"refined-coastline-error: {ex}")
        info = {
            "mask_source": " + ".join(sources) if sources else "no-land-source-available",
            "coastline_source": cinfo.get("source"), "coastline_vintage": cinfo.get("vintage"),
            "coastline_datum": cinfo.get("datum") or "MHW (approx; not LAT)",
            "n_land_px": int(land.sum()), "land_frac": float(land.mean()),
            "port_veto_applied": bool(port_info and port_info.get("source") not in (None, "no-port-tags-in-bbox")),
            "refined_applied": bool(refined_info and refined_info.get("source") == "refined-coastline"),
        }
        if cache:
            _GLOBAL_CUT_CACHE[key] = (land, info)

    # MASK-NDWI (2026-07-10, user priority override #2): imagery-evidence
    # additive land layer, NOT cached (varies per-call with the actual scene
    # fetched). Deliberately OUTSIDE the vector cache block above.
    if ndwi is not None:
        try:
            from backend.ndwi_mask import ndwi_land_mask as _ndwi_land_mask
        except ImportError:
            from ndwi_mask import ndwi_land_mask as _ndwi_land_mask  # type: ignore
        try:
            extra_land, ndwi_stats = _ndwi_land_mask(ndwi, land, res_m=ndwi_res_m)
            land = land | extra_land
            info = dict(info)
            info["mask_source"] = info.get("mask_source", "") + f" + ndwi_10m(+{ndwi_stats['extra_land_px']} px, thr={ndwi_stats['threshold_used']}, otsu_diag={ndwi_stats['otsu_threshold_diagnostic']})"
            info["ndwi_stats"] = ndwi_stats
            info["n_land_px"] = int(land.sum())
            info["land_frac"] = float(land.mean())
        except Exception as ex:
            info = dict(info)
            info["mask_source"] = info.get("mask_source", "") + f" + ndwi_10m-error({ex})"

    if land.shape != depth_grid.shape:
        # Grid-size mismatch (rare, e.g. an odd-shaped per-scene crop) — resize the
        # land mask via nearest-neighbour rather than skip the cut.
        from PIL import Image as _PILImage
        land_r = np.asarray(_PILImage.fromarray(land.astype(np.uint8) * 255).resize(
            (W, H), _PILImage.NEAREST)) > 127
    else:
        land_r = land

    cut = depth_grid.copy()
    cut[land_r] = np.nan
    n_cut = int((land_r & np.isfinite(depth_grid)).sum())
    out_info = dict(info)
    out_info["n_depth_px_cut_this_call"] = n_cut
    return cut, land_r, out_info


if __name__ == "__main__":
    # One-time OSMCoastline cache prep (unzip + .qix + regional GPKGs). See
    # prepare_coastline_cache() docstring for the download command.
    import sys as _sys
    force = "--force" in _sys.argv
    out = prepare_coastline_cache(force=force)
    print(json.dumps(out, indent=2))
