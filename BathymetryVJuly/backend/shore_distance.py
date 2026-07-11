"""
Shore-distance rasters via the GDAL CLI — adapted from the original
Bathy.py pipeline (create_distance) used on the Moroccan ROIs.

Reference script (verbatim, simplified):
    gdal.Open(B02.tiff)   -> geotransform, CRS
    gdal.Translate(..., projWin=[…])   # clip global mask to S2 tile extent
    gdal.Warp(..., dstSRS=EPSG:<s2>)   # reproject to S2 CRS
    gdal_calc.py   --calc="A*(A==2)"   # isolate water class
    gdal_proximity.py -values 0        # distance-to-shore

We replicate the same chain here with the system GDAL CLI
(`gdal_translate`, `gdalwarp`, `gdal_calc.py`, `gdal_proximity.py`) via
subprocess — no Python `osgeo` binding required. The "Mask_wgs84" raster
is built from OSM coastline + breakwater polygons the first time an ROI
is seen and cached on disk, so port structures (breakwaters, reclaimed
land, quays) are captured instead of the 1 km coarse coastline.

Public entry point:
    create_distance_raster(b02: np.ndarray, bbox, out_dir, tag,
                           pixel_size_m=10.0) -> (water_mask, dist_m)
"""
from __future__ import annotations
import os, json, hashlib, logging, subprocess, shutil
from pathlib import Path
import numpy as np

L = logging.getLogger("bathy.shore_dist")

# Persistent cache for Mask_wgs84-equivalent rasters (one per ROI hash).
# OSM polygons rarely change day-to-day; cache-invalidate by deleting the
# directory if you pull new coastline data.
CACHE_DIR = Path(os.environ.get(
    "BATHY_SHORE_CACHE",
    os.path.expanduser("~/Bathymetry_VMarch/cache/shore_mask")
))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# 1. Write B02 as a proper georeferenced GeoTIFF (the "S2_name_B02.tiff"
#    input that Bathy.py expects). We use rasterio because it wraps the
#    same GDAL C library, just without the `osgeo` python binding.
# ═══════════════════════════════════════════════════════════

def _write_b02_geotiff(b02: np.ndarray, bbox, path: Path) -> Path:
    """Save the B02 array as an EPSG:4326 GeoTIFF covering `bbox`."""
    import rasterio
    from rasterio.transform import from_bounds
    H, W = b02.shape
    w, s, e, n = bbox
    transform = from_bounds(w, s, e, n, W, H)
    with rasterio.open(
        str(path), "w", driver="GTiff",
        height=H, width=W, count=1, dtype=b02.dtype,
        crs="EPSG:4326", transform=transform,
        compress="LZW",
    ) as dst:
        dst.write(b02, 1)
    return path


# ═══════════════════════════════════════════════════════════
# 2. Build the Mask_wgs84.tiff equivalent from OSM.
#    Classes used (same convention as Bathy.py — water = 2, land = 1):
#       pixel = 2  →  water (natural=water, natural=bay, ocean interior)
#       pixel = 1  →  land  (everything the coastline encloses, plus
#                            man_made=breakwater / pier polygons)
# ═══════════════════════════════════════════════════════════

def _bbox_key(bbox) -> str:
    return hashlib.md5(json.dumps([round(x, 5) for x in bbox]).encode()).hexdigest()[:12]


def _fetch_osm_polygons(bbox, buffer_deg: float = 0.05) -> list:
    """Return OSM polygons (lat/lon WGS84) classified as 'water' or 'land'
    covering the bbox + small buffer. Uses Overpass API.
    Output: [{ 'class': 'land'|'water', 'coords': [(lon,lat), …] }, …]
    """
    import requests
    w, s, e, n = bbox
    w -= buffer_deg; e += buffer_deg; s -= buffer_deg; n += buffer_deg
    # Overpass QL query — coastline way + man_made breakwater/pier + natural=water
    q = f"""
[out:json][timeout:60];
(
  way["natural"="coastline"]({s},{w},{n},{e});
  way["man_made"="breakwater"]({s},{w},{n},{e});
  way["man_made"="pier"]({s},{w},{n},{e});
  relation["natural"="water"]({s},{w},{n},{e});
  way["natural"="water"]({s},{w},{n},{e});
  relation["landuse"="reservoir"]({s},{w},{n},{e});
  relation["place"="island"]({s},{w},{n},{e});
  way["place"="island"]({s},{w},{n},{e});
  way["place"="islet"]({s},{w},{n},{e});
);
(._;>;);
out body;
"""
    urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.osm.ch/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]
    headers = {
        "User-Agent": "bathymetry-vmarch/1.0 (shore-distance)",
        "Accept": "application/json",
    }
    data = None
    last_err = ""
    for url in urls:
        try:
            # Overpass servers expect the query as URL-encoded "data=<QL>"
            # body, not as JSON or multipart. requests uses form encoding
            # when data is a dict, which matches.
            r = requests.post(url, data={"data": q}, headers=headers, timeout=240)
            if r.ok:
                try:
                    parsed = r.json()
                except Exception as jx:
                    last_err = f"{url}: bad JSON ({jx})"
                    L.warning(last_err); continue
                n_el = len(parsed.get("elements", []))
                L.info(f"Overpass: got {n_el} elements from {url}")
                if n_el == 0:
                    last_err = f"{url}: 0 elements (try next mirror)"
                    L.warning(last_err); continue
                data = parsed
                break
            else:
                last_err = f"{url}: HTTP {r.status_code}"
                L.warning(last_err)
        except Exception as ex:
            last_err = f"{url}: {ex}"
            L.warning(last_err)
    if data is None:
        raise RuntimeError(f"Overpass API unreachable / empty on all mirrors — {last_err}")
    # Build node lookup
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"] == "node"}
    polys = []
    for el in data["elements"]:
        if el["type"] != "way":
            continue
        tags = el.get("tags", {})
        if "coastline" in tags.get("natural", ""):
            klass = "coastline"
        elif tags.get("man_made") in ("breakwater", "pier"):
            klass = "land"
        elif tags.get("natural") == "water" or tags.get("landuse") == "reservoir":
            klass = "water"
        elif tags.get("place") in ("island", "islet"):
            klass = "land"
        else:
            continue
        coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
        if len(coords) < 2:
            continue
        polys.append({"class": klass, "coords": coords})
    L.info(f"OSM: {len(polys)} polygons for bbox")
    return polys


def _build_mask_wgs84(bbox, px_per_deg: int = 20000) -> Path:
    """Rasterise OSM polygons into a uint8 GeoTIFF covering `bbox`.
    Pixel convention: 2 = water, 1 = land (any other = no-data).
    Cached in CACHE_DIR under the bbox hash.
    """
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize
    import rasterio
    from shapely.geometry import Polygon, LineString
    from shapely.ops import polygonize, unary_union

    key = _bbox_key(bbox)
    out = CACHE_DIR / f"mask_wgs84_{key}.tiff"
    if out.exists():
        L.info(f"Mask_wgs84 cache hit: {out.name}")
        return out

    w, s, e, n = bbox
    polys = _fetch_osm_polygons(bbox)
    land_shapes = []
    coast_lines = []
    water_shapes = []
    for p in polys:
        coords = p["coords"]
        if p["class"] == "land" and len(coords) >= 3:
            try:
                g = Polygon(coords)
                if g.is_valid:
                    land_shapes.append(g)
            except Exception:
                pass
        elif p["class"] == "water" and len(coords) >= 3:
            try:
                g = Polygon(coords)
                if g.is_valid:
                    water_shapes.append(g)
            except Exception:
                pass
        elif p["class"] == "coastline":
            coast_lines.append(LineString(coords))

    # Coastline is a set of open lines that, taken together, enclose land.
    # polygonize() turns them into polygons; whichever polygon contains the
    # centroid of the bbox is probably sea, the complement is land.
    if coast_lines:
        try:
            merged = unary_union(coast_lines)
            for g in polygonize(merged):
                if g.is_valid:
                    land_shapes.append(g)
        except Exception as ex:
            L.warning(f"polygonize coastline: {ex}")

    # Raster grid
    lon_span = e - w; lat_span = n - s
    W = max(128, int(lon_span * px_per_deg))
    H = max(128, int(lat_span * px_per_deg))
    # Cap at something sane — 20k/deg * 0.15 deg ~ 3000 px per side
    W = min(W, 6000); H = min(H, 6000)
    transform = from_bounds(w, s, e, n, W, H)

    # Paint: start everything water (2), then burn land (1) on top
    mask = np.full((H, W), 2, dtype=np.uint8)
    if land_shapes:
        burned = rasterize(
            [(g, 1) for g in land_shapes if not g.is_empty],
            out_shape=(H, W), transform=transform, fill=0, dtype="uint8",
        )
        mask = np.where(burned == 1, 1, mask)
    # Water polygons over the top (rare but possible — inland lakes inside land)
    if water_shapes:
        wburn = rasterize(
            [(g, 2) for g in water_shapes if not g.is_empty],
            out_shape=(H, W), transform=transform, fill=0, dtype="uint8",
        )
        mask = np.where(wburn == 2, 2, mask)

    with rasterio.open(
        str(out), "w", driver="GTiff",
        height=H, width=W, count=1, dtype="uint8",
        crs="EPSG:4326", transform=transform,
        compress="LZW", nodata=0,
    ) as dst:
        dst.write(mask, 1)
    L.info(f"Mask_wgs84 built {W}x{H} @ {out}")
    return out


# ═══════════════════════════════════════════════════════════
# 3. The full Bathy.py create_distance pipeline (adapted).
# ═══════════════════════════════════════════════════════════

def _run(cmd: list, check: bool = True):
    L.info("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = f"{cmd[0]} failed ({r.returncode}): {r.stderr[-400:]}"
        if check:
            raise RuntimeError(msg)
        L.warning(msg)
    return r


def create_distance_raster(
    b02: np.ndarray,
    bbox,
    out_dir,
    tag: str,
    pixel_size_m: float = 10.0,
):
    """GDAL-based shore-distance pipeline (adapted from Bathy.py).

    Parameters
    ----------
    b02 : numpy array of the S2 blue band (used only to fix the output
          grid size — the pipeline operates on pixel geometry)
    bbox : [w, s, e, n] in EPSG:4326
    out_dir : working directory (per-scene scratch)
    tag   : prefix for output filenames (e.g. "scene1_2024-01")
    pixel_size_m : target pixel size in metres (S2 native is 10 m)

    Returns
    -------
    (water_mask: bool HxW, dist_m: float32 HxW)  both on the B02 grid.
    """
    for tool in ("gdal_translate", "gdalwarp", "gdal_calc.py", "gdal_proximity.py"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not in PATH — install GDAL CLI")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    b02_tif    = out_dir / f"{tag}_B02.tiff"
    mask_wgs   = _build_mask_wgs84(bbox)              # persistent cache
    clipped    = out_dir / f"{tag}_mask_clip.tiff"    # gdal_translate output
    warped     = out_dir / f"{tag}_mask_warp.tiff"    # gdalwarp output (S2 grid)
    water_only = out_dir / f"{tag}_Inter.tiff"        # gdal_calc output
    dist_tif   = out_dir / f"{tag}_Distance.tiff"     # gdal_proximity output

    _write_b02_geotiff(b02, bbox, b02_tif)
    H, W = b02.shape
    w, s, e, n = bbox

    # Step 1 — clip the global mask to S2 tile extent (gdal_translate -projwin)
    _run([
        "gdal_translate",
        "-projwin", str(w), str(n), str(e), str(s),
        "-projwin_srs", "EPSG:4326",
        str(mask_wgs), str(clipped),
    ])

    # Step 2 — reproject + resample to exactly the B02 grid (gdalwarp)
    _run([
        "gdalwarp",
        "-overwrite",
        "-t_srs", "EPSG:4326",
        "-te", str(w), str(s), str(e), str(n),
        "-ts", str(W), str(H),
        "-r", "near",
        str(clipped), str(warped),
    ])

    # Step 3 — isolate water class (gdal_calc.py --calc="A*(A==2)")
    _run([
        "gdal_calc.py",
        "-A", str(warped),
        "--outfile", str(water_only),
        "--calc", "A*(A==2)",
        "--NoDataValue", "0",
        "--overwrite",
    ])

    # Step 4 — distance-to-shore (gdal_proximity.py -values 0 = distance
    # from each pixel to the nearest land = 0-valued pixel).
    # Units = pixels in the source SRS — we multiply by pixel_size_m below.
    _run([
        "gdal_proximity.py",
        str(water_only), str(dist_tif),
        "-values", "0",
        "-distunits", "PIXEL",
        "-ot", "Float32",
    ])

    # Read results back
    import rasterio
    with rasterio.open(str(warped)) as ds:
        mask_arr = ds.read(1)
    water_mask = (mask_arr == 2)
    with rasterio.open(str(dist_tif)) as ds:
        dist_px = ds.read(1).astype(np.float32)
    dist_m = dist_px * float(pixel_size_m)

    L.info(f"GDAL shore-dist {tag}: water={int(water_mask.sum())}/{H*W} "
           f"· max_dist={float(dist_m.max()):.0f} m "
           f"· files in {out_dir}")
    return water_mask, dist_m
