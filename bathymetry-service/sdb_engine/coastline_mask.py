"""Vector coastline LAND cut from committed OSMCoastline regional GPKGs.

Vendored from Bathymetry_Production/backend/osm_land_mask.py
(coastline_land_for / _coastline_source_path / rasterize polygon branch),
rewritten on fiona (bbox-filtered read) + rasterio.features.rasterize — no
geopandas/osmnx/Overpass and no network. Only the committed UAE + Morocco
regional subsets ship; outside those regions the caller's NDWI-only
behavior is unchanged.

Datum honesty: OSM natural=coastline approximates Mean High Water — a
few-metre-accurate HIGH-water line, NOT LAT. It is authoritative for LAND
only: unioned into the existing NDWI land cut, never adding water.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

L = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "coastline"
VINTAGE_JSON = DATA_DIR / "vintage.json"

# (w, s, e, n) EPSG:4326 boxes covered by each committed GPKG (actual data
# extents; a tile missing land only skips the cut — NDWI stays the base mask).
COASTLINE_REGIONS = {
    "uae":     (50.0, 22.0, 58.0, 27.0),
    "morocco": (-17.35, 20.5, -7.0, 33.9),
}
COASTLINE_GPKG = {
    "uae":     DATA_DIR / "coastline_land_uae.gpkg",
    "morocco": DATA_DIR / "coastline_land_morocco.gpkg",
}

DATUM_NOTE = "MHW (approx; not LAT)"


def _bbox_in_region(bbox) -> Optional[str]:
    """Region key whose box fully contains `bbox`, else None."""
    w, s, e, n = [float(x) for x in bbox]
    for key, (rw, rs, re_, rn) in COASTLINE_REGIONS.items():
        if w >= rw and s >= rs and e <= re_ and n <= rn:
            return key
    return None


def _coastline_source_path(bbox) -> Tuple[Optional[Path], Optional[str]]:
    """(gpkg_path, region_key) for `bbox`, or (None, None) outside coverage."""
    region = _bbox_in_region(bbox)
    if region is not None:
        gpkg = COASTLINE_GPKG.get(region)
        if gpkg is not None and gpkg.exists():
            return gpkg, region
    return None, None


def coastline_vintage() -> Optional[str]:
    if VINTAGE_JSON.exists():
        try:
            return json.load(open(VINTAGE_JSON)).get("vintage")
        except Exception:
            pass
    return None


def coastline_land_for(bbox, out_shape) -> Tuple[Optional[np.ndarray], Dict]:
    """Rasterize the committed OSMCoastline land polygons over
    `bbox` = [w, s, e, n] (EPSG:4326) to a boolean (H, W) LAND mask.

    Returns (land bool (H, W) or None, info). land is None when the bbox is
    outside the committed regions (caller keeps NDWI-only behavior). info
    always carries source, vintage, n_polys, datum, region, osm_land_frac."""
    H, W = out_shape
    info = {"source": "osm-coastline-unavailable", "vintage": coastline_vintage(),
            "n_polys": 0, "datum": DATUM_NOTE, "region": None,
            "osm_land_frac": None}
    path, region = _coastline_source_path(bbox)
    if path is None:
        return None, info
    info["region"] = region
    try:
        import fiona
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        w, s, e, n = [float(x) for x in bbox]
        geoms = []
        with fiona.open(str(path)) as src:
            for feat in src.filter(bbox=(w, s, e, n)):
                geom = feat["geometry"]
                if geom is None:
                    continue
                gtype = geom["type"]
                if gtype not in ("Polygon", "MultiPolygon"):
                    continue
                geoms.append({"type": gtype, "coordinates": geom["coordinates"]})

        info["source"] = "osm-coastline-landpoly"
        info["n_polys"] = len(geoms)
        if not geoms:
            info["osm_land_frac"] = 0.0
            return np.zeros((H, W), dtype=bool), info

        transform = from_bounds(w, s, e, n, W, H)
        land = rasterize([(g, 1) for g in geoms], out_shape=(H, W),
                         transform=transform, fill=0, dtype="uint8").astype(bool)
        info["osm_land_frac"] = float(land.mean())
        return land, info
    except Exception as ex:
        L.warning("coastline read/rasterize failed: %s", ex)
        info["source"] = f"osm-coastline-error: {ex}"
        return None, info
