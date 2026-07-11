"""
Global land mask from Natural Earth 10m shapefiles.
Uses prepared geometry for fast vectorized point-in-polygon.
"""
import os
import numpy as np

_LAND_GEOM = None


def _load():
    global _LAND_GEOM
    if _LAND_GEOM is not None:
        return _LAND_GEOM
    import geopandas as gpd
    from shapely.prepared import prep
    shp = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocean",
                       "ne_10m_land", "ne_10m_land.shp")
    if not os.path.exists(shp):
        raise FileNotFoundError(f"Natural Earth land shapefile not found: {shp}")
    gdf = gpd.read_file(shp)
    union = gdf.union_all()
    _LAND_GEOM = prep(union)
    print("[LANDMASK] Global Natural Earth 10m land loaded")
    return _LAND_GEOM


def is_land_global(lon, lat):
    """Check if a single point is on land (global)."""
    from shapely.geometry import Point
    return _load().contains(Point(lon, lat))


def make_ocean_mask(lons_1d, lats_1d):
    """
    Build a 2D boolean ocean mask for a lat/lon grid.
    Returns: (ny, nx) array, True = ocean, False = land.
    Uses vectorized multi-point query for speed.
    """
    from shapely.geometry import MultiPoint
    geom = _load()
    ny, nx = len(lats_1d), len(lons_1d)
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    flat_lons = lon_grid.ravel()
    flat_lats = lat_grid.ravel()

    # Batch query: build all points, test against prepared geometry
    # For very large grids, subsample then interpolate
    total = ny * nx
    if total > 250000:
        # Subsample to ~500x500, then nearest-neighbor upscale
        step = max(1, int(np.sqrt(total / 250000)))
        sub_lats = lats_1d[::step]
        sub_lons = lons_1d[::step]
        sub_mask = _query_grid(geom, sub_lons, sub_lats)
        # Upscale via nearest neighbor
        from scipy.ndimage import zoom
        zy = ny / len(sub_lats)
        zx = nx / len(sub_lons)
        ocean = zoom(sub_mask.astype(float), (zy, zx), order=0) > 0.5
    else:
        ocean = _query_grid(geom, lons_1d, lats_1d)

    return ocean


def _query_grid(prepared_geom, lons_1d, lats_1d):
    """Query each grid point against prepared land geometry."""
    from shapely.geometry import Point
    ny, nx = len(lats_1d), len(lons_1d)
    ocean = np.ones((ny, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            if prepared_geom.contains(Point(lons_1d[j], lats_1d[i])):
                ocean[i, j] = False
    return ocean
