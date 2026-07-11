"""
Land mask for the Strait of Hormuz region (24.5-27.5N, 54-58E).

Uses simplified coastline polygons to distinguish land from sea.
Points inside any land polygon return True (is_land).
"""

import numpy as np

# Simplified coastline polygons for the Hormuz region.
# Each polygon is a list of (lon, lat) vertices defining a land area.
# Traced from OpenStreetMap / natural earth coastline data.

# Iranian coastline (northern shore of Persian Gulf + Strait of Hormuz)
IRAN_COAST = [
    (54.0, 27.5),   # NW corner (border area)
    (54.0, 26.65),
    (54.3, 26.55),
    (54.6, 26.50),
    (54.8, 26.42),
    (55.0, 26.35),
    (55.2, 26.30),
    (55.4, 26.25),
    (55.6, 26.22),
    (55.8, 26.20),
    (55.95, 26.20),
    (56.05, 26.25),
    (56.10, 26.30),
    (56.15, 26.35),  # Qeshm island approach
    (56.20, 26.50),
    (56.25, 26.55),
    (56.30, 26.60),
    (56.35, 26.65),
    (56.40, 26.70),
    (56.50, 26.80),
    (56.60, 26.85),
    (56.80, 26.90),
    (57.00, 27.00),
    (57.10, 27.05),
    (57.20, 27.10),
    (57.30, 27.15),
    (57.40, 27.20),
    (57.60, 27.20),
    (57.80, 27.15),
    (58.0, 27.10),
    (58.0, 27.5),    # NE corner
]

# Qeshm Island
QESHM = [
    (55.80, 26.55),
    (55.90, 26.50),
    (56.00, 26.52),
    (56.10, 26.55),
    (56.20, 26.60),
    (56.27, 26.70),
    (56.30, 26.78),
    (56.28, 26.85),
    (56.22, 26.88),
    (56.15, 26.88),
    (56.05, 26.85),
    (55.95, 26.80),
    (55.85, 26.75),
    (55.78, 26.68),
    (55.76, 26.62),
    (55.78, 26.57),
]

# Hormuz Island (small)
HORMUZ_ISLAND = [
    (56.44, 27.04),
    (56.48, 27.02),
    (56.50, 27.05),
    (56.48, 27.08),
    (56.44, 27.08),
    (56.42, 27.06),
]

# Larak Island
LARAK = [
    (56.33, 26.85),
    (56.38, 26.83),
    (56.40, 26.86),
    (56.38, 26.89),
    (56.33, 26.88),
]

# Musandam Peninsula (Oman, juts into the strait)
MUSANDAM = [
    (56.00, 26.15),
    (56.05, 26.10),
    (56.10, 26.05),
    (56.15, 26.00),
    (56.20, 25.95),
    (56.25, 26.00),
    (56.30, 26.05),
    (56.35, 26.10),
    (56.38, 26.17),
    (56.40, 26.25),
    (56.42, 26.30),
    (56.38, 26.35),
    (56.30, 26.38),
    (56.25, 26.35),
    (56.20, 26.30),
    (56.15, 26.25),
    (56.10, 26.20),
    (56.05, 26.18),
]

# UAE coastline (south side of Persian Gulf)
UAE_COAST = [
    (54.0, 24.5),    # SW corner
    (54.0, 24.85),
    (54.3, 24.65),
    (54.5, 24.55),
    (54.7, 24.48),
    (55.0, 24.50),
    (55.2, 24.60),
    (55.3, 24.80),
    (55.4, 25.00),
    (55.5, 25.15),
    (55.6, 25.25),
    (55.7, 25.35),
    (55.8, 25.40),
    (55.9, 25.50),
    (55.95, 25.60),
    (56.0, 25.70),
    (56.0, 25.85),
    (56.0, 25.95),   # meets Musandam
    (56.00, 26.15),  # Musandam base
    # Back along Musandam east side down to Oman coast
    (56.42, 26.30),
    (56.45, 26.20),
    (56.48, 26.10),
    (56.50, 26.00),
    (56.50, 25.80),
    (56.55, 25.60),
    (56.60, 25.40),
    (56.80, 25.20),
    (57.00, 25.05),
    (57.20, 24.90),
    (57.40, 24.70),
    (57.60, 24.60),
    (57.80, 24.55),
    (58.0, 24.5),    # SE corner
]

LAND_POLYGONS = [IRAN_COAST, QESHM, HORMUZ_ISLAND, LARAK, MUSANDAM, UAE_COAST]


def _point_in_polygon(px, py, polygon):
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def is_land(lon, lat):
    """Check if a single point is on land."""
    for poly in LAND_POLYGONS:
        if _point_in_polygon(lon, lat, poly):
            return True
    return False


def make_land_mask(lons, lats):
    """
    Create a 2D boolean land mask for a grid.
    Returns: numpy array of shape (len(lats), len(lons)), True = land.
    """
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask = np.zeros(lon_grid.shape, dtype=bool)

    for i in range(len(lats)):
        for j in range(len(lons)):
            mask[i, j] = is_land(lons[j], lats[i])

    return mask


def get_land_mask_json(bbox, resolution=0.08):
    """Return land mask as JSON-serializable dict."""
    lats = np.arange(bbox["lat_min"], bbox["lat_max"], resolution)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"], resolution)
    mask = make_land_mask(lons, lats)

    return {
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "mask": mask.tolist(),  # True = land
        "ny": len(lats),
        "nx": len(lons),
    }
