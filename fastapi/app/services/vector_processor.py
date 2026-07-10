"""Vector file processing using Fiona and Shapely."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import fiona
from fiona.transform import transform_geom
from shapely.geometry import mapping, shape
from shapely.validation import make_valid


# Magic bytes / extension mapping
_FORMAT_SIGNATURES: dict[bytes, str] = {
    b'{"type"': "GeoJSON",
    b"PK": "ESRI Shapefile",  # ZIP containing shp
    b"\x00\x00\x27\x0a": "ESRI Shapefile",
    b"<?xml": "KML",          # KML or KMZ
    b"SQLite": "GPKG",
    b"GPSBabel": "GPX",
}

_EXT_DRIVERS: dict[str, str] = {
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".kml": "KML",
    ".kmz": "KML",
    ".shp": "ESRI Shapefile",
    ".zip": "ESRI Shapefile",
    ".gpkg": "GPKG",
    ".gpx": "GPX",
    ".gdb": "OpenFileGDB",
}


def detect_format(file_path: str, filename: str) -> str:
    """Detect vector file format from magic bytes and extension.

    Returns Fiona driver name.
    """
    ext = Path(filename).suffix.lower()

    # Try magic bytes first
    with open(file_path, "rb") as f:
        header = f.read(16)

    for sig, driver in _FORMAT_SIGNATURES.items():
        if header.startswith(sig):
            # Distinguish KML vs KMZ
            if driver == "KML" and ext == ".kmz":
                return "KML"  # Fiona handles KMZ via KML driver after extract
            return driver

    # Check for GeoJSON by looking for JSON structure
    if header.lstrip().startswith(b"{"):
        return "GeoJSON"

    # Check for zipped shapefile
    if ext == ".zip" or header[:2] == b"PK":
        return "ESRI Shapefile"

    # Fallback to extension
    driver = _EXT_DRIVERS.get(ext)
    if driver:
        return driver

    raise ValueError(f"Unsupported vector format: {filename}")


def process_vector_file(
    file_path: str,
    filename: str,
) -> dict[str, Any]:
    """Process a vector file: read, reproject to 4326, validate geometries.

    Returns dict with:
        - geometry_types: list of geometry type strings
        - properties_schema: dict of property name -> type string
        - feature_count: int
        - bbox: [minx, miny, maxx, maxy]
        - features: list of (wkt, geometry_type, properties) tuples
    """
    driver = detect_format(file_path, filename)

    # Handle zipped shapefiles
    open_path = file_path
    if driver == "ESRI Shapefile" and zipfile.is_zipfile(file_path):
        open_path = f"zip://{file_path}"

    with fiona.open(open_path, driver=driver) as src:
        src_crs = src.crs
        need_reproject = src_crs and str(src_crs).upper() not in (
            "EPSG:4326", "{'INIT': 'EPSG:4326'}", "WGS84",
        )
        # Also check for epsg code
        if src_crs:
            try:
                epsg = src_crs.get("init", "").upper()
                if epsg == "EPSG:4326":
                    need_reproject = False
            except (AttributeError, TypeError):
                pass
            try:
                if hasattr(src_crs, "to_epsg") and src_crs.to_epsg() == 4326:
                    need_reproject = False
            except Exception:
                pass

        geometry_types: set[str] = set()
        properties_schema: dict[str, str] = {}
        features: list[tuple[str, str, dict]] = []

        # Discover property schema from Fiona schema
        if src.schema and "properties" in src.schema:
            for prop_name, prop_type in src.schema["properties"].items():
                properties_schema[prop_name] = str(prop_type)

        for feat in src:
            geom = feat.get("geometry")
            props = dict(feat.get("properties", {}))

            if geom is None:
                continue

            # Reproject to 4326 if needed
            if need_reproject and src_crs:
                geom = transform_geom(src_crs, "EPSG:4326", geom)

            # Validate and fix geometry with shapely
            shp = shape(geom)
            if not shp.is_valid:
                shp = make_valid(shp)

            geom_type = shp.geom_type
            geometry_types.add(geom_type)
            wkt = shp.wkt

            features.append((wkt, geom_type, props))

    # Calculate bbox from all features
    if features:
        from shapely import wkt as shapely_wkt
        all_bounds = []
        for wkt_str, _, _ in features:
            g = shapely_wkt.loads(wkt_str)
            all_bounds.append(g.bounds)

        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        bbox = [minx, miny, maxx, maxy]
    else:
        bbox = None

    return {
        "geometry_types": sorted(geometry_types),
        "properties_schema": properties_schema,
        "feature_count": len(features),
        "bbox": bbox,
        "features": features,
    }


def export_features(
    features: list[dict[str, Any]],
    fmt: str,
    output_path: str,
) -> str:
    """Export features to GeoJSON, KML, or Shapefile.

    Args:
        features: list of dicts with 'geometry' (GeoJSON dict) and 'properties'.
        fmt: 'geojson', 'kml', or 'shapefile'.
        output_path: destination file path.

    Returns:
        The output file path.
    """
    if fmt == "geojson":
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": f["geometry"],
                    "properties": f.get("properties", {}),
                }
                for f in features
            ],
        }
        with open(output_path, "w") as fh:
            json.dump(fc, fh)
        return output_path

    # Determine schema from first feature
    if not features:
        raise ValueError("No features to export")

    sample_props = features[0].get("properties", {})
    schema_props = {}
    for k, v in sample_props.items():
        if isinstance(v, int):
            schema_props[k] = "int"
        elif isinstance(v, float):
            schema_props[k] = "float"
        else:
            schema_props[k] = "str"

    sample_geom_type = features[0]["geometry"]["type"]

    driver_map = {
        "kml": "KML",
        "shapefile": "ESRI Shapefile",
    }
    driver = driver_map.get(fmt)
    if not driver:
        raise ValueError(f"Unsupported export format: {fmt}")

    schema = {
        "geometry": sample_geom_type,
        "properties": schema_props,
    }

    actual_path = output_path
    if fmt == "shapefile":
        # Write to a temp dir then zip
        tmpdir = tempfile.mkdtemp()
        shp_path = os.path.join(tmpdir, "export.shp")
        with fiona.open(shp_path, "w", driver=driver, schema=schema, crs="EPSG:4326") as dst:
            for f in features:
                dst.write({
                    "geometry": f["geometry"],
                    "properties": {k: f.get("properties", {}).get(k) for k in schema_props},
                })
        # Zip up all shapefile components
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for child in Path(tmpdir).iterdir():
                zf.write(child, child.name)
        return output_path

    with fiona.open(actual_path, "w", driver=driver, schema=schema, crs="EPSG:4326") as dst:
        for f in features:
            dst.write({
                "geometry": f["geometry"],
                "properties": {k: f.get("properties", {}).get(k) for k in schema_props},
            })

    return output_path
