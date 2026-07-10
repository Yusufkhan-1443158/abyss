"""Vendor-agnostic raster metadata extraction.

Real-world imagery folders ship a main raster plus vendor-specific sidecars:
Maxar (.IMD + .RPB), Airbus DIMAP (DIM_*.XML), Planet (_metadata.json),
Landsat MTL, Sentinel MTD_*, GDAL PAM (.aux.xml). Each vendor uses different
field names for the same concept ("firstLineTime" vs "IMAGING_DATE" vs
"acquired"). This module detects the vendor, parses its files, and returns a
normalized dict so downstream code does not have to know vendor spellings.

Keep one parser per vendor. Unknown fields go into raw_by_file so that when
a new vendor is encountered we can write a parser and re-ingest without
re-uploading.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

NORMALIZED_KEYS = (
    "vendor",
    "product_id",
    "acquired_at",
    "sensor",
    "platform",
    "cloud_cover_pct",
    "sun_elevation_deg",
    "sun_azimuth_deg",
    "off_nadir_deg",
    "gsd_m",
    "bands",
    "classification",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_files(folder: str) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(folder):
        for f in files:
            paths.append(os.path.join(root, f))
    return paths


def _has_ext(files: list[str], ext: str) -> bool:
    ext = ext.lower()
    return any(f.lower().endswith(ext) for f in files)


def _read_text(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _first_xml_root_tag(path: str) -> str | None:
    try:
        for event, elem in ET.iterparse(path, events=("start",)):
            return re.sub(r"^\{.*\}", "", elem.tag)
    except (ET.ParseError, OSError):
        return None
    return None


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Maxar / DigitalGlobe
# ---------------------------------------------------------------------------

_IMD_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?);?\s*$")

# Maxar IMD files use a flat key = "value"; structure with BEGIN_GROUP blocks.
_MAXAR_ALIASES = {
    "acquired_at":       ["firstLineTime", "earliestAcqTime", "TLCTime"],
    "sensor":            ["satId", "SATID"],
    "platform":          ["platform"],
    "cloud_cover_pct":   ["cloudCover"],
    "sun_elevation_deg": ["meanSunEl", "sunElevationAngle", "meanSunElevation"],
    "sun_azimuth_deg":   ["meanSunAz", "sunAzimuth"],
    "off_nadir_deg":     ["meanOffNadirViewAngle", "offNadirAngle"],
    "gsd_m":             ["meanCollectedGSD", "groundSampleDistance"],
    "product_id":        ["productCatalogId", "productOrderId", "catId"],
}


def match_maxar(folder: str, files: list[str]) -> bool:
    return _has_ext(files, ".imd") and (_has_ext(files, ".rpb") or _has_ext(files, ".til"))


def _parse_imd(path: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    text = _read_text(path)
    for line in text.splitlines():
        m = _IMD_KEY_RE.match(line)
        if m:
            k, v = m.group(1), m.group(2).strip().strip('"')
            result[k] = v
    return result


def parse_maxar(folder: str, files: list[str]) -> dict[str, Any]:
    imd_path = next((f for f in files if f.lower().endswith(".imd")), None)
    imd = _parse_imd(imd_path) if imd_path else {}

    out: dict[str, Any] = {"vendor": "maxar", "raw_by_file": {}}
    if imd_path:
        out["raw_by_file"][os.path.basename(imd_path)] = imd

    for norm_key, aliases in _MAXAR_ALIASES.items():
        for a in aliases:
            if a in imd:
                out[norm_key] = imd[a]
                break

    for k in ("cloud_cover_pct", "sun_elevation_deg", "sun_azimuth_deg",
              "off_nadir_deg", "gsd_m"):
        if k in out:
            out[k] = _to_float(out[k])
    if out.get("cloud_cover_pct") is not None and 0 <= out["cloud_cover_pct"] <= 1:
        out["cloud_cover_pct"] *= 100.0

    return out


# ---------------------------------------------------------------------------
# Airbus DIMAP (Pleiades, SPOT)
# ---------------------------------------------------------------------------

def match_dimap(folder: str, files: list[str]) -> bool:
    for f in files:
        if f.lower().endswith(".xml") and _first_xml_root_tag(f) == "Dimap_Document":
            return True
    return False


def _xml_find_text(root: ET.Element, *paths: str) -> str | None:
    for p in paths:
        el = root.find(p)
        if el is not None and el.text:
            return el.text.strip()
    return None


def parse_dimap(folder: str, files: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"vendor": "airbus_dimap", "raw_by_file": {}}
    xml_path = next(
        (f for f in files
         if f.lower().endswith(".xml") and _first_xml_root_tag(f) == "Dimap_Document"),
        None,
    )
    if not xml_path:
        return out
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return out

    out["product_id"] = _xml_find_text(root, ".//DATASET_NAME", ".//PRODUCT_ID")
    out["sensor"] = _xml_find_text(root, ".//MISSION_INDEX", ".//INSTRUMENT")
    out["platform"] = _xml_find_text(root, ".//MISSION")
    date = _xml_find_text(root, ".//IMAGING_DATE", ".//START_TIME")
    time = _xml_find_text(root, ".//IMAGING_TIME")
    if date and time:
        out["acquired_at"] = f"{date}T{time}"
    elif date:
        out["acquired_at"] = date
    out["cloud_cover_pct"] = _to_float(_xml_find_text(root, ".//CLOUD_COVERAGE", ".//Cloud_Cover"))
    out["sun_elevation_deg"] = _to_float(_xml_find_text(root, ".//SUN_ELEVATION"))
    out["sun_azimuth_deg"] = _to_float(_xml_find_text(root, ".//SUN_AZIMUTH"))
    out["off_nadir_deg"] = _to_float(_xml_find_text(root, ".//INCIDENCE_ANGLE", ".//VIEWING_ANGLE"))
    return out


# ---------------------------------------------------------------------------
# Planet Labs
# ---------------------------------------------------------------------------

def match_planet(folder: str, files: list[str]) -> bool:
    return any(f.lower().endswith("_metadata.json") for f in files)


def parse_planet(folder: str, files: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"vendor": "planet", "raw_by_file": {}}
    meta_path = next((f for f in files if f.lower().endswith("_metadata.json")), None)
    if not meta_path:
        return out
    try:
        data = json.loads(_read_text(meta_path, max_bytes=4_000_000))
    except json.JSONDecodeError:
        return out

    props = data.get("properties") or data
    out["raw_by_file"][os.path.basename(meta_path)] = props
    out["product_id"] = props.get("id")
    out["acquired_at"] = props.get("acquired")
    out["sensor"] = props.get("instrument")
    out["platform"] = props.get("satellite_id") or props.get("provider")
    out["cloud_cover_pct"] = _to_float(props.get("cloud_cover"))
    out["sun_elevation_deg"] = _to_float(props.get("sun_elevation"))
    out["sun_azimuth_deg"] = _to_float(props.get("sun_azimuth"))
    out["off_nadir_deg"] = _to_float(props.get("view_angle"))
    out["gsd_m"] = _to_float(props.get("gsd"))
    if out.get("cloud_cover_pct") is not None and 0 <= out["cloud_cover_pct"] <= 1:
        out["cloud_cover_pct"] *= 100.0
    return out


# ---------------------------------------------------------------------------
# USGS Landsat MTL
# ---------------------------------------------------------------------------

def match_landsat(folder: str, files: list[str]) -> bool:
    return any(f.upper().endswith("_MTL.TXT") or f.upper().endswith("_MTL.XML")
               for f in files)


def parse_landsat(folder: str, files: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"vendor": "landsat", "raw_by_file": {}}
    mtl = next((f for f in files if f.upper().endswith("_MTL.TXT")), None)
    if mtl:
        kv: dict[str, Any] = {}
        for line in _read_text(mtl).splitlines():
            m = _IMD_KEY_RE.match(line)
            if m:
                kv[m.group(1)] = m.group(2).strip().strip('"')
        out["raw_by_file"][os.path.basename(mtl)] = kv
        out["product_id"] = kv.get("LANDSAT_PRODUCT_ID") or kv.get("LANDSAT_SCENE_ID")
        out["sensor"] = kv.get("SENSOR_ID")
        out["platform"] = kv.get("SPACECRAFT_ID")
        d, t = kv.get("DATE_ACQUIRED"), kv.get("SCENE_CENTER_TIME")
        if d and t:
            out["acquired_at"] = f"{d}T{t}"
        elif d:
            out["acquired_at"] = d
        out["cloud_cover_pct"] = _to_float(kv.get("CLOUD_COVER"))
        out["sun_elevation_deg"] = _to_float(kv.get("SUN_ELEVATION"))
        out["sun_azimuth_deg"] = _to_float(kv.get("SUN_AZIMUTH"))
    return out


# ---------------------------------------------------------------------------
# GDAL PAM (.aux.xml)
# ---------------------------------------------------------------------------

def match_pam(folder: str, files: list[str]) -> bool:
    for f in files:
        if f.lower().endswith(".aux.xml") and _first_xml_root_tag(f) == "PAMDataset":
            return True
    return False


def parse_pam(folder: str, files: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"vendor": "pam", "raw_by_file": {}}
    aux_files = [f for f in files if f.lower().endswith(".aux.xml")]
    for aux in aux_files:
        try:
            tree = ET.parse(aux)
            root = tree.getroot()
        except ET.ParseError:
            continue
        mdi: dict[str, str] = {}
        for md in root.findall(".//Metadata"):
            for item in md.findall("MDI"):
                key = item.get("key")
                if key and item.text is not None:
                    mdi[key] = item.text.strip()
        out["raw_by_file"][os.path.basename(aux)] = mdi
        for candidate in ("ACQUISITIONDATETIME", "TIFFTAG_DATETIME", "AcquisitionDateTime"):
            if candidate in mdi and "acquired_at" not in out:
                out["acquired_at"] = mdi[candidate]
    return out


# ---------------------------------------------------------------------------
# Generic XML fallback
# ---------------------------------------------------------------------------

_GENERIC_FIELD_HINTS = {
    "acquired_at":       re.compile(r"(acqui[a-z]*|first[- ]?line[- ]?time|imaging[- ]?date|datetime)", re.I),
    "cloud_cover_pct":   re.compile(r"cloud", re.I),
    "sun_elevation_deg": re.compile(r"sun[- ]?el", re.I),
    "sun_azimuth_deg":   re.compile(r"sun[- ]?az", re.I),
    "sensor":            re.compile(r"sensor|sat[- ]?id|mission", re.I),
    "gsd_m":             re.compile(r"gsd|ground[- ]?sample", re.I),
    "off_nadir_deg":     re.compile(r"off[- ]?nadir|incid|view[- ]?angle", re.I),
}


def parse_generic(folder: str, files: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"vendor": "generic", "raw_by_file": {}}
    xml_files = [f for f in files
                 if f.lower().endswith(".xml") or f.lower().endswith(".aux.xml")]
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        flat: dict[str, str] = {}
        for el in tree.iter():
            if el.text and el.text.strip():
                tag = re.sub(r"^\{.*\}", "", el.tag)
                flat[tag] = el.text.strip()
        out["raw_by_file"][os.path.basename(xml_path)] = flat
        for norm_key, pattern in _GENERIC_FIELD_HINTS.items():
            if norm_key in out:
                continue
            for tag, val in flat.items():
                if pattern.search(tag):
                    out[norm_key] = val
                    break

    for k in ("cloud_cover_pct", "sun_elevation_deg", "sun_azimuth_deg",
              "off_nadir_deg", "gsd_m"):
        if k in out:
            v = _to_float(out[k])
            if v is not None:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

VENDORS: list[tuple[str, Callable[[str, list[str]], bool], Callable[[str, list[str]], dict]]] = [
    ("maxar",   match_maxar,   parse_maxar),
    ("dimap",   match_dimap,   parse_dimap),
    ("planet",  match_planet,  parse_planet),
    ("landsat", match_landsat, parse_landsat),
    ("pam",     match_pam,     parse_pam),
]


def extract_vendor_metadata(folder: str) -> tuple[dict[str, Any], list[str]]:
    """Detect vendor and extract normalized metadata from a folder.

    Returns (normalized_dict, warnings). Falls back to parse_generic for
    unknown vendors. Multi-vendor folders merge with later vendors winning
    non-null fields, and disagreements land in warnings.
    """
    files = _list_files(folder)
    warnings: list[str] = []
    merged: dict[str, Any] = {"vendor": None, "raw_by_file": {}}
    vendors_seen: list[str] = []

    for name, matcher, parser in VENDORS:
        try:
            if matcher(folder, files):
                vendors_seen.append(name)
                parsed = parser(folder, files)
                for k, v in parsed.items():
                    if k == "raw_by_file":
                        merged["raw_by_file"].update(v)
                        continue
                    if v in (None, ""):
                        continue
                    existing = merged.get(k)
                    if existing in (None, "") or k == "vendor":
                        merged[k] = v
                    elif existing != v and k != "vendor":
                        warnings.append(f"{k} conflict: {existing!r} vs {v!r} (from {name})")
        except Exception as exc:
            warnings.append(f"{name} parser failed: {exc}")

    if not vendors_seen:
        merged = parse_generic(folder, files)
    elif len(vendors_seen) > 1:
        merged["vendor"] = "+".join(vendors_seen)

    present_files = sorted(os.path.relpath(f, folder) for f in files)
    merged["bundle_files"] = present_files
    return merged, warnings
