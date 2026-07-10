"""Build template-based bathymetry reports.

A report is an ordered list of typed sections (the `bathymetry_v1` template),
auto-filled from the pipeline output — NOT freeform text. Section content is
baked into `bathymetry_reports.sections` (JSONB) so the client renderer just
iterates and dispatches on `type`. Prose is generated from fixed Python
templates (no LLM), so this is deterministic and airgap-safe.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session


def _summary_text(site: str, stats: dict, model: str) -> str:
    return (
        f"This bathymetric survey of {site} derives water-depth estimates from the "
        f"source imagery using the {model} model. Estimated depths range from "
        f"{stats.get('min_m', '?')} m in the shallows to {stats.get('max_m', '?')} m "
        f"at the deepest sounding, with a mean depth of {stats.get('mean_m', '?')} m "
        f"(σ {stats.get('std_m', '?')} m). Valid depth coverage spans "
        f"{stats.get('coverage_pct', '?')}% of the scene. The colour-mapped depth "
        f"surface and the extracted cross-section below summarise the bathymetry."
    )


def _methodology_text(model: str) -> str:
    return (
        "Satellite-Derived Bathymetry (SDB). Free Sentinel-2 L2A surface-reflectance "
        "imagery is pulled for the survey ROI (Microsoft Planetary Computer), and a "
        f"23-feature turbidity-aware MLP ({model}) estimates per-pixel water depth from "
        "log-band ratios, NDWI, local texture, Nechad SPM and bottom-type clustering. "
        "Output is IHO-calibrated, clamped to 0–25 m, colour-mapped (shallow cyan → "
        "abyssal navy) and re-ingested as a georeferenced depth layer; MC-Dropout gives "
        "per-pixel uncertainty and IHO S-44 compliance. Depth values are model estimates "
        "and should be validated against in-situ soundings before navigational use."
    )


def build_report(
    db: Session,
    source_raster,
    depth_raster_id: str,
    infer_data: dict,
    created_by: str | None = None,
    site_name: str | None = None,
    cache_key: str | None = None,
) -> tuple[str, str]:
    """Create a `bathymetry_reports` row with fully-filled sections. Returns
    (report_id, report_code). `source_raster` may be None for an ROI-first run
    (the studio draws an area with no uploaded source image)."""
    report_id = str(uuid.uuid4())
    report_code = "ABY-" + uuid.uuid4().hex[:8].upper()
    site = site_name or (getattr(source_raster, "name", None) if source_raster else None) or "Survey area"
    src_crs = getattr(source_raster, "crs", None) if source_raster else None
    stats = infer_data.get("stats", {})
    model = infer_data.get("model", "dl-pro-v3")
    classification = "UNCLASSIFIED"
    generated_at = datetime.now(timezone.utc)

    acquisition = None
    if source_raster is not None and getattr(source_raster, "acquisition_date", None):
        acquisition = source_raster.acquisition_date.isoformat()

    # Source-imagery section: the uploaded scene's thumbnail when present,
    # else a note pointing at the fetched Sentinel-2 granule.
    if source_raster is not None:
        source_section = {
            "type": "image", "title": "Source Imagery",
            "data": {"url": f"/api/rasters/{source_raster.id}/thumbnail",
                     "caption": "Source scene used for depth inference."},
        }
    else:
        source_section = {
            "type": "kv", "title": "Source Imagery",
            "data": {"rows": [
                ["Sentinel-2 scene", infer_data.get("scene_id") or "—"],
                ["Acquired", infer_data.get("acquired") or "—"],
                ["Cloud cover", f"{infer_data.get('cloud_cover', '—')} %"],
                ["Provider", "Microsoft Planetary Computer (free)"],
            ]},
        }

    sections = [
        {
            "type": "cover",
            "title": f"Bathymetric Survey — {site}",
            "data": {
                "report_code": report_code,
                "site_name": site,
                "classification": classification,
                "model": model,
                "generated_at": generated_at.isoformat(),
            },
        },
        {
            "type": "kv",
            "title": "Survey Metadata",
            "data": {"rows": [
                ["Report code", report_code],
                ["Survey area", site],
                ["Model", f"{model} {infer_data.get('model_version') or ''}".strip()],
                ["Sentinel-2 scene", infer_data.get("scene_id") or "—"],
                ["Acquired", (infer_data.get("acquired") or acquisition or "—")],
                ["Cloud cover", f"{infer_data.get('cloud_cover', '—')} %"],
                ["CRS", infer_data.get("crs") or src_crs or "EPSG:4326"],
                ["Depth coverage", f"{stats.get('coverage_pct', '—')} %"],
                ["Max modelled depth", f"{infer_data.get('max_depth_m', '—')} m"],
                ["Model hold-out RMSE", f"{(infer_data.get('holdout_metrics') or {}).get('rmse', '—')} m"],
            ]},
        },
        {
            "type": "prose",
            "title": "Executive Summary",
            "data": {"text": _summary_text(site, stats, model)},
        },
        source_section,
        {
            "type": "raster",
            "title": "Depth Map",
            "data": {"raster_id": str(depth_raster_id),
                     "url": f"/api/rasters/{depth_raster_id}/thumbnail",
                     "legend": "depth",
                     "max_depth_m": infer_data.get("max_depth_m"),
                     "caption": "Colour-mapped water depth (shallow → abyssal)."},
        },
        {
            "type": "chart",
            "title": "Depth Cross-section",
            "data": {"profile": infer_data.get("profile", {"distance_m": [], "depth_m": []})},
        },
        {
            "type": "stats",
            "title": "Depth Statistics",
            "data": {"stats": stats},
        },
        {
            "type": "kv",
            "title": "IHO S-44 Compliance (model baseline)",
            "data": {"rows": [
                [k.replace("_", " "), f"{v} %"]
                for k, v in (infer_data.get("iho_s44_pct") or {}).items()
            ] or [["—", "no compliance data"]]},
        },
        {
            "type": "prose",
            "title": "Methodology & Caveats",
            "data": {"text": _methodology_text(model)},
        },
    ]

    db.execute(
        text(
            """
            INSERT INTO bathymetry_reports
                (id, report_code, source_raster_id, depth_raster_id, template_id,
                 title, site_name, classification_level, geom, status, sections,
                 statistics, generated_at, created_by, cache_key)
            VALUES
                (:id, :code, :src, :depth, 'bathymetry_v1',
                 :title, :site, :cls,
                 (SELECT center_point FROM raster_catalog WHERE id = :depth),
                 'ready', CAST(:sections AS jsonb),
                 CAST(:stats AS jsonb), :gen, :uid, :ckey)
            """
        ),
        {
            "id": report_id,
            "code": report_code,
            "src": str(source_raster.id) if source_raster is not None else None,
            "depth": str(depth_raster_id),
            "title": f"Bathymetric Survey — {site}",
            "site": site,
            "cls": classification,
            "sections": json.dumps(sections),
            # statistics carries the headline numbers + IHO + the downsampled
            # depth/sigma grid the Studio uses for 2D/3D/profile rendering.
            "stats": json.dumps({
                **stats,
                "iho_s44_pct": infer_data.get("iho_s44_pct"),
                "holdout_metrics": infer_data.get("holdout_metrics"),
                "grid": infer_data.get("grid"),
                "scene_id": infer_data.get("scene_id"),
                "max_depth_m": infer_data.get("max_depth_m"),
            }),
            "gen": generated_at,
            "uid": created_by,
            "ckey": cache_key,
        },
    )
    db.commit()
    return report_id, report_code
