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


def _methodology_text(model: str, infer_data: dict | None = None) -> str:
    d = infer_data or {}
    calib = d.get("calibration") or {}
    if model == "vmarch-sdb":
        blend = calib.get("uae_blend") or {}
        bias = calib.get("bias_correction") or {}
        refs = calib.get("refs") or {}
        inside = blend.get("inside_predefined")
        bias_txt = (
            f"3-stage bias correction applied ({bias.get('stage1', 'isotonic')} "
            f"recalibration → per-pixel kNN-IDW residual field → ±3 m clamp; "
            f"training RMSE {bias.get('pre_rmse_m', '—')}→"
            f"{bias.get('post_rmse_m', '—')} m on {bias.get('n_train_used', '—')} "
            "reference anchors)" if bias.get("applied") else
            "3-stage bias correction skipped (insufficient reference anchors "
            "in the ROI)")
        return (
            "Satellite-Derived Bathymetry — VMarch production method (as deployed "
            "on the Bathymetry-from-Space Railway platform). Per-pixel depth is a "
            "weighted fusion of (1) Lyzenga (1985/2006) multi-band log-linear WLS "
            "on deep-water-subtracted reflectance, (2) Stumpf (2003) blue/green "
            "and blue/red log-ratios, combined by inverse-RMSE ensemble "
            f"({calib.get('ls_ensemble', '—')}), and (3) the UAE-clustered "
            "RF/MLP pretrained ensemble blended with REGION-AWARE weights "
            f"(w_UAE={blend.get('w_uae', '—')}; "
            + ("inside a calibrated UAE region → 90% Lyzenga+Stumpf / 10% "
               "pretrained" if inside else
               "outside the calibration envelopes → 50/50 transfer-learning "
               "blend") +
            f"). Calibration anchors: {refs.get('user', 0)} user points, "
            f"{refs.get('gebco', 0)} GEBCO fall-through points. " + bias_txt +
            ". Output clamped to 0–25 m; land cut with SCL/NDWI + vector "
            "coastline; per-pixel σ is a TPU-style total "
            "(model/tide/refraction/reference/georeferencing terms). Depth "
            "values are model estimates and must be validated against in-situ "
            "soundings before navigational use."
        )
    if model == "uae-sdb-ensemble":
        return (
            "Satellite-Derived Bathymetry (SDB). Per-pixel depth is estimated from the "
            "ingested scene's own multispectral bands by a cluster ensemble (K-means "
            "optical-regime clustering with per-cluster Random Forest and MLP "
            "regressors) over a 13-feature spectral stack: raw reflectances, Lyzenga "
            "log-bands and depth-invariant index, Stumpf blue/green and blue/red "
            "log-ratios, and NDWI. Land is cut with an NDWI water mask; output is "
            "clamped to 0–25 m and colour-mapped (shallow cyan → abyssal navy). The "
            f"model registry is v{calib.get('registry_version', 1)}, calibrated on "
            f"{calib.get('domain', 'UAE coastal waters (8 calibration regions)')} "
            f"(in-sample R² {calib.get('in_sample_r2', 0.978)}, RMSE "
            f"{calib.get('in_sample_rmse_m', 0.735)} m); outside the UAE calibration "
            "envelope the output is indicative only. Depths are referenced to the "
            "calibration soundings' survey datum and are not tide-corrected. A "
            "per-pixel uncertainty channel combines per-cluster calibration RMSE and "
            "ensemble disagreement. Depth values are model estimates and must be "
            "validated against in-situ soundings before navigational use."
        )
    if model == "stumpf-log-ratio":
        return (
            "UNCALIBRATED pseudo-depth (low confidence). The source scene carries only "
            "RGB bands (no NIR/multispectral), so no calibrated depth model applies. "
            "The depth surface is the Stumpf (2003) blue/green log-ratio, "
            "percentile-stretched to a literature-anchored 0–25 m range: it preserves "
            "RELATIVE shallow-to-deep structure but is NOT metric depth. Land is cut "
            "with a blue-water index. Do not use for navigation, engineering or any "
            "quantitative purpose; ingest multispectral imagery (e.g. Sentinel-2 "
            "B02/B03/B04/B08) for calibrated output."
        )
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


def _metadata_rows(report_code, site, model, infer_data, stats, acquisition,
                   src_crs):
    rows = [
        ["Report code", report_code],
        ["Survey area", site],
        ["Model", infer_data.get("model_label")
         or f"{model} {infer_data.get('model_version') or ''}".strip()],
    ]
    if infer_data.get("method_card"):
        rows.append(["Method preset", infer_data["method_card"]])
    if infer_data.get("vmarch_core"):
        vc = infer_data["vmarch_core"]
        ver = vc.get("platform_version")
        rows.append(["Computed by",
                     "Bathymetry-from-Space platform"
                     + (f" v{ver}" if ver else "")
                     + (f" — {vc.get('endpoint')}" if vc.get("endpoint") else "")])
    if infer_data.get("method"):
        rows.append(["Method", infer_data["method"]])
    if infer_data.get("resolution_m"):
        rows.append(["Output resolution", f"{infer_data['resolution_m']} m"])
    if infer_data.get("engine_fallback"):
        rows.append(["Engine fallback", infer_data["engine_fallback"]])
    if infer_data.get("scene_id"):
        rows += [
            ["Sentinel-2 scene", infer_data.get("scene_id")],
            ["Cloud cover", f"{infer_data.get('cloud_cover', '—')} %"],
        ]
    rows += [
        ["Acquired", (infer_data.get("acquired") or acquisition or "—")],
        ["CRS", infer_data.get("crs") or src_crs or "EPSG:4326"],
        ["Depth coverage", f"{stats.get('coverage_pct', '—')} %"],
        ["Max modelled depth", f"{infer_data.get('max_depth_m', '—')} m"],
    ]
    calib = infer_data.get("calibration") or {}
    holdout_rmse = (infer_data.get("holdout_metrics") or {}).get("rmse")
    if holdout_rmse is not None:
        rows.append(["Model hold-out RMSE", f"{holdout_rmse} m"])
    elif calib.get("in_sample_rmse_m") is not None:
        rows += [
            ["Calibration domain", calib.get("domain", "—")],
            ["Calibration fit (in-sample)",
             f"R² {calib.get('in_sample_r2', '—')}, RMSE {calib.get('in_sample_rmse_m', '—')} m"],
            ["Datum", "Calibration survey datum (not tide-corrected)"],
        ]
    if infer_data.get("calibrated") is False:
        rows.append(["Confidence", infer_data.get("confidence") or
                     "LOW — uncalibrated pseudo-depth"])
    tide = infer_data.get("tide") or {}
    if tide:
        if tide.get("applied"):
            rows += [
                ["Tide correction",
                 f"{tide.get('applied_m', 0.0):+.2f} m (Open-Meteo/CMEMS)"],
                ["Vertical datum", "MSL (tide-corrected)"],
            ]
        else:
            reason = tide.get("reason") or tide.get("method") or "unavailable"
            rows.append(["Tide correction", f"not applied — {reason}"])
    return rows


VALIDATION_SECTION_TITLE = "Reference Validation (IHO S-44)"


def _validation_section(validation: dict) -> dict:
    iho_orders = validation.get("iho_orders") or {}
    val_rows = [
        ["Reference soundings matched", validation.get("n_pairs", "—")],
        ["RMSE", f"{validation.get('rmse', '—')} m"],
        ["MAE", f"{validation.get('mae', '—')} m"],
        ["Bias", f"{validation.get('bias', '—')} m"],
        ["R²", validation.get("r2", "—")],
        ["p95 error", f"{validation.get('p95_error_m', '—')} m"],
        ["IHO order (p95 ≤ TVU)", validation.get("order_label") or "—"],
        ["CATZOC", validation.get("catzoc") or "—"],
    ] + [
        [f"TVU pass ({k})", f"{v.get('pass_pct', '—')} %"]
        for k, v in iho_orders.items()
    ]
    if validation.get("detection_capability_note"):
        val_rows.append(["Detection capability",
                         validation["detection_capability_note"]])
    if validation.get("tpu_caveat"):
        val_rows.append(["TPU caveat", validation["tpu_caveat"]])
    return {"type": "kv", "title": VALIDATION_SECTION_TITLE,
            "data": {"rows": val_rows}}


def attach_validation(db: Session, ref: str, validation: dict) -> None:
    """Persist a post-hoc soundings validation into an existing report:
    statistics.validation + the S-44 section (replaced if already present).
    The full pair list is dropped from the stored copy to keep the row slim."""
    row = db.execute(
        text("SELECT id, sections, statistics FROM bathymetry_reports "
             "WHERE report_code = :r OR CAST(id AS text) = :r LIMIT 1"),
        {"r": ref},
    ).mappings().first()
    if not row:
        raise ValueError("report not found")
    slim = {k: v for k, v in validation.items()
            if k not in ("pairs", "pair_stats")}
    section = _validation_section(slim)
    sections = list(row["sections"] or [])
    sections = [s for s in sections
                if (s or {}).get("title") != VALIDATION_SECTION_TITLE]
    sections.insert(max(0, len(sections) - 1), section)
    stats = dict(row["statistics"] or {})
    stats["validation"] = slim
    db.execute(
        text("UPDATE bathymetry_reports "
             "SET sections = CAST(:sec AS jsonb), statistics = CAST(:st AS jsonb) "
             "WHERE id = :id"),
        {"sec": json.dumps(sections), "st": json.dumps(stats), "id": str(row["id"])},
    )
    db.commit()


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
            "data": {"rows": _metadata_rows(report_code, site, model, infer_data,
                                            stats, acquisition, src_crs)},
        },
        {
            "type": "prose",
            "title": "Executive Summary",
            "data": {"text": _summary_text(site, stats, model) + (
                " NOTE: this product is an uncalibrated relative pseudo-depth "
                "(RGB-only source) — depth values are not metric."
                if infer_data.get("calibrated") is False else "")},
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
            "data": {"text": _methodology_text(model, infer_data)},
        },
    ]

    composite = infer_data.get("composite")
    if composite:
        per_scene = composite.get("per_scene") or []
        comp_rows = [
            ["Scenes used", f"{composite.get('n_scenes_used', '—')} of "
                            f"{composite.get('n_scenes_requested', '—')} requested"],
            ["Scene dates", ", ".join(str(p.get("acquired") or "?")[:10]
                                      for p in per_scene if p.get("kept")) or "—"],
            ["Per-scene σ (median)",
             "; ".join(f"{str(p.get('acquired') or p.get('scene_id') or '?')[:10]}: "
                       f"{p.get('sigma_med', '—')} m"
                       + ("" if p.get("kept") else " (dropped: glint)")
                       for p in per_scene) or "—"],
            ["Cross-scene agreement (median px σ)",
             f"{composite.get('px_std_median_m', '—')} m"],
            ["Cross-scene agreement (p95 px σ)",
             f"{composite.get('px_std_p95_m', '—')} m"],
        ]
        sections.insert(-1, {"type": "kv", "title": "Multi-scene Composite",
                             "data": {"rows": comp_rows}})

    validation = infer_data.get("validation")
    if validation:
        sections.insert(-1, _validation_section(validation))

    calibration_local = infer_data.get("calibration_local")
    if calibration_local:
        hold = calibration_local.get("holdout") or {}
        before, after = hold.get("before") or {}, hold.get("after") or {}
        shift = calibration_local.get("shift_px") or {}
        lin = calibration_local.get("linear") or {}
        cal_rows = [
            ["Provenance", calibration_local.get("provenance", "local_points")],
            ["Soundings used", f"{calibration_local.get('n_points', '—')} "
                               f"({calibration_local.get('n_train', '—')} fit / "
                               f"{calibration_local.get('n_holdout', '—')} holdout)"],
            ["Steps", ", ".join(calibration_local.get("steps") or []) or "—"],
            ["Co-registration shift", f"dr {shift.get('dr', 0)} px, "
                                      f"dc {shift.get('dc', 0)} px"],
            ["Robust linear fit", f"depth' = {lin.get('a', '—')} × depth + "
                                  f"{lin.get('b', '—')} m"],
            ["Holdout RMSE (before → after)",
             f"{before.get('rmse_m', '—')} m → {after.get('rmse_m', '—')} m"],
            ["Holdout MAE (before → after)",
             f"{before.get('mae_m', '—')} m → {after.get('mae_m', '—')} m"],
            ["Holdout bias (before → after)",
             f"{before.get('bias_m', '—')} m → {after.get('bias_m', '—')} m"],
        ]
        if calibration_local.get("note"):
            cal_rows.append(["Note", calibration_local["note"]])
        sections.insert(-1, {"type": "kv",
                             "title": "Local Calibration (user soundings)",
                             "data": {"rows": cal_rows}})

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
                "model": model,
                "model_version": infer_data.get("model_version"),
                "model_label": infer_data.get("model_label"),
                "method_card": infer_data.get("method_card"),
                "resolution_m": infer_data.get("resolution_m"),
                "engine": infer_data.get("engine"),
                "method": infer_data.get("method"),
                "calibrated": infer_data.get("calibrated"),
                "calibration": infer_data.get("calibration"),
                # infer-time raster id → locates depth/{id}/depth.tif for
                # post-hoc validation / local calibration.
                "infer_raster_id": infer_data.get("raster_id"),
                "depth_raw_key": infer_data.get("depth_raw_key"),
                **({"tide": infer_data["tide"]} if infer_data.get("tide") else {}),
                # Untouched provenance block from the real VMarch platform
                # (vmarch-core) — method labels/metrics verbatim.
                **({"vmarch_core": infer_data["vmarch_core"]}
                   if infer_data.get("vmarch_core") else {}),
                **({"composite": composite} if composite else {}),
                **({"validation": validation} if validation else {}),
                **({"calibration_local": calibration_local} if calibration_local else {}),
            }),
            "gen": generated_at,
            "uid": created_by,
            "ckey": cache_key,
        },
    )
    db.commit()
    return report_id, report_code
