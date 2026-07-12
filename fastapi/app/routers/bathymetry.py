"""Proxy router for the Abyss bathymetry inference service.

Mirrors the (dropped) YOLO proxy pattern: a thin, auth-guarded reverse proxy
to the standalone bathymetry-service container. The depth model is swapped in
later by replacing bathymetry-service/main.py's /infer body — this router and
the rest of the pipeline are model-agnostic.
"""

import json
import os
import uuid as _uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..routers.auth import get_current_user
from ..database import get_db
from ..models import User

router = APIRouter()

BATHYMETRY_SERVICE_URL = os.getenv("BATHYMETRY_SERVICE_URL", "http://bathymetry:8002")


@router.post("/run/{raster_id}")
async def run_bathymetry_for_raster(
    raster_id: str,
    current_user: User = Depends(get_current_user),
):
    """Manually trigger the bathymetry pipeline for an existing source raster.
    (Ingest also auto-triggers this for non-derived rasters.)"""
    from ..services.bathymetry_tasks import run_bathymetry
    run_bathymetry.apply_async(args=[raster_id], queue="sw_bg")
    return {"status": "queued", "raster_id": raster_id}


@router.post("/roi")
async def run_roi(
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """ROI-first run for the Studio: draw an area → fetch Sentinel-2 → DL-Pro →
    depth + report. Returns a job_id to poll via GET /jobs/{job_id}.

    Cached: an identical request (same area + dates + current model version) is
    served instantly from the stored report — no re-fetch, no re-inference.
    Pass {"force": true} to regenerate anyway (still reuses cached imagery).
    """
    from ..services.bathymetry_cache import normalize_bbox, result_key, current_model

    bbox = normalize_bbox(payload.get("bbox"))
    if bbox is None:
        raise HTTPException(status_code=400, detail="bbox required ({west,south,east,north} or [w,s,e,n])")
    name = (payload.get("name") or "Survey area")[:120]
    sd = payload.get("start_date", "2023-01-01")
    ed = payload.get("end_date", "2024-12-31")
    force = bool(payload.get("force"))
    try:
        n_scenes = max(1, min(int(payload.get("n_scenes") or 1), 7))
    except (TypeError, ValueError):
        n_scenes = 1
    engine = (payload.get("engine") or "").strip().lower() or None
    if engine not in (None, "uae-sdb-ensemble", "dl-pro-v3"):
        raise HTTPException(status_code=400,
                            detail="engine must be 'uae-sdb-ensemble' or 'dl-pro-v3'")

    model, version = current_model(BATHYMETRY_SERVICE_URL)
    ckey = result_key(bbox, sd, ed, engine or model, version)
    if n_scenes > 1:  # composite products dedup separately from single-scene
        ckey = f"{ckey}-n{n_scenes}"

    # Dedup: return the stored product for an identical (area, dates, model) run.
    if not force:
        hit = db.execute(
            text("SELECT id, report_code, depth_raster_id FROM bathymetry_reports "
                 "WHERE cache_key = :k AND status = 'ready' "
                 "ORDER BY generated_at DESC NULLS LAST LIMIT 1"),
            {"k": ckey},
        ).mappings().first()
        if hit:
            job_id = str(_uuid.uuid4())
            db.execute(
                text("INSERT INTO bathymetry_jobs "
                     "(id, status, model, requested_by, depth_raster_id, report_id, metadata) "
                     "VALUES (:id, 'done', :model, :uid, :depth, :rep, CAST(:meta AS jsonb))"),
                {"id": job_id, "model": model, "uid": str(current_user.id),
                 "depth": str(hit["depth_raster_id"]) if hit["depth_raster_id"] else None,
                 "rep": str(hit["id"]),
                 "meta": json.dumps({"report_code": hit["report_code"], "cached": True})},
            )
            db.commit()
            return {"job_id": job_id, "status": "done",
                    "report_code": hit["report_code"], "cached": True}

    job_id = str(_uuid.uuid4())
    db.execute(
        text("INSERT INTO bathymetry_jobs (id, status, model, requested_by) "
             "VALUES (:id, 'running', :model, :uid)"),
        {"id": job_id, "model": model, "uid": str(current_user.id)},
    )
    db.commit()
    from ..services.bathymetry_tasks import run_bathymetry_roi
    run_bathymetry_roi.apply_async(args=[job_id, bbox, sd, ed, name, ckey,
                                         n_scenes, engine],
                                   queue="sw_bg")
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}")
async def job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = db.execute(
        text("SELECT status, error_message, report_id, depth_raster_id, metadata "
             "FROM bathymetry_jobs WHERE id = :id"),
        {"id": job_id},
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    meta = row["metadata"] or {}
    return {
        "job_id": job_id,
        "status": row["status"],
        "error": row["error_message"],
        "report_id": str(row["report_id"]) if row["report_id"] else None,
        "report_code": meta.get("report_code"),
        "cached": bool(meta.get("cached")),
        "depth_raster_id": str(row["depth_raster_id"]) if row["depth_raster_id"] else None,
    }


def _load_report(db: Session, ref: str):
    row = db.execute(
        text("SELECT id, report_code, site_name, statistics FROM bathymetry_reports "
             "WHERE report_code = :r OR CAST(id AS text) = :r LIMIT 1"),
        {"r": ref},
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="report not found")
    return row


@router.post("/validate")
async def validate_soundings(
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """IHO S-44 validation of a produced depth layer against user-uploaded
    reference soundings. Pass report_code to resolve the stored depth product
    and persist the result into that report's statistics + sections."""
    report_code = payload.pop("report_code", None)
    if report_code and not payload.get("raster_id") and not payload.get("predicted"):
        row = _load_report(db, report_code)
        rid = (row["statistics"] or {}).get("infer_raster_id")
        if not rid:
            raise HTTPException(
                status_code=400,
                detail="this survey predates stored depth products — re-run it "
                       "or retry (the page will fall back to grid samples)")
        payload["raster_id"] = rid

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        try:
            resp = await client.post(f"{BATHYMETRY_SERVICE_URL}/bathymetry/validate",
                                     json=payload)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Bathymetry service unavailable")
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    result = resp.json()

    if report_code:
        from ..services.report_builder import attach_validation
        try:
            attach_validation(db, report_code, result)
            result["saved_to_report"] = report_code
        except Exception:
            result["saved_to_report"] = None
    return result


@router.post("/calibrate")
async def calibrate_to_points(
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fit a robust local calibration of a survey's depth layer to uploaded
    soundings (80/20 holdout) and ingest the calibrated grid as a NEW derived
    layer + report. The original layer and report are untouched."""
    import io
    from ..services import minio_storage, report_builder
    from ..services.tasks import process_raster_upload
    from ..models import RasterCatalog

    report_code = payload.get("report_code")
    observed = payload.get("observed") or []
    if not report_code or not observed:
        raise HTTPException(status_code=400,
                            detail="report_code and observed points required")
    row = _load_report(db, report_code)
    stats = row["statistics"] or {}
    rid = stats.get("infer_raster_id")
    if not rid:
        raise HTTPException(status_code=400,
                            detail="this survey predates calibration support — "
                                   "re-run it first")

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        try:
            resp = await client.post(
                f"{BATHYMETRY_SERVICE_URL}/bathymetry/calibrate",
                json={"raster_id": rid, "observed": observed})
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Bathymetry service unavailable")
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    data = resp.json()

    # carry model provenance from the source survey
    data.setdefault("model", stats.get("model") or "dl-pro-v3")
    data.setdefault("model_version", stats.get("model_version"))
    data.setdefault("engine", stats.get("engine"))
    site = (row["site_name"] or "Survey area")

    derived_id = str(_uuid.uuid4())
    raw_key = f"{derived_id}/depth_rgb.tif"
    try:
        blob = minio_storage.download_file(data.get("depth_bucket", "depth"),
                                           data["depth_rgb_key"])
        minio_storage.upload_file("raw", raw_key, io.BytesIO(blob),
                                  content_type="image/tiff", length=len(blob))
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"calibrated product transfer failed: {e}")

    cal = data.get("calibration_local") or {}
    derived = RasterCatalog(
        id=derived_id, name=f"Bathymetry — {site} (calibrated)",
        description=f"Depth product for {site}, locally calibrated to "
                    f"{cal.get('n_points', '?')} user soundings",
        original_filename=f"depth_{rid}_calibrated.tif", original_format="GTiff",
        minio_raw_path=raw_key, processing_status="pending",
        uploaded_by=current_user.id,
        metadata_={"source_kind": "bathymetry", "parent_raster_id": None,
                   "bathymetry": {"model": data.get("model"),
                                  "stats": data.get("stats"),
                                  "max_depth_m": data.get("max_depth_m"),
                                  "calibration": "local_points",
                                  "calibration_local": cal,
                                  "derived_from_report": row["report_code"]}},
    )
    db.add(derived)
    db.commit()
    process_raster_upload.delay(derived_id)

    report_id, new_code = report_builder.build_report(
        db, None, derived_id, data,
        created_by=str(current_user.id),
        site_name=f"{site} · calibrated")
    return {"status": "done", "report_code": new_code, "report_id": report_id,
            "depth_raster_id": derived_id,
            "calibration": cal,
            "source_report_code": row["report_code"]}


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def bathymetry_proxy(
    path: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Proxy requests to the bathymetry inference service with JWT validation."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        url = f"{BATHYMETRY_SERVICE_URL}/bathymetry/{path}"
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "authorization")
        }
        headers["X-User"] = str(current_user.id)

        body = await request.body()

        try:
            if request.headers.get("accept") == "text/event-stream":
                async def stream_bathy():
                    async with client.stream(
                        request.method, url, headers=headers, content=body
                    ) as resp:
                        async for line in resp.aiter_lines():
                            yield line + "\n"

                return StreamingResponse(
                    stream_bathy(), media_type="text/event-stream"
                )
            else:
                resp = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=body,
                    params=dict(request.query_params),
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers={
                        "content-type": resp.headers.get(
                            "content-type", "application/json"
                        )
                    },
                )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503, detail="Bathymetry service unavailable"
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
