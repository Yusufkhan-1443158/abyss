"""Seed-on-empty sample results so a fresh/empty deploy is usable out of the box.

The durable results store (``backend/jobs_store.py``, index under
``Very_HR_Results/_jobs/_results/results_index.json``) is *runtime state* that
is NOT shipped with the repo (gitignored outputs). On a clean clone or a
Railway redeploy the Results panel therefore shows "FINISHED JOBS · 0".

This module ships a couple of SMALL real depth GeoTIFFs under
``samples/results/`` (bundled, non-gitignored) plus ``seed_manifest.json`` with
the honest per-result metrics. ``seed_sample_results_if_empty()`` registers
those samples through the normal ``register_result`` path — but ONLY when the
store currently has zero ``done`` results. It is fully idempotent (fixed result
ids, de-duped by ``register_result``) and never overwrites real user results.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    from backend import jobs_store as _store
except ImportError:  # pragma: no cover - script-style import
    import jobs_store as _store  # type: ignore

try:
    from backend import model_card as _model_card
except ImportError:  # pragma: no cover - script-style import
    import model_card as _model_card  # type: ignore

L = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = _REPO_ROOT / "samples" / "results"
MANIFEST_PATH = SAMPLES_DIR / "seed_manifest.json"


def _load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except Exception as ex:  # pragma: no cover
        L.warning("seed_results: cannot parse manifest %s: %s", MANIFEST_PATH, ex)
        return []
    return data.get("results", []) if isinstance(data, dict) else []


def _has_done_results() -> bool:
    try:
        return any(e.get("status") == "done" for e in _store.list_results(limit=1000))
    except Exception:
        return False


def seed_sample_results_if_empty() -> int:
    """Register bundled sample results iff the store has zero 'done' results.

    Returns the number of results seeded (0 = no-op, store already populated or
    nothing to seed). Idempotent and safe to call at startup and per-request.
    """
    if _has_done_results():
        return 0  # store already has real (or previously-seeded) results — no-op.

    entries = _load_manifest()
    if not entries:
        return 0

    seeded = 0
    for item in entries:
        try:
            src = SAMPLES_DIR / item["file"]
            if not src.exists():
                L.warning("seed_results: sample tif missing: %s", src)
                continue
            rid = item["id"]
            # Skip if this exact seed id is already registered (race / re-entry).
            if _store.get_result(rid) is not None:
                continue
            # Copy the bundled tif into the store's OUTPUTS_DIR so the download /
            # resolve_output path is identical to a real finished job. We pass
            # output_bytes so register_result stores it under <rid>__<name>.
            _store.register_result(
                result_id=rid,
                name=item["name"],
                roi_bbox=item.get("roi_bbox") or [],
                resolution=item.get("resolution") or "10m",
                status="done",
                output_bytes=src.read_bytes(),
                output_filename=item["file"],
                metrics=_metrics_with_mask(item),
                job_id=None,
                error=None,
            )
            # Restore the manifest's honest 'date' (register_result stamps now()).
            _backdate(rid, item.get("date"))
            seeded += 1
        except Exception as ex:  # pragma: no cover
            L.warning("seed_results: failed to seed %s: %s", item.get("id"), ex)

    if seeded:
        L.info("seed_results: seeded %d sample result(s) into empty store.", seeded)
    return seeded


def _metrics_with_mask(item: dict[str, Any]) -> dict[str, Any] | None:
    """Attach the absolute server-side path of the bundled optical_valid mask
    (if the manifest item declares `mask_file` and it exists) into the metrics
    under `mask_path`, so GET /api/results/<id>/mask can resolve it without a
    separate schema change. Returns a (possibly augmented) metrics dict."""
    metrics = dict(item.get("metrics") or {})
    mf = item.get("mask_file")
    if mf:
        mp = SAMPLES_DIR / mf
        if mp.exists():
            metrics["mask_path"] = str(mp)
            if item.get("mask_classes"):
                metrics["mask_classes"] = item["mask_classes"]
    # Multi-raster products (e.g. the DL-fusion marawah LAT set: depth_lat
    # primary + sigma_lat / depth_d1a / lat_offset companions). Each companion
    # raster's absolute server-side path is recorded under `extra_downloads`
    # (same in-place-serve pattern as `mask_path`); the download route resolves
    # it by `key`. Recomputed each seed so the path is runtime-correct.
    extra = item.get("extra_files") or []
    if extra:
        dl = []
        for ef in extra:
            try:
                fn = ef.get("filename")
                ep = SAMPLES_DIR / fn if fn else None
                if ep is None or not ep.exists():
                    continue
                dl.append({
                    "key": ef.get("key") or ep.stem,
                    "label": ef.get("label") or ep.name,
                    "filename": ep.name,
                    "path": str(ep),
                    "size_bytes": ep.stat().st_size,
                })
            except Exception:
                continue
        if dl:
            metrics["extra_downloads"] = dl
    # Fold manifest-level descriptive fields into metrics so they survive the
    # fixed register_result schema and reach the UI (register_result preserves
    # the metrics dict verbatim). Additive — never overwrites a metric.
    for k in ("is_default", "vhr_action", "units", "source_note"):
        if k in item and k not in metrics:
            metrics[k] = item[k]
    # DL_WEB_LOG.md §4 / AC-3 / AC-10: for a default DL region (Khalifa, OMC)
    # the manifest carries the flat headline numbers + a `_dl_card_region`
    # marker; attach the FULL auditable §4 provenance block built from the
    # committed model meta (single source of truth) so the UI Model-Card panel
    # renders for the default cards exactly as it does for fresh user runs.
    # Lyzenga/recon defaults (Jbel inline card / Abu Al Abyad) are left untouched.
    region = metrics.get("_dl_card_region")
    if region and "model_card" not in metrics:
        try:
            card = _model_card.build_dl_model_card(
                region=str(region),
                resolution_m=int(str(item.get("resolution", "10m")).replace("m", "") or 10),
                region_training="trained",
                catzoc=metrics.get("catzoc"),
                date_processed=None,
            )
            metrics["model_card"] = card
            # Mirror the suppression flag onto the flat metrics so the inline
            # R² cell also reads "suppressed" (OMC) without the panel open.
            _cm = card.get("metrics") or {}
            if _cm.get("r2_suppressed"):
                metrics.setdefault("r2_suppressed", True)
        except Exception as ex:  # pragma: no cover
            L.warning("seed_results: model-card build failed for %s: %s",
                      item.get("id"), ex)
    return metrics or None


def _seed_one(item: dict[str, Any]) -> dict[str, Any] | None:
    """Register a single manifest item into the store (idempotent). Returns the
    stored entry, or None if the sample tif is missing."""
    src = SAMPLES_DIR / item["file"]
    if not src.exists():
        L.warning("seed_results: sample tif missing: %s", src)
        return None
    rid = item["id"]
    existing = _store.get_result(rid)
    if existing is not None:
        # DEFAULT regions are AUTHORITATIVE from the manifest, not user data. If
        # the manifest's metrics differ from what is stored (e.g. an existing
        # deploy seeded the pre-DL Lyzenga numbers and we've since switched the
        # default cards to the DL PatchCNN §4 model card), REFRESH the stored
        # default entry in place so the new provenance/numbers actually appear.
        # Non-default (user) results are never touched here. This is the fix that
        # makes AC-3/AC-10 hold on a box whose results_index.json predates this
        # round. Idempotent: once refreshed, the metrics compare equal → no-op.
        if item.get("is_default") or item.get("always_seed"):
            # If the manifest's PRIMARY raster changed for an authoritative
            # entry (e.g. the marawah CERTIFIED entry was repointed from the LAT
            # raster to the surface-relative D1a raster in the DLF-R13b label
            # split), the in-place metric patch below would leave the OLD output
            # file bytes attached. Detect the mismatch by expected stored name
            # and RE-REGISTER (re-copies bytes + refreshes metrics/name). Only
            # fires when the primary file actually differs → idempotent no-op
            # otherwise.
            try:
                expected_out = f"{rid}__{_store._safe_name(item['file'])}"  # type: ignore[attr-defined]
                if existing.get("output_name") != expected_out:
                    src = SAMPLES_DIR / item["file"]
                    if src.exists():
                        _store.register_result(
                            result_id=rid,
                            name=item["name"],
                            roi_bbox=item.get("roi_bbox") or [],
                            resolution=item.get("resolution") or "10m",
                            status="done",
                            output_bytes=src.read_bytes(),
                            output_filename=item["file"],
                            metrics=_metrics_with_mask(item),
                            job_id=None,
                            error=None,
                        )
                        _backdate(rid, item.get("date"))
                        existing = _store.get_result(rid) or existing
                        L.info("seed_results: re-registered %s (primary raster changed → %s).",
                               rid, item["file"])
                        return existing
            except Exception as ex:
                L.warning("seed_results: primary-file refresh failed for %s: %s", rid, ex)
            try:
                desired = _metrics_with_mask(item) or {}
                cur = existing.get("metrics") or {}
                if (cur != desired) or (existing.get("name") != item.get("name")):
                    with _store._Lock():  # type: ignore[attr-defined]
                        items = _store._load()  # type: ignore[attr-defined]
                        for e in items:
                            if e.get("id") == rid:
                                e["metrics"] = desired
                                e["name"] = item.get("name", e.get("name"))
                                if item.get("resolution"):
                                    e["resolution"] = item["resolution"]
                                if item.get("roi_bbox"):
                                    e["roi_bbox"] = item["roi_bbox"]
                        _store._save(items)  # type: ignore[attr-defined]
                    _backdate(rid, item.get("date"))
                    existing = _store.get_result(rid) or existing
                    L.info("seed_results: refreshed default %s metrics from manifest.", rid)
            except Exception as ex:
                L.warning("seed_results: default refresh failed for %s: %s", rid, ex)
            return existing
        # Non-default idempotent no-op, EXCEPT: backfill the optical_valid
        # mask_path into the metrics if the manifest now declares a mask the
        # stored entry predates.
        try:
            cur = existing.get("metrics") or {}
            if item.get("mask_file") and not cur.get("mask_path"):
                refreshed = _metrics_with_mask(item)
                if refreshed and refreshed.get("mask_path"):
                    with _store._Lock():  # type: ignore[attr-defined]
                        items = _store._load()  # type: ignore[attr-defined]
                        for e in items:
                            if e.get("id") == rid:
                                m = dict(e.get("metrics") or {})
                                m["mask_path"] = refreshed["mask_path"]
                                if refreshed.get("mask_classes"):
                                    m["mask_classes"] = refreshed["mask_classes"]
                                e["metrics"] = m
                        _store._save(items)  # type: ignore[attr-defined]
                    existing = _store.get_result(rid) or existing
        except Exception:
            L.warning("seed_results: mask_path backfill failed for %s", rid)
        return existing
    entry = _store.register_result(
        result_id=rid,
        name=item["name"],
        roi_bbox=item.get("roi_bbox") or [],
        resolution=item.get("resolution") or "10m",
        status="done",
        output_bytes=src.read_bytes(),
        output_filename=item["file"],
        metrics=_metrics_with_mask(item),
        job_id=None,
        error=None,
    )
    _backdate(rid, item.get("date"))
    return _store.get_result(rid) or entry


def seed_default_regions() -> int:
    """Register the 4 DEFAULT auto-run regions (manifest items with
    is_default=true) ALWAYS — not gated on an empty store — so they appear as
    frozen precomputed Results on every load (DEFAULT_REGIONS_LOG.md AC-1).

    Idempotent via fixed ids + `_seed_one`'s get_result guard. Returns the count
    of default entries present after seeding. Never overwrites real user results
    (different ids) and never recomputes the USER-VALIDATED metrics — they are
    copied verbatim from the manifest.
    """
    n = 0
    for item in _load_manifest():
        if not item.get("is_default"):
            continue
        try:
            if _seed_one(item) is not None:
                n += 1
        except Exception as ex:  # pragma: no cover
            L.warning("seed_results: failed to seed default %s: %s",
                      item.get("id"), ex)
    if n:
        L.info("seed_results: ensured %d default region result(s).", n)
    return n


def seed_always_products() -> int:
    """Register curated NON-default products marked ``always_seed=true`` (e.g. the
    certified DL-fusion marawah / bu_tinah products) on every load — like the
    default regions but WITHOUT the ``is_default`` flag, so they list under
    FINISHED JOBS rather than the Default-Regions section. Idempotent via fixed
    ids; metrics are refreshed authoritatively from the manifest. Returns the
    count seeded/ensured. Never touches real user results (different ids)."""
    n = 0
    for item in _load_manifest():
        if item.get("is_default") or not item.get("always_seed"):
            continue
        try:
            if _seed_one(item) is not None:
                n += 1
        except Exception as ex:  # pragma: no cover
            L.warning("seed_results: failed to seed always-product %s: %s",
                      item.get("id"), ex)
    if n:
        L.info("seed_results: ensured %d always-seed product(s).", n)
    return n


def seed_named_sample(seed_id: str) -> dict[str, Any] | None:
    """Register ONE bundled sample by its manifest id, regardless of whether the
    store already has other results. Idempotent. Used by the R2 'Khalifa Port
    (High-Resolution)' button to surface the pre-rendered VHR product instantly.
    Returns the stored result entry, or None if not found in the manifest."""
    for item in _load_manifest():
        if item.get("id") == seed_id:
            return _seed_one(item)
    return None


def _backdate(rid: str, date: Any) -> None:
    """Best-effort: overwrite the auto-stamped date with the manifest date so
    the seeded entry carries its honest original timestamp."""
    if not date:
        return
    try:
        with _store._Lock():  # type: ignore[attr-defined]
            items = _store._load()  # type: ignore[attr-defined]
            for e in items:
                if e.get("id") == rid:
                    e["date"] = float(date)
            _store._save(items)  # type: ignore[attr-defined]
    except Exception:
        pass
