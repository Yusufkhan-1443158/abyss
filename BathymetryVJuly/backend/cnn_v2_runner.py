"""
CNN_v2 runner — BP Neural Network pipeline after Guo et al. 2022
(Water 14(23), 3862). Mirrors the structure of `accurate_runner.py`
so it plugs into the same SSE streaming endpoint the UI already uses,
but swaps the stratified ridge / Patent-MLP stages for a single BP-NN
trained on Sentinel-2 reflectances + Stumpf ratios with ICESat-2 +
observed XYZ as the ground truth.

Goal: R² > 0.90, RMSE < 1.5 m on coastal reef-style ROIs (paper's
reported performance on the Changhua-Yongxing Islands).
"""

from __future__ import annotations

import logging, time, uuid, io, traceback
from pathlib import Path
import numpy as np

L = logging.getLogger("bathymetry.cnn_v2")

# Re-use the job registry + emit + DOWNLOAD_DIR from accurate_runner so
# SSE streaming works identically.
try:
    from backend.accurate_runner import (
        create_job, get_job, _emit, DOWNLOAD_DIR,
        _fetch_icesat2, _fetch_gebco,
    )
except ImportError:
    from accurate_runner import (
        create_job, get_job, _emit, DOWNLOAD_DIR,
        _fetch_icesat2, _fetch_gebco,
    )


def run_cnn_v2_job(job_id: str, bbox: tuple, region: str = "custom",
                    epochs: int = 500, res: int = 10, cloud: int = 25):
    """Synchronous runner executed in a background thread. Emits SSE
    events via `_emit` the same way the accurate pipeline does."""
    j = get_job(job_id)
    if j is None:
        return
    j["status"] = "running"

    # Bias-correction & calibration options from the sidebar payload
    bc = j.get("bc", {}) if j is not None else {}
    bc_mode        = bc.get("mode", "single")
    bc_window_days = int(bc.get("window_days", 180))
    bc_n_scenes    = int(bc.get("n_scenes", 4))
    bc_tide        = bool(bc.get("tide", True))
    bc_wave        = bool(bc.get("wave", True))
    bc_geoid       = bool(bc.get("geoid", True))
    bc_calib       = bc.get("calib", "shift+bias+local")
    bc_deep        = bool(bc.get("deep", True))

    try:
        _emit(job_id, "start", 1,
              f"CNN_v2 (Guo 2022 BP-NN) starting · region={j.get('region', region)} · "
              f"mode={bc_mode} · window={bc_window_days}d · "
              f"tide={bc_tide} wave={bc_wave} geoid={bc_geoid} · calib={bc_calib}")

        # ── References: GEBCO + ICESat-2 + observed XYZ ──
        _emit(job_id, "gebco", 6, "pulling GEBCO 2024 prior")
        gebco, gebco_err = _fetch_gebco(bbox)
        if gebco_err:
            _emit(job_id, "gebco", 10, f"GEBCO unavailable ({gebco_err})")
            gebco = []
        else:
            _emit(job_id, "gebco", 10, f"GEBCO: {len(gebco)} prior points")

        _emit(job_id, "icesat2", 12,
              "querying ICESat-2 ATL03 via SlideRule (45 s cap)")
        _hb = {"tick": 0}
        def _hbeat(msg):
            _hb["tick"] += 1
            pct = min(22.0, 12.0 + 1.1 * _hb["tick"])
            _emit(job_id, "icesat2", pct, msg)
        ice, ice_err = _fetch_icesat2(bbox, heartbeat=_hbeat, timeout_s=45)
        if ice_err:
            _emit(job_id, "icesat2", 23,
                  f"ICESat-2 unavailable ({ice_err}) — continuing")
            ice = []
        else:
            _emit(job_id, "icesat2", 23,
                  f"ICESat-2: {len(ice)} refraction-corrected depths (raw, pre-align)")

        # ── Tidal_Alignment_Module: retide each ICESat-2 segment from its own
        #    t1 to the Sentinel-2 composite overpass time t2. Obeys the bc_*
        #    toggles from the sidebar: bc_tide = apply Open-Meteo retide,
        #    bc_wave = apply +Hs/2 if waves at t2 > 0.5 m, bc_geoid = use
        #    EGM2008 for ellipsoidal→MSL (only meaningful if photon_heights
        #    are passed; for segment-level depth the geoid is a no-op).
        tide_align_info = None
        if ice and (bc_tide or bc_wave):
            try:
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                # S2 composite centre time: midpoint of the date window.
                t2 = _dt.combine(_dt.utcnow().date() - _td(days=int(bc_window_days/2)),
                                 _dt.min.time(), tzinfo=_tz.utc) + _td(hours=6, minutes=30)
                try:
                    from backend.tidal_alignment import tidal_alignment_pipeline
                except ImportError:
                    from tidal_alignment import tidal_alignment_pipeline
                aligned, tide_align_info = tidal_alignment_pipeline(
                    list(bbox), t2, ice, photon_heights=False,
                )
                # Only adopt the alignment if the user actually requested tide.
                # If bc_tide is False and bc_wave is True, we zero the tide
                # component post-hoc but keep the wave correction.
                if not bc_tide and tide_align_info is not None:
                    # Undo the (T2-T1) adjustment — keep only the wave term
                    wc = float(tide_align_info.get("wave_correction_m", 0.0))
                    for p in aligned:
                        p["depth_aligned"] = float(p["depth"] + wc)
                        p["delta_m"] = 0.0
                if not bc_wave and tide_align_info is not None:
                    wc = float(tide_align_info.get("wave_correction_m", 0.0))
                    if wc > 0:
                        for p in aligned:
                            p["depth_aligned"] = float(p["depth_aligned"] - wc)
                        tide_align_info["wave_correction_m"] = 0.0
                # Replace raw depths with (possibly adjusted) aligned ones
                ice = []
                for p in aligned:
                    d = float(p.get("depth_aligned", p.get("depth")))
                    if 0.2 < d <= 25.0:
                        ice.append({**p, "depth": d})
                _emit(job_id, "icesat2", 24,
                      f"Tidal_Alignment applied · n={tide_align_info.get('n_points')} · "
                      f"tide={'on' if bc_tide else 'off'} · wave={'on' if bc_wave else 'off'} · "
                      f"mean Δ={tide_align_info.get('delta_mean_m', 0):+.2f} m · "
                      f"Hs={tide_align_info.get('wave_Hs_t2_m', 0):.2f} m · "
                      f"wave corr={tide_align_info.get('wave_correction_m', 0):+.2f} m")
            except Exception as _tex:
                _emit(job_id, "icesat2", 24,
                      f"tidal alignment skipped ({_tex}) — using raw ICESat-2")
        elif ice:
            _emit(job_id, "icesat2", 24,
                  "Tidal/Wave alignment disabled by user — using raw ICESat-2")

        obs_payload = j.get("user_points") or []
        observed = []
        for p in obs_payload:
            try:
                d = float(p.get("depth", 0))
                pc = p.get("photon_class", "")
                if d <= 0 or d > 25:
                    continue
                if pc not in ("observed", "icesat2_cshelph", "bathymetry"):
                    continue
                observed.append({"lat": float(p["lat"]), "lon": float(p["lon"]),
                                  "depth": d, "source": pc})
            except Exception:
                continue

        # Benchmark/validate-only mode: observed XYZ is NOT fed to training;
        # it's held 100% out and used only for honest evaluation at the end.
        validate_only_mode = bool(j.get("validate_only_observed", False))
        if validate_only_mode and observed:
            _emit(job_id, "observed", 25,
                  f"VALIDATE-ONLY mode — {len(observed)} observed pts held out of training")
            training_observed = []
            validate_only_pts = observed
        else:
            training_observed = observed
            validate_only_pts = []
            if observed:
                _emit(job_id, "observed", 25, f"{len(observed)} observed pts merged into training")

        # ── Auto-fallback: if there's no observed survey data, prefer
        # inference with a pre-trained regional model over training on
        # GEBCO (too coarse at 450 m) + a handful of ICESat-2 photons.
        # GEBCO + sparse ICESat-2 training collapses the BP-NN to a mean
        # prediction with high RMSE and negative R². A pre-trained model
        # transferred from a nearby surveyed region is far more useful.
        # This mirrors how EOMAP / Fugro handle "new ROI, no survey".
        _need_fallback = (len(training_observed) == 0)
        if _need_fallback:
            try:
                try:
                    from backend import bp_inference as BI
                except ImportError:
                    import bp_inference as BI
                nm = BI.nearest_model(list(bbox), prefer_deep=True)
            except Exception as _ex:
                nm = None
            if nm is not None:
                weights_file = nm["weights_file"]
                mf = nm.get("manifest") or {}
                met = mf.get("metrics") or {}
                sel = nm.get("_selection") or {}
                _emit(job_id, "fallback", 40,
                      f"no observed + no ICESat-2 — switching to INFERENCE using "
                      f"pre-trained model {nm.get('filename')} "
                      f"(R²={met.get('r2')}, RMSE={met.get('rmse_m')} m, "
                      f"dist={sel.get('distance_deg')}°, "
                      f"contains_roi={sel.get('contains_roi')})")
                # Fast-forward: do the S2 fetch + inference + export only
                # We reuse the rest of the pipeline below (stitch / export) but
                # skip BP training entirely.
                j["_inference_override"] = {
                    "weights_file": weights_file,
                    "manifest": mf,
                    "selection": sel,
                    "model_filename": nm.get("filename"),
                }
            else:
                _emit(job_id, "warning", 40,
                      "no pre-trained model found for fallback — attempting "
                      "GEBCO-only training (expect near-constant output)")

        refs = training_observed + ice + gebco
        if len(refs) < 5:
            # Last-ditch GEBCO prior: if the ROI is pure offshore with no ICESat-2
            # and no observed, query GEBCO again with a wider buffer so we at
            # least have a coarse depth prior. This beats aborting with 500.
            try:
                gebco_wide, _ = _fetch_gebco([bbox[0]-0.05, bbox[1]-0.05,
                                                bbox[2]+0.05, bbox[3]+0.05])
                gebco = gebco + gebco_wide
                refs = training_observed + ice + gebco
                _emit(job_id, "gebco", 17,
                      f"widened GEBCO search by 5 km: now {len(gebco)} pts (was sparse)")
            except Exception:
                pass
        if len(refs) < 5:
            raise RuntimeError(
                f"only {len(refs)} reference points after widened fallback — "
                f"aborting (training_observed={len(training_observed)}, "
                f"icesat2={len(ice)}, gebco={len(gebco)}). This ROI has no "
                f"usable bathymetric priors; either upload observed XYZ or "
                f"choose a coastal ROI where ICESat-2 has coverage.")
        if len(refs) < 20:
            _emit(job_id, "warning", 26,
                  f"only {len(refs)} training refs "
                  f"(training_observed={len(training_observed)}, icesat2={len(ice)}, "
                  f"gebco={len(gebco)}) — predicting anyway but expect a smooth, "
                  f"GEBCO-dominated map. Upload observed XYZ for a proper result.")

        # ── Sentinel-2 — single composite OR multi-image stack ──
        import backend.app as A
        import datetime as dt
        ed = dt.date.today().isoformat()
        sd = (dt.date.today() - dt.timedelta(days=bc_window_days)).isoformat()

        if bc_mode == "multi" and bc_n_scenes >= 2:
            # Split the window into N equal chunks, fetch one scene per chunk,
            # build a per-scene mosaic by averaging features (keeps water mask
            # as intersection). Each scene has its own acquisition date, so
            # ICESat-2 photons can later be retided to the CLOSEST scene.
            from datetime import datetime as _dt, timedelta as _td
            _emit(job_id, "sentinel2", 26,
                  f"fetching {bc_n_scenes} S2 scenes across {bc_window_days}-day window @ {res} m (multi-image)")
            scenes = []
            scene_dates = []
            dt_sd = _dt.strptime(sd, "%Y-%m-%d")
            dt_ed = _dt.strptime(ed, "%Y-%m-%d")
            total_days = max((dt_ed - dt_sd).days, 14)
            chunk = max(7, total_days // bc_n_scenes)
            for i in range(bc_n_scenes):
                c_sd = dt_sd + _td(days=i * chunk)
                c_ed = min(c_sd + _td(days=chunk - 1), dt_ed)
                if c_sd >= dt_ed: break
                try:
                    sc = A._fetch_s2_single(list(bbox), c_sd.strftime("%Y-%m-%d"),
                                             c_ed.strftime("%Y-%m-%d"), res=res, cloud=cloud)
                    scenes.append(sc)
                    scene_dates.append(c_sd.strftime("%Y-%m-%d"))
                    _emit(job_id, "sentinel2", 26 + (10 * (i + 1)) // bc_n_scenes,
                          f"S2 scene {i+1}/{bc_n_scenes}: {c_sd.strftime('%Y-%m-%d')} "
                          f"({sc['width']}x{sc['height']})")
                except Exception as _ex:
                    _emit(job_id, "sentinel2", 26, f"scene {i+1} failed ({_ex})")
            if not scenes:
                raise RuntimeError(f"all {bc_n_scenes} S2 fetches failed over {sd}→{ed}")
            # Average features across scenes; combined_water = AND
            H, W = scenes[0]["red"].shape
            combined_water = scenes[0]["water_mask"].copy()
            for sc in scenes[1:]:
                combined_water &= sc["water_mask"]
            import numpy as _np
            avg_bands = {}
            for k in ("blue", "green", "red", "nir", "coastal", "swir", "ndwi"):
                if k not in scenes[0]: continue
                try:
                    avg_bands[k] = _np.mean(_np.stack([sc[k].astype(_np.float64)
                                                         for sc in scenes if k in sc]), axis=0)
                except Exception:
                    avg_bands[k] = scenes[0][k]
            s2 = {**scenes[0], **avg_bands, "water_mask": combined_water,
                   "n_scenes_used": len(scenes), "scene_dates": scene_dates}
            water = s2["water_mask"]
            _emit(job_id, "sentinel2", 38,
                  f"multi-image stack: {len(scenes)} scenes · "
                  f"dates {scene_dates[0]}…{scene_dates[-1]} · water {int(water.sum())*100//(H*W)}%")
        else:
            _emit(job_id, "sentinel2", 28,
                  f"fetching Sentinel-2 L2A median composite @ {res} m "
                  f"({bc_window_days}-day window)")
            s2 = A.fetch_s2(bbox, sd, ed, res=res, cloud=cloud)
            H, W = s2["red"].shape
            water = A.make_water_mask(list(bbox), H, W, s2=s2)
            s2["water_mask"] = water
            _emit(job_id, "sentinel2", 38,
                  f"S2 {W}×{H} @ {res} m · water {int(water.sum())*100//(H*W)}%")

        # ── BP NN training ──
        _emit(job_id, "train", 42,
              f"training BP neural network (Guo 2022: 6->32->16->1) on {len(refs)} refs"
              + (f" | validate-only: {len(validate_only_pts)} observed pts"
                 if validate_only_pts else ""))
        try:
            from backend import bp_network as BP
        except ImportError:
            import bp_network as BP
        ref_bp = {
            "lats":   np.array([p["lat"] for p in refs]),
            "lons":   np.array([p["lon"] for p in refs]),
            "depths": np.array([p["depth"] for p in refs]),
        }
        val_only_bp = None
        if validate_only_pts:
            val_only_bp = {
                "lats":   np.array([p["lat"] for p in validate_only_pts]),
                "lons":   np.array([p["lon"] for p in validate_only_pts]),
                "depths": np.array([p["depth"] for p in validate_only_pts]),
            }

        # Model-weight filename — persistent across runs so the user can reuse
        # / fine-tune it later.
        from pathlib import Path
        model_dir = Path(__file__).parent / "ocean" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        weights_path = model_dir / f"bp_network_{region}.pt"
        _emit(job_id, "train", 44,
              f"model weights file: backend/ocean/models/bp_network_{region}.pt"
              + (" (will warm-start from existing checkpoint)" if weights_path.exists() else " (new)"))

        # Live per-epoch progress — forwards every ~5 epochs to the SSE stream
        # so the frontend progress bar creeps from 44% to 78% during training.
        _last_emit = {"t": 0.0}
        def _epoch_cb(ep, total, tr_loss, v_loss):
            # Map epoch fraction to 44..78 % band
            pct = 44.0 + 34.0 * (ep + 1) / max(1, total)
            now = time.time()
            if ep == 0 or (ep + 1) % 5 == 0 or (ep + 1) == total:
                _emit(job_id, "train", pct,
                      f"BP-NN epoch {ep+1}/{total} · train_loss={tr_loss:.5f} · val_loss={v_loss:.5f}")
                _last_emit["t"] = now

        t0 = time.time()
        # If we decided earlier to fall back to a pre-trained model, skip
        # training and run pure inference with it.
        override = j.get("_inference_override") if j is not None else None
        if override:
            try:
                try:
                    from backend import bp_inference as BI
                except ImportError:
                    import bp_inference as BI
                net_pre, ck_pre = BI.load_model(override["weights_file"])
                depth, info_inf = BI.predict(net_pre, ck_pre, s2, list(bbox))
                bp_info = {
                    "method": f"INFERENCE (pre-trained): {override.get('model_filename')}",
                    "architecture": info_inf.get("architecture"),
                    "deep": info_inf.get("deep", False),
                    "n_train": override["manifest"].get("n_observed", 0),
                    "n_val":   override["manifest"].get("n_val_pixels", 0),
                    "epochs_used": override["manifest"].get("epochs_used", 0),
                    "epochs_max":  override["manifest"].get("epochs_used", 0),
                    "rmse_m":  override["manifest"].get("metrics", {}).get("rmse_m", 0.0),
                    "mae_m":   override["manifest"].get("metrics", {}).get("mae_m", 0.0),
                    "bias_m":  override["manifest"].get("metrics", {}).get("bias_m", 0.0),
                    "r2":      override["manifest"].get("metrics", {}).get("r2", 0.0),
                    "feature_names": info_inf.get("feature_names"),
                    "feature_min":   override["manifest"].get("feature_min"),
                    "feature_max":   override["manifest"].get("feature_max"),
                    "target_scale_m": info_inf.get("target_scale_m", 25.0),
                    "citation": "Pre-trained BP-NN, region-transferred",
                    "warm_started": True,
                    "saved_to": override["weights_file"],
                    "inference_only": True,
                    "selected_model": override,
                    "validate_only": None,
                }
                _emit(job_id, "train", 78,
                      f"INFERENCE done with {override.get('model_filename')} · "
                      f"R²={bp_info['r2']} · RMSE={bp_info['rmse_m']} m "
                      f"(metrics from manifest; ROI may differ)")
            except Exception as _ex:
                _emit(job_id, "error", 78, f"inference fallback failed: {_ex}")
                raise
        else:
            depth, bp_info = BP.bp_predict(
                s2, ref_bp, list(bbox),
                epochs=epochs, verbose=False,
                save_path=str(weights_path),
                load_path=str(weights_path) if weights_path.exists() else None,
                validate_only_refs=val_only_bp,
                progress_cb=_epoch_cb,
                deep=bc_deep,
            )
        if depth is None:
            raise RuntimeError(f"BP-NN training failed: {bp_info.get('error', 'unknown')}")
        dt_train = time.time() - t0
        # Preferred metrics for display: the honest held-out set if available.
        vo = bp_info.get("validate_only")
        if vo is not None:
            _emit(job_id, "train", 78,
                  f"BP-NN trained in {dt_train:.0f}s · HONEST held-out: "
                  f"R²={vo['r2']:.3f} · RMSE={vo['rmse_m']:.2f} m · "
                  f"MAE={vo['mae_m']:.2f} m · bias={vo['bias_m']:+.2f} m · "
                  f"n={vo['n_pairs']}")
        else:
            _emit(job_id, "train", 78,
                  f"BP-NN trained in {dt_train:.0f}s · internal val: "
                  f"R²={bp_info['r2']:.3f} · RMSE={bp_info['rmse_m']:.2f} m · "
                  f"MAE={bp_info['mae_m']:.2f} m")

        # ── Post-prediction calibration (Step-3 layer) ──
        # Professional bias cleanup: pixel-shift co-registration + global
        # mean-residual + IDW local residual field. Uses only TRAINING
        # observations (not held-out) to avoid data leakage.
        calibration_info = None
        if bc_calib and bc_calib != "none" and training_observed:
            try:
                try:
                    from backend.depth_calibration import calibrate_to_observed
                except ImportError:
                    from depth_calibration import calibrate_to_observed
                cal_lats = np.array([p["lat"] for p in training_observed])
                cal_lons = np.array([p["lon"] for p in training_observed])
                cal_deps = np.array([p["depth"] for p in training_observed])
                depth, calibration_info = calibrate_to_observed(
                    depth, list(bbox),
                    obs_lats=cal_lats, obs_lons=cal_lons, obs_depths=cal_deps,
                    steps=bc_calib, max_shift_px=5,
                    idw_power=2.0, clamp_local_m=2.0, verbose=False,
                )
                post = calibration_info.get("post_calibration", {})
                pre  = calibration_info.get("pre_calibration", {})
                sh   = calibration_info.get("shift", {})
                _emit(job_id, "calibration", 88,
                      f"post-calib ({bc_calib}) · shift=({sh.get('dr',0):+d},{sh.get('dc',0):+d}) px · "
                      f"RMSE {pre.get('rmse_m','?')}→{post.get('rmse_m','?')} m · "
                      f"MAE {pre.get('mae_m','?')}→{post.get('mae_m','?')} m · "
                      f"|bias| {abs(pre.get('bias_m',0)):.2f}→{abs(post.get('bias_m',0)):.2f} m")
            except Exception as _cex:
                _emit(job_id, "calibration", 88, f"post-calibration skipped: {_cex}")
        else:
            _emit(job_id, "calibration", 88,
                  f"post-calibration {'disabled' if bc_calib=='none' else 'skipped (no training observed)'}")

        # Uncertainty band: 1σ ≈ held-out RMSE, scaled mildly with depth
        unc = np.full_like(depth, float(bp_info["rmse_m"]), dtype=np.float32)

        _emit(job_id, "stitch", 90, "applying water mask")
        depth = np.where(water & np.isfinite(depth), depth, np.nan)

        # ── Write GeoTIFF + PNG ──
        _emit(job_id, "export", 94, "writing GeoTIFF + PNG")
        try:
            import rasterio
            from rasterio.transform import from_bounds
            w_, s_, e_, n_ = bbox
            transform = from_bounds(w_, s_, e_, n_, W, H)
            depth_tif = DOWNLOAD_DIR / f"cnn_v2_{region}_{job_id}.tif"
            unc_tif   = DOWNLOAD_DIR / f"cnn_v2_{region}_{job_id}_uncertainty.tif"
            for p, arr in [(depth_tif, depth), (unc_tif, unc)]:
                data = np.where(np.isfinite(arr), arr, -9999).astype(np.float32)
                with rasterio.open(
                    p, "w", driver="GTiff", height=H, width=W, count=1,
                    dtype="float32", crs="EPSG:4326", transform=transform,
                    nodata=-9999, compress="deflate", tiled=True,
                ) as ds:
                    ds.write(data, 1)
        except Exception as ex:
            _emit(job_id, "export", 94, f"geotiff failed: {ex}")
            depth_tif = unc_tif = None

        png_path = DOWNLOAD_DIR / f"cnn_v2_{region}_{job_id}.png"
        try:
            from PIL import Image
            finite = np.isfinite(depth) & (depth > 0)
            if finite.any():
                vmax = float(np.nanpercentile(depth[finite], 98))
                t = np.clip(depth / max(vmax, 1), 0, 1)
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
                rgb[..., 0][finite] = np.clip(240 - 200 * t[finite], 0, 255).astype(np.uint8)
                rgb[..., 1][finite] = np.clip(249 - 190 * t[finite], 0, 255).astype(np.uint8)
                rgb[..., 2][finite] = np.clip(255 - 220 * t[finite], 0, 255).astype(np.uint8)
                Image.fromarray(rgb).save(png_path)
        except Exception as ex:
            _emit(job_id, "export", 95, f"png failed: {ex}")

        # ── Final event ──
        metrics = {
            "r2": bp_info["r2"],
            "rmse_m": bp_info["rmse_m"],
            "mae_m":  bp_info["mae_m"],
            "bias_m": bp_info["bias_m"],
            "depth_min_m": float(np.nanmin(depth)) if np.isfinite(depth).any() else None,
            "depth_max_m": float(np.nanmax(depth)) if np.isfinite(depth).any() else None,
            "depth_mean_m": float(np.nanmean(depth)) if np.isfinite(depth).any() else None,
            "n_water_px": int(water.sum()),
            "depth_scale_m": bp_info["target_scale_m"],
            "n_train": bp_info["n_train"],
            "n_val":   bp_info["n_val"],
            "epochs_used": bp_info["epochs_used"],
        }
        sources = {
            "n_iboating": 0,
            "n_observed_total":   len(observed),
            "n_observed_trained": len(training_observed),
            "n_observed_heldout": len(validate_only_pts),
            "n_icesat2":  len(ice),
            "n_gebco":    len(gebco),
            "n_total_refs": len(refs),
            "validate_only_mode": validate_only_mode,
        }
        j["result"] = {
            "architecture": bp_info["architecture"],
            "citation": bp_info["citation"],
            "method": bp_info["method"],
            "sources": sources,
            "metrics": metrics,
            "bp_info": bp_info,
            "validate_only": bp_info.get("validate_only"),
            "calibration":   calibration_info,
            "tidal_alignment": tide_align_info,
            "bias_correction_settings": {
                "mode": bc_mode, "window_days": bc_window_days,
                "n_scenes": bc_n_scenes, "tide": bc_tide,
                "wave": bc_wave, "geoid": bc_geoid, "calib": bc_calib,
                "scenes_dates": s2.get("scene_dates") if isinstance(s2, dict) else None,
                "inference_fallback_used":  bp_info.get("inference_only", False),
                "inference_fallback_model": (bp_info.get("selected_model") or {}).get("model_filename"),
            },
            "weights_file": {
                "path": str(weights_path),
                "relative_path": f"backend/ocean/models/bp_network_{region}.pt",
                "filename": f"bp_network_{region}.pt",
                "warm_started": bp_info.get("warm_started", False),
                "note": "Reused on next run for the same region — delete the file to re-train from scratch.",
            },
            "downloads": {
                "geotiff":     f"/downloads/{depth_tif.name}" if depth_tif else None,
                "uncertainty": f"/downloads/{unc_tif.name}"   if unc_tif   else None,
                "preview":     f"/downloads/{png_path.name}"  if png_path.exists() else None,
            },
            "job_id": job_id,
        }
        j["status"] = "done"
        _emit(job_id, "done", 100, "complete", extra={"result": j["result"]})

    except Exception as ex:
        j["status"] = "error"
        j["error"] = str(ex)
        _emit(job_id, "error", 0, str(ex),
              extra={"traceback": traceback.format_exc()})
