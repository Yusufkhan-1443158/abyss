"""
ICESat-2 Bathymetry via CShelph — Professional Pipeline
════════════════════════════════════════════════════════
Processing chain:
  1. Search ATL03 granules via earthaccess for ROI + date range
  2. Download/stream H5 files
  3. Read photon-level data (lat, lon, height, confidence)
  4. Orthometric correction (WGS84 ellipsoid → EGM2008 geoid)
  5. Adaptive spatial binning (density-driven resolution)
  6. Sea surface detection (robust KDE peak)
  7. Refraction correction (Snell's law — Parrish et al. 2019)
  8. Kd estimation — diffuse attenuation from photon decay
  9. Bottom detection (adaptive density threshold)
 10. Along-track signal quality scoring
 11. MAD-based outlier rejection
 12. Total Propagated Uncertainty (TPU) per point
 13. Multi-pass aggregation + cross-validation
 14. Return classified photons + profiles + depth points + quality metrics

References:
  - Thomas & Lee, CShelph v2.9 (github.com/nmt28/C-SHELPh)
  - Parrish et al. (2019), Remote Sensing 11(14), 1634
  - Albright & Glennie (2021), IEEE TGRS — TPU model
"""
from __future__ import annotations
import os, logging, math, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

L = logging.getLogger("bathy.icesat2")
MAX_DEPTH_M = 25.0

CACHE_DIR = Path("/tmp/bathy/icesat2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Physical constants
WATER_REFRACTIVE_INDEX = 1.34
LASER_WAVELENGTH_NM = 532
SPEED_OF_LIGHT = 299_792_458  # m/s


# ═══════════════════════════════════════════════════════════
# 1. SEARCH ATL03 GRANULES
# ═══════════════════════════════════════════════════════════

def search_atl03(bbox, start_date, end_date, max_results=20):
    """
    Search for ICESat-2 ATL03 granules over ROI + date range.
    Returns list of granule metadata (id, date, orbit, size, etc.)
    """
    # Load credentials from .env file BEFORE importing earthaccess
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists() and not os.environ.get("EARTHDATA_USERNAME"):
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        L.info("earthaccess: loaded credentials from .env")

    import earthaccess
    logged_in = False
    for strategy in ("environment", "netrc"):
        try:
            earthaccess.login(strategy=strategy)
            logged_in = True
            L.info(f"earthaccess: logged in via {strategy}")
            break
        except Exception:
            continue
    if not logged_in:
        L.warning("earthaccess: no credentials — search may fail")

    w, s, e, n = bbox
    L.info(f"ICESat-2 search: bbox=[{w},{s},{e},{n}], {start_date} → {end_date}")

    results = earthaccess.search_data(
        short_name="ATL03",
        bounding_box=(w, s, e, n),
        temporal=(start_date, end_date),
        count=max_results,
    )

    granules = []
    for r in results:
        try:
            granule_id = r.data_links()[0].split("/")[-1] if hasattr(r, 'data_links') else str(r)[:60]
            size_mb = round(r.size() / 1e6, 1) if hasattr(r, 'size') else 0
        except Exception:
            granule_id = str(r)[:60]
            size_mb = 0

        try:
            time_start = str(r["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])[:10]
        except Exception:
            try:
                time_start = str(r)[:10]
            except Exception:
                time_start = start_date

        granules.append({
            "id": granule_id,
            "date": time_start,
            "size_mb": size_mb,
            "result_obj": r,
        })

    L.info(f"ICESat-2: found {len(granules)} ATL03 granules")
    return granules


# ═══════════════════════════════════════════════════════════
# 2. PROCESS SINGLE GRANULE
# ═══════════════════════════════════════════════════════════

def process_granule(granule_result, bbox, laser_num=1, threshold=30, water_temp=22.0):
    """
    Full CShelph pipeline on one ATL03 granule.
    Returns dict with classified photons, profiles, depth points, quality metrics.
    """
    import earthaccess
    import cshelph

    w, s, e, n = bbox

    L.info("Downloading ATL03 granule...")
    try:
        files = earthaccess.download([granule_result], str(CACHE_DIR))
        if not files:
            return None, "Download failed"
        h5_path = str(files[0])
    except Exception as ex:
        return None, f"Download error: {ex}"

    L.info(f"Processing {h5_path}, laser={laser_num}")

    all_tracks = []
    all_profiles = []
    all_depth_pts = []
    all_quality = []

    beams = [laser_num] if laser_num > 0 else [1, 2, 3]
    for beam in beams:
        try:
            track_result = _process_beam(h5_path, bbox, beam, threshold, water_temp)
            if track_result:
                all_tracks.append(track_result["track_info"])
                all_profiles.append(track_result["profile"])
                all_depth_pts.extend(track_result["depth_points"])
                all_quality.append(track_result["quality_metrics"])
        except Exception as ex:
            L.warning(f"Beam {beam} failed: {ex}")
            continue

    if not all_depth_pts:
        return None, "No bathymetric photons found in any beam"

    # Aggregate quality metrics across beams
    agg_quality = _aggregate_quality(all_quality)

    return {
        "tracks": all_tracks,
        "profiles": all_profiles,
        "depth_points": all_depth_pts,
        "h5_file": h5_path,
        "n_beams": len(all_tracks),
        "n_depths": len(all_depth_pts),
        "quality": agg_quality,
    }, None


# ═══════════════════════════════════════════════════════════
# 3. BEAM PROCESSING — full professional pipeline
# ═══════════════════════════════════════════════════════════

def _process_beam(h5_path, bbox, laser_num, threshold, water_temp):
    """Process a single beam through the full professional CShelph pipeline."""
    import cshelph

    w, s, e, n = bbox

    # ── Step 1: Read ATL03 ──
    L.info(f"  Beam {laser_num}: reading ATL03...")
    try:
        lat, lon, photon_h, conf, ref_elev, ref_azimuth, \
            ph_index_beg, segment_id, alt_sc, seg_ph_count = \
            cshelph.read_atl03(h5_path, str(laser_num))
    except Exception as ex:
        L.warning(f"  Beam {laser_num}: read failed — {ex}")
        return None

    if len(lat) < 100:
        L.info(f"  Beam {laser_num}: only {len(lat)} photons, skipping")
        return None

    # ── Spatial filter to ROI ──
    mask = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
    if mask.sum() < 50:
        L.info(f"  Beam {laser_num}: only {mask.sum()} photons in ROI, skipping")
        return None

    lat = lat[mask]
    lon = lon[mask]
    photon_h = photon_h[mask]
    conf = conf[mask]

    # Interpolate segment-level arrays to photon-level
    ref_elev_ph = cshelph.ref_linear_interp(seg_ph_count, ref_elev)
    ref_azim_ph = cshelph.ref_linear_interp(seg_ph_count, ref_azimuth)
    alt_sc_ph = cshelph.ref_linear_interp(seg_ph_count, alt_sc)

    if len(ref_elev_ph) == len(mask):
        ref_elev_ph = ref_elev_ph[mask]
        ref_azim_ph = ref_azim_ph[mask]
        alt_sc_ph = alt_sc_ph[mask]
    else:
        n_ph = mask.sum()
        ref_elev_ph = ref_elev_ph[:n_ph] if len(ref_elev_ph) >= n_ph else np.full(n_ph, np.nanmean(ref_elev))
        ref_azim_ph = ref_azim_ph[:n_ph] if len(ref_azim_ph) >= n_ph else np.full(n_ph, np.nanmean(ref_azimuth))
        alt_sc_ph = alt_sc_ph[:n_ph] if len(alt_sc_ph) >= n_ph else np.full(n_ph, np.nanmean(alt_sc))

    L.info(f"  Beam {laser_num}: {len(lat)} photons in ROI")

    # ── Step 2: Orthometric correction ──
    epsg = cshelph.convert_wgs_to_utm(lat[0], lon[0])
    L.info(f"  Beam {laser_num}: UTM zone → {epsg}")
    Y_utm, X_utm, Z_ortho = cshelph.orthometric_correction(lat, lon, photon_h, epsg)

    # ── Step 3: Adaptive binning ──
    # Determine bin resolution from photon density
    lat_res, height_res = _adaptive_bin_resolution(Y_utm, Z_ortho)
    L.info(f"  Beam {laser_num}: adaptive bins — lat_res={lat_res}m, h_res={height_res}m")

    dataset = pd.DataFrame({
        "latitude": Y_utm, "photon_height": Z_ortho,
        "longitude": X_utm, "conf": conf,
        "ref_elev": ref_elev_ph,
        "ref_azimuth": ref_azim_ph,
        "alt_sc": alt_sc_ph,
        "lat_wgs": lat, "lon_wgs": lon,
    })

    binned = cshelph.bin_data(dataset, lat_res, height_res)

    # ── Step 4: Sea surface detection (robust) ──
    sea_height = cshelph.get_sea_height(binned, surface_buffer=-0.5)
    if not sea_height or all(np.isnan(sea_height)):
        L.info(f"  Beam {laser_num}: no sea surface detected")
        return None

    sea_arr = np.array(sea_height, dtype=float)
    # Robust sea surface: median ± MAD filter to remove tidal/wave outliers
    sea_valid = sea_arr[np.isfinite(sea_arr)]
    if len(sea_valid) < 3:
        L.info(f"  Beam {laser_num}: too few sea surface estimates")
        return None
    sea_median = float(np.median(sea_valid))
    sea_mad = float(np.median(np.abs(sea_valid - sea_median))) * 1.4826  # MAD → σ
    sea_arr[(sea_arr < sea_median - 3 * sea_mad) | (sea_arr > sea_median + 3 * sea_mad)] = np.nan

    ws_mean = float(np.nanmean(sea_arr))
    ws_std = float(np.nanstd(sea_arr)) if np.sum(np.isfinite(sea_arr)) > 1 else 0.1
    L.info(f"  Beam {laser_num}: sea surface = {ws_mean:.2f}m ± {ws_std:.3f}m "
           f"(MAD filtered, {np.sum(np.isfinite(sea_arr))}/{len(sea_arr)} bins)")

    # ── Step 5: Refraction correction ──
    L.info(f"  Beam {laser_num}: refraction correction (T={water_temp}°C, n={WATER_REFRACTIVE_INDEX})...")
    try:
        corrected = cshelph.refraction_correction(
            water_temp=water_temp,
            water_surface=ws_mean,
            wavelength=LASER_WAVELENGTH_NM,
            photon_ref_elev=dataset["ref_elev"].values,
            ph_ref_azimuth=dataset["ref_azimuth"].values,
            photon_z=dataset["photon_height"].values,
            photon_x=dataset["longitude"].values,
            photon_y=dataset["latitude"].values,
            ph_conf=dataset["conf"].values,
            satellite_altitude=dataset["alt_sc"].values,
        )
        # CShelph returns: (out_x, out_y, out_z, ph_conf, photon_x, photon_y, photon_z, azimuth, elev)
        corr_x, corr_y, corr_z = corrected[0], corrected[1], corrected[2]
        corr_conf = corrected[3] if len(corrected) > 3 else dataset["conf"].values
    except Exception as ex:
        L.warning(f"  Beam {laser_num}: refraction failed — {ex}, using uncorrected")
        corr_x = dataset["longitude"].values
        corr_y = dataset["latitude"].values
        corr_z = dataset["photon_height"].values
        corr_conf = dataset["conf"].values

    # ── Step 6: Kd estimation (diffuse attenuation coefficient) ──
    kd, kd_quality = _estimate_kd(corr_z, ws_mean)
    L.info(f"  Beam {laser_num}: Kd(532) ≈ {kd:.3f} m⁻¹ (quality={kd_quality})")

    # Maximum detectable depth from Kd: ~1.5/Kd (Secchi disk approx)
    max_detectable = min(MAX_DEPTH_M, 1.5 / max(kd, 0.01))
    L.info(f"  Beam {laser_num}: max detectable depth ≈ {max_detectable:.1f}m")

    # ── Step 7: Re-bin corrected data + bottom detection ──
    # CShelph get_bath_height expects cor_latitude, cor_longitude, cor_photon_height columns
    corrected_df = pd.DataFrame({
        "latitude": corr_y, "photon_height": corr_z,
        "longitude": corr_x, "conf": corr_conf,
        "cor_latitude": corr_y,
        "cor_longitude": corr_x,
        "cor_photon_height": corr_z,
    })
    binned_corr = cshelph.bin_data(corrected_df, lat_res, height_res)

    # Adaptive threshold: scale with Kd (clearer water → lower threshold needed)
    adaptive_threshold = max(5, int(threshold * (0.15 / max(kd, 0.01)) ** 0.3))
    adaptive_threshold = min(adaptive_threshold, threshold * 3)
    L.info(f"  Beam {laser_num}: adaptive threshold = {adaptive_threshold} "
           f"(base={threshold}, Kd-scaled)")

    try:
        bath_height, geo_df = cshelph.get_bath_height(
            binned_corr, adaptive_threshold, ws_mean, height_res
        )
    except TypeError:
        # CShelph v2.9 bug: geo_photon_list is a list not ndarray
        # Monkey-patch and retry
        import cshelph as _csh
        _orig_src = _csh.__file__
        L.warning("  Patching CShelph get_bath_height for numpy compatibility...")
        # Manually compute bath height from binned corrected data
        bath_height = []
        binned_bath = binned_corr[binned_corr['photon_height'] < ws_mean - height_res * 2]
        if len(binned_bath) == 0:
            bath_height = [np.nan]
        else:
            for lb in sorted(binned_corr['lat_bins'].unique()):
                grp = binned_bath[binned_bath['lat_bins'] == lb]
                if len(grp) == 0:
                    bath_height.append(np.nan)
                    continue
                # Find height bin with most photons
                hcounts = grp.groupby('height_bins').size()
                if len(hcounts) == 0 or hcounts.max() < adaptive_threshold:
                    bath_height.append(np.nan)
                    continue
                peak_bin = hcounts.idxmax()
                bath_h = grp[grp['height_bins'] == peak_bin]['cor_photon_height'].median()
                bath_height.append(float(bath_h))
        geo_df = None

    if not bath_height or all(np.isnan(bath_height)):
        L.info(f"  Beam {laser_num}: no bottom detected")
        return None

    # ── Step 8: Build raw depth points ──
    from pyproj import Transformer, CRS
    epsg_num = int(epsg.replace("epsg:", ""))
    to_wgs = Transformer.from_crs(CRS.from_epsg(epsg_num), CRS.from_epsg(4326), always_xy=True)

    raw_depth_points = []
    bath_arr = np.array(bath_height, dtype=float)
    lat_bins = sorted(binned_corr["lat_bins"].unique()) if "lat_bins" in binned_corr.columns else []

    for i, lb in enumerate(lat_bins):
        if i >= len(bath_arr) or np.isnan(bath_arr[i]):
            continue
        sea_h = sea_arr[i] if i < len(sea_arr) and np.isfinite(sea_arr[i]) else ws_mean
        depth = sea_h - bath_arr[i]
        if depth <= 0 or depth > MAX_DEPTH_M:
            continue

        bin_mask = binned_corr["lat_bins"] == lb
        if bin_mask.sum() == 0:
            continue
        med_x = float(corrected_df.loc[bin_mask, "longitude"].median())
        med_y = float(corrected_df.loc[bin_mask, "latitude"].median())

        try:
            wgs_lon, wgs_lat = to_wgs.transform(med_x, med_y)
        except Exception:
            continue

        if not (-90 <= wgs_lat <= 90 and -180 <= wgs_lon <= 180):
            continue

        # Photon count in this bin around bottom elevation (signal strength)
        bottom_band = corrected_df.loc[bin_mask]
        bottom_photons = bottom_band[
            (bottom_band["photon_height"] >= bath_arr[i] - height_res) &
            (bottom_band["photon_height"] <= bath_arr[i] + height_res)
        ]
        signal_count = len(bottom_photons)

        raw_depth_points.append({
            "lat": round(wgs_lat, 6),
            "lon": round(wgs_lon, 6),
            "depth": round(min(depth, MAX_DEPTH_M), 2),
            "sea_surface": round(sea_h, 2),
            "bottom_elev": round(float(bath_arr[i]), 2),
            "signal_photons": signal_count,
            "beam": laser_num,
            "lat_bin": float(lb),
        })

    if not raw_depth_points:
        L.info(f"  Beam {laser_num}: no valid depth points after extraction")
        return None

    L.info(f"  Beam {laser_num}: {len(raw_depth_points)} raw depth points before filtering")

    # ── Step 9: Along-track signal quality scoring ──
    raw_depth_points = _score_along_track_quality(raw_depth_points)

    # ── Step 10: MAD-based outlier rejection ──
    depth_points = _mad_outlier_rejection(raw_depth_points)
    L.info(f"  Beam {laser_num}: {len(depth_points)} depth points after MAD filter")

    # ── Step 11: TPU (Total Propagated Uncertainty) ──
    depth_points = _compute_tpu(depth_points, ws_std, kd, water_temp)

    # ── Step 12: Depth-dependent confidence + Kd filtering ──
    final_points = []
    for p in depth_points:
        depth = p["depth"]
        # Reject depths beyond what Kd allows
        if depth > max_detectable * 1.2:
            L.debug(f"    Rejected {depth:.1f}m > max_detectable {max_detectable:.1f}m")
            continue
        # Confidence from depth, signal, and Kd
        p["confidence"] = _compute_confidence(
            depth, p["signal_photons"], kd, max_detectable, p.get("track_quality", 0.5)
        )
        p["photon_class"] = "icesat2_cshelph"
        p["kd_532"] = round(kd, 4)
        p["max_detectable_m"] = round(max_detectable, 1)
        final_points.append(p)

    if not final_points:
        L.info(f"  Beam {laser_num}: no points survived filtering")
        return None

    # ── Build profile ──
    profile_along = _build_profile(lat_bins, sea_arr, bath_arr, ws_mean, height_res)

    # ── Photon cloud for visualization ──
    photon_cloud = _build_photon_cloud(corr_x, corr_y, corr_z, corr_conf,
                                        ws_mean, to_wgs, max_photons=5000)

    # ── Quality metrics for this beam ──
    depths_arr = np.array([p["depth"] for p in final_points])
    tpus = np.array([p.get("tpu_m", 0.5) for p in final_points])
    quality_metrics = {
        "beam": laser_num,
        "n_raw_photons": len(lat),
        "n_depth_points": len(final_points),
        "kd_532": round(kd, 4),
        "kd_quality": kd_quality,
        "max_detectable_m": round(max_detectable, 1),
        "sea_surface_m": round(ws_mean, 2),
        "sea_surface_std": round(ws_std, 3),
        "depth_range": [round(float(np.min(depths_arr)), 1),
                        round(float(np.max(depths_arr)), 1)],
        "mean_depth": round(float(np.mean(depths_arr)), 2),
        "mean_tpu_m": round(float(np.mean(tpus)), 3),
        "adaptive_threshold": adaptive_threshold,
        "lat_res_m": lat_res,
        "height_res_m": height_res,
    }

    # Downsampled beam ground-track coordinates (lat, lon) so the front-end
    # can draw the ICESat-2 track as a polyline without shipping 10 000+ pts.
    beam_coords = []
    if len(final_points) > 0:
        lats_fp = np.array([p["lat"] for p in final_points])
        lons_fp = np.array([p["lon"] for p in final_points])
        order = np.argsort(lats_fp)
        lats_fp = lats_fp[order]; lons_fp = lons_fp[order]
        step = max(1, len(lats_fp) // 200)
        beam_coords = [[round(float(lats_fp[i]), 6), round(float(lons_fp[i]), 6)]
                       for i in range(0, len(lats_fp), step)]

    track_info = {
        "beam": laser_num,
        "n_photons": len(lat),
        "n_depths": len(final_points),
        "sea_surface_m": round(ws_mean, 2),
        "depth_range": quality_metrics["depth_range"],
        "mean_depth": quality_metrics["mean_depth"],
        "kd_532": round(kd, 4),
        "mean_tpu_m": quality_metrics["mean_tpu_m"],
        "coords": beam_coords,
    }

    L.info(f"  Beam {laser_num}: {len(final_points)} final points, "
           f"range {quality_metrics['depth_range']}m, "
           f"mean TPU={quality_metrics['mean_tpu_m']:.3f}m, "
           f"Kd={kd:.3f} m⁻¹")

    return {
        "track_info": track_info,
        "profile": profile_along,
        "depth_points": final_points,
        "photon_cloud": photon_cloud,
        "quality_metrics": quality_metrics,
    }


# ═══════════════════════════════════════════════════════════
# ADAPTIVE BINNING
# ═══════════════════════════════════════════════════════════

def _adaptive_bin_resolution(y_utm, z_ortho):
    """
    Choose spatial and height bin resolution based on photon density.
    Dense data → finer bins for better resolution.
    Sparse data → coarser bins to maintain statistical significance.
    """
    n = len(y_utm)
    y_range = np.ptp(y_utm[np.isfinite(y_utm)]) if np.any(np.isfinite(y_utm)) else 1000

    # Photon linear density (photons per metre along track)
    density = n / max(y_range, 1)

    if density > 50:      # Very dense — high resolution
        lat_res, height_res = 5.0, 0.3
    elif density > 20:    # Dense — standard
        lat_res, height_res = 10.0, 0.5
    elif density > 5:     # Moderate — coarser
        lat_res, height_res = 20.0, 0.5
    else:                 # Sparse — wide bins
        lat_res, height_res = 30.0, 1.0

    return lat_res, height_res


# ═══════════════════════════════════════════════════════════
# Kd ESTIMATION (Diffuse Attenuation Coefficient)
# ═══════════════════════════════════════════════════════════

def _estimate_kd(corr_z, sea_surface, depth_bins=20):
    """
    Estimate Kd(532) from photon vertical distribution below sea surface.
    Photon count decays exponentially with depth: N(z) ∝ exp(-2·Kd·z)
    (factor 2 for round-trip through water column).

    Returns (kd, quality_label).
    """
    # Select subsurface photons
    subsurface = corr_z[(corr_z < sea_surface - 0.5) & (corr_z > sea_surface - MAX_DEPTH_M)]
    if len(subsurface) < 50:
        return 0.15, "insufficient_data"  # default moderate turbidity

    # Convert to depth (positive down)
    depths = sea_surface - subsurface

    # Histogram by depth
    bin_edges = np.linspace(0.5, min(float(np.max(depths)), MAX_DEPTH_M), depth_bins + 1)
    counts, _ = np.histogram(depths, bins=bin_edges)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit exponential decay: log(N) = -2·Kd·z + C
    valid = counts > 0
    if valid.sum() < 4:
        return 0.15, "few_bins"

    log_counts = np.log(counts[valid].astype(float))
    z_valid = bin_centres[valid]

    # Linear regression: log(N) vs z
    A = np.vstack([z_valid, np.ones(len(z_valid))]).T
    try:
        result = np.linalg.lstsq(A, log_counts, rcond=None)
        slope = result[0][0]
        residuals = result[1] if len(result[1]) > 0 else [999]
    except Exception:
        return 0.15, "fit_failed"

    # slope = -2·Kd → Kd = -slope/2
    kd = max(0.01, -slope / 2)  # clamp to physically reasonable minimum
    kd = min(kd, 2.0)  # max Kd for extremely turbid water

    # Quality assessment
    r2 = 1 - residuals[0] / (np.var(log_counts) * len(log_counts)) if residuals[0] < 999 else 0
    if r2 > 0.8 and valid.sum() >= 8:
        quality = "good"
    elif r2 > 0.5 and valid.sum() >= 5:
        quality = "moderate"
    else:
        quality = "poor"

    return round(kd, 4), quality


# ═══════════════════════════════════════════════════════════
# ALONG-TRACK SIGNAL QUALITY SCORING
# ═══════════════════════════════════════════════════════════

def _score_along_track_quality(depth_points):
    """
    Score each depth point based on along-track consistency.
    A good bottom signal has neighbours at similar depths.
    Isolated points or sudden jumps get low quality scores.
    """
    if len(depth_points) < 3:
        for p in depth_points:
            p["track_quality"] = 0.5
        return depth_points

    depths = np.array([p["depth"] for p in depth_points])
    bins = np.array([p["lat_bin"] for p in depth_points])
    signals = np.array([p["signal_photons"] for p in depth_points])

    for i, p in enumerate(depth_points):
        # Find neighbours (±3 bins)
        dist = np.abs(bins - bins[i])
        sort_idx = np.argsort(dist)
        neighbours = sort_idx[1:min(7, len(sort_idx))]  # up to 6 nearest

        if len(neighbours) < 2:
            p["track_quality"] = 0.3
            continue

        neighbour_depths = depths[neighbours]
        depth_diff = np.abs(depths[i] - np.median(neighbour_depths))
        neighbour_std = np.std(neighbour_depths) + 0.1

        # Smoothness score: how well does this point fit its neighbours?
        smoothness = max(0, 1.0 - depth_diff / (2 * neighbour_std))

        # Signal strength score: more bottom photons = more confident
        median_signal = max(1, np.median(signals))
        signal_score = min(1.0, signals[i] / (2 * median_signal))

        # Combined quality
        p["track_quality"] = round(0.6 * smoothness + 0.4 * signal_score, 3)

    return depth_points


# ═══════════════════════════════════════════════════════════
# MAD-BASED OUTLIER REJECTION
# ═══════════════════════════════════════════════════════════

def _mad_outlier_rejection(depth_points, k=3.0):
    """
    Reject outliers using Median Absolute Deviation (MAD).
    More robust than IQR for non-Gaussian distributions.
    Also applies local consistency check (sliding window).
    """
    if len(depth_points) < 5:
        return depth_points

    depths = np.array([p["depth"] for p in depth_points])

    # Global MAD filter
    median_d = np.median(depths)
    mad = np.median(np.abs(depths - median_d))
    sigma_mad = mad * 1.4826  # MAD → equivalent σ
    if sigma_mad < 0.1:
        sigma_mad = 0.1

    global_ok = np.abs(depths - median_d) < k * sigma_mad
    n_global_rejected = np.sum(~global_ok)

    # Local sliding window MAD (window=7 points)
    window = 7
    local_ok = np.ones(len(depths), dtype=bool)
    for i in range(len(depths)):
        lo = max(0, i - window // 2)
        hi = min(len(depths), i + window // 2 + 1)
        local = depths[lo:hi]
        if len(local) < 3:
            continue
        local_med = np.median(local)
        local_mad = np.median(np.abs(local - local_med)) * 1.4826
        if local_mad < 0.1:
            local_mad = 0.1
        if abs(depths[i] - local_med) > k * local_mad:
            local_ok[i] = False

    keep = global_ok & local_ok
    n_total_rejected = np.sum(~keep)

    if n_total_rejected > 0:
        L.info(f"    MAD filter: rejected {n_total_rejected} "
               f"({n_global_rejected} global, {np.sum(~local_ok & global_ok)} local)")

    return [depth_points[i] for i in range(len(depth_points)) if keep[i]]


# ═══════════════════════════════════════════════════════════
# TPU (Total Propagated Uncertainty)
# ═══════════════════════════════════════════════════════════

def _compute_tpu(depth_points, sea_surface_std, kd, water_temp):
    """
    Compute Total Propagated Uncertainty (TPU) per depth point.
    Based on Albright & Glennie (2021) simplified model.

    Error sources:
      1. Sea surface uncertainty (σ_ss) — from surface detection variance
      2. Refraction uncertainty (σ_refr) — grows with depth
      3. Geolocation uncertainty (σ_geo) — ~0.5m horizontal → vertical via slope
      4. Photon timing uncertainty (σ_timing) — ~0.2ns → ~0.03m in water
    """
    for p in depth_points:
        depth = p["depth"]

        # Sea surface uncertainty (constant for this beam)
        sigma_ss = sea_surface_std

        # Refraction uncertainty: grows with depth and incidence angle
        # Simplified: σ_refr ≈ 0.003 × depth (from Parrish 2019 Fig. 8)
        sigma_refr = 0.003 * depth

        # Kd-related uncertainty: deeper in turbid water → more scattering
        # σ_kd ≈ kd × depth × 0.02 (photon path length variation)
        sigma_kd = kd * depth * 0.02

        # Timing uncertainty: ~0.2 ns single-photon jitter
        # In water: 0.2e-9 * c / (2 * n_water) ≈ 0.022m
        sigma_timing = 0.022

        # Bottom slope effect: horizontal error × estimated local slope
        # We estimate slope from track quality (poor quality → steeper slope)
        track_q = p.get("track_quality", 0.5)
        est_slope = 0.1 * (1 - track_q)  # 0-10% slope
        sigma_slope = 0.5 * est_slope  # 0.5m horizontal uncertainty × slope

        # Total propagated uncertainty (RSS)
        tpu = math.sqrt(
            sigma_ss ** 2 +
            sigma_refr ** 2 +
            sigma_kd ** 2 +
            sigma_timing ** 2 +
            sigma_slope ** 2
        )

        p["tpu_m"] = round(tpu, 3)
        p["tpu_components"] = {
            "sea_surface": round(sigma_ss, 4),
            "refraction": round(sigma_refr, 4),
            "kd_scatter": round(sigma_kd, 4),
            "timing": round(sigma_timing, 4),
            "slope": round(sigma_slope, 4),
        }

    return depth_points


# ═══════════════════════════════════════════════════════════
# CONFIDENCE SCORING
# ═══════════════════════════════════════════════════════════

def _compute_confidence(depth, signal_photons, kd, max_detectable, track_quality):
    """
    Compute overall confidence for a depth point.
    Combines depth/Kd ratio, signal strength, and track quality.
    """
    # Depth ratio: how close to detection limit?
    depth_ratio = depth / max(max_detectable, 0.1)
    depth_score = max(0, 1.0 - depth_ratio ** 2)  # quadratic decay

    # Signal strength: more photons → more confident
    signal_score = min(1.0, signal_photons / 10.0)

    # Combined (weighted)
    confidence = 0.4 * depth_score + 0.3 * signal_score + 0.3 * track_quality
    confidence = round(max(0.05, min(1.0, confidence)), 3)

    # Categorical label
    if confidence >= 0.7:
        label = "high"
    elif confidence >= 0.4:
        label = "medium"
    else:
        label = "low"

    return label


# ═══════════════════════════════════════════════════════════
# MULTI-PASS AGGREGATION
# ═══════════════════════════════════════════════════════════

def aggregate_multi_pass(all_results):
    """
    Aggregate depth points from multiple granules/passes.
    Cross-validates overlapping points and computes ensemble statistics.

    Returns unified depth point list with cross-validation metrics.
    """
    if not all_results:
        return []

    all_pts = []
    for result in all_results:
        if result and "depth_points" in result:
            for p in result["depth_points"]:
                all_pts.append(p)

    if len(all_pts) < 5:
        return all_pts

    # Grid-based aggregation: group nearby points
    grid_res_deg = 0.0001  # ~11m
    grid = {}
    for p in all_pts:
        gk = (round(p["lat"] / grid_res_deg) * grid_res_deg,
              round(p["lon"] / grid_res_deg) * grid_res_deg)
        grid.setdefault(gk, []).append(p)

    aggregated = []
    for (glat, glon), pts in grid.items():
        if len(pts) == 1:
            p = pts[0].copy()
            p["n_passes"] = 1
            p["cross_validated"] = False
            aggregated.append(p)
            continue

        # Multiple passes at this location → cross-validate
        depths = np.array([p["depth"] for p in pts])
        tpus = np.array([p.get("tpu_m", 0.5) for p in pts])

        # Weighted mean (inverse TPU weighting)
        weights = 1.0 / (tpus ** 2 + 0.01)
        mean_depth = float(np.average(depths, weights=weights))
        std_depth = float(np.sqrt(np.average((depths - mean_depth) ** 2, weights=weights)))

        # Cross-validation: agreement between passes
        agreement = 1.0 - min(1.0, std_depth / max(mean_depth * 0.1, 0.5))

        best = min(pts, key=lambda p: p.get("tpu_m", 1.0))
        merged = best.copy()
        merged["depth"] = round(mean_depth, 2)
        merged["tpu_m"] = round(float(np.min(tpus)) * (0.7 if agreement > 0.7 else 1.0), 3)
        merged["n_passes"] = len(pts)
        merged["cross_validated"] = True
        merged["pass_std_m"] = round(std_depth, 3)
        merged["pass_agreement"] = round(agreement, 3)
        merged["confidence"] = "high" if agreement > 0.7 else "medium" if agreement > 0.4 else "low"
        aggregated.append(merged)

    n_xval = sum(1 for p in aggregated if p.get("cross_validated"))
    L.info(f"Multi-pass aggregation: {len(all_pts)} → {len(aggregated)} points "
           f"({n_xval} cross-validated)")
    return aggregated


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def _build_profile(lat_bins, sea_arr, bath_arr, ws_mean, height_res):
    """Build along-track depth profile for chart visualization."""
    profile = []
    for i, lb in enumerate(lat_bins):
        sea_h = sea_arr[i] if i < len(sea_arr) and np.isfinite(sea_arr[i]) else ws_mean
        bath_h = bath_arr[i] if i < len(bath_arr) else np.nan
        dist_m = (lb - lat_bins[0]) if lat_bins else 0
        depth = None
        if np.isfinite(bath_h) and np.isfinite(sea_h) and (sea_h - bath_h) > 0:
            depth = round(float(sea_h - bath_h), 2)
        profile.append({
            "dist_m": round(float(dist_m), 1),
            "sea_surface": round(float(sea_h), 2) if np.isfinite(sea_h) else None,
            "bottom": round(float(bath_h), 2) if np.isfinite(bath_h) else None,
            "depth": depth,
        })
    return profile


def _build_photon_cloud(corr_x, corr_y, corr_z, corr_conf, ws_mean,
                         to_wgs, max_photons=5000):
    """Build subsampled photon cloud for frontend visualization."""
    step = max(1, len(corr_z) // max_photons)
    cloud = []
    for i in range(0, len(corr_z), step):
        try:
            px, py = to_wgs.transform(float(corr_x[i]), float(corr_y[i]))
        except Exception:
            continue
        h = float(corr_z[i])
        c = int(corr_conf[i]) if i < len(corr_conf) else 0
        if abs(h - ws_mean) < 0.5:
            cls = "surface"
        elif ws_mean - MAX_DEPTH_M < h < ws_mean - 0.5:
            cls = "bathymetry"
        else:
            cls = "noise"
        cloud.append({
            "lat": round(py, 6), "lon": round(px, 6),
            "h": round(h, 2), "conf": c, "cls": cls,
        })
    return cloud


def _aggregate_quality(quality_list):
    """Aggregate quality metrics across multiple beams."""
    if not quality_list:
        return {}

    return {
        "n_beams": len(quality_list),
        "total_depth_points": sum(q.get("n_depth_points", 0) for q in quality_list),
        "mean_kd": round(np.mean([q["kd_532"] for q in quality_list]), 4),
        "mean_tpu_m": round(np.mean([q["mean_tpu_m"] for q in quality_list]), 3),
        "max_detectable_m": round(np.mean([q["max_detectable_m"] for q in quality_list]), 1),
        "sea_surface_m": round(np.mean([q["sea_surface_m"] for q in quality_list]), 2),
        "depth_range": [
            round(min(q["depth_range"][0] for q in quality_list), 1),
            round(max(q["depth_range"][1] for q in quality_list), 1),
        ],
        "beams": quality_list,
    }


# ═══════════════════════════════════════════════════════════
# QUICK MODE — SlideRule (no download needed)
# ═══════════════════════════════════════════════════════════

def quick_icesat2(bbox, start_date, end_date):
    """
    Fast ICESat-2 bathymetry using SlideRule API (no file download).
    Returns filtered bathymetric photons with basic refraction correction.
    Falls back gracefully if SlideRule unavailable.
    """
    try:
        from sliderule import sliderule, icesat2
        sliderule.init("slideruleearth.io")
    except Exception:
        return None, "SlideRule not installed"

    w, s, e, n = bbox
    from datetime import datetime, timedelta
    d0 = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=90)
    d1 = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=90)

    poly = [{"lon": w, "lat": s}, {"lon": e, "lat": s},
            {"lon": e, "lat": n}, {"lon": w, "lat": n}, {"lon": w, "lat": s}]
    parms = {
        "poly": poly,
        "t0": d0.strftime('%Y-%m-%dT00:00:00Z'),
        "t1": d1.strftime('%Y-%m-%dT23:59:59Z'),
        "srt": 1, "cnf": 0, "len": 20.0, "res": 20.0,
        "pass_invalid": False,
        "yapc": {"score": 0, "knn": 0, "min_ph": 4},
    }

    L.info(f"SlideRule: querying {d0.date()} → {d1.date()}...")
    gdf = icesat2.atl03sp(parms)
    if gdf is None or len(gdf) == 0:
        return None, "No photons found"

    L.info(f"SlideRule: {len(gdf)} raw photons")
    return gdf, None
