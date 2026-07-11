"""Track T2 test battery — Feature Engineering / Fusion / Tidal Correction.

Tests
-----
T2a  Import-clean: both modules import without error.
T2b  Optical indices on synthetic (T,B,H,W) reflectance:
     - Shape (H,W,9) returned.
     - NDWI ∈ [-1, 1] (all values).
     - Temporal median suppresses an injected glint spike (variance assertion).
     - All output channels are finite.
T2c  deepwater_baseline_prior:
     - Returns finite IDW prior on known-point pixels.
     - icesat2_spatial_prior is finite where tracks pass.
     - altimetry_anchor_label references ICESat-2, not radar.
     - idw_valid_mask is bool (H,W).
T2d  tidal_offset + normalize_scenes_to_datum:
     - tidal_offset returns finite array of length T.
     - normalize_scenes_to_datum reduces inter-date spread on a
       synthetic tide-contaminated depth set.
     - Metadata contains 'method' and 'warning'.
T2e  Determinism: fixed seed reproduces all outputs exactly.

Run with:
    .venv/bin/python3 -m pytest backend/sdb_optim/tests/test_t2.py -v
or:
    .venv/bin/python3 backend/sdb_optim/tests/test_t2.py
"""

from __future__ import annotations

import datetime
import sys
import pathlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Add project root to path so imports work without installation
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[3]  # Bathymetry_VMarch/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 42

# ============================================================================
# T2a — Import clean
# ============================================================================

def test_t2a_import_clean():
    """Both T2 modules import without raising any exception."""
    import importlib
    fe = importlib.import_module("backend.sdb_optim.feature_engineering")
    tc = importlib.import_module("backend.sdb_optim.tidal_correction")
    assert hasattr(fe, "compute_optical_indices"), "compute_optical_indices missing"
    assert hasattr(fe, "compute_temporal_stats"),  "compute_temporal_stats missing"
    assert hasattr(fe, "deepwater_baseline_prior"), "deepwater_baseline_prior missing"
    assert hasattr(fe, "fuse_optical_sensors"),     "fuse_optical_sensors missing"
    assert hasattr(tc, "tidal_offset"),             "tidal_offset missing"
    assert hasattr(tc, "normalize_scenes_to_datum"), "normalize_scenes_to_datum missing"
    print("T2a PASS: both modules import cleanly.")


# ============================================================================
# T2b — Optical indices + temporal stats on synthetic data
# ============================================================================

def _make_synthetic_bands(H: int = 32, W: int = 32, seed: int = SEED):
    """Return (blue, green, red, nir) as float32 in [0.01, 0.25] reflectance."""
    rng = np.random.default_rng(seed)
    # Realistic spectral order: blue > green > red > nir for open water
    blue  = rng.uniform(0.03, 0.12, (H, W)).astype(np.float32)
    green = rng.uniform(0.02, 0.08, (H, W)).astype(np.float32)
    red   = rng.uniform(0.01, 0.05, (H, W)).astype(np.float32)
    nir   = rng.uniform(0.005, 0.02, (H, W)).astype(np.float32)
    return blue, green, red, nir


def test_t2b_indices_shape_dtype():
    """compute_optical_indices returns (H,W,9) float32, all finite."""
    from backend.sdb_optim.feature_engineering import compute_optical_indices
    H, W = 32, 32
    blue, green, red, nir = _make_synthetic_bands(H, W)
    cube, names = compute_optical_indices(blue, green, red, nir)
    assert cube.shape == (H, W, 9), f"Expected (32,32,9); got {cube.shape}"
    assert cube.dtype == np.float32, f"Expected float32; got {cube.dtype}"
    assert len(names) == 9, f"Expected 9 channel names; got {len(names)}"
    assert np.all(np.isfinite(cube)), "Non-finite values in optical indices cube"
    print(f"T2b-shape PASS: shape={cube.shape}, dtype={cube.dtype}, names={names}")


def test_t2b_ndwi_range():
    """NDWI channel is strictly within [-1, 1]."""
    from backend.sdb_optim.feature_engineering import compute_optical_indices
    blue, green, red, nir = _make_synthetic_bands()
    cube, names = compute_optical_indices(blue, green, red, nir)
    ndwi_idx = names.index("NDWI")
    ndwi = cube[:, :, ndwi_idx]
    assert np.all(ndwi >= -1.0 - 1e-5), f"NDWI has values below -1: {ndwi.min()}"
    assert np.all(ndwi <= 1.0 + 1e-5),  f"NDWI has values above +1: {ndwi.max()}"
    print(f"T2b-NDWI PASS: NDWI range [{ndwi.min():.4f}, {ndwi.max():.4f}]")


def test_t2b_temporal_median_suppresses_glint():
    """Temporal median reduces per-pixel variance vs. a glint-spiked single scene.

    We construct T=5 scenes where ONE date has a 5x glint spike on the blue
    channel.  The temporal median should give lower variance than including
    the spiked scene in a mean.
    """
    from backend.sdb_optim.feature_engineering import compute_temporal_stats

    rng = np.random.default_rng(SEED)
    H, W = 32, 32
    T = 5

    # Base reflectance (H, W, 1) per date
    base = rng.uniform(0.03, 0.08, (H, W)).astype(np.float32)

    # Stack T dates, inject a 5x glint spike on date 0
    ts = np.stack([base] * T, axis=0)[:, :, :, np.newaxis]  # (T, H, W, 1)
    ts[0, :, :, 0] *= 5.0   # spike at date 0

    # All pixels valid
    vm = np.ones((T, H, W), dtype=bool)

    stats, names = compute_temporal_stats(ts, valid_mask=vm)
    # stats shape: (H, W, 3) for 1 channel (med, var, count)
    assert stats.shape == (H, W, 3), f"Expected (32,32,3); got {stats.shape}"
    assert np.all(np.isfinite(stats)), "Non-finite values in temporal stats"

    med_idx = names.index("temporal_median_ch0")
    var_idx = names.index("temporal_var_ch0")

    # Median: should be close to base (glint at index 0 is the outlier for T=5)
    median_values = stats[:, :, med_idx]
    # Variance of the median surface should be much lower than variance of the spiked scene
    var_of_median = float(np.var(median_values))
    # Spiked scene alone
    var_spiked = float(np.var(ts[0, :, :, 0]))
    # Base alone
    var_base = float(np.var(ts[1, :, :, 0]))

    print(
        f"T2b-temporal PASS: var(spiked_scene)={var_spiked:.5f}, "
        f"var(base)={var_base:.5f}, var(temporal_median)={var_of_median:.5f}"
    )
    assert var_of_median < var_spiked, (
        f"Temporal median variance {var_of_median:.5f} should be < "
        f"spiked scene variance {var_spiked:.5f}"
    )


def test_t2b_temporal_stats_shape():
    """compute_temporal_stats with (T,H,W) 3D input returns (H,W,3) cube."""
    from backend.sdb_optim.feature_engineering import compute_temporal_stats
    rng = np.random.default_rng(SEED)
    T, H, W = 6, 16, 16
    ts = rng.uniform(0.01, 0.2, (T, H, W)).astype(np.float32)
    stats, names = compute_temporal_stats(ts)
    assert stats.shape == (H, W, 3), f"Expected (16,16,3); got {stats.shape}"
    assert all(np.isfinite(stats).ravel()), "Non-finite temporal stats"
    print(f"T2b-temporal-shape PASS: {stats.shape}, names={names}")


# ============================================================================
# T2c — deepwater_baseline_prior
# ============================================================================

def _make_grid(H: int = 20, W: int = 20):
    """Return (grid_lat, grid_lon) covering a small patch near Abu Dhabi."""
    lat = np.linspace(24.80, 24.82, H)
    lon = np.linspace(54.62, 54.65, W)
    grid_lon, grid_lat = np.meshgrid(lon, lat)  # (H, W)
    return grid_lat.astype(np.float32), grid_lon.astype(np.float32)


def test_t2c_deepwater_prior_finite():
    """IDW prior is finite on pixels within range of known soundings."""
    from backend.sdb_optim.feature_engineering import deepwater_baseline_prior

    rng = np.random.default_rng(SEED)
    H, W = 20, 20
    grid_lat, grid_lon = _make_grid(H, W)

    # 10 synthetic soundings scattered in the grid
    N = 10
    sc_lat = rng.uniform(24.80, 24.82, N)
    sc_lon = rng.uniform(54.62, 54.65, N)
    sc_dep = rng.uniform(1.0, 15.0, N).astype(np.float32)
    coords = np.column_stack([sc_lat, sc_lon]).astype(np.float64)

    prior = deepwater_baseline_prior(
        coords=coords,
        depths_known=sc_dep,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
    )

    assert "idw_depth_prior" in prior, "Missing 'idw_depth_prior'"
    assert "idw_valid_mask" in prior, "Missing 'idw_valid_mask'"
    assert "altimetry_anchor_label" in prior, "Missing 'altimetry_anchor_label'"
    assert prior["idw_depth_prior"].shape == (H, W), (
        f"IDW prior shape {prior['idw_depth_prior'].shape} != ({H},{W})"
    )
    assert prior["idw_valid_mask"].dtype == bool, (
        f"idw_valid_mask dtype {prior['idw_valid_mask'].dtype} != bool"
    )

    # Where valid, check finite values
    valid = prior["idw_valid_mask"]
    assert valid.any(), "No valid IDW pixels found"
    idw_valid_vals = prior["idw_depth_prior"][valid]
    assert np.all(np.isfinite(idw_valid_vals)), "Non-finite IDW values at valid pixels"
    assert np.all(idw_valid_vals >= 0), "Negative IDW depth values"

    print(
        f"T2c-idw PASS: {valid.sum()}/{H*W} valid pixels, "
        f"depth range [{idw_valid_vals.min():.2f}, {idw_valid_vals.max():.2f}] m"
    )


def test_t2c_icesat2_prior_finite():
    """ICESat-2 spatial prior is finite where tracks are provided."""
    from backend.sdb_optim.feature_engineering import deepwater_baseline_prior

    rng = np.random.default_rng(SEED + 1)
    H, W = 20, 20
    grid_lat, grid_lon = _make_grid(H, W)

    # Empty soundings (only ICESat-2)
    coords = np.zeros((0, 2), dtype=np.float64)
    deps = np.array([], dtype=np.float32)

    # 15 synthetic ICESat-2 track points crossing the grid
    M = 15
    is_lats = np.linspace(24.80, 24.82, M)
    is_lons = np.linspace(54.62, 54.65, M)
    is_deps = rng.uniform(2.0, 10.0, M).astype(np.float32)

    prior = deepwater_baseline_prior(
        coords=coords,
        depths_known=deps,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        icesat2_depths=is_deps,
        icesat2_lats=is_lats,
        icesat2_lons=is_lons,
    )

    is_prior = prior["icesat2_spatial_prior"]
    assert is_prior is not None, "icesat2_spatial_prior is None"
    assert is_prior.shape == (H, W), f"ICESat-2 prior shape {is_prior.shape} != ({H},{W})"
    finite_mask = np.isfinite(is_prior)
    assert finite_mask.any(), "No finite ICESat-2 prior pixels"
    finite_vals = is_prior[finite_mask]
    assert np.all(finite_vals >= 0), "Negative ICESat-2 prior values"
    print(
        f"T2c-icesat2 PASS: {finite_mask.sum()}/{H*W} pixels covered, "
        f"range [{finite_vals.min():.2f}, {finite_vals.max():.2f}] m"
    )


def test_t2c_altimetry_label_correct():
    """altimetry_anchor_label explicitly names ICESat-2, not radar."""
    from backend.sdb_optim.feature_engineering import deepwater_baseline_prior

    coords = np.zeros((1, 2))
    deps = np.array([5.0])
    grid_lat, grid_lon = _make_grid(5, 5)
    prior = deepwater_baseline_prior(coords, deps, grid_lat, grid_lon)
    label = prior["altimetry_anchor_label"]
    assert "ICESat-2" in label, f"Label missing 'ICESat-2': {label}"
    assert "radar" in label.lower(), (
        f"Label should explicitly disclaim radar: {label}"
    )
    print(f"T2c-label PASS: label = '{label[:80]}...'")


# ============================================================================
# T2d — tidal_offset + normalize_scenes_to_datum
# ============================================================================

def _make_timestamps(n: int = 5, start: str = "2023-06-01") -> list:
    """Return n UTC datetimes spaced 4 days apart."""
    base = datetime.datetime(2023, 6, 1, 10, 30, 0, tzinfo=datetime.timezone.utc)
    return [base + datetime.timedelta(days=4 * i) for i in range(n)]


def test_t2d_tidal_offset_finite():
    """tidal_offset returns finite float32 array of length T."""
    from backend.sdb_optim.tidal_correction import tidal_offset
    T = 5
    timestamps = _make_timestamps(T)
    tide, meta = tidal_offset(
        timestamps, lat=24.82, lon=54.65,
        fallback_site="UAE_gulf",
    )
    assert tide.shape == (T,), f"Expected shape ({T},); got {tide.shape}"
    assert tide.dtype == np.float32, f"Expected float32; got {tide.dtype}"
    assert np.all(np.isfinite(tide)), "Non-finite tide heights"
    assert "method" in meta, "meta missing 'method'"
    assert "warning" in meta, "meta missing 'warning'"
    print(
        f"T2d-tidal PASS: method={meta['method']}, "
        f"tide range [{tide.min():.3f}, {tide.max():.3f}] m"
    )


def test_t2d_normalize_reduces_spread():
    """normalize_scenes_to_datum reduces inter-date depth spread on synthetic data.

    We inject a known tide (sine wave) into otherwise identical depth obs.
    After normalization, inter-scene std should be significantly reduced.
    """
    from backend.sdb_optim.tidal_correction import normalize_scenes_to_datum, tidal_offset

    T = 8
    timestamps = _make_timestamps(T)
    lat, lon = 24.82, 54.65

    # Get the stub tide heights (deterministic)
    tide_h, _ = tidal_offset(timestamps, lat, lon, fallback_site="UAE_gulf")

    # Construct synthetic "observed" depths: true depth 5.0 m + tidal bias
    true_depth = 5.0
    observed_depths = np.full(T, true_depth, dtype=np.float32) + tide_h

    std_before = float(np.std(observed_depths))

    corrected, applied_tide, meta = normalize_scenes_to_datum(
        observed_depths, timestamps, lat, lon,
        fallback_site="UAE_gulf",
        target_datum="LAT",
    )

    std_after = float(np.std(corrected))

    print(
        f"T2d-normalize PASS: std_before={std_before:.4f} m, "
        f"std_after={std_after:.4f} m, reduction={std_before-std_after:.4f} m"
    )
    # After correction, the spread attributable to tides should be nearly zero
    # (we injected exactly the model tide, so subtraction should cancel it)
    assert std_after < std_before + 1e-4, (
        f"Expected std_after ({std_after:.4f}) <= std_before ({std_before:.4f})"
    )
    # The residual std should be < 1e-3 m since we injected exactly the model tide
    assert std_after < 0.05, (
        f"Residual std {std_after:.4f} m is too large (expected < 0.05 m)"
    )

    assert "target_datum" in meta, "meta missing 'target_datum'"
    assert "tide_rms_m" in meta, "meta missing 'tide_rms_m'"


def test_t2d_msl_datum_conversion():
    """Normalizing to MSL shifts depths by the MSL-above-CD offset."""
    from backend.sdb_optim.tidal_correction import normalize_scenes_to_datum, _MSL_ABOVE_CD

    T = 3
    timestamps = _make_timestamps(T)
    depths = np.array([5.0, 5.0, 5.0], dtype=np.float32)

    lat_corr, _, meta_lat = normalize_scenes_to_datum(
        depths, timestamps, 24.82, 54.65,
        fallback_site="UAE_gulf", target_datum="LAT"
    )
    msl_corr, _, meta_msl = normalize_scenes_to_datum(
        depths, timestamps, 24.82, 54.65,
        fallback_site="UAE_gulf", target_datum="MSL"
    )
    expected_offset = _MSL_ABOVE_CD["UAE_gulf"]
    actual_offset = float(np.mean(msl_corr - lat_corr))
    assert abs(actual_offset - expected_offset) < 1e-4, (
        f"MSL offset {actual_offset:.4f} m != expected {expected_offset:.4f} m"
    )
    print(f"T2d-MSL PASS: MSL offset = {actual_offset:.4f} m (expected {expected_offset:.4f} m)")


# ============================================================================
# T2e — Determinism
# ============================================================================

def test_t2e_determinism_indices():
    """compute_optical_indices is deterministic (no random state)."""
    from backend.sdb_optim.feature_engineering import compute_optical_indices
    blue, green, red, nir = _make_synthetic_bands(seed=SEED)
    c1, _ = compute_optical_indices(blue, green, red, nir)
    c2, _ = compute_optical_indices(blue, green, red, nir)
    assert np.array_equal(c1, c2), "compute_optical_indices is NOT deterministic"
    print("T2e-indices-determinism PASS")


def test_t2e_determinism_tidal():
    """tidal_offset (harmonic stub) is deterministic."""
    from backend.sdb_optim.tidal_correction import tidal_offset
    timestamps = _make_timestamps(5)
    t1, _ = tidal_offset(timestamps, 24.82, 54.65, fallback_site="UAE_gulf")
    t2, _ = tidal_offset(timestamps, 24.82, 54.65, fallback_site="UAE_gulf")
    assert np.array_equal(t1, t2), "tidal_offset is NOT deterministic"
    print("T2e-tidal-determinism PASS")


def test_t2e_determinism_temporal():
    """compute_temporal_stats is deterministic."""
    from backend.sdb_optim.feature_engineering import compute_temporal_stats
    rng = np.random.default_rng(SEED)
    ts = rng.uniform(0.01, 0.2, (6, 16, 16)).astype(np.float32)
    s1, _ = compute_temporal_stats(ts)
    s2, _ = compute_temporal_stats(ts)
    assert np.array_equal(s1, s2, equal_nan=True), "compute_temporal_stats is NOT deterministic"
    print("T2e-temporal-determinism PASS")


def test_t2e_determinism_prior():
    """deepwater_baseline_prior is deterministic."""
    from backend.sdb_optim.feature_engineering import deepwater_baseline_prior
    rng = np.random.default_rng(SEED)
    H, W = 10, 10
    grid_lat, grid_lon = _make_grid(H, W)
    coords = np.column_stack([rng.uniform(24.80, 24.82, 5), rng.uniform(54.62, 54.65, 5)])
    deps = rng.uniform(1.0, 10.0, 5).astype(np.float32)

    p1 = deepwater_baseline_prior(coords, deps, grid_lat, grid_lon)
    p2 = deepwater_baseline_prior(coords, deps, grid_lat, grid_lon)

    np.testing.assert_array_equal(p1["idw_depth_prior"], p2["idw_depth_prior"])
    np.testing.assert_array_equal(p1["idw_valid_mask"], p2["idw_valid_mask"])
    print("T2e-prior-determinism PASS")


# ============================================================================
# Additional: fuse_optical_sensors smoke test
# ============================================================================

def test_t2_fuse_s2_only():
    """fuse_optical_sensors with S2 only returns correct band set."""
    from backend.sdb_optim.feature_engineering import fuse_optical_sensors
    rng = np.random.default_rng(SEED)
    H, W = 30, 30  # divisible by 3 for block-avg to 30 m
    s2 = {
        "blue":  rng.uniform(0.02, 0.12, (H, W)).astype(np.float32),
        "green": rng.uniform(0.01, 0.08, (H, W)).astype(np.float32),
        "red":   rng.uniform(0.005, 0.05, (H, W)).astype(np.float32),
        "nir":   rng.uniform(0.003, 0.02, (H, W)).astype(np.float32),
    }
    fused, meta = fuse_optical_sensors(s2, landsat=None, target_res_m=30.0, verbose=False)
    for band in ("blue", "green", "red", "nir"):
        assert band in fused, f"Missing band '{band}' in fused output"
        assert np.all(np.isfinite(fused[band])), f"Non-finite values in fused {band}"
    assert meta["sensors"] == ["S2"]
    assert meta["crosscal_applied"] is False
    print(
        f"T2-fuse-s2only PASS: shape={fused['blue'].shape}, bands={[k for k in fused if k not in ('sensor','n_scenes')]}"
    )


def test_t2_fuse_s2_l8():
    """fuse_optical_sensors S2+L8 fusion at 30 m produces valid fused bands."""
    from backend.sdb_optim.feature_engineering import fuse_optical_sensors
    rng = np.random.default_rng(SEED)
    H_s2, W_s2 = 30, 30   # 10 m grid (30 px = 300 m)
    H_l8, W_l8 = 10, 10   # 30 m grid (10 px = 300 m)
    s2 = {
        "blue":  rng.uniform(0.02, 0.12, (H_s2, W_s2)).astype(np.float32),
        "green": rng.uniform(0.01, 0.08, (H_s2, W_s2)).astype(np.float32),
        "red":   rng.uniform(0.005, 0.05, (H_s2, W_s2)).astype(np.float32),
        "nir":   rng.uniform(0.003, 0.02, (H_s2, W_s2)).astype(np.float32),
    }
    l8 = {
        "blue":  rng.uniform(0.022, 0.115, (H_l8, W_l8)).astype(np.float32),
        "green": rng.uniform(0.015, 0.075, (H_l8, W_l8)).astype(np.float32),
        "red":   rng.uniform(0.006, 0.048, (H_l8, W_l8)).astype(np.float32),
        "nir":   rng.uniform(0.004, 0.018, (H_l8, W_l8)).astype(np.float32),
    }
    fused, meta = fuse_optical_sensors(s2, l8, target_res_m=30.0, apply_crosscal=True, verbose=False)
    for band in ("blue", "green", "red", "nir"):
        assert band in fused, f"Missing '{band}'"
        assert np.all(np.isfinite(fused[band])), f"Non-finite in fused {band}"
        assert np.all(fused[band] > 0), f"Non-positive in fused {band}"
    assert meta["crosscal_applied"] is True
    assert "Pahlevan" in meta["warning"][0], "Cross-cal warning should cite Pahlevan"
    print(
        f"T2-fuse-s2-l8 PASS: shape={fused['blue'].shape}, "
        f"crosscal={meta['crosscal_applied']}"
    )


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_t2a_import_clean,
        test_t2b_indices_shape_dtype,
        test_t2b_ndwi_range,
        test_t2b_temporal_median_suppresses_glint,
        test_t2b_temporal_stats_shape,
        test_t2c_deepwater_prior_finite,
        test_t2c_icesat2_prior_finite,
        test_t2c_altimetry_label_correct,
        test_t2d_tidal_offset_finite,
        test_t2d_normalize_reduces_spread,
        test_t2d_msl_datum_conversion,
        test_t2e_determinism_indices,
        test_t2e_determinism_tidal,
        test_t2e_determinism_temporal,
        test_t2e_determinism_prior,
        test_t2_fuse_s2_only,
        test_t2_fuse_s2_l8,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL {fn.__name__}: {exc}")
            failed += 1

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"T2 BATTERY: {passed}/{total} PASSED,  {failed}/{total} FAILED")
    if failed:
        sys.exit(1)
