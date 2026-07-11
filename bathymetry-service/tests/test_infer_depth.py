"""Unit tests for sdb_engine.infer_depth on synthetic coastal scenes.

Builds georeferenced GeoTIFFs with a known west-east depth gradient and a
land strip, then checks the stub contract ((C,H,W) in -> (H,W) float32 out,
NaN = land/nodata), the land cut, the 0-25 m clip and that the recovered
depth field is monotonic with the true gradient.

Run:  pytest bathymetry-service/tests/test_infer_depth.py -v
"""
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sdb_engine  # noqa: E402

H, W = 120, 200
LAND_COLS = 30           # west strip is land
MAX_TRUE_DEPTH = 20.0


def _true_depth():
    z = np.zeros((H, W), dtype=np.float64)
    cols = np.arange(W - LAND_COLS, dtype=np.float64)
    z[:, LAND_COLS:] = 0.5 + (MAX_TRUE_DEPTH - 0.5) * cols / cols.max()
    return z


def _beer_lambert(z, r_bottom, r_inf, k):
    return r_inf + (r_bottom - r_inf) * np.exp(-2.0 * k * z)


def _write_tif(path, bands, dtype, descriptions=None):
    transform = from_bounds(54.60, 24.78, 54.66, 24.82, W, H)
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=len(bands),
        dtype=dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        for i, b in enumerate(bands, start=1):
            dst.write(b.astype(dtype), i)
            if descriptions:
                dst.set_band_description(i, descriptions[i - 1])
    return path


@pytest.fixture
def multispectral_tif(tmp_path):
    """4-band S2-like scene (B,G,R,NIR uint16 DN, reflectance*10000)."""
    rng = np.random.default_rng(7)
    z = _true_depth()
    blue = _beer_lambert(z, 0.120, 0.050, 0.08)
    green = _beer_lambert(z, 0.140, 0.040, 0.12)
    red = _beer_lambert(z, 0.100, 0.010, 0.50)
    nir = np.full((H, W), 0.008)
    for arr, land_val in ((blue, 0.10), (green, 0.18), (red, 0.20), (nir, 0.35)):
        arr[:, :LAND_COLS] = land_val
    bands = []
    for arr in (blue, green, red, nir):
        dn = arr * 10000.0 * (1.0 + rng.normal(0, 0.01, (H, W)))
        bands.append(np.clip(dn, 1, 12000))
    return _write_tif(tmp_path / "s2_scene.tif", bands, "uint16",
                      ["blue", "green", "red", "nir"])


@pytest.fixture
def rgb_tif(tmp_path):
    """Plain 3-band RGB uint8 scene (R,G,B band order)."""
    rng = np.random.default_rng(11)
    z = _true_depth()
    blue = _beer_lambert(z, 0.120, 0.050, 0.08)
    green = _beer_lambert(z, 0.140, 0.040, 0.12)
    red = _beer_lambert(z, 0.100, 0.010, 0.50)
    for arr, land_val in ((blue, 0.10), (green, 0.18), (red, 0.35)):
        arr[:, :LAND_COLS] = land_val
    bands = []
    for arr in (red, green, blue):
        u8 = arr / 0.40 * 255.0 * (1.0 + rng.normal(0, 0.01, (H, W)))
        bands.append(np.clip(u8, 0, 255))
    return _write_tif(tmp_path / "rgb_scene.tif", bands, "uint8")


def _read(path):
    with rasterio.open(path) as src:
        return (src.read(masked=True).astype(np.float32).filled(np.nan),
                list(src.descriptions), src.dtypes[0])


def _column_monotonicity(depth, true_z):
    """Spearman rank correlation between column-mean predicted and true depth."""
    pred = np.nanmean(depth[:, LAND_COLS:], axis=0)
    true = true_z[:, LAND_COLS:].mean(axis=0)
    ok = np.isfinite(pred)
    pr = np.argsort(np.argsort(pred[ok])).astype(np.float64)
    tr = np.argsort(np.argsort(true[ok])).astype(np.float64)
    return float(np.corrcoef(pr, tr)[0, 1])


def test_multispectral_contract_and_monotonicity(multispectral_tif):
    bands, names, dtype = _read(multispectral_tif)
    result = sdb_engine.infer(bands, band_names=names, src_dtype=dtype)
    depth = result["depth"]

    assert depth.shape == (H, W)
    assert depth.dtype == np.float32
    assert result["calibrated"] is True
    assert "uae" in result["model"]
    # land strip cut to NaN, water mostly predicted
    assert np.isnan(depth[:, :LAND_COLS]).all()
    water_frac = np.isfinite(depth[:, LAND_COLS:]).mean()
    assert water_frac > 0.9
    # physical range
    vals = depth[np.isfinite(depth)]
    assert vals.min() >= 0.0 and vals.max() <= 25.0
    # depth increases with the true gradient
    rho = _column_monotonicity(depth, _true_depth())
    assert rho > 0.5, f"monotonicity too weak: rho={rho:.2f}"
    # uncertainty channel present and positive over water
    sigma = result["sigma"]
    assert sigma is not None and np.nanmin(sigma) >= 0.0
    # stub-compatible entry point returns the same grid
    d2 = sdb_engine.infer_depth(bands, band_names=names)
    assert np.array_equal(np.isfinite(d2), np.isfinite(depth))


def test_rgb_fallback_is_labeled_uncalibrated(rgb_tif):
    bands, _, dtype = _read(rgb_tif)
    result = sdb_engine.infer(bands, src_dtype=dtype)
    depth = result["depth"]

    assert depth.shape == (H, W)
    assert depth.dtype == np.float32
    assert result["calibrated"] is False
    assert result["model"] == "stumpf-log-ratio"
    assert "LOW" in result["confidence"]
    assert np.isnan(depth[:, :LAND_COLS]).all()
    vals = depth[np.isfinite(depth)]
    assert len(vals) > 0
    assert vals.min() >= 0.0 and vals.max() <= 25.0
    rho = _column_monotonicity(depth, _true_depth())
    assert rho > 0.5, f"monotonicity too weak: rho={rho:.2f}"


def test_too_few_bands_rejected():
    with pytest.raises(ValueError):
        sdb_engine.infer(np.zeros((2, 10, 10), dtype=np.float32))
