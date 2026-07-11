"""SCI-R3 acceptance: vendored regional coastline land cut (no network)."""
import numpy as np

from conftest import FakeMinio  # noqa: F401  (ensures sys.path setup)

import sdb_engine
from sdb_engine.coastline_mask import coastline_land_for, _bbox_in_region

KHALIFA = [54.60, 24.78, 54.72, 24.90]
CASABLANCA = [-7.70, 33.55, -7.55, 33.65]
BAHAMAS = [-77.5, 24.0, -77.3, 24.2]

H, W = 120, 160


def _all_water_bands():
    """4-band S2-like stack whose NDWI is water everywhere (green >> nir)."""
    rng = np.random.default_rng(3)
    blue = np.full((H, W), 1500.0) + rng.normal(0, 10, (H, W))
    green = np.full((H, W), 2000.0) + rng.normal(0, 10, (H, W))
    red = np.full((H, W), 1000.0) + rng.normal(0, 10, (H, W))
    nir = np.full((H, W), 100.0) + rng.normal(0, 5, (H, W))
    return np.stack([blue, green, red, nir]).astype(np.float32)


def test_khalifa_bbox_uses_uae_gpkg():
    land, info = coastline_land_for(KHALIFA, (H, W))
    assert land is not None
    assert info["region"] == "uae"
    assert info["source"] == "osm-coastline-landpoly"
    assert info["n_polys"] > 0
    assert 0.0 < info["osm_land_frac"] < 1.0
    assert info["datum"] == "MHW (approx; not LAT)"
    assert info["vintage"]


def test_casablanca_picks_morocco():
    assert _bbox_in_region(CASABLANCA) == "morocco"
    land, info = coastline_land_for(CASABLANCA, (64, 64))
    assert land is not None
    assert info["region"] == "morocco"


def test_infer_cuts_exactly_coastline_land():
    bands = _all_water_bands()
    names = ["blue", "green", "red", "nir"]
    res_plain = sdb_engine.infer(bands, band_names=names, src_dtype=np.uint16)
    res_cut = sdb_engine.infer(bands, band_names=names, src_dtype=np.uint16,
                               bbox4326=KHALIFA)

    land, _ = coastline_land_for(KHALIFA, (H, W))
    assert land is not None and land.any()

    valid_plain = np.isfinite(res_plain["depth"])
    valid_cut = np.isfinite(res_cut["depth"])
    assert not valid_cut[land].any()                       # NaN under land polys
    assert np.array_equal(valid_cut, valid_plain & ~land)  # loses exactly those
    assert res_cut["mask"]["coastline"]["region"] == "uae"


def test_bahamas_unchanged_outside_regions():
    land, info = coastline_land_for(BAHAMAS, (H, W))
    assert land is None
    assert info["region"] is None

    bands = _all_water_bands()
    names = ["blue", "green", "red", "nir"]
    res_plain = sdb_engine.infer(bands, band_names=names, src_dtype=np.uint16)
    res_bbox = sdb_engine.infer(bands, band_names=names, src_dtype=np.uint16,
                                bbox4326=BAHAMAS)
    assert np.array_equal(res_plain["depth"], res_bbox["depth"], equal_nan=True)
