"""SCI-R1 acceptance: Open-Meteo tide correction, offline (requests mocked)."""
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from conftest import make_s2, install_fake_dl_pro, PREEXISTING_ROI_KEYS

import tide

BBOX = [-77.5, 24.0, -77.3, 24.2]
ACQ = datetime(2024, 4, 16, 10, 0, tzinfo=timezone.utc)


class _Resp:
    def __init__(self, js):
        self._js = js

    def raise_for_status(self):
        pass

    def json(self):
        return self._js


def _hourly_series(peak_hour=10, peak=0.6, base=0.2):
    times = [f"2024-04-16T{h:02d}:00" for h in range(24)]
    heights = [peak if h == peak_hour else base for h in range(24)]
    return {"hourly": {"time": times, "sea_level_height_msl": heights}}


@pytest.fixture(autouse=True)
def _clear_tide_cache():
    tide._CACHE.clear()
    yield
    tide._CACHE.clear()


def test_tide_lowers_constant_grid(monkeypatch):
    monkeypatch.setattr(tide.requests, "get",
                        lambda url, timeout=15: _Resp(_hourly_series()))
    grid = np.full((50, 60), 10.0, np.float32)
    grid[:5, :] = np.nan

    corr, tide_m, info = tide.apply_tide_correction(grid, BBOX, ACQ)

    assert info["method"] == "open-meteo/cmems"
    assert abs(tide_m - 0.6) < 1e-9
    valid = np.isfinite(corr)
    assert np.isnan(corr[:5, :]).all()          # NaNs preserved
    assert np.abs(corr[valid] - 9.4).max() < 1e-3   # 10 m -> 9.4 m +/- 1 mm


def test_tide_api_down_leaves_depth_unchanged(monkeypatch):
    def _boom(url, timeout=15):
        raise ConnectionError("dns failure")
    monkeypatch.setattr(tide.requests, "get", _boom)
    grid = np.full((20, 20), 10.0, np.float32)

    corr, tide_m, info = tide.apply_tide_correction(grid, BBOX, ACQ)

    assert info["method"] == "unavailable"
    assert tide_m == 0.0
    assert info["applied_m"] == 0.0
    assert np.array_equal(corr, grid)


def test_infer_payload_keys_and_tide_block(monkeypatch, fake_minio):
    from fastapi.testclient import TestClient
    import main
    import s2_fetch

    install_fake_dl_pro(monkeypatch)
    monkeypatch.setattr(
        s2_fetch, "fetch_s2",
        lambda bbox, sd, ed, max_cloud=40, **kw: make_s2(bbox))
    monkeypatch.setattr(
        tide, "get_tide_height",
        lambda lat, lon, utc_dt=None: (0.6, {
            "method": "open-meteo/cmems", "utc": utc_dt.isoformat(),
            "tide_range_that_day_m": 0.9}))

    client = TestClient(main.app)
    r = client.post("/bathymetry/infer",
                    json={"raster_id": "tide-test", "bbox": BBOX})
    assert r.status_code == 200, r.text
    payload = r.json()

    missing = PREEXISTING_ROI_KEYS - set(payload)
    assert not missing, f"pre-existing keys lost: {missing}"
    assert "composite" not in payload
    t = payload["tide"]
    assert t["applied"] is True
    assert abs(t["applied_m"] - 0.6) < 1e-9
    assert t["datum"] == "MSL"
    assert t["method"] == "open-meteo/cmems"
    assert abs(payload["stats"]["mean_m"] - 9.4) < 1e-2


def test_infer_tide_kill_switch(monkeypatch, fake_minio):
    from fastapi.testclient import TestClient
    import main
    import s2_fetch

    install_fake_dl_pro(monkeypatch)
    monkeypatch.setattr(
        s2_fetch, "fetch_s2",
        lambda bbox, sd, ed, max_cloud=40, **kw: make_s2(bbox))
    monkeypatch.setenv("TIDE_CORRECTION", "0")
    monkeypatch.setattr(
        tide, "get_tide_height",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API called")))

    client = TestClient(main.app)
    r = client.post("/bathymetry/infer",
                    json={"raster_id": "tide-off", "bbox": BBOX})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["tide"]["applied"] is False
    assert payload["tide"]["datum"] == "uncorrected"
    assert abs(payload["stats"]["mean_m"] - 10.0) < 1e-2
