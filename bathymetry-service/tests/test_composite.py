"""SCI-R4 acceptance: inverse-variance compositing + glint gate."""
import numpy as np

from conftest import make_s2, install_fake_dl_pro, PREEXISTING_ROI_KEYS

from composite import compose

BBOX = [-77.5, 24.0, -77.3, 24.2]
SHAPE = (8, 10)


def _scene(depth, sigma, scene_id, glint=None, acquired="2024-04-16T10:00:00Z"):
    return {"depth": np.full(SHAPE, depth, np.float32),
            "sigma": np.full(SHAPE, sigma, np.float32),
            "scene_id": scene_id, "acquired": acquired,
            "cloud_cover": 5.0, "glint": glint, "tide_m": None}


def test_inverse_variance_math():
    results = [_scene(10.0, 0.5, "a"), _scene(11.0, 1.0, "b"),
               _scene(12.0, 2.0, "c")]
    out = compose(results)

    iv = np.array([1 / 0.5 ** 2, 1 / 1.0 ** 2, 1 / 2.0 ** 2])
    z_expect = float(np.sum(np.array([10, 11, 12]) * iv) / iv.sum())
    s_expect = float(1.0 / np.sqrt(iv.sum()))

    assert np.abs(out["depth"] - z_expect).max() < 1e-5
    assert np.abs(out["sigma"] - s_expect).max() < 1e-5
    agr = out["agreement"]
    assert agr["n_scenes_used"] == 3
    assert agr["n_scenes_grid_median"] == 3
    assert all(p["kept"] for p in agr["per_scene"])
    assert agr["px_std_median_m"] is not None


def test_glint_gate_drops_scene():
    results = [_scene(10.0, 0.5, "a", glint=0.01),
               _scene(11.0, 1.0, "b", glint=0.05),   # glinty -> dropped
               _scene(12.0, 2.0, "c", glint=0.02)]
    out = compose(results)
    agr = out["agreement"]
    assert agr["n_scenes_used"] == 2
    kept = {p["scene_id"]: p["kept"] for p in agr["per_scene"]}
    assert kept == {"a": True, "b": False, "c": True}

    iv = np.array([1 / 0.5 ** 2, 1 / 2.0 ** 2])
    z_expect = float(np.sum(np.array([10, 12]) * iv) / iv.sum())
    assert np.abs(out["depth"] - z_expect).max() < 1e-5


def test_glint_gate_always_keeps_one():
    results = [_scene(10.0, 0.5, "a", glint=0.08),
               _scene(11.0, 1.0, "b", glint=0.05)]
    out = compose(results)
    agr = out["agreement"]
    assert agr["n_scenes_used"] == 1
    kept = {p["scene_id"]: p["kept"] for p in agr["per_scene"]}
    assert kept == {"a": False, "b": True}          # lowest glint survives


def test_sigma_clamp():
    out = compose([_scene(10.0, 0.05, "a")])
    assert np.abs(out["sigma"] - 0.5).max() < 1e-6  # clamped up to 0.5 m


def test_infer_n_scenes_1_has_no_composite_key(monkeypatch, fake_minio):
    from fastapi.testclient import TestClient
    import main
    import s2_fetch
    import tide

    install_fake_dl_pro(monkeypatch)
    monkeypatch.setattr(
        s2_fetch, "fetch_s2",
        lambda bbox, sd, ed, max_cloud=40, **kw: make_s2(bbox))
    monkeypatch.setattr(
        s2_fetch, "fetch_s2_scenes",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("multi-scene fetch called")))
    monkeypatch.setattr(
        tide, "get_tide_height",
        lambda lat, lon, utc_dt=None: (0.0, {"method": "unavailable",
                                             "reason": "offline test"}))

    client = TestClient(main.app)
    r = client.post("/bathymetry/infer",
                    json={"raster_id": "single", "bbox": BBOX, "n_scenes": 1,
                          "engine": "dl-pro-v3"})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert "composite" not in payload
    missing = PREEXISTING_ROI_KEYS - set(payload)
    assert not missing, f"pre-existing keys lost: {missing}"


def test_infer_n_scenes_3_composites(monkeypatch, fake_minio):
    from fastapi.testclient import TestClient
    import main
    import s2_fetch
    import tide

    install_fake_dl_pro(monkeypatch)
    scenes = [make_s2(BBOX, scene_id="s1", acquired="2024-01-05T10:00:00Z",
                      test_depth=10.0),
              make_s2(BBOX, scene_id="s2", acquired="2024-03-05T10:00:00Z",
                      test_depth=11.0),
              make_s2(BBOX, scene_id="s3", acquired="2024-05-05T10:00:00Z",
                      test_depth=12.0)]
    for s2_, sig in zip(scenes, (0.5, 1.0, 2.0)):
        s2_["test_sigma"] = sig
    monkeypatch.setattr(s2_fetch, "fetch_s2_scenes",
                        lambda *a, **k: scenes)
    monkeypatch.setattr(
        tide, "get_tide_height",
        lambda lat, lon, utc_dt=None: (0.6, {
            "method": "open-meteo/cmems", "utc": utc_dt.isoformat(),
            "tide_range_that_day_m": 0.9}))

    client = TestClient(main.app)
    r = client.post("/bathymetry/infer",
                    json={"raster_id": "multi", "bbox": BBOX, "n_scenes": 3,
                          "engine": "dl-pro-v3"})
    assert r.status_code == 200, r.text
    payload = r.json()
    comp = payload["composite"]
    assert comp["n_scenes_used"] == 3
    assert len(comp["per_scene"]) == 3
    assert payload["scene_id"] == "s1"
    assert payload["tide"]["applied"] is True

    iv = np.array([1 / 0.5 ** 2, 1 / 1.0 ** 2, 1 / 2.0 ** 2])
    z_expect = float(np.sum((np.array([10, 11, 12]) - 0.6) * iv) / iv.sum())
    assert abs(payload["stats"]["mean_m"] - round(z_expect, 2)) < 1e-6
