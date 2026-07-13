"""ROI engine selection: the full VMarch chain is the default, DL-Pro stays
available on explicit request, and local calibration emits a new product."""
import numpy as np

from conftest import make_s2, install_fake_dl_pro

BBOX = [54.30, 24.40, 54.40, 24.50]


def _patch_fetch(monkeypatch):
    import s2_fetch
    monkeypatch.setattr(
        s2_fetch, "fetch_s2",
        lambda bbox, sd, ed, max_cloud=40, **kw: make_s2(bbox))


def test_roi_default_engine_is_vmarch(monkeypatch, fake_minio):
    from fastapi.testclient import TestClient
    import main

    _patch_fetch(monkeypatch)
    monkeypatch.setenv("TIDE_CORRECTION", "0")
    client = TestClient(main.app)
    r = client.post("/bathymetry/infer",
                    json={"raster_id": "eng-default", "bbox": BBOX})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["model"] == "vmarch-sdb"
    assert p["model_version"] == "registry-v1"
    assert "VMarch SDB (Lyzenga+Stumpf+UAE ensemble" in p["model_label"]
    assert p["engine"] == "vmarch-sdb"
    assert p["engine_requested"] == "vmarch-sdb"
    calib = p["calibration"]
    assert calib["uae_blend"]["used"] is True
    # BBOX contains the Old Mussafah preset → region-aware 90/10 blend and
    # bundled in-situ anchors drive the 3-stage bias correction.
    assert calib["uae_blend"]["inside_predefined"] is True
    assert calib["uae_blend"]["w_uae"] == 0.10
    assert calib["refs"]["in_situ"] >= 30
    assert calib["bias_correction"]["applied"] is True
    assert p["holdout_metrics"].get("n", 0) >= 5
    assert np.isfinite(np.array(
        [v for row in p["grid"]["depth"] for v in row if v is not None])).all()


def test_roi_engine_alias_maps_to_vmarch(monkeypatch, fake_minio):
    from fastapi.testclient import TestClient
    import main

    _patch_fetch(monkeypatch)
    monkeypatch.setenv("TIDE_CORRECTION", "0")
    client = TestClient(main.app)
    r = client.post("/bathymetry/infer",
                    json={"raster_id": "eng-alias", "bbox": BBOX,
                          "engine": "uae-sdb-ensemble"})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "vmarch-sdb"


def test_roi_explicit_dl_pro_still_works(monkeypatch, fake_minio):
    from fastapi.testclient import TestClient
    import main

    install_fake_dl_pro(monkeypatch)
    _patch_fetch(monkeypatch)
    monkeypatch.setenv("TIDE_CORRECTION", "0")
    client = TestClient(main.app)
    r = client.post("/bathymetry/infer",
                    json={"raster_id": "eng-dlpro", "bbox": BBOX,
                          "engine": "dl-pro-v3"})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["model"] == "dl-pro-v3"
    assert p["engine"] == "dl-pro-v3"
    assert abs(p["stats"]["mean_m"] - 10.0) < 1e-2

    r = client.post("/bathymetry/infer",
                    json={"raster_id": "eng-bad", "bbox": BBOX,
                          "engine": "nope"})
    assert r.status_code == 400


def test_calibrate_emits_new_product_with_holdout(monkeypatch, fake_minio):
    from fastapi.testclient import TestClient
    from rasterio.transform import from_bounds
    import main

    H, W = 60, 80
    w, s, e, n = BBOX
    rng = np.random.default_rng(7)
    depth = (5.0 + 10.0 * rng.random((H, W))).astype(np.float32)
    transform = from_bounds(w, s, e, n, W, H)
    main._put_geotiff(fake_minio, "src-prod/depth.tif", depth, transform, 1,
                      "float32", nodata=-9999.0)

    # truth = prediction + 1.2 m constant bias -> calibration must recover it
    rows = rng.integers(2, H - 2, 300)
    cols = rng.integers(2, W - 2, 300)
    from rasterio.transform import xy
    xs, ys = xy(transform, rows, cols)
    observed = [{"lat": float(y), "lon": float(x),
                 "depth": float(depth[r, c] + 1.2)}
                for r, c, x, y in zip(rows, cols, xs, ys)]

    client = TestClient(main.app)
    r = client.post("/bathymetry/calibrate",
                    json={"raster_id": "src-prod", "observed": observed})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["raster_id"] != "src-prod"
    assert p["depth_raw_key"].endswith("/depth.tif")
    assert ("depth", "src-prod/depth.tif") in fake_minio.store  # original kept
    cal = p["calibration_local"]
    assert cal["provenance"] == "local_points"
    assert cal["n_train"] + cal["n_holdout"] == cal["n_points"]
    before = cal["holdout"]["before"]["rmse_m"]
    after = cal["holdout"]["after"]["rmse_m"]
    assert before > 1.0
    assert after < before
    assert after < 0.5

    r = client.post("/bathymetry/calibrate",
                    json={"raster_id": "src-prod",
                          "observed": observed[:4]})
    assert r.status_code == 400
