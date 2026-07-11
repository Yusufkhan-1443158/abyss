"""SCI-R2 acceptance: IHO S-44 validation math on a synthetic tilted plane."""
import numpy as np
import pytest

from conftest import FakeMinio  # noqa: F401  (ensures sys.path setup)

from validation import validate_points, iho_order_from_p95, ValidationError

BBOX = [54.60, 24.78, 54.61, 24.79]
NX, NY = 100, 100
SIGMA = 0.2
N_OBS = 500


def _plane_points():
    lons = np.linspace(BBOX[0], BBOX[2], NX)
    lats = np.linspace(BBOX[1], BBOX[3], NY)
    predicted, truth = [], {}
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            d = 2.0 + 16.0 * j / (NX - 1)
            predicted.append({"lat": float(lat), "lon": float(lon), "depth": d})
            truth[(i, j)] = d
    return predicted, truth, lats, lons


def test_synthetic_plane_stats():
    rng = np.random.default_rng(42)
    predicted, truth, lats, lons = _plane_points()
    idx = rng.choice(NX * NY, size=N_OBS, replace=False)
    observed = []
    for k in idx:
        i, j = divmod(int(k), NX)
        observed.append({"lat": float(lats[i]), "lon": float(lons[j]),
                         "depth": truth[(i, j)] + float(rng.normal(0, SIGMA))})

    res = validate_points(predicted, observed, resolution_m=10.0)

    assert res["n_pairs"] == N_OBS
    assert 0.15 < res["rmse"] < 0.25, res["rmse"]
    assert abs(res["bias"]) < 0.05, res["bias"]
    assert res["iho_orders"]["order1a"]["pass_pct"] > 95
    assert res["catzoc"] in ("A1", "A2/B", "C")
    assert res["order_label"] is not None
    assert res["r2"] > 0.99
    assert res["detection_capability_note"] is not None  # 1a >= 95 %
    assert len(res["residual_histogram"]) == 25
    assert res["spatial_residuals"]["h"] == 12
    assert res["class_confusion"]["agreement_pct"] > 90
    assert len(res["stratified"]) >= 3
    assert res["match_distance_stats_m"]["max"] < 1.0


def test_datum_mismatch_flagged():
    predicted, truth, lats, lons = _plane_points()
    observed = [{"lat": float(lats[i]), "lon": float(lons[j]),
                 "depth": truth[(i, j)]}
                for i, j in [(0, 0), (10, 10), (20, 20), (30, 30), (40, 40),
                             (50, 50), (60, 60), (70, 70), (80, 80), (90, 90)]]
    res = validate_points(predicted, observed, observed_vertical_datum="MSL")
    assert res["datum_mismatch"] is True
    assert res["bias_note"] is not None


def test_iho_order_from_p95_tiers():
    assert iho_order_from_p95(0.2, 5.0)[1] == "A1"
    assert iho_order_from_p95(0.4, 5.0)[1] == "A2/B"
    assert iho_order_from_p95(0.8, 5.0)[1] == "C"
    assert iho_order_from_p95(3.0, 5.0)[0] == "Below Order 2"


def test_empty_observed_raises():
    with pytest.raises(ValidationError):
        validate_points([{"lat": 0.0, "lon": 0.0, "depth": 5.0}], [])


def test_validate_endpoint_400_and_200(fake_minio):
    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    r = client.post("/bathymetry/validate",
                    json={"predicted": [{"lat": 0, "lon": 0, "depth": 5}],
                          "observed": []})
    assert r.status_code == 400

    predicted, truth, lats, lons = _plane_points()
    observed = [{"lat": float(lats[i]), "lon": float(lons[i]),
                 "depth": truth[(i, i)]} for i in range(0, 100, 5)]
    r = client.post("/bathymetry/validate",
                    json={"predicted": predicted, "observed": observed})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_pairs"] == len(observed)
    assert body["rmse"] == 0.0
    assert body["iho_orders"]["order1a"]["pass_pct"] == 100.0
