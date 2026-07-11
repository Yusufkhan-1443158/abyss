"""Shared offline fixtures: fake MinIO, fake Sentinel-2 scenes and a fake
DL-Pro engine so /bathymetry/infer runs with no network and no torch."""
import io
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeMinioResponse:
    def __init__(self, blob):
        self._blob = blob

    def read(self):
        return self._blob

    def close(self):
        pass

    def release_conn(self):
        pass


class FakeMinio:
    def __init__(self):
        self.store = {}

    def bucket_exists(self, bucket):
        return True

    def make_bucket(self, bucket):
        pass

    def put_object(self, bucket, key, data, length=None, content_type=None):
        self.store[(bucket, key)] = data.read()

    def get_object(self, bucket, key):
        if (bucket, key) not in self.store:
            raise FileNotFoundError(f"{bucket}/{key}")
        return FakeMinioResponse(self.store[(bucket, key)])


@pytest.fixture
def fake_minio(monkeypatch):
    import main
    fm = FakeMinio()
    monkeypatch.setattr(main, "_minio", lambda: fm)
    return fm


def make_s2(bbox, H=40, W=50, scene_id="S2A_TEST_SCENE",
            acquired="2024-04-16T10:00:00Z", cloud=3.2, nir_dn=100.0,
            test_depth=10.0):
    from rasterio.transform import from_bounds
    w, s, e, n = bbox
    green = np.full((H, W), 2000.0, np.float32)
    nir = np.full((H, W), float(nir_dn), np.float32)
    ndwi = ((green - nir) / (green + nir + 1e-6)).astype(np.float32)
    return {
        "blue": np.full((H, W), 1500.0, np.float32), "green": green,
        "red": np.full((H, W), 1000.0, np.float32), "nir": nir,
        "coastal": np.full((H, W), 1400.0, np.float32),
        "ndwi": ndwi, "water_mask": np.ones((H, W), bool),
        "width": W, "height": H,
        "transform": from_bounds(w, s, e, n, W, H),
        "bounds": [w, s, e, n], "crs": "EPSG:4326",
        "scene_id": scene_id, "cloud_cover": cloud, "acquired": acquired,
        "test_depth": float(test_depth),
        "test_sigma": None,
    }


def install_fake_dl_pro(monkeypatch):
    """Inject a torch-free dl_pro_engine whose depth/sigma come from the
    scene's test_depth/test_sigma fields."""
    mod = types.ModuleType("dl_pro_engine")

    def predict_dl_pro(s2, bbox=None, **kw):
        H, W = s2["red"].shape
        depth = np.full((H, W), s2.get("test_depth", 10.0), np.float32)
        sig = s2.get("test_sigma")
        sigma = np.full((H, W), 0.8 if sig is None else float(sig), np.float32)
        return {
            "depth": depth, "sigma": sigma, "sigma_eff": None,
            "water_mask": np.ones((H, W), bool),
            "calibration": {"alpha": 0.0, "beta": 1.0, "r2": 0.9,
                            "n_refs": 0, "source": "baseline"},
            "iho_s44": {"Order_1a": 87.0},
            "model_meta": {"version": "3.0-test",
                           "holdout_metrics": {"rmse": 0.9}},
        }

    mod.predict_dl_pro = predict_dl_pro
    monkeypatch.setitem(sys.modules, "dl_pro_engine", mod)
    return mod


PREEXISTING_ROI_KEYS = {
    "raster_id", "depth_rgb_key", "depth_raw_key", "depth_bucket", "crs",
    "bbox", "width", "height", "units", "max_depth_m", "stats", "grid",
    "profile", "model", "model_version", "iho_s44_pct", "holdout_metrics",
    "calibration", "scene_id", "cloud_cover", "acquired",
}
