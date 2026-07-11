# Integrate the real Bathymetry_VMarch SDB engine into `infer_depth()`

## What changed

This PR replaces the remaining stub behaviour of the bathymetry service with the
production Satellite-Derived Bathymetry engine, vendored self-contained under
`bathymetry-service/sdb_engine/`. Uploaded rasters are now inferred from **their
own pixels**; the drawn-ROI Sentinel-2 path (DL-Pro v3) is unchanged.

### `bathymetry-service/`
- **New `sdb_engine/` package** — feature engineering (13-feature Lyzenga/Stumpf/NDWI
  stack), the UAE-calibrated cluster ensemble (per-cluster Random Forest + MLP),
  the NDWI land cut, the depth colour ramp, and the `infer_depth()` entry point.
  Model bundles (`models/uae_clustered_rf.pkl`, `models/uae_clustered_cnn.pkl`)
  load via a module-remapping unpickler, so no training package is required.
- **`main.py`** — `POST /bathymetry/infer` now honours `source_bucket`/`source_key`
  (previously accepted but ignored): the source raster is read from MinIO and
  routed by its band content:
  - **Multispectral (B/G/R + NIR, Sentinel-2-like)** → calibrated UAE ensemble,
    NDWI land cut, 0–25 m clip, per-pixel uncertainty channel (per-cluster
    calibration RMSE ⊕ RF/MLP disagreement) exposed in `grid.sigma`.
  - **Plain RGB (3 bands)** → Stumpf blue/green log-ratio pseudo-depth
    (percentile-stretched), blue-water-index land cut, explicitly returned as
    `calibrated: false` / low confidence — never presented as metric depth.
  - Band identification by band descriptions when present, else position
    (3 = R,G,B; 4 = B,G,R,NIR; 5+ = coastal,B,G,R,NIR); 0–1 reflectance and
    8-bit inputs are rescaled to the S2 DN scale.
  - Product writing/stats/profile/grid factored into a shared helper used by
    both paths; GeoTIFF products keep the source CRS; `bbox` is always reported
    in EPSG:4326.
- **Dependencies** — `numpy>=2,<3`, `scikit-learn>=1.7,<1.8` (the environment
  the bundles were pickled in) and torch `2.2.2 → 2.7.0` (numpy-2-compatible;
  `dl_pro_engine` already loads with `weights_only=False`).

### `fastapi/`
- `services/bathymetry_tasks.py` — `run_bathymetry` posts
  `source_bucket="raw"` + the raster's `minio_raw_path`, so ingest-triggered
  inference uses the uploaded scene (bbox is kept as georeferencing fallback);
  method/confidence recorded in the derived layer's metadata and the job row.
- `services/report_builder.py` — reports are method-aware and honest: model +
  method + calibration domain/fit + datum rows in Survey Metadata, a matching
  Methodology & Caveats section per method, and an explicit uncalibrated
  warning in the Executive Summary for RGB fallback products.

## Model provenance

| Item | Value |
|---|---|
| Engine | UAE cluster ensemble (`uae-sdb-ensemble`, registry v1) |
| Members | K-means (6 optical regimes) → per-cluster Random Forest + per-cluster MLP (13→64→64→32→1) |
| Features | blue, green, red, coastal, NIR reflectances; Lyzenga log-bands + depth-invariant index; Stumpf B/G + B/R log-ratios; NDWI |
| Training data | ~443k sounding-matched Sentinel-2 pixels, 8 UAE regions (Khalifa Port, Old Mussafah, Jbel Dhanna, Bu Tinah, Yas Lagoon, Ras Ghanada, Eastern Mangroves, Lulu Lagoon) |
| Fit (in-sample) | R² 0.978, RMSE 0.735 m |
| Validity | UAE-calibrated; **outside the UAE the output is indicative only** |
| Depth range | 0–25 m (clamped) |
| Datum | Calibration soundings' survey datum; not tide-corrected |
| RGB fallback | Stumpf (2003) log-ratio pseudo-depth, uncalibrated, relative only |

## How to test

```bash
# Unit tests (synthetic 4-band coastal scene + synthetic RGB scene):
cd bathymetry-service && pytest tests/test_infer_depth.py -v

# Service image builds:
docker build -t abyss-bathymetry bathymetry-service/

# Full pipeline (dev): upload a multispectral GeoTIFF; the report's Survey
# Metadata should show "uae-sdb-ensemble registry-v1" + calibration rows.
# Upload a plain RGB image; the report must carry the uncalibrated warning.
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
curl -F file=@scene.tif "http://localhost:9088/api/rasters/upload?name=Scene"
```

Unit results on the synthetic gradient scene (0.5→20 m, land strip):
4-band → calibrated ensemble, land strip all-NaN, depths within 0–25 m,
column-mean Spearman ρ vs true gradient 0.87, sigma channel populated;
RGB → `stumpf-log-ratio`/uncalibrated, ρ 0.82, land cut applied.
