# Bathymetry VMarch — Calibrated Sentinel-2 SDB

Production satellite-derived-bathymetry (SDB) platform calibrated on UAE coastal
waters. A user draws an ROI on the map (or picks a predefined region), the
backend pulls Sentinel-2 imagery, runs a multi-stage Lyzenga + Stumpf +
UAE-pretrained ensemble with bias correction, and returns a depth grid +
uncertainty map + IHO-S52 chart + 3D view.

Stack: **Flask + React + PyTorch + scikit-learn**, deployed via **Docker on
Railway**.

---

## Architecture at a glance

```
                  ┌──────────────────────────────────────┐
                  │        React frontend (CRA)          │
                  │  Sidebar · MapPanel · ResultsPanel · │
                  │  IHOChartView · ThreeDView (R3F)     │
                  └────────────────┬─────────────────────┘
                                   │  POST /api/...
                  ┌────────────────▼─────────────────────┐
                  │            Flask backend             │
                  │                                      │
                  │  /api/sdb-pro          fast S2 path  │
                  │  /api/very-hr-clustered  → fast path │
                  │  /api/very-hr-mle       multi-scene  │
                  │  /api/sdb-pro/geotiff   downloadable │
                  │                                      │
                  │  ┌──────────────────────────────┐    │
                  │  │ _run_s2_lyzenga_fast()       │    │
                  │  │  • _fetch_s2_cached (3 scn)  │    │
                  │  │  • Lyzenga + Stumpf WLS      │    │
                  │  │  • UAE Cluster+CNN/RF blend  │    │
                  │  │  • Isotonic + IDW bias-corr  │    │
                  │  │  • depth + water + render    │    │
                  │  └──────────────────────────────┘    │
                  └──────────────────────────────────────┘
```

---

## Key features

### Sensors and inputs
* **Sentinel-2 L2A** via Sentinel-Hub (Copernicus CDSE), falling back to
  Google Earth Engine when Sentinel-Hub credentials are unavailable.
  Single-scene SIMPLE mosaicking with `leastCC` ordering, then per-band
  median across **3 sub-windows** of the user date range — kills sun
  glint, cloud edges and ephemeral wakes that bias log-ratio SDB.
* Disk LRU cache at `cache/s2_fast/<sha1>.pkl`, TTL 7 days.
  Background warm-up of the four predefined regions on Flask boot.
* Bundled in-situ surveys (Khalifa Port basin, Old Mussafah Channel,
  Jbel Dhanna) plus ICESat-2 ATL24 bathymetric photons and OCR'd chart
  soundings, fused with source-quality weighting — see [Model registry](#model-registry).
* GEBCO per-tile fall-through (4×4 grid) when in-situ density is low.

### Depth estimation
* **Lyzenga 1985 / 2006** — multi-band log-linear regression on
  deep-water-subtracted reflectance, Tikhonov-regularised WLS.
* **Stumpf 2003 + Caballero & Stumpf 2020** — log-ratio bands `B/G` and
  `B/R`, each independently calibrated.
* **UAE pretrained model** (`backend/uae_pretrained{,_cnn}.py`)
  * 13 spectral features → `StandardScaler` → `KMeans` (6 clusters)
    → per-cluster regressor.
  * Two backends: **Random Forest** and **PyTorch MLP** — runtime
    auto-prefers the CNN when present.
* **Inverse-RMSE ensemble** of Lyzenga + Stumpf_BG + Stumpf_BR.
* **Region-aware blend**:
  * Inside a predefined region → **90 % Lyzenga (in-situ-calibrated) /
    10 % UAE pretrained**.
  * Outside → 50 / 50 transfer-learning blend.
  * With user-supplied depths → shift another 10 % onto Lyzenga.
* **Darkbox in-situ injection** — when the ROI is inside a calibration
  region, the per-cluster CNN heads get a quick fine-tune on the
  region's in-situ pool before prediction, adapting to that region's
  spectral regime per request with no on-disk state change.

### Bias correction (3 stages)
1. **Stage 1 — Isotonic regression** (`sklearn.isotonic`). Non-parametric
   monotonic mapping `pred → true`, captures the shallow / deep
   nonlinearity that a single linear `(a, b)` recalibration misses.
2. **Stage 2 — Per-pixel residual IDW**. cKDTree on training-point
   coordinates; each water pixel's correction is the inverse-distance²
   weighted residual from its nearest training points, fading smoothly
   to the global median residual far from any training point.
3. **Stage 3 — Clamp** the per-pixel correction to ±3 m, final clip
   `[0, MAX_DEPTH_M=25 m]`.

### Multi-scene MLE (whole-year stack)
`POST /api/very-hr-mle` runs the per-scene fast pipeline on **3 / 5 / 7
evenly-spaced ±15 d windows** across a calendar year, then per-pixel
inverse-variance Maximum Likelihood combines them:

```
z_hat[r,c] = Σ_i z_i[r,c] / σ_i²   /   Σ_i 1 / σ_i²
σ_hat[r,c] = 1 / sqrt(Σ_i 1 / σ_i²)
```

`σ_i` is each scene's held-out RMSE clamped to `[0.5, 5] m`. Returns the
standard `sdb-pro` shape plus a green→red **uncertainty PNG** on `σ_hat`,
and a per-scene table with per-scene overlay, outlier flag and tidal
correction summary.

### Water mask
Triple intersection (intersection where the strict mask leaves ≥ 30 %
of the spectral mask, otherwise spectral alone):
1. **Spectral** water/land test on NDWI/MNDWI/NIR/SWIR with cloud /
   shadow / snow rejection and a 3×3 morphological opening.
2. **Coverage** cross-check against global land-polygon data and
   S2 brightness/blue-fraction, catching reclaimed land and wet sand.
3. **Imagery-evidence NDWI cut** — native-resolution NDWI from the same
   Sentinel-2 fetch flags land the vector land sources haven't mapped
   yet (e.g. recently reclaimed port pads), safety-gated against a
   sounding-retention check.
4. **Morphological opening** (3×3) — removes ship specks, wakes,
   floating debris.

### Region calibration (auto on click)
Predefined regions carry a calibrated standard Sentinel-2 date in
`backend/calibration_data/per_region_best_dates.json`. Clicking a
predefined region in the Sidebar sets both the ROI and the date
automatically.

### Frontend
* **Sidebar** — ROI panel, predefined-region presets, Sentinel-2 date
  picker, Multi-scene MLE panel, Planet VHR backup status.
* **MapPanel** — Mapbox base map + Leaflet draw + result-card popup,
  per-scene overlay swap, distance-measurement tool.
* **ResultsPanel** — Advanced tab (depth distribution histogram),
  IHO chart tab (depth points over an IHO S-52 palette with isobath
  contours), MLE scene group, tidal-correction disclosure.
* **ThreeDView** — `@react-three/fiber` real-time WebGL seabed/water
  scene with orbit controls.
* **IHOChartView** — marching-squares isobaths + S-52 sounding labels.

### Hardening
* **40 km² ROI cap** server-side on every extraction endpoint.
* **`_json_safe()`** scrubs NaN / Inf and numpy scalars from every
  response so `JSON.parse` on the frontend never trips.
* **GeoTIFF download** is opt-in via `POST /api/sdb-pro/geotiff` —
  keeps the main JSON response small and avoids proxy body-size issues.
* Every served depth product passes through a universal land-cut
  (global coastline vector + imagery-evidence NDWI) before rendering
  or export.

---

## Model registry

Production inference weights (`backend/models/uae_clustered_{rf,cnn}.pkl`)
are versioned under `backend/models/registry/v<N>/`, each version carrying
a `model_card.json` with:

* Training-data provenance — per-source point counts, bounding boxes and
  a content fingerprint (`sha256` of the sorted point set) — **never raw
  coordinates**.
* Held-out spatial-block cross-validation metrics per calibration site
  (RMSE / bias / R² / decile slope), not in-sample numbers.
* Weight file checksums, CATZOC tier and IHO Order mapping derived from
  the held-out metrics.

`backend/models/registry/CURRENT` points at the active version; the
predefined-region metadata the app serves (sounding counts, bounding
boxes, datum) is read from that version's `model_card.json`, not by
re-parsing raw survey files at request time.

Regenerating a version: `scripts/train_uae_pretrained.py` and
`scripts/train_uae_cnn.py` retrain from the fused reference pool;
`scripts/retrain_uae_pretrained.py` warm-starts an existing version and
rejects the candidate if any previously-certified site's accuracy
regresses beyond a fixed tolerance; `scripts/compute_registry_metrics.py`
computes the held-out spatial-block metrics that go into the card.

---

## Running locally

```bash
# 1. Backend (Python venv)
source .venv/bin/activate
source .env                    # SH_CLIENT_ID, SH_CLIENT_SECRET, GEMINI_API_KEY
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 600 backend.app:app

# 2. Frontend (separate terminal)
cd frontend
REACT_APP_API_URL=http://localhost:5000 PORT=3000 npx react-scripts start

# Open  http://localhost:3000
```

Regenerate the production model registry (only needed when the reference
data pool changes):

```bash
source .venv/bin/activate && source .env
python scripts/train_uae_pretrained.py   # → backend/models/uae_clustered_rf.pkl
python scripts/train_uae_cnn.py          # → backend/models/uae_clustered_cnn.pkl
python backend/models/build_registry_v1.py
```

Run the per-region date sweep (selects the best Sentinel-2 date for
each calibration region):

```bash
python scripts/all_regions_date_sweep.py
# writes backend/calibration_data/per_region_best_dates.json
```

---

## Deploying to Railway

`railway.toml` + `Dockerfile` build:

* Frontend stage: `node:20-slim` → `npm run build` → static bundle.
* Backend stage: `python:3.11-slim` + `chromium` (Playwright) +
  `gdal-dev` / `geos-dev` / `proj-dev`.
* `cache/` is empty on first boot; a background warm-up thread
  pre-fetches the predefined region scenes so the first user click
  lands on a warm cache.

---

## API summary

| Endpoint | Purpose |
|---|---|
| `POST /api/sdb-pro` | Fast Sentinel-2 SDB on a single ±10 d window |
| `POST /api/sdb-pro/geotiff` | Downloadable GeoTIFF for the same ROI / window |
| `POST /api/very-hr-clustered` | Routes through the fast path (drop-in) |
| `POST /api/very-hr-mle` | Multi-scene whole-year MLE stack (3/5/7) |
| `POST /api/extract` | Legacy unified pipeline (CNN + Caballero etc.) |
| `POST /api/export` | CSV / GeoJSON / GeoTIFF export, land-cut applied |
| `GET  /api/planet-backup/status` | Planet VHR backup imagery status |
| `GET  /api/health` | Backend health + capability flags |

All depth-grid endpoints return:

```json
{
  "method": "Calibrated bathymetry",
  "stats":   {"mean_depth": ..., "max_depth": ..., "min_depth": ...,
              "std_depth": ..., "grid_points": ..., "resolution_m": ...,
              "confidence": ..., "n_train": ...},
  "metrics": {"rmse_m": ..., "bias_m": ..., "r2": ...,
              "s44_1a_pct": ..., "s44_order2_pct": ..., "n_test": ...},
  "tide_correction": {"applied": false, "source": "EOT20", "height_m": ...},
  "depth_png_b64": "...", "raster_bounds": [[s,w],[n,e]],
  "uncertainty_png_b64": "...",
  "interpolated_points": [{"lat": ..., "lon": ..., "depth": ..., "photon_class": "interpolated"}],
  "contours": [{"depth": 5.0, "coords": [...]}]
}
```

`metrics` keys are populated only when held-out test points exist.
`uncertainty_png_b64` and the per-scene table are populated only by
`/api/very-hr-mle`.

---

## Layout

```
backend/
├── app.py                          Flask app, all endpoints
├── lyzenga_sliderule.py            Lyzenga + Stumpf + SlideRule estimator
├── uae_pretrained.py               KMeans + RandomForest cluster model
├── uae_pretrained_cnn.py           KMeans + per-cluster MLP variant
├── icesat2_bathy.py                ATL03 / SlideRule wrapper
├── osm_land_mask.py                Global coastline / land vector cut
├── ndwi_mask.py                    Imagery-evidence NDWI land cut
├── wave_bathy.py                   Almar wave-dispersion bathymetry
├── monitoring.py                   SQLite-backed change detection
└── models/
    ├── uae_clustered_rf.pkl
    ├── uae_clustered_cnn.pkl
    ├── registry_schema.py          model_card.json validator
    ├── registry_hash.py            Point-set content fingerprint
    ├── registry_privacy_lint.py    Coordinate-leak lint
    └── registry/
        ├── CURRENT
        └── v1/model_card.json

frontend/src/
├── App.js
├── components/
│   ├── Sidebar.js                  ROI / date / MLE controls
│   ├── MapPanel.js                 Leaflet + result popup
│   ├── ResultsPanel.js             Advanced + IHO + Pro tabs
│   ├── IHOChartView.js             Marching-squares isobaths
│   ├── ThreeDView.js               R3F 3D scene
│   ├── MleSceneGroup.js            Per-scene MLE result group
│   ├── PlanetBackupPanel.js        Planet VHR backup status
│   └── TideChip.js                 Shared tidal-correction chip

scripts/
├── train_uae_pretrained.py         RF training driver
├── train_uae_cnn.py                CNN training driver
├── retrain_uae_pretrained.py       Warm-start retrain + forgetting guard
├── compute_registry_metrics.py     Held-out spatial-block CV metrics
├── khalifa_date_sweep.py
└── all_regions_date_sweep.py

backend/calibration_data/
└── per_region_best_dates.json      Per-region calibrated standard dates

cache/                              Disk LRU (gitignored)
└── s2_fast/<sha1>_n3.pkl           Cached 3-scene S2 medians
```

---

## References

* **Lyzenga, D. R.** (1985). Shallow-water bathymetry using combined
  lidar and passive multispectral scanner data. *IJRS* 6(1).
* **Lyzenga, D. R. et al.** (2006). Multispectral bathymetry using a
  simple physically based algorithm. *IEEE TGARS* 44(8).
* **Stumpf, R. P. et al.** (2003). Determination of water depth with
  high-resolution satellite imagery over variable bottom types.
  *L&O* 48(1).
* **Caballero, I. & Stumpf, R. P.** (2020). Towards routine mapping of
  shallow bathymetry in environments with variable turbidity.
  *Remote Sensing* 12(3), 451.
* **Hedley, J. D. et al.** (2005). Simple and robust removal of sun
  glint for mapping shallow-water benthos. *IJRS* 26(10).
* **Almar, R. et al.** Wave-dispersion bathymetry from S2 inter-band
  time-offsets (`backend/wave_bathy.py`).
* **Parrish, C. E. et al.** (2025). ICESat-2 ATL24 bathymetric product.
* **IHO S-44** (2020) — Total Vertical Uncertainty for hydrographic
  surveys; used for Order 1A / Order 2 pass-rate scoring.
