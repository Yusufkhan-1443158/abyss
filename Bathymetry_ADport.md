# Bathymetry_ADport

Satellite-derived bathymetry (SDB) for AD Ports — the Bathymetry from Space science
stack delivered through the Abyss platform GUI.

## What this project is
- **Basis / GUI**: Abyss (`github.com/Yusufkhan-1443158/abyss`, branch `BathymetryVJuly`)
  — image ingest → AI depth → report. UI is Abyss's own (Stella theme); no UI imported
  from the science stack.
- **Science / engine**: the Bathymetry_VMarch production method (as deployed on Railway):
  Sentinel-2 → Lyzenga/Stumpf spectral features → UAE-clustered RF+CNN ensemble
  (registry v1, trained on 542,874 pts / 8 UAE regions incl. ICESat-2 ATL24 class-40
  photons and i-Boating soundings) → bias correction → land/ocean cut (OSM coastline +
  NDWI) → tide correction (per-scene) → multi-scene inverse-variance MLE composite.
- **Validation**: IHO S-44 statistics (Special/1a/2 pass %, CATZOC/order label,
  RMSE/MAE/MedAE/bias/R²/Pearson) against user soundings; honest caveats
  (detection-capability, TPU) surfaced in report.
- **Calibration**: "calibrate to my points" — robust local fit (Huber + residual field,
  80/20 holdout with before/after RMSE) emitting a new derived layer with provenance.

## Repositories & branches
| What | Where |
|---|---|
| Delivery branch (GUI + engine, admin PR target) | `Yusufkhan-1443158/abyss` → `BathymetryVJuly` |
| Science source (clean single commit) | `Wassim1313/Bathymetry_VMarch` → `production` |
| Full development history | `Wassim1313/Bathymetry_VMarch` → `adports/doc-compliance` |

## Local test
```bash
cd ~/abyss
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
# http://localhost:9088/bathymetry-studio.html  (dev auto-login)
```

## Key numbers (verified)
- Registry v1 held-out spatial-block CV: Khalifa 1.61 m RMSE, Old Mussafah 0.94 m (CATZOC B).
- Multi-scene composite reproduces analytic inverse-variance mean to 1e-6.
- Calibration holdout demo: RMSE 0.965 → 0.408 m, bias −0.882 → +0.029 m.
- Planetary-Computer S2 harmonization offset (+1000 DN, baseline ≥04.00) detected and
  corrected — without it absolute depths are wrong by construction.

## vmarch-core — the REAL platform computes the products
The port alone was judged not acceptable: the products must come from the actual
Bathymetry_VMarch code. The full platform checked in at `BathymetryVJuly/` now runs as
its own service (**vmarch-core**, gunicorn `backend.app:app` on :8080, internal only)
and the studio's "live platform" method cards delegate to its REAL endpoints through a
thin adapter in `bathymetry-service` (`vmarch_core_client.py`):

| Studio method card | engine id | vmarch-core endpoint |
|---|---|---|
| VMarch SDB Standard | `vmarch-core-standard` | `POST /api/sdb-pro` (`include_geotiff:true`) |
| Very-HR Clustered | `vmarch-core-clustered` | `POST /api/very-hr-clustered` (S2 fast path) |
| Multi-Scene MLE | `vmarch-core-mle` | `POST /api/very-hr-mle` (n_scenes 3/5/7, year = end-date year) → composite GeoTIFF via `/downloads/` |
| Wave-Kinematics SDB | `vmarch-core-wave` | `POST /api/s2shores-bathymetry` (tested working here via GEE L1C; scenes → `top_k`) |
| VMarch SDB (port) | `vmarch-sdb` | — vendored `vmarch_engine.py`, offline fallback (auto-used with `engine_fallback` provenance when vmarch-core is unreachable) |
| DL-Pro v3 | `dl-pro-v3` | — vendored fallback |
| ICESat-2 Photon | — | honest disabled card: creds OK and granule search verified in this env, but the product is along-track soundings (not a survey grid) — not wired to the raster pipeline |

The adapter ingests the platform's own GeoTIFF (float32, EPSG:4326, nodata −9999,
positive-down) into the normal raster+report pipeline. Method labels, metrics,
tide-correction disclosure, augmentation and water-mask provenance are passed through
**verbatim** (`stats.vmarch_core` block in the report; "Computed by … platform vX.Y"
metadata row). Abyss applies NO second tide correction / masking / calibration on top.

Minimal, justified changes inside `BathymetryVJuly/` (the platform folder):
1. `backend/requirements.txt`: `+ earthengine-api` — the platform's own GEE imagery
   fallback needs it and the Sentinel-Hub creds are 401 in this env.
2. `backend/app.py`: `/api/sdb-pro` and `/api/very-hr-clustered` accept
   `include_geotiff` (+ `resolution_m` on sdb-pro) from the request body so a
   service-to-service caller gets the georeferenced product inline (no second run).

### vmarch-core environment (RunPod / any deployment)
Local stack: gitignored `secure-config/vmarch-core.env` (create it from your
Bathymetry_VMarch `.env`). The dev overlay additionally mounts `~/.config/earthengine`
for GEE user credentials. Variables:

| Var | Purpose |
|---|---|
| `SH_CLIENT_ID` / `SH_CLIENT_SECRET` | Sentinel Hub (CDSE) S2 imagery — primary fetch |
| `S2_GEE_FALLBACK=1` | enable the GEE imagery fallback when SH is unavailable |
| `EE_PROJECT` (default `ee-wassimehtp`) | GEE project |
| `GEE_SERVICE_ACCOUNT_JSON` / `GOOGLE_APPLICATION_CREDENTIALS` | GEE service-account auth for headless deployments (base64 accepted) |
| `MAPBOX_TOKEN` | VHR imagery + VHR water-mask refinement |
| `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | NASA EarthData (ICESat-2 / SlideRule augmentation) |
| `GEMINI_API_KEY` | chart digitisation / i-Boating OCR features |
| `PLANET_API_KEY` | Planet backup imagery (optional) |
| `TIDE_REDUCE_TO_MSL` | default off — tide disclosure-only, exactly like production |

RunPod single-container image: `/opt/venv-vmarch` env + `/app/vmarch-core` code +
supervisord `[program:vmarch-core]` (API-only — no React build; the studio reaches it
through the adapter). `VMARCH_CORE_URL=http://127.0.0.1:8080`.

## Status / next
- [x] Full-fidelity VMarch method port into the Abyss ROI path — kept as the honest
      offline "(port)" fallback engine.
- [x] vmarch-core service: the actual platform code computes the studio products.
- [ ] Admin PR review/merge `BathymetryVJuly` → `main`, then RunPod rebuild.
- [ ] Planet imagery harvest (blocked: Insights OAuth / Orders provisioning).
- [ ] Renew CDSE Sentinel-Hub credentials (currently 401 → GEE fallback path in use).
