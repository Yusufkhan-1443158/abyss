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

## Status / next
- [ ] Full-fidelity VMarch method port into the Abyss ROI path (Lyzenga WLS + Stumpf +
      region-aware cluster blend + 3-stage bias correction), golden-output-verified
      against the Railway implementation.
- [ ] Admin PR review/merge `BathymetryVJuly` → `main`, then RunPod rebuild.
- [ ] Planet imagery harvest (blocked: Insights OAuth / Orders provisioning).
