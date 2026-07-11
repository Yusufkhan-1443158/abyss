# Deploying Bathymetry_VMarch (SDB web app) to Railway

Flask backend (`backend.app:app`) + React frontend (CRA, served as static files by
Flask from `frontend/build/`). Single container, single public port.

## What is already wired (no changes needed)

| Concern | Where | Status |
|---|---|---|
| Production server | `Dockerfile` CMD → `gunicorn --bind 0.0.0.0:8080 --workers 2 --timeout 600 backend.app:app` | ready |
| PORT env var | `backend/app.py` (`app.run(port=int(os.environ.get('PORT',8080)))`) + Dockerfile `ENV PORT=8080`. Railway injects `$PORT`; gunicorn binds 8080 — set the Railway service port to 8080. | ready |
| Frontend build | Dockerfile multi-stage (`node:20-slim` → `npm run build` → copied to `/app/frontend/build`) | ready |
| Static + SPA routing | `backend/app.py` auto-detects `frontend/build`; catch-all `/<path:path>` + 404 handler serve `index.html` for client-side routes | ready |
| Healthcheck | `GET /api/health` (200 JSON); `railway.toml` `healthcheckPath=/api/health` | ready |
| Builder | `railway.toml` `builder=DOCKERFILE` | ready |
| Secrets | none committed; `.env` gitignored; `ee-*.json`/`*-service-account.json` gitignored | ready |

## Smoke test (verified locally 2026-06-10, gunicorn 1 worker)

```
/api/health  -> 200 {"v":"9.3","sklearn":true,"scipy":true,"cnn_unet":true,...,"max_depth_m":25.0}
/            -> 200 text/html  (React index, <title>Bathymetry From Space</title>)
/results     -> 200            (SPA deep link served via catch-all)
/static/js/main.<hash>.js -> 200, 1.5 MB bundle
/api/nope    -> 404 {"error":"Not found"}
```

## Environment variables to set in the Railway service

REQUIRED for full functionality (the app boots without them but falls back / disables features):

| Var | Purpose | Notes |
|---|---|---|
| `EARTHDATA_USERNAME` | NASA EarthData (ICESat-2 / ATL via SlideRule) | |
| `EARTHDATA_PASSWORD` | NASA EarthData | |
| `SH_CLIENT_ID` | Sentinel-Hub OAuth | creds expired → app uses GEE fallback; safe to omit |
| `SH_CLIENT_SECRET` | Sentinel-Hub OAuth | secret — Railway var only, never commit |
| `MAPBOX_TOKEN` | Mapbox satellite tiles (VHR path) | public `pk.` token |
| `GEMINI_API_KEY` | Gemini OCR (i-Boating sounding harvest) | secret |

Platform / optional:

| Var | Default | Notes |
|---|---|---|
| `PORT` | 8080 | Railway sets this; keep service port = 8080 |
| `PYTHONUNBUFFERED` | 1 | logging |
| `APP_ENV` | production | |
| `DATA_DIR` | /tmp/bathymetry_data | writable scratch |
| `PLANET_API_KEY` | — | Planet Data API (PSScene search / future VHR harvest); secret — Railway var only |

## Client tier (AD Ports / hydrographic build)

`REACT_APP_CLIENT_TIER=hydro` is a **build-time** (not runtime) variable: it must be
present when `npm run build` runs inside the Docker build. It un-hides the IHO
S-52 chart tab, the S-44 validation tab, and CATZOC/S-44 terminology for the
hydrographic client (see `frontend/src/components/internalMode.js`,
`SHOW_IHO_SURFACE`). Without it the app builds in the redacted consumer tier.

On Railway: set `REACT_APP_CLIENT_TIER=hydro` as a service variable **and** make
sure the Dockerfile forwards it into the frontend build stage:

```dockerfile
ARG REACT_APP_CLIENT_TIER
ENV REACT_APP_CLIENT_TIER=$REACT_APP_CLIENT_TIER
```

(add the two lines above to the node build stage before `RUN npm run build` if not
already present). Verify after deploy: the Results panel must show the
`IHO CHART` and `VALIDATION` tabs and the CATZOC verdict banner.

GEE: if using a service account, mount the key via a Railway file/secret and point
`GOOGLE_APPLICATION_CREDENTIALS` at it. Do NOT commit the JSON (gitignored as `ee-*.json`).

## Deploy

The Railway CLI is installed (`~/.nvm/.../bin/railway`) but **NOT authenticated**
(`railway whoami` → Unauthorized). After `railway login` in a browser session, the
one-command deploy from the repo root:

```bash
cd /home/wassi/Bathymetry_VMarch
railway login                 # interactive, one-time
railway init                  # or: railway link   (to an existing project)
railway up                    # builds the Dockerfile and deploys
# then set the env vars above:  railway variables --set SH_CLIENT_SECRET=... (repeat)
# verify:  curl https://<service>.up.railway.app/api/health
```

Alternatively connect the GitHub repo (`Wassim1313/Bathymetry_VMarch`, branch
`deploy/railway` or `main`) in the Railway dashboard for auto-deploy on push — the
`Dockerfile` + `railway.toml` are picked up automatically.

## GitHub

`gh` CLI is not installed on this host. Push uses the git `origin` remote
(`https://github.com/Wassim1313/Bathymetry_VMarch.git`). Branch: `deploy/railway`.
