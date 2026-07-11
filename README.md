# Abyss — by Orbion

A distilled, bathymetry-focused fork of Glyph (IMINT). One pipeline, three steps:

> **Image ingested/pulled → AI generates bathymetric data → user generates a report.**

Adopts the **Stella** visual theme (near-black canvas, neon-cyan, glass morphism,
orbital depth glow), with a working **day/night** toggle and an **Abyss by Orbion**
logo.

## What's inside

- **Instant-load ingest (Goal #1).** Drag an image in and it's viewable in
  *seconds*, not ~30s. At ingest we build the Cloud-Optimized GeoTIFF straight
  into the shared tile cache and mark the raster `ready` immediately; persisting
  the COG to object storage and generating the deep-zoom HTJ2K happen in the
  background on a low-priority queue (`finalize_raster`). The first map tile is
  already warm. See `fastapi/app/services/tasks.py` (Stage A/B) and
  `fastapi/app/routers/rasters.py:_get_cached_raster`.
- **Bathymetry pipeline.** On ingest (or via `POST /api/bathymetry/run/{id}`),
  `run_bathymetry` calls the inference service, re-ingests a colour-mapped depth
  raster as a derived layer, and builds a template report. The depth model is the
  **real Bathymetry_VMarch SDB engine** (`bathymetry-service/sdb_engine/`): a
  UAE-calibrated cluster ensemble (RF + MLP) for multispectral sources, with an
  explicitly-labelled uncalibrated Stumpf fallback for plain RGB. See
  `bathymetry-service/README.md` for provenance and caveats.
- **Template reports.** `bathymetry_reports` stores an ordered list of typed
  sections (cover / metadata / summary / source imagery / depth map / cross-section
  / statistics / methodology), auto-filled by `services/report_builder.py` and
  rendered by `apps/report-view.html` (print-to-PDF via the browser).

## Run it

`docker-compose.yml` is the **operational** (hardened) baseline; a thin
`docker-compose.dev.yml` overlay restores the zero-friction local-dev behaviour.

### Operational (enforced login · TLS · file-based secrets)
```bash
cd Abyss
# 1. Generate the Docker-secret files (JWT/PG/Redis/MinIO + seed-account creds).
bash secure-config/keys/GENERATE.sh
# 2. Self-signed TLS cert for the demo (real certs go in the same dir).
mkdir -p secure-config/certs
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout secure-config/certs/abyss.key -out secure-config/certs/abyss.crt \
  -subj "/CN=abyss.local"
# 3. Bring it up. (If you ran the stack before, prefix `docker compose down -v` so
#    Postgres/MinIO re-initialise with the new secret passwords — DESTROYS data.)
docker compose up --build -d
xdg-open https://localhost:9443/     # http://localhost:9088 → 301 https
```
The app aborts boot (fail-fast) if `DEV_NO_AUTH=false` but the JWT secret is empty
or a known dev default — so a missing/weak secret can't ship silently.

### Local dev (zero-login bypass · plaintext creds · no TLS)
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
xdg-open http://localhost:9088/      # login auto-forwards (DEV_NO_AUTH)
```
No secrets or certs need generating — the overlay re-points every secret at an
empty placeholder and supplies plaintext credentials.

Ports: nginx **9443** (app, TLS) / **9088** (→301, or the dev app), fastapi
**8000**, bathymetry **8002**. The data services (postgres/redis/minio) are
internal to the `abyss-net` network.

### Try the pipeline (dev)
```bash
curl -F file=@/path/to/scene.tif "http://localhost:9088/api/rasters/upload?name=MyScene"
# → ready in ~1s; bathymetry runs automatically; a report appears in Reports.
```
(Operationally the upload requires a bearer token — log in first and send
`Authorization: Bearer <access_token>`.)

## Auth & demo accounts
Operationally, real JWT login is enforced (HS256 access+refresh, HttpOnly+`Secure`
cookies, Redis JTI blacklist + refresh rotation, role guards). Three accounts are
seeded deterministically (idempotent upsert each boot while `SEED_FORCE=true`):

| user | role | password |
|------|------|----------|
| `admin`   | admin   | `secure-config/keys/seed-admin-password.txt` |
| `analyst` | analyst | `secure-config/keys/seed-analyst-password.txt` |
| `viewer`  | viewer  | `secure-config/keys/seed-viewer-password.txt` |

In **dev** the same accounts are `admin/abyss`, `analyst/analyst`, `viewer/viewer`
and login is bypassed entirely (`DEV_NO_AUTH=true`). Set `SEED_FORCE=false` once
accounts are admin-managed so manual password changes survive restarts.

## Security & operations
- **Secrets** — `secure-config/keys/GENERATE.sh` emits the secret files;
  `docker-compose.yml` mounts them as Docker secrets (`/run/secrets/*`). `*.txt`
  is git-ignored. `config.py` resolves every credential as *env var → secret file
  → safe default*.
- **TLS** — nginx terminates TLS on :443 (HSTS, TLSv1.2/1.3); :80 301-redirects to
  https. Cookies set `Secure` via `COOKIE_SECURE`.
- **Rate limiting** — slowapi + Redis (db /4), XFF-aware behind nginx (uvicorn runs
  with `--proxy-headers`). Global `RATE_LIMIT_DEFAULT` + strict `RATE_LIMIT_LOGIN`
  (5/min) on login/refresh.
- **Data-at-rest** — MinIO buckets get SSE-S3 (built-in KMS via `minio-kms-key.txt`)
  so imagery/depth/report payloads are encrypted. Passwords are bcrypt. For full
  at-rest coverage of Postgres/MinIO/Redis volumes, run the Docker data root (or a
  dedicated partition holding `postgres-data`/`minio-data`/`redis-data`) on a
  **LUKS** device — protects all three stores with zero app change.
- **Tests** — auth/RBAC regression suite runs in-container against live PG/Redis:
  ```bash
  docker compose exec fastapi pytest
  ```

## The real depth model
The Bathymetry_VMarch SDB engine is vendored under
`bathymetry-service/sdb_engine/` (feature engineering, pretrained UAE model
bundles, NDWI land cut, colour ramp) and wired into `infer_depth()`. Uploaded
rasters are inferred from their own pixels; drawn ROIs still use the free
Sentinel-2 fetch. Model provenance, calibration domain and honesty caveats are
in `bathymetry-service/README.md` and surfaced in every generated report.

## Known follow-ups (cosmetic / next pass)
- master-home's inline bottom dock still lists a couple of dropped apps.
- `GET /api/rasters/` (trailing slash) 404s a home-page "recent collections" probe.
- Client-side `geotiff.js` drag-in (true sub-second local preview) is a later phase.
