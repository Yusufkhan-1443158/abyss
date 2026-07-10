# Abyss Bathymetry Service

Standalone inference microservice for Abyss. Reads a source raster from MinIO,
produces a georeferenced **depth product** (RGB colour-mapped for display +
float32 metres for analysis), and returns summary statistics and a cross-section
transect.

## The model is a stub (for now)

`infer_depth(bands, max_depth)` in `main.py` currently synthesizes depth from
image luminance (darker water → deeper). It exists so the **entire Abyss
pipeline runs end-to-end today**: image ingest → instant display → depth layer →
auto-filled template report.

### Swapping in the real model

Replace **only** the body of `infer_depth()`:

```python
def infer_depth(bands: np.ndarray, max_depth: float) -> np.ndarray:
    # bands: (C, H, W) source pixels;  return: (H, W) float32 depth in metres, NaN = nodata
    return my_model.predict(bands)
```

Everything else — the MinIO I/O, the georeferenced GeoTIFF writers, the colour
ramp, the stats/transect, the proxy router, the Celery chaining, and the report
template — is model-agnostic and stays unchanged.

> The target model lives in a private repo (`Wassim1313/Bathymetry_VMarch`).
> Grant the Abyss build access (make public, add the `huznv` GitHub account as a
> collaborator, or drop the weights/inference code into this folder) and wire it
> into `infer_depth()`.

## API
- `POST /bathymetry/infer` — `{raster_id, source_bucket, source_key, max_depth?}`
- `GET  /bathymetry/health`
- `GET  /bathymetry/models`

## Performance & tuning (large AOIs)

The scene grid is capped at `max_px=1024` per side (`s2_fetch._target_grid`), so a
larger area just gets a coarser resolution — inference never grows unbounded. On
top of that the hot path was optimised so big/coastal areas run fast **without
changing any output value**:

- **Water-only inference** — the MC-Dropout MLP runs only on water-mask pixels
  (land/cloud carry no depth and were discarded anyway). Cost now scales with the
  water area, not the whole scene. Coastal AOIs (lots of land) speed up ~2–3×.
- **Single turbidity/k-means pass** — the Nechad-SPM + NDTI + bottom-cluster
  stack (the priciest non-MLP step, with a k-means over every water pixel) is
  computed once and reused for both the feature stack and the σ-map (was twice).
- **Concurrent band fetch** — the 6 Sentinel-2 bands are warped in parallel
  (independent remote-COG reads, GIL released during I/O), overlapping the
  network-bound reads that dominate a fresh fetch.
- **Imagery cache** — a fetched+warped band stack is stored in MinIO
  (`s2cache`), so repeat/re-runs of an area skip Planetary Computer entirely.

### Env knobs (defaults preserve current behaviour exactly)

| Env var | Default | Effect |
|---|---|---|
| `DL_PRO_BATCH_PIXELS` | `100000` | Pixels per MLP forward-batch (peak RAM). Lower on tight-memory hosts. |
| `DL_PRO_N_MC` | `30` | MC-Dropout passes for the σ estimate. Lower → faster, slightly noisier σ. |
| `DL_PRO_KMEANS_NINIT` | `10` | k-means restarts for bottom-type clusters. Lower → faster clustering on huge scenes. |
| `DL_PRO_TORCH_THREADS` | *(torch default)* | CPU threads for inference. Set to the host's core count on a many-core server. |
| `S2_FETCH_WORKERS` | `6` | Concurrent band-warp workers for the S2 fetch. |

Changing `DL_PRO_N_MC` or `DL_PRO_KMEANS_NINIT` alters the numeric output
slightly (different σ / cluster fit); the other knobs never change output.
