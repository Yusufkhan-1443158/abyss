# Abyss Bathymetry Service

Standalone inference microservice for Abyss. Reads a source raster from MinIO
(or fetches Sentinel-2 for a drawn ROI), produces a georeferenced **depth
product** (RGB colour-mapped for display + float32 metres for analysis), and
returns summary statistics, a downsampled grid and a cross-section transect.

## The real SDB engine (`sdb_engine/`)

`infer_depth()` is the production Bathymetry_VMarch engine, vendored
self-contained under `sdb_engine/`:

```python
from sdb_engine import infer_depth
depth = infer_depth(bands)   # bands: (C, H, W) -> (H, W) float32 metres, NaN = land/nodata
```

Band handling is honest about what the input supports:

- **Multispectral (blue/green/red + NIR, Sentinel-2-like)** — the calibrated
  path: a 13-feature spectral stack (raw reflectances, Lyzenga log-bands +
  depth-invariant index, Stumpf B/G + B/R log-ratios, NDWI) feeds a K-means
  cluster ensemble with per-cluster Random Forest and MLP regressors
  (`sdb_engine/models/uae_clustered_rf.pkl` + `uae_clustered_cnn.pkl`,
  registry v1, trained on ~443k soundings across 8 UAE regions; in-sample
  R² 0.978 / RMSE 0.735 m). Land is cut with an NDWI water mask; depth is
  clipped to 0–25 m; a per-pixel uncertainty channel combines per-cluster
  calibration RMSE and ensemble disagreement. **UAE-calibrated** — outside
  that envelope the output is indicative.
- **Plain RGB (3 bands)** — the physically defensible fallback: Stumpf
  blue/green log-ratio pseudo-depth, percentile-stretched. Relative structure
  only, explicitly flagged `calibrated: false` / low confidence in the
  response and the report. Never presented as metric depth.

Band identification uses band descriptions when present (`blue`/`B02` …),
else position: 3 bands = R,G,B; 4 = B,G,R,NIR; 5+ = coastal,B,G,R,NIR.
Reflectance (0–1) and 8-bit inputs are rescaled to the S2 DN scale.

Depths are referenced to the calibration soundings' survey datum and are not
tide-corrected. Validate against in-situ soundings before navigational use.

The ROI path (no uploaded source) still fetches free Sentinel-2 L2A from
Microsoft Planetary Computer and runs the DL-Pro v3 turbidity-aware MLP.

## API
- `POST /bathymetry/infer` — `{raster_id, source_bucket?, source_key?, bbox?,
  start_date?, end_date?, max_cloud?, max_depth?}`; `source_*` selects the
  ingested-raster engine, `bbox` alone selects the Sentinel-2 ROI path.
- `GET  /bathymetry/health`
- `GET  /bathymetry/models`

## Tests
```bash
pytest tests/test_infer_depth.py -v
```

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
