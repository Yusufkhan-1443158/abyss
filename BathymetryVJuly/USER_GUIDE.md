# Bathymetry from Space — User Guide

Satellite-derived bathymetry (SDB) platform for shallow coastal waters.
Coordinates: **WGS84**. Vertical datum: **LAT** (adjustable on upload). Depths are capped at **25 m**
(the physical limit of optical SDB in Gulf waters). Maximum area per run: **40 km²**.

---

## 1. Typical workflow (5 steps)

1. **Draw a Region of Interest (ROI)** — use the rectangle tool on the map, click a preset
   region button, or edit the rectangle corners. The N/S/E/W boxes in the sidebar update live.
2. **Upload reference data (optional)** — survey soundings (`.xyz`, `.csv`, `.shp`) to calibrate
   the model to your area. Skip this step in a predefined region: calibration is built in.
3. **Choose a depth method** — the resolution selector (20 m / 10 m / High-Res) for the standard
   run, the High-Resolution Depth panel for ~1–2 m products, the Multi-Scene MLE for a whole-year
   composite with uncertainty, or Wave-Based Depth where reference data is unavailable.
4. **Run the extraction** — click the run button. Progress appears in the LOGS tab; long
   high-resolution jobs appear in the JOBS window with a real percentage.
5. **Validate and export** — upload independent survey points to get RMSE / bias / IHO S-44
   pass rates, then download the grid as GeoTIFF, CSV or GeoJSON.

---

## 2. Region of Interest

- **Draw**: use the rectangle draw control in the top-right of the map.
- **Presets**: Khalifa Port, Old Mussafah, Abu Al Abyad, Jbel Dhanna. Clicking a preset sets the
  ROI *and* the calibrated imagery date for that site, so you land on the validated optimum.
- **Zoom to ROI** re-centres the map on the current rectangle.
- ROIs larger than the 40 km² cap are rejected server-side; split large areas into tiles.

## 3. Reference / calibration data

- **Formats**: `.xyz`, `.txt`, `.csv` point lists, or Esri Shapefile (`.shp` + `.dbf`/`.shx`/`.prj`,
  or a `.zip` of them).
- **Options at upload**: coordinate order (Lat,Lon,Z or Lon,Lat,Z), UTM zone (EPSG) if the file is
  projected, horizontal datum, and vertical datum (LAT / MSL / other).
- Points falling inside the ROI are used to calibrate the depth model; a held-out fraction (20 %)
  is kept for honest validation.
- Predefined regions already contain survey-grade calibration — no upload needed there.

## 4. Run Bathymetry (standard run)

- **Resolution**: `20 m` = fast coarse overview; `10 m` = standard production depth (native
  Sentinel-2); `High-Res` = ~1 m base raster for fine detail (slower).
- The imagery used is a multi-scene cloud-free median around the chosen date, which suppresses
  glint, wakes and haze.
- Inside a calibrated region the run automatically uses the in-region trained model
  ("region-calibrated" badge); outside, a general UAE model is used with reduced weight.

## 5. High-Resolution Depth panel

- **CLUSTERS (K)**: number of seabed classes used to group similar pixels before per-cluster
  regression. K = 8 is the calibrated default.
- **Augment**: automatically adds satellite-lidar and nautical-chart reference depths in bands
  where your calibration data is sparse.
- **Imagery date**: the least-cloudy scene within ±10 days of the chosen date is selected.
- **Imagery source**: High-Res (~2 m/px) base raster, or Standard (10 m/px — faster, always
  available).
- **Run High-Resolution Depth (~1 m)** submits a *local job* (processed offline, result uploaded
  later). **Run High-Resolution PRO (multi-scene)** runs server-side and reports a real progress
  percentage. Both appear in the **JOBS** window.

## 6. Multi-Scene MLE (whole year)

Combines N evenly-spaced scenes across a calendar year into a single depth product using
per-pixel inverse-variance weighting (maximum likelihood). Output includes a per-pixel
uncertainty map. Choose the year and 3 / 5 / 7 scenes; runtime is roughly N × 5–15 s.

## 7. Wave-Based Depth

Reference-free bathymetry from surface-wave physics (linear dispersion): wave celerity measured
from inter-band time offsets is inverted to depth. Use it where no calibration data exists and
where a visible swell/wave field is present; it is not suited to flat, sheltered water.

## 8. Lidar depth points (ICESat-2)

Search and process satellite laser-altimeter depth points over any ROI. Results are drawn on the
map colour-coded per beam and are used automatically as extra reference data by the extraction.

## 9. Results

- **Report card** (sidebar): quality badge, depth map, per-depth-band accuracy, references used,
  and the download row.
- **Views** (header): `MAP` (depth overlay + isobath chart tab), `3D` (WebGL seabed), `SPLIT`
  (map + 3D), `ASCII` (text-grid preview), `MONITOR`, `UAE RESULTS`.
- **RESULTS catalogue** (header button): persistent list of every finished job — reopen,
  analyse or download past runs.

## 10. Validation

Upload an independent survey file (`.shp`/`.csv`) in the Validation panel, then click
**Run Validation**. Reported: RMSE, MAE, bias, R², IHO S-44 pass %, and the number of matched
pairs. Validation points are never used for calibration.

## 11. Export

Every result can be downloaded as:

| Format  | Contents |
|---------|----------|
| GeoTIFF | Georeferenced depth raster (WGS84), ready for GIS/CAD |
| CSV     | `lon, lat, depth_m` point list |
| GeoJSON | Depth points/contours for web mapping |

Multi-file products (PRO runs) offer a one-click **.zip of all GeoTIFFs** with a manifest.

## 12. Monitoring

The **MONITOR** view stores successive surveys per zone and computes change between epochs:
depth/volume differences, shoaling or deepening trends, and alerts (e.g. sudden shoaling in a
navigation channel). Use it to schedule re-surveys and track dredging needs.

## 13. Ordering a High-Resolution product

The **HIGH-RES** header button opens the order panel: select an area and request a ~1 m depth
product; delivery lands in the RESULTS catalogue.

---

## Troubleshooting

- **Run button greyed out** — no ROI yet: draw a rectangle or click a preset region.
- **"ROI too large"** — the 40 km² cap; split the area.
- **Noisy/blank patches** — clouds or turbidity on the chosen date; move the imagery date or use
  the multi-scene MLE, which averages them out.
- **Depths look offset by a constant** — check the vertical datum chosen at upload (LAT vs MSL).
- **Turbidity warning on the report** — optical depth retrieval is degraded; treat results as
  indicative and prefer a clearer date.

## Glossary

- **SDB** — Satellite-Derived Bathymetry.
- **ROI** — Region of Interest (the rectangle you draw).
- **LAT** — Lowest Astronomical Tide, the chart datum used for depths.
- **IHO S-44** — international accuracy standard for hydrographic surveys; the pass % states how
  many validation points fall within the allowed error for their depth.
- **MLE** — Maximum-Likelihood Estimate: multi-scene, uncertainty-weighted depth composite.
