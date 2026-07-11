import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet-draw';
import { rampCss, rampGradientCss, rampTicks } from './depthRamp';
import { INTERNAL_ANALYSIS } from './internalMode';

// Depth → colour using the shared perceptually-uniform ramp so map markers,
// the colorbar legend, the 2D Results colorbar and the 3D seabed all agree.
function depthColor(depth, maxD) {
  return rampCss(Math.abs(depth) / (maxD || 25));
}

// Client-portal marker labels hide the data source; internal shows it.
const PT_LABEL = {
  gebco:  INTERNAL_ANALYSIS ? 'GEBCO' : 'Reference',
  icesat: INTERNAL_ANALYSIS ? 'ICESat-2' : 'Survey point',
  lidar:  INTERNAL_ANALYSIS ? 'ICESat-2 lidar' : 'Lidar depth',
  insitu: INTERNAL_ANALYSIS ? 'In-situ' : 'Survey point',
};

export default function MapPanel({ roi, setRoi, results, csvPreview, icesat2Results, overlayOverride, onShowOverlay, style }) {
  const mapRef = useRef(null);
  const mapInst = useRef(null);
  const layersRef = useRef({});
  // Remember the last raster_bounds we fit-bounded to so that toggling
  // visibility checkboxes (which re-runs the render effect) does NOT snap
  // the map back, only a genuinely new result does.
  const lastFitKey = useRef(null);
  const [vis, setVis] = useState({ depth: true, gebco: true, icesat: true, cshelph: true });
  // optical_valid deep-abstain mask overlay (Khalifa VHR honesty-by-design).
  // null = no mask shown. When set, we draw the legend + (if a raster is
  // available) an image overlay where the deep-abstain region is greyed/hatched.
  const [maskInfo, setMaskInfo] = useState(null);
  const [maskVisible, setMaskVisible] = useState(true);
  const [showM2, setShowM2] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [trainMode, setTrainMode] = useState(false);
  const [trainPts, setTrainPts] = useState([]);
  // ── Measure-distance tool (user-requested: "give opportunity to measure
  // distance for me to check" — to verify the land-mask offset at quays
  // himself). Click to add vertices, live polyline with per-segment +
  // cumulative distance labels, double-click/ESC to finish, Clear button.
  const [measuring, setMeasuring] = useState(false);
  const [measurePts, setMeasurePts] = useState([]); // [{lat,lng}, ...]
  const measuringRef = useRef(false);
  useEffect(() => { measuringRef.current = measuring; }, [measuring]);

  // Auto-show stats popup when any results arrive
  useEffect(() => {
    if (results?.ml_stats?.caballero_stats) setShowM2(true);
    else if (results?.stats) setShowStats(true);
  }, [results]);

  // Init map
  useEffect(() => {
    if (mapInst.current) return;
    const map = L.map(mapRef.current, { center: [24.5, 54.0], zoom: 9, zoomControl: false, attributionControl: false });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap' }).addTo(map);
    L.control.zoom({ position: 'topright' }).addTo(map);

    const drawn = new L.FeatureGroup(); map.addLayer(drawn);
    window._drawnGroup = drawn;
    const dc = new L.Control.Draw({ position: 'topright', draw: { rectangle: { shapeOptions: { color: '#22d3ee', weight: 2, fillOpacity: 0.08, dashArray: '8,6' } }, polygon: false, circle: false, circlemarker: false, marker: false, polyline: false }, edit: { featureGroup: drawn } });
    map.addControl(dc);
    const _pushRoiFromLayer = (layer) => {
      const b = layer.getBounds();
      setRoi({ north: b.getNorth(), south: b.getSouth(), east: b.getEast(), west: b.getWest() });
    };
    map.on('draw:created', e => {
      drawn.clearLayers(); drawn.addLayer(e.layer); _pushRoiFromLayer(e.layer);
      // Auto-centre + auto-zoom on the freshly drawn ROI (zoom level derives
      // from the rectangle size via flyToBounds — small ROI → closer zoom).
      map.flyToBounds(e.layer.getBounds(), { padding: [40, 40], duration: 0.8 });
    });
    map.on('draw:edited', e => {
      // Rectangle was edited via the edit control — push updated bounds back to the sidebar fields.
      e.layers.eachLayer(layer => _pushRoiFromLayer(layer));
    });
    map.on('draw:editvertex', e => {
      drawn.eachLayer(layer => _pushRoiFromLayer(layer));
    });
    // Rectangles fire editmove/editresize (NOT editvertex) while their edit
    // handles are dragged — sync N/S/E/W sidebar fields in real time.
    map.on('draw:editmove', e => { if (e.layer?.getBounds) _pushRoiFromLayer(e.layer); });
    map.on('draw:editresize', e => { if (e.layer?.getBounds) _pushRoiFromLayer(e.layer); });
    map.on('draw:deleted', () => setRoi(null));

    // ── Measure-distance tool click handlers ──
    map.on('click', (e) => {
      if (!measuringRef.current) return;
      setMeasurePts(prev => [...prev, { lat: e.latlng.lat, lng: e.latlng.lng }]);
    });
    map.on('dblclick', (e) => {
      if (!measuringRef.current) return;
      L.DomEvent.stop(e); // don't let dblclick also zoom the map
      setMeasuring(false);
    });
    const _escHandler = (ev) => { if (ev.key === 'Escape' && measuringRef.current) setMeasuring(false); };
    window.addEventListener('keydown', _escHandler);
    window._measureEscHandler = _escHandler;

    // Listen for programmatic ROI changes (preset buttons, observed-data loads)
    // → draw rectangle on map + fly to that bbox so the user actually sees the area.
    const _roiListener = (ev) => {
      const bb = ev.detail;
      if (!bb || bb.north === undefined) return;
      drawn.clearLayers();
      const rect = L.rectangle(
        [[bb.south, bb.west], [bb.north, bb.east]],
        { color: '#22d3ee', weight: 2, fillOpacity: 0.08, dashArray: '8,6' }
      );
      drawn.addLayer(rect);
      map.flyToBounds([[bb.south, bb.west], [bb.north, bb.east]], { padding: [40, 40], duration: 0.8 });
    };
    window.addEventListener('setRoi', _roiListener);
    window.addEventListener('zoomToRoi', _roiListener);

    // Satellite toggle
    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 19 });

    // OpenSeaMap nautical overlay (free, no API key needed)
    const seamarks = L.tileLayer('https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png', {
      maxZoom: 18, opacity: 0.85, attribution: '© OpenSeaMap'
    });
    // NOAA nautical charts
    const noaaChart = L.tileLayer('https://tileservice.charts.noaa.gov/tiles/50000_1/{z}/{x}/{y}.png', {
      maxZoom: 18, opacity: 0.7, attribution: '© NOAA'
    });
    // Depth contour overlay (OpenSeaMap)
    const iboatDepth = L.tileLayer('https://depth.openseamap.org/gebco/{z}/{x}/{y}.png', {
      maxZoom: 18, opacity: 0.7, attribution: '© OpenSeaMap Depth'
    });

    // Layer toggle buttons
    const lc = L.control({ position: 'bottomright' });
    lc.onAdd = () => {
      const d = L.DomUtil.create('div');
      d.style.cssText = 'display:flex;flex-direction:column;gap:4px';
      d.innerHTML = `
        <button id="sat-btn" style="padding:5px 10px;font-size:9px;font-family:monospace;font-weight:600;background:rgba(255,255,255,0.92);border:1px solid #cbd5e1;color:#475569;border-radius:6px;cursor:pointer;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.1)">SATELLITE</button>
        <button id="sea-btn" style="padding:5px 10px;font-size:9px;font-family:monospace;font-weight:600;background:rgba(255,255,255,0.92);border:1px solid #cbd5e1;color:#92400e;border-radius:6px;cursor:pointer;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.1)">SEAMARKS</button>
        <button id="depth-btn" style="padding:5px 10px;font-size:9px;font-family:monospace;font-weight:600;background:rgba(255,255,255,0.92);border:1px solid #cbd5e1;color:#0284c7;border-radius:6px;cursor:pointer;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.1)">DEPTH CHART</button>
        <button id="noaa-btn" style="padding:5px 10px;font-size:9px;font-family:monospace;font-weight:600;background:rgba(255,255,255,0.92);border:1px solid #cbd5e1;color:#0e7490;border-radius:6px;cursor:pointer;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.1)">NOAA</button>
        <a id="iboat-btn" href="#" title="Open the reference chart in a new tab for this location" style="padding:5px 10px;font-size:9px;font-family:monospace;font-weight:700;background:rgba(255,255,255,0.92);border:1px solid #065f46;color:#065f46;border-radius:6px;cursor:pointer;white-space:nowrap;text-decoration:none;display:block;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.1)">REFERENCE CHART</a>
        <button id="train-btn" style="padding:5px 10px;font-size:9px;font-family:monospace;font-weight:700;background:rgba(255,255,255,0.92);border:2px solid #7c3aed;color:#7c3aed;border-radius:6px;cursor:pointer;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.1)">+ TRAIN PT</button>
        <button id="load-train-btn" style="padding:5px 10px;font-size:9px;font-family:monospace;font-weight:600;background:rgba(255,255,255,0.92);border:1px solid #7c3aed;color:#7c3aed;border-radius:6px;cursor:pointer;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.1)">LOAD SAVED</button>
        <button id="export-train-btn" style="padding:5px 10px;font-size:9px;font-family:monospace;font-weight:600;background:rgba(255,255,255,0.92);border:1px solid #059669;color:#059669;border-radius:6px;cursor:pointer;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,0.1)">EXPORT CSV</button>
      `;
      let satOn=false, seaOn=false, noaaOn=false, depthOn=false;
      d.querySelector('#sat-btn').onclick = () => { satOn ? map.removeLayer(satellite) : satellite.addTo(map); satOn=!satOn; d.querySelector('#sat-btn').style.background=satOn?'#dbeafe':'rgba(255,255,255,0.92)'; };
      d.querySelector('#sea-btn').onclick = () => { seaOn ? map.removeLayer(seamarks) : seamarks.addTo(map); seaOn=!seaOn; d.querySelector('#sea-btn').style.background=seaOn?'#fef3c7':'rgba(255,255,255,0.92)'; };
      d.querySelector('#depth-btn').onclick = () => { depthOn ? map.removeLayer(iboatDepth) : iboatDepth.addTo(map); depthOn=!depthOn; d.querySelector('#depth-btn').style.background=depthOn?'#bfdbfe':'rgba(255,255,255,0.92)'; };
      d.querySelector('#noaa-btn').onclick = () => { noaaOn ? map.removeLayer(noaaChart) : noaaChart.addTo(map); noaaOn=!noaaOn; d.querySelector('#noaa-btn').style.background=noaaOn?'#cffafe':'rgba(255,255,255,0.92)'; };
      d.querySelector('#iboat-btn').onclick = (e) => { e.preventDefault(); const c=map.getCenter(); const z=map.getZoom(); window.open('https://fishing-app.gpsnauticalcharts.com/i-boating-fishing-web-app/fishing-marine-charts-navigation.html#'+z+'/'+c.lat.toFixed(4)+'/'+c.lng.toFixed(4),'_blank'); };
      // Train mode toggle
      d.querySelector('#train-btn').onclick = () => {
        window._trainMode = !window._trainMode;
        const btn = d.querySelector('#train-btn');
        btn.style.background = window._trainMode ? '#7c3aed' : 'rgba(255,255,255,0.92)';
        btn.style.color = window._trainMode ? '#fff' : '#7c3aed';
        btn.textContent = window._trainMode ? 'CLICK MAP' : '+ TRAIN PT';
        map.getContainer().style.cursor = window._trainMode ? 'crosshair' : '';
      };
      // Load saved training points onto map
      d.querySelector('#load-train-btn').onclick = async () => {
        try {
          const tg = window._trainGroup;
          const r = await fetch('/api/training/all');
          const data = await r.json();
          if (tg) tg.clearLayers();
          const step = Math.max(1, Math.floor((data.points||[]).length / 2000));
          (data.points||[]).forEach((p, i) => {
            if (i % step !== 0 || !tg) return;
            tg.addLayer(L.circleMarker([p.lat, p.lon], {
              radius: 4, fillColor: p.source === 'map_click' || p.source === 'manual' ? '#7c3aed' : p.source?.startsWith('iboating') || p.source === 'etopo_coastal' ? '#2563eb' : '#f59e0b',
              fillOpacity: 0.7, stroke: true, weight: 0.5, color: '#fff'
            }).bindPopup(`<div style="font-family:monospace;font-size:11px"><b style="color:#7c3aed">${p.depth.toFixed(1)}m</b><br><span style="font-size:8px;color:#94a3b8">${p.source}</span></div>`));
          });
          d.querySelector('#load-train-btn').textContent = `${data.count} PTS`;
          setTimeout(() => { d.querySelector('#load-train-btn').textContent = 'LOAD SAVED'; }, 3000);
        } catch(e) { console.warn('Load training failed:', e); }
      };
      // Export training points as CSV
      d.querySelector('#export-train-btn').onclick = () => {
        window.open('/api/training/export', '_blank');
      };

      L.DomEvent.disableClickPropagation(d);
      return d;
    };
    lc.addTo(map);

    // Click-to-add training point
    const trainGroup = L.layerGroup().addTo(map);
    window._trainGroup = trainGroup;
    map.on('click', async (e) => {
      if (!window._trainMode) return;
      const {lat, lng} = e.latlng;
      const depth = prompt(`Depth at (${lat.toFixed(5)}, ${lng.toFixed(5)})?\nEnter depth in meters (read from chart):`);
      if (!depth || isNaN(parseFloat(depth))) return;
      const d = Math.abs(parseFloat(depth));
      // Save to backend store
      try {
        await fetch('/api/training/add', {method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({points:[{lat, lon:lng, depth:d}], source:'map_click'})});
      } catch(ex) { console.warn('Training save failed:', ex); }
      // Show marker on map
      const marker = L.circleMarker([lat, lng], {
        radius: 7, fillColor: '#7c3aed', fillOpacity: 0.9, stroke: true, weight: 2, color: '#fff'
      }).bindPopup(`<div style="font-family:monospace;font-size:12px;text-align:center"><b style="color:#7c3aed">${d.toFixed(1)}m</b><br><span style="font-size:9px;color:#64748b">Training point</span></div>`);
      trainGroup.addLayer(marker);
      marker.openPopup();
    });
    L.control.scale({ position: 'bottomleft', imperial: false }).addTo(map);

    // Coords
    const cc = L.control({ position: 'bottomleft' });
    cc.onAdd = () => { const d = L.DomUtil.create('div'); d.id = 'coords'; d.style.cssText = 'padding:3px 8px;font-size:9px;font-family:monospace;background:rgba(255,255,255,0.9);border:1px solid #cbd5e1;border-radius:6px;color:#374151;box-shadow:0 1px 3px rgba(0,0,0,0.1)'; d.innerHTML = '—'; return d; };
    cc.addTo(map);
    map.on('mousemove', e => { const el = document.getElementById('coords'); if (el) el.innerHTML = `${e.latlng.lat.toFixed(5)}° ${e.latlng.lng.toFixed(5)}°`; });

    mapInst.current = map;
    return () => {
      window.removeEventListener('setRoi', _roiListener);
      window.removeEventListener('zoomToRoi', _roiListener);
      if (window._measureEscHandler) { window.removeEventListener('keydown', window._measureEscHandler); delete window._measureEscHandler; }
      map.remove();
      mapInst.current = null;
    };
  }, [setRoi]);

  // ── Measure-tool render effect: polyline + per-segment/cumulative labels ──
  const haversineM = (a, b) => {
    const R = 6371000, toRad = (d) => d * Math.PI / 180;
    const dLat = toRad(b.lat - a.lat), dLng = toRad(b.lng - a.lng);
    const s = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(s));
  };
  const fmtDist = (m) => (m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(1)} m`);
  useEffect(() => {
    const map = mapInst.current;
    if (!map) return;
    if (layersRef.current.measureLayer) { map.removeLayer(layersRef.current.measureLayer); delete layersRef.current.measureLayer; }
    if (measurePts.length === 0) return;
    const grp = L.layerGroup();
    const latlngs = measurePts.map(p => [p.lat, p.lng]);
    L.polyline(latlngs, { color: '#f59e0b', weight: 2.5, dashArray: measuring ? '6,5' : null }).addTo(grp);
    let cumulative = 0;
    measurePts.forEach((p, i) => {
      L.circleMarker([p.lat, p.lng], { radius: 4, fillColor: '#f59e0b', fillOpacity: 1, stroke: true, weight: 1.5, color: '#fff' }).addTo(grp);
      if (i > 0) {
        const segM = haversineM(measurePts[i - 1], p);
        cumulative += segM;
        const mid = { lat: (measurePts[i - 1].lat + p.lat) / 2, lng: (measurePts[i - 1].lng + p.lng) / 2 };
        L.marker([mid.lat, mid.lng], {
          icon: L.divIcon({
            className: 'measure-label',
            html: `<div style="background:rgba(15,23,42,0.88);color:#fbbf24;font-family:monospace;font-size:9.5px;font-weight:700;padding:2px 6px;border-radius:4px;white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,0.3)">${fmtDist(segM)}</div>`,
            iconSize: [0, 0],
          }),
        }).addTo(grp);
      }
    });
    if (measurePts.length > 1) {
      const last = measurePts[measurePts.length - 1];
      L.marker([last.lat, last.lng], {
        icon: L.divIcon({
          className: 'measure-label-total',
          html: `<div style="background:#f59e0b;color:#1e293b;font-family:monospace;font-size:10px;font-weight:800;padding:3px 8px;border-radius:5px;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.35);transform:translateY(-22px)">total ${fmtDist(cumulative)}</div>`,
          iconSize: [0, 0],
        }),
      }).addTo(grp);
    }
    grp.addTo(map);
    layersRef.current.measureLayer = grp;
  }, [measurePts, measuring]);

  // Render data
  useEffect(() => {
    const map = mapInst.current;
    if (!map) return;

    // Clear old
    ['depthLayer', 'rasterLayer', 'gebcoLayer', 'icesatLayer'].forEach(k => {
      if (layersRef.current[k]) { map.removeLayer(layersRef.current[k]); delete layersRef.current[k]; }
    });

    if (!results || !results.points) return;

    const pts = results.points;
    const gebcoG = L.layerGroup();
    const icesatG = L.layerGroup();

    // ── RASTER OVERLAY (primary depth display) ──
    if (results.raster_png && results.raster_bounds) {
      const bounds = results.raster_bounds; // [[south,west],[north,east]]
      const raster = L.imageOverlay(
        `data:image/png;base64,${results.raster_png}`,
        bounds,
        { opacity: 0.95, interactive: false, className: 'depth-raster' }
      );
      layersRef.current.rasterLayer = raster;
      if (vis.depth) raster.addTo(map);
      // Only fit bounds the first time we see THIS specific raster — toggling
      // visibility checkboxes must not re-snap the map (caused the "map jumps
      // back" instability).
      const fitKey = JSON.stringify(bounds) + '::' + (results.raster_png?.length || 0);
      if (lastFitKey.current !== fitKey) {
        map.fitBounds(bounds, { padding: [40, 40] });
        lastFitKey.current = fitKey;
      }
    }

    // ── Reference points (GEBCO, ICESat-2, chart, in-situ) — keep as markers ──
    pts.forEach(p => {
      if (p.photon_class === 'gebco') {
        gebcoG.addLayer(L.circleMarker([p.lat, p.lon], {
          radius: 4, fillColor: '#fbbf24', fillOpacity: 0.5,
          stroke: true, weight: 0.5, color: 'rgba(255,255,255,0.2)',
        }).bindPopup(`<div style="font-family:monospace;font-size:11px"><b style="color:#fbbf24">${PT_LABEL.gebco}</b><br>Depth: <b>${p.depth.toFixed(1)}m</b></div>`));
      } else if (p.photon_class === 'bathymetry' || p.photon_class === 'icesat2_cshelph') {
        icesatG.addLayer(L.circleMarker([p.lat, p.lon], {
          radius: 3.5, fillColor: '#ef4444', fillOpacity: 0.9,
          stroke: true, weight: 0.5, color: '#fff',
        }).bindPopup(`<div style="font-family:monospace;font-size:11px"><b style="color:#ef4444">${PT_LABEL.icesat}</b><br>Depth: <b>${p.depth.toFixed(2)}m</b>${(INTERNAL_ANALYSIS && p.beam)?`<br>Beam: ${p.beam}`:''}</div>`));
      } else if (p.photon_class === 'chart') {
        gebcoG.addLayer(L.circleMarker([p.lat, p.lon], {
          radius: 5, fillColor: '#a78bfa', fillOpacity: 0.85,
          stroke: true, weight: 1, color: '#fff',
        }).bindPopup(`<div style="font-family:monospace;font-size:11px"><b style="color:#a78bfa">Reference library</b><br>Depth: <b>${p.depth.toFixed(1)}m</b></div>`));
      } else if (p.photon_class === 'insitu' || p.photon_class === 'observed') {
        gebcoG.addLayer(L.circleMarker([p.lat, p.lon], {
          radius: 3, fillColor: '#f59e0b', fillOpacity: 0.6,
          stroke: false,
        }));
      }
    });

    layersRef.current.gebcoLayer = gebcoG;
    layersRef.current.icesatLayer = icesatG;

    if (vis.gebco) gebcoG.addTo(map);
    if (vis.icesat) icesatG.addTo(map);
  }, [results, vis]);

  // ── ADPorts F1/F2: per-scene / Planet-backup overlay OVERRIDE ──
  // Swaps the visible raster to overlayOverride.png_b64 at overlayOverride.bounds
  // WITHOUT touching lastFitKey / calling fitBounds — the camera must never
  // re-snap when the user is just paging through MLE scenes or Planet dates
  // that share the same geographic extent as the composite.
  useEffect(() => {
    const map = mapInst.current;
    if (!map) return;
    if (layersRef.current.overlayOverrideLayer) {
      map.removeLayer(layersRef.current.overlayOverrideLayer);
      delete layersRef.current.overlayOverrideLayer;
    }
    const base = layersRef.current.rasterLayer;
    if (overlayOverride && overlayOverride.png_b64 && overlayOverride.bounds) {
      if (base && map.hasLayer(base)) map.removeLayer(base);
      const ov = L.imageOverlay(
        `data:image/png;base64,${overlayOverride.png_b64}`,
        overlayOverride.bounds,
        { opacity: 0.95, interactive: false, className: 'depth-raster depth-raster-override' }
      );
      ov.addTo(map);
      layersRef.current.overlayOverrideLayer = ov;
    } else if (base && vis.depth && !map.hasLayer(base)) {
      base.addTo(map);
    }
  }, [overlayOverride, vis.depth]);

  // ─── ICESat-2 CShelph dedicated layer ─────────────────────────────────
  //   • depth points coloured by depth
  //   • ground-track polylines per beam
  //   • auto-fit map bounds on first render
  useEffect(() => {
    const map = mapInst.current; if (!map) return;
    ['cshelphPts', 'cshelphTracks'].forEach(k => {
      if (layersRef.current[k]) { map.removeLayer(layersRef.current[k]); delete layersRef.current[k]; }
    });
    if (!icesat2Results || !icesat2Results.depth_points?.length) return;

    const pts = icesat2Results.depth_points;
    const maxD = icesat2Results.quality?.max_detectable_m || 25;

    // Depth points (high-visibility — larger radius, white halo)
    const ptsGroup = L.layerGroup();
    const beamColours = { 1: '#ef4444', 2: '#f97316', 3: '#fbbf24' };
    pts.forEach(p => {
      const beamCol = beamColours[p.beam] || '#ef4444';
      const fill = depthColor(p.depth, maxD);
      const marker = L.circleMarker([p.lat, p.lon], {
        radius: p.cross_validated ? 5.5 : 4,
        fillColor: fill, fillOpacity: 0.95,
        stroke: true, weight: 1.2, color: beamCol,
      });
      const xvalRow = p.cross_validated
        ? `<div style="color:#10b981;font-weight:700;margin-top:3px">✓ cross-validated</div>`
        : '';
      marker.bindPopup(
        `<div style="font-family:monospace;font-size:11px;min-width:140px">
           <div style="font-weight:800;color:${fill};font-size:13px;margin-bottom:4px">${PT_LABEL.lidar}</div>
           <div><b>Depth</b>: ${p.depth.toFixed(2)} m</div>
           ${INTERNAL_ANALYSIS ? `<div style="color:#64748b;font-size:10px">Beam ${p.beam}</div>` : ''}
           ${xvalRow}
         </div>`
      );
      ptsGroup.addLayer(marker);
    });
    layersRef.current.cshelphPts = ptsGroup;
    if (vis.cshelph) ptsGroup.addTo(map);

    // Beam ground-tracks as thin polylines
    const tracksGroup = L.layerGroup();
    (icesat2Results.tracks || []).forEach(t => {
      if (!t.coords || t.coords.length < 2) return;
      const latlngs = t.coords.map(c => [c[0], c[1]]);
      const line = L.polyline(latlngs, {
        color: beamColours[t.beam] || '#ef4444',
        weight: 2.5, opacity: 0.6, dashArray: '6,4',
      });
      line.bindTooltip(`Beam ${t.beam} · ${t.n_depths} depths`, { sticky: true });
      tracksGroup.addLayer(line);
    });
    if (tracksGroup.getLayers().length > 0) {
      layersRef.current.cshelphTracks = tracksGroup;
      if (vis.cshelph) tracksGroup.addTo(map);
    }

    // Auto-fit map bounds to the CShelph points (padding) on first result load
    const lats = pts.map(p => p.lat), lons = pts.map(p => p.lon);
    if (lats.length > 0) {
      map.flyToBounds(
        [[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]],
        { padding: [50, 50], duration: 0.6, maxZoom: 14 }
      );
    }
  }, [icesat2Results, vis]);

  // CSV preview (show uploaded points on map before extract)
  useEffect(() => {
    const map = mapInst.current; if (!map) return;
    if (layersRef.current.csvLayer) { map.removeLayer(layersRef.current.csvLayer); delete layersRef.current.csvLayer; }
    if (csvPreview && csvPreview.length > 0) {
      const g = L.layerGroup();
      const step = Math.max(1, Math.floor(csvPreview.length / 500));
      csvPreview.forEach((p, i) => { if (i % step !== 0) return;
        g.addLayer(L.circleMarker([p.lat, p.lon], { radius: 3, fillColor: '#f59e0b', fillOpacity: 0.7, stroke: true, weight: 0.5, color: '#fff' })
          .bindPopup(`<div style="font-family:monospace;font-size:11px"><b style="color:#f59e0b">${PT_LABEL.insitu}</b><br>Depth: <b>${p.depth.toFixed(1)}m</b></div>`));
      });
      g.addTo(map); layersRef.current.csvLayer = g;
      const lats = csvPreview.map(p => p.lat), lons = csvPreview.map(p => p.lon);
      if (lats.length > 0) map.fitBounds([[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]], { padding: [40, 40] });
    }
  }, [csvPreview]);

  // ─── optical_valid deep-abstain mask overlay ──────────────────────────────
  //   Honesty by design: on Analyse of a Khalifa-VHR (in-situ-anchored) result,
  //   render the optical_valid mask so the dredged DEEP BASIN looks visibly
  //   "not satellite-measured" — greyed + hatched, distinct from the shallow
  //   optical-valid zone and its collar. Wired to GET /api/results/<id>/mask.
  //
  //   Contract (all OPTIONAL; degrades gracefully — never fabricates a mask):
  //     event 'showResultMask' detail = {
  //       id, name,
  //       maskUrl,                 // GET endpoint for the mask (PNG / JSON)
  //       apiBase,                 // base URL to resolve relative urls
  //       bbox: [w,s,e,n],         // overlay extent (required to place a raster)
  //       zOpt, zCollar,           // legend depth thresholds
  //       counts: {valid,collar,deep}   // optional pixel counts for the legend
  //     }
  //   The mask endpoint may reply with { png_b64 | mask_png_b64, bbox?,
  //   z_opt_m?, z_opt_collar_m?, counts? } or { mask_url, bbox? }. If no raster
  //   is available we still show the LEGEND + an honest "deep basin = not
  //   satellite-measured" note (no synthetic overlay).
  useEffect(() => {
    const map = mapInst.current; if (!map) return undefined;

    const clearMask = () => {
      if (layersRef.current.maskLayer) { map.removeLayer(layersRef.current.maskLayer); delete layersRef.current.maskLayer; }
    };

    const onShow = async (ev) => {
      const det = (ev && ev.detail) || {};
      const apiBase = det.apiBase || '';
      let bbox = Array.isArray(det.bbox) && det.bbox.length >= 4 ? det.bbox : null;
      let pngB64 = null;
      let counts = det.counts || null;
      let zOpt = det.zOpt, zCollar = det.zCollar;
      let rasterShown = false;

      // Try to fetch the actual mask raster from the backend (may 404 until the
      // developer ships GET /api/results/<id>/mask — degrade gracefully).
      if (det.maskUrl) {
        try {
          const r = await fetch(`${apiBase}${det.maskUrl}`);
          const ct = (r.headers.get('content-type') || '').toLowerCase();
          if (r.ok && ct.includes('json')) {
            const d = await r.json();
            pngB64 = d.png_b64 || d.mask_png_b64 || null;
            if (!bbox && Array.isArray(d.bbox) && d.bbox.length >= 4) bbox = d.bbox;
            if (d.counts) counts = d.counts;
            if (zOpt == null && d.z_opt_m != null) zOpt = d.z_opt_m;
            if (zCollar == null && d.z_opt_collar_m != null) zCollar = d.z_opt_collar_m;
            // A relative raster URL inside the JSON (e.g. a PNG tile) is also fine.
            if (!pngB64 && d.mask_url) {
              clearMask();
              if (bbox) {
                const bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]];
                layersRef.current.maskLayer = L.imageOverlay(`${apiBase}${d.mask_url}`, bounds, { opacity: 0.7, interactive: false, className: 'optical-valid-mask' });
                if (maskVisible) layersRef.current.maskLayer.addTo(map);
                rasterShown = true;
              }
            }
          } else if (r.ok && ct.includes('image')) {
            // Direct image bytes — overlay the URL itself.
            clearMask();
            if (bbox) {
              const bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]];
              layersRef.current.maskLayer = L.imageOverlay(`${apiBase}${det.maskUrl}`, bounds, { opacity: 0.7, interactive: false, className: 'optical-valid-mask' });
              if (maskVisible) layersRef.current.maskLayer.addTo(map);
              rasterShown = true;
            }
          }
        } catch (_) { /* offline / not yet shipped — fall through to legend-only */ }
      }

      if (!rasterShown && pngB64 && bbox) {
        clearMask();
        const bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]];
        layersRef.current.maskLayer = L.imageOverlay(`data:image/png;base64,${pngB64}`, bounds, { opacity: 0.7, interactive: false, className: 'optical-valid-mask' });
        if (maskVisible) layersRef.current.maskLayer.addTo(map);
        rasterShown = true;
      }

      setMaskVisible(true);
      setMaskInfo({
        id: det.id, name: det.name,
        zOpt: zOpt != null ? zOpt : 10.5,
        zCollar: zCollar != null ? zCollar : 12.0,
        counts,
        rasterShown,
      });
      if (bbox) map.flyToBounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], { padding: [40, 40], duration: 0.7 });
    };

    const onClear = () => { clearMask(); setMaskInfo(null); };

    window.addEventListener('showResultMask', onShow);
    window.addEventListener('clearResultMask', onClear);
    return () => {
      window.removeEventListener('showResultMask', onShow);
      window.removeEventListener('clearResultMask', onClear);
    };
  }, [maskVisible]);

  const toggleMask = () => {
    const map = mapInst.current; if (!map) return;
    const layer = layersRef.current.maskLayer;
    setMaskVisible(v => {
      const nv = !v;
      if (layer) { nv ? layer.addTo(map) : map.removeLayer(layer); }
      return nv;
    });
  };

  const toggle = (k) => {
    const nv = { ...vis, [k]: !vis[k] };
    setVis(nv);
    const map = mapInst.current; if (!map) return;
    const lk = { depth: 'rasterLayer', gebco: 'gebcoLayer', icesat: 'icesatLayer', cshelph: 'cshelphPts' };
    const layer = layersRef.current[lk[k]];
    if (layer) { nv[k] ? layer.addTo(map) : map.removeLayer(layer); }
    // Also toggle CShelph tracks together with the CShelph points
    if (k === 'cshelph') {
      const tr = layersRef.current.cshelphTracks;
      if (tr) { nv[k] ? tr.addTo(map) : map.removeLayer(tr); }
    }
  };

  return (
    <div style={{ ...style, position: 'relative' }}>
      <div ref={mapRef} style={{ width: '100%', height: '100%' }} />

      {/* Measure-distance tool toggle + clear (user-requested, to verify mask offset at quays) */}
      <div style={{ position: 'absolute', bottom: '12px', left: '12px', zIndex: 1000, display: 'flex', gap: 6 }}>
        <button onClick={() => { if (measuring) { setMeasuring(false); } else { setMeasurePts([]); setMeasuring(true); } }}
          title="Measure distance — click to add points, double-click or ESC to finish"
          style={{
            fontSize: '10px', fontFamily: 'monospace', fontWeight: 700, padding: '6px 10px',
            borderRadius: '8px', border: measuring ? '1.5px solid #f59e0b' : '1px solid rgba(0,0,0,0.12)',
            background: measuring ? '#f59e0b' : 'rgba(255,255,255,0.95)',
            color: measuring ? '#1e293b' : '#374151', cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          }}>
          📏 {measuring ? 'Measuring… (dblclick/ESC to finish)' : 'Measure distance'}
        </button>
        {measurePts.length > 0 && (
          <button onClick={() => { setMeasurePts([]); setMeasuring(false); }}
            style={{
              fontSize: '10px', fontFamily: 'monospace', fontWeight: 700, padding: '6px 10px',
              borderRadius: '8px', border: '1px solid rgba(0,0,0,0.12)', background: 'rgba(255,255,255,0.95)',
              color: '#dc2626', cursor: 'pointer', boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
            }}>
            Clear
          </button>
        )}
      </div>

      {/* ADPorts F1/F2 — floating chip while a per-scene/Planet-date overlay is active */}
      {overlayOverride && (
        <div style={{
          position: 'absolute', top: '12px', left: '50%', transform: 'translateX(-50%)',
          zIndex: 1001, background: 'rgba(2,132,199,0.95)', backdropFilter: 'blur(12px)',
          borderRadius: '8px', border: '1px solid rgba(255,255,255,0.25)', padding: '6px 10px',
          display: 'flex', alignItems: 'center', gap: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.25)',
        }}>
          <span style={{ fontSize: '9.5px', fontFamily: 'monospace', color: '#fff', fontWeight: 700 }}>
            {overlayOverride.label || 'Scene overlay'}
          </span>
          <button onClick={() => onShowOverlay && onShowOverlay(null)} style={{
            fontSize: '9px', fontFamily: 'monospace', fontWeight: 700, color: '#0284c7',
            background: '#fff', border: 'none', borderRadius: '6px', padding: '3px 8px', cursor: 'pointer',
          }}>
            ← back to composite
          </button>
        </div>
      )}

      {(results || icesat2Results) && (
        <div style={{ position: 'absolute', top: '12px', left: '12px', zIndex: 1000, background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(12px)', borderRadius: '8px', border: '1px solid rgba(0,0,0,0.12)', padding: '10px', display: 'flex', flexDirection: 'column', gap: '2px', boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}>
          <p style={{ fontSize: '8px', fontFamily: 'monospace', color: '#374151', fontWeight: 700, letterSpacing: '0.12em', marginBottom: '4px' }}>LAYERS</p>
          {(INTERNAL_ANALYSIS ? [
            { key: 'depth', label: 'Bathymetry', color: '#1491d2' },
            { key: 'gebco', label: 'GEBCO Ref', color: '#d97706' },
            { key: 'icesat', label: 'ICESat-2', color: '#dc2626' },
            { key: 'cshelph', label: 'CShelph depths', color: '#ef4444' },
          ] : [
            { key: 'depth', label: 'Depth', color: '#1491d2' },
            { key: 'gebco', label: 'Reference points', color: '#d97706' },
            { key: 'icesat', label: 'Survey points', color: '#dc2626' },
            { key: 'cshelph', label: 'Lidar depths', color: '#ef4444' },
          ]).map(l => (
            <button key={l.key} onClick={() => toggle(l.key)} style={{
              display: 'flex', alignItems: 'center', gap: '7px', padding: '4px 8px',
              fontSize: '10px', fontFamily: 'monospace', fontWeight: 500,
              background: vis[l.key] ? 'rgba(20,145,210,0.08)' : 'transparent',
              border: 'none', borderRadius: '4px', cursor: 'pointer',
              color: vis[l.key] ? '#1e293b' : '#94a3b8',
            }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '3px', background: vis[l.key] ? l.color : '#cbd5e1', opacity: vis[l.key] ? 1 : 0.4 }} />
              {l.label}
            </button>
          ))}
        </div>
      )}

      {results && (() => {
        const maxDepth = results.raster_max_depth || results.stats?.max_depth || 25;
        const ticks = rampTicks(maxDepth, 5);
        return (
        <div style={{ position: 'absolute', bottom: '50px', right: '12px', zIndex: 1000, background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(12px)', borderRadius: '8px', border: '1px solid rgba(0,0,0,0.12)', padding: '12px 14px', boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}>
          <p style={{ fontSize: '9px', fontFamily: 'monospace', color: '#1e293b', fontWeight: 700, letterSpacing: '0.08em', marginBottom: '1px' }}>DEPTH</p>
          <p style={{ fontSize: '8px', fontFamily: 'monospace', color: '#64748b', fontWeight: 600, marginBottom: '8px' }}>metres · MSL</p>
          <div style={{ display: 'flex', alignItems: 'stretch', gap: '6px' }}>
            <div style={{ width: '14px', borderRadius: '3px', background: rampGradientCss('to bottom'), border: '1px solid rgba(0,0,0,0.08)' }} />
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', fontSize: '9px', fontFamily: 'monospace', color: '#374151', fontWeight: 600, height: '100px' }}>
              {ticks.map((t, i) => <span key={i}>{t.depth.toFixed(t.depth < 10 ? 1 : 0)}</span>)}
            </div>
          </div>
          {INTERNAL_ANALYSIS && results.ml_stats?.fusion_stats && (
            <div style={{ marginTop: '8px', paddingTop: '6px', borderTop: '1px solid #e2e8f0', fontSize: '8px', fontFamily: 'monospace', color: '#64748b' }}>
              <div>Bias: <b style={{color:'#059669'}}>{results.ml_stats.fusion_stats.final_bias?.toFixed(2)}m</b></div>
              <div>RMSE: <b style={{color:'#dc2626'}}>{results.ml_stats.fusion_stats.final_rmse?.toFixed(2)}m</b></div>
            </div>
          )}
        </div>
        );
      })()}
      {/* ═══ optical_valid deep-abstain mask LEGEND (Khalifa VHR honesty) ═══ */}
      {maskInfo && (() => {
        const c = maskInfo.counts || {};
        const tot = (c.valid || 0) + (c.collar || 0) + (c.deep || 0);
        const pct = (n) => (tot > 0 && n != null ? `${((n / tot) * 100).toFixed(0)}%` : null);
        const rows = INTERNAL_ANALYSIS ? [
          { key: 'valid',  label: `Optical-valid (≤ ${maskInfo.zOpt} m)`, swatch: 'rgba(14,116,144,0.55)', hatch: false, n: c.valid },
          { key: 'collar', label: `Collar (${maskInfo.zOpt}–${maskInfo.zCollar} m)`, swatch: 'rgba(161,98,7,0.55)', hatch: false, n: c.collar },
          { key: 'deep',   label: `Deep-abstain (> ${maskInfo.zCollar} m · CATZOC C-D)`, swatch: 'rgba(107,114,128,0.55)', hatch: true, n: c.deep },
        ] : [
          { key: 'valid',  label: `Measured (≤ ${maskInfo.zOpt} m)`, swatch: 'rgba(14,116,144,0.55)', hatch: false, n: c.valid },
          { key: 'collar', label: `Transition (${maskInfo.zOpt}–${maskInfo.zCollar} m)`, swatch: 'rgba(161,98,7,0.55)', hatch: false, n: c.collar },
          { key: 'deep',   label: `Charted (> ${maskInfo.zCollar} m)`, swatch: 'rgba(107,114,128,0.55)', hatch: true, n: c.deep },
        ];
        return (
          <div style={{ position: 'absolute', top: '12px', right: '12px', zIndex: 1000, width: 230, background: 'rgba(255,255,255,0.96)', backdropFilter: 'blur(12px)', borderRadius: '8px', border: '1px solid rgba(202,138,4,0.45)', padding: '10px 12px', boxShadow: '0 2px 10px rgba(0,0,0,0.18)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
              <p style={{ margin: 0, fontSize: '8.5px', fontFamily: 'monospace', fontWeight: 800, letterSpacing: '0.08em', color: '#854d0e' }}>{INTERNAL_ANALYSIS ? 'OPTICAL-VALID MASK' : 'DEPTH COVERAGE'}</p>
              <div style={{ display: 'flex', gap: 4 }}>
                {maskInfo.rasterShown && (
                  <button onClick={toggleMask} title="Toggle the mask overlay" style={{ fontSize: '8px', fontFamily: 'monospace', fontWeight: 700, padding: '2px 6px', borderRadius: 4, border: '1px solid #cbd5e1', background: maskVisible ? '#fef3c7' : 'rgba(255,255,255,0.9)', color: '#92400e', cursor: 'pointer' }}>{maskVisible ? 'HIDE' : 'SHOW'}</button>
                )}
                <button onClick={() => window.dispatchEvent(new CustomEvent('clearResultMask'))} title="Remove the mask layer" style={{ fontSize: '8px', fontFamily: 'monospace', fontWeight: 700, padding: '2px 6px', borderRadius: 4, border: '1px solid #cbd5e1', background: 'rgba(255,255,255,0.9)', color: '#64748b', cursor: 'pointer' }}>✕</button>
              </div>
            </div>
            {rows.map(r => (
              <div key={r.key} style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '2px 0' }}>
                <span style={{ flexShrink: 0, width: 13, height: 13, borderRadius: 3, border: '1px solid rgba(0,0,0,0.15)',
                  background: r.hatch
                    ? `repeating-linear-gradient(45deg, ${r.swatch}, ${r.swatch} 3px, rgba(75,85,99,0.85) 3px, rgba(75,85,99,0.85) 6px)`
                    : r.swatch }} />
                <span style={{ flex: 1, fontSize: '8.5px', fontFamily: 'monospace', color: '#334155', lineHeight: 1.25 }}>{r.label}</span>
                {pct(r.n) && <span style={{ fontSize: '8px', fontFamily: 'monospace', fontWeight: 700, color: '#64748b' }}>{pct(r.n)}</span>}
              </div>
            ))}
            <p style={{ margin: '6px 0 0', paddingTop: 6, borderTop: '1px solid #f1d9a3', fontSize: '7.5px', fontFamily: 'monospace', lineHeight: 1.4, color: '#92400e' }}>
              {INTERNAL_ANALYSIS
                ? <>The deep basin is <b>not satellite-measured</b> — depth there is in-situ-anchored (charted/maintained), not an SDB retrieval.</>
                : <>Deep areas show <b>charted depth</b> rather than directly measured depth.</>}
              {INTERNAL_ANALYSIS && !maskInfo.rasterShown && <> <span style={{ color: '#b45309' }}>Mask raster pending backend <code>/api/results/&lt;id&gt;/mask</code>; legend shown from pixel counts.</span></>}
            </p>
          </div>
        );
      })()}

      {/* ═══ General Results Popup — any method ═══ */}
      {results?.stats && !results?.ml_stats?.caballero_stats && showStats && (
        <div style={{ position:'absolute', top:0, left:0, right:0, bottom:0, zIndex:2000, background:'rgba(0,0,0,0.5)', display:'flex', alignItems:'center', justifyContent:'center' }}
             onClick={(e) => { if(e.target === e.currentTarget) setShowStats(false); }}>
          <div style={{ background:'#fff', borderRadius:'12px', padding:'24px', maxWidth:'500px', width:'95%', maxHeight:'85vh', overflow:'auto', boxShadow:'0 20px 60px rgba(0,0,0,0.3)' }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'16px' }}>
              <div>
                <h3 style={{ margin:0, fontSize:'16px', fontFamily:'system-ui', fontWeight:800, color:'#1e293b' }}>
                  Bathymetry Results
                </h3>
                <p style={{ margin:'2px 0 0', fontSize:'11px', fontFamily:'monospace', color:'#64748b' }}>
                  {results.stats?.grid_points?.toLocaleString()} depth points @ {results.stats?.resolution_m || '?'}m
                </p>
              </div>
              <button onClick={() => setShowStats(false)} style={{ background:'none', border:'none', fontSize:'20px', cursor:'pointer', color:'#94a3b8' }}>x</button>
            </div>
            <div style={{ display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap:'10px', marginBottom:'16px' }}>
              {(() => {
                // Pull stats robustly from any of the shapes the various
                // pipelines return, so this card never renders "undefinedm".
                const s = results.stats || {};
                const ml = results.ml_stats || {};
                const m = results.metrics || {};
                const fmt = (v, d=1, suffix='m') =>
                  (v === null || v === undefined || Number.isNaN(Number(v))) ? '-' : `${Number(v).toFixed(d)}${suffix}`;
                const meanV = s.mean_depth ?? m.depth_mean_m ?? m.mean_depth_m;
                const maxV  = s.max_depth  ?? m.depth_max_m  ?? m.max_depth_m;
                const minV  = s.min_depth  ?? m.depth_min_m  ?? m.min_depth_m;
                const stdV  = s.std_depth  ?? m.depth_std_m  ?? m.std_depth_m;
                const r2V   = ml.r2 ?? m.r2;
                const nT    = ml.n_train ?? m.n_train ?? s.n_train;
                return [
                  // R\u00B2 / N Train are accuracy/training internals \u2014 INTERNAL only.
                  ...(INTERNAL_ANALYSIS ? [{ label:'R\u00B2', value: (r2V === null || r2V === undefined) ? '-' : Number(r2V).toFixed(3), color:'#2563eb' }] : []),
                  { label:'Mean Depth',   value: fmt(meanV, 1), color:'#0891b2' },
                  { label:'Max Depth',    value: fmt(maxV,  1), color:'#dc2626' },
                  { label:'Min Depth',    value: fmt(minV,  1), color:'#059669' },
                  { label:'Std Dev',      value: fmt(stdV,  2), color:'#7c3aed' },
                  ...(INTERNAL_ANALYSIS ? [{ label:'N Train', value: (nT === null || nT === undefined) ? '-' : Number(nT).toLocaleString(), color:'#d97706' }] : []),
                ];
              })().map((s,i) => (
                <div key={i} style={{ background:'#f8fafc', borderRadius:'8px', padding:'10px 12px', border:'1px solid #e2e8f0' }}>
                  <div style={{ fontSize:'9px', fontFamily:'monospace', color:'#94a3b8', fontWeight:600, letterSpacing:'0.08em' }}>{s.label}</div>
                  <div style={{ fontSize:'18px', fontFamily:'monospace', fontWeight:800, color:s.color, marginTop:'2px' }}>{s.value}</div>
                </div>
              ))}
            </div>
            {/* Source pills hidden — UI shows only the depth result, not the
                technical pipeline / data provenance. */}
            {false && results.sources_used && (
              <div style={{ marginBottom:'14px' }}>
                <p style={{ fontSize:'9px', fontFamily:'monospace', color:'#64748b', fontWeight:700, marginBottom:'6px' }}>SOURCES</p>
                <div style={{ display:'flex', flexWrap:'wrap', gap:'4px' }}>
                  {results.sources_used.map((s,i) => (
                    <span key={i} style={{ fontSize:'9px', fontFamily:'monospace', color:'#0d9488', background:'rgba(13,148,136,0.08)', padding:'3px 8px', borderRadius:'4px' }}>{s}</span>
                  ))}
                </div>
              </div>
            )}
            {/* Export */}
            {results.geotiff_b64 ? (
              <button onClick={() => {
                const bin = atob(results.geotiff_b64); const arr = new Uint8Array(bin.length);
                for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
                const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([arr],{type:'image/tiff'})); a.download='bathymetry.tif'; a.click();
              }} style={{ width:'100%', padding:'14px', fontSize:'13px', fontFamily:'monospace', fontWeight:800, borderRadius:'8px', border:'none', cursor:'pointer', background:'linear-gradient(135deg,#0284c7,#0d9488)', color:'#fff', boxShadow:'0 4px 12px rgba(2,132,199,0.3)' }}>
                Export GeoTIFF
              </button>
            ) : (
              <p style={{ fontSize:'10px', fontFamily:'monospace', color:'#94a3b8', textAlign:'center' }}>GeoTIFF generating...</p>
            )}
          </div>
        </div>
      )}

      {/* ═══ Method 2 stats + scatter — INTERNAL ONLY (accuracy/method/validation) ═══ */}
      {INTERNAL_ANALYSIS && results?.ml_stats?.caballero_stats && (() => {
        const cs = results.ml_stats.caballero_stats;
        const coeff = results.ml_stats.coefficients || {};
        const sc = results.scatter || {};
        const enhanced = cs.cnn_enhanced;
        const maxD = Math.max(...(sc.reference||[1]), ...(sc.predicted||[1]), 1);
        return showM2 && (
        <div style={{ position:'absolute', top:0, left:0, right:0, bottom:0, zIndex:2000, background:'rgba(0,0,0,0.5)', display:'flex', alignItems:'center', justifyContent:'center' }}
             onClick={(e) => { if(e.target === e.currentTarget) setShowM2(false); }}>
          <div style={{ background:'#fff', borderRadius:'12px', padding:'24px', maxWidth:'680px', width:'95%', maxHeight:'90vh', overflow:'auto', boxShadow:'0 20px 60px rgba(0,0,0,0.3)' }}>
            {/* Header */}
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'16px' }}>
              <div>
                <h3 style={{ margin:0, fontSize:'16px', fontFamily:'system-ui', fontWeight:800, color:'#1e293b' }}>
                  {enhanced ? 'Band-ratio + CNN residual' : 'Band-ratio depth'}
                </h3>
                <p style={{ margin:'2px 0 0', fontSize:'11px', fontFamily:'monospace', color:'#64748b' }}>
                  {enhanced ? 'Band-ratio + residual correction' : 'Band-ratio bathymetry'}
                </p>
              </div>
              <button onClick={() => setShowM2(false)} style={{ background:'none', border:'none', fontSize:'20px', cursor:'pointer', color:'#94a3b8' }}>x</button>
            </div>

            {/* Stats grid */}
            <div style={{ display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap:'10px', marginBottom:'18px' }}>
              {[
                { label:'R\u00B2', value: (cs.final_r2 ?? cs.r2)?.toFixed(3), color:'#2563eb' },
                { label:'RMSE', value: (cs.final_rmse ?? cs.rmse)?.toFixed(2)+'m', color:'#dc2626' },
                { label:'MedAE', value: (cs.final_medae ?? cs.medae)?.toFixed(2)+'m', color:'#059669' },
                { label:'Bias', value: (cs.final_bias ?? cs.bias)?.toFixed(3)+'m', color:'#7c3aed' },
                { label:'IQR', value: cs.iqr?.toFixed(2)+'m', color:'#d97706' },
                { label:'N pts', value: cs.n_val, color:'#0891b2' },
              ].map((s,i) => (
                <div key={i} style={{ background:'#f8fafc', borderRadius:'8px', padding:'10px 12px', border:'1px solid #e2e8f0' }}>
                  <div style={{ fontSize:'9px', fontFamily:'monospace', color:'#94a3b8', fontWeight:600, letterSpacing:'0.08em' }}>{s.label}</div>
                  <div style={{ fontSize:'18px', fontFamily:'monospace', fontWeight:800, color:s.color, marginTop:'2px' }}>{s.value}</div>
                </div>
              ))}
            </div>

            {/* Switching zones */}
            <div style={{ display:'flex', gap:'8px', marginBottom:'18px' }}>
              {[
                { label:'Shallow (<2m)', n: cs.n_shallow_sdbred, color:'#ef4444' },
                { label:'Blend (2-3.5m)', n: cs.n_transition, color:'#f59e0b' },
                { label:'Deep (>3.5m)', n: cs.n_deep_sdbgreen, color:'#10b981' },
              ].map((z,i) => (
                <div key={i} style={{ flex:1, background:`${z.color}10`, border:`1px solid ${z.color}30`, borderRadius:'6px', padding:'8px', textAlign:'center' }}>
                  <div style={{ fontSize:'14px', fontWeight:800, fontFamily:'monospace', color:z.color }}>{(z.n||0).toLocaleString()}</div>
                  <div style={{ fontSize:'8px', fontFamily:'monospace', color:'#64748b' }}>{z.label}</div>
                </div>
              ))}
            </div>

            {/* Scatter plot — SVG */}
            {sc.predicted && sc.predicted.length > 0 && (
              <div style={{ marginBottom:'14px' }}>
                <p style={{ fontSize:'10px', fontFamily:'monospace', fontWeight:700, color:'#374151', marginBottom:'6px' }}>
                  Predicted vs Reference (n={sc.predicted.length})
                </p>
                <svg viewBox="0 0 300 300" style={{ width:'100%', maxWidth:'400px', background:'#f8fafc', borderRadius:'8px', border:'1px solid #e2e8f0' }}>
                  {/* Grid lines */}
                  {[0.2,0.4,0.6,0.8].map(f => (
                    <React.Fragment key={f}>
                      <line x1={30} y1={270*f+15} x2={285} y2={270*f+15} stroke="#e2e8f0" strokeWidth="0.5"/>
                      <line x1={270*f+30} y1={15} x2={270*f+30} y2={285} stroke="#e2e8f0" strokeWidth="0.5"/>
                    </React.Fragment>
                  ))}
                  {/* 1:1 line */}
                  <line x1={30} y1={285} x2={285} y2={30} stroke="#94a3b8" strokeWidth="1" strokeDasharray="4,3"/>
                  {/* Points */}
                  {sc.predicted.map((p, i) => {
                    const x = 30 + (sc.reference[i] / maxD) * 255;
                    const y = 285 - (p / maxD) * 255;
                    return <circle key={i} cx={x} cy={y} r="2.5" fill="#2563eb" fillOpacity="0.4" stroke="#1d4ed8" strokeWidth="0.3"/>;
                  })}
                  {/* Axes labels */}
                  <text x={150} y={298} textAnchor="middle" fontSize="9" fontFamily="monospace" fill="#64748b">Reference (m)</text>
                  <text x={8} y={150} textAnchor="middle" fontSize="9" fontFamily="monospace" fill="#64748b" transform="rotate(-90,8,150)">Predicted (m)</text>
                  {/* Tick labels */}
                  <text x={30} y={296} fontSize="8" fontFamily="monospace" fill="#94a3b8">0</text>
                  <text x={280} y={296} fontSize="8" fontFamily="monospace" fill="#94a3b8">{maxD.toFixed(0)}</text>
                  <text x={4} y={288} fontSize="8" fontFamily="monospace" fill="#94a3b8">0</text>
                  <text x={4} y={20} fontSize="8" fontFamily="monospace" fill="#94a3b8">{maxD.toFixed(0)}</text>
                </svg>
              </div>
            )}

            {/* Coefficients */}
{/* model coefficients hidden from UI — kept internal for reproducibility */}

            {/* Export GeoTIFF */}
            {results.geotiff_b64 ? (
              <button onClick={() => {
                const bin = atob(results.geotiff_b64);
                const arr = new Uint8Array(bin.length);
                for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
                const blob = new Blob([arr], {type:'image/tiff'});
                const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'bathymetry.tif'; a.click();
              }} style={{ width:'100%', padding:'14px', fontSize:'13px', fontFamily:'monospace', fontWeight:800, borderRadius:'8px', border:'none', cursor:'pointer', background:'linear-gradient(135deg,#0284c7,#0d9488)', color:'#fff', boxShadow:'0 4px 12px rgba(2,132,199,0.3)' }}>
                Export GeoTIFF
              </button>
            ) : (
              <p style={{ fontSize:'10px', fontFamily:'monospace', color:'#94a3b8', textAlign:'center' }}>GeoTIFF not available — run extraction first</p>
            )}
          </div>
        </div>
        ); })()}
    </div>
  );
}
