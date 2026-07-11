import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';

// IHO S-52 depth classes — colour per class (sounding text + legend)
const IHO_PALETTE = [
  { max: 2.0,  label: '0–2m',   color: '#ca8a04', tint: 'rgba(253,224,71,0.28)' },
  { max: 5.0,  label: '2–5m',   color: '#ea580c', tint: 'rgba(251,146,60,0.25)' },
  { max: 10.0, label: '5–10m',  color: '#0ea5e9', tint: 'rgba(14,165,233,0.22)' },
  { max: 15.0, label: '10–15m', color: '#2563eb', tint: 'rgba(37,99,235,0.22)' },
  { max: 20.0, label: '15–20m', color: '#1e3a8a', tint: 'rgba(30,58,138,0.25)' },
  { max: 9e9,  label: '>20m',   color: '#312e81', tint: 'rgba(49,46,129,0.28)' },
];
const classOf = (d) => {
  for (let i = 0; i < IHO_PALETTE.length; i++) if (d < IHO_PALETTE[i].max) return i;
  return IHO_PALETTE.length - 1;
};

// ── Marching-squares isobath generator on a regular grid ───────────────
//   grid: Float32Array (row-major), w, h, NaN for no-data
//   Returns a list of {level, segs: [[[x0,y0],[x1,y1]], ...]} in grid coords
function marchingSquares(grid, w, h, levels) {
  const result = [];
  for (const L of levels) {
    const segs = [];
    const at = (r, c) => grid[r * w + c];
    for (let r = 0; r < h - 1; r++) {
      for (let c = 0; c < w - 1; c++) {
        const v00 = at(r, c), v01 = at(r, c + 1), v11 = at(r + 1, c + 1), v10 = at(r + 1, c);
        if (!Number.isFinite(v00) || !Number.isFinite(v01) || !Number.isFinite(v10) || !Number.isFinite(v11)) continue;
        let idx = 0;
        if (v00 >= L) idx |= 1;
        if (v01 >= L) idx |= 2;
        if (v11 >= L) idx |= 4;
        if (v10 >= L) idx |= 8;
        if (idx === 0 || idx === 15) continue;
        const lerp = (va, vb, a, b) => {
          const t = (L - va) / (vb - va || 1e-10);
          return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        };
        const top    = () => lerp(v00, v01, [c, r], [c + 1, r]);
        const right  = () => lerp(v01, v11, [c + 1, r], [c + 1, r + 1]);
        const bottom = () => lerp(v10, v11, [c, r + 1], [c + 1, r + 1]);
        const left   = () => lerp(v00, v10, [c, r], [c, r + 1]);
        // Simple (ignoring saddle-point disambiguation, OK for smooth bathy)
        switch (idx) {
          case 1:  case 14: segs.push([top(), left()]); break;
          case 2:  case 13: segs.push([top(), right()]); break;
          case 3:  case 12: segs.push([left(), right()]); break;
          case 4:  case 11: segs.push([bottom(), right()]); break;
          case 5:          segs.push([top(), left()]); segs.push([bottom(), right()]); break;
          case 6:  case 9:  segs.push([top(), bottom()]); break;
          case 7:  case 8:  segs.push([left(), bottom()]); break;
          case 10:         segs.push([top(), right()]); segs.push([left(), bottom()]); break;
          default: break;
        }
      }
    }
    result.push({ level: L, segs });
  }
  return result;
}

export default function IHOChartView({ points, bbox, stats, rasterPng, rasterBounds, metadata, onPointClick }) {
  const canvasRef = useRef(null);
  const wrapperRef = useRef(null);
  const [rasterImg, setRasterImg] = useState(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState(null);
  const [hover, setHover] = useState(null);
  const [showSoundings, setShowSoundings] = useState(true);
  const [showRaster, setShowRaster] = useState(true);
  const [showContours, setShowContours] = useState(true);
  const [contourInterval, setContourInterval] = useState(5);  // meters; default 5m
  const [soundingStep, setSoundingStep] = useState('auto');   // auto | 1 | 2 | 5 (denser/sparser)

  // Load raster PNG as an Image once it arrives
  useEffect(() => {
    if (!rasterPng) { setRasterImg(null); return; }
    const img = new Image();
    img.onload = () => setRasterImg(img);
    img.onerror = () => setRasterImg(null);
    img.src = rasterPng.startsWith('data:') ? rasterPng : `data:image/png;base64,${rasterPng}`;
  }, [rasterPng]);

  // Sort / filter soundings
  const soundings = useMemo(() => {
    if (!points) return [];
    return points.filter(p =>
      (p.photon_class === 'interpolated' || p.photon_class === 'bathymetry' || p.photon_class === 'gebco')
      && Number.isFinite(p.depth) && p.depth > 0);
  }, [points]);

  // Build a depth grid on (lon, lat) for marching squares
  // Uses max density of soundings — sparse cells are filled via simple NN from nearest cell
  const depthGrid = useMemo(() => {
    if (!bbox || !soundings.length) return null;
    const GW = 160, GH = 160;
    const gsum = new Float32Array(GW * GH);
    const gcnt = new Int32Array(GW * GH);
    const lonSpan = bbox.east - bbox.west + 1e-10;
    const latSpan = bbox.north - bbox.south + 1e-10;
    for (const p of soundings) {
      const cx = Math.floor((p.lon - bbox.west) / lonSpan * (GW - 1));
      const cy = Math.floor((bbox.north - p.lat) / latSpan * (GH - 1));
      if (cx < 0 || cx >= GW || cy < 0 || cy >= GH) continue;
      const k = cy * GW + cx;
      gsum[k] += p.depth; gcnt[k]++;
    }
    const grid = new Float32Array(GW * GH).fill(NaN);
    for (let k = 0; k < GW * GH; k++) if (gcnt[k] > 0) grid[k] = gsum[k] / gcnt[k];
    // Fill empty cells by propagation (2 passes of nearest valid neighbour)
    for (let pass = 0; pass < 2; pass++) {
      const next = grid.slice();
      for (let r = 0; r < GH; r++) {
        for (let c = 0; c < GW; c++) {
          const k = r * GW + c;
          if (Number.isFinite(grid[k])) continue;
          let sum = 0, n = 0;
          for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
            const rr = r + dr, cc = c + dc;
            if (rr < 0 || rr >= GH || cc < 0 || cc >= GW) continue;
            const v = grid[rr * GW + cc];
            if (Number.isFinite(v)) { sum += v; n++; }
          }
          if (n > 0) next[k] = sum / n;
        }
      }
      for (let k = 0; k < GW * GH; k++) grid[k] = next[k];
    }
    return { grid, w: GW, h: GH };
  }, [soundings, bbox]);

  // Generate contours at the requested interval
  const contours = useMemo(() => {
    if (!depthGrid || !showContours) return [];
    const { grid, w, h } = depthGrid;
    const maxD = stats?.max_depth || 25;
    const levels = [];
    for (let L = contourInterval; L < maxD + contourInterval; L += contourInterval) levels.push(L);
    return marchingSquares(grid, w, h, levels);
  }, [depthGrid, contourInterval, showContours, stats]);

  // Observe container size
  useEffect(() => {
    if (!wrapperRef.current) return;
    const ro = new ResizeObserver(entries => {
      for (const e of entries) {
        const r = e.contentRect;
        setSize({ w: Math.max(400, Math.floor(r.width)), h: Math.max(400, Math.floor(r.height)) });
      }
    });
    ro.observe(wrapperRef.current);
    return () => ro.disconnect();
  }, []);

  const project = useCallback((lon, lat) => {
    if (!bbox) return [0, 0];
    const sx = (lon - bbox.west) / (bbox.east - bbox.west + 1e-10) * size.w;
    const sy = (bbox.north - lat) / (bbox.north - bbox.south + 1e-10) * size.h;
    return [sx * zoom + pan.x, sy * zoom + pan.y];
  }, [bbox, size, zoom, pan]);

  const unproject = useCallback((cx, cy) => {
    if (!bbox) return [0, 0];
    const sx = (cx - pan.x) / zoom;
    const sy = (cy - pan.y) / zoom;
    return [bbox.west + sx / size.w * (bbox.east - bbox.west),
            bbox.north - sy / size.h * (bbox.north - bbox.south)];
  }, [bbox, size, zoom, pan]);

  // Render loop
  useEffect(() => {
    const cv = canvasRef.current; if (!cv || !bbox) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size.w * dpr; cv.height = size.h * dpr;
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Chart paper background (S-52 neutral land colour)
    ctx.fillStyle = '#f2efe4';
    ctx.fillRect(0, 0, size.w, size.h);

    // 1) GeoTIFF raster (pre-rendered by backend) — snaps to rasterBounds
    if (showRaster && rasterImg) {
      const rb = rasterBounds && rasterBounds.west !== undefined ? rasterBounds : bbox;
      const [x0, y0] = project(rb.west, rb.north);
      const [x1, y1] = project(rb.east, rb.south);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.globalAlpha = 0.92;
      ctx.drawImage(rasterImg, x0, y0, x1 - x0, y1 - y0);
      ctx.globalAlpha = 1;
    } else if (showRaster && depthGrid) {
      // Fallback: render tint from the in-memory grid
      const { grid, w: gw, h: gh } = depthGrid;
      const cellW = size.w * zoom / gw;
      const cellH = size.h * zoom / gh;
      for (let r = 0; r < gh; r++) {
        for (let c = 0; c < gw; c++) {
          const v = grid[r * gw + c];
          if (!Number.isFinite(v)) continue;
          ctx.fillStyle = IHO_PALETTE[classOf(v)].tint;
          ctx.fillRect(c * cellW + pan.x, r * cellH + pan.y, cellW + 0.5, cellH + 0.5);
        }
      }
    }

    // 2) Contours at user-chosen interval
    if (showContours && depthGrid && contours.length) {
      const { w: gw, h: gh } = depthGrid;
      const lonSpan = bbox.east - bbox.west, latSpan = bbox.north - bbox.south;
      const toScreen = (gx, gy) => {
        const lon = bbox.west + (gx / (gw - 1)) * lonSpan;
        const lat = bbox.north - (gy / (gh - 1)) * latSpan;
        return project(lon, lat);
      };
      for (const c of contours) {
        const cls = IHO_PALETTE[classOf(c.level)];
        const major = Math.round(c.level) % 10 === 0;  // bolder at 10m multiples
        ctx.strokeStyle = cls.color + (major ? 'ff' : 'cc');
        ctx.lineWidth = major ? 1.2 : 0.7;
        ctx.beginPath();
        for (const [[x0g, y0g], [x1g, y1g]] of c.segs) {
          const [sx0, sy0] = toScreen(x0g, y0g);
          const [sx1, sy1] = toScreen(x1g, y1g);
          ctx.moveTo(sx0, sy0); ctx.lineTo(sx1, sy1);
        }
        ctx.stroke();
      }
      // Contour depth labels along a few segments
      ctx.font = '9px JetBrains Mono, ui-monospace, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (const c of contours) {
        const cls = IHO_PALETTE[classOf(c.level)];
        const stride = Math.max(1, Math.floor(c.segs.length / 4));
        for (let i = 0; i < c.segs.length; i += stride) {
          const [[x0g, y0g]] = c.segs[i];
          const [lx, ly] = toScreen(x0g, y0g);
          if (lx < 0 || lx > size.w || ly < 0 || ly > size.h) continue;
          const txt = `${c.level}m`;
          ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,0.96)';
          ctx.strokeText(txt, lx, ly);
          ctx.fillStyle = cls.color; ctx.fillText(txt, lx, ly);
        }
      }
    }

    // 3) Graticule
    ctx.strokeStyle = 'rgba(100,116,139,0.18)'; ctx.lineWidth = 0.5;
    ctx.setLineDash([3, 4]);
    for (let f = 0.25; f < 1.0; f += 0.25) {
      const x = f * size.w * zoom + pan.x;
      const y = f * size.h * zoom + pan.y;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, size.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(size.w, y); ctx.stroke();
    }
    ctx.setLineDash([]);

    // 4) Soundings with LoD (collision-avoidance in screen space)
    let nDrawn = 0;
    if (showSoundings) {
      const labelW = 22, labelH = 13;
      const cellGridW = Math.ceil(size.w / labelW);
      const cellGridH = Math.ceil(size.h / labelH);
      const drawn = new Uint8Array(cellGridW * cellGridH);
      ctx.font = 'bold 10px JetBrains Mono, ui-monospace, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      // Priority by shallow-first (safety)
      const prioritized = soundings.slice().sort((a, b) => a.depth - b.depth);
      const stride = soundingStep === 'auto' ? 1 : parseInt(soundingStep, 10);
      for (let i = 0; i < prioritized.length; i += stride) {
        const p = prioritized[i];
        const [x, y] = project(p.lon, p.lat);
        if (x < 0 || x > size.w || y < 0 || y > size.h) continue;
        const gx = Math.floor(x / labelW);
        const gy = Math.floor(y / labelH);
        const k = gy * cellGridW + gx;
        if (drawn[k]) continue;
        drawn[k] = 1;
        if (gx + 1 < cellGridW) drawn[k + 1] = 1;
        if (gx > 0) drawn[k - 1] = 1;
        const cls = IHO_PALETTE[classOf(p.depth)];
        const label = p.depth < 10 ? p.depth.toFixed(1) : Math.round(p.depth).toString();
        ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,0.94)';
        ctx.strokeText(label, x, y);
        ctx.fillStyle = cls.color; ctx.fillText(label, x, y);
        nDrawn++;
        if (nDrawn > 9000) break;
      }
    }

    // 5) Status badge
    ctx.font = '10px JetBrains Mono, ui-monospace, monospace';
    ctx.fillStyle = 'rgba(15,23,42,0.75)';
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    const badge = `zoom ${zoom.toFixed(1)}× │ ${nDrawn}/${soundings.length} soundings │ contours every ${contourInterval}m`;
    ctx.fillText(badge, 8, 8);

    // 6) Hover crosshair
    if (hover) {
      ctx.strokeStyle = 'rgba(239,68,68,0.7)'; ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(hover.x, 0); ctx.lineTo(hover.x, size.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, hover.y); ctx.lineTo(size.w, hover.y); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [soundings, bbox, size, zoom, pan, showRaster, showSoundings, showContours,
      contours, depthGrid, rasterImg, rasterBounds, contourInterval, soundingStep, hover, project]);

  // ── Interaction
  const onWheel = useCallback((e) => {
    e.preventDefault();
    const cv = canvasRef.current; if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0016);
    const newZoom = Math.max(0.5, Math.min(40, zoom * factor));
    const dx = (mx - pan.x) * (newZoom / zoom - 1);
    const dy = (my - pan.y) * (newZoom / zoom - 1);
    setPan({ x: pan.x - dx, y: pan.y - dy });
    setZoom(newZoom);
  }, [zoom, pan]);

  const onMouseDown = (e) => { setDragging(true); setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y }); };
  const onMouseMove = (e) => {
    const cv = canvasRef.current; if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    if (dragging && dragStart) setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    setHover({ x: mx, y: my });
  };
  const onMouseUp = () => setDragging(false);
  const onMouseLeave = () => { setHover(null); setDragging(false); };

  const onClick = useCallback((e) => {
    if (!bbox) return;
    const cv = canvasRef.current; if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const [lon, lat] = unproject(mx, my);
    let best = null, bestD = 14 * 14;
    for (const p of soundings) {
      const [px, py] = project(p.lon, p.lat);
      const dx = px - mx, dy = py - my;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD) { bestD = d2; best = p; }
    }
    if (onPointClick) onPointClick({ point: best, lat, lon, metadata });
  }, [bbox, soundings, project, unproject, metadata, onPointClick]);

  useEffect(() => {
    const cv = canvasRef.current; if (!cv) return;
    const handler = (ev) => onWheel(ev);
    cv.addEventListener('wheel', handler, { passive: false });
    return () => cv.removeEventListener('wheel', handler);
  }, [onWheel]);

  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  const hoverReadout = useMemo(() => {
    if (!hover || !bbox) return null;
    const [lon, lat] = unproject(hover.x, hover.y);
    return { lat: lat.toFixed(5), lon: lon.toFixed(5) };
  }, [hover, bbox, unproject]);

  if (!bbox || !points || points.length === 0) {
    return <div style={{ padding: 20, color: 'var(--text-dim)' }}>No depth points to display.</div>;
  }

  return (
    <div ref={wrapperRef} style={{ position: 'relative', width: '100%', height: '640px', background: '#f2efe4', borderRadius: 10, overflow: 'hidden', border: '1px solid #d4c9a8' }}>
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h, display: 'block', cursor: dragging ? 'grabbing' : 'crosshair' }}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp} onMouseLeave={onMouseLeave}
        onClick={onClick}
      />
      {/* Toolbar */}
      <div style={{ position: 'absolute', top: 8, right: 8, display: 'flex', gap: 4, background: 'rgba(255,255,255,0.94)', padding: 6, borderRadius: 8, border: '1px solid #cbd5e1', fontFamily: 'JetBrains Mono, monospace', fontSize: 10, flexWrap: 'wrap', maxWidth: 460 }}>
        <button onClick={() => setZoom(z => Math.min(40, z * 1.4))} style={btnStyle}>＋</button>
        <button onClick={() => setZoom(z => Math.max(0.5, z / 1.4))} style={btnStyle}>−</button>
        <button onClick={resetView} style={btnStyle}>reset</button>
        <span style={{ width: 1, background: '#cbd5e1', margin: '0 2px' }} />
        <label style={lblStyle}><input type="checkbox" checked={showRaster} onChange={e => setShowRaster(e.target.checked)} /> GeoTIFF</label>
        <label style={lblStyle}><input type="checkbox" checked={showContours} onChange={e => setShowContours(e.target.checked)} /> isobaths</label>
        <label style={lblStyle}><input type="checkbox" checked={showSoundings} onChange={e => setShowSoundings(e.target.checked)} /> soundings</label>
        <span style={{ width: 1, background: '#cbd5e1', margin: '0 2px' }} />
        <label style={lblStyle}>contour interval:
          <select value={contourInterval} onChange={e => setContourInterval(parseFloat(e.target.value))}
            style={{ marginLeft: 4, fontFamily: 'inherit', fontSize: 10 }}>
            {[1, 2, 5, 10, 20].map(v => <option key={v} value={v}>{v}m</option>)}
          </select>
        </label>
        <label style={lblStyle}>density:
          <select value={soundingStep} onChange={e => setSoundingStep(e.target.value)}
            style={{ marginLeft: 4, fontFamily: 'inherit', fontSize: 10 }}>
            <option value="auto">auto</option>
            <option value="1">all</option>
            <option value="2">½</option>
            <option value="5">⅕</option>
          </select>
        </label>
      </div>
      {/* Legend */}
      <div style={{ position: 'absolute', bottom: 8, left: 8, background: 'rgba(255,255,255,0.93)', padding: 8, borderRadius: 8, border: '1px solid #cbd5e1', fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
        <div style={{ fontWeight: 700, color: '#0f172a', marginBottom: 4, letterSpacing: '0.06em' }}>IHO S-52 DEPTH</div>
        {IHO_PALETTE.map(cls => (
          <div key={cls.label} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
            <div style={{ width: 14, height: 10, background: cls.tint, border: `1.5px solid ${cls.color}` }} />
            <span style={{ color: cls.color, fontWeight: 700 }}>{cls.label}</span>
          </div>
        ))}
      </div>
      {hoverReadout && (
        <div style={{ position: 'absolute', top: 8, left: 8, background: 'rgba(255,255,255,0.92)', padding: '4px 8px', borderRadius: 6, fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#0f172a', border: '1px solid #cbd5e1' }}>
          lat {hoverReadout.lat} │ lon {hoverReadout.lon}
        </div>
      )}
    </div>
  );
}

const btnStyle = { background: '#fff', border: '1px solid #cbd5e1', borderRadius: 5, padding: '3px 8px', fontFamily: 'JetBrains Mono, monospace', fontSize: 10, cursor: 'pointer', color: '#0f172a', fontWeight: 700 };
const lblStyle = { display: 'flex', alignItems: 'center', gap: 3, cursor: 'pointer', padding: '2px 4px', color: '#0f172a' };
