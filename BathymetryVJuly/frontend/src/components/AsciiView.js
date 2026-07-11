import React, { useMemo, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// AsciiView — colored monospace grid showing Modeled vs Observed depths.
//   • Two side-by-side grids (MOD / OBS); optional DIFF grid (mod − obs).
//   • Each cell is monospace text on a depth-colored background.
//   • Zoom slider increases grid density so finer detail and more numbers
//     show up as you zoom in (matches the user's "zoom → more numbers" ask).
// ─────────────────────────────────────────────────────────────────────────────

const MOD_CLASSES = new Set(['interpolated', 'bathymetry']);
const OBS_CLASSES = new Set(['observed', 'insitu', 'icesat2_cshelph', 'chart', 'gebco']);

function depthColor(d, maxD) {
  if (!Number.isFinite(d) || d <= 0) return null;
  const t = Math.min(d / (maxD || 25), 1);
  if (t < 0.1) return '#caf0f8';
  if (t < 0.2) return '#90e0ef';
  if (t < 0.35) return '#48cae4';
  if (t < 0.5) return '#00b4d8';
  if (t < 0.65) return '#0096c7';
  if (t < 0.8) return '#0077b6';
  return '#023e8a';
}

function diffColor(diff) {
  if (!Number.isFinite(diff)) return null;
  const t = Math.max(-1, Math.min(1, diff / 5));  // ±5 m saturates
  if (t < -0.6) return '#1d4ed8';   // model too shallow (deep blue)
  if (t < -0.2) return '#60a5fa';   // model slightly shallow
  if (t < 0.2)  return '#a7f3d0';   // ≈ match (green)
  if (t < 0.6)  return '#fbbf24';   // model slightly deep
  return '#dc2626';                  // model too deep (red)
}

function textColor(bg) {
  if (!bg) return '#475569';
  // dark blues need light text
  if (['#0077b6', '#023e8a', '#0096c7', '#1d4ed8', '#dc2626'].includes(bg)) return '#f8fafc';
  return '#0f172a';
}

function fmt(d) {
  if (!Number.isFinite(d)) return '·';
  if (Math.abs(d) >= 100) return Math.round(d).toString();
  if (Math.abs(d) >= 10)  return d.toFixed(1);
  return d.toFixed(2);
}

function getBbox(results) {
  const b = results?.bbox;
  if (b && typeof b === 'object' && 'west' in b) {
    return { w: b.west, s: b.south, e: b.east, n: b.north };
  }
  if (Array.isArray(b) && b.length === 4) {
    return { w: b[0], s: b[1], e: b[2], n: b[3] };
  }
  // fallback: derive from points
  let w = +Infinity, s = +Infinity, e = -Infinity, n = -Infinity;
  for (const p of (results?.points || [])) {
    if (Number.isFinite(p.lat) && Number.isFinite(p.lon)) {
      if (p.lon < w) w = p.lon; if (p.lon > e) e = p.lon;
      if (p.lat < s) s = p.lat; if (p.lat > n) n = p.lat;
    }
  }
  if (!Number.isFinite(w)) return null;
  return { w, s, e, n };
}

function bin(points, classSet, bbox, nx, ny) {
  const sum = new Float64Array(nx * ny);
  const cnt = new Uint32Array(nx * ny);
  const { w, s, e, n } = bbox;
  const dx = (e - w) / nx;
  const dy = (n - s) / ny;
  if (!(dx > 0) || !(dy > 0)) return { sum, cnt };
  for (const p of points) {
    if (!classSet.has(p.photon_class)) continue;
    if (!Number.isFinite(p.depth) || p.depth <= 0) continue;
    const ix = Math.floor((p.lon - w) / dx);
    const iy = Math.floor((p.lat - s) / dy);
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) continue;
    const k = iy * nx + ix;
    sum[k] += p.depth;
    cnt[k] += 1;
  }
  return { sum, cnt };
}

export default function AsciiView({ results }) {
  const [density, setDensity] = useState(28);  // cells per side
  const [showDiff, setShowDiff] = useState(true);

  const bbox = useMemo(() => getBbox(results), [results]);
  const pts = results?.points || [];
  const maxD = results?.stats?.max_depth || 25;

  const grids = useMemo(() => {
    if (!bbox || pts.length === 0) return null;
    // Lat-aspect-correct cell count so cells look closer to square
    const latSpan = bbox.n - bbox.s;
    const lonSpan = bbox.e - bbox.w;
    const latMid = (bbox.s + bbox.n) / 2;
    const meanPerDeg = Math.max(0.01, Math.cos(latMid * Math.PI / 180));
    const aspect = (lonSpan * meanPerDeg) / Math.max(0.0001, latSpan);
    const nx = Math.max(4, Math.round(density * Math.sqrt(aspect)));
    const ny = Math.max(4, Math.round(density / Math.sqrt(aspect)));
    const mod = bin(pts, MOD_CLASSES, bbox, nx, ny);
    const obs = bin(pts, OBS_CLASSES, bbox, nx, ny);
    return { nx, ny, mod, obs };
  }, [bbox, pts, density]);

  if (!results || !grids) {
    return (
      <div style={{
        height:'100%', display:'flex', alignItems:'center', justifyContent:'center',
        background:'var(--bg-secondary)', padding:'24px',
      }}>
        <p style={{fontFamily:'var(--font-mono)',fontSize:'12px',color:'var(--text-dim)',textAlign:'center'}}>
          Run a depth extraction first — the ASCII view renders modeled vs observed depths once results arrive.
        </p>
      </div>
    );
  }

  const { nx, ny, mod, obs } = grids;
  const modCells = mod.cnt.reduce((a, c) => a + (c > 0 ? 1 : 0), 0);
  const obsCells = obs.cnt.reduce((a, c) => a + (c > 0 ? 1 : 0), 0);

  // Stats vs obs (paired cells only)
  let nPaired = 0, sumDiff = 0, sumAbs = 0, sumSq = 0;
  for (let k = 0; k < mod.cnt.length; k++) {
    if (mod.cnt[k] > 0 && obs.cnt[k] > 0) {
      const m = mod.sum[k] / mod.cnt[k];
      const o = obs.sum[k] / obs.cnt[k];
      const d = m - o;
      nPaired++; sumDiff += d; sumAbs += Math.abs(d); sumSq += d * d;
    }
  }
  const bias = nPaired ? sumDiff / nPaired : null;
  const mae = nPaired ? sumAbs / nPaired : null;
  const rmse = nPaired ? Math.sqrt(sumSq / nPaired) : null;

  return (
    <div style={{
      height:'100%', display:'flex', flexDirection:'column', overflow:'hidden',
      background:'var(--bg-secondary)',
    }}>
      {/* Toolbar */}
      <div style={{
        padding:'10px 16px', display:'flex', alignItems:'center', gap:'16px', flexWrap:'wrap',
        background:'var(--bg-primary)', borderBottom:'1px solid var(--border-dim)',
      }}>
        <span style={{fontFamily:'var(--font-mono)',fontSize:'11px',fontWeight:800,letterSpacing:'0.08em',color:'#0f172a'}}>
          ASCII · MOD vs OBS
        </span>
        <label style={{display:'flex',alignItems:'center',gap:'8px',fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text-dim)'}}>
          ZOOM
          <input type="range" min="8" max="80" step="2" value={density}
            onChange={e => setDensity(parseInt(e.target.value, 10))}
            style={{width:'160px'}} />
          <b style={{color:'#0f172a',minWidth:'40px'}}>{nx}×{ny}</b>
        </label>
        <label style={{display:'flex',alignItems:'center',gap:'5px',fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text-dim)'}}>
          <input type="checkbox" checked={showDiff} onChange={e=>setShowDiff(e.target.checked)} />
          DIFF
        </label>
        <span style={{flex:1}} />
        <span style={{fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text-dim)'}}>
          mod cells <b style={{color:'#0f172a'}}>{modCells}</b> ·
          obs cells <b style={{color:'#0f172a'}}>{obsCells}</b> ·
          paired <b style={{color:'#0f172a'}}>{nPaired}</b>
          {rmse !== null && (
            <> · RMSE <b style={{color:'#0f172a'}}>{rmse.toFixed(2)}m</b>
               · MAE <b style={{color:'#0f172a'}}>{mae.toFixed(2)}m</b>
               · bias <b style={{color:'#0f172a'}}>{bias.toFixed(2)}m</b></>
          )}
        </span>
      </div>

      {/* Grid panels */}
      <div style={{
        flex:1, overflow:'auto', padding:'16px',
        display:'grid',
        gridTemplateColumns: showDiff ? 'repeat(3, minmax(0,1fr))' : 'repeat(2, minmax(0,1fr))',
        gap:'16px',
      }}>
        <GridPanel title="MODELED" tone="#0369a1" nx={nx} ny={ny} grid={mod} maxD={maxD} />
        <GridPanel title="OBSERVED" tone="#92400e" nx={nx} ny={ny} grid={obs} maxD={maxD} />
        {showDiff && <DiffPanel nx={nx} ny={ny} mod={mod} obs={obs} />}
      </div>
    </div>
  );
}

function GridPanel({ title, tone, nx, ny, grid, maxD }) {
  // Render rows top-to-bottom = north-to-south for chart-like orientation.
  const rows = [];
  for (let iy = ny - 1; iy >= 0; iy--) {
    const cells = [];
    for (let ix = 0; ix < nx; ix++) {
      const k = iy * nx + ix;
      const c = grid.cnt[k];
      if (c > 0) {
        const d = grid.sum[k] / c;
        const bg = depthColor(d, maxD);
        cells.push(
          <td key={ix} title={`d=${d.toFixed(2)}m · n=${c}`}
            style={{background:bg,color:textColor(bg),padding:'1px 3px',textAlign:'center',fontSize:'10px',lineHeight:1}}>
            {fmt(d)}
          </td>
        );
      } else {
        cells.push(<td key={ix} style={{background:'rgba(15,23,42,0.03)',color:'#cbd5e1',padding:'1px 3px',textAlign:'center',fontSize:'10px',lineHeight:1}}>·</td>);
      }
    }
    rows.push(<tr key={iy}>{cells}</tr>);
  }
  return (
    <div style={{display:'flex',flexDirection:'column',minWidth:0,minHeight:0}}>
      <h3 style={{margin:'0 0 8px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:800,letterSpacing:'0.12em',color:tone}}>
        {title}
      </h3>
      <div style={{
        overflow:'auto',background:'var(--bg-primary)',border:'1px solid var(--border-dim)',
        borderRadius:'8px',padding:'4px',
      }}>
        <table style={{borderCollapse:'collapse',fontFamily:'var(--font-mono)',whiteSpace:'nowrap'}}>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>
  );
}

function DiffPanel({ nx, ny, mod, obs }) {
  const rows = [];
  for (let iy = ny - 1; iy >= 0; iy--) {
    const cells = [];
    for (let ix = 0; ix < nx; ix++) {
      const k = iy * nx + ix;
      if (mod.cnt[k] > 0 && obs.cnt[k] > 0) {
        const m = mod.sum[k] / mod.cnt[k];
        const o = obs.sum[k] / obs.cnt[k];
        const diff = m - o;
        const bg = diffColor(diff);
        const sign = diff >= 0 ? '+' : '';
        cells.push(
          <td key={ix} title={`mod=${m.toFixed(2)}  obs=${o.toFixed(2)}  Δ=${sign}${diff.toFixed(2)}m`}
            style={{background:bg,color:textColor(bg),padding:'1px 3px',textAlign:'center',fontSize:'10px',lineHeight:1}}>
            {sign}{Math.abs(diff) < 10 ? diff.toFixed(2) : diff.toFixed(1)}
          </td>
        );
      } else if (mod.cnt[k] > 0) {
        cells.push(<td key={ix} title="modeled only" style={{background:'rgba(2,132,199,0.10)',color:'#94a3b8',padding:'1px 3px',textAlign:'center',fontSize:'10px',lineHeight:1}}>·</td>);
      } else if (obs.cnt[k] > 0) {
        cells.push(<td key={ix} title="observed only" style={{background:'rgba(146,64,14,0.10)',color:'#94a3b8',padding:'1px 3px',textAlign:'center',fontSize:'10px',lineHeight:1}}>·</td>);
      } else {
        cells.push(<td key={ix} style={{background:'rgba(15,23,42,0.03)',color:'#cbd5e1',padding:'1px 3px',textAlign:'center',fontSize:'10px',lineHeight:1}}>·</td>);
      }
    }
    rows.push(<tr key={iy}>{cells}</tr>);
  }
  return (
    <div style={{display:'flex',flexDirection:'column',minWidth:0,minHeight:0}}>
      <h3 style={{margin:'0 0 8px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:800,letterSpacing:'0.12em',color:'#7c3aed'}}>
        DIFF (mod − obs)
      </h3>
      <div style={{
        overflow:'auto',background:'var(--bg-primary)',border:'1px solid var(--border-dim)',
        borderRadius:'8px',padding:'4px',
      }}>
        <table style={{borderCollapse:'collapse',fontFamily:'var(--font-mono)',whiteSpace:'nowrap'}}>
          <tbody>{rows}</tbody>
        </table>
      </div>
      <div style={{marginTop:'6px',display:'flex',gap:'10px',fontFamily:'var(--font-mono)',fontSize:'9px',color:'var(--text-dim)',flexWrap:'wrap'}}>
        <Legend bg="#1d4ed8" label="mod ≫ shallower (-)" />
        <Legend bg="#60a5fa" label="mod < obs" />
        <Legend bg="#a7f3d0" label="≈ match" />
        <Legend bg="#fbbf24" label="mod > obs" />
        <Legend bg="#dc2626" label="mod ≫ deeper (+)" />
      </div>
    </div>
  );
}

function Legend({ bg, label }) {
  return (
    <span style={{display:'inline-flex',alignItems:'center',gap:'4px'}}>
      <span style={{width:'10px',height:'10px',background:bg,borderRadius:'2px',display:'inline-block'}} />
      {label}
    </span>
  );
}
