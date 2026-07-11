import React, { useEffect, useRef, useState, useMemo } from 'react';
import L from 'leaflet';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine, Legend,
} from 'recharts';
import { INTERNAL_ANALYSIS } from './internalMode';

const MONO = 'var(--font-mono)';
const GATE_PASS = '#059669';
const GATE_FAIL = '#e11d48';

// status -> visual treatment. abstain / not-viable are NOT hidden.
const STATUS_STYLE = {
  PASS:       { bg: 'rgba(5,150,105,0.12)',  fg: '#047857', label: 'ALL GATES PASS' },
  PARTIAL:    { bg: 'rgba(217,119,6,0.12)',  fg: '#b45309', label: 'HONEST PARTIAL' },
  ABSTAIN:    { bg: 'rgba(225,29,72,0.12)',  fg: '#be123c', label: 'HONEST ABSTAIN' },
  NOT_VIABLE: { bg: 'rgba(225,29,72,0.12)',  fg: '#be123c', label: 'NOT VIABLE' },
  BENCHMARK:  { bg: 'rgba(37,99,235,0.12)',  fg: '#1d4ed8', label: 'BENCHMARK (control)' },
};

// depth colour ramp (m) -> a blue->teal->amber scale, shallow=light
function depthColor(d) {
  const stops = [
    [0, [186, 230, 253]], [2, [56, 189, 248]], [4, [14, 165, 233]],
    [6, [13, 148, 136]], [10, [5, 150, 105]], [16, [217, 119, 6]], [25, [120, 53, 15]],
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (d >= stops[i][0] && d <= stops[i + 1][0]) { lo = stops[i]; hi = stops[i + 1]; break; }
  }
  const t = hi[0] === lo[0] ? 0 : (d - lo[0]) / (hi[0] - lo[0]);
  const c = lo[1].map((v, k) => Math.round(v + (hi[1][k] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function fmt(x, d = 2) { return (x === null || x === undefined || Number.isNaN(x)) ? '—' : Number(x).toFixed(d); }

function StatusChip({ status }) {
  const s = STATUS_STYLE[status] || STATUS_STYLE.ABSTAIN;
  return (
    <span style={{ padding: '3px 9px', borderRadius: '5px', background: s.bg, color: s.fg,
      fontFamily: MONO, fontSize: '9px', fontWeight: 800, letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>
      {s.label}
    </span>
  );
}

function GateChip({ pass, label }) {
  return (
    <div title={label} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '5px 8px',
      borderRadius: '6px', border: `1px solid ${pass ? 'rgba(5,150,105,0.3)' : 'rgba(225,29,72,0.35)'}`,
      background: pass ? 'rgba(5,150,105,0.07)' : 'rgba(225,29,72,0.09)' }}>
      <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: pass ? GATE_PASS : GATE_FAIL,
        boxShadow: pass ? 'none' : '0 0 6px rgba(225,29,72,0.6)' }} />
      <span style={{ fontFamily: MONO, fontSize: '9px', fontWeight: 700, color: pass ? '#065f46' : '#9f1239' }}>
        {pass ? 'PASS' : 'FAIL'}
      </span>
      <span style={{ fontFamily: MONO, fontSize: '9px', color: 'var(--text-secondary)' }}>{label}</span>
    </div>
  );
}

// ---- the per-site OOF map (raw leaflet, matching MapPanel's stack) ----
function SiteMap({ site, mode }) {
  const ref = useRef(null);
  const mapRef = useRef(null);
  const layerRef = useRef(null);

  useEffect(() => {
    if (!ref.current || mapRef.current) return;
    const m = L.map(ref.current, { center: site.center, zoom: 12, zoomControl: true, attributionControl: false });
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19 }).addTo(m);
    mapRef.current = m;
    return () => { m.remove(); mapRef.current = null; };
  }, [site.key]);

  useEffect(() => {
    const m = mapRef.current;
    if (!m) return;
    if (layerRef.current) { m.removeLayer(layerRef.current); layerRef.current = null; }
    const cells = (site.cells || []).filter(c => c.lat !== undefined && c.lon !== undefined);
    if (!cells.length) return;
    const g = L.layerGroup();
    const pts = [];
    cells.forEach(c => {
      const val = mode === 'pred' ? c.p : (mode === 'error' ? (c.p - c.t) : c.t);
      const color = mode === 'error'
        ? (val >= 0 ? `rgba(225,29,72,${Math.min(1, 0.25 + Math.abs(val) / 4)})`
                    : `rgba(37,99,235,${Math.min(1, 0.25 + Math.abs(val) / 4)})`)
        : depthColor(val);
      L.circleMarker([c.lat, c.lon], { radius: 3, color, fillColor: color, fillOpacity: 0.85, weight: 0 })
        .bindTooltip(`true ${fmt(c.t)} m · pred ${fmt(c.p)} m · err ${fmt(c.p - c.t)} m`, { sticky: true })
        .addTo(g);
      pts.push([c.lat, c.lon]);
    });
    g.addTo(m);
    layerRef.current = g;
    try { m.fitBounds(L.latLngBounds(pts).pad(0.15)); } catch (e) { /* noop */ }
  }, [site.key, mode]);

  const hasLL = (site.cells || []).some(c => c.lat !== undefined);
  if (!hasLL) {
    return (
      <div style={{ height: '320px', display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg-secondary)', borderRadius: '8px', border: '1px solid var(--border-dim)' }}>
        <p style={{ fontFamily: MONO, fontSize: '11px', color: 'var(--text-dim)', textAlign: 'center', padding: '20px' }}>
          No per-cell coordinates in this run's data<br />(see scatter &amp; band table).
        </p>
      </div>
    );
  }
  return <div ref={ref} style={{ height: '320px', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border-dim)' }} />;
}

// ---- pred-vs-true scatter with 1:1 line + density (alpha) ----
function PredTrueScatter({ site }) {
  const data = useMemo(() => (site.cells || []).map(c => ({ x: c.t, y: c.p })), [site.key]);
  const maxV = useMemo(() => {
    let mx = 1; data.forEach(d => { mx = Math.max(mx, d.x, d.y); }); return Math.ceil(mx);
  }, [data]);
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 10, right: 16, bottom: 28, left: 4 }}>
        <CartesianGrid stroke="var(--border-dim)" />
        <XAxis type="number" dataKey="x" name="True depth" domain={[0, maxV]} unit=" m"
          tick={{ fontSize: 9, fontFamily: MONO, fill: 'var(--text-dim)' }}
          label={{ value: 'True depth (m)', position: 'bottom', offset: 8, fontSize: 10, fontFamily: MONO, fill: 'var(--text-secondary)' }} />
        <YAxis type="number" dataKey="y" name="Predicted" domain={[0, maxV]} unit=" m"
          tick={{ fontSize: 9, fontFamily: MONO, fill: 'var(--text-dim)' }}
          label={{ value: 'Pred (m)', angle: -90, position: 'insideLeft', fontSize: 10, fontFamily: MONO, fill: 'var(--text-secondary)' }} />
        <ZAxis range={[6, 6]} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontFamily: MONO, fontSize: '10px' }}
          formatter={(v) => `${Number(v).toFixed(2)} m`} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: maxV, y: maxV }]} stroke="#0f172a" strokeDasharray="6 4" ifOverflow="extendDomain" />
        <Scatter data={data} fill="rgba(2,132,199,0.35)" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ---- per-2m-band RMSE + bias bars ----
function BandChart({ site }) {
  const data = (site.per_band || []).map(b => ({
    band: b.band, rmse: b.rmse_m, bias: b.bias_m, n: b.n,
    biasFail: b.n >= 50 && b.bias_m !== null && Math.abs(b.bias_m) >= 0.8,
  }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 10, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--border-dim)" vertical={false} />
        <XAxis dataKey="band" tick={{ fontSize: 9, fontFamily: MONO, fill: 'var(--text-dim)' }} />
        <YAxis tick={{ fontSize: 9, fontFamily: MONO, fill: 'var(--text-dim)' }} unit=" m" />
        <Tooltip contentStyle={{ fontFamily: MONO, fontSize: '10px' }}
          formatter={(v, n) => [`${Number(v).toFixed(2)} m`, n.toUpperCase()]} />
        <ReferenceLine y={0} stroke="#94a3b8" />
        <Legend wrapperStyle={{ fontFamily: MONO, fontSize: '9px' }} />
        <Bar dataKey="rmse" name="RMSE" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill="#0891b2" />)}
        </Bar>
        <Bar dataKey="bias" name="Bias" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.biasFail ? GATE_FAIL : '#d97706'} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function HeadlineRmse({ site }) {
  const p = site.pooled || {};
  const isFull = site.headline_metric === 'full';
  const val = isFull ? p.rmse_full : p.rmse_012;
  const range = isFull ? 'full-range (all cells 4–22 m)' : '0–12 m band';
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
      <span style={{ fontSize: '30px', fontWeight: 800, fontFamily: MONO, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
        {fmt(val)}<span style={{ fontSize: '13px', color: 'var(--text-dim)' }}> m</span>
      </span>
      <span style={{ fontFamily: MONO, fontSize: '10px', color: 'var(--text-secondary)' }}>
        pooled RMSE · {range}
      </span>
    </div>
  );
}

function Leaderboard({ index, activeKey, onPick }) {
  const rows = [...(index.sites || [])].sort((a, b) => (a.headline_rmse ?? 99) - (b.headline_rmse ?? 99));
  return (
    <div style={{ background: 'var(--bg-card)', borderRadius: '10px', border: '1px solid var(--border-dim)', overflow: 'hidden' }}>
      <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border-dim)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ fontFamily: MONO, fontSize: '11px', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '0.04em', margin: 0 }}>
          RMSE LEADERBOARD · SUB-1M CAMPAIGN
        </h3>
        <span style={{ fontFamily: MONO, fontSize: '8px', color: 'var(--text-dim)' }}>headline = pooled OOF</span>
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: MONO, fontSize: '10px' }}>
        <thead>
          <tr style={{ color: 'var(--text-dim)', textAlign: 'left' }}>
            {['SITE', 'RMSE', 'BAND', 'GATES', 'n', 'STATUS'].map(h => (
              <th key={h} style={{ padding: '6px 12px', fontWeight: 700, borderBottom: '1px solid var(--border-dim)' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(r => {
            const active = r.key === activeKey;
            const gatesOk = r.n_gates_pass === 7;
            return (
              <tr key={r.key} onClick={() => onPick(r.key)} style={{ cursor: 'pointer',
                background: active ? 'rgba(2,132,199,0.08)' : 'transparent', borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '8px 12px', fontWeight: 700, color: 'var(--text-primary)' }}>{r.name}</td>
                <td style={{ padding: '8px 12px', fontWeight: 800,
                  color: (r.headline_rmse ?? 9) < 0.95 ? GATE_PASS : 'var(--text-primary)' }}>{fmt(r.headline_rmse)} m</td>
                <td style={{ padding: '8px 12px', color: 'var(--text-dim)' }}>{r.headline_metric === 'full' ? 'full' : '0–12 m'}</td>
                <td style={{ padding: '8px 12px', fontWeight: 800, color: gatesOk ? GATE_PASS : GATE_FAIL }}>{r.n_gates_pass}/7</td>
                <td style={{ padding: '8px 12px', color: 'var(--text-secondary)' }}>{r.n_cells?.toLocaleString?.() ?? r.n_cells}</td>
                <td style={{ padding: '8px 12px' }}><StatusChip status={r.status} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function UaeResultsPanel() {
  // This explorer is an internal accuracy/OOF benchmark dashboard (RMSE
  // leaderboard, per-band error, gates). It is NOT a client-portal surface —
  // gate it behind INTERNAL_ANALYSIS. Client portal sees a neutral placeholder.
  if (!INTERNAL_ANALYSIS) {
    return (
      <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-secondary)' }}>
        <p style={{ fontFamily: MONO, fontSize: '12px', color: 'var(--text-dim)', textAlign: 'center', maxWidth: '420px' }}>
          Depth results are available in the Results catalogue (📊 RESULTS).
        </p>
      </div>
    );
  }
  return <UaeResultsExplorer />;
}

function UaeResultsExplorer() {
  const [index, setIndex] = useState(null);
  const [activeKey, setActiveKey] = useState(null);
  const [site, setSite] = useState(null);
  const [mapMode, setMapMode] = useState('true'); // true | pred | error
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch('/data/index.json').then(r => r.json()).then(d => {
      setIndex(d);
      if (d.sites && d.sites.length) {
        const best = [...d.sites].sort((a, b) => (a.headline_rmse ?? 99) - (b.headline_rmse ?? 99))[0];
        setActiveKey(best.key);
      }
    }).catch(e => setErr('Could not load /data/index.json — run CNN_iboating_v2/export_vis_data.py'));
  }, []);

  useEffect(() => {
    if (!activeKey) return;
    setSite(null);
    fetch(`/data/${activeKey}.json`).then(r => r.json()).then(setSite)
      .catch(() => setErr(`Could not load /data/${activeKey}.json`));
  }, [activeKey]);

  if (err) {
    return <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-secondary)' }}>
      <p style={{ fontFamily: MONO, fontSize: '12px', color: GATE_FAIL, textAlign: 'center', maxWidth: '420px' }}>{err}</p>
    </div>;
  }
  if (!index) {
    return <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-secondary)' }}>
      <p style={{ fontFamily: MONO, fontSize: '12px', color: 'var(--text-dim)' }}>Loading UAE results…</p>
    </div>;
  }

  const sStyle = site ? (STATUS_STYLE[site.status] || STATUS_STYLE.ABSTAIN) : null;

  return (
    <div style={{ height: '100%', overflowY: 'auto', background: 'var(--bg-base)', padding: '18px 20px' }}>
      <div style={{ marginBottom: '6px', display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ fontSize: '17px', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.01em', margin: 0 }}>
            UAE Results Explorer — SUB-1M Campaign
          </h2>
          <p style={{ fontFamily: MONO, fontSize: '9px', color: 'var(--text-dim)', margin: '3px 0 0' }}>
            Real OOF artifacts only · pooled cell-OOF is the headline (fold-mean shown separately) · generated {index.generated_at}
          </p>
        </div>
        <div style={{ fontFamily: MONO, fontSize: '9px', color: 'var(--text-secondary)', background: 'rgba(217,119,6,0.10)', padding: '5px 10px', borderRadius: '6px', border: '1px solid rgba(217,119,6,0.25)' }}>
          GATE: 0–12 m RMSE &lt; {index.gate_rmse_012} m · honest-abstain allowed
        </div>
      </div>

      <div style={{ margin: '14px 0' }}>
        <Leaderboard index={index} activeKey={activeKey} onPick={setActiveKey} />
      </div>

      {!site && <p style={{ fontFamily: MONO, fontSize: '11px', color: 'var(--text-dim)' }}>Loading site…</p>}

      {site && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {/* site header card */}
          <div style={{ background: sStyle.bg, borderRadius: '10px', border: `1px solid ${sStyle.fg}33`, padding: '14px 16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '10px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>{site.name}</h3>
                  <StatusChip status={site.status} />
                </div>
                <p style={{ fontFamily: MONO, fontSize: '9px', color: 'var(--text-secondary)', margin: '4px 0 8px' }}>
                  {site.region} · GT: {site.gt_source} · {site.n_dates ? `${site.n_dates} date(s)` : ''} · {site.n_cells?.toLocaleString?.()} cells · {site.n_folds || '—'} folds
                </p>
                <HeadlineRmse site={site} />
                <p style={{ fontFamily: MONO, fontSize: '10px', color: sStyle.fg, fontWeight: 700, margin: '8px 0 0' }}>{site.status_note}</p>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '22px', fontWeight: 800, fontFamily: MONO, color: site.n_gates_pass === 7 ? GATE_PASS : GATE_FAIL }}>
                  {site.n_gates_pass}/7
                </div>
                <div style={{ fontFamily: MONO, fontSize: '8px', color: 'var(--text-dim)' }}>GATES PASS</div>
                {site.r4_benchmark_rmse && (
                  <div style={{ marginTop: '6px', fontFamily: MONO, fontSize: '8px', color: 'var(--text-secondary)' }}>
                    R4 closed benchmark: <b>{fmt(site.r4_benchmark_rmse)} m</b> (fold-mean)
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* honesty: pooled vs fold-mean distinction */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: '10px' }}>
            <Stat label="Pooled full-range RMSE" value={`${fmt(site.pooled?.rmse_full)} m`} />
            <Stat label="Pooled 0–12 m RMSE" value={`${fmt(site.pooled?.rmse_012)} m`}
              hi={site.pooled?.rmse_012 != null && site.pooled.rmse_012 < 0.95} />
            <Stat label="Full-range R²" value={fmt(site.pooled?.r2_full)} />
            <Stat label="Decile slope" value={fmt(site.pooled?.slope)} />
            <Stat label="Pred/GT std-ratio" value={fmt(site.pooled?.std_ratio)} />
            {site.fold_mean?.band_0_12m && (
              <Stat label="Fold-mean 0–12 m (context)" sub
                value={`${fmt(site.fold_mean.band_0_12m.rmse_mean)} ± ${fmt(site.fold_mean.band_0_12m.rmse_sd)} m`} />
            )}
          </div>

          {/* 7-gate table */}
          <div style={{ background: 'var(--bg-card)', borderRadius: '10px', border: '1px solid var(--border-dim)', padding: '14px 16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h4 style={{ fontFamily: MONO, fontSize: '11px', fontWeight: 800, color: 'var(--text-primary)', margin: 0, letterSpacing: '0.04em' }}>7-GATE VERDICT</h4>
              <span style={{ fontFamily: MONO, fontSize: '8px', color: 'var(--text-dim)' }}>source: {site.gates_source}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: '8px' }}>
              {(site.gate_labels || []).map(g => (
                <GateChip key={g.key} pass={!!site.gates?.[g.key]} label={g.label} />
              ))}
            </div>
          </div>

          {/* charts grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(360px,1fr))', gap: '14px' }}>
            <Card title="OOF DEPTH MAP">
              <div style={{ display: 'flex', gap: '4px', marginBottom: '8px' }}>
                {[['true', 'TRUE'], ['pred', 'PRED'], ['error', 'ERROR']].map(([m, lab]) => (
                  <button key={m} onClick={() => setMapMode(m)} style={{
                    padding: '4px 12px', borderRadius: '6px', border: '1px solid var(--border-dim)', cursor: 'pointer',
                    fontFamily: MONO, fontSize: '9px', fontWeight: 700,
                    background: mapMode === m ? 'linear-gradient(135deg,#0891b2,#2563eb)' : 'var(--bg-primary)',
                    color: mapMode === m ? '#fff' : 'var(--text-dim)',
                  }}>{lab}</button>
                ))}
              </div>
              <SiteMap site={site} mode={mapMode} />
            </Card>

            <Card title="PRED vs TRUE (1:1 line)">
              <PredTrueScatter site={site} />
              <p style={{ fontFamily: MONO, fontSize: '8px', color: 'var(--text-dim)', margin: '6px 0 0' }}>
                Points on the dashed 1:1 line = perfect. Vertical compression toward the mean = regression-to-mean failure.
              </p>
            </Card>

            <Card title="PER-2 m-BAND RMSE & BIAS">
              <BandChart site={site} />
              <p style={{ fontFamily: MONO, fontSize: '8px', color: 'var(--text-dim)', margin: '6px 0 0' }}>
                Red bias bars = |bias| ≥ 0.8 m with n ≥ 50 (band-bias gate FAIL). Source: {site.per_band_source}.
              </p>
            </Card>

            <Card title="PER-2 m-BAND TABLE">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: MONO, fontSize: '10px' }}>
                <thead><tr style={{ color: 'var(--text-dim)', textAlign: 'left' }}>
                  {['BAND', 'n', 'RMSE', 'BIAS'].map(h => <th key={h} style={{ padding: '5px 8px', borderBottom: '1px solid var(--border-dim)' }}>{h}</th>)}
                </tr></thead>
                <tbody>
                  {(site.per_band || []).map((b, i) => {
                    const fail = b.n >= 50 && b.bias_m !== null && Math.abs(b.bias_m) >= 0.8;
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)', opacity: b.n < 50 ? 0.55 : 1 }}>
                        <td style={{ padding: '6px 8px', fontWeight: 700, color: 'var(--text-primary)' }}>{b.band}</td>
                        <td style={{ padding: '6px 8px', color: 'var(--text-secondary)' }}>{b.n}{b.n < 50 ? ' *' : ''}</td>
                        <td style={{ padding: '6px 8px', color: 'var(--text-primary)' }}>{fmt(b.rmse_m)}</td>
                        <td style={{ padding: '6px 8px', fontWeight: 700, color: fail ? GATE_FAIL : 'var(--text-secondary)' }}>{b.bias_m > 0 ? '+' : ''}{fmt(b.bias_m)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p style={{ fontFamily: MONO, fontSize: '8px', color: 'var(--text-dim)', margin: '6px 0 0' }}>* n &lt; 50 → band-bias gate not evaluated (low sample).</p>
            </Card>
          </div>

          <p style={{ fontFamily: MONO, fontSize: '8px', color: 'var(--text-dim)', margin: '4px 0 24px' }}>
            Artifact: {site.artifact}{site.results_json ? ` · ${site.results_json}` : ''} · OOF cells emitted: {site.n_cells_emitted?.toLocaleString?.()} / {site.n_cells?.toLocaleString?.()}
          </p>
        </div>
      )}
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div style={{ background: 'var(--bg-card)', borderRadius: '10px', border: '1px solid var(--border-dim)', padding: '14px 16px' }}>
      <h4 style={{ fontFamily: MONO, fontSize: '11px', fontWeight: 800, color: 'var(--text-primary)', margin: '0 0 10px', letterSpacing: '0.04em' }}>{title}</h4>
      {children}
    </div>
  );
}

function Stat({ label, value, hi, sub }) {
  return (
    <div style={{ background: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--border-dim)', padding: '10px 12px' }}>
      <div style={{ fontFamily: MONO, fontSize: '8px', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</div>
      <div style={{ fontFamily: MONO, fontSize: sub ? '13px' : '16px', fontWeight: 800, marginTop: '3px',
        color: hi ? GATE_PASS : 'var(--text-primary)' }}>{value}</div>
    </div>
  );
}
