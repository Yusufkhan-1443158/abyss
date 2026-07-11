import React, { useState, useMemo } from 'react';
import IHOChartView from './IHOChartView';
import DepthColorbar from './DepthColorbar';
import TideChip from './TideChip';
import MleSceneGroup from './MleSceneGroup';
import { INTERNAL_ANALYSIS, SHOW_IHO_SURFACE } from './internalMode';

// ── Advanced Tab: deep stats, calibration, sources, metadata, histograms ──
function AdvancedTab({ results, validationResults }) {
  const s = results?.stats || {};
  const ml = results?.ml_stats || {};
  const md = results?.image_metadata || {};
  const metrics = results?.metrics || {};           // CNN_v2 / Accurate runner metrics
  const tiled = results?.stats_tiled || null;
  const calib = (results?.calibration)              // CNN_v2 / Accurate calibration
    || (tiled?.calibration)                          // Quick-tiled calibration (legacy)
    || null;
  const tidal = results?.tidal_alignment || null;
  const bc    = results?.bias_correction_settings || null;
  const sources = results?.sources_used || [];
  const points = results?.points || [];
  const depthVals = useMemo(() =>
    (points || []).filter(p => p && (p.photon_class === 'interpolated' || p.photon_class === 'bathymetry') && p.depth > 0).map(p => p.depth),
  [points]);
  // Depth histogram bins
  const bins = useMemo(() => {
    if (depthVals.length === 0) return [];
    const maxD = Math.max(...depthVals);
    const B = 25;
    const step = Math.max(0.5, maxD / B);
    const arr = new Array(B).fill(0).map((_, i) => ({ x: i * step + step / 2, count: 0, lo: i * step, hi: (i + 1) * step }));
    for (const v of depthVals) {
      const i = Math.min(B - 1, Math.floor(v / step));
      arr[i].count++;
    }
    return arr;
  }, [depthVals]);
  const maxCount = bins.reduce((a, b) => Math.max(a, b.count), 1);

  // Percentiles
  const pct = useMemo(() => {
    if (depthVals.length === 0) return null;
    const v = depthVals.slice().sort((a, b) => a - b);
    const pick = (f) => v[Math.max(0, Math.min(v.length - 1, Math.floor(f * v.length)))];
    return { p5: pick(0.05), p25: pick(0.25), p50: pick(0.5), p75: pick(0.75), p95: pick(0.95) };
  }, [depthVals]);

  return (
    <div>
      {/* Top summary grid */}
      <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.08em', marginBottom: 10 }}>
        ADVANCED ANALYTICS · PIPELINE STATE · CALIBRATION
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8, marginBottom: 16 }}>
        <BigStat label="Water pixels" value={(s.grid_points || metrics.n_water_px || depthVals.length || 0).toLocaleString()} color="#0284c7" />
        <BigStat label="Mean" value={(s.mean_depth ?? metrics.depth_mean_m)?.toFixed(2)} unit="m" color="#0891b2" />
        <BigStat label="Max"  value={(s.max_depth  ?? metrics.depth_max_m)?.toFixed(2)}  unit="m" color="#1d4ed8" />
        <BigStat label="Min"  value={(s.min_depth  ?? metrics.depth_min_m)?.toFixed(2)}  unit="m" color="#6366f1" />
        <BigStat label="RMSE" value={metrics.rmse_m?.toFixed(2)} unit="m" color="#e11d48" />
        <BigStat label="R²"    value={metrics.r2?.toFixed(3)} color="#7c3aed" />
        <BigStat label="Resolution" value={s.resolution_m ?? md.resolution_m ?? bc?.window_days ? bc.window_days : '—'} unit={s.resolution_m || md.resolution_m ? 'm' : ''} color="#0f766e" />
        <BigStat label="n train" value={metrics.n_train ?? ml.n_train ?? '—'} color="#be185d" />
      </div>

      {/* Bias-correction settings summary (from /api/cnn-v2/start) */}
      {bc && (
        <div style={{ marginBottom: 16, padding: 10, borderRadius: 10, border: '1px solid rgba(14,165,233,0.25)', background: 'rgba(14,165,233,0.03)' }}>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#0369a1', fontWeight: 800, letterSpacing: '0.08em', marginBottom: 6 }}>
            ⚙ BIAS-CORRECTION SETTINGS (applied on this run)
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 6, fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
            <MetaKV k="mode"        v={bc.mode} />
            <MetaKV k="window"      v={`${bc.window_days} d`} />
            <MetaKV k="n_scenes"    v={bc.mode === 'multi' ? bc.n_scenes : '—'} />
            <MetaKV k="tide"        v={bc.tide ? 'on' : 'off'} />
            <MetaKV k="wave"        v={bc.wave ? 'on' : 'off'} />
            <MetaKV k="geoid"       v={bc.geoid ? 'on' : 'off'} />
            <MetaKV k="post-calib"  v={bc.calib} />
            {bc.scenes_dates && <MetaKV k="scenes" v={bc.scenes_dates.join(', ')} />}
          </div>
        </div>
      )}

      {/* Tidal_Alignment_Module summary */}
      {tidal && (
        <div style={{ marginBottom: 16, padding: 10, borderRadius: 10, border: '1px solid rgba(2,132,199,0.22)', background: 'rgba(2,132,199,0.03)' }}>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#075985', fontWeight: 800, letterSpacing: '0.08em', marginBottom: 6 }}>
            🌊 TIDAL_ALIGNMENT (at-source retide)
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 6, fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
            <MetaKV k="tide backend"  v={tidal.tide_backend} />
            <MetaKV k="wave backend"  v={tidal.wave_backend} />
            <MetaKV k="geoid backend" v={tidal.geoid_backend} />
            <MetaKV k="n points"      v={tidal.n_points} />
            <MetaKV k="t2"            v={tidal.t2_iso?.slice(0, 16)} />
            <MetaKV k="T2 at centre"  v={`${tidal.roi_centre?.T2_m?.toFixed?.(2) ?? '—'} m`} />
            <MetaKV k="tide Δ mean"   v={`${tidal.delta_mean_m ?? '—'} m`} />
            <MetaKV k="tide Δ range"  v={`[${tidal.delta_min_m ?? '—'}, ${tidal.delta_max_m ?? '—'}] m`} />
            <MetaKV k="wave Hs(t2)"   v={`${tidal.wave_Hs_t2_m ?? '—'} m`} />
            <MetaKV k="wave corr"     v={`${tidal.wave_correction_m ?? 0} m`} />
          </div>
        </div>
      )}

      {/* Post-prediction calibration pre/post */}
      {calib && calib.pre_calibration && calib.post_calibration && (
        <div style={{ marginBottom: 16, padding: 10, borderRadius: 10, border: '1px solid rgba(5,150,105,0.3)', background: 'rgba(5,150,105,0.03)' }}>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#065f46', fontWeight: 800, letterSpacing: '0.08em', marginBottom: 6 }}>
            🎯 POST-PREDICTION CALIBRATION · {(calib.steps_applied || []).join(' + ')}
          </p>
          <table style={{ width: '100%', fontFamily: 'JetBrains Mono, monospace', fontSize: 10, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-dim)', textAlign: 'left' }}>
                <th style={{ padding: '4px 6px' }}>Stage</th>
                <th style={{ padding: '4px 6px', textAlign: 'right' }}>RMSE (m)</th>
                <th style={{ padding: '4px 6px', textAlign: 'right' }}>MAE (m)</th>
                <th style={{ padding: '4px 6px', textAlign: 'right' }}>bias (m)</th>
                <th style={{ padding: '4px 6px', textAlign: 'right' }}>R²</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['pre',       calib.pre_calibration],
                ...(calib.shift ? [[`shift (${calib.shift.dr >= 0 ? '+' : ''}${calib.shift.dr}, ${calib.shift.dc >= 0 ? '+' : ''}${calib.shift.dc}) px`, calib.shift.after_shift]] : []),
                ...(calib.bias ? [[`bias (${calib.bias.mean_shift_m?.toFixed(2)} m)`, calib.bias.after_bias]] : []),
                ...(calib.local ? [[`local IDW [${calib.local.field_min_m?.toFixed(2)}..${calib.local.field_max_m?.toFixed(2)}] m`, calib.local.after_local]] : []),
                ['post', calib.post_calibration],
              ].map(([label, s], i) => (
                <tr key={i} style={{ borderTop: '1px solid rgba(5,150,105,0.12)', color: label === 'post' ? '#065f46' : 'var(--text-primary)', fontWeight: label === 'post' ? 700 : 400 }}>
                  <td style={{ padding: '4px 6px' }}>{label}</td>
                  <td style={{ padding: '4px 6px', textAlign: 'right' }}>{s?.rmse_m ?? '—'}</td>
                  <td style={{ padding: '4px 6px', textAlign: 'right' }}>{s?.mae_m ?? '—'}</td>
                  <td style={{ padding: '4px 6px', textAlign: 'right' }}>{s?.bias_m ?? '—'}</td>
                  <td style={{ padding: '4px 6px', textAlign: 'right' }}>{s?.r2 ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {calib.improvement && (
            <p style={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace', color: '#065f46', marginTop: 6 }}>
              Improvement: RMSE {calib.improvement.rmse_m >= 0 ? '−' : '+'}{Math.abs(calib.improvement.rmse_m)} m ·
              MAE {calib.improvement.mae_m >= 0 ? '−' : '+'}{Math.abs(calib.improvement.mae_m)} m ·
              |bias| {calib.improvement['|bias|_m'] >= 0 ? '−' : '+'}{Math.abs(calib.improvement['|bias|_m'])} m
            </p>
          )}
        </div>
      )}

      {/* Stratified (Chen 2026 TGRS-inspired) per-stratum breakdown */}
      {results?.stratified?.strata && results.stratified.strata.length > 0 && (
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 10, border: '1.5px solid rgba(14,165,233,0.35)', background: 'rgba(14,165,233,0.04)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#0369a1', fontWeight: 800, letterSpacing: '0.08em' }}>
              STRATIFIED BATHYMETRY (Chen 2026 TGRS) — per depth stratum
            </p>
            <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
              overall RMSE {results.stratified.overall?.rmse_m} m · MAE {results.stratified.overall?.mae_m} m · R² {results.stratified.overall?.r2}
            </span>
          </div>
          <div style={{ borderRadius: 8, border: '1px solid rgba(14,165,233,0.2)', overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}>
              <thead>
                <tr style={{ background: 'rgba(14,165,233,0.08)', borderBottom: '1px solid rgba(14,165,233,0.25)' }}>
                  {['Stratum', 'Depth (m)', 'n_train', 'n_val', 'RMSE', 'MAE', 'Bias', 'R²', 'Model'].map(h => (
                    <th key={h} style={{ padding: '5px 8px', textAlign: 'left', color: '#0369a1', fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.stratified.strata.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid rgba(14,165,233,0.12)' }}>
                    <td style={{ padding: '4px 8px', fontWeight: 700, color: '#0f172a' }}>{s.name}</td>
                    <td style={{ padding: '4px 8px' }}>{s.lo}–{s.hi}</td>
                    <td style={{ padding: '4px 8px' }}>{s.n_train ?? s.n ?? '—'}</td>
                    <td style={{ padding: '4px 8px' }}>{s.n_val ?? '—'}</td>
                    <td style={{ padding: '4px 8px', color: '#e11d48', fontWeight: 600 }}>{s.rmse ?? '—'}</td>
                    <td style={{ padding: '4px 8px', color: '#d97706' }}>{s.mae ?? '—'}</td>
                    <td style={{ padding: '4px 8px', color: (s.bias ?? 0) >= 0 ? '#7c3aed' : '#0284c7' }}>{s.bias !== undefined ? (s.bias >= 0 ? '+' : '') + s.bias : '—'}</td>
                    <td style={{ padding: '4px 8px' }}>{s.r2 ?? '—'}</td>
                    <td style={{ padding: '4px 8px', fontSize: 9, color: 'var(--text-dim)' }}>{s.model}{s.bootstrap ? ' + boot' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginTop: 8, lineHeight: 1.5 }}>
            Depth-balanced bootstrap activates automatically for strata with &lt; 200 training points (GAN substitute from the paper). Soft-blended mosaic across stratum boundaries prevents seams.
          </p>
        </div>
      )}

      {/* Observation-based calibration panel */}
      {calib && (
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 10, border: '1.5px solid rgba(5,150,105,0.3)', background: 'rgba(5,150,105,0.03)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#065f46', fontWeight: 800, letterSpacing: '0.08em' }}>
              OBSERVATION-BASED BIAS CALIBRATION ✓
            </p>
            <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>{calib.method}</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8 }}>
            <CalibCell label="n observations" value={calib.n_obs_used} tone="#0f172a" />
            <CalibCell label="bias before" value={`${(calib.pre_bias_m >= 0 ? '+' : '')}${calib.pre_bias_m?.toFixed(2)} m`} tone="#d97706" />
            <CalibCell label="bias after" value={`${(calib.post_bias_m >= 0 ? '+' : '')}${calib.post_bias_m?.toFixed(2)} m`} tone={Math.abs(calib.post_bias_m) < Math.abs(calib.pre_bias_m) ? '#059669' : '#dc2626'} />
            <CalibCell label="RMSE before → after" value={`${calib.pre_rmse_m?.toFixed(2)} → ${calib.post_rmse_m?.toFixed(2)} m`} tone="#0284c7" />
          </div>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginTop: 8, lineHeight: 1.5 }}>
            Observed points were used (a) at weight {calib.observation_weight}× versus GEBCO in each tile's ridge regression, and (b) to estimate a residual (obs − pred) field, interpolated via IDW and added back to the mosaic.
          </p>
        </div>
      )}

      {/* Depth distribution histogram */}
      {bins.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, marginBottom: 6 }}>
            DEPTH DISTRIBUTION ({depthVals.length.toLocaleString()} pixels)
          </p>
          <svg width="100%" height={140} style={{ background: '#fff', borderRadius: 8, border: '1px solid #e2e8f0' }} viewBox="0 0 500 140" preserveAspectRatio="none">
            <line x1={40} y1={110} x2={495} y2={110} stroke="#94a3b8" strokeWidth="1" />
            {bins.map((b, i) => {
              const bw = (495 - 40) / bins.length;
              const h = (b.count / maxCount) * 95;
              return (
                <g key={i}>
                  <rect x={40 + i * bw} y={110 - h} width={bw - 0.6} height={h} fill="#0284c7" fillOpacity={0.7} />
                </g>
              );
            })}
            {pct && ['p5', 'p50', 'p95'].map((k, i) => {
              const maxD = Math.max(...depthVals);
              const xk = 40 + (pct[k] / maxD) * (495 - 40);
              const col = i === 1 ? '#dc2626' : '#7c3aed';
              return (
                <g key={k}>
                  <line x1={xk} y1={10} x2={xk} y2={110} stroke={col} strokeWidth="1" strokeDasharray="4,3" />
                  <text x={xk} y={8} textAnchor="middle" fontSize="8" fill={col} fontFamily="JetBrains Mono" fontWeight="700">{k} {pct[k].toFixed(1)}</text>
                </g>
              );
            })}
            <text x={40} y={124} fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">0 m</text>
            <text x={495} y={124} textAnchor="end" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">{Math.max(...depthVals).toFixed(1)} m</text>
            <text x={38} y={112} textAnchor="end" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">0</text>
            <text x={38} y={18} textAnchor="end" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">{maxCount}</text>
          </svg>
          {pct && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6, marginTop: 6, fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}>
              {['p5', 'p25', 'p50', 'p75', 'p95'].map(k => (
                <div key={k} style={{ padding: '4px 6px', background: 'var(--bg-card)', border: '1px solid var(--border-dim)', borderRadius: 6, textAlign: 'center' }}>
                  <div style={{ fontSize: 8, color: 'var(--text-dim)' }}>{k.toUpperCase()}</div>
                  <div style={{ fontWeight: 700, color: 'var(--accent-primary)' }}>{pct[k].toFixed(2)} m</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tiling / pipeline state */}
      {tiled && (
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 10, border: '1px solid var(--border-dim)', background: 'var(--bg-card)' }}>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, marginBottom: 6 }}>4×4 TILED PIPELINE</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
            <MetaKV k="Grid" v={tiled.grid} />
            <MetaKV k="Tiles OK" v={`${tiled.tiles_ok}/${tiled.tiles_total}`} />
            <MetaKV k="Scenes total" v={tiled.total_scenes} />
            <MetaKV k="Workers" v={tiled.workers} />
          </div>
        </div>
      )}

      {/* Image metadata */}
      {md && (md.sensor || md.acquisition_dates) && (
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 10, border: '1px solid var(--border-dim)', background: 'var(--bg-card)' }}>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, marginBottom: 6 }}>IMAGE METADATA</p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
            <MetaKV k="Sensor" v={md.sensor} />
            <MetaKV k="Resolution" v={`${md.resolution_m} m`} />
            <MetaKV k="Dates" v={(md.acquisition_dates || []).join(', ') || '—'} />
            <MetaKV k="Scenes" v={md.n_scenes ?? '—'} />
            <MetaKV k="Method" v={md.method} />
            <MetaKV k="Max cloud" v={`${md.cloud_max_pct}%`} />
            <MetaKV k="CRS" v={md.crs} />
            <MetaKV k="Bands" v={(md.bands || []).join(', ')} />
            {md.turbidity && <MetaKV k="Turbidity" v={`${md.turbidity.class} (NDTI ${md.turbidity.ndti_mean})`} />}
            {md.turbidity?.secchi_depth_est_m && <MetaKV k="Secchi est." v={`${md.turbidity.secchi_depth_est_m} m`} />}
          </div>
        </div>
      )}

      {/* Sources used */}
      {sources.length > 0 && (
        <div style={{ marginBottom: 16, padding: 12, borderRadius: 10, border: '1px solid var(--border-dim)', background: 'var(--bg-card)' }}>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, marginBottom: 6 }}>SOURCES USED (IN ORDER)</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {sources.map((src, i) => (
              <span key={i} style={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace', padding: '3px 8px', background: 'var(--bg-secondary)', border: '1px solid var(--border-dim)', borderRadius: 4, color: 'var(--text-primary)' }}>
                {src}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Quick validation recap */}
      {validationResults && !validationResults.error && validationResults.n_pairs > 0 && (
        <div style={{ padding: 12, borderRadius: 10, border: '1.5px solid rgba(2,132,199,0.3)', background: 'rgba(2,132,199,0.03)' }}>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#075985', fontWeight: 800, marginBottom: 6 }}>
            VALIDATION AGAINST OBSERVED ({validationResults.n_pairs} pairs)
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(90px, 1fr))', gap: 6, fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
            <MetaKV k="RMSE" v={`${validationResults.rmse} m`} />
            <MetaKV k="MAE" v={`${validationResults.mae} m`} />
            <MetaKV k="Bias" v={`${validationResults.bias >= 0 ? '+' : ''}${validationResults.bias} m`} />
            <MetaKV k="R²" v={validationResults.r2} />
            {validationResults.iho_orders?.order1a && <MetaKV k="S-44 1a" v={`${validationResults.iho_orders.order1a.pass_pct}%`} />}
            {validationResults.class_confusion && <MetaKV k="Class agr." v={`${validationResults.class_confusion.agreement_pct}%`} />}
          </div>
        </div>
      )}
    </div>
  );
}

function BigStat({ label, value, unit, color = '#0284c7' }) {
  return (
    <div style={{ padding: 10, borderRadius: 10, background: 'var(--bg-card)', border: '1px solid var(--border-dim)', textAlign: 'center' }}>
      <div style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.1em' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 800, color }}>{value}{unit && <span style={{ fontSize: 12, marginLeft: 3, color: 'var(--text-dim)' }}>{unit}</span>}</div>
    </div>
  );
}
function CalibCell({ label, value, tone }) {
  return (
    <div style={{ padding: 6, background: 'var(--bg-secondary)', borderRadius: 6, textAlign: 'center' }}>
      <div style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>{label}</div>
      <div style={{ fontSize: 13, fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color: tone }}>{value}</div>
    </div>
  );
}
function MetaKV({ k, v }) {
  return (
    <div style={{ display: 'flex', gap: 6 }}>
      <span style={{ color: 'var(--text-dim)', minWidth: 90 }}>{k}:</span>
      <span style={{ color: 'var(--text-primary)', fontWeight: 600, wordBreak: 'break-word' }}>{v ?? '—'}</span>
    </div>
  );
}

// ── IHO Chart Tab: zoomable chart + image-metadata popup on click ──
function IHOChartTab({ results }) {
  const [selected, setSelected] = useState(null);
  const meta = useMemo(() => {
    const r = results || {};
    const stats = r.stats || {};
    const ml = r.ml_stats || {};
    const md = r.image_metadata || {};
    // Pull acquisition dates from multiple possible fields
    const dates =
      md.acquisition_dates ||
      ml.scene_dates ||
      (r.sources_used || []).filter(s => /dates=/.test(s)).map(s => (s.match(/dates=([^)]+)/) || [])[1]).filter(Boolean)[0]?.split(',') ||
      [];
    return {
      dates,
      bands: md.bands || ['B01 coastal', 'B02 blue', 'B03 green', 'B04 red', 'B08 NIR', 'B11 SWIR', 'SCL'],
      resolution_m: md.resolution_m || stats.resolution_m || r.resolution_m || '—',
      method: md.method || ml.method || 'Spectral ridge + MLE fusion',
      cloud_max_pct: md.cloud_max_pct ?? 30,
      sensor: md.sensor || 'Sentinel-2 L2A (ESA / CDSE)',
      crs: md.crs || 'EPSG:4326',
      mosaicking: md.mosaicking || 'leastCC per sub-window',
      sun_elev_deg: md.sun_elev_deg,
      view_zenith_deg: md.view_zenith_deg,
      n_scenes: ml.n_scenes || md.n_scenes,
      tile_grid: r.stats_tiled?.grid || md.tile_grid,
      tiles_ok: r.stats_tiled?.tiles_ok,
      workers: r.stats_tiled?.workers,
      turbidity: md.turbidity || null,
    };
  }, [results]);
  const handleClick = (info) => setSelected(info);
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.08em' }}>
          IHO S-52 ELECTRONIC CHART — WHEEL TO ZOOM · DRAG TO PAN · CLICK A SOUNDING FOR IMAGE METADATA
        </p>
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#0284c7' }}>
          {(results.interpolated_points || []).length || (results.points || []).length} pts
        </span>
      </div>
      <IHOChartView
        points={results.points}
        bbox={results.bbox}
        stats={results.stats}
        rasterPng={results.raster_png}
        rasterBounds={results.raster_bounds}
        metadata={meta}
        onPointClick={handleClick}
      />
      {/* Image metadata panel (shown after first click) */}
      <div style={{ marginTop: 12, padding: 12, background: 'var(--bg-card)', border: '1px solid var(--border-dim)', borderRadius: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.08em' }}>
            {selected ? 'CLICKED SOUNDING & IMAGE METADATA' : 'IMAGE METADATA — CLICK A SOUNDING FOR LOCATION-SPECIFIC DETAILS'}
          </p>
          {selected?.point && (
            <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color: '#0284c7' }}>
              {selected.point.depth.toFixed(1)} m @ {selected.point.lat.toFixed(5)}, {selected.point.lon.toFixed(5)}
            </span>
          )}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
          <MetaRow k="Sensor" v={meta.sensor} />
          <MetaRow k="Resolution" v={`${meta.resolution_m} m`} />
          <MetaRow k="Acquisition dates" v={meta.dates.length ? meta.dates.join(', ') : '—'} />
          <MetaRow k="Scenes fused" v={meta.n_scenes ?? '—'} />
          <MetaRow k="Method" v={meta.method} />
          <MetaRow k="Tile grid" v={meta.tile_grid ? `${meta.tile_grid} (${meta.tiles_ok}/16 OK, ${meta.workers}w)` : 'single tile'} />
          <MetaRow k="Bands" v={meta.bands.join(', ')} />
          <MetaRow k="Mosaicking" v={meta.mosaicking} />
          <MetaRow k="Max cloud" v={`${meta.cloud_max_pct}%`} />
          <MetaRow k="CRS" v={meta.crs} />
          {meta.sun_elev_deg != null && <MetaRow k="Sun elevation" v={`${meta.sun_elev_deg.toFixed(1)}°`} />}
          {meta.view_zenith_deg != null && <MetaRow k="View zenith" v={`${meta.view_zenith_deg.toFixed(1)}°`} />}
          {meta.turbidity && (
            <>
              <MetaRow k="Turbidity NDTI" v={`${meta.turbidity.ndti_mean?.toFixed(3)} (${meta.turbidity.class})`} />
              <MetaRow k="Red/Blue ratio" v={meta.turbidity.r_b_ratio?.toFixed(3)} />
              <MetaRow k="Secchi depth est." v={meta.turbidity.secchi_depth_est_m ? `${meta.turbidity.secchi_depth_est_m.toFixed(1)} m` : '—'} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function MetaRow({ k, v }) {
  return (
    <div style={{ display: 'flex', gap: 6 }}>
      <span style={{ color: 'var(--text-dim)', minWidth: 110 }}>{k}:</span>
      <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{v}</span>
    </div>
  );
}

function StatCard({ label, value, unit, accent, icon }) {
  return (
    <div style={{
      padding: '12px', background: 'var(--bg-card)', borderRadius: 'var(--radius)',
      border: '1px solid var(--border-subtle)', transition: 'border-color 0.2s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px' }}>
        {icon && <span style={{ fontSize: '10px' }}>{icon}</span>}
        <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600, letterSpacing: '0.08em' }}>{label}</p>
      </div>
      <p style={{ fontSize: '20px', fontWeight: 800, color: accent || 'var(--accent-primary)', lineHeight: 1.1, letterSpacing: '-0.02em' }}>
        {value}<span style={{ fontSize: '10px', fontWeight: 500, color: 'var(--text-dim)', marginLeft: '3px' }}>{unit}</span>
      </p>
    </div>
  );
}

function DepthHistogram({ points, maxDepth }) {
  const bins = useMemo(() => {
    const bp = (points || []).filter(p => ['bathymetry','interpolated'].includes(p.photon_class));
    const nBins = 24;
    const bw = (maxDepth || 30) / nBins;
    const counts = new Array(nBins).fill(0);
    bp.forEach(p => {
      const b = Math.min(Math.floor(Math.abs(p.depth) / bw), nBins - 1);
      counts[b]++;
    });
    const mx = Math.max(...counts);
    return counts.map((c, i) => ({
      depth: (i * bw + bw / 2).toFixed(1),
      count: c,
      pct: mx > 0 ? c / mx : 0,
    }));
  }, [points, maxDepth]);

  return (
    <div>
      <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '8px' }}>DEPTH DISTRIBUTION</p>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '2px', height: '80px' }}>
        {bins.map((b, i) => (
          <div key={i} title={`${b.depth}m: ${b.count} photons`} style={{
            flex: 1,
            height: `${Math.max(b.pct * 100, 2)}%`,
            borderRadius: '2px 2px 0 0',
            background: `linear-gradient(to top, #312e81, #6366f1, #38bdf8)`,
            opacity: 0.25 + b.pct * 0.75,
            cursor: 'pointer',
            transition: 'opacity 0.15s',
          }} />
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
        <span>0m</span><span>{(maxDepth / 2).toFixed(0)}m</span><span>{maxDepth.toFixed(0)}m</span>
      </div>
    </div>
  );
}

function DepthProfile({ profile, title, color }) {
  if (!profile || profile.length === 0) return null;
  const valid = profile.filter(p => p.height !== null && p.height !== undefined);
  if (valid.length < 2) return null;
  const minH = Math.min(...valid.map(p => p.height));
  const maxH = Math.max(...valid.map(p => p.height));
  const range = maxH - minH || 1;

  const pathData = valid.map((p, i) => {
    const x = (i / (valid.length - 1)) * 300;
    const y = ((p.height - minH) / range) * 40 + 2;
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');

  const areaPath = `${pathData} L 300 42 L 0 42 Z`;

  return (
    <div style={{ marginBottom: '12px' }}>
      <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: color || 'var(--text-secondary)', fontWeight: 600, marginBottom: '4px' }}>{title}</p>
      <svg width="100%" height="44" viewBox="0 0 300 44" preserveAspectRatio="none" style={{ display: 'block' }}>
        <defs>
          <linearGradient id={`grad-${title.replace(/\s/g, '')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color || '#38bdf8'} stopOpacity="0.3" />
            <stop offset="100%" stopColor="#312e81" stopOpacity="0.6" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill={`url(#grad-${title.replace(/\s/g, '')})`} />
        <path d={pathData} fill="none" stroke={color || '#38bdf8'} strokeWidth="1.5" strokeOpacity="0.7" />
      </svg>
    </div>
  );
}

function DiffMapSVG({ points, bbox, stats }) {
  if (!points || !bbox) return null;
  const w = 380, h = 340;
  const spanLat = bbox.north - bbox.south || 0.01, spanLon = bbox.east - bbox.west || 0.01;
  const maxAbs = Math.max(Math.abs(stats.max_diff), Math.abs(stats.min_diff), 1);
  const dColor = (d) => {
    const t = Math.max(-1, Math.min(1, d / maxAbs));
    if (Math.abs(t) < 0.1) return '#e2e8f0';
    if (t > 0) return `rgba(225,29,72,${Math.min(0.9, 0.2 + t * 0.7)})`;
    return `rgba(5,150,105,${Math.min(0.9, 0.2 + Math.abs(t) * 0.7)})`;
  };
  const gx = 40, gy = 40;
  const grid = Array.from({ length: gy }, () => Array(gx).fill(NaN));
  const cnt = Array.from({ length: gy }, () => Array(gx).fill(0));
  points.forEach(p => {
    const c = Math.floor((p.lon - bbox.west) / spanLon * (gx - 1));
    const r = Math.floor((bbox.north - p.lat) / spanLat * (gy - 1));
    if (r >= 0 && r < gy && c >= 0 && c < gx) {
      grid[r][c] = cnt[r][c] === 0 ? p.diff : (grid[r][c] * cnt[r][c] + p.diff) / (cnt[r][c] + 1);
      cnt[r][c]++;
    }
  });
  const cw = w / gx, ch = h / gy;
  return (
    <svg width={w} height={h} style={{ borderRadius: '10px', border: '1px solid #e2e8f0', background: '#fafafa' }}>
      {Array.from({ length: gy }, (_, r) => Array.from({ length: gx }, (_, c) => {
        if (isNaN(grid[r][c])) return null;
        return <rect key={`${r}-${c}`} x={c * cw} y={r * ch} width={cw + 0.5} height={ch + 0.5} fill={dColor(grid[r][c])} />;
      }))}
      <rect x={8} y={8} width={110} height={22} rx={4} fill="rgba(255,255,255,0.9)" stroke="#cbd5e1" strokeWidth="0.5" />
      <text x={14} y={22} fontSize="8" fontWeight="700" fill="#334155" fontFamily="JetBrains Mono">DEPTH CHANGE MAP</text>
    </svg>
  );
}

function NauticalChartSVG({ points, bbox, stats }) {
  if (!points || !bbox) return null;
  const w = 380, h = 500;
  const spanLat = bbox.north - bbox.south || 0.01, spanLon = bbox.east - bbox.west || 0.01;
  const maxD = stats?.max_depth || 25;
  const dp = points.filter(p => ['interpolated','bathymetry','chart'].includes(p.photon_class) && p.depth > 0);
  const step = Math.max(1, Math.floor(dp.length / 200));
  const dColor = (d) => { const t = Math.min(d / maxD, 1); return t < 0.05 ? '#c6ecff' : t < 0.15 ? '#a8daf0' : t < 0.3 ? '#7ec8e3' : t < 0.5 ? '#3da5d9' : t < 0.7 ? '#2176ae' : '#1a5276'; };
  return (
    <svg width={w} height={h} style={{ fontFamily: 'JetBrains Mono, monospace' }}>
      <rect x="0" y="0" width={w} height={h} fill="#e8f4fc" />
      {[0.25,0.5,0.75].map(f => (
        <g key={f}>
          <line x1={f*w} y1={0} x2={f*w} y2={h} stroke="#c8dce8" strokeWidth="0.5" strokeDasharray="4,4" />
          <line x1={0} y1={f*h} x2={w} y2={f*h} stroke="#c8dce8" strokeWidth="0.5" strokeDasharray="4,4" />
          <text x={f*w+2} y={h-4} fontSize="6" fill="#8ba5b5">{(bbox.west+f*spanLon).toFixed(3)}°E</text>
          <text x={2} y={f*h-2} fontSize="6" fill="#8ba5b5">{(bbox.north-f*spanLat).toFixed(3)}°N</text>
        </g>
      ))}
      {(() => {
        const gx=40,gy=50,cells=[],grid=Array.from({length:gy},()=>Array(gx).fill(NaN)),cnt=Array.from({length:gy},()=>Array(gx).fill(0));
        dp.forEach(p=>{const c=Math.floor((p.lon-bbox.west)/spanLon*(gx-1)),r=Math.floor((bbox.north-p.lat)/spanLat*(gy-1));if(r>=0&&r<gy&&c>=0&&c<gx){grid[r][c]=cnt[r][c]===0?p.depth:(grid[r][c]*cnt[r][c]+p.depth)/(cnt[r][c]+1);cnt[r][c]++;}});
        const cw=w/gx,ch=h/gy;
        for(let r=0;r<gy;r++)for(let c=0;c<gx;c++)if(!isNaN(grid[r][c])&&grid[r][c]>0)cells.push(<rect key={`${r}-${c}`} x={c*cw} y={r*ch} width={cw+0.5} height={ch+0.5} fill={dColor(grid[r][c])} opacity="0.7"/>);
        return cells;
      })()}
      {dp.filter((_,i)=>i%step===0).map((p,i)=>{
        const x=((p.lon-bbox.west)/spanLon)*w,y=((bbox.north-p.lat)/spanLat)*h;
        if(x<5||x>w-15||y<8||y>h-8)return null;
        return <text key={i} x={x} y={y} fontSize="7" fontWeight="600" fill="#1a3c5e" textAnchor="middle">{p.depth.toFixed(1)}</text>;
      })}
      <rect x={8} y={8} width={160} height={36} rx={4} fill="rgba(255,255,255,0.9)" stroke="#8ba5b5" strokeWidth="0.5"/>
      <text x={14} y={22} fontSize="8" fontWeight="700" fill="#1a3c5e">BATHYMETRIC CHART</text>
      <text x={14} y={34} fontSize="6" fill="#5b7d8e">{INTERNAL_ANALYSIS ? 'Generated — Sentinel-2 + CNN' : 'Generated depth product'}</text>
      <rect x={8} y={h-70} width={60} height={62} rx={4} fill="rgba(255,255,255,0.9)" stroke="#8ba5b5" strokeWidth="0.5"/>
      <text x={14} y={h-56} fontSize="6" fontWeight="600" fill="#5b7d8e">DEPTH (m)</text>
      {[0,0.25,0.5,0.75,1].map((f,i)=>(<g key={i}><rect x={14} y={h-50+i*9} width={10} height={7} fill={dColor(maxD*f)}/><text x={28} y={h-44+i*9} fontSize="6" fill="#5b7d8e">{(maxD*f).toFixed(0)}</text></g>))}
      <circle cx={w-24} cy={24} r={12} fill="rgba(255,255,255,0.8)" stroke="#8ba5b5" strokeWidth="0.5"/>
      <text x={w-24} y={17} textAnchor="middle" fontSize="7" fontWeight="700" fill="#e11d48">N</text>
      <line x1={w-24} y1={14} x2={w-24} y2={34} stroke="#1a3c5e" strokeWidth="0.8"/>
    </svg>
  );
}

function ContourMapSVG({ points, contours, bbox, stats, professional }) {
  if (!bbox) return null;
  const w = 420, h = 520;
  const pad = { t: 48, r: 20, b: 60, l: 50 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const spanLat = bbox.north - bbox.south || 0.01, spanLon = bbox.east - bbox.west || 0.01;
  const maxD = stats?.max_depth || 25;

  const toX = (lon) => pad.l + ((lon - bbox.west) / spanLon) * pw;
  const toY = (lat) => pad.t + ((bbox.north - lat) / spanLat) * ph;

  // Build depth grid for background fill
  const dp = (points || []).filter(p => ['interpolated','bathymetry','chart'].includes(p.photon_class) && p.depth > 0);
  const gx = 60, gy = 70;
  const grid = Array.from({ length: gy }, () => Array(gx).fill(NaN));
  const cnt = Array.from({ length: gy }, () => Array(gx).fill(0));
  dp.forEach(p => {
    const c = Math.floor((p.lon - bbox.west) / spanLon * (gx - 1));
    const r = Math.floor((bbox.north - p.lat) / spanLat * (gy - 1));
    if (r >= 0 && r < gy && c >= 0 && c < gx) {
      grid[r][c] = cnt[r][c] === 0 ? p.depth : (grid[r][c] * cnt[r][c] + p.depth) / (cnt[r][c] + 1);
      cnt[r][c]++;
    }
  });
  const cw = pw / gx, ch = ph / gy;

  const dColor = (d) => {
    const t = Math.min(d / maxD, 1);
    if (t < 0.04) return '#e0f7fa';
    if (t < 0.1) return '#b2ebf2';
    if (t < 0.2) return '#80deea';
    if (t < 0.3) return '#4dd0e1';
    if (t < 0.4) return '#26c6da';
    if (t < 0.5) return '#00acc1';
    if (t < 0.6) return '#0097a7';
    if (t < 0.7) return '#00838f';
    if (t < 0.8) return '#006064';
    if (t < 0.9) return '#004d5e';
    return '#003545';
  };

  const contourColor = (d) => {
    const t = Math.min(d / maxD, 1);
    if (t < 0.3) return '#01579b';
    if (t < 0.6) return '#b71c1c';
    return '#1b5e20';
  };

  const contourLines = (contours || []).map((c, idx) => {
    if (!c.coords || c.coords.length < 2) return null;
    const pathD = c.coords.map((pt, i) => {
      const x = toX(pt.lon);
      const y = toY(pt.lat);
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    // Label position at midpoint
    const mid = c.coords[Math.floor(c.coords.length / 2)];
    const mx = toX(mid.lon);
    const my = toY(mid.lat);
    return (
      <g key={idx}>
        <path d={pathD} fill="none" stroke={contourColor(c.depth)} strokeWidth={c.depth % 5 === 0 ? '1.8' : '0.9'} strokeOpacity={c.depth % 5 === 0 ? '0.9' : '0.6'} strokeLinecap="round" strokeLinejoin="round" />
        {c.depth % 1 === 0 && (
          <g>
            <rect x={mx - 10} y={my - 6} width="20" height="11" rx="2" fill="rgba(255,255,255,0.88)" stroke={contourColor(c.depth)} strokeWidth="0.4" />
            <text x={mx} y={my + 3} textAnchor="middle" fontSize="7" fontWeight="700" fill={contourColor(c.depth)} fontFamily="JetBrains Mono">{c.depth}</text>
          </g>
        )}
      </g>
    );
  }).filter(Boolean);

  // Tick marks
  const nTicksX = 5, nTicksY = 5;
  const ticks = [];
  for (let i = 0; i <= nTicksX; i++) {
    const lon = bbox.west + (i / nTicksX) * spanLon;
    const x = toX(lon);
    ticks.push(<g key={`tx${i}`}><line x1={x} y1={pad.t + ph} x2={x} y2={pad.t + ph + 5} stroke="#90a4ae" strokeWidth="0.5" /><text x={x} y={pad.t + ph + 16} textAnchor="middle" fontSize="7" fill="#607d8b" fontFamily="JetBrains Mono">{lon.toFixed(3)}°E</text></g>);
    ticks.push(<line key={`gx${i}`} x1={x} y1={pad.t} x2={x} y2={pad.t + ph} stroke="#cfd8dc" strokeWidth="0.3" strokeDasharray="3,3" />);
  }
  for (let i = 0; i <= nTicksY; i++) {
    const lat = bbox.north - (i / nTicksY) * spanLat;
    const y = toY(lat);
    ticks.push(<g key={`ty${i}`}><line x1={pad.l - 5} y1={y} x2={pad.l} y2={y} stroke="#90a4ae" strokeWidth="0.5" /><text x={pad.l - 8} y={y + 3} textAnchor="end" fontSize="7" fill="#607d8b" fontFamily="JetBrains Mono">{lat.toFixed(3)}°N</text></g>);
    ticks.push(<line key={`gy${i}`} x1={pad.l} y1={y} x2={pad.l + pw} y2={y} stroke="#cfd8dc" strokeWidth="0.3" strokeDasharray="3,3" />);
  }

  // Legend
  const legendSteps = 8;

  return (
    <svg width={w} height={h} style={{ fontFamily: 'JetBrains Mono, monospace' }}>
      {/* Background */}
      <rect x="0" y="0" width={w} height={h} fill="#fafbfc" rx="10" />
      {/* Border frame */}
      <rect x={pad.l} y={pad.t} width={pw} height={ph} fill="#e8f0f7" stroke="#90a4ae" strokeWidth="0.5" />
      {/* Grid lines */}
      {ticks}
      {/* Depth color fill */}
      {Array.from({ length: gy }, (_, r) => Array.from({ length: gx }, (_, c) => {
        if (isNaN(grid[r][c]) || grid[r][c] <= 0) return null;
        return <rect key={`g${r}-${c}`} x={pad.l + c * cw} y={pad.t + r * ch} width={cw + 0.5} height={ch + 0.5} fill={dColor(grid[r][c])} opacity="0.75" />;
      }))}
      {/* Contour lines */}
      {contourLines}
      {/* Title block */}
      <rect x={pad.l + 4} y={pad.t + 4} width={200} height={34} rx={4} fill="rgba(255,255,255,0.92)" stroke="#78909c" strokeWidth="0.5" />
      <text x={pad.l + 12} y={pad.t + 18} fontSize="10" fontWeight="800" fill="#263238">BATHYMETRIC CONTOUR MAP</text>
      <text x={pad.l + 12} y={pad.t + 30} fontSize="7" fill="#607d8b">{(INTERNAL_ANALYSIS && professional?.method) ? `${professional.method} — Isobaths in metres` : 'Isobaths in metres'}</text>
      {/* Compass */}
      <circle cx={pad.l + pw - 18} cy={pad.t + 22} r={14} fill="rgba(255,255,255,0.85)" stroke="#90a4ae" strokeWidth="0.5" />
      <text x={pad.l + pw - 18} y={pad.t + 15} textAnchor="middle" fontSize="8" fontWeight="800" fill="#b71c1c">N</text>
      <line x1={pad.l + pw - 18} y1={pad.t + 10} x2={pad.l + pw - 18} y2={pad.t + 34} stroke="#263238" strokeWidth="0.8" />
      <polygon points={`${pad.l + pw - 18},${pad.t + 9} ${pad.l + pw - 21},${pad.t + 15} ${pad.l + pw - 15},${pad.t + 15}`} fill="#263238" />
      {/* Color bar legend */}
      <rect x={pad.l + pw - 72} y={pad.t + ph - 115} width={68} height={110} rx={4} fill="rgba(255,255,255,0.92)" stroke="#90a4ae" strokeWidth="0.5" />
      <text x={pad.l + pw - 67} y={pad.t + ph - 100} fontSize="7" fontWeight="700" fill="#455a64">DEPTH (m)</text>
      {Array.from({ length: legendSteps }, (_, i) => {
        const d = (maxD / legendSteps) * (i + 0.5);
        return (
          <g key={`leg${i}`}>
            <rect x={pad.l + pw - 67} y={pad.t + ph - 92 + i * 10} width={14} height={9} rx="1" fill={dColor(d)} />
            <text x={pad.l + pw - 49} y={pad.t + ph - 84 + i * 10} fontSize="7" fill="#455a64">{d.toFixed(1)}</text>
          </g>
        );
      })}
      {/* Scale bar */}
      <line x1={pad.l + 8} y1={h - 22} x2={pad.l + 68} y2={h - 22} stroke="#263238" strokeWidth="1.5" />
      <line x1={pad.l + 8} y1={h - 25} x2={pad.l + 8} y2={h - 19} stroke="#263238" strokeWidth="1" />
      <line x1={pad.l + 68} y1={h - 25} x2={pad.l + 68} y2={h - 19} stroke="#263238" strokeWidth="1" />
      <text x={pad.l + 38} y={h - 12} textAnchor="middle" fontSize="7" fill="#455a64">{(spanLon * 111 * Math.cos(((bbox.north + bbox.south) / 2) * Math.PI / 180) * 60 / pw).toFixed(2)} km/div</text>
      {/* Datum note */}
      <text x={w - 8} y={h - 8} textAnchor="end" fontSize="6" fill="#90a4ae">Datum: WGS84 | Depths: MSL | Max {maxD.toFixed(1)}m</text>
    </svg>
  );
}

function ProfessionalEstimation({ professional }) {
  if (!professional || !professional.depth_stats) return null;
  const p = professional;
  const ds = p.depth_stats;

  return (
    <div>
      {/* Confidence banner */}
      <div style={{
        padding: '14px', borderRadius: '10px', marginBottom: '14px',
        background: `linear-gradient(135deg, ${p.confidence_color}10, ${p.confidence_color}05)`,
        border: `1.5px solid ${p.confidence_color}40`,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.12em' }}>CATZOC / IHO ORDER</p>
            <p style={{ fontSize: '22px', fontWeight: 900, color: p.confidence_color, letterSpacing: '-0.02em', marginTop: '2px' }}>
              {p.confidence_level}
              {(p.iho_assessment && p.iho_assessment.order_label) && (
                <span style={{ fontSize: '13px', fontWeight: 600 }}> · {p.iho_assessment.order_label}</span>
              )}
            </p>
          </div>
          {INTERNAL_ANALYSIS && (
          <div style={{ textAlign: 'right' }}>
            <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>{p.method}</p>
            <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>R² = {p.r2?.toFixed(4) || '—'}</p>
            <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>{p.n_training_points} ref pts</p>
          </div>
          )}
        </div>
      </div>

      {/* Survey metadata */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '6px', marginBottom: '14px' }}>
        {[
          ['AREA', `${p.area_km2} km²`, 'var(--accent-primary)'],
          ['RESOLUTION', `${p.resolution_m} m`, 'var(--accent-teal)'],
          ['GRID', p.grid_size, 'var(--accent-blue)'],
          ['COVERAGE', `${p.coverage_pct}%`, 'var(--accent-emerald)'],
          ['WATER PX', p.water_pixels?.toLocaleString(), 'var(--accent-violet)'],
          ['TOTAL PX', p.total_pixels?.toLocaleString(), 'var(--text-dim)'],
        ].map(([label, val, color], i) => (
          <div key={i} style={{ padding: '8px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--border-dim)', textAlign: 'center' }}>
            <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600 }}>{label}</p>
            <p style={{ fontSize: '13px', fontWeight: 700, color }}>{val}</p>
          </div>
        ))}
      </div>

      {/* Depth statistics */}
      <div style={{ padding: '14px', background: 'var(--bg-card)', borderRadius: 'var(--radius)', border: '1px solid var(--border-subtle)', marginBottom: '14px' }}>
        <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '10px' }}>DEPTH STATISTICS</p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '8px', marginBottom: '10px' }}>
          {[
            ['Mean', ds.mean, 'm'], ['Median', ds.median, 'm'], ['Std Dev', ds.std, 'm'], ['Range', `${ds.min}-${ds.max}`, 'm'],
          ].map(([l, v, u], i) => (
            <div key={i} style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600 }}>{l}</p>
              <p style={{ fontSize: '14px', fontWeight: 700, color: 'var(--accent-primary)' }}>{typeof v === 'number' ? v.toFixed(2) : v}<span style={{ fontSize: '8px', color: 'var(--text-dim)' }}>{u}</span></p>
            </div>
          ))}
        </div>
        {/* Percentile bar */}
        <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600, marginBottom: '6px' }}>PERCENTILES</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '2px', height: '24px', borderRadius: '4px', overflow: 'hidden', background: 'var(--bg-secondary)' }}>
          {[
            { label: 'P5', val: ds.percentiles.p5, width: 20, color: '#e0f7fa' },
            { label: 'P25', val: ds.percentiles.p25, width: 20, color: '#80deea' },
            { label: 'P50', val: ds.percentiles.p50, width: 20, color: '#26c6da' },
            { label: 'P75', val: ds.percentiles.p75, width: 20, color: '#00838f' },
            { label: 'P95', val: ds.percentiles.p95, width: 20, color: '#004d5e' },
          ].map((pc, i) => (
            <div key={i} style={{ flex: 1, background: pc.color, height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ fontSize: '6px', fontWeight: 700, color: i < 3 ? '#004d5e' : '#e0f7fa' }}>{pc.label}</span>
              <span style={{ fontSize: '8px', fontWeight: 600, color: i < 3 ? '#004d5e' : '#e0f7fa' }}>{pc.val}m</span>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
          <span>Skewness: {ds.skewness}</span>
          <span>Kurtosis: {ds.kurtosis}</span>
        </div>
      </div>

      {/* IHO S-44 Assessment — INTERNAL ONLY (IHO accuracy/compliance) */}
      {INTERNAL_ANALYSIS && p.iho_assessment && (
      <div style={{ padding: '14px', background: 'rgba(5,150,105,0.03)', borderRadius: 'var(--radius)', border: '1px solid rgba(5,150,105,0.15)', marginBottom: '14px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: '#059669', fontWeight: 700, letterSpacing: '0.1em' }}>IHO S-44 COMPLIANCE (p95 ≤ TVU)</p>
          <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', padding: '2px 8px', borderRadius: '4px', background: (p.iho_assessment.zones_passing_order2 === p.iho_assessment.zones_with_data && p.iho_assessment.zones_with_data > 0) ? 'rgba(5,150,105,0.15)' : 'rgba(225,29,72,0.12)', color: (p.iho_assessment.zones_passing_order2 === p.iho_assessment.zones_with_data && p.iho_assessment.zones_with_data > 0) ? '#059669' : '#e11d48', fontWeight: 700 }}>
            {p.iho_assessment.zones_passing_order2}/{p.iho_assessment.zones_with_data} ZONES PASS ORDER 2
          </span>
        </div>
        <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginBottom: '8px' }}>
          {p.iho_assessment.standard} — {p.iho_assessment.pass_criterion}
          {p.iho_assessment.catzoc_tier && <> · CATZOC {p.iho_assessment.catzoc_tier}</>}
        </p>
        {/* Zone detail table — reference TVUs + null-safe per-zone RMSE/p95 (ITEM 6) */}
        <div style={{ maxHeight: '200px', overflow: 'auto', borderRadius: '6px', border: '1px solid rgba(5,150,105,0.1)' }}>
          <table style={{ width: '100%', fontSize: '8px', fontFamily: 'var(--font-mono)', borderCollapse: 'collapse' }}>
            <thead><tr style={{ background: 'rgba(5,150,105,0.05)', position: 'sticky', top: 0 }}>
              {['Zone', 'Range', '%', 'Mean', 'TVU(1a)', 'TVU(2)', 'Zone RMSE', 'Zone p95', 'Order met'].map(h => (
                <th key={h} style={{ padding: '4px 5px', textAlign: 'left', color: '#059669', fontWeight: 700 }}>{h}</th>
              ))}
            </tr></thead>
            <tbody>{(p.zones_detailed || []).map((z, i) => {
              const hasData = z.rmse_at_zone != null;
              const fail = hasData && z.p95_at_zone > z.tvu_s44_order2;
              return (
              <tr key={i} style={{ borderBottom: '1px solid rgba(5,150,105,0.06)', background: fail ? 'rgba(225,29,72,0.03)' : 'transparent' }}>
                <td style={{ padding: '3px 5px', fontWeight: 600 }}><span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, background: z.color, marginRight: 4, verticalAlign: 'middle' }} />{z.label}</td>
                <td style={{ padding: '3px 5px' }}>{z.range}</td>
                <td style={{ padding: '3px 5px' }}>{z.pct}%</td>
                <td style={{ padding: '3px 5px', color: 'var(--accent-primary)' }}>{z.mean}m</td>
                <td style={{ padding: '3px 5px' }}>{z.tvu_s44_order1a}m</td>
                <td style={{ padding: '3px 5px' }}>{z.tvu_s44_order2}m</td>
                <td style={{ padding: '3px 5px', color: hasData ? 'var(--text-secondary)' : 'var(--text-dim)' }}>{hasData ? `${z.rmse_at_zone}m` : 'N/A'}</td>
                <td style={{ padding: '3px 5px', color: hasData ? (fail ? '#e11d48' : '#059669') : 'var(--text-dim)' }}>{hasData ? `${z.p95_at_zone}m` : 'N/A'}</td>
                <td style={{ padding: '3px 5px', fontWeight: 700, color: !hasData ? 'var(--text-dim)' : (fail ? '#e11d48' : '#059669') }}>{hasData ? z.order_met : 'N/A — RMSE n/a'}</td>
              </tr>
            );})}</tbody>
          </table>
        </div>
      </div>
      )}

      {/* Residuals — INTERNAL ONLY (accuracy/error metrics) */}
      {INTERNAL_ANALYSIS && p.residuals && (
        <div style={{ padding: '14px', background: 'rgba(124,58,237,0.03)', borderRadius: 'var(--radius)', border: '1px solid rgba(124,58,237,0.15)', marginBottom: '14px' }}>
          <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: '#7c3aed', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '10px' }}>REFERENCE RESIDUAL ANALYSIS</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
            {[
              ['RMSE', p.residuals.rmse, 'm'], ['MAE', p.residuals.mae, 'm'],
              ['Bias', p.residuals.bias, 'm'], ['P95 Err', p.residuals.p95_error, 'm'],
            ].map(([l, v, u], i) => (
              <div key={i} style={{ padding: '6px', borderRadius: '6px', background: 'var(--bg-card)', textAlign: 'center' }}>
                <p style={{ fontSize: '7px', color: 'var(--text-dim)', fontWeight: 600 }}>{l}</p>
                <p style={{ fontSize: '13px', fontWeight: 700, color: '#7c3aed' }}>{v.toFixed(3)}<span style={{ fontSize: '8px', color: 'var(--text-dim)' }}>{u}</span></p>
              </div>
            ))}
          </div>
          <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginTop: '6px' }}>
            Based on {p.residuals.n} reference point comparisons · Max abs error: {p.residuals.max_abs}m
          </p>
        </div>
      )}

      {/* Slope analysis */}
      {p.slope && (
        <div style={{ padding: '14px', background: 'var(--bg-card)', borderRadius: 'var(--radius)', border: '1px solid var(--border-subtle)' }}>
          <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '8px' }}>SEABED MORPHOLOGY</p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600 }}>MEAN SLOPE</p>
              <p style={{ fontSize: '15px', fontWeight: 700, color: 'var(--accent-teal)' }}>{p.slope.mean_slope_deg}°</p>
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600 }}>MAX SLOPE</p>
              <p style={{ fontSize: '15px', fontWeight: 700, color: 'var(--accent-blue)' }}>{p.slope.max_slope_deg}°</p>
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600 }}>GRADIENT</p>
              <p style={{ fontSize: '15px', fontWeight: 700, color: 'var(--accent-violet)' }}>{p.slope.mean_gradient}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ResultsPanel({ results, resultsB, diffResults, validationResults, onClose, onExport, onShowOverlay, activeOverlayKey }) {
  const [tab, setTab] = useState('overview');
  const [fullscreen, setFullscreen] = useState(false);
  const stats = results?.stats || {};
  const maxD = stats.max_depth || 30;
  const hasDiff = diffResults && diffResults.points && diffResults.points.length > 0;
  const hasVal = !!validationResults && !validationResults.error && Array.isArray(validationResults.pairs) && validationResults.pairs.length > 0;
  const hasValError = !!validationResults && (validationResults.error || (Array.isArray(validationResults.pairs) && validationResults.pairs.length === 0));

  const containerStyle = fullscreen
    ? {
        position: 'fixed', inset: 0, zIndex: 10000, background: 'var(--bg-secondary)',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        animation: 'fadeIn 0.2s var(--ease-out)',
      }
    : {
        gridRow: '2', background: 'var(--bg-secondary)',
        borderLeft: '1px solid var(--border-dim)',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        animation: 'slideInRight 0.4s var(--ease-out)',
      };

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={{
        padding: '12px 16px', borderBottom: '1px solid var(--border-dim)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <div>
          <h3 style={{ fontSize: fullscreen ? '18px' : '14px', fontWeight: 800, color: 'var(--accent-primary)', letterSpacing: '-0.02em' }}>
            Analysis Results {fullscreen && <span style={{ fontSize: 10, color: 'var(--text-dim)', fontWeight: 500, marginLeft: 8 }}>FULLSCREEN</span>}
          </h3>
          <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginTop: '2px' }}>
            {stats.grid_points || stats.bathy_photons || 0} depth points · {stats.confidence ? `${stats.confidence}% confidence` : `${stats.granules_processed || 0} sources`}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <button onClick={() => setFullscreen(f => !f)}
            title={fullscreen ? 'Exit fullscreen' : 'Open in fullscreen (big window + advanced tab)'}
            style={{
              height: 28, padding: '0 10px', borderRadius: 8,
              border: '1px solid var(--accent-primary)',
              background: fullscreen ? 'var(--accent-primary)' : 'var(--bg-card)',
              color: fullscreen ? '#fff' : 'var(--accent-primary)',
              cursor: 'pointer', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
            }}>
            {fullscreen ? '⟱ EXIT' : '⛶ FULLSCREEN'}
          </button>
          <button onClick={onClose} style={{
            width: '28px', height: '28px', borderRadius: '8px',
            border: '1px solid var(--border-dim)', background: 'var(--bg-card)',
            color: 'var(--text-dim)', cursor: 'pointer', fontSize: '14px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>×</button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border-dim)', overflowX: 'auto' }}>
        {['overview',
          ...(SHOW_IHO_SURFACE ? ['iho_chart'] : []),
          'contour', 'nautical',
          ...(hasDiff ? ['diff'] : []),
          ...((SHOW_IHO_SURFACE && (hasVal || hasValError)) ? ['validation'] : []),
          ...(INTERNAL_ANALYSIS ? ['advanced'] : []),
          'profiles', 'data'].map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            flex: 1, padding: '8px', fontSize: '9px', fontFamily: 'var(--font-mono)',
            fontWeight: 600, border: 'none', cursor: 'pointer', letterSpacing: '0.06em',
            background: tab === t ? 'var(--bg-tertiary)' : 'transparent',
            color: tab === t ? 'var(--accent-primary)' : 'var(--text-dim)',
            borderBottom: tab === t ? '2px solid var(--accent-primary)' : '2px solid transparent',
          }}>{t === 'iho_chart' ? 'IHO CHART' : t.toUpperCase()}</button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>

        {/* ══════ OVERVIEW TAB ══════ */}
        {tab === 'overview' && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '16px' }}>
              <StatCard label="MEAN DEPTH" value={stats.mean_depth?.toFixed(2)} unit="m" />
              <StatCard icon="⬇" label="MAX DEPTH" value={stats.max_depth?.toFixed(2)} unit="m" accent="var(--accent-blue)" />
              <StatCard label="DEPTH PIXELS" value={(stats.grid_points || stats.sdb_pixels || 0).toLocaleString()} accent="var(--accent-teal)" />
              <StatCard label="CONFIDENCE" value={stats.confidence ? `${stats.confidence}` : '—'} unit="%" accent="var(--accent-amber)" />
              <StatCard icon="📈" label="STD DEV" value={stats.std_depth?.toFixed(2)} unit="m" accent="var(--accent-emerald)" />
              <StatCard icon="⬆" label="MIN DEPTH" value={stats.min_depth?.toFixed(2)} unit="m" accent="var(--text-secondary)" />
            </div>

            {/* ── Depth colorbar + honest σ / CATZOC summary ── */}
            {(() => {
              const m = results?.metrics || {};
              const ml = results?.ml_stats || {};
              const sigma = stats.mean_sigma ?? m.sigma_median ?? ml.sigma_median ?? null;
              // CATZOC / IHO S-44 are gated to SHOW_IHO_SURFACE (internal OR
              // hydro-client tier, IHO-R1) — plain consumer portal shows only
              // the σ uncertainty (no quality-tier / standard disclosure).
              const catzoc = SHOW_IHO_SURFACE ? (m.catzoc ?? results?.iho?.catzoc ?? null) : null;
              const s44 = SHOW_IHO_SURFACE ? (m.s44_1a_pct ?? null) : null;
              // ADPorts F3 — tide chip renders for ALL tiers (safety info, not IHO-gated).
              const tc = results?.tide_correction;
              const hasHonesty = sigma != null || catzoc != null || s44 != null || !!tc;
              return (
                <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap', alignItems: 'stretch' }}>
                  <div style={{ flex: '0 0 auto', padding: '12px 14px', background: 'var(--bg-card)', borderRadius: 'var(--radius)', border: '1px solid var(--border-dim)' }}>
                    <DepthColorbar maxD={maxD} height={110} nTicks={6} />
                  </div>
                  {hasHonesty && (
                    <div style={{ flex: '1 1 160px', padding: '12px 14px', background: 'var(--bg-card)', borderRadius: 'var(--radius)', border: '1px solid var(--border-dim)', display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 10 }}>
                      <p style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 800, letterSpacing: '0.1em', margin: 0 }}>HONEST UNCERTAINTY</p>
                      {sigma != null && (
                        <div>
                          <div style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700 }}>σ (per-pixel std)</div>
                          <div style={{ fontSize: 18, fontWeight: 800, color: '#0891b2', fontFamily: 'var(--font-display)' }}>±{Number(sigma).toFixed(2)} <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-dim)' }}>m</span></div>
                        </div>
                      )}
                      {catzoc != null && (
                        <div>
                          <div style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700 }}>CATZOC (IHO S-57)</div>
                          <div style={{ fontSize: 13, fontWeight: 800, color: '#2563eb', fontFamily: 'var(--font-mono)' }}>{catzoc}</div>
                        </div>
                      )}
                      {s44 != null && (
                        <div>
                          <div style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700 }}>IHO S-44 Order 1A pass</div>
                          <div style={{ fontSize: 13, fontWeight: 800, color: s44 >= 90 ? '#059669' : s44 >= 70 ? '#0284c7' : '#b45309', fontFamily: 'var(--font-mono)' }}>{Number(s44).toFixed(0)}%</div>
                        </div>
                      )}
                      {tc && (
                        <div>
                          <div style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, marginBottom: 3 }}>Tidal reduction</div>
                          <TideChip tc={tc} variant="full" />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* ── Multi-scene MLE result group (ADPorts F1) ── */}
            {results.stability && !results.stability.error && (
              <MleSceneGroup results={results} onShowOverlay={onShowOverlay} activeOverlayKey={activeOverlayKey} />
            )}

            {/* Calibration info — INTERNAL ONLY (data-source + accuracy) */}
            {INTERNAL_ANALYSIS && results.ml_stats?.calibration && (
              <div style={{ padding: '12px', background: 'rgba(45,212,191,0.05)', borderRadius: 'var(--radius)', border: '1px solid rgba(45,212,191,0.15)', marginBottom: '16px' }}>
                <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '8px' }}>GEBCO CALIBRATION</p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '6px', fontSize: '10px', fontFamily: 'var(--font-mono)' }}>
                  <div><span style={{ color: 'var(--text-dim)' }}>R²: </span><span style={{ color: 'var(--accent-teal)', fontWeight: 700 }}>{results.ml_stats.calibration.r2?.toFixed(3)}</span></div>
                  <div><span style={{ color: 'var(--text-dim)' }}>RMSE before: </span><span>{results.ml_stats.calibration.rmse_before?.toFixed(2)}m</span></div>
                  <div><span style={{ color: 'var(--text-dim)' }}>RMSE after: </span><span style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>{results.ml_stats.calibration.rmse_after?.toFixed(2)}m</span></div>
                </div>
              </div>
            )}

            {/* S2Shores — wave-dispersion physics — INTERNAL ONLY (method disclosure) */}
            {INTERNAL_ANALYSIS && results.s2shores && (() => {
              const s = results.s2shores;
              const ml = results.ml_stats || {};
              return (
                <div style={{ padding: '14px', background: 'rgba(6,182,212,0.05)', borderRadius: 'var(--radius)', border: '1px solid rgba(6,182,212,0.25)', marginBottom: '16px' }}>
                  <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: '#0e7490', fontWeight: 800, letterSpacing: '0.1em', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: 6 }}>
                    🌊 S2SHORES · WAVE-DISPERSION PHYSICS
                    <span style={{ fontSize: 8, color: 'var(--text-dim)', fontWeight: 600 }}>Almar et al. 2024 · CNES</span>
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '8px', fontSize: '10px', fontFamily: 'var(--font-mono)', marginBottom: '10px' }}>
                    <div><span style={{ color: 'var(--text-dim)' }}>res: </span><span style={{ color: '#0e7490', fontWeight: 700 }}>{s.resolution_m}m</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>window: </span><span style={{ fontWeight: 700 }}>{s.window_m}m</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>step: </span><span style={{ fontWeight: 700 }}>{s.step_m}m</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>src: </span>
                      <span style={{ fontSize: 8, padding: '0 6px', borderRadius: 3,
                        background: s.source?.startsWith('GEE') ? 'rgba(34,197,94,0.15)' : 'rgba(2,132,199,0.12)',
                        color: s.source?.startsWith('GEE') ? '#15803d' : '#0369a1', fontWeight: 700 }}>{s.source}</span>
                    </div>
                    <div><span style={{ color: 'var(--text-dim)' }}>valid: </span><span style={{ fontWeight: 700 }}>{s.valid_windows}/{s.total_windows}</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>λ̄: </span><span style={{ fontWeight: 700 }}>{s.mean_wavelength_m}m</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>c̄: </span><span style={{ fontWeight: 700 }}>{s.mean_celerity_m_s}m/s</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>T̄: </span><span style={{ fontWeight: 700 }}>{s.mean_period_s}s</span></div>
                  </div>
                  {s.scene_id && (
                    <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', borderTop: '1px dashed rgba(6,182,212,0.2)', paddingTop: 6 }}>
                      scene: <span style={{ color: '#0e7490' }}>{s.scene_id}</span>
                      {s.scene_time_ms && <span style={{ marginLeft: 8 }}>t={new Date(s.scene_time_ms).toISOString().slice(0, 16).replace('T', ' ')}Z</span>}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* CBR — Cluster-Based Regression — INTERNAL ONLY (method + accuracy) */}
            {INTERNAL_ANALYSIS && results.cbr && (() => {
              const cbr = results.cbr;
              const m = cbr.metrics || {};
              const classes = cbr.per_class || [];
              return (
                <div style={{ padding: '14px', background: 'rgba(234,88,12,0.05)', borderRadius: 'var(--radius)', border: '1px solid rgba(234,88,12,0.2)', marginBottom: '16px' }}>
                  <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: '#c2410c', fontWeight: 800, letterSpacing: '0.1em', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: 6 }}>
                    🪸 CBR · CLUSTER-BASED REGRESSION
                    <span style={{ fontSize: 8, color: 'var(--text-dim)', fontWeight: 600, letterSpacing: '0.05em' }}>Geyman &amp; Maloof 2019</span>
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '8px', fontSize: '10px', fontFamily: 'var(--font-mono)', marginBottom: '10px' }}>
                    <div><span style={{ color: 'var(--text-dim)' }}>K: </span><span style={{ color: '#c2410c', fontWeight: 700 }}>{cbr.n_clusters}</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>R²: </span><span style={{ color: '#c2410c', fontWeight: 700 }}>{m.r2}</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>RMSE: </span><span style={{ color: '#dc2626', fontWeight: 700 }}>{m.rmse}m</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>MAE: </span><span style={{ color: '#059669', fontWeight: 700 }}>{m.mae}m</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>Bias: </span><span style={{ fontWeight: 600 }}>{m.bias}m</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>N train: </span><span style={{ fontWeight: 600 }}>{m.n_train}</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>User: </span><span style={{ fontWeight: 600 }}>{cbr.n_user_points || 0}</span></div>
                    <div><span style={{ color: 'var(--text-dim)' }}>ICESat-2: </span><span style={{ fontWeight: 600 }}>{cbr.n_icesat2 || 0}</span></div>
                  </div>
                  {m.iho_s44 && (
                    <div style={{ padding: '6px 10px', borderRadius: 6, background: 'rgba(5,150,105,0.08)', border: '1px solid rgba(5,150,105,0.2)', fontSize: 10, fontFamily: 'var(--font-mono)', color: '#047857', fontWeight: 700, marginBottom: 6 }}>
                      IHO: {m.iho_s44}
                    </div>
                  )}
                  {cbr.mle && (
                    <div style={{ padding: '8px 10px', borderRadius: 6, background: 'rgba(162,28,175,0.06)', border: '1px solid rgba(162,28,175,0.2)', marginBottom: 6 }}>
                      <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#a21caf', fontWeight: 700, marginBottom: 4 }}>
                        MLE FUSION · {cbr.mle.n_scenes_used}/{cbr.mle.n_scenes_requested} scenes · mean coverage {cbr.mle.mean_coverage_per_px?.toFixed(2)} scenes/px
                        {cbr.mle.n_rescued > 0 && <span style={{ color: '#b45309', marginLeft: 8 }}>· nanmean-rescued {cbr.mle.n_rescued.toLocaleString()} px</span>}
                      </div>
                      {Array.isArray(cbr.per_scene) && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto auto auto', gap: '2px 8px', fontSize: 9, fontFamily: 'var(--font-mono)' }}>
                          <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>date range</span>
                          <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>src</span>
                          <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>R²</span>
                          <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>RMSE</span>
                          <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>env</span>
                          {cbr.per_scene.map((ps, i) => (
                            <React.Fragment key={i}>
                              <span style={{ color: ps.status === 'ok' ? 'var(--text-primary)' : '#b45309' }}>{ps.date_range}</span>
                              <span style={{ fontSize: 8, padding: '0 4px', borderRadius: 3,
                                background: ps.source === 'GEE' ? 'rgba(34,197,94,0.15)' : 'rgba(2,132,199,0.12)',
                                color: ps.source === 'GEE' ? '#15803d' : '#0369a1', fontWeight: 700, textAlign: 'center' }}>
                                {ps.source || '—'}
                              </span>
                              <span>{ps.status === 'ok' ? ps.r2 : '—'}</span>
                              <span>{ps.status === 'ok' ? `${ps.rmse}m` : ps.status.slice(0, 18)}</span>
                              <span style={{ color: ps.status === 'ok' ? '#c2410c' : '#b45309' }}>{ps.envelope_stage || '—'}</span>
                            </React.Fragment>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  {(m.envelope_stage || m.rw_floor != null) && (
                    <div style={{ padding: '6px 10px', borderRadius: 6, background: 'rgba(234,88,12,0.05)', border: '1px solid rgba(234,88,12,0.15)', fontSize: 9, fontFamily: 'var(--font-mono)', color: '#9a3412', marginBottom: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                      {m.envelope_stage && (
                        <span>envelope: <b style={{color: m.envelope_stage==='strict'?'#047857':m.envelope_stage==='rw-only'?'#b45309':'#c2410c'}}>{m.envelope_stage}</b></span>
                      )}
                      {m.rw_floor != null && <span>Rw floor: <b>{m.rw_floor}</b></span>}
                      {m.n_shallow_pixels != null && m.n_water_pixels != null && (
                        <span>bottom-detectable: <b>{m.n_shallow_pixels.toLocaleString()}</b> / {m.n_water_pixels.toLocaleString()} px ({((m.n_shallow_pixels/Math.max(1,m.n_water_pixels))*100).toFixed(1)}%)</span>
                      )}
                    </div>
                  )}
                  {cbr.retide?.applied && (
                    <div style={{ padding: '6px 10px', borderRadius: 6, background: 'rgba(14,165,233,0.05)', border: '1px solid rgba(14,165,233,0.2)', marginBottom: 8, fontSize: 9, fontFamily: 'var(--font-mono)', color: '#0369a1', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                      <span style={{ fontWeight: 700 }}>⏱ RETIDED TO S2</span>
                      {cbr.retide.s2_reference_utc && <span>t₂={cbr.retide.s2_reference_utc.slice(0, 16).replace('T', ' ')}Z</span>}
                      <span>tide@S2=<b>{cbr.retide.tide_s2_m}m</b></span>
                      <span>n={cbr.retide.n_retided}/{cbr.retide.n_input}</span>
                      <span>median Δ=<b>{cbr.retide.tide_shift_median_m}m</b></span>
                      <span>|Δ|avg=<b>{cbr.retide.tide_shift_abs_mean_m}m</b></span>
                      {cbr.retide.n_wave_corrected > 0 && <span>wave: {cbr.retide.n_wave_corrected} · |Hs/2|avg=<b>{cbr.retide.wave_shift_abs_mean_m}m</b></span>}
                    </div>
                  )}
                  {(m.validation_in_regime || m.validation_full) && (
                    <div style={{ padding: '8px 10px', borderRadius: 6, background: 'rgba(59,130,246,0.05)', border: '1px solid rgba(59,130,246,0.2)', marginBottom: 8 }}>
                      <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#1d4ed8', fontWeight: 700, marginBottom: 4 }}>TWO-PASS VALIDATION</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'auto auto auto auto auto auto', gap: '2px 10px', fontSize: 9, fontFamily: 'var(--font-mono)' }}>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>pass</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>cap</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>R²</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>RMSE</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>MAE</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>N</span>
                        {m.validation_in_regime && (
                          <React.Fragment>
                            <span style={{ color: '#047857', fontWeight: 700 }}>in-regime</span>
                            <span>≤{m.validation_in_regime.depth_cap_m}m</span>
                            <span style={{ fontWeight: 700 }}>{m.validation_in_regime.r2}</span>
                            <span style={{ fontWeight: 700, color: '#047857' }}>{m.validation_in_regime.rmse}m</span>
                            <span>{m.validation_in_regime.mae}m</span>
                            <span>{m.validation_in_regime.n}</span>
                          </React.Fragment>
                        )}
                        {m.validation_full && (
                          <React.Fragment>
                            <span style={{ color: '#b45309' }}>full set</span>
                            <span>≤{m.validation_full.depth_cap_m}m</span>
                            <span>{m.validation_full.r2}</span>
                            <span>{m.validation_full.rmse}m</span>
                            <span>{m.validation_full.mae}m</span>
                            <span>{m.validation_full.n}</span>
                          </React.Fragment>
                        )}
                      </div>
                    </div>
                  )}
                  {(m.features_mode || m.calibration) && (
                    <div style={{ padding: '6px 10px', borderRadius: 6, background: 'rgba(5,150,105,0.05)', border: '1px solid rgba(5,150,105,0.2)', fontSize: 9, fontFamily: 'var(--font-mono)', color: '#065f46', marginBottom: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                      {m.features_mode && <span>features: <b>{m.features_mode}</b></span>}
                      {m.depth_weighting != null && <span>depth-w: <b>{m.depth_weighting ? 'on' : 'off'}</b></span>}
                      {m.calibration?.mode && <span>post-calib: <b>{m.calibration.mode}</b></span>}
                      {m.calibration?.global_shift_m != null && <span>bias shift: <b>{m.calibration.global_shift_m}m</b></span>}
                      {m.calibration?.local_mean_abs_m != null && <span>IDW |Δ|: <b>{m.calibration.local_mean_abs_m}m</b></span>}
                      {m.ref_max_depth_m != null && <span>ref cap: <b>{m.ref_max_depth_m}m</b></span>}
                      {(m.n_ref_raw != null && m.n_ref_after_depth_cap != null) && (
                        <span>refs: {m.n_ref_raw} → cap {m.n_ref_after_depth_cap}{m.n_ref_mad_trimmed>0?` → MAD −${m.n_ref_mad_trimmed}`:''}</span>
                      )}
                    </div>
                  )}
                  {classes.length > 0 && (
                    <div style={{ borderTop: '1px solid rgba(234,88,12,0.15)', paddingTop: 8 }}>
                      <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#9a3412', fontWeight: 700, marginBottom: 6 }}>PER-CLUSTER FIT</p>
                      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr 1fr 1fr auto', gap: '4px 10px', fontSize: 9, fontFamily: 'var(--font-mono)', alignItems: 'center' }}>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>#</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>N</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>RMSE</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>MAE</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>R²</span>
                        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>fit</span>
                        {classes.map(c => (
                          <React.Fragment key={c.id}>
                            <span style={{ color: '#c2410c', fontWeight: 700 }}>{c.id}</span>
                            <span>{c.n_train}</span>
                            <span>{c.rmse}m</span>
                            <span>{c.mae}m</span>
                            <span>{c.r2}</span>
                            <span style={{ fontSize: 8, padding: '1px 6px', borderRadius: 3,
                              background: c.fallback ? 'rgba(250,204,21,0.15)' : 'rgba(5,150,105,0.15)',
                              color: c.fallback ? '#92400e' : '#047857', fontWeight: 700 }}>
                              {c.fallback ? 'global' : 'per-class'}
                            </span>
                          </React.Fragment>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Method 2 results — INTERNAL ONLY (method + accuracy disclosure) */}
            {INTERNAL_ANALYSIS && results.ml_stats?.caballero_stats && (() => {
              const cs = results.ml_stats.caballero_stats;
              const coeff = results.ml_stats.coefficients || {};
              return (
              <div style={{ padding: '14px', background: 'rgba(217,119,6,0.05)', borderRadius: 'var(--radius)', border: '1px solid rgba(217,119,6,0.2)', marginBottom: '16px' }}>
                <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: '#d97706', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '10px' }}>
                  BAND-RATIO METHOD
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', fontSize: '10px', fontFamily: 'var(--font-mono)', marginBottom: '10px' }}>
                  <div><span style={{ color: 'var(--text-dim)' }}>R²: </span><span style={{ color: '#d97706', fontWeight: 700 }}>{cs.r2?.toFixed(3)}</span></div>
                  <div><span style={{ color: 'var(--text-dim)' }}>RMSE: </span><span style={{ color: '#dc2626', fontWeight: 700 }}>{cs.rmse?.toFixed(2)}m</span></div>
                  <div><span style={{ color: 'var(--text-dim)' }}>MedAE: </span><span style={{ color: '#059669', fontWeight: 700 }}>{cs.medae?.toFixed(2)}m</span></div>
                  <div><span style={{ color: 'var(--text-dim)' }}>Bias: </span><span style={{ fontWeight: 600 }}>{cs.bias?.toFixed(2)}m</span></div>
                  <div><span style={{ color: 'var(--text-dim)' }}>IQR: </span><span style={{ fontWeight: 600 }}>{cs.iqr?.toFixed(2)}m</span></div>
                  <div><span style={{ color: 'var(--text-dim)' }}>N: </span><span style={{ fontWeight: 600 }}>{cs.n_val}</span></div>
                </div>
                <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', borderTop: '1px solid rgba(217,119,6,0.15)', paddingTop: '8px' }}>
                  <p style={{ fontWeight: 700, marginBottom: '4px', color: '#92400e' }}>Depth zone coverage</p>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px' }}>
                    <div>Shallow (&lt;2m): <b>{cs.n_shallow_sdbred?.toLocaleString()}</b>px</div>
                    <div>Blend (2-3.5m): <b>{cs.n_transition?.toLocaleString()}</b>px</div>
                    <div>Deep (&gt;3.5m): <b>{cs.n_deep_sdbgreen?.toLocaleString()}</b>px</div>
                  </div>
                </div>
              </div>
              ); })()}

            {/* Depth zone breakdown */}
            <div style={{
              padding: '14px', background: 'var(--bg-card)', borderRadius: 'var(--radius)',
              border: '1px solid var(--border-subtle)', marginBottom: '16px',
            }}>
              <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '10px' }}>
                DEPTH ZONES
              </p>
              {(() => {
                const zc = results.ml_stats?.zone_counts || results.ml_stats?.sdb_stats?.zc || {};
                const zones = [
                  { label: 'Very Shallow (0-3m)', key: 'vs', color: '#90e0ef' },
                  { label: 'Shallow (3-8m)', key: 'sh', color: '#00b4d8' },
                  { label: 'Moderate (8-15m)', key: 'md', color: '#0077b6' },
                  { label: 'Deep (15-25m)', key: 'dp', color: '#03045e' },
                ];
                const total = zones.reduce((s, z) => s + (zc[z.key] || 0), 0) || 1;
                return zones.map(z => {
                  const count = zc[z.key] || 0;
                  const pct = (count / total) * 100;
                  return (
                    <div key={z.key} style={{ marginBottom: '8px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
                        <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: z.color, fontWeight: 600 }}>{z.label}</span>
                        <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                          {count.toLocaleString()} px ({pct.toFixed(1)}%)
                        </span>
                      </div>
                      <div style={{ height: '4px', background: 'var(--bg-secondary)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ width: `${pct}%`, height: '100%', background: z.color, borderRadius: '2px', transition: 'width 0.5s' }} />
                      </div>
                    </div>
                  );
                });
              })()}

              {/* Source pills hidden in the user-facing result panel — the
                  bathymetry map is presented as a single calibrated depth
                  product, not a list of underlying methods. */}
            </div>

            {/* Depth histogram */}
            <div style={{
              padding: '14px', background: 'var(--bg-card)', borderRadius: 'var(--radius)',
              border: '1px solid var(--border-subtle)', marginBottom: '16px',
            }}>
              <DepthHistogram points={results.points} maxDepth={maxD} />
            </div>

            {/* ICESat-2 acquisition dates — INTERNAL ONLY (data-source disclosure) */}
            {INTERNAL_ANALYSIS && results.icesat2_dates && results.icesat2_dates.length > 0 && (
              <div style={{ padding: '14px', background: 'rgba(2,132,199,0.03)', borderRadius: 'var(--radius)', border: '1px solid rgba(2,132,199,0.15)', marginBottom: '16px' }}>
                <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '8px' }}>
                  🛰 ICESat-2 ACQUISITION DATES (±180 days)
                </p>
                <div style={{ maxHeight: '120px', overflow: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10px', fontFamily: 'var(--font-mono)' }}>
                    <thead><tr style={{ borderBottom: '1px solid rgba(2,132,199,0.15)' }}>
                      <th style={{ padding: '4px 8px', textAlign: 'left', color: 'var(--text-dim)', fontWeight: 600 }}>#</th>
                      <th style={{ padding: '4px 8px', textAlign: 'left', color: 'var(--text-dim)', fontWeight: 600 }}>Date</th>
                      <th style={{ padding: '4px 8px', textAlign: 'left', color: 'var(--text-dim)', fontWeight: 600 }}>Photons</th>
                    </tr></thead>
                    <tbody>{results.icesat2_dates.filter(d => d).map((dt, i) => {
                      const nph = (results.points || []).filter(p => p.acq_date === dt && p.photon_class === 'bathymetry').length;
                      return (
                        <tr key={i} style={{ borderBottom: '1px solid rgba(2,132,199,0.06)' }}>
                          <td style={{ padding: '3px 8px', color: 'var(--text-dim)' }}>{i + 1}</td>
                          <td style={{ padding: '3px 8px', color: 'var(--accent-primary)', fontWeight: 600 }}>{dt}</td>
                          <td style={{ padding: '3px 8px', color: 'var(--text-secondary)' }}>{nph}</td>
                        </tr>
                      );
                    })}</tbody>
                  </table>
                </div>
              </div>
            )}

          </div>
        )}

        {/* ══════ IHO CHART TAB — zoomable canvas with LoD soundings ══════ */}
        {tab === 'iho_chart' && results?.bbox && (
          <IHOChartTab results={results} />
        )}

        {/* ══════ ADVANCED TAB — deep stats, calibration, sources, metadata ══════ */}
        {tab === 'advanced' && (
          <AdvancedTab results={results} validationResults={validationResults} />
        )}

        {/* ══════ CONTOUR MAP + PROFESSIONAL ESTIMATION ══════ */}
        {tab === 'contour' && results?.bbox && (
          <div>
            {/* Contour Map */}
            <div style={{
              padding: '8px', background: '#fff', borderRadius: '12px',
              border: '1px solid var(--border-dim)', boxShadow: 'var(--shadow-sm)',
              marginBottom: '16px', display: 'flex', justifyContent: 'center',
            }}>
              <ContourMapSVG
                points={results.points}
                contours={results.contours}
                bbox={results.bbox}
                stats={results.stats}
                professional={results.professional}
              />
            </div>

            {/* Contour levels legend */}
            {results.contour_levels && results.contour_levels.length > 0 && (
              <div style={{
                padding: '12px', background: 'var(--bg-card)', borderRadius: 'var(--radius)',
                border: '1px solid var(--border-subtle)', marginBottom: '16px',
              }}>
                <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '8px' }}>
                  ISOBATH LEVELS ({results.contour_levels.length})
                </p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                  {results.contour_levels.map((l, i) => (
                    <span key={i} style={{
                      fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700,
                      padding: '3px 10px', borderRadius: '4px',
                      background: `rgba(0, ${Math.max(0, 150 - l * 6)}, ${Math.max(0, 180 - l * 5)}, 0.1)`,
                      color: `rgb(0, ${Math.max(0, 120 - l * 5)}, ${Math.max(0, 150 - l * 4)})`,
                      border: `1px solid rgba(0, ${Math.max(0, 150 - l * 6)}, ${Math.max(0, 180 - l * 5)}, 0.25)`,
                    }}>{l}m</span>
                  ))}
                </div>
                <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginTop: '6px' }}>
                  {(results.contours || []).length} contour segments generated
                </p>
              </div>
            )}

            {/* Professional Estimation */}
            <ProfessionalEstimation professional={results.professional} />
          </div>
        )}

        {/* ══════ NAUTICAL COMPARISON ══════ */}
        {tab === 'nautical' && results?.bbox && (
          <div>
            <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.08em', marginBottom: '10px' }}>
              NAUTICAL CHART COMPARISON
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', height: 'calc(100vh - 240px)' }}>
              <div style={{ borderRadius: '10px', border: '1px solid var(--border-dim)', overflow: 'hidden', display: 'flex', flexDirection: 'column', boxShadow: 'var(--shadow-sm)' }}>
                <div style={{ padding: '8px 12px', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-dim)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-primary)' }} />
                  <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>OUR CHART (S2 + CNN)</span>
                </div>
                <div style={{ flex: 1, overflow: 'auto', background: '#f0f7ff', padding: '4px', display: 'flex', justifyContent: 'center' }}>
                  <NauticalChartSVG points={results.points} bbox={results.bbox} stats={results.stats} />
                </div>
              </div>
              <div style={{ borderRadius: '10px', border: '1px solid var(--border-dim)', overflow: 'hidden', display: 'flex', flexDirection: 'column', boxShadow: 'var(--shadow-sm)' }}>
                <div style={{ padding: '8px 12px', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-dim)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-amber)' }} />
                  <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>NAUTICAL CHART</span>
                </div>
                <iframe title="Nautical chart" src={`https://fishing-app.gpsnauticalcharts.com/i-boating-fishing-web-app/fishing-marine-charts-navigation.html#14/${((results.bbox.north+results.bbox.south)/2).toFixed(4)}/${((results.bbox.east+results.bbox.west)/2).toFixed(4)}`}
                  style={{ flex: 1, border: 'none', width: '100%' }} />
              </div>
            </div>
          </div>
        )}

        {/* ══════ DIFF TAB ══════ */}
        {tab === 'diff' && hasDiff && (
          <div>
            <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.08em', marginBottom: '10px' }}>
              DEPTH DIFFERENCE (EPOCH A − EPOCH B)
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
              <div style={{ padding: '10px', borderRadius: '8px', background: 'rgba(2,132,199,0.04)', border: '1px solid rgba(2,132,199,0.15)', textAlign: 'center' }}>
                <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600 }}>MEAN Δ</p>
                <p style={{ fontSize: '18px', fontWeight: 800, color: diffResults.mean_diff > 0 ? '#e11d48' : '#059669' }}>{diffResults.mean_diff > 0 ? '+' : ''}{diffResults.mean_diff.toFixed(2)}m</p>
              </div>
              <div style={{ padding: '10px', borderRadius: '8px', background: 'rgba(124,58,237,0.04)', border: '1px solid rgba(124,58,237,0.15)', textAlign: 'center' }}>
                <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600 }}>STD Δ</p>
                <p style={{ fontSize: '18px', fontWeight: 800, color: 'var(--accent-violet)' }}>±{diffResults.std_diff.toFixed(2)}m</p>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '6px', marginBottom: '12px' }}>
              <div style={{ padding: '8px', borderRadius: '6px', background: 'var(--bg-card)', border: '1px solid var(--border-dim)', textAlign: 'center' }}>
                <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>MAX DEEPER</p>
                <p style={{ fontSize: '13px', fontWeight: 700, color: '#e11d48' }}>+{diffResults.max_diff.toFixed(2)}m</p>
              </div>
              <div style={{ padding: '8px', borderRadius: '6px', background: 'var(--bg-card)', border: '1px solid var(--border-dim)', textAlign: 'center' }}>
                <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>MAX SHALLOWER</p>
                <p style={{ fontSize: '13px', fontWeight: 700, color: '#059669' }}>{diffResults.min_diff.toFixed(2)}m</p>
              </div>
              <div style={{ padding: '8px', borderRadius: '6px', background: 'var(--bg-card)', border: '1px solid var(--border-dim)', textAlign: 'center' }}>
                <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>POINTS</p>
                <p style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{diffResults.n_points}</p>
              </div>
            </div>
            {/* Diff color map */}
            <DiffMapSVG points={diffResults.points} bbox={results.bbox} stats={diffResults} />
            {/* Legend */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', marginTop: '8px', fontSize: '9px', fontFamily: 'var(--font-mono)' }}>
              <span style={{ color: '#059669' }}>■ Shallower (A&lt;B)</span>
              <span style={{ color: '#94a3b8' }}>■ No change</span>
              <span style={{ color: '#e11d48' }}>■ Deeper (A&gt;B)</span>
            </div>
          </div>
        )}

        {/* ══════ VALIDATION TAB ══════ */}
        {tab === 'validation' && hasValError && (
          <div style={{ padding: 20, background: 'rgba(225,29,72,0.04)', border: '1px dashed rgba(225,29,72,0.3)', borderRadius: 10 }}>
            <p style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: '#be123c', fontWeight: 700, marginBottom: 8 }}>VALIDATION COULD NOT RUN</p>
            <p style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: '#64748b', lineHeight: 1.6 }}>
              {validationResults.error || 'No predicted/observed pairs matched within the 50 m search radius.'}<br/>
              Make sure the uploaded XYZ and the current ROI overlap, and that you ran Extract / Quick Analyse first.
            </p>
          </div>
        )}
        {tab === 'validation' && hasVal && (
          <div>
            <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.08em', marginBottom: '10px' }}>
              IN-SITU VALIDATION — PREDICTED vs OBSERVED {validationResults.method ? `· ${validationResults.method}` : ''}
            </p>
            {/* Headline verdict: CATZOC tier + IHO order + pass criterion (R3),
                shown before the per-order strip and the stat cards. */}
            {(validationResults.catzoc || validationResults.order_label) && (() => {
              const cz = (validationResults.catzoc || '').toUpperCase();
              const vc = cz.startsWith('A') ? '#059669' : cz.startsWith('B') ? '#0284c7' : cz.startsWith('C') ? '#d97706' : cz.startsWith('D') ? '#e11d48' : '#475569';
              return (
                <div style={{ display: 'flex', alignItems: 'stretch', gap: 12, padding: '12px 14px', borderRadius: 10, marginBottom: 12, background: `${vc}0d`, border: `1.5px solid ${vc}44` }}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minWidth: 58, padding: '6px 10px', borderRadius: 8, background: `${vc}1a`, border: `1px solid ${vc}55` }}>
                    <span style={{ fontSize: 7, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.06em' }}>CATZOC</span>
                    <span style={{ fontSize: 24, fontWeight: 800, color: vc, lineHeight: 1.05 }}>{validationResults.catzoc || '—'}</span>
                  </div>
                  <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    {validationResults.order_label && (
                      <p style={{ fontSize: 13, fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>{validationResults.order_label}</p>
                    )}
                    <p style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', margin: '3px 0 0', lineHeight: 1.5, whiteSpace: 'normal', wordBreak: 'break-word' }}>
                      {validationResults.pass_criterion || 'p95 ≤ TVU (IHO S-44 §3.3.1)'} · vertical-accuracy assessment only
                      {/* ADPorts F3 — datum suffix, honest either way */}
                      {results?.tide_correction && (
                        results.tide_correction.applied
                          ? ` · reduced to ${results.tide_correction.datum_after || 'MSL'} (${results.tide_correction.source})`
                          : ' · UNCORRECTED — instantaneous sea level'
                      )}
                    </p>
                  </div>
                </div>
              );
            })()}
            {/* IHO S-44 multi-order compliance strip */}
            {validationResults.iho_orders && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, marginBottom: 10 }}>
                {[
                  { k: 'special', label: 'S-44 SPECIAL', sub: 'a=0.25 b=0.0075' },
                  { k: 'order1a', label: 'S-44 ORDER 1a',  sub: 'a=0.5 b=0.013' },
                  { k: 'order2',  label: 'S-44 ORDER 2',   sub: 'a=1.0 b=0.023' },
                ].map(o => {
                  const r = validationResults.iho_orders[o.k];
                  const p = r?.pass_pct ?? 0;
                  const c = p >= 95 ? '#059669' : p >= 80 ? '#d97706' : '#e11d48';
                  return (
                    <div key={o.k} style={{ padding: 8, borderRadius: 8, background: 'var(--bg-card)', border: `1.5px solid ${c}33`, textAlign: 'center' }}>
                      <p style={{ fontSize: 7, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700 }}>{o.label}</p>
                      <p style={{ fontSize: 18, fontWeight: 800, color: c }}>{p.toFixed(1)}%</p>
                      <p style={{ fontSize: 7, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>{r?.n_pass}/{(r?.n_pass || 0) + (r?.n_fail || 0)} · {o.sub}</p>
                    </div>
                  );
                })}
              </div>
            )}
            {/* CATZOC verdict shown in the headline banner above; detection-
                capability + TPU caveats render as footnotes below the numbers. */}
            {/* Stats cards */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '6px', marginBottom: '12px' }}>
              {[['RMSE', validationResults.rmse, 'm', '#e11d48'],['MAE', validationResults.mae, 'm', '#d97706'],['MedAE', validationResults.medae, 'm', '#b45309'],
                ['Bias', validationResults.bias, 'm', '#7c3aed'],['Std', validationResults.std, 'm', '#6366f1'],['R²', validationResults.r2, '', '#0284c7'],
                ['Pearson r', validationResults.pearson_r, '', '#0891b2'],['MAPE', validationResults.mape_pct, '%', '#0d9488'],['Pairs', validationResults.n_pairs, '', '#475569']
              ].filter(r => r[1] !== undefined).map(([label, val, unit, color], i) => (
                <div key={i} style={{ padding: '8px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--border-dim)', textAlign: 'center' }}>
                  <p style={{ fontSize: '7px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 600 }}>{label}</p>
                  <p style={{ fontSize: '15px', fontWeight: 800, color }}>{typeof val === 'number' ? val.toFixed(label === 'R²' ? 3 : label === 'Pairs' ? 0 : 2) : val}{unit}</p>
                </div>
              ))}
            </div>
            {/* ── Caveats & disclosures (footnotes, beneath the numbers) ── */}
            {(() => {
              const io = validationResults.iho_orders || {};
              const specialPct = io.special?.pass_pct ?? 0;
              const order1aPct = io.order1a?.pass_pct ?? 0;
              const showDetection = specialPct >= 95 || order1aPct >= 95;
              const md = validationResults.match_distance_stats_m;
              const hasMd = md && (md.mean != null || md.p95 != null || md.max != null);
              const tc = results?.tide_correction;
              if (!showDetection && !validationResults.tpu_caveat && !hasMd && !tc) return null;
              const resTxt = validationResults.resolution_m != null ? `${validationResults.resolution_m} m` : 'the selected';
              const note = validationResults.detection_capability_note ||
                (`TVU pass-rate only; Special Order / Order 1a additionally require full-seafloor `
                  + `object-detection capability (cubic features ≥1 m / ≥2 m, S-44 Table 1) that a `
                  + `${resTxt} gridded SDB product cannot demonstrate — treat any 'Special'/'1a' badge `
                  + `above as vertical-accuracy-only, not full IHO Order compliance.`);
              const fmt = (v) => (typeof v === 'number' ? v.toFixed(1) : v);
              return (
                <div style={{ marginBottom: 12, padding: '10px 12px', borderRadius: 8, background: 'var(--bg-card)', border: '1px solid var(--border-dim)' }}>
                  <p style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.08em', margin: '0 0 7px' }}>DISCLOSURES</p>
                  {showDetection && (
                    <div data-testid="detection-capability-note" style={{ display: 'flex', gap: 7, marginBottom: (validationResults.tpu_caveat || hasMd) ? 8 : 0 }}>
                      <span style={{ color: '#b45309', fontSize: 10, lineHeight: 1.4 }}>▲</span>
                      <p style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: '#92400e', lineHeight: 1.55, margin: 0, whiteSpace: 'normal', wordBreak: 'break-word' }}>{note}</p>
                    </div>
                  )}
                  {validationResults.tpu_caveat && (
                    <div style={{ display: 'flex', gap: 7, marginBottom: hasMd ? 8 : 0 }}>
                      <span style={{ color: 'var(--text-dim)', fontSize: 10, lineHeight: 1.4 }}>ⓘ</span>
                      <p style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', lineHeight: 1.55, margin: 0, whiteSpace: 'normal', wordBreak: 'break-word' }}>{validationResults.tpu_caveat}</p>
                    </div>
                  )}
                  {hasMd && (
                    <div style={{ display: 'flex', gap: 7, marginBottom: tc ? 8 : 0 }}>
                      <span style={{ color: 'var(--text-dim)', fontSize: 10, lineHeight: 1.4 }}>⌖</span>
                      <p style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', lineHeight: 1.55, margin: 0 }}>
                        Predicted↔observed match distance: mean {fmt(md.mean)} m · p95 {fmt(md.p95)} m · max {fmt(md.max)} m
                        {validationResults.n_excluded_by_distance ? ` · ${validationResults.n_excluded_by_distance} pair(s) excluded beyond radius` : ''}.
                      </p>
                    </div>
                  )}
                  {/* ADPorts F3 — tidal-correction disclosure row (same styling as tpu_caveat/amber-warning) */}
                  {tc && (
                    <div style={{ display: 'flex', gap: 7 }}>
                      <span style={{ color: tc.applied ? 'var(--text-dim)' : '#b45309', fontSize: 10, lineHeight: 1.4 }}>{tc.applied ? 'ⓘ' : '▲'}</span>
                      <p style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: tc.applied ? 'var(--text-dim)' : '#92400e', lineHeight: 1.55, margin: 0, whiteSpace: 'normal', wordBreak: 'break-word' }}>{tc.note}</p>
                    </div>
                  )}
                </div>
              );
            })()}
            {/* Scatter plot: Observed vs Predicted */}
            <svg width={380} height={280} style={{ background: '#fff', borderRadius: '10px', border: '1px solid #e2e8f0', marginBottom: '12px' }}>
              <text x={190} y={16} textAnchor="middle" fontSize="10" fontWeight="700" fill="#0f172a" fontFamily="Plus Jakarta Sans">Observed vs Predicted Depth</text>
              {(() => {
                const pad = { t: 28, r: 16, b: 36, l: 44 };
                const pw = 380 - pad.l - pad.r, ph = 280 - pad.t - pad.b;
                const pairs = validationResults.pairs || [];
                const maxVal = Math.max(...pairs.map(p => Math.max(p.obs_depth, p.pred_depth)), 1) * 1.1;
                return (<>
                  {/* 1:1 line */}
                  <line x1={pad.l} y1={pad.t + ph} x2={pad.l + pw} y2={pad.t} stroke="#cbd5e1" strokeWidth="1" strokeDasharray="4,4" />
                  {/* Axes */}
                  <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + ph} stroke="#94a3b8" strokeWidth="1" />
                  <line x1={pad.l} y1={pad.t + ph} x2={pad.l + pw} y2={pad.t + ph} stroke="#94a3b8" strokeWidth="1" />
                  <text x={190} y={275} textAnchor="middle" fontSize="8" fill="#64748b" fontFamily="JetBrains Mono">Observed (m)</text>
                  <text x={10} y={pad.t + ph / 2} textAnchor="middle" fontSize="8" fill="#64748b" fontFamily="JetBrains Mono" transform={`rotate(-90,10,${pad.t + ph / 2})`}>Predicted (m)</text>
                  {[0, 0.25, 0.5, 0.75, 1].map((f, i) => {
                    const v = maxVal * f;
                    return (<g key={i}>
                      <text x={pad.l - 4} y={pad.t + ph * (1 - f) + 3} textAnchor="end" fontSize="7" fill="#94a3b8" fontFamily="JetBrains Mono">{v.toFixed(0)}</text>
                      <text x={pad.l + pw * f} y={pad.t + ph + 12} textAnchor="middle" fontSize="7" fill="#94a3b8" fontFamily="JetBrains Mono">{v.toFixed(0)}</text>
                    </g>);
                  })}
                  {/* Points */}
                  {pairs.map((p, i) => {
                    const x = pad.l + (p.obs_depth / maxVal) * pw;
                    const y = pad.t + ph - (p.pred_depth / maxVal) * ph;
                    return <circle key={i} cx={x} cy={y} r={3.5} fill={p.s44_pass ? 'rgba(5,150,105,0.7)' : 'rgba(225,29,72,0.7)'} stroke="#fff" strokeWidth="0.5" />;
                  })}
                </>);
              })()}
            </svg>
            {/* S-44 compliance map */}
            <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, marginBottom: '6px' }}>IHO S-44 ORDER 1 COMPLIANCE</p>
            <div style={{ display: 'flex', gap: '6px', marginBottom: '6px', fontSize: '9px', fontFamily: 'var(--font-mono)' }}>
              <span style={{ color: '#059669' }}>● Pass ({(validationResults.pairs||[]).filter(p => p.s44_pass).length})</span>
              <span style={{ color: '#e11d48' }}>● Fail ({(validationResults.pairs||[]).filter(p => !p.s44_pass).length})</span>
              <span style={{ color: '#64748b' }}>TVU = √(0.5² + (0.013×d)²)</span>
            </div>
            {/* Residual histogram */}
            {validationResults.residual_histogram && validationResults.residual_histogram.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <p style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, marginBottom: 6 }}>
                  RESIDUAL DISTRIBUTION (pred − obs)
                </p>
                <svg width={380} height={120} style={{ background: '#fff', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                  {(() => {
                    const bins = validationResults.residual_histogram;
                    const maxC = Math.max(...bins.map(b => b.count), 1);
                    const pad = { t: 10, r: 12, b: 22, l: 24 };
                    const pw = 380 - pad.l - pad.r, ph = 120 - pad.t - pad.b;
                    const bw = pw / bins.length;
                    return (<>
                      <line x1={pad.l} y1={pad.t + ph} x2={pad.l + pw} y2={pad.t + ph} stroke="#94a3b8" strokeWidth="1" />
                      {/* zero line */}
                      {(() => {
                        const xs = bins.map(b => b.x);
                        const xmin = xs[0], xmax = xs[xs.length - 1];
                        const z = pad.l + (0 - xmin) / (xmax - xmin + 1e-10) * pw;
                        return <line x1={z} y1={pad.t} x2={z} y2={pad.t + ph} stroke="#ef4444" strokeDasharray="3,3" strokeWidth="0.8" />;
                      })()}
                      {bins.map((b, i) => {
                        const h = (b.count / maxC) * ph;
                        const x = pad.l + i * bw;
                        const y = pad.t + ph - h;
                        const c = b.x < 0 ? '#7c3aed' : b.x > 0 ? '#d97706' : '#059669';
                        return <rect key={i} x={x} y={y} width={bw - 0.5} height={h} fill={c} fillOpacity={0.7} />;
                      })}
                      {[0, Math.floor(bins.length / 2), bins.length - 1].map(i => (
                        <text key={i} x={pad.l + i * bw + bw / 2} y={pad.t + ph + 12} textAnchor="middle" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">{bins[i].x.toFixed(1)}m</text>
                      ))}
                      <text x={pad.l - 2} y={pad.t + 6} textAnchor="end" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">{maxC}</text>
                      <text x={pad.l - 2} y={pad.t + ph} textAnchor="end" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">0</text>
                    </>);
                  })()}
                </svg>
              </div>
            )}
            {/* Depth-stratified stats */}
            {validationResults.stratified && validationResults.stratified.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <p style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, marginBottom: 6 }}>
                  DEPTH-STRATIFIED STATS (by observed depth class)
                </p>
                <div style={{ borderRadius: 8, border: '1px solid #e2e8f0', overflow: 'hidden' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 9, fontFamily: 'JetBrains Mono' }}>
                    <thead><tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                      {['Range (m)', 'N', 'RMSE', 'MAE', 'Bias', 'S-44 1a'].map(h => (
                        <th key={h} style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b', fontWeight: 700 }}>{h}</th>
                      ))}
                    </tr></thead>
                    <tbody>
                      {validationResults.stratified.map((s, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '4px 8px', fontWeight: 700, color: '#0f172a' }}>{s.range}</td>
                          <td style={{ padding: '4px 8px' }}>{s.n}</td>
                          <td style={{ padding: '4px 8px', color: '#e11d48', fontWeight: 600 }}>{s.rmse}</td>
                          <td style={{ padding: '4px 8px', color: '#d97706' }}>{s.mae}</td>
                          <td style={{ padding: '4px 8px', color: s.bias >= 0 ? '#7c3aed' : '#0284c7' }}>{s.bias >= 0 ? '+' : ''}{s.bias}</td>
                          <td style={{ padding: '4px 8px', color: s.pass_1a >= 95 ? '#059669' : s.pass_1a >= 80 ? '#d97706' : '#e11d48', fontWeight: 700 }}>{s.pass_1a}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            {/* IHO class confusion matrix */}
            {validationResults.class_confusion && (
              <div style={{ marginBottom: 12 }}>
                <p style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, marginBottom: 6 }}>
                  IHO S-52 DEPTH-CLASS AGREEMENT · {validationResults.class_confusion.agreement_pct}% MATCH
                </p>
                <div style={{ borderRadius: 8, border: '1px solid #e2e8f0', overflow: 'auto' }}>
                  <table style={{ borderCollapse: 'collapse', fontSize: 9, fontFamily: 'JetBrains Mono', margin: '0 auto' }}>
                    <thead><tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                      <th style={{ padding: '4px 6px', color: '#64748b' }}>obs ↓ / pred →</th>
                      {validationResults.class_confusion.labels.map(l => (
                        <th key={l} style={{ padding: '4px 6px', color: '#64748b', fontWeight: 700 }}>{l}</th>
                      ))}
                    </tr></thead>
                    <tbody>
                      {validationResults.class_confusion.matrix.map((row, i) => {
                        const rowSum = row.reduce((a, b) => a + b, 0) || 1;
                        return (
                          <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                            <td style={{ padding: '4px 6px', fontWeight: 700, background: '#f8fafc' }}>{validationResults.class_confusion.labels[i]}</td>
                            {row.map((v, j) => {
                              const f = v / rowSum;
                              return (
                                <td key={j} style={{ padding: '4px 6px', textAlign: 'center', background: i === j ? `rgba(5,150,105,${0.1 + f * 0.6})` : f > 0 ? `rgba(225,29,72,${0.08 + f * 0.4})` : 'transparent', color: i === j ? '#065f46' : (v > 0 ? '#9f1239' : '#94a3b8'), fontWeight: i === j ? 700 : 500 }}>{v}</td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            {/* Pair table */}
            <div style={{ maxHeight: '180px', overflow: 'auto', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '9px', fontFamily: 'JetBrains Mono' }}>
                <thead><tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0 }}>
                  {['Lat', 'Lon', 'Obs(m)', 'Pred(m)', 'Δ(m)', 'S-44'].map(h => (
                    <th key={h} style={{ padding: '4px 6px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr></thead>
                <tbody>{(validationResults.pairs || []).slice(0, 500).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: p.s44_pass ? 'rgba(5,150,105,0.03)' : 'rgba(225,29,72,0.03)' }}>
                    <td style={{ padding: '3px 6px' }}>{p.lat.toFixed(4)}</td>
                    <td style={{ padding: '3px 6px' }}>{p.lon.toFixed(4)}</td>
                    <td style={{ padding: '3px 6px' }}>{p.obs_depth}</td>
                    <td style={{ padding: '3px 6px' }}>{p.pred_depth}</td>
                    <td style={{ padding: '3px 6px', color: Math.abs(p.diff) > 1 ? '#e11d48' : '#059669', fontWeight: 600 }}>{p.diff > 0 ? '+' : ''}{p.diff}</td>
                    <td style={{ padding: '3px 6px', color: p.s44_pass ? '#059669' : '#e11d48', fontWeight: 700 }}>{p.s44_pass ? '✓' : '✗'}</td>
                  </tr>
                ))}</tbody>
              </table>
              {(validationResults.pairs || []).length > 500 && (
                <p style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', padding: '6px', textAlign: 'center' }}>
                  Showing 500 of {validationResults.pairs.length} pairs
                </p>
              )}
            </div>
          </div>
        )}

        {/* ══════ PROFILES TAB ══════ */}
        {tab === 'profiles' && (
          <div>
            <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '12px' }}>
              ALONG-TRACK DEPTH PROFILES
            </p>
            {(results.bath_profiles || []).map((bp, i) => (
              <DepthProfile key={i} profile={bp.profile} title={bp.track} color="#38bdf8" />
            ))}
            {(results.sea_profiles || []).map((sp, i) => (
              <DepthProfile key={`sea-${i}`} profile={sp.profile} title={`Sea Surface — ${sp.track}`} color="#34d399" />
            ))}

            {results.tracks && (
              <div style={{ padding: '12px', background: 'var(--bg-card)', borderRadius: 'var(--radius)', border: '1px solid var(--border-subtle)', marginTop: '12px' }}>
                <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '8px' }}>
                  TRACK SUMMARY
                </p>
                <table style={{ width: '100%', fontSize: '9px', fontFamily: 'var(--font-mono)', borderCollapse: 'collapse' }}>
                  <thead><tr style={{ color: 'var(--text-dim)' }}>
                    <th style={{ textAlign: 'left', padding: '4px 0', fontWeight: 600 }}>Track</th>
                    <th style={{ textAlign: 'right', padding: '4px 0', fontWeight: 600 }}>Bathy</th>
                    <th style={{ textAlign: 'right', padding: '4px 0', fontWeight: 600 }}>Mean</th>
                    <th style={{ textAlign: 'right', padding: '4px 0', fontWeight: 600 }}>Max</th>
                  </tr></thead>
                  <tbody>
                    {results.tracks.filter(t => !t.error).map((t, i) => (
                      <tr key={i} style={{ color: 'var(--text-secondary)', borderTop: '1px solid var(--border-subtle)' }}>
                        <td style={{ padding: '3px 0', maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.track_id}</td>
                        <td style={{ textAlign: 'right', padding: '3px 0' }}>{t.n_bathy}</td>
                        <td style={{ textAlign: 'right', padding: '3px 0', color: 'var(--accent-primary)' }}>{t.mean_depth?.toFixed(1)}m</td>
                        <td style={{ textAlign: 'right', padding: '3px 0', color: 'var(--accent-blue)' }}>{t.max_depth?.toFixed(1)}m</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ══════ DATA TAB ══════ */}
        {tab === 'data' && (
          <div>
            <p style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.1em', marginBottom: '10px' }}>
              CLASSIFIED BATHYMETRY PHOTONS
            </p>
            <div style={{ maxHeight: '400px', overflow: 'auto', borderRadius: 'var(--radius)', border: '1px solid var(--border-dim)' }}>
              <table style={{ width: '100%', fontSize: '9px', fontFamily: 'var(--font-mono)', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-card)', color: 'var(--text-dim)', position: 'sticky', top: 0 }}>
                    <th style={{ textAlign: 'left', padding: '7px 6px', fontWeight: 600 }}>Lat</th>
                    <th style={{ textAlign: 'left', padding: '7px 6px', fontWeight: 600 }}>Lon</th>
                    <th style={{ textAlign: 'right', padding: '7px 6px', fontWeight: 600 }}>Depth</th>
                    <th style={{ textAlign: 'right', padding: '7px 6px', fontWeight: 600 }}>Height</th>
                  </tr>
                </thead>
                <tbody>
                  {(results.points || []).filter(p => ['bathymetry','interpolated'].includes(p.photon_class)).slice(0, 200).map((p, i) => (
                    <tr key={i} style={{ color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '4px 6px' }}>{p.lat.toFixed(5)}</td>
                      <td style={{ padding: '4px 6px' }}>{p.lon.toFixed(5)}</td>
                      <td style={{ textAlign: 'right', padding: '4px 6px', color: 'var(--accent-primary)', fontWeight: 600 }}>{p.depth.toFixed(2)}m</td>
                      <td style={{ textAlign: 'right', padding: '4px 6px' }}>{p.height?.toFixed(2) || '—'}m</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Download all TIFFs (.zip) — a re-opened persisted result carries
                its download-all endpoint; streams every raster it references. */}
            {results._bathy_result?.download_all_url && (
              <a href={results._bathy_result.download_all_url} target="_blank" rel="noreferrer"
                title="Download EVERY GeoTIFF this result references in one .zip (with a MANIFEST)"
                style={{
                  display: 'block', textAlign: 'center', marginTop: '14px', padding: '10px',
                  fontSize: '11px', fontFamily: 'var(--font-display)', fontWeight: 800, letterSpacing: '0.04em',
                  borderRadius: 'var(--radius)', textDecoration: 'none',
                  border: '1px solid #0891b2', background: 'rgba(8,145,178,0.14)', color: '#0e7490',
                }}>🗂 Download all TIFFs (.zip)</a>
            )}

            {/* Export buttons */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginTop: '14px' }}>
              {[
                { fmt: 'geojson', label: 'GeoJSON', color: 'var(--accent-primary)' },
                { fmt: 'csv', label: 'CSV', color: 'var(--accent-teal)' },
                { fmt: 'geotiff', label: 'GeoTIFF', color: 'var(--accent-blue)' },
                { fmt: 'netcdf', label: 'NetCDF', color: 'var(--accent-violet)' },
              ].map(e => (
                <button key={e.fmt} onClick={() => onExport(e.fmt)} style={{
                  padding: '10px', fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 700,
                  borderRadius: 'var(--radius)', cursor: 'pointer', letterSpacing: '0.04em',
                  border: `1px solid color-mix(in srgb, ${e.color} 30%, transparent)`,
                  background: `color-mix(in srgb, ${e.color} 6%, transparent)`,
                  color: e.color, transition: 'all 0.15s',
                }}>↓ {e.label}</button>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
