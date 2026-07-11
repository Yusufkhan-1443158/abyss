import React from 'react';
import { rampGradientCss, rampTicks } from './depthRamp';

// ─────────────────────────────────────────────────────────────────────────────
// DepthColorbar — a clean, labelled bathymetric colorbar WITH UNITS (m).
// Uses the shared perceptually-uniform ramp (depthRamp.js) so the legend
// matches the 2D map overlay and the 3D seabed surface exactly.
//
// Props:
//   maxD     max depth in metres (defaults 25)
//   height   bar height px (default 120)
//   nTicks   number of labelled ticks incl. endpoints (default 6)
//   title    header label (default "DEPTH")
//   horizontal  render a horizontal bar instead of vertical
// ─────────────────────────────────────────────────────────────────────────────
export default function DepthColorbar({ maxD = 25, height = 120, nTicks = 6, title = 'DEPTH', horizontal = false }) {
  const ticks = rampTicks(maxD, nTicks);

  if (horizontal) {
    return (
      <div style={{ fontFamily: 'JetBrains Mono, monospace' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontSize: 9, fontWeight: 800, color: '#0f172a', letterSpacing: '0.08em' }}>{title}</span>
          <span style={{ fontSize: 8.5, fontWeight: 700, color: '#64748b' }}>metres (MSL)</span>
        </div>
        <div style={{ height: 12, borderRadius: 3, background: rampGradientCss('to right'), border: '1px solid rgba(0,0,0,0.08)' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3, fontSize: 8.5, fontWeight: 600, color: '#475569' }}>
          {ticks.map((t, i) => <span key={i}>{t.depth.toFixed(t.depth < 10 ? 1 : 0)}</span>)}
        </div>
      </div>
    );
  }

  return (
    <div style={{ fontFamily: 'JetBrains Mono, monospace' }}>
      <div style={{ fontSize: 9, fontWeight: 800, color: '#0f172a', letterSpacing: '0.08em', marginBottom: 6 }}>
        {title} <span style={{ fontWeight: 600, color: '#64748b' }}>(m · MSL)</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'stretch', gap: 7 }}>
        <div style={{ width: 16, height, borderRadius: 3, background: rampGradientCss('to bottom'), border: '1px solid rgba(0,0,0,0.08)' }} />
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height, fontSize: 9, fontWeight: 600, color: '#475569' }}>
          {ticks.map((t, i) => (
            <span key={i}>{t.depth.toFixed(t.depth < 10 ? 1 : 0)}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
