import React from 'react';

// ADPorts F3 — shared tidal-correction disclosure chip.
// Renders for ALL client tiers (safety info, never IHO/hydro-gated).
//
// Props:
//   tc       — a `tide_correction` object {applied, height_m, source, datum_after, note, ...}
//   height_m — (compact variant) a raw signed tide height when no full `tc` is available
//   phase    — (compact variant) e.g. "high-rising"
//   variant  — 'full' | 'compact' (default 'full')
export default function TideChip({ tc, height_m, phase, variant = 'full' }) {
  if (variant === 'compact') {
    const h = tc ? tc.height_m : height_m;
    if (h == null) {
      return (
        <span style={{
          fontSize: '8.5px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)',
          borderRadius: 'var(--radius-sm)', padding: '1px 6px', border: '1px solid var(--border-dim)',
        }}>tide —</span>
      );
    }
    const sign = h >= 0 ? '+' : '';
    return (
      <span title={tc?.note || phase || ''} style={{
        fontSize: '8.5px', fontFamily: 'var(--font-mono)', fontWeight: 700,
        color: h >= 0 ? '#0284c7' : '#7c3aed',
        background: h >= 0 ? 'rgba(2,132,199,0.08)' : 'rgba(124,58,237,0.08)',
        border: `1px solid ${h >= 0 ? 'rgba(2,132,199,0.3)' : 'rgba(124,58,237,0.3)'}`,
        borderRadius: 'var(--radius-sm)', padding: '1px 6px',
      }}>
        tide {sign}{Number(h).toFixed(2)} m{phase ? ` · ${phase}` : ''}
      </span>
    );
  }

  // 'full' variant — needs a real tide_correction object.
  if (!tc) return null;
  const applied = !!tc.applied;
  const height = tc.height_m != null ? `${tc.height_m >= 0 ? '+' : ''}${Number(tc.height_m).toFixed(2)} m` : null;

  if (!applied) {
    return (
      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 10px',
        background: 'rgba(217,119,6,0.08)', border: '1px solid rgba(217,119,6,0.35)',
        borderRadius: 'var(--radius-sm)', color: '#b45309',
      }}>
        <span style={{ fontSize: 12 }}>⚠</span>
        <span style={{ fontSize: '9.5px', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.02em' }}>
          UNCORRECTED — INSTANTANEOUS SEA LEVEL
          {height ? <span style={{ fontWeight: 500, marginLeft: 6, color: '#92400e' }}>(tide {height} {tc.source && tc.source !== 'none' ? tc.source : 'n/a'})</span> : null}
        </span>
      </div>
    );
  }

  return (
    <div style={{
      display: 'inline-flex', flexDirection: 'column', gap: 1, padding: '5px 10px',
      background: 'rgba(5,150,105,0.08)', border: '1px solid rgba(5,150,105,0.30)',
      borderRadius: 'var(--radius-sm)', color: '#047857',
    }}>
      <span style={{ fontSize: '9.5px', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.02em' }}>
        REDUCED TO {tc.datum_after || 'MSL'}
      </span>
      <span style={{ fontSize: '8.5px', fontFamily: 'var(--font-mono)', color: '#059669' }}>
        {tc.source || '—'} · h {height || '—'}
      </span>
    </div>
  );
}
