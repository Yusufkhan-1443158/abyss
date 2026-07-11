import React from 'react';
import { INTERNAL_ANALYSIS } from './internalMode';

export default function Header({ view, setView, hasResults, onReset, activeModule, setActiveModule, onOpenHelp, onOpenResults, onOpenVhrPurchase }) {
  // Functional processing-mode selector. IDs/handlers unchanged; the client
  // portal shows neutral labels (no data-source disclosure), internal shows the
  // source-specific labels.
  const modes = INTERNAL_ANALYSIS ? [
    { id: 'gebco_s2', label: 'GEBCO + S2', desc: 'No NASA needed' },
    { id: 'icesat2', label: 'ICESat-2', desc: 'Needs EarthData' },
    { id: 'fusion', label: 'Fusion', desc: 'All sources' },
  ] : [
    { id: 'gebco_s2', label: 'Standard', desc: 'Standard depth processing' },
    { id: 'icesat2', label: 'Enhanced', desc: 'Enhanced depth processing' },
    { id: 'fusion', label: 'Combined', desc: 'Combined depth processing' },
  ];
  const views = [
    { id: 'map', label: 'MAP', desc: 'Interactive map — draw the ROI and view depth layers' },
    { id: '3d', label: '3D', desc: '3D seabed view of the current result' },
    { id: 'split', label: 'SPLIT', desc: 'Map and 3D side by side' },
    { id: 'ascii', label: 'ASCII', desc: 'Text-grid preview of the depth raster' },
    { id: 'monitoring', label: 'MONITOR', desc: 'Change detection — surveys over time, trends and alerts' },
    { id: 'uae', label: 'UAE RESULTS', desc: 'Pre-computed UAE regional results' },
  ];

  return (
    <header style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px', background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-dim)', zIndex: 100 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ width: '30px', height: '30px', borderRadius: '6px', background: '#0f172a', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#e2e8f0', fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: '13px', letterSpacing: '-0.05em' }}>B</div>
        <div>
          <h1 style={{ fontSize: '14px', fontWeight: 700, letterSpacing: '-0.01em', color: '#0f172a' }}>Bathymetry From Satellite</h1>
          <p style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', letterSpacing: '0.08em', fontWeight: 500 }}>{INTERNAL_ANALYSIS ? 'Professional depth mapping · WGS84 · LAT' : 'Professional depth mapping'}</p>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '1px', background: 'var(--bg-primary)', borderRadius: '8px', padding: '3px', border: '1px solid var(--border-subtle)' }}>
        {modes.map(m => (
          <button key={m.id} onClick={() => setActiveModule(m.id)} title={m.desc} style={{
            padding: '5px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer',
            fontFamily: 'var(--font-mono)', fontSize: '9px', fontWeight: 600, letterSpacing: '0.04em',
            background: activeModule === m.id ? 'rgba(56,189,248,0.12)' : 'transparent',
            color: activeModule === m.id ? '#38bdf8' : 'var(--text-dim)',
          }}>{m.label}</button>
        ))}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{ display: 'flex', background: 'var(--bg-primary)', borderRadius: '8px', padding: '3px', border: '1px solid var(--border-subtle)' }}>
          {views.map(v => (
            <button key={v.id} onClick={() => setView(v.id)} title={v.desc} disabled={!['map','monitoring','uae'].includes(v.id) && !hasResults} style={{
              padding: '5px 12px', borderRadius: '6px', border: 'none', cursor: !['map','monitoring','uae'].includes(v.id) && !hasResults ? 'not-allowed' : 'pointer',
              fontFamily: 'var(--font-mono)', fontSize: '9px', fontWeight: 600, opacity: !['map','monitoring','uae'].includes(v.id) && !hasResults ? 0.25 : 1,
              background: view === v.id ? 'linear-gradient(135deg, #38bdf8, #3b82f6)' : 'transparent',
              color: view === v.id ? '#fff' : 'var(--text-dim)',
            }}>{v.label}</button>
          ))}
        </div>
        <button onClick={onOpenResults} title="Open the persistent Results catalogue (queue compute, download & analyse finished jobs)" style={{
          padding: '5px 14px', fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 800,
          borderRadius: '6px', border: '1px solid rgba(37,99,235,0.35)',
          background: 'rgba(37,99,235,0.10)', color: '#1d4ed8', cursor: 'pointer',
          letterSpacing: '0.04em',
        }}>📊 RESULTS</button>
        <button onClick={onOpenVhrPurchase} title="Order a High-Resolution depth product for an area" style={{
          padding: '5px 14px', fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 800,
          borderRadius: '6px', border: '1px solid rgba(14,165,233,0.40)',
          background: 'linear-gradient(135deg, rgba(14,165,233,0.12), rgba(37,99,235,0.10))', color: '#0369a1', cursor: 'pointer',
          letterSpacing: '0.04em',
        }}>🛰️ HIGH-RES</button>
        <button onClick={onOpenHelp} title="Open user guide" style={{
          padding: '5px 12px', fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 700,
          borderRadius: '6px', border: '1px solid var(--border-dim)',
          background: 'var(--bg-primary)', color: 'var(--text-secondary)', cursor: 'pointer',
          letterSpacing: '0.04em',
        }}>Help</button>
        {hasResults && <button onClick={onReset} style={{ padding: '5px 12px', fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 600, borderRadius: '6px', border: '1px solid rgba(251,113,133,0.2)', background: 'rgba(251,113,133,0.06)', color: '#fb7185', cursor: 'pointer' }}>RESET</button>}
        <a href="/ocean/index.html" style={{ padding: '5px 14px', borderRadius: '6px', border: '1px solid rgba(139,92,246,0.3)', background: 'rgba(139,92,246,0.08)', color: '#a78bfa', fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700, letterSpacing: '0.04em', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '6px', transition: 'all 0.2s' }}>
          <span style={{ fontSize: '12px' }}>&#x1F30A;</span> OCEAN MONITOR
        </a>
        <div style={{ padding: '4px 10px', background: 'var(--bg-primary)', borderRadius: '6px', border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: '#34d399', boxShadow: '0 0 6px #34d399' }} />
          <span style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>v4.1 · Pro Extract</span>
        </div>
      </div>
    </header>
  );
}
