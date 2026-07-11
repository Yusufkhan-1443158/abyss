import React, { useEffect, useState, useRef } from 'react';
import HelpIcon from './HelpIcon';
import TideChip from './TideChip';

const API = process.env.REACT_APP_API_URL || '';

// ADPorts F2 — honest Planet VHR backup status panel (Sidebar).
// Self-fetches GET /api/planet-backup/status; renders for ALL client tiers
// (this is read-only provisioning status, not IHO-gated). Never renders a
// thumbnail/image until the backend reports `available:true` with a real
// asset URL — no fake placeholders that could read as imagery.
export default function PlanetBackupPanel({ onShowOverlay }) {
  const [state, setState] = useState({ loading: true, error: null, data: null });
  const pollRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      fetch(`${API}/api/planet-backup/status?site=khalifa_port`)
        .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
        .then(d => { if (!cancelled) setState({ loading: false, error: null, data: d }); })
        .catch(e => { if (!cancelled) setState(s => ({ loading: false, error: String(e), data: s.data })); });
    };
    load();
    return () => { cancelled = true; };
  }, []);

  // Poll every 60s only while imagery is not fully ready yet.
  useEffect(() => {
    const status = state.data?.overall_status;
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (status && status !== 'ready') {
      pollRef.current = setInterval(() => {
        fetch(`${API}/api/planet-backup/status?site=khalifa_port`)
          .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
          .then(d => setState({ loading: false, error: null, data: d }))
          .catch(() => {});
      }, 60000);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [state.data?.overall_status]);

  const { loading, error, data } = state;

  if (loading && !data) {
    return (
      <div style={{ marginBottom: '14px', padding: '10px 12px', borderRadius: 'var(--radius)', border: '1px dashed rgba(2,132,199,0.45)', background: 'rgba(2,132,199,0.04)' }}>
        <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>Loading Planet VHR status…</p>
      </div>
    );
  }
  if (error && !data) {
    return (
      <div style={{ marginBottom: '14px', padding: '10px 12px', borderRadius: 'var(--radius)', border: '1px dashed rgba(2,132,199,0.45)', background: 'rgba(2,132,199,0.04)' }}>
        <p style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>Planet selection manifest unavailable</p>
      </div>
    );
  }
  if (!data) return null;

  const ready = data.overall_status === 'ready';
  const badgeStyle = ready
    ? { background: 'rgba(5,150,105,0.10)', border: '1px solid rgba(5,150,105,0.35)', color: '#047857' }
    : { background: 'rgba(217,119,6,0.10)', border: '1px solid rgba(217,119,6,0.35)', color: '#b45309' };

  return (
    <div style={{ marginBottom: '14px', padding: '12px 14px', borderRadius: 'var(--radius)', border: '1px dashed rgba(2,132,199,0.45)', background: 'rgba(2,132,199,0.04)' }}>
      <p style={{ fontSize: 10.5, fontFamily: 'var(--font-mono)', color: '#0369a1', fontWeight: 800, letterSpacing: '0.06em', margin: '0 0 6px', display: 'flex', alignItems: 'center' }}>
        PLANET VHR — KHALIFA ({data.n_dates || 0} DATES)
        <HelpIcon text="Very-High-Resolution PlanetScope backup imagery, ordered for 5 dates spanning the tidal range. Depths are not re-derived from this imagery — it is a spatial-detail backup only." />
      </p>
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '4px 9px', borderRadius: 'var(--radius-sm)', marginBottom: 8, ...badgeStyle }}>
        <span style={{ fontSize: 11 }}>{ready ? '✓' : '⏳'}</span>
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{data.status_label || data.overall_status}</span>
      </div>
      <div style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', marginBottom: 4 }}>
        Planet quota {data.quota?.account_month_quota_km2 ?? '—'} km²/mo · remaining {data.quota?.remaining_km2 ?? '—'} km²
      </div>
      <div style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginBottom: 8 }}>
        {data.item_type || '—'} · {data.product_bundle || '—'} · AOI {data.aoi?.area_km2 ?? '—'} km²
        {data.generated_utc ? ` · selected ${String(data.generated_utc).slice(0, 10)}` : ''}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {(data.dates || []).map((d, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 6px', borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
            {d.available && d.assets?.thumb_url ? (
              <img src={d.assets.thumb_url} alt={d.date} style={{ width: 36, height: 36, objectFit: 'cover', borderRadius: 4, imageRendering: 'pixelated', border: '1px solid var(--border-dim)' }} />
            ) : (
              <div title="awaiting Planet provisioning" style={{
                width: 36, height: 36, borderRadius: 4, background: 'var(--bg-secondary)',
                border: '1px dashed var(--border-dim)', display: 'flex', alignItems: 'center',
                justifyContent: 'center', fontSize: 14, color: 'var(--text-dim)', flexShrink: 0,
              }}>⏳</div>
            )}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 9.5, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>{d.date}</span>
                {d.tide_m_msl != null && <TideChip variant="compact" height_m={d.tide_m_msl} phase={d.tide_phase} />}
                {d.clear_percent != null && (
                  <span style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', border: '1px solid var(--border-dim)', borderRadius: 'var(--radius-sm)', padding: '1px 5px' }}>
                    {d.clear_percent}% clear
                  </span>
                )}
                <span style={{
                  fontSize: 7.5, fontWeight: 800, letterSpacing: '0.04em', borderRadius: 3, padding: '1px 5px',
                  color: d.available ? '#047857' : '#64748b',
                  background: d.available ? 'rgba(5,150,105,0.12)' : 'rgba(100,116,139,0.12)',
                }}>{d.available ? 'AVAILABLE' : 'AWAITING'}</span>
              </div>
            </div>
            <div>
              {d.available && d.assets?.geotiff_url ? (
                <div style={{ display: 'flex', gap: 4 }}>
                  <button onClick={() => onShowOverlay && d.assets.overlay_png_b64 && onShowOverlay({ png_b64: d.assets.overlay_png_b64, bounds: d.assets.bounds, label: `Planet ${d.date}`, key: `planet_${d.date}` })}
                    disabled={!d.assets.overlay_png_b64}
                    style={{ fontSize: 8, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-blue)', background: 'transparent', border: '1px solid color-mix(in srgb, var(--accent-blue) 30%, transparent)', borderRadius: 'var(--radius-sm)', padding: '2px 6px', cursor: d.assets.overlay_png_b64 ? 'pointer' : 'not-allowed', opacity: d.assets.overlay_png_b64 ? 1 : 0.4 }}>
                    View
                  </button>
                  <a href={d.assets.geotiff_url} target="_blank" rel="noreferrer" style={{ fontSize: 8, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-blue)', border: '1px solid color-mix(in srgb, var(--accent-blue) 30%, transparent)', borderRadius: 'var(--radius-sm)', padding: '2px 6px', textDecoration: 'none' }}>
                    Download
                  </a>
                </div>
              ) : (
                <span style={{ fontSize: 9, color: 'var(--text-dim)' }}>—</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
