import React from 'react';
import TideChip from './TideChip';

// ADPorts F1 — Multi-Scene MLE per-scene results as a result GROUP.
// Replaces the old flat "Multi-scene MLE stability" table (ResultsPanel.js
// ~1062-1157) with a hero (composite) card + one row per contributing scene,
// each viewable on the map (via onShowOverlay) and individually downloadable.
//
// Props:
//   results          — the MLE result object (top-level: depth_png_b64,
//                       raster_bounds, stability{...}, download_all_zip, ...)
//   onShowOverlay(p) — p = {png_b64, bounds, label, key} | null (null restores composite)
//   activeOverlayKey — 'composite' | scene date string, for row highlighting
export default function MleSceneGroup({ results, onShowOverlay, activeOverlayKey }) {
  const st = results?.stability;
  if (!st || st.error) return null;

  const dls = st.downloads || results.downloads || [];
  const zipUrl = results.download_all_zip || st.download_all_zip;
  const keptLabel = results.scenes_kept_label
    || (st.screening && st.screening.applied
         ? `${st.screening.n_kept} of ${st.screening.n_candidates} scenes kept (glint/turbidity QC)`
         : null);
  const sc = st.shared_cal;
  const wm = results.water_mask_meta || {};
  const bounds = results.raster_bounds;
  const perScene = Array.isArray(st.per_scene) ? st.per_scene : [];
  const threshold = st.agreement_threshold_m ?? st.unstable_threshold_m;

  return (
    <div style={{ padding: '12px 14px', background: 'var(--bg-card)', borderRadius: 'var(--radius)', border: '1px solid var(--border-dim)', marginBottom: '16px' }}>
      <p style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 800, letterSpacing: '0.1em', margin: '0 0 8px' }}>
        MULTI-SCENE MLE RESULT GROUP · {st.n_scenes} SCENES
      </p>

      {/* ── HeroRow (composite) ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10, padding: '8px 10px', marginBottom: 10,
        borderLeft: '3px solid var(--accent-blue)', borderRadius: 6,
        background: (!activeOverlayKey || activeOverlayKey === 'composite') ? 'rgba(2,132,199,0.08)' : 'transparent',
        border: (!activeOverlayKey || activeOverlayKey === 'composite') ? '1px solid rgba(2,132,199,0.35)' : '1px solid transparent',
        borderLeftWidth: 3, borderLeftColor: 'var(--accent-blue)',
      }}>
        {results.depth_png_b64 && (
          <img src={`data:image/png;base64,${results.depth_png_b64}`} alt="composite"
            style={{ width: 56, height: 56, objectFit: 'cover', borderRadius: 4, imageRendering: 'pixelated', border: '1px solid var(--border-dim)' }} />
        )}
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{ fontSize: 10.5, fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>MLE COMPOSITE · {st.n_scenes} SCENES</p>
          {keptLabel && (
            <p style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: '#0369a1', fontWeight: 700, margin: '2px 0 0' }}>
              ⛁ {keptLabel}
              {sc && sc.applied && <span style={{ color: '#a21caf', marginLeft: 8 }}>· shared-cal ON (tide-corrected)</span>}
            </p>
          )}
          <div style={{ display: 'flex', gap: 8, marginTop: 4, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
              median inter-scene σ <b style={{ color: '#0891b2' }}>{st.median_interscene_sigma_m != null ? `${st.median_interscene_sigma_m} m` : '—'}</b>
            </span>
            <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
              unstable <b style={{ color: st.unstable_fraction > 0.25 ? '#b45309' : '#059669' }}>{st.unstable_fraction != null ? `${(st.unstable_fraction * 100).toFixed(1)}%` : '—'}</b>
            </span>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'flex-end' }}>
          {bounds && results.depth_png_b64 && (
            <button onClick={() => onShowOverlay && onShowOverlay(null)} style={{
              fontSize: 9, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-blue)',
              background: 'transparent', border: '1px solid color-mix(in srgb, var(--accent-blue) 30%, transparent)',
              borderRadius: 'var(--radius-sm)', padding: '3px 8px', cursor: 'pointer',
            }}>View on map</button>
          )}
          {zipUrl && (
            <a href={zipUrl} target="_blank" rel="noreferrer" style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#fff', background: 'var(--accent-blue)', fontWeight: 700, textDecoration: 'none', padding: '3px 8px', borderRadius: 'var(--radius-sm)' }}>
              ⬇ Download all (zip)
            </a>
          )}
        </div>
      </div>

      {dls.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
          {dls.map((e, i) => (
            <a key={i} href={e.url} target="_blank" rel="noreferrer" style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)', textDecoration: 'none', padding: '3px 7px', border: '1px solid color-mix(in srgb, var(--accent-blue) 30%, transparent)', borderRadius: 'var(--radius)' }}>↓ {e.label}</a>
          ))}
        </div>
      )}

      {/* ── Per-scene agreement caption + rows ── */}
      {perScene.length > 0 && (
        <>
          <p style={{ fontSize: 8, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.05em', margin: '4px 0 6px' }}>
            PER-SCENE AGREEMENT VS COMPOSITE — ensures the median is not far from any single image
            {threshold != null && <span> · outlier threshold |Δ| &gt; {threshold.toFixed ? threshold.toFixed(2) : threshold} m or r &lt; 0.7</span>}
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 8 }}>
            {perScene.map((s, i) => {
              const isActive = activeOverlayKey === s.date;
              const isOutlier = !!s.outlier;
              return (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: '5px 8px', borderRadius: 6,
                  background: isActive ? 'rgba(2,132,199,0.08)' : 'transparent',
                  border: isActive ? '1px solid rgba(2,132,199,0.35)' : '1px solid var(--border-subtle)',
                  borderLeft: isOutlier ? '2px solid #d97706' : (isActive ? '1px solid rgba(2,132,199,0.35)' : '1px solid var(--border-subtle)'),
                }}>
                  {s.overlay_png_b64 ? (
                    <img src={`data:image/png;base64,${s.overlay_png_b64}`} alt={s.date}
                      style={{ width: 40, height: 40, objectFit: 'cover', borderRadius: 4, imageRendering: 'pixelated', border: '1px solid var(--border-dim)' }} />
                  ) : (
                    <div style={{ width: 40, height: 40, borderRadius: 4, background: 'var(--bg-secondary)', border: '1px dashed var(--border-dim)' }} />
                  )}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 9.5, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>{s.date}</span>
                      {s.tide_m != null && <TideChip variant="compact" height_m={s.tide_m} />}
                      {isOutlier && (
                        <span style={{
                          fontSize: 8, fontWeight: 800, letterSpacing: '0.02em', color: '#92400e',
                          background: 'rgba(180,83,9,0.12)', border: '1px solid rgba(180,83,9,0.4)',
                          borderRadius: 4, padding: '1px 6px',
                        }}>
                          ⚠ OUTLIER · |Δ| {s.rms_dev_from_composite_m != null ? `${s.rms_dev_from_composite_m} m` : '—'}
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: 8, marginTop: 2, flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>|Δ| {s.rms_dev_from_composite_m != null ? `${s.rms_dev_from_composite_m} m` : '—'}</span>
                      <span style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>r {s.corr_vs_composite != null ? s.corr_vs_composite : '—'}</span>
                      <span style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>valid {s.valid_pct != null ? `${s.valid_pct.toFixed ? s.valid_pct.toFixed(1) : s.valid_pct}%` : '—'}</span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 4 }}>
                    {s.overlay_png_b64 && bounds && (
                      <button onClick={() => onShowOverlay && onShowOverlay({ png_b64: s.overlay_png_b64, bounds, label: `Scene ${s.date}`, key: s.date })}
                        style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-blue)', background: 'transparent', border: '1px solid color-mix(in srgb, var(--accent-blue) 30%, transparent)', borderRadius: 'var(--radius-sm)', padding: '2px 6px', cursor: 'pointer' }}>
                        View
                      </button>
                    )}
                    {s.geotiff_url && (
                      <a href={s.geotiff_url} target="_blank" rel="noreferrer"
                        style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-blue)', border: '1px solid color-mix(in srgb, var(--accent-blue) 30%, transparent)', borderRadius: 'var(--radius-sm)', padding: '2px 6px', textDecoration: 'none' }}>
                        Download
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* 3.U2 — mask QA (source honesty + preview overlay) */}
      {(wm.mask_source || wm.mask_preview_png) && (
        <div style={{ padding: '7px 9px', marginBottom: 4, borderRadius: 'var(--radius)', background: 'rgba(245,160,40,0.06)', border: '1px solid rgba(245,160,40,0.25)' }}>
          <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#b45309', fontWeight: 700, marginBottom: 4, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
            <span>MASK QA · {wm.mask_source || '—'}</span>
            {(wm.low_confidence || (wm.osm_coverage_class && wm.osm_coverage_class !== 'rich')) && (
              <span title="OSM coverage inadequate for authoritative masking — the coastline is decided by the AI (GMM) segmentation here."
                style={{ fontSize: 8, fontWeight: 800, letterSpacing: '0.05em', color: '#fff', background: '#b45309', borderRadius: 3, padding: '1px 5px' }}>
                ⚠ OSM {wm.osm_coverage_class || 'low'}
                {wm.osm_land_recall != null && wm.osm_coverage_class === 'sparse'
                  ? ` · land recall ${Math.round(wm.osm_land_recall * 100)}%` : ''}
              </span>
            )}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 10px', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
            {wm.osm_coverage_class != null && <div>OSM coverage: <b>{wm.osm_coverage_class}</b>{wm.osm_land_recall != null ? ` (recall ${Math.round(wm.osm_land_recall * 100)}%)` : ''}</div>}
            {wm.retention_subtidal_pct != null && <div>subtidal retention: <b>{wm.retention_subtidal_pct}%</b></div>}
            {wm.intertidal_excluded_frac != null && <div>intertidal excluded: <b>{(wm.intertidal_excluded_frac * 100).toFixed(2)}%</b></div>}
            {wm.reclaimed_px != null && <div>blue-reclaimed px: <b>{wm.reclaimed_px}</b></div>}
            {wm.osm_veto_px != null && <div>OSM-veto px: <b>{wm.osm_veto_px}</b></div>}
          </div>
          {wm.mask_preview_png && (
            <div style={{ marginTop: 6 }}>
              <a href={wm.mask_preview_png} target="_blank" rel="noreferrer" style={{ display: 'inline-block' }}>
                <img src={wm.mask_preview_png} alt="mask QA overlay" style={{ maxWidth: '100%', maxHeight: 160, border: '1px solid var(--border-dim)', borderRadius: 4, imageRendering: 'pixelated' }} />
              </a>
              <div style={{ fontSize: 8, color: 'var(--text-dim)', marginTop: 2 }}>teal=water kept · grey=land · red=OSM veto · orange=intertidal excluded</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
