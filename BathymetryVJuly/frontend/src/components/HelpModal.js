import React from 'react';
import { INTERNAL_ANALYSIS } from './internalMode';

const API = process.env.REACT_APP_API_URL || '';

/**
 * Short, plain-English user guide. Keep it scannable — no methodology.
 */
export default function HelpModal({ open, onClose }) {
  if (!open) return null;
  const Sec = ({ title, children }) => (
    <section style={{ marginBottom: 18 }}>
      <h3 style={{
        fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 800,
        letterSpacing: '0.08em', color: '#0284c7', margin: '0 0 6px',
      }}>{title}</h3>
      <div style={{ fontSize: 12, lineHeight: 1.55, color: '#1e293b' }}>{children}</div>
    </section>
  );
  return (
    <div onClick={e => { if (e.target === e.currentTarget) onClose && onClose(); }} style={{
      position: 'fixed', inset: 0, zIndex: 3000,
      background: 'rgba(15,23,42,0.55)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        background: '#fff', borderRadius: 14, width: '92%', maxWidth: 720,
        maxHeight: '92vh', overflow: 'auto',
        padding: '22px 26px 26px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.35)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div>
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 800, color: '#0f172a' }}>
              User guide
            </h2>
            <p style={{ margin: '2px 0 0', fontSize: 11, fontFamily: 'var(--font-mono)', color: '#64748b' }}>
              WGS84 · LAT datum (adjustable) · depths capped at 25 m
            </p>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', fontSize: 22, cursor: 'pointer', color: '#94a3b8',
          }} aria-label="Close">×</button>
        </div>

        <Sec title="1 · Set a Region of Interest">
          Draw a rectangle on the map, pick a preset, or edit vertices — N/S/E/W update live.
          Use <b>Zoom to ROI</b> to re-center the map.
        </Sec>

        <Sec title="2 · Load survey data (optional)">
          Supported: <code>.xyz</code>, <code>.csv</code>, <code>.shp</code> (+<code>.dbf</code>/<code>.prj</code>).
          Pick the coordinate order (Lat,Lon,Z or Lon,Lat,Z), UTM zone if projected, and vertical
          datum. Four preset sites are available: Khalifa Port, Old Mussafah, Abu Al Abyad, Jbel Dhanna.
        </Sec>

        <Sec title="3 · Pro Extract (recommended)">
          Pick a preset or draw a ROI, then click <b>Pro Extract</b>. The app picks the best
          reference data for the area automatically and returns a depth map{INTERNAL_ANALYSIS ? ' with R² / RMSE metrics' : ''}.
        </Sec>

        <Sec title={INTERNAL_ANALYSIS ? '4 · ICESat-2 lidar' : '4 · Lidar depth points'}>
          Search and run lidar depth points over any ROI. Results appear on the map
          colour-coded per beam and feed into Pro Extract automatically.
        </Sec>

        <Sec title="5 · Validation & export">
          {INTERNAL_ANALYSIS
            ? <>Upload an observed-depth file to compare against the prediction (RMSE, R², S-44 pass %). Export to GeoJSON, CSV or GeoTIFF.</>
            : <>Export the depth result to GeoJSON, CSV or GeoTIFF.</>}
        </Sec>

        <Sec title="6 · Monitoring">
          Store successive surveys and view change, trends, and alerts in the <b>MONITOR</b> tab.
        </Sec>

        <div style={{
          marginTop: 4, paddingTop: 12, borderTop: '1px solid #e2e8f0',
          fontSize: 12, color: '#1e293b', display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
        }}>
          <span>Need more detail? The full step-by-step manual covers every module:</span>
          <a href={`${API}/api/user-guide`} target="_blank" rel="noreferrer" style={{
            fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700, color: '#0284c7',
            padding: '4px 10px', borderRadius: 6, border: '1px solid rgba(2,132,199,0.35)',
            background: 'rgba(2,132,199,0.06)', textDecoration: 'none',
          }}>View user guide</a>
          <a href={`${API}/api/user-guide?download=1`} style={{
            fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700, color: '#fff',
            padding: '4px 12px', borderRadius: 6, border: '1px solid #0284c7',
            background: '#0284c7', textDecoration: 'none',
          }}>⬇ Download</a>
        </div>
      </div>
    </div>
  );
}
