import React, { useState, useEffect, useCallback } from 'react';
import HelpIcon from './HelpIcon';

const API = process.env.REACT_APP_API_URL || '';

// Mini SVG chart for time series
function TimeSeriesChart({ data, field, color, label, height = 80 }) {
  if (!data || data.length < 2) return null;
  const values = data.map(d => d[field]).filter(v => v != null && v > 0);
  if (values.length < 2) return null;
  const mn = Math.min(...values), mx = Math.max(...values);
  const range = mx - mn || 1;
  const w = 280, h = height, pad = 20;

  const points = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * (w - 2 * pad);
    const y = h - pad - ((v - mn) / range) * (h - 2 * pad);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div style={{ marginBottom: '8px' }}>
      <div style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginBottom: '2px' }}>{label}</div>
      <svg width={w} height={h} style={{ background: 'var(--bg-primary)', borderRadius: '4px', border: '1px solid var(--border-dim)' }}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map(f => {
          const y = h - pad - f * (h - 2 * pad);
          return <line key={f} x1={pad} y1={y} x2={w - pad} y2={y} stroke="var(--border-dim)" strokeWidth="0.5" strokeDasharray="2,2" />;
        })}
        {/* Line */}
        <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
        {/* Points */}
        {values.map((v, i) => {
          const x = pad + (i / (values.length - 1)) * (w - 2 * pad);
          const y = h - pad - ((v - mn) / range) * (h - 2 * pad);
          return <circle key={i} cx={x} cy={y} r="3" fill={color} opacity="0.8" />;
        })}
        {/* Y axis labels */}
        <text x={pad - 2} y={pad} fontSize="7" fill="var(--text-dim)" textAnchor="end" fontFamily="var(--font-mono)">{mx.toFixed(1)}</text>
        <text x={pad - 2} y={h - pad} fontSize="7" fill="var(--text-dim)" textAnchor="end" fontFamily="var(--font-mono)">{mn.toFixed(1)}</text>
        {/* X axis labels */}
        {data.length > 0 && <text x={pad} y={h - 4} fontSize="6" fill="var(--text-dim)" fontFamily="var(--font-mono)">{data[0].date}</text>}
        {data.length > 1 && <text x={w - pad} y={h - 4} fontSize="6" fill="var(--text-dim)" textAnchor="end" fontFamily="var(--font-mono)">{data[data.length - 1].date}</text>}
      </svg>
    </div>
  );
}

// Alert badge
function AlertBadge({ severity, count }) {
  const colors = { critical: '#ef4444', warning: '#f59e0b', info: '#3b82f6' };
  if (!count) return null;
  return (
    <span style={{
      display: 'inline-block', padding: '1px 6px', borderRadius: '8px', fontSize: '8px',
      fontFamily: 'var(--font-mono)', fontWeight: 700, marginLeft: '4px',
      background: colors[severity] || '#666', color: '#fff',
    }}>{count}</span>
  );
}

export default function MonitoringPanel({ roi, results, onStoreSurvey, addLog }) {
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(false);
  const [storeDate, setStoreDate] = useState(new Date().toISOString().split('T')[0]);
  const [storeName, setStoreName] = useState('');
  const [compareA, setCompareA] = useState('');
  const [compareB, setCompareB] = useState('');
  const [changeResult, setChangeResult] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [zoneName, setZoneName] = useState('');
  const [zoneThreshold, setZoneThreshold] = useState(0.5);

  // Fetch dashboard data
  const fetchDashboard = useCallback(async () => {
    setLoading(true);
    try {
      const body = roi ? JSON.stringify({ bbox: roi }) : '{}';
      const r = await fetch(`${API}/api/monitoring/dashboard`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body
      });
      const d = await r.json();
      setDashboard(d);
      if (d.surveys?.length >= 2 && !compareA) {
        setCompareA(d.surveys[1]?.survey_id || '');
        setCompareB(d.surveys[0]?.survey_id || '');
      }
    } catch (e) { console.error('Dashboard fetch:', e); }
    setLoading(false);
  }, [roi, compareA]);

  useEffect(() => { fetchDashboard(); }, [fetchDashboard]);

  // Store current results as survey
  const handleStore = async () => {
    if (!results?.interpolated_points?.length) {
      addLog?.('No results to store — run extraction first', 'error');
      return;
    }
    const surveyId = storeName || `survey_${Date.now()}`;
    try {
      // Reconstruct a simplified grid from interpolated points
      const pts = results.interpolated_points;
      const lats = pts.map(p => p.lat), lons = pts.map(p => p.lon);
      const minLat = Math.min(...lats), maxLat = Math.max(...lats);
      const minLon = Math.min(...lons), maxLon = Math.max(...lons);
      const gridSize = Math.ceil(Math.sqrt(pts.length));
      const grid = Array(gridSize).fill(null).map(() => Array(gridSize).fill(0));
      pts.forEach(p => {
        const r = Math.min(gridSize - 1, Math.floor((maxLat - p.lat) / (maxLat - minLat + 1e-10) * gridSize));
        const c = Math.min(gridSize - 1, Math.floor((p.lon - minLon) / (maxLon - minLon + 1e-10) * gridSize));
        grid[r][c] = p.depth || 0;
      });

      const r = await fetch(`${API}/api/monitoring/store`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          survey_id: surveyId, bbox: roi, date: storeDate,
          depth_grid: grid, method: results.ml_stats?.method || 'cnn',
          r2: results.ml_stats?.r2 || 0, rmse: results.ml_stats?.quality?.rmse || 0,
          sources: (results.sources_used || []).join(', '),
        })
      });
      const d = await r.json();
      if (d.stored) {
        addLog?.(`Survey "${surveyId}" stored (${d.stats?.n_points} pts, mean=${d.stats?.mean_depth}m)`, 'success');
        fetchDashboard();
      } else {
        addLog?.(d.error || 'Store failed', 'error');
      }
    } catch (e) { addLog?.(e.message, 'error'); }
  };

  // Compare two surveys
  const handleCompare = async () => {
    if (!compareA || !compareB) return;
    try {
      const r = await fetch(`${API}/api/monitoring/change`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ survey_a: compareA, survey_b: compareB, threshold })
      });
      const d = await r.json();
      if (d.error) { addLog?.(d.error, 'error'); return; }
      setChangeResult(d);
      addLog?.(`Change: mean=${d.statistics?.mean_change?.toFixed(2)}m, ` +
               `RMSD=${d.statistics?.rmsd?.toFixed(2)}m, alert=${d.statistics?.alert_level}`, 'success');
    } catch (e) { addLog?.(e.message, 'error'); }
  };

  // Create monitoring zone
  const handleCreateZone = async () => {
    if (!roi || !zoneName) return;
    try {
      const r = await fetch(`${API}/api/monitoring/zones`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: zoneName, bbox: roi, threshold: zoneThreshold })
      });
      const d = await r.json();
      if (d.created) { addLog?.(`Zone "${zoneName}" created`, 'success'); fetchDashboard(); }
    } catch (e) { addLog?.(e.message, 'error'); }
  };

  // Acknowledge alert
  const handleAckAlert = async (id) => {
    try {
      await fetch(`${API}/api/monitoring/alerts/${id}/acknowledge`, { method: 'POST' });
      fetchDashboard();
    } catch (e) { console.error(e); }
  };

  const mono = { fontFamily: 'var(--font-mono)' };
  const label = { fontSize: '7px', color: 'var(--text-dim)', display: 'block', marginBottom: '2px', ...mono };
  const input = {
    width: '100%', padding: '5px', fontSize: '9px', background: 'var(--bg-primary)',
    border: '1px solid var(--border-dim)', borderRadius: '4px', color: 'var(--text-primary)', ...mono
  };
  const statCard = (title, value, color) => (
    <div style={{ padding: '6px', borderRadius: '4px', background: 'var(--bg-primary)', border: '1px solid var(--border-dim)', textAlign: 'center' }}>
      <div style={{ fontSize: '7px', color: 'var(--text-dim)', ...mono }}>{title}</div>
      <div style={{ fontSize: '14px', fontWeight: 700, color, ...mono }}>{value}</div>
    </div>
  );

  return (
    <div style={{ padding: '12px', fontSize: '9px', ...mono, overflowY: 'auto', maxHeight: '100%' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <h3 style={{ fontSize: '12px', fontWeight: 800, color: '#06b6d4', margin: 0, display: 'flex', alignItems: 'center' }}>
          BATHYMETRY MONITORING
          <HelpIcon text="Stores each survey run for a zone and tracks change over time: depth/volume differences, shoaling or deepening trends, and alerts (e.g. sudden shoaling in a channel)." />
        </h3>
        <button onClick={fetchDashboard} disabled={loading} style={{
          padding: '4px 8px', fontSize: '8px', borderRadius: '4px', border: '1px solid var(--border-dim)',
          background: 'var(--bg-secondary)', color: 'var(--text-primary)', cursor: 'pointer', ...mono
        }}>{loading ? '...' : 'Refresh'}</button>
      </div>

      {/* Overview Stats */}
      {dashboard && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px', marginBottom: '12px' }}>
          {statCard('SURVEYS', dashboard.total_surveys, '#06b6d4')}
          {statCard('ALERTS', (dashboard.alert_counts?.critical || 0) + (dashboard.alert_counts?.warning || 0), '#ef4444')}
          {statCard('ZONES', dashboard.zones?.length || 0, '#10b981')}
        </div>
      )}

      {/* Alert Banner */}
      {dashboard?.alerts?.length > 0 && (
        <div style={{ marginBottom: '12px', padding: '8px', borderRadius: '6px',
          border: '1px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.05)' }}>
          <div style={{ fontSize: '9px', fontWeight: 700, color: '#ef4444', marginBottom: '4px' }}>
            ACTIVE ALERTS
            <AlertBadge severity="critical" count={dashboard.alert_counts?.critical} />
            <AlertBadge severity="warning" count={dashboard.alert_counts?.warning} />
          </div>
          {dashboard.alerts.slice(0, 5).map(a => (
            <div key={a.id} style={{
              padding: '4px', marginBottom: '2px', borderRadius: '3px', fontSize: '8px',
              background: a.severity === 'critical' ? 'rgba(239,68,68,0.08)' : 'rgba(245,158,11,0.08)',
              display: 'flex', alignItems: 'center', gap: '4px',
            }}>
              <span style={{ color: a.severity === 'critical' ? '#ef4444' : '#f59e0b', fontWeight: 700 }}>
                {a.severity.toUpperCase()}
              </span>
              <span style={{ flex: 1, color: 'var(--text-primary)' }}>{a.message?.slice(0, 80)}</span>
              <button onClick={() => handleAckAlert(a.id)} style={{
                fontSize: '7px', padding: '1px 4px', borderRadius: '2px', border: '1px solid var(--border-dim)',
                background: 'var(--bg-secondary)', color: 'var(--text-dim)', cursor: 'pointer', ...mono
              }}>ACK</button>
            </div>
          ))}
        </div>
      )}

      {/* Time Series Charts */}
      {dashboard?.time_series?.length >= 2 && (
        <div style={{ marginBottom: '12px' }}>
          <TimeSeriesChart data={dashboard.time_series} field="mean_depth" color="#06b6d4" label="Mean Depth (m) over time" />
          <TimeSeriesChart data={dashboard.time_series} field="r2" color="#10b981" label="Model R\u00b2 over time" height={60} />
        </div>
      )}

      {/* Store Survey */}
      <div style={{ marginBottom: '12px', padding: '10px', borderRadius: '6px',
        border: '1px solid rgba(6,182,212,0.2)', background: 'rgba(6,182,212,0.03)' }}>
        <div style={{ fontSize: '9px', fontWeight: 700, color: '#06b6d4', marginBottom: '6px' }}>STORE SURVEY</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', marginBottom: '4px' }}>
          <div>
            <label style={label}>SURVEY ID</label>
            <input value={storeName} onChange={e => setStoreName(e.target.value)}
              placeholder="auto-generated" style={input} />
          </div>
          <div>
            <label style={label}>DATE</label>
            <input type="date" value={storeDate} onChange={e => setStoreDate(e.target.value)} style={input} />
          </div>
        </div>
        <button onClick={handleStore} disabled={!results} style={{
          width: '100%', padding: '8px', fontSize: '9px', fontWeight: 700, borderRadius: '4px',
          border: 'none', cursor: results ? 'pointer' : 'not-allowed',
          background: results ? 'linear-gradient(135deg, #06b6d4, #0891b2)' : 'var(--bg-secondary)',
          color: results ? '#fff' : 'var(--text-dim)', ...mono
        }}>Store Current Results</button>
      </div>

      {/* Change Detection */}
      <div style={{ marginBottom: '12px', padding: '10px', borderRadius: '6px',
        border: '1px solid rgba(168,85,247,0.2)', background: 'rgba(168,85,247,0.03)' }}>
        <div style={{ fontSize: '9px', fontWeight: 700, color: '#a855f7', marginBottom: '6px' }}>CHANGE DETECTION</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', marginBottom: '4px' }}>
          <div>
            <label style={label}>SURVEY A (earlier)</label>
            <select value={compareA} onChange={e => setCompareA(e.target.value)} style={input}>
              <option value="">Select...</option>
              {dashboard?.surveys?.map(s => (
                <option key={s.survey_id} value={s.survey_id}>{s.date} — {s.survey_id}</option>
              ))}
            </select>
          </div>
          <div>
            <label style={label}>SURVEY B (later)</label>
            <select value={compareB} onChange={e => setCompareB(e.target.value)} style={input}>
              <option value="">Select...</option>
              {dashboard?.surveys?.map(s => (
                <option key={s.survey_id} value={s.survey_id}>{s.date} — {s.survey_id}</option>
              ))}
            </select>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center', marginBottom: '4px' }}>
          <label style={{ ...label, marginBottom: 0, whiteSpace: 'nowrap' }}>THRESHOLD</label>
          <input type="number" min={0.1} max={5} step={0.1} value={threshold}
            onChange={e => setThreshold(+e.target.value)} style={{ ...input, width: '60px' }} />
          <span style={{ fontSize: '7px', color: 'var(--text-dim)' }}>m</span>
          <button onClick={handleCompare} disabled={!compareA || !compareB} style={{
            flex: 1, padding: '6px', fontSize: '9px', fontWeight: 700, borderRadius: '4px',
            border: 'none', cursor: (compareA && compareB) ? 'pointer' : 'not-allowed',
            background: (compareA && compareB) ? 'linear-gradient(135deg, #a855f7, #7c3aed)' : 'var(--bg-secondary)',
            color: (compareA && compareB) ? '#fff' : 'var(--text-dim)', ...mono
          }}>Compare</button>
        </div>

        {/* Change Results */}
        {changeResult?.statistics && (
          <div style={{ padding: '6px', borderRadius: '4px', background: 'var(--bg-secondary)', border: '1px solid var(--border-dim)', marginTop: '4px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px', marginBottom: '4px' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '7px', color: 'var(--text-dim)' }}>MEAN</div>
                <div style={{ fontSize: '11px', fontWeight: 700, color: changeResult.statistics.mean_change > 0 ? '#10b981' : '#ef4444' }}>
                  {changeResult.statistics.mean_change > 0 ? '+' : ''}{changeResult.statistics.mean_change?.toFixed(2)}m
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '7px', color: 'var(--text-dim)' }}>RMSD</div>
                <div style={{ fontSize: '11px', fontWeight: 700, color: '#f59e0b' }}>{changeResult.statistics.rmsd?.toFixed(2)}m</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '7px', color: 'var(--text-dim)' }}>CHANGED</div>
                <div style={{ fontSize: '11px', fontWeight: 700, color: '#a855f7' }}>{changeResult.statistics.pct_changed}%</div>
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '7px' }}>
              <span style={{ color: '#ef4444' }}>Erosion: {Math.abs(changeResult.statistics.max_erosion || 0).toFixed(1)}m ({changeResult.statistics.n_erosion} cells)</span>
              <span style={{ color: '#10b981' }}>Deposition: {(changeResult.statistics.max_deposition || 0).toFixed(1)}m ({changeResult.statistics.n_deposition} cells)</span>
            </div>
            {changeResult.statistics.alert_level !== 'none' && (
              <div style={{
                marginTop: '4px', padding: '3px 6px', borderRadius: '3px', fontSize: '8px', fontWeight: 700, textAlign: 'center',
                background: changeResult.statistics.alert_level === 'critical' ? 'rgba(239,68,68,0.1)' : 'rgba(245,158,11,0.1)',
                color: changeResult.statistics.alert_level === 'critical' ? '#ef4444' : '#f59e0b',
              }}>ALERT: {changeResult.statistics.alert_level.toUpperCase()}</div>
            )}
          </div>
        )}
      </div>

      {/* Monitoring Zones */}
      <div style={{ marginBottom: '12px', padding: '10px', borderRadius: '6px',
        border: '1px solid rgba(16,185,129,0.2)', background: 'rgba(16,185,129,0.03)' }}>
        <div style={{ fontSize: '9px', fontWeight: 700, color: '#10b981', marginBottom: '6px' }}>MONITORING ZONES</div>
        {dashboard?.zones?.map(z => (
          <div key={z.id} style={{
            padding: '4px', marginBottom: '2px', borderRadius: '3px', fontSize: '8px',
            background: 'var(--bg-primary)', border: '1px solid var(--border-dim)',
            display: 'flex', justifyContent: 'space-between',
          }}>
            <span style={{ fontWeight: 600 }}>{z.name}</span>
            <span style={{ color: 'var(--text-dim)' }}>th={z.alert_threshold_m}m / {z.check_interval_days}d</span>
          </div>
        ))}
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '4px', marginTop: '4px' }}>
          <input value={zoneName} onChange={e => setZoneName(e.target.value)}
            placeholder="Zone name" style={input} />
          <input type="number" min={0.1} max={5} step={0.1} value={zoneThreshold}
            onChange={e => setZoneThreshold(+e.target.value)} style={input} />
        </div>
        <button onClick={handleCreateZone} disabled={!roi || !zoneName} style={{
          width: '100%', padding: '6px', marginTop: '4px', fontSize: '8px', fontWeight: 700, borderRadius: '4px',
          border: '1px solid rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.05)',
          color: '#10b981', cursor: (roi && zoneName) ? 'pointer' : 'not-allowed', ...mono
        }}>Create Zone from ROI</button>
      </div>

      {/* Survey History */}
      {dashboard?.surveys?.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '9px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>SURVEY HISTORY</div>
          <div style={{ maxHeight: '120px', overflow: 'auto', borderRadius: '4px', border: '1px solid var(--border-dim)' }}>
            {dashboard.surveys.map((s, i) => (
              <div key={i} style={{
                padding: '4px 6px', fontSize: '8px', borderBottom: '1px solid var(--border-dim)',
                display: 'grid', gridTemplateColumns: '70px 1fr 50px 40px', gap: '4px', alignItems: 'center',
                background: i % 2 === 0 ? 'var(--bg-primary)' : 'transparent',
              }}>
                <span style={{ fontWeight: 600 }}>{s.date}</span>
                <span style={{ color: 'var(--text-dim)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.survey_id}</span>
                <span>{s.mean_depth?.toFixed(1)}m</span>
                <span style={{ color: '#10b981' }}>R\u00b2={s.r2?.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
