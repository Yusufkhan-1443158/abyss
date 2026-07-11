import React from 'react';

export default function StatusBar({ roi, results, loading, activeModule, lastLog }) {
  return (
    <div style={{
      gridColumn: '1 / -1',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 16px',
      background: 'var(--bg-secondary)',
      borderTop: '1px solid var(--border-subtle)',
      fontSize: '9px', fontFamily: 'var(--font-mono)',
      color: 'var(--text-dim)',
    }}>
      {/* Left: status indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{
          width: '6px', height: '6px', borderRadius: '50%',
          background: loading ? 'var(--accent-amber)' : results ? 'var(--accent-emerald)' : 'var(--text-dim)',
          boxShadow: loading ? '0 0 6px var(--accent-amber)' : results ? '0 0 6px var(--accent-emerald)' : 'none',
          animation: loading ? 'pulse 1s infinite' : 'none',
        }} />
        <span>
          {loading ? 'Processing...' : results ? 'Ready' : 'Idle'}
        </span>

        {roi && (
          <span style={{ color: 'var(--text-dim)', marginLeft: '8px' }}>
            ROI: {roi.south.toFixed(2)}°–{roi.north.toFixed(2)}°N, {roi.west.toFixed(2)}°–{roi.east.toFixed(2)}°E
          </span>
        )}
      </div>

      {/* Center: last log message */}
      <div style={{ flex: 1, textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', padding: '0 20px' }}>
        {lastLog && (
          <span style={{
            color: lastLog.type === 'error' ? 'var(--accent-rose)'
              : lastLog.type === 'success' ? 'var(--accent-emerald)'
              : 'var(--text-dim)',
          }}>
            {lastLog.msg}
          </span>
        )}
      </div>

      {/* Right: module + version */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span>Module: {activeModule.toUpperCase()}</span>
        <span style={{ color: 'var(--text-faint)' }}>|</span>
        <span>Bathymetry From Space v2.0 Phase II</span>
      </div>
    </div>
  );
}
