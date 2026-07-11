import React from 'react';
import { INTERNAL_ANALYSIS } from './internalMode';

export default function LoadingOverlay({ message, step, total, activeModule }) {
  const progress = total > 0 ? ((step + 1) / total) * 100 : 0;

  return (
    <div style={{
      position: 'absolute', inset: 0,
      background: 'rgba(6, 9, 15, 0.92)',
      backdropFilter: 'blur(12px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexDirection: 'column', gap: '28px', zIndex: 1000,
    }}>
      {/* Animated orbital spinner */}
      <div style={{ position: 'relative', width: '80px', height: '80px' }}>
        {/* Outer ring */}
        <div style={{
          position: 'absolute', inset: 0,
          border: '2px solid var(--border-dim)',
          borderTopColor: 'var(--accent-primary)',
          borderRadius: '50%',
          animation: 'spin 1.2s linear infinite',
        }} />
        {/* Inner ring */}
        <div style={{
          position: 'absolute', inset: '10px',
          border: '2px solid var(--border-subtle)',
          borderBottomColor: 'var(--accent-teal)',
          borderRadius: '50%',
          animation: 'spin 1.8s linear infinite reverse',
        }} />
        {/* Center icon */}
        <div style={{
          position: 'absolute', inset: '20px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '20px',
          animation: 'pulse 2s infinite',
        }}>
          {activeModule === 'swot' ? '🛰' : activeModule === 'fusion' ? '🔬' : '⚓'}
        </div>
      </div>

      {/* Text */}
      <div style={{ textAlign: 'center', maxWidth: '360px' }}>
        <p style={{
          fontSize: '15px', fontWeight: 700, letterSpacing: '-0.02em',
          background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-teal))',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          marginBottom: '10px',
        }}>
          {activeModule === 'swot' ? 'SWOT Processing' : 'Bathymetry Pipeline'}
        </p>
        <p style={{
          fontFamily: 'var(--font-mono)', fontSize: '11px',
          color: 'var(--text-secondary)', lineHeight: 1.5,
          animation: 'fadeIn 0.3s',
        }}>
          {message}
        </p>
      </div>

      {/* Progress bar */}
      {total > 0 && (
        <div style={{ width: '280px' }}>
          <div style={{
            height: '3px', background: 'var(--bg-tertiary)',
            borderRadius: '2px', overflow: 'hidden',
          }}>
            <div style={{
              width: `${progress}%`, height: '100%',
              background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-teal))',
              borderRadius: '2px',
              transition: 'width 0.5s var(--ease-out)',
            }} />
          </div>
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            marginTop: '6px', fontSize: '9px', fontFamily: 'var(--font-mono)',
          }}>
            <span style={{ color: 'var(--text-dim)' }}>Step {step + 1} of {total}</span>
            <span style={{ color: 'var(--accent-primary)' }}>{progress.toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Note — adapt to the actual step the pipeline is in */}
      <p style={{
        fontFamily: 'var(--font-mono)', fontSize: '9px',
        color: 'var(--text-dim)', opacity: 0.6, maxWidth: '340px', textAlign: 'center',
      }}>
        {!INTERNAL_ANALYSIS
          ? 'Processing. Progress bar updates after each phase.'
          : (/icesat2|sliderule|atl03/i.test(message || '')
            ? 'Streaming ICESat-2 ATL03 photons via SlideRule (90 s cap — continues without ICESat-2 if slow).'
            : /sentinel|s2|cdse|copernicus/i.test(message || '')
              ? 'Fetching Sentinel-2 L2A imagery from Copernicus CDSE.'
              : /gebco/i.test(message || '')
                ? 'Pulling GEBCO 2024 bathymetric prior.'
                : /train|epoch|cnn/i.test(message || '')
                  ? 'Training CNN / physical prior — progress logged per epoch.'
                  : 'Pipeline running. Progress bar updates after each phase.')}
      </p>
    </div>
  );
}
