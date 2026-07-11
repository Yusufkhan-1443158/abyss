import React, { useState } from 'react';

/**
 * Small inline info icon with an on-hover tooltip.
 * Use: <HelpIcon text="…long explanation…" />
 */
export default function HelpIcon({ text, size = 13 }) {
  const [show, setShow] = useState(false);
  return (
    <span
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onFocus={() => setShow(true)}
      onBlur={() => setShow(false)}
      tabIndex={0}
      role="button"
      aria-label="Info"
      style={{
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: size, height: size,
        flex: '0 0 auto',
        marginLeft: 6,
        borderRadius: '50%',
        background: 'rgba(2,132,199,0.12)',
        color: '#0284c7',
        fontSize: Math.max(9, size - 4),
        fontFamily: 'var(--font-mono)',
        fontWeight: 800,
        cursor: 'help',
        userSelect: 'none',
      }}
    >
      i
      {show && (
        <span
          style={{
            position: 'absolute',
            top: size + 4,
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 2100,
            whiteSpace: 'normal',
            width: 240,
            padding: '8px 10px',
            borderRadius: 6,
            background: 'rgba(15,23,42,0.96)',
            color: '#fff',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            fontWeight: 500,
            lineHeight: 1.5,
            boxShadow: '0 4px 12px rgba(0,0,0,0.25)',
            pointerEvents: 'none',
          }}
        >
          {text}
        </span>
      )}
    </span>
  );
}
