// ─────────────────────────────────────────────────────────────────────────────
// depthRamp.js — single source of truth for the bathymetric color ramp.
//
// Agency-grade EO visualisation: a perceptually-uniform, colorblind-safe
// sequential ramp (light cyan shallow → deep navy), shared across the 2D
// Results colorbar, the Leaflet MapPanel overlay legend, and the R3F 3D
// seabed surface so a depth reads as the SAME colour everywhere in the app.
//
// The stops below are a tuned, monotonic-luminance blue ramp (cmocean
// "deep"-style, reversed so shallow=light). t ∈ [0,1] where 0 = surface,
// 1 = max depth.
// ─────────────────────────────────────────────────────────────────────────────

// [t, [r,g,b]] with r,g,b in 0..1 — monotonic decreasing luminance.
const STOPS = [
  [0.00, [0.918, 0.969, 0.992]],  // #EAF7FD  surface / very shallow
  [0.10, [0.706, 0.902, 0.965]],  // #B4E6F6
  [0.22, [0.470, 0.812, 0.937]],  // #78CFEF
  [0.34, [0.275, 0.706, 0.890]],  // #46B4E3
  [0.46, [0.137, 0.588, 0.812]],  // #2396CF
  [0.58, [0.063, 0.471, 0.706]],  // #1078B4
  [0.70, [0.039, 0.357, 0.580]],  // #0A5B94
  [0.82, [0.031, 0.255, 0.435]],  // #08416F
  [0.92, [0.027, 0.165, 0.302]],  // #072A4D
  [1.00, [0.020, 0.090, 0.180]],  // #04172E  deepest
];

// Return [r,g,b] in 0..1 for t ∈ [0,1].
export function rampRGB(t) {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (x <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const a = (x - t0) / (t1 - t0 || 1e-6);
      return [
        c0[0] + a * (c1[0] - c0[0]),
        c0[1] + a * (c1[1] - c0[1]),
        c0[2] + a * (c1[2] - c0[2]),
      ];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

// CSS rgb() string for t ∈ [0,1].
export function rampCss(t) {
  const [r, g, b] = rampRGB(t);
  return `rgb(${(r * 255) | 0}, ${(g * 255) | 0}, ${(b * 255) | 0})`;
}

// CSS rgb() for an absolute depth (m) given a max depth (m).
export function depthCss(depth, maxD) {
  return rampCss((depth || 0) / (maxD || 25));
}

// A `linear-gradient(...)` string (top = surface, bottom = deep) for CSS bars.
export function rampGradientCss(direction = 'to bottom', n = 12) {
  const parts = [];
  for (let i = 0; i <= n; i++) parts.push(rampCss(i / n));
  return `linear-gradient(${direction}, ${parts.join(', ')})`;
}

// Evenly-spaced tick objects {frac, depth, css} for a max depth — handy for
// rendering a labelled colorbar. `nTicks` includes both endpoints.
export function rampTicks(maxD, nTicks = 5) {
  const out = [];
  const N = Math.max(1, nTicks - 1);
  for (let i = 0; i <= N; i++) {
    const frac = i / N;
    out.push({ frac, depth: (maxD || 25) * frac, css: rampCss(frac) });
  }
  return out;
}
