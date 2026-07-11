// ─────────────────────────────────────────────────────────────────────────────
// internalMode — single source of truth for the CLIENT-PORTAL vs INTERNAL gate.
//
// The client-facing portal shows ONLY results (depth figures + numbers). The
// full method / data-source / provenance / accuracy-error surface is INTERNAL
// only and is gated behind INTERNAL_ANALYSIS (off by default). Nothing is
// deleted from the data layer — internal reviewers re-enable the expert surface
// with ?internal=1 (or localStorage.bathyInternal="1") WITHOUT any code change.
// ─────────────────────────────────────────────────────────────────────────────
export const INTERNAL_ANALYSIS = (() => {
  try {
    if (typeof window === 'undefined') return false;
    const qs = new URLSearchParams(window.location.search || '');
    if (qs.get('internal') === '1' || qs.get('internal') === 'true') return true;
    if (window.localStorage && window.localStorage.getItem('bathyInternal') === '1') return true;
  } catch (_) { /* ignore */ }
  return false;
})();

// ─────────────────────────────────────────────────────────────────────────────
// HYDRO_CLIENT — third tier for hydrographic-authority client builds (e.g. AD
// Ports/Noatum). Unlike INTERNAL_ANALYSIS (full internal method/data-source
// surface), HYDRO_CLIENT un-gates ONLY the IHO S-44/S-52 compliance surface
// (the 'iho_chart' + 'validation' tabs, the S-44/CATZOC badges) that a
// hydrographic authority contractually needs to see, while leaving the
// engine/data-source redactions (Sentinel-2, ICESat-2, Lyzenga, etc. — IP
// concerns, see sanitizeClientText in App.js) untouched. Set at build time via
// `REACT_APP_CLIENT_TIER=hydro`, or per-session via `?tier=hydro` /
// `localStorage.bathyTier='hydro'` (no rebuild needed for demos/testing).
// INTERNAL_ANALYSIS always implies the hydro surface too.
// ─────────────────────────────────────────────────────────────────────────────
export const HYDRO_CLIENT = (() => {
  try {
    if (typeof process !== 'undefined' && process.env && process.env.REACT_APP_CLIENT_TIER === 'hydro') return true;
    if (typeof window === 'undefined') return false;
    const qs = new URLSearchParams(window.location.search || '');
    if (qs.get('tier') === 'hydro') return true;
    if (window.localStorage && window.localStorage.getItem('bathyTier') === 'hydro') return true;
  } catch (_) { /* ignore */ }
  return false;
})();

// Convenience: true whenever the IHO S-44/S-52 compliance surface should
// render (either the full internal build, or the hydro-client tier).
export const SHOW_IHO_SURFACE = INTERNAL_ANALYSIS || HYDRO_CLIENT;

export default INTERNAL_ANALYSIS;
