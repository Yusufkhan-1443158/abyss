import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ThreeDView from './ThreeDView';
import { INTERNAL_ANALYSIS } from './internalMode';

// ─────────────────────────────────────────────────────────────────────────────
// BathyResultsPanel — persistent, always-reachable Results space (R2/R3/R4).
//
//  • R2: a one-click "Khalifa Port (High-Resolution)" button that POSTs
//        /api/bathy-job/start {site_key:"khalifa_port", resolution:"vhr"}.
//  • R3: a durable list of ALL finished jobs from /api/results (name, ROI,
//        resolution, date, status, file + size). Each row has Download and
//        Analyse actions. Polls /api/results so it survives reloads.
//  • R4: a small "Compute bathymetry" form — pick one or many ROIs + a
//        resolution (VHR or 10 m) and queue them via /start or /start-batch.
//
// Honesty (R1): running jobs surface progress only as the backend reports it
// (real % or an indeterminate spinner) — never a time-extrapolated bar.
// ─────────────────────────────────────────────────────────────────────────────

// CLIENT PORTAL vs INTERNAL ANALYSIS gate — see ./internalMode. The client-facing
// portal shows ONLY results (depth figures + numbers); the full method /
// provenance / model-card + RMSE/R²/CATZOC surface is INTERNAL-only (?internal=1).
// Nothing is deleted from the data layer — gated surfaces simply do not mount.

// Best-effort depth range for a result, read defensively from whatever the
// backend attached. Returns [min,max] in metres or null — never fabricated.
function depthRange(res) {
  const m = (res && res.metrics) || {};
  const s = (res && (res.depth_stats || res.stats)) || {};
  const lo = pick(s, 'min', 'min_depth', 'depth_min') ?? pick(m, 'depth_min_m', 'z_min')
    ?? (Array.isArray(m.depth_range_m) ? m.depth_range_m[0] : undefined);
  const hi = pick(s, 'max', 'max_depth', 'depth_max') ?? pick(m, 'depth_max_m', 'z_max')
    ?? (Array.isArray(m.depth_range_m) ? m.depth_range_m[1] : undefined);
  if (lo == null && hi == null) return null;
  return [lo, hi];
}

function fmtArea(res) {
  const m = (res && res.metrics) || {};
  const s = (res && (res.depth_stats || res.stats)) || {};
  const a = pick(s, 'area_km2', 'area') ?? pick(m, 'area_km2');
  return a != null ? `${num(a, 2)} km²` : null;
}

function fmtCoverage(res) {
  const m = (res && res.metrics) || {};
  const s = (res && (res.depth_stats || res.stats)) || {};
  const c = pick(s, 'coverage_pct') ?? pick(m, 'coverage_pct');
  return c != null ? `${num(c, 0)}%` : null;
}

// Known calibrated sites (mirrors the Sidebar presets / backend _KNOWN_SITES).
const KNOWN_SITES = [
  { key: 'khalifa_port',  label: 'Khalifa Port',  bbox: { west: 54.636,  south: 24.785,  east: 54.690,  north: 24.840 } },
  { key: 'old_mussafah',  label: 'Old Mussafah',  bbox: { west: 54.355,  south: 24.412,  east: 54.412,  north: 24.466 } },
  { key: 'abu_al_abyad',  label: 'Abu Al Abyad',  bbox: { west: 53.852,  south: 24.213,  east: 53.910,  north: 24.267 } },
  { key: 'jbel_dhanna',   label: 'Jbel Dhanna',   bbox: { west: 52.5692, south: 24.1989, east: 52.6186, north: 24.2440 } },
];

function fmtDate(epochSeconds) {
  if (!epochSeconds) return '—';
  try {
    const d = new Date(epochSeconds * 1000);
    return d.toLocaleString(undefined, { year: 'numeric', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
  } catch (_) { return '—'; }
}

function fmtSize(bytes) {
  if (!bytes && bytes !== 0) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fmtRoi(bbox) {
  if (!Array.isArray(bbox) || bbox.length < 4) return '—';
  const [w, s, e, n] = bbox;
  return `${w.toFixed(3)},${s.toFixed(3)} → ${e.toFixed(3)},${n.toFixed(3)}`;
}

// Collect every download format the backend offers for a result. The GeoTIFF
// is the canonical artefact (download_url); CSV / GeoJSON / mask are surfaced
// only when the backend actually exposes them — read defensively from several
// possible shapes (a `downloads:[{format,url,size_bytes?}]` array, or discrete
// *_url fields). Never invents a link the backend didn't provide.
function downloadFormats(res) {
  const out = [];
  const seen = new Set();
  const add = (fmt, url, label) => {
    if (!url || seen.has(fmt)) return;
    seen.add(fmt);
    out.push({ fmt, url, label });
  };
  if (res.download_url) add('geotiff', res.download_url, 'GeoTIFF (.tif)');
  if (Array.isArray(res.downloads)) {
    res.downloads.forEach(d => {
      if (!d || !d.url) return;
      const f = String(d.format || d.fmt || '').toLowerCase();
      const label = d.label || ({
        geotiff: 'GeoTIFF (.tif)', tif: 'GeoTIFF (.tif)', tiff: 'GeoTIFF (.tif)',
        csv: 'CSV (.csv)', geojson: 'GeoJSON (.geojson)', json: 'GeoJSON (.geojson)',
        mask: 'Optical-valid mask (.tif)',
      }[f] || (f ? f.toUpperCase() : 'Download'));
      add(f || `fmt${out.length}`, d.url, label);
    });
  }
  add('csv', res.csv_url || res.download_csv_url, 'CSV (.csv)');
  add('geojson', res.geojson_url || res.download_geojson_url, 'GeoJSON (.geojson)');
  add('mask', res.mask_url || res.mask_download_url, 'Optical-valid mask (.tif)');
  return out;
}

// The "Download all TIFFs (.zip)" endpoint for a result. The backend serves a
// deterministic route per id (streams EVERY raster the result references +
// MANIFEST.txt), so we can derive it from the id even if the list payload
// predates the `download_all_url` field. Works for every result type
// (certified / reconnaissance / MLE / stability / persisted jobs).
function zipUrlFor(res) {
  if (!res) return null;
  if (res.download_all_url) return res.download_all_url;
  const id = res.id || res.result_id || res.stability_id;
  return id ? `/api/results/${id}/download-all.zip` : null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Khalifa VHR HONESTY product detection + card (K1/K2; KHALIFA_PANEL_LOG).
//
// The Khalifa VHR 1 m product is in-situ-anchored (multibeam EDT-IDW), with an
// `optical_valid` mask: shallow (≤ z_opt ≈ 10.5 m) optical-valid, a collar
// (10.5–12 m, low-confidence) and a DEEP-ABSTAIN region (>12 m, optically
// saturated, CATZOC C-D). On the dredged deep basin satellite SDB has NO honest
// skill (the photon/optical ceiling) — the UI must NEVER imply sub-metre
// satellite-derived skill there. We render an explicit honesty card instead.
//
// Backend contract this card reads (all OPTIONAL; degrades gracefully):
//   result.metrics.honesty_card = {
//     line_1_shallow_SDB_skill: str, line_2_deep_abstain: str,
//     z_opt_m, z_opt_collar_m, kd490, secchi_m }
//   result.metrics.optical_valid = {            // pixel counts from the mask .tif
//     n_valid_le_zopt, n_collar, n_deep_abstain }
//   result.metrics.product = "khalifa_vhr"      // explicit product tag (preferred)
//   result.metrics.shallow_rmse_m, .n_insitu, .n_insitu_anchors  (provenance)
// If the explicit tag is absent we fall back to a name/resolution heuristic so a
// VHR result named "Khalifa …" still surfaces the honesty framing.
// ─────────────────────────────────────────────────────────────────────────────
function khalifaHonesty(res) {
  if (!res) return null;
  const m = res.metrics || {};
  const card = m.honesty_card || (res.honesty_card) || null;
  const mask = m.optical_valid || m.mask_summary || null;
  const tagged = m.product === 'khalifa_vhr' || m.is_khalifa_vhr === true || res.product === 'khalifa_vhr';
  const name = String(res.name || '').toLowerCase();
  const isVhr = res.resolution === 'vhr';
  const looksKhalifaVhr = isVhr && name.includes('khalifa');
  if (!(card || mask || tagged || looksKhalifaVhr)) return null;
  const zOpt = (card && card.z_opt_m) ?? m.z_opt_m ?? 10.5;
  const zCollar = (card && card.z_opt_collar_m) ?? m.z_opt_collar_m ?? 12.0;
  return {
    present: true,
    // honest, backend-substantiated when available; otherwise the documented K2 defaults
    line1: (card && card.line_1_shallow_SDB_skill)
      || `Shallow optically-valid (≤ ${zOpt} m): satellite SDB shows NO honest skill (leakage-safe spatial-block RMSE collapses to the mean, R² ≪ 0). Depth here is the in-situ-anchored IDW carrier, not an SDB retrieval.`,
    line2: (card && card.line_2_deep_abstain)
      || `Deep (> ${zCollar} m): OPTICALLY SATURATED — in-situ-anchored charted/maintained depth (CATZOC C-D), NOT an SDB retrieval. Full-fit deep RMSE is a map fidelity number, never quoted as skill.`,
    zOpt, zCollar,
    kd490: (card && card.kd490) ?? m.kd490 ?? null,
    secchi: (card && card.secchi_m) ?? m.secchi_m ?? null,
    shallowRmse: m.shallow_rmse_m ?? m.headline_shallow_rmse_m ?? (card && card.shallow_rmse_m) ?? null,
    nInsitu: m.n_insitu ?? null,
    mask: mask ? {
      valid: mask.n_valid_le_zopt ?? mask.valid ?? null,
      collar: mask.n_collar ?? mask.collar ?? null,
      deep: mask.n_deep_abstain ?? mask.deep ?? null,
    } : null,
    backed: !!(card || mask),   // false ⇒ we are showing the documented framing, not a per-run payload
  };
}

// Compact honesty banner shown inline on a Khalifa-VHR result row.
function KhalifaHonestyBanner({ h }) {
  if (!h) return null;
  return (
    <div style={{
      margin: '0 0 8px', padding: '8px 10px', borderRadius: '7px',
      border: '1px solid rgba(202,138,4,0.45)',
      background: 'repeating-linear-gradient(45deg, rgba(202,138,4,0.06), rgba(202,138,4,0.06) 7px, rgba(202,138,4,0.12) 7px, rgba(202,138,4,0.12) 14px)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '3px' }}>
        <span style={{ fontSize: '11px' }}>⚠️</span>
        <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.06em', color: '#854d0e' }}>
          IN-SITU-ANCHORED 1 m · NOT SATELLITE SDB ON THE DEEP BASIN
        </span>
      </div>
      <p style={{ margin: 0, fontSize: '9.5px', lineHeight: 1.5, fontFamily: 'var(--font-mono)', color: '#713f12' }}>
        Shallow (≤ {h.zOpt} m) optical-valid · collar {h.zOpt}–{h.zCollar} m · deep (&gt; {h.zCollar} m) <b>optically-saturated abstain</b> (CATZOC C-D). Open <b>Full analysis</b> for the honesty card.
      </p>
    </div>
  );
}

// Full honesty card for the Khalifa VHR product (shown in FullAnalysisModal).
// Honesty by design: the deep-abstain region is rendered hatched/distinct, and
// no satellite-derived sub-metre skill is implied anywhere on the deep basin.
function KhalifaHonestyCard({ h, stats }) {
  if (!h) return null;
  const total = h.mask ? (h.mask.valid || 0) + (h.mask.collar || 0) + (h.mask.deep || 0) : 0;
  const pct = (n) => (total > 0 && n != null ? `${((n / total) * 100).toFixed(1)}%` : null);
  const zones = [
    {
      key: 'valid', name: `Shallow optical-valid (≤ ${h.zOpt} m)`, color: '#0e7490',
      bg: 'rgba(14,116,144,0.10)', hatch: false, count: h.mask?.valid,
      note: `Kd490 ≈ ${h.kd490 ?? '0.22'}/m, Secchi ≈ ${h.secchi ?? '7'} m. Even here, leakage-safe SDB skill is absent — depth is the in-situ-anchored IDW carrier.`,
    },
    {
      key: 'collar', name: `Collar (${h.zOpt}–${h.zCollar} m)`, color: '#a16207',
      bg: 'rgba(161,98,7,0.12)', hatch: false, count: h.mask?.collar,
      note: 'Low-confidence transition band near the optical floor.',
    },
    {
      key: 'deep', name: `Deep abstain (> ${h.zCollar} m)`, color: '#6b7280',
      bg: 'rgba(107,114,128,0.10)', hatch: true, count: h.mask?.deep,
      note: 'Optically saturated — satellite SDB has NO honest skill (photon/optical ceiling). Depth = in-situ-anchored charted/maintained value, CATZOC C-D. NOT an SDB retrieval.',
    },
  ];
  return (
    <div style={{
      padding: '14px 16px', borderRadius: 10,
      border: '1px solid rgba(202,138,4,0.45)',
      background: 'linear-gradient(135deg, rgba(202,138,4,0.05), rgba(180,83,9,0.04))',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', padding: '2px 7px', borderRadius: 5, background: '#a16207', color: '#fff' }}>HONESTY CARD</span>
        <span style={{ fontSize: 12, fontFamily: 'var(--font-display)', fontWeight: 800, color: '#854d0e' }}>
          Khalifa VHR — in-situ-anchored 1 m (multibeam EDT-IDW)
        </span>
      </div>

      <p style={{ margin: '0 0 6px', fontSize: 11, lineHeight: 1.55, color: 'var(--text-secondary)' }}>
        <b style={{ color: '#854d0e' }}>1 · Shallow SDB skill:</b> {h.line1}
        {h.shallowRmse != null && (
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}> (shallow leakage-safe RMSE {num(h.shallowRmse, 2)} m — a mean-collapse number, not skill)</span>
        )}
      </p>
      <p style={{ margin: '0 0 10px', fontSize: 11, lineHeight: 1.55, color: 'var(--text-secondary)' }}>
        <b style={{ color: '#854d0e' }}>2 · Deep abstain:</b> {h.line2}
        {h.nInsitu != null && (
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}> (n_insitu = {h.nInsitu.toLocaleString()})</span>
        )}
      </p>

      {/* optical_valid mask zones — deep abstain is hatched/greyed */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
        {zones.map(z => (
          <div key={z.key} style={{
            padding: '8px 10px', borderRadius: 8,
            border: `1px solid ${z.color}55`,
            background: z.hatch
              ? `repeating-linear-gradient(45deg, ${z.bg}, ${z.bg} 6px, rgba(107,114,128,0.22) 6px, rgba(107,114,128,0.22) 12px)`
              : z.bg,
          }}>
            <div style={{ fontSize: 10, fontFamily: 'var(--font-display)', fontWeight: 800, color: z.color, marginBottom: 3 }}>
              {z.hatch ? '▦ ' : ''}{z.name}
            </div>
            {z.count != null && (
              <div style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginBottom: 3 }}>
                {z.count.toLocaleString()} px{pct(z.count) ? ` · ${pct(z.count)}` : ''}
              </div>
            )}
            <div style={{ fontSize: 9, lineHeight: 1.45, color: 'var(--text-secondary)' }}>{z.note}</div>
          </div>
        ))}
      </div>

      {!h.backed && (
        <p style={{ margin: '8px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
          Note: this run did not return a per-result honesty payload; the framing above is the documented
          Khalifa VHR product contract (z_opt ≈ {h.zOpt} m). It lights up with exact pixel counts once the
          backend ships <code>metrics.honesty_card</code> / <code>metrics.optical_valid</code>.
        </p>
      )}
    </div>
  );
}

export default function BathyResultsPanel({ open, onClose, API, roi, addLog, onAnalyse }) {
  const [results, setResults] = useState([]);
  const [loadingList, setLoadingList] = useState(false);
  const [listError, setListError] = useState(null);
  const [busy, setBusy] = useState(false);                 // a start/batch call is in flight
  const [activeMetricsId, setActiveMetricsId] = useState(null);

  // Full-analysis modal (per-result deep dive + 3D seabed)
  const [analysisFor, setAnalysisFor] = useState(null);    // the result meta being opened
  const [analysisData, setAnalysisData] = useState(null);  // /analyse payload
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);

  // R4 compute-form state
  const [formRes, setFormRes] = useState('10m');           // "10m" | "vhr"
  const [selectedSites, setSelectedSites] = useState({ khalifa_port: true });
  const [useDrawnRoi, setUseDrawnRoi] = useState(false);

  const pollRef = useRef(null);

  const log = useCallback((m, t) => { if (addLog) addLog(m, t); }, [addLog]);

  // ── R3: durable catalogue fetch ──────────────────────────────────────────
  const fetchResults = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoadingList(true);
    try {
      const r = await fetch(`${API}/api/results`);
      const d = await r.json();
      if (d && Array.isArray(d.results)) {
        setResults(d.results);
        setListError(null);
      }
    } catch (_) {
      setListError('Backend offline — cannot reach /api/results');
    } finally {
      if (showSpinner) setLoadingList(false);
    }
  }, [API]);

  // Poll while open (persists across reloads — the store is server-side).
  useEffect(() => {
    if (!open) return undefined;
    fetchResults(true);
    pollRef.current = setInterval(() => fetchResults(false), 5000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [open, fetchResults]);

  // ── job queueing helpers (R2/R4) ─────────────────────────────────────────
  const startSingle = useCallback(async (body, niceName) => {
    setBusy(true);
    try {
      const r = await fetch(`${API}/api/bathy-job/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok || d.error) {
        const msg = d.error || `HTTP ${r.status}`;
        log(`Compute start failed (${niceName}): ${msg}`, 'error');
        return false;
      }
      log(`Queued ${niceName} · job ${d.id} · ${d.area_km2 ?? '?'} km² @ ${body.resolution}`, 'success');
      // Refresh soon so the run shows up as it lands in the catalogue.
      setTimeout(() => fetchResults(false), 1500);
      return true;
    } catch (e) {
      log(`Compute start failed (${niceName}): ${e.message}`, 'error');
      return false;
    } finally {
      setBusy(false);
    }
  }, [API, log, fetchResults]);

  // R2 — Khalifa Port High-Resolution (VHR) one-click.
  const launchKhalifaHR = useCallback(() => {
    return startSingle(
      { site_key: 'khalifa_port', resolution: 'vhr', label: 'Khalifa Port (High-Resolution)' },
      'Khalifa Port (High-Resolution)',
    );
  }, [startSingle]);

  // R4 — queue the form selection. 1 ROI → /start; ≥2 → /start-batch.
  const launchCompute = useCallback(async () => {
    const items = [];
    KNOWN_SITES.forEach(s => {
      if (selectedSites[s.key]) {
        items.push({ site_key: s.key, resolution: formRes, label: `${s.label} (${formRes === 'vhr' ? 'High-Resolution' : '10 m'})` });
      }
    });
    if (useDrawnRoi) {
      if (Array.isArray(roi) && roi.length >= 4) {
        items.push({ bbox: { west: roi[0], south: roi[1], east: roi[2], north: roi[3] }, resolution: formRes, label: `Drawn ROI (${formRes === 'vhr' ? 'High-Resolution' : '10 m'})` });
      } else {
        log('Use-drawn-ROI is checked but no ROI is drawn on the map.', 'error');
        return;
      }
    }
    if (items.length === 0) {
      log('Select at least one ROI (a known site or the drawn ROI) before computing.', 'error');
      return;
    }
    if (items.length === 1) {
      await startSingle(items[0], items[0].label);
      return;
    }
    // Batch
    setBusy(true);
    try {
      const r = await fetch(`${API}/api/bathy-job/start-batch`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobs: items, resolution: formRes }),
      });
      const d = await r.json();
      if (!r.ok) { log(`Batch start failed: HTTP ${r.status}`, 'error'); return; }
      const errs = (d.errors || []).length;
      log(`Queued ${d.queued ?? items.length} bathymetry job(s) @ ${formRes}${errs ? ` · ${errs} error(s)` : ''}`, errs ? 'error' : 'success');
      setTimeout(() => fetchResults(false), 1500);
    } catch (e) {
      log(`Batch start failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  }, [API, selectedSites, useDrawnRoi, roi, formRes, startSingle, log, fetchResults]);

  // ── R3 row actions ────────────────────────────────────────────────────────
  const handleDownload = useCallback((res, fmt) => {
    // fmt is an optional {fmt,url,label} chosen from the format menu; default
    // to the canonical GeoTIFF (download_url).
    const url = fmt && fmt.url ? fmt.url : res.download_url;
    if (!url) { log(`No file to download for "${res.name}".`, 'error'); return; }
    // Stream as attachment — open in a new tab so the browser saves it.
    window.open(`${API}${url}`, '_blank', 'noopener');
    log(`Downloading ${fmt ? fmt.label : (res.output_name || res.name)}…`, 'info');
  }, [API, log]);

  const handleAnalyse = useCallback(async (res) => {
    try {
      const r = await fetch(`${API}${res.analyse_url}`);
      const d = await r.json();
      if (!r.ok || d.error) { log(`Analyse failed: ${d.error || `HTTP ${r.status}`}`, 'error'); return; }
      setActiveMetricsId(res.id);
      const bbox = Array.isArray(d.roi_bbox) && d.roi_bbox.length >= 4 ? d.roi_bbox : res.roi_bbox;
      // Khalifa-VHR (in-situ-anchored): render the optical_valid mask so the
      // deep basin is visibly "not satellite-measured" (honesty by design).
      const honesty = khalifaHonesty(d) || khalifaHonesty(res);
      if (honesty) {
        window.dispatchEvent(new CustomEvent('showResultMask', { detail: {
          id: res.id, name: res.name, apiBase: API,
          maskUrl: d.mask_url || `/api/results/${res.id}/mask`,
          bbox,
          zOpt: honesty.zOpt, zCollar: honesty.zCollar,
          counts: honesty.mask ? { valid: honesty.mask.valid, collar: honesty.mask.collar, deep: honesty.mask.deep } : null,
        } }));
      } else {
        window.dispatchEvent(new CustomEvent('clearResultMask'));
      }
      // Centre the map on the result ROI (reuse the app-wide zoom event).
      if (Array.isArray(bbox) && bbox.length >= 4) {
        const [w, s, e, n] = bbox;
        window.dispatchEvent(new CustomEvent('zoomToRoi', { detail: { west: w, south: s, east: e, north: n } }));
      }
      if (onAnalyse) onAnalyse(d);
      const m = d.metrics || {};
      const bits = [];
      // CLIENT PORTAL: figures only (depth range + coverage). The accuracy/error
      // metrics (RMSE/R²/CATZOC) are emitted to the log only in internal mode.
      if (m.coverage_pct != null) bits.push(`coverage ${m.coverage_pct}%`);
      if (INTERNAL_ANALYSIS) {
        if (m.rmse_m != null) bits.push(`RMSE ${m.rmse_m} m`);
        if (m.r2 != null) bits.push(`R² ${m.r2}`);
        if (m.catzoc) bits.push(`CATZOC ${m.catzoc}`);
      }
      log(`Analysing "${d.name}"${bits.length ? ` · ${bits.join(' · ')}` : ''}`, 'success');
    } catch (e) {
      log(`Analyse failed: ${e.message}`, 'error');
    }
  }, [API, log, onAnalyse]);

  // Open the full-analysis modal (deep stats + histogram + IHO + 3D seabed).
  const openFullAnalysis = useCallback(async (res) => {
    setAnalysisFor(res);
    setAnalysisData(null);
    setAnalysisError(null);
    setAnalysisLoading(true);
    try {
      const r = await fetch(`${API}${res.analyse_url}`);
      const d = await r.json();
      if (!r.ok || d.error) {
        setAnalysisError(d.error || `HTTP ${r.status}`);
      } else {
        setAnalysisData(d);
      }
    } catch (e) {
      setAnalysisError(e.message || 'request failed');
    } finally {
      setAnalysisLoading(false);
    }
  }, [API]);

  const closeFullAnalysis = useCallback(() => {
    setAnalysisFor(null);
    setAnalysisData(null);
    setAnalysisError(null);
  }, []);

  const handleDelete = useCallback(async (res) => {
    try {
      await fetch(`${API}/api/results/${res.id}`, { method: 'DELETE' });
      setResults(prev => prev.filter(x => x.id !== res.id));
      log(`Removed result "${res.name}".`, 'info');
    } catch (e) {
      log(`Delete failed: ${e.message}`, 'error');
    }
  }, [API, log]);

  // ── "Very HR (Mapbox)" action for a DEFAULT region (task 2) ─────────────────
  // POSTs the result's vhr_url → backend queues a zoom-17 Mapbox re-inference
  // job and returns the honest RGB-upsample caveat. The job lands in Results.
  const handleVeryHr = useCallback(async (res) => {
    const url = res.vhr_url || `/api/results/${res.id}/very-hr`;
    setBusy(true);
    try {
      const r = await fetch(`${API}${url}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const d = await r.json();
      if (!r.ok || d.error) {
        log(`Very HR failed (${res.name}): ${d.error || `HTTP ${r.status}`}`, 'error');
        return;
      }
      const jid = (d.job && d.job.id) || d.job_id || '?';
      log(`Queued High-Resolution (~1 m) for "${res.name}" · job ${jid}.`, 'success');
      // Refresh soon so the VHR job lands in the catalogue.
      setTimeout(() => fetchResults(false), 1500);
    } catch (e) {
      log(`Very HR failed (${res.name}): ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  }, [API, log, fetchResults]);

  // Partition: the 4 precomputed default regions render in their own prominent
  // section; everything else lists in "Finished jobs".
  const defaultRegions = useMemo(
    () => results.filter(r => (r.is_default === true) || (r.metrics && r.metrics.is_default === true)),
    [results],
  );
  const otherResults = useMemo(
    () => results.filter(r => !((r.is_default === true) || (r.metrics && r.metrics.is_default === true))),
    [results],
  );

  const counts = useMemo(() => {
    const done = results.filter(r => r.status === 'done').length;
    const failed = results.filter(r => r.status === 'failed').length;
    return { done, failed, total: results.length };
  }, [results]);

  if (!open) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(15,23,42,0.55)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px',
        backdropFilter: 'blur(3px)',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: 'min(860px, 100%)', maxHeight: '92vh', display: 'flex', flexDirection: 'column',
          background: 'var(--bg-primary)', borderRadius: '14px',
          border: '1px solid var(--border-dim)',
          boxShadow: '0 20px 60px rgba(15,23,42,0.35)',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: 'linear-gradient(135deg, rgba(14,165,233,0.10), rgba(37,99,235,0.06))',
          borderBottom: '1px solid var(--border-dim)',
        }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '14px', fontFamily: 'var(--font-display)', fontWeight: 800, letterSpacing: '0.06em', color: '#1d4ed8' }}>
              📊 BATHYMETRY RESULTS
            </h2>
            <p style={{ margin: '4px 0 0', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
              {counts.done} done · {counts.failed} failed · {counts.total} total {listError ? `· ${listError}` : '· persists across reloads'}
            </p>
          </div>
          <button onClick={onClose} aria-label="Close"
            style={{
              width: '30px', height: '30px', borderRadius: '8px', border: '1px solid var(--border-dim)',
              background: 'var(--bg-secondary)', color: 'var(--text-dim)', cursor: 'pointer',
              fontSize: '14px', fontWeight: 700,
            }}
          >✕</button>
        </div>

        {/* Body */}
        <div style={{ padding: '16px 20px', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>

          {/* R2 — Khalifa Port High-Resolution */}
          <div style={{
            padding: '14px', borderRadius: '10px',
            border: '1px solid rgba(14,165,233,0.30)',
            background: 'linear-gradient(135deg, rgba(14,165,233,0.06), rgba(37,99,235,0.04))',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap',
          }}>
            <div style={{ minWidth: 0 }}>
              <p style={{ margin: 0, fontSize: '12px', fontFamily: 'var(--font-display)', fontWeight: 800, color: '#0f172a' }}>
                Khalifa Port (High-Resolution)
              </p>
              <p style={{ margin: '3px 0 0', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                Generates the 1 m depth product over the Khalifa Port area.
              </p>
            </div>
            <button onClick={launchKhalifaHR} disabled={busy}
              title="Khalifa Port — High-Resolution (1 m) depth product"
              style={{
                padding: '10px 18px', fontSize: '12px', fontFamily: 'var(--font-display)', fontWeight: 800,
                borderRadius: '8px', border: 'none', letterSpacing: '0.04em',
                background: busy ? 'var(--bg-secondary)' : 'linear-gradient(135deg,#0ea5e9,#2563eb)',
                color: busy ? 'var(--text-dim)' : '#fff', cursor: busy ? 'not-allowed' : 'pointer',
                whiteSpace: 'nowrap',
              }}>
              Khalifa Port (High-Resolution)
            </button>
          </div>

          {/* R4 — Compute bathymetry form */}
          <div style={{
            padding: '14px', borderRadius: '10px',
            border: '1px solid var(--border-dim)', background: 'var(--bg-secondary)',
          }}>
            <p style={{ margin: '0 0 10px', fontSize: '11px', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: 'var(--text-secondary)' }}>
              COMPUTE BATHYMETRY
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '10px' }}>
              {KNOWN_SITES.map(s => {
                const on = !!selectedSites[s.key];
                return (
                  <button key={s.key}
                    onClick={() => setSelectedSites(prev => ({ ...prev, [s.key]: !prev[s.key] }))}
                    style={{
                      padding: '6px 12px', fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 700,
                      borderRadius: '6px', cursor: 'pointer',
                      border: on ? '1.5px solid #2563eb' : '1px solid var(--border-dim)',
                      background: on ? 'rgba(37,99,235,0.12)' : 'var(--bg-primary)',
                      color: on ? '#1d4ed8' : 'var(--text-secondary)',
                    }}>
                    {on ? '✓ ' : ''}{s.label}
                  </button>
                );
              })}
              <button
                onClick={() => setUseDrawnRoi(v => !v)}
                title={Array.isArray(roi) && roi.length >= 4 ? 'Use the ROI currently drawn on the map' : 'Draw an ROI on the map first'}
                style={{
                  padding: '6px 12px', fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 700,
                  borderRadius: '6px', cursor: 'pointer',
                  border: useDrawnRoi ? '1.5px solid #7c3aed' : '1px dashed var(--border-dim)',
                  background: useDrawnRoi ? 'rgba(124,58,237,0.12)' : 'var(--bg-primary)',
                  color: useDrawnRoi ? '#6d28d9' : 'var(--text-dim)',
                }}>
                {useDrawnRoi ? '✓ ' : ''}Drawn ROI
              </button>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', gap: '4px', background: 'var(--bg-primary)', borderRadius: '8px', padding: '3px', border: '1px solid var(--border-subtle)' }}>
                {[{ id: '10m', label: '10 m' }, { id: 'vhr', label: 'High-Resolution (~1 m)' }].map(opt => (
                  <button key={opt.id} onClick={() => setFormRes(opt.id)}
                    style={{
                      padding: '6px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer',
                      fontFamily: 'var(--font-mono)', fontSize: '10px', fontWeight: 700,
                      background: formRes === opt.id ? 'linear-gradient(135deg,#0ea5e9,#2563eb)' : 'transparent',
                      color: formRes === opt.id ? '#fff' : 'var(--text-dim)',
                    }}>{opt.label}</button>
                ))}
              </div>
              <button onClick={launchCompute} disabled={busy}
                style={{
                  padding: '9px 18px', fontSize: '11px', fontFamily: 'var(--font-display)', fontWeight: 800,
                  borderRadius: '8px', border: '1px solid #2563eb', letterSpacing: '0.04em',
                  background: busy ? 'var(--bg-secondary)' : '#2563eb',
                  color: busy ? 'var(--text-dim)' : '#fff', cursor: busy ? 'not-allowed' : 'pointer',
                }}>
                {busy ? 'Queueing…' : 'Queue compute'}
              </button>
            </div>
          </div>

          {/* DEFAULT REGIONS — the 4 precomputed standard-date products, shown
              on load WITHOUT drawing an ROI (DEFAULT_REGIONS_LOG.md task 1). */}
          {defaultRegions.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: '8px', gap: '8px', flexWrap: 'wrap' }}>
                <h3 style={{ margin: 0, fontSize: '11px', fontFamily: 'var(--font-display)', fontWeight: 800, letterSpacing: '0.08em', color: '#0369a1' }}>
                  ⭐ REGIONS · {defaultRegions.length}
                </h3>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {defaultRegions.map(res => (
                  <DefaultRegionCard key={res.id} res={res}
                    active={activeMetricsId === res.id}
                    onDownload={(fmt) => handleDownload(res, fmt)}
                    onAnalyse={() => handleAnalyse(res)}
                    onFullAnalysis={() => openFullAnalysis(res)}
                    onVeryHr={() => handleVeryHr(res)}
                    busy={busy} />
                ))}
              </div>
            </div>
          )}

          {/* R3 — durable results list (defaults excluded; they have their own section) */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <h3 style={{ margin: 0, fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.1em', color: '#1d4ed8' }}>
                FINISHED JOBS · {otherResults.length}
              </h3>
              <button onClick={() => fetchResults(true)} disabled={loadingList}
                style={{
                  padding: '5px 10px', fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700,
                  borderRadius: '5px', border: '1px solid var(--border-dim)',
                  background: 'var(--bg-secondary)', color: 'var(--text-dim)', cursor: 'pointer',
                }}>{loadingList ? '…' : '↻ Refresh'}</button>
            </div>

            {otherResults.length === 0 ? (
              <div style={{ padding: '26px', textAlign: 'center', border: '1px dashed var(--border-dim)', borderRadius: '10px' }}>
                <p style={{ margin: 0, fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                  No computed results yet.
                </p>
                <p style={{ margin: '6px 0 0', fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                  Explore the <b>Default Regions</b> above, or use the compute form to queue your own job.
                </p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {otherResults.map(res => (
                  <ResultRow key={res.id} res={res}
                    active={activeMetricsId === res.id}
                    onDownload={(fmt) => handleDownload(res, fmt)}
                    onAnalyse={() => handleAnalyse(res)}
                    onFullAnalysis={() => openFullAnalysis(res)}
                    onDelete={() => handleDelete(res)} />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={{
          padding: '10px 20px', borderTop: '1px solid var(--border-dim)',
          background: 'var(--bg-secondary)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <p style={{ margin: 0, fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
            Results are server-persisted — they survive reloads and backend restarts.
          </p>
          <button onClick={onClose} style={{
            padding: '7px 14px', fontSize: '11px', fontFamily: 'var(--font-display)', fontWeight: 700,
            borderRadius: '6px', border: '1px solid var(--border-dim)',
            background: 'var(--bg-primary)', color: 'var(--text-secondary)', cursor: 'pointer',
          }}>Close</button>
        </div>
      </div>

      {/* Full-analysis modal — opened from a done row's "🔍 Full analysis" / "🧊 3D" */}
      {analysisFor && (
        <FullAnalysisModal
          meta={analysisFor}
          data={analysisData}
          loading={analysisLoading}
          error={analysisError}
          apiBase={API}
          onClose={closeFullAnalysis}
        />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DefaultRegionCard — one of the 4 precomputed standard-date default products.
// Renders: name, standard date, resolution, and EITHER a validated metrics
// table (RMSE / Bias / R² + CATZOC) OR — for abu_al_abyad — a RECONNAISSANCE
// label + honesty card (no fabricated numbers). Actions: Download, Analyse
// (map overlay), Very HR (Mapbox). Numbers are shown VERBATIM from the backend
// — never recomputed or extrapolated.
// ─────────────────────────────────────────────────────────────────────────────
function fmtSigned(v, unit = ' m') {
  if (v == null || Number.isNaN(v)) return '—';
  const s = v > 0 ? '+' : '';
  return `${s}${v}${unit}`;
}

function DefaultRegionCard({ res, active, onDownload, onAnalyse, onFullAnalysis, onVeryHr, busy }) {
  const m = res.metrics || {};
  const card = m.honesty_card || {};
  const isRecon = m.reconnaissance === true || (typeof m.catzoc === 'string' && m.catzoc.toLowerCase().includes('reconnaissance'));
  const stdDate = m.standard_date || '—';
  const resLabel = res.resolution === 'vhr' ? 'High-Resolution (~1 m)' : res.resolution === '10m' ? '10 m' : (res.resolution || '—');
  const formats = downloadFormats(res);
  const zipUrl = zipUrlFor(res);
  const modelCard = readModelCard(res);
  const [showCard, setShowCard] = useState(false);

  const accent = isRecon ? '#b45309' : '#0369a1';
  const accentBg = isRecon ? 'rgba(180,83,9,0.06)' : 'rgba(3,105,161,0.05)';
  const accentBorder = isRecon ? 'rgba(180,83,9,0.30)' : 'rgba(3,105,161,0.28)';

  // CLIENT PORTAL: figures + numbers only — depth range, coverage, area, date.
  // No method/engine, no data source, no RMSE/accuracy, no CATZOC/provenance.
  const range = depthRange(res);
  const cov = fmtCoverage(res);
  const area = fmtArea(res);
  const portalFigs = [
    range ? { label: 'DEPTH', value: `${num(range[0], 1)}–${num(range[1], 1)} m` } : null,
    cov ? { label: 'COVERAGE', value: cov } : null,
    area ? { label: 'AREA', value: area } : null,
    { label: 'DATE', value: stdDate !== '—' ? stdDate : fmtDate(res.date) },
  ].filter(Boolean);

  return (
    <div style={{
      padding: '13px 15px', borderRadius: '11px',
      border: active ? `1.5px solid ${accent}` : `1px solid ${accentBorder}`,
      background: accentBg,
    }}>
      {/* Header line: name + resolution + ROI (no method/provenance badges) */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px', flexWrap: 'wrap', marginBottom: '8px' }}>
        <div style={{ minWidth: 0, flex: 1 }}>
          <span style={{ fontSize: '13px', fontFamily: 'var(--font-display)', fontWeight: 800, color: '#0f172a' }}>
            {res.name || 'Region'}
          </span>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '5px' }}>
            <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700, padding: '2px 7px', borderRadius: '5px', background: 'rgba(37,99,235,0.12)', color: '#1d4ed8' }}>{resLabel}</span>
            <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700, padding: '2px 7px', borderRadius: '5px', background: 'rgba(15,23,42,0.07)', color: '#334155' }}>ROI {fmtRoi(res.roi_bbox)}</span>
          </div>
        </div>
      </div>

      {/* CLIENT PORTAL figures — depth range / coverage / area / date only.
          (Reconnaissance regions render with NO numbers — same as before.) */}
      {!isRecon && portalFigs.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(4, portalFigs.length)}, 1fr)`, gap: '6px' }}>
          {portalFigs.map(c => (
            <div key={c.label} style={{ padding: '7px 8px', borderRadius: '6px', background: 'rgba(255,255,255,0.6)', border: '1px solid var(--border-dim)', textAlign: 'center' }}>
              <div style={{ fontSize: '8.5px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-dim)', letterSpacing: '0.06em' }}>{c.label}</div>
              <div style={{ fontSize: '12px', fontFamily: 'var(--font-display)', fontWeight: 800, color: '#0f172a', marginTop: '2px' }}>{c.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* INTERNAL-ONLY: full metrics + honesty card (gated; hidden on portal) */}
      {INTERNAL_ANALYSIS && (
        isRecon ? (
          <ReconnaissanceCard m={m} card={card} />
        ) : (
          <DefaultMetricsTable m={m} card={card} modelCard={modelCard} />
        )
      )}
      {INTERNAL_ANALYSIS && card && (card.headline || card.catzoc_note) && (
        <div style={{ marginTop: '8px' }}>
          {card.headline && (
            <p style={{ margin: 0, fontSize: '10px', fontFamily: 'var(--font-mono)', color: '#475569', lineHeight: 1.45 }}>
              {card.headline}
            </p>
          )}
          <button onClick={() => setShowCard(s => !s)}
            style={{ marginTop: '6px', padding: '3px 8px', fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700, borderRadius: '5px', border: `1px solid ${accentBorder}`, background: 'transparent', color: accent, cursor: 'pointer' }}>
            {showCard ? '▴ Hide honesty card' : '▾ Honesty card'}
          </button>
          {showCard && <DefaultHonestyCard card={card} m={m} accent={accent} />}
        </div>
      )}

      {/* Actions: Download · Analyse (map) · High-Resolution */}
      <div style={{ display: 'flex', gap: '6px', marginTop: '10px', flexWrap: 'wrap' }}>
        <button onClick={() => onDownload(formats[0])} disabled={formats.length === 0}
          title="Download the depth GeoTIFF"
          style={{
            flex: '1 1 110px', padding: '7px', fontSize: '10px', fontFamily: 'var(--font-display)', fontWeight: 700,
            borderRadius: '5px', letterSpacing: '0.4px',
            border: '1px solid ' + (formats.length ? '#0ea5e9' : 'var(--border-dim)'),
            background: formats.length ? 'rgba(14,165,233,0.10)' : 'var(--bg-primary)',
            color: formats.length ? '#0369a1' : 'var(--text-dim)',
            cursor: formats.length ? 'pointer' : 'not-allowed',
          }}>⬇ Download</button>
        {zipUrl && (
          <button onClick={() => onDownload({ fmt: 'zip', url: zipUrl, label: 'all TIFFs (.zip)' })}
            title="Download EVERY GeoTIFF this product references (depth + companions + mask) in one .zip, with a MANIFEST"
            style={{
              flex: '1 1 130px', padding: '7px', fontSize: '10px', fontFamily: 'var(--font-display)', fontWeight: 800,
              borderRadius: '5px', letterSpacing: '0.4px',
              border: '1px solid #0891b2', background: 'rgba(8,145,178,0.14)', color: '#0e7490', cursor: 'pointer',
            }}>🗂 All TIFFs (.zip)</button>
        )}
        <button onClick={onAnalyse}
          title="Overlay this region's depth on the map and centre on its ROI"
          style={{
            flex: '1 1 90px', padding: '7px', fontSize: '10px', fontFamily: 'var(--font-display)', fontWeight: 700,
            borderRadius: '5px', letterSpacing: '0.4px',
            border: '1px solid #2563eb', background: 'rgba(37,99,235,0.10)', color: '#1d4ed8', cursor: 'pointer',
          }}>🗺 Analyse</button>
        <button onClick={onFullAnalysis}
          title="Depth distribution, statistics + 3D seabed"
          style={{
            flex: '1 1 90px', padding: '7px', fontSize: '10px', fontFamily: 'var(--font-display)', fontWeight: 700,
            borderRadius: '5px', letterSpacing: '0.4px',
            border: '1px solid #2563eb', background: '#2563eb', color: '#fff', cursor: 'pointer',
          }}>🔍 Details · 🧊 3D</button>
        <button onClick={onVeryHr} disabled={busy}
          title="High-Resolution: ~1 m product for added spatial detail"
          style={{
            flex: '1 1 120px', padding: '7px', fontSize: '10px', fontFamily: 'var(--font-display)', fontWeight: 700,
            borderRadius: '5px', letterSpacing: '0.4px',
            border: '1px solid #7c3aed',
            background: busy ? 'var(--bg-secondary)' : 'rgba(124,58,237,0.10)',
            color: busy ? 'var(--text-dim)' : '#6d28d9',
            cursor: busy ? 'not-allowed' : 'pointer',
          }}>🛰️ High-Resolution</button>
      </div>
      <p style={{ margin: '6px 0 0', fontSize: '8.5px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', lineHeight: 1.4 }}>
        High-Resolution = ~1 m product for added spatial detail.
      </p>
    </div>
  );
}

function DefaultMetricsTable({ m, card, modelCard }) {
  // Honest metric source: prefer the §4 model-card metrics when present (DL
  // product numbers measured on the DL product); fall back to the legacy stamp.
  const mc = modelCard && modelCard.present ? modelCard.metrics : null;
  const rmse = mc && mc.rmse != null ? mc.rmse : m.rmse_m;
  const bias = mc && mc.bias != null ? mc.bias : m.bias_m;
  const r2Suppressed = mc ? mc.r2Suppressed : (m.r2_suppressed === true || m.single_mode === true);
  const r2Val = mc && !mc.r2Suppressed ? mc.r2 : (mc ? null : m.r2);
  // R² shown ONLY as a full-range value, OR "suppressed" for single-mode bands
  // (never a per-2 m-band R²).
  const r2Cell = r2Suppressed
    ? { label: 'R²', value: 'suppressed', color: '#b45309', tip: 'Full-range R² suppressed — narrow depth range (single-mode); judge by RMSE + decile slope.' }
    : { label: 'R²', value: r2Val != null ? `${num(r2Val, 3)}` : '–', color: '#059669', tip: 'Full-range R² under spatial-block CV.' };
  const cells = [
    { label: 'RMSE', value: rmse != null ? `${num(rmse, 2)} m` : '–', color: '#dc2626' },
    { label: 'Bias', value: fmtSigned(bias != null ? Number(num(bias, 2)) : bias), color: '#334155' },
    r2Cell,
    { label: 'CATZOC', value: (modelCard && modelCard.catzoc) || m.catzoc || '–', color: '#2563eb' },
  ];
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
        {cells.map(c => (
          <div key={c.label} title={c.tip || undefined} style={{ padding: '7px 8px', borderRadius: '6px', background: 'rgba(255,255,255,0.6)', border: '1px solid var(--border-dim)', textAlign: 'center' }}>
            <div style={{ fontSize: '8.5px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-dim)', letterSpacing: '0.06em' }}>{c.label}</div>
            <div style={{ fontSize: c.value === 'suppressed' ? '10px' : '13px', fontFamily: 'var(--font-display)', fontWeight: 800, color: c.color, marginTop: c.value === 'suppressed' ? '4px' : '2px' }}>{c.value}</div>
          </div>
        ))}
      </div>
      {(m.datum || m.ground_truth || m.train_provenance) && (
        <p style={{ margin: '6px 0 0', fontSize: '8.5px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', lineHeight: 1.4 }}>
          {m.datum && <>Datum {m.datum}. </>}
          {m.train_provenance && <><b>{m.train_provenance}</b>. </>}
          {m.ground_truth && <>GT: {m.ground_truth}.</>}
        </p>
      )}
    </div>
  );
}

function ReconnaissanceCard({ m, card }) {
  const sources = m.sources || {};
  const cal = m.calibration || {};
  const ice = sources.icesat2_sliderule || {};
  const ib = sources.iboating || {};
  return (
    <div style={{ padding: '9px 11px', borderRadius: '8px', background: 'rgba(180,83,9,0.07)', border: '1px solid rgba(180,83,9,0.22)' }}>
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '6px' }}>
        <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 800, padding: '2px 7px', borderRadius: '5px', background: 'rgba(180,83,9,0.16)', color: '#92400e' }}>RMSE / Bias / R² — none (no in-situ)</span>
        {m.catzoc && <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 800, padding: '2px 7px', borderRadius: '5px', background: 'rgba(15,23,42,0.07)', color: '#334155' }}>CATZOC {m.catzoc}</span>}
        {m.z_opt_m != null && <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700, padding: '2px 7px', borderRadius: '5px', background: 'rgba(15,23,42,0.07)', color: '#334155' }}>z_opt {m.z_opt_m} m</span>}
      </div>
      <p style={{ margin: 0, fontSize: '10px', fontFamily: 'var(--font-mono)', color: '#7c2d12', lineHeight: 1.45 }}>
        {card.headline || 'RECONNAISSANCE estimate — no held-out in-situ, so no RMSE/Bias/R² is reported.'}
      </p>
      {/* Sources used + per-source flag */}
      <div style={{ marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '3px', fontSize: '9px', fontFamily: 'var(--font-mono)', color: '#475569' }}>
        <div>• <b>S2 optical prior</b> {sources.optical_prior && sources.optical_prior.uncalibrated ? '(uncalibrated — depth shape only)' : ''}</div>
        {ice && (ice.n_in_bbox != null || ice.n_photon_px != null) && (
          <div>• <b>ICESat-2 ATL24/SlideRule</b> photons{ice.n_in_bbox != null ? ` n≈${ice.n_in_bbox} in bbox` : ''}{ice.n_photon_px != null ? `, ${ice.n_photon_px} px owned` : ''} — shallow zone</div>
        )}
        {ib && ib.used === false && (
          <div>• <b>i-Boating DROPPED</b>{ib.n != null ? ` (n=${ib.n})` : ''} — {ib.reason || 'datum-unreconcilable (S12)'}</div>
        )}
        {sources.gebco && (
          <div>• <b>GEBCO floor</b> — {sources.gebco.available ? 'applied below z_opt' : (sources.gebco.note || 'unavailable; deep gaps stay nodata (not extrapolated)')}</div>
        )}
        {(cal.anchor_applied === false) && (
          <div>• <b>ATL24 affine anchor REJECTED</b> (corr {cal.fit_corr} &lt; gate {cal.corr_gate}) — no fabricated relationship applied.</div>
        )}
      </div>
      {/* Per-pixel source legend */}
      {card.source_legend && (
        <div style={{ marginTop: '6px', display: 'flex', gap: '8px', flexWrap: 'wrap', fontSize: '8.5px', fontFamily: 'var(--font-mono)', color: '#475569' }}>
          {Object.entries(card.source_legend).map(([k, v]) => (
            <span key={k}>[{k}] {v}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function DefaultHonestyCard({ card, m, accent }) {
  const lines = [];
  if (card.catzoc_note) lines.push(['CATZOC', card.catzoc_note]);
  if (card.error_attribution) lines.push(['Error attribution', card.error_attribution]);
  if (card.r2_caveat) lines.push(['R² caveat', card.r2_caveat]);
  if (card.train_caveat) lines.push(['Training', card.train_caveat]);
  if (card.bias_note) lines.push(['Bias', card.bias_note]);
  if (card.anchor_decision) lines.push(['Anchor decision', card.anchor_decision]);
  if (card.iboating_decision) lines.push(['i-Boating', card.iboating_decision]);
  if (card.comparability) lines.push(['Comparability', card.comparability]);
  if (card.vhr_caveat) lines.push(['Very HR', card.vhr_caveat]);
  return (
    <div style={{ marginTop: '6px', padding: '9px 11px', borderRadius: '8px', background: 'rgba(255,255,255,0.65)', border: `1px solid ${accent}33`, display: 'flex', flexDirection: 'column', gap: '5px' }}>
      {lines.map(([k, v]) => (
        <div key={k} style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: '#475569', lineHeight: 1.45 }}>
          <b style={{ color: accent }}>{k}:</b> {v}
        </div>
      ))}
    </div>
  );
}

function ResultRow({ res, active, onDownload, onAnalyse, onFullAnalysis, onDelete }) {
  const isDone = res.status === 'done';
  const isFailed = res.status === 'failed';
  const statusColor = isDone ? '#065f46' : isFailed ? '#991b1b' : '#4338ca';
  const statusBg = isDone ? 'rgba(5,150,105,0.18)' : isFailed ? 'rgba(220,38,38,0.16)' : 'rgba(79,70,229,0.18)';
  const m = res.metrics || {};
  const resLabel = res.resolution === 'vhr' ? 'High-Resolution (~1 m)' : res.resolution === '10m' ? '10 m' : (res.resolution || '—');
  const honesty = khalifaHonesty(res);
  const formats = downloadFormats(res);
  const zipUrl = zipUrlFor(res);
  const modelCard = readModelCard(res);
  const [fmtMenuOpen, setFmtMenuOpen] = useState(false);
  // CLIENT PORTAL figures (no method/source/accuracy).
  const range = depthRange(res);
  const cov = fmtCoverage(res);

  return (
    <div style={{
      padding: '12px 14px', borderRadius: '10px',
      border: active ? '1.5px solid #2563eb' : '1px solid var(--border-dim)',
      background: active ? 'rgba(37,99,235,0.05)' : 'var(--bg-secondary)',
    }}>
      <div
        onClick={isDone ? onFullAnalysis : undefined}
        title={isDone ? 'Open full analysis + 3D seabed' : undefined}
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', gap: '8px', flexWrap: 'wrap', cursor: isDone ? 'pointer' : 'default' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0, flex: 1 }}>
          <span style={{ fontSize: '12px', fontFamily: 'var(--font-display)', fontWeight: 800, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {res.name || 'Bathymetry result'}
          </span>
          <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700, padding: '2px 7px', borderRadius: '5px', background: 'rgba(37,99,235,0.12)', color: '#1d4ed8', whiteSpace: 'nowrap' }}>
            {resLabel}
          </span>
          {INTERNAL_ANALYSIS && <MethodBadge card={modelCard} />}
        </div>
        <span style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700, padding: '3px 7px', borderRadius: '5px', background: statusBg, color: statusColor, letterSpacing: '0.05em' }}>
          {(res.status || '').toUpperCase()}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginBottom: '8px' }}>
        <div><span>ROI </span><b style={{ color: 'var(--text-primary)' }}>{fmtRoi(res.roi_bbox)}</b></div>
        <div><span>Date </span><b style={{ color: 'var(--text-primary)' }}>{fmtDate(res.date)}</b></div>
        <div><span>File </span><b style={{ color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'inline-block', maxWidth: '100%', verticalAlign: 'bottom' }}>{res.output_name || '—'}</b></div>
        <div><span>Size </span><b style={{ color: 'var(--text-primary)' }}>{fmtSize(res.size_bytes)}</b></div>
      </div>

      {isFailed && res.error && (
        <p style={{ margin: '0 0 8px', fontSize: '10px', fontFamily: 'var(--font-mono)', color: '#991b1b', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          Error: {String(res.error).slice(0, 160)}
        </p>
      )}

      {/* CLIENT PORTAL figures (depth range / coverage) — no accuracy metrics. */}
      {isDone && (range || cov) && (
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', padding: '8px 10px', marginBottom: '8px', borderRadius: '6px', background: 'rgba(37,99,235,0.06)', border: '1px solid rgba(37,99,235,0.20)', fontSize: '10px', fontFamily: 'var(--font-mono)' }}>
          {range && <div>Depth <b style={{ color: '#0f172a' }}>{num(range[0], 1)}–{num(range[1], 1)} m</b></div>}
          {cov && <div>Coverage <b style={{ color: '#0f172a' }}>{cov}</b></div>}
        </div>
      )}

      {/* INTERNAL-ONLY: honesty banner + full accuracy metrics (gated). */}
      {INTERNAL_ANALYSIS && <KhalifaHonestyBanner h={honesty} />}
      {INTERNAL_ANALYSIS && (active && isDone) && (
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', padding: '8px 10px', marginBottom: '8px', borderRadius: '6px', background: 'rgba(37,99,235,0.06)', border: '1px solid rgba(37,99,235,0.20)', fontSize: '10px', fontFamily: 'var(--font-mono)' }}>
          {m.rmse_m != null && <div>RMSE <b style={{ color: '#dc2626' }}>{m.rmse_m} m</b></div>}
          {(modelCard && modelCard.present && modelCard.metrics.r2Suppressed)
            ? <div title="Full-range R² suppressed — single-mode (narrow range); judge by RMSE + decile slope.">R² <b style={{ color: '#b45309' }}>suppressed</b></div>
            : (m.r2 != null && <div>R² <b style={{ color: '#059669' }}>{m.r2}</b></div>)}
          {m.mae_m != null && <div>MAE <b style={{ color: 'var(--text-primary)' }}>{m.mae_m} m</b></div>}
          {m.bias_m != null && <div>Bias <b style={{ color: 'var(--text-primary)' }}>{m.bias_m} m</b></div>}
          {m.coverage_pct != null && <div>Coverage <b style={{ color: 'var(--text-primary)' }}>{m.coverage_pct}%</b></div>}
          {m.catzoc && <div>CATZOC <b style={{ color: '#2563eb' }}>{m.catzoc}</b></div>}
          {m.n_test != null && <div>n <b style={{ color: 'var(--text-primary)' }}>{m.n_test}</b></div>}
        </div>
      )}

      <div style={{ display: 'flex', gap: '6px' }}>
        {/* Download — single button when only the GeoTIFF exists; a small
            format menu (GeoTIFF / CSV / GeoJSON / mask) when the backend
            exposes more. Honest: only formats the backend actually serves. */}
        <div style={{ flex: 1, position: 'relative' }}>
          <button
            onClick={() => { if (formats.length <= 1) { onDownload(formats[0]); } else { setFmtMenuOpen(o => !o); } }}
            disabled={formats.length === 0}
            title={formats.length > 1 ? `Download (${formats.length} formats)` : 'Download the GeoTIFF'}
            style={{
              width: '100%', padding: '7px', fontSize: '10px', fontFamily: 'var(--font-display)', fontWeight: 700,
              borderRadius: '5px', letterSpacing: '0.4px',
              border: '1px solid ' + (formats.length ? '#0ea5e9' : 'var(--border-dim)'),
              background: formats.length ? 'rgba(14,165,233,0.10)' : 'var(--bg-primary)',
              color: formats.length ? '#0369a1' : 'var(--text-dim)',
              cursor: formats.length ? 'pointer' : 'not-allowed',
            }}>⬇ Download{formats.length > 1 ? ` ▾ (${formats.length})` : ''}</button>
          {fmtMenuOpen && formats.length > 1 && (
            <div style={{ position: 'absolute', bottom: 'calc(100% + 4px)', left: 0, right: 0, zIndex: 20,
              background: 'var(--bg-primary)', border: '1px solid #0ea5e9', borderRadius: 6,
              boxShadow: '0 8px 24px rgba(15,23,42,0.25)', overflow: 'hidden' }}>
              {formats.map(f => (
                <button key={f.fmt}
                  onClick={() => { onDownload(f); setFmtMenuOpen(false); }}
                  style={{ display: 'block', width: '100%', textAlign: 'left', padding: '7px 10px',
                    fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 600,
                    border: 'none', borderBottom: '1px solid var(--border-subtle, #eef2f7)',
                    background: 'transparent', color: '#0369a1', cursor: 'pointer' }}>
                  ⬇ {f.label}
                </button>
              ))}
            </div>
          )}
        </div>
        <button onClick={onAnalyse} disabled={!isDone}
          title={'Overlay this result on the map and centre on its area'}
          style={{
            flex: 1, padding: '7px', fontSize: '10px', fontFamily: 'var(--font-display)', fontWeight: 700,
            borderRadius: '5px', letterSpacing: '0.4px',
            border: '1px solid ' + (isDone ? '#2563eb' : 'var(--border-dim)'),
            background: isDone ? 'rgba(37,99,235,0.10)' : 'var(--bg-primary)',
            color: isDone ? '#1d4ed8' : 'var(--text-dim)',
            cursor: isDone ? 'pointer' : 'not-allowed',
          }}>🗺 Map</button>
        <button onClick={onFullAnalysis} disabled={!isDone}
          title="Depth distribution, statistics + 3D seabed"
          style={{
            flex: 1.4, padding: '7px', fontSize: '10px', fontFamily: 'var(--font-display)', fontWeight: 700,
            borderRadius: '5px', letterSpacing: '0.4px',
            border: '1px solid ' + (isDone ? '#2563eb' : 'var(--border-dim)'),
            background: isDone ? '#2563eb' : 'var(--bg-primary)',
            color: isDone ? '#fff' : 'var(--text-dim)',
            cursor: isDone ? 'pointer' : 'not-allowed',
          }}>🔍 Details · 🧊 3D</button>
        <button onClick={onDelete}
          style={{
            padding: '7px 12px', fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 700,
            borderRadius: '5px', border: '1px solid var(--border-dim)',
            background: 'var(--bg-primary)', color: 'var(--text-dim)', cursor: 'pointer',
          }}>Remove</button>
      </div>
      {/* Download all TIFFs — every raster this result references (depth +
          companions + mask/σ + MLE per-scene set) in one .zip with a MANIFEST. */}
      {isDone && zipUrl && (
        <button onClick={() => onDownload({ fmt: 'zip', url: zipUrl, label: 'all TIFFs (.zip)' })}
          title="Download EVERY GeoTIFF this result references in one .zip (with a MANIFEST listing what was included / missing)"
          style={{
            width: '100%', marginTop: '6px', padding: '7px', fontSize: '10px',
            fontFamily: 'var(--font-display)', fontWeight: 800, letterSpacing: '0.4px',
            borderRadius: '5px', border: '1px solid #0891b2',
            background: 'rgba(8,145,178,0.14)', color: '#0e7490', cursor: 'pointer',
          }}>🗂 Download all TIFFs (.zip)</button>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FullAnalysisModal — the per-result deep dive opened from a done row.
//
// Fed by GET /api/results/<id>/analyse. Designed to the enriched schema the
// backend is producing in parallel:
//   { id, name, roi_bbox, resolution, date, units?,
//     depth_stats:{min,max,mean,median,std,p5,p25,p50,p75,p95,area_km2?},
//     histogram:{bin_edges:[...], counts:[...]},
//     per_band:[{from,to,count,pct}] | coverage_bands,
//     iho:{rmse_m,r2,bias_m,mae_m,n_test,order,catzoc,note},
//     grid:{values(2D or row-major), bbox, nrows, ncols, nodata, z_min, z_max} }
// Every field is read defensively (falls back to the legacy `metrics` block and
// to roi_bbox), so the modal degrades gracefully before the backend ships the
// full payload — it never fabricates numbers it wasn't given.
// ─────────────────────────────────────────────────────────────────────────────

const BAND_COLORS = [
  '#bae6fd', '#7dd3fc', '#38bdf8', '#0ea5e9', '#0284c7',
  '#0369a1', '#075985', '#0c4a6e', '#082f49', '#041f33',
];

function num(v, dp = 2) {
  if (v == null || !Number.isFinite(Number(v))) return '—';
  return Number(v).toFixed(dp);
}

function pick(obj, ...keys) {
  for (const k of keys) {
    if (obj && obj[k] != null) return obj[k];
  }
  return undefined;
}

// ─────────────────────────────────────────────────────────────────────────────
// Model-card / provenance (DL_WEB_LOG §4). The backend attaches an auditable
// `metrics.model_card` (a.k.a. `provenance`) block to every result; we render
// it on the Analyse/Full view and a compact method badge on each card. Every
// field is read defensively — the panel degrades gracefully (renders only what
// is present) before the backend ships the full block, and NEVER fabricates a
// number. Method is normalised to dl | lyzenga | reconnaissance so the user
// always knows whether a result is a DL retrieval, a Lyzenga product, or a
// reconnaissance estimate — DL accuracy is never implied on a non-DL product.
// ─────────────────────────────────────────────────────────────────────────────
function readModelCard(res) {
  if (!res) return null;
  const m = res.metrics || {};
  const mc = m.model_card || m.provenance || res.model_card || res.provenance || null;
  // Region-training stamp (always present on the 4 defaults per the prior round).
  const rt = m.region_training || {};
  const accuracyKind = m.accuracy_kind || rt.accuracy_kind || (mc && mc.accuracy_kind) || null;

  // Method classification. Prefer an explicit method string from the card, else
  // infer from the legacy stamps. recon wins (no metric attribution).
  const methodStr = (mc && mc.method) || m.method || '';
  const ml = String(methodStr).toLowerCase();
  const isRecon = (m.reconnaissance === true)
    || accuracyKind === 'reconnaissance'
    || ml.includes('reconnaissance') || ml.includes('recon')
    || (typeof m.catzoc === 'string' && m.catzoc.toLowerCase().includes('reconnaissance'));
  let kind;
  if (isRecon) kind = 'reconnaissance';
  else if (ml.includes('dl') || ml.includes('patchcnn') || ml.includes('cnn') || ml.includes('deep')) kind = 'dl';
  else if (ml.includes('lyzenga') || ml.includes('stumpf') || ml.includes('fallback')) kind = 'lyzenga';
  else kind = mc ? 'dl' : 'unknown';   // a present model_card implies the DL product

  if (!mc) {
    // No explicit §4 block — return a minimal descriptor so the badge + a short
    // honest method label still render from the legacy stamps.
    if (kind === 'unknown') return null;
    return { kind, present: false, method: methodStr || null, accuracyKind, regionTraining: rt.region_label || rt.region_key || null };
  }

  const unc = mc.uncertainty || {};
  const met = mc.metrics || {};
  const bsg = mc.band_shuffle_guard || mc.band_shuffle || null;
  const iho = mc.iho_s44 || mc.iho || {};
  // Honest R²: full-range only. A backend "suppressed" marker (or an explicit
  // null + single-mode flag) means we show "R² suppressed", never a per-band R².
  const r2Raw = pick(met, 'full_range_r2', 'r2');
  const r2Suppressed = met.r2_suppressed === true
    || met.full_range_r2 === 'suppressed'
    || met.single_mode === true
    || (typeof r2Raw === 'string' && String(r2Raw).toLowerCase().includes('suppress'));

  return {
    kind, present: true,
    method: mc.method || methodStr || null,
    modelVersion: mc.model_version || null,
    architecture: mc.architecture || null,
    engineFile: mc.engine_file || null,
    inputBands: Array.isArray(mc.input_bands) ? mc.input_bands : null,
    derivedFeatures: Array.isArray(mc.derived_features) ? mc.derived_features : null,
    trainingProvenance: mc.training_provenance || null,
    regionTraining: mc.region_training || rt.region_label || rt.region_key || null,
    resolutionM: mc.resolution_m ?? null,
    dateProcessed: mc.date_processed || null,
    sceneDates: Array.isArray(mc.s2_scene_dates) ? mc.s2_scene_dates : null,
    datum: mc.datum || null,
    maxDepthM: mc.max_depth_m ?? null,
    accuracyKind,
    sigma: {
      mean: pick(unc, 'sigma_mean_m'),
      calibration: unc.sigma_calibration || null,
      cov95Target: pick(unc, 'coverage_95_target'),
      cov95Observed: pick(unc, 'coverage_95_observed'),
      cov95Raw: pick(unc, 'coverage_95_raw'),
    },
    // ITEM 1 — Total Propagated Uncertainty (not just model σ).
    tpu: (unc.tpu_m != null) ? {
      tpuM: pick(unc, 'tpu_m'),
      uModel: pick(unc, 'u_model_sigma_mean_m'),
      uDatum: pick(unc, 'u_datum_m'),
      uTide: pick(unc, 'u_tide_m'),
      uRefraction: pick(unc, 'u_refraction_m'),
      uCoreg: pick(unc, 'u_coreg_m'),
      tidalRangeM: pick(unc, 'tidal_range_m'),
      tideAssumedZero: unc.u_tide_assumed_zero === true,
      note: unc.tpu_note || null,
    } : null,
    // ITEM 5 — datum transform provenance.
    datumInfo: mc.datum_info || null,
    // ITEM 4 — shoal-bias safety flag.
    shoalSafe: (mc.shoal_safe === true || mc.shoal_safe === false) ? mc.shoal_safe : null,
    shoalWarning: mc.shoal_warning || null,
    metrics: {
      split: met.split || null,
      nTest: pick(met, 'n_test'),
      rmse: pick(met, 'rmse_m'),
      bias: pick(met, 'bias_m'),
      r2: r2Suppressed ? null : (Number.isFinite(Number(r2Raw)) ? Number(r2Raw) : null),
      r2Suppressed,
      decileSlope: pick(met, 'decile_slope'),
    },
    bandShuffle: bsg ? {
      orig: pick(bsg, 'rmse_orig_m', 'rmse_orig'),
      shuffled: pick(bsg, 'rmse_shuffled_m', 'rmse_shuffled'),
      deltaPct: pick(bsg, 'delta_pct'),
      verdict: bsg.verdict || null,
      pass: (bsg.verdict ? /skill present|pass/i.test(bsg.verdict) : null)
        ?? (pick(bsg, 'delta_pct') != null ? Number(pick(bsg, 'delta_pct')) > 10 : null),
    } : null,
    iho: {
      order1a: pick(iho, 'Order_1a_pct', 'order1a_pct', 'order_1a_pct'),
      order2: pick(iho, 'Order_2_pct', 'order2_pct', 'order_2_pct'),
      orderLabel: iho.order_label || null,
      p95: pick(iho, 'p95_error_m'),
      tvuEval: pick(iho, 'tvu_at_eval_m'),
      passCriterion: iho.pass_criterion || null,
      catzoc: iho.catzoc || null,
    },
    catzoc: mc.catzoc || met.catzoc || null,
    limitationFlag: mc.limitation_flag || null,
    dataSources: Array.isArray(mc.data_sources) ? mc.data_sources : null,
    disclaimer: mc.disclaimer || 'Reconnaissance-grade SDB. Not to be used for navigation.',
  };
}

// Visual identity per method kind (badge colour + short/long label).
const METHOD_STYLE = {
  dl:             { label: 'DL', long: 'Deep-learning retrieval', color: '#6d28d9', bg: 'rgba(124,58,237,0.14)', border: 'rgba(124,58,237,0.45)' },
  lyzenga:        { label: 'Lyzenga', long: 'Lyzenga/Stumpf physics', color: '#0369a1', bg: 'rgba(3,105,161,0.12)', border: 'rgba(3,105,161,0.40)' },
  reconnaissance: { label: 'RECON', long: 'Reconnaissance estimate', color: '#92400e', bg: 'rgba(180,83,9,0.16)', border: 'rgba(180,83,9,0.42)' },
  unknown:        { label: 'SDB', long: 'Satellite-derived bathymetry', color: '#334155', bg: 'rgba(15,23,42,0.07)', border: 'var(--border-dim)' },
};

// Compact method badge for a result card. Always visible so the method is never
// ambiguous. For DL it prints the model version when available.
function MethodBadge({ card }) {
  if (!card) return null;
  const st = METHOD_STYLE[card.kind] || METHOD_STYLE.unknown;
  const ver = card.kind === 'dl' && card.modelVersion ? ` · ${card.modelVersion}` : '';
  const title = card.method
    ? `${card.method}${card.modelVersion ? ` (${card.modelVersion})` : ''}`
    : st.long;
  return (
    <span title={title}
      style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 800, padding: '2px 7px',
        borderRadius: '5px', background: st.bg, color: st.color, border: `1px solid ${st.border}`,
        letterSpacing: '0.04em', whiteSpace: 'nowrap', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <span>{card.kind === 'dl' ? '◆' : card.kind === 'lyzenga' ? '∿' : card.kind === 'reconnaissance' ? '◷' : '◇'}</span>
      {st.label}{ver}
    </span>
  );
}

// Band-shuffle PASS/FAIL badge (DL leakage guard). Honest: only renders when
// the backend ships the guard; PASS means shuffling the input bands degraded
// RMSE > 10% (spectral depth skill present, not bottom-type/texture).
function BandShuffleBadge({ bs }) {
  if (!bs || (bs.deltaPct == null && bs.pass == null)) return null;
  const pass = bs.pass !== false;
  const color = pass ? '#065f46' : '#991b1b';
  const bg = pass ? 'rgba(5,150,105,0.15)' : 'rgba(220,38,38,0.14)';
  const delta = bs.deltaPct != null ? `+${num(bs.deltaPct, 0)}%` : '';
  return (
    <span title={bs.orig != null && bs.shuffled != null
      ? `Band-shuffle leakage guard: RMSE ${num(bs.orig, 2)} → ${num(bs.shuffled, 2)} m (${delta}). ${bs.verdict || (pass ? 'spectral depth skill present' : 'leakage suspect')}`
      : (bs.verdict || 'Band-shuffle leakage guard')}
      style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 800, padding: '2px 7px',
        borderRadius: '5px', background: bg, color, letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>
      {pass ? '✓ BAND-SHUFFLE PASS' : '✗ BAND-SHUFFLE FAIL'}{delta ? ` ${delta}` : ''}
    </span>
  );
}

// Limitation-flag chip (data-/algorithm-/physics-limited).
function LimitationChip({ flag }) {
  if (!flag) return null;
  const f = String(flag).toLowerCase();
  const color = f.includes('physics') ? '#b91c1c' : f.includes('algorithm') ? '#b45309' : '#0e7490';
  return (
    <span title="Honest limitation attribution for this product"
      style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700, padding: '2px 7px',
        borderRadius: '5px', background: `${color}1f`, color, border: `1px solid ${color}55`,
        letterSpacing: '0.03em' }}>
      ⚑ {flag}
    </span>
  );
}

// Normalise the depth-stats block across naming variants.
function readStats(d) {
  const s = d?.depth_stats || d?.stats || {};
  const m = d?.metrics || {};
  return {
    min:    pick(s, 'min', 'min_depth', 'depth_min'),
    max:    pick(s, 'max', 'max_depth', 'depth_max'),
    mean:   pick(s, 'mean', 'avg'),
    median: pick(s, 'median', 'p50'),
    std:    pick(s, 'std', 'stddev', 'sigma'),
    p5:     pick(s, 'p5', 'pct5'),
    p25:    pick(s, 'p25', 'pct25'),
    p50:    pick(s, 'p50', 'median'),
    p75:    pick(s, 'p75', 'pct75'),
    p95:    pick(s, 'p95', 'pct95'),
    area:   pick(s, 'area_km2', 'area') ?? pick(m, 'area_km2'),
    coverage: pick(s, 'coverage_pct') ?? pick(m, 'coverage_pct'),
  };
}

// Normalise the IHO/CATZOC block (falls back to the legacy `metrics`).
function readIho(d) {
  const i = d?.iho || {};
  const m = d?.metrics || {};
  return {
    rmse:  pick(i, 'rmse_m', 'rmse') ?? pick(m, 'rmse_m'),
    r2:    pick(i, 'r2') ?? pick(m, 'r2'),
    bias:  pick(i, 'bias_m', 'bias') ?? pick(m, 'bias_m'),
    mae:   pick(i, 'mae_m', 'mae') ?? pick(m, 'mae_m'),
    n:     pick(i, 'n_test', 'n') ?? pick(m, 'n_test'),
    order: pick(i, 'order', 'iho_order'),
    catzoc: pick(i, 'catzoc') ?? pick(m, 'catzoc'),
    note:  pick(i, 'note', 'caveat'),
    reconnaissance: pick(i, 'reconnaissance'),
    evalDepth: pick(i, 'eval_depth_m'),
    tvu: pick(i, 'tvu_at_median_m'),
    ordersTable: Array.isArray(i?.orders_table) ? i.orders_table : null,
  };
}

// ── INSTRUCTIVE LAYER ───────────────────────────────────────────────────────
// Plain-language, number-driven interpretations + quality colour-coding so a
// non-expert understands what each metric means. Nothing is fabricated — every
// sentence is generated from the actual value, and falls silent when null.

const Q_GOOD = '#059669', Q_OK = '#b45309', Q_POOR = '#dc2626', Q_NEUTRAL = '#475569';

// CATZOC A1→D ordered scale (IHO S-57 zones of confidence).
const CATZOC_SCALE = [
  { tier: 'A1', color: '#047857', short: 'Survey-grade',     desc: 'Highest confidence — full-coverage, controlled survey. Safe for navigation.' },
  { tier: 'A2', color: '#0d9488', short: 'Survey-grade',     desc: 'High confidence — assessed survey. Suitable for navigation.' },
  { tier: 'B',  color: '#0891b2', short: 'Assessed',         desc: 'Moderate confidence — assessed but with larger uncertainty. Navigate with care.' },
  { tier: 'C',  color: '#d97706', short: 'Low confidence',   desc: 'Low confidence — depth/position uncertainty significant. Not for precise navigation.' },
  { tier: 'D',  color: '#dc2626', short: 'Poor / unassessed', desc: 'Poor or unassessed quality — RECONNAISSANCE only. NOT safe for navigation.' },
];

function catzocInfo(tier) {
  const t = String(tier || '').toUpperCase().trim();
  return CATZOC_SCALE.find(c => c.tier === t) || null;
}

// Quality verdict for each metric, returning {color, text} or null.
function rmseQuality(rmse) {
  if (rmse == null) return null;
  const v = Number(rmse);
  const color = v <= 1 ? Q_GOOD : v <= 3 ? Q_OK : Q_POOR;
  return { color, text: `A typical depth estimate here is within ±${num(v, 1)} m of the true depth (root-mean-square error). Smaller is better; survey-grade work needs roughly ≤1 m.` };
}
function r2Quality(r2) {
  if (r2 == null) return null;
  const v = Number(r2);
  const pct = Math.round(Math.max(0, Math.min(1, v)) * 100);
  let band, color;
  if (v >= 0.7) { band = 'good skill'; color = Q_GOOD; }
  else if (v >= 0.4) { band = 'moderate skill'; color = Q_OK; }
  else if (v >= 0) { band = 'weak skill'; color = Q_POOR; }
  else { band = 'no skill (worse than guessing the mean depth)'; color = Q_POOR; }
  return { color, text: `The model explains about ${pct}% of the real depth variation (${band}; >0.70 is considered good, <0 means it beats nothing).` };
}
function biasQuality(bias) {
  if (bias == null) return null;
  const v = Number(bias);
  const mag = Math.abs(v);
  const color = mag <= 0.5 ? Q_GOOD : mag <= 1.5 ? Q_OK : Q_POOR;
  let dir;
  if (mag < 0.05) dir = 'essentially unbiased on average.';
  else if (v > 0) dir = `over-deep by ${num(mag, 1)} m on average (positive-down) — i.e. it tends to report water deeper than it is; for navigation that is the unsafe direction, so treat shoals cautiously.`;
  else dir = `shallower than truth by ${num(mag, 1)} m on average — it tends to report water shallower than it is, which is the conservative/safer direction for navigation.`;
  return { color, text: `Average systematic offset: ${dir} (Bias is separate from RMSE — it is the consistent lean, not the random scatter.)` };
}
function maeQuality(mae) {
  if (mae == null) return null;
  return { color: Q_NEUTRAL, text: `On average each estimate is off by ${num(mae, 1)} m (mean absolute error — like RMSE but less sensitive to a few big misses).` };
}
function coverageQuality(cov) {
  if (cov == null) return null;
  const v = Number(cov);
  const color = v >= 80 ? Q_GOOD : v >= 50 ? Q_OK : Q_POOR;
  return { color, text: `${num(v, 0)}% of the ROI returned a usable depth; the rest was masked (cloud, land, glint, or too deep/turbid for optical light to reach the seabed).` };
}

// A "?" caption row — tiny, consistent with the mono caption style.
function HelpCaption({ children }) {
  return (
    <p style={{ margin: '5px 0 0', fontSize: 9, lineHeight: 1.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', display: 'flex', gap: 5, alignItems: 'flex-start' }}>
      <span style={{ flexShrink: 0, fontWeight: 800, color: '#2563eb' }}>?</span>
      <span>{children}</span>
    </p>
  );
}

// One "metric → what it means" line.
function MetricLine({ label, value, unit, q }) {
  if (value == null && !q) return null;
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', padding: '7px 0', borderBottom: '1px solid var(--border-subtle, #eef2f7)' }}>
      <div style={{ width: 92, flexShrink: 0 }}>
        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.04em', color: 'var(--text-secondary)' }}>{label}</span>
      </div>
      <div style={{ width: 78, flexShrink: 0, textAlign: 'right' }}>
        <span style={{ fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 800, color: q ? q.color : '#1d4ed8' }}>
          {value}{value != null && value !== '—' && unit ? ` ${unit}` : ''}
        </span>
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <span style={{ fontSize: 10, lineHeight: 1.5, fontFamily: 'var(--font-sans, var(--font-mono))', color: 'var(--text-secondary)' }}>
          {q ? q.text : ''}
        </span>
      </div>
    </div>
  );
}

// Collapsible "Learn more" section — experts stay collapsed, novices expand.
function Collapsible({ title, defaultOpen = false, children, accent = '#1d4ed8' }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ border: '1px solid var(--border-dim)', borderRadius: 8, overflow: 'hidden' }}>
      <button onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '9px 12px', cursor: 'pointer', border: 'none', textAlign: 'left',
          background: 'var(--bg-secondary)',
          fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.06em', color: accent,
        }}>
        <span>{title}</span>
        <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{open ? '▾ hide' : '▸ learn more'}</span>
      </button>
      {open && <div style={{ padding: '12px', background: 'var(--bg-primary)' }}>{children}</div>}
    </div>
  );
}

// CATZOC A1→D tier scale, highlighting where THIS result sits.
function CatzocScale({ tier }) {
  const here = String(tier || '').toUpperCase().trim();
  return (
    <div>
      <div style={{ display: 'flex', gap: 4 }}>
        {CATZOC_SCALE.map(c => {
          const isHere = c.tier === here;
          return (
            <div key={c.tier} title={c.desc}
              style={{
                flex: 1, textAlign: 'center', padding: '6px 2px', borderRadius: 6,
                background: isHere ? c.color : 'var(--bg-secondary)',
                border: isHere ? `2px solid ${c.color}` : '1px solid var(--border-dim)',
                boxShadow: isHere ? `0 0 0 2px ${c.color}33` : 'none',
              }}>
              <div style={{ fontSize: 12, fontWeight: 800, fontFamily: 'var(--font-mono)', color: isHere ? '#fff' : c.color }}>{c.tier}</div>
              <div style={{ fontSize: 7.5, fontFamily: 'var(--font-mono)', color: isHere ? 'rgba(255,255,255,0.9)' : 'var(--text-dim)', marginTop: 2 }}>{c.short}</div>
              {isHere && <div style={{ fontSize: 7, fontWeight: 800, color: '#fff', marginTop: 2, letterSpacing: '0.08em' }}>◀ THIS</div>}
            </div>
          );
        })}
      </div>
      <p style={{ margin: '8px 0 0', fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
        Worse ◀ ─────────────── CATZOC scale (IHO S-57 zones of confidence) ─────────────── ▶ Better
      </p>
    </div>
  );
}

// The synthesized "What can I use this for?" verdict.
function buildVerdict(iho, stats) {
  const rmse = iho.rmse != null ? Number(iho.rmse) : null;
  const r2 = iho.r2 != null ? Number(iho.r2) : null;
  const cz = catzocInfo(iho.catzoc);
  const tier = cz ? cz.tier : (iho.catzoc || '—');
  // Grade by the weakest of CATZOC / RMSE.
  const survey = tier === 'A1' || tier === 'A2';
  const usable = tier === 'B' || (rmse != null && rmse <= 2);
  let headline, color, goodFor, notFor;
  if (survey) {
    color = Q_GOOD;
    headline = `Survey-grade bathymetry (CATZOC ${tier}${rmse != null ? `, RMSE ${num(rmse, 1)} m` : ''}).`;
    goodFor = 'navigation planning, depth charts, dredging estimates and engineering reference.';
    notFor = 'safety-of-life navigation without confirming against the official chart and local Notices to Mariners.';
  } else if (usable) {
    color = Q_OK;
    headline = `Engineering-aware reconnaissance (CATZOC ${tier}${rmse != null ? `, RMSE ${num(rmse, 1)} m` : ''}).`;
    goodFor = 'planning, relative depth trends, spotting shoals and channels, and pre-survey scoping.';
    notFor = 'navigation, dredging volumes, or engineering design — those need survey-grade (CATZOC A) data.';
  } else {
    color = Q_POOR;
    headline = `Reconnaissance-grade bathymetry (CATZOC ${tier}${rmse != null ? `, RMSE ${num(rmse, 1)} m` : ''}${r2 != null ? `, R² ${num(r2, 2)}` : ''}).`;
    goodFor = 'planning, understanding relative depth trends, and identifying potential shoals or features to investigate.';
    notFor = 'navigation, dredging volumes, or engineering — those require survey-grade (CATZOC A) data.';
  }
  return { headline, color, goodFor, notFor };
}

// Build per-2 m coverage bands from per_band if present, else from the
// histogram, else null. Never invents data.
function readBands(d) {
  const explicit = d?.per_band || d?.coverage_bands;
  if (Array.isArray(explicit) && explicit.length) {
    const total = explicit.reduce((a, b) => a + (Number(b.count) || 0), 0) || 1;
    return explicit.map((b) => ({
      label: b.label || `${num(pick(b, 'from', 'lo', 'min'), 0)}–${num(pick(b, 'to', 'hi', 'max'), 0)} m`,
      pct: b.pct != null ? Number(b.pct) : (Number(b.count) || 0) / total * 100,
    }));
  }
  const h = d?.histogram;
  if (h && Array.isArray(h.bin_edges) && Array.isArray(h.counts) && h.counts.length) {
    const edges = h.bin_edges, counts = h.counts;
    const total = counts.reduce((a, b) => a + (Number(b) || 0), 0) || 1;
    // Re-bin the histogram into 2 m bands.
    const maxD = edges[edges.length - 1];
    const nBands = Math.max(1, Math.ceil(maxD / 2));
    const acc = new Array(nBands).fill(0);
    for (let i = 0; i < counts.length; i++) {
      const mid = (edges[i] + edges[i + 1]) / 2;
      const bi = Math.min(nBands - 1, Math.floor(mid / 2));
      acc[bi] += Number(counts[i]) || 0;
    }
    return acc.map((c, i) => ({ label: `${i * 2}–${i * 2 + 2} m`, pct: c / total * 100 }));
  }
  return null;
}

// Depth-distribution histogram — hand-rolled SVG to match ResultsPanel.js
// (the app's existing histogram convention; no extra chart dep).
function HistogramChart({ histogram, stats }) {
  if (!histogram || !Array.isArray(histogram.counts) || !histogram.counts.length) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', border: '1px dashed var(--border-dim)', borderRadius: 8, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
        No depth-distribution histogram in this analyse payload.
      </div>
    );
  }
  const counts = histogram.counts.map(Number);
  const edges = (histogram.bin_edges || []).map(Number);
  const maxCount = Math.max(...counts, 1);
  const total = counts.reduce((a, b) => a + b, 0);
  const maxD = edges.length ? edges[edges.length - 1] : (stats.max || 1);
  const W = 500, H = 150, x0 = 42, y0 = 118, x1 = 495;
  const bw = (x1 - x0) / counts.length;
  const pcts = [['p5', stats.p5], ['p50', stats.p50], ['p95', stats.p95]];
  return (
    <div>
      <svg width="100%" height={H} style={{ background: '#fff', borderRadius: 8, border: '1px solid #e2e8f0' }} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <line x1={x0} y1={y0} x2={x1} y2={y0} stroke="#94a3b8" strokeWidth="1" />
        {counts.map((c, i) => {
          const h = (c / maxCount) * 100;
          return <rect key={i} x={x0 + i * bw} y={y0 - h} width={Math.max(0.5, bw - 0.6)} height={h} fill="#0284c7" fillOpacity={0.72} />;
        })}
        {maxD ? pcts.map(([k, v], i) => {
          if (v == null || !Number.isFinite(Number(v))) return null;
          const xk = x0 + (Number(v) / maxD) * (x1 - x0);
          const col = i === 1 ? '#dc2626' : '#7c3aed';
          return (
            <g key={k}>
              <line x1={xk} y1={14} x2={xk} y2={y0} stroke={col} strokeWidth="1" strokeDasharray="4,3" />
              <text x={xk} y={11} textAnchor="middle" fontSize="8" fill={col} fontFamily="JetBrains Mono" fontWeight="700">{k} {num(v, 1)}</text>
            </g>
          );
        }) : null}
        <text x={x0} y={132} fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">0 m</text>
        <text x={x1} y={132} textAnchor="end" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">{num(maxD, 1)} m</text>
        <text x={x0 - 4} y={y0} textAnchor="end" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">0</text>
        <text x={x0 - 4} y={20} textAnchor="end" fontSize="7" fill="#64748b" fontFamily="JetBrains Mono">{maxCount}</text>
      </svg>
      <p style={{ margin: '4px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
        {total ? `${total.toLocaleString()} samples` : ''} · depth positive-down (m)
      </p>
    </div>
  );
}

function StatChip({ label, value, unit }) {
  return (
    <div style={{ padding: '6px 8px', background: 'var(--bg-primary)', border: '1px solid var(--border-dim)', borderRadius: 6, textAlign: 'center' }}>
      <div style={{ fontSize: 8, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', letterSpacing: '0.06em' }}>{label}</div>
      <div style={{ fontSize: 12, fontWeight: 800, color: '#1d4ed8', fontFamily: 'var(--font-mono)' }}>
        {value}{value !== '—' && unit ? ` ${unit}` : ''}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelCardPanel — the auditable "Method & Provenance / Model Card" (§4).
// Agency-grade: method + version, architecture, inputs, training provenance +
// region-training status, σ with observed-vs-target 95% coverage, the
// spatial-block metrics (RMSE / signed bias / full-range R² OR "suppressed"),
// decile slope, the band-shuffle PASS badge, IHO S-44 + CATZOC chip, the
// limitation flag, data sources, datum = LAT, and a "not for navigation"
// disclaimer. Renders only the fields the backend actually supplied.
// ─────────────────────────────────────────────────────────────────────────────
function MCRow({ k, v, mono = true }) {
  if (v == null || v === '' || v === '—') return null;
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', padding: '5px 0', borderBottom: '1px solid var(--border-subtle, #eef2f7)' }}>
      <div style={{ width: 132, flexShrink: 0, fontSize: 9, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.03em', color: 'var(--text-dim)' }}>{k}</div>
      <div style={{ flex: 1, minWidth: 0, fontSize: 10, lineHeight: 1.5, fontFamily: mono ? 'var(--font-mono)' : 'inherit', color: 'var(--text-secondary)', wordBreak: 'break-word' }}>{v}</div>
    </div>
  );
}

function ModelCardPanel({ card, iho }) {
  if (!card) return null;
  const st = METHOD_STYLE[card.kind] || METHOD_STYLE.unknown;
  const mm = card.metrics || {};
  const sig = card.sigma || {};
  const covObs = sig.cov95Observed, covTgt = sig.cov95Target ?? 0.95;
  const covWarn = covObs != null && Math.abs(Number(covObs) - Number(covTgt)) > 0.10;
  // R² rendering: full-range value, OR an explicit "suppressed" note for
  // single-mode (narrow-range) regions. NEVER a per-2 m-band R².
  const r2Text = card.kind === 'reconnaissance'
    ? 'n/a (reconnaissance — no held-out in-situ)'
    : (mm.r2Suppressed
        ? 'suppressed — single-mode (narrow depth range); judge by RMSE + decile slope'
        : (mm.r2 != null ? num(mm.r2, 3) : null));

  return (
    <div style={{ border: `1px solid ${st.border}`, borderRadius: 10, overflow: 'hidden' }}>
      {/* Header band */}
      <div style={{ padding: '11px 14px', background: st.bg, borderBottom: `1px solid ${st.border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
          <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', padding: '2px 7px', borderRadius: 5, background: st.color, color: '#fff' }}>MODEL CARD</span>
          <span style={{ fontSize: 12, fontFamily: 'var(--font-display)', fontWeight: 800, color: st.color }}>
            {card.method || st.long}{card.modelVersion ? ` — ${card.modelVersion}` : ''}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {card.bandShuffle && <BandShuffleBadge bs={card.bandShuffle} />}
          {card.limitationFlag && <LimitationChip flag={card.limitationFlag} />}
        </div>
      </div>

      <div style={{ padding: '6px 14px 12px', background: 'var(--bg-primary)' }}>
        {/* Method & architecture */}
        <MCRow k="Method" v={card.method || st.long} mono={false} />
        <MCRow k="Architecture" v={card.architecture} />
        <MCRow k="Engine" v={card.engineFile} />
        <MCRow k="Input bands" v={card.inputBands ? card.inputBands.join(', ') : null} />
        <MCRow k="Derived features" v={card.derivedFeatures ? card.derivedFeatures.join(' · ') : null} />
        <MCRow k="Training" v={card.trainingProvenance} mono={false} />
        <MCRow k="Region training" v={typeof card.regionTraining === 'string'
          ? card.regionTraining
          : (card.regionTraining ? (card.regionTraining.region_label || card.regionTraining.region_key) : null)} />
        <MCRow k="Resolution" v={card.resolutionM != null ? `${card.resolutionM} m` : null} />
        <MCRow k="Scene dates" v={card.sceneDates ? card.sceneDates.join(', ') : null} />
        <MCRow k="Processed" v={card.dateProcessed ? String(card.dateProcessed).slice(0, 19).replace('T', ' ') : null} />

        {/* Validation metrics — spatial-block CV, honest R² */}
        {card.kind !== 'reconnaissance' && (mm.rmse != null || mm.bias != null || r2Text != null || mm.decileSlope != null) && (
          <div style={{ marginTop: 10, padding: '9px 11px', borderRadius: 8, background: 'var(--bg-secondary)', border: '1px solid var(--border-dim)' }}>
            <p style={{ margin: '0 0 7px', fontSize: 8.5, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: 'var(--text-dim)' }}>
              VALIDATION{mm.split ? ` · ${mm.split}` : ''}{mm.nTest != null ? ` · n_test ${Number(mm.nTest).toLocaleString()}` : ''}
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(86px, 1fr))', gap: 6 }}>
              <StatChip label="RMSE" value={num(mm.rmse, 2)} unit="m" />
              <StatChip label="BIAS" value={mm.bias != null ? fmtSigned(Number(num(mm.bias, 2))) : '—'} />
              <StatChip label="DECILE SLOPE" value={num(mm.decileSlope, 3)} />
            </div>
            <div style={{ marginTop: 7, fontSize: 9.5, fontFamily: 'var(--font-mono)', lineHeight: 1.5, color: 'var(--text-secondary)' }}>
              <b style={{ color: mm.r2Suppressed ? '#b45309' : '#059669' }}>Full-range R²:</b>{' '}
              {r2Text != null ? r2Text : '—'}
            </div>
            {mm.bias != null && Math.abs(Number(mm.bias)) > 0.3 && (
              <p style={{ margin: '5px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', lineHeight: 1.45 }}>
                Bias reported separately from RMSE (signed, positive-down): {fmtSigned(Number(num(mm.bias, 2)))}{' '}
                {Number(mm.bias) < 0 ? '— under-deepening (conservative direction).' : '— over-deepening.'}
              </p>
            )}
          </div>
        )}

        {/* Uncertainty σ + observed-vs-target 95% coverage */}
        {(sig.mean != null || covObs != null) && (
          <div style={{ marginTop: 8, padding: '9px 11px', borderRadius: 8, background: 'rgba(8,145,178,0.06)', border: '1px solid rgba(8,145,178,0.25)' }}>
            <p style={{ margin: '0 0 6px', fontSize: 8.5, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: '#0e7490' }}>UNCERTAINTY (σ)</p>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 10, fontFamily: 'var(--font-mono)' }}>
              {sig.mean != null && <div>σ mean <b style={{ color: '#0891b2' }}>±{num(sig.mean, 2)} m</b></div>}
              {covObs != null && (
                <div>95% coverage <b style={{ color: covWarn ? '#b45309' : '#059669' }}>{num(Number(covObs) * (covObs <= 1 ? 100 : 1), 0)}%</b>
                  <span style={{ color: 'var(--text-dim)' }}> / target {num(Number(covTgt) * (covTgt <= 1 ? 100 : 1), 0)}%</span>
                </div>
              )}
            </div>
            {sig.calibration && <p style={{ margin: '5px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>Calibration: {sig.calibration}</p>}
            {sig.cov95Raw != null && Number(sig.cov95Raw) !== Number(covObs) && (
              <p style={{ margin: '5px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', lineHeight: 1.45 }}>
                Raw (pre-recalibration) coverage was {num(Number(sig.cov95Raw) * (sig.cov95Raw <= 1 ? 100 : 1), 0)}% — the displayed coverage reflects the operative (recalibrated) σ. Shown transparently.
              </p>
            )}
            {covWarn && (
              <p style={{ margin: '5px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: '#b45309', lineHeight: 1.45 }}>
                Observed coverage differs from target — {Number(covObs) < Number(covTgt) ? 'the model is under-confident here (intervals too narrow in deep water)' : 'intervals are wider than needed'}. Shown honestly, not hidden.
              </p>
            )}
          </div>
        )}

        {/* ITEM 1 — Total Propagated Uncertainty (TPU), not just model σ */}
        {card.tpu && (
          <div style={{ marginTop: 8, padding: '9px 11px', borderRadius: 8, background: 'rgba(124,58,237,0.05)', border: '1px solid rgba(124,58,237,0.25)' }}>
            <p style={{ margin: '0 0 6px', fontSize: 8.5, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: '#6d28d9' }}>TOTAL PROPAGATED UNCERTAINTY (IHO S-44 §3.3.1)</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(78px, 1fr))', gap: 6, fontSize: 9.5, fontFamily: 'var(--font-mono)' }}>
              <div>u_model <b style={{ color: '#6d28d9' }}>{card.tpu.uModel != null ? num(card.tpu.uModel, 2) : '—'} m</b></div>
              <div>u_tide <b style={{ color: card.tpu.tideAssumedZero ? '#059669' : '#b45309' }}>{num(card.tpu.uTide, 2)} m</b></div>
              <div>u_datum <b>{num(card.tpu.uDatum, 2)} m</b></div>
              <div>u_refr <b>{card.tpu.uRefraction != null ? num(card.tpu.uRefraction, 2) : '—'} m</b></div>
              <div>u_coreg <b>{num(card.tpu.uCoreg, 2)} m</b></div>
              <div>TPU <b style={{ color: '#6d28d9' }}>±{num(card.tpu.tpuM, 2)} m</b></div>
            </div>
            {!card.tpu.tideAssumedZero && (
              <p style={{ margin: '6px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: '#b45309', lineHeight: 1.45 }}>
                ⚠ Tide NOT corrected — tidal range ≈ {num(card.tpu.tidalRangeM, 1)} m (Gulf); u_tide is an assumed component, not measured.
              </p>
            )}
            {card.tpu.note && (
              <p style={{ margin: '5px 0 0', fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', lineHeight: 1.45 }}>{card.tpu.note}</p>
            )}
          </div>
        )}

        {/* ITEM 4 — shoal-bias safety-of-navigation warning (internal, prominent red) */}
        {card.shoalSafe === false && (
          <div style={{ marginTop: 8, padding: '9px 11px', borderRadius: 8, background: 'rgba(225,29,72,0.10)', border: '1.5px solid rgba(225,29,72,0.55)' }}>
            <p style={{ margin: 0, fontSize: 9.5, fontFamily: 'var(--font-mono)', fontWeight: 800, color: '#e11d48', lineHeight: 1.5 }}>
              ⛔ SHOAL-BIAS UNSAFE — {card.shoalWarning || 'model over-deepens (> +0.20 m); unsafe direction. Do NOT use for navigation.'}
            </p>
          </div>
        )}

        {/* IHO S-44 + CATZOC — honest p95-vs-TVU classification (ITEM 1/2) */}
        {(card.iho.orderLabel || card.iho.order1a != null || card.iho.order2 != null || card.catzoc) && (
          <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {card.iho.orderLabel && (
              <span style={{ fontSize: 9.5, fontFamily: 'var(--font-mono)', fontWeight: 700, padding: '3px 9px', borderRadius: 6, background: /below/i.test(card.iho.orderLabel) ? 'rgba(225,29,72,0.10)' : 'rgba(37,99,235,0.10)', color: /below/i.test(card.iho.orderLabel) ? '#e11d48' : '#1d4ed8' }}>
                IHO S-44 {card.iho.orderLabel} <span style={{ fontWeight: 500 }}>(model σ only)</span>
              </span>
            )}
            {card.iho.order1a != null && (
              <span style={{ fontSize: 9.5, fontFamily: 'var(--font-mono)', fontWeight: 700, padding: '3px 9px', borderRadius: 6, background: 'rgba(37,99,235,0.10)', color: '#1d4ed8' }}>
                IHO S-44 Order 1a {num(card.iho.order1a, 0)}%
              </span>
            )}
            {(card.catzoc || (iho && iho.catzoc)) && (
              <span style={{ fontSize: 9.5, fontFamily: 'var(--font-mono)', fontWeight: 800, padding: '3px 9px', borderRadius: 6, background: (card.catzoc || iho.catzoc) === 'D' ? 'rgba(225,29,72,0.10)' : 'rgba(15,23,42,0.07)', color: (card.catzoc || iho.catzoc) === 'D' ? '#e11d48' : '#0f172a' }}>
                CATZOC {card.catzoc || iho.catzoc}
              </span>
            )}
            {card.iho.p95 != null && (
              <span style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', flex: '1 1 100%', lineHeight: 1.4 }}>
                p95 ≈ {num(card.iho.p95, 2)} m vs TVU {num(card.iho.tvuEval, 2)} m · {card.iho.passCriterion || 'p95 ≤ TVU (S-44 §3.3.1)'}
              </span>
            )}
          </div>
        )}

        {/* Data sources */}
        {card.dataSources && (
          <div style={{ marginTop: 8 }}>
            <p style={{ margin: '0 0 4px', fontSize: 8.5, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: 'var(--text-dim)' }}>DATA SOURCES</p>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 9.5, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {card.dataSources.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
        )}

        {/* Datum + disclaimer */}
        <p style={{ margin: '9px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', lineHeight: 1.5 }}>
          Datum: <b>{card.datum || 'LAT (positive-down)'}</b>{card.maxDepthM != null ? ` · cap ${card.maxDepthM} m` : ''}.
        </p>
        {/* ITEM 5 — amber "DATUM ASSUMED" chip when no explicit transform applied */}
        {card.datumInfo && card.datumInfo.datum_transform_applied === false && (
          <div style={{ marginTop: 5, display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', fontWeight: 800, padding: '2px 8px', borderRadius: 5, background: 'rgba(180,83,9,0.12)', color: '#b45309' }}>
              DATUM ASSUMED — not explicitly transformed
            </span>
            {card.datumInfo.datum_offset_note && (
              <span style={{ fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', flex: '1 1 100%', lineHeight: 1.4 }}>{card.datumInfo.datum_offset_note}</span>
            )}
          </div>
        )}
        <p style={{ margin: '4px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: '#b45309', fontStyle: 'italic', lineHeight: 1.5 }}>
          ⚠ {card.disclaimer || 'Reconnaissance-grade SDB. Not to be used for navigation.'}
        </p>
        {!card.present && (
          <p style={{ margin: '6px 0 0', fontSize: 8.5, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', lineHeight: 1.45 }}>
            Note: this result did not return the full <code>metrics.model_card</code> provenance block — the
            method label above is from the region-training stamp. The panel fills in fully once the backend
            attaches the §4 model card.
          </p>
        )}
      </div>
    </div>
  );
}

function FullAnalysisModal({ meta, data, loading, error, apiBase, onClose }) {
  const zipUrl = zipUrlFor(data || meta);
  const stats = readStats(data);
  const iho = readIho(data);
  const bands = readBands(data);
  const grid = data?.grid;
  const roi = data?.roi_bbox || meta?.roi_bbox;
  const resLabel = (meta?.resolution || data?.resolution) === 'vhr' ? 'High-Resolution (~1 m)'
    : (meta?.resolution || data?.resolution) === '10m' ? '10 m' : (meta?.resolution || '—');
  const units = INTERNAL_ANALYSIS ? (data?.units || 'm, positive-down, LAT') : 'm';
  const honesty = khalifaHonesty(data) || khalifaHonesty(meta);
  const modelCard = readModelCard(data) || readModelCard(meta);

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1100,
        background: 'rgba(15,23,42,0.62)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px',
        backdropFilter: 'blur(3px)',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(1080px, 100%)', maxHeight: '94vh', display: 'flex', flexDirection: 'column',
          background: 'var(--bg-primary)', borderRadius: '14px',
          border: '1px solid var(--border-dim)', boxShadow: '0 24px 70px rgba(15,23,42,0.45)', overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '14px 20px', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12,
          background: 'linear-gradient(135deg, rgba(14,165,233,0.12), rgba(37,99,235,0.07))',
          borderBottom: '1px solid var(--border-dim)',
        }}>
          <div style={{ minWidth: 0 }}>
            <h2 style={{ margin: 0, fontSize: '14px', fontFamily: 'var(--font-display)', fontWeight: 800, color: '#1d4ed8', letterSpacing: '0.04em' }}>
              🔍 {meta?.name || data?.name || 'Bathymetry analysis'}
            </h2>
            <p style={{ margin: '5px 0 0', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
              {resLabel} · ROI {fmtRoi(roi)} · {fmtDate(meta?.date || data?.date)}
            </p>
            <p style={{ margin: '2px 0 0', fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
              area {stats.area != null ? `${num(stats.area, 2)} km²` : '—'} · depth {num(stats.min, 1)}–{num(stats.max, 1)} m · units {units}
            </p>
          </div>
          <button onClick={onClose} aria-label="Close"
            style={{ width: 30, height: 30, borderRadius: 8, border: '1px solid var(--border-dim)', background: 'var(--bg-secondary)', color: 'var(--text-dim)', cursor: 'pointer', fontSize: 14, fontWeight: 700, flexShrink: 0 }}>✕</button>
        </div>

        {/* Body */}
        <div style={{ padding: '16px 20px', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
          {loading && (
            <div style={{ padding: 40, textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-dim)' }}>
              <span style={{ display: 'inline-block', width: 16, height: 16, border: '2px solid var(--border-dim)', borderTopColor: '#2563eb', borderRadius: '50%', animation: 'spin 0.8s linear infinite', verticalAlign: 'middle', marginRight: 8 }} />
              Loading full analysis…
            </div>
          )}
          {!loading && error && (
            <div style={{ padding: 24, textAlign: 'center', border: '1px solid rgba(220,38,38,0.3)', borderRadius: 8, background: 'rgba(220,38,38,0.06)', fontFamily: 'var(--font-mono)', fontSize: 12, color: '#991b1b' }}>
              Could not load analysis: {String(error)}
            </div>
          )}

          {!loading && !error && data && (() => {
            const verdict = buildVerdict(iho, stats);
            const cz = catzocInfo(iho.catzoc);
            return (
            <>
              {/* ── VERDICT / "What can I use this for?" (INTERNAL ONLY) ───── */}
              {INTERNAL_ANALYSIS && (
              <div style={{
                padding: '12px 14px', borderRadius: 10,
                border: `1px solid ${verdict.color}55`,
                background: `${verdict.color}0e`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', padding: '2px 7px', borderRadius: 5, background: verdict.color, color: '#fff' }}>VERDICT</span>
                  <span style={{ fontSize: 12, fontFamily: 'var(--font-display)', fontWeight: 800, color: verdict.color }}>{verdict.headline}</span>
                </div>
                <p style={{ margin: 0, fontSize: 11, lineHeight: 1.55, color: 'var(--text-secondary)' }}>
                  <b style={{ color: Q_GOOD }}>Good for:</b> {verdict.goodFor}<br />
                  <b style={{ color: Q_POOR }}>Not for:</b> {verdict.notFor}
                </p>
                {iho.reconnaissance && (
                  <p style={{ margin: '8px 0 0', fontSize: 10, fontWeight: 700, color: '#991b1b', fontFamily: 'var(--font-mono)' }}>
                    ⚠ RECONNAISSANCE flag set — this product is for indicative use only and must NOT be used for navigation.
                  </p>
                )}
              </div>
              )}

              {/* ── METHOD & PROVENANCE / MODEL CARD (§4) — INTERNAL ONLY ──── */}
              {INTERNAL_ANALYSIS && modelCard && (
                <div>
                  <p style={{ margin: '0 0 8px', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: '#1d4ed8' }}>
                    METHOD &amp; PROVENANCE
                  </p>
                  <ModelCardPanel card={modelCard} iho={iho} />
                </div>
              )}

              {/* ── Honesty card (shallow vs deep) — INTERNAL ONLY ─────────── */}
              {INTERNAL_ANALYSIS && honesty && <KhalifaHonestyCard h={honesty} stats={stats} />}

              {/* CLIENT PORTAL headline figures — depth range / coverage / area */}
              {!INTERNAL_ANALYSIS && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 8 }}>
                  <StatChip label="DEPTH RANGE" value={`${num(stats.min, 1)}–${num(stats.max, 1)}`} unit="m" />
                  <StatChip label="COVERAGE" value={stats.coverage != null ? num(stats.coverage, 0) : '—'} unit="%" />
                  <StatChip label="AREA" value={stats.area != null ? num(stats.area, 2) : '—'} unit="km²" />
                </div>
              )}

              {/* Grid: left = charts/stats, right = 3D */}
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.05fr) minmax(0, 1fr)', gap: 16, alignItems: 'stretch' }}>
                {/* LEFT column */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
                  {/* Histogram */}
                  <div>
                    <p style={{ margin: '0 0 6px', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: '#1d4ed8' }}>DEPTH DISTRIBUTION</p>
                    <HistogramChart histogram={data.histogram} stats={stats} />
                    <HelpCaption>
                      How to read: each bar is a slice of depth; tall bars mean most of the seabed area sits at that depth.
                      The dashed lines mark the shallowest 5% (p5), the middle (median, p50) and the deepest 5% (p95) of the area.
                    </HelpCaption>
                  </div>

                  {/* Stats */}
                  <div>
                    <p style={{ margin: '0 0 6px', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: '#1d4ed8' }}>STATISTICS (m)</p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6 }}>
                      <StatChip label="MIN" value={num(stats.min, 2)} />
                      <StatChip label="MAX" value={num(stats.max, 2)} />
                      <StatChip label="MEAN" value={num(stats.mean, 2)} />
                      <StatChip label="MEDIAN" value={num(stats.median, 2)} />
                      <StatChip label="STD" value={num(stats.std, 2)} />
                      <StatChip label="P5" value={num(stats.p5, 2)} />
                      <StatChip label="P95" value={num(stats.p95, 2)} />
                      <StatChip label="COVERAGE" value={stats.coverage != null ? num(stats.coverage, 1) : '—'} unit="%" />
                    </div>
                    <HelpCaption>
                      The seabed here runs {num(stats.min, 1)}–{num(stats.max, 1)} m deep, typically around {num(stats.median, 1)} m (median).
                      STD ({num(stats.std, 1)} m) is how spread-out the depths are; half the area lies between p25 and p75.
                    </HelpCaption>
                  </div>

                  {/* Per-2 m coverage bars */}
                  {bands && (
                    <div>
                      <p style={{ margin: '0 0 6px', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: '#1d4ed8' }}>COVERAGE BY 2 m BAND</p>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {bands.map((b, i) => (
                          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 9, fontFamily: 'var(--font-mono)' }}>
                            <span style={{ width: 58, color: 'var(--text-dim)', textAlign: 'right' }}>{b.label}</span>
                            <div style={{ flex: 1, height: 12, background: 'var(--bg-secondary)', borderRadius: 3, overflow: 'hidden', border: '1px solid var(--border-dim)' }}>
                              <div style={{ width: `${Math.max(0, Math.min(100, b.pct))}%`, height: '100%', background: BAND_COLORS[Math.min(BAND_COLORS.length - 1, i)] }} />
                            </div>
                            <span style={{ width: 38, color: 'var(--text-secondary)', textAlign: 'right', fontWeight: 700 }}>{num(b.pct, 1)}%</span>
                          </div>
                        ))}
                      </div>
                      <HelpCaption>
                        How much of the mapped seabed falls in each 2 m depth band — useful for spotting where the shallow shoals
                        and the deeper channels are concentrated.
                      </HelpCaption>
                    </div>
                  )}
                </div>

                {/* RIGHT column — 3D seabed */}
                <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                  <p style={{ margin: '0 0 6px', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: '#1d4ed8' }}>🧊 3D SEABED</p>
                  <div style={{ flex: 1, minHeight: 320, height: '100%', borderRadius: 10, overflow: 'hidden', border: '1px solid var(--border-dim)', background: '#dbeafe' }}>
                    {grid && grid.values ? (
                      <ThreeDView gridPayload={grid} />
                    ) : (
                      <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: 16, fontFamily: 'var(--font-mono)', fontSize: 11, color: '#475569' }}>
                        No downsampled depth grid in this analyse payload — the 3D seabed needs the backend `grid` block.
                      </div>
                    )}
                  </div>
                  <HelpCaption>
                    How to read: drag to orbit, scroll to zoom. Colour = depth (light = shallow, dark navy = deep).
                    The vertical relief is exaggerated to make seabed shape visible — it is NOT true vertical scale.
                  </HelpCaption>
                </div>
              </div>

              {/* ── IHO S-44 / CATZOC — value + plain-language interpretation ─
                  INTERNAL ONLY: error/accuracy metrics never reach the portal. */}
              {INTERNAL_ANALYSIS && (
              <div>
                <p style={{ margin: '0 0 8px', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: '#1d4ed8' }}>ACCURACY — WHAT EACH NUMBER MEANS</p>
                <div style={{ border: '1px solid var(--border-dim)', borderRadius: 8, padding: '4px 12px 10px', background: 'var(--bg-primary)' }}>
                  <MetricLine label="RMSE" value={num(iho.rmse, 2)} unit="m" q={rmseQuality(iho.rmse)} />
                  <MetricLine label="R²" value={num(iho.r2, 2)} q={r2Quality(iho.r2)} />
                  <MetricLine label="Bias" value={num(iho.bias, 2)} unit="m" q={biasQuality(iho.bias)} />
                  <MetricLine label="MAE" value={num(iho.mae, 2)} unit="m" q={maeQuality(iho.mae)} />
                  <MetricLine label="Coverage" value={stats.coverage != null ? num(stats.coverage, 0) : null} unit="%" q={coverageQuality(stats.coverage)} />
                  <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', padding: '7px 0' }}>
                    <div style={{ width: 92, flexShrink: 0 }}><span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--text-secondary)' }}>n test</span></div>
                    <div style={{ width: 78, flexShrink: 0, textAlign: 'right' }}><span style={{ fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 800, color: '#1d4ed8' }}>{iho.n != null ? Number(iho.n).toLocaleString() : '—'}</span></div>
                    <div style={{ flex: 1, minWidth: 0 }}><span style={{ fontSize: 10, lineHeight: 1.5, color: 'var(--text-secondary)' }}>Number of independent in-situ soundings the estimate was checked against — more is more trustworthy.</span></div>
                  </div>
                </div>
              </div>
              )}

              {/* ── CATZOC tier scale + explanation — INTERNAL ONLY ───────── */}
              {INTERNAL_ANALYSIS && (
              <div>
                <p style={{ margin: '0 0 8px', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: '#1d4ed8' }}>
                  IHO QUALITY TIER · ORDER {iho.order || '—'} · CATZOC {iho.catzoc || '—'}
                </p>
                <CatzocScale tier={iho.catzoc} />
                {cz && (
                  <p style={{ margin: '8px 0 0', fontSize: 10.5, lineHeight: 1.55, color: 'var(--text-secondary)' }}>
                    <b style={{ color: cz.color }}>This result is CATZOC {cz.tier} — {cz.short}.</b> {cz.desc}
                    {iho.tvu != null && <> The achieved vertical uncertainty at the median depth is about {num(iho.tvu, 2)} m.</>}
                  </p>
                )}
                {iho.note && (
                  <p style={{ margin: '6px 0 0', fontSize: 9.5, fontFamily: 'var(--font-mono)', color: '#b45309', fontStyle: 'italic' }}>
                    ⚠ {iho.note}
                  </p>
                )}

                {iho.ordersTable && iho.ordersTable.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <Collapsible title="IHO S-44 ORDER TABLE — which survey tiers this run meets">
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 10 }}>
                        <thead>
                          <tr style={{ background: 'var(--bg-secondary)', color: 'var(--text-dim)' }}>
                            {['Order', 'CATZOC', 'Allowed TVU @ median', 'Met?'].map(h => (
                              <th key={h} style={{ padding: '6px 8px', textAlign: 'left', fontWeight: 700, borderBottom: '1px solid var(--border-dim)' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {iho.ordersTable.map((o, i) => (
                            <tr key={i} style={{ color: 'var(--text-primary)', borderBottom: '1px solid var(--border-subtle, #eef2f7)' }}>
                              <td style={{ padding: '6px 8px' }}>{o.order || '—'}</td>
                              <td style={{ padding: '6px 8px', fontWeight: 700 }}>{o.catzoc || '—'}</td>
                              <td style={{ padding: '6px 8px' }}>{o.tvu_at_median_m != null ? `≤ ${num(o.tvu_at_median_m, 2)} m` : '—'}</td>
                              <td style={{ padding: '6px 8px', fontWeight: 800, color: o.meets ? Q_GOOD : Q_POOR }}>{o.meets ? '✓ yes' : '✗ no'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <p style={{ margin: '8px 0 0', fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                        A tier is "met" when this run's RMSE ({num(iho.rmse, 2)} m) is within the IHO-allowed Total Vertical Uncertainty (TVU = √(a² + (b·depth)²)) at the median depth ({iho.evalDepth != null ? num(iho.evalDepth, 1) : num(stats.median, 1)} m).
                      </p>
                    </Collapsible>
                  </div>
                )}
              </div>
              )}

              {/* ── DATA PROVENANCE + METHOD + CAVEATS — INTERNAL ONLY ────── */}
              {INTERNAL_ANALYSIS && (
              <Collapsible title="DATA SOURCES, METHOD & CAVEATS — read before trusting these numbers" accent="#7c3aed">
                <div style={{ fontSize: 10.5, lineHeight: 1.6, color: 'var(--text-secondary)' }}>
                  <p style={{ margin: '0 0 8px' }}>
                    <b style={{ color: '#7c3aed' }}>Data sources & method.</b> {meta?.name || data?.name || ''} — Satellite-Derived Bathymetry (SDB):
                    Sentinel-2 L2A optical imagery (multi-scene cloud/glint-masked median composite) inverted to depth with the
                    Lyzenga (log-linear) and Stumpf (band-ratio) physics models, calibrated and bias-corrected against
                    ICESat-2 / ATL24 laser-altimetry photons and i-Boating chart soundings as reference depths.
                  </p>
                  <p style={{ margin: '0 0 8px' }}>
                    <b style={{ color: '#7c3aed' }}>Datum & units.</b> Depths are in metres, <b>positive-down</b>, referenced to <b>LAT</b> (Lowest Astronomical Tide).
                    A larger number means deeper water; bias is reported on the same convention ({units}).
                  </p>
                  <p style={{ margin: 0 }}>
                    <b style={{ color: '#7c3aed' }}>Key caveats (honest uncertainty).</b>
                  </p>
                  <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                    <li>Optical SDB only works where light reaches the seabed — accuracy degrades with depth and water turbidity, and fails entirely beyond the optical limit.</li>
                    <li>Reference soundings are sparse and may sit off-site; there is no dense in-situ control everywhere, so local errors can exceed the average RMSE.</li>
                    <li><b>Bias is reported separately from RMSE</b> — a low bias does not mean low scatter, and vice-versa.</li>
                    <li>RMSE/R²/CATZOC here are computed on held-out test soundings; treat them as indicative, not a guarantee for any single pixel.</li>
                    <li>This is not an official hydrographic survey and must not replace charted depths for navigation.</li>
                  </ul>
                </div>
              </Collapsible>
              )}
            </>
            );
          })()}
        </div>

        {/* Footer */}
        <div style={{ padding: '10px 20px', borderTop: '1px solid var(--border-dim)', background: 'var(--bg-secondary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <p style={{ margin: 0, fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
            {INTERNAL_ANALYSIS
              ? 'Depths are positive-down (m), referenced to LAT. Drag the 3D view to orbit · scroll to zoom.'
              : 'Depths in metres. Drag the 3D view to orbit · scroll to zoom.'}
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            {zipUrl && (
              <button onClick={() => window.open(`${apiBase || ''}${zipUrl}`, '_blank', 'noopener')}
                title="Download EVERY GeoTIFF this result references in one .zip (with a MANIFEST)"
                style={{ padding: '7px 14px', fontSize: 11, fontFamily: 'var(--font-display)', fontWeight: 800, borderRadius: 6, border: '1px solid #0891b2', background: 'rgba(8,145,178,0.14)', color: '#0e7490', cursor: 'pointer' }}>
                🗂 Download all TIFFs (.zip)
              </button>
            )}
            <button onClick={onClose} style={{ padding: '7px 14px', fontSize: 11, fontFamily: 'var(--font-display)', fontWeight: 700, borderRadius: 6, border: '1px solid var(--border-dim)', background: 'var(--bg-primary)', color: 'var(--text-secondary)', cursor: 'pointer' }}>Close</button>
          </div>
        </div>
      </div>
    </div>
  );
}
