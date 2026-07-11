import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// VhrPurchasePanel — "Buy Very-High-Resolution Imagery" (commercial-data order).
//
// Flow (matches the BathyResultsPanel design language: modal over a blurred
// backdrop, mono/display type, sky→blue gradients):
//
//   1. ROI  → POST /api/vhr-purchase/quote   → live area (km²), unit price
//             (≈5 €/km²), TOTAL (area × 5 €), provider, resolution, min-area.
//   2. CTA  → POST /api/vhr-purchase/checkout → redirect to checkout_url
//             (Stripe TEST Checkout, or a simulated success route the backend
//             returns when no Stripe key is configured).
//   3. On return (?vhr_purchase=success&session_id=…) → POST /confirm → status.
//   4. Orders list → GET /api/vhr-purchase/orders with payment_status +
//             sh_status chips. The Sentinel Hub order runs DRY-RUN / SANDBOX by
//             default → an honest "sandbox" badge; no real imagery is charged.
//
// Honesty: TEST MODE badge on payment; SANDBOX badge on the SH order — we never
// imply a real image was delivered.
//
// Backend contract (paths under /api/vhr-purchase) — reconcile with the
// developer's APP_LOOP_LOG.md post when they publish it:
//   POST /quote    {bbox:{west,south,east,north}}
//     → {area_km2, unit_price_eur, currency, total_eur, provider,
//        resolution_m, min_area_km2, min_price_eur, billable_area_km2, note}
//   POST /checkout {bbox, area_km2, total_eur}
//     → {checkout_url, session_id, test_mode, simulated}
//   POST /confirm  {session_id}
//     → <order>
//   GET  /orders   → {orders:[<order>]}
//   <order> = {id, created_at, area_km2, total_eur, currency, provider,
//              resolution_m, bbox:[w,s,e,n], payment_status:"PENDING"|"PAID",
//              sh_status:"sandbox"|"created"|"confirmed"|"failed",
//              sandbox:true, checkout_url, error}
// ─────────────────────────────────────────────────────────────────────────────

const UNIT_PRICE_EUR = 5;          // ≈ 5 € / km² (display fallback; quote is authoritative)
const DEFAULT_PROVIDER = 'Airbus Pléiades';
const DEFAULT_RES_M = 0.5;
const DEFAULT_MIN_AREA = 1;        // km² (display fallback)

const eur = (n) => {
  if (n == null || Number.isNaN(Number(n))) return '—';
  try {
    return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'EUR', maximumFractionDigits: 2 }).format(Number(n));
  } catch (_) { return `€${Number(n).toFixed(2)}`; }
};

const km2 = (n) => (n == null || Number.isNaN(Number(n)) ? '—' : `${Number(n).toFixed(2)} km²`);

function fmtDate(epochSeconds) {
  if (!epochSeconds) return '—';
  try { return new Date(epochSeconds * 1000).toLocaleString(undefined, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' }); }
  catch (_) { return '—'; }
}

function roiToBbox(roi) {
  // App-wide ROI is {west,south,east,north}; tolerate a [w,s,e,n] array too.
  if (!roi) return null;
  if (Array.isArray(roi) && roi.length >= 4) return { west: roi[0], south: roi[1], east: roi[2], north: roi[3] };
  if (typeof roi === 'object' && roi.west != null) return { west: roi.west, south: roi.south, east: roi.east, north: roi.north };
  return null;
}

// Local haversine area so the panel still shows an honest km² readout if the
// quote endpoint is briefly unreachable (clearly tagged "est." when used).
function approxAreaKm2(bbox) {
  if (!bbox) return null;
  const R = 6371.0088;
  const dLat = (bbox.north - bbox.south) * Math.PI / 180;
  const latMid = ((bbox.north + bbox.south) / 2) * Math.PI / 180;
  const dLon = (bbox.east - bbox.west) * Math.PI / 180;
  const h = R * dLat;
  const w = R * Math.cos(latMid) * dLon;
  return Math.abs(h * w);
}

export default function VhrPurchasePanel({ open, onClose, API, roi, addLog }) {
  const [quote, setQuote] = useState(null);
  const [quoting, setQuoting] = useState(false);
  const [quoteErr, setQuoteErr] = useState(null);
  const [checkingOut, setCheckingOut] = useState(false);
  const [orders, setOrders] = useState([]);
  const [ordersErr, setOrdersErr] = useState(null);
  const [toast, setToast] = useState(null);   // {msg, kind:'ok'|'err'}
  const [confirming, setConfirming] = useState(false);

  const pollRef = useRef(null);
  const bbox = useMemo(() => roiToBbox(roi), [roi]);

  const log = useCallback((m, t) => { if (addLog) addLog(m, t); }, [addLog]);
  const flash = useCallback((msg, kind = 'ok') => {
    setToast({ msg, kind });
    setTimeout(() => setToast(null), 4200);
  }, []);

  // ── R2: live quote on ROI change ──────────────────────────────────────────
  const fetchQuote = useCallback(async (bb) => {
    if (!bb) { setQuote(null); return; }
    setQuoting(true); setQuoteErr(null);
    try {
      const r = await fetch(`${API}/api/vhr-purchase/quote`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox: bb }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
      setQuote(d);
    } catch (e) {
      // Honest fallback: compute an *estimate* locally so the card still renders,
      // and label it clearly as an estimate (no silent fabrication).
      const area = approxAreaKm2(bb);
      const billable = area != null ? Math.max(area, DEFAULT_MIN_AREA) : null;
      setQuote(area != null ? {
        area_km2: area,
        billable_area_km2: billable,
        unit_price_eur: UNIT_PRICE_EUR,
        total_eur: billable != null ? billable * UNIT_PRICE_EUR : null,
        currency: 'EUR',
        provider: DEFAULT_PROVIDER,
        resolution_m: DEFAULT_RES_M,
        min_area_km2: DEFAULT_MIN_AREA,
        note: 'Estimate (quote endpoint unreachable) — final price confirmed at checkout.',
        _estimate: true,
      } : null);
      setQuoteErr('Quote endpoint unreachable — showing a local estimate.');
    } finally {
      setQuoting(false);
    }
  }, [API]);

  useEffect(() => {
    if (!open) return;
    fetchQuote(bbox);
  }, [open, bbox, fetchQuote]);

  // ── R5: orders list (poll while open) ─────────────────────────────────────
  const fetchOrders = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/vhr-purchase/orders`);
      const d = await r.json();
      if (Array.isArray(d?.orders)) { setOrders(d.orders); setOrdersErr(null); }
    } catch (_) {
      setOrdersErr('Cannot reach /api/vhr-purchase/orders');
    }
  }, [API]);

  useEffect(() => {
    if (!open) return undefined;
    fetchOrders();
    pollRef.current = setInterval(fetchOrders, 6000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [open, fetchOrders]);

  // ── R3/R4: checkout → redirect ────────────────────────────────────────────
  const startCheckout = useCallback(async () => {
    if (!bbox || !quote) { flash('Select an ROI first.', 'err'); return; }
    setCheckingOut(true);
    try {
      const r = await fetch(`${API}/api/vhr-purchase/checkout`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox, area_km2: quote.area_km2, total_eur: quote.total_eur }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
      if (!d.checkout_url) throw new Error('No checkout_url returned');
      // Remember the pending session so the return handler can confirm it even
      // if the backend success route forgets the query param.
      try { sessionStorage.setItem('vhr_purchase_session', d.session_id || ''); } catch (_) {}
      log(`VHR purchase: redirecting to ${d.simulated ? 'simulated' : 'Stripe TEST'} checkout for ${eur(quote.total_eur)}`, 'info');
      window.location.assign(d.checkout_url);
    } catch (e) {
      flash(`Checkout failed: ${e.message}`, 'err');
      log(`VHR purchase checkout failed: ${e.message}`, 'error');
      setCheckingOut(false);
    }
  }, [API, bbox, quote, flash, log]);

  // ── R3: handle the checkout return ────────────────────────────────────────
  // The backend /success route already FINALISES the order (payment→paid, SH
  // order placed, Results entry created) before redirecting the SPA to
  // /?vhr_order=<id>&vhr_paid=1 (or /?vhr_cancelled=1). So on return we just
  // fetch the finished order by id; we only fall back to POST /confirm when we
  // somehow have a Stripe session id but no order id (legacy path).
  const finaliseReturn = useCallback(async (orderId, sessionId) => {
    setConfirming(true);
    try {
      let o;
      if (orderId) {
        const r = await fetch(`${API}/api/vhr-purchase/orders/${orderId}`);
        o = await r.json();
        if (!r.ok || o.error) throw new Error(o.error || `HTTP ${r.status}`);
      } else if (sessionId) {
        const r = await fetch(`${API}/api/vhr-purchase/confirm`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId }),
        });
        o = await r.json();
        if (!r.ok || o.error) throw new Error(o.error || `HTTP ${r.status}`);
      } else {
        flash('Returned from checkout, but no order reference — refreshing orders.', 'err');
        fetchOrders();
        return;
      }
      const sb = o.sh_status === 'sandbox';
      flash(`Payment confirmed — order ${o.order_id || ''} placed (${o.sh_status || 'created'}${sb ? ', sandbox' : ''}).`, 'ok');
      log(`VHR purchase confirmed: order ${o.order_id} · ${o.payment_status} · SH ${o.sh_status}${sb ? ' (sandbox)' : ''}`, 'success');
      fetchOrders();
    } catch (e) {
      flash(`Confirm failed: ${e.message}`, 'err');
      log(`VHR purchase confirm failed: ${e.message}`, 'error');
    } finally {
      setConfirming(false);
      try { sessionStorage.removeItem('vhr_purchase_session'); } catch (_) {}
    }
  }, [API, flash, log, fetchOrders]);

  // Detect the checkout return once on mount (panel can be auto-opened by App
  // when the URL carries the paid/cancelled flag).
  useEffect(() => {
    let params;
    try { params = new URLSearchParams(window.location.search); } catch (_) { return; }
    const cancelled = params.get('vhr_cancelled') === '1' || params.get('vhr_purchase') === 'cancel';
    const paid = params.get('vhr_paid') === '1' || params.get('vhr_purchase') === 'success';
    if (!paid && !cancelled) return;
    if (cancelled) {
      flash('Checkout cancelled — no payment was made.', 'err');
    } else {
      const orderId = params.get('vhr_order');
      let sid = params.get('session_id');
      if (!orderId && !sid) { try { sid = sessionStorage.getItem('vhr_purchase_session'); } catch (_) {} }
      finaliseReturn(orderId, sid);
    }
    // Clean the URL so a reload doesn't re-confirm.
    try {
      const url = new URL(window.location.href);
      ['vhr_purchase', 'vhr_paid', 'vhr_order', 'vhr_cancelled', 'session_id'].forEach((k) => url.searchParams.delete(k));
      window.history.replaceState({}, '', url.toString());
    } catch (_) {}
  }, [finaliseReturn, fetchOrders, flash]);

  if (!open) return null;

  const total = quote?.total_eur;
  const unit = quote?.unit_price_eur ?? UNIT_PRICE_EUR;
  const area = quote?.area_km2;
  const billable = quote?.billable_area_km2 ?? area;
  const provider = quote?.provider ?? DEFAULT_PROVIDER;
  const resM = quote?.resolution_m ?? DEFAULT_RES_M;
  const minArea = quote?.min_area_km2 ?? DEFAULT_MIN_AREA;
  const belowMin = area != null && minArea != null && area < minArea;
  const canBuy = !!bbox && !!quote && total != null && !checkingOut;

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(15,23,42,0.55)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px', backdropFilter: 'blur(3px)',
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        width: 'min(820px, 100%)', maxHeight: '92vh', display: 'flex', flexDirection: 'column',
        background: 'var(--bg-primary)', borderRadius: '14px', border: '1px solid var(--border-dim)',
        boxShadow: '0 20px 60px rgba(15,23,42,0.35)', overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: 'linear-gradient(135deg, rgba(14,165,233,0.10), rgba(37,99,235,0.06))',
          borderBottom: '1px solid var(--border-dim)',
        }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '14px', fontFamily: 'var(--font-display)', fontWeight: 800, letterSpacing: '0.06em', color: '#1d4ed8', display: 'flex', alignItems: 'center', gap: '8px' }}>
              🛰️ BUY VERY-HR IMAGERY
              <span style={badge('#92400e', 'rgba(245,158,11,0.18)')}>TEST MODE</span>
              <span style={badge('#3730a3', 'rgba(99,102,241,0.16)')}>SANDBOX</span>
            </h2>
            <p style={{ margin: '4px 0 0', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
              Commercial-data order · {provider} · ~{resM} m · ≈{eur(unit)}/km² · Stripe TEST · no real imagery charged
            </p>
          </div>
          <button onClick={onClose} aria-label="Close" style={{
            width: '30px', height: '30px', borderRadius: '8px', border: '1px solid var(--border-dim)',
            background: 'var(--bg-secondary)', color: 'var(--text-dim)', cursor: 'pointer', fontSize: '14px', fontWeight: 700,
          }}>✕</button>
        </div>

        {/* Body */}
        <div style={{ padding: '16px 20px', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>

          {/* ── Pricing card ──────────────────────────────────────────────── */}
          <div style={{
            borderRadius: '12px', border: '1px solid rgba(14,165,233,0.30)',
            background: 'linear-gradient(135deg, rgba(14,165,233,0.06), rgba(37,99,235,0.04))',
            overflow: 'hidden',
          }}>
            {/* ROI readout */}
            <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--border-dim)' }}>
              <p style={{ margin: 0, fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em', color: 'var(--text-secondary)' }}>
                REGION OF INTEREST
              </p>
              {bbox ? (
                <div style={{ marginTop: '6px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {[['W', bbox.west], ['S', bbox.south], ['E', bbox.east], ['N', bbox.north]].map(([d, v]) => (
                    <span key={d} style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', padding: '3px 8px', borderRadius: '5px', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)' }}>
                      {d} <b>{Number(v).toFixed(4)}°</b>
                    </span>
                  ))}
                </div>
              ) : (
                <p style={{ margin: '6px 0 0', fontSize: '11px', fontFamily: 'var(--font-mono)', color: '#b45309' }}>
                  Draw a rectangle on the map (or pick a preset ROI) to get a price.
                </p>
              )}
            </div>

            {/* area → price breakdown */}
            <div style={{ padding: '14px', display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: '4px 18px', fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                <span>Area</span>
                <b style={{ color: 'var(--text-primary)' }}>{quoting ? '…' : km2(area)}{quote?._estimate ? ' (est.)' : ''}</b>
                <span>Unit price</span>
                <b style={{ color: 'var(--text-primary)' }}>{eur(unit)} / km²</b>
                {billable != null && billable !== area && (<><span>Billable area</span><b style={{ color: 'var(--text-primary)' }}>{km2(billable)}{belowMin ? ' (min)' : ''}</b></>)}
                <span>Provider</span>
                <b style={{ color: 'var(--text-primary)' }}>{provider}</b>
                <span>Resolution</span>
                <b style={{ color: 'var(--text-primary)' }}>~{resM} m</b>
              </div>
              <div style={{ textAlign: 'right' }}>
                <p style={{ margin: 0, fontSize: '9px', fontFamily: 'var(--font-mono)', letterSpacing: '0.1em', color: 'var(--text-dim)' }}>TOTAL</p>
                <p style={{ margin: '2px 0 0', fontSize: '30px', fontFamily: 'var(--font-display)', fontWeight: 800, lineHeight: 1, color: '#1d4ed8' }}>
                  {quoting ? '…' : eur(total)}
                </p>
              </div>
            </div>

            {/* notes / errors */}
            <div style={{ padding: '0 14px 12px' }}>
              {belowMin && (
                <p style={{ margin: '0 0 6px', fontSize: '10px', fontFamily: 'var(--font-mono)', color: '#b45309' }}>
                  ⚠ ROI below the {minArea} km² minimum — billed at the minimum order size.
                </p>
              )}
              {quote?.note && (
                <p style={{ margin: '0 0 6px', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>{quote.note}</p>
              )}
              {quoteErr && (
                <p style={{ margin: '0 0 6px', fontSize: '10px', fontFamily: 'var(--font-mono)', color: '#b45309' }}>{quoteErr}</p>
              )}
              <button onClick={startCheckout} disabled={!canBuy} style={{
                width: '100%', padding: '13px', fontSize: '14px', fontFamily: 'var(--font-display)', fontWeight: 800,
                borderRadius: '9px', border: 'none', letterSpacing: '0.04em',
                background: canBuy ? 'linear-gradient(135deg,#0ea5e9,#2563eb)' : 'var(--bg-secondary)',
                color: canBuy ? '#fff' : 'var(--text-dim)', cursor: canBuy ? 'pointer' : 'not-allowed',
                boxShadow: canBuy ? '0 6px 16px rgba(37,99,235,0.30)' : 'none',
              }}>
                {checkingOut ? 'Redirecting to checkout…' : `💳 Purchase — ${eur(total)}`}
              </button>
              <p style={{ margin: '8px 0 0', fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', textAlign: 'center' }}>
                Pay by card via Stripe <b>TEST</b> Checkout. Use test card <b>4242 4242 4242 4242</b>, any future date / CVC.
                The order runs in <b>sandbox / dry-run</b> — no real imagery is purchased.
              </p>
            </div>
          </div>

          {/* ── Orders ────────────────────────────────────────────────────── */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <h3 style={{ margin: 0, fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.1em', color: '#1d4ed8' }}>
                YOUR ORDERS · {orders.length}
              </h3>
              <button onClick={fetchOrders} style={{
                padding: '5px 10px', fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700,
                borderRadius: '5px', border: '1px solid var(--border-dim)', background: 'var(--bg-secondary)', color: 'var(--text-dim)', cursor: 'pointer',
              }}>{confirming ? 'confirming…' : '↻ Refresh'}</button>
            </div>

            {orders.length === 0 ? (
              <div style={{ padding: '22px', textAlign: 'center', border: '1px dashed var(--border-dim)', borderRadius: '10px' }}>
                <p style={{ margin: 0, fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                  {ordersErr || 'No imagery orders yet. Pick an ROI and purchase above.'}
                </p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {orders.map(o => <OrderRow key={o.id} o={o} />)}
              </div>
            )}
            <p style={{ margin: '8px 0 0', fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
              Orders are server-persisted and also surface in the 📊 Results catalogue.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div style={{ padding: '10px 20px', borderTop: '1px solid var(--border-dim)', background: 'var(--bg-secondary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <p style={{ margin: 0, fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
            Stripe TEST mode · sandbox — nothing is really charged or delivered.
          </p>
          <button onClick={onClose} style={{
            padding: '7px 14px', fontSize: '11px', fontFamily: 'var(--font-display)', fontWeight: 700,
            borderRadius: '6px', border: '1px solid var(--border-dim)', background: 'var(--bg-primary)', color: 'var(--text-secondary)', cursor: 'pointer',
          }}>Close</button>
        </div>

        {/* Toast */}
        {toast && (
          <div style={{
            position: 'absolute', bottom: '64px', left: '50%', transform: 'translateX(-50%)',
            padding: '10px 16px', borderRadius: '8px', fontSize: '11px', fontFamily: 'var(--font-mono)', fontWeight: 700,
            color: '#fff', background: toast.kind === 'err' ? '#dc2626' : '#059669',
            boxShadow: '0 8px 20px rgba(15,23,42,0.30)', maxWidth: '90%', textAlign: 'center',
          }}>{toast.msg}</div>
        )}
      </div>
    </div>
  );
}

function badge(color, bg) {
  return {
    fontSize: '8px', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em',
    padding: '2px 6px', borderRadius: '5px', color, background: bg,
  };
}

function OrderRow({ o }) {
  const paid = o.payment_status === 'PAID';
  const payColor = paid ? '#065f46' : '#92400e';
  const payBg = paid ? 'rgba(5,150,105,0.18)' : 'rgba(245,158,11,0.18)';
  const sh = (o.sh_status || (o.sandbox ? 'sandbox' : '—'));
  const shConfirmed = sh === 'confirmed' || sh === 'created';
  const shColor = sh === 'failed' ? '#991b1b' : shConfirmed ? '#1d4ed8' : '#3730a3';
  const shBg = sh === 'failed' ? 'rgba(220,38,38,0.16)' : shConfirmed ? 'rgba(37,99,235,0.14)' : 'rgba(99,102,241,0.16)';

  return (
    <div style={{ padding: '11px 13px', borderRadius: '10px', border: '1px solid var(--border-dim)', background: 'var(--bg-secondary)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', flexWrap: 'wrap', marginBottom: '6px' }}>
        <span style={{ fontSize: '11px', fontFamily: 'var(--font-display)', fontWeight: 800, color: 'var(--text-primary)' }}>
          {o.provider || 'VHR'} · {km2(o.area_km2)} · {eur(o.total_eur)}
        </span>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          <span style={badge(payColor, payBg)}>{o.payment_status || 'PENDING'}</span>
          <span style={badge(shColor, shBg)}>SH {String(sh).toUpperCase()}</span>
          {o.sandbox && <span style={badge('#3730a3', 'rgba(99,102,241,0.16)')}>SANDBOX</span>}
        </div>
      </div>
      <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
        <span>#{String(o.id || '').slice(0, 10)}</span>
        <span>{fmtDate(o.created_at)}</span>
        {o.resolution_m != null && <span>~{o.resolution_m} m</span>}
        {Array.isArray(o.bbox) && o.bbox.length >= 4 && (
          <span>ROI {o.bbox[0].toFixed(3)},{o.bbox[1].toFixed(3)} → {o.bbox[2].toFixed(3)},{o.bbox[3].toFixed(3)}</span>
        )}
      </div>
      {o.error && <p style={{ margin: '6px 0 0', fontSize: '9px', fontFamily: 'var(--font-mono)', color: '#991b1b' }}>Error: {String(o.error).slice(0, 140)}</p>}
      {!paid && o.checkout_url && (
        <a href={o.checkout_url} style={{ display: 'inline-block', marginTop: '6px', fontSize: '9px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#1d4ed8' }}>
          Resume checkout →
        </a>
      )}
    </div>
  );
}
