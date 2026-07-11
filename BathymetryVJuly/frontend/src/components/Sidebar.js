import React, { useState, useRef } from 'react';
import HelpIcon from './HelpIcon';
import PlanetBackupPanel from './PlanetBackupPanel';
import { INTERNAL_ANALYSIS, SHOW_IHO_SURFACE } from './internalMode';

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar — production UI exposes exactly TWO methods:
//   1. Clustered SDB  (Mapbox VHR + 12-band S2 + OSM + K-means + HGB ensemble)
//   2. Almar Wave Physics  (S2Shores: wave-dispersion bathymetry)
// All other historical method handlers are accepted via props but no longer
// rendered, so existing wiring remains intact.
// ─────────────────────────────────────────────────────────────────────────────
export default function Sidebar({
  roi, params, setParams,
  onClusteredExtract, onAlmar, onVeryHrMle,
  loading, error, results,
  onExport,
  logs,
  userPoints,
  onUploadObserved, onLoadObserved, observedData, onLoadObservedMeta,
  onUploadShapefile, validationPts, onValidate, validationResults,
  vhrJobs, onVhrJobStart, onOpenVhrJobsModal,
  onVhrMlePro,
  onIBoatingPipeline,
  onShowOverlay,
  // (legacy props not rendered)
}) {
  // Resolution run mode for the CLEAR run flow: 10m | 20m | 50m | 100m | vhr.
  const resolution = params.resolution || '10m';
  const RES_OPTS = INTERNAL_ANALYSIS ? [
    { k: '10m',  label: '10 m',    hint: 'Standard Sentinel-2 (production SDB)' },
    { k: '20m',  label: '20 m',    hint: 'Fast overview · coarse Sentinel-2 grid' },
    { k: '50m',  label: '50 m',    hint: 'Regional overview · very fast' },
    { k: '100m', label: '100 m',   hint: 'Basin-scale overview · fastest' },
    { k: 'vhr',  label: 'Very HR', hint: 'Mapbox ~1 m detail · slower' },
  ] : [
    { k: '10m',  label: '10 m',           hint: 'Standard production depth' },
    { k: '20m',  label: '20 m',           hint: 'Fast overview · coarse grid' },
    { k: '50m',  label: '50 m',           hint: 'Regional overview · very fast' },
    { k: '100m', label: '100 m',          hint: 'Basin-scale overview · fastest' },
    { k: 'vhr',  label: 'High-Res',       hint: '~1 m detail · slower' },
  ];
  const PRESET_KEYS = ['khalifa_port', 'old_mussafah', 'abu_al_abyad', 'jbel_dhanna'];
  const usesTraining = PRESET_KEYS.includes(params.pro_site_key);
  const [tab, setTab] = useState('data');
  // i-Boating chart pipeline is HIDDEN by default — the user must explicitly
  // opt in. Public-chart OCR quality varies by area, so FUSION mode is the
  // recommended path for AD Ports operational areas.
  const [showIboating, setShowIboating] = useState(false);
  // Two-way ROI binding: the N/S/E/W fields are editable — typing a new
  // coordinate and pressing Enter (or leaving the field) redraws the ROI
  // rectangle on the map via the global 'setRoi' event. While a field is
  // being typed its draft value wins; otherwise the live roi value is shown
  // (so drag/resize on the map still updates the fields in real time).
  const [roiDraft, setRoiDraft] = useState({});
  const commitRoiField = (k, raw) => {
    setRoiDraft(d => { const nd = { ...d }; delete nd[k]; return nd; });
    const v = parseFloat(raw);
    if (!roi || !Number.isFinite(v) || v === roi[k]) return;
    const nb = { ...roi, [k]: v };
    if (nb.north <= nb.south || nb.east <= nb.west) return; // reject degenerate box
    window.dispatchEvent(new CustomEvent('setRoi', { detail: nb }));
  };
  const observedRef = useRef(null);
  const shpRef = useRef(null);
  const activeJobs = (vhrJobs || []);
  const runningCount = activeJobs.filter(j => j.status === 'processing').length;
  const doneCount = activeJobs.filter(j => j.status === 'done').length;
  const observedPts = (userPoints || []).filter(p => p?.photon_class === 'observed').length;

  // Train fraction fixed at 80 % per UX spec — no slider exposed.
  const trainFrac = 0.80;
  const nClusters = typeof params.n_clusters === 'number' ? params.n_clusters : 8;
  const augment = params.augment !== false;
  const imagerySource = params.imagery_source === 's2' ? 's2' : 'vhr';

  const m = results?.metrics;
  const aug = results?.augmentation?.counts;
  const perBand = results?.per_band;
  const s44Img = results?.s44_bands_png_b64;
  const overlayImg = results?.overlay_png_b64;
  const depthImg = results?.depth_png_b64;
  const vhrImg = results?.vhr_image_b64;
  const scatterImg = results?.scatter_png_b64;

  // Quality badge based on the S-44 Order 1A pass rate (95% = IHO target)
  const s44_1a = m?.s44_1a_pct;
  let badge = null;
  if (typeof s44_1a === 'number') {
    if (s44_1a >= 95) badge = { label: SHOW_IHO_SURFACE ? 'S-44 ORDER 1a — PASS' : 'Highest confidence', color: '#059669', bg: 'rgba(5,150,105,0.12)' };
    else if (s44_1a >= 80) badge = { label: 'High confidence', color: '#0284c7', bg: 'rgba(2,132,199,0.12)' };
    else if (s44_1a >= 60) badge = { label: 'Moderate confidence', color: '#d97706', bg: 'rgba(217,119,6,0.12)' };
    else badge = { label: 'Indicative only', color: '#b45309', bg: 'rgba(180,83,9,0.12)' };
  }

  return (
    <aside style={{ background:'var(--bg-primary)', borderRight:'1px solid var(--border-dim)', display:'flex', flexDirection:'column', overflow:'hidden', gridRow:'2', boxShadow:'2px 0 8px rgba(0,0,0,0.04)' }}>
      <div style={{ display:'flex', borderBottom:'1px solid var(--border-dim)', background:'var(--bg-secondary)' }}>
        {[{id:'data',label:'DATA'},{id:'logs',label:'LOGS'}].map(t => (
          <button key={t.id} onClick={()=>setTab(t.id)} style={{
            flex:1, padding:'12px', fontSize:'10px', fontFamily:'var(--font-mono)', fontWeight:600, border:'none', cursor:'pointer', letterSpacing:'0.08em',
            background:tab===t.id?'var(--bg-primary)':'transparent', color:tab===t.id?'var(--accent-primary)':'var(--text-dim)',
            borderBottom:tab===t.id?'2px solid var(--accent-primary)':'2px solid transparent',
          }}>{t.label}</button>
        ))}
      </div>

      <div style={{ flex:1, overflow:'auto', padding:'20px' }}>
        {tab==='data' && (
          <div>
            {/* ─────── Region of interest ─────── */}
            <h3 style={{ fontSize:'13px', fontWeight:700, color:'var(--text-primary)', marginBottom:'12px', display:'flex', alignItems:'center' }}>
              Region of Interest
              <HelpIcon text="Draw a rectangle on the map, click a preset, or edit the rectangle. The chosen ROI is what both methods run on." />
            </h3>
            <div style={{ padding:'14px', borderRadius:'var(--radius)', marginBottom:'16px', border:`1px solid ${roi?'rgba(5,150,105,0.2)':'var(--border-dim)'}`, background:roi?'rgba(5,150,105,0.03)':'var(--bg-secondary)' }}>
              {roi ? (
                <>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'6px', fontFamily:'var(--font-mono)', fontSize:'11px' }}>
                    {[['N','north'],['S','south'],['E','east'],['W','west']].map(([d,k])=>(
                      <div key={d} style={{ padding:'6px 10px', background:'var(--bg-primary)', borderRadius:'6px', display:'flex', justifyContent:'space-between', alignItems:'center', border:'1px solid var(--border-dim)' }}>
                        <span style={{color:'var(--text-dim)',fontWeight:600}}>{d}</span>
                        <span style={{display:'flex',alignItems:'center'}}>
                          <input
                            type="text" inputMode="decimal"
                            value={roiDraft[k] !== undefined ? roiDraft[k] : (roi[k] !== undefined && roi[k] !== null ? roi[k].toFixed(4) : '')}
                            onChange={e=>setRoiDraft(dr=>({...dr,[k]:e.target.value}))}
                            onBlur={e=>commitRoiField(k, e.target.value)}
                            onKeyDown={e=>{ if(e.key==='Enter') e.target.blur(); }}
                            title={`Edit the ${d} boundary — Enter redraws the ROI on the map`}
                            style={{ width:'62px', textAlign:'right', border:'none', outline:'none', background:'transparent',
                              fontFamily:'var(--font-mono)', fontSize:'11px', color:'var(--text-primary)', fontWeight:500, padding:0 }}
                          />
                          <span style={{color:'var(--text-primary)',fontWeight:500}}>°</span>
                        </span>
                      </div>
                    ))}
                  </div>
                  <button
                    onClick={()=>window.dispatchEvent(new CustomEvent('zoomToRoi',{detail:roi}))}
                    title="Fly to current ROI"
                    style={{ marginTop:'8px', width:'100%', padding:'8px',
                      fontSize:'10px', fontFamily:'var(--font-mono)', fontWeight:700,
                      borderRadius:'6px', cursor:'pointer',
                      border:'1px solid rgba(2,132,199,0.3)',
                      background:'rgba(2,132,199,0.06)', color:'#0284c7' }}
                  >Zoom to ROI</button>
                </>
              ) : (
                <p style={{ textAlign:'center', color:'var(--text-dim)', fontSize:'12px', padding:'8px 0' }}>Draw a rectangle on the map</p>
              )}
              <div style={{display:'flex',gap:'3px',marginTop:'8px',flexWrap:'wrap'}}>
                {[
                  // Predefined regions are tuned to ≈10 m native S2; larger
                  // user-drawn ROIs are still accepted and rendered at
                  // coarser native resolution (20 / 30 / 50 m).
                  {key:'khalifa_port',label:'Khalifa Port',bbox:{west:54.636,south:24.785,east:54.690,north:24.840}, standard_date:'2024-10-15'},
                  {key:'old_mussafah',label:'Old Mussafah',bbox:{west:54.355,south:24.412,east:54.412,north:24.466}, standard_date:'2024-08-15'},
                  {key:'abu_al_abyad',label:'Abu Al Abyad',bbox:{west:53.852,south:24.213,east:53.910,north:24.267}, standard_date:'2024-08-15'},
                  {key:'jbel_dhanna',label:'Jbel Dhanna',bbox:{west:52.5692,south:24.1989,east:52.6186,north:24.2440}, standard_date:'2024-02-15'},
                ].map(site=>(
                  <button key={site.key} onClick={()=>{
                    window.dispatchEvent(new CustomEvent('setRoi',{detail:site.bbox}));
                    const t = new Date(site.standard_date + 'T00:00:00Z');
                    const lo = new Date(t.getTime() - 10*86400000);
                    const hi = new Date(t.getTime() + 10*86400000);
                    const iso = (x) => x.toISOString().slice(0,10);
                    setParams(p=>({...p, pro_site_key:site.key,
                      s2_date:site.standard_date,
                      start_date:iso(lo), end_date:iso(hi)}));
                  }} title={`Set ROI to ${site.label} (calibrated ${site.standard_date})`} style={{
                    padding:'5px 10px',fontSize:'9px',fontFamily:'var(--font-mono)',fontWeight:600,borderRadius:'4px',cursor:'pointer',
                    border:(params.pro_site_key===site.key)?'1px solid #0f172a':'1px solid var(--border-dim)',
                    background:(params.pro_site_key===site.key)?'#0f172a':'var(--bg-primary)',
                    color:(params.pro_site_key===site.key)?'#fff':'var(--text-secondary)',
                  }}>{site.label}</button>
                ))}
              </div>
            </div>

            {/* ─────── CLEAR RUN FLOW · resolution selector + run ─────── */}
            <div style={{marginBottom:'16px',padding:'16px',borderRadius:'var(--radius)',
                 border:'2px solid rgba(2,132,199,0.45)',
                 background:'linear-gradient(135deg, rgba(2,132,199,0.08), rgba(13,148,136,0.05))',
                 boxShadow:'0 2px 12px rgba(2,132,199,0.12)'}}>
              <div style={{display:'flex',alignItems:'center',gap:'8px',marginBottom:'8px'}}>
                <span style={{display:'flex',alignItems:'center',justifyContent:'center',width:'20px',height:'20px',borderRadius:'50%',background:'#0284c7',color:'#fff',fontSize:'11px',fontWeight:800,fontFamily:'var(--font-mono)'}}>1</span>
                <h3 style={{fontSize:'13px',fontWeight:800,color:'#0c4a6e',margin:0,letterSpacing:'-0.01em'}}>Run Bathymetry</h3>
              </div>
              <p style={{fontSize:'10px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',lineHeight:1.5,margin:'0 0 12px'}}>
                Pick a region / ROI → choose a resolution → run.
              </p>

              {/* Step label */}
              <label style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'#0c4a6e',fontWeight:800,letterSpacing:'0.08em',display:'flex',alignItems:'center',gap:'5px',marginBottom:'6px'}}>
                RESOLUTION
                <HelpIcon text={INTERNAL_ANALYSIS ? "Output grid resolution: 10 m = standard Sentinel-2 production SDB, 20/50/100 m = progressively faster coarse overviews. Very HR = Mapbox ~1 m base raster for fine detail (slower)." : "Output grid resolution: 10 m = standard production depth, 20/50/100 m = faster coarse overviews. High-Res = ~1 m detail (slower)."} />
              </label>
              <div style={{display:'grid',gridTemplateColumns:`repeat(${RES_OPTS.length},1fr)`,gap:'5px',marginBottom:'8px'}}>
                {RES_OPTS.map(o=>(
                  <button key={o.k} type="button" onClick={()=>setParams(p=>({...p,resolution:o.k,
                    // keep the numeric output-resolution param in sync for every
                    // endpoint that reads resolution_m (quick-analyse, extract, DL Pro)
                    resolution_m:o.k==='vhr'?p.resolution_m:parseInt(o.k,10)}))} style={{
                    padding:'9px 4px',fontSize:'11px',fontFamily:'var(--font-mono)',fontWeight:800,borderRadius:'7px',cursor:'pointer',
                    border:resolution===o.k?'2px solid #0284c7':'1px solid var(--border-dim)',
                    background:resolution===o.k?'rgba(2,132,199,0.14)':'var(--bg-primary)',
                    color:resolution===o.k?'#0369a1':'var(--text-dim)',
                    boxShadow:resolution===o.k?'0 1px 6px rgba(2,132,199,0.25)':'none',
                  }}>{o.label}</button>
                ))}
              </div>
              <p style={{fontSize:'9.5px',fontFamily:'var(--font-mono)',color:'#475569',lineHeight:1.5,margin:'0 0 12px',minHeight:'14px'}}>
                {RES_OPTS.find(o=>o.k===resolution)?.hint}
              </p>

              {/* In-region training badge */}
              {usesTraining && (
                <div style={{display:'flex',alignItems:'center',gap:'6px',marginBottom:'10px',padding:'7px 9px',borderRadius:'6px',
                     background:'rgba(16,185,129,0.08)',border:'1px solid rgba(16,185,129,0.3)'}}>
                  <span style={{fontSize:'12px'}}>✓</span>
                  <span style={{fontSize:'9px',fontFamily:'var(--font-mono)',fontWeight:700,color:'#047857',lineHeight:1.4}}>
                    {INTERNAL_ANALYSIS
                      ? `Uses in-region training — ${params.pro_site_key.replace(/_/g,' ')} pretrained model + in-situ calibration`
                      : `Region-calibrated — ${params.pro_site_key.replace(/_/g,' ')}`}
                  </span>
                </div>
              )}

              <button
                onClick={()=>onClusteredExtract && onClusteredExtract(
                  usesTraining ? params.pro_site_key : null,
                  { resolution }
                )}
                disabled={!roi || loading}
                title={INTERNAL_ANALYSIS ? "Run satellite-derived bathymetry on the selected ROI at the chosen resolution. Default regions auto-use their in-region training." : "Run the depth computation on the selected area at the chosen resolution."}
                style={{
                  width:'100%', padding:'14px', fontSize:'13px', fontFamily:'var(--font-display)', fontWeight:800, borderRadius:'var(--radius)', border:'none',
                  cursor:(roi&&!loading)?'pointer':'not-allowed',
                  opacity:(roi&&!loading)?1:0.45,
                  background:(roi&&!loading)?'linear-gradient(135deg,#0284c7 0%,#0d9488 100%)':'var(--bg-secondary)',
                  color:(roi&&!loading)?'#fff':'var(--text-dim)',
                  boxShadow:(roi&&!loading)?'0 4px 16px rgba(2,132,199,0.4)':'none',
                  letterSpacing:'0.4px',
                }}
              >{loading ? '⟳ Computing…' : `▶ Run @ ${RES_OPTS.find(o=>o.k===resolution)?.label}${usesTraining ? (INTERNAL_ANALYSIS ? ' · in-region training' : ' · region-calibrated') : ''}`}</button>
              {!roi && (
                <p style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',textAlign:'center',margin:'7px 0 0'}}>
                  Draw an ROI or pick a region above to enable run.
                </p>
              )}
            </div>

            {/* ─────── Predefined regions / custom upload ─────── */}
            <div style={{marginBottom:'16px',padding:'16px',borderRadius:'var(--radius)',border:'1px solid rgba(16,185,129,0.25)',background:'var(--bg-secondary)'}}>
              <h4 style={{fontSize:'11px',fontFamily:'var(--font-mono)',fontWeight:700,color:'#10b981',marginBottom:'6px',display:'flex',alignItems:'center'}}>
                PREDEFINED REGIONS
                <HelpIcon text={INTERNAL_ANALYSIS ? "Calibrated UAE bathymetry — pick a region and the model runs against Sentinel-2 with no upload needed. You can also upload custom calibration data for any region." : "Calibrated regions — pick a region and the depth runs immediately, no upload needed. You can also upload custom calibration data for any region."} />
              </h4>
              <p style={{fontSize:'10px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',lineHeight:1.6,marginBottom:'8px'}}>
                {INTERNAL_ANALYSIS
                  ? 'One-click predefined regions — calibrated UAE bathymetry runs immediately on Sentinel-2.'
                  : 'One-click calibrated regions — depth runs immediately.'}
              </p>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'4px',marginBottom:'8px'}}>
                {[
                  // bbox tuned for ≤40 km² (10 m native res) + standard_date = lowest-RMSE month from
                  // the per-region date sweep (validated on 20 % held-out
                  // in-situ). Clicking a region sets ROI + date so the
                  // user lands on the calibrated optimum immediately.
                  {key:'khalifa_port',label:'Khalifa Port',bbox:{west:54.636,south:24.785,east:54.690,north:24.840},
                   standard_date:'2024-10-15', metrics:{rmse:1.96, bias:-0.04, r2:0.62}},
                  {key:'old_mussafah',label:'Old Mussafah',bbox:{west:54.355,south:24.412,east:54.412,north:24.466},
                   standard_date:'2024-08-15', metrics:{rmse:0.73, bias:+0.14, r2:0.22}},
                  {key:'abu_al_abyad',label:'Abu Al Abyad',bbox:{west:53.852,south:24.213,east:53.910,north:24.267},
                   standard_date:'2024-08-15'},  // no in-situ for sweep — pick clear-month default
                  {key:'jbel_dhanna',label:'Jbel Dhanna',bbox:{west:52.5692,south:24.1989,east:52.6186,north:24.2440},
                   standard_date:'2024-02-15', metrics:{rmse:0.80, bias:+0.15, r2:0.83}},
                ].map(s=>(
                  <button key={s.key} onClick={()=>{
                    // Predefined regions set ROI + the calibrated standard
                    // date for that location (and ±10 day window). No
                    // in-situ points are rendered on the map.
                    window.dispatchEvent(new CustomEvent('setRoi',{detail:s.bbox}));
                    const t = new Date(s.standard_date + 'T00:00:00Z');
                    const lo = new Date(t.getTime() - 10*86400000);
                    const hi = new Date(t.getTime() + 10*86400000);
                    const iso = (x) => x.toISOString().slice(0,10);
                    setParams(p=>({...p, pro_site_key:s.key,
                      s2_date:s.standard_date,
                      start_date:iso(lo), end_date:iso(hi)}));
                    // Fetch dataset metadata (datum / CRS / count / bbox) for the sidebar card
                    if (onLoadObservedMeta) onLoadObservedMeta(s.key);
                  }} disabled={loading}
                     title={(INTERNAL_ANALYSIS && s.metrics)
                       ? `Calibrated ${s.label} · ${s.standard_date} · RMSE ${s.metrics.rmse} m · bias ${s.metrics.bias>=0?'+':''}${s.metrics.bias} m · R² ${s.metrics.r2}`
                       : `Calibrated ${s.label} · ${s.standard_date}`}
                     style={{
                    padding:'8px 4px',fontSize:'9px',fontFamily:'var(--font-mono)',fontWeight:700,borderRadius:'6px',cursor:loading?'wait':'pointer',
                    border:(params.pro_site_key===s.key)?'1.5px solid #10b981':'1px solid rgba(16,185,129,0.3)',
                    background:(params.pro_site_key===s.key)?'rgba(16,185,129,0.18)':'rgba(16,185,129,0.04)',
                    color:'#10b981',opacity:loading?0.5:1,
                  }}>{s.label}</button>
                ))}
              </div>
              <div style={{display:'inline-flex',alignItems:'center',gap:'5px',marginBottom:'8px',padding:'4px 8px',borderRadius:'5px',
                   background:'rgba(16,185,129,0.07)',border:'1px solid rgba(16,185,129,0.25)'}}>
                <span style={{fontSize:'10px'}}>✓</span>
                <span style={{fontSize:'8.5px',fontFamily:'var(--font-mono)',fontWeight:700,color:'#047857',letterSpacing:'0.02em'}}>
                  {INTERNAL_ANALYSIS ? 'uses in-region training (pretrained + in-situ calibration)' : 'region-calibrated'}
                </span>
              </div>

              {/* ── Upload reference system: coordinate order / CRS / vertical datum ── */}
              <div style={{marginBottom:'8px',padding:'10px',borderRadius:'6px',background:'var(--bg-primary)',border:'1px solid var(--border-dim)'}}>
                <div style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-secondary)',fontWeight:800,letterSpacing:'0.08em',display:'flex',alignItems:'center',marginBottom:'8px',paddingBottom:'6px',borderBottom:'1px solid var(--border-dim)'}}>
                  UPLOAD REFERENCE SYSTEM
                  <HelpIcon text="How the coordinates and depths in your uploaded survey file are interpreted before they are reprojected to WGS84 and compared against the model." />
                </div>
                <label style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.06em',display:'flex',alignItems:'center',marginBottom:'4px'}}>
                  COORDINATE ORDER
                  <HelpIcon text="Column order of your XYZ/CSV file. 'Lat, Lon, Z' is the default; pick 'Lon, Lat, Z' for traditional GIS X,Y,Z files. Shapefiles ignore this (axis order is fixed by the format)." />
                </label>
                <div style={{display:'flex',gap:'3px',marginBottom:'6px'}}>
                  {[{v:'latlon',l:'Lat, Lon, Z'},{v:'lonlat',l:'Lon, Lat, Z'}].map(o=>(
                    <button key={o.v} onClick={()=>setParams(p=>({...p,obs_coord_order:o.v}))} style={{
                      flex:1,padding:'5px',fontSize:'9px',fontFamily:'var(--font-mono)',fontWeight:700,borderRadius:'5px',cursor:'pointer',
                      border:(params.obs_coord_order||'latlon')===o.v?'1.5px solid #10b981':'1px solid var(--border-dim)',
                      background:(params.obs_coord_order||'latlon')===o.v?'rgba(16,185,129,0.12)':'var(--bg-secondary)',
                      color:(params.obs_coord_order||'latlon')===o.v?'#10b981':'var(--text-dim)',
                    }}>{o.l}</button>
                  ))}
                </div>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'6px'}}>
                  <div>
                    <label style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.06em',display:'flex',alignItems:'center',marginBottom:'4px'}}>
                      UTM / EPSG
                      <HelpIcon text="EPSG code of the source projection (e.g. 32640 for UTM Zone 40N). Leave empty for geographic degrees (WGS84). Points are reprojected to WGS84 (EPSG:4326) on upload." />
                    </label>
                    <input type="text" inputMode="numeric" placeholder="e.g. 32640 (blank = degrees)"
                      value={params.obs_utm_epsg||''}
                      onChange={e=>{const v=e.target.value.replace(/[^0-9]/g,'');setParams(p=>({...p,obs_utm_epsg:v}));}}
                      style={{width:'100%',padding:'5px 6px',fontSize:'9px',fontFamily:'var(--font-mono)',borderRadius:'5px',
                        border:'1px solid var(--border-dim)',background:'var(--bg-secondary)',color:'var(--text)'}} />
                  </div>
                  <div>
                    <label style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.06em',display:'flex',alignItems:'center',marginBottom:'4px'}}>
                      VERTICAL DATUM
                      <HelpIcon text="Vertical reference of your depths: LAT (Lowest Astronomical Tide), MSL (Mean Sea Level), CD (Chart Datum) or WGS84 ellipsoid. Stored with the dataset and propagated to exports." />
                    </label>
                    <select value={params.obs_v_datum||'LAT'} onChange={e=>setParams(p=>({...p,obs_v_datum:e.target.value}))}
                      style={{width:'100%',padding:'5px 6px',fontSize:'9px',fontFamily:'var(--font-mono)',borderRadius:'5px',
                        border:'1px solid var(--border-dim)',background:'var(--bg-secondary)',color:'var(--text)'}}>
                      {['LAT','MSL','CD','WGS84'].map(d=><option key={d} value={d}>{d}</option>)}
                    </select>
                  </div>
                </div>
              </div>

              <input ref={observedRef} type="file" accept=".xyz,.txt,.csv,.shp,.dbf,.shx,.prj,.cpg,.zip" multiple style={{display:'none'}}
                onChange={e=>{
                  if(e.target.files?.length && onUploadObserved){
                    onUploadObserved(e.target.files, {
                      coord_order: params.obs_coord_order || 'latlon',
                      utm_epsg: params.obs_utm_epsg || '',
                      vertical_datum: params.obs_v_datum || 'LAT',
                      horizontal_datum: params.obs_h_datum || 'WGS84',
                    });
                  }
                }} />
              <button onClick={()=>observedRef.current?.click()} disabled={loading} style={{
                width:'100%',padding:'10px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:600,borderRadius:'var(--radius-sm)',cursor:loading?'wait':'pointer',
                border:'1px dashed rgba(16,185,129,0.3)',background:'rgba(16,185,129,0.03)',color:'#10b981',opacity:loading?0.5:1,
              }}>Upload custom calibration data</button>

              {(observedPts > 0 || observedData) && (
                <div style={{marginTop:'8px',padding:'8px',borderRadius:'6px',background:'var(--bg-primary)',border:'1px solid var(--border-dim)',fontSize:'10px',fontFamily:'var(--font-mono)'}}>
                  <div style={{display:'flex',justifyContent:'space-between',marginBottom:observedData?'6px':0}}>
                    <span style={{fontWeight:700,color:'#10b981'}}>
                      {(observedData?.count ?? observedPts)} calibration pts{observedPts>0?' loaded':''}
                    </span>
                    {observedData?.site && <span style={{color:'var(--text-dim)'}}>{observedData.site}</span>}
                  </div>
                  {observedData && (
                    <div style={{display:'grid',gridTemplateColumns:'auto 1fr',gap:'2px 8px',fontSize:'9px',color:'var(--text-dim)'}}>
                      <span>Vertical datum</span>
                      <span style={{color:'var(--text)',fontWeight:600,textAlign:'right'}}>{observedData.vertical_datum||'LAT'}</span>
                      <span>Horizontal datum</span>
                      <span style={{color:'var(--text)',fontWeight:600,textAlign:'right'}}>{observedData.horizontal_datum||'WGS84'}</span>
                      <span>Original CRS</span>
                      <span style={{color:'var(--text)',fontWeight:600,textAlign:'right'}}>{observedData.crs||(observedData.source_epsg?`EPSG:${observedData.source_epsg}`:'WGS84 (EPSG:4326)')}</span>
                      {observedData.bbox && (<>
                        <span style={{gridColumn:'1 / -1',marginTop:'2px'}}>Bounding box</span>
                        <span style={{gridColumn:'1 / -1',color:'var(--text)',fontWeight:600,whiteSpace:'normal',wordBreak:'break-word',lineHeight:1.4}}>
                          {Number(observedData.bbox.west).toFixed(4)}–{Number(observedData.bbox.east).toFixed(4)}°E · {Number(observedData.bbox.south).toFixed(4)}–{Number(observedData.bbox.north).toFixed(4)}°N
                        </span>
                      </>)}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ─────── METHOD 1: Very HR Bathymetry ─────── */}
            <div style={{marginBottom:'12px',padding:'16px',borderRadius:'var(--radius)',
                border:'2px solid #059669',
                background:'linear-gradient(135deg, rgba(5,150,105,0.10), rgba(16,185,129,0.06))',
                boxShadow:'0 2px 12px rgba(5,150,105,0.18)'}}>
              <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'4px'}}>
                <p style={{fontSize:'11px',fontFamily:'var(--font-mono)',color:'#065f46',fontWeight:800,letterSpacing:'0.08em',margin:0,display:'flex',alignItems:'center'}}>
                  ✦ HIGH-RESOLUTION DEPTH
                  <HelpIcon text={INTERNAL_ANALYSIS
                    ? "Method 1 — clustered SDB: Mapbox VHR base raster + 12-band Sentinel-2 + K-means seabed clusters + per-cluster ensemble. Set the options below, then run as a local job or the multi-scene PRO."
                    : "Method 1 — high-resolution depth from satellite imagery. Set the options below, then run; long jobs report progress in the JOBS window."} />
                </p>
              </div>
              <p style={{fontSize:'10px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',lineHeight:1.5,marginBottom:'10px'}}>
                <b>High-Resolution computation</b> — takes a long time to give a result
                at very high resolution.
              </p>

              {/* Knobs — train fraction is fixed at 80 % internally so the
                  user can't dial it down. Clusters knob retained for the
                  legacy CBR pipeline only. */}
              <div style={{display:'grid',gridTemplateColumns:'1fr',gap:'8px',marginBottom:'10px'}}>
                <div>
                  <label style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.06em',display:'flex',alignItems:'center',marginBottom:'4px'}}>
                    CLUSTERS
                    <HelpIcon text="Number of seabed classes (K-means) used to group similar-looking pixels before per-cluster depth regression. K=8 is the calibrated default." />
                  </label>
                  <div style={{display:'flex',gap:'3px'}}>
                    {[6,8,10].map(k=>(
                      <button key={k} onClick={()=>setParams(p=>({...p,n_clusters:k}))} style={{
                        flex:1,padding:'6px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:700,borderRadius:'5px',cursor:'pointer',
                        border:nClusters===k?'1.5px solid #059669':'1px solid var(--border-dim)',
                        background:nClusters===k?'rgba(5,150,105,0.12)':'var(--bg-primary)',
                        color:nClusters===k?'#065f46':'var(--text-dim)',
                      }}>K={k}</button>
                    ))}
                  </div>
                </div>
              </div>

              <label style={{display:'flex',alignItems:'center',gap:'6px',marginBottom:'10px',fontSize:'10px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',cursor:'pointer'}}>
                <input type="checkbox" checked={augment}
                  onChange={e=>setParams(p=>({...p,augment:e.target.checked}))}/>
                <span><b>Augment</b> sparse depth bands automatically</span>
                <HelpIcon text="Automatically adds extra reference depths (satellite lidar / nautical-chart soundings) in depth bands where your calibration data is sparse." />
              </label>

              {/* Sentinel-2 date — single-day calendar pick. Backend uses
                  ±10 days as the search window for the least-cloudy scene. */}
              <div style={{marginBottom:'10px'}}>
                <label style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.06em',display:'flex',alignItems:'center',marginBottom:'4px'}}>
                  {INTERNAL_ANALYSIS ? 'SENTINEL-2 DATE' : 'IMAGERY DATE'}  <span style={{fontWeight:500,color:'var(--text-dim)',marginLeft:'4px'}}>(±10 day search window)</span>
                  <HelpIcon text="Target acquisition date. The least-cloudy scene within ±10 days of this date is selected automatically." />
                </label>
                <input
                  type="date"
                  value={params.s2_date || '2024-10-15'}
                  onChange={e=>{
                    const d = e.target.value;
                    if (!d) return;
                    const t = new Date(d + 'T00:00:00Z');
                    const lo = new Date(t.getTime() - 10*86400000);
                    const hi = new Date(t.getTime() + 10*86400000);
                    const iso = (x) => x.toISOString().slice(0,10);
                    setParams(p => ({...p, s2_date:d, start_date:iso(lo), end_date:iso(hi)}));
                  }}
                  style={{
                    width:'100%', padding:'6px 10px', fontSize:'11px',
                    fontFamily:'var(--font-mono)', fontWeight:600,
                    borderRadius:'5px', border:'1px solid var(--border-dim)',
                    background:'var(--bg-primary)', color:'var(--text-primary)',
                  }}
                />
                <p style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',margin:'4px 0 0',lineHeight:1.4}}>
                  Window: <b>{params.start_date}</b> → <b>{params.end_date}</b> (least-cloudy scene chosen)
                </p>
              </div>

              {/* Imagery source toggle: Mapbox VHR (default, ≈2 m) vs Sentinel-2 only (≈10 m, free) */}
              <div style={{marginBottom:'10px'}}>
                <label style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.06em',display:'flex',alignItems:'center',marginBottom:'4px'}}>
                  IMAGERY SOURCE
                  <HelpIcon text={INTERNAL_ANALYSIS
                    ? "Mapbox VHR (~2 m/px) base raster for fine detail, or free Sentinel-2-only (10 m/px, no Mapbox quota)."
                    : "High-Res (~2 m/px) base imagery for finer detail, or Standard (10 m/px) — faster and always available."} />
                </label>
                <div style={{display:'flex',gap:'3px'}}>
                  {(INTERNAL_ANALYSIS ? [
                    {k:'vhr', label:'Mapbox VHR (~2 m)'},
                    {k:'s2',  label:'Sentinel-2 only (10 m)'},
                  ] : [
                    {k:'vhr', label:'High-Res (~2 m)'},
                    {k:'s2',  label:'Standard (10 m)'},
                  ]).map(o=>(
                    <button key={o.k} onClick={()=>setParams(p=>({...p,imagery_source:o.k}))} style={{
                      flex:1,padding:'6px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:700,borderRadius:'5px',cursor:'pointer',
                      border:imagerySource===o.k?'1.5px solid #059669':'1px solid var(--border-dim)',
                      background:imagerySource===o.k?'rgba(5,150,105,0.12)':'var(--bg-primary)',
                      color:imagerySource===o.k?'#065f46':'var(--text-dim)',
                    }}>{o.label}</button>
                  ))}
                </div>
                <p style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',margin:'4px 0 0',lineHeight:1.4}}>
                  {INTERNAL_ANALYSIS
                    ? (imagerySource==='s2'
                        ? 'Free Sentinel-2 L2A 10 m grid as base raster. No Mapbox quota. Resolution ≈10 m/px.'
                        : 'Mapbox satellite mosaic (~2 m/px) as base raster, with S2 + ERA-5/SST features.')
                    : (imagerySource==='s2'
                        ? 'Standard 10 m base raster. Resolution ≈10 m/px.'
                        : 'High-resolution base raster (~2 m/px) for finer detail.')}
                </p>
              </div>

              {/* ── Very High Resolution (~1 m) button — clicking sends a LOCAL JOB ── */}
              <button
                onClick={()=>onVhrJobStart && onVhrJobStart({
                  label: `High-Resolution Depth${params.pro_site_key?` · ${params.pro_site_key}`:''}`,
                  params: { trainFrac, nClusters, augment, imagerySource, target_res_m: 1.0 },
                })}
                disabled={!roi}
                title="Submit this area as a High-Resolution (~1 m) local-processing job. You'll process it offline and upload the result."
                style={{
                  width:'100%', padding:'14px', fontSize:'12px', fontFamily:'var(--font-display)', fontWeight:800, borderRadius:'var(--radius)', border:'none',
                  cursor:roi?'pointer':'not-allowed',
                  opacity:roi?1:0.45,
                  background: roi ? 'linear-gradient(135deg,#059669 0%,#0d9488 100%)' : 'var(--bg-secondary)',
                  color: roi ? '#fff' : 'var(--text-dim)',
                  boxShadow: roi ? '0 4px 16px rgba(5,150,105,0.4)' : 'none',
                  letterSpacing:'0.5px',
                }}
              >Run High-Resolution Depth (~1 m)</button>

              {/* ── Very HR + MLE PRO — server-side: 10-scene S2 median + Mapbox VHR + MLE.
                  Reports a REAL progress percentage via the same JOBS panel. ── */}
              <button
                onClick={()=>onVhrMlePro && onVhrMlePro({
                  n_scenes: params.vhr_mle_n_scenes || 10,
                  year: params.vhr_mle_year || 2024,
                  target_res_m: params.vhr_mle_res_m || 2.0,
                  max_cloud: params.vhr_mle_cloud || 20,
                })}
                disabled={!roi}
                title={INTERNAL_ANALYSIS ? "Server-side: 10-scene Sentinel-2 median composite + Mapbox VHR + per-pixel inverse-variance MLE. Reports REAL % progress." : "Server-side multi-scene high-resolution depth product with per-pixel uncertainty. Reports real % progress."}
                style={{
                  marginTop:'8px', width:'100%', padding:'14px', fontSize:'12px',
                  fontFamily:'var(--font-display)', fontWeight:800, borderRadius:'var(--radius)',
                  border:'none', cursor:roi?'pointer':'not-allowed',
                  opacity:roi?1:0.45,
                  background: roi ? 'linear-gradient(135deg,#7c3aed 0%,#4f46e5 100%)' : 'var(--bg-secondary)',
                  color: roi ? '#fff' : 'var(--text-dim)',
                  boxShadow: roi ? '0 4px 16px rgba(124,58,237,0.40)' : 'none',
                  letterSpacing:'0.5px',
                }}
              >▶ Run High-Resolution PRO (multi-scene)</button>

              {/* JOBS button — opens the popup window listing every local job */}
              <button
                onClick={()=>onOpenVhrJobsModal && onOpenVhrJobsModal()}
                style={{
                  marginTop:'8px', width:'100%', padding:'11px', fontSize:'11px',
                  fontFamily:'var(--font-display)', fontWeight:800, borderRadius:'var(--radius)',
                  border:'1px solid rgba(79,70,229,0.45)',
                  background: activeJobs.length>0 ? 'linear-gradient(135deg,#7c3aed 0%,#4f46e5 100%)' : 'rgba(79,70,229,0.06)',
                  color: activeJobs.length>0 ? '#fff' : '#4f46e5',
                  cursor:'pointer', letterSpacing:'0.4px',
                  display:'flex', alignItems:'center', justifyContent:'space-between', gap:'6px',
                }}
              >
                <span>📦 JOBS · {activeJobs.length}</span>
                <span style={{fontSize:'9px',fontFamily:'var(--font-mono)',opacity:0.9}}>
                  {runningCount>0?`${runningCount} running`:''}{runningCount>0 && doneCount>0?' · ':''}{doneCount>0?`${doneCount} done`:''}
                  {runningCount===0 && doneCount===0?'open window →':''}
                </span>
              </button>

              {/* ── Multi-scene MLE: full-year stack ── */}
              <div style={{marginTop:'14px',padding:'12px',borderRadius:'var(--radius)',
                   border:'1px dashed rgba(2,132,199,0.45)',
                   background:'rgba(2,132,199,0.04)'}}>
                <p style={{fontSize:'10px',fontFamily:'var(--font-mono)',color:'#0369a1',fontWeight:800,letterSpacing:'0.08em',margin:'0 0 6px 0',display:'flex',alignItems:'center'}}>
                  ✦ MULTI-SCENE MLE (whole year)
                  <HelpIcon text="Stacks N scenes spread across the chosen year and combines them per pixel (inverse-variance) into one depth product with a per-pixel uncertainty map. More scenes = smoother but slower." />
                </p>
                <p style={{fontSize:'9.5px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',lineHeight:1.5,marginBottom:'8px'}}>
                  {INTERNAL_ANALYSIS
                    ? 'N evenly-spaced single-scene fetches across a calendar year, then per-pixel inverse-variance Maximum Likelihood (Almar-style). Runtime ≈ N × 5–15 s.'
                    : 'Combines N evenly-spaced scenes across a year into a single depth product with per-pixel uncertainty. Runtime ≈ N × 5–15 s.'}
                </p>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'6px',marginBottom:'8px'}}>
                  <div>
                    <label style={{fontSize:'8px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.06em',display:'block',marginBottom:'3px'}}>
                      YEAR
                    </label>
                    <input type="number" min="2017" max="2025" step="1"
                      value={params.mle_year || 2024}
                      onChange={e=>setParams(p=>({...p, mle_year:parseInt(e.target.value, 10)||2024}))}
                      style={{width:'100%',padding:'5px 8px',fontSize:'11px',fontFamily:'var(--font-mono)',fontWeight:600,
                              borderRadius:'5px',border:'1px solid var(--border-dim)',background:'var(--bg-primary)'}} />
                  </div>
                  <div>
                    <label style={{fontSize:'8px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.06em',display:'block',marginBottom:'3px'}}>
                      SCENES / YEAR
                    </label>
                    <div style={{display:'flex',gap:'3px'}}>
                      {[3,5,7].map(n=>(
                        <button key={n} type="button" onClick={()=>setParams(p=>({...p, mle_n_scenes:n}))} style={{
                          flex:1,padding:'5px 0',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:700,borderRadius:'5px',cursor:'pointer',
                          border:(params.mle_n_scenes||5)===n?'1.5px solid #0369a1':'1px solid var(--border-dim)',
                          background:(params.mle_n_scenes||5)===n?'rgba(2,132,199,0.12)':'var(--bg-primary)',
                          color:(params.mle_n_scenes||5)===n?'#0369a1':'var(--text-dim)',
                        }}>{n}</button>
                      ))}
                    </div>
                  </div>
                </div>
                <button onClick={()=>onVeryHrMle && onVeryHrMle({
                    year: params.mle_year || 2024,
                    n_scenes: params.mle_n_scenes || 5,
                  })}
                  disabled={!roi || loading} style={{
                    width:'100%', padding:'11px', fontSize:'11px', fontFamily:'var(--font-display)', fontWeight:700, borderRadius:'var(--radius)', border:'none',
                    cursor:(roi&&!loading)?'pointer':'not-allowed', opacity:(roi&&!loading)?1:0.45,
                    background: roi ? 'linear-gradient(135deg,#0284c7 0%,#0369a1 100%)' : 'var(--bg-secondary)',
                    color: roi ? '#fff' : 'var(--text-dim)',
                    boxShadow: roi ? '0 3px 10px rgba(2,132,199,0.30)' : 'none',
                    letterSpacing:'0.4px',
                  }}>
                  {loading?'⟳ MLE-stacking scenes…':`Run MLE · ${params.mle_n_scenes||5} scenes / ${params.mle_year||2024}`}
                </button>
              </div>
            </div>

            {/* ─────── Planet VHR backup status (ADPorts F2) ─────── */}
            <PlanetBackupPanel onShowOverlay={onShowOverlay} />

            {/* ─────── METHOD 2: Almar Wave Physics ─────── */}
            <div style={{marginBottom:'14px',padding:'14px',borderRadius:'var(--radius)',
                 border:'1.5px solid rgba(168,85,247,0.45)',
                 background:'linear-gradient(135deg, rgba(168,85,247,0.08), rgba(99,102,241,0.06))'}}>
              <p style={{fontSize:'11px',fontFamily:'var(--font-mono)',color:'#7e22ce',fontWeight:800,letterSpacing:'0.08em',margin:'0 0 4px 0',display:'flex',alignItems:'center'}}>
                {INTERNAL_ANALYSIS ? '⌇ ALMAR WAVE PHYSICS' : '⌇ WAVE-BASED DEPTH'}
                <HelpIcon text="Method 2 — depth from surface-wave physics; needs no reference data. Works best in open swell / visible wave fields; not suited to flat, sheltered water." />
              </p>
              <p style={{fontSize:'9.5px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',lineHeight:1.5,marginBottom:'8px'}}>
                {INTERNAL_ANALYSIS
                  ? <>S2 inter-band time-offset → wave-celerity → linear-dispersion <i> ω² = g·k·tanh(kh)</i> → depth.  Reference-free; works in open swell/wave fields.</>
                  : <>Reference-free depth from surface wave patterns. Works in open swell/wave fields.</>}
              </p>
              <button onClick={()=>onAlmar && onAlmar({resolutionM:50})}
                disabled={!roi||loading} style={{
                  width:'100%', padding:'12px', fontSize:'11px', fontFamily:'var(--font-display)', fontWeight:700, borderRadius:'var(--radius)', border:'none',
                  cursor:roi&&!loading?'pointer':'not-allowed', opacity:roi&&!loading?1:0.45,
                  background: roi ? 'linear-gradient(135deg,#a855f7 0%,#6366f1 100%)' : 'var(--bg-secondary)',
                  color: roi ? '#fff' : 'var(--text-dim)',
                  boxShadow: roi ? '0 3px 12px rgba(168,85,247,0.35)' : 'none',
                }}>{loading?'⟳ Computing…':(INTERNAL_ANALYSIS?'Run Almar Wave Physics':'Run Wave-Based Depth')}</button>
            </div>

            {/* ─────── Professional results report ─────── */}
            {m && (
              <div style={{marginBottom:'14px',borderRadius:'var(--radius)',
                   background:'#fff',border:'1px solid var(--border-dim)',
                   boxShadow:'0 4px 16px rgba(15,23,42,0.06)',overflow:'hidden'}}>

                {/* Report header */}
                <div style={{padding:'14px 16px',borderBottom:'1px solid var(--border-dim)',
                     background:'linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%)',color:'#fff'}}>
                  <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'4px'}}>
                    <span style={{fontSize:'10px',fontFamily:'var(--font-mono)',letterSpacing:'0.12em',opacity:0.7,fontWeight:700}}>
                      BATHYMETRY REPORT
                    </span>
                    {badge && (
                      <span style={{fontSize:'9px',fontFamily:'var(--font-mono)',fontWeight:700,
                            padding:'4px 10px',borderRadius:'12px',
                            background:badge.bg,color:'#fff',
                            border:`1px solid ${badge.color}`,letterSpacing:'0.06em'}}>
                        {badge.label}
                      </span>
                    )}
                  </div>
                  <div style={{fontSize:'14px',fontFamily:'var(--font-display)',fontWeight:700,letterSpacing:'-0.01em'}}>
                    {results.site && results.site !== 'custom' ? results.site.replace(/_/g,' ') : 'Custom ROI'}
                  </div>
                  <div style={{fontSize:'10px',fontFamily:'var(--font-mono)',opacity:0.65,marginTop:'2px'}}>
                    Calibrated bathymetry · grid ≈ {results.resolution_m} m/px · {results.elapsed_s}s
                    {results.augmentation?.n_scenes > 1 && (
                      <> · {results.augmentation.n_scenes}-scene median</>
                    )}
                  </div>
                  {INTERNAL_ANALYSIS && results.augmentation?.bias_correction?.applied && (
                    <div style={{fontSize:'9px',fontFamily:'var(--font-mono)',opacity:0.55,marginTop:'2px'}}>
                      bias-correct: RMSE {results.augmentation.bias_correction.pre_rmse_m}→
                      <b style={{color:'#059669'}}>{results.augmentation.bias_correction.post_rmse_m}</b> m
                      {' · '}
                      bias {results.augmentation.bias_correction.pre_bias_m>=0?'+':''}
                      {results.augmentation.bias_correction.pre_bias_m}→
                      <b style={{color:'#059669'}}>
                        {results.augmentation.bias_correction.post_bias_m>=0?'+':''}
                        {results.augmentation.bias_correction.post_bias_m}
                      </b> m
                    </div>
                  )}
                </div>

                {/* Turbidity / quality warnings */}
                {results?.turbidity_warning && (
                  <div style={{padding:'10px 14px',background:'rgba(217,119,6,0.06)',
                       borderBottom:'1px solid rgba(217,119,6,0.3)',
                       fontSize:'10px',fontFamily:'var(--font-mono)',color:'#92400e',lineHeight:1.5}}>
                    {results.turbidity_warning}
                  </div>
                )}

                {/* Big stat tiles — INTERNAL ONLY (accuracy/error metrics). The
                    client portal shows depth figures + maps, never RMSE/R²/IHO. */}
                {INTERNAL_ANALYSIS && (
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'1px',
                     background:'var(--border-dim)',padding:'1px',
                     borderBottom:'1px solid var(--border-dim)'}}>
                  {[
                    {k:'RMSE',v:m.rmse_m,u:'m',color:'#0f172a'},
                    {k:'MAE',v:m.mae_m,u:'m',color:'#0f172a'},
                    {k:'R²',v:m.r2,u:'',color:'#0f172a'},
                    {k:'Bias',v:m.bias_m,u:'m',color:Math.abs(m.bias_m||0)<0.3?'#059669':'#b45309'},
                    {k:'IHO 1A pass',v:m.s44_1a_pct,u:'%',color:s44_1a>=90?'#059669':(s44_1a>=70?'#0284c7':'#b45309')},
                    {k:'IHO Order 2 pass',v:m.s44_order2_pct,u:'%',color:'#0284c7'},
                  ].map((t,i)=>(
                    <div key={i} style={{padding:'12px 10px',background:'#fff'}}>
                      <div style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',letterSpacing:'0.08em',fontWeight:700,marginBottom:'2px'}}>
                        {t.k}
                      </div>
                      <div style={{fontSize:'18px',fontFamily:'var(--font-display)',fontWeight:700,color:t.color,letterSpacing:'-0.02em'}}>
                        {t.v ?? '—'}<span style={{fontSize:'10px',fontWeight:500,color:'var(--text-dim)',marginLeft:'2px'}}>{t.u}</span>
                      </div>
                    </div>
                  ))}
                </div>
                )}

                {/* Optional ≥ 5 m subset metrics — INTERNAL ONLY */}
                {INTERNAL_ANALYSIS && m.overall_excluding_shallow && (
                  <div style={{padding:'10px 14px',background:'rgba(2,132,199,0.04)',borderBottom:'1px solid var(--border-dim)',
                       fontSize:'10px',fontFamily:'var(--font-mono)',color:'#0c4a6e',lineHeight:1.6}}>
                    <span style={{fontWeight:700,letterSpacing:'0.06em'}}>≥ 5 m only</span>
                    {' — '}
                    RMSE <b>{m.overall_excluding_shallow.rmse_m} m</b> · R² <b>{m.overall_excluding_shallow.r2}</b> · IHO 1A <b>{m.overall_excluding_shallow.s44_1a_pct}%</b>
                  </div>
                )}

                {/* Side-by-side images */}
                {(overlayImg || depthImg || vhrImg) && (
                  <div style={{padding:'10px 12px',borderBottom:'1px solid var(--border-dim)'}}>
                    <div style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.08em',marginBottom:'6px'}}>
                      DEPTH MAP
                    </div>
                    <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'6px'}}>
                      {vhrImg && (
                        <figure style={{margin:0}}>
                          <img src={`data:image/png;base64,${vhrImg}`} alt="Mosaic"
                               style={{width:'100%',borderRadius:'6px',border:'1px solid var(--border-dim)',display:'block'}} />
                          <figcaption style={{fontSize:'8.5px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',marginTop:'2px',textAlign:'center'}}>
                            High-resolution mosaic
                          </figcaption>
                        </figure>
                      )}
                      {(overlayImg || depthImg) && (
                        <figure style={{margin:0}}>
                          <img src={`data:image/png;base64,${overlayImg || depthImg}`} alt="Depth"
                               style={{width:'100%',borderRadius:'6px',border:'1px solid var(--border-dim)',display:'block'}} />
                          <figcaption style={{fontSize:'8.5px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',marginTop:'2px',textAlign:'center'}}>
                            Predicted depth (0–25 m)
                          </figcaption>
                        </figure>
                      )}
                    </div>
                  </div>
                )}

                {/* S-44 bar chart — INTERNAL ONLY (IHO accuracy) */}
                {INTERNAL_ANALYSIS && s44Img && (
                  <div style={{padding:'10px 12px',borderBottom:'1px solid var(--border-dim)'}}>
                    <div style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.08em',marginBottom:'6px'}}>
                      IHO S-44 COMPLIANCE — PER DEPTH BAND
                    </div>
                    <img src={`data:image/png;base64,${s44Img}`} alt="S-44 bands"
                         style={{width:'100%',borderRadius:'6px',border:'1px solid var(--border-dim)',display:'block'}} />
                  </div>
                )}

                {/* Per-band accuracy table — INTERNAL ONLY */}
                {INTERNAL_ANALYSIS && Array.isArray(perBand) && perBand.length > 0 && (
                  <div style={{padding:'10px 12px',borderBottom:'1px solid var(--border-dim)'}}>
                    <div style={{fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700,letterSpacing:'0.08em',marginBottom:'6px'}}>
                      ACCURACY PER DEPTH BAND
                    </div>
                    <div style={{display:'grid',gridTemplateColumns:'80px 60px 60px 1fr',rowGap:'3px',columnGap:'8px',fontSize:'10px',fontFamily:'var(--font-mono)'}}>
                      <span style={{color:'var(--text-dim)',fontWeight:700}}>Band</span>
                      <span style={{color:'var(--text-dim)',fontWeight:700,textAlign:'right'}}>n</span>
                      <span style={{color:'var(--text-dim)',fontWeight:700,textAlign:'right'}}>RMSE</span>
                      <span style={{color:'var(--text-dim)',fontWeight:700}}>IHO 1A pass</span>
                      {perBand.map((b,i)=>{
                        const [lo,hi]=b.range||[0,0];
                        const c = b.s44_1a_pct>=90?'#059669':(b.s44_1a_pct>=70?'#0284c7':(b.s44_1a_pct>=40?'#d97706':'#b45309'));
                        const w = Math.min(100, Math.max(0, b.s44_1a_pct||0));
                        return (
                          <React.Fragment key={i}>
                            <span style={{color:'var(--text-secondary)'}}>{lo}–{hi} m</span>
                            <span style={{textAlign:'right'}}>{(b.n||0).toLocaleString()}</span>
                            <span style={{textAlign:'right',color:'var(--text-secondary)'}}>{b.rmse_m!=null?`${b.rmse_m} m`:'—'}</span>
                            <span style={{display:'flex',alignItems:'center',gap:'4px'}}>
                              <span style={{flex:1,height:'6px',borderRadius:'3px',background:'var(--bg-secondary)',overflow:'hidden'}}>
                                <span style={{display:'block',width:`${w}%`,height:'100%',background:c}}/>
                              </span>
                              <span style={{minWidth:'34px',color:c,fontWeight:700,textAlign:'right'}}>
                                {b.s44_1a_pct!=null?`${b.s44_1a_pct}%`:'—'}
                              </span>
                            </span>
                          </React.Fragment>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Source breakdown — INTERNAL ONLY (data-source disclosure) */}
                {INTERNAL_ANALYSIS && aug && (
                  <div style={{padding:'10px 14px',borderBottom:'1px solid var(--border-dim)',
                       fontSize:'9.5px',fontFamily:'var(--font-mono)',color:'var(--text-secondary)',lineHeight:1.6}}>
                    <span style={{fontWeight:700,letterSpacing:'0.06em',color:'var(--text-dim)'}}>REFERENCES USED:</span>
                    {' '}{aug.insitu>0 && <span><b>{aug.insitu.toLocaleString()}</b> in-situ</span>}
                    {(aug.sliderule>0) && <span> · <b>{aug.sliderule}</b> ICESat-2</span>}
                    {(aug.iboating>0) && <span> · <b>{aug.iboating}</b> chart</span>}
                    {(aug.gebco>0) && <span> · <b>{aug.gebco}</b> GEBCO</span>}
                    {(m.n_test>0) && <span style={{color:'var(--text-dim)'}}>  ·  validated on {m.n_test.toLocaleString()} held-out points</span>}
                  </div>
                )}

                {/* Download row */}
                <div style={{padding:'10px 12px',display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:'6px',background:'var(--bg-secondary)'}}>
                  {[
                    {fmt:'geotiff',label:'GeoTIFF'},
                    {fmt:'csv',label:'CSV'},
                    {fmt:'geojson',label:'GeoJSON'},
                  ].map(d=>(
                    <button key={d.fmt} onClick={()=>onExport&&onExport(d.fmt)}
                      title={`Download the current depth grid as ${d.label}`} style={{
                      padding:'8px',fontSize:'9px',fontFamily:'var(--font-mono)',fontWeight:700,
                      borderRadius:'6px',cursor:'pointer',
                      border:'1px solid var(--border-dim)',background:'#fff',color:'var(--text-secondary)',
                      letterSpacing:'0.06em',
                    }}>↓ {d.label}</button>
                  ))}
                </div>
              </div>
            )}

            {error && (
              <div style={{padding:'10px',borderRadius:'var(--radius-sm)',background:'rgba(225,29,72,0.06)',border:'1px solid rgba(225,29,72,0.15)',fontSize:'11px',fontFamily:'var(--font-mono)',color:'var(--accent-rose)',marginBottom:'12px'}}>
                {error}
              </div>
            )}

            {/* ─────── Validation against external survey — INTERNAL ONLY ───────
                 Accuracy/error metrics (RMSE/MAE/Bias/R²/S-44) + in-situ
                 comparison are expert affordances, hidden from the client portal. */}
            {INTERNAL_ANALYSIS && (results || observedPts > 0) && (
              <div style={{marginBottom:'16px',padding:'14px',borderRadius:'var(--radius)',border:'1px solid rgba(225,29,72,0.15)',background:'rgba(225,29,72,0.02)'}}>
                <h4 style={{fontSize:'11px',fontFamily:'var(--font-mono)',fontWeight:700,color:'var(--accent-rose)',marginBottom:'10px',display:'flex',alignItems:'center'}}>
                  VALIDATION — In-Situ Comparison
                  <HelpIcon text="Upload independent survey points (.shp/.csv) and compare them against the predicted grid: RMSE, MAE, bias, R² and IHO S-44 pass rate." />
                </h4>
                <input ref={shpRef} type="file" accept=".shp,.dbf,.shx,.prj,.cpg,.csv,.zip" multiple style={{display:'none'}}
                  onChange={e=>{ if(e.target.files?.length && onUploadShapefile) onUploadShapefile(e.target.files); }} />
                <button onClick={()=>shpRef.current?.click()} style={{
                  width:'100%',padding:'10px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:700,borderRadius:'var(--radius-sm)',cursor:'pointer',
                  border:'1px dashed rgba(225,29,72,0.3)',background:'rgba(225,29,72,0.03)',color:'var(--accent-rose)',marginBottom:'6px',
                }}>Upload Shapefile (.shp+…) or CSV</button>
                {validationPts && validationPts.length>0 && (
                  <>
                    <p style={{fontSize:'10px',fontFamily:'var(--font-mono)',color:'var(--accent-emerald)',fontWeight:600,marginBottom:'6px'}}>
                      {validationPts.length} observed points loaded
                    </p>
                    <button onClick={onValidate} disabled={loading} style={{
                      width:'100%',padding:'12px',fontSize:'11px',fontFamily:'var(--font-mono)',fontWeight:800,borderRadius:'var(--radius-sm)',cursor:'pointer',
                      border:'none',background:'var(--bg-secondary)',color:'#fff',opacity:loading?0.5:1,
                    }}>Run Validation (RMSE + S-44)</button>
                  </>
                )}
                {validationResults && (
                  <div style={{marginTop:'8px',padding:'10px',borderRadius:'6px',background:'var(--bg-secondary)',border:'1px solid var(--border-dim)',fontSize:'10px',fontFamily:'var(--font-mono)'}}>
                    <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'4px'}}>
                      <span style={{color:'var(--text-dim)'}}>RMSE:</span><span style={{fontWeight:700,color:'var(--accent-rose)'}}>{validationResults.rmse?.toFixed(2)}m</span>
                      <span style={{color:'var(--text-dim)'}}>MAE:</span><span style={{fontWeight:700}}>{validationResults.mae?.toFixed(2)}m</span>
                      <span style={{color:'var(--text-dim)'}}>Bias:</span><span style={{fontWeight:700}}>{validationResults.bias?.toFixed(2)}m</span>
                      <span style={{color:'var(--text-dim)'}}>R²:</span><span style={{fontWeight:700,color:'var(--accent-primary)'}}>{validationResults.r2?.toFixed(3)}</span>
                      <span style={{color:'var(--text-dim)'}}>S-44 Pass:</span><span style={{fontWeight:700,color:'var(--accent-emerald)'}}>{validationResults.s44_pass_pct?.toFixed(0)}%</span>
                      <span style={{color:'var(--text-dim)'}}>Pairs:</span><span style={{fontWeight:700}}>{validationResults.n_pairs}</span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ─────── i-Boating chart pipeline — OPT-IN ONLY ───────
                 Hidden by default per AD Ports compliance: results depend on
                 public-chart availability/quality, so the user must enable it
                 explicitly and is steered towards FUSION mode instead. */}
            <div style={{marginBottom:'16px'}}>
              <label style={{display:'flex',alignItems:'center',gap:'6px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:600,color:'var(--text-dim)',cursor:'pointer'}}>
                <input type="checkbox" checked={showIboating} onChange={e=>setShowIboating(e.target.checked)} />
                Show experimental i-Boating chart pipeline
                <HelpIcon text="OCR-extracts depth soundings from public i-Boating nautical charts and trains a model on them. Experimental — hidden by default." />
              </label>
              {showIboating && (
                <div style={{marginTop:'8px',padding:'14px',borderRadius:'var(--radius)',borderLeft:'3px solid #d97706',border:'1px solid rgba(217,119,6,0.25)',borderLeftWidth:'3px',background:'rgba(217,119,6,0.04)'}}>
                  <div style={{display:'inline-flex',alignItems:'center',gap:'6px',marginBottom:'8px',padding:'3px 8px',borderRadius:'4px',background:'rgba(217,119,6,0.14)',border:'1px solid rgba(217,119,6,0.3)'}}>
                    <span style={{fontSize:'11px'}}>⚠</span>
                    <span style={{fontSize:'9px',fontFamily:'var(--font-mono)',fontWeight:800,color:'#b45309',letterSpacing:'0.06em'}}>
                      EXPERIMENTAL · i-BOATING CHART PIPELINE
                    </span>
                  </div>
                  <p style={{fontSize:'10px',fontFamily:'var(--font-mono)',color:'#92400e',lineHeight:1.6,marginBottom:'10px'}}>
                    Depths are OCR-extracted from public i-Boating nautical charts.
                    Results depend entirely on public-chart availability and quality for
                    this area — coverage can be sparse, outdated or on a different
                    vertical datum. Drying heights (underlined / "0" labels) are filtered
                    out automatically.
                  </p>
                  <div style={{display:'flex',gap:'7px',marginBottom:'10px',padding:'8px 10px',borderRadius:'6px',background:'rgba(2,132,199,0.06)',border:'1px solid rgba(2,132,199,0.22)'}}>
                    <span style={{fontSize:'11px',lineHeight:1.4}}>💡</span>
                    <p style={{fontSize:'10px',fontFamily:'var(--font-mono)',color:'#0c4a6e',lineHeight:1.6,margin:0}}>
                      <b>Recommended for AD Ports operational areas:</b> use <b>FUSION mode</b> —
                      upload observed XYZ soundings and fuse them with GEBCO + ICESat-2 for a
                      documented, datum-aware reference set.
                    </p>
                  </div>
                  <button onClick={()=>onIBoatingPipeline&&onIBoatingPipeline()} disabled={loading||!roi} style={{
                    width:'100%',padding:'10px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:700,borderRadius:'var(--radius-sm)',
                    cursor:(loading||!roi)?'not-allowed':'pointer',
                    border:'1px solid rgba(217,119,6,0.35)',background:'#fff',color:'#b45309',
                    opacity:(loading||!roi)?0.5:1,letterSpacing:'0.06em',
                  }}>{roi?'Run i-Boating Pipeline (chart OCR)':'Draw an ROI first'}</button>
                </div>
              )}
            </div>

            {/* Export removed — the report card above now exposes a unified
                 Download row, and validation has its own panel. */}
          </div>
        )}

        {tab==='logs' && (
          <div>
            <h3 style={{fontSize:'13px',fontWeight:700,marginBottom:'12px'}}>Processing Logs</h3>
            <div style={{background:'var(--bg-secondary)',borderRadius:'var(--radius)',border:'1px solid var(--border-dim)',padding:'8px',maxHeight:'500px',overflow:'auto'}}>
              {(!logs || logs.length===0) ? (
                <p style={{fontSize:'11px',color:'var(--text-dim)',textAlign:'center',padding:'20px'}}>No logs yet</p>
              ) : (
                logs.map((log,i)=>(
                  <div key={i} style={{display:'flex',gap:'8px',padding:'4px',borderBottom:'1px solid var(--border-dim)',fontSize:'10px',fontFamily:'var(--font-mono)'}}>
                    <span style={{color:'var(--text-dim)',minWidth:'65px'}}>{log.time}</span>
                    <span style={{color:log.type==='error'?'var(--accent-rose)':log.type==='success'?'var(--accent-emerald)':'var(--text-secondary)'}}>{log.msg}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}

