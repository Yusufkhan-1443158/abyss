import React, { useState, useCallback, useEffect } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import MapPanel from './components/MapPanel';
import ThreeDView from './components/ThreeDView';
import ResultsPanel from './components/ResultsPanel';
import LoadingOverlay from './components/LoadingOverlay';
import MonitoringPanel from './components/MonitoringPanel';
import StatusBar from './components/StatusBar';
import HelpModal from './components/HelpModal';
import VhrJobsModal from './components/VhrJobsModal';
import BathyResultsPanel from './components/BathyResultsPanel';
import VhrPurchasePanel from './components/VhrPurchasePanel';
import AsciiView from './components/AsciiView';
import UaeResultsPanel from './components/UaeResultsPanel';
import { INTERNAL_ANALYSIS, HYDRO_CLIENT } from './components/internalMode';

const API = process.env.REACT_APP_API_URL || '';

// CLIENT PORTAL log/loading sanitizer. The Logs tab + LoadingOverlay surface
// pipeline messages that name the method/engine and the data sources. On the
// client portal (INTERNAL_ANALYSIS=false) we redact those tokens to neutral
// wording at the single render chokepoint, so no source/method is disclosed in
// the visible activity log — without rewriting every addLog call site (and
// nothing is removed from the data layer; internal mode shows the full text).
// Engine/data-source names — redacted for both the plain client portal AND the
// hydro-client tier (IP concern, unrelated to IHO compliance).
const _CLIENT_REDACTIONS_ENGINE = [
  [/Sentinel-2 L2A|Sentinel-2|Sentinel|Copernicus(?: CDSE)?|CDSE/gi, 'imagery'],
  [/Mapbox VHR \+ S2|Mapbox VHR|Mapbox/gi, 'high-resolution imagery'],
  [/ICESat-2 ATL03|ICESat-2|ICESat2|ATL03|ATL24|SlideRule/gi, 'survey reference'],
  [/\bGEBCO\b/gi, 'reference depths'],
  [/Lyzenga\/Stumpf|Lyzenga|Stumpf|band-ratio/gi, 'depth model'],
  [/CShelph/gi, 'lidar'],
  [/multibeam/gi, 'survey'],
  [/satellite-derived bathymetry|satellite-derived/gi, 'depth'],
  [/\bin-situ\b/gi, 'reference'],
  [/\bCNN\b|PatchCNN|U-Net|BP-NN|MLP/gi, 'model'],
];
// IHO S-44/CATZOC terminology — redacted only on the plain consumer portal.
// The hydro-client tier (REACT_APP_CLIENT_TIER=hydro, see IHO-R1) needs the
// real terms since AD Ports is a hydrographic authority that explicitly
// requested S-44/CATZOC language.
const _CLIENT_REDACTIONS_IHO = [
  [/\bCATZOC\b/gi, 'confidence'],
  [/\bIHO S-?44\b|\bS-?44\b|\bIHO\b/gi, 'quality'],
];
function sanitizeClientText(s) {
  if (INTERNAL_ANALYSIS || typeof s !== 'string') return s;
  let out = s;
  for (const [re, rep] of _CLIENT_REDACTIONS_ENGINE) out = out.replace(re, rep);
  if (!HYDRO_CLIENT) {
    for (const [re, rep] of _CLIENT_REDACTIONS_IHO) out = out.replace(re, rep);
  }
  return out;
}

// Compute an ROI bbox from uploaded training/observed points (5 % margin,
// ≥0.002° so a tight cluster still yields a visible rectangle). Uploading a
// data file auto-sets the ROI from the data extent — the MapPanel 'setRoi'
// listener then draws the rectangle and flies/zooms to it.
function roiFromPoints(pts) {
  if (!pts || pts.length === 0) return null;
  let n = -90, s = 90, e = -180, w = 180;
  for (const p of pts) {
    if (!Number.isFinite(p?.lat) || !Number.isFinite(p?.lon)) continue;
    if (p.lat > n) n = p.lat; if (p.lat < s) s = p.lat;
    if (p.lon > e) e = p.lon; if (p.lon < w) w = p.lon;
  }
  if (n < s || e < w) return null;
  const mLat = Math.max((n - s) * 0.05, 0.002);
  const mLon = Math.max((e - w) * 0.05, 0.002);
  return { north: n + mLat, south: s - mLat, east: e + mLon, west: w - mLon };
}

function App() {
  const [roi, setRoi] = useState(null);
  const [results, setResults] = useState(null);
  const [resultsB, setResultsB] = useState(null);
  const [diffResults, setDiffResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [loadingStep, setLoadingStep] = useState(0);
  const [loadingTotal, setLoadingTotal] = useState(0);
  const [error, setError] = useState(null);
  const [view, setView] = useState('map');
  const [showResults, setShowResults] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [activeModule, setActiveModule] = useState('gebco_s2');
  const [logs, setLogs] = useState([]);
  const [userPoints, setUserPoints] = useState([]);
  const [csvPreview, setCsvPreview] = useState(null);
  const [validationPts, setValidationPts] = useState([]);
  // Loaded observed/calibration dataset metadata (datum, CRS, count, bbox)
  const [observedData, setObservedData] = useState(null);
  const [validationResults, setValidationResults] = useState(null);
  const [autoEvalResults, setAutoEvalResults] = useState(null);
  const [digitiseResults, setDigitiseResults] = useState(null);
  const [icesat2Results, setIcesat2Results] = useState(null);
  const [icesat2Granules, setIcesat2Granules] = useState([]);
  const [mleResults, setMleResults] = useState(null);
  // ADPorts F1/F2: per-scene / Planet-backup map overlay override — when
  // non-null, MapPanel renders THIS raster instead of `results`' own
  // overlay, without re-fitting the camera (all scenes/dates share the
  // same geographic extent as the composite/site bbox). null = restore
  // the default (composite) overlay.
  const [mapOverlay, setMapOverlay] = useState(null); // { png_b64, bounds, label, key } | null
  const onShowOverlay = (payload) => setMapOverlay(payload);
  const [showHelp, setShowHelp] = useState(false);
  // Very-HR local-processing jobs — server is source of truth, refreshed every 5 s.
  const [vhrJobs, setVhrJobs] = useState([]);
  const [showVhrJobsModal, setShowVhrJobsModal] = useState(false);
  // Persistent Results catalogue (R2/R3/R4) — durable bathy-job outputs.
  const [showBathyResults, setShowBathyResults] = useState(false);
  const [showVhrPurchase, setShowVhrPurchase] = useState(false);
  const [params, setParams] = useState({
    mode: 'gebco_s2', chart_zoom: 14, max_cloud: 20,
    ref_source: 'gebco', depth_method: 'cnn',
    s2_date: '2024-10-15',
    start_date: '2024-10-05', end_date: '2024-10-25',
    start_date_b: '2023-05-01', end_date_b: '2023-09-30',
    mle_scenes_per_epoch: 6,
    mle_use_icesat2: true,
    mle_ice_max_granules: 4,
    ice_start: '2022-01-01', ice_end: '2024-12-31',
    ice_laser: 0, ice_threshold: 30, ice_water_temp: 22,
    resolution_m: 10,  // output raster resolution: 10/20/50/100 m
    resolution: '10m', // run mode: '20m' | '10m' | 'vhr' (clear-UI selector)
    tiled: true,       // enable 4x4 parallel tiling for large ROIs
    // Bias-correction + calibration (applies to CNN_v2 + Accurate pipelines)
    bc_mode: 'single',            // 'single' composite or 'multi' per-scene
    bc_window_days: 180,          // Sentinel-2 time window
    bc_n_scenes: 4,               // number of scenes when multi-image
    bc_tide: true,                // at-source ICESat-2 tide retide
    bc_wave: true,                // wave Hs/2 correction at t2
    bc_geoid: true,               // EGM2008 / Gulf fit ellipsoidal→MSL
    bc_calib: 'shift+bias+local', // post-prediction calibration steps
    // Inference mode — use a saved BP-NN checkpoint, no retraining
    bc_inference_only: false,     // toggle "Inference only"
    bc_inference_model: '',       // relative path of chosen weights file
    bc_deep: true,                // default to deep BP-NN for robustness
  });
  const [availableModels, setAvailableModels] = useState([]);

  const addLog = useCallback((msg, type='info') => {
    setLogs(p => [...p.slice(-80), { time: new Date().toISOString().split('T')[1].split('.')[0], msg: sanitizeClientText(msg), type }]);
  }, []);
  const setMode = useCallback((m) => { setActiveModule(m); setParams(p => ({...p, mode: m})); }, []);

  // Auto-open the VHR purchase panel when returning from Stripe/simulated checkout.
  // Backend /success redirects to ?vhr_paid=1&vhr_order=<id> (or ?vhr_cancelled=1);
  // the panel itself reads those params and fetches the finished order.
  useEffect(() => {
    try {
      const p = new URLSearchParams(window.location.search);
      if (p.get('vhr_paid') === '1' || p.get('vhr_cancelled') === '1'
          || p.get('vhr_purchase') === 'success' || p.get('vhr_purchase') === 'cancel') {
        setShowVhrPurchase(true);
      }
    } catch (_) {}
  }, []);

  // Listen for quick-set ROI events from sidebar
  useEffect(() => {
    const handler = (e) => { setRoi(e.detail); addLog(`ROI set to preset: ${JSON.stringify(e.detail)}`, 'success'); };
    window.addEventListener('setRoi', handler);
    return () => window.removeEventListener('setRoi', handler);
  }, [addLog]);

  const searchTracks = useCallback(async () => {
    if (!roi) return;
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API}/api/search-tracks`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({bbox:roi}) });
      const d = await r.json(); setSearchResults(d); addLog('Sources found','success');
    } catch(e) { setError(e.message); }
    finally { setLoading(false); }
  }, [roi, addLog]);

  // Generic extract function
  const doExtract = useCallback(async (sd, ed, label) => {
    if (!roi) return null;
    setLoading(true); setError(null);
    const steps = ['Gathering reference depths...','Downloading Sentinel-2...','Training CNN model...','Predicting bathymetry...','Building results...'];
    setLoadingTotal(steps.length); setLoadingStep(0); setLoadingMsg(`${label}: ${steps[0]}`);
    addLog(`Extracting ${label} (${sd} → ${ed}, ${activeModule})...`);
    let si=0; const iv=setInterval(()=>{si++;if(si<steps.length){setLoadingStep(si);setLoadingMsg(`${label}: ${steps[si]}`);}},8000);
    try {
      const r = await fetch(`${API}/api/extract`, {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({bbox:roi,params:{...params,mode:activeModule},start_date:sd,end_date:ed,user_points:userPoints})});
      const d = await r.json();
      clearInterval(iv);
      if (d.error && !d.interpolated_points?.length) throw new Error(d.error);
      const res = {...d, points:[...(d.points||[]),...(d.interpolated_points||[])]};
      addLog(`${label} done! ${d.interpolated_points?.length||0} pts. Sources: ${(d.sources_used||[]).join(', ')}`,'success');
      return res;
    } catch(e) { clearInterval(iv); setError(e.message); addLog(`${label}: ${e.message}`,'error'); return null; }
    finally { setLoading(false); }
  }, [roi, params, activeModule, addLog, userPoints]);

  const extract = useCallback(async () => {
    const res = await doExtract(params.start_date, params.end_date, 'Epoch A');
    if (res) { setResults(res); setShowResults(true); setCsvPreview(null); setResultsB(null); setDiffResults(null); }
  }, [doExtract, params.start_date, params.end_date]);

  const extractEpochB = useCallback(async () => {
    const res = await doExtract(params.start_date_b, params.end_date_b, 'Epoch B');
    if (res) { setResultsB(res); addLog(`Epoch B ready for comparison`,'success'); }
  }, [doExtract, params.start_date_b, params.end_date_b, addLog]);

  // Quick Analyse — GEBCO + Spectral (< 1 min)
  const quickAnalyse = useCallback(async () => {
    if (!roi) return;
    setLoading(true); setError(null);
    const steps = ['Fetching GEBCO depths...','Downloading Sentinel-2 @20m...','Calibrating spectral indices...','Bayesian fusion...'];
    setLoadingTotal(steps.length); setLoadingStep(0); setLoadingMsg(`Quick: ${steps[0]}`);
    addLog(`Quick Analyse: GEBCO + Spectral indices (fast mode)...`);
    let si=0; const iv=setInterval(()=>{si++;if(si<steps.length){setLoadingStep(si);setLoadingMsg(`Quick: ${steps[si]}`);}},5000);
    try {
      const r = await fetch(`${API}/api/quick-analyse`, {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({bbox:roi,start_date:params.start_date,end_date:params.end_date,
          resolution_m: params.resolution_m || 10,
          tiled: params.tiled !== false,
          user_points: (userPoints || []).filter(p =>
            p?.photon_class === 'observed' || p?.photon_class === 'icesat2_cshelph' || p?.photon_class === 'bathymetry')})});
      const d = await r.json(); clearInterval(iv);
      if (d.error && !d.interpolated_points?.length) throw new Error(d.error);
      const res = {...d, points:[...(d.points||[]),...(d.interpolated_points||[])]};
      addLog(`Quick done! ${d.interpolated_points?.length||0} pts in ${d.processing_time_sec||'?'}s. Sources: ${(d.sources_used||[]).join(', ')}`,'success');
      setResults(res); setShowResults(true); setCsvPreview(null); setResultsB(null); setDiffResults(null);
    } catch(e) { clearInterval(iv); setError(e.message); addLog(`Quick Analyse: ${e.message}`,'error'); }
    finally { setLoading(false); }
  }, [roi, params, userPoints, addLog]);

  // Smart CNN — flexible multi-layer CNN (SlideRule + user XYZ + tide/wave)
  // ─────────── DL Pro (default model, MC-Dropout MLP + IHO calibration) ───────────
  const runDlPro = useCallback(async ({ refPts = null, fineTune = false } = {}) => {
    if (!roi) return;
    setLoading(true); setError(null);
    setLoadingMsg('DL Pro: fetching Sentinel-2 + computing features...');
    setLoadingStep(0); setLoadingTotal(refPts ? 4 : 3);
    addLog(`DL Pro: ${refPts ? `${refPts.length} refs supplied${fineTune ? ' (fine-tune ON)' : ''}` : 'no refs (baseline calibration)'}…`);
    try {
      const r = await fetch(`${API}/api/predict_dl_pro`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bbox: roi,
          start_date: params.start_date,
          end_date: params.end_date,
          cloud: params.cloud || 30,
          resolution_m: params.resolution_m || 20,
          ref_pts: refPts || [],
          fine_tune: !!fineTune,
        }),
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      const ihoMsg = d.iho_s44 ?
        ` | IHO 1a=${(d.iho_s44.Order_1a ?? d.iho_s44.Order_1a?.pct_within_TVU ?? 0).toFixed?.(1) ?? d.iho_s44.Order_1a}% O2=${(d.iho_s44.Order_2 ?? d.iho_s44.Order_2?.pct_within_TVU ?? 0).toFixed?.(1) ?? d.iho_s44.Order_2}%` : '';
      addLog(
        `DL Pro done: depth median=${d.stats?.depth_median?.toFixed?.(2)}m | σ_med=${d.stats?.sigma_median?.toFixed?.(2)}m | cal α=${d.calibration?.alpha?.toFixed?.(3)} β=${d.calibration?.beta?.toFixed?.(3)} (${d.calibration?.source}, n=${d.calibration?.n_refs})` + ihoMsg,
        'success'
      );
      setResults({
        method: 'DL Pro',
        bbox: d.bbox,
        date_window: d.date_window,
        calibration: d.calibration,
        fine_tune: d.fine_tune,
        iho_s44: d.iho_s44,
        stats: d.stats,
        model_meta: d.model_meta,
        geotiff_b64: d.geotiff_b64?.depth,
        lower95_b64: d.geotiff_b64?.lower95,
        upper95_b64: d.geotiff_b64?.upper95,
        sigma_b64: d.geotiff_b64?.sigma,
      });
      setShowResults(true); setCsvPreview(null); setResultsB(null); setDiffResults(null);
    } catch (e) {
      setError(e.message); addLog(`DL Pro: ${e.message}`, 'error');
    } finally { setLoading(false); }
  }, [roi, params.start_date, params.end_date, params.cloud, params.resolution_m]);

  const runSmartCnn = useCallback(async ({ retrain = false, userWeight = 20.0 } = {}) => {
    if (!roi) return;
    setLoading(true); setError(null);
    const steps = retrain
      ? ['Fetching Sentinel-2 + deep-water glint correction...',
         'Querying SlideRule ICESat-2 (high-confidence)...',
         'Tide-harmonising all labels to S2 epoch...',
         'Building 14-layer feature stack (bands + canal prob + distances + tide/wave)...',
         'Training Attention U-Net (CBAM + ASPP, weighted loss)...',
         'MC+TTA inference + TPS residual correction...']
      : ['Fetching Sentinel-2...',
         'Building feature stack...',
         'Loading cached model + MC+TTA inference...'];
    setLoadingTotal(steps.length); setLoadingStep(0); setLoadingMsg(`SmartCNN: ${steps[0]}`);
    addLog(`SmartCNN ${retrain ? '(retrain)' : '(cached)'}: ${retrain ? 'train+predict' : 'predict only'}...`);
    let si = 0; const iv = setInterval(() => {
      si++; if (si < steps.length) { setLoadingStep(si); setLoadingMsg(`SmartCNN: ${steps[si]}`); }
    }, retrain ? 20000 : 8000);
    try {
      const r = await fetch(`${API}/api/smart-cnn`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bbox: roi,
          s2_start: params.start_date,
          s2_end: params.end_date,
          retrain,
          user_weight: userWeight,
          mc_passes: 6,
        }),
      });
      const d = await r.json(); clearInterval(iv);
      if (d.error) throw new Error(d.error);
      addLog(
        `SmartCNN done: ${d.used_cache ? 'CACHED' : 'TRAINED'} | layers=${d.n_layers} | refs u/sr/xyz/gebco=${d.n_user_refs}/${d.n_sliderule}/${d.n_xyz_store}/${d.n_gebco}` +
        (d.validation ? ` | R²=${d.validation.r2} RMSE=${d.validation.rmse_m}m MAE=${d.validation.mae_m}m` : ''),
        'success'
      );
      setResults({
        ...d,
        points: [],
        stats: {
          grid_points: 0,
          mean_depth: d.depth_stats?.mean_m || 0,
          max_depth: d.depth_stats?.max_m || 0,
          min_depth: d.depth_stats?.min_m || 0,
        },
        sources_used: [
          d.n_user_refs ? `user×${d.user_weight}` : null,
          d.n_sliderule ? `sliderule=${d.n_sliderule}` : null,
          d.n_xyz_store ? `xyz_store=${d.n_xyz_store}` : null,
          d.n_gebco ? `gebco=${d.n_gebco}` : null,
        ].filter(Boolean),
        raster_png: d.raster_png_base64,
        raster_bounds: d.raster_bounds,
        raster_max_depth: d.raster_max_depth,
      });
      setShowResults(true); setCsvPreview(null); setResultsB(null); setDiffResults(null);
    } catch (e) {
      clearInterval(iv); setError(e.message); addLog(`SmartCNN: ${e.message}`, 'error');
    } finally { setLoading(false); }
  }, [roi, params, addLog]);

  // BOA-CNN-BiLSTM — Zhu et al. 2025 JSTARS
  const runBoaCnnBilstm = useCallback(async ({ useBoa = true } = {}) => {
    if (!roi) return;
    setLoading(true); setError(null);
    const steps = useBoa
      ? ['Fetching Sentinel-2 L2A surface reflectance...',
         'Fetching SlideRule ATL03 + refraction + tide harmonisation...',
         'Selecting 6 top-Pearson spectral features...',
         'Bayesian hyperparameter optimisation (GP + Matern 5/2)...',
         'Final CNN-BiLSTM fit (Adam, MSE, LR step decay)...',
         'Per-pixel inference over the S2 grid...']
      : ['Fetching Sentinel-2...', 'Fetching SlideRule + harmonising...',
         'Selecting features + training CNN-BiLSTM...',
         'Per-pixel inference...'];
    setLoadingTotal(steps.length); setLoadingStep(0);
    setLoadingMsg(`BOA-CNN-BiLSTM: ${steps[0]}`);
    addLog(`BOA-CNN-BiLSTM (Zhu et al. 2025): ${useBoa ? 'with' : 'without'} Bayesian optimisation...`);
    let si = 0; const iv = setInterval(() => {
      si++; if (si < steps.length) { setLoadingStep(si); setLoadingMsg(`BOA-CNN-BiLSTM: ${steps[si]}`); }
    }, useBoa ? 25000 : 10000);
    try {
      const r = await fetch(`${API}/api/boa-cnn-bilstm`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bbox: roi,
          s2_start: params.start_date,
          s2_end:   params.end_date,
          use_boa:  useBoa,
          boa_calls: useBoa ? 20 : 0,
          boa_cv_folds: 3,
        }),
      });
      const d = await r.json(); clearInterval(iv);
      if (d.error) throw new Error(d.error);
      const v = d.validation || {};
      addLog(
        `BOA-CNN-BiLSTM done: n_sr=${d.n_sliderule} features=[${(d.feature_names||[]).join(',')}] ` +
        `r=${v.r} R²=${v.r2} RMSE=${v.rmse_m}m MAE=${v.mae_m}m MRE=${v.mre_pct}% ` +
        `hp=ks${d.hyperparameters?.kernel_size}/nf${d.hyperparameters?.num_filters}/u${d.hyperparameters?.bilstm_units}`,
        'success'
      );
      setResults({
        ...d,
        points: [],
        stats: {
          grid_points: 0,
          mean_depth: d.depth_stats?.mean_m || 0,
          max_depth: d.depth_stats?.max_m || 0,
          min_depth: d.depth_stats?.min_m || 0,
        },
        sources_used: [`sliderule=${d.n_sliderule}`, d.boa?.method ? `BOA=${d.boa.method}` : null].filter(Boolean),
        raster_png: d.raster_png_base64,
        raster_bounds: d.raster_bounds,
        raster_max_depth: d.raster_max_depth,
      });
      setShowResults(true); setCsvPreview(null); setResultsB(null); setDiffResults(null);
    } catch (e) {
      clearInterval(iv); setError(e.message); addLog(`BOA-CNN-BiLSTM: ${e.message}`, 'error');
    } finally { setLoading(false); }
  }, [roi, params, addLog]);

  // Multi-Epoch Maximum Likelihood — many S2 scenes × 2 epochs
  const multiEpochMle = useCallback(async () => {
    if (!roi) return;
    if (!params.start_date || !params.end_date || !params.start_date_b || !params.end_date_b) {
      setError('Set Epoch A and Epoch B date ranges first'); return;
    }
    setLoading(true); setError(null); setMleResults(null);
    const nsc = params.mle_scenes_per_epoch || 6;
    const steps = [
      'Loading GEBCO + in-situ XYZ refs...',
      `Fetching ${nsc} S2 scenes for Epoch A @10m...`,
      'Per-scene ridge regression (weighted by σ)...',
      `Fetching ${nsc} S2 scenes for Epoch B @10m...`,
      'Per-pixel Gaussian MLE fusion...',
      'HR land masking + B−A diff + ASCII preview...',
    ];
    setLoadingTotal(steps.length); setLoadingStep(0); setLoadingMsg(`MLE: ${steps[0]}`);
    addLog(`Multi-Epoch MLE: ${nsc} scenes × 2 epochs, HR land mask`);
    let si = 0;
    const iv = setInterval(() => { si++; if (si < steps.length) { setLoadingStep(si); setLoadingMsg(`MLE: ${steps[si]}`); } }, 8000);
    try {
      // ICESat-2 wide window = union of A and B
      const iceStart = [params.start_date, params.start_date_b].filter(Boolean).sort()[0];
      const iceEnd   = [params.end_date,   params.end_date_b  ].filter(Boolean).sort().reverse()[0];
      const r = await fetch(`${API}/api/multi-epoch-mle`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          bbox: roi,
          epoch_a: { start: params.start_date, end: params.end_date },
          epoch_b: { start: params.start_date_b, end: params.end_date_b },
          n_scenes_per_epoch: nsc,
          use_icesat2: !!params.mle_use_icesat2,
          ice_start: iceStart,
          ice_end: iceEnd,
          ice_max_granules: params.mle_ice_max_granules || 4,
        }),
      });
      const d = await r.json(); clearInterval(iv);
      if (d.error) throw new Error(d.error);
      setMleResults(d);
      // Also hydrate the regular results slots so MapPanel overlays work
      const pointsA = [...(d.epoch_a?.points || [])];
      const pointsB = [...(d.epoch_b?.points || [])];
      setResults({
        ...d.epoch_a,
        points: pointsA,
        stats: { grid_points: pointsA.length, mean_depth: 0, max_depth: 0, min_depth: 0 },
        sources_used: d.sources_used || [],
        raster_png: d.epoch_a?.raster?.png,
        raster_bounds: d.epoch_a?.raster?.bounds,
        raster_max_depth: d.epoch_a?.raster?.max_depth,
        geotiff_b64: d.epoch_a?.geotiff_b64,
      });
      setResultsB({
        ...d.epoch_b,
        points: pointsB,
        stats: { grid_points: pointsB.length },
        raster_png: d.epoch_b?.raster?.png,
        raster_bounds: d.epoch_b?.raster?.bounds,
        raster_max_depth: d.epoch_b?.raster?.max_depth,
        geotiff_b64: d.epoch_b?.geotiff_b64,
      });
      setShowResults(true);
      const s = d.diff?.stats || {};
      const ice = d.icesat2 || {};
      if (ice.used) {
        addLog(`ICESat-2 (fixed ${ice.window?.start}→${ice.window?.end}): ${ice.n_photons} photons from ${ice.n_granules} granules${ice.error?' — '+ice.error:''}`, ice.error?'warn':'info');
      }
      addLog(`MLE done in ${d.processing_time_sec}s. A=${d.epoch_a?.n_scenes_used}sc R²=${d.epoch_a?.r2} B=${d.epoch_b?.n_scenes_used}sc R²=${d.epoch_b?.r2}. Δmean=${s.mean_change_m}m RMSD=${s.rmsd_m}m`,'success');
    } catch (e) {
      clearInterval(iv); setError(e.message); addLog(`Multi-Epoch MLE: ${e.message}`,'error');
    } finally { setLoading(false); }
  }, [roi, params.start_date, params.end_date, params.start_date_b, params.end_date_b, params.mle_scenes_per_epoch, addLog]);

  // GPU Mosaic — large region tiled @10m
  const extractMosaic = useCallback(async () => {
    if (!roi) return;
    setLoading(true); setError(null);
    const steps = ['Searching best S2 scenes (last 4 weeks)...','Splitting region into sub-tiles...','Fetching S2 @10m per tile...','Running GPU CNN per tile...','Feather-blending mosaic...','Building output grid...'];
    setLoadingTotal(steps.length); setLoadingStep(0); setLoadingMsg(steps[0]);
    addLog(`GPU Mosaic: splitting large region into tiles @10m...`);
    let si=0; const iv=setInterval(()=>{si++;if(si<steps.length){setLoadingStep(si);setLoadingMsg(steps[si]);}},15000);
    try {
      const r = await fetch(`${API}/api/extract-mosaic`, {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({bbox:roi,params:{...params,n_weeks:4,max_cloud:params.max_cloud||20,tile_deg:0.08,overlap_deg:0.008}})});
      const d = await r.json(); clearInterval(iv);
      if (d.error && !d.interpolated_points?.length) throw new Error(d.error);
      const res = {...d, points:[...(d.points||[]),...(d.interpolated_points||[])]};
      addLog(`GPU Mosaic done! ${d.interpolated_points?.length||0} pts @10m. ${d.stats?.tiles_ok||0}/${d.stats?.tiles_total||0} tiles. Device: ${d.stats?.device||'?'}. Sources: ${(d.sources_used||[]).join(', ')}`,'success');
      setResults(res); setShowResults(true); setCsvPreview(null); setResultsB(null); setDiffResults(null);
    } catch(e) { clearInterval(iv); setError(e.message); addLog(`GPU Mosaic: ${e.message}`,'error'); }
    finally { setLoading(false); }
  }, [roi, params, addLog]);

  // Accurate Estimation — HPC pipeline with live SSE progress
  // Fuses ICESat-2 ATL03 + GEBCO + iBoating (backend-only) → BathyNetPro
  // trained, then runs 32-way parallel tiled CNN inference and streams
  // progress to the UI via Server-Sent Events.
  const runAccurateBathymetry = useCallback(async () => {
    if (!roi) return;
    setLoading(true); setError(null);
    setLoadingTotal(100); setLoadingStep(0);
    setLoadingMsg('Starting accurate pipeline...');
    addLog('Accurate: submitting job (ICESat-2 45s-cap + GEBCO + observed XYZ → Patent-MLP)');
    try {
      const startR = await fetch(`${API}/api/accurate-bathymetry/start`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          bbox: roi,
          region: params.site_key || 'custom',
          epochs: params.epochs || 40,
          workers: 32,
          // Bias-correction & calibration options from the Sidebar panel
          bc_mode:        params.bc_mode || 'single',
          bc_window_days: params.bc_window_days || 180,
          bc_n_scenes:    params.bc_n_scenes || 4,
          bc_tide:        params.bc_tide  !== false,
          bc_wave:        params.bc_wave  !== false,
          bc_geoid:       params.bc_geoid !== false,
          bc_calib:       params.bc_calib || 'shift+bias+local',
          user_points: (userPoints || []).filter(p =>
            p?.photon_class === 'observed' || p?.photon_class === 'icesat2_cshelph' || p?.photon_class === 'bathymetry'),
        }),
      });
      const s = await startR.json();
      if (s.error) { setError(s.error); addLog(`Accurate: ${s.error}`, 'error'); setLoading(false); return; }
      addLog(`Job ${s.job_id} submitted — opening SSE stream`);
      const es = new EventSource(`${API}${s.stream_url}`);
      const handle = (ev) => {
        try {
          const d = JSON.parse(ev.data);
          if (typeof d.progress === 'number') {
            setLoadingStep(Math.max(0, Math.min(100, Math.round(d.progress))));
          }
          if (d.message) setLoadingMsg(`[${d.phase}] ${d.message}`);
          if (d.phase === 'done' && d.result) {
            const r = d.result;
            const dl = r.downloads || {};
            const m = r.metrics || {};
            const src = r.sources || {};
            addLog(
              `Accurate done · ${src.n_total_refs||0} refs (iboating ${src.n_iboating||0} [backend-only] + ICESat-2 ${src.n_icesat2||0} + GEBCO ${src.n_gebco||0}) · ` +
              `R²=${m.r2?.toFixed(3)} · RMSE=${m.rmse_m?.toFixed(2)}m · ${r.n_tiles} tiles on ${r.n_workers} CPUs`,
              'success');
            if (dl.geotiff) addLog(`GeoTIFF: ${API}${dl.geotiff}`, 'success');
            if (dl.preview) addLog(`Preview PNG: ${API}${dl.preview}`, 'success');
            // Surface download links through a result object the UI already knows
            setResults({
              architecture: r.architecture,
              citation: r.citation,
              metrics: m, sources: src,
              accurate_downloads: {
                geotiff_url: `${API}${dl.geotiff}`,
                uncertainty_url: dl.uncertainty ? `${API}${dl.uncertainty}` : null,
                preview_url: dl.preview ? `${API}${dl.preview}` : null,
              },
              job_id: r.job_id,
              n_tiles: r.n_tiles, n_workers: r.n_workers,
            });
            setShowResults(true);
            if (m.r2 != null) setAutoEvalResults({
              rmse: m.rmse_m, mae: m.mae_m, bias: 0, r2: m.r2,
              n_pairs: src.n_total_refs,
              source: `Accurate · ${r.architecture}`,
            });
            es.close(); setLoading(false);
          }
          if (d.phase === 'error') {
            addLog(`Accurate: ${d.message}`, 'error');
            setError(d.message); es.close(); setLoading(false);
          }
        } catch (e) {
          addLog(`SSE parse: ${e.message}`, 'error');
        }
      };
      // Listen to every phase as an event name (SSE convention)
      ['start','ingest','icesat2','gebco','sentinel2','watermask','train',
       'prep_tiles','infer','stitch','export','done','error','info'].forEach(p => {
        es.addEventListener(p, handle);
      });
      es.onerror = (e) => {
        // EventSource auto-retries; close after DONE to avoid leaks.
        if (es.readyState === 2) es.close();
      };
    } catch (e) {
      setError(e.message); addLog(`Accurate: ${e.message}`, 'error'); setLoading(false);
    }
  }, [roi, params, userPoints, addLog]);

  // ── One-click Khalifa Port Test: load observed XYZ → CNN_v2 → show result ──
  // This mirrors the server-side self-test we ran (test_cnn_v2_khalifa.py).
  const runKhalifaTest = useCallback(async () => {
    setLoading(true); setError(null);
    addLog('Khalifa Test: loading KP Basin + EMAL observed XYZ …');
    try {
      const r = await fetch(`${API}/api/load-observed`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({site: 'khalifa_port'}),
      });
      const d = await r.json();
      if (d.error) { throw new Error(d.error); }
      const bbox = d.bbox;
      window.dispatchEvent(new CustomEvent('setRoi', {detail: bbox}));
      setUserPoints(d.points || []);
      setCsvPreview(d.points || []);
      setValidationPts(d.points || []);
      setObservedData(d);
      addLog(`Khalifa Test: ${d.count} observed pts loaded · firing CNN_v2 (Guo 2022) …`, 'success');

      // Fire CNN_v2 directly with Khalifa data (can't rely on stale params state)
      setLoadingTotal(100); setLoadingStep(0);
      setLoadingMsg('Khalifa Test · CNN_v2 (Guo 2022 BP-NN) starting …');
      const startR = await fetch(`${API}/api/cnn-v2/start`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          bbox,
          region: 'khalifa_port',
          epochs: 150,                      // down from 500 — converges fine with cosine LR
          validate_only_observed: true,     // honest eval: XYZ held out of training
          user_points: (d.points || []).filter(p =>
            p?.photon_class === 'observed' || p?.photon_class === 'icesat2_cshelph' || p?.photon_class === 'bathymetry'),
        }),
      });
      const s = await startR.json();
      if (s.error) throw new Error(s.error);
      addLog(`Khalifa Test: CNN_v2 job ${s.job_id} — streaming progress`);
      const es = new EventSource(`${API}${s.stream_url}`);
      const handle = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (typeof msg.progress === 'number') setLoadingStep(Math.max(0, Math.min(100, Math.round(msg.progress))));
          if (msg.message) setLoadingMsg(`[${msg.phase}] ${msg.message}`);
          if (msg.phase === 'done' && msg.result) {
            const rr = msg.result; const dl = rr.downloads || {}; const m = rr.metrics || {};
            addLog(`Khalifa Test DONE · R²=${m.r2?.toFixed(3)} · RMSE=${m.rmse_m?.toFixed(2)}m · MAE=${m.mae_m?.toFixed(2)}m · bias=${m.bias_m>=0?'+':''}${m.bias_m?.toFixed(2)}m`, 'success');
            setResults({
              architecture: rr.architecture, citation: rr.citation, method: rr.method,
              metrics: m, sources: rr.sources, bp_info: rr.bp_info, bbox,
              accurate_downloads: {
                geotiff_url: dl.geotiff ? `${API}${dl.geotiff}` : null,
                uncertainty_url: dl.uncertainty ? `${API}${dl.uncertainty}` : null,
                preview_url: dl.preview ? `${API}${dl.preview}` : null,
              },
              job_id: rr.job_id,
            });
            setShowResults(true); setLoading(false); es.close();
          } else if (msg.phase === 'error') {
            setError(msg.message || 'error'); addLog(`Khalifa Test: ${msg.message}`, 'error');
            setLoading(false); es.close();
          }
        } catch(_) {}
      };
      es.addEventListener('progress', handle); es.addEventListener('done', handle);
      es.addEventListener('error', () => { if (es.readyState === 2) es.close(); });
    } catch (e) {
      setError(e.message); addLog(`Khalifa Test: ${e.message}`, 'error'); setLoading(false);
    }
  }, [addLog]);

  // ── Refresh the list of saved BP-NN checkpoints ──
  const refreshModels = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/cnn-v2/models`);
      const d = await r.json();
      setAvailableModels(d.models || []);
      addLog(`${(d.models||[]).length} saved BP-NN checkpoint(s) available`, 'success');
    } catch (e) { addLog(`models: ${e.message}`, 'error'); }
  }, [addLog]);
  // Auto-refresh on first load
  useEffect(() => { refreshModels(); }, [refreshModels]);

  // ── Inference-only run (no training, uses a saved checkpoint) ──
  const runInferenceOnly = useCallback(async () => {
    if (!roi) return;
    if (!params.bc_inference_model) {
      addLog('Pick a saved model in the Bias & Calibration panel first', 'warn');
      return;
    }
    setLoading(true); setError(null); setLoadingMsg('Inference (no training)…');
    addLog(`CNN_v2 INFERENCE · model=${params.bc_inference_model.split('/').pop()}`);
    try {
      const r = await fetch(`${API}/api/cnn-v2/inference`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ bbox: roi, weights_file: params.bc_inference_model }),
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      addLog(`Inference done · grid ${d.grid_shape?.join('×')} · depth ${d.metrics?.depth_min_m?.toFixed(2)}–${d.metrics?.depth_max_m?.toFixed(2)}m`, 'success');
      setResults({
        method: d.method, architecture: d.architecture,
        metrics: d.metrics, bbox: roi,
        raster_png: d.preview_b64, raster_bounds: {
          west: roi.west, east: roi.east, south: roi.south, north: roi.north
        },
        geotiff_b64: d.geotiff_b64,
        weights_file: { path: d.weights_file, relative_path: d.weights_file, filename: d.weights_file.split('/').pop() },
        bias_correction_settings: { inference_only: true, model: d.weights_file },
      });
      setShowResults(true);
    } catch (e) { setError(e.message); addLog(`Inference: ${e.message}`, 'error'); }
    finally { setLoading(false); }
  }, [roi, params.bc_inference_model, addLog]);

  // ── CNN_v2 — BP Neural Network (Guo et al. 2022) ──
  const runCnnV2 = useCallback(async () => {
    if (!roi) return;
    // Branch: inference-only mode short-circuits to the no-training endpoint
    if (params.bc_inference_only) { return runInferenceOnly(); }
    setLoading(true); setError(null);
    setLoadingTotal(100); setLoadingStep(0);
    setLoadingMsg('Starting CNN_v2 (Guo 2022 BP-NN) ...');
    addLog('CNN_v2: submitting job (S2 B/G/R/NIR + Stumpf ratios → BP-NN, ICESat-2 + observed labels)');
    try {
      const startR = await fetch(`${API}/api/cnn-v2/start`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          bbox: roi,
          region: params.site_key || 'custom',
          epochs: params.cnn_v2_epochs || 150,
          // Bias-correction & calibration options from Sidebar
          bc_mode:        params.bc_mode || 'single',
          bc_window_days: params.bc_window_days || 180,
          bc_n_scenes:    params.bc_n_scenes || 4,
          bc_tide:        params.bc_tide  !== false,
          bc_wave:        params.bc_wave  !== false,
          bc_geoid:       params.bc_geoid !== false,
          bc_calib:       params.bc_calib || 'shift+bias+local',
          bc_deep:        params.bc_deep !== false,
          user_points: (userPoints || []).filter(p =>
            p?.photon_class === 'observed' || p?.photon_class === 'icesat2_cshelph' || p?.photon_class === 'bathymetry'),
        }),
      });
      const s = await startR.json();
      if (s.error) { setError(s.error); addLog(`CNN_v2: ${s.error}`, 'error'); setLoading(false); return; }
      addLog(`CNN_v2 job ${s.job_id} submitted — opening SSE stream`);
      const es = new EventSource(`${API}${s.stream_url}`);
      const handle = (ev) => {
        try {
          const d = JSON.parse(ev.data);
          if (typeof d.progress === 'number') {
            setLoadingStep(Math.max(0, Math.min(100, Math.round(d.progress))));
          }
          if (d.message) setLoadingMsg(`[${d.phase}] ${d.message}`);
          if (d.phase === 'done' && d.result) {
            const r = d.result;
            const dl = r.downloads || {};
            const m = r.metrics || {};
            addLog(`CNN_v2 done · R²=${m.r2?.toFixed(3)} · RMSE=${m.rmse_m?.toFixed(2)}m · MAE=${m.mae_m?.toFixed(2)}m · n_train=${m.n_train} · epochs_used=${m.epochs_used}`, 'success');
            if (dl.geotiff) addLog(`GeoTIFF: ${API}${dl.geotiff}`, 'success');
            if (dl.preview) addLog(`Preview PNG: ${API}${dl.preview}`, 'success');
            setResults({
              architecture: r.architecture,
              citation: r.citation,
              method: r.method,
              metrics: m, sources: r.sources,
              bp_info: r.bp_info,
              accurate_downloads: {
                geotiff_url: dl.geotiff ? `${API}${dl.geotiff}` : null,
                uncertainty_url: dl.uncertainty ? `${API}${dl.uncertainty}` : null,
                preview_url: dl.preview ? `${API}${dl.preview}` : null,
              },
              job_id: r.job_id,
            });
            setShowResults(true);
            setLoading(false); es.close();
          } else if (d.phase === 'error') {
            setError(d.message || 'error'); addLog(`CNN_v2: ${d.message}`, 'error');
            setLoading(false); es.close();
          }
        } catch (e) { /* ignore malformed SSE frames */ }
      };
      es.addEventListener('progress', handle);
      es.addEventListener('done', handle);
      es.addEventListener('error', (ev) => { if (es.readyState === 2) es.close(); });
    } catch (e) {
      setError(e.message); addLog(`CNN_v2: ${e.message}`, 'error'); setLoading(false);
    }
  }, [roi, params, userPoints, addLog]);

  // i-Boating full pipeline
  const runIBoatingPipeline = useCallback(async () => {
    if (!roi) return;
    setLoading(true); setError(null);
    const steps = ['Opening i-Boating chart...','Waiting for tiles to render...','AI reading depth soundings...','Training CNN on chart depths (80%)...','Predicting full bathymetry...','Validating on held-out 20%...'];
    setLoadingTotal(steps.length); setLoadingStep(0); setLoadingMsg(steps[0]);
    addLog(`i-Boating pipeline: zoom ${params.chart_zoom||14}, split 80/20`);
    let si=0; const iv=setInterval(()=>{si++;if(si<steps.length){setLoadingStep(si);setLoadingMsg(steps[si]);}},12000);
    try {
      const r = await fetch(`${API}/api/iboating-pipeline`, {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({bbox:roi,params,start_date:params.start_date,end_date:params.end_date,zoom:params.chart_zoom||14})});
      const d = await r.json(); clearInterval(iv);
      if (d.error && !d.interpolated_points?.length) { setError(d.error); addLog(`i-Boating: ${d.error}`,'error'); return; }
      const res = {...d, points:[...(d.points||[]),...(d.interpolated_points||[])]};
      setResults(res); setShowResults(true);
      const ib = d.iboating || {};
      addLog(`i-Boating done! ${ib.n_chart||0} chart pts → ${ib.n_train||0} train / ${ib.n_val||0} val. Sources: ${(d.sources_used||[]).join(', ')}`,'success');
      if (ib.validation) {
        addLog(`Validation (20% held-out): RMSE=${ib.validation.rmse?.toFixed(2)}m, R²=${ib.validation.r2?.toFixed(3)}, S-44=${ib.validation.s44_pass_pct?.toFixed(0)}%`,'success');
        setAutoEvalResults({...ib.validation, source:'i-Boating chart (20% held-out)', n_chart:ib.n_chart});
      }
    } catch(e) { clearInterval(iv); setError(e.message); addLog(e.message,'error'); }
    finally { setLoading(false); }
  }, [roi, params, addLog]);

  // Compute depth difference A - B
  const computeDiff = useCallback(() => {
    if (!results || !resultsB) return;
    addLog('Computing depth difference (A − B)...');
    const ptsA = (results.points||[]).filter(p => p.photon_class === 'interpolated' && p.depth > 0);
    const ptsB = (resultsB.points||[]).filter(p => p.photon_class === 'interpolated' && p.depth > 0);
    // Build grid lookup for B
    const gridB = {};
    ptsB.forEach(p => { gridB[`${p.lat.toFixed(4)}_${p.lon.toFixed(4)}`] = p.depth; });
    const diffs = [];
    ptsA.forEach(p => {
      const key = `${p.lat.toFixed(4)}_${p.lon.toFixed(4)}`;
      if (gridB[key]) {
        diffs.push({ lat: p.lat, lon: p.lon, depthA: p.depth, depthB: gridB[key], diff: p.depth - gridB[key], photon_class: 'diff' });
      }
    });
    if (diffs.length === 0) { addLog('No overlapping points found','error'); return; }
    const diffVals = diffs.map(d => d.diff);
    const dr = {
      points: diffs, n_points: diffs.length,
      mean_diff: diffVals.reduce((a,b)=>a+b,0)/diffs.length,
      max_diff: Math.max(...diffVals), min_diff: Math.min(...diffVals),
      std_diff: Math.sqrt(diffVals.reduce((s,v)=>s+(v-diffVals.reduce((a,b)=>a+b,0)/diffs.length)**2,0)/diffs.length),
    };
    setDiffResults(dr);
    addLog(`Difference: mean=${dr.mean_diff.toFixed(2)}m, max=${dr.max_diff.toFixed(2)}m, ${dr.n_points} pts`,'success');
  }, [results, resultsB, addLog]);

  const exportData = useCallback(async (format) => {
    // ROB-R5: surface a visible warning instead of silently no-op'ing when
    // the user clicks Export before running an analysis.
    if (!results) { addLog('Export: run an analysis first (no result to export).', 'warn'); return; }
    try {
      // GeoTIFF: prefer the inline base64 if we have it, else fall back
      // to the on-demand /api/sdb-pro/geotiff endpoint (the fast path
      // omits the inline geotiff_b64 to keep the JSON response small).
      if (format === 'geotiff') {
        if (results.geotiff_b64) {
          const bin = atob(results.geotiff_b64);
          const arr = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          const a = document.createElement('a');
          a.href = URL.createObjectURL(new Blob([arr], {type:'image/tiff'}));
          a.download = 'bathymetry.tif'; a.click();
          return;
        }
        const bbox = results.bbox || roi;
        const bd = Array.isArray(bbox)
          ? { west: bbox[0], south: bbox[1], east: bbox[2], north: bbox[3] }
          : bbox;
        addLog('Generating GeoTIFF…');
        const r = await fetch(`${API}/api/sdb-pro/geotiff`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            bbox: bd,
            start_date: params.start_date,
            end_date: params.end_date,
          }),
        });
        if (!r.ok) {
          const d = await r.json().catch(() => ({error:'GeoTIFF download failed'}));
          throw new Error(d.error);
        }
        const blob = await r.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'bathymetry.tif'; a.click();
        addLog('GeoTIFF downloaded', 'success');
        return;
      }
      // CSV / GeoJSON / NetCDF: use whatever depth-point set we have.
      const pts = (results.points && results.points.length)
        ? results.points
        : (results.interpolated_points || []);
      // Reference-system metadata rides along so exports are self-describing.
      // GIS-R4: the exported `points` here are the MODEL grid (results.points),
      // not the uploaded survey — so `vertical_datum` MUST be the product's
      // own datum ('LAT' unless a model result explicitly overrides it), never
      // the uploaded observedData's. The uploaded survey's datum (if any) is
      // still worth recording — it rides along under a distinct field so the
      // backend can tag it separately (REFERENCE_SURVEY_DATUM) without ever
      // asserting it as the product's VERTICAL_DATUM.
      const r = await fetch(`${API}/api/export`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({points:pts,format,bbox:results.bbox||roi,
        vertical_datum:results.vertical_datum||'LAT',
        reference_survey_datum:observedData?.vertical_datum||null,
        source_crs:observedData?.crs||'WGS84 (EPSG:4326)'})});
      if(!r.ok){const d=await r.json().catch(()=>({error:`Export failed (HTTP ${r.status})`}));throw new Error(d.error);}
      const blob=await r.blob();const url=URL.createObjectURL(blob);
      const ext={geojson:'geojson',csv:'csv',geotiff:'tif'}[format]||format;
      const a=document.createElement('a');a.href=url;a.download=`bathymetry.${ext}`;a.click();
    } catch(e) { setError(e.message); addLog(`Export ${format}: ${e.message}`, 'error'); }
  }, [results, roi, params.start_date, params.end_date, observedData, addLog]);

  const uploadCSV = useCallback(async (file) => {
    const fd=new FormData(); fd.append('file',file);
    try {
      addLog(`Uploading ${file.name}...`);
      const r=await fetch(`${API}/api/upload-csv`,{method:'POST',body:fd});
      const d=await r.json(); if(d.error) throw new Error(d.error);
      setUserPoints(d.points||[]); setCsvPreview(d.points||[]);
      // Auto-set the ROI from the uploaded data extent (rectangle + auto-zoom).
      const bb = roiFromPoints(d.points);
      if (bb) window.dispatchEvent(new CustomEvent('setRoi', { detail: bb }));
      addLog(`${d.count} depth points loaded`,'success');
    } catch(e) { setError(e.message); }
  }, [addLog]);

  const uploadChart = useCallback(async (file) => {
    if(!roi){setError('Draw ROI first');return;}
    const fd=new FormData();fd.append('file',file);fd.append('bbox',JSON.stringify(roi));
    try {
      addLog('Gemini Vision reading depth numbers...');
      setLoading(true);setLoadingMsg('AI reading nautical chart depths...');
      const r=await fetch(`${API}/api/extract-chart`,{method:'POST',body:fd});
      const d=await r.json();
      if(d.points?.length>0){
        setUserPoints(prev=>[...prev,...d.points]);setCsvPreview(prev=>[...(prev||[]),...d.points]);
        addLog(`✓ ${d.count} depth soundings from chart`,'success');
      } else { addLog(d.error||d.message||'No depths found',d.error?'error':'info'); }
    } catch(e){setError(e.message);}
    finally{setLoading(false);}
  }, [roi, addLog]);

  const uploadShapefile = useCallback(async (files) => {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    try {
      addLog('Uploading shapefile...');
      setLoading(true); setLoadingMsg('Parsing shapefile (UTM → WGS84)...');
      const r = await fetch(`${API}/api/upload-shapefile`, { method: 'POST', body: fd });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      setValidationPts(d.points || []);
      setCsvPreview(prev => [...(prev || []), ...(d.points || [])]);
      // Auto-set the ROI from the uploaded data extent (rectangle + auto-zoom).
      const bb = roiFromPoints(d.points);
      if (bb) window.dispatchEvent(new CustomEvent('setRoi', { detail: bb }));
      addLog(`✓ ${d.count} in-situ points loaded (${d.crs || 'WGS84'})`, 'success');
    } catch (e) { setError(e.message); addLog(e.message, 'error'); }
    finally { setLoading(false); }
  }, [addLog]);

  const validateResults = useCallback(async () => {
    if (!results) { addLog('Validation needs a result — run Quick Analyse or Extract first.', 'warn'); return; }
    // Accept observed points from the dedicated validationPts OR from userPoints (photon_class='observed')
    const observed = validationPts.length > 0
      ? validationPts
      : (userPoints || []).filter(p => p?.photon_class === 'observed' && p.depth > 0);
    if (observed.length === 0) { addLog('No observed points — upload an XYZ / shapefile first.', 'warn'); return; }
    addLog(`Computing validation (RMSE, bias, S-44) · ${observed.length} obs × ${(results.points||[]).length} pred ...`);
    try {
      // IHO-R7: thread the actual run resolution through so the R5/R6
      // detection-capability disclosure states a concrete figure instead of
      // silently degrading to generic wording. results.stats.resolution_m is
      // populated on every run path (see setResults calls e.g. App.js:1352,
      // 1413, 1584); params.resolution_m (the Sidebar selector) is the
      // fallback for any path that hasn't echoed it back from the backend yet.
      const resolutionM = results?.stats?.resolution_m ?? params?.resolution_m ?? null;
      // GIS-R3: propagate the uploaded survey's vertical datum so the backend
      // can flag an MSL/CD-vs-LAT-assumed mismatch instead of silently
      // folding the offset into bias/RMSE. 'LAT' is what every model
      // prediction is asserted at (_DATUM_NOTE, backend/app.py).
      const r = await fetch(`${API}/api/validate-xyz`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          predicted: results.points, observed, resolution_m: resolutionM,
          observed_vertical_datum: observedData?.vertical_datum ?? null,
          predicted_vertical_datum: 'LAT',
        })
      });
      // ROB-R1: a non-finite value anywhere in the payload would otherwise
      // make this throw a cryptic "Unexpected token 'N' ..." SyntaxError —
      // the backend now sanitises NaN/Infinity to null before responding, so
      // this parse is safe; keep the try/catch as defense-in-depth against
      // any future regression or a truly non-JSON error page.
      let d;
      try { d = await r.json(); }
      catch (e) { addLog('Validation: server returned an unparseable response (non-JSON) — please retry or report this.', 'error'); setError('Unparseable validation response'); return; }
      if (d.error) { addLog(`Validation: ${d.error}`, 'error'); setError(d.error); return; }
      setValidationResults(d);
      addLog(`Validation: RMSE=${d.rmse?.toFixed(2)}m, bias=${d.bias?.toFixed(2)}m, ${d.n_pairs} pairs, S-44-1a ${d.iho_orders?.order1a?.pass_pct?.toFixed(0)}%`
        + (d.datum_mismatch ? ` ⚠ datum mismatch (${d.datum_mismatch_note || 'see validation panel'})` : ''), 'success');
    } catch (e) { setError(e.message); addLog(e.message, 'error'); }
  }, [results, validationPts, userPoints, addLog, params?.resolution_m, observedData]);

  // Auto-run validation as soon as both a prediction AND observed points are present
  useEffect(() => {
    const hasObs = validationPts.length > 0
      || (userPoints || []).some(p => p?.photon_class === 'observed' && p.depth > 0);
    if (results && hasObs && !validationResults) {
      validateResults();
    }
  }, [results, validationPts, userPoints, validationResults, validateResults]);

  const openChart = useCallback(() => {
    if(!roi) return;
    const z=params.chart_zoom||14;
    const lat=((roi.north+roi.south)/2).toFixed(4);
    const lon=((roi.east+roi.west)/2).toFixed(4);
    window.open(`https://fishing-app.gpsnauticalcharts.com/i-boating-fishing-web-app/fishing-marine-charts-navigation.html#${z}/${lat}/${lon}`,'iboating','width=1400,height=900');
    addLog(`Opened i-Boating at zoom ${z}`);
  }, [roi, params, addLog]);

  useEffect(() => {
    const handlePaste = (e) => {
      if(!roi) return;
      const items=e.clipboardData?.items;if(!items) return;
      for(const item of items){
        if(item.type.startsWith('image/')){
          const file=item.getAsFile();
          if(file){addLog('📋 Pasted chart screenshot...');uploadChart(file);}
          break;
        }
      }
    };
    document.addEventListener('paste',handlePaste);
    return ()=>document.removeEventListener('paste',handlePaste);
  }, [roi, uploadChart, addLog]);

  // Smart Chart Digitiser → full bathymetry map
  const smartDigitise = useCallback(async (file) => {
    if(!roi){setError('Draw ROI first');return;}
    const fd=new FormData();fd.append('file',file);fd.append('bbox',JSON.stringify(roi));
    try {
      addLog('Chart → Bathymetry: AI reading every depth sounding...');
      setLoading(true);setLoadingMsg('AI reading depth soundings from chart...');
      const steps=['Reading chart depths...','Georeferencing points...','Interpolating bathymetry grid...'];
      setLoadingTotal(steps.length);setLoadingStep(0);
      let si=0;const iv=setInterval(()=>{si++;if(si<steps.length){setLoadingStep(si);setLoadingMsg(steps[si]);}},6000);
      const r=await fetch(`${API}/api/smart-chart-digitise`,{method:'POST',body:fd});
      const d=await r.json();clearInterval(iv);
      if(d.error && !d.points?.length){addLog(d.error,'error');setLoading(false);return;}
      setDigitiseResults(d);
      // Feed chart points as training reference for future CNN runs
      if(d.points?.length>0){
        setUserPoints(prev=>[...prev,...d.points]);
        setCsvPreview(prev=>[...(prev||[]),...d.points]);
      }
      // Load as full results (same format as /api/extract) so map + 3D + results panel all work
      if(d.interpolated_points?.length>0){
        const res={...d, points:[...(d.points||[]),...(d.interpolated_points||[])]};
        setResults(res);setShowResults(true);
      }
      addLog(`Chart digitised: ${d.soundings||0} soundings + ${d.contours||0} contours + ${d.color_pts||0} colour → ${d.grid_points||0} grid pts (max ${d.max_depth_cap}m)`,'success');
    } catch(e){setError(e.message);addLog(e.message,'error');}
    finally{setLoading(false);}
  }, [roi, addLog]);

  // ICESat-2 CShelph — search granules
  const icesat2Search = useCallback(async () => {
    if (!roi) return;
    setLoading(true);setLoadingMsg('Searching ICESat-2 ATL03 granules...');
    addLog(`ICESat-2: searching ${params.ice_start} → ${params.ice_end}...`);
    try {
      const r=await fetch(`${API}/api/icesat2-search`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({bbox:roi,start_date:params.ice_start,end_date:params.ice_end})});
      const d=await r.json();
      if(d.error){addLog(d.error,'error');}
      else{setIcesat2Granules(d.granules||[]);addLog(`Found ${d.count} ATL03 granules`,'success');}
    } catch(e){addLog(e.message,'error');}
    finally{setLoading(false);}
  }, [roi, params.ice_start, params.ice_end, addLog]);

  // ICESat-2 CShelph — process granule
  const icesat2Process = useCallback(async (granuleIdx=0) => {
    if (!roi) return;
    setLoading(true);setError(null);
    const multiPass=!!params.ice_multi_pass;
    const steps=multiPass
      ?['Downloading ATL03...','Reading photons...','Orthometric correction...','Adaptive binning...','KDE sea surface...','Refraction correction...','Kd estimation...','Bottom detection...','MAD filtering...','TPU calculation...','Multi-pass aggregation...']
      :['Downloading ATL03...','Reading photons...','Orthometric correction...','Adaptive binning...','KDE sea surface...','Refraction correction...','Kd estimation...','Bottom detection...','MAD filtering...','TPU calculation...'];
    setLoadingTotal(steps.length);setLoadingStep(0);setLoadingMsg(steps[0]);
    addLog(`ICESat-2 CShelph: laser=${params.ice_laser||'all'}, threshold=${params.ice_threshold}, T=${params.ice_water_temp}°C${multiPass?' [MULTI-PASS]':''}`);
    let si=0;const iv=setInterval(()=>{si++;if(si<steps.length){setLoadingStep(si);setLoadingMsg(steps[si]);}},6000);
    try {
      const r=await fetch(`${API}/api/icesat2-process`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({bbox:roi,start_date:params.ice_start,end_date:params.ice_end,
          laser:params.ice_laser||0,threshold:params.ice_threshold||30,
          water_temp:params.ice_water_temp||22,granule_index:granuleIdx,
          multi_pass:multiPass,n_granules:params.ice_n_granules||3})});
      const d=await r.json();clearInterval(iv);
      if(d.error && !d.depth_points?.length){addLog(d.error,'error');setLoading(false);return;}
      setIcesat2Results(d);
      // Add depth points as reference data for CNN training
      if(d.depth_points?.length>0){
        setUserPoints(prev=>[...prev,...d.depth_points]);
        setCsvPreview(prev=>[...(prev||[]),...d.depth_points]);
      }
      if(d.available_granules)setIcesat2Granules(d.available_granules);
      addLog(`ICESat-2 CShelph: ${d.n_depths} depth points from ${d.n_beams} beams. Granule: ${d.granule?.id} (${d.granule?.date})`,'success');
      d.tracks?.forEach(t=>addLog(`  Beam ${t.beam}: ${t.n_depths} depths [${t.depth_range[0]}-${t.depth_range[1]}m], Kd=${t.kd_532||'?'} m⁻¹, TPU=${t.mean_tpu_m||'?'}m`,'success'));
      if(d.quality)addLog(`  Quality: Kd=${d.quality.mean_kd} m⁻¹, TPU=${d.quality.mean_tpu_m}m, max_detect=${d.quality.max_detectable_m}m`,'success');
      if(d.processed_granules?.length>1)addLog(`  Multi-pass: ${d.processed_granules.length} granules aggregated`,'success');
    } catch(e){clearInterval(iv);setError(e.message);addLog(e.message,'error');}
    finally{setLoading(false);}
  }, [roi, params, addLog]);

  const autoEvaluate = useCallback(async () => {
    if (!results || !roi) return;
    addLog('Auto-evaluating against nautical chart (OpenSeaMap + Gemini)...');
    try {
      setLoading(true); setLoadingMsg('Fetching chart tiles & extracting depths...');
      const r = await fetch(`${API}/api/auto-evaluate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox: roi, predicted: results.points, zoom: params.chart_zoom || 14 })
      });
      const d = await r.json();
      if (d.error && !d.n_pairs) { addLog(d.error, 'error'); setAutoEvalResults(d); }
      else {
        setAutoEvalResults(d);
        if (d.chart_points?.length) {
          setUserPoints(prev => [...prev, ...d.chart_points]);
          setCsvPreview(prev => [...(prev || []), ...d.chart_points]);
        }
        addLog(`Auto-eval: RMSE=${d.rmse?.toFixed(2)}m, bias=${d.bias?.toFixed(2)}m, R²=${d.r2?.toFixed(3)}, S-44=${d.s44_pass_pct?.toFixed(0)}% (${d.n_pairs} pairs from ${d.n_chart} chart pts)`, 'success');
      }
    } catch (e) { setError(e.message); addLog(e.message, 'error'); }
    finally { setLoading(false); }
  }, [results, roi, params.chart_zoom, addLog]);

  // High-res satellite capture
  // Load known observed bathymetry site — just loads points + sets ROI, user clicks Extract themselves
  const loadObserved = useCallback(async (siteKey) => {
    setLoading(true); setLoadingMsg('Loading observed bathymetry...');
    addLog(`Loading observed bathymetry: ${siteKey}...`);
    try {
      const r = await fetch(`${API}/api/load-observed`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ site: siteKey }) });
      const d = await r.json();
      if (d.error) { addLog(d.error, 'error'); setLoading(false); return; }
      // Set ROI on map from data extent
      if (d.bbox) {
        const ev = new CustomEvent('setRoi', { detail: d.bbox });
        window.dispatchEvent(ev);
      }
      // Store as training data + validation data, auto-set fusion mode
      setUserPoints(d.points || []);
      setCsvPreview(d.points || []);
      setValidationPts(d.points || []);
      setObservedData(d);
      setParams(p => ({...p, ref_source: 'fusion', img_source: 'mapbox'}));
      addLog(`Loaded ${d.count} observed points (${d.site}). FUSION mode (Mapbox + GEBCO + ICESat-2 + In-situ). Draw ROI → Extract.`, 'success');
    } catch (e) { addLog(e.message, 'error'); }
    setLoading(false);
  }, [addLog]);

  // Lightweight dataset-metadata fetch for the predefined regions — shows
  // vertical/horizontal datum, original CRS, point count and bbox in the
  // sidebar without loading the full point cloud onto the map.
  const loadObservedMeta = useCallback(async (siteKey) => {
    try {
      const r = await fetch(`${API}/api/load-observed`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ site: siteKey, meta_only: true }),
      });
      const d = await r.json();
      if (!d.error) setObservedData(d);
    } catch (e) { /* metadata card is best-effort — never block the run */ }
  }, []);

  // CBR — Cluster-Based Regression (Geyman & Maloof 2019) + SlideRule ICESat-2
  const runCbr = useCallback(async ({ nClusters = 5, smoothBoundaries = true, minPtsPerCluster = 8, nScenes = 1, featuresMode = 'poly', depthWeighting = true, calibrate = 'local', retide = true, waveCorrect = true } = {}) => {
    if (!roi) return;
    const bbox = { west: roi.west, south: roi.south, east: roi.east, north: roi.north };
    // CBR wants bathymetric reference points — uploaded survey, CShelph, or SlideRule-tagged photons.
    const user_points = (userPoints || []).filter(p =>
      p?.photon_class === 'observed' ||
      p?.photon_class === 'icesat2_cshelph' ||
      p?.photon_class === 'bathymetry'
    ).map(p => ({ lat: p.lat, lon: p.lon, depth: Math.abs(p.depth) }));

    setLoading(true); setError(null);
    const steps = nScenes > 1 ? [
      user_points.length ? `Using ${user_points.length} user reference points...` : 'Querying SlideRule ICESat-2 ATL03 (multi-year)...',
      `Fetching ${nScenes} Sentinel-2 scenes across the window...`,
      `Running CBR per scene (K=${nClusters}) on deglinted reflectance...`,
      'Gaussian MLE fusion (inverse-variance, 3-pass IRLS outlier rejection)...',
      'Validating fused depth vs references...',
    ] : [
      'Fetching Sentinel-2 (deglinted + deep-water subtracted)...',
      user_points.length ? `Using ${user_points.length} user reference points...` : 'Querying SlideRule ICESat-2 ATL03...',
      `Running K-means (K=${nClusters}) on bottom-type spectrum...`,
      'Fitting per-cluster robust linear regression...',
      smoothBoundaries ? 'Predicting depth with soft cluster blending...' : 'Predicting depth per cluster...',
    ];
    setLoadingTotal(steps.length); setLoadingStep(0); setLoadingMsg(`CBR: ${steps[0]}`);
    addLog(`CBR · K=${nClusters} · smooth=${smoothBoundaries} · scenes=${nScenes} · ${user_points.length || 'auto'} refs`);
    let si = 0; const iv = setInterval(() => {
      si++; if (si < steps.length) { setLoadingStep(si); setLoadingMsg(`CBR: ${steps[si]}`); }
    }, 6000);
    try {
      const r = await fetch(`${API}/api/cbr-bathymetry`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bbox,
          start_date: params.start_date,
          end_date: params.end_date,
          cloud: params.max_cloud || 25,
          resolution_m: params.resolution_m || 20,
          n_clusters: nClusters,
          smooth_boundaries: smoothBoundaries,
          min_pts_per_cluster: minPtsPerCluster,
          n_scenes: nScenes,
          features_mode: featuresMode,
          depth_weighting: depthWeighting,
          calibrate,
          retide,
          wave_correct: waveCorrect,
          user_points,
        }),
      });
      const d = await r.json(); clearInterval(iv);
      if (d.error) {
        if (Array.isArray(d.icesat2_debug) && d.icesat2_debug.length) {
          d.icesat2_debug.forEach(line => addLog(`  SlideRule · ${line}`, 'info'));
        }
        throw new Error(d.error);
      }
      setResults({ ...d, points: [...(d.points || []), ...(d.interpolated_points || [])] });
      setShowResults(true); setCsvPreview(null); setResultsB(null); setDiffResults(null);
      const m = d.ml_stats || {};
      const mle = d.cbr?.mle;
      const rt = d.cbr?.retide;
      addLog(`CBR done · R²=${m.r2} · RMSE=${m.rmse}m · MAE=${m.mae}m · ${m.iho_s44} · K=${d.cbr?.n_clusters}` +
        (mle ? ` · MLE ${mle.n_scenes_used}/${mle.n_scenes_requested} scenes · cov=${mle.mean_coverage_per_px?.toFixed(1)}${mle.n_rescued?` · rescued=${mle.n_rescued}`:''}` : '') +
        (rt?.applied ? ` · retided ${rt.n_retided}photons · med Δ=${rt.tide_shift_median_m}m · |Δ|avg=${rt.tide_shift_abs_mean_m}m${rt.wave_shift_abs_mean_m?` · wave avg ${rt.wave_shift_abs_mean_m}m`:''}`:''), 'success');
      if (d.cbr?.per_scene) {
        d.cbr.per_scene.forEach(ps => addLog(`  scene ${ps.date_range}: ${ps.status === 'ok' ? `R²=${ps.r2} RMSE=${ps.rmse}m` : ps.status}`, ps.status === 'ok' ? 'info' : 'error'));
      }
    } catch (e) { clearInterval(iv); setError(e.message); addLog(`CBR: ${e.message}`, 'error'); }
    finally { setLoading(false); }
  }, [roi, params.start_date, params.end_date, params.max_cloud, params.resolution_m, userPoints, addLog]);

  // S2Shores — physical wave-dispersion bathymetry (Almar et al. / CNES)
  const runS2Shores = useCallback(async ({ resolutionM = 50, windowM = null, stepM = null, source = 'auto' } = {}) => {
    if (!roi) return;
    setLoading(true); setError(null);
    const steps = [
      `Picking least-cloudy S2 L1C tile (${source.toUpperCase()}) at ${resolutionM}m...`,
      'Extracting B02+B04 (the ~1.005s inter-detector offset carries the wave celerity signal)...',
      'Detrending + high-pass filtering bands; Radon transform for wave direction per window...',
      'Cross-spectral analysis on sinograms → local wavelength, celerity, period...',
      'Inverting linear dispersion ω² = g·k·tanh(k·h) → depth grid...',
    ];
    setLoadingTotal(steps.length); setLoadingStep(0); setLoadingMsg(`S2Shores: ${steps[0]}`);
    addLog(`S2Shores · res=${resolutionM}m · window=${windowM||8*resolutionM}m · step=${stepM||2*resolutionM}m · src=${source}`);
    let si = 0; const iv = setInterval(() => {
      si++; if (si < steps.length) { setLoadingStep(si); setLoadingMsg(`S2Shores: ${steps[si]}`); }
    }, 9000);
    try {
      const r = await fetch(`${API}/api/s2shores-bathymetry`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bbox: { west: roi.west, south: roi.south, east: roi.east, north: roi.north },
          start_date: params.start_date,
          end_date: params.end_date,
          cloud: params.max_cloud || 15,
          resolution_m: resolutionM,
          window_m: windowM || 8 * resolutionM,
          step_m: stepM || 2 * resolutionM,
          source,
        }),
      });
      const d = await r.json(); clearInterval(iv);
      if (d.error) throw new Error(d.error);
      setResults({ ...d, points: [...(d.points || []), ...(d.interpolated_points || [])] });
      setShowResults(true); setCsvPreview(null); setResultsB(null); setDiffResults(null);
      const s = d.s2shores || {};
      addLog(`S2Shores done · ${s.valid_windows}/${s.total_windows} windows valid · λ̄=${s.mean_wavelength_m}m · c̄=${s.mean_celerity_m_s}m/s · T̄=${s.mean_period_s}s · src=${s.source}`, 'success');
    } catch (e) { clearInterval(iv); setError(e.message); addLog(`S2Shores: ${e.message}`, 'error'); }
    finally { setLoading(false); }
  }, [roi, params.start_date, params.end_date, params.max_cloud, addLog]);

  // Clustered SDB — Mapbox VHR + 12-band S2 + OSM + K-means/HGB ensemble +
  // band-aware augmentation (in-situ → SlideRule + i-Boating → GEBCO).
  const clusteredExtract = useCallback(async (siteKey = null, opts = {}) => {
    const obs = (userPoints || [])
      .filter(p => p?.photon_class === 'observed' || p?.photon_class === 'icesat2_cshelph')
      .map(p => ({ lat: p.lat, lon: p.lon, depth: p.depth }));
    // Resolution run mode drives both the imagery source and the output grid.
    //   vhr → Mapbox VHR base raster, ≈1 m target grid
    //   10m → Sentinel-2 only, 10 m grid (production S2 Lyzenga/Stumpf)
    //   20m → Sentinel-2 only, 20 m (coarser/aggregate fast overview)
    // Caller can still override imagery_source explicitly via opts.imagerySource.
    const resolution = opts.resolution || params.resolution || '10m';
    let src, targetResM;
    if (resolution === 'vhr') { src = 'vhr'; targetResM = opts.targetResM || 1.0; }
    else if (resolution === '20m') { src = 's2'; targetResM = 20.0; }
    else if (resolution === '50m') { src = 's2'; targetResM = 50.0; }
    else if (resolution === '100m') { src = 's2'; targetResM = 100.0; }
    else { src = 's2'; targetResM = 10.0; }
    if (opts.imagerySource === 's2') src = 's2';
    if (opts.imagerySource === 'vhr') src = 'vhr';
    const payload = {
      bbox: roi,
      site_key: siteKey,
      resolution,                 // '10m' | '20m' | '50m' | '100m' | 'vhr' — backend run param
      // User-selected output resolution (10/20/50/100 m) for the S2 fast path
      resolution_m: src === 's2' ? Math.round(targetResM) : undefined,
      train_frac: typeof opts.trainFrac === 'number' ? opts.trainFrac : 0.20,
      n_clusters: opts.nClusters || 8,
      max_iter: opts.maxIter || 1200,
      n_estimators: opts.nEstimators || 5,
      target_res_m: targetResM,
      augment: opts.augment !== false,
      aug_min_per_band: opts.augMinPerBand || 150,
      s2_start_date: params.start_date,
      s2_end_date: params.end_date,
      user_points: obs,
      imagery_source: src,
    };
    if (!payload.bbox && !payload.site_key) return;
    setLoading(true);
    const resLabel = resolution === 'vhr'
      ? (INTERNAL_ANALYSIS ? 'Very HR (Mapbox ~1 m)' : 'High-Resolution (~1 m)')
      : resolution === '20m' ? '20 m overview'
      : resolution === '50m' ? '50 m overview'
      : resolution === '100m' ? '100 m overview'
      : (INTERNAL_ANALYSIS ? '10 m (Sentinel-2)' : '10 m');
    setLoadingMsg(resolution === 'vhr'
      ? (INTERNAL_ANALYSIS ? 'Very HR — Mapbox ~1 m, computing depth grid…' : 'High-Resolution ~1 m, computing depth grid…')
      : `Computing depth grid at ${resLabel}…`);
    setError(null);
    addLog(INTERNAL_ANALYSIS
      ? `Bathymetry run${siteKey ? ` · ${siteKey}` : ''} · resolution=${resolution} (${src === 's2' ? 'Sentinel-2' : 'Mapbox VHR + S2'})…`
      : `Depth run${siteKey ? ` · ${siteKey}` : ''} · resolution=${resolution}…`);
    try {
      const r = await fetch(`${API}/api/very-hr-clustered`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.error) {
        addLog(d.error, 'error'); setError(d.error);
      } else {
        const m = d.metrics || {};
        // The MapPanel renders results.raster_png at results.raster_bounds.
        // Very HR returns depth_png_b64 + bbox=[w,s,e,n] — translate to the
        // raster_* shape so the depth grid actually shows on the map.
        const bb = d.bbox && d.bbox.length === 4
          ? [[d.bbox[1], d.bbox[0]], [d.bbox[3], d.bbox[2]]]   // [[s,w],[n,e]]
          : null;
        // Preserve every stat/ml_stat the backend ships and only top up
        // the keys the legacy CBR shape used to expose. Without the spread
        // the calibrated default profile (no held-out user pts) leaves
        // every BigStat / StatCard showing NaN.
        const backendStats = d.stats || {};
        const backendMl = d.ml_stats || {};
        // Fast S2 path returns depth points in `interpolated_points`. The
        // legacy CBR returned them in `points`. ResultsPanel, IHOChartView
        // and DepthHistogram all read `results.points`, so we copy the
        // interpolated array over when the legacy slot is empty.
        const fallbackPoints = (d.points && d.points.length)
          ? d.points
          : (d.interpolated_points || []).map(p => ({
              ...p,
              photon_class: p.photon_class || 'interpolated',
            }));
        // Normalise bbox to {west,south,east,north} for IHOChartView; the
        // fast path returns it as a [w,s,e,n] array.
        const bboxObj = Array.isArray(d.bbox) && d.bbox.length === 4
          ? { west: d.bbox[0], south: d.bbox[1], east: d.bbox[2], north: d.bbox[3] }
          : (d.bbox || null);
        setResults({
          ...d,
          bbox: bboxObj,
          points: fallbackPoints,
          raster_png: d.overlay_png_b64 || d.depth_png_b64 || null,
          raster_bounds: bb,
          ml_stats: {
            ...backendMl,
            rmse: m.rmse_m ?? backendMl.rmse,
            mae: m.mae_m ?? backendMl.mae,
            r2: m.r2 ?? backendMl.r2,
            bias: m.bias_m ?? backendMl.bias,
            n_pairs: m.n_test,
            n_train: backendMl.n_train ?? m.n_train,
          },
          method: d.method,
          stats: {
            ...backendStats,
            resolution_m: d.resolution_m || backendStats.resolution_m,
          },
        });
        setShowResults(true);
        // Only mention RMSE / S-44 when there were actual held-out points.
        const hasGT = (m.n_test ?? 0) >= 5 && m.rmse_m !== null && m.rmse_m !== undefined;
        if (hasGT) {
          addLog(
            `Very HR done · ${d.elapsed_s}s · RMSE=${m.rmse_m}m R²=${m.r2} bias=${m.bias_m}m S-44 1A=${m.s44_1a_pct}% O2=${m.s44_order2_pct}%`,
            'success'
          );
        } else {
          addLog(
            `Very HR done · ${d.elapsed_s}s · ${backendStats.grid_points || 0} px · mean ${backendStats.mean_depth ?? '–'}m`,
            'success'
          );
        }
      }
    } catch (e) { setError(e.message); addLog(e.message, 'error'); }
    finally { setLoading(false); }
  }, [roi, params.start_date, params.end_date, params.resolution, userPoints, addLog]);

  // ── Multi-scene MLE — full-year stack ──
  const veryHrMle = useCallback(async (opts = {}) => {
    if (!roi) return;
    const year = opts.year || 2024;
    const n_scenes = opts.n_scenes || 5;
    setLoading(true);
    setLoadingMsg(`MLE — stacking ${n_scenes} scenes across ${year}…`);
    setError(null);
    addLog(`MLE: ${n_scenes} scenes · year ${year}`);
    try {
      const r = await fetch(`${API}/api/very-hr-mle`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox: roi, year, n_scenes }),
      });
      const d = await r.json();
      if (d.error) { addLog(d.error, 'error'); setError(d.error); return; }
      const bb = d.bbox && d.bbox.length === 4
        ? [[d.bbox[1], d.bbox[0]], [d.bbox[3], d.bbox[2]]]
        : null;
      const bboxObj = Array.isArray(d.bbox) && d.bbox.length === 4
        ? { west: d.bbox[0], south: d.bbox[1], east: d.bbox[2], north: d.bbox[3] }
        : (d.bbox || null);
      // ResultsPanel + IHOChartView + DepthHistogram + ThreeDView all
      // read results.points. Back-fill from interpolated_points so the
      // 3D view, depth histogram and IHO chart populate after MLE.
      const fallbackPoints = (d.points && d.points.length)
        ? d.points
        : (d.interpolated_points || []).map(p => ({
            ...p,
            photon_class: p.photon_class || 'interpolated',
          }));
      setResults({
        ...d,
        bbox: bboxObj,
        points: fallbackPoints,
        raster_png: d.overlay_png_b64 || d.depth_png_b64 || null,
        raster_bounds: bb,
        ml_stats: { ...(d.ml_stats || {}), method: d.method },
        method: d.method,
        stats: { ...(d.stats || {}), resolution_m: d.resolution_m || (d.stats || {}).resolution_m },
      });
      setShowResults(true);
      const s = d.stats || {};
      addLog(
        `MLE done · ${d.elapsed_s}s · ${(d.augmentation || {}).mle_scenes?.length || 0} scenes · `
        + `mean depth ${s.mean_depth} m · σ̄ ${s.mean_sigma} m`,
        'success'
      );
    } catch (e) {
      setError(e.message); addLog(`MLE: ${e.message}`, 'error');
    } finally { setLoading(false); }
  }, [roi, addLog]);

  // ─────────────────────────────────────────────────────────────────────────
  // Very-HR local-processing job: register a job for an ROI, poll its
  // HONEST status (real progress_pct when the backend emits one, else an
  // indeterminate spinner — never a time-extrapolated bar), and accept a
  // result upload that flips it to "done".
  // ─────────────────────────────────────────────────────────────────────────
  const refreshVhrJobs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/very-hr-job/list`);
      const d = await r.json();
      if (d && Array.isArray(d.jobs)) setVhrJobs(d.jobs);
    } catch (_) { /* backend offline — leave list as-is */ }
  }, []);

  const vhrJobStart = useCallback(async (opts = {}) => {
    if (!roi) { addLog('Draw an ROI before queueing a Very-HR job', 'error'); return; }
    try {
      addLog('Queueing Very-HR local job…');
      const r = await fetch(`${API}/api/very-hr-job/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox: roi, label: opts.label || 'Very HR Bathymetry', params: opts.params || {} }),
      });
      if (!r.ok) {
        const txt = await r.text();
        const msg = `Job start HTTP ${r.status} — is the backend running on ${API || 'this origin'}? ${txt.slice(0,120)}`;
        addLog(msg, 'error'); setError(msg); return;
      }
      const d = await r.json();
      if (d.error) { addLog(`Job start: ${d.error}`, 'error'); setError(d.error); return; }
      setVhrJobs(prev => [d, ...prev.filter(j => j.id !== d.id)]);
      addLog(`Very-HR job ${d.id} queued · ${d.area_km2} km² (local processing)`, 'success');
    } catch (e) {
      const msg = `Job start failed: ${e.message}. Make sure the Flask backend is running.`;
      addLog(msg, 'error'); setError(msg);
    }
  }, [roi, addLog]);

  const vhrJobUpload = useCallback(async (jobId, file, notes) => {
    if (!jobId || !file) return;
    try {
      const fd = new FormData();
      fd.append('file', file);
      if (notes) fd.append('notes', notes);
      const r = await fetch(`${API}/api/very-hr-job/${jobId}/upload`, { method: 'POST', body: fd });
      const d = await r.json();
      if (d.error) { addLog(`Upload: ${d.error}`, 'error'); return; }
      setVhrJobs(prev => prev.map(j => j.id === jobId ? d : j));
      addLog(`Very-HR job ${jobId} done · ${d.result_filename} (${Math.round((d.result_size_bytes||0)/1024)} KB)`, 'success');
    } catch (e) {
      addLog(`Upload: ${e.message}`, 'error');
    }
  }, [addLog]);

  const vhrJobCancel = useCallback(async (jobId) => {
    if (!jobId) return;
    try {
      await fetch(`${API}/api/very-hr-job/${jobId}/cancel`, { method: 'POST' });
    } catch (_) { /* swallow */ }
    setVhrJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'cancelled', progress_pct: 0 } : j));
    addLog(`Very-HR job ${jobId} cancelled`, 'info');
  }, [addLog]);

  const vhrJobDelete = useCallback(async (jobId) => {
    if (!jobId) return;
    try {
      await fetch(`${API}/api/very-hr-job/${jobId}`, { method: 'DELETE' });
    } catch (_) { /* swallow */ }
    setVhrJobs(prev => prev.filter(j => j.id !== jobId));
  }, []);

  // Very HR + MLE PRO — kicks off a server-side job that runs 10 S2 median
  // scenes + Mapbox VHR + per-pixel inverse-variance MLE. The same job
  // poller surfaces REAL progress (`progress_source === 'real'`).
  const vhrMlePro = useCallback(async (opts = {}) => {
    if (!roi) { addLog('Draw an ROI before queueing Very HR + MLE PRO', 'error'); return; }
    const obs = (userPoints || [])
      .filter(p => p?.photon_class === 'observed' || p?.photon_class === 'icesat2_cshelph')
      .map(p => ({ lat: p.lat, lon: p.lon, depth: p.depth }));
    const body = {
      bbox: roi,
      year: opts.year || 2024,
      n_scenes: opts.n_scenes || 10,
      max_cloud: opts.max_cloud || 20,
      target_res_m: opts.target_res_m || 2.0,
      user_points: opts.user_points || obs,
      label: `Very HR + MLE PRO · ${opts.n_scenes || 10} S2 scenes (${opts.year || 2024})`,
    };
    try {
      addLog(`VHR + MLE PRO · queueing ${body.n_scenes}-scene fusion @ ${body.target_res_m} m/px…`);
      const r = await fetch(`${API}/api/very-hr-mle-pro/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const txt = await r.text();
        const msg = `VHR-MLE-PRO HTTP ${r.status} — ${txt.slice(0,160)}`;
        addLog(msg, 'error'); setError(msg); return;
      }
      const d = await r.json();
      if (d.error) { addLog(`VHR-MLE-PRO: ${d.error}`, 'error'); setError(d.error); return; }
      setVhrJobs(prev => [d, ...prev.filter(j => j.id !== d.id)]);
      addLog(`VHR-MLE-PRO ${d.id} started · ${d.area_km2} km² · server-side real-progress`, 'success');
    } catch (e) {
      const msg = `VHR-MLE-PRO start failed: ${e.message}`;
      addLog(msg, 'error'); setError(msg);
    }
  }, [roi, userPoints, addLog]);

  // Refresh jobs once on mount, then every 2 s while any job is running, 5 s otherwise.
  useEffect(() => {
    refreshVhrJobs();
    const anyRunning = (vhrJobs || []).some(j => j.status === 'processing');
    const period = anyRunning ? 2000 : 5000;
    const t = setInterval(refreshVhrJobs, period);
    return () => clearInterval(t);
  }, [refreshVhrJobs, vhrJobs]);

  // Almar Wave Physics (= S2Shores wave-dispersion bathymetry)
  // Aliased so the Sidebar can call a single semantic name.
  // (definition lives at the existing `runS2Shores` further down.)

  // Very HR Extract — Mapbox VHR or Sentinel-2-only (imagerySource = 'vhr' | 's2')
  const veryHrExtract = useCallback(async (siteKey = null, trainFrac = 0.20, imagerySource = 'vhr') => {
    const obs = (userPoints || [])
      .filter(p => p?.photon_class === 'observed' || p?.photon_class === 'icesat2_cshelph')
      .map(p => ({ lat: p.lat, lon: p.lon, depth: p.depth }));
    const src = (imagerySource === 's2') ? 's2' : 'vhr';
    const payload = {
      bbox: roi,
      site_key: siteKey,
      train_frac: trainFrac,
      target_res_m: 1.5,
      user_points: obs,
      imagery_source: src,
    };
    if (!payload.bbox && !payload.site_key) return;
    setLoading(true);
    setLoadingMsg(src === 's2'
      ? 'Very HR — Sentinel-2 (10 m), training on in-situ…'
      : 'Very HR — fetching Mapbox mosaic, training on in-situ…');
    setError(null);
    addLog(`Very HR Extract${siteKey ? ` · ${siteKey}` : ''} — ${src === 's2' ? 'Sentinel-2 only' : 'Mapbox VHR + S2'} + ${Math.round(trainFrac*100)}% train`);
    try {
      const r = await fetch(`${API}/api/very-hr-extract`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.error) {
        addLog(d.error, 'error'); setError(d.error);
      } else {
        const m = d.metrics || {};
        setResults({
          ...d,
          points: [],
          ml_stats: { rmse: m.rmse_m, mae: m.mae_m, r2: m.r2, bias: m.bias_m, n_pairs: m.n_test },
          method: d.method,
          stats: { resolution_m: d.resolution_m },
        });
        setShowResults(true);
        addLog(`Very HR done · ${d.elapsed_s}s · RMSE=${m.rmse_m}m R²=${m.r2} bias=${m.bias_m}m S-44=${m.s44_1b_pass_pct}% (n_train=${m.n_train}, n_test=${m.n_test})`, 'success');
      }
    } catch (e) { setError(e.message); addLog(e.message, 'error'); }
    finally { setLoading(false); }
  }, [roi, userPoints, addLog]);

  // Pro Extract — one-click orchestrated pipeline
  const proExtract = useCallback(async (siteKey = null) => {
    const payload = {
      bbox: roi,
      site_key: siteKey,
      start_date: params.start_date,
      end_date: params.end_date,
      // Accept both uploaded survey ("observed") and CShelph-derived depths ("icesat2_cshelph")
      user_points: (userPoints || []).filter(p =>
        p?.photon_class === 'observed' || p?.photon_class === 'icesat2_cshelph'
      ),
    };
    if (!payload.bbox && !payload.site_key) return;
    setLoading(true); setLoadingMsg('Pro Extract — gathering references, training model...'); setError(null);
    addLog(`Pro Extract${siteKey ? ` · ${siteKey}` : ''} — auto-fusing available references...`);
    try {
      const r = await fetch(`${API}/api/pro-extract`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.error) {
        addLog(d.error, 'error');
        setError(d.error);
      } else {
        setResults(d); setShowResults(true);
        if (d.bbox) {
          const ev = new CustomEvent('setRoi', { detail: d.bbox });
          window.dispatchEvent(ev);
        }
        const pro = d.pro || {};
        addLog(`Pro Extract done · refs: survey=${pro.n_insitu||0} lib=${pro.n_reference_library||0} ICESat2=${pro.n_icesat2||0} GEBCO=${pro.n_gebco||0} · R²=${d.ml_stats?.r2}`, 'success');
      }
    } catch (e) { setError(e.message); addLog(e.message, 'error'); }
    finally { setLoading(false); }
  }, [roi, params.start_date, params.end_date, userPoints, addLog]);

  // Upload user observed bathymetry (XYZ/SHP) — just loads, user runs Extract
  const uploadObserved = useCallback(async (files, opts={}) => {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    // Sidebar sends explicit coord-order / UTM / datum metadata so we never silently swap axes.
    fd.append('coord_order', opts.coord_order || 'latlon');
    if (opts.utm_epsg) fd.append('utm_epsg', String(opts.utm_epsg));
    fd.append('vertical_datum', opts.vertical_datum || 'LAT');
    fd.append('horizontal_datum', opts.horizontal_datum || 'WGS84');
    setLoading(true); setLoadingMsg('Parsing observed bathymetry...');
    addLog(`Uploading observed bathymetry (order=${opts.coord_order||'latlon'}${opts.utm_epsg?`, UTM EPSG:${opts.utm_epsg}`:''})...`);
    try {
      const r = await fetch(`${API}/api/upload-observed`, { method: 'POST', body: fd });
      let d;
      try { d = await r.json(); }
      catch (e) { addLog('Upload: server returned an unparseable response (non-JSON) — please retry or report this.', 'error'); setLoading(false); return; }
      if (d.error) { addLog(d.error, 'error'); setLoading(false); return; }
      if (d.bbox) {
        const ev = new CustomEvent('setRoi', { detail: d.bbox });
        window.dispatchEvent(ev);
      }
      setUserPoints(d.points || []);
      setCsvPreview(d.points || []);
      setValidationPts(d.points || []);
      setObservedData(d);
      setParams(p => ({...p, ref_source: 'fusion', img_source: 'mapbox'}));
      // GIS-R2 / ROB-R2: never present a plausibility failure as a plain
      // success — a wrong coord_order/UTM zone lands the survey on another
      // continent with no other visible symptom.
      if (d.in_expected_aoi === false) {
        addLog(`⚠ ${d.plausibility_warning || `Uploaded points centre at ${d.centroid?.lat}, ${d.centroid?.lon} — outside the expected UAE/Gulf area. Check coord_order / UTM zone before trusting this upload.`}`, 'error');
      } else {
        addLog(`Uploaded ${d.count} observed points. FUSION mode (Mapbox + GEBCO + ICESat-2 + In-situ). Draw ROI → Extract.`, 'success');
      }
      if (Array.isArray(d.warnings) && d.warnings.length) {
        d.warnings.forEach(w => addLog(`⚠ ${w}`, 'warn'));
      }
    } catch (e) { addLog(e.message, 'error'); }
    setLoading(false);
  }, [addLog]);

  const [satCapture, setSatCapture] = useState(null);
  // HR Satellite → run full bathymetry extraction with VHR Mapbox imagery
  const captureSatellite = useCallback(async () => {
    if (!roi) return;
    setLoading(true); setError(null);
    const steps = ['Fetching VHR satellite (~0.5m)...','Building water mask...','Fusing GEBCO + ICESat-2 + observed...','IDW + GBR regression...','Hillshade + contours...'];
    setLoadingTotal(steps.length); setLoadingStep(0); setLoadingMsg(steps[0]);
    addLog('HR Extract: VHR satellite + all reference sources...');
    let si=0; const iv=setInterval(()=>{si++;if(si<steps.length){setLoadingStep(si);setLoadingMsg(steps[si]);}},2000);
    try {
      const r = await fetch(`${API}/api/extract`, {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({bbox:roi,params:{...params,mode:activeModule,ref_source:'fusion',img_source:'mapbox_hr'},
          start_date:params.start_date,end_date:params.end_date,user_points:userPoints})});
      const d = await r.json(); clearInterval(iv);
      if (d.error && !d.interpolated_points?.length) throw new Error(d.error);
      const res = {...d, points:[...(d.points||[]),...(d.interpolated_points||[])]};
      setResults(res); setShowResults(true); setCsvPreview(null); setResultsB(null); setDiffResults(null);
      addLog(`HR Extract done! ${d.interpolated_points?.length||0} pts. Sources: ${(d.sources_used||[]).join(', ')}`,'success');
    } catch(e) { clearInterval(iv); setError(e.message); addLog(e.message,'error'); }
    finally { setLoading(false); }
  }, [roi, params, activeModule, addLog, userPoints]);

  // Analyse a finished bathy-job result on the map (R3). The store gives us
  // the ROI bbox + metrics; the GeoTIFF itself is downloadable. We surface
  // the metrics in the results panel and fit the map to the ROI (the
  // BathyResultsPanel already dispatches `zoomToRoi`).
  const analyseBathyResult = useCallback((meta) => {
    if (!meta) return;
    const m = meta.metrics || {};
    setResults({
      stats: {
        grid_points: m.n_test || 0,
        resolution_m: meta.resolution === 'vhr' ? 1 : 10,
        max_depth: 25,
      },
      metrics: m,
      ml_stats: (m.rmse_m != null || m.bias_m != null) ? {
        fusion_stats: { final_rmse: m.rmse_m, final_bias: m.bias_m },
      } : undefined,
      points: [],
      _bathy_result: { id: meta.id, name: meta.name, roi_bbox: meta.roi_bbox, resolution: meta.resolution, download_url: meta.download_url, download_all_url: meta.download_all_url || (meta.id ? `/api/results/${meta.id}/download-all.zip` : null) },
    });
    setShowResults(true);
    setShowBathyResults(false);
    addLog(`Loaded "${meta.name}" metrics onto the analysis panel (download the GeoTIFF for the full raster).`, 'info');
  }, [addLog]);

  // Auto-show the 4 DEFAULT regions on first load (DEFAULT_REGIONS_LOG.md task 1).
  // The Results panel surfaces the precomputed `is_default` entries WITHOUT the
  // user drawing an ROI. One-shot per browser session so it never nags after the
  // user has dismissed it once.
  useEffect(() => {
    let shown = false;
    try { shown = sessionStorage.getItem('defaultsAutoShown') === '1'; } catch (_) {}
    if (shown) return;
    const t = setTimeout(() => {
      setShowBathyResults(true);
      try { sessionStorage.setItem('defaultsAutoShown', '1'); } catch (_) {}
    }, 600);
    return () => clearTimeout(t);
  }, []);

  const reset = useCallback(() => {
    setRoi(null);setResults(null);setResultsB(null);setDiffResults(null);setShowResults(false);setError(null);setSearchResults(null);setLogs([]);setUserPoints([]);setCsvPreview(null);setValidationPts([]);setValidationResults(null);setAutoEvalResults(null);setDigitiseResults(null);setIcesat2Results(null);setIcesat2Granules([]);setSatCapture(null);setObservedData(null);
  }, []);

  return (
    <div style={{height:'100vh',display:'grid',gridTemplateRows:'56px 1fr 32px',gridTemplateColumns:showResults?'400px 1fr 420px':'400px 1fr',background:'var(--bg-base)',overflow:'hidden'}}>
      <Header view={view} setView={setView} hasResults={!!results} onReset={reset} activeModule={activeModule} setActiveModule={setMode} onOpenHelp={()=>setShowHelp(true)} onOpenResults={()=>setShowBathyResults(true)} onOpenVhrPurchase={()=>setShowVhrPurchase(true)} />
      <HelpModal open={showHelp} onClose={()=>setShowHelp(false)} />
      <VhrJobsModal
        open={showVhrJobsModal}
        onClose={()=>setShowVhrJobsModal(false)}
        jobs={vhrJobs}
        onUpload={vhrJobUpload}
        onCancel={vhrJobCancel}
        onDelete={vhrJobDelete}
      />
      <BathyResultsPanel
        open={showBathyResults}
        onClose={()=>setShowBathyResults(false)}
        API={API}
        roi={roi}
        addLog={addLog}
        onAnalyse={analyseBathyResult}
      />
      <VhrPurchasePanel
        open={showVhrPurchase}
        onClose={()=>setShowVhrPurchase(false)}
        API={API}
        roi={roi}
        addLog={addLog}
      />
      <Sidebar roi={roi} params={params} setParams={setParams} onExtract={extract} onExtractMosaic={extractMosaic} onSearch={searchTracks} loading={loading} error={error} results={results} searchResults={searchResults} onExport={exportData} activeModule={activeModule} logs={logs} onUploadCSV={uploadCSV} userPoints={userPoints} onUploadChart={uploadChart} onOpenChart={openChart}
        onExtractEpochB={extractEpochB} resultsB={resultsB} onComputeDiff={computeDiff} diffResults={diffResults}
        onUploadShapefile={uploadShapefile} validationPts={validationPts} onValidate={validateResults} validationResults={validationResults}
        onAutoEvaluate={autoEvaluate} autoEvalResults={autoEvalResults}
        onIBoatingPipeline={runIBoatingPipeline}
        onAccurateBathymetry={runAccurateBathymetry} onCnnV2={runCnnV2} onKhalifaTest={runKhalifaTest}
        availableModels={availableModels} onRefreshModels={refreshModels}
        onSmartDigitise={smartDigitise} digitiseResults={digitiseResults}
        onIcesat2Search={icesat2Search} onIcesat2Process={icesat2Process}
        icesat2Results={icesat2Results} icesat2Granules={icesat2Granules}
        onCaptureSatellite={captureSatellite} satCapture={satCapture}
        onLoadObserved={loadObserved} onUploadObserved={uploadObserved} observedData={observedData} onLoadObservedMeta={loadObservedMeta}
        onProExtract={proExtract} onVeryHrExtract={veryHrExtract}
        onClusteredExtract={clusteredExtract} onAlmar={runS2Shores} onVeryHrMle={veryHrMle}
        vhrJobs={vhrJobs} onVhrJobStart={vhrJobStart} onVhrJobUpload={vhrJobUpload} onVhrJobCancel={vhrJobCancel} onVhrJobDelete={vhrJobDelete}
        onVhrMlePro={vhrMlePro}
        onOpenVhrJobsModal={()=>setShowVhrJobsModal(true)}
        onShowOverlay={onShowOverlay}
        onQuickAnalyse={quickAnalyse}
        onDlPro={runDlPro}
        onSmartCnn={runSmartCnn}
        onBoaCnnBilstm={runBoaCnnBilstm}
        onCbr={runCbr}
        onS2Shores={runS2Shores}
        onMultiEpochMle={multiEpochMle} mleResults={mleResults} />
      <div style={{position:'relative',overflow:'hidden'}}>
        {view==='monitoring'&&<MonitoringPanel roi={roi} results={results} addLog={addLog} />}
        {view==='uae'&&<UaeResultsPanel />}
        {view==='ascii'&&<AsciiView results={results} />}
        {(view==='map'||view==='split')&&<MapPanel roi={roi} setRoi={setRoi} results={results} csvPreview={csvPreview} icesat2Results={icesat2Results} overlayOverride={mapOverlay} onShowOverlay={onShowOverlay} style={{height:view==='split'?'50%':'100%',width:'100%'}} />}
        {(view==='3d'||view==='split')&&results&&<div style={{height:view==='split'?'50%':'100%',width:'100%'}}><ThreeDView data={results} /></div>}
        {view==='3d'&&!results&&<div style={{height:'100%',display:'flex',alignItems:'center',justifyContent:'center',background:'var(--bg-secondary)'}}><p style={{fontFamily:'var(--font-mono)',fontSize:'13px',color:'var(--text-dim)'}}>Draw ROI → Extract → View 3D</p></div>}
        {loading&&<LoadingOverlay message={sanitizeClientText(loadingMsg)} step={loadingStep} total={loadingTotal} activeModule={activeModule} />}
        {/* High-res satellite capture popup */}
        {satCapture&&satCapture.image_b64&&(
          <div style={{position:'absolute',top:0,left:0,right:0,bottom:0,background:'rgba(0,0,0,0.85)',zIndex:500,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',padding:'20px'}}>
            <div style={{background:'var(--bg-secondary)',borderRadius:'12px',padding:'16px',maxWidth:'90%',maxHeight:'90%',overflow:'auto',border:'1px solid var(--border-dim)'}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'12px'}}>
                <div>
                  <h3 style={{fontFamily:'var(--font-mono)',fontSize:'13px',fontWeight:800,color:'#06b6d4',margin:0}}>HIGH-RES COMMERCIAL SATELLITE</h3>
                  <p style={{fontFamily:'var(--font-mono)',fontSize:'9px',color:'var(--text-dim)',margin:'4px 0 0'}}>
                    {satCapture.width}x{satCapture.height}px | {satCapture.resolution_m}m/px | {satCapture.size_km?.[0]}x{satCapture.size_km?.[1]}km
                  </p>
                </div>
                <button onClick={()=>setSatCapture(null)} style={{padding:'6px 12px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:700,borderRadius:'6px',border:'1px solid rgba(251,113,133,0.3)',background:'rgba(251,113,133,0.08)',color:'#fb7185',cursor:'pointer'}}>CLOSE</button>
              </div>
              <img src={`data:image/png;base64,${satCapture.image_b64}`} alt="Satellite" style={{maxWidth:'100%',borderRadius:'8px',border:'1px solid var(--border-dim)'}} />
              <div style={{marginTop:'8px',display:'grid',gridTemplateColumns:'1fr 1fr 1fr 1fr',gap:'8px',fontFamily:'var(--font-mono)',fontSize:'8px'}}>
                <div style={{padding:'6px',borderRadius:'4px',background:'var(--bg-primary)',textAlign:'center'}}>
                  <div style={{color:'var(--text-dim)'}}>RESOLUTION</div>
                  <div style={{fontSize:'12px',fontWeight:700,color:'#06b6d4'}}>{satCapture.resolution_m}m</div>
                </div>
                <div style={{padding:'6px',borderRadius:'4px',background:'var(--bg-primary)',textAlign:'center'}}>
                  <div style={{color:'var(--text-dim)'}}>SIZE</div>
                  <div style={{fontSize:'12px',fontWeight:700,color:'#10b981'}}>{satCapture.size_km?.[0]}x{satCapture.size_km?.[1]}km</div>
                </div>
                <div style={{padding:'6px',borderRadius:'4px',background:'var(--bg-primary)',textAlign:'center'}}>
                  <div style={{color:'var(--text-dim)'}}>PIXELS</div>
                  <div style={{fontSize:'12px',fontWeight:700,color:'#f59e0b'}}>{satCapture.width}x{satCapture.height}</div>
                </div>
                <div style={{padding:'6px',borderRadius:'4px',background:'var(--bg-primary)',textAlign:'center'}}>
                  <div style={{color:'var(--text-dim)'}}>CENTER</div>
                  <div style={{fontSize:'10px',fontWeight:600,color:'var(--text-primary)'}}>{satCapture.center?.[0].toFixed(3)}N {satCapture.center?.[1].toFixed(3)}E</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      {showResults&&results&&<ResultsPanel results={results} resultsB={resultsB} diffResults={diffResults} validationResults={validationResults} autoEvalResults={autoEvalResults} onClose={()=>setShowResults(false)} onExport={exportData} onShowOverlay={onShowOverlay} activeOverlayKey={mapOverlay?mapOverlay.key:'composite'} />}
      {/* Floating "Open Results" button when results exist but panel is closed */}
      {!showResults && results && (
        <button
          onClick={() => setShowResults(true)}
          title="Reopen the advanced analysis results (fullscreen view available inside)"
          style={{
            position: 'fixed', bottom: 52, right: 20, zIndex: 9999,
            padding: '12px 18px', fontSize: 12, fontWeight: 800,
            fontFamily: 'var(--font-mono)', letterSpacing: '0.06em',
            borderRadius: 999, border: 'none', cursor: 'pointer',
            background: 'linear-gradient(135deg,#0ea5e9,#2563eb)',
            color: '#fff', boxShadow: '0 8px 24px rgba(37,99,235,0.45)',
            display: 'flex', alignItems: 'center', gap: 8,
          }}
        >
          📊 SHOW RESULTS
          <span style={{ fontSize: 10, opacity: 0.85, background: 'rgba(255,255,255,0.15)', padding: '2px 8px', borderRadius: 999 }}>
            {(results?.stats?.grid_points || results?.interpolated_points?.length || 0).toLocaleString()} pts
          </span>
        </button>
      )}
      <StatusBar roi={roi} results={results} loading={loading} activeModule={activeModule} lastLog={logs[logs.length-1]} />
    </div>
  );
}
export default App;
