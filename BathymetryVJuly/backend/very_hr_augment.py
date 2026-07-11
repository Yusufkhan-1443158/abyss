"""Band-wise reference-data augmentation for Clustered SDB.

Philosophy
----------
Slice the depth range into 2 m bands (0-2, 2-4, …, 24-26 m).  For each band:

1. Use in-situ measurements first (``photon_class='observed'``).
2. If a band has fewer than ``min_per_band`` points, top-up from
   **i-Boating** chart digitisation (only the soundings whose depth lands in
   that band are kept).
3. If still under-populated, fall back to **GEBCO** (cropped to the band).

Augmented points are tagged with a source label and a sample-weight:
``insitu = 6.0`` ≫ ``iboating = 2.0`` > ``gebco = 1.0``.  Downstream the
HistGradientBoostingRegressor uses these weights so high-confidence in-situ
observations dominate where available.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

L = logging.getLogger("very_hr_augment")
if not L.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
    L.addHandler(h)
    L.setLevel(logging.INFO)

W_INSITU = 6.0
W_SLIDERULE = 5.0      # ICESat-2 ATL03 bathymetric photons via SlideRule
W_IBOATING = 2.0
W_GEBCO = 1.0
SLIDERULE_MIN_DEPTH_M = 5.0  # exclude SlideRule for very-shallow bands (refraction noise)
# ATL24 class_ph: 40 = bathymetry, 41 = sea surface (harvest_r2_atl24.py CORRECTION,
# 2026-07-03). Shared constant so this and harvest_r2_atl24.py cannot drift again.
ATL24_CLASS_BATHY = 40


def band_index(depth: np.ndarray, band_w: float = 2.0) -> np.ndarray:
    return np.clip((depth / band_w).astype(int), 0, int(round(26.0 / band_w)) - 1)


def band_histogram(depths: np.ndarray, band_w: float = 2.0,
                   max_d: float = 26.0) -> Dict[Tuple[float, float], int]:
    bins = np.arange(0, max_d + band_w / 2, band_w)
    h, _ = np.histogram(depths, bins=bins)
    return {(float(bins[i]), float(bins[i + 1])): int(h[i]) for i in range(len(h))}


def _filter_to_band(lats: np.ndarray, lons: np.ndarray, deps: np.ndarray,
                    bbox: List[float], lo: float, hi: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, s, e, n = bbox
    m = (deps >= lo) & (deps < hi) & (lats >= s) & (lats <= n) & (lons >= w) & (lons <= e)
    return lats[m], lons[m], deps[m]


# ════════════════════════════════════════════════════════════════════════
# i-Boating augmentation — uses backend.iboating.run_iboating_pipeline if
# Playwright + Gemini are available; cached on disk per bbox+zoom.
# ════════════════════════════════════════════════════════════════════════
def fetch_iboating_points(bbox: List[float],
                           cache_dir: Optional[Path] = None,
                           zoom: int = 14,
                           min_confidence: float = 0.5,
                           ) -> Optional[Dict[str, np.ndarray]]:
    """Capture the i-Boating chart for ``bbox`` and extract every digitised
    sounding (single-shot, no S2 fusion).  Returns dict with lats/lons/depths
    or ``None`` on failure.
    """
    cache_dir = Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parent.parent / "cache" / "iboating_aug")
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    import json
    key = hashlib.md5(json.dumps([bbox, zoom], sort_keys=True).encode()).hexdigest()[:12]
    cache_path = cache_dir / f"iboating_{key}_c{int(min_confidence*100):03d}.npz"
    if cache_path.exists():
        try:
            d = np.load(cache_path)
            L.info(f"i-Boating cache hit  {cache_path.name}  "
                   f"({len(d['depths'])} pts)")
            return {"lats": d["lats"], "lons": d["lons"], "depths": d["depths"]}
        except Exception as ex:
            L.warning(f"i-Boating cache read failed: {ex}")

    have_groq = bool(os.getenv("GROQ_API_KEY"))
    have_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if not (have_groq or have_gemini):
        L.warning("i-Boating skipped: no GROQ_API_KEY / GEMINI_API_KEY set")
        return None
    try:
        try:
            from playwright.sync_api import sync_playwright as _ck  # noqa: F401
        except Exception as ex:
            L.warning(f"i-Boating skipped: playwright not importable ({ex})")
            return None
        try:
            from backend.iboating import _multi_zoom_capture, _pixels_to_geo
        except ImportError:
            from iboating import _multi_zoom_capture, _pixels_to_geo  # type: ignore
        # _multi_zoom_capture → [(path, b64, w, h, z), ...]
        captures = _multi_zoom_capture(bbox, zooms=(zoom, max(12, zoom - 2)))
        if not captures:
            L.warning("i-Boating capture returned nothing")
            return None
        L.info(f"i-Boating: {len(captures)} chart capture(s) ready")

        all_sounds: List[Dict] = []
        for path, b64, img_w, img_h, z in captures:
            tier_label = "?"; pts = None; err = None

            # Tier A — Groq vision (fast, multimodal Llama-4)
            if have_groq:
                try:
                    try:
                        from backend.iboating_groq import extract_depths_from_chart_groq
                    except ImportError:
                        from iboating_groq import extract_depths_from_chart_groq  # type: ignore
                    pts, err = extract_depths_from_chart_groq(
                        b64, bbox, img_w, img_h,
                        min_confidence=float(min_confidence))
                    tier_label = "Groq"
                except Exception as ex:
                    L.warning(f"Groq backend errored on z{z}: {ex}")
                    pts = None
                    err = str(ex)

            # Tier B — Gemini fallback
            if not pts and have_gemini:
                try:
                    try:
                        from backend.iboating import _extract_depths_from_chart
                    except ImportError:
                        from iboating import _extract_depths_from_chart  # type: ignore
                    res = _extract_depths_from_chart(b64, bbox, img_w, img_h)
                    if isinstance(res, tuple):
                        pts, err = res
                    else:
                        pts, err = res, None
                    tier_label = "Gemini" if pts else tier_label
                except Exception as ex:
                    L.warning(f"Gemini backend errored on z{z}: {ex}")
                    err = str(ex)

            if pts:
                geo = _pixels_to_geo(pts, bbox, img_w, img_h)
                L.info(f"  z{z}: {len(geo)} pts via {tier_label}")
                all_sounds.extend(geo)
            else:
                L.warning(f"  z{z}: 0 pts ({err or 'unknown'})")

        soundings = all_sounds

        if not soundings:
            L.warning("i-Boating extraction yielded 0 soundings (all backends)")
            return None
        lats = np.array([float(s["lat"]) for s in soundings])
        lons = np.array([float(s["lon"]) for s in soundings])
        deps = np.array([abs(float(s["depth"])) for s in soundings])
        np.savez_compressed(cache_path, lats=lats, lons=lons, depths=deps)
        L.info(f"i-Boating: {len(deps)} soundings cached → {cache_path.name}")
        return {"lats": lats, "lons": lons, "depths": deps}
    except Exception as ex:
        L.warning(f"i-Boating fetch failed: {ex}")
        return None


# ════════════════════════════════════════════════════════════════════════
# SlideRule (ICESat-2 ATL03) augmentation
# ════════════════════════════════════════════════════════════════════════
def fetch_sliderule_points(bbox: List[float],
                            start_date: str = "2020-01-01",
                            end_date: str = "2025-12-31",
                            cache_dir: Optional[Path] = None,
                            high_confidence: bool = False,
                            min_seafloor_depth_m: float = 0.0,
                            ) -> Optional[Dict[str, np.ndarray]]:
    """Run the project's SlideRule wrapper to grab ATL03 bathymetry photons.

    Returns ``{lats, lons, depths}`` or ``None`` when SlideRule is unavailable.
    Cached on disk per (bbox, dates, filters).

    ``min_seafloor_depth_m`` (v6 ATL24-proxy) drops photons shallower than
    the threshold from the returned set.  This is used to suppress
    air-water-interface contamination that the in-house YAPC + KDE
    sea-surface detector leaves behind: typically 20-30 % of
    "bathymetry"-classified photons in the 0-2 m bin are actually surface
    returns or refraction-correction residuals at the air-water boundary.
    NASA's ATL24 ensemble model handles this natively; this is the
    quick-and-dirty proxy for sites/clients without ATL24 access.
    """
    cache_dir = Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parent.parent / "cache" / "sliderule_aug")
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    import json
    key = hashlib.md5(json.dumps([bbox, start_date, end_date,
                                    bool(high_confidence),
                                    float(min_seafloor_depth_m)],
                                   sort_keys=True).encode()).hexdigest()[:12]
    cache_path = cache_dir / f"sliderule_{key}.npz"
    if cache_path.exists():
        try:
            d = np.load(cache_path)
            L.info(f"SlideRule cache hit  {cache_path.name}  ({len(d['depths'])} pts)")
            return {"lats": d["lats"], "lons": d["lons"], "depths": d["depths"]}
        except Exception as ex:
            L.warning(f"SlideRule cache read failed: {ex}")

    try:
        try:
            from backend.app import run_sliderule
        except ImportError:
            from app import run_sliderule  # type: ignore
        ice, msg = run_sliderule(list(bbox), start_date, end_date)
        if not ice:
            L.warning(f"SlideRule returned no points: {msg}")
            return None
        bathy = [p for p in ice if p.get("photon_class") == "bathymetry"
                 and float(p.get("depth", 0)) > 0]
        if not bathy:
            L.warning("SlideRule: 0 bathymetry photons in window")
            return None
        lats = np.array([float(p["lat"]) for p in bathy])
        lons = np.array([float(p["lon"]) for p in bathy])
        deps = np.array([abs(float(p["depth"])) for p in bathy])
        # Strip the cap pile-up: depths within 0.3 m of MAX_DEPTH_M are
        # almost always clipped values, not real seafloor.
        cap_mask = deps >= 24.7
        n_cap = int(cap_mask.sum())
        if n_cap and n_cap > max(5, 0.05 * len(deps)):
            keep = ~cap_mask
            L.info(f"SlideRule: dropping {n_cap} suspected cap-pile-up photons "
                   f"(depth>=24.7 m, {100*n_cap/len(deps):.1f}%)")
            lats, lons, deps = lats[keep], lons[keep], deps[keep]
        if high_confidence:
            # Stricter trim: drop the deepest 5% (typically refraction outliers
            # at depth) and the shallowest 5% (sub-pixel surface ambiguity).
            if len(deps) >= 40:
                lo_q = float(np.quantile(deps, 0.05))
                hi_q = float(np.quantile(deps, 0.95))
                keep = (deps >= lo_q) & (deps <= hi_q)
                L.info(f"SlideRule HC: keeping inner 5–95% quantile "
                       f"(depth window {lo_q:.2f}–{hi_q:.2f} m, "
                       f"{int(keep.sum())}/{len(deps)})")
                lats, lons, deps = lats[keep], lons[keep], deps[keep]
        if min_seafloor_depth_m > 0 and len(deps) > 0:
            # ATL24-proxy: drop suspected air-water-interface photons.  The
            # in-house YAPC + KDE classifier leaves a tall density peak in
            # the first ~1-2 m of "bathymetry" photons; these are mostly
            # refraction-correction residuals at the surface.  NASA's ATL24
            # ensemble drops them via its trained classifier; we just cut
            # them by depth threshold.
            #
            # ``min_seafloor_depth_m`` semantics:
            #   > 0  : hard floor (legacy behaviour — universal threshold)
            #   < 0  : adaptive — find the photon-density local minimum
            #          between the surface peak and the seafloor distribution
            #          (per-site, capped at |value| as the maximum threshold).
            if min_seafloor_depth_m < 0 and len(deps) >= 200:
                cap = -float(min_seafloor_depth_m)
                # Fine 0.1 m bin histogram, smoothed
                edges = np.arange(0.0, max(deps.max() + 0.2, cap + 0.5), 0.1)
                hist, _ = np.histogram(deps, bins=edges)
                # 7-tap triangular smoother
                kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
                kernel /= kernel.sum()
                sm = np.convolve(hist.astype(np.float64), kernel, mode="same")
                centres = (edges[:-1] + edges[1:]) / 2
                # Find the surface peak (highest density in 0-1.5 m)
                in_surface = centres < 1.5
                if in_surface.sum() > 5 and sm[in_surface].max() > 0:
                    surf_idx = int(np.argmax(sm * in_surface))
                    # Walk right from surface peak; find first local minimum
                    # before reaching the cap
                    end_idx = int(np.searchsorted(centres, cap, side="right"))
                    end_idx = min(end_idx, len(sm) - 2)
                    floor = None
                    for i in range(surf_idx + 2, end_idx):
                        if sm[i] < sm[i - 1] and sm[i] <= sm[i + 1]:
                            # local min — and it must be *substantially* lower
                            # than the surface peak (< 60% of peak)
                            if sm[i] < 0.6 * sm[surf_idx]:
                                floor = float(centres[i])
                                break
                    if floor is None:
                        floor = float(min(cap, centres[surf_idx] + 0.8))
                    threshold = floor
                    L.info(f"SlideRule ATL24-proxy [adaptive]: surface peak at "
                           f"{centres[surf_idx]:.2f} m; density-min floor at "
                           f"{floor:.2f} m (cap={cap:.2f} m)")
                else:
                    threshold = 0.5  # fallback
                    L.info(f"SlideRule ATL24-proxy [adaptive]: no clear surface "
                           f"peak; falling back to {threshold:.2f} m floor")
            else:
                threshold = float(min_seafloor_depth_m)
            keep = deps >= threshold
            n_drop = int((~keep).sum())
            if n_drop:
                L.info(f"SlideRule ATL24-proxy: dropping {n_drop} photons "
                       f"with depth < {threshold:.2f} m "
                       f"({100*n_drop/len(deps):.1f}% of pool — suspected "
                       f"air-water-interface contamination)")
                lats, lons, deps = lats[keep], lons[keep], deps[keep]
        np.savez_compressed(cache_path, lats=lats, lons=lons, depths=deps)
        L.info(f"SlideRule: {len(deps)} ATL03 bathy photons cached → {cache_path.name}")
        return {"lats": lats, "lons": lons, "depths": deps}
    except Exception as ex:
        L.warning(f"SlideRule fetch failed: {ex}")
        return None


# ════════════════════════════════════════════════════════════════════════
# ATL24 — official NASA bathymetric-photon product (Parrish/Magruder 2025)
# ════════════════════════════════════════════════════════════════════════
def fetch_atl24_points(bbox: List[float],
                        start_date: str = "2019-07-01",
                        end_date: str = "2025-12-31",
                        cache_dir: Optional[Path] = None,
                        max_granules: int = 0,
                        min_confidence: float = 0.0,
                        ) -> Optional[Dict[str, np.ndarray]]:
    """Fetch ATL24 class=bathymetry photons over ``bbox`` via NASA Earthdata.

    ATL24 (Parrish et al. 2025, doi:10.1029/2025EA004391; Magruder et al.
    2025, doi:10.1029/2025EA004390) is NASA's ensemble-classified,
    refraction-corrected (n applied in-product) bathymetric-photon product.
    Per-photon orthometric seafloor height ``ortho_h`` and modelled water
    ``surface_h`` are already refraction-corrected, so depth is simply
    ``surface_h - ortho_h`` — no manual n=1.34 step and no 24.7 m pile-up
    cap (both were artefacts of the in-house ATL03 + YAPC + histogram path
    in ``backend.app.run_sliderule``).

    We pull granules directly from Earthdata (the project ``~/.netrc`` already
    holds urs.earthdata.nasa.gov creds) rather than via SlideRule because the
    pinned SlideRule client (v5.3.2) does not expose the ``atl24g`` wrapper
    and the deployed server's ``atl24g.lua`` orchestration script fails to
    load (HTTP 500); the raw ``atl24x`` endpoint returns 0 photons without
    that orchestration.  Reading the standard product directly is the robust,
    honest path and yields identical photons.

    Keeps class==40 (bathymetry) photons, drops ``low_confidence_flag`` and
    ``sensor_depth_exceeded``, applies an optional ``min_confidence`` cut.
    Returns ``{lats, lons, depths, sigma_tvu}`` or ``None`` on failure.
    Cached on disk per (bbox, dates, filters).
    """
    cache_dir = Path(cache_dir) if cache_dir else (
        Path(__file__).resolve().parent.parent / "cache" / "sliderule_aug")
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    import json
    key = hashlib.md5(json.dumps([bbox, start_date, end_date,
                                   int(max_granules), float(min_confidence),
                                   "v2class40"],
                                  sort_keys=True).encode()).hexdigest()[:12]
    cache_path = cache_dir / f"atl24_{key}.npz"
    if cache_path.exists():
        try:
            d = np.load(cache_path)
            L.info(f"ATL24 cache hit  {cache_path.name}  ({len(d['depths'])} bathy photons)")
            return {"lats": d["lats"], "lons": d["lons"], "depths": d["depths"],
                    "sigma_tvu": d["sigma_tvu"] if "sigma_tvu" in d else
                    np.full(len(d["depths"]), np.nan)}
        except Exception as ex:
            L.warning(f"ATL24 cache read failed: {ex}")

    try:
        import h5py
    except Exception as ex:
        L.warning(f"ATL24: h5py unavailable ({ex})")
        return None

    w, s, e, n = bbox
    dl_dir = cache_dir.parent / "atl24"
    dl_dir.mkdir(parents=True, exist_ok=True)

    # ── Granule acquisition ────────────────────────────────────────────────
    # ROOT CAUSE of the round-2 hang: ``earthaccess.download`` re-validates and
    # re-fetches FULL ATL24 .h5 granules (~5.4 GB) with NO network timeout, so
    # a stalled NSIDC/cloud connection blocks the whole pipeline indefinitely
    # (process alive at 1% CPU, no progress).  Fix: if granules already exist
    # locally, consume them directly and NEVER touch the network.  Also bound
    # any real download with a per-request timeout and a granule cap.
    #   ATL24_LOCAL_ONLY=1  → never hit the network; read cache/atl24/*.h5 only.
    #   ATL24_NET_TIMEOUT_S → per-request network timeout (default 120 s).
    import os as _os
    local_only = _os.environ.get("ATL24_LOCAL_ONLY", "0") == "1"
    net_timeout = float(_os.environ.get("ATL24_NET_TIMEOUT_S", "120"))
    cached_h5 = sorted(dl_dir.glob("ATL24_*.h5"))

    paths: list = []
    if local_only or cached_h5:
        # Consume the granules already on disk — no search, no download.
        paths = [str(p) for p in cached_h5]
        L.info(f"ATL24: consuming {len(paths)} cached granule(s) in {dl_dir} "
               f"(local_only={local_only}); skipping network fetch")
        if not paths and local_only:
            L.warning("ATL24: ATL24_LOCAL_ONLY=1 but no cached .h5 granules found")
            return None
    if not paths:
        try:
            import earthaccess
        except Exception as ex:
            L.warning(f"ATL24: earthaccess unavailable and no cache ({ex})")
            return None
        try:
            auth = earthaccess.login(strategy="netrc")
            if not getattr(auth, "authenticated", False):
                L.warning("ATL24: Earthdata auth failed (.netrc)")
                return None
            results = earthaccess.search_data(
                short_name="ATL24",
                bounding_box=(float(w), float(s), float(e), float(n)),
                temporal=(start_date, end_date))
            if not results:
                L.warning("ATL24: 0 granules from CMR over bbox")
                return None
            if max_granules and len(results) > max_granules:
                results = results[:max_granules]
            L.info(f"ATL24: {len(results)} granules; downloading to {dl_dir} "
                   f"(timeout {net_timeout:.0f}s/req)")
            # Bound the network call with a hard wall-clock timeout so a stalled
            # connection can never hang the pipeline again.
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                fut = _ex.submit(earthaccess.download, results, str(dl_dir))
                try:
                    paths = fut.result(timeout=net_timeout * max(1, len(results)))
                except _cf.TimeoutError:
                    L.warning(f"ATL24: download timed out after "
                              f"{net_timeout * max(1, len(results)):.0f}s; "
                              f"falling back to whatever landed in {dl_dir}")
                    paths = [str(p) for p in sorted(dl_dir.glob('ATL24_*.h5'))]
            if not paths:
                L.warning("ATL24: download yielded no granules")
                return None
        except Exception as ex:
            L.warning(f"ATL24 acquisition failed: {ex}")
            paths = [str(p) for p in sorted(dl_dir.glob('ATL24_*.h5'))]
            if not paths:
                return None

    try:

        all_lat, all_lon, all_dep, all_sig = [], [], [], []
        n_files = 0
        for p in paths:
            try:
                with h5py.File(p, "r") as hf:
                    beams = [k for k in hf.keys() if k.startswith("gt")]
                    for bm in beams:
                        g = hf[bm]
                        if "class_ph" not in g:
                            continue
                        cls = g["class_ph"][:]
                        bathy = cls == ATL24_CLASS_BATHY
                        if not bathy.any():
                            continue
                        lat = g["lat_ph"][bathy].astype(np.float64)
                        lon = g["lon_ph"][bathy].astype(np.float64)
                        # depth = modelled water surface − seafloor ortho_h
                        surf = g["surface_h"][bathy].astype(np.float64)
                        sea = g["ortho_h"][bathy].astype(np.float64)
                        dep = surf - sea
                        sig = (g["sigma_tvu"][bathy].astype(np.float64)
                               if "sigma_tvu" in g else np.full(len(dep), np.nan))
                        # QC: clip to bbox, drop flagged photons
                        ok = ((lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
                              & np.isfinite(dep) & (dep > 0.0))
                        if "low_confidence_flag" in g:
                            ok &= (g["low_confidence_flag"][bathy] == 0)
                        if "sensor_depth_exceeded" in g:
                            ok &= (g["sensor_depth_exceeded"][bathy] == 0)
                        if min_confidence > 0 and "confidence" in g:
                            ok &= (g["confidence"][bathy].astype(np.float64)
                                   >= float(min_confidence))
                        if ok.any():
                            all_lat.append(lat[ok]); all_lon.append(lon[ok])
                            all_dep.append(dep[ok]); all_sig.append(sig[ok])
                n_files += 1
            except Exception as ex:
                L.warning(f"ATL24: failed reading {Path(p).name}: {ex}")

        if not all_dep:
            L.warning("ATL24: 0 bathymetry photons after QC over bbox")
            return None
        lats = np.concatenate(all_lat)
        lons = np.concatenate(all_lon)
        deps = np.concatenate(all_dep)
        sigs = np.concatenate(all_sig)
        n_deep = int((deps > 14.0).sum())
        L.info(f"ATL24: {len(deps)} bathy photons from {n_files} granules "
               f"({n_deep} deeper than 14 m); depth "
               f"{deps.min():.2f}–{deps.max():.2f} m (median {np.median(deps):.2f})")
        np.savez_compressed(cache_path, lats=lats, lons=lons, depths=deps,
                            sigma_tvu=sigs)
        return {"lats": lats, "lons": lons, "depths": deps, "sigma_tvu": sigs}
    except Exception as ex:
        L.warning(f"ATL24 fetch failed: {ex}")
        return None


# ════════════════════════════════════════════════════════════════════════
# GEBCO augmentation — leverage backend.app.fetch_gebco
# ════════════════════════════════════════════════════════════════════════
def fetch_gebco_points(bbox: List[float]) -> Optional[Dict[str, np.ndarray]]:
    try:
        try:
            from backend.app import fetch_gebco
        except ImportError:
            from app import fetch_gebco  # type: ignore
        g = fetch_gebco(bbox)
        if not g or len(g.get("depths", [])) == 0:
            return None
        L.info(f"GEBCO: {len(g['depths'])} pts retrieved")
        return {"lats": np.asarray(g["lats"], dtype=np.float64),
                "lons": np.asarray(g["lons"], dtype=np.float64),
                "depths": np.asarray(g["depths"], dtype=np.float64)}
    except Exception as ex:
        L.warning(f"GEBCO fetch failed: {ex}")
        return None


# ════════════════════════════════════════════════════════════════════════
# Top-level augmenter
# ════════════════════════════════════════════════════════════════════════
def augment_in_situ(
    lats: np.ndarray, lons: np.ndarray, deps: np.ndarray,
    bbox: List[float],
    band_w: float = 2.0,
    max_d: float = 26.0,
    min_per_band: int = 80,
    use_iboating: bool = True,
    use_sliderule: bool = True,
    use_gebco: bool = True,
    sliderule_start: str = "2020-01-01",
    sliderule_end: str = "2025-12-31",
    min_aug_confidence: float = 0.5,
    high_confidence_only: bool = False,
) -> Dict:
    """Returns dict with combined arrays + sample weights + per-source counts.

    Output keys: ``lats, lons, depths, weights, sources`` (sources is a numpy
    array of strings from {'insitu', 'iboating', 'gebco'}).
    """
    in_lats = np.asarray(lats, dtype=np.float64)
    in_lons = np.asarray(lons, dtype=np.float64)
    in_deps = np.clip(np.asarray(deps, dtype=np.float64), 0, max_d)

    bins = np.arange(0, max_d + band_w / 2, band_w)
    n_bands = len(bins) - 1
    in_hist, _ = np.histogram(in_deps, bins=bins)
    L.info(f"In-situ band histogram (band_w={band_w} m, threshold={min_per_band}):")
    for i in range(n_bands):
        L.info(f"  {bins[i]:>4.1f}-{bins[i+1]:<4.1f} m: {in_hist[i]:>5,} "
               f"{'OK' if in_hist[i] >= min_per_band else 'SPARSE → augment'}")

    # Decide which bands need augmentation
    need = in_hist < min_per_band
    sparse_bands = [(float(bins[i]), float(bins[i + 1])) for i in range(n_bands) if need[i]]
    deficit = {b: int(min_per_band - in_hist[i]) for i, b in
               enumerate([(float(bins[k]), float(bins[k + 1])) for k in range(n_bands)])
               if need[i]}

    # Tier 2 sources (parallel: i-Boating + SlideRule)
    iboat: Optional[Dict] = None
    if sparse_bands and use_iboating:
        iboat = fetch_iboating_points(bbox, min_confidence=min_aug_confidence)
    slr: Optional[Dict] = None
    if sparse_bands and use_sliderule:
        slr = fetch_sliderule_points(bbox, sliderule_start, sliderule_end,
                                       high_confidence=high_confidence_only)
    # Tier 3 (4th in user nomenclature): GEBCO  — disabled in HC mode
    gebco: Optional[Dict] = None
    if sparse_bands and use_gebco and not high_confidence_only:
        gebco = fetch_gebco_points(bbox)

    aug_lats: List[np.ndarray] = []
    aug_lons: List[np.ndarray] = []
    aug_deps: List[np.ndarray] = []
    aug_wts: List[np.ndarray] = []
    aug_src: List[np.ndarray] = []
    counts = {"insitu": int(len(in_deps)), "iboating": 0, "sliderule": 0, "gebco": 0}

    rng = np.random.default_rng(42)

    def _take(la_, lo_, de_, src_label, weight, n_max):
        if len(de_) > n_max:
            sel = rng.choice(len(de_), n_max, replace=False)
            la_, lo_, de_ = la_[sel], lo_[sel], de_[sel]
        aug_lats.append(la_); aug_lons.append(lo_); aug_deps.append(de_)
        aug_wts.append(np.full_like(de_, weight, dtype=np.float64))
        aug_src.append(np.full(len(de_), src_label, dtype=object))
        counts[src_label] += int(len(de_))
        return int(len(de_))

    for (lo, hi) in sparse_bands:
        need_n = deficit[(lo, hi)]
        log_parts = []

        # 2a) SlideRule first within tier 2 (higher accuracy), but only when
        #     band centre ≥ SLIDERULE_MIN_DEPTH_M — skip for very shallow.
        band_centre = 0.5 * (lo + hi)
        if slr is not None and need_n > 0 and band_centre >= SLIDERULE_MIN_DEPTH_M:
            la, lo_, de = _filter_to_band(slr["lats"], slr["lons"],
                                           slr["depths"], bbox, lo, hi)
            if len(de) > 0:
                added = _take(la, lo_, de, "sliderule", W_SLIDERULE, need_n)
                need_n = max(0, need_n - added)
                log_parts.append(f"+{added} SlideRule")

        # 2b) i-Boating (fills the rest of tier 2)
        if iboat is not None and need_n > 0:
            la, lo_, de = _filter_to_band(iboat["lats"], iboat["lons"],
                                           iboat["depths"], bbox, lo, hi)
            if len(de) > 0:
                added = _take(la, lo_, de, "iboating", W_IBOATING, need_n)
                need_n = max(0, need_n - added)
                log_parts.append(f"+{added} i-Boating")

        # 4) GEBCO last
        if gebco is not None and need_n > 0:
            la, lo_, de = _filter_to_band(gebco["lats"], gebco["lons"],
                                           gebco["depths"], bbox, lo, hi)
            if len(de) > 0:
                added = _take(la, lo_, de, "gebco", W_GEBCO, need_n)
                log_parts.append(f"+{added} GEBCO")

        if not log_parts:
            log_parts.append("(no augmentation source returned data)")
        L.info(f"  band {lo:>4.1f}-{hi:<4.1f} m  (deficit {deficit[(lo,hi)]:>3}): "
               + ", ".join(log_parts))

    # Combine
    all_lats = np.concatenate([in_lats] + aug_lats) if aug_lats else in_lats
    all_lons = np.concatenate([in_lons] + aug_lons) if aug_lons else in_lons
    all_deps = np.concatenate([in_deps] + aug_deps) if aug_deps else in_deps
    all_wts = np.concatenate(
        [np.full(len(in_deps), W_INSITU, dtype=np.float64)] + aug_wts
    ) if aug_wts else np.full(len(in_deps), W_INSITU, dtype=np.float64)
    all_src = np.concatenate(
        [np.full(len(in_deps), "insitu", dtype=object)] + aug_src
    ) if aug_src else np.full(len(in_deps), "insitu", dtype=object)

    # Final histogram for the report
    final_hist, _ = np.histogram(all_deps, bins=bins)
    band_summary = []
    for i in range(n_bands):
        band_summary.append({
            "lo": float(bins[i]), "hi": float(bins[i + 1]),
            "n_insitu": int(in_hist[i]),
            "n_total": int(final_hist[i]),
            "augmented": int(final_hist[i] - in_hist[i]),
        })
    L.info(
        f"Augmentation complete: in-situ {counts['insitu']:,}  "
        f"+ SlideRule {counts['sliderule']:,}  + i-Boating {counts['iboating']:,}  "
        f"+ GEBCO {counts['gebco']:,}  = total {len(all_deps):,}"
    )
    return {
        "lats": all_lats, "lons": all_lons, "depths": all_deps,
        "weights": all_wts.astype(np.float64),
        "sources": all_src,
        "counts": counts,
        "per_band": band_summary,
        "min_per_band": min_per_band,
        "band_w": band_w,
    }
