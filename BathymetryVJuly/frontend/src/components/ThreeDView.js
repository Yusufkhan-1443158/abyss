import React, { useMemo, useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';
import { rampRGB, rampCss } from './depthRamp';

/**
 * Realistic 3D bathymetric viewer.
 *
 *   • Seabed mesh: regular grid, vertices displaced by depth, smooth-shaded
 *     PBR surface with a per-vertex bathymetric color gradient.
 *   • Water surface: semi-transparent plane at z = 0 with subtle animated
 *     waves and a tiled normal-map for refraction-style sparkle.
 *   • Lighting: directional sun + ambient hemisphere, contact shadows.
 *   • Orbit controls (drag to rotate, wheel to zoom, right-click to pan).
 *
 * Replaces the previous canvas2D implementation. Uses three.js via
 * @react-three/fiber + @react-three/drei (already in package.json).
 */

const SCENE_SPAN = 100;        // metres in scene units, both X & Y
const GRID_RES = 64;           // grid resolution per side (was 96 — too heavy)
const WATER_RES = 32;          // water-surface plane resolution (was 80)
const MAX_DEPTH_FALLBACK = 25; // metres if stats.max_depth missing

// Bathymetric color ramp — shared perceptually-uniform ramp (depthRamp.js)
// so the 3D seabed matches the 2D colorbar and map overlay exactly.
// t in [0, 1] (0 = surface, 1 = max depth).
const depthColor = (t) => rampRGB(t);

// Build a (GRID_RES × GRID_RES) depth grid from the input points + a bbox.
// Returns { grid, hasData, maxD } in (lat-row × lon-col) order.
function buildDepthGrid(points, bbox, maxDepth) {
  const G = GRID_RES;
  const grid = new Float32Array(G * G);
  const cnt = new Int32Array(G * G);
  grid.fill(NaN);

  if (!points || !points.length) return { grid, hasData: false, maxD: maxDepth };

  // Derive bbox from points if not given
  let west, east, south, north;
  if (bbox && bbox.west !== undefined) {
    west = bbox.west; east = bbox.east; south = bbox.south; north = bbox.north;
  } else {
    let mnLa = +Infinity, mxLa = -Infinity, mnLo = +Infinity, mxLo = -Infinity;
    for (const p of points) {
      if (p.lat < mnLa) mnLa = p.lat;
      if (p.lat > mxLa) mxLa = p.lat;
      if (p.lon < mnLo) mnLo = p.lon;
      if (p.lon > mxLo) mxLo = p.lon;
    }
    west = mnLo; east = mxLo; south = mnLa; north = mxLa;
  }
  const lonSpan = east - west || 1e-6;
  const latSpan = north - south || 1e-6;

  for (const p of points) {
    if (!Number.isFinite(p.depth) || p.depth <= 0.05) continue;
    if (p.photon_class && !['interpolated', 'bathymetry', 'gebco', 'observed'].includes(p.photon_class)) continue;
    const c = Math.floor((p.lon - west) / lonSpan * (G - 1));
    const r = Math.floor((north - p.lat) / latSpan * (G - 1));
    if (r < 0 || r >= G || c < 0 || c >= G) continue;
    const k = r * G + c;
    if (cnt[k] === 0) { grid[k] = p.depth; cnt[k] = 1; }
    else { grid[k] = (grid[k] * cnt[k] + p.depth) / (cnt[k] + 1); cnt[k]++; }
  }

  // Fill empty cells by 8-neighbour averaging — but ONLY 2 passes.
  // Any cell more than 2 grid steps from an actual depth sample stays
  // NaN, so the Seabed mesh treats it as land (terrain-style grey)
  // instead of bleeding water depth onto buildings, piers and inland
  // pixels along the coastline.
  for (let pass = 0; pass < 2; pass++) {
    const next = grid.slice();
    let filled = 0;
    for (let r = 0; r < G; r++) {
      for (let c = 0; c < G; c++) {
        const k = r * G + c;
        if (Number.isFinite(grid[k])) continue;
        let sum = 0, n = 0;
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const rr = r + dr, cc = c + dc;
            if (rr < 0 || rr >= G || cc < 0 || cc >= G) continue;
            const v = grid[rr * G + cc];
            if (Number.isFinite(v)) { sum += v; n++; }
          }
        }
        if (n >= 3) { next[k] = sum / n; filled++; }
      }
    }
    grid.set(next);
    if (filled === 0) break;
  }

  return { grid, hasData: true, maxD: maxDepth };
}

// Resample an arbitrary (nrows × ncols) depth grid (row-major Float array or
// 2D array, NaN/nodata = land/empty) onto the GRID_RES × GRID_RES mesh grid
// the Seabed renderer expects. Used by the Results "🧊 3D" view, which is fed
// a downsampled depth `grid` straight from /api/results/<id>/analyse instead of
// a point cloud. North-up: row 0 = north edge, matching buildDepthGrid.
function resampleGrid(values, nrows, ncols, nodata) {
  const G = GRID_RES;
  const out = new Float32Array(G * G);
  out.fill(NaN);
  if (!values || !nrows || !ncols) return out;
  const at = (r, c) => {
    const v = Array.isArray(values[0]) ? values[r][c] : values[r * ncols + c];
    if (v == null) return NaN;
    const n = Number(v);
    if (!Number.isFinite(n)) return NaN;
    if (nodata != null && n === nodata) return NaN;
    return n;
  };
  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      const sr = Math.min(nrows - 1, Math.floor((r / (G - 1)) * (nrows - 1)));
      const sc = Math.min(ncols - 1, Math.floor((c / (G - 1)) * (ncols - 1)));
      out[r * G + c] = at(sr, sc);
    }
  }
  return out;
}

// Seabed mesh — vertices displaced by depth, vertex colors from depth ramp.
function Seabed({ grid, maxD, zScale }) {
  const meshRef = useRef();

  const geometry = useMemo(() => {
    const G = GRID_RES;
    const geo = new THREE.PlaneGeometry(SCENE_SPAN, SCENE_SPAN, G - 1, G - 1);
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);

    // Deterministic 2D hash → smooth pseudo-noise in [0, 1]. Cheap
    // replacement for a real DEM that gives land vertices a terrain-
    // like bump instead of a flat slab.
    const hash2 = (a, b) => {
      const x = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
      return x - Math.floor(x);
    };
    const smoothNoise = (r, c) => {
      const r0 = Math.floor(r), c0 = Math.floor(c);
      const tr = r - r0, tc = c - c0;
      const e = (t) => t * t * (3 - 2 * t);  // smoothstep
      const v00 = hash2(r0,     c0    );
      const v10 = hash2(r0 + 1, c0    );
      const v01 = hash2(r0,     c0 + 1);
      const v11 = hash2(r0 + 1, c0 + 1);
      const a = v00 + (v10 - v00) * e(tr);
      const b = v01 + (v11 - v01) * e(tr);
      return a + (b - a) * e(tc);
    };
    // Two octaves of smooth-noise for slightly bumpy terrain
    const terrainNoise = (r, c) =>
      0.65 * smoothNoise(r * 0.18, c * 0.18) +
      0.35 * smoothNoise(r * 0.06, c * 0.06);
    // Maximum land elevation in metres above the water plane. Coastal
    // UAE is mostly low-lying so 4 m is a reasonable visual anchor.
    const LAND_MAX_M = 4.0;

    for (let i = 0; i < pos.count; i++) {
      const r = Math.floor(i / G);
      const c = i % G;
      const k = r * G + c;
      const d = grid[k];
      if (Number.isFinite(d) && d > 0.05) {
        // Sea floor — z below the water plane
        pos.setZ(i, -d * zScale);
        const t = Math.min(d / maxD, 1);
        const [r0, g0, b0] = depthColor(t);
        colors[i * 3] = r0; colors[i * 3 + 1] = g0; colors[i * 3 + 2] = b0;
      } else {
        // Land — terrain-style bumpy elevation above water (always > 0)
        // and a grey palette (rocky / urban look).
        const e_m = 0.4 + LAND_MAX_M * terrainNoise(r, c);  // 0.4–4.4 m
        pos.setZ(i, e_m * zScale);
        // Grey palette with a subtle warm tint that brightens with
        // elevation, so higher land reads as sandy / rocky highlight
        // and lower land as dark grey shoreline.
        const u = Math.min(1.0, (e_m - 0.4) / LAND_MAX_M);
        const base = 0.46 + 0.30 * u;             // 0.46 → 0.76
        colors[i * 3]     = base + 0.02;
        colors[i * 3 + 1] = base + 0.01;
        colors[i * 3 + 2] = base - 0.03;          // slightly cooler in blue
      }
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return geo;
  }, [grid, maxD, zScale]);

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} receiveShadow castShadow>
      <meshStandardMaterial
        vertexColors
        roughness={0.85}
        metalness={0.05}
        flatShading={false}
        side={THREE.FrontSide}
      />
    </mesh>
  );
}

// Animated water surface plane at z = 0
//
// Throttled to ~30 FPS and only updates a *single* z-channel array — much
// cheaper than the previous per-vertex Vector3 set. Drops the
// MeshPhysicalMaterial transmission path (which forces a second pass on
// the GPU and was the main reason the 3D tab "blocked" the page on
// integrated GPUs); MeshStandardMaterial with transparency is plenty
// realistic and 5-10× cheaper to render.
function WaterSurface({ animate = true }) {
  const ref = useRef();
  const t = useRef(0);
  const lastTick = useRef(0);
  useFrame((_, dt) => {
    if (!animate || !ref.current) return;
    t.current += dt;
    lastTick.current += dt;
    if (lastTick.current < 1 / 30) return;     // throttle ≤30 FPS
    lastTick.current = 0;
    const pos = ref.current.geometry.attributes.position;
    const arr = pos.array;
    for (let i = 0; i < pos.count; i++) {
      const x = arr[i * 3], y = arr[i * 3 + 1];
      arr[i * 3 + 2] =
        Math.sin(x * 0.18 + t.current * 0.7) * 0.16 +
        Math.cos(y * 0.22 + t.current * 0.85) * 0.12;
    }
    pos.needsUpdate = true;
  });
  const geometry = useMemo(
    () => new THREE.PlaneGeometry(SCENE_SPAN * 1.05, SCENE_SPAN * 1.05, WATER_RES, WATER_RES),
    []
  );
  return (
    <mesh ref={ref} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
      <meshStandardMaterial
        color="#3ec5dc"
        roughness={0.22}
        metalness={0.10}
        transparent
        opacity={0.45}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// Depth scale legend in screen space
function DepthLegend({ maxD }) {
  const stops = [0, 0.25, 0.5, 0.75, 1];
  return (
    <div style={{
      position: 'absolute', bottom: 12, left: 12,
      background: 'rgba(255,255,255,0.94)',
      borderRadius: 10, padding: '10px 12px',
      border: '1px solid rgba(0,0,0,0.08)',
      boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
      fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
      minWidth: 110,
    }}>
      <div style={{ fontWeight: 800, color: '#0f172a', marginBottom: 1, letterSpacing: '0.06em' }}>DEPTH</div>
      <div style={{ fontWeight: 600, color: '#64748b', marginBottom: 6, fontSize: 8.5 }}>metres · MSL</div>
      {stops.slice().reverse().map((f, i) => {
        const d = maxD * f;
        return (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
            <div style={{
              width: 22, height: 12,
              background: rampCss(f),
              borderRadius: 2,
              border: '1px solid rgba(0,0,0,0.06)',
            }} />
            <span style={{ color: '#475569', fontWeight: 600 }}>{d.toFixed(d < 10 ? 1 : 0)}</span>
          </div>
        );
      })}
    </div>
  );
}

// Camera-rig helper — sets a nice oblique starting view
function CameraRig() {
  // No-op; OrbitControls handles initial position via its `target`.
  return null;
}

export default function ThreeDView({ data, gridPayload }) {
  const stats = data?.stats || {};
  const points = data?.points || data?.interpolated_points || [];
  // Vertical exaggeration — user-controllable depth amplification for clarity.
  const [zScale, setZScale] = useState(5);
  const Z_OPTS = [2, 5, 10, 20];

  // Two ingest paths:
  //  • gridPayload  — the analyse `grid` ({values,nrows,ncols,nodata,z_min,
  //    z_max}) from /api/results/<id>/analyse → resample straight onto the mesh.
  //  • data.points  — the live-compute point cloud (legacy v1 path).
  const maxD = gridPayload
    ? (Number(gridPayload.z_max) || MAX_DEPTH_FALLBACK)
    : (Number(stats.max_depth) || MAX_DEPTH_FALLBACK);

  const { grid, hasData } = useMemo(() => {
    if (gridPayload && gridPayload.values) {
      const g = resampleGrid(
        gridPayload.values, gridPayload.nrows, gridPayload.ncols, gridPayload.nodata,
      );
      const has = g.some((v) => Number.isFinite(v) && v > 0.05);
      return { grid: g, hasData: has, maxD };
    }
    return buildDepthGrid(points, data?.bbox, maxD);
  }, [gridPayload, points, data?.bbox, maxD]);

  if (!hasData) {
    return (
      <div style={{
        width: '100%', height: '100%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'linear-gradient(180deg, #dbeafe 0%, #bfdbfe 70%, #f0f9ff 100%)',
        color: '#475569', fontFamily: 'JetBrains Mono, monospace', fontSize: 13,
      }}>
        Run a depth analysis to see the 3D view.
      </div>
    );
  }

  // Initial camera placed roughly south-west of scene, looking down ~25°.
  const camPos = [SCENE_SPAN * 0.95, SCENE_SPAN * 0.55, SCENE_SPAN * 0.95];

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#dbeafe' }}>
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: camPos, fov: 38, near: 1, far: 1000 }}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      >
        {/* Sky gradient via a large background sphere */}
        <color attach="background" args={['#cfe6f5']} />
        <fog attach="fog" args={['#cfe6f5', SCENE_SPAN * 1.4, SCENE_SPAN * 3]} />

        {/* Lighting */}
        <hemisphereLight args={['#f0f9ff', '#0c4a6e', 0.55]} />
        <ambientLight intensity={0.18} />
        <directionalLight
          position={[60, 90, 40]}
          intensity={1.25}
          castShadow
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
          shadow-camera-near={0.5}
          shadow-camera-far={500}
          shadow-camera-left={-SCENE_SPAN}
          shadow-camera-right={SCENE_SPAN}
          shadow-camera-top={SCENE_SPAN}
          shadow-camera-bottom={-SCENE_SPAN}
        />

        {/* Scene */}
        <Seabed grid={grid} maxD={maxD} zScale={zScale} />
        <WaterSurface />

        {/* Soft contact shadow at the seabed footprint */}
        <ContactShadows
          position={[0, -maxD * zScale - 0.05, 0]}
          opacity={0.30} scale={SCENE_SPAN * 1.3} blur={2.0} far={SCENE_SPAN}
        />

        {/*
         * Removed <Environment preset="sunset"> — the HDR cubemap it
         * fetched (~1 MB) was the main reason the 3D tab "blocked"
         * the page. The hemisphere + directional + ambient lighting
         * above is enough for a clean PBR look, and the seabed renders
         * instantly.
         */}

        <OrbitControls
          enablePan
          enableZoom
          enableRotate
          minDistance={SCENE_SPAN * 0.4}
          maxDistance={SCENE_SPAN * 2.4}
          target={[0, -maxD * zScale * 0.5, 0]}
          maxPolarAngle={Math.PI / 2.05}
        />
        <CameraRig />
      </Canvas>

      {/* Hint + vertical-exaggeration control (top-right) */}
      <div style={{
        position: 'absolute', top: 12, right: 12,
        background: 'rgba(255,255,255,0.95)',
        padding: '8px 12px', borderRadius: 10,
        fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
        color: '#475569', border: '1px solid rgba(0,0,0,0.08)',
        boxShadow: '0 2px 12px rgba(0,0,0,0.08)', minWidth: 150,
      }}>
        <div style={{ fontWeight: 800, color: '#0f172a', letterSpacing: '0.06em', marginBottom: 6 }}>
          VERTICAL EXAGGERATION
        </div>
        <div style={{ display: 'flex', gap: 3, marginBottom: 6 }}>
          {Z_OPTS.map((z) => (
            <button key={z} onClick={() => setZScale(z)} style={{
              flex: 1, padding: '5px 0', fontSize: 10, fontWeight: 800,
              fontFamily: 'JetBrains Mono, monospace', borderRadius: 5, cursor: 'pointer',
              border: zScale === z ? '1.5px solid #0284c7' : '1px solid #cbd5e1',
              background: zScale === z ? 'rgba(2,132,199,0.12)' : '#fff',
              color: zScale === z ? '#0369a1' : '#64748b',
            }}>{z}×</button>
          ))}
        </div>
        <div style={{ fontSize: 9, color: '#94a3b8' }}>Drag • Scroll • Right-click pan</div>
      </div>
      <DepthLegend maxD={maxD} />
    </div>
  );
}
