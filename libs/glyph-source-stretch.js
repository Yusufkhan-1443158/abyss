/*
 * glyph-source-stretch.js — platform-wide WebGPU 16-bit SOURCE-tile re-stretch.
 *
 * Fetches a canonical 16-bit SOURCE tile (the SWS1 blob) from the super-worker's
 * /superworker/source/{z}/{x}/{y}.bin endpoint, uploads it ONCE into an
 * rgba16uint integer texture, and runs the stretch+band-combo compute shader
 * (stretch.wgsl) to an rgba8unorm output. Re-stretching (contrast / gamma /
 * band-combo) only rewrites a uniform buffer and re-dispatches the SAME pipeline
 * against the ALREADY-RESIDENT texture — ZERO server round-trip.
 *
 * This is a SHARED module (not per-page). It mirrors the WebGPU patterns already
 * in libs/anime4k-webgpu.js (compute pipeline, @workgroup_size(8,8),
 * textureLoad/textureStore), but uses an INTEGER texture (texture_2d<u32>,
 * rgba16uint) sampled with textureLoad — the correct lossless carrier for >11-bit
 * integer source (rgba16float would lose precision; uint16->f32 in the shader is
 * lossless because 65535 < 2^24).
 *
 * KEY CORRECTNESS NOTES (match the server oracle exactly):
 *  - 3-band source is shipped as a 4-BAND blob (R,G,B,coverage) by the server, so
 *    it uploads directly into rgba16uint (8 bytes/texel, 2048 B/row at 256 — a
 *    256-multiple, clean writeTexture). No rgb16uint format exists. The 4th band
 *    is the per-pixel coverage/alpha mask -> shader alpha.
 *  - The default Stretch uniform is initialized from the X-SW-Default-Stretch
 *    header (server-baked guard: hi=lo+1.0 when hi<=lo), so first paint == server
 *    default 8-bit PNG at 0 LSB (the shader floor-quantizes to match the server's
 *    truncating .astype(uint8)).
 *  - n_rasters > 1 means client re-stretch is APPROXIMATE for non-top rasters
 *    (the blob stores the composite); the demo flags this and can fall back to a
 *    server re-derive.
 */

export const STRETCH_WGSL = `
struct Stretch {
  lo      : vec4<f32>,
  hi      : vec4<f32>,
  gamma   : vec4<f32>,
  bandmap : vec4<u32>,
};

@group(0) @binding(0) var src : texture_2d<u32>;
@group(0) @binding(1) var outTex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> p : Stretch;

fn stretch_one(v: f32, lo: f32, hi: f32, g: f32) -> f32 {
  // span guard matches the CPU oracle: divide by (hi-lo), force 1.0 ONLY if hi<=lo
  let span = select(hi - lo, 1.0, hi <= lo);
  let norm = clamp((v - lo) / span, 0.0, 1.0);
  let gv   = pow(norm, 1.0 / g);
  // FLOOR-quantize so rgba8unorm's round reproduces the server's truncating cast
  return floor(clamp(gv, 0.0, 1.0) * 255.0) / 255.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(outTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let pix = vec2<i32>(i32(gid.x), i32(gid.y));
  let raw: vec4<u32> = textureLoad(src, pix, 0);
  let r = stretch_one(f32(raw[p.bandmap.x]), p.lo.x, p.hi.x, p.gamma.x);
  let g = stretch_one(f32(raw[p.bandmap.y]), p.lo.y, p.hi.y, p.gamma.y);
  let b = stretch_one(f32(raw[p.bandmap.z]), p.lo.z, p.hi.z, p.gamma.z);
  let cov = raw[p.bandmap.w];
  let a = select(0.0, 1.0, cov > 0u);
  textureStore(outTex, pix, vec4<f32>(r, g, b, a));
}
`;

const MAGIC = 0x53575331; // "SWS1" big-endian read of 4 ASCII bytes

/* ----------------------------- SWS1 parsing ----------------------------- */

// Inflate zlib payloads in-browser without a dependency, via DecompressionStream
// (widely supported in modern browsers). raw passes through; zstd is not browser-
// native, so the server should emit zlib/raw for the browser path (see README).
async function inflate(bytes, codec /* 0 raw, 1 zlib, 2 zstd */) {
  if (codec === 0) return bytes;
  if (codec === 1) {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('zlib source blob but DecompressionStream unavailable; ' +
                      'configure the server to emit raw for this client');
    }
    const ds = new DecompressionStream('deflate');
    const stream = new Blob([bytes]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  throw new Error('zstd payload cannot be inflated in-browser; server must emit ' +
                  'zlib or raw for the browser path (X-SW-Codec)');
}

export async function parseSWS1(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  if (dv.byteLength < 16) throw new Error('blob too short for SWS1 header');
  const magic = dv.getUint32(0, false); // ASCII bytes, byte-order-agnostic
  if (magic !== MAGIC) throw new Error('bad SWS1 magic');
  // ALL multi-byte header fields are LITTLE-ENDIAN (per the SWS1 spec).
  const bands  = dv.getUint8(4);
  const dtype  = dv.getUint8(5);   // 1 = uint16
  const codec  = dv.getUint8(6);   // 0 raw / 1 zlib / 2 zstd
  const layout = dv.getUint8(7);   // 0 interleaved
  const width  = dv.getUint16(8, true);
  const height = dv.getUint16(10, true);
  const nodata = dv.getUint16(12, true);
  const nRasters = dv.getUint8(14);
  const payload = new Uint8Array(arrayBuffer, 16);
  const rawBytes = await inflate(payload, codec);
  // interleaved uint16 little-endian -> Uint16Array view
  const px = new Uint16Array(rawBytes.buffer, rawBytes.byteOffset,
                             rawBytes.byteLength >> 1);
  return { bands, dtype, codec, layout, width, height, nodata, nRasters, px };
}

/* --------------------- pad to 4-band rgba16uint ------------------------- */
// rgba16uint requires 4 channels (8 bytes/texel). The server already emits a
// 4-band blob (R,G,B,coverage), but if a 1/2/3-band blob is ever supplied we pad
// here (broadcast colour, coverage=65535) so the upload stays a single texture.
function toRGBA16(px, bands, width, height) {
  if (bands === 4) return px; // already RGBA
  const out = new Uint16Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const r = px[i * bands + 0];
    const g = bands >= 2 ? px[i * bands + 1] : r;
    const b = bands >= 3 ? px[i * bands + 2] : r;
    out[i * 4 + 0] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 65535; // assume covered when no explicit coverage band
  }
  return out;
}

/* --------------------------- the renderer ------------------------------- */

export class SourceStretchRenderer {
  constructor(device) {
    this.device = device;
    this.pipeline = null;
    this.srcTexture = null;
    this.outTexture = null;
    this.uniformBuf = null;
    this.bindGroup = null;
    this.width = 0;
    this.height = 0;
    this.networkRequests = 0; // counter the demo asserts never increments on re-stretch
  }

  async init() {
    const module = this.device.createShaderModule({ code: STRETCH_WGSL });
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    // uniform: 3x vec4<f32> + 1x vec4<u32> = 4*16 = 64 bytes
    this.uniformBuf = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /* Fetch ONE source tile and upload it into a resident rgba16uint texture.
   * Returns { header, defaultStretch } parsed from the X-SW-* response headers. */
  async loadTile(url, fetchOpts = {}) {
    this.networkRequests += 1; // THE only place this increments
    const resp = await fetch(url, fetchOpts);
    const cache = resp.headers.get('X-SW-Cache');
    const bandsHdr = resp.headers.get('X-SW-Bands');
    const buf = await resp.arrayBuffer();
    // empty/no-raster tile: sub-magic-length body -> transparent, skip upload
    if (!bandsHdr || bandsHdr === '0' || buf.byteLength < 16) {
      return { empty: true, cache };
    }
    const blob = await parseSWS1(buf);
    const rgba = toRGBA16(blob.px, blob.bands, blob.width, blob.height);
    this.width = blob.width;
    this.height = blob.height;

    this.srcTexture = this.device.createTexture({
      size: [blob.width, blob.height, 1],
      format: 'rgba16uint',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture: this.srcTexture },
      rgba,
      { bytesPerRow: blob.width * 4 * 2, rowsPerImage: blob.height }, // 2048 @256
      [blob.width, blob.height, 1],
    );

    this.outTexture = this.device.createTexture({
      size: [blob.width, blob.height, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.srcTexture.createView() },
        { binding: 1, resource: this.outTexture.createView() },
        { binding: 2, resource: { buffer: this.uniformBuf } },
      ],
    });

    // default stretch from the header so first paint == server default
    let defaultStretch = [];
    try { defaultStretch = JSON.parse(resp.headers.get('X-SW-Default-Stretch') || '[]'); }
    catch (_) { defaultStretch = []; }
    const header = {
      cache, bands: blob.bands, nRasters: blob.nRasters,
      width: blob.width, height: blob.height, nodata: blob.nodata,
      codec: resp.headers.get('X-SW-Codec'),
    };
    return { empty: false, header, defaultStretch };
  }

  /* Write the stretch uniform. stretch = [{band,lo,hi,gamma}...] (R,G,B order
   * after bandmap is applied), bandmap = [rSrc,gSrc,bSrc,coverageSrc].
   * This is what an analyst's slider/dropdown calls — NO refetch. */
  setStretch(stretch, bandmap = [0, 1, 2, 3]) {
    const f = new Float32Array(12); // lo[4], hi[4], gamma[4]
    const u = new Uint32Array(4);   // bandmap
    for (let i = 0; i < 3; i++) {
      const s = stretch[i] || { lo: 0, hi: 65535, gamma: 1.0 };
      f[i] = s.lo; f[4 + i] = s.hi; f[8 + i] = s.gamma == null ? 1.0 : s.gamma;
    }
    f[3] = 0; f[7] = 1; f[11] = 1; // unused 4th lanes (keep gamma!=0)
    u[0] = bandmap[0]; u[1] = bandmap[1]; u[2] = bandmap[2]; u[3] = bandmap[3];
    // pack into the 64-byte uniform: 48 bytes float + 16 bytes uint
    const bytes = new ArrayBuffer(64);
    new Float32Array(bytes, 0, 12).set(f);
    new Uint32Array(bytes, 48, 4).set(u);
    this.device.queue.writeBuffer(this.uniformBuf, 0, bytes);
  }

  /* Re-dispatch the compute pass against the RESIDENT texture. No network.
   * Returns the GPU-only elapsed ms (timestamped by the caller via performance.now). */
  dispatch() {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /* Read the rgba8unorm output back to a Uint8ClampedArray for canvas/ImageData
   * or pixel-diff against the server PNG. */
  async readPixels() {
    const bytesPerRow = Math.ceil(this.width * 4 / 256) * 256;
    const buf = this.device.createBuffer({
      size: bytesPerRow * this.height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: this.outTexture },
      { buffer: buf, bytesPerRow, rowsPerImage: this.height },
      [this.width, this.height, 1],
    );
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    const out = new Uint8ClampedArray(this.width * this.height * 4);
    for (let y = 0; y < this.height; y++) {
      out.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + this.width * 4),
              y * this.width * 4);
    }
    buf.unmap();
    return new ImageData(out, this.width, this.height);
  }
}

export async function requestDevice() {
  if (!navigator.gpu) throw new Error('WebGPU not available in this browser');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  return adapter.requestDevice();
}
