/* ndjson-stream.js — line-buffered fetch reader for application/x-ndjson.
 *
 * Usage:
 *   const ctrl = new AbortController();
 *   const result = await streamNdjson('/api/reports/stream?bbox=...', {
 *     signal: ctrl.signal,
 *     etag: cache.getMeta('etag'),       // optional; sent as If-None-Match
 *     onManifest: (m) => updateProgressUI(m),
 *     onRow: (row) => bufferRow(row),
 *     onBatch: (rows) => flushToCluster(rows),  // called every batchSize rows
 *     batchSize: 250,
 *     headers: { Authorization: 'Bearer ' + token },
 *   });
 *   // result === { status: 200|304, manifest, rowCount, etag } |
 *   //           { status: 304, etag } when If-None-Match matched.
 *
 * The first line of a 200 response MUST be a manifest object (the server
 * always emits one). Subsequent lines are JSON-per-line rows. If the response
 * is 304 Not Modified, returns immediately with { status: 304, etag }.
 */

(function (root) {
  async function streamNdjson(url, opts = {}) {
    const {
      signal,
      etag,
      headers = {},
      onManifest,
      onRow,
      onBatch,
      batchSize = 250,
      method = 'GET',
      body,
    } = opts;

    const reqHeaders = {
      Accept: 'application/x-ndjson',
      ...headers,
    };
    if (etag) reqHeaders['If-None-Match'] = etag;

    const r = await fetch(url, {
      method,
      headers: reqHeaders,
      body,
      signal,
      credentials: 'same-origin',
    });

    if (r.status === 304) {
      return { status: 304, etag: r.headers.get('ETag') || etag };
    }
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error(`stream failed ${r.status}: ${text.slice(0, 200)}`);
    }
    if (!r.body) {
      throw new Error('stream response has no body');
    }

    const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    let manifest = null;
    let rowCount = 0;
    const batch = [];

    const flush = () => {
      if (batch.length === 0) return;
      if (onBatch) onBatch(batch.slice());
      batch.length = 0;
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;
      let nl;
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        let obj;
        try { obj = JSON.parse(line); }
        catch (e) {
          console.warn('[ndjson] bad line', e, line.slice(0, 120));
          continue;
        }
        if (manifest === null) {
          manifest = obj;
          if (onManifest) onManifest(manifest);
          continue;
        }
        rowCount++;
        if (onRow) onRow(obj);
        if (onBatch) {
          batch.push(obj);
          if (batch.length >= batchSize) flush();
        }
      }
    }
    // Trailing chunk
    if (buffer.trim()) {
      try {
        const obj = JSON.parse(buffer);
        if (manifest === null) {
          manifest = obj;
          if (onManifest) onManifest(manifest);
        } else {
          rowCount++;
          if (onRow) onRow(obj);
          if (onBatch) batch.push(obj);
        }
      } catch (e) { /* ignore tail garbage */ }
    }
    flush();

    return {
      status: r.status,
      manifest,
      rowCount,
      etag: r.headers.get('ETag') || (manifest && manifest.etag),
    };
  }

  root.NdjsonStream = { streamNdjson };
})(typeof window !== 'undefined' ? window : globalThis);
