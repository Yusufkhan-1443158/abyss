/* coverage-set.js — track which (bbox, [start,end]) slices have been fully
 * loaded into the client cache, so pan/scrubber can be additive.
 *
 * - Quantizes bbox to a fixed-size grid (default 0.5° cells) so coverage is
 *   tile-shaped, not arbitrary polygons. This keeps set arithmetic O(cells)
 *   and dedupes across users panning over similar areas.
 * - Tracks coverage per cell as a list of date ranges. `missing(bbox, range)`
 *   returns the list of (cell-bbox, range-segment) deltas the caller still
 *   needs to fetch.
 * - Persists optionally via an IdbCache .meta slot (key 'coverage:v1'); the
 *   payload is a small JSON blob (cell -> list of ranges).
 *
 * Usage:
 *   const cov = new CoverageSet({ cellDeg: 0.5 });
 *   await cov.restore(cache);  // optional
 *   const missing = cov.missing([w,s,e,n], ['2025-04-27','2026-04-27']);
 *   for (const slice of missing) await fetchAndCache(slice);
 *   cov.add([w,s,e,n], ['2025-04-27','2026-04-27']);
 *   await cov.persist(cache);
 */

(function (root) {
  function dayKey(d) {
    if (d == null) return null;
    if (typeof d === 'string') return d.slice(0, 10);
    return new Date(d).toISOString().slice(0, 10);
  }
  function dayCmp(a, b) { return a < b ? -1 : a > b ? 1 : 0; }
  // Inclusive integer-day distance helper (for stitch/merge gap = 0).
  function nextDay(iso) {
    const d = new Date(iso + 'T00:00:00Z');
    d.setUTCDate(d.getUTCDate() + 1);
    return d.toISOString().slice(0, 10);
  }

  class RangeSet {
    constructor() { this.ranges = []; }      // sorted, non-overlapping [start,end] inclusive
    add(start, end) {
      if (start > end) return;
      const r = [start, end];
      this.ranges.push(r);
      this.ranges.sort((a, b) => dayCmp(a[0], b[0]));
      const merged = [];
      for (const cur of this.ranges) {
        if (!merged.length) { merged.push(cur); continue; }
        const top = merged[merged.length - 1];
        if (cur[0] <= top[1] || cur[0] === nextDay(top[1])) {
          if (cur[1] > top[1]) top[1] = cur[1];
        } else {
          merged.push(cur);
        }
      }
      this.ranges = merged;
    }
    // Return the parts of [s,e] not already covered.
    missing(s, e) {
      if (s > e) return [];
      const gaps = [];
      let cursor = s;
      for (const [rs, re] of this.ranges) {
        if (re < cursor) continue;
        if (rs > e) break;
        if (rs > cursor) gaps.push([cursor, dayCmp(rs, e) > 0 ? e : rs > s ? prevDay(rs) : rs]);
        cursor = dayCmp(re, cursor) > 0 ? nextDay(re) : cursor;
        if (cursor > e) break;
      }
      if (cursor <= e) gaps.push([cursor, e]);
      return gaps;
    }
    contains(s, e) { return this.missing(s, e).length === 0; }
    toJSON() { return this.ranges; }
    static fromJSON(arr) {
      const r = new RangeSet();
      r.ranges = (arr || []).map(([s, e]) => [s, e]);
      return r;
    }
  }
  function prevDay(iso) {
    const d = new Date(iso + 'T00:00:00Z');
    d.setUTCDate(d.getUTCDate() - 1);
    return d.toISOString().slice(0, 10);
  }

  class CoverageSet {
    constructor(opts = {}) {
      this.cellDeg = opts.cellDeg || 0.5;     // 0.5° cells (~55km at equator)
      this.cells = new Map();                 // cellKey -> RangeSet
    }

    _cellsFor(bbox) {
      const [w, s, e, n] = bbox;
      const sz = this.cellDeg;
      const x0 = Math.floor(w / sz) * sz;
      const x1 = Math.ceil(e / sz) * sz;
      const y0 = Math.floor(s / sz) * sz;
      const y1 = Math.ceil(n / sz) * sz;
      const out = [];
      for (let x = x0; x < x1; x += sz) {
        for (let y = y0; y < y1; y += sz) {
          out.push({
            key: `${x.toFixed(3)}:${y.toFixed(3)}`,
            bbox: [x, y, x + sz, y + sz],
          });
        }
      }
      return out;
    }

    add(bbox, [start, end]) {
      const s = dayKey(start), e = dayKey(end);
      if (!s || !e) return;
      for (const c of this._cellsFor(bbox)) {
        let rs = this.cells.get(c.key);
        if (!rs) { rs = new RangeSet(); this.cells.set(c.key, rs); }
        rs.add(s, e);
      }
    }

    // Returns array of {bbox: [w,s,e,n], range: [start,end]} chunks the caller
    // still needs to fetch. Aggregates contiguous cells that share the SAME
    // missing range so the network sees fewer, fatter requests.
    missing(bbox, [start, end]) {
      const s = dayKey(start), e = dayKey(end);
      if (!s || !e) return [];
      const perCell = this._cellsFor(bbox).map(c => {
        const rs = this.cells.get(c.key);
        const gaps = rs ? rs.missing(s, e) : [[s, e]];
        return { ...c, gaps };
      });
      const out = [];
      for (const c of perCell) {
        for (const g of c.gaps) out.push({ bbox: c.bbox, range: g });
      }
      return out;
    }

    contains(bbox, [start, end]) {
      return this.missing(bbox, [start, end]).length === 0;
    }

    toJSON() {
      const o = {};
      for (const [k, rs] of this.cells) o[k] = rs.toJSON();
      return { cellDeg: this.cellDeg, cells: o };
    }

    static fromJSON(j) {
      const cs = new CoverageSet({ cellDeg: j.cellDeg });
      for (const [k, ranges] of Object.entries(j.cells || {})) {
        cs.cells.set(k, RangeSet.fromJSON(ranges));
      }
      return cs;
    }

    async restore(cache, key = 'coverage:v1') {
      try {
        const blob = await cache.getMeta(key);
        if (blob) {
          const j = JSON.parse(blob);
          this.cellDeg = j.cellDeg || this.cellDeg;
          this.cells.clear();
          for (const [k, ranges] of Object.entries(j.cells || {})) {
            this.cells.set(k, RangeSet.fromJSON(ranges));
          }
        }
      } catch (e) { console.warn('[coverage] restore failed', e); }
      return this;
    }

    async persist(cache, key = 'coverage:v1') {
      try {
        await cache.setMeta(key, JSON.stringify(this.toJSON()));
      } catch (e) { console.warn('[coverage] persist failed', e); }
    }
  }

  root.CoverageSet = CoverageSet;
})(typeof window !== 'undefined' ? window : globalThis);
