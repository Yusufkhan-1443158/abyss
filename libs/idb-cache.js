/* idb-cache.js — tiny IndexedDB wrapper for the streaming-cache flow.
 *
 * Two object stores per cache: rows (keyPath: configurable) and meta (key/value
 * for ETags, timestamps, etc). Browser-only, no deps. Falls through to an
 * in-memory shim if IndexedDB is unavailable (incognito, denied) so callers
 * don't need to special-case it.
 *
 * Usage:
 *   const cache = await openCache('reports', { keyPath: 'report_id' });
 *   const cached = await cache.getAll({ limit: 200, orderBy: 'report_date', dir: 'desc' });
 *   await cache.putBatch(rows);
 *   await cache.setMeta('etag', 'W/"...');
 *   const etag = await cache.getMeta('etag');
 *   await cache.evictOlderThan(90);  // days; uses _last_seen_at
 *   await cache.size();
 */

(function (root) {
  const DB_PREFIX = 'glyph-cache:';

  function memShim(opts) {
    const rows = new Map();
    const meta = new Map();
    return {
      isMemory: true,
      async getAll({ limit = 1000, orderBy, dir = 'desc' } = {}) {
        let arr = Array.from(rows.values());
        if (orderBy) {
          arr.sort((a, b) => (a[orderBy] > b[orderBy] ? 1 : -1));
          if (dir === 'desc') arr.reverse();
        }
        return arr.slice(0, limit);
      },
      async get(key) { return rows.get(key); },
      async putBatch(items) {
        const now = Date.now();
        for (const it of items) {
          it._last_seen_at = now;
          rows.set(it[opts.keyPath], it);
        }
      },
      async delete(key) { rows.delete(key); },
      async clear() { rows.clear(); },
      async size() { return rows.size; },
      async evictOlderThan(days) {
        const cutoff = Date.now() - days * 86400000;
        let removed = 0;
        for (const [k, v] of rows) {
          if ((v._last_seen_at ?? 0) < cutoff) { rows.delete(k); removed++; }
        }
        return removed;
      },
      async getMeta(k) { return meta.get(k); },
      async setMeta(k, v) { meta.set(k, v); },
    };
  }

  function openCache(name, opts = {}) {
    const keyPath = opts.keyPath || 'id';
    const version = opts.version || 1;

    if (typeof indexedDB === 'undefined') {
      console.warn('[idb-cache] IndexedDB unavailable; falling back to memory cache for', name);
      return Promise.resolve(memShim({ keyPath }));
    }

    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_PREFIX + name, version);
      req.onerror = () => {
        console.warn('[idb-cache] open failed; using memory shim:', req.error);
        resolve(memShim({ keyPath }));
      };
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('rows')) {
          const store = db.createObjectStore('rows', { keyPath });
          store.createIndex('_last_seen_at', '_last_seen_at', { unique: false });
          if (opts.indexes) {
            for (const ix of opts.indexes) {
              store.createIndex(ix, ix, { unique: false });
            }
          }
        }
        if (!db.objectStoreNames.contains('meta')) {
          db.createObjectStore('meta');
        }
      };
      req.onsuccess = () => resolve(buildApi(req.result, keyPath));
    });
  }

  function buildApi(db, keyPath) {
    function tx(store, mode) {
      return db.transaction(store, mode).objectStore(store);
    }
    function awaitReq(r) {
      return new Promise((resolve, reject) => {
        r.onsuccess = () => resolve(r.result);
        r.onerror = () => reject(r.error);
      });
    }

    return {
      isMemory: false,
      async getAll({ limit = 1000, orderBy, dir = 'desc' } = {}) {
        if (orderBy && db.transaction('rows').objectStore('rows').indexNames.contains(orderBy)) {
          const out = [];
          return new Promise((resolve, reject) => {
            const idx = tx('rows', 'readonly').index(orderBy);
            const cursor = idx.openCursor(null, dir === 'desc' ? 'prev' : 'next');
            cursor.onsuccess = (e) => {
              const c = e.target.result;
              if (!c || out.length >= limit) return resolve(out);
              out.push(c.value);
              c.continue();
            };
            cursor.onerror = () => reject(cursor.error);
          });
        }
        const all = await awaitReq(tx('rows', 'readonly').getAll());
        if (orderBy) {
          all.sort((a, b) => (a[orderBy] > b[orderBy] ? 1 : -1));
          if (dir === 'desc') all.reverse();
        }
        return all.slice(0, limit);
      },
      async get(key) {
        return awaitReq(tx('rows', 'readonly').get(key));
      },
      async putBatch(items) {
        const t = db.transaction('rows', 'readwrite');
        const store = t.objectStore('rows');
        const now = Date.now();
        for (const it of items) {
          it._last_seen_at = now;
          store.put(it);
        }
        return new Promise((resolve, reject) => {
          t.oncomplete = () => resolve(items.length);
          t.onerror = () => reject(t.error);
          t.onabort = () => reject(t.error);
        });
      },
      async delete(key) {
        return awaitReq(tx('rows', 'readwrite').delete(key));
      },
      async clear() {
        return awaitReq(tx('rows', 'readwrite').clear());
      },
      async size() {
        return awaitReq(tx('rows', 'readonly').count());
      },
      async evictOlderThan(days) {
        const cutoff = Date.now() - days * 86400000;
        const t = db.transaction('rows', 'readwrite');
        const idx = t.objectStore('rows').index('_last_seen_at');
        const range = IDBKeyRange.upperBound(cutoff);
        return new Promise((resolve, reject) => {
          let removed = 0;
          const cursor = idx.openCursor(range);
          cursor.onsuccess = (e) => {
            const c = e.target.result;
            if (!c) return resolve(removed);
            c.delete();
            removed++;
            c.continue();
          };
          cursor.onerror = () => reject(cursor.error);
        });
      },
      async getMeta(k) {
        return awaitReq(tx('meta', 'readonly').get(k));
      },
      async setMeta(k, v) {
        return awaitReq(tx('meta', 'readwrite').put(v, k));
      },
    };
  }

  root.IdbCache = { openCache };
})(typeof window !== 'undefined' ? window : globalThis);
