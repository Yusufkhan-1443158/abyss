/* glyph-upload-queue.js
 *
 * Cross-page upload state for Intel Globe.
 *
 * Solves: a user starts an upload then navigates away or closes the page,
 * the upload silently dies, no warning. This lib:
 *   - Hooks beforeunload when uploads are in flight (browser-native confirm)
 *   - Aborts in-flight XHRs cleanly on confirmed exit (server doesn't hang)
 *   - Renders a fixed bottom-right tray showing per-upload progress so the
 *     user always knows what's happening
 *   - Persists the queue to sessionStorage so a same-tab navigation can
 *     restore the tray UI (the actual transfer dies — service-worker resume
 *     is documented as follow-up, intentionally out of scope here)
 *
 * Public API: window.GlyphUploadQueue
 *   .enqueue({ id?, label, file, url, headers?, onProgress?, onDone?, onError? })
 *       → returns { id, abort() }
 *   .cancel(id)
 *   .status()   → { active: [...], queued: [...], completed: [...], failed: [...] }
 *   .clearCompleted()
 *   .subscribe(fn) → returns unsubscribe()
 *
 * Loaded globally — see master-home / login bootstrap.
 */
(function () {
    'use strict';
    if (window.GlyphUploadQueue) return;   // idempotent (master-home + page may both include)

    var STORAGE_KEY = 'glyph_upload_queue';
    var MAX_CONCURRENT = 3;
    var jobs = [];          // {id, label, file (kept only while active), url, headers, status, progress, error, xhr}
    var subscribers = [];

    // ---- internal helpers ----------------------------------------------------
    function uuid() {
        return (crypto && crypto.randomUUID) ? crypto.randomUUID()
            : 'u-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
    }
    function notify() {
        // Persist a minimal snapshot for same-tab restore (no File objects —
        // those don't survive serialization, the tray UI just shows what was queued).
        try {
            sessionStorage.setItem(STORAGE_KEY, JSON.stringify(jobs.map(function (j) {
                return {
                    id: j.id, label: j.label, status: j.status,
                    progress: j.progress, error: j.error,
                    fileName: j.file && j.file.name, fileSize: j.file && j.file.size
                };
            })));
        } catch (e) { /* storage full or disabled */ }
        for (var i = 0; i < subscribers.length; i++) {
            try { subscribers[i](groupedStatus()); } catch (e) { /* don't let one subscriber break others */ }
        }
        renderTray();
    }
    function groupedStatus() {
        return {
            active:    jobs.filter(function (j) { return j.status === 'active'; }),
            queued:    jobs.filter(function (j) { return j.status === 'queued'; }),
            completed: jobs.filter(function (j) { return j.status === 'completed'; }),
            failed:    jobs.filter(function (j) { return j.status === 'failed'; })
        };
    }
    function pump() {
        var active = jobs.filter(function (j) { return j.status === 'active'; }).length;
        while (active < MAX_CONCURRENT) {
            var next = jobs.find(function (j) { return j.status === 'queued'; });
            if (!next) break;
            startJob(next);
            active++;
        }
        notify();
    }
    function startJob(job) {
        job.status = 'active';
        job.progress = 0;
        var xhr = new XMLHttpRequest();
        job.xhr = xhr;
        var fd = new FormData();
        fd.append('file', job.file, job.file.name);
        if (job.extraFields) {
            Object.keys(job.extraFields).forEach(function (k) { fd.append(k, job.extraFields[k]); });
        }
        xhr.upload.addEventListener('progress', function (ev) {
            if (ev.lengthComputable) {
                job.progress = Math.round((ev.loaded / ev.total) * 100);
                if (job.onProgress) try { job.onProgress(job.progress, ev.loaded, ev.total); } catch (e) {}
                notify();
            }
        });
        xhr.addEventListener('load', function () {
            job.xhr = null;
            if (xhr.status >= 200 && xhr.status < 300) {
                job.status = 'completed';
                job.progress = 100;
                var resp = null;
                try { resp = JSON.parse(xhr.responseText); } catch (e) { resp = xhr.responseText; }
                if (job.onDone) try { job.onDone(resp, xhr.status); } catch (e) {}
            } else {
                job.status = 'failed';
                job.error = 'HTTP ' + xhr.status + ': ' + (xhr.responseText || '').slice(0, 200);
                if (job.onError) try { job.onError(job.error, xhr.status); } catch (e) {}
            }
            // Free the File reference to release memory; tray still shows the label.
            job.file = null;
            pump();
        });
        xhr.addEventListener('error', function () {
            job.xhr = null;
            job.status = 'failed';
            job.error = 'Network error';
            if (job.onError) try { job.onError(job.error, 0); } catch (e) {}
            job.file = null;
            pump();
        });
        xhr.addEventListener('abort', function () {
            job.xhr = null;
            job.status = 'failed';
            job.error = 'Aborted';
            job.file = null;
            pump();
        });
        xhr.open('POST', job.url, true);
        if (job.headers) {
            Object.keys(job.headers).forEach(function (k) { xhr.setRequestHeader(k, job.headers[k]); });
        }
        xhr.send(fd);
    }
    function abortJob(job) {
        if (job.xhr) {
            try { job.xhr.abort(); } catch (e) { /* ignore */ }
        }
    }

    // ---- beforeunload guard --------------------------------------------------
    window.addEventListener('beforeunload', function (ev) {
        var hasActive = jobs.some(function (j) { return j.status === 'active' || j.status === 'queued'; });
        if (!hasActive) return;
        var msg = (window.GlyphI18n && window.GlyphI18n.t)
            ? window.GlyphI18n.t('upload.queue.unsaved_warning')
            : 'You have uploads in progress. Leave anyway?';
        ev.preventDefault();
        ev.returnValue = msg;   // legacy browsers
        return msg;
    });
    // pagehide fires when the user actually leaves (after the confirm).
    window.addEventListener('pagehide', function () {
        jobs.filter(function (j) { return j.status === 'active'; }).forEach(abortJob);
    });

    // ---- bell + popup UI -----------------------------------------------------
    // Always-visible bell in the top-right of every page. Click → dropdown
    // popup with the upload list. Outside-click closes it. Active count is
    // shown as a badge on the bell so the user always knows there's activity
    // even when the popup is closed. Replaces the bottom-right tray pattern.
    var bellEl = null, popupEl = null, popupOpen = false;

    function ensureBell() {
        if (bellEl) return bellEl;
        if (!document.getElementById('glyph-upload-bell-style')) {
            var st = document.createElement('style');
            st.id = 'glyph-upload-bell-style';
            st.textContent = ''
                /* Default = static placement inside .glyph-topbar-right, .top-bar-right (set
                   below). The .floating modifier kicks in only when the topbar
                   container isn't found on a page (e.g. login.html), in which
                   case we fall back to a fixed top-right position. */
                + '#glyph-upload-bell{position:relative;width:34px;height:34px;border-radius:50%;background:rgba(15,15,25,0.6);border:1px solid rgba(120,140,180,0.3);color:#c9d4f5;font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;backdrop-filter:blur(8px);transition:all 0.15s;margin:0 4px}'
                + '#glyph-upload-bell.floating{position:fixed;top:14px;right:16px;z-index:9000}'
                + '#glyph-upload-bell:hover{border-color:#00f5ff;color:#00f5ff;background:rgba(0,245,255,0.08)}'
                + '#glyph-upload-bell.has-active{border-color:#00f5ff;color:#00f5ff}'
                + '#glyph-upload-bell.has-failed{border-color:#ef4444;color:#ef4444}'
                + '#glyph-upload-bell .badge{position:absolute;top:-4px;right:-4px;min-width:16px;height:16px;padding:0 4px;border-radius:8px;background:#00f5ff;color:#05122e;font-size:10px;font-weight:700;font-family:ui-monospace,monospace;display:flex;align-items:center;justify-content:center;line-height:1;pointer-events:none}'
                + '#glyph-upload-bell.has-failed .badge{background:#ef4444;color:#fff}'
                + '#glyph-upload-bell .badge.hidden{display:none}'
                + '#glyph-upload-popup{position:fixed;top:54px;right:16px;width:340px;max-height:420px;background:rgba(15,15,25,0.96);border:1px solid rgba(120,140,180,0.4);border-radius:10px;color:#e7ecff;font:12px/1.4 system-ui,sans-serif;z-index:9001;backdrop-filter:blur(12px);box-shadow:0 6px 28px rgba(0,0,0,0.5);overflow:hidden;display:none;flex-direction:column}'
                + '#glyph-upload-popup.show{display:flex}'
                + '#glyph-upload-popup .head{padding:10px 14px;background:rgba(0,245,255,0.08);border-bottom:1px solid rgba(120,140,180,0.3);display:flex;align-items:center;gap:10px}'
                + '#glyph-upload-popup .head .title{font-weight:600;letter-spacing:.3px}'
                + '#glyph-upload-popup .head .count{color:#8fb6ff;font-family:ui-monospace,monospace;font-size:11px;flex:1;text-align:right}'
                + '#glyph-upload-popup .head .clear-btn{background:none;border:none;color:#8a97c2;cursor:pointer;font-size:11px;padding:2px 6px;border-radius:4px;font-family:inherit}'
                + '#glyph-upload-popup .head .clear-btn:hover{color:#fff;background:rgba(255,255,255,0.08)}'
                + '#glyph-upload-popup .body{overflow-y:auto;max-height:340px}'
                + '#glyph-upload-popup .empty{padding:24px 14px;text-align:center;color:#8a97c2;font-size:11.5px}'
                + '#glyph-upload-popup .item{padding:10px 14px;border-bottom:1px solid rgba(120,140,180,0.12)}'
                + '#glyph-upload-popup .item:last-child{border-bottom:none}'
                + '#glyph-upload-popup .item .lbl{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}'
                + '#glyph-upload-popup .item .name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,monospace;font-size:11px}'
                + '#glyph-upload-popup .item .pct{color:#8fb6ff;margin-left:8px;font-family:ui-monospace,monospace;font-size:11px;min-width:48px;text-align:right}'
                + '#glyph-upload-popup .item .bar{height:3px;background:rgba(120,140,180,0.15);border-radius:2px;overflow:hidden}'
                + '#glyph-upload-popup .item .fill{height:100%;background:linear-gradient(90deg,#00f5ff,#a855f7);transition:width 0.2s}'
                + '#glyph-upload-popup .item.completed .fill{background:#22c55e}'
                + '#glyph-upload-popup .item.failed .fill{background:#ef4444;width:100%}'
                + '#glyph-upload-popup .item.failed .err{color:#ef4444;font-size:10.5px;margin-top:4px;font-family:ui-monospace,monospace}'
                + '#glyph-upload-popup .item .cancel{margin-left:6px;background:none;border:none;color:#8a97c2;cursor:pointer;font-size:14px;padding:0 4px}'
                + '#glyph-upload-popup .item .cancel:hover{color:#ef4444}'
                + '[dir="rtl"] #glyph-upload-bell{right:auto;left:240px}'
                + '[dir="rtl"] #glyph-upload-popup{right:auto;left:16px}';
            document.head.appendChild(st);
        }
        bellEl = document.createElement('button');
        bellEl.id = 'glyph-upload-bell';
        bellEl.title = 'Uploads';
        bellEl.setAttribute('aria-label', 'Uploads');
        // Inline SVG bell icon — consistent across browsers, no emoji-font dep
        bellEl.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg><span class="badge hidden" id="glyph-upload-bell-badge"></span>';
        bellEl.addEventListener('click', function (ev) {
            ev.stopPropagation();
            popupOpen = !popupOpen;
            renderPopup();
        });
        // Place inside the topbar's right-side cluster so it lays out inline
        // with the existing controls (Arabic toggle, theme, gear, user). Falls
        // back to a fixed top-right position if no topbar is present (login).
        var topbarRight = document.querySelector('.glyph-topbar-right, .top-bar-right');
        if (topbarRight) {
            // Insert as the FIRST child so the bell sits to the LEFT of the
            // existing controls (LTR) — out of the way of the user-info cluster
            // which is the page-identity anchor most users look for first.
            topbarRight.insertBefore(bellEl, topbarRight.firstChild);
        } else {
            bellEl.classList.add('floating');
            document.body.appendChild(bellEl);
        }

        popupEl = document.createElement('div');
        popupEl.id = 'glyph-upload-popup';
        popupEl.innerHTML = ''
            + '<div class="head">'
            +   '<span class="title" data-i18n="upload.queue.tray_title">Uploads</span>'
            +   '<span class="count" id="gup-count"></span>'
            +   '<button class="clear-btn" id="gup-clear" title="Clear completed">Clear done</button>'
            + '</div>'
            + '<div class="body" id="gup-body"></div>';
        document.body.appendChild(popupEl);
        // Stop propagation so clicking inside doesn't fall through to the
        // body listener and close the popup.
        popupEl.addEventListener('click', function (ev) { ev.stopPropagation(); });
        document.getElementById('gup-clear').addEventListener('click', function () {
            api.clearCompleted();
        });
        // Outside-click closes.
        document.addEventListener('click', function () {
            if (popupOpen) { popupOpen = false; renderPopup(); }
        });
        // Esc closes.
        document.addEventListener('keydown', function (ev) {
            if (ev.key === 'Escape' && popupOpen) { popupOpen = false; renderPopup(); }
        });
        return bellEl;
    }
    function renderTray() {
        ensureBell();
        var grouped = groupedStatus();
        var totalLive = grouped.active.length + grouped.queued.length;
        var hasFailed = grouped.failed.length > 0;
        // Update bell state (color + badge).
        bellEl.classList.toggle('has-active', totalLive > 0);
        bellEl.classList.toggle('has-failed', !totalLive && hasFailed);
        var badge = document.getElementById('glyph-upload-bell-badge');
        var badgeCount = totalLive || hasFailed ? (totalLive || grouped.failed.length) : 0;
        badge.textContent = badgeCount > 99 ? '99+' : String(badgeCount);
        badge.classList.toggle('hidden', badgeCount === 0);
        if (popupOpen) renderPopup();
    }
    function renderPopup() {
        ensureBell();
        if (!popupOpen) { popupEl.classList.remove('show'); return; }
        popupEl.classList.add('show');

        var grouped = groupedStatus();
        var totalLive = grouped.active.length + grouped.queued.length;
        var countText = totalLive ? (totalLive + ' active')
                       : grouped.failed.length ? (grouped.failed.length + ' failed')
                       : grouped.completed.length ? (grouped.completed.length + ' done')
                       : '';
        document.getElementById('gup-count').textContent = countText;

        var body = document.getElementById('gup-body');
        if (jobs.length === 0) {
            body.innerHTML = '<div class="empty">No uploads yet.</div>';
            return;
        }
        body.innerHTML = jobs.slice().reverse().map(function (j) {
            var pctText = j.status === 'queued' ? 'queued'
                       : j.status === 'completed' ? 'done'
                       : j.status === 'failed' ? 'failed'
                       : (j.progress | 0) + '%';
            var fillW = j.status === 'completed' ? 100
                      : j.status === 'failed' ? 100
                      : (j.progress | 0);
            var cancel = (j.status === 'active' || j.status === 'queued')
                ? '<button class="cancel" data-cancel="' + j.id + '" title="Cancel">&times;</button>' : '';
            var err = j.status === 'failed' && j.error
                ? '<div class="err">' + escapeHtml(j.error) + '</div>' : '';
            return ''
                + '<div class="item ' + j.status + '">'
                +   '<div class="lbl">'
                +     '<span class="name" title="' + escapeHtml(j.label || j.fileName || j.id) + '">' + escapeHtml(j.label || j.fileName || j.id) + '</span>'
                +     '<span class="pct">' + pctText + '</span>'
                +     cancel
                +   '</div>'
                +   '<div class="bar"><div class="fill" style="width:' + fillW + '%"></div></div>'
                +   err
                + '</div>';
        }).join('');
        body.querySelectorAll('[data-cancel]').forEach(function (b) {
            b.addEventListener('click', function (ev) {
                ev.stopPropagation();
                api.cancel(this.dataset.cancel);
            });
        });
    }
    function escapeHtml(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
            return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c];
        });
    }

    // ---- public API ----------------------------------------------------------
    var api = {
        // Lower-level: register an upload that some OTHER caller is doing
        // (its own XHR). The lib does NOT do the network; it only tracks
        // the upload's existence so the beforeunload guard fires and the
        // tray UI shows it. Returns a handle with setProgress/done/fail/abort.
        // Used by pages that already have their own upload UI (collections-browse
        // modal queue) and want to additively get the cross-page guard + tray.
        track: function (opts) {
            opts = opts || {};
            /* tray was replaced by bell — no dismissal state needed */
            var job = {
                id: opts.id || uuid(),
                label: opts.label || (opts.fileName || 'upload'),
                file: null,                              // external; lib doesn't hold the bytes
                fileName: opts.fileName || opts.label,
                fileSize: opts.fileSize || 0,
                url: opts.url || null,
                status: 'active',
                progress: 0,
                error: null,
                xhr: null,                               // we don't own it
                _externalAbort: opts.abort || null,      // caller's abort fn (for cancel)
                _external: true
            };
            jobs.push(job);
            notify();
            return {
                id: job.id,
                setProgress: function (pct) {
                    job.progress = Math.max(0, Math.min(100, pct | 0));
                    notify();
                },
                setDone: function () {
                    job.status = 'completed';
                    job.progress = 100;
                    notify();
                },
                setError: function (msg) {
                    job.status = 'failed';
                    job.error = msg || 'failed';
                    notify();
                },
                abort: function () {
                    if (job._externalAbort) try { job._externalAbort(); } catch (e) {}
                    job.status = 'failed';
                    job.error = 'Canceled';
                    notify();
                }
            };
        },
        enqueue: function (opts) {
            opts = opts || {};
            if (!opts.file || !opts.url) throw new Error('GlyphUploadQueue.enqueue: file + url required');
            /* tray was replaced by bell — no dismissal state needed */
            var job = {
                id: opts.id || uuid(),
                label: opts.label || opts.file.name,
                file: opts.file,
                url: opts.url,
                headers: opts.headers || {},
                extraFields: opts.extraFields || null,
                onProgress: opts.onProgress || null,
                onDone: opts.onDone || null,
                onError: opts.onError || null,
                status: 'queued',
                progress: 0,
                error: null,
                xhr: null
            };
            jobs.push(job);
            pump();
            return {
                id: job.id,
                abort: function () { api.cancel(job.id); }
            };
        },
        cancel: function (id) {
            var job = jobs.find(function (j) { return j.id === id; });
            if (!job) return;
            if (job.status === 'queued') {
                job.status = 'failed';
                job.error = 'Canceled';
            } else if (job.status === 'active') {
                abortJob(job);
            }
            notify();
        },
        status: groupedStatus,
        clearCompleted: function () {
            jobs = jobs.filter(function (j) { return j.status !== 'completed'; });
            notify();
        },
        subscribe: function (fn) {
            subscribers.push(fn);
            return function unsub() {
                subscribers = subscribers.filter(function (s) { return s !== fn; });
            };
        }
    };
    window.GlyphUploadQueue = api;

    // Inject the bell as soon as the DOM is ready, even with zero jobs, so
    // the icon is always discoverable in the top-right of every page.
    // The topbar is INJECTED by abyss-theme.js post-load; if we don't find
    // .glyph-topbar-right, .top-bar-right on first try, retry briefly so the bell ends up
    // INSIDE the topbar's right cluster instead of floating-and-overlapping.
    function _bootBell() {
        try { ensureBell(); renderTray(); } catch (e) { /* DOM not ready yet */ }
    }
    function _bellRetry() {
        var b = document.getElementById('glyph-upload-bell');
        var topbar = document.querySelector('.glyph-topbar-right, .top-bar-right');
        if (b && b.classList.contains('floating') && topbar) {
            // Topbar appeared after our initial mount — re-parent into it.
            topbar.insertBefore(b, topbar.firstChild);
            b.classList.remove('floating');
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            _bootBell();
            // Retry up to 3s in case the topbar is injected late.
            var tries = 0;
            var iv = setInterval(function () {
                tries++;
                _bellRetry();
                var b = document.getElementById('glyph-upload-bell');
                if ((b && !b.classList.contains('floating')) || tries >= 30) clearInterval(iv);
            }, 100);
        });
    } else {
        _bootBell();
        var tries = 0;
        var iv = setInterval(function () {
            tries++;
            _bellRetry();
            var b = document.getElementById('glyph-upload-bell');
            if ((b && !b.classList.contains('floating')) || tries >= 30) clearInterval(iv);
        }, 100);
    }

    // Re-hydrate session jobs (if there were uploads in flight before a
    // same-tab navigation, the popup reappears with them marked failed —
    // gives the user closure that "no, the upload didn't survive").
    try {
        var snap = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || '[]');
        if (Array.isArray(snap) && snap.length) {
            snap.forEach(function (s) {
                if (s.status === 'active' || s.status === 'queued') {
                    s.status = 'failed';
                    s.error = 'Lost on navigation';
                    s.file = null;
                }
                jobs.push(Object.assign({}, s, { xhr: null }));
            });
            renderTray();
        }
    } catch (e) { /* corrupt snapshot, ignore */ }
})();
