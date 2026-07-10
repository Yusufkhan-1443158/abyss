/**
 * AbyssAuth — shared client auth/session for Abyss apps.
 *
 * Single source of truth for token storage, silent refresh, identity load,
 * and logout. Supersedes the duplicated inline checkAuth/loadUser/logout in the
 * app pages and the localStorage-only sign-out handler in abyss-theme.js.
 *
 *   <script src="/libs/abyss-auth.js"></script>
 *   if (!AbyssAuth.checkAuth()) return;          // redirect to login if no session
 *   AbyssAuth.startAutoRefresh();                // keep the access token fresh
 *   AbyssAuth.loadUser();                        // fill avatar/name from /api/auth/me
 *   const res = await AbyssAuth.authFetch('/api/reports');   // Bearer + auto-refresh
 *   AbyssAuth.logout();                          // POST /api/auth/logout then redirect
 *
 * Token model: the access JWT lives BOTH in localStorage['intel_globe_token']
 * (read directly by legacy app code for Bearer headers + expiry checks) AND in
 * an HttpOnly cookie set by the server. The refresh token is an HttpOnly cookie
 * scoped to /api/auth and is never exposed to JS — so refresh() simply POSTs to
 * /api/auth/refresh with credentials:'include' and the browser supplies it.
 */
(function (global) {
  'use strict';

  var TOKEN_KEY    = 'intel_globe_token';
  var USER_KEY     = 'intel_globe_user';
  var REMEMBER_KEY = 'intel_globe_remember';
  var ALIVE_KEY    = 'intel_globe_alive';   // sessionStorage sentinel (per browser session)
  var LOGOUT_KEY   = 'abyss_logged_out';    // sessionStorage flag set on explicit logout
  var LOGIN_PAGE   = 'login.html';

  // ---- storage helpers ------------------------------------------------
  function getToken() {
    try { return localStorage.getItem(TOKEN_KEY) || ''; } catch (e) { return ''; }
  }
  function getUser() {
    try { var raw = localStorage.getItem(USER_KEY); return raw ? JSON.parse(raw) : null; }
    catch (e) { return null; }
  }
  function clearSession() {
    try { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY); } catch (e) {}
  }
  /**
   * Persist a session. token/user may be null to update only one of them.
   * `remember` (boolean) records whether the session should survive the browser
   * session closing; omitted = leave the existing preference untouched.
   */
  function setSession(token, user, remember) {
    try {
      if (token) localStorage.setItem(TOKEN_KEY, token);
      if (user) localStorage.setItem(USER_KEY, typeof user === 'string' ? user : JSON.stringify(user));
      if (typeof remember === 'boolean') localStorage.setItem(REMEMBER_KEY, remember ? '1' : '0');
      sessionStorage.setItem(ALIVE_KEY, '1');
    } catch (e) { /* storage disabled — ignore */ }
  }

  // ---- identity helpers (lifted from abyss-theme.js:81-97) ------------
  function getInitials(user) {
    if (!user) return '??';
    if (user.display_name) {
      var parts = user.display_name.trim().split(/\s+/);
      return (parts[0][0] + (parts[1] ? parts[1][0] : '')).toUpperCase();
    }
    if (user.username) return user.username.slice(0, 2).toUpperCase();
    return '??';
  }
  function getDisplayName(user) {
    if (!user) return 'User';
    return user.display_name || user.full_name || user.username || 'User';
  }

  // ---- JWT expiry (pattern from reports-browse.html:1668-1669) --------
  function decodePayload(token) {
    try { return JSON.parse(atob(token.split('.')[1])); } catch (e) { return null; }
  }
  // A real access JWT has 3 dot-separated parts. Anything else ('dev', 'cookie',
  // empty) is a cookie-backed sentinel — auth rides the HttpOnly cookie (real
  // session) or the DEV_NO_AUTH bypass, so no Bearer header is sent for it.
  function isSentinel(token) {
    return !token || token.split('.').length !== 3;
  }
  function isExpired(token, skewSeconds) {
    token = token || getToken();
    if (!token) return true;
    if (isSentinel(token)) return false;      // cookie-backed → server validates the cookie
    var p = decodePayload(token);
    if (!p || !p.exp) return false;           // opaque / no exp → let the server decide
    var skew = (skewSeconds == null ? 30 : skewSeconds);
    return (p.exp * 1000) < (Date.now() + skew * 1000);
  }

  // ---- refresh (single-flight) ----------------------------------------
  var _refreshPromise = null;
  function refresh() {
    if (_refreshPromise) return _refreshPromise;            // coalesce concurrent 401s
    _refreshPromise = fetch('/api/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',                                // sends the /api/auth refresh cookie
      body: '{}'
    }).then(function (res) {
      if (!res.ok) throw new Error('refresh failed: ' + res.status);
      return res.json();
    }).then(function (data) {
      var token = data.access_token || data.token;
      if (token) {
        var p = decodePayload(token) || {};
        var existing = getUser() || {};
        setSession(token, Object.assign({}, existing, {
          username: p.username || existing.username,
          role: p.role || existing.role
        }));
      }
      return token;
    });
    // clear the in-flight handle whether it resolves or rejects
    _refreshPromise.then(function () { _refreshPromise = null; }, function () { _refreshPromise = null; });
    return _refreshPromise;
  }

  // ---- authFetch: Bearer + credentials, one transparent refresh on 401 -
  function authFetch(url, opts) {
    opts = opts || {};
    function doFetch(tok) {
      var headers = Object.assign({}, opts.headers || {});
      if (!isSentinel(tok)) headers['Authorization'] = 'Bearer ' + tok;  // else rely on the cookie
      return fetch(url, Object.assign({}, opts, { headers: headers, credentials: 'include' }));
    }
    return doFetch(getToken()).then(function (res) {
      if (res.status !== 401) return res;
      return refresh().then(function (newTok) {
        return doFetch(newTok || getToken());                // retry once with the fresh token
      }).catch(function () {
        return res;                                          // refresh failed → surface the 401
      });
    });
  }

  // ---- session guards --------------------------------------------------
  function onLoginPage() {
    return (location.pathname.split('/').pop() || '') === LOGIN_PAGE;
  }
  function redirectToLogin() {
    if (!onLoginPage()) location.href = LOGIN_PAGE;
  }

  /**
   * Ensure a usable session exists, repairing a cookie-only session (valid
   * HttpOnly cookies but no localStorage token) via a silent refresh instead of
   * bouncing to login. Resolves true if authenticated, false after redirecting.
   */
  function ensureSession() {
    var token = getToken();
    if (token && !isExpired(token)) return Promise.resolve(true);
    return refresh().then(function (tok) {
      if (tok) return true;
      clearSession(); redirectToLogin(); return false;
    }).catch(function () {
      clearSession(); redirectToLogin(); return false;
    });
  }

  /**
   * Synchronous guard preserving existing `if (!checkAuth()) return;` call sites.
   * - valid token  → true (proceed)
   * - expired token → true, refresh in the background (authFetch also recovers)
   * - no token      → false; repair a cookie session (then reload) or redirect
   */
  function checkAuth() {
    var token = getToken();
    if (!token) {
      ensureSession().then(function (ok) { if (ok) location.reload(); });
      return false;
    }
    if (isExpired(token)) refresh().catch(function () {});
    return true;
  }

  // ---- identity load ---------------------------------------------------
  function fillIdentity(user, nameSel) {
    var name = getDisplayName(user);
    var initials = getInitials(user);
    var nameNodes = [];
    if (nameSel) nameNodes = nameNodes.concat(Array.prototype.slice.call(document.querySelectorAll(nameSel)));
    nameNodes = nameNodes.concat(Array.prototype.slice.call(document.querySelectorAll('[data-abyss-user-name]')));
    nameNodes.forEach(function (n) { n.textContent = name; });
    document.querySelectorAll('[data-abyss-user-initials]').forEach(function (n) { n.textContent = initials; });
    document.querySelectorAll('[data-abyss-user-role]').forEach(function (n) {
      if (user && user.role) n.textContent = user.role;
    });
  }

  /** Refresh canonical identity from /api/auth/me and fill the UI. */
  function loadUser(nameSel) {
    return authFetch('/api/auth/me').then(function (res) {
      return res.ok ? res.json() : null;
    }).then(function (user) {
      if (user) {
        var existing = getUser() || {};
        setSession(null, Object.assign({}, existing, user));
        fillIdentity(user, nameSel || '#userDisplay');
      } else {
        fillIdentity(getUser(), nameSel || '#userDisplay');     // fall back to cached identity
      }
      return user;
    }).catch(function () {
      fillIdentity(getUser(), nameSel || '#userDisplay');
      return null;
    });
  }

  // ---- logout (the real end-to-end flow) ------------------------------
  function logout() {
    return authFetch('/api/auth/logout', { method: 'POST' })
      .catch(function () { /* best-effort: clear client state regardless */ })
      .then(function () {
        clearSession();
        try { sessionStorage.setItem(LOGOUT_KEY, '1'); } catch (e) {}
        location.href = LOGIN_PAGE;
      });
  }

  // ---- proactive auto-refresh -----------------------------------------
  var _autoTimer = null;
  function startAutoRefresh() {
    if (_autoTimer) return;
    function schedule() {
      var p = decodePayload(getToken());
      var delay = 4 * 60 * 1000;                              // default cadence if exp unknown
      if (p && p.exp) delay = Math.max(30000, (p.exp * 1000) - Date.now() - 120000); // 2 min early
      _autoTimer = setTimeout(function () {
        refresh().catch(function () {}).then(schedule);
      }, delay);
    }
    schedule();
  }

  // ---- remember-me enforcement (runs once at load) --------------------
  // If the session was created with "remember me" unchecked and the browser
  // session sentinel is gone (browser/tab session was closed and reopened),
  // drop the stored token so the user must log in again. Documented trade-off:
  // a brand-new tab during an ephemeral session lacks the per-tab sentinel and
  // will re-prompt.
  (function enforceRemember() {
    try {
      if (!localStorage.getItem(TOKEN_KEY)) return;
      var remember = localStorage.getItem(REMEMBER_KEY);
      if (remember === '0' && !sessionStorage.getItem(ALIVE_KEY)) {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
      } else {
        sessionStorage.setItem(ALIVE_KEY, '1');
      }
    } catch (e) { /* ignore */ }
  })();

  global.AbyssAuth = {
    getToken: getToken,
    getUser: getUser,
    getInitials: getInitials,
    getDisplayName: getDisplayName,
    decodePayload: decodePayload,
    isExpired: isExpired,
    setSession: setSession,
    clearSession: clearSession,
    refresh: refresh,
    authFetch: authFetch,
    ensureSession: ensureSession,
    checkAuth: checkAuth,
    loadUser: loadUser,
    fillIdentity: fillIdentity,
    logout: logout,
    startAutoRefresh: startAutoRefresh
  };
})(window);
