/* Abyss app-permission gate (cosmetic client-side RBAC).
 *
 * The REAL enforcement is server-side (require_admin / require_role in
 * fastapi/app/middleware/auth_middleware.py) — every protected API returns 403
 * for an unauthorized role regardless of what the client shows. This script is
 * UX only: it hides apps a user's group can't reach and redirects away from a
 * disallowed page.
 *
 * Contract: preserves the synchronous `checkAppPermission(appId) -> true`
 * signature the app pages call during init, so nothing breaks if this resolves
 * late. The actual allow/deny check runs asynchronously and redirects on deny.
 * Always graceful-degrades (allows) on any error so an API hiccup never locks a
 * legitimate user out — matching the inline behaviour in master-home.html and
 * abyss-theme.js.
 */
(function (global) {
  var _cache = null;     // cached permissions payload for this page load
  var _inflight = null;  // de-dupe concurrent calls

  function getToken() {
    try { return localStorage.getItem('intel_globe_token') || ''; } catch (e) { return ''; }
  }
  function getUsername() {
    try { var u = JSON.parse(localStorage.getItem('intel_globe_user') || 'null'); return u && u.username; }
    catch (e) { return null; }
  }

  function fetchPerms() {
    if (_cache) return Promise.resolve(_cache);
    if (_inflight) return _inflight;
    var username = getUsername();
    var token = getToken();
    if (!username || !token) return Promise.resolve(null);
    // Reuse AbyssAuth.authFetch (Bearer + silent refresh) when present.
    var doFetch = (global.AbyssAuth && global.AbyssAuth.authFetch)
      ? global.AbyssAuth.authFetch
      : function (u) { return fetch(u, { headers: { 'Authorization': 'Bearer ' + token } }); };
    _inflight = doFetch('/api/permissions/' + encodeURIComponent(username))
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { _cache = d; _inflight = null; return d; })
      .catch(function () { _inflight = null; return null; });
    return _inflight;
  }

  function hideAdminFor(data) {
    if (data && data.is_admin) return;
    var nodes = document.querySelectorAll('[data-app="admin"]');
    for (var i = 0; i < nodes.length; i++) nodes[i].style.display = 'none';
  }

  global.checkAppPermission = function (appId) {
    // Non-blocking: resolve permissions, then enforce. Returns true synchronously
    // so existing call sites (`if (!checkAppPermission('x')) ...`) are unaffected.
    fetchPerms().then(function (data) {
      if (!data) return;                 // API unavailable → allow (graceful)
      hideAdminFor(data);
      if (data.is_admin) return;         // admin sees everything
      if (appId === 'admin') { global.location.replace('master-home.html'); return; }
      var allowed = (data.allowed_apps || []).map(function (a) { return a.app_name; });
      if (allowed.length === 0) return;  // no restrictions configured → allow
      if (appId && allowed.indexOf(appId) === -1) {
        global.location.replace('master-home.html');
      }
    });
    return true;
  };
})(window);
