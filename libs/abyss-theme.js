/**
 * GLYPH THEME - Auto-injector for Map 2.0 pages
 *
 * Usage:
 *   <link rel="stylesheet" href="/libs/abyss-theme.css">
 *   <script src="/libs/abyss-theme.js" data-opacity="15"></script>
 *
 * Options (via data attributes on the script tag):
 *   data-opacity  - Globe background opacity: 5|10|15|20|25|30|40|50|60|75|100 (default: 15)
 *   data-no-bg    - Set to "true" to skip the orbital globe background
 *   data-no-topbar - Set to "true" to skip the top bar
 *   data-no-dock  - Set to "true" to skip the topbar app icons
 *   data-no-wrap  - Set to "true" to skip wrapping body content
 */
(function () {
  'use strict';

  // ======================== THEME INIT (runs immediately) ========================
  (function initTheme() {
    // Dark by default; honor a stored light preference (glyph_theme key).
    var t = null;
    try { t = localStorage.getItem('glyph_theme'); } catch (e) {}
    document.documentElement.setAttribute('data-theme', t === 'light' ? 'light' : 'dark');
  })();

  // Shared theme API — pages and the topbar toggle both go through this.
  window.AbyssTheme = {
    get: function () {
      return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    },
    set: function (t) {
      t = t === 'light' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', t);
      try { localStorage.setItem('glyph_theme', t); } catch (e) {}
      try { document.dispatchEvent(new CustomEvent('abyss-theme-change', { detail: { theme: t } })); } catch (e) {}
    },
    toggle: function () {
      window.AbyssTheme.set(window.AbyssTheme.get() === 'light' ? 'dark' : 'light');
    }
  };

  // ======================== LANGUAGE INIT (runs immediately) ========================
  (function initLang() {
    var lang = localStorage.getItem('glyph_lang') || 'en';
    if (lang === 'ar') {
      document.documentElement.setAttribute('dir', 'rtl');
      document.documentElement.setAttribute('lang', 'ar');
    } else {
      document.documentElement.setAttribute('dir', 'ltr');
      document.documentElement.setAttribute('lang', 'en');
    }
  })();

  // ======================== CONFIG ========================
  const scriptTag = document.currentScript || document.querySelector('script[src*="abyss-theme"]');
  const cfg = {
    opacity: (scriptTag && scriptTag.getAttribute('data-opacity')) || '15',
    noBg: scriptTag && scriptTag.hasAttribute('data-no-bg'),
    noTopbar: scriptTag && scriptTag.hasAttribute('data-no-topbar'),
    noDock: scriptTag && scriptTag.hasAttribute('data-no-dock'),
    noWrap: scriptTag && scriptTag.hasAttribute('data-no-wrap'),
  };

  // ======================== HELPERS ========================
  function qs(sel) { return document.querySelector(sel); }
  function qsa(sel) { return document.querySelectorAll(sel); }
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'class') node.className = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    });
    if (children) {
      if (typeof children === 'string') node.innerHTML = children;
      else if (Array.isArray(children)) children.forEach(c => { if (c) node.appendChild(c); });
    }
    return node;
  }

  // Parse stored user info
  function getUserInfo() {
    try {
      const raw = localStorage.getItem('intel_globe_user');
      if (raw) return JSON.parse(raw);
    } catch (e) { /* ignore */ }
    return null;
  }

  function getToken() {
    return localStorage.getItem('intel_globe_token') || '';
  }

  function getInitials(user) {
    if (!user) return '??';
    if (user.display_name) {
      const parts = user.display_name.trim().split(/\s+/);
      return (parts[0][0] + (parts[1] ? parts[1][0] : '')).toUpperCase();
    }
    if (user.username) return user.username.slice(0, 2).toUpperCase();
    return '??';
  }

  function getDisplayName(user) {
    if (!user) return 'User';
    if (user.display_name) return user.display_name;
    if (user.full_name) return user.full_name;
    if (user.username) return user.username;
    return 'User';
  }

  // Detect current page for active dock highlight
  function getCurrentApp() {
    const path = window.location.pathname.split('/').pop() || '';
    const map = {
      'global-browse.html': 'global-browse',
      'map-viewer.html': 'map-viewer',
      'reports-browse.html': 'reports-browse',
      'collections-browse.html': 'collections-browse',
      'projects-browse.html': 'projects-browse',
      'oob-browse.html': 'oob-browse',
      'timelines.html': 'timelines',
      'slideshows.html': 'slideshows',
      'stories-browse.html': 'stories-browse',
      'admin.html': 'admin',
      'master-home.html': 'home',
    };
    return map[path] || '';
  }

  // ======================== SVG ICONS ========================
  const ICONS = {
    globe: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    map: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>',
    file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    briefcase: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/></svg>',
    folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
    crosshair: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/></svg>',
    clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    monitor: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>',
    book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
    gear: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    pin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>',
  };

  // Dock items config: [app-id, label, icon-key, href]
  // Abyss distilled dock — only the apps that ship in the bathymetry pipeline.
  const DOCK_ITEMS = [
    ['collections-browse', 'Collections', 'briefcase', 'collections-browse.html'],
    ['bathymetry-studio', 'CORE', 'pin', 'bathymetry-studio.html'],
    ['reports-browse',     'Reports',     'file',      'reports-browse.html'],
    'sep', // separator
    ['admin',              'Admin',       'gear',      'admin.html'],
  ];

  // ======================== 1. INJECT ORBITAL GLOBE BACKGROUND ========================
  function injectGlobeBg() {
    if (cfg.noBg || document.getElementById('glyph-globe-bg')) return;

    const bg = el('div', { id: 'glyph-globe-bg', 'class': 'glyph-globe-bg', 'data-opacity': cfg.opacity });
    bg.innerHTML = [
      '<div class="glyph-globe-grid"></div>',
      '<div class="glyph-continent glyph-continent-1"></div>',
      '<div class="glyph-continent glyph-continent-2"></div>',
      '<div class="glyph-continent glyph-continent-3"></div>',
      '<div class="glyph-continent glyph-continent-4"></div>',
      '<div class="glyph-continent glyph-continent-5"></div>',
      '<div class="glyph-globe-sphere"></div>',
      '<div class="glyph-orbital-ring glyph-orbital-ring-1"></div>',
      '<div class="glyph-orbital-ring glyph-orbital-ring-2"></div>',
      '<div class="glyph-orbital-ring glyph-orbital-ring-3"></div>',
      '<div class="glyph-orbital-ring glyph-orbital-ring-4"></div>',
      // 12 dots (nth-child offsets start at 7 because of the 6 elements above sphere+rings)
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-dot"></div>',
      '<div class="glyph-globe-ambient"></div>',
    ].join('');

    document.body.insertBefore(bg, document.body.firstChild);
  }

  // ======================== 2. INJECT TOP BAR ========================
  function injectTopbar() {
    if (cfg.noTopbar || document.getElementById('glyph-topbar')) return;

    const user = getUserInfo();
    const initials = getInitials(user);
    const displayName = getDisplayName(user);
    const currentApp = getCurrentApp();

    // Build app icons HTML
    var appsHtml = '';
    DOCK_ITEMS.forEach(function (item) {
      if (item === 'sep') {
        appsHtml += '<div class="glyph-topbar-apps-sep"></div>';
        return;
      }
      var appId = item[0], label = item[1], iconKey = item[2], href = item[3];
      var activeClass = currentApp === appId ? ' active' : '';
      appsHtml += '<a class="glyph-topbar-app-icon' + activeClass + '" href="' + href + '" title="' + label + '" data-app="' + appId + '">' +
        '<div class="glyph-topbar-app-svg">' + ICONS[iconKey] + '</div>' +
        '<span class="glyph-topbar-app-label">' + label + '</span>' +
        '</a>';
    });

    const topbar = el('div', { id: 'glyph-topbar', 'class': 'glyph-topbar' });
    topbar.innerHTML = `
      <a class="glyph-topbar-left" href="master-home.html">
        <span class="glyph-topbar-logo" id="glyph-topbar-logo" aria-label="Orbion Maritime"></span>
        <div class="glyph-classification-badge" id="glyph-classification-badge">UNCLASSIFIED</div>
      </a>
      <div class="glyph-app-dock" id="glyph-app-dock">
        ${appsHtml}
      </div>
      <div class="glyph-topbar-right">
        <button class="glyph-theme-toggle" id="glyph-theme-toggle" title="Toggle light/dark theme" aria-label="Toggle light/dark theme"></button>
        <button class="glyph-lang-toggle" id="glyph-lang-toggle" title="${localStorage.getItem('glyph_lang') === 'ar' ? 'التبديل إلى الإنجليزية' : 'Switch to Arabic (العربية)'}" aria-label="${localStorage.getItem('glyph_lang') === 'ar' ? 'Switch to English' : 'Switch to Arabic'}"><span aria-hidden="true" style="font-size:13px;opacity:.7">🌐</span>&nbsp;${localStorage.getItem('glyph_lang') === 'ar' ? 'EN' : 'AR'}</button>
        <div class="glyph-user-info">
          <div class="glyph-user-avatar">${initials}</div>
          <span>${displayName}</span>
        </div>
        <button class="glyph-btn-logout" id="glyph-signout">Sign Out</button>
      </div>
    `;

    var dockBar = topbar.querySelector('.glyph-app-dock');

    document.body.insertBefore(topbar, document.body.firstChild);

    // Brand: render the Orbion Maritime lockup from the shared logo lib
    // (libs/abyss-logo.js is the single source of truth for the app logo).
    (function renderTopbarLogo() {
      var slot = document.getElementById('glyph-topbar-logo');
      if (!slot) return;
      function paint() {
        if (window.AbyssLogo) slot.innerHTML = window.AbyssLogo.lockup({ size: 28 });
      }
      if (window.AbyssLogo) { paint(); return; }
      var s = document.querySelector('script[src*="abyss-logo.js"]');
      if (!s) {
        s = document.createElement('script');
        s.src = '/libs/abyss-logo.js';
        document.head.appendChild(s);
      }
      s.addEventListener('load', paint);
    })();

    // Dock magnification effect
    initAppDockMagnification(dockBar);

    // Sign out handler — delegate to the shared lib so the server session
    // (HttpOnly cookies + JTI blacklist) is actually invalidated. Falls back to
    // a local clear if abyss-auth.js somehow isn't present.
    document.getElementById('glyph-signout').addEventListener('click', function () {
      if (window.AbyssAuth && typeof window.AbyssAuth.logout === 'function') {
        window.AbyssAuth.logout();
        return;
      }
      localStorage.removeItem('intel_globe_token');
      localStorage.removeItem('intel_globe_user');
      window.location.href = 'login.html';
    });

    // Theme (day/night) toggle handler
    var themeBtn = document.getElementById('glyph-theme-toggle');
    if (themeBtn) {
      themeBtn.addEventListener('click', function () {
        window.AbyssTheme.toggle();
      });
    }

    // Language toggle handler
    var langBtn = document.getElementById('glyph-lang-toggle');
    if (langBtn) {
      langBtn.addEventListener('click', function () {
        var current = localStorage.getItem('glyph_lang') || 'en';
        var next = current === 'ar' ? 'en' : 'ar';
        localStorage.setItem('glyph_lang', next);
        // Full reload to apply dir/lang and re-translate — cleanest approach
        location.reload();
      });
    }

    // Fetch classification + broadcast
    fetchClassificationAndBroadcast();

    // Permission-based filtering of topbar app icons
    applyTopbarPermissions();
  }

  function fetchClassificationAndBroadcast() {
    var badgeEl = document.getElementById('glyph-classification-badge');

    fetch('/api/settings/public')
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data) return;

        // Set classification badge
        if (data.default_classification && badgeEl) {
          var cls = data.default_classification.toUpperCase();
          badgeEl.textContent = cls;
          // Apply color class
          if (cls.indexOf('TOP SECRET') !== -1) {
            badgeEl.className = 'glyph-classification-badge cls-ts';
          } else if (cls.indexOf('SECRET') !== -1) {
            badgeEl.className = 'glyph-classification-badge cls-s';
          } else if (cls.indexOf('CONFIDENTIAL') !== -1) {
            badgeEl.className = 'glyph-classification-badge cls-c';
          } else {
            badgeEl.className = 'glyph-classification-badge cls-u';
          }
        }

        // If there is a broadcast message, inject a thin bar below the topbar
        if (data.broadcast) {
          var existing = document.getElementById('glyph-broadcast-bar');
          if (!existing) {
            var bar = el('div', { id: 'glyph-broadcast-bar', 'class': 'glyph-broadcast-bar' });
            bar.innerHTML = '<span class="glyph-broadcast-live">LIVE</span> ' + data.broadcast;
            var topbar = document.getElementById('glyph-topbar');
            if (topbar && topbar.parentNode) {
              topbar.parentNode.insertBefore(bar, topbar.nextSibling);
            }
          }
        }
      })
      .catch(function () {
        // Silently fail -- keep default text
      });
  }

  function initAppDockMagnification(dock) {
    var items = dock.querySelectorAll('.glyph-topbar-app-icon');
    dock.addEventListener('mousemove', function (e) {
      var mouseX = e.clientX;
      items.forEach(function (item) {
        var rect = item.getBoundingClientRect();
        var center = rect.left + rect.width / 2;
        var distance = Math.abs(mouseX - center);
        var maxDist = 100;
        if (distance < maxDist) {
          var ratio = 1 - distance / maxDist;
          var scale = 1 + ratio * 0.25;
          var ty = ratio * 3;
          item.style.transform = 'scale(' + scale + ') translateY(' + ty + 'px)';
        } else {
          item.style.transform = '';
        }
      });
    });
    dock.addEventListener('mouseleave', function () {
      items.forEach(function (item) { item.style.transform = ''; });
    });
  }

  function applyTopbarPermissions() {
    var appsContainer = document.getElementById('glyph-app-dock');
    if (!appsContainer) return;

    var user = getUserInfo();
    var token = getToken();
    if (!user || !user.username || !token) return;

    fetch('/api/permissions/' + encodeURIComponent(user.username), {
      headers: { 'Authorization': 'Bearer ' + token }
    })
      .then(function (res) {
        if (!res.ok) return null; // 404 or other error — show everything
        return res.json();
      })
      .then(function (data) {
        if (!data) return; // API unavailable — show everything

        var isAdmin = !!data.is_admin;
        var allowedApps = (data.allowed_apps || []).map(function (a) { return a.app_name; });

        // Hide admin icon for non-admins regardless of allowed_apps
        if (!isAdmin) {
          var adminIcon = appsContainer.querySelector('.glyph-topbar-app-icon[data-app="admin"]');
          if (adminIcon) adminIcon.style.display = 'none';
        }

        // If admin, show all — done
        if (isAdmin) return;

        // Empty allowed_apps means no restrictions configured — show all (except admin, handled above)
        if (allowedApps.length === 0) return;

        var allowed = new Set(allowedApps);

        appsContainer.querySelectorAll('.glyph-topbar-app-icon[data-app]').forEach(function (item) {
          var appName = item.getAttribute('data-app');
          if (appName === 'admin') return; // already handled above
          if (!allowed.has(appName)) {
            item.style.display = 'none';
          }
        });
      })
      .catch(function () {
        // Permission check failed — show everything (graceful degradation)
      });
  }

  // ======================== TRANSLATION ENGINE ========================
  var _translations = null;

  function loadTranslations(callback) {
    var lang = localStorage.getItem('glyph_lang') || 'en';
    if (lang === 'en') {
      _translations = null;
      if (callback) callback(null);
      return;
    }
    fetch('/libs/i18n/' + lang + '.json')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        _translations = data;
        if (callback) callback(data);
      })
      .catch(function () {
        _translations = null;
        if (callback) callback(null);
      });
  }

  function applyTranslations(translations) {
    if (!translations) return;
    // Translate all elements with data-i18n attribute
    qsa('[data-i18n]').forEach(function (el) {
      var key = el.getAttribute('data-i18n');
      if (translations[key]) {
        // Check if the element has child elements (icons etc.) — only replace text nodes
        if (el.childElementCount === 0) {
          el.textContent = translations[key];
        } else {
          // Find the last text node and replace it
          var nodes = el.childNodes;
          for (var i = nodes.length - 1; i >= 0; i--) {
            if (nodes[i].nodeType === 3 && nodes[i].textContent.trim()) {
              nodes[i].textContent = ' ' + translations[key];
              break;
            }
          }
        }
      }
    });

    // Translate placeholder attributes
    qsa('[data-i18n-placeholder]').forEach(function (el) {
      var key = el.getAttribute('data-i18n-placeholder');
      if (translations[key]) {
        el.placeholder = translations[key];
      }
    });

    // Translate title attributes
    qsa('[data-i18n-title]').forEach(function (el) {
      var key = el.getAttribute('data-i18n-title');
      if (translations[key]) {
        el.title = translations[key];
      }
    });

    // Translate topbar dock labels (injected by abyss-theme.js, no data-i18n)
    var dockLabelMap = {
      'Browse': translations['nav.browse'],
      'Reports': translations['nav.reports'],
      'Collections': translations['nav.collections'],
      'Projects': translations['nav.projects'],
      'OOB': translations['nav.oob'],
      'Regions': translations['nav.regions'],
      'Stories': translations['nav.stories'],
      'Timelines': translations['nav.timelines'],
      'Slideshows': translations['nav.slideshows'],
      'Admin': translations['nav.admin'],
    };
    qsa('.glyph-topbar-app-label').forEach(function (label) {
      var text = label.textContent.trim();
      if (dockLabelMap[text]) label.textContent = dockLabelMap[text];
    });

    // Translate the sign-out button
    var signOutBtn = document.getElementById('glyph-signout');
    if (signOutBtn && translations['nav.signout']) {
      signOutBtn.textContent = translations['nav.signout'];
    }
  }

  // Expose globally for pages to use after dynamic content loads
  window.GlyphI18n = {
    t: function (key) {
      return (_translations && _translations[key]) || key;
    },
    translations: function () { return _translations; },
    apply: function () { applyTranslations(_translations); },
    isRTL: function () { return document.documentElement.getAttribute('dir') === 'rtl'; }
  };

  // ======================== 3. WRAP PAGE CONTENT ========================
  function wrapPageContent() {
    if (cfg.noWrap || document.querySelector('.glyph-page-content')) return;

    // If page has an .app container with its own grid/flex layout, DON'T wrap —
    // just add the animation class and let CSS handle spacing
    var appEl = document.querySelector('.app');
    if (appEl) {
      appEl.classList.add('glyph-page-enter');
      return;
    }

    // Collect body children that are NOT theme elements
    var themeIds = new Set(['glyph-globe-bg', 'glyph-topbar', 'glyph-broadcast-bar']);
    var children = Array.from(document.body.children).filter(function (child) {
      // Skip theme elements, scripts, links
      if (themeIds.has(child.id)) return false;
      if (child.tagName === 'SCRIPT') return false;
      if (child.tagName === 'LINK') return false;
      if (child.tagName === 'STYLE') return false;
      return true;
    });

    if (children.length === 0) return;

    var wrapper = el('div', { 'class': 'glyph-page-content glyph-page-enter' });
    // Insert wrapper before the first content child
    children[0].parentNode.insertBefore(wrapper, children[0]);
    children.forEach(function (child) {
      wrapper.appendChild(child);
    });
  }

  // ======================== 4. HIDE EXISTING NAV/HEADER ========================
  function hideExistingNav() {
    // Common selectors for navigation that the theme replaces
    var selectors = [
      'header:not(.glyph-topbar)',
      'nav:not(.glyph-dock)',
      '.header',
      '.nav-bar',
      '.top-nav',
      '.navbar',
      '.site-header',
      '.app-header',
    ];

    selectors.forEach(function (sel) {
      qsa(sel).forEach(function (node) {
        // Don't hide if it's inside the glyph wrapper or is a glyph element
        if (node.closest('.glyph-topbar') || node.closest('.glyph-dock')) return;
        if (node.id && node.id.startsWith('glyph-')) return;
        node.style.display = 'none';
      });
    });
  }

  // ======================== INIT ========================
  function init() {
    // Add body class for base styling. Both names are applied: the CSS layout
    // rules (topbar offset, .content-area, etc.) are authored as
    // `body.glyph-themed`, so this class must be present or the injected 60px
    // topbar overlaps page content (blank/squashed collections & reports).
    document.body.classList.add('abyss-themed');
    document.body.classList.add('glyph-themed');

    // Injection order matters: bg first (behind everything), topbar, then wrap content
    injectGlobeBg();
    injectTopbar();
    // Only hide existing nav when we injected a topbar to replace it
    if (!cfg.noTopbar) {
      hideExistingNav();
    }
    wrapPageContent();

    // Load translations and apply to data-i18n elements
    loadTranslations(function (data) {
      if (data) applyTranslations(data);
    });
  }

  // Run when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // ======================== VERSION CHECK & AUTO-RELOAD ========================
  (function() {
    var _appVersion = null;
    var _versionUrl = '/libs/version.json';

    function checkVersion() {
      try {
        fetch(_versionUrl + '?_t=' + Date.now())
          .then(function(r) { return r.json(); })
          .then(function(data) {
            if (_appVersion === null) {
              _appVersion = data.version;
            } else if (data.version !== _appVersion) {
              location.reload(true);
            }
          })
          .catch(function() {});
      } catch(e) {}
    }

    window.addEventListener('load', function() {
      checkVersion();
      setInterval(checkVersion, 30000);
    });
  })();

  // Shared auth lib — ensure AbyssAuth is available on every themed page so the
  // sign-out handler (and any page code) can invalidate the server session.
  // Idempotent; pages may also include it explicitly before abyss-theme.js.
  (function loadAuthLib() {
    if (window.AbyssAuth) return;
    if (document.querySelector('script[src*="abyss-auth.js"]')) return;
    var s = document.createElement('script');
    s.src = '/libs/abyss-auth.js';
    document.head.appendChild(s);
  })();

  // Universal upload-queue bell — every page that loads abyss-theme.js gets
  // the bell + popup. Idempotent: the queue lib's IIFE no-ops if already
  // loaded. Pages can still <script src="..."> it explicitly without
  // breaking anything (the lib guards on window.GlyphUploadQueue).
  (function loadUploadQueue() {
    if (window.GlyphUploadQueue) return;
    if (document.querySelector('script[src*="glyph-upload-queue.js"]')) return;
    var s = document.createElement('script');
    s.src = '/libs/glyph-upload-queue.js';
    s.async = true;
    document.head.appendChild(s);
  })();

})();
