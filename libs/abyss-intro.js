/* ============================================================================
   ABYSS — brand intro ("sonar bloom → Netflix zoom").
   Plays a one-shot welcome over the dashboard: the sounding origin pulses, the
   depth-contour arcs ping outward in sequence (the echo), the wordmark settles,
   then the whole mark scales up and dissolves to reveal the page behind it.

   Wiring:
     login.html  → on successful login: sessionStorage.setItem('abyss_intro','1')
     dashboard   → <script src="/libs/abyss-logo.js"></script>
                   <script src="/libs/abyss-intro.js"></script>   (auto-plays if flagged)
     Manual:     AbyssIntro.play()  /  AbyssIntro.play({force:true})
   Respects prefers-reduced-motion (instant reveal).
   ============================================================================ */
(function (global) {
  'use strict';
  var FLAG = 'abyss_intro';
  var styled = false;

  function injectStyle() {
    if (styled) return; styled = true;
    var css = `
    .abyss-intro{position:fixed;inset:0;z-index:99999;display:grid;place-items:center;overflow:hidden;
      background:radial-gradient(120% 90% at 50% 42%, #0c1422 0%, #070a11 55%, #04060b 100%);
      opacity:1}
    .abyss-intro::before{content:'';position:absolute;inset:-50%;
      background-image:linear-gradient(rgba(var(--accent-rgb,0,245,255),.05) 1px,transparent 1px),linear-gradient(90deg,rgba(var(--accent-rgb,0,245,255),.05) 1px,transparent 1px);
      background-size:54px 54px;-webkit-mask-image:radial-gradient(circle at 50% 42%,#000 0%,transparent 62%);
      mask-image:radial-gradient(circle at 50% 42%,#000 0%,transparent 62%);opacity:.5}
    :root[data-theme="light"] .abyss-intro{
      background:radial-gradient(120% 90% at 50% 42%, #f7fafc 0%, #eef1f6 55%, #e4eaf2 100%)}
    .abyss-intro.done{opacity:0;transition:opacity .45s ease}
    .abyss-intro-stage{position:relative;display:flex;flex-direction:column;align-items:center;gap:18px;
      will-change:transform,opacity;transform:scale(.92);opacity:0}
    .abyss-intro-mark{position:relative;color:var(--accent,#00f5ff);filter:drop-shadow(0 0 26px rgba(var(--accent-rgb,0,245,255),.35))}
    :root[data-theme="light"] .abyss-intro-mark{filter:drop-shadow(0 0 22px rgba(var(--accent-rgb,2,138,158),.25))}
    .abyss-intro-mark svg{width:168px;height:168px}
    /* sonar rings emanate from the sounding origin (top-centre of the mark) */
    .abyss-intro-rings{position:absolute;left:50%;top:33px;transform:translate(-50%,-50%);pointer-events:none}
    .abyss-intro-rings span{position:absolute;left:0;top:0;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;
      border:1.5px solid rgba(var(--accent-rgb,0,245,255),.6);opacity:0}
    .abyss-intro-word{text-align:center;opacity:0;transform:translateY(8px);color:#eafcff}
    :root[data-theme="light"] .abyss-intro-word{color:#0b1526}
    .abyss-intro-word svg{display:block}

    /* --- play timeline (triggered by .playing) --- */
    .abyss-intro.playing{animation:ai-bg 3s cubic-bezier(.16,1,.3,1) forwards}
    .abyss-intro.playing .abyss-intro-stage{animation:ai-stage 3s cubic-bezier(.16,1,.3,1) forwards}
    .abyss-intro.playing .abyss-intro-word{animation:ai-word .7s ease forwards .95s}
    .abyss-intro.playing .abyss-intro-rings span:nth-child(1){animation:ai-ring 1.5s ease-out .15s}
    .abyss-intro.playing .abyss-intro-rings span:nth-child(2){animation:ai-ring 1.6s ease-out .5s}
    .abyss-intro.playing .abyss-intro-rings span:nth-child(3){animation:ai-ring 1.7s ease-out .9s}
    /* origin dot pulse + each echo-arc draws in sequence */
    .abyss-intro .abyss-intro-mark svg > circle:first-of-type{transform-box:fill-box;transform-origin:center;opacity:0}
    .abyss-intro.playing .abyss-intro-mark svg > circle:first-of-type{animation:ai-origin .8s ease forwards}
    .abyss-intro .abyss-intro-mark svg g path{stroke-dasharray:90;stroke-dashoffset:90}
    .abyss-intro.playing .abyss-intro-mark svg g path:nth-child(1){animation:ai-draw .65s ease forwards .25s}
    .abyss-intro.playing .abyss-intro-mark svg g path:nth-child(2){animation:ai-draw .65s ease forwards .45s}
    .abyss-intro.playing .abyss-intro-mark svg g path:nth-child(3){animation:ai-draw .65s ease forwards .65s}
    .abyss-intro.playing .abyss-intro-mark svg g path:nth-child(4){animation:ai-draw .65s ease forwards .85s}

    @keyframes ai-stage{
      0%{transform:scale(.92);opacity:0}
      14%{opacity:1}
      58%{transform:scale(1);opacity:1}
      70%{transform:scale(1.04);opacity:1}
      100%{transform:scale(7);opacity:0}}
    @keyframes ai-bg{0%,66%{opacity:1}100%{opacity:0}}
    @keyframes ai-word{to{opacity:1;transform:translateY(0)}}
    @keyframes ai-origin{0%{opacity:0;transform:scale(.2)}55%{opacity:1;transform:scale(1.35)}100%{opacity:1;transform:scale(1)}}
    @keyframes ai-draw{to{stroke-dashoffset:0}}
    @keyframes ai-ring{0%{opacity:.0;transform:scale(.3)}12%{opacity:.7}100%{opacity:0;transform:scale(13)}}
    @media (prefers-reduced-motion:reduce){
      .abyss-intro.playing{animation:none;opacity:1}
      .abyss-intro.playing .abyss-intro-stage{animation:none;transform:scale(1);opacity:1}
      .abyss-intro.playing .abyss-intro-word{animation:none;opacity:1;transform:none}
      .abyss-intro .abyss-intro-mark svg g path{stroke-dashoffset:0}
      .abyss-intro .abyss-intro-mark svg > circle:first-of-type{opacity:1}}
    `;
    var s = document.createElement('style'); s.id = 'abyss-intro-style'; s.textContent = css;
    document.head.appendChild(s);
  }

  function build() {
    var markSvg = (global.AbyssLogo && AbyssLogo.mark) ? AbyssLogo.mark(168)
      : '<svg viewBox="0 0 56 56" width="168" height="168"></svg>';
    var wordSvg = (global.AbyssLogo && AbyssLogo.wordmark) ? AbyssLogo.wordmark(46)
      : '<span style="font-weight:800;letter-spacing:.2em">ORBION MARITIME</span>';
    var ov = document.createElement('div');
    ov.id = 'abyss-intro'; ov.className = 'abyss-intro'; ov.setAttribute('aria-hidden', 'true');
    ov.innerHTML =
      '<div class="abyss-intro-stage">' +
        '<div class="abyss-intro-mark">' + markSvg +
          '<div class="abyss-intro-rings"><span></span><span></span><span></span></div>' +
        '</div>' +
        '<div class="abyss-intro-word">' + wordSvg + '</div>' +
      '</div>';
    return ov;
  }

  function play(opts) {
    opts = opts || {};
    injectStyle();
    if (document.getElementById('abyss-intro')) return;
    var ov = build();
    (document.body || document.documentElement).appendChild(ov);
    var reduce = global.matchMedia && global.matchMedia('(prefers-reduced-motion:reduce)').matches;
    var hold = reduce ? 500 : 3000;          // total runtime before reveal
    // double-rAF so initial styles commit before the .playing animations start
    requestAnimationFrame(function () { requestAnimationFrame(function () { ov.classList.add('playing'); }); });
    setTimeout(function () {
      ov.classList.add('done');
      setTimeout(function () { if (ov && ov.parentNode) ov.parentNode.removeChild(ov); if (opts.onDone) opts.onDone(); }, 650);
    }, hold);
  }

  function pending() { try { return sessionStorage.getItem(FLAG) === '1'; } catch (e) { return false; } }
  function arm()     { try { sessionStorage.setItem(FLAG, '1'); } catch (e) {} }
  function consume() { try { sessionStorage.removeItem(FLAG); } catch (e) {} }

  global.AbyssIntro = {
    play: function (o) { o = o || {}; if (o.force || pending()) { consume(); play(o); } },
    arm: arm, pending: pending
  };

  // Auto-play once if armed by the login flow.
  if (pending()) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { global.AbyssIntro.play(); });
    else global.AbyssIntro.play();
  }
})(window);
