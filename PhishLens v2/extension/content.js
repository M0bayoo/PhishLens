// content.js — PhishLens phase 2 page reader.
// Extracts what a HUMAN sees (computed-style filtered) + credential-UI signals,
// then hands identity text to the background worker for brand-consistency judging.

(function () {
  function isVisible(el) {
    const st = window.getComputedStyle(el);
    if (!st) return false;
    if (st.display === 'none' || st.visibility === 'hidden') return false;
    if (parseFloat(st.opacity || '1') === 0) return false;
    const fs = parseFloat(st.fontSize || '16');
    if (fs && fs < 4) return false;                       // zero/near-zero font = hidden junk
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return false;     // collapsed
    if (r.bottom < -2000 || r.right < -2000) return false; // shoved far off-screen
    return true;
  }

  function visibleText(el) {
    if (!el || !isVisible(el)) return '';
    let out = '';
    for (const node of el.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) out += ' ' + node.textContent;
      else if (node.nodeType === Node.ELEMENT_NODE) out += ' ' + visibleText(node);
    }
    return out;
  }

  // IDENTITY text only: what the page presents itself AS (not body content).
  function identityText() {
    const bits = [];
    if (document.title) bits.push(document.title);
    const og = document.querySelector('meta[property="og:site_name"]');
    if (og && og.content) bits.push(og.content);
    const ogt = document.querySelector('meta[property="og:title"]');
    if (ogt && ogt.content) bits.push(ogt.content);
    document.querySelectorAll('h1').forEach(h => { if (isVisible(h)) bits.push(visibleText(h)); });
    // logo alt-text: common places brands name themselves
    document.querySelectorAll(
      'img[alt*="logo" i], a[class*="logo" i] img, header img[alt], [id*="logo" i] img'
    ).forEach(img => { if (img.alt) bits.push(img.alt); });
    return bits.join(' ').replace(/\s+/g, ' ').trim().slice(0, 400);
  }

  function domSignals() {
    let hidden = 0;
    const all = document.querySelectorAll('*');
    for (let i = 0; i < all.length && hidden <= 500; i++) {
      const st = window.getComputedStyle(all[i]);
      if (st && (st.display === 'none' || st.visibility === 'hidden')) hidden++;
    }
    const pwd = document.querySelectorAll('input[type="password"]').length;
    const forms = document.querySelectorAll('form').length;
    const iframes = document.querySelectorAll('iframe').length;
    // external resource ratio (same definition as Python script 14)
    let ext = 0, total = 0;
    const here = location.host.toLowerCase();
    document.querySelectorAll('script[src],link[href],img[src]').forEach(el => {
      const u = el.src || el.href; if (!u) return;
      let d; try { d = new URL(u, location.href).host.toLowerCase(); } catch { return; }
      if (!d) return; total++;
      if (d !== here && !d.endsWith('.' + here)) ext++;
    });
    return {
      numForms: forms, numPwd: pwd, iframeCount: iframes, hiddenCount: hidden,
      extRatio: total ? +(ext / total).toFixed(4) : 0
    };
  }

  function report() {
    try {
      const payload = Object.assign(
        { type: 'pageInfo', url: location.href, identity: identityText() },
        domSignals()
      );
      chrome.runtime.sendMessage(payload);
    } catch (e) { /* extension context may be gone; ignore */ }
  }

  // Report after load, and once more shortly after (SPAs fill the DOM late).
  if (document.readyState === 'complete') report();
  else window.addEventListener('load', report);
  setTimeout(report, 1500);
})();
