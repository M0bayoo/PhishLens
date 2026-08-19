chrome.runtime.sendMessage({ type: 'getVerdict' }, resp => {
  const V = document.getElementById('verdict');
  if (!resp || !resp.verdict) { V.textContent = 'Not scoreable'; return; }
  document.getElementById('url').textContent = resp.url || '';
  const band = resp.verdict.band, p = resp.verdict.p;
  let label, cls, color;
  if (band === 'ok')        { label = '\u2713 Looks legitimate';       cls='ok';   color='#2e7d32'; }
  else if (band === 'warn') { label = '\u26a0 Uncertain \u2014 checking'; cls='warn'; color='#f9a825'; }
  else                      { label = '\u26a0 Likely phishing';        cls='bad';  color='#d32f2f'; }
  const pct = (p != null) ? ` (${(p*100).toFixed(1)}% legit)` : '';
  V.textContent = label + pct; V.className = cls;
  const fill = document.getElementById('fill');
  fill.style.width = ((p != null ? p : 0.5)*100) + '%'; fill.style.background = color;

  const d = [];
  if (resp.klass) d.push('URL class: ' + resp.klass);
  d.push('Reason: ' + resp.verdict.reason);
  if (resp.page) {
    if (resp.page.brand) d.push('Page claims brand: ' + resp.page.brand + (resp.page.mismatch ? ' (mismatch!)' : ' (ok)'));
    d.push('Login form: ' + (resp.page.numPwd > 0 ? 'yes' : 'no') + ', forms: ' + resp.page.numForms);
  } else {
    d.push('Page: not yet read (or dead/blank)');
  }
  document.getElementById('detail').innerHTML = d.join('<br>');
  document.getElementById('mode').textContent = 'Fusion mode ' + (resp.mode || '?') + ' \u00b7 runs on-device';
});
