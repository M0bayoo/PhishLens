chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
  const url = tabs[0] && tabs[0].url;
  const v = document.getElementById('verdict');
  if (!url || !/^https?:/.test(url)) { v.textContent = 'Not scoreable'; return; }
  document.getElementById('url').textContent = url;
  chrome.runtime.sendMessage({ type: 'scoreUrl', url }, resp => {
    if (!resp || resp.pLegit == null) { v.textContent = 'Error: ' + (resp && resp.error || 'no score'); return; }
    const p = resp.pLegit;
    let label, color;
    if (p >= 0.6)      { label = '\u2713 Looks legitimate'; color = '#2e7d32'; }
    else if (p >= 0.4) { label = '\u26a0 Caution \u2014 uncertain'; color = '#f9a825'; }
    else               { label = '\u26a0 Likely phishing'; color = '#d32f2f'; }
    v.textContent = `${label} (${(p*100).toFixed(1)}% legit)`;
    v.style.color = color;
    const fill = document.getElementById('fill');
    fill.style.width = Math.round(p*100) + '%';
    fill.style.background = color;
  });
});
