const KW = ['login','signin','verify','secure','account','update','confirm',
            'bank','paypal','password','webscr','wallet'];
let MODEL = null;
async function getModel() {
  if (!MODEL) MODEL = await (await fetch(chrome.runtime.getURL('model/forest_model_e.json'))).json();
  return MODEL;
}
function entropy(s) {
  if (!s) return 0;
  const c = {};
  for (const ch of s) c[ch] = (c[ch] || 0) + 1;
  let e = 0;
  for (const k in c) { const p = c[k] / s.length; e -= p * Math.log2(p); }
  return e;
}
const cnt = (s, ch) => s.split(ch).length - 1;
function lex(u) {
  let p;
  try { p = new URL(u); } catch { return null; }
  if (p.protocol !== 'http:' && p.protocol !== 'https:') return null;
  const host = p.host;
  let path = p.pathname;
  if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^\/?#]*\//.test(u)) path = '';
  const q = p.search ? p.search.slice(1) : '';
  return {
    url_length: u.length, domain_length: host.length,
    path_length: path.length, query_length: q.length,
    num_dots: cnt(u,'.'), num_hyphens: cnt(u,'-'),
    num_underscores: cnt(u,'_'), num_slashes: cnt(u,'/'),
    num_at: cnt(u,'@'), num_question_marks: cnt(u,'?'),
    num_equals: cnt(u,'='), num_ampersands: cnt(u,'&'),
    num_percent: cnt(u,'%'),
    num_digits_in_domain: (host.match(/\d/g)||[]).length,
    num_subdomains: Math.max(cnt(host,'.')-1, 0),
    has_ip_in_url: /^\d{1,3}(\.\d{1,3}){3}$/.test(host.split(':')[0]) ? 1 : 0,
    has_https: p.protocol==='https:' ? 1 : 0,
    has_port: host.includes(':') ? 1 : 0,
    has_double_slash: u.slice(8).includes('//') ? 1 : 0,
    has_at_symbol: u.includes('@') ? 1 : 0,
    has_hex_encoding: u.includes('%') ? 1 : 0,
    has_suspicious_kw: KW.some(k=>u.toLowerCase().includes(k)) ? 1 : 0,
    url_entropy: entropy(u), domain_entropy: entropy(host)
  };
}
function scoreForest(model, x) {
  let sum = 0;
  for (const tr of model.trees) {
    let n = 0;
    while (tr.f[n] !== -2) n = x[tr.f[n]] <= tr.th[n] ? tr.L[n] : tr.R[n];
    sum += tr.v[n];
  }
  return sum / model.trees.length;
}
async function scoreUrl(u) {
  const f = lex(u);
  if (!f) return null;
  const model = await getModel();
  const x = model.features.map(name => f[name]);
  return scoreForest(model, x);
}
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  const url = changeInfo.url || (changeInfo.status === 'loading' ? tab.url : null);
  if (!url || !/^https?:/.test(url)) return;
  try {
    const pLegit = await scoreUrl(url);
    if (pLegit == null) return;
    const band = pLegit >= 0.6 ? 'ok' : (pLegit >= 0.4 ? 'warn' : 'bad');
    chrome.action.setBadgeText({ tabId, text: band==='ok' ? 'OK' : band==='warn' ? '?' : '!' });
    chrome.action.setBadgeBackgroundColor({ tabId, color: band==='ok' ? '#2e7d32' : band==='warn' ? '#f9a825' : '#d32f2f' });
  } catch(e) { console.error('PhishLens scoring failed:', e); }
});
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'scoreUrl') {
    scoreUrl(msg.url).then(p => sendResponse({pLegit: p}))
                     .catch(e => sendResponse({error: String(e)}));
    return true;
  }
});

