// background.js â€” PhishLens phase 1 (gate + E5 forest) + phase 2 fusion.
// ================= FUSION MODE TOGGLE =================
// 'A' = cautious : page brand-mismatch -> AMBER by default; RED only if phase-1 also flags it
// 'B' = assertive: page brand-mismatch -> RED  by default; reinforced if phase-1 also flags it
const FUSION_MODE = 'A';   // change to 'B' and reload to test the other policy
// =====================================================

// phishlens_core.js â€” ports Python scripts 34/35 to JS. Parity-tested before deploy.

const BRANDS = {
 'paypal':['paypal.com','paypal.co.uk','paypal.me'],
 'amazon':['amazon.com','amazon.co.uk','amazon.de','amazon.fr','amazon.it','amazon.es',
           'amazon.ca','amazon.co.jp','amazon.in','amazon.com.br','amazon.com.au'],
 'apple':['apple.com','icloud.com'], 'icloud':['icloud.com','apple.com'],
 'microsoft':['microsoft.com','live.com','office.com','outlook.com','microsoftonline.com'],
 'outlook':['outlook.com','live.com','microsoft.com'],
 'onedrive':['onedrive.live.com','microsoft.com'],
 'google':['google.com','google.co.uk','goo.gl','youtube.com','gmail.com','googleusercontent.com'],
 'gmail':['gmail.com','google.com'],
 'facebook':['facebook.com','fb.com'], 'instagram':['instagram.com'], 'whatsapp':['whatsapp.com'],
 'netflix':['netflix.com'], 'spotify':['spotify.com'], 'linkedin':['linkedin.com'],
 'allegro':['allegro.pl'], 'allegrolokalnie':['allegrolokalnie.pl'],
 'shopee':['shopee.com','shopee.co.id','shopee.sg','shopee.tw','shopee.com.br','shopee.vn'],
 'ebay':['ebay.com','ebay.co.uk','ebay.de'], 'aliexpress':['aliexpress.com'],
 'dhl':['dhl.com','dhl.de'], 'fedex':['fedex.com'], 'usps':['usps.com'],
 'royalmail':['royalmail.com'], 'dpd':['dpd.com','dpd.co.uk'], 'evri':['evri.com'],
 'hsbc':['hsbc.com','hsbc.co.uk'], 'barclays':['barclays.co.uk'], 'natwest':['natwest.com'],
 'lloyds':['lloydsbank.com'], 'santander':['santander.co.uk','santander.com'],
 'monzo':['monzo.com'], 'revolut':['revolut.com'], 'halifax':['halifax.co.uk'],
 'nationwide':['nationwide.co.uk'], 'chase':['chase.com'], 'wellsfargo':['wellsfargo.com'],
 'citibank':['citi.com','citibank.com'], 'binance':['binance.com'], 'coinbase':['coinbase.com'],
 'metamask':['metamask.io'], 'dropbox':['dropbox.com'], 'wetransfer':['wetransfer.com'],
 'docusign':['docusign.com'], 'adobe':['adobe.com'], 'steam':['steampowered.com'],
 'roblox':['roblox.com'], 'hmrc':['hmrc.gov.uk'], 'dvla':['dvla.gov.uk'],
 'nhs':['nhs.uk'], 'tesco':['tesco.com'], 'argos':['argos.co.uk'],
};
const LEGIT_DOMAINS = new Set();
for (const b in BRANDS) for (const d of BRANDS[b]) LEGIT_DOMAINS.add(d);

const PLATFORM = ['github.io','netlify.app','wixsite.com','wixstudio.com','firebaseapp.com',
 'herokuapp.com','vercel.app','pages.dev','weebly.com','glitch.me','amazonaws.com',
 'cloudfront.net','googleusercontent.com','windows.net','web.app','workers.dev','r2.dev',
 'repl.co','myshopify.com','square.site','000webhostapp.com','duckdns.org','sharepoint.com',
 'blogspot.com','wordpress.com','sites.google.com','translate.goog','azurewebsites.net'];
const HIGH_RISK_TLD = new Set(['buzz','top','cfd','sbs','xyz','tk','ml','ga','cf','gq','icu','click','link',
 'work','rest','fit','loan','date','racing','win','bid','stream','download',
 'pro','online','site','website','space','store','shop','life','world']);
const MED_RISK_TLD = new Set(['info','biz','us','cc','me','io','app','dev','page','one','today','digital','live']);
const MULTI_SUFFIX = new Set(['co.uk','org.uk','ac.uk','gov.uk','net.uk','com.br','com.au','co.jp','co.nz',
 'com.tr','co.in','com.mx','com.ar','co.za','com.pl','net.au','org.au','co.id','com.sg',
 'com.vn','co.kr','com.cn','co.il','com.hk','com.tw','co.th','com.my','co.ke']);
const KW = ['login','signin','verify','secure','account','update','confirm',
            'bank','paypal','password','webscr','wallet'];

function regDomain(host) {
  host = host.toLowerCase().split(':')[0].replace(/^\.+|\.+$/g,'');
  const parts = host.split('.');
  if (parts.length <= 2) return host;
  if (MULTI_SUFFIX.has(parts.slice(-2).join('.')) && parts.length >= 3) return parts.slice(-3).join('.');
  return parts.slice(-2).join('.');
}
function toks(s) { return s.toLowerCase().split(/[.\-_]/).filter(t => t); }
function lev(a, b) {
  if (Math.abs(a.length - b.length) > 2) return 99;
  let prev = Array.from({length: b.length + 1}, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur.push(Math.min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (a[i-1] !== b[j-1] ? 1 : 0)));
    }
    prev = cur;
  }
  return prev[b.length];
}
function hostOf(u) { try { return new URL(u).host.toLowerCase().split(':')[0]; } catch { return ''; } }
function partsOf(u) {
  let p; try { p = new URL(u); } catch { return null; }
  const host = p.host.toLowerCase().split(':')[0];
  let path = p.pathname;
  if (!/^[a-zA-Z][a-zA-Z0-9+.\-]*:\/\/[^/?#]*\//.test(u)) path = '';
  const q = p.search ? p.search.slice(1) : '';
  return { p, host, path, q, scheme: p.protocol.replace(':','') };
}

function brandSignals(u) {
  const pp = partsOf(u);
  if (!pp || !pp.host) return {b_host:0,b_mis:0,b_sub:0,b_look:0,b_path:0};
  const host = pp.host, rd = regDomain(host), rdFirst = rd.split('.')[0], rdToks = toks(rdFirst);
  const sub = host.endsWith(rd) ? host.slice(0, host.length - rd.length).replace(/\.+$/,'') : '';
  const subToks = toks(sub);
  const pathQ = (pp.path + '?' + pp.q).toLowerCase();
  for (const b in BRANDS) {
    const legit = BRANDS[b];
    if (rdToks.includes(b)) {
      if (legit.some(d => rd === d || rd.endsWith('.'+d))) return {b_host:1,b_mis:0,b_sub:0,b_look:0,b_path:0};
      if (rdFirst === b) return {b_host:1,b_mis:0,b_sub:0,b_look:0,b_path:0};
      return {b_host:1,b_mis:1,b_sub:0,b_look:0,b_path:0};
    }
    if (subToks.includes(b)) {
      if (legit.some(d => rd === d || rd.endsWith('.'+d))) return {b_host:1,b_mis:0,b_sub:0,b_look:0,b_path:0};
      return {b_host:1,b_mis:1,b_sub:1,b_look:0,b_path:0};
    }
  }
  for (const b in BRANDS) {
    if (b.length < 5) continue;
    for (const t of rdToks.concat(subToks)) {
      if (t.length >= 5 && t !== b && lev(t, b) <= 1) return {b_host:1,b_mis:1,b_sub:0,b_look:1,b_path:0};
    }
  }
  for (const b in BRANDS) {
    if (b.length >= 5 && pathQ.includes(b)) return {b_host:0,b_mis:0,b_sub:0,b_look:0,b_path:1};
  }
  return {b_host:0,b_mis:0,b_sub:0,b_look:0,b_path:0};
}
function brandIsLegit(u) {
  const rd = regDomain(hostOf(u));
  if (!rd) return 0;
  for (const d of LEGIT_DOMAINS) if (rd === d || rd.endsWith('.'+d)) return 1;
  return 0;
}
function tldRisk(u) {
  const host = hostOf(u); const tld = host.includes('.') ? host.split('.').pop() : '';
  if (HIGH_RISK_TLD.has(tld)) return 2;
  if (MED_RISK_TLD.has(tld)) return 1;
  return 0;
}
function isPlatform(u) {
  const rd = regDomain(hostOf(u));
  return PLATFORM.some(d => rd === d || rd.endsWith('.'+d));
}
function entropy(s) {
  if (!s) return 0;
  const c = {}; for (const ch of s) c[ch] = (c[ch]||0)+1;
  let e = 0; for (const k in c) { const p = c[k]/s.length; e -= p*Math.log2(p); }
  return e;
}
const cnt = (s, ch) => s.split(ch).length - 1;

function lex(u) {
  u = String(u); const pp = partsOf(u);
  const host = pp ? pp.host : '', path = pp ? pp.path : '', q = pp ? pp.q : '';
  const bs = brandSignals(u);
  return {
    url_length:u.length, domain_length:host.length, path_length:path.length, query_length:q.length,
    num_dots:cnt(u,'.'), num_hyphens:cnt(u,'-'), num_underscores:cnt(u,'_'), num_slashes:cnt(u,'/'),
    num_at:cnt(u,'@'), num_question_marks:cnt(u,'?'), num_equals:cnt(u,'='), num_ampersands:cnt(u,'&'),
    num_percent:cnt(u,'%'), num_digits_in_domain:(host.match(/\d/g)||[]).length,
    num_subdomains:Math.max(cnt(host,'.')-1,0),
    has_ip_in_url:/^\d{1,3}(\.\d{1,3}){3}$/.test(host.split(':')[0])?1:0,
    has_https:(pp&&pp.scheme==='https')?1:0, has_port:host.includes(':')?1:0,
    has_double_slash:u.slice(8).includes('//')?1:0, has_at_symbol:u.includes('@')?1:0,
    has_hex_encoding:u.includes('%')?1:0,
    has_suspicious_kw:KW.some(k=>u.toLowerCase().includes(k))?1:0,
    url_entropy:entropy(u), domain_entropy:entropy(host),
    brand_in_host:bs.b_host, brand_domain_mismatch:bs.b_mis, brand_in_subdomain:bs.b_sub,
    brand_lookalike:bs.b_look, brand_in_path:bs.b_path, brand_is_legit:brandIsLegit(u),
    tld_risk:tldRisk(u), reg_domain_length:regDomain(host).length, is_platform:isPlatform(u)?1:0,
    subdomain_depth:Math.max(host.split('.').length - regDomain(host).split('.').length, 0)
  };
}

function urlClass(u) {
  const pp = partsOf(u); if (!pp) return 'ORDINARY';
  const host = pp.host; const bs = brandSignals(u);
  if (isPlatform(u)) return 'PLATFORM';
  if (bs.b_mis) return 'BRAND_MISMATCH';
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(host) || u.includes('@') || host.includes('xn--')) return 'STRUCTURAL';
  if (brandIsLegit(u)) return 'KNOWN_BRAND';
  if (pp.q.length > 40 || entropy(u) > 4.6) return 'LONG_COMPLEX';
  if (pp.path.includes('%')) return 'ENCODED';
  if (host.split('.').length <= 2 && pp.path.replace(/\//g,'').length === 0) return 'BARE_ROOT';
  return 'ORDINARY';
}
function gate(u, pLegit) {
  const k = urlClass(u);
  if (k === 'BRAND_MISMATCH' || k === 'STRUCTURAL') return ['RED_RULE', k];
  if (k === 'KNOWN_BRAND') return ['ACCEPT_GUARD', k];
  if (k === 'LONG_COMPLEX' || k === 'PLATFORM' || k === 'ENCODED') return ['ESCALATE', k];
  if (k === 'BARE_ROOT' && tldRisk(u) >= 2) return ['ESCALATE', k];
  return [(pLegit >= 0.5 ? 'ACCEPT' : 'RED_URL'), k];
}

// ---------- E5 forest (loaded from model/forest_model_e.json) ----------
let MODEL = null;
async function getModel() {
  if (!MODEL) MODEL = await (await fetch(chrome.runtime.getURL('model/forest_model_e.json'))).json();
  return MODEL;
}
function scoreForest(model, feat) {
  const x = model.features.map(n => feat[n]);
  let sum = 0;
  for (const tr of model.trees) {
    let n = 0;
    while (tr.f[n] !== -2) n = x[tr.f[n]] <= tr.th[n] ? tr.L[n] : tr.R[n];
    sum += tr.v[n];
  }
  return sum / model.trees.length;   // P(legit)
}

// ---------- phase-2 page-brand detection ----------
function detectPageBrand(identityText) {
  const t = ' ' + String(identityText).toLowerCase().replace(/[^a-z0-9]+/g, ' ') + ' ';
  const hits = [];
  for (const b in BRANDS) { if (b.length >= 4 && t.includes(' ' + b + ' ')) hits.push(b); }
  return hits.length === 1 ? hits[0] : null;   // 0 or >1 brands named = no single claim
}
function pageBrandMismatch(url, identityText, hasPassword) {
  const brand = detectPageBrand(identityText);
  if (!brand) return { brand: null, mismatch: false };
  let rd; try { rd = regDomain(new URL(url).host); } catch { return { brand, mismatch: false }; }
  const ok = BRANDS[brand].some(d => rd === d || rd.endsWith('.' + d));
  return { brand, mismatch: (!ok && !!hasPassword) };   // credential UI required
}

// ---------- verdict model ----------
// bands: 'ok' (green), 'warn' (amber), 'bad' (red)
async function phase1(url) {
  const model = await getModel();
  const t0 = performance.now(); const p = scoreForest(model, lex(url)); const t1 = performance.now(); console.log('[PhishLens timing] phase1 scoring:', (t1-t0).toFixed(3), 'ms for', url);
  const [action, klass] = gate(url, p);
  let band;
  if (action === 'RED_RULE' || action === 'RED_URL') band = 'bad';
  else if (action === 'ACCEPT_GUARD' || action === 'ACCEPT') band = 'ok';
  else band = 'warn';                                   // ESCALATE -> amber until phase 2
  return { p, action, klass, band };
}

function fuse(ph1, page) {
  // page = {mismatch, brand} or null (no page info yet / dead page)
  // Rule reds and guard greens from phase 1 are never overridden by phase 2.
  if (ph1.action === 'RED_RULE') return { band: 'bad', reason: 'url_rule:' + ph1.klass, p: ph1.p };
  if (ph1.action === 'ACCEPT_GUARD') return { band: 'ok', reason: 'known_brand', p: ph1.p };

  const pageMismatch = page && page.mismatch;
  if (pageMismatch) {
    if (FUSION_MODE === 'B') {
      // assertive: page decides -> red; phase-1 low score just reinforces
      return { band: 'bad', reason: 'page_brand_mismatch(B):' + page.brand, p: ph1.p };
    } else {
      // cautious: amber unless phase-1 ALSO flags (score below red line)
      const band = (ph1.p < 0.40) ? 'bad' : 'warn';
      return { band, reason: 'page_brand_mismatch(A):' + page.brand, p: ph1.p };
    }
  }

  // no page mismatch -> fall back to phase-1 outcome
  if (ph1.action === 'ESCALATE') {
    // escalated but page cleared / silent -> honest amber, show score
    return { band: (page ? 'ok' : 'warn'), reason: page ? 'page_clean' : 'awaiting_page', p: ph1.p };
  }
  return { band: ph1.band, reason: 'url_score', p: ph1.p };
}

// ---------- per-tab state + badge ----------
const tabState = {};   // tabId -> {url, ph1, page}
function setBadge(tabId, band) {
  const text = band === 'ok' ? 'OK' : band === 'warn' ? '?' : '!';
  const color = band === 'ok' ? '#2e7d32' : band === 'warn' ? '#f9a825' : '#d32f2f';
  chrome.action.setBadgeText({ tabId, text });
  chrome.action.setBadgeBackgroundColor({ tabId, color });
}

async function evaluate(tabId, url) {
  if (!url || !/^https?:/.test(url)) return;
  const ph1 = await phase1(url);
  const st = tabState[tabId] || {};
  st.url = url; st.ph1 = ph1; st.page = (st.page && st.page.url === url) ? st.page : null;
  tabState[tabId] = st;
  const v = fuse(ph1, st.page);
  st.verdict = v;
  setBadge(tabId, v.band);
}

chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  const url = info.url || (info.status === 'loading' ? tab.url : null);
  if (url) { tabState[tabId] = { url }; evaluate(tabId, url); }
});
chrome.tabs.onRemoved.addListener(tabId => { delete tabState[tabId]; });

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === 'pageInfo' && sender.tab) {
    const tabId = sender.tab.id;
    const st = tabState[tabId] || {};
    if (!st.ph1 || st.url !== msg.url) { st.url = msg.url; }
    const pm = pageBrandMismatch(msg.url, msg.identity, msg.numPwd > 0);
    st.page = { url: msg.url, mismatch: pm.mismatch, brand: pm.brand,
                numForms: msg.numForms, numPwd: msg.numPwd,
                extRatio: msg.extRatio, hiddenCount: msg.hiddenCount };
    tabState[tabId] = st;
    (async () => {
      if (!st.ph1) st.ph1 = await phase1(msg.url);
      const v = fuse(st.ph1, st.page);
      st.verdict = v;
      setBadge(tabId, v.band);
    })();
    return;
  }
  if (msg && msg.type === 'getVerdict') {
    // popup asks for the active tab's current verdict
    chrome.tabs.query({ active: true, currentWindow: true }, async tabs => {
      const tab = tabs[0]; if (!tab) { sendResponse({}); return; }
      const st = tabState[tab.id];
      if (st && st.verdict) {
        sendResponse({ url: st.url, verdict: st.verdict, page: st.page || null,
                       mode: FUSION_MODE, klass: st.ph1 ? st.ph1.klass : null });
      } else if (tab.url && /^https?:/.test(tab.url)) {
        const ph1 = await phase1(tab.url);
        const v = fuse(ph1, null);
        sendResponse({ url: tab.url, verdict: v, page: null, mode: FUSION_MODE, klass: ph1.klass });
      } else sendResponse({ url: tab.url, verdict: null });
    });
    return true;   // async
  }
});

