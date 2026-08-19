# Lexical and structural feature extraction for PhishLens.
# Converts a URL into a 34-dimensional feature vector, computable from the
# URL string alone with no network access. Imported by 04_train_model.py.

import re
import math
from urllib.parse import urlparse
from collections import Counter


# Known brands mapped to the domains they legitimately own. Used both to
# recognise genuine brand traffic and to detect impersonation of it.
BRANDS = {
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
}

REAL_BRAND_DOMAINS = []
for domain_list in BRANDS.values():
    REAL_BRAND_DOMAINS.extend(domain_list)

# Platforms hosting user-published content on a shared domain. The
# registered domain therefore carries no reputation signal either way.
FREE_HOSTING = ('github.io','netlify.app','wixsite.com','wixstudio.com','firebaseapp.com',
 'herokuapp.com','vercel.app','pages.dev','weebly.com','glitch.me','amazonaws.com',
 'cloudfront.net','googleusercontent.com','windows.net','web.app','workers.dev','r2.dev',
 'repl.co','myshopify.com','square.site','000webhostapp.com','duckdns.org','sharepoint.com',
 'blogspot.com','wordpress.com','sites.google.com','translate.goog','azurewebsites.net')

# Hosts owned by a known brand but serving user-uploaded content. The
# domain is genuine while the page on it is not vouched for by the brand,
# so these must not satisfy the known-brand guard: phishing kits are
# routinely hosted on them precisely to inherit the domain's reputation.
USER_CONTENT_HOSTS = (
    'sites.google.com', 'docs.google.com', 'drive.google.com',
    'script.google.com', 'groups.google.com', 'forms.gle',
    'googleusercontent.com', 'express.adobe.com', 'spark.adobe.com',
    'documentcloud.adobe.com', 'dropbox.com', 'dropboxusercontent.com',
    'onedrive.live.com', '1drv.ms', 'sharepoint.com', 'forms.office.com',
    'notion.site', 'canva.site', 'figma.site')

# Top-level domains disproportionately represented in phishing corpora,
# generally those available cheaply and with minimal registration checks.
HIGH_RISK_TLD = {'buzz','top','cfd','sbs','xyz','tk','ml','ga','cf','gq','icu','click','link',
                 'work','rest','fit','loan','date','racing','win','bid','stream','download',
                 'pro','online','site','website','space','store','shop','life','world'}

MED_RISK_TLD = {'info','biz','us','cc','me','io','app','dev','page','one','today','digital','live'}

# Public suffixes spanning two labels, where the registered domain is the
# final three labels rather than the final two (bbc.co.uk, not co.uk).
TWO_PART_ENDINGS = ('co.uk','org.uk','ac.uk','gov.uk','net.uk','com.br','com.au','co.jp','co.nz',
 'com.tr','co.in','com.mx','com.ar','co.za','com.pl','net.au','org.au','co.id','com.sg',
 'com.vn','co.kr','com.cn','co.il','com.hk','com.tw','co.th','com.my','co.ke')

# Terms characteristic of credential-harvesting and payment-fraud lures.
SCAM_WORDS = ('login','signin','verify','secure','account','update','confirm',
              'bank','paypal','password','webscr','wallet')


# Returns the registered domain: 'shop.news.bbc.co.uk' -> 'bbc.co.uk'
def main_domain(host):
    host = host.lower().split(':')[0].strip('.')
    pieces = host.split('.')
    if len(pieces) <= 2:
        return host
    last_two = '.'.join(pieces[-2:])
    if last_two in TWO_PART_ENDINGS and len(pieces) >= 3:
        return '.'.join(pieces[-3:])
    return last_two


# Splits on domain separators: 'my-paypal_site' -> ['my','paypal','site']
def split_words(text):
    return [word for word in re.split(r'[.\-_]', text.lower()) if word]


# Levenshtein distance <= 1, i.e. a single-character edit apart.
# Detects typosquatting such as 'paypa1' against 'paypal'.
def is_one_typo_away(word_a, word_b):
    if abs(len(word_a) - len(word_b)) > 2:
        return False
    previous_row = list(range(len(word_b) + 1))
    for i, letter_a in enumerate(word_a, 1):
        current_row = [i]
        for j, letter_b in enumerate(word_b, 1):
            cost = 0 if letter_a == letter_b else 1
            current_row.append(min(previous_row[j] + 1,
                                   current_row[j - 1] + 1,
                                   previous_row[j - 1] + cost))
        previous_row = current_row
    return previous_row[-1] <= 1


# Whether the given brand legitimately owns the given domain
def owns_domain(brand, domain):
    return any(domain == real or domain.endswith('.' + real)
               for real in BRANDS[brand])


# Shannon entropy of the character distribution. Higher values indicate
# algorithmically generated strings.
def randomness(text):
    if not text:
        return 0.0
    letter_counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total)
                for count in letter_counts.values())


# 1 if the host serves user-uploaded content on a brand-owned domain
def on_user_content_host(url):
    try:
        host = urlparse(url).netloc.lower().split(':')[0]
    except Exception:
        return 0
    return int(any(host == h or host.endswith('.' + h)
                   for h in USER_CONTENT_HOSTS))


# 1 if the registered domain belongs to a known brand, else 0. Pages on
# user-content hosts are excluded: the domain is genuine but the content
# is not, so treating them as known-brand traffic would auto-accept
# phishing hosted on services such as Google Sites or Adobe Express.
def is_genuine_brand_site(url):
    if on_user_content_host(url):
        return 0
    try:
        domain = main_domain(urlparse(url).netloc)
    except Exception:
        return 0
    if not domain:
        return 0
    return int(any(domain == real or domain.endswith('.' + real)
                   for real in REAL_BRAND_DOMAINS))


# Classifies how the URL references a known brand, if at all. Returns five
# flags:
#   brand_in_host          a brand name appears in the hostname
#   brand_domain_mismatch  that brand does not own this domain
#   brand_in_subdomain     the brand name sits in the subdomain only
#   brand_lookalike        the name is a single-character misspelling
#   brand_in_path          the brand appears only in the path or query
def check_brand_abuse(url):
    no_flags = {'brand_in_host': 0, 'brand_domain_mismatch': 0,
                'brand_in_subdomain': 0, 'brand_lookalike': 0, 'brand_in_path': 0}
    try:
        parts = urlparse(url)
    except Exception:
        return no_flags

    host = parts.netloc.lower().split(':')[0]
    if not host:
        return no_flags

    domain = main_domain(host)
    domain_name = domain.split('.')[0]
    domain_words = split_words(domain_name)
    subdomain = host[:-len(domain)].rstrip('.') if host.endswith(domain) else ''
    subdomain_words = split_words(subdomain)
    path_and_query = (parts.path + '?' + parts.query).lower()

    # Brand name in the registered domain: genuine if the brand owns it,
    # impersonation otherwise (paypal.com vs paypal-secure.com).
    for brand in BRANDS:
        if brand in domain_words:
            if owns_domain(brand, domain) or domain_name == brand:
                return {**no_flags, 'brand_in_host': 1}
            return {**no_flags, 'brand_in_host': 1, 'brand_domain_mismatch': 1}

        # Brand name in the subdomain of a domain it does not own
        # (paypal.evil-site.com).
        if brand in subdomain_words:
            if owns_domain(brand, domain):
                return {**no_flags, 'brand_in_host': 1}
            return {**no_flags, 'brand_in_host': 1,
                    'brand_domain_mismatch': 1, 'brand_in_subdomain': 1}

    # Misspelled brand name. Restricted to names of five characters or
    # more, below which single-edit matches are too often coincidental.
    for brand in BRANDS:
        if len(brand) < 5:
            continue
        for word in domain_words + subdomain_words:
            if len(word) >= 5 and word != brand and is_one_typo_away(word, brand):
                return {**no_flags, 'brand_in_host': 1,
                        'brand_domain_mismatch': 1, 'brand_lookalike': 1}

    # Brand referenced only after the hostname (evil.com/paypal/login).
    for brand in BRANDS:
        if len(brand) >= 5 and brand in path_and_query:
            return {**no_flags, 'brand_in_path': 1}

    return no_flags


# 2 for high-risk TLDs, 1 for medium-risk, 0 otherwise
def tld_risk_score(url):
    try:
        host = urlparse(url).netloc.lower().split(':')[0]
    except Exception:
        return 0
    ending = host.split('.')[-1] if '.' in host else ''
    if ending in HIGH_RISK_TLD:
        return 2
    if ending in MED_RISK_TLD:
        return 1
    return 0


# 1 if hosted on a shared user-content platform, else 0. The full hostname
# is checked before reducing to the registered domain, since hosts such as
# sites.google.com reduce to google.com and would otherwise be missed.
def on_free_hosting(url):
    try:
        host = urlparse(url).netloc.lower().split(':')[0]
        domain = main_domain(host)
    except Exception:
        return 0
    if on_user_content_host(url):
        return 1
    return int(any(domain == platform or domain.endswith('.' + platform)
                   for platform in FREE_HOSTING))


# Converts a URL into its 34-feature vector
def url_to_features(url):
    url = str(url)
    parts = urlparse(url)
    host = parts.netloc
    path = parts.path
    query = parts.query
    brand = check_brand_abuse(url)
    domain = main_domain(host)

    return {
        # Component lengths
        'url_length':    len(url),
        'domain_length': len(host),
        'path_length':   len(path),
        'query_length':  len(query),

        # Frequency of structurally significant characters
        'num_dots':             url.count('.'),
        'num_hyphens':          url.count('-'),
        'num_underscores':      url.count('_'),
        'num_slashes':          url.count('/'),
        'num_at':               url.count('@'),
        'num_question_marks':   url.count('?'),
        'num_equals':           url.count('='),
        'num_ampersands':       url.count('&'),
        'num_percent':          url.count('%'),
        'num_digits_in_domain': sum(c.isdigit() for c in host),
        'num_subdomains':       max(host.count('.') - 1, 0),

        # Binary indicators. has_at_symbol and has_double_slash cover
        # authority-confusion tricks; has_hex_encoding covers percent-
        # encoding used to obscure the visible address.
        'has_ip_in_url':     int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$',
                                               host.split(':')[0]))),
        'has_https':         int(parts.scheme == 'https'),
        'has_port':          int(':' in host),
        'has_double_slash':  int('//' in url[8:]),
        'has_at_symbol':     int('@' in url),
        'has_hex_encoding':  int('%' in url),
        'has_suspicious_kw': int(any(w in url.lower() for w in SCAM_WORDS)),

        # Character-distribution randomness
        'url_entropy':    randomness(url),
        'domain_entropy': randomness(host),

        # Brand-consistency signals (see check_brand_abuse). brand_is_legit
        # is the symmetric counterpart to the mismatch flags: it permits
        # genuine brand traffic rather than only rejecting impersonation.
        'brand_in_host':         brand['brand_in_host'],
        'brand_domain_mismatch': brand['brand_domain_mismatch'],
        'brand_in_subdomain':    brand['brand_in_subdomain'],
        'brand_lookalike':       brand['brand_lookalike'],
        'brand_in_path':         brand['brand_in_path'],
        'brand_is_legit':        is_genuine_brand_site(url),

        # Domain reputation and structure
        'tld_risk':          tld_risk_score(url),
        'reg_domain_length': len(domain),
        'is_platform':       on_free_hosting(url),
        'subdomain_depth':   max(len(host.split('.')) - len(domain.split('.')), 0),
    }
