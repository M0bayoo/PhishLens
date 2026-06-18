"""
PhishLens - Feature Category Mapping
=====================================
Maps PhiUSIIL's 55 actual columns (confirmed via 02_explore_data.py)
to PhishLens's 5 feature categories defined in Methodology Section 3.5.

This corrects an earlier assumption in the methodology: PhiUSIIL DOES
provide most rendered-page features. The actual gap is narrower —
specifically TLS certificate data and domain/WHOIS age data.

Use this mapping when building the merge pipeline (03_merge_pipeline.py).
"""

# ---------------------------------------------------------------------------
# CATEGORY 1: URL-Lexical
# Fully covered by PhiUSIIL.
# ---------------------------------------------------------------------------
URL_LEXICAL_PHIUSIIL = [
    "URLLength",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",       # [sic] - PhiUSIIL's own spelling
    "DegitRatioInURL",       # [sic]
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",  # [sic]
    "IsHTTPS",
]

# ---------------------------------------------------------------------------
# CATEGORY 2: Host / DNS
# PARTIALLY covered by PhiUSIIL — domain length and IP-as-domain flag only.
# Domain AGE and WHOIS registrar data are NOT in PhiUSIIL — must be
# extracted fresh for PhishTank/Tranco URLs.
# ---------------------------------------------------------------------------
HOST_DNS_PHIUSIIL = [
    "DomainLength",
    "IsDomainIP",
    "TLD",
]
HOST_DNS_MISSING_FROM_PHIUSIIL = [
    "domain_age_days",       # requires WHOIS lookup
    "whois_registrar",       # requires WHOIS lookup
    "dns_ttl",                # requires DNS lookup
]

# ---------------------------------------------------------------------------
# CATEGORY 3: TLS Certificate
# NOT covered by PhiUSIIL at all. This is the clearest, confirmed gap.
# Must be extracted fresh for PhishTank/Tranco URLs (live sites only —
# PhiUSIIL's 2021 phishing URLs are mostly dead and cannot be certificate-
# checked retroactively).
# ---------------------------------------------------------------------------
TLS_CERTIFICATE_MISSING_FROM_PHIUSIIL = [
    "cert_issuer_type",       # DV / OV / EV
    "cert_age_days",
    "cert_validity_days",
    "is_lets_encrypt",
    "san_mismatch",
]

# ---------------------------------------------------------------------------
# CATEGORY 4: Rendered Page
# MOSTLY covered by PhiUSIIL — this is the correction. PhiUSIIL has
# extensive page-structure features already extracted at collection time.
# ---------------------------------------------------------------------------
RENDERED_PAGE_PHIUSIIL = [
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]

# ---------------------------------------------------------------------------
# CATEGORY 5: Behavioural (redirect depth)
# Covered by PhiUSIIL via NoOfURLRedirect / NoOfSelfRedirect (see Category 4).
# No separate extraction needed for this category specifically.
# ---------------------------------------------------------------------------
BEHAVIOURAL_PHIUSIIL = [
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
]

# ---------------------------------------------------------------------------
# Columns NOT used for modelling (identifiers / raw text, not features)
# ---------------------------------------------------------------------------
EXCLUDED_COLUMNS = [
    "URL",      # raw URL string — used to derive features, not a feature itself
    "Domain",   # raw domain string — same as above
    "Title",    # raw page title text — same as above
]

TARGET_COLUMN = "label"
# CONFIRMED via UCI dataset documentation (archive.ics.uci.edu/dataset/967):
#   label = 1  ->  LEGITIMATE URL
#   label = 0  ->  PHISHING URL
# This matches the observed distribution: 134,850 rows at label=1 (legitimate)
# and 100,945 rows at label=0 (phishing) — exactly matching PhiUSIIL's
# documented composition. This is the OPPOSITE convention to many other
# phishing datasets (which often use 1=phishing) - handle with care when
# merging with PhishTank (all phishing -> must be relabelled to 0) and
# Tranco (all legitimate -> must be relabelled to 1).


if __name__ == "__main__":
    total_mapped = (
        len(URL_LEXICAL_PHIUSIIL)
        + len(HOST_DNS_PHIUSIIL)
        + len(RENDERED_PAGE_PHIUSIIL)
    )
    print(f"URL-Lexical features from PhiUSIIL: {len(URL_LEXICAL_PHIUSIIL)}")
    print(f"Host/DNS features from PhiUSIIL: {len(HOST_DNS_PHIUSIIL)} "
          f"(+ {len(HOST_DNS_MISSING_FROM_PHIUSIIL)} to extract fresh)")
    print(f"TLS Certificate features from PhiUSIIL: 0 "
          f"(all {len(TLS_CERTIFICATE_MISSING_FROM_PHIUSIIL)} must be extracted fresh)")
    print(f"Rendered Page features from PhiUSIIL: {len(RENDERED_PAGE_PHIUSIIL)}")
    print(f"Behavioural features from PhiUSIIL: {len(BEHAVIOURAL_PHIUSIIL)} (overlaps with Rendered Page)")
    print(f"\nTotal PhiUSIIL columns used as features: {total_mapped}")
    print(f"Total PhiUSIIL columns available: 55")
    print(f"\nLabel convention CONFIRMED: label=1 -> legitimate, label=0 -> phishing.")
    print(f"PhishTank rows (all phishing) will be assigned label=0.")
    print(f"Tranco rows (all legitimate) will be assigned label=1.")
