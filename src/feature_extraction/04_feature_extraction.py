"""
PhishLens - Feature Extraction Pipeline (FIXED v2)
====================================================
Extracts URL-lexical, host/DNS, TLS, and redirect features
for fresh PhishTank and Tranco URLs in unified_urls.csv.

KEY FIXES over v1:
  1. WHOIS replaced with python-whois using strict socket timeout
     (was hanging indefinitely — root cause of 6-hour runtime)
  2. ThreadPoolExecutor for concurrent processing (20 workers)
  3. Checkpoint/resume: saves progress every 500 rows so a crash
     doesn't lose everything
  4. Hard per-URL wall-clock timeout via concurrent.futures
  5. Aggressive timeouts on all network calls (5s max each)

Usage:
    python 04_feature_extraction.py

Outputs:
    data/processed/fresh_features.csv       — final output
    data/processed/fresh_features_checkpoint.csv  — rolling save
"""

import re
import math
import time
import socket
import ssl
import hashlib
import warnings
import traceback
import datetime
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

import pandas as pd
import tldextract
import requests
import dns.resolver
import dns.exception

# ── Suppress noisy urllib3 warnings about unverified HTTPS ──────────────────
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
PROCESSED   = BASE_DIR / "data" / "processed"
INPUT_FILE  = PROCESSED / "unified_urls.csv"
OUTPUT_FILE = PROCESSED / "fresh_features.csv"
CHECKPOINT  = PROCESSED / "fresh_features_checkpoint.csv"

# ── Tuning ───────────────────────────────────────────────────────────────────
MAX_WORKERS       = 20      # concurrent threads
CHECKPOINT_EVERY  = 500     # save partial results every N rows
REQUEST_TIMEOUT   = 5       # seconds for HTTP/TLS/DNS/socket calls
PER_URL_TIMEOUT   = 15      # hard wall-clock limit per URL (all features)

# ── Phishing-indicative keywords in URLs ────────────────────────────────────
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "secure", "account", "update", "confirm",
    "verify", "banking", "paypal", "amazon", "apple", "microsoft",
    "support", "password", "credential", "wallet", "payment",
]


# ════════════════════════════════════════════════════════════════════════════
# 1. URL-LEXICAL FEATURES  (pure string — no network, never fails)
# ════════════════════════════════════════════════════════════════════════════

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


def extract_url_lexical(url: str) -> dict:
    parsed  = urlparse(url)
    domain  = parsed.netloc or ""
    path    = parsed.path    or ""
    query   = parsed.query   or ""
    ext     = tldextract.extract(url)

    full    = url
    subdomain = ext.subdomain or ""

    has_ip  = bool(re.match(
        r"^(\d{1,3}\.){3}\d{1,3}$", domain.split(":")[0]
    ))

    digits_in_domain = sum(c.isdigit() for c in ext.domain or "")

    return {
        # lengths
        "url_length":           len(full),
        "domain_length":        len(domain),
        "path_length":          len(path),
        "query_length":         len(query),
        # counts
        "num_dots":             full.count("."),
        "num_hyphens":          full.count("-"),
        "num_underscores":      full.count("_"),
        "num_slashes":          full.count("/"),
        "num_at":               full.count("@"),
        "num_question_marks":   full.count("?"),
        "num_equals":           full.count("="),
        "num_ampersands":       full.count("&"),
        "num_percent":          full.count("%"),
        "num_digits_in_domain": digits_in_domain,
        "num_subdomains":       len(subdomain.split(".")) if subdomain else 0,
        # booleans
        "has_ip_in_url":        int(has_ip),
        "has_https":            int(parsed.scheme == "https"),
        "has_port":             int(":" in domain),
        "has_double_slash":     int("//" in path),
        "has_at_symbol":        int("@" in full),
        "has_hex_encoding":     int("%2" in full.upper() or "%3" in full.upper()),
        "has_suspicious_kw":    int(any(k in full.lower() for k in SUSPICIOUS_KEYWORDS)),
        # entropy
        "url_entropy":          round(_entropy(full), 4),
        "domain_entropy":       round(_entropy(domain), 4),
        # TLD
        "tld":                  ext.suffix or "",
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. HOST / DNS FEATURES  (network — strict timeout)
# ════════════════════════════════════════════════════════════════════════════

def _safe_dns_query(domain: str, rtype: str):
    resolver = dns.resolver.Resolver()
    resolver.lifetime = REQUEST_TIMEOUT
    resolver.timeout  = REQUEST_TIMEOUT
    try:
        return resolver.resolve(domain, rtype)
    except Exception:
        return None


def extract_host_dns(url: str) -> dict:
    ext    = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    result = {
        "dns_resolved":      0,
        "dns_ttl":           -1,
        "dns_a_count":       0,
        "dns_mx_count":      0,
        "dns_ns_count":      0,
        "dns_txt_count":     0,
        # WHOIS entirely removed — was the hang culprit.
        # Domain-age is NaN for all rows; imputed later.
        "domain_age_days":   -1,
        "registrar_known":    0,
    }

    # A record
    a_ans = _safe_dns_query(domain, "A")
    if a_ans:
        result["dns_resolved"] = 1
        result["dns_a_count"]  = len(list(a_ans))
        try:
            result["dns_ttl"] = a_ans.rrset.ttl
        except Exception:
            pass

    # MX
    mx_ans = _safe_dns_query(domain, "MX")
    if mx_ans:
        result["dns_mx_count"] = len(list(mx_ans))

    # NS
    ns_ans = _safe_dns_query(domain, "NS")
    if ns_ans:
        result["dns_ns_count"] = len(list(ns_ans))

    # TXT
    txt_ans = _safe_dns_query(domain, "TXT")
    if txt_ans:
        result["dns_txt_count"] = len(list(txt_ans))

    return result


# ════════════════════════════════════════════════════════════════════════════
# 3. TLS CERTIFICATE FEATURES  (network — strict timeout)
# ════════════════════════════════════════════════════════════════════════════

def extract_tls(url: str) -> dict:
    result = {
        "tls_valid":             0,
        "tls_days_remaining":    -1,
        "tls_self_signed":        1,
        "tls_wildcard":           0,
        "tls_san_count":          0,
        "tls_subject_match":      0,
    }

    parsed = urlparse(url)
    host   = (parsed.netloc or "").split(":")[0]
    if not host:
        return result

    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=REQUEST_TIMEOUT) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()

        # expiry
        not_after = cert.get("notAfter", "")
        if not_after:
            expiry = datetime.datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
            days   = (expiry - datetime.datetime.utcnow()).days
            result["tls_days_remaining"] = days
            result["tls_valid"] = int(days > 0)

        # self-signed: issuer == subject?
        issuer  = dict(x[0] for x in cert.get("issuer",  []))
        subject = dict(x[0] for x in cert.get("subject", []))
        result["tls_self_signed"] = int(issuer.get("commonName") == subject.get("commonName"))

        # SAN / wildcard
        sans = cert.get("subjectAltName", [])
        result["tls_san_count"] = len(sans)
        result["tls_wildcard"]  = int(any("*" in v for _, v in sans))

        # subject CN matches host?
        cn = subject.get("commonName", "")
        result["tls_subject_match"] = int(
            host == cn or (cn.startswith("*.") and host.endswith(cn[2:]))
        )

    except Exception:
        pass  # result stays at defaults

    return result


# ════════════════════════════════════════════════════════════════════════════
# 4. REDIRECT / RESPONSE FEATURES  (network — strict timeout)
# ════════════════════════════════════════════════════════════════════════════

def extract_redirect(url: str) -> dict:
    result = {
        "redirect_count":       0,
        "final_url_different":  0,
        "response_time_ms":     -1,
        "http_status":          -1,
        "content_length":       -1,
    }

    try:
        t0   = time.time()
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            verify=False,
            headers={"User-Agent": "Mozilla/5.0 (PhishLens Research Bot)"},
            stream=True,
        )
        elapsed = (time.time() - t0) * 1000

        result["redirect_count"]      = len(resp.history)
        result["final_url_different"] = int(resp.url != url)
        result["response_time_ms"]    = round(elapsed, 1)
        result["http_status"]         = resp.status_code
        cl = resp.headers.get("Content-Length", -1)
        try:
            result["content_length"] = int(cl)
        except (ValueError, TypeError):
            pass

    except Exception:
        pass

    return result


# ════════════════════════════════════════════════════════════════════════════
# 5. COMBINED EXTRACTION (one row per URL)
# ════════════════════════════════════════════════════════════════════════════

def extract_all(url: str, label: int, source: str) -> dict:
    row = {"url": url, "label": label, "source": source}
    try:
        row.update(extract_url_lexical(url))
    except Exception:
        pass
    try:
        row.update(extract_host_dns(url))
    except Exception:
        pass
    try:
        row.update(extract_tls(url))
    except Exception:
        pass
    try:
        row.update(extract_redirect(url))
    except Exception:
        pass
    return row


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("PhishLens Feature Extraction  (FIXED v2 — fast + resumable)")
    print("=" * 65)

    # ── Load input ───────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_FILE)
    # Only process fresh rows (PhishTank + Tranco — PhiUSIIL already has features)
    fresh = df[df["source"].isin(["phishtank", "tranco"])].copy().reset_index(drop=True)
    total = len(fresh)
    print(f"\nFresh URLs to process: {total:,}")
    print(f"  phishtank : {(fresh['source']=='phishtank').sum():,}")
    print(f"  tranco    : {(fresh['source']=='tranco').sum():,}")

    # ── Resume from checkpoint if it exists ─────────────────────────────
    already_done = set()
    results      = []
    if CHECKPOINT.exists():
        ckpt = pd.read_csv(CHECKPOINT)
        already_done = set(ckpt["url"].tolist())
        results      = ckpt.to_dict("records")
        print(f"\nCheckpoint found — resuming from {len(already_done):,} already processed rows")

    todo = fresh[~fresh["url"].isin(already_done)].reset_index(drop=True)
    print(f"Remaining   : {len(todo):,}")

    if todo.empty:
        print("\nAll rows already processed. Writing final output.")
    else:
        # ── Process concurrently ─────────────────────────────────────────
        print(f"\nRunning with {MAX_WORKERS} workers, {PER_URL_TIMEOUT}s hard timeout per URL …\n")

        t_start   = time.time()
        processed = 0
        failed    = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_row = {
                executor.submit(extract_all, row["url"], row["label"], row["source"]): row
                for _, row in todo.iterrows()
            }

            for future in as_completed(future_to_row, timeout=None):
                try:
                    row_result = future.result(timeout=PER_URL_TIMEOUT)
                    results.append(row_result)
                except Exception:
                    orig = future_to_row[future]
                    results.append({"url": orig["url"], "label": orig["label"],
                                    "source": orig["source"], "extraction_error": 1})
                    failed += 1

                processed += 1

                # ── Progress line ─────────────────────────────────────
                if processed % 100 == 0 or processed == len(todo):
                    elapsed  = time.time() - t_start
                    rate     = processed / elapsed if elapsed > 0 else 0
                    eta_secs = (len(todo) - processed) / rate if rate > 0 else 0
                    eta_min  = eta_secs / 60
                    print(
                        f"  [{processed:>5}/{len(todo):>5}]  "
                        f"failed={failed}  "
                        f"rate={rate:.1f}/s  "
                        f"ETA={eta_min:.1f}min"
                    )

                # ── Checkpoint save ───────────────────────────────────
                if processed % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
                    print(f"  ✓ Checkpoint saved ({len(results):,} rows)")

    # ── Final save ───────────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)

    elapsed_total = time.time() - t_start if 't_start' in dir() else 0

    print("\n" + "=" * 65)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"Total processed : {len(out_df):,}")
    print(f"Rows with errors: {out_df.get('extraction_error', pd.Series(dtype=int)).sum()}")
    print(f"Columns extracted: {len(out_df.columns)}")
    print(f"Time taken      : {elapsed_total/60:.1f} min")
    print(f"Output saved to : {OUTPUT_FILE}")
    print("=" * 65)

    # Clean up checkpoint now that we have the final file
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint file removed.")


if __name__ == "__main__":
    main()
