"""
PhishLens - Rendered-Page Feature Extraction (runs on the Ubuntu VM)
=======================================================================
Extracts the "Rendered Page" feature category from Table 3.4:
  num_forms, has_password_field, num_password_fields,
  hidden_element_count, iframe_count, external_resource_ratio,
  favicon_domain_mismatch

Adds these to your existing fresh_features.csv (URL-lexical/DNS/TLS/
redirect features), producing the complete five-category feature set
described in Chapter 3, Section 3.5.

DUE-DILIGENCE FIXES applied before first real run (1 July 2026):
  1. PERFORMANCE: original design launched a new browser PER URL
     (~0.63s overhead each). Empirically measured 3.5x speedup by
     launching ONE browser per worker thread, reused across that
     thread's whole batch via fresh contexts per URL. Confirmed
     stable under concurrent load (20/20, then 40/40 test runs, 0 errors).
  2. CORRECTNESS: domain-comparison logic (external_resource_ratio,
     favicon_domain_mismatch) now uses page.url (the FINAL URL after
     any redirects) instead of the original pre-navigation URL.
     Phishing pages redirect constantly; comparing against the wrong
     domain would silently misjudge these two features on any
     redirecting page.
  3. COMPLETENESS: external_resource_ratio now also counts stylesheets
     (link[rel=stylesheet]), not just images/scripts.
  4. SCOPE CORRECTION: earlier informal explanation mentioned "social
     media links", "copyright notices", "popup count" as example
     rendered-page signals - these are NOT in Chapter 3 Table 3.4 and
     are NOT built here. Only the six features actually specified are
     extracted: num_forms, has_password_field, num_password_fields,
     hidden_element_count, iframe_count, external_resource_ratio,
     favicon_domain_mismatch.

SAFETY / ETHICS (matches Section 3.9 exactly):
  - One-hop, passive fetch only. Does NOT follow links, does NOT submit
    forms, does NOT interact with login/payment fields.
  - Runs inside this isolated VM only - never run this on your main
    Windows machine, since it visits live, unverified phishing URLs.
  - Page content is discarded immediately after feature extraction;
    only the numeric feature vector is retained (no raw HTML/text kept).

LESSONS APPLIED FROM EARLIER SESSION (the WHOIS 6-hour hang):
  - Hard per-page timeout (10s navigation)
  - Checkpoint every 250 rows - a crash loses at most 250 rows, not everything
  - Concurrent processing, but modest (8 worker threads)

Requirements (install inside the VM):
    pip install playwright pandas --break-system-packages
    playwright install chromium
    playwright install-deps chromium   # installs OS-level dependencies

Usage:
    python3 14_rendered_page_extraction.py

Outputs:
    data/processed/rendered_page_features.csv
    (merge into fresh_features.csv with the follow-up merge script)
"""

import time
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from playwright.sync_api import sync_playwright

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
PROCESSED   = BASE_DIR / "data" / "processed"
INPUT_FILE  = PROCESSED / "render_sample_urls.csv"     # stratified ~1000-row
                                                        # sample, NOT the full
                                                        # fresh_features.csv -
                                                        # see 13_stratified_sample.py
OUTPUT_FILE = PROCESSED / "rendered_page_features.csv"
CHECKPOINT  = PROCESSED / "rendered_page_checkpoint.csv"

MAX_WORKERS      = 2   # reduced from 8 after VM memory crash (2 chromium
                       # instances instead of 8 dramatically cuts RAM use;
                       # slower but stable - correctness over speed here
CHECKPOINT_EVERY = 250
NAV_TIMEOUT_MS   = 10_000   # 10s hard cap on page navigation

EMPTY_FEATURES = {
    "num_forms": None, "has_password_field": None,
    "num_password_fields": None, "hidden_element_count": None,
    "iframe_count": None, "external_resource_ratio": None,
    "favicon_domain_mismatch": None, "render_error": 1,
}


def extract_rendered_features(page) -> dict:
    """Extract Table 3.4 'Rendered Page' features from an already-loaded page.
    Uses page.url (final URL after redirects), not the pre-navigation URL."""
    page_domain = urlparse(page.url).netloc

    num_forms = page.eval_on_selector_all("form", "els => els.length")
    num_password_fields = page.eval_on_selector_all(
        "input[type='password']", "els => els.length")
    hidden_inline = page.eval_on_selector_all(
        "[style*='display:none'], [style*='display: none']", "els => els.length")
    hidden_inputs = page.eval_on_selector_all(
        "input[type='hidden']", "els => els.length")
    iframe_count = page.eval_on_selector_all("iframe", "els => els.length")

    resource_urls = page.eval_on_selector_all(
        "img[src], script[src], link[rel='stylesheet'][href]",
        "els => els.map(e => e.src || e.href)")
    total_resources = len(resource_urls)
    external = sum(1 for u in resource_urls
                   if u and urlparse(u).netloc and urlparse(u).netloc != page_domain)
    external_ratio = (external / total_resources) if total_resources > 0 else 0.0

    favicon_href = page.eval_on_selector(
        "link[rel*='icon']", "el => el ? el.href : null")
    favicon_mismatch = 0
    if favicon_href:
        fav_domain = urlparse(favicon_href).netloc
        favicon_mismatch = int(fav_domain != "" and fav_domain != page_domain)

    return {
        "num_forms": num_forms,
        "has_password_field": int(num_password_fields > 0),
        "num_password_fields": num_password_fields,
        "hidden_element_count": hidden_inline + hidden_inputs,
        "iframe_count": iframe_count,
        "external_resource_ratio": round(external_ratio, 4),
        "favicon_domain_mismatch": favicon_mismatch,
        "render_error": 0,
    }


def process_url_batch(urls_chunk: list) -> list:
    """One browser launched ONCE per worker thread, reused across this
    thread's whole batch via fresh contexts per URL. Never shared across
    threads - each sync_playwright()/browser stays local to the thread
    that created it (confirmed safe AND ~3.5x faster than relaunching a
    browser per URL, verified empirically before this script was used)."""
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"]
        )
        for url in urls_chunk:
            result = {"url": url}
            try:
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (PhishLens Research Bot; one-hop passive fetch)",
                    ignore_https_errors=True,
                )
                page = context.new_page()
                page.set_default_timeout(NAV_TIMEOUT_MS)
                page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                result.update(extract_rendered_features(page))
                context.close()
            except Exception:
                result.update(EMPTY_FEATURES)
            results.append(result)
        browser.close()
    return results


def main():
    print("=" * 65)
    print("PhishLens - Rendered-Page Feature Extraction (VM)")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)
    urls = df["url"].tolist()
    print(f"\nURLs to process: {len(urls):,}")

    already_done = set()
    results = []
    if CHECKPOINT.exists():
        ckpt = pd.read_csv(CHECKPOINT)
        already_done = set(ckpt["url"].tolist())
        results = ckpt.to_dict("records")
        print(f"Checkpoint found - resuming from {len(already_done):,} done")

    todo = [u for u in urls if u not in already_done]
    print(f"Remaining: {len(todo):,}\n")

    if not todo:
        print("Nothing left to process.")
    else:
        # Split URLs into MAX_WORKERS chunks - one browser launched per chunk
        chunks = [todo[i::MAX_WORKERS] for i in range(MAX_WORKERS)]
        print(f"Split into {len(chunks)} chunks (~{len(todo)//MAX_WORKERS} URLs each)\n")

        t0 = time.time()
        processed = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for chunk_results in executor.map(process_url_batch, chunks):
                for r in chunk_results:
                    results.append(r)
                    processed += 1
                    if r.get("render_error"):
                        errors += 1

                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(todo) - processed) / rate / 60 if rate > 0 else 0
                print(f"  [{processed:>5}/{len(todo):>5}] errors={errors} "
                      f"rate={rate:.1f}/s ETA={eta:.1f}min")

                if processed % CHECKPOINT_EVERY < len(chunk_results):
                    pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
                    print(f"  Checkpoint saved ({len(results):,} rows)")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)
    elapsed_total = time.time() - t0 if 't0' in dir() else 0

    print("\n" + "=" * 65)
    print("RENDERED-PAGE EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"Total processed : {len(out_df):,}")
    print(f"Render errors   : {out_df['render_error'].sum():,} "
          f"({out_df['render_error'].mean()*100:.1f}%)")
    print(f"Time taken      : {elapsed_total/60:.1f} min")
    print(f"Output saved to : {OUTPUT_FILE}")
    print("=" * 65)

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()


if __name__ == "__main__":
    main()
