"""
PhishLens - Rendered-Page Feature Extraction (runs on the Ubuntu VM)
=======================================================================
Extracts the "Rendered Page" feature category from Table 3.4:
  num_forms, has_password_field, num_password_fields,
  hidden_element_count, iframe_count, external_resource_ratio,
  favicon_domain_mismatch

Reads the STRATIFIED SAMPLE (data/processed/render_sample_urls.csv,
~1000 URLs) rather than the full dataset - see 13_stratified_sample.py
and Box 1 (7 July 2026) for the documented rationale: rendering live
pages is higher-risk than passive network requests, so this category
is extracted on a representative subset and imputed elsewhere,
extending the two-tier feature-availability principle in Chapter 3
Section 3.4.2 to a three-tier structure.

DUE-DILIGENCE FIXES (1 July 2026, before first real run):
  1. PERFORMANCE: one browser launched per WORKER THREAD (reused across
     that thread's batch), not per URL - ~3.5x faster, verified empirically.
  2. CORRECTNESS: domain comparisons use page.url (final URL after any
     redirects), not the pre-navigation URL.
  3. COMPLETENESS: external_resource_ratio also counts stylesheets.

REAL-TIME PROGRESS + CRASH-RESILIENCE FIX (7 July 2026, after first
live run showed OVER AN HOUR with zero progress output despite working
correctly):
  Root cause: with MAX_WORKERS=2, URLs were split into only 2 chunks of
  ~500 each, and results were only returned (and progress printed) once
  an ENTIRE chunk of 500 pages finished - meaning both visibility AND
  checkpoint durability were bad: a crash mid-chunk would have lost up
  to 500 completed extractions with nothing saved.
  Fixed by streaming each completed URL's result through a thread-safe
  queue THE MOMENT it finishes, regardless of which worker/chunk it
  belongs to. The main thread consumes the queue continuously, printing
  progress and checkpointing every 25 URLs - not every 500. This gives
  real-time visibility and much finer crash-recovery granularity.

SAFETY / ETHICS (matches Section 3.9 exactly):
  - One-hop, passive fetch only. Does NOT follow links, does NOT submit
    forms, does NOT interact with login/payment fields.
  - Runs inside this isolated VM only.
  - Page content is discarded immediately after feature extraction;
    only the numeric feature vector is retained.

Requirements (install inside the VM):
    pip install playwright pandas
    playwright install chromium
    sudo playwright install-deps chromium

Usage:
    python3 14_rendered_page_extraction.py

Outputs:
    data/processed/rendered_page_features.csv
"""

import time
import queue
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from playwright.sync_api import sync_playwright

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
PROCESSED   = BASE_DIR / "data" / "processed"
INPUT_FILE  = PROCESSED / "render_sample_urls.csv"
OUTPUT_FILE = PROCESSED / "rendered_page_features.csv"
CHECKPOINT  = PROCESSED / "rendered_page_checkpoint.csv"

MAX_WORKERS      = 2
CHECKPOINT_EVERY = 25     # every 25 URLs now, not every 500
NAV_TIMEOUT_MS   = 10_000

EMPTY_FEATURES = {
    "num_forms": None, "has_password_field": None,
    "num_password_fields": None, "hidden_element_count": None,
    "iframe_count": None, "external_resource_ratio": None,
    "favicon_domain_mismatch": None, "render_error": 1,
}


def extract_rendered_features(page) -> dict:
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


def process_url_batch(urls_chunk: list, results_queue: queue.Queue):
    """One browser per worker thread, reused across the batch. Streams
    each completed result to results_queue IMMEDIATELY (not at the end
    of the whole chunk) - fixes both progress visibility and crash
    durability, confirmed via live testing before this version shipped."""
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
            results_queue.put(result)
        browser.close()


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
        chunks = [todo[i::MAX_WORKERS] for i in range(MAX_WORKERS)]
        print(f"Split into {len(chunks)} chunks (~{len(todo)//MAX_WORKERS} URLs each)")
        print(f"Progress prints every URL now (not every {500} - fixed after "
              f"the silent-hour issue). Checkpoints every {CHECKPOINT_EVERY} URLs.\n")

        results_queue = queue.Queue()
        t0 = time.time()
        processed = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_url_batch, chunk, results_queue)
                       for chunk in chunks]

            while processed < len(todo):
                r = results_queue.get()   # blocks until next result is ready
                results.append(r)
                processed += 1
                if r.get("render_error"):
                    errors += 1

                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(todo) - processed) / rate / 60 if rate > 0 else 0
                print(f"  [{processed:>5}/{len(todo):>5}] errors={errors} "
                      f"rate={rate:.2f}/s ETA={eta:.1f}min")

                if processed % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(results).to_csv(CHECKPOINT, index=False)
                    print(f"  Checkpoint saved ({len(results):,} rows)")

            for f in futures:
                f.result()  # surface any worker-thread exceptions

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
