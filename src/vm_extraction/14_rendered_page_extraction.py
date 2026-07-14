"""
PhishLens - Rendered-Page Feature Extraction (VM) - v2 FIXED
=======================================================================
Extracts the "Rendered Page" feature category from Table 3.4:
  num_forms, has_password_field, num_password_fields,
  hidden_element_count, iframe_count, external_resource_ratio,
  favicon_domain_mismatch

ALL FIXES APPLIED (13 July 2026):
  1. Realistic Chrome user agent (was "PhishLens Research Bot" -
     legit sites' bot protection was rejecting it, causing the
     inverted failure pattern where Tranco failed MORE than phishing)
  2. NAV_TIMEOUT raised 10s -> 15s
  3. wait_until="domcontentloaded" (never full "load" - heavy legit
     pages burn the timeout waiting for ads/trackers)
  4. www-retry: Tranco lists bare domains; on ERR_NAME_NOT_RESOLVED
     retries once with www. prefix
  5. MAX_WORKERS reduced 8 -> 4 (8 parallel headless contexts from one
     VPN IP looks bot-like and triggers rate limiting)
  6. Timezone-aware timestamps (fixes utcnow() deprecation warning)
  7. Built-in --test mode: 10 Tranco + 5 PhishTank URLs, sequential,
     prints real error text. RUN THIS FIRST, EVERY TIME.

SAFETY / ETHICS (matches Section 3.9):
  - One-hop, passive fetch only. No link-following, no form submission.
  - Run inside the isolated VM only.
  - Page content discarded after feature extraction; only the numeric
    feature vector is retained.

METHODOLOGY NOTE (for Chapter 3): a standard desktop Chrome user-agent
string is used because commercial bot protection on legitimate sites
and cloaking logic in phishing kits both suppress content served to
self-identified automated clients, which would bias feature extraction.

Usage:
    python3 14_rendered_page_extraction.py --test    # 2-min sanity check
    python3 14_rendered_page_extraction.py           # full run
"""

import sys
import time
import datetime
import threading
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from playwright.sync_api import sync_playwright

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
PROCESSED   = BASE_DIR / "data" / "processed"
SAMPLE_FILE = PROCESSED / "render_sample_urls.csv"
OUTPUT_FILE = PROCESSED / "rendered_page_features.csv"
CHECKPOINT  = PROCESSED / "rendered_page_checkpoint.csv"

MAX_WORKERS      = 4          # FIX 5: was 8
CHECKPOINT_EVERY = 25
NAV_TIMEOUT_MS   = 15_000     # FIX 2: was 10_000

# FIX 1: realistic desktop Chrome UA (was "PhishLens Research Bot")
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")

EMPTY_FEATURES = {
    "num_forms": None, "has_password_field": None,
    "num_password_fields": None, "hidden_element_count": None,
    "iframe_count": None, "external_resource_ratio": None,
    "favicon_domain_mismatch": None,
}

_print_lock = threading.Lock()


def now_iso() -> str:
    """FIX 6: timezone-aware UTC timestamp (no deprecation warning)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def navigate_with_retry(page, url: str):
    """FIX 3 + FIX 4: domcontentloaded, and www-retry on DNS failure."""
    try:
        return page.goto(url, wait_until="domcontentloaded",
                         timeout=NAV_TIMEOUT_MS), url
    except Exception as first_err:
        if "ERR_NAME_NOT_RESOLVED" in str(first_err) and "://www." not in url:
            www_url = url.replace("://", "://www.", 1)
            return page.goto(www_url, wait_until="domcontentloaded",
                             timeout=NAV_TIMEOUT_MS), www_url
        raise


def extract_rendered_features(page, page_url: str) -> dict:
    """Extract Table 3.4 'Rendered Page' features from a loaded page.
    Uses the final URL after redirects, not the pre-navigation URL."""
    final_domain = urlparse(page.url).netloc.lower()

    counts = page.evaluate("""() => {
        const forms = document.querySelectorAll('form').length;
        const pwds  = document.querySelectorAll('input[type=password]').length;
        const iframes = document.querySelectorAll('iframe').length;

        let hidden = 0;
        for (const el of document.querySelectorAll('*')) {
            const st = window.getComputedStyle(el);
            if (st.display === 'none' || st.visibility === 'hidden') hidden++;
            if (hidden > 500) break;   // cap the walk on huge pages
        }

        const res = [];
        for (const el of document.querySelectorAll('script[src],link[href],img[src]')) {
            const u = el.src || el.href;
            if (u) res.push(u);
        }

        let favicon = null;
        const fav = document.querySelector(
            'link[rel~="icon"], link[rel="shortcut icon"]');
        if (fav && fav.href) favicon = fav.href;

        return {forms, pwds, iframes, hidden, res, favicon};
    }""")

    ext = 0
    total = 0
    for r in counts["res"]:
        try:
            d = urlparse(r).netloc.lower()
        except Exception:
            continue
        if not d:
            continue
        total += 1
        if d != final_domain and not d.endswith("." + final_domain):
            ext += 1
    ratio = round(ext / total, 4) if total else 0.0

    fav_mismatch = 0
    if counts["favicon"]:
        try:
            fd = urlparse(counts["favicon"]).netloc.lower()
            if fd and fd != final_domain and not fd.endswith("." + final_domain):
                fav_mismatch = 1
        except Exception:
            pass

    return {
        "num_forms": counts["forms"],
        "has_password_field": 1 if counts["pwds"] > 0 else 0,
        "num_password_fields": counts["pwds"],
        "hidden_element_count": counts["hidden"],
        "iframe_count": counts["iframes"],
        "external_resource_ratio": ratio,
        "favicon_domain_mismatch": fav_mismatch,
    }


def process_url(playwright, url: str) -> dict:
    """Render one URL and return its feature row. Never raises."""
    row = {"url": url, "processed_at": now_iso()}
    browser = None
    try:
        browser = playwright.chromium.launch(
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(
            user_agent=USER_AGENT,           # FIX 1
            ignore_https_errors=True,
            viewport={"width": 1366, "height": 768},
        )
        page = context.new_page()
        page.set_default_timeout(NAV_TIMEOUT_MS)
        _, final_input_url = navigate_with_retry(page, url)   # FIX 3+4
        feats = extract_rendered_features(page, final_input_url)
        row.update(feats)
        row["render_error"] = 0
        row["error_type"] = ""
    except Exception as e:
        row.update(EMPTY_FEATURES)
        row["render_error"] = 1
        # keep a short machine-readable error class for later analysis
        msg = str(e)
        if "ERR_NAME_NOT_RESOLVED" in msg:
            row["error_type"] = "dns"
        elif "ERR_CONNECTION_REFUSED" in msg:
            row["error_type"] = "refused"
        elif "Timeout" in msg or "timeout" in msg:
            row["error_type"] = "timeout"
        elif "ERR_CONNECTION_RESET" in msg:
            row["error_type"] = "reset"
        elif "ERR_ABORTED" in msg:
            row["error_type"] = "aborted"
        else:
            row["error_type"] = "other"
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass
    return row


def run_test_mode():
    """FIX 7: 2-minute sanity check before any full run."""
    print("=" * 66)
    print("TEST MODE - 10 Tranco + 5 PhishTank URLs, sequential")
    print("=" * 66)
    df = pd.read_csv(SAMPLE_FILE)
    urls = (df[df["source"] == "tranco"]["url"].head(10).tolist()
            + df[df["source"] == "phishtank"]["url"].head(5).tolist())
    ok_tranco = 0
    with sync_playwright() as p:
        for i, url in enumerate(urls, 1):
            src = "tranco   " if i <= 10 else "phishtank"
            row = process_url(p, url)
            if row["render_error"] == 0:
                if i <= 10:
                    ok_tranco += 1
                print(f"[{i:2d}] {src} OK    {url[:60]}")
            else:
                print(f"[{i:2d}] {src} FAIL  {url[:60]}  ({row['error_type']})")
    print("-" * 66)
    print(f"Tranco success: {ok_tranco}/10")
    if ok_tranco >= 7:
        print("PASS - fix confirmed. Run the full extraction now:")
        print("    python3 src/vm_extraction/14_rendered_page_extraction.py")
    else:
        print("FAIL - do NOT start the full run. Paste this output to Claude.")
    print("=" * 66)


def main():
    if "--test" in sys.argv:
        run_test_mode()
        return

    print("=" * 66)
    print("PhishLens - Rendered-Page Feature Extraction (VM) - v2 FIXED")
    print("=" * 66)

    df = pd.read_csv(SAMPLE_FILE)
    urls = df["url"].tolist()
    print(f"\nURLs to process: {len(urls):,}")

    done_rows = []
    done_urls = set()
    if CHECKPOINT.exists():
        cp = pd.read_csv(CHECKPOINT)
        done_rows = cp.to_dict("records")
        done_urls = set(cp["url"])
        print(f"Checkpoint found - resuming from {len(done_urls)} done")

    todo = [u for u in urls if u not in done_urls]
    print(f"Remaining: {len(todo)}")
    print(f"Workers: {MAX_WORKERS} | Timeout: {NAV_TIMEOUT_MS//1000}s | "
          f"UA: realistic Chrome | Checkpoints every {CHECKPOINT_EVERY} URLs\n")

    if not todo:
        print("Nothing to do - all URLs already processed.")
    else:
        results = list(done_rows)
        start = time.time()
        errors = sum(r.get("render_error", 0) for r in done_rows)
        completed = 0

        def worker(url):
            with sync_playwright() as p:
                return process_url(p, url)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(worker, u): u for u in todo}
            for fut in as_completed(futures):
                row = fut.result()
                results.append(row)
                completed += 1
                errors += row["render_error"]
                rate = completed / max(time.time() - start, 1)
                eta = (len(todo) - completed) / max(rate, 0.01) / 60
                with _print_lock:
                    print(f"  [{completed:5d}/{len(todo):5d}] "
                          f"errors={errors} rate={rate:.2f}/s ETA={eta:.1f}min")
                if completed % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

        out = pd.DataFrame(results)
        # carry source/label columns through for downstream scripts
        out = out.merge(df.drop_duplicates("url"), on="url", how="left")
        out.to_csv(OUTPUT_FILE, index=False)
        if CHECKPOINT.exists():
            CHECKPOINT.unlink()

        n_err = int(out["render_error"].sum())
        print("\n" + "=" * 66)
        print("RENDERED-PAGE EXTRACTION COMPLETE")
        print("=" * 66)
        print(f"Total processed : {len(out):,}")
        print(f"Render errors   : {n_err} ({n_err/len(out)*100:.1f}%)")
        print(f"Time taken      : {(time.time()-start)/60:.1f} min")
        print(f"Output saved to : {OUTPUT_FILE}")
        print("\nError breakdown by source:")
        print(out.groupby("source")["render_error"]
                 .agg(["sum", "count", "mean"]).round(3))
        print("\nError types:")
        print(out[out["render_error"] == 1]["error_type"]
                 .value_counts().to_string())
        print("=" * 66)


if __name__ == "__main__":
    main()
