"""
PhishLens - Refresh Stale PhishTank URLs for Rendering
==========================================================
Your original PhishTank URLs (collected weeks ago) are now ~80% dead -
phishing pages are typically taken down within hours to days of being
reported, so this is expected, not a bug (see Box 1, 7 July 2026).

This script replaces the PhishTank rows in render_sample_urls.csv with
FRESH, currently-live PhishTank URLs, using their online-valid feed
(which by definition only contains currently-online, verified phishes).
Tranco (legitimate) rows are left untouched - those were rendering fine.

Run this on WINDOWS, not the VM - it's a plain HTTP download and
liveness check, no page rendering/JS execution involved, so there is
no elevated risk requiring VM isolation.

Requirements:
    pip install requests pandas

Usage:
    python src/vm_extraction/15_refresh_phishtank_sample.py

Outputs:
    data/processed/render_sample_urls.csv   (overwritten with fresh
                                              PhishTank rows + original
                                              Tranco rows)
"""

import time
import requests
import pandas as pd
from pathlib import Path
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
PROCESSED  = BASE_DIR / "data" / "processed"
SAMPLE_FILE = PROCESSED / "render_sample_urls.csv"

PHISHTANK_FEED = "http://data.phishtank.com/data/online-valid.csv"
# If you have a PhishTank API key from your original data collection,
# paste it here for a higher rate limit. Leave blank for anonymous access.
API_KEY = ""

# FALLBACK: PhishTank has a documented history of blocking automated/
# script-based downloads (curl, bots) with 403/SSL errors even with a
# valid API key, while the identical URL works fine in a normal browser
# (confirmed via multiple independent reports, e.g. MISP issue #9855,
# pfSense forum threads). If the automated fetch below fails, manually
# download the CSV via your browser from the URL above (or with your
# API key inserted) and save it here - the script will use it directly
# instead of attempting the network call again:
MANUAL_DOWNLOAD_FALLBACK = PROCESSED / "online-valid.csv"

LIVENESS_TIMEOUT = 5
MAX_WORKERS = 15
# Browser-like User-Agent as a mitigation for the automated-download
# blocking issue described above (not guaranteed to help, but reports
# suggest PhishTank's blocking is often keyed on obviously bot-like UAs)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def fetch_phishtank_feed() -> pd.DataFrame:
    # Check manual-download fallback FIRST if it exists (means a previous
    # automated attempt failed and you downloaded it by hand instead)
    if MANUAL_DOWNLOAD_FALLBACK.exists():
        print(f"Using manually-downloaded file: {MANUAL_DOWNLOAD_FALLBACK.name}")
        df = pd.read_csv(MANUAL_DOWNLOAD_FALLBACK)
        print(f"  Feed contains {len(df):,} currently-online verified phishes")
        return df

    url = PHISHTANK_FEED
    if API_KEY:
        url = f"https://data.phishtank.com/data/{API_KEY}/online-valid.csv"

    print(f"Fetching PhishTank online-valid feed ...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        print(f"  Feed contains {len(df):,} currently-online verified phishes")
        return df
    except Exception as e:
        print(f"\n  AUTOMATED FETCH FAILED: {e}")
        print(f"  This is a known issue - PhishTank sometimes blocks automated")
        print(f"  downloads even with a valid API key, while browser access works.")
        print(f"\n  FALLBACK: open this URL in your normal web browser:")
        print(f"    {url}")
        print(f"  Save the downloaded file as:")
        print(f"    {MANUAL_DOWNLOAD_FALLBACK}")
        print(f"  Then re-run this script - it will use the saved file automatically.")
        raise SystemExit(1)


def check_alive(url: str) -> bool:
    try:
        r = requests.head(url, timeout=LIVENESS_TIMEOUT, headers=HEADERS,
                          allow_redirects=True)
        if r.status_code >= 400:
            # some servers reject HEAD; retry with GET before giving up
            r = requests.get(url, timeout=LIVENESS_TIMEOUT, headers=HEADERS,
                             stream=True)
        return r.status_code < 400
    except Exception:
        return False


def main():
    print("=" * 62)
    print("PhishLens - Refresh Stale PhishTank URLs")
    print("=" * 62)

    sample = pd.read_csv(SAMPLE_FILE)
    n_phishtank_needed = (sample["source"] == "phishtank").sum()
    tranco_rows = sample[sample["source"] == "tranco"].copy()
    print(f"\nCurrent sample: {len(sample):,} rows")
    print(f"  PhishTank rows to replace: {n_phishtank_needed:,}")
    print(f"  Tranco rows kept as-is   : {len(tranco_rows):,}")

    feed = fetch_phishtank_feed()
    candidate_urls = feed["url"].dropna().unique().tolist()
    # Shuffle isn't needed - feed order is roughly submission-time, mixing
    # is fine for our purposes since we just need N live ones.
    print(f"\nChecking liveness of candidates (need {n_phishtank_needed:,} "
          f"live URLs) ...")

    live_urls = []
    t0 = time.time()
    checked = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        idx = 0
        # submit an initial batch
        batch_size = min(300, len(candidate_urls))
        for u in candidate_urls[:batch_size]:
            futures[executor.submit(check_alive, u)] = u
        idx = batch_size

        while futures and len(live_urls) < n_phishtank_needed:
            for future in as_completed(list(futures.keys())):
                url = futures.pop(future)
                checked += 1
                try:
                    if future.result():
                        live_urls.append(url)
                except Exception:
                    pass

                if checked % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  Checked {checked:,} | Live found: {len(live_urls):,}"
                          f"/{n_phishtank_needed:,} | {elapsed:.0f}s elapsed")

                if len(live_urls) >= n_phishtank_needed:
                    break

                # top up the queue if running low and more candidates exist
                if len(futures) < 20 and idx < len(candidate_urls):
                    next_batch = candidate_urls[idx:idx+100]
                    for u in next_batch:
                        futures[executor.submit(check_alive, u)] = u
                    idx += len(next_batch)

    print(f"\nFound {len(live_urls):,} live PhishTank URLs "
          f"(checked {checked:,} candidates)")

    if len(live_urls) < n_phishtank_needed:
        print(f"  WARNING: only found {len(live_urls)}, needed {n_phishtank_needed}. "
              f"Using all found; consider a larger candidate batch.")

    fresh_phishtank = pd.DataFrame({
        "url": live_urls[:n_phishtank_needed],
        "label": 0,
        "source": "phishtank",
    })

    updated_sample = pd.concat([fresh_phishtank, tranco_rows], ignore_index=True)
    updated_sample.to_csv(SAMPLE_FILE, index=False)

    print(f"\nSaved refreshed sample -> {SAMPLE_FILE}")
    print(f"  New PhishTank rows: {len(fresh_phishtank):,} (freshly verified live)")
    print(f"  Tranco rows kept  : {len(tranco_rows):,}")
    print("\n" + "=" * 62)
    print("Refresh complete. Re-run 14_rendered_page_extraction.py on the")
    print("VM against this updated sample - PhishTank rows should now")
    print("render successfully instead of mostly failing.")
    print("=" * 62)


if __name__ == "__main__":
    main()
