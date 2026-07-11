"""
PhishLens - Diagnostic: Why Are Renders Failing?
====================================================
Tests the first 10 PhishTank URLs from the sample ONE AT A TIME,
printing the REAL error message for each failure - not just
render_error=1. Fast (~1-2 min), run this instead of guessing blindly.

Usage:
    python3 16_diagnose_render_failures.py
"""

import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLE   = BASE_DIR / "data" / "processed" / "render_sample_urls.csv"

df = pd.read_csv(SAMPLE)
phishtank_urls = df[df["source"] == "phishtank"]["url"].head(10).tolist()

print("=" * 70)
print("Diagnostic: testing 10 PhishTank URLs, showing REAL error messages")
print("=" * 70)

with sync_playwright() as p:
    browser = p.chromium.launch(
        args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"]
    )
    for i, url in enumerate(phishtank_urls, 1):
        print(f"\n[{i}] {url}")
        try:
            context = browser.new_context(
                user_agent="Mozilla/5.0 (PhishLens Research Bot; one-hop passive fetch)",
                ignore_https_errors=True,
            )
            page = context.new_page()
            page.set_default_timeout(10000)
            resp = page.goto(url, wait_until="domcontentloaded", timeout=10000)
            status = resp.status if resp else "no response object"
            final_url = page.url
            title = page.title()
            print(f"    SUCCESS - status={status} final_url={final_url[:80]}")
            print(f"    title={title!r}")
            context.close()
        except Exception as e:
            print(f"    FAILED - {type(e).__name__}: {str(e)[:200]}")
    browser.close()

print("\n" + "=" * 70)
print("Paste this FULL output back - the real error text tells us exactly")
print("what's going wrong (timeout? DNS? bot-blocking? something else).")
print("=" * 70)
