"""
PhishLens - Phase 1b: Filter to Website Records & Extract Visible Text
========================================================================
combined_reduced.json mixes HTML pages, bare URLs, and other text with
no source field distinguishing them. This script filters to records that
are GENUINE HTML webpages (by structural heuristic), then strips HTML
down to visible text (title, headings, body) using BeautifulSoup -
matching what the Chrome extension will extract from a live page.

Usage:
    python src/semantic_trained_experiment/03_filter_extract_website_text.py

Requirements:
    pip install beautifulsoup4 lxml

Outputs:
    data/processed/website_texts.csv   - url_like_index, visible_text, label
"""

import json
import re
from pathlib import Path
from collections import Counter

from bs4 import BeautifulSoup
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW      = BASE_DIR / "data" / "raw"
PROCESSED = BASE_DIR / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

INPUT_FILE  = RAW / "combined_reduced.json"
OUTPUT_FILE = PROCESSED / "website_texts.csv"

# Heuristic markers that indicate genuine HTML page source
HTML_MARKERS = ["<!doctype", "<html", "<body", "<head", "<title", "<meta", "<div"]
MIN_HTML_MARKERS = 2       # require at least this many markers to count as HTML
MIN_TEXT_LENGTH   = 50     # after stripping tags, require this many chars of real text


def looks_like_html(text: str) -> bool:
    if not isinstance(text, str) or len(text) < 100:
        return False
    lower = text.lower()
    hits = sum(1 for marker in HTML_MARKERS if marker in lower)
    return hits >= MIN_HTML_MARKERS


def extract_visible_text(html: str) -> str:
    """Strip HTML down to visible text - title, headings, body paragraphs."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Remove non-visible elements entirely
    for tag in soup(["script", "style", "noscript", "head", "meta", "link"]):
        tag.decompose()

    # Title separately (often carries strong signal, e.g. "Verify Your Account")
    title = soup.title.get_text(strip=True) if soup.title else ""

    body_text = soup.get_text(separator=" ", strip=True)
    combined = f"{title} . {body_text}" if title else body_text

    # Collapse excess whitespace
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


def main():
    print("=" * 66)
    print("PhishLens - Filter to Website Records & Extract Visible Text")
    print("=" * 66)

    print(f"\nLoading {INPUT_FILE.name} ...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Total records: {len(data):,}")

    print(f"\nFiltering to genuine HTML records "
          f"(>= {MIN_HTML_MARKERS} structural markers) ...")
    html_records = [d for d in data if looks_like_html(d.get("text", ""))]
    print(f"  HTML-like records found: {len(html_records):,} "
          f"({len(html_records)/len(data)*100:.1f}% of total)")

    label_counts = Counter(d["label"] for d in html_records)
    print(f"  Label distribution among HTML records: {dict(label_counts)}")

    if len(html_records) < 500:
        print("\n  WARNING: very few HTML records found. Check MIN_HTML_MARKERS")
        print("  or inspect a few raw samples to adjust the heuristic.")

    print(f"\nExtracting visible text from {len(html_records):,} HTML records ...")
    print("(This may take a few minutes for large record counts)")

    rows = []
    errors = 0
    for i, rec in enumerate(html_records):
        try:
            visible = extract_visible_text(rec["text"])
            if len(visible) >= MIN_TEXT_LENGTH:
                rows.append({
                    "index": i,
                    "visible_text": visible,
                    "label": rec["label"],
                })
        except Exception:
            errors += 1

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1:,}/{len(html_records):,} "
                  f"(kept {len(rows):,}, errors {errors})")

    print(f"\n  Extraction complete. Kept {len(rows):,} records "
          f"(dropped {len(html_records)-len(rows):,} too-short/errored).")

    df = pd.DataFrame(rows)
    print(f"\nFinal label distribution:")
    print(f"  label=0: {(df['label']==0).sum():,}")
    print(f"  label=1: {(df['label']==1).sum():,}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")

    print("\n" + "=" * 66)
    print("Sample extracted visible text (first HTML record):")
    print("-" * 66)
    if len(df) > 0:
        print(df.iloc[0]["visible_text"][:300])
    print("=" * 66)
    print("\nReview the label balance and sample text above, then paste this")
    print("output back before we proceed to the training step.")


if __name__ == "__main__":
    main()
