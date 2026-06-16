"""
PhishLens - Data Acquisition Script
=====================================
Downloads and performs initial loading of the three dataset sources:
  1. PhiUSIIL (historical base, ~235,795 URLs with 54-56 pre-extracted features)
  2. PhishTank (live phishing URLs, via public API)
  3. Tranco Top 1M (live legitimate URLs, via weekly list)

Run this on YOUR machine (not in a restricted sandbox) since it requires
access to:
  - archive.ics.uci.edu (UCI ML Repository)
  - data.phishtank.com (PhishTank API)
  - tranco-list.eu (Tranco)

Usage:
    python 01_download_data.py

Outputs (saved to data/raw/):
    phiusiil_raw.csv
    phishtank_raw.csv
    tranco_top1m.csv
"""

import os
import csv
import json
import time
import zipfile
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. PhiUSIIL — via ucimlrepo
# ---------------------------------------------------------------------------
def download_phiusiil():
    """
    Downloads PhiUSIIL (UCI dataset id=967) using the ucimlrepo package.
    Install first with: pip install ucimlrepo
    """
    print("\n[1/3] Downloading PhiUSIIL dataset (UCI repo id=967)...")
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("  ucimlrepo not installed. Run: pip install ucimlrepo")
        return False

    try:
        dataset = fetch_ucirepo(id=967)
        X = dataset.data.features
        y = dataset.data.targets

        # Combine features and target into one DataFrame
        df = X.copy()
        df["label"] = y

        out_path = RAW_DIR / "phiusiil_raw.csv"
        df.to_csv(out_path, index=False)

        print(f"  Saved {len(df):,} rows to {out_path}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)[:10]} ...")
        print(f"  Label distribution:\n{df['label'].value_counts()}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Fallback: manually download from")
        print("  https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset")
        print(f"  and place the CSV at: {RAW_DIR / 'phiusiil_raw.csv'}")
        return False


# ---------------------------------------------------------------------------
# 2. PhishTank — via public API (online valid phishing feed, JSON)
# ---------------------------------------------------------------------------
def download_phishtank(app_key=None):
    """
    Downloads the PhishTank 'online valid' phishing feed.

    PhishTank provides a free data feed at:
        http://data.phishtank.com/data/online-valid.json

    An app_key is optional but recommended (register at phishtank.org)
    to avoid stricter rate limiting:
        http://data.phishtank.com/data/<app_key>/online-valid.json
    """
    print("\n[2/3] Downloading PhishTank feed (verified phishing URLs)...")

    if app_key:
        url = f"http://data.phishtank.com/data/{app_key}/online-valid.json"
    else:
        url = "http://data.phishtank.com/data/online-valid.json"

    out_path = RAW_DIR / "phishtank_raw.csv"

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "PhishLens-Research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        print(f"  Retrieved {len(data):,} verified phishing entries")

        # Flatten into a simple CSV: url, target, submission_time, verified
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "phish_id", "target", "submission_time", "verified", "online"])
            for entry in data:
                writer.writerow([
                    entry.get("url", ""),
                    entry.get("phish_id", ""),
                    entry.get("target", ""),
                    entry.get("submission_time", ""),
                    entry.get("verified", ""),
                    entry.get("online", ""),
                ])

        print(f"  Saved to {out_path}")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Fallback: manually download from")
        print("  https://www.phishtank.com/developer_info.php")
        print(f"  and place the JSON/CSV at: {out_path}")
        return False


# ---------------------------------------------------------------------------
# 3. Tranco Top 1M — via tranco-list.eu
# ---------------------------------------------------------------------------
def download_tranco():
    """
    Downloads the latest Tranco Top 1M list.

    Tranco provides a stable, citable list at:
        https://tranco-list.eu/top-1m.csv.zip
    (this is a redirect to the latest list)
    """
    print("\n[3/3] Downloading Tranco Top 1M list...")

    url = "https://tranco-list.eu/top-1m.csv.zip"
    zip_path = RAW_DIR / "tranco_top1m.csv.zip"
    out_path = RAW_DIR / "tranco_top1m.csv"

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "PhishLens-Research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(zip_path, "wb") as f:
                f.write(resp.read())

        with zipfile.ZipFile(zip_path, "r") as z:
            # Tranco zips usually contain a single CSV
            names = z.namelist()
            csv_name = names[0]
            z.extract(csv_name, RAW_DIR)
            extracted_path = RAW_DIR / csv_name
            if extracted_path != out_path:
                extracted_path.rename(out_path)

        zip_path.unlink(missing_ok=True)

        # Quick sanity check
        with open(out_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)

        print(f"  Saved {line_count:,} ranked domains to {out_path}")
        print("  Format: rank,domain (no header)")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Fallback: manually download from https://tranco-list.eu")
        print(f"  and place the CSV at: {out_path}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PhishLens Data Acquisition")
    print("=" * 60)
    print(f"Output directory: {RAW_DIR}")

    results = {
        "PhiUSIIL": download_phiusiil(),
        "PhishTank": download_phishtank(app_key=None),  # add your app_key if you have one
        "Tranco": download_tranco(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED (see fallback instructions above)"
        print(f"  {name}: {status}")

    print("\nNext step: run 02_explore_data.py to inspect the downloaded files.")
