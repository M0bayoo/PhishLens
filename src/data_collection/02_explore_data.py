"""
PhishLens - Data Exploration Script
=====================================
Run this AFTER 01_download_data.py to inspect the raw datasets:
  - Column names and types
  - Row counts and label balance
  - Sample rows
  - Missing value summary

This helps confirm the actual schema of each source before we build
the feature-mapping and merge pipeline (02b_merge_pipeline.py).

Usage:
    python 02_explore_data.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def explore_phiusiil():
    path = RAW_DIR / "phiusiil_raw.csv"
    print("\n" + "=" * 60)
    print("PhiUSIIL")
    print("=" * 60)

    if not path.exists():
        print(f"  NOT FOUND: {path}")
        print("  Run 01_download_data.py first.")
        return

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    if "label" in df.columns:
        print(f"\n  Label distribution:")
        print(df["label"].value_counts())

    print(f"\n  First 3 rows (selected columns):")
    preview_cols = [c for c in df.columns[:8]]
    print(df[preview_cols].head(3).to_string())

    print(f"\n  Missing values per column (top 10):")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(10) if missing.sum() > 0 else "  None")


def explore_phishtank():
    path = RAW_DIR / "phishtank_raw.csv"
    print("\n" + "=" * 60)
    print("PhishTank")
    print("=" * 60)

    if not path.exists():
        print(f"  NOT FOUND: {path}")
        print("  Run 01_download_data.py first.")
        return

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())

    if "online" in df.columns:
        print(f"\n  'online' status distribution:")
        print(df["online"].value_counts())


def explore_tranco():
    path = RAW_DIR / "tranco_top1m.csv"
    print("\n" + "=" * 60)
    print("Tranco Top 1M")
    print("=" * 60)

    if not path.exists():
        print(f"  NOT FOUND: {path}")
        print("  Run 01_download_data.py first.")
        return

    df = pd.read_csv(path, header=None, names=["rank", "domain"])
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\n  First 5 (top ranked):")
    print(df.head(5).to_string(index=False))
    print(f"\n  Sample from 10,000-100,000 tier:")
    print(df.iloc[10000:10005].to_string(index=False))
    print(f"\n  Sample from 100,000-1,000,000 tier:")
    print(df.iloc[100000:100005].to_string(index=False))


if __name__ == "__main__":
    explore_phiusiil()
    explore_phishtank()
    explore_tranco()

    print("\n" + "=" * 60)
    print("Exploration complete.")
    print("Share this output so we can build the feature-mapping pipeline")
    print("based on the ACTUAL columns available in PhiUSIIL.")
    print("=" * 60)
