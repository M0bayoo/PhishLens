"""
PhishLens - Dataset Merge Pipeline
=====================================
Combines the three raw data sources into one unified dataset:

  1. PhiUSIIL    - historical base, already has 55 features + label
  2. PhishTank    - live phishing URLs (label assigned = 0 / phishing)
  3. Tranco       - live legitimate domains (label assigned = 1 / legitimate)

This script does NOT yet run full feature extraction on PhishTank/Tranco
(that happens in 04_feature_extraction.py). Here we:
  - Load and standardise all three sources
  - Confirm/repair label convention (1=legitimate, 0=phishing)
  - Sample Tranco across rank tiers per methodology Section 3.4.1
  - Tag every row with its source, for traceability
  - Save an intermediate "unified_raw.csv" that feature extraction will
    then enrich with the columns PhiUSIIL is missing (TLS, domain age, etc.)

Usage:
    python 03_merge_pipeline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Label convention CONFIRMED via UCI documentation (archive.ics.uci.edu/dataset/967):
#   1 = legitimate, 0 = phishing
LABEL_LEGITIMATE = 1
LABEL_PHISHING = 0

# How many fresh URLs to sample from PhishTank and Tranco.
# Kept moderate for the first pass — full feature extraction (TLS lookups,
# DNS queries, headless rendering) is comparatively slow, so we don't want
# to commit to extracting features for hundreds of thousands of fresh URLs
# before the pipeline is validated end-to-end on a smaller batch.
N_PHISHTANK_SAMPLE = 5000
N_TRANCO_PER_TIER = 2000  # x3 tiers = 6000 legitimate fresh URLs


def load_phiusiil():
    print("\n[1/3] Loading PhiUSIIL...")
    path = RAW_DIR / "phiusiil_raw.csv"
    df = pd.read_csv(path)

    # Sanity check the label convention against known composition
    counts = df["label"].value_counts()
    print(f"  Loaded {len(df):,} rows")
    print(f"  label=1 (expected legitimate): {counts.get(1, 0):,}")
    print(f"  label=0 (expected phishing):   {counts.get(0, 0):,}")

    if counts.get(1, 0) < counts.get(0, 0):
        print("  WARNING: label=1 count is smaller than label=0. "
              "This contradicts the documented PhiUSIIL composition "
              "(134,850 legitimate vs 100,945 phishing). Verify before proceeding.")

    df["source"] = "phiusiil"
    df["url_col"] = df["URL"]  # standardised reference column across sources
    return df


def load_phishtank_sample(n=N_PHISHTANK_SAMPLE, seed=42):
    print(f"\n[2/3] Loading PhishTank sample (n={n})...")
    path = RAW_DIR / "phishtank_raw.csv"
    df = pd.read_csv(path)
    print(f"  Full PhishTank feed: {len(df):,} rows")

    # Keep only entries confirmed verified AND currently online —
    # this matches the "recent, confirmed phishing URLs" framing in
    # Methodology Section 3.4.1.
    if "verified" in df.columns:
        df = df[df["verified"].astype(str).str.lower() == "yes"]
    if "online" in df.columns:
        df = df[df["online"].astype(str).str.lower() == "yes"]
    print(f"  Verified + online: {len(df):,} rows")

    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    df["label"] = LABEL_PHISHING
    df["source"] = "phishtank"
    df["url_col"] = df["url"]
    return df


def load_tranco_sample(n_per_tier=N_TRANCO_PER_TIER, seed=42):
    print(f"\n[3/3] Loading Tranco sample ({n_per_tier} per tier x 3 tiers)...")
    path = RAW_DIR / "tranco_top1m.csv"
    df = pd.read_csv(path, header=None, names=["rank", "domain"])
    print(f"  Full Tranco list: {len(df):,} rows")

    # Sample across rank tiers per Methodology Section 3.4.1:
    #   top 10,000 / 10,000-100,000 / 100,000-1,000,000
    tier_1 = df[df["rank"] <= 10_000]
    tier_2 = df[(df["rank"] > 10_000) & (df["rank"] <= 100_000)]
    tier_3 = df[(df["rank"] > 100_000) & (df["rank"] <= 1_000_000)]

    rng = np.random.RandomState(seed)
    sample_1 = tier_1.sample(n=min(n_per_tier, len(tier_1)), random_state=rng)
    sample_2 = tier_2.sample(n=min(n_per_tier, len(tier_2)), random_state=rng)
    sample_3 = tier_3.sample(n=min(n_per_tier, len(tier_3)), random_state=rng)

    combined = pd.concat([sample_1, sample_2, sample_3], ignore_index=True)
    print(f"  Sampled: tier1={len(sample_1)}, tier2={len(sample_2)}, tier3={len(sample_3)}")

    # Tranco gives bare domains, not full URLs - add https:// scheme
    # (most modern sites enforce HTTPS; this is a starting point that the
    # feature extractor will validate/correct during fresh-URL processing)
    combined["url_col"] = "https://" + combined["domain"].astype(str)
    combined["label"] = LABEL_LEGITIMATE
    combined["source"] = "tranco"
    return combined


def main():
    print("=" * 60)
    print("PhishLens Dataset Merge Pipeline")
    print("=" * 60)

    phiusiil = load_phiusiil()
    phishtank = load_phishtank_sample()
    tranco = load_tranco_sample()

    # Build a unified frame with just the columns needed at this stage:
    # url, label, source. Full feature extraction for phishtank/tranco
    # rows happens in the next script — this keeps merge logic simple
    # and auditable, and avoids silently losing PhiUSIIL's 55 columns
    # by trying to force-fit everything into one wide table too early.
    unified = pd.concat([
        phiusiil[["url_col", "label", "source"]].rename(columns={"url_col": "url"}),
        phishtank[["url_col", "label", "source"]].rename(columns={"url_col": "url"}),
        tranco[["url_col", "label", "source"]].rename(columns={"url_col": "url"}),
    ], ignore_index=True)

    print("\n" + "=" * 60)
    print("UNIFIED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(unified):,}")
    print(f"\nBy source:")
    print(unified["source"].value_counts())
    print(f"\nBy label (1=legitimate, 0=phishing):")
    print(unified["label"].value_counts())
    print(f"\nDuplicate URLs found: {unified['url'].duplicated().sum()}")

    # Drop exact duplicate URLs, keeping the first occurrence
    before = len(unified)
    unified = unified.drop_duplicates(subset="url", keep="first").reset_index(drop=True)
    print(f"Removed {before - len(unified)} duplicate rows")

    out_path = PROCESSED_DIR / "unified_urls.csv"
    unified.to_csv(out_path, index=False)
    print(f"\nSaved unified URL list (pre-feature-extraction) to:\n  {out_path}")

    # Also save the full PhiUSIIL feature table separately, since it
    # already has all 55 columns extracted — no need to redo this work.
    phiusiil_features_path = PROCESSED_DIR / "phiusiil_features.csv"
    phiusiil.to_csv(phiusiil_features_path, index=False)
    print(f"Saved PhiUSIIL full feature table to:\n  {phiusiil_features_path}")

    print("\nNext step: run 04_feature_extraction.py to extract the full")
    print("five-category feature set for the PhishTank and Tranco rows in")
    print("unified_urls.csv (PhiUSIIL rows already have features).")


if __name__ == "__main__":
    main()
