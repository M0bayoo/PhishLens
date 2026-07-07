"""
PhishLens - Stratified Sample for Rendered-Page Extraction
==============================================================
Selects a stratified subset (~1000 URLs) from the full fresh dataset,
preserving the same phishing/legit and PhishTank/Tranco proportions,
for safe VM-based rendered-page/DOM feature extraction.

WHY A SUBSET (documented decision, see Box 1 / Chapter 3 revision):
  Rendering live, unverified pages executes their JavaScript - a
  materially higher risk than the passive network requests used for
  URL/DNS/TLS features. Extracting this category on a representative
  subset rather than the full 10,823 rows keeps the VM extraction
  fast and safe, extending the two-tier feature-availability principle
  (Section 3.4.2) to a three-tier structure: rows outside this subset
  get median-imputed rendered-page values (Section 3.4.3), consistent
  with how PhiUSIIL rows already lack TLS/domain-age data.

  NOTE: a supplementary SHAP sub-analysis on ONLY this subset (where
  rendered-page data is real, not imputed) is required later to show
  true predictive value decoupled from the imputation dilution effect
  on the full-dataset analysis. See Box 1 note, 7 July 2026.

Usage:
    python src/vm_extraction/13_stratified_sample.py

Outputs:
    data/processed/render_sample_urls.csv   - the ~1000 URLs to render
"""

import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
PROCESSED  = BASE_DIR / "data" / "processed"
INPUT_FILE = PROCESSED / "fresh_features.csv"
OUTPUT_FILE = PROCESSED / "render_sample_urls.csv"

TARGET_SAMPLE_SIZE = 1000
RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("PhishLens - Stratified Sample for Rendered-Page Extraction")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nFull dataset: {len(df):,} rows")
    print("\nFull proportions (source x label):")
    print(df.groupby(["source", "label"]).size())

    frac = TARGET_SAMPLE_SIZE / len(df)
    sample = df.groupby(["source", "label"], group_keys=False).sample(
        frac=frac, random_state=RANDOM_STATE
    )

    print(f"\nSample size: {len(sample):,}")
    print("Sample proportions (source x label):")
    print(sample.groupby(["source", "label"]).size())

    full_prop = df.groupby(["source", "label"]).size() / len(df)
    samp_prop = sample.groupby(["source", "label"]).size() / len(sample)
    max_diff = (full_prop - samp_prop).abs().max()
    print(f"\nMax proportion difference: {max_diff:.4f} "
          f"({'OK - well matched' if max_diff < 0.02 else 'CHECK - drifted'})")

    # Keep only what the VM extraction script needs
    out = sample[["url", "label", "source"]].reset_index(drop=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("Sample ready. This file, NOT fresh_features.csv, is what")
    print("14_rendered_page_extraction.py should read on the VM.")
    print("=" * 60)


if __name__ == "__main__":
    main()
