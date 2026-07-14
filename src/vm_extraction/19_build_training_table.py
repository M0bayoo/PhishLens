"""
PhishLens - Build Master Training Table
=========================================================
Left-joins the rendered-page features (1,000-URL VM subset) onto the
full URL/DNS/TLS feature table (11,269 rows), producing ONE master
table that supports both planned models:

  Model A (URL/DNS/TLS only)  - trains on ALL rows
  Model B (full feature set)  - trains on the subset where
                                in_render_sample == 1 AND render_error == 0

Design decisions (recorded for Chapter 3):
  - Rows outside the render sample keep NaN in rendered columns and
    in_render_sample = 0. They are NOT imputed - Model A never uses
    rendered columns, Model B never uses these rows.
  - Rows where rendering failed (render_error == 1) keep NaN rendered
    features. They stay usable for Model A; excluded from Model B.
  - label/source come from fresh_features (authoritative); duplicates
    from the rendered file are dropped after a consistency check.

Usage:
    python3 19_build_training_table.py

Output:
    data/processed/training_table.csv
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
FRESH     = PROCESSED / "fresh_features.csv"
RENDERED  = PROCESSED / "rendered_page_features.csv"
OUTPUT    = PROCESSED / "training_table.csv"

RENDER_FEATURES = [
    "num_forms", "has_password_field", "num_password_fields",
    "hidden_element_count", "iframe_count", "external_resource_ratio",
    "favicon_domain_mismatch",
]


def main():
    print("=" * 66)
    print("PhishLens - Build Master Training Table")
    print("=" * 66)

    fresh = pd.read_csv(FRESH)
    rendered = pd.read_csv(RENDERED)

    print(f"\nfresh_features.csv          : {fresh.shape}")
    print(f"rendered_page_features.csv  : {rendered.shape}")

    # ── Safety rail 1: no duplicate URLs in either table ──
    for name, df in [("fresh", fresh), ("rendered", rendered)]:
        dups = df["url"].duplicated().sum()
        if dups:
            print(f"WARNING: {dups} duplicate URLs in {name} - keeping first")
    fresh = fresh.drop_duplicates("url", keep="first")
    rendered = rendered.drop_duplicates("url", keep="first")

    # ── Safety rail 2: every rendered URL must now exist in fresh ──
    missing = ~rendered["url"].isin(set(fresh["url"]))
    if missing.sum() > 0:
        print(f"\nERROR: {missing.sum()} rendered URLs still missing from "
              f"fresh_features. Gap-fill incomplete. NOT writing output.")
        print(rendered[missing][["url", "source"]].head(10).to_string())
        sys.exit(1)
    print("Overlap check: all rendered URLs present in fresh_features ✓")

    # ── Safety rail 3: label consistency between the two files ──
    check = rendered[["url", "label"]].merge(
        fresh[["url", "label"]], on="url", suffixes=("_r", "_f"))
    conflicts = (check["label_r"] != check["label_f"]).sum()
    if conflicts:
        print(f"\nERROR: {conflicts} URLs have conflicting labels between "
              f"files. NOT writing output. Paste this to Claude.")
        bad = check[check["label_r"] != check["label_f"]]
        print(bad.head(10).to_string())
        sys.exit(1)
    print("Label consistency check: no conflicts ✓")

    # ── Merge ──
    keep = ["url"] + RENDER_FEATURES + ["render_error"]
    rend_slim = rendered[keep]

    master = fresh.merge(rend_slim, on="url", how="left")
    master["in_render_sample"] = master["url"].isin(set(rendered["url"])).astype(int)

    master.to_csv(OUTPUT, index=False)

    # ── Report ──
    n_render = int(master["in_render_sample"].sum())
    model_b = master[(master["in_render_sample"] == 1) &
                     (master["render_error"] == 0)]

    print("\n" + "=" * 66)
    print("TRAINING TABLE BUILT")
    print("=" * 66)
    print(f"Total rows              : {len(master):,}")
    print(f"Total columns           : {len(master.columns)}")
    print(f"\nModel A (URL/DNS/TLS, all rows):")
    print(f"  Rows: {len(master):,}")
    print(master["label"].value_counts().rename(
        {1: "  phishing", 0: "  legitimate"}).to_string())
    print(f"\nModel B (full features, rendered-and-alive subset):")
    print(f"  Rows: {len(model_b):,}")
    print(model_b["label"].value_counts().rename(
        {1: "  phishing", 0: "  legitimate"}).to_string())
    print(f"\nRendered-feature completeness in Model B subset:")
    null_counts = model_b[RENDER_FEATURES].isna().sum()
    print(null_counts.to_string())
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
