"""
PhishLens - Gap-Fill: URL/DNS/TLS features for fresh PhishTank URLs
=====================================================================
The 446 PhishTank URLs refreshed by script 15 were rendered by script
14 but never went through script 04, so they exist in
rendered_page_features.csv with NO url-lexical/DNS/TLS features.
A naive merge would give the model 446 phishing rows with mostly-empty
columns - it would learn "missing DNS data = phishing". This script
fills the gap.

CRITICAL DESIGN RULE: this wrapper IMPORTS extract_all() from the
original 04_feature_extraction.py and reuses it unchanged. It does NOT
reimplement any feature, so the new rows are computed identically to
the existing 10,823.

Safety rails:
  - Backs up fresh_features.csv before touching it
  - Refuses to append if the produced columns don't exactly match
  - Skips URLs already present (safe to rerun)
  - Checkpoints every 25 URLs

Requirements (VM venv):
    pip install tldextract dnspython python-whois requests

Usage:
    python3 18_gapfill_phishing_features.py
"""

import sys
import time
import shutil
import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
PROCESSED  = BASE_DIR / "data" / "processed"
FRESH      = PROCESSED / "fresh_features.csv"
RENDERED   = PROCESSED / "rendered_page_features.csv"
CHECKPOINT = PROCESSED / "gapfill_checkpoint.csv"
BACKUP     = PROCESSED / "fresh_features_BACKUP_pre_gapfill.csv"

SCRIPT_04 = BASE_DIR / "src" / "feature_extraction" / "04_feature_extraction.py"

MAX_WORKERS      = 10
CHECKPOINT_EVERY = 25


def load_extract_all():
    """Import extract_all() from the numerically-named script 04."""
    spec = importlib.util.spec_from_file_location("fe04", SCRIPT_04)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fe04"] = mod
    spec.loader.exec_module(mod)
    return mod.extract_all


def main():
    print("=" * 66)
    print("PhishLens - Gap-Fill: features for fresh PhishTank URLs")
    print("=" * 66)

    extract_all = load_extract_all()
    print(f"Imported extract_all() from {SCRIPT_04.name} - no reimplementation")

    fresh = pd.read_csv(FRESH)
    rendered = pd.read_csv(RENDERED)

    missing = rendered[~rendered["url"].isin(set(fresh["url"]))]
    missing = missing[["url", "label", "source"]].drop_duplicates("url")
    print(f"\nURLs in rendered set missing from {FRESH.name}: {len(missing)}")
    src_counts = missing["source"].value_counts().to_dict()
    print(f"By source: {src_counts}")

    done_rows = []
    done_urls = set()
    if CHECKPOINT.exists():
        cp = pd.read_csv(CHECKPOINT)
        done_rows = cp.to_dict("records")
        done_urls = set(cp["url"])
        print(f"Checkpoint found - resuming from {len(done_urls)} done")

    todo = missing[~missing["url"].isin(done_urls)].to_dict("records")
    print(f"Remaining to extract: {len(todo)}\n")

    results = list(done_rows)
    if todo:
        start = time.time()
        completed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(extract_all, r["url"], r["label"], r["source"]): r
                for r in todo
            }
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception as e:
                    r = futures[fut]
                    print(f"  ROW FAILED ENTIRELY: {r['url'][:60]} - "
                          f"{type(e).__name__}: {str(e)[:80]}")
                    continue
                results.append(row)
                completed += 1
                rate = completed / max(time.time() - start, 1)
                eta = (len(todo) - completed) / max(rate, 0.01) / 60
                print(f"  [{completed:4d}/{len(todo):4d}] "
                      f"rate={rate:.2f}/s ETA={eta:.1f}min")
                if completed % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

    new_df = pd.DataFrame(results)

    # ── Safety rail: exact column match before touching fresh_features ──
    fresh_cols = fresh.columns.tolist()
    new_cols = new_df.columns.tolist()
    if set(new_cols) != set(fresh_cols):
        only_fresh = [c for c in fresh_cols if c not in new_cols]
        only_new = [c for c in new_cols if c not in fresh_cols]
        print("\nCOLUMN MISMATCH - NOT appending. Paste this to Claude:")
        print(f"  In fresh_features but not produced: {only_fresh}")
        print(f"  Produced but not in fresh_features: {only_new}")
        new_df.to_csv(PROCESSED / "gapfill_UNMERGED.csv", index=False)
        print(f"  Raw results saved to gapfill_UNMERGED.csv for inspection")
        sys.exit(1)

    new_df = new_df[fresh_cols]  # identical column order

    # ── Backup, then append ──
    shutil.copy(FRESH, BACKUP)
    print(f"\nBackup written: {BACKUP.name}")

    combined = pd.concat([fresh, new_df], ignore_index=True)
    combined = combined.drop_duplicates("url", keep="first")
    combined.to_csv(FRESH, index=False)

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print("=" * 66)
    print("GAP-FILL COMPLETE")
    print("=" * 66)
    print(f"fresh_features.csv: {len(fresh):,} -> {len(combined):,} rows")
    print(f"Label distribution now:")
    print(combined["label"].value_counts().to_string())
    print(f"\nDNS-resolved rate of the new rows: "
          f"{new_df['dns_resolved'].mean()*100:.1f}%")
    print("\nNext: verify overlap, then merge script for training table.")


if __name__ == "__main__":
    main()
