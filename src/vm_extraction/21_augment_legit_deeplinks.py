"""
PhishLens - Augment Legitimate Class with Deep-Link URLs
==========================================================
PROBLEM (found via feature-importance audit, 15 Jul 2026):
Tranco provides bare homepage domains (path_length = 0) while
PhishTank provides deep links. path_length + num_slashes +
num_subdomains carried ~57% of Model A's importance - the model was
partly learning "has a URL path = phishing", a dataset-construction
artifact that would misclassify legitimate deep links (e.g. webmail,
e-commerce checkout pages) in deployment.

FIX: augment the legitimate class with real deep-link legitimate URLs
sampled from PhiUSIIL (already in unified_urls.csv, label 1), and
extract their features with the ORIGINAL extract_all() from script 04
- no reimplementation, identical computation to all other rows.

Notes recorded for Chapter 3:
  - PhiUSIIL is a 2024 research corpus; some URLs will be dead, so
    liveness features for these rows are weaker. The A vs A-minus
    comparison in script 20 controls for this.
  - New rows get source = 'phiusiil_legit' so their provenance stays
    auditable and they can be ablated in evaluation.

Requirements (same as gap-fill): tldextract dnspython python-whois

Usage:
    python3 21_augment_legit_deeplinks.py            # default 2,500 URLs
    python3 21_augment_legit_deeplinks.py 4000       # custom count

After completion: rerun 19 then 20.
"""

import sys
import time
import shutil
import importlib.util
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
PROCESSED  = BASE_DIR / "data" / "processed"
UNIFIED    = PROCESSED / "unified_urls.csv"
FRESH      = PROCESSED / "fresh_features.csv"
CHECKPOINT = PROCESSED / "augment_checkpoint.csv"
BACKUP     = PROCESSED / "fresh_features_BACKUP_pre_augment.csv"

SCRIPT_04 = BASE_DIR / "src" / "feature_extraction" / "04_feature_extraction.py"

MAX_WORKERS      = 15
CHECKPOINT_EVERY = 50
SEED             = 42
DEFAULT_N        = 2500
SOURCE_TAG       = "phiusiil_legit"


def load_extract_all():
    spec = importlib.util.spec_from_file_location("fe04", SCRIPT_04)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fe04"] = mod
    spec.loader.exec_module(mod)
    return mod.extract_all


def has_deep_path(url: str) -> bool:
    try:
        p = urlparse(url)
        return len(p.path.strip("/")) > 3 or bool(p.query)
    except Exception:
        return False


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N

    print("=" * 66)
    print("PhishLens - Augment Legitimate Class with Deep-Link URLs")
    print("=" * 66)

    extract_all = load_extract_all()
    print(f"Imported extract_all() from {SCRIPT_04.name}")

    unified = pd.read_csv(UNIFIED)
    fresh = pd.read_csv(FRESH)

    # PhiUSIIL legitimate URLs with real paths, not already in fresh_features
    cand = unified[(unified["source"] == "phiusiil") & (unified["label"] == 1)]
    cand = cand[~cand["url"].isin(set(fresh["url"]))]
    cand = cand[cand["url"].map(has_deep_path)]
    print(f"PhiUSIIL legitimate deep-link candidates: {len(cand):,}")

    sample = cand.sample(n=min(n_target, len(cand)), random_state=SEED)
    print(f"Sampled for extraction: {len(sample):,}")
    pl = sample["url"].map(lambda u: len(urlparse(u).path))
    print(f"Sampled path length: mean={pl.mean():.1f}, median={pl.median():.0f}")

    done_rows, done_urls = [], set()
    if CHECKPOINT.exists():
        cp = pd.read_csv(CHECKPOINT)
        done_rows = cp.to_dict("records")
        done_urls = set(cp["url"])
        print(f"Checkpoint found - resuming from {len(done_urls)} done")

    todo = sample[~sample["url"].isin(done_urls)]["url"].tolist()
    print(f"Remaining to extract: {len(todo)}\n")

    results = list(done_rows)
    if todo:
        start = time.time()
        completed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(extract_all, u, 1, SOURCE_TAG): u
                       for u in todo}
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  ROW FAILED: {futures[fut][:60]} - "
                          f"{type(e).__name__}: {str(e)[:80]}")
                    continue
                results.append(row)
                completed += 1
                rate = completed / max(time.time() - start, 1)
                eta = (len(todo) - completed) / max(rate, 0.01) / 60
                if completed % 10 == 0 or completed == len(todo):
                    print(f"  [{completed:4d}/{len(todo):4d}] "
                          f"rate={rate:.2f}/s ETA={eta:.1f}min")
                if completed % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

    new_df = pd.DataFrame(results)

    # Safety rail: exact column match
    if set(new_df.columns) != set(fresh.columns):
        only_fresh = [c for c in fresh.columns if c not in new_df.columns]
        only_new = [c for c in new_df.columns if c not in fresh.columns]
        print("\nCOLUMN MISMATCH - NOT appending. Paste this to Claude:")
        print(f"  In fresh but not produced: {only_fresh}")
        print(f"  Produced but not in fresh: {only_new}")
        new_df.to_csv(PROCESSED / "augment_UNMERGED.csv", index=False)
        sys.exit(1)

    new_df = new_df[fresh.columns.tolist()]

    shutil.copy(FRESH, BACKUP)
    print(f"\nBackup written: {BACKUP.name}")

    combined = pd.concat([fresh, new_df], ignore_index=True)
    combined = combined.drop_duplicates("url", keep="first")
    combined.to_csv(FRESH, index=False)

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print("=" * 66)
    print("AUGMENTATION COMPLETE")
    print("=" * 66)
    print(f"fresh_features.csv: {len(fresh):,} -> {len(combined):,} rows")
    print("Label distribution (0=phishing, 1=legitimate):")
    print(combined["label"].value_counts().to_string())
    print("\nPath length by label AFTER augmentation:")
    print(combined.groupby("label")["path_length"].agg(
        ["mean", "median"]).round(1).to_string())
    print(f"\nDNS-resolved rate of new rows: "
          f"{new_df['dns_resolved'].mean()*100:.1f}%")
    print("\nNext: rerun 19_build_training_table.py, then 20_train_models.py")


if __name__ == "__main__":
    main()
