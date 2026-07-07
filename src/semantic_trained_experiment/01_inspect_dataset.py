"""
PhishLens - Phase 1b: Dataset Download & Inspection (run this FIRST)
=====================================================================
Downloads the phishing text/HTML dataset and prints its structure so
we build the training pipeline against the REAL data, not assumptions.
Takes under a minute. Run this before 12_train_semantic_classifier.py.

Usage:
    python src/semantic/12_inspect_dataset.py
"""

import json
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID  = "ealvaradob/phishing-dataset"
FILENAME = "texts.json"


def main():
    print("=" * 64)
    print("PhishLens - Dataset Download & Inspection")
    print("=" * 64)

    print(f"\nDownloading {FILENAME} from {REPO_ID} ...")
    print("(~52 MB, should take under a minute)")
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        print(f"  Downloaded to: {path}")
    except Exception as e:
        print(f"\nERROR downloading: {e}")
        print("If this fails, try: pip install --upgrade huggingface_hub")
        return

    print("\nLoading and inspecting structure ...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nTop-level type: {type(data)}")
    if isinstance(data, list):
        print(f"Number of records: {len(data):,}")
        print(f"\nFirst record keys: {list(data[0].keys()) if data else 'EMPTY'}")
        print(f"\n--- Sample record [0] ---")
        for k, v in data[0].items():
            preview = str(v)[:150]
            print(f"  {k}: {preview}")
        print(f"\n--- Sample record [1] ---")
        for k, v in data[1].items():
            preview = str(v)[:150]
            print(f"  {k}: {preview}")

        # Try to find label distribution
        possible_label_keys = ["label", "labels", "class", "type", "source"]
        for key in possible_label_keys:
            if key in data[0]:
                values = [d.get(key) for d in data[:5000]]  # sample first 5000
                from collections import Counter
                counts = Counter(values)
                print(f"\nDistribution of '{key}' (first 5000 records): {dict(counts)}")

    elif isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")
        for k in list(data.keys())[:3]:
            print(f"\n--- data['{k}'] preview ---")
            v = data[k]
            print(f"  type: {type(v)}")
            if isinstance(v, list) and v:
                print(f"  length: {len(v)}")
                print(f"  first item: {str(v[0])[:200]}")

    print("\n" + "=" * 64)
    print("Inspection complete. Paste this full output back to Claude")
    print("so the training pipeline can be built to match the real structure.")
    print("=" * 64)


if __name__ == "__main__":
    main()
