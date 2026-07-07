"""
PhishLens - Diagnostic: Word-Frequency Skew Check
====================================================
Checks whether certain words (e.g. "bank") appear disproportionately
in phishing vs legitimate rows of the training data - which would
explain a classifier learning a "shortcut" association rather than
genuine phishing-language understanding.

Usage:
    python src/semantic_trained_experiment/06_check_word_skew.py
"""

import re
import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "website_texts.csv"

# Words to check for label-conditional skew
WATCH_WORDS = ["bank", "account", "payment", "verify", "password", "login",
               "security", "card", "confirm", "billing"]


def main():
    print("=" * 62)
    print("Word-Frequency Skew Diagnostic")
    print("=" * 62)

    df = pd.read_csv(INPUT_FILE)
    df["visible_text"] = df["visible_text"].astype(str).str.lower()

    n_phish = (df["label"] == 1).sum()
    n_legit = (df["label"] == 0).sum()
    print(f"\nTotal rows: {len(df):,}  (phishing={n_phish:,}, legit={n_legit:,})")

    print(f"\n{'Word':<12}{'In Phish %':>12}{'In Legit %':>12}{'Skew ratio':>12}")
    print("-" * 50)

    results = []
    for word in WATCH_WORDS:
        pattern = re.compile(r"\b" + re.escape(word) + r"\b")
        phish_hits = df[df["label"] == 1]["visible_text"].str.contains(pattern).sum()
        legit_hits = df[df["label"] == 0]["visible_text"].str.contains(pattern).sum()

        phish_pct = phish_hits / n_phish * 100
        legit_pct = legit_hits / n_legit * 100
        ratio = (phish_pct + 0.01) / (legit_pct + 0.01)  # avoid div by zero

        print(f"{word:<12}{phish_pct:>11.1f}%{legit_pct:>11.1f}%{ratio:>12.2f}")
        results.append((word, phish_pct, legit_pct, ratio))

    print("\n" + "=" * 62)
    print("Reading this table:")
    print("  Skew ratio > 2  -> word appears much more in PHISHING rows")
    print("                     -> classifier may over-associate this word")
    print("                        with phishing regardless of context")
    print("  Skew ratio ~ 1  -> word appears similarly in both classes")
    print("                     -> classifier's behaviour on this word is")
    print("                        NOT explained by simple frequency skew")
    print("=" * 62)

    # Flag the strongest skews
    sorted_results = sorted(results, key=lambda x: -x[3])
    print("\nStrongest phishing-skewed words (highest ratio):")
    for word, p, l, r in sorted_results[:5]:
        print(f"  {word:<12} ratio={r:.2f}  (phish={p:.1f}%, legit={l:.1f}%)")


if __name__ == "__main__":
    main()
