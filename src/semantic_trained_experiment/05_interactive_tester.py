"""
PhishLens - Phase 1b: Interactive Trained-Classifier Tester
==============================================================
Type any text and see the TRAINED classifier's phishing probability,
using the MiniLM embedding + Logistic Regression classifier trained
in 04_train_classifier.py.

This is the trained-classifier counterpart to
src/semantic/10b_interactive_tester.py (zero-shot) - use both side
by side to compare behaviour on the same input.

Usage:
    python src/semantic_trained_experiment/05_interactive_tester.py
Type 'quit' to exit.
"""

import re
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS   = BASE_DIR / "models_experiment"

MODEL_NAME = "all-MiniLM-L6-v2"

# ── Obfuscation detector (same deterministic logic as the zero-shot track,
# src/semantic/10_semantic_phase.py) - applied here too for a fair
# best-effort comparison between the two Phase 1 approaches. This layer
# is independent of any trained model, so it works identically regardless
# of which classifier sits underneath it.
OBFUSCATION_WATCHWORDS = ["account", "verify", "password", "login", "security",
                          "bank", "confirm", "update", "suspend", "billing",
                          "payment", "card", "invoice", "refund", "transfer"]
OBFUSCATION_SUBS = {
    'o': '0', 'i': '1!', 'l': '1', 'e': '3',
    'a': '4@', 's': '5$', 't': '7', 'b': '8',
}
OBFUSCATION_BOOST = 0.35


def _build_obfuscation_pattern(word: str) -> re.Pattern:
    parts = [f"[{re.escape(ch + OBFUSCATION_SUBS.get(ch, ''))}]" for ch in word]
    return re.compile(r"[\s\-_.]*".join(parts), re.IGNORECASE)


_OBFUSCATION_PATTERNS = {w: _build_obfuscation_pattern(w)
                         for w in OBFUSCATION_WATCHWORDS}


def detect_obfuscation(text: str):
    text_lower = text.lower()
    findings = []
    for word, pattern in _OBFUSCATION_PATTERNS.items():
        for m in pattern.finditer(text_lower):
            matched = m.group()
            has_sep = bool(re.search(r"[\s\-_.]", matched))
            has_sub = any(c.isdigit() or c in "!@$" for c in matched)
            if has_sep or has_sub:
                kind = "+".join(k for k, v in
                                [("char-substitution", has_sub),
                                 ("letter-spacing", has_sep)] if v)
                findings.append((word, matched, kind))
    return findings


def verdict(risk: float) -> str:
    if risk >= 0.70:
        return "RED   (high phishing risk)"
    elif risk >= 0.30:
        return "AMBER (suspicious)"
    else:
        return "GREEN (looks fine)"


def main():
    print("=" * 62)
    print("PhishLens - Interactive Trained-Classifier Tester")
    print("=" * 62)

    clf_path = MODELS / "classifier.pkl"
    if not clf_path.exists():
        print(f"\nERROR: {clf_path} not found.")
        print("Run 04_train_classifier.py first.")
        return

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    print(f"\nLoading MiniLM ({MODEL_NAME}) ...")
    model = SentenceTransformer(MODEL_NAME)
    print("Loaded. Classifier ready.\n")
    print("Type any sentence or page text to test. 'quit' to exit.\n")

    while True:
        text = input(">> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not text:
            continue

        vec = model.encode([text], normalize_embeddings=True)
        proba = clf.predict_proba(vec)[0]
        base_p_phish = float(proba[1])

        obf = detect_obfuscation(text)
        p_phish = min(1.0, base_p_phish + OBFUSCATION_BOOST) if obf else base_p_phish
        pred = int(p_phish >= 0.5)

        print(f"   Risk score : {p_phish:.2f}   ->  {verdict(p_phish)}")
        if obf:
            print(f"   Base risk  : {base_p_phish:.2f}  (+{OBFUSCATION_BOOST:.2f} obfuscation boost)")
        print(f"   Prediction : {'PHISHING' if pred == 1 else 'LEGIT'}")
        print(f"   P(legit)   : {1-p_phish:.3f}   P(phishing): {p_phish:.3f}")
        if obf:
            for word, matched, kind in obf:
                print(f"   OBFUSCATION: '{matched}' looks like disguised '{word}' ({kind})")
        print()


if __name__ == "__main__":
    main()
