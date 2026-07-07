"""
PhishLens - Phase 1: Interactive Semantic Tester v3
=====================================================
Type any text and see its semantic phishing risk live: contrastive
similarity score PLUS obfuscation/character-substitution detection.

Run 10_semantic_phase.py first (it saves the artifacts this loads).

Usage:
    python src/semantic/10b_interactive_tester.py
Type 'quit' to exit.
"""

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS   = BASE_DIR / "models"


def build_obfuscation_pattern(word, subs):
    parts = [f"[{re.escape(ch + subs.get(ch, ''))}]" for ch in word]
    return re.compile(r"[\s\-_.]*".join(parts), re.IGNORECASE)


def detect_obfuscation(text, patterns):
    text_lower = text.lower()
    findings = []
    for word, pattern in patterns.items():
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


def main():
    print("=" * 64)
    print("PhishLens - Interactive Semantic Tester v3 (+ obfuscation check)")
    print("=" * 64)

    required = [
        MODELS / "phishing_phrases.json",
        MODELS / "phishing_phrase_vectors.npy",
        MODELS / "legit_anchor_vectors.npy",
        MODELS / "semantic_scoring_config.json",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        print(f"\nERROR: missing artifacts: {missing}")
        print("Run 10_semantic_phase.py first to generate them.")
        return

    with open(MODELS / "phishing_phrases.json") as f:
        pdata = json.load(f)
    phrases    = pdata["phrases"]
    categories = pdata["categories"]
    phish_vecs = np.load(MODELS / "phishing_phrase_vectors.npy")
    legit_vecs = np.load(MODELS / "legit_anchor_vectors.npy")
    with open(MODELS / "semantic_scoring_config.json") as f:
        cfg = json.load(f)

    TOP_K        = cfg["top_k"]
    MARGIN_FLOOR = cfg["margin_floor"]
    MARGIN_CEIL  = cfg["margin_ceil"]
    OBF_WORDS    = cfg["obfuscation_watchwords"]
    OBF_SUBS     = cfg["obfuscation_substitutions"]
    OBF_BOOST    = cfg["obfuscation_boost"]
    obf_patterns = {w: build_obfuscation_pattern(w, OBF_SUBS) for w in OBF_WORDS}

    def topk_mean(sims, k=TOP_K):
        k = min(k, len(sims))
        return float(np.sort(sims)[-k:].mean())

    def calibrate(margin):
        scaled = (margin - MARGIN_FLOOR) / (MARGIN_CEIL - MARGIN_FLOOR)
        return float(min(1.0, max(0.0, scaled)))

    def verdict(risk):
        if risk >= 0.70:
            return "RED   (high phishing risk)"
        elif risk >= 0.30:
            return "AMBER (suspicious)"
        else:
            return "GREEN (looks fine)"

    print(f"\nLoading MiniLM ({cfg['model']}) ...")
    model = SentenceTransformer(cfg["model"])
    print(f"Loaded. {len(phrases)} phishing prototypes / "
          f"{legit_vecs.shape[0]} legit anchors / "
          f"{len(OBF_WORDS)} obfuscation watchwords ready.\n")
    print("Type any sentence to test. 'quit' to exit.\n")

    while True:
        text = input(">> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not text:
            continue

        vec   = model.encode([text], normalize_embeddings=True)[0]
        ps    = phish_vecs @ vec
        ls    = legit_vecs @ vec
        p_top = topk_mean(ps)
        l_top = topk_mean(ls)
        margin = p_top - l_top
        base_risk = calibrate(margin)
        best   = int(ps.argmax())

        obf = detect_obfuscation(text, obf_patterns)
        risk = min(1.0, base_risk + OBF_BOOST) if obf else base_risk

        print(f"   Risk score    : {risk:.2f}   ->  {verdict(risk)}")
        if obf:
            print(f"   Base risk     : {base_risk:.2f}  (+{OBF_BOOST:.2f} obfuscation boost)")
        print(f"   Phishing sim  : {p_top:.3f}   (top-{TOP_K} mean)")
        print(f"   Legit sim     : {l_top:.3f}   (top-{TOP_K} mean)")
        print(f"   Margin        : {margin:+.3f}")
        print(f"   Tactic matched: {categories[best]}")
        print(f"   Closest phrase: \"{phrases[best]}\"")
        if obf:
            for word, matched, kind in obf:
                print(f"   OBFUSCATION   : '{matched}' looks like disguised '{word}' ({kind})")
        print()


if __name__ == "__main__":
    main()
