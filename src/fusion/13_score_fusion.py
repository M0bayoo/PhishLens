"""
PhishLens - Score Fusion Module
=================================
Combines Phase 1 (semantic) + Phase 2 (structural) scores into the
final Green/Amber/Red verdict, per Chapter 3 spec:
  Green < 0.30 | Amber 0.30-0.69 | Red >= 0.70 | Red+block >= 0.90

Weighted linear fusion (structural weighted higher - it's the more
reliable signal, 99.77% CV accuracy vs zero-shot's qualitative nature),
with a single-signal override: if either score alone is very high
(>=0.85), the fused verdict escalates to at least Red, so a confident
signal from one phase can't be diluted away by a weak reading on the
other (validated against realistic scenarios - see demo below).

This is the reference implementation, later ported to JavaScript for
the Chrome extension so both share identical logic.

Usage:
    python src/fusion/13_score_fusion.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS   = BASE_DIR / "models"

W_SEMANTIC   = 0.4
W_STRUCTURAL = 0.6
OVERRIDE_THRESHOLD = 0.85   # either score alone triggers escalation
GREEN_MAX  = 0.30
AMBER_MAX  = 0.70
BLOCK_MIN  = 0.90


def fuse(semantic_risk: float, structural_risk: float) -> float:
    fused = W_SEMANTIC * semantic_risk + W_STRUCTURAL * structural_risk
    if semantic_risk >= OVERRIDE_THRESHOLD or structural_risk >= OVERRIDE_THRESHOLD:
        fused = max(fused, AMBER_MAX)  # guarantee at least Red-boundary
    return min(1.0, max(0.0, fused))


def verdict(fused_score: float) -> str:
    if fused_score >= BLOCK_MIN:
        return "RED_BLOCKED"
    elif fused_score >= AMBER_MAX:
        return "RED"
    elif fused_score >= GREEN_MAX:
        return "AMBER"
    else:
        return "GREEN"


def evaluate(semantic_risk: float, structural_risk: float) -> dict:
    f = fuse(semantic_risk, structural_risk)
    return {
        "semantic_risk": semantic_risk,
        "structural_risk": structural_risk,
        "fused_score": round(f, 4),
        "verdict": verdict(f),
    }


def main():
    print("=" * 60)
    print("PhishLens - Score Fusion Module")
    print("=" * 60)

    config = {
        "w_semantic": W_SEMANTIC,
        "w_structural": W_STRUCTURAL,
        "override_threshold": OVERRIDE_THRESHOLD,
        "green_max": GREEN_MAX,
        "amber_max": AMBER_MAX,
        "block_min": BLOCK_MIN,
        "rule": "fused = w_semantic*semantic + w_structural*structural; "
                "if either score >= override_threshold, fused = max(fused, amber_max)",
    }
    with open(MODELS / "fusion_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved -> models/fusion_config.json")

    scenarios = [
        ("Real phishing (bad structure + bad text)", 0.95, 0.98),
        ("Real legit site (good structure + good text)", 0.05, 0.02),
        ("Legit site, ambiguous text (structure saves it)", 0.65, 0.05),
        ("Fake domain, clean text (structure catches it)", 0.10, 0.92),
        ("Borderline both", 0.45, 0.50),
        ("Confident phishing text, unknown domain", 0.90, 0.40),
    ]

    print(f"\n{'Scenario':<50}{'Sem':>6}{'Struct':>7}{'Fused':>7}  Verdict")
    print("-" * 92)
    for name, sem, struct in scenarios:
        r = evaluate(sem, struct)
        print(f"{name:<50}{r['semantic_risk']:>6.2f}{r['structural_risk']:>7.2f}"
              f"{r['fused_score']:>7.2f}  {r['verdict']}")

    print("\n" + "=" * 60)
    print("Fusion module ready. Config saved for extension to load.")
    print("=" * 60)


if __name__ == "__main__":
    main()
