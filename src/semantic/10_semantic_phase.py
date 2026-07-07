"""
PhishLens - Phase 1: Semantic Analysis (Zero-Shot Contrastive Similarity) v2
=============================================================================
Reads the visible text of a page and produces a semantic risk score in [0,1].

IMPROVEMENTS OVER v1:
  1. 80 phishing prototype phrases across 12 tactic categories
     (v1 had 20, heavily urgency-biased).
  2. CONTRASTIVE scoring: 35 legitimate anchor phrases added. Score is
     based on the MARGIN (phishing similarity minus legitimate similarity),
     not raw phishing similarity alone. Benign transactional language
     ("send your details", "make a payment") sits close to BOTH sets,
     so its margin ~0 -> low risk. Genuine phishing sits close to the
     phishing set only -> high margin -> high risk.
     This directly fixes the false-Amber behaviour found in manual testing.
  3. Top-k mean similarity (k=3) instead of single max: one lucky word
     overlap can no longer dominate the score; also smooths the
     punctuation sensitivity observed in v1.
  4. Category-aware explanation: reports the phishing TACTIC matched
     (e.g. "credential harvesting"), feeding the explanation layer.
  5. All scoring parameters saved to semantic_scoring_config.json so the
     Chrome extension replicates scoring exactly.

Requirements:
    pip install sentence-transformers

Usage:
    python src/semantic/10_semantic_phase.py

Outputs:
    models/phishing_phrases.json          - phrases with categories
    models/phishing_phrase_vectors.npy    - phishing prototype embeddings
    models/legit_anchor_phrases.json      - legitimate anchor phrases
    models/legit_anchor_vectors.npy       - legitimate anchor embeddings
    models/semantic_scoring_config.json   - scoring parameters (contract)
    reports/semantic_phase_demo.txt       - demo scores
"""

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS   = BASE_DIR / "models"
REPORTS  = BASE_DIR / "reports"
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

# ── Phishing prototypes: 80 phrases across 12 tactic categories ──────────────
PHISHING_PHRASES = {
    "urgency / account threat": [
        "your account has been suspended",
        "your account will be permanently closed",
        "immediate action is required to avoid suspension",
        "your account will be deactivated within 24 hours",
        "failure to respond will result in account termination",
        "your access has been temporarily restricted",
        "act now before your account is locked",
    ],
    "credential harvesting": [
        "verify your account immediately",
        "confirm your password to continue",
        "re-enter your login credentials to verify your identity",
        "update your security information now",
        "confirm your identity to restore access",
        "sign in to verify recent changes to your account",
        "your session has expired please log in again to continue",
        "enter your username and password to unlock your account",
    ],
    "payment / billing": [
        "your payment could not be processed",
        "update your billing information to avoid interruption",
        "your card has been declined please update payment details",
        "there was a problem with your recent payment",
        "your invoice is overdue pay now to avoid penalties",
        "confirm your card details to complete the transaction",
        "your subscription payment failed update your card",
    ],
    "security alert impersonation": [
        "unusual sign-in activity detected on your account",
        "we detected a suspicious login attempt",
        "security alert your account may be compromised",
        "someone tried to access your account",
        "a new device has signed in to your account verify it was you",
        "we noticed unusual activity please secure your account",
    ],
    "fake delivery / parcel": [
        "your package could not be delivered",
        "a delivery attempt was missed schedule redelivery now",
        "pay a small customs fee to release your parcel",
        "your shipment is on hold pending address confirmation",
        "track your undelivered package by confirming your details",
    ],
    "tech support scam": [
        "your computer has been infected with a virus",
        "call our support line immediately to remove malware",
        "your device has been blocked for security reasons",
        "critical windows alert do not shut down your computer",
        "your antivirus subscription has expired renew immediately",
    ],
    "prize / lottery lure": [
        "you have won a prize claim it now",
        "congratulations you are our lucky winner",
        "you have been selected for an exclusive reward",
        "claim your gift card before it expires",
        "you are eligible for a cash prize confirm to receive it",
    ],
    "tax / government impersonation": [
        "you have a pending tax refund claim it now",
        "final notice regarding your unpaid taxes",
        "your national insurance number has been suspended",
        "legal action will be taken unless you respond immediately",
        "you are entitled to a government rebate confirm your details",
    ],
    "hr / payroll scam": [
        "your salary payment requires verification",
        "update your direct deposit information immediately",
        "your payroll details need to be confirmed by end of day",
        "action required review the attached employment document",
    ],
    "crypto / investment scam": [
        "double your bitcoin with our guaranteed investment plan",
        "limited time crypto giveaway send to receive double back",
        "your wallet has been flagged verify to avoid suspension",
        "claim your free tokens by connecting your wallet",
    ],
    "document / file share lure": [
        "a secure document has been shared with you sign in to view",
        "you have received an encrypted message click to open",
        "review and sign the attached document immediately",
        "a file has been shared with you via secure link",
    ],
    "generic pressure / fear": [
        "this is your final warning",
        "do not ignore this message",
        "respond within 24 hours or face consequences",
        "your immediate attention is required",
        "failure to comply will result in permanent closure",
    ],
    "coercion / extortion / threat": [
        "pay now or your access will be blocked",
        "if you do not pay you will lose access immediately",
        "failure to pay will result in your account being blocked",
        "we will restrict your access unless you pay immediately",
        "pay the fee or your data will be deleted",
        "you must pay now or face permanent restriction",
        "send payment immediately or this will escalate",
        "comply now or further action will be taken against you",
        "your files will remain locked until payment is received",
        "ignoring this demand will result in serious consequences",
    ],
}

# ── Legitimate anchors: normal web language across common page types ─────────
LEGIT_ANCHORS = [
    # e-commerce
    "browse our product catalogue with free delivery on orders over fifty pounds",
    "add items to your basket and check out when you are ready",
    "read customer reviews before you buy",
    "view your order history and track current orders in your account",
    "sign up for our newsletter to receive offers and updates",
    # content / news / blog
    "read the latest news and analysis from our editorial team",
    "explore our blog for recipes tips and guides",
    "subscribe to our channel for weekly videos",
    "share this article with your friends",
    # documentation / software
    "this guide explains how to install and configure the software",
    "see the api reference for detailed usage examples",
    "download the latest release from the official website",
    "report bugs and request features on our issue tracker",
    # support / contact
    "contact our support team monday to friday",
    "we aim to respond to all enquiries within two working days",
    "visit our help centre for frequently asked questions",
    "chat with our customer service team for assistance",
    # legitimate account / banking language
    "log in to view your account balance and recent transactions",
    "you can change your password at any time in account settings",
    "manage your payment methods in your account dashboard",
    "set up a standing order or direct debit in online banking",
    "your statement for this month is now available to view",
    # education / community
    "enrol in our online course and learn at your own pace",
    "join the discussion in our community forum",
    "register for the upcoming webinar",
    # travel / booking
    "search flights hotels and car hire in one place",
    "manage your booking or check in online",
    # general corporate
    "learn more about our company and our mission",
    "we are hiring view open positions on our careers page",
    "read our privacy policy and terms of service",
    "cookie preferences can be updated at any time",
    # transactional but benign
    "your order has been dispatched and is on its way",
    "thank you for your purchase a receipt has been emailed to you",
    "your booking is confirmed we look forward to seeing you",
    "your appointment reminder for next tuesday at ten am",
    # legitimate verification / security flows (benign counterpart to
    # "verify your account" phishing phrases - added after live testing
    # showed the bare phrase alone was mistaken for phishing)
    "verify your email address to complete your registration",
    "verify your phone number to enable two factor authentication",
    "please verify your account to unlock additional features",
    "we sent a verification code to your registered email",
    "confirm your email to activate your new account",
    "verify your identity to complete account setup",
    "click the link in your email to verify your new account",
    "enter the code we sent you to confirm your account",
]

# ── Obfuscation / character-substitution detector ────────────────────────────
# Detects deliberate disguising of security-sensitive words (a known phishing
# evasion tactic, e.g. "verify y0ur acc0unt" or "s e c u r i t y alert") that
# would otherwise slip past exact keyword filters while remaining human
# readable. Discovered as a real gap via manual adversarial testing
# (1 July 2026) - MiniLM's subword tokenization is somewhat typo-tolerant,
# meaning it treats obfuscated text as a near-miss rather than a red flag.
# This is a lightweight, deterministic complement to the semantic score,
# not a replacement for it.
OBFUSCATION_WATCHWORDS = ["account", "verify", "password", "login", "security",
                          "bank", "confirm", "update", "suspend", "billing"]
OBFUSCATION_SUBS = {
    'o': '0', 'i': '1!', 'l': '1', 'e': '3',
    'a': '4@', 's': '5$', 't': '7', 'b': '8',
}
OBFUSCATION_BOOST = 0.35   # added to semantic risk score if obfuscation found


def _build_obfuscation_pattern(word: str) -> re.Pattern:
    parts = [f"[{re.escape(ch + OBFUSCATION_SUBS.get(ch, ''))}]" for ch in word]
    return re.compile(r"[\s\-_.]*".join(parts), re.IGNORECASE)


_OBFUSCATION_PATTERNS = {w: _build_obfuscation_pattern(w)
                         for w in OBFUSCATION_WATCHWORDS}


def detect_obfuscation(text: str):
    """Return list of (word, matched_text, kind) for disguised sensitive words."""
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


# ── Scoring parameters ────────────────────────────────────────────────────────
TOP_K        = 3      # top-k mean similarity instead of single max
MARGIN_FLOOR = 0.05   # margin at/below this -> risk 0
MARGIN_CEIL  = 0.32   # margin at/above this -> risk 1
# CALIBRATION NOTE (updated after live adversarial testing, 1 July 2026):
# History: original ceiling 0.40 (untested estimate) -> lowered to 0.28
# after finding canonical phishing text ("verify your account immediately")
# under-scored into Amber. But 0.28 then caused a false positive: the
# BENIGN phrase "verify your account" alone scored Red, because the
# phishing prototype list had generic verify-phrases with no legitimate
# counterpart to contrast against (a one-sided phrase bank, not a
# calibration problem). Fixed properly by adding legitimate verification/
# security anchors (email verification, 2FA, account setup) so the
# CONTRAST does the separating. Ceiling restored to a middle value (0.32)
# since the anchor fix - not the ceiling - is the correct mechanism for
# distinguishing "verify your account" (benign) from "verify your account
# immediately, or face suspension" (phishing, urgency-qualified).
# Lesson: ceiling tuning is a blunt global instrument that trades recall
# against false positives; prototype/anchor balance is the precise fix.
# margin = topk_mean(phish sims) - topk_mean(legit sims)
# Benign transactional text is similar to BOTH sets -> margin ~0 -> low risk.


def topk_mean(sims: np.ndarray, k: int = TOP_K) -> float:
    k = min(k, len(sims))
    return float(np.sort(sims)[-k:].mean())


def calibrate(margin: float) -> float:
    scaled = (margin - MARGIN_FLOOR) / (MARGIN_CEIL - MARGIN_FLOOR)
    return float(min(1.0, max(0.0, scaled)))


def main():
    print("=" * 68)
    print("PhishLens - Phase 1: Semantic Analysis v2 (contrastive similarity)")
    print("=" * 68)

    # Flatten phishing phrases, remembering category per phrase
    phish_texts, phish_cats = [], []
    for cat, plist in PHISHING_PHRASES.items():
        for p in plist:
            phish_texts.append(p)
            phish_cats.append(cat)

    print(f"\nPhishing prototypes : {len(phish_texts)} phrases "
          f"across {len(PHISHING_PHRASES)} tactic categories")
    print(f"Legitimate anchors  : {len(LEGIT_ANCHORS)} phrases")

    # ── Load MiniLM ───────────────────────────────────────────────────────
    print(f"\nLoading MiniLM ({MODEL_NAME}) ...")
    model = SentenceTransformer(MODEL_NAME)
    print("  Loaded.")

    # ── Embed both prototype sets ─────────────────────────────────────────
    print("\nEmbedding prototypes ...")
    phish_vecs = model.encode(phish_texts, normalize_embeddings=True)
    legit_vecs = model.encode(LEGIT_ANCHORS, normalize_embeddings=True)
    print(f"  Phishing matrix : {phish_vecs.shape}")
    print(f"  Legit matrix    : {legit_vecs.shape}")

    # ── Save artifacts for the extension ─────────────────────────────────
    with open(MODELS / "phishing_phrases.json", "w") as f:
        json.dump({"phrases": phish_texts, "categories": phish_cats}, f, indent=2)
    np.save(MODELS / "phishing_phrase_vectors.npy", phish_vecs)
    with open(MODELS / "legit_anchor_phrases.json", "w") as f:
        json.dump(LEGIT_ANCHORS, f, indent=2)
    np.save(MODELS / "legit_anchor_vectors.npy", legit_vecs)
    with open(MODELS / "semantic_scoring_config.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "top_k": TOP_K,
            "margin_floor": MARGIN_FLOOR,
            "margin_ceil": MARGIN_CEIL,
            "rule": "risk = clip((topk_mean(phish_sims) - topk_mean(legit_sims) "
                    "- margin_floor) / (margin_ceil - margin_floor), 0, 1)",
            "obfuscation_watchwords": OBFUSCATION_WATCHWORDS,
            "obfuscation_substitutions": OBFUSCATION_SUBS,
            "obfuscation_boost": OBFUSCATION_BOOST,
            "obfuscation_rule": "if disguised watchword found (char-substitution "
                                "or letter-spacing), add obfuscation_boost to risk, "
                                "capped at 1.0",
        }, f, indent=2)
    print("  Saved: phishing_phrases.json, phishing_phrase_vectors.npy,")
    print("         legit_anchor_phrases.json, legit_anchor_vectors.npy,")
    print("         semantic_scoring_config.json")

    # ── Scoring function ──────────────────────────────────────────────────
    def semantic_risk(text: str):
        vec  = model.encode([text], normalize_embeddings=True)[0]
        ps   = phish_vecs @ vec
        ls   = legit_vecs @ vec
        p_top = topk_mean(ps)
        l_top = topk_mean(ls)
        margin = p_top - l_top
        base_risk = calibrate(margin)
        best  = int(ps.argmax())

        obf = detect_obfuscation(text)
        final_risk = min(1.0, base_risk + OBFUSCATION_BOOST) if obf else base_risk

        return {
            "risk":   final_risk,
            "base_risk": base_risk,
            "p_sim":  p_top,
            "l_sim":  l_top,
            "margin": margin,
            "phrase": phish_texts[best],
            "tactic": phish_cats[best],
            "obfuscation": obf,
        }

    # ── Demo: original examples + the v1 problem cases ────────────────────
    demo = [
        ("PHISHING", "Your account has been suspended due to unusual activity. "
                     "Verify your password immediately to restore access."),
        ("PHISHING", "Security alert: we detected a suspicious login attempt. "
                     "Confirm your identity now or your account will be closed."),
        ("PHISHING", "Your payment could not be processed. Update your billing "
                     "information within 24 hours to avoid service interruption."),
        ("PHISHING", "Your parcel is on hold. Pay a small customs fee now to "
                     "release your delivery."),
        ("PHISHING", "Congratulations! You have been selected for a cash prize. "
                     "Confirm your details to receive it."),
        ("LEGIT",    "Welcome to our online bookstore. Browse thousands of titles "
                     "with free delivery on orders over twenty pounds."),
        ("LEGIT",    "This guide explains how to set up your development environment "
                     "and run your first project."),
        ("LEGIT",    "Contact our support team Monday to Friday. We aim to respond "
                     "to all enquiries within two working days."),
        ("LEGIT",    "Log in to view your account balance and recent transactions."),
        # v1 false-Amber cases from interactive testing:
        ("BENIGN?",  "can you send the money now?"),
        ("BENIGN?",  "what are your account details?"),
        ("BENIGN?",  "send your details"),
        ("PHISHING", "click the link and input your password and username immediately"),
        # Coercion case that exposed the gap in manual testing:
        ("COERCION", "i will block you now if you dont make payment i asked you"),
        ("COERCION", "i will block you now if you dont make payment i asked you "
                     "you will have to make payment now by force"),
        # Benign verify-language that exposed the one-sided phrase bank
        # (should now score LOW after adding legit verification anchors):
        ("BENIGN?",  "verify your account"),
        ("BENIGN?",  "click link to verify login"),
        ("BENIGN?",  "verify your email to complete registration"),
        # Obfuscation / character-substitution evasion (added after
        # manual testing exposed MiniLM's typo-tolerance as a blind spot):
        ("OBFUSCATED", "verify y0ur acc0unt immediately"),
        ("OBFUSCATED", "s e c u r i t y alert act now"),
        ("CLEAN",       "our verification process is secure and fast"),
    ]

    print("\n" + "-" * 68)
    print("Demo: v2 contrastive scores (incl. v1 false-Amber cases)")
    print("-" * 68)
    print(f"\n{'Type':<11}{'Risk':>6}{'Base':>6}{'Phish':>7}{'Legit':>7}{'Margin':>8}  Tactic / Obfuscation")
    print("-" * 88)

    lines = []
    for label, text in demo:
        r = semantic_risk(text)
        obf_note = ""
        if r["obfuscation"]:
            w, matched, kind = r["obfuscation"][0]
            obf_note = f"  [OBFUSCATION: '{matched}' -> {w}, {kind}]"
        print(f"{label:<11}{r['risk']:>6.2f}{r['base_risk']:>6.2f}{r['p_sim']:>7.3f}"
              f"{r['l_sim']:>7.3f}{r['margin']:>+8.3f}  {r['tactic']}{obf_note}")
        lines.append(f"{label} | risk={r['risk']:.2f} base={r['base_risk']:.2f} "
                     f"margin={r['margin']:+.3f} | tactic={r['tactic']}{obf_note} "
                     f"| {text[:55]}")

    with open(REPORTS / "semantic_phase_demo.txt", "w") as f:
        f.write("PhishLens Phase 1 v2 - Contrastive Semantic Demo\n")
        f.write("=" * 55 + "\n")
        f.write(f"Model: {MODEL_NAME} | top_k={TOP_K} | "
                f"margin range [{MARGIN_FLOOR}, {MARGIN_CEIL}]\n")
        f.write(f"Phishing prototypes: {len(phish_texts)} in "
                f"{len(PHISHING_PHRASES)} categories\n")
        f.write(f"Legit anchors: {len(LEGIT_ANCHORS)}\n\n")
        f.write("\n".join(lines))
    print("\n  Demo report saved -> reports/semantic_phase_demo.txt")

    print("\n" + "=" * 68)
    print("Phase 1 v2 ready. Key expectations:")
    print("  - PHISHING rows: high risk (margin strongly positive)")
    print("  - LEGIT rows: risk ~0 (margin near zero or negative)")
    print("  - BENIGN? rows (v1 false-Ambers): should now be LOWER than v1")
    print("=" * 68)


if __name__ == "__main__":
    main()
