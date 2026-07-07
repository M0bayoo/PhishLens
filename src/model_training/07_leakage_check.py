"""
PhishLens - Data Leakage & Model Honesty Check
================================================
Answers the real question: is the 0.9999 accuracy genuine, or is the
model exploiting a shortcut (data leakage / source memorisation)?

Three checks:
  1. LEAKAGE: does any single feature predict the label almost perfectly?
  2. SOURCE MEMORISATION: can the model distinguish dataset source?
     And does train-on-one-source / test-on-other collapse?
  3. HONEST BASELINE: accuracy on fresh PhishTank/Tranco data ALONE.

Usage:
    python src/model_training/07_leakage_check.py

Outputs:
    reports/leakage_report.txt
    reports/feature_label_correlation.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
REPORTS   = BASE_DIR / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["url", "source", "tld", "extraction_error"]
CLIP = 1e7
RS = 42

report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(msg)

log("=" * 66)
log("PhishLens - Data Leakage & Model Honesty Check")
log("=" * 66)

# ── Load with source labels preserved ────────────────────────────────────
phi   = pd.read_csv(PROCESSED / "phiusiil_features.csv")
fresh = pd.read_csv(PROCESSED / "fresh_features.csv")

# Tag origin before merge
phi   = phi.copy();   phi["_origin"]   = "phiusiil"
fresh = fresh.copy(); fresh["_origin"] = "fresh"

comb = pd.concat([phi, fresh], ignore_index=True, sort=False)
origin = comb["_origin"].values
y_all  = comb["label"].astype(int).values

X_all = comb.drop(columns=[c for c in DROP_COLS + ["label", "_origin"]
                           if c in comb.columns]).select_dtypes(include=[np.number]).copy()
X_all = X_all.clip(lower=-CLIP, upper=CLIP)
feature_cols = list(X_all.columns)

log(f"\nTotal rows: {len(X_all):,}  |  features: {len(feature_cols)}")
log(f"  phiusiil rows: {(origin=='phiusiil').sum():,}")
log(f"  fresh rows   : {(origin=='fresh').sum():,}")

# ════════════════════════════════════════════════════════════════════════
# CHECK 1 — Single-feature leakage
# Which features, ALONE, can separate phishing from legit almost perfectly?
# ════════════════════════════════════════════════════════════════════════
log("\n" + "-" * 66)
log("[CHECK 1] Single-feature leakage")
log("-" * 66)
log("A feature that alone scores >0.95 accuracy is a leakage suspect.\n")

imp_all = SimpleImputer(strategy="median")
X_imp   = imp_all.fit_transform(X_all)

# Point-biserial correlation of each feature with the label
corrs = []
for i, col in enumerate(feature_cols):
    c = np.corrcoef(X_imp[:, i], y_all)[0, 1]
    corrs.append((col, abs(c) if not np.isnan(c) else 0.0))

corrs.sort(key=lambda t: -t[1])
log("Top 12 features by |correlation| with label:")
for col, c in corrs[:12]:
    flag = "  <-- SUSPECT" if c > 0.9 else ""
    log(f"  {col:<32} |r| = {c:.3f}{flag}")

# Also: best single-feature decision-stump accuracy for top suspects
log("\nSingle-feature accuracy (decision stump) for top 5:")
from sklearn.tree import DecisionTreeClassifier
for col, c in corrs[:5]:
    idx = feature_cols.index(col)
    stump = DecisionTreeClassifier(max_depth=1, random_state=RS)
    Xi = X_imp[:, idx].reshape(-1, 1)
    stump.fit(Xi, y_all)
    acc = stump.score(Xi, y_all)
    flag = "  <-- near-perfect ALONE" if acc > 0.95 else ""
    log(f"  {col:<32} acc = {acc:.3f}{flag}")

# ════════════════════════════════════════════════════════════════════════
# CHECK 2 — Source memorisation
# ════════════════════════════════════════════════════════════════════════
log("\n" + "-" * 66)
log("[CHECK 2] Source memorisation")
log("-" * 66)

# 2a: Can a model tell phiusiil from fresh using the features?
log("\n2a. Can the model identify which DATASET a row came from?")
y_src = (origin == "phiusiil").astype(int)
clf_src = RandomForestClassifier(n_estimators=50, random_state=RS, n_jobs=-1)
src_scores = cross_val_score(clf_src, X_imp, y_src, cv=3, scoring="accuracy")
log(f"   Source-identification accuracy: {src_scores.mean():.3f}")
if src_scores.mean() > 0.95:
    log("   -> Features STRONGLY encode source. Leakage risk is real:")
    log("      the model can tell datasets apart, so it may be using")
    log("      source as a proxy for the label.")
else:
    log("   -> Features do not strongly encode source. Good sign.")

# 2b: Train on one source, test on the other
log("\n2b. Cross-source generalisation (train on one, test on other):")
for train_src, test_src in [("phiusiil", "fresh"), ("fresh", "phiusiil")]:
    tr = origin == train_src
    te = origin == test_src
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_all[tr])
    Xte = imp.transform(X_all[te])
    clf = RandomForestClassifier(n_estimators=50, class_weight="balanced",
                                  random_state=RS, n_jobs=-1).fit(Xtr, y_all[tr])
    acc = accuracy_score(y_all[te], clf.predict(Xte))
    f1  = f1_score(y_all[te], clf.predict(Xte))
    verdict = "GENERALISES" if acc > 0.85 else "COLLAPSES -> was memorising"
    log(f"   train {train_src:<9} -> test {test_src:<9}: acc {acc:.3f} f1 {f1:.3f}  [{verdict}]")

# ════════════════════════════════════════════════════════════════════════
# CHECK 3 — Honest baseline on fresh data alone
# ════════════════════════════════════════════════════════════════════════
log("\n" + "-" * 66)
log("[CHECK 3] Honest baseline: fresh PhishTank/Tranco data ONLY")
log("-" * 66)

fresh_mask = origin == "fresh"
Xf = X_all[fresh_mask].reset_index(drop=True)
yf = y_all[fresh_mask]

# Drop columns that are entirely empty for fresh data (the phiusiil-only cols)
non_empty = Xf.columns[Xf.notna().any()].tolist()
Xf = Xf[non_empty]
log(f"\nFresh-only usable features (non-empty): {len(non_empty)} of {len(feature_cols)}")

Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
    Xf, yf, test_size=0.2, random_state=RS, stratify=yf
)
impf = SimpleImputer(strategy="median")
Xf_tr_i = impf.fit_transform(Xf_tr)
Xf_te_i = impf.transform(Xf_te)
clf_f = RandomForestClassifier(n_estimators=50, class_weight="balanced",
                                random_state=RS, n_jobs=-1).fit(Xf_tr_i, yf_tr)
acc_f = accuracy_score(yf_te, clf_f.predict(Xf_te_i))
f1_f  = f1_score(yf_te, clf_f.predict(Xf_te_i))
log(f"  Accuracy on fresh-only data: {acc_f:.4f}")
log(f"  F1 on fresh-only data      : {f1_f:.4f}")
log("  (This is the most honest estimate of real-world performance,")
log("   since fresh data has no phiusiil-vs-fresh structure to exploit.)")

# ── Correlation plot ─────────────────────────────────────────────────────
top = corrs[:15][::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([t[0] for t in top], [t[1] for t in top],
        color=["crimson" if t[1] > 0.9 else "steelblue" for t in top])
ax.axvline(0.9, color="red", linestyle="--", alpha=0.5, label="leakage threshold (0.9)")
ax.set_xlabel("|correlation| with label")
ax.set_title("PhishLens - Feature vs Label Correlation (leakage check)")
ax.legend()
plt.tight_layout()
plt.savefig(REPORTS / "feature_label_correlation.png", dpi=150)
plt.close()
log(f"\nCorrelation plot saved -> reports/feature_label_correlation.png")

# ── Verdict ──────────────────────────────────────────────────────────────
log("\n" + "=" * 66)
log("SUMMARY")
log("=" * 66)
n_suspect = sum(1 for _, c in corrs if c > 0.9)
log(f"  Single-feature leakage suspects (|r|>0.9): {n_suspect}")
log(f"  Source-identification accuracy           : {src_scores.mean():.3f}")
log(f"  Fresh-only honest accuracy               : {acc_f:.4f}")
log("\n  Interpret:")
log("  - If suspects=0, source-id is low, cross-source generalises,")
log("    and fresh-only stays high -> your 0.9999 is likely GENUINE.")
log("  - If source-id is high AND cross-source collapses -> the merged")
log("    accuracy is inflated by source structure; report fresh-only instead.")
log("=" * 66)

with open(REPORTS / "leakage_report.txt", "w") as f:
    f.write("\n".join(report_lines))
print(f"\nFull report saved -> reports/leakage_report.txt")
