"""
PhishLens - ONNX Divergence Diagnostic
========================================
Finds exactly which rows/columns cause sklearn and ONNX to disagree.
Run this to identify the root cause of the 93.65% agreement issue.

Usage:
    python src/model_training/diagnose_onnx.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import onnxruntime as rt

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
MODELS    = BASE_DIR / "models"

DROP_COLS = ["url", "source", "tld", "extraction_error"]

print("=" * 60)
print("ONNX Divergence Diagnostic")
print("=" * 60)

# Rebuild the exact test set
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

phiusiil = pd.read_csv(PROCESSED / "phiusiil_features.csv")
fresh    = pd.read_csv(PROCESSED / "fresh_features.csv")
combined = pd.concat([phiusiil, fresh], ignore_index=True, sort=False)

drop = [c for c in DROP_COLS if c in combined.columns]
combined = combined.drop(columns=drop)
y = combined["label"].astype(int)
X = combined.drop(columns=["label"]).select_dtypes(include=[np.number])
feature_cols = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Load saved imputer + model
with open(MODELS / "imputer.pkl", "rb") as f:
    imputer = pickle.load(f)
with open(MODELS / "phishlens_rf.pkl", "rb") as f:
    clf = pickle.load(f)

X_test_i = imputer.transform(X_test).astype(np.float32)

# Predictions
sess = rt.InferenceSession(str(MODELS / "phishlens_rf.onnx"))
inp  = sess.get_inputs()[0].name
on_preds = sess.run(None, {inp: X_test_i})[0]
sk_preds = clf.predict(X_test_i)

disagree = sk_preds != on_preds
n_dis = disagree.sum()
print(f"\nTotal test rows : {len(sk_preds):,}")
print(f"Disagreements   : {n_dis:,} ({n_dis/len(sk_preds)*100:.2f}%)")

# ── Check 1: value ranges (float32 overflow / precision loss) ──────────
print("\n--- Feature value ranges ---")
X_test_raw = X_test.copy()
ranges = []
for i, col in enumerate(feature_cols):
    col_max = np.nanmax(np.abs(X_test_raw[col].values.astype(np.float64)))
    ranges.append((col, col_max))

# Columns with very large values lose precision in float32
big = [(c, m) for c, m in ranges if m > 1e7]
if big:
    print("Columns with values > 1e7 (float32 precision risk):")
    for c, m in sorted(big, key=lambda x: -x[1]):
        print(f"  {c:<30} max abs = {m:,.0f}")
else:
    print("No columns exceed 1e7 — float32 precision is not the issue.")

# ── Check 2: float32 vs float64 prediction divergence ──────────────────
print("\n--- float64 vs float32 sklearn divergence ---")
X_test_i64 = imputer.transform(X_test).astype(np.float64)
sk_preds64 = clf.predict(X_test_i64)
sk_div = (sk_preds64 != sk_preds).sum()
print(f"sklearn float64 vs float32 disagreements: {sk_div:,}")
if sk_div > 0:
    print("  -> The model itself is sensitive to float32 casting.")
    print("  -> This is the root cause. Fix: keep ONNX in float64 (DoubleTensorType).")

# ── Check 3: how close are disagreement probabilities to 0.5? ──────────
if n_dis > 0:
    proba = clf.predict_proba(X_test_i)[:, 1]
    dis_proba = proba[disagree]
    print(f"\n--- Disagreement probabilities ---")
    print(f"  Mean P(legit) on disagreements: {dis_proba.mean():.4f}")
    print(f"  Min: {dis_proba.min():.4f}  Max: {dis_proba.max():.4f}")
    near_boundary = np.sum((dis_proba > 0.45) & (dis_proba < 0.55))
    print(f"  Within 0.45-0.55 (borderline): {near_boundary}/{n_dis}")

print("\n" + "=" * 60)
print("Diagnosis complete.")
print("=" * 60)
