"""
PhishLens - ONNX Parity Root-Cause Diagnostic (fresh model)
=============================================================
Finds the REAL cause of the 90% ONNX parity by examining exactly
which rows disagree and what they have in common.

Uses the already-saved fresh model + data. Fast (~30s).

Usage:
    python src/model_training/09_diagnose_fresh_onnx.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
import onnxruntime as rt

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
MODELS    = BASE_DIR / "models"
DROP_COLS = ["url", "source", "tld", "extraction_error"]
CLIP = 1e7
RS = 42

print("=" * 64)
print("ONNX Parity Root-Cause Diagnostic (fresh model)")
print("=" * 64)

# Rebuild exact fresh test set
df = pd.read_csv(PROCESSED / "fresh_features.csv")
y  = df["label"].astype(int)
X  = df.drop(columns=[c for c in DROP_COLS + ["label"] if c in df.columns])
X  = X.select_dtypes(include=[np.number]).copy()
X  = X.drop(columns=X.columns[X.isna().all()].tolist())
X  = X.clip(lower=-CLIP, upper=CLIP)
feature_cols = list(X.columns)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)

with open(MODELS / "phishlens_fresh.pkl", "rb") as f:
    clf = pickle.load(f)
with open(MODELS / "imputer_fresh.pkl", "rb") as f:
    imputer = pickle.load(f)

X_te_i = imputer.transform(X_te).astype(np.float32)

# Get ONNX predictions
sess = rt.InferenceSession(str(MODELS / "phishlens_fresh.onnx"))
inp  = sess.get_inputs()[0].name
on_pred   = sess.run(None, {inp: X_te_i})[0]
on_proba  = sess.run(None, {inp: X_te_i})[1]   # probabilities
sk_pred   = clf.predict(X_te_i)
sk_proba  = clf.predict_proba(X_te_i)

disagree = sk_pred != on_pred
n_dis = disagree.sum()
print(f"\nTest rows: {len(sk_pred):,}  Disagreements: {n_dis:,} ({n_dis/len(sk_pred)*100:.2f}%)")

# ── Are disagreements at the decision boundary? ──────────────────────────
print("\n--- Where do disagreements happen? ---")
sk_p1 = sk_proba[:, 1]
if n_dis > 0:
    dis_p = sk_p1[disagree]
    print(f"  sklearn P(legit) on disagreeing rows:")
    print(f"    min {dis_p.min():.4f}  max {dis_p.max():.4f}  mean {dis_p.mean():.4f}")
    near_half = np.sum((dis_p > 0.4) & (dis_p < 0.6))
    print(f"    within 0.4-0.6 (boundary): {near_half}/{n_dis} ({near_half/n_dis*100:.0f}%)")

# ── Compare PROBABILITIES not just labels ────────────────────────────────
print("\n--- Probability agreement (not just labels) ---")
proba_diff = np.abs(sk_proba[:, 1] - on_proba[:, 1])
print(f"  Max |P_sklearn - P_onnx| : {proba_diff.max():.6f}")
print(f"  Mean|P_sklearn - P_onnx| : {proba_diff.mean():.6f}")
print(f"  Rows with diff > 0.5     : {np.sum(proba_diff > 0.5)}")
print(f"  Rows with diff > 0.1     : {np.sum(proba_diff > 0.1)}")
print(f"  Rows with diff < 0.01    : {np.sum(proba_diff < 0.01)}")

# ── Test: does float64 throughout fix THIS data? ─────────────────────────
print("\n--- Fix test: float64 export + float64 inference ---")
X_te_i64 = imputer.transform(X_te).astype(np.float64)
it64 = [("input", DoubleTensorType([None, len(feature_cols)]))]
opts = {id(clf): {"zipmap": False}}
onx64 = convert_sklearn(clf, initial_types=it64, target_opset=15, options=opts)
p = MODELS / "_diag64.onnx"; p.write_bytes(onx64.SerializeToString())
sess64 = rt.InferenceSession(str(p))
inp64 = sess64.get_inputs()[0].name
on64 = sess64.run(None, {inp64: X_te_i64})[0]
sk64 = clf.predict(X_te_i64)
par64 = np.mean(sk64 == on64) * 100
print(f"  float64 throughout parity: {par64:.2f}%")
p.unlink()

# ── Test: round inputs to fewer decimals (kill float noise) ──────────────
print("\n--- Fix test: round inputs to 5 decimals before both ---")
X_te_r = np.round(X_te_i, 5)
on_r = sess.run(None, {inp: X_te_r})[0]
sk_r = clf.predict(X_te_r)
print(f"  rounded-input parity: {np.mean(sk_r==on_r)*100:.2f}%")

print("\n" + "=" * 64)
print("READING THE RESULT:")
print("  - If proba diffs are tiny (<0.01) but labels flip -> pure")
print("    boundary tie-breaking; harmless, fixable by using ONNX")
print("    probabilities + same threshold as sklearn.")
print("  - If float64 parity ~100% -> export float64 is the fix.")
print("  - If proba diffs are LARGE -> real conversion bug, investigate.")
print("=" * 64)
