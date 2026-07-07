"""
PhishLens - Definitive ONNX Divergence Test
=============================================
Tests FOUR hypotheses about the 93.65% agreement, all on YOUR real data,
to find which factor actually causes it. Retrains small models so it's fast.

Hypotheses tested:
  A. class_weight='balanced'      -> retrain with class_weight=None
  B. predict_proba vs predict     -> compare via ONNX probability output
  C. skl2onnx version issue       -> try to_onnx() helper instead
  D. opset version                -> try opset 12 and 18

Usage:
    python src/model_training/definitive_test.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
import onnxruntime as rt

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
DROP_COLS = ["url", "source", "tld", "extraction_error"]
CLIP = 1e7

print("=" * 62)
print("Definitive ONNX divergence test")
print("=" * 62)

# Load + prep real data
phi   = pd.read_csv(PROCESSED / "phiusiil_features.csv")
fresh = pd.read_csv(PROCESSED / "fresh_features.csv")
comb  = pd.concat([phi, fresh], ignore_index=True, sort=False)
comb  = comb.drop(columns=[c for c in DROP_COLS if c in comb.columns])
y = comb["label"].astype(int)
X = comb.drop(columns=["label"]).select_dtypes(include=[np.number]).copy()
X = X.clip(lower=-CLIP, upper=CLIP)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
imp = SimpleImputer(strategy="median")
Xtr_i = imp.fit_transform(Xtr).astype(np.float32)
Xte_i = imp.transform(Xte).astype(np.float32)
Xte_i64 = imp.transform(Xte).astype(np.float64)
n_feat = Xtr_i.shape[1]

def onnx_agreement(clf, Xte_arr, tensor_type, opset, label):
    try:
        it = [("input", tensor_type([None, n_feat]))]
        opts = {id(clf): {"zipmap": False}}
        onx = convert_sklearn(clf, initial_types=it, target_opset=opset, options=opts)
        p = Path("_tmp.onnx"); p.write_bytes(onx.SerializeToString())
        sess = rt.InferenceSession(str(p))
        inp = sess.get_inputs()[0].name
        on = sess.run(None, {inp: Xte_arr})[0]
        sk = clf.predict(Xte_arr)
        agree = np.mean(sk == on) * 100
        p.unlink()
        print(f"  {label:<48} {agree:7.4f}%")
        return agree
    except Exception as e:
        print(f"  {label:<48} ERROR: {str(e)[:30]}")
        return None

# ── Hypothesis A: class_weight ──────────────────────────────────────────
print("\n[A] class_weight effect (float32, opset 15):")
clf_bal = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1).fit(Xtr_i, ytr)
onnx_agreement(clf_bal, Xte_i, FloatTensorType, 15, "class_weight='balanced'")

clf_none = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
            class_weight=None, random_state=42, n_jobs=-1).fit(Xtr_i, ytr)
onnx_agreement(clf_none, Xte_i, FloatTensorType, 15, "class_weight=None")

clf_sub = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1).fit(Xtr_i, ytr)
onnx_agreement(clf_sub, Xte_i, FloatTensorType, 15, "class_weight='balanced_subsample'")

# ── Hypothesis D: opset versions (using balanced model) ─────────────────
print("\n[D] opset version effect (balanced, float32):")
for op in [12, 15, 18]:
    onnx_agreement(clf_bal, Xte_i, FloatTensorType, op, f"opset {op}")

# ── Hypothesis B: does class_weight=None + opset combos hit 100? ────────
print("\n[B] class_weight=None across opsets (float32):")
for op in [12, 15, 18]:
    onnx_agreement(clf_none, Xte_i, FloatTensorType, op, f"None, opset {op}")

# ── Hypothesis C: min_samples_leaf=1 (default trees) ───────────────────
print("\n[C] min_samples_leaf effect (balanced, float32, opset 15):")
clf_leaf1 = RandomForestClassifier(n_estimators=200, min_samples_leaf=1,
            class_weight="balanced", random_state=42, n_jobs=-1).fit(Xtr_i, ytr)
onnx_agreement(clf_leaf1, Xte_i, FloatTensorType, 15, "min_samples_leaf=1")

# ── Smaller forest test ─────────────────────────────────────────────────
print("\n[E] forest size effect (balanced, float32, opset 15):")
for n in [10, 50]:
    c = RandomForestClassifier(n_estimators=n, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1).fit(Xtr_i, ytr)
    onnx_agreement(c, Xte_i, FloatTensorType, 15, f"n_estimators={n}")

print("\n" + "=" * 62)
print("Look for the row that hits ~100%. That isolates the cause.")
print("Paste this entire output back.")
print("=" * 62)
