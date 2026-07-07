"""
PhishLens - Model Complexity Selection
========================================
Principled selection of Random Forest size for real-time browser deployment.
Evaluates n_estimators against:
  - Accuracy, Precision, Recall, F1 (does performance hold?)
  - Model file size (browser download weight)
  - Inference latency (real-time constraint)
  - ONNX prediction parity (deployment fidelity)

Produces:
  reports/model_selection.csv          - full results table
  reports/model_selection.png          - accuracy + ONNX parity vs n_trees
  Console recommendation for the smallest forest retaining full performance.

This is a methodological step: it justifies the chosen model complexity
for the dissertation rather than picking 200 trees arbitrarily.

Usage:
    python src/model_training/06_model_selection.py
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
REPORTS   = BASE_DIR / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["url", "source", "tld", "extraction_error"]
CLIP      = 1e7
RANDOM_STATE = 42

# Forest sizes to evaluate
TREE_SIZES = [10, 25, 50, 75, 100, 150, 200]

print("=" * 70)
print("PhishLens - Model Complexity Selection")
print("=" * 70)

# ── Load & prep (same pipeline as training) ──────────────────────────────
print("\nLoading data ...")
phi   = pd.read_csv(PROCESSED / "phiusiil_features.csv")
fresh = pd.read_csv(PROCESSED / "fresh_features.csv")
comb  = pd.concat([phi, fresh], ignore_index=True, sort=False)
comb  = comb.drop(columns=[c for c in DROP_COLS if c in comb.columns])
y = comb["label"].astype(int)
X = comb.drop(columns=["label"]).select_dtypes(include=[np.number]).copy()
X = X.clip(lower=-CLIP, upper=CLIP)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
imp   = SimpleImputer(strategy="median")
Xtr_i = imp.fit_transform(Xtr).astype(np.float32)
Xte_i = imp.transform(Xte).astype(np.float32)
n_feat = Xtr_i.shape[1]
print(f"  Train {len(Xtr_i):,} | Test {len(Xte_i):,} | Features {n_feat}")

# ── Evaluate each forest size ────────────────────────────────────────────
print(f"\nEvaluating {len(TREE_SIZES)} forest sizes ...\n")
print(f"  {'trees':>6} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} "
      f"{'size_KB':>8} {'infer_ms':>9} {'onnx%':>8}")
print("  " + "-" * 66)

results = []
for n_est in TREE_SIZES:
    clf = RandomForestClassifier(
        n_estimators=n_est, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ).fit(Xtr_i, ytr)

    # Metrics
    pred = clf.predict(Xte_i)
    acc  = accuracy_score(yte, pred)
    prec = precision_score(yte, pred)
    rec  = recall_score(yte, pred)
    f1   = f1_score(yte, pred)

    # ONNX export + size + parity
    it   = [("input", FloatTensorType([None, n_feat]))]
    opts = {id(clf): {"zipmap": False}}
    onx  = convert_sklearn(clf, initial_types=it, target_opset=15, options=opts)
    tmp  = REPORTS / "_sel_tmp.onnx"
    tmp.write_bytes(onx.SerializeToString())
    size_kb = tmp.stat().st_size / 1024

    sess = rt.InferenceSession(str(tmp))
    inp  = sess.get_inputs()[0].name

    # Inference latency (per-sample, averaged over test set)
    t0 = time.time()
    on_pred = sess.run(None, {inp: Xte_i})[0]
    infer_ms = (time.time() - t0) / len(Xte_i) * 1000

    onnx_parity = np.mean(clf.predict(Xte_i) == on_pred) * 100
    tmp.unlink()

    print(f"  {n_est:>6} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} "
          f"{size_kb:>8.0f} {infer_ms:>9.4f} {onnx_parity:>8.2f}")

    results.append({
        "n_estimators": n_est, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "size_kb": size_kb,
        "infer_ms_per_sample": infer_ms, "onnx_parity_pct": onnx_parity,
    })

res_df = pd.DataFrame(results)
res_df.to_csv(REPORTS / "model_selection.csv", index=False)
print(f"\n  Results table saved -> reports/model_selection.csv")

# ── Recommendation logic ─────────────────────────────────────────────────
# Smallest forest within 0.1% of best accuracy AND >= 99% ONNX parity
best_acc = res_df["accuracy"].max()
eligible = res_df[
    (res_df["accuracy"] >= best_acc - 0.001) &
    (res_df["onnx_parity_pct"] >= 99.0)
]
if len(eligible) > 0:
    rec = eligible.iloc[0]
    print(f"\n  RECOMMENDATION: n_estimators = {int(rec['n_estimators'])}")
    print(f"    accuracy {rec['accuracy']:.4f} | f1 {rec['f1']:.4f} | "
          f"ONNX parity {rec['onnx_parity_pct']:.2f}% | {rec['size_kb']:.0f} KB")
    print(f"    -> smallest forest with full accuracy AND >=99% ONNX parity")
else:
    # Fall back: best ONNX parity among top-accuracy models
    top = res_df[res_df["accuracy"] >= best_acc - 0.001]
    rec = top.loc[top["onnx_parity_pct"].idxmax()]
    print(f"\n  No size reached 99% parity at full accuracy.")
    print(f"  Best available: n_estimators = {int(rec['n_estimators'])} "
          f"({rec['onnx_parity_pct']:.2f}% parity, {rec['accuracy']:.4f} acc)")

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(res_df["n_estimators"], res_df["accuracy"]*100,
         "o-", color="steelblue", label="Accuracy (%)")
ax1.plot(res_df["n_estimators"], res_df["f1"]*100,
         "s--", color="navy", label="F1 (%)")
ax1.set_xlabel("Number of trees (n_estimators)")
ax1.set_ylabel("Accuracy / F1 (%)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_ylim(min(res_df["accuracy"].min()*100 - 1, 98), 100.2)

ax2 = ax1.twinx()
ax2.plot(res_df["n_estimators"], res_df["onnx_parity_pct"],
         "^-", color="crimson", label="ONNX parity (%)")
ax2.set_ylabel("ONNX parity (%)", color="crimson")
ax2.tick_params(axis="y", labelcolor="crimson")

ax1.set_title("PhishLens - Model Complexity vs Performance & ONNX Parity")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
plt.tight_layout()
plt.savefig(REPORTS / "model_selection.png", dpi=150)
plt.close()
print(f"  Plot saved -> reports/model_selection.png")

print("\n" + "=" * 70)
print("Selection complete. Review the table and recommendation above.")
print("=" * 70)
