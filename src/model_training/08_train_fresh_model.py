"""
PhishLens - Final Model Training (FRESH DATA ONLY - leakage-free)
==================================================================
Trains the deployed model on the fresh PhishTank/Tranco URLs only,
using the features extracted by 04_feature_extraction.py.

WHY FRESH-ONLY:
  The leakage check (07_leakage_check.py) showed the merged PhiUSIIL+fresh
  dataset let the model identify SOURCE (100% source-id accuracy) and
  cross-source generalisation collapsed to ~0.55. The merged 0.9999 was
  inflated by dataset structure, not genuine phishing signal.
  Training on fresh data ALONE removes that fingerprint and gives an
  honest result (~0.997) that also matches what the Chrome extension
  will actually compute at runtime (same feature extractor).

Pipeline:
  1. Load fresh_features.csv only
  2. Drop columns that are entirely empty (none, since fresh is self-contained)
  3. 5-fold cross-validation -> proves the score is robust, not a lucky split
  4. Final 80/20 train/test split, median imputation (no leakage)
  5. Train Random Forest
  6. Evaluate + confusion matrix + feature importances + SHAP
  7. Export ONNX (float32, zipmap=False) + verify parity
  8. Sanity check on real URLs

Usage:
    python src/model_training/08_train_fresh_model.py

Outputs (all suffixed _fresh to keep separate from merged model):
    models/phishlens_fresh.onnx
    models/phishlens_fresh.pkl
    models/imputer_fresh.pkl
    models/feature_columns_fresh.json
    reports/fresh_classification_report.txt
    reports/fresh_confusion_matrix.png
    reports/fresh_feature_importances.png
    reports/fresh_shap_summary.png
    reports/fresh_cross_validation.txt
"""

import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import shap

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
MODELS    = BASE_DIR / "models"
REPORTS   = BASE_DIR / "reports"
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

FRESH_FILE = PROCESSED / "fresh_features.csv"
DROP_COLS  = ["url", "source", "tld", "extraction_error"]
CLIP       = 1e7
RS         = 42

RF_PARAMS = {
    "n_estimators":     100,    # fresh data is smaller; 100 trees is a safe,
                                # conventional choice (no over-provisioning concern
                                # here, and ONNX parity is fine at this dataset size)
    "max_depth":        None,
    "min_samples_leaf": 2,
    "class_weight":     "balanced",
    "random_state":     RS,
    "n_jobs":           -1,
}


def main():
    t0 = time.time()
    print("=" * 66)
    print("PhishLens - Final Model (FRESH DATA ONLY - leakage-free)")
    print("=" * 66)

    # ── 1. Load fresh data ────────────────────────────────────────────────
    print("\n[1/8] Loading fresh feature data ...")
    df = pd.read_csv(FRESH_FILE)
    print(f"  Rows: {len(df):,}")
    print(f"  label=1 (legitimate): {(df['label']==1).sum():,}")
    print(f"  label=0 (phishing)  : {(df['label']==0).sum():,}")

    # ── 2. Prepare features ───────────────────────────────────────────────
    print("\n[2/8] Preparing features ...")
    y = df["label"].astype(int)
    X = df.drop(columns=[c for c in DROP_COLS + ["label"] if c in df.columns])
    X = X.select_dtypes(include=[np.number]).copy()
    # Drop any fully-empty columns
    empty = X.columns[X.isna().all()].tolist()
    if empty:
        X = X.drop(columns=empty)
        print(f"  Dropped {len(empty)} fully-empty columns")
    X = X.clip(lower=-CLIP, upper=CLIP)
    feature_cols = list(X.columns)
    print(f"  Usable features: {len(feature_cols)}")
    print(f"  Missing values : {X.isna().sum().sum():,} "
          f"({X.isna().mean().mean()*100:.1f}%)")

    # ── 3. Cross-validation (robustness proof) ────────────────────────────
    print("\n[3/8] 5-fold cross-validation (proves score is not a lucky split) ...")
    imp_cv = SimpleImputer(strategy="median")
    X_cv   = imp_cv.fit_transform(X).astype(np.float32)
    clf_cv = RandomForestClassifier(**RF_PARAMS)
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    acc_scores = cross_val_score(clf_cv, X_cv, y, cv=skf, scoring="accuracy")
    f1_scores  = cross_val_score(clf_cv, X_cv, y, cv=skf, scoring="f1")
    print(f"  CV Accuracy: {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}")
    print(f"  CV F1      : {f1_scores.mean():.4f} +/- {f1_scores.std():.4f}")
    print(f"  Per-fold accuracy: {[f'{s:.4f}' for s in acc_scores]}")
    with open(REPORTS / "fresh_cross_validation.txt", "w") as f:
        f.write("PhishLens Fresh-Only Model - 5-Fold Cross-Validation\n")
        f.write("=" * 52 + "\n")
        f.write(f"CV Accuracy: {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}\n")
        f.write(f"CV F1      : {f1_scores.mean():.4f} +/- {f1_scores.std():.4f}\n")
        f.write(f"Per-fold accuracy: {list(acc_scores)}\n")
        f.write(f"Per-fold F1      : {list(f1_scores)}\n")
    print("  Saved -> reports/fresh_cross_validation.txt")

    # ── 4. Train/test split + impute ──────────────────────────────────────
    print("\n[4/8] Train/test split & imputation ...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    X_tr_i  = imputer.fit_transform(X_tr).astype(np.float32)
    X_te_i  = imputer.transform(X_te).astype(np.float32)
    print(f"  Train: {len(X_tr_i):,}  Test: {len(X_te_i):,}")
    with open(MODELS / "imputer_fresh.pkl", "wb") as f:
        pickle.dump(imputer, f)
    with open(MODELS / "feature_columns_fresh.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print("  Saved imputer_fresh.pkl + feature_columns_fresh.json")

    # ── 5. Train ──────────────────────────────────────────────────────────
    print("\n[5/8] Training Random Forest ...")
    clf = RandomForestClassifier(**RF_PARAMS).fit(X_tr_i, y_tr)
    with open(MODELS / "phishlens_fresh.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(f"  Trained {RF_PARAMS['n_estimators']} trees. Saved phishlens_fresh.pkl")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    print("\n[6/8] Evaluating on held-out test set ...")
    y_pred  = clf.predict(X_te_i)
    y_proba = clf.predict_proba(X_te_i)[:, 1]
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, y_proba)
    cm   = confusion_matrix(y_te, y_pred)
    cr   = classification_report(y_te, y_pred,
                                  target_names=["Phishing (0)", "Legitimate (1)"])
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{cr}")

    with open(REPORTS / "fresh_classification_report.txt", "w") as f:
        f.write("PhishLens Fresh-Only Model - Evaluation\n")
        f.write("=" * 45 + "\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n")
        f.write(f"ROC-AUC   : {auc:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n{cr}\n")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=["Phishing","Legitimate"],
           yticklabels=["Phishing","Legitimate"],
           xlabel="Predicted", ylabel="True",
           title="PhishLens (Fresh-Only) - Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS / "fresh_confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top = importances.nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(8,6))
    top.plot(kind="barh", ax=ax, color="seagreen")
    ax.set_title("PhishLens (Fresh-Only) - Top 20 Features")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS / "fresh_feature_importances.png", dpi=150)
    plt.close()
    print("  Saved confusion matrix + feature importance plots")

    # SHAP
    try:
        rng = np.random.default_rng(RS)
        idx = rng.choice(len(X_tr_i), size=min(2000, len(X_tr_i)), replace=False)
        X_sub = X_tr_i[idx]
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_sub)
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        plt.figure(figsize=(10,8))
        shap.summary_plot(sv, X_sub, feature_names=feature_cols, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(REPORTS / "fresh_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved SHAP summary plot")
    except Exception as e:
        print(f"  SHAP skipped ({str(e)[:40]})")

    # ── 7. Export ONNX + verify ───────────────────────────────────────────
    print("\n[7/8] Exporting to ONNX + verifying ...")
    it   = [("float_input", FloatTensorType([None, len(feature_cols)]))]
    opts = {id(clf): {"zipmap": False}}
    onx  = convert_sklearn(clf, initial_types=it, target_opset=15, options=opts)
    onnx_path = MODELS / "phishlens_fresh.onnx"
    onnx_path.write_bytes(onx.SerializeToString())
    size_kb = onnx_path.stat().st_size / 1024
    import onnxruntime as rt
    sess = rt.InferenceSession(str(onnx_path))
    inp  = sess.get_inputs()[0].name
    on_pred = sess.run(None, {inp: X_te_i})[0]
    parity  = np.mean(clf.predict(X_te_i) == on_pred) * 100
    print(f"  Saved phishlens_fresh.onnx ({size_kb:.0f} KB)")
    print(f"  ONNX parity: {parity:.2f}%")

    # ── 8. Sanity check ───────────────────────────────────────────────────
    print("\n[8/8] Sanity check on real URLs ...")
    df_full = pd.read_csv(FRESH_FILE)
    phish = df_full[df_full["label"]==0].head(3)
    legit = df_full[df_full["label"]==1].head(3)
    sample = pd.concat([phish, legit], ignore_index=True)
    Xs = sample.drop(columns=[c for c in DROP_COLS + ["label"] if c in sample.columns])
    Xs = Xs.select_dtypes(include=[np.number]).copy()
    for c in feature_cols:
        if c not in Xs.columns:
            Xs[c] = np.nan
    Xs = Xs[feature_cols].clip(lower=-CLIP, upper=CLIP)
    Xs_i = imputer.transform(Xs).astype(np.float32)
    preds = clf.predict(Xs_i)
    proba = clf.predict_proba(Xs_i)[:,1]
    print(f"\n  {'URL':<50} {'True':>5} {'Pred':>5} {'P(legit)':>9}")
    print("  " + "-"*72)
    correct = 0
    for url, t, p, pr in zip(sample["url"], sample["label"], preds, proba):
        ok = "OK   " if t==p else "WRONG"
        if t==p: correct += 1
        print(f"  {ok} {str(url)[:47]:<47} {t:>5} {p:>5} {pr:>9.4f}")
    print(f"\n  {correct}/{len(preds)} correct")

    elapsed = time.time() - t0
    print("\n" + "=" * 66)
    print("FRESH-ONLY MODEL COMPLETE")
    print("=" * 66)
    print(f"  CV Accuracy   : {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test F1       : {f1:.4f}")
    print(f"  ONNX parity   : {parity:.2f}%")
    print(f"  Features      : {len(feature_cols)}")
    print(f"  Time          : {elapsed/60:.1f} min")
    print("=" * 66)


if __name__ == "__main__":
    main()
