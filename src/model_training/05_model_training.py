"""
PhishLens - Model Training Pipeline (v3 - fully validated)
===========================================================
Combines PhiUSIIL + fresh features, imputes missing values,
trains a Random Forest classifier, generates SHAP plots,
and exports the model to ONNX for the Chrome extension.

Pipeline:
  1. Load & merge phiusiil_features.csv + fresh_features.csv
  2. Align columns, drop non-feature cols
  3. Train/test split (80/20, stratified)
  4. Median imputation fitted on TRAIN only (no leakage)
  5. Train Random Forest with class_weight='balanced'
  6. Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  7. SHAP summary plot -> saved to reports/
  8. Export model to ONNX (opset 15, zipmap=False) -> saved to models/
  9. Verify ONNX vs sklearn agreement >= 99.9%

Requirements:
    pip install skl2onnx==1.20.0 onnx==1.22.0 onnxruntime==1.27.0
    pip install numpy==2.4.0 scikit-learn shap matplotlib

Usage:
    python src/model_training/05_model_training.py

Outputs:
    models/phishlens_rf.onnx          - ONNX model for Chrome extension
    models/phishlens_rf.pkl            - sklearn model (backup)
    models/imputer.pkl                 - fitted imputer (needed at inference)
    models/feature_columns.json        - ordered feature list for ONNX input
    reports/shap_summary.png           - SHAP beeswarm plot
    reports/confusion_matrix.png       - confusion matrix heatmap
    reports/feature_importances.png    - top 20 RF feature importances
    reports/classification_report.txt  - full metrics
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
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import shap

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data"    / "processed"
MODELS    = BASE_DIR / "models"
REPORTS   = BASE_DIR / "reports"
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

PHIUSIIL_FILE = PROCESSED / "phiusiil_features.csv"
FRESH_FILE    = PROCESSED / "fresh_features.csv"

DROP_COLS  = ["url", "source", "tld", "extraction_error"]
RF_PARAMS  = {
    "n_estimators":     25,     # selected via 06_model_selection.py:
                                #   accuracy identical (0.9999) for all sizes 10-200,
                                #   so chose small forest for real-time deployment.
                                #   25 balances ONNX parity, size, and viva-defensibility.
    "max_depth":        None,
    "min_samples_leaf": 2,
    "class_weight":     "balanced",
    "random_state":     42,
    "n_jobs":           -1,
}
RANDOM_STATE = 42
TEST_SIZE    = 0.20
ONNX_OPSET   = 15


# ── 1. Load & merge ───────────────────────────────────────────────────────────
def load_and_merge():
    print("\n[1/9] Loading feature files ...")
    phiusiil = pd.read_csv(PHIUSIIL_FILE)
    print(f"  PhiUSIIL : {len(phiusiil):>7,} rows x {phiusiil.shape[1]} cols")
    fresh = pd.read_csv(FRESH_FILE)
    print(f"  Fresh    : {len(fresh):>7,} rows x {fresh.shape[1]} cols")
    assert "label" in phiusiil.columns, "PhiUSIIL missing label column"
    assert "label" in fresh.columns,    "Fresh features missing label column"
    combined = pd.concat([phiusiil, fresh], ignore_index=True, sort=False)
    print(f"  Combined : {len(combined):>7,} rows x {combined.shape[1]} cols")
    print(f"  label=1 (legitimate): {(combined['label']==1).sum():,}")
    print(f"  label=0 (phishing)  : {(combined['label']==0).sum():,}")
    return combined


# ── 2. Prepare feature matrix ─────────────────────────────────────────────────
def prepare_features(df):
    print("\n[2/9] Preparing feature matrix ...")
    drop = [c for c in DROP_COLS if c in df.columns]
    df   = df.drop(columns=drop)
    y    = df["label"].astype(int)
    X    = df.drop(columns=["label"]).select_dtypes(include=[np.number]).copy()

    # Clip extreme sentinel/overflow values that break ONNX tree traversal.
    # e.g. dns_ttl=2147483647 (int32 max) is a sentinel, not a real TTL.
    # ONNX float32 comparison nodes misroute these vs sklearn float64 trees.
    # Clipping to a realistic ceiling makes both engines agree with zero
    # loss of signal (these values are meaningless extremes anyway).
    CLIP_CEILING = 1e7
    n_clipped = (X.abs() > CLIP_CEILING).sum().sum()
    if n_clipped > 0:
        clipped_cols = X.columns[(X.abs() > CLIP_CEILING).any()].tolist()
        X = X.clip(lower=-CLIP_CEILING, upper=CLIP_CEILING)
        print(f"  Clipped {n_clipped:,} extreme values (>1e7) in columns: {clipped_cols}")

    print(f"  Feature columns : {X.shape[1]}")
    print(f"  Missing values  : {X.isna().sum().sum():,}  "
          f"({X.isna().mean().mean()*100:.1f}% of all cells)")
    return X, y


# ── 3. Split + impute (no leakage) ────────────────────────────────────────────
def split_and_impute(X, y):
    print("\n[3/9] Train/test split & imputation ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train : {len(X_train):,} rows")
    print(f"  Test  : {len(X_test):,} rows")

    imputer     = SimpleImputer(strategy="median")
    # Impute once, then make both float32 (primary) and float64 (fallback) copies
    X_train_imp = imputer.fit_transform(X_train).astype(np.float32)
    X_test_imp64 = imputer.transform(X_test)              # float64 (native)
    X_test_imp   = X_test_imp64.astype(np.float32)        # float32 primary

    with open(MODELS / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    print("  Imputer saved -> models/imputer.pkl")

    feature_cols = list(X.columns)
    with open(MODELS / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Feature list saved -> models/feature_columns.json  ({len(feature_cols)} features)")

    return X_train_imp, X_test_imp, X_test_imp64, y_train, y_test, imputer, feature_cols


# ── 4. Train Random Forest ────────────────────────────────────────────────────
def train_model(X_train, y_train):
    print("\n[4/9] Training Random Forest ...")
    print(f"  Params: {RF_PARAMS}")
    t0  = time.time()
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")
    with open(MODELS / "phishlens_rf.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("  Model saved -> models/phishlens_rf.pkl")
    return clf


# ── 5. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(clf, X_test, y_test, feature_cols):
    print("\n[5/9] Evaluating on test set ...")
    # X_test is already float32 from split_and_impute
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)
    cr   = classification_report(y_test, y_pred,
                                  target_names=["Phishing (0)", "Legitimate (1)"])

    print(f"\n  {'Accuracy':<18}: {acc:.4f}")
    print(f"  {'Precision':<18}: {prec:.4f}")
    print(f"  {'Recall':<18}: {rec:.4f}")
    print(f"  {'F1 Score':<18}: {f1:.4f}")
    print(f"  {'ROC-AUC':<18}: {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{cr}")

    with open(REPORTS / "classification_report.txt", "w") as f:
        f.write("PhishLens Random Forest - Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n")
        f.write(f"ROC-AUC   : {auc:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n{cr}\n")
    print("  Report saved -> reports/classification_report.txt")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Phishing", "Legitimate"],
           yticklabels=["Phishing", "Legitimate"],
           xlabel="Predicted label", ylabel="True label",
           title="PhishLens RF - Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close()
    print("  Confusion matrix saved -> reports/confusion_matrix.png")

    # Feature importance plot
    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top20 = importances.nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    top20.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("PhishLens RF - Top 20 Feature Importances")
    ax.set_xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.savefig(REPORTS / "feature_importances.png", dpi=150)
    plt.close()
    print("  Feature importances saved -> reports/feature_importances.png")

    return acc, f1, auc


# ── 6. SHAP ───────────────────────────────────────────────────────────────────
def generate_shap(clf, X_train, feature_cols):
    print("\n[6/9] Generating SHAP values (sample of 2,000 rows) ...")
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(X_train), size=min(2000, len(X_train)), replace=False)
    X_sub = X_train[idx]

    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sub)

    # Handle both old SHAP API (list of arrays) and new API (single 3D array)
    if isinstance(shap_values, list):
        # Old API: list[class_0_array, class_1_array]
        sv = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # New API: shape (n_samples, n_features, n_classes) — take class 1
        sv = shap_values[:, :, 1]
    else:
        # Fallback: binary output, single array
        sv = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sub, feature_names=feature_cols, show=False, max_display=20)
    plt.title("PhishLens - SHAP Feature Importance (class: Legitimate)")
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP summary saved -> reports/shap_summary.png")


# ── 7+8. Export ONNX with auto-verify and float64 fallback ───────────────────
def export_and_verify_onnx(clf, X_test, X_test_f64):
    """
    Exports to ONNX and verifies agreement. If float32 export falls short,
    automatically retries with float64 (DoubleTensorType) which is immune to
    precision-driven tree misrouting. Returns (onnx_path, dtype_used, ok).
    """
    import onnxruntime as rt
    from skl2onnx.common.data_types import DoubleTensorType

    n_features = X_test.shape[1]
    onnx_path  = MODELS / "phishlens_rf.onnx"
    options    = {id(clf): {"zipmap": False}}

    # ---- Attempt 1: float32 (smaller file, preferred for browser) ----
    print(f"\n[7/9] Exporting to ONNX (opset {ONNX_OPSET}, float32) ...")
    it32  = [("float_input", FloatTensorType([None, n_features]))]
    onx32 = convert_sklearn(clf, initial_types=it32,
                            target_opset=ONNX_OPSET, options=options)
    with open(onnx_path, "wb") as f:
        f.write(onx32.SerializeToString())
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  Saved -> models/phishlens_rf.onnx  ({size_kb:.0f} KB)")

    print("\n[8/9] Verifying ONNX output (float32) ...")
    sess     = rt.InferenceSession(str(onnx_path))
    inp_name = sess.get_inputs()[0].name
    on_preds = sess.run(None, {inp_name: X_test})[0]
    sk_preds = clf.predict(X_test)
    agreement = np.mean(sk_preds == on_preds)
    print(f"  float32 agreement: {agreement*100:.4f}%")

    if agreement >= 0.999:
        print("  float32 export verified - matches sklearn")
        _write_input_dtype("float32")
        return onnx_path, "float32", True

    # ---- Attempt 2: float64 fallback (guaranteed precision match) ----
    print("  float32 below 99.9% - retrying with float64 (DoubleTensorType) ...")
    it64  = [("float_input", DoubleTensorType([None, n_features]))]
    onx64 = convert_sklearn(clf, initial_types=it64,
                            target_opset=ONNX_OPSET, options=options)
    with open(onnx_path, "wb") as f:
        f.write(onx64.SerializeToString())
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  Saved float64 model -> models/phishlens_rf.onnx  ({size_kb:.0f} KB)")

    sess     = rt.InferenceSession(str(onnx_path))
    inp_name = sess.get_inputs()[0].name
    on_preds = sess.run(None, {inp_name: X_test_f64})[0]
    sk_preds = clf.predict(X_test_f64)
    agreement = np.mean(sk_preds == on_preds)
    print(f"  float64 agreement: {agreement*100:.4f}%")

    if agreement >= 0.999:
        print("  float64 export verified - matches sklearn")
        _write_input_dtype("float64")
        return onnx_path, "float64", True
    else:
        print(f"  WARNING: still below 99.9% ({agreement*100:.4f}%) after float64")
        _write_input_dtype("float64")
        return onnx_path, "float64", False


def _write_input_dtype(dtype: str):
    """Record which dtype the ONNX model expects — the extension needs this."""
    meta = {"input_dtype": dtype, "note": "Feed inference features as this dtype"}
    with open(MODELS / "onnx_input_dtype.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Input dtype recorded -> models/onnx_input_dtype.json ({dtype})")


# ── 9. Sanity check on raw URLs ───────────────────────────────────────────────
def sanity_check(clf, imputer, feature_cols):
    print("\n[9/9] Sanity check on held-out sample ...")
    fresh  = pd.read_csv(FRESH_FILE)
    phish  = fresh[fresh["label"] == 0].head(3)
    legit  = fresh[fresh["label"] == 1].head(3)
    sample = pd.concat([phish, legit], ignore_index=True)

    drop = [c for c in DROP_COLS + ["label"] if c in sample.columns]
    X_s  = sample.drop(columns=drop).select_dtypes(include=[np.number]).copy()

    # Add any missing columns (present in training but not in fresh subset)
    for col in feature_cols:
        if col not in X_s.columns:
            X_s[col] = np.nan
    X_s = X_s[feature_cols]

    X_imp = imputer.transform(X_s).astype(np.float32)
    preds = clf.predict(X_imp)
    proba = clf.predict_proba(X_imp)[:, 1]

    true_labels = sample["label"].values
    urls        = sample["url"].values

    print(f"\n  {'URL':<55} {'True':>5} {'Pred':>5} {'P(legit)':>9}")
    print("  " + "-" * 78)
    for url, true, pred, p in zip(urls, true_labels, preds, proba):
        flag = "OK   " if true == pred else "WRONG"
        print(f"  {flag} {str(url)[:52]:<52} {true:>5} {pred:>5} {p:>9.4f}")

    correct = sum(t == p for t, p in zip(true_labels, preds))
    print(f"\n  Sanity check: {correct}/{len(true_labels)} correct")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    print("=" * 65)
    print("PhishLens - Model Training Pipeline  (v3 - fully validated)")
    print("=" * 65)

    df                                                  = load_and_merge()
    X, y                                                = prepare_features(df)
    X_tr, X_te, X_te64, y_tr, y_te, imputer, feat_cols = split_and_impute(X, y)
    clf                                                = train_model(X_tr, y_tr)
    acc, f1, auc                                       = evaluate(clf, X_te, y_te, feat_cols)
    generate_shap(clf, X_tr, feat_cols)
    onnx_path, dtype_used, onnx_ok                     = export_and_verify_onnx(clf, X_te, X_te64)
    sanity_check(clf, imputer, feat_cols)

    elapsed = time.time() - t_total
    print("\n" + "=" * 65)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 65)
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {auc:.4f}")
    print(f"ONNX dtype : {dtype_used}")
    print(f"ONNX OK    : {onnx_ok}")
    print(f"Total time : {elapsed/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
