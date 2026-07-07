"""
PhishLens - FINAL Model Training (corrected, leakage-free, deployment-honest)
==============================================================================
This supersedes 05_model_training.py and 08_train_fresh_model.py.

CORRECTIONS BAKED IN (and WHY):
  1. FRESH DATA ONLY. The merged PhiUSIIL+fresh dataset caused source leakage
     (07_leakage_check.py: 100% source-identifiability, cross-source collapse
     to ~0.55). Training on fresh data alone removes the dataset fingerprint
     and matches what the Chrome extension actually computes at runtime.

  2. 200 TREES. Earlier the forest was cut to 10 trees to "fix" ONNX parity.
     That was a MISDIAGNOSIS - ONNX probabilities were always identical to
     sklearn (09_diagnose_fresh_onnx.py: max prob diff = 0.000000). The parity
     gap was purely a label-thresholding difference, not a model/precision
     problem. With that reason void, we use 200 trees (conventional strong
     default, robust, not under-provisioned).

  3. NO ONNX "FIXES". Removed the float64 fallback and the clipping-as-ONNX-fix.
     ONNX never needed fixing. The extension reads the PROBABILITY output and
     applies one fixed threshold (saved as an artifact) - that gives full
     parity with the reference model.

  4. SENTINEL VALUES -> MISSING (data-quality fix, not ONNX). Values like
     dns_ttl=2147483647 (int32 max) and -1 placeholders are garbage sentinels,
     not real measurements. We convert them to NaN and let the imputer handle
     them properly, rather than clipping to an arbitrary ceiling.

  5. THRESHOLD-CONSISTENT METRICS. Reported metrics are computed by applying
     the SAME fixed decision threshold the extension will use, so the
     dissertation numbers match real deployment behaviour.

  6. CROSS-VALIDATED HEADLINE METRICS. With ~10.8k rows, headline accuracy/F1
     come from 5-fold cross-validation (robust), with a single held-out split
     as confirmation.

Usage:
    python src/model_training/05_train_final_model.py

Key outputs:
    models/phishlens_final.onnx            - deployed model
    models/phishlens_final.pkl             - sklearn reference
    models/imputer_final.pkl
    models/feature_columns_final.json
    models/decision_threshold.json         - threshold the extension MUST use
    reports/final_metrics.txt
    reports/final_confusion_matrix.png
    reports/final_feature_importances.png
    reports/final_shap_summary.png
    reports/final_cross_validation.txt
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
from sklearn.model_selection import (
    train_test_split, cross_val_predict, StratifiedKFold
)
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

# Garbage sentinel values to treat as missing (NOT real measurements)
SENTINELS  = [2147483647, -2147483648]
# Note: -1 is used as a legitimate "not applicable" in some columns
# (e.g. tls_days_remaining when no TLS), so we do NOT blanket-null -1.
# Only the int32 overflow sentinels are unambiguously garbage.

DECISION_THRESHOLD = 0.5   # P(phishing) >= this  -> flag as phishing
RS = 42

RF_PARAMS = {
    "n_estimators":     200,   # conventional strong default (see correction #2)
    "max_depth":        None,
    "min_samples_leaf": 2,
    "class_weight":     "balanced",
    "random_state":     RS,
    "n_jobs":           -1,
}


def clean_features(X):
    """Convert garbage sentinels to NaN so the imputer handles them properly."""
    X = X.copy()
    n_sent = 0
    for s in SENTINELS:
        hit = (X == s)
        n_sent += int(hit.sum().sum())
        X[hit] = np.nan
    return X, n_sent


def main():
    t0 = time.time()
    print("=" * 68)
    print("PhishLens - FINAL Model (corrected, leakage-free, deploy-honest)")
    print("=" * 68)

    # ── 1. Load fresh data ────────────────────────────────────────────────
    print("\n[1/8] Loading fresh feature data ...")
    df = pd.read_csv(FRESH_FILE)
    print(f"  Rows: {len(df):,}")
    print(f"  label=1 (legitimate): {(df['label']==1).sum():,}")
    print(f"  label=0 (phishing)  : {(df['label']==0).sum():,}")

    # ── 2. Prepare + clean features ───────────────────────────────────────
    print("\n[2/8] Preparing & cleaning features ...")
    y = df["label"].astype(int)
    X = df.drop(columns=[c for c in DROP_COLS + ["label"] if c in df.columns])
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.drop(columns=X.columns[X.isna().all()].tolist())  # drop empty cols
    X, n_sent = clean_features(X)
    feature_cols = list(X.columns)
    print(f"  Usable features         : {len(feature_cols)}")
    print(f"  Sentinel values -> NaN  : {n_sent}")
    print(f"  Total missing after     : {X.isna().sum().sum():,} "
          f"({X.isna().mean().mean()*100:.2f}%)")

    # ── 3. Cross-validated headline metrics (threshold-consistent) ────────
    print("\n[3/8] 5-fold cross-validation (headline metrics) ...")
    imp_cv = SimpleImputer(strategy="median")
    X_cv   = imp_cv.fit_transform(X).astype(np.float32)
    clf_cv = RandomForestClassifier(**RF_PARAMS)
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    # Out-of-fold probabilities, then apply the SAME threshold the extension uses
    cv_proba = cross_val_predict(clf_cv, X_cv, y, cv=skf,
                                 method="predict_proba", n_jobs=-1)[:, 1]
    cv_pred  = (cv_proba >= (1 - DECISION_THRESHOLD)).astype(int)  # P(legit) form
    # NOTE: model's positive class is label=1 (legit). P(phishing)=1-P(legit).
    # Flag phishing when P(phishing) >= DECISION_THRESHOLD  <=> P(legit) <= 1-thr.
    # cv_pred here is in label space (1=legit) for metric computation:
    cv_pred_label = (cv_proba >= (1 - DECISION_THRESHOLD)).astype(int)

    cv_acc = accuracy_score(y, cv_pred_label)
    cv_f1  = f1_score(y, cv_pred_label)
    cv_auc = roc_auc_score(y, cv_proba)
    print(f"  CV Accuracy (thr={DECISION_THRESHOLD}): {cv_acc:.4f}")
    print(f"  CV F1       (thr={DECISION_THRESHOLD}): {cv_f1:.4f}")
    print(f"  CV ROC-AUC                  : {cv_auc:.4f}")

    with open(REPORTS / "final_cross_validation.txt", "w") as f:
        f.write("PhishLens FINAL Model - 5-Fold Cross-Validation\n")
        f.write("=" * 50 + "\n")
        f.write(f"Decision threshold (P_phishing>=): {DECISION_THRESHOLD}\n")
        f.write(f"CV Accuracy: {cv_acc:.4f}\n")
        f.write(f"CV F1      : {cv_f1:.4f}\n")
        f.write(f"CV ROC-AUC : {cv_auc:.4f}\n")
    print("  Saved -> reports/final_cross_validation.txt")

    # ── 4. Final split + impute ───────────────────────────────────────────
    print("\n[4/8] Held-out split & imputation (confirmation) ...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    X_tr_i  = imputer.fit_transform(X_tr).astype(np.float32)
    X_te_i  = imputer.transform(X_te).astype(np.float32)
    print(f"  Train: {len(X_tr_i):,}  Test: {len(X_te_i):,}")

    with open(MODELS / "imputer_final.pkl", "wb") as f:
        pickle.dump(imputer, f)
    with open(MODELS / "feature_columns_final.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS / "decision_threshold.json", "w") as f:
        json.dump({
            "decision_threshold_phishing": DECISION_THRESHOLD,
            "positive_class": "label=1 is legitimate",
            "rule": "flag phishing when P(phishing) >= decision_threshold_phishing",
            "note": "Extension MUST use the probability output + this threshold "
                    "to match the reference model exactly."
        }, f, indent=2)
    print("  Saved imputer_final.pkl, feature_columns_final.json, decision_threshold.json")

    # ── 5. Train final model ──────────────────────────────────────────────
    print("\n[5/8] Training final Random Forest (200 trees) ...")
    clf = RandomForestClassifier(**RF_PARAMS).fit(X_tr_i, y_tr)
    with open(MODELS / "phishlens_final.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("  Saved phishlens_final.pkl")

    # ── 6. Evaluate at the deployed threshold ─────────────────────────────
    print("\n[6/8] Evaluating (at deployed threshold) ...")
    proba_legit = clf.predict_proba(X_te_i)[:, 1]
    y_pred = (proba_legit >= (1 - DECISION_THRESHOLD)).astype(int)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, proba_legit)
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

    with open(REPORTS / "final_metrics.txt", "w") as f:
        f.write("PhishLens FINAL Model - Held-out Evaluation\n")
        f.write("=" * 50 + "\n")
        f.write(f"Decision threshold (P_phishing>=): {DECISION_THRESHOLD}\n\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n")
        f.write(f"ROC-AUC   : {auc:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n{cr}\n")
        f.write(f"\nCross-validated (5-fold) headline:\n")
        f.write(f"  CV Accuracy: {cv_acc:.4f}\n  CV F1: {cv_f1:.4f}\n")
    print("  Saved -> reports/final_metrics.txt")

    # Plots
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=["Phishing","Legitimate"],
           yticklabels=["Phishing","Legitimate"],
           xlabel="Predicted", ylabel="True",
           title="PhishLens FINAL - Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS / "final_confusion_matrix.png", dpi=150)
    plt.close()

    importances = pd.Series(clf.feature_importances_, index=feature_cols)
    top = importances.nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(8,6))
    top.plot(kind="barh", ax=ax, color="seagreen")
    ax.set_title("PhishLens FINAL - Top 20 Features")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS / "final_feature_importances.png", dpi=150)
    plt.close()
    print("  Saved confusion matrix + feature importance plots")

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
        plt.savefig(REPORTS / "final_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved SHAP summary plot")
    except Exception as e:
        print(f"  SHAP skipped ({str(e)[:40]})")

    # ── 7. Export ONNX + verify PROBABILITY parity (the correct measure) ──
    print("\n[7/8] Exporting ONNX + verifying PROBABILITY parity ...")
    it   = [("float_input", FloatTensorType([None, len(feature_cols)]))]
    opts = {id(clf): {"zipmap": False}}
    onx  = convert_sklearn(clf, initial_types=it, target_opset=15, options=opts)
    onnx_path = MODELS / "phishlens_final.onnx"
    onnx_path.write_bytes(onx.SerializeToString())
    size_kb = onnx_path.stat().st_size / 1024

    import onnxruntime as rt
    sess = rt.InferenceSession(str(onnx_path))
    inp  = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {inp: X_te_i})
    onnx_proba = onnx_out[1][:, 1]          # P(legit) from ONNX
    sk_proba   = clf.predict_proba(X_te_i)[:, 1]

    max_prob_diff = np.abs(onnx_proba - sk_proba).max()
    # Label parity AFTER applying the same threshold to both
    onnx_label = (onnx_proba >= (1 - DECISION_THRESHOLD)).astype(int)
    sk_label   = (sk_proba   >= (1 - DECISION_THRESHOLD)).astype(int)
    label_parity = np.mean(onnx_label == sk_label) * 100

    print(f"  Saved phishlens_final.onnx ({size_kb:.0f} KB)")
    print(f"  Max |P_onnx - P_sklearn|     : {max_prob_diff:.8f}")
    print(f"  Label parity @ same threshold: {label_parity:.2f}%")
    if max_prob_diff < 1e-5:
        print("  -> Probabilities identical. ONNX is a faithful copy.")
    if label_parity >= 99.9:
        print("  -> Full label parity once a consistent threshold is applied.")

    # ── 8. Sanity check ───────────────────────────────────────────────────
    print("\n[8/8] Sanity check on real URLs ...")
    phish = df[df["label"]==0].head(3)
    legit = df[df["label"]==1].head(3)
    sample = pd.concat([phish, legit], ignore_index=True)
    Xs = sample.drop(columns=[c for c in DROP_COLS + ["label"] if c in sample.columns])
    Xs = Xs.select_dtypes(include=[np.number]).copy()
    Xs, _ = clean_features(Xs)
    for c in feature_cols:
        if c not in Xs.columns:
            Xs[c] = np.nan
    Xs = Xs[feature_cols]
    Xs_i = imputer.transform(Xs).astype(np.float32)
    p_legit = clf.predict_proba(Xs_i)[:, 1]
    p_phish = 1 - p_legit
    pred = (p_phish >= DECISION_THRESHOLD).astype(int)  # 1 = flagged phishing
    print(f"\n  {'URL':<48} {'True':>5} {'P(phish)':>9} {'Flag':>5}")
    print("  " + "-"*70)
    correct = 0
    for url, t, pp in zip(sample["url"], sample["label"], p_phish):
        flagged = int(pp >= DECISION_THRESHOLD)
        # true phishing is label==0; flagged==1 means we call it phishing
        is_right = (flagged == 1 and t == 0) or (flagged == 0 and t == 1)
        if is_right: correct += 1
        mark = "OK   " if is_right else "WRONG"
        print(f"  {mark} {str(url)[:42]:<42} {t:>5} {pp:>9.4f} {flagged:>5}")
    print(f"\n  {correct}/{len(sample)} correct")

    elapsed = time.time() - t0
    print("\n" + "=" * 68)
    print("FINAL MODEL COMPLETE")
    print("=" * 68)
    print(f"  CV Accuracy    : {cv_acc:.4f}")
    print(f"  CV F1          : {cv_f1:.4f}")
    print(f"  Test Accuracy  : {acc:.4f}")
    print(f"  Test F1        : {f1:.4f}")
    print(f"  Max prob diff  : {max_prob_diff:.8f}  (ONNX fidelity)")
    print(f"  Label parity   : {label_parity:.2f}%  (at deployed threshold)")
    print(f"  Trees          : {RF_PARAMS['n_estimators']}")
    print(f"  Features       : {len(feature_cols)}")
    print(f"  Threshold      : {DECISION_THRESHOLD}")
    print(f"  Time           : {elapsed/60:.1f} min")
    print("=" * 68)


if __name__ == "__main__":
    main()
