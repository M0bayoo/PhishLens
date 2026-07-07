"""
PhishLens - Phase 1b: Trained Classifier on MiniLM Embeddings
================================================================
Embeds website text with MiniLM, trains a classifier head (Logistic
Regression) on the embeddings, evaluates it properly (cross-validation
+ held-out test), and exports the classifier to ONNX.

This is the TRAINED-CLASSIFIER alternative to the zero-shot similarity
approach in src/semantic/ - kept fully separate for honest comparison.

Pipeline:
  1. Load website_texts.csv (from 03_filter_extract_website_text.py)
  2. Embed all texts with MiniLM (all-MiniLM-L6-v2, 384-dim)
  3. 5-fold cross-validation (robust headline metrics)
  4. Held-out train/test split, train final classifier
  5. Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  6. Export classifier to ONNX (zipmap=False) + verify parity
  7. Qualitative check: most-confident-correct / most-confident-wrong examples

Requirements:
    pip install sentence-transformers scikit-learn skl2onnx onnxruntime

Usage:
    python src/semantic_trained_experiment/04_train_classifier.py

Outputs:
    models_experiment/embeddings_cache.npy       - cached MiniLM embeddings
    models_experiment/classifier.pkl             - trained sklearn classifier
    models_experiment/classifier.onnx            - ONNX export
    reports_experiment/classifier_metrics.txt
    reports_experiment/classifier_confusion_matrix.png
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

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
MODELS    = BASE_DIR / "models_experiment"
REPORTS   = BASE_DIR / "reports_experiment"
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

INPUT_FILE     = PROCESSED / "website_texts.csv"
EMBEDDING_CACHE = MODELS / "embeddings_cache.npy"
MODEL_NAME     = "all-MiniLM-L6-v2"
RS             = 42
MAX_CHARS      = 2000   # truncate very long page text before embedding


def main():
    t0 = time.time()
    print("=" * 66)
    print("PhishLens - Trained Classifier on MiniLM Embeddings (experiment)")
    print("=" * 66)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/7] Loading website text data ...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["visible_text"]).reset_index(drop=True)
    print(f"  Rows: {len(df):,}")
    print(f"  label=0 (legit)   : {(df['label']==0).sum():,}")
    print(f"  label=1 (phishing): {(df['label']==1).sum():,}")

    texts = df["visible_text"].astype(str).apply(lambda t: t[:MAX_CHARS]).tolist()
    y = df["label"].astype(int).values

    # ── 2. Embed with MiniLM (cache to disk - expensive step) ─────────────
    if EMBEDDING_CACHE.exists():
        print(f"\n[2/7] Loading cached embeddings from {EMBEDDING_CACHE.name} ...")
        X = np.load(EMBEDDING_CACHE)
        if len(X) != len(texts):
            print("  Cache size mismatch - re-embedding.")
            X = None
        else:
            print(f"  Loaded cached embeddings: {X.shape}")
    else:
        X = None

    if X is None:
        print(f"\n[2/7] Embedding {len(texts):,} texts with MiniLM ({MODEL_NAME}) ...")
        print("  (This is the slow step - may take several minutes)")
        model = SentenceTransformer(MODEL_NAME)
        X = model.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                         batch_size=64)
        np.save(EMBEDDING_CACHE, X)
        print(f"  Embeddings shape: {X.shape}. Cached -> {EMBEDDING_CACHE.name}")

    # ── 3. Cross-validation (headline metrics) ────────────────────────────
    print("\n[3/7] 5-fold cross-validation ...")
    clf_cv = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RS)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    cv_proba = cross_val_predict(clf_cv, X, y, cv=skf, method="predict_proba",
                                 n_jobs=-1)[:, 1]
    cv_pred = (cv_proba >= 0.5).astype(int)
    cv_acc = accuracy_score(y, cv_pred)
    cv_f1  = f1_score(y, cv_pred)
    cv_auc = roc_auc_score(y, cv_proba)
    print(f"  CV Accuracy: {cv_acc:.4f}")
    print(f"  CV F1      : {cv_f1:.4f}")
    print(f"  CV ROC-AUC : {cv_auc:.4f}")

    with open(REPORTS / "classifier_cross_validation.txt", "w") as f:
        f.write("Trained Classifier (MiniLM + LogisticRegression) - 5-Fold CV\n")
        f.write("=" * 55 + "\n")
        f.write(f"CV Accuracy: {cv_acc:.4f}\nCV F1: {cv_f1:.4f}\nCV ROC-AUC: {cv_auc:.4f}\n")

    # ── 4. Held-out split + train final model ─────────────────────────────
    print("\n[4/7] Held-out split & training final classifier ...")
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=RS, stratify=y
    )
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RS)
    clf.fit(X_tr, y_tr)
    with open(MODELS / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}  Saved classifier.pkl")

    # ── 5. Evaluate ─────────────────────────────────────────────────────────
    print("\n[5/7] Evaluating on held-out test set ...")
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, y_proba)
    cm   = confusion_matrix(y_te, y_pred)
    cr   = classification_report(y_te, y_pred, target_names=["Legit (0)", "Phishing (1)"])

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{cr}")

    with open(REPORTS / "classifier_metrics.txt", "w") as f:
        f.write("Trained Classifier - Held-out Evaluation\n" + "=" * 45 + "\n")
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\n")
        f.write(f"F1: {f1:.4f}\nROC-AUC: {auc:.4f}\n\nConfusion Matrix:\n{cm}\n\n{cr}\n")
        f.write(f"\nCross-validated: CV Acc={cv_acc:.4f} CV F1={cv_f1:.4f}\n")

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap=plt.cm.Purples)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Legit","Phishing"],
           yticklabels=["Legit","Phishing"], xlabel="Predicted", ylabel="True",
           title="Trained Classifier - Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS / "classifier_confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved confusion matrix -> reports_experiment/classifier_confusion_matrix.png")

    # ── 6. Export ONNX + verify ─────────────────────────────────────────────
    print("\n[6/7] Exporting to ONNX + verifying parity ...")
    options = {id(clf): {"zipmap": False}}
    it = [("float_input", FloatTensorType([None, X.shape[1]]))]
    onx = convert_sklearn(clf, initial_types=it, target_opset=15, options=options)
    onnx_path = MODELS / "classifier.onnx"
    onnx_path.write_bytes(onx.SerializeToString())

    import onnxruntime as rt
    sess = rt.InferenceSession(str(onnx_path))
    inp = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {inp: X_te.astype(np.float32)})
    onnx_pred = onnx_out[0]
    onnx_proba = onnx_out[1][:, 1]
    label_parity = np.mean(onnx_pred == y_pred) * 100
    max_prob_diff = np.abs(onnx_proba - y_proba).max()
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  Saved classifier.onnx ({size_kb:.0f} KB)")
    print(f"  Label parity  : {label_parity:.2f}%")
    print(f"  Max prob diff : {max_prob_diff:.8f}")

    # ── 7. Qualitative check: most confident correct / wrong ───────────────
    print("\n[7/7] Qualitative check: most confident predictions ...")
    test_texts = df.iloc[idx_te]["visible_text"].values
    results_df = pd.DataFrame({
        "text": test_texts, "true": y_te, "pred": y_pred, "proba_phish": 1-y_proba,
    })
    results_df["correct"] = results_df["true"] == results_df["pred"]
    results_df["confidence"] = np.abs(results_df["proba_phish"] - 0.5)

    print("\n  Most confident CORRECT predictions:")
    top_correct = results_df[results_df["correct"]].nlargest(3, "confidence")
    for _, row in top_correct.iterrows():
        label = "PHISHING" if row["true"]==1 else "LEGIT"
        print(f"    [{label}] P(phish)={row['proba_phish']:.3f} | {row['text'][:70]}")

    print("\n  Most confident WRONG predictions (worth reviewing):")
    top_wrong = results_df[~results_df["correct"]].nlargest(3, "confidence")
    if len(top_wrong) == 0:
        print("    (none - all errors were low-confidence)")
    for _, row in top_wrong.iterrows():
        label = "PHISHING" if row["true"]==1 else "LEGIT"
        print(f"    [true={label}] P(phish)={row['proba_phish']:.3f} | {row['text'][:70]}")

    elapsed = time.time() - t0
    print("\n" + "=" * 66)
    print("TRAINED CLASSIFIER EXPERIMENT COMPLETE")
    print("=" * 66)
    print(f"  CV Accuracy    : {cv_acc:.4f}")
    print(f"  CV F1          : {cv_f1:.4f}")
    print(f"  Test Accuracy  : {acc:.4f}")
    print(f"  Test F1        : {f1:.4f}")
    print(f"  ONNX parity    : {label_parity:.2f}%")
    print(f"  Training rows  : {len(df):,}")
    print(f"  Time           : {elapsed/60:.1f} min")
    print("=" * 66)


if __name__ == "__main__":
    main()
