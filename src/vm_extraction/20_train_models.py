"""
PhishLens - Train Models A and B (Random Forest)
==================================================================
Model A : URL/DNS/TLS features, all 11,269 rows
Model A-: leakage check - Model A WITHOUT liveness features
          (dns_resolved, http_status, response_time_ms,
          content_length). Dead-at-extraction phishing URLs could
          let a model learn "dead = phishing" (dataset-ageing
          leakage). If A- performs close to A, no material leakage.
Model B : URL/DNS/TLS + rendered-page features, rendered-and-alive
          subset (in_render_sample==1 & render_error==0)

Methodological safeguards:
  - Domain-grouped train/test split (GroupShuffleSplit on registered
    domain): no domain appears in both train and test, preventing
    near-duplicate URL leakage.
  - Stratification verified after grouping; class balance reported.
  - tld encoded by frequency (top-20 kept, rest 'other') - no
    high-cardinality one-hot explosion.
  - Missing numerics filled with -1 sentinel (RF-safe) + report of
    missingness so nothing is silently imputed.

Requirements:
    pip install scikit-learn joblib

Usage:
    python3 20_train_models.py

Outputs:
    models/model_A.joblib, models/model_A_minus.joblib,
    models/model_B.joblib
    reports/training_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
MODELS    = BASE_DIR / "models"
REPORTS   = BASE_DIR / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

TABLE = PROCESSED / "training_table.csv"
REPORT_OUT = REPORTS / "training_report.txt"

RENDER_FEATURES = [
    "num_forms", "has_password_field", "num_password_fields",
    "hidden_element_count", "iframe_count", "external_resource_ratio",
    "favicon_domain_mismatch",
]
LIVENESS_FEATURES = [
    "dns_resolved", "http_status", "response_time_ms", "content_length",
]
NON_FEATURES = ["url", "label", "source", "render_error",
                "in_render_sample", "tld"]

SEED = 42
lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def registered_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower().split(":")[0]
        parts = netloc.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else netloc
    except Exception:
        return url


def prepare(df: pd.DataFrame):
    df = df.copy()
    top_tlds = df["tld"].value_counts().head(20).index
    df["tld_enc"] = df["tld"].where(df["tld"].isin(top_tlds), "other")
    tld_dummies = pd.get_dummies(df["tld_enc"], prefix="tld")
    feats = [c for c in df.columns
             if c not in NON_FEATURES + ["tld_enc"] + list(tld_dummies.columns)]
    X = pd.concat([df[feats], tld_dummies], axis=1)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    miss = X.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        log("Missing values filled with -1 sentinel:")
        log(miss.to_string())
    X = X.fillna(-1)
    return X, df["label"].astype(int), df["url"].map(registered_domain)


def train_eval(name, X, y, groups, drop_cols=None):
    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

    log(f"\n{'='*66}\n{name}\n{'='*66}")
    log(f"Features: {X.shape[1]} | Train: {len(Xtr):,} | Test: {len(Xte):,}")
    log(f"Train balance: {ytr.value_counts().to_dict()} | "
        f"Test balance: {yte.value_counts().to_dict()}")

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    log(f"Accuracy : {accuracy_score(yte, pred):.4f}")
    log(f"Precision: {precision_score(yte, pred):.4f}")
    log(f"Recall   : {recall_score(yte, pred):.4f}")
    log(f"F1       : {f1_score(yte, pred):.4f}")
    log(f"AUC-ROC  : {roc_auc_score(yte, proba):.4f}")
    log(f"FPR      : {fp/(fp+tn):.4f}   FNR: {fn/(fn+tp):.4f}")
    log(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

    imp = (pd.Series(clf.feature_importances_, index=X.columns)
             .sort_values(ascending=False).head(15))
    log("\nTop 15 feature importances:")
    log(imp.round(4).to_string())
    return clf


def main():
    log("=" * 66)
    log("PhishLens - Model Training (Random Forest)")
    log("=" * 66)

    df = pd.read_csv(TABLE)
    log(f"Training table: {df.shape}")

    # ── Model A: all rows, URL/DNS/TLS features ──
    dfA = df.drop(columns=RENDER_FEATURES)
    XA, yA, gA = prepare(dfA)
    mA = train_eval("MODEL A - URL/DNS/TLS features (all rows)", XA, yA, gA)
    joblib.dump(mA, MODELS / "model_A.joblib")

    # ── Model A-minus: leakage check ──
    mAm = train_eval("MODEL A-MINUS - leakage check (liveness features "
                     "removed)", XA, yA, gA, drop_cols=LIVENESS_FEATURES)
    joblib.dump(mAm, MODELS / "model_A_minus.joblib")

    # ── Model B: rendered-and-alive subset, full features ──
    dfB = df[(df["in_render_sample"] == 1) & (df["render_error"] == 0)]
    XB, yB, gB = prepare(dfB)
    mB = train_eval("MODEL B - full feature set (rendered-alive subset)",
                    XB, yB, gB)
    joblib.dump(mB, MODELS / "model_B.joblib")

    log(f"\nModels saved to {MODELS}/")
    REPORT_OUT.write_text("\n".join(lines))
    log(f"Report saved to {REPORT_OUT}")
    log("\nINTERPRETATION GUIDE:")
    log("- If Model A-minus accuracy is within ~2% of Model A, liveness")
    log("  leakage is not material - cite this in Chapter 4.")
    log("- If Model A-minus drops sharply, the model was exploiting dead")
    log("  infrastructure signals - Chapter 4 must use A-minus as primary.")
    log("- Compare Model B vs Model A on the same subset for the rendered-")
    log("  features ablation claim.")


if __name__ == "__main__":
    main()
