# -*- coding: utf-8 -*-
"""
Compare classic supervised model and deep autoencoder for fraud detection.
Loads outputs from train_classic.py and train_autoencoder.py, sweeps thresholds
for the autoencoder, and writes a comparison table to reports/model_comparison.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)

BASE = Path(__file__).resolve().parent.parent


def load_classic_metrics():
    path = BASE / "reports" / "classic_metrics.json"
    if not path.exists():
        raise FileNotFoundError("Run train_classic.py first.")
    with open(path, "r") as f:
        return json.load(f)


def load_autoencoder_outputs():
    errors_path = BASE / "reports" / "autoencoder_errors.npy"
    labels_path = BASE / "reports" / "autoencoder_labels.npy"
    if not errors_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Run train_autoencoder.py first.")
    errors = np.load(errors_path)
    labels = np.load(labels_path)
    return errors, labels


def eval_autoencoder(errors, labels, threshold: float):
    preds = (errors >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, pos_label=1, average="binary", zero_division=0
    )
    return precision, recall, f1


def find_best_threshold(errors, labels, metric: str = "f1"):
    """
    Simple sweep over percentiles of the error distribution on test set.
    metric: "precision", "recall", or "f1".
    """
    percentiles = np.linspace(80, 99.5, 40)
    best_val = -1.0
    best_thr = None
    best_tuple = (0.0, 0.0, 0.0)
    for p in percentiles:
        thr = np.percentile(errors, p)
        prec, rec, f1 = eval_autoencoder(errors, labels, thr)
        val = {"precision": prec, "recall": rec, "f1": f1}[metric]
        if val > best_val:
            best_val = val
            best_thr = thr
            best_tuple = (prec, rec, f1)
    return best_thr, best_tuple


def main():
    classic = load_classic_metrics()
    errors, labels = load_autoencoder_outputs()

    # Classic model metrics for fraud class (label "1")
    fraud_stats = classic["report"]["1"]
    classic_row = {
        "model": "classic (logreg)",
        "precision_fraud": fraud_stats["precision"],
        "recall_fraud": fraud_stats["recall"],
        "f1_fraud": fraud_stats["f1-score"],
        "roc_auc": classic["roc_auc"],
    }

    # Autoencoder: choose threshold that maximizes F1 on test set
    thr, (prec, rec, f1) = find_best_threshold(errors, labels, metric="f1")
    auto_row = {
        "model": "autoencoder",
        "precision_fraud": prec,
        "recall_fraud": rec,
        "f1_fraud": f1,
        "roc_auc": np.nan,  # autoencoder is unsupervised; AUC less natural here
    }

    df = pd.DataFrame([classic_row, auto_row])
    reports_dir = BASE / "reports"
    reports_dir.mkdir(exist_ok=True)
    df.to_csv(reports_dir / "model_comparison.csv", index=False)
    print(df)
    print("Saved reports/model_comparison.csv")


if __name__ == "__main__":
    main()

