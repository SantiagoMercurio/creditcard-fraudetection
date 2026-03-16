# -*- coding: utf-8 -*-
"""
Train a classic supervised model for fraud detection on creditcard.csv.
Handles class imbalance with class_weight and saves metrics + model.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
# You could also switch to XGBoost here if desired.

from prepare_data import load_data, get_feature_matrix


BASE = Path(__file__).resolve().parent.parent


def train(save_model: bool = True):
    df = load_data()
    X_train, X_test, y_train, y_test = get_feature_matrix(df)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_proba)
    print("ROC–AUC:", round(auc, 4))

    reports_dir = BASE / "reports"
    reports_dir.mkdir(exist_ok=True)

    metrics = {
        "roc_auc": float(auc),
        "report": classification_report(
            y_test, y_pred, digits=4, output_dict=True
        ),
    }
    with open(reports_dir / "classic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved reports/classic_metrics.json")

    if save_model:
        import joblib

        models_dir = BASE / "models"
        models_dir.mkdir(exist_ok=True)
        joblib.dump(clf, models_dir / "classic_model.joblib")
        print("Saved models/classic_model.joblib")

    return clf, (X_test, y_test, y_pred, y_proba)


if __name__ == "__main__":
    train()

