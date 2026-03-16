# -*- coding: utf-8 -*-
"""
Helpers to load and prepare the credit card fraud dataset.
Expected file: data/creditcard.csv (from Kaggle).
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent.parent


def load_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = BASE / "data" / "creditcard.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. "
            "Download creditcard.csv from Kaggle and put it in the data/ folder."
        )
    df = pd.read_csv(csv_path)
    return df


def train_test_split_scaled(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split into train/test with stratification on Class.
    Scale Amount and (optionally) Time; V1..V28 are already scaled.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    for col in ["Amount", "Time"]:
        if col in X_train.columns:
            X_train[col] = scaler.fit_transform(X_train[[col]])
            X_test[col] = scaler.transform(X_test[[col]])

    return X_train, X_test, y_train, y_test


def get_feature_matrix(df: pd.DataFrame):
    """Return X, y as numpy arrays, scaling Amount/Time."""
    X_train, X_test, y_train, y_test = train_test_split_scaled(df)
    return X_train, X_test, y_train, y_test

