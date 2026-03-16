# -*- coding: utf-8 -*-
"""
Streamlit dashboard for fraud detection project.
- Overview: class imbalance and basic stats for Amount.
- Model comparison: table from reports/model_comparison.csv.
- Autoencoder threshold explorer: move threshold and see precision/recall/F1.

Run from project root with:
    streamlit run src/dashboard.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from prepare_data import load_data
from evaluate import load_autoencoder_outputs, eval_autoencoder
from evaluate import load_classic_metrics


BASE = Path(__file__).resolve().parent.parent


def load_model_comparison():
    path = BASE / "reports" / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


st.title("Fraud detection dashboard")
st.write(
    "Quick glance at class imbalance, model comparison (classic vs autoencoder), "
    "and a simple threshold explorer for the autoencoder."
)

tab_overview, tab_models, tab_auto = st.tabs(
    ["Overview", "Models", "Autoencoder threshold"]
)


with tab_overview:
    st.subheader("Dataset overview")
    try:
        df = load_data()
    except FileNotFoundError:
        st.warning("Put creditcard.csv in the data/ folder to view the dashboard.")
        st.stop()

    n_total = len(df)
    counts = df["Class"].value_counts()
    n_fraud = counts.get(1, 0)
    fraud_rate = n_fraud / n_total * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total transactions", n_total)
    with col2:
        st.metric("Fraud rate (%)", f"{fraud_rate:.3f}")

    st.write("Class distribution (0 = normal, 1 = fraud):")
    chart_df = counts.rename_axis("Class").reset_index(name="count")
    st.bar_chart(chart_df.set_index("Class"))

    st.subheader("Amount by class")
    stats = df.groupby("Class")["Amount"].describe()[["mean", "50%", "max"]]
    stats = stats.rename(columns={"50%": "median"})
    st.table(stats)


with tab_models:
    st.subheader("Classic vs autoencoder")
    comp = load_model_comparison()
    if comp is None:
        st.info(
            "Run `python src/train_classic.py`, `python src/train_autoencoder.py` "
            "and `python src/evaluate.py` first to generate reports/model_comparison.csv."
        )
    else:
        st.dataframe(comp)
        st.write(
            "- `precision_fraud`: how many flagged as fraud are really fraud.\n"
            "- `recall_fraud`: how many real frauds we catch.\n"
            "- `f1_fraud`: balance between precision and recall."
        )


with tab_auto:
    st.subheader("Autoencoder threshold explorer")
    try:
        errors, labels = load_autoencoder_outputs()
    except FileNotFoundError:
        st.info("Run `python src/train_autoencoder.py` first to generate AE outputs.")
    else:
        st.write(
            "Move the threshold to trade off between precision and recall. "
            "Higher threshold → fewer alerts (higher precision, lower recall)."
        )
        min_thr = float(np.percentile(errors, 80))
        max_thr = float(np.percentile(errors, 99.9))
        thr = st.slider(
            "Reconstruction error threshold",
            min_value=min_thr,
            max_value=max_thr,
            value=float(np.percentile(errors, 95)),
            step=(max_thr - min_thr) / 100.0,
        )
        prec, rec, f1 = eval_autoencoder(errors, labels, thr)
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision (fraud)", f"{prec:.3f}")
        col2.metric("Recall (fraud)", f"{rec:.3f}")
        col3.metric("F1 (fraud)", f"{f1:.3f}")

        st.markdown("---")
        st.markdown("**Classic model reference (fixed threshold)**")
        try:
            classic = load_classic_metrics()
            stats = classic["report"]["1"]
            c_prec = stats["precision"]
            c_rec = stats["recall"]
            c_f1 = stats["f1-score"]
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Classic precision (fraud)", f"{c_prec:.3f}")
            cc2.metric("Classic recall (fraud)", f"{c_rec:.3f}")
            cc3.metric("Classic F1 (fraud)", f"{c_f1:.3f}")
            st.caption(
                "Use the slider above to see when the autoencoder gets closer to or "
                "beats these fixed classic-model metrics."
            )
        except FileNotFoundError:
            st.caption("Classic metrics not found. Run `python src/train_classic.py` first.")
