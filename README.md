# Credit card fraud detection (ML + Deep Learning)

Detect fraudulent credit card transactions on an imbalanced dataset using both classic ML models and a deep autoencoder.

---

## Dataset

- **Credit Card Fraud Detection (Kaggle)** – `creditcard.csv`  
  284,807 transactions, 492 frauds (highly imbalanced), PCA-transformed features `V1`…`V28`, plus `Time`, `Amount`, and `Class` (0 = normal, 1 = fraud).

Put the CSV in the `data/` folder once downloaded from Kaggle.

---

## What this project shows

- **EDA & imbalance analysis** – class distribution, statistics for `Amount` and `Time`, visualization of how fraud vs non-fraud differ.
- **Classic ML model** – Logistic Regression / XGBoost with strategies for imbalance (class weights / undersampling).
- **Deep Learning autoencoder** – trained on normal transactions only, using reconstruction error as an anomaly score.
- **Anomaly detection** – comparison between supervised model and autoencoder-based detection.
- **Evaluation focus** – Precision, Recall, F1 for the fraud class and ROC–AUC, plus business discussion of false positives vs false negatives.

---

## How to run

From the project root (`03_fraud_detection/`):

```bash
pip install -r requirements.txt

# 1. Train classic model (logistic regression / XGBoost)
python src/train_classic.py

# 2. Train deep autoencoder on normal transactions
python src/train_autoencoder.py

# 3. Evaluate and compare models
python src/evaluate.py
```

After running, you’ll get:

- **reports/classic_metrics.json** – metrics for the classic model.  
- **reports/autoencoder_metrics.json** – metrics for the autoencoder.  
- **reports/model_comparison.csv** – side-by-side comparison.  

---

### How to run the dashboard

From the same project root:

```bash
cd 03_fraud_detection
streamlit run src/dashboard.py
```

- **Overview** – class imbalance and basic stats for `Amount`.  
- **Models** – table comparing classic model vs autoencoder (precision/recall/F1 for fraud).  
- **Autoencoder threshold** – slider to change the reconstruction-error threshold and see how precision/recall/F1 move.

---

## Results (example – to be filled after running)

- **Classic model (e.g. XGBoost)**  
  - Precision (fraud): …  
  - Recall (fraud): …  
  - F1 (fraud): …  
  - ROC–AUC: …

- **Autoencoder**  
  - Precision (fraud): …  
  - Recall (fraud): …  
  - F1 (fraud): …  

Once you’ve run the scripts, update this section with the actual numbers.

---

## Project structure

```
03_fraud_detection/
├── data/                # creditcard.csv (not in repo)
├── models/              # saved classic model + autoencoder
├── notebooks/
│   ├── 01_eda_fraud.ipynb
│   └── 02_modeling_fraud.ipynb
├── reports/             # metrics, comparison tables, plots
├── src/
│   ├── prepare_data.py
│   ├── train_classic.py
│   ├── train_autoencoder.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

