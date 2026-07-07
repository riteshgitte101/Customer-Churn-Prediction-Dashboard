# Customer Churn Prediction Dashboard

A machine learning project that predicts whether a telecom customer is likely to churn, 
built on the IBM Telco Customer Churn dataset. Includes model training (XGBoost), 
SHAP-based explainability, and a Streamlit dashboard to explore predictions interactively.

## Why I built this

Most subscription businesses lose revenue not from lacking customers, but from failing 
to catch the ones about to leave. This project trains a model to flag at-risk customers 
and explains why, rather than just outputting a probability.

## Setup

\```bash
pip install -r requirements.txt
streamlit run app.py
\```

Open `http://localhost:8501`. The model trains automatically on first launch (~10-30 seconds).

## Structure

\```
├── app.py                    # entry point, kicks off training on first run
├── pages/
│   ├── 01_Overview.py        # churn stats, revenue at risk
│   ├── 02_Analytics.py       # customer segments, service usage
│   ├── 03_Predictions.py     # predict for one customer or batch CSV
│   ├── 04_Explainability.py  # SHAP plots
│   └── 05_Recommendations.py # rule-based retention suggestions
├── core/
│   ├── data.py
│   ├── model.py
│   ├── explainer.py
│   └── recommender.py
├── data/
├── models/
└── requirements.txt
\```

## Stack

Streamlit, XGBoost, scikit-learn, SHAP, Plotly, pandas/numpy.

## Model

XGBoost classifier with early stopping on validation log-loss, `scale_pos_weight` for 
class imbalance. Evaluated on accuracy, precision, recall, F1, AUC-ROC — precision/recall 
matter more here since churners are the minority class.

## Dataset

IBM Telco Customer Churn dataset, 7,043 customers, 20 features — 
[Kaggle link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).