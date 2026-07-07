"""
app.py — Home Page
Telco Customer Churn Analysis
Dataset Summary · Model Performance
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.data import load_raw_data, preprocess, get_X_y, split_data
from core.model import (
    MODEL_PATH,
    evaluate_model,
    load_model,
    model_exists,
    save_model,
    train_model,
)

st.set_page_config(
    page_title="Telco Churn Analysis",
    page_icon="📊",
    layout="wide",
)

# ── Cached data & model loaders ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_data():
    df_raw = load_raw_data()
    df_enc, encoders, feature_names = preprocess(df_raw)
    X, y = get_X_y(df_enc)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return df_raw, df_enc, X, y, X_train, X_test, y_train, y_test, feature_names, encoders


@st.cache_resource(show_spinner=False)
def _get_model():
    return load_model(MODEL_PATH)


def auto_train():
    if model_exists():
        return
    with st.spinner("Training XGBoost model on the Telco dataset…"):
        _, _, _, _, X_train, X_test, y_train, y_test, _, _ = _load_data()
        model = train_model(X_train, y_train, X_test, y_test)
        save_model(model)
        st.cache_resource.clear()


# ── Load data & model ─────────────────────────────────────────────────────────
try:
    auto_train()
    (
        df_raw, df_enc, X, y,
        X_train, X_test, y_train, y_test,
        feature_names, encoders,
    ) = _load_data()
    model = _get_model()
    metrics = evaluate_model(model, X_test, y_test)
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()


# ── Page layout ───────────────────────────────────────────────────────────────
st.title("Telco Customer Churn Analysis")
st.caption("Machine Learning project · XGBoost + SHAP · IBM Telco Customer Churn Dataset")
st.divider()

# ── 1. Dataset Summary ────────────────────────────────────────────────────────
st.header("1. Dataset Summary")

n_customers = len(df_raw)
n_features = len(feature_names)
churn_rate = (df_raw["Churn"] == "Yes").mean()
n_missing = df_raw.isnull().sum().sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{n_customers:,}")
c2.metric("Number of Features", n_features)
c3.metric("Churn Rate", f"{churn_rate:.1%}")
c4.metric("Missing Values", int(n_missing))

st.write("")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sample Records")
    st.dataframe(df_raw.head(8), use_container_width=True)

with col_right:
    st.subheader("Feature Types")
    dtype_df = pd.DataFrame({
        "Feature": df_raw.columns.tolist(),
        "Type": df_raw.dtypes.astype(str).tolist(),
        "Unique Values": [df_raw[c].nunique() for c in df_raw.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, height=280)

st.divider()

# ── 2. Churn Distribution ─────────────────────────────────────────────────────
st.header("2. Churn Distribution")

churn_counts = df_raw["Churn"].value_counts().reset_index()
churn_counts.columns = ["Churn", "Count"]

col_pie, col_bar = st.columns(2)

with col_pie:
    fig_pie = px.pie(
        churn_counts,
        names="Churn",
        values="Count",
        color="Churn",
        color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
        title="Churn vs Retained Customers",
    )
    fig_pie.update_layout(template="plotly_white")
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    fig_bar = px.bar(
        churn_counts,
        x="Churn",
        y="Count",
        color="Churn",
        color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
        title="Customer Count by Churn Status",
        text="Count",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── 3. Model Performance ──────────────────────────────────────────────────────
st.header("3. Model Performance (Test Set)")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("AUC-ROC",   f"{metrics['auc_roc']:.4f}")
m2.metric("F1 Score",  f"{metrics['f1']:.4f}")
m3.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
m4.metric("Precision", f"{metrics['precision']:.4f}")
m5.metric("Recall",    f"{metrics['recall']:.4f}")

st.write("")

col_cm, col_roc = st.columns(2)

with col_cm:
    cm = metrics["confusion_matrix"]
    labels = ["No", "Yes"]
    fig_cm = ff.create_annotated_heatmap(
        z=cm[::-1],
        x=["Predicted No", "Predicted Yes"],
        y=["Actual Yes", "Actual No"],
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(
        title="Confusion Matrix",
        template="plotly_white",
        height=350,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_roc:
    fpr = metrics["roc_fpr"]
    tpr = metrics["roc_tpr"]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"AUC = {metrics['auc_roc']:.4f}",
        line=dict(color="#1f77b4", width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color="gray", dash="dash"),
    ))
    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=350,
        legend=dict(x=0.6, y=0.1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("- **Home** — Dataset & model summary")
    st.markdown("- **EDA** — Exploratory data analysis")
    st.markdown("- **Predict** — Prediction, SHAP & recommendations")
    st.divider()
    st.markdown(f"Dataset: **{n_customers:,} customers**")
    st.markdown(f"Churn rate: **{churn_rate:.1%}**")
    st.markdown(f"Model AUC-ROC: **{metrics['auc_roc']:.4f}**")
