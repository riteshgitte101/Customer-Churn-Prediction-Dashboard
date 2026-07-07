"""
pages/01_Overview.py — Exploratory Data Analysis
Churn distributions, feature charts, correlation heatmap.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.data import load_raw_data, preprocess, get_X_y, split_data
from core.model import load_model, model_exists

st.set_page_config(
    page_title="EDA · Telco Churn",
    page_icon="📊",
    layout="wide",
)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load():
    df_raw = load_raw_data()
    df_enc, _, feature_names = preprocess(df_raw)
    return df_raw, df_enc, feature_names

if not model_exists():
    st.warning("Model not trained yet. Go to the Home page first.")
    st.stop()

try:
    df, df_enc, feature_names = _load()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Page ──────────────────────────────────────────────────────────────────────
st.title("Exploratory Data Analysis")
st.caption("IBM Telco Customer Churn Dataset")
st.divider()

# ── Churn by Contract ─────────────────────────────────────────────────────────
st.header("Churn Rate by Contract Type")

contract_churn = (
    df.groupby("Contract")["Churn"]
    .value_counts(normalize=True)
    .rename("Rate")
    .reset_index()
)
contract_churn_yes = contract_churn[contract_churn["Churn"] == "Yes"].copy()
contract_churn_yes["Rate (%)"] = (contract_churn_yes["Rate"] * 100).round(1)

fig_contract = px.bar(
    contract_churn_yes,
    x="Contract",
    y="Rate",
    text="Rate (%)",
    title="Churn Rate by Contract Type",
    labels={"Rate": "Churn Rate", "Contract": "Contract Type"},
    color="Rate",
    color_continuous_scale="Reds",
)
fig_contract.update_traces(texttemplate="%{text}%", textposition="outside")
fig_contract.update_layout(
    template="plotly_white",
    showlegend=False,
    coloraxis_showscale=False,
    yaxis_tickformat=".0%",
)
st.plotly_chart(fig_contract, use_container_width=True)

st.divider()

# ── Tenure & Monthly Charges ──────────────────────────────────────────────────
st.header("Tenure and Monthly Charges by Churn Status")

col_hist, col_box = st.columns(2)

with col_hist:
    fig_tenure = px.histogram(
        df,
        x="tenure",
        color="Churn",
        barmode="overlay",
        nbins=36,
        color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
        title="Tenure Distribution (months)",
        labels={"tenure": "Tenure (months)", "count": "Number of Customers"},
        opacity=0.75,
    )
    fig_tenure.update_layout(template="plotly_white", legend_title="Churn")
    st.plotly_chart(fig_tenure, use_container_width=True)

with col_box:
    fig_box = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
        title="Monthly Charges by Churn Status",
        labels={"MonthlyCharges": "Monthly Charges ($)", "Churn": "Churn"},
        points="outliers",
    )
    fig_box.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── Internet Service & Payment Method ────────────────────────────────────────
st.header("Churn by Service Type and Payment Method")

col_inet, col_pay = st.columns(2)

with col_inet:
    inet_counts = (
        df.groupby(["InternetService", "Churn"])
        .size()
        .reset_index(name="Count")
    )
    fig_inet = px.bar(
        inet_counts,
        x="InternetService",
        y="Count",
        color="Churn",
        barmode="group",
        color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
        title="Customers by Internet Service",
        labels={"InternetService": "Internet Service", "Count": "Number of Customers"},
    )
    fig_inet.update_layout(template="plotly_white", legend_title="Churn")
    st.plotly_chart(fig_inet, use_container_width=True)

with col_pay:
    pay_churn = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .reset_index()
    )
    pay_churn.columns = ["PaymentMethod", "ChurnRate"]
    pay_churn = pay_churn.sort_values("ChurnRate", ascending=True)

    fig_pay = px.bar(
        pay_churn,
        x="ChurnRate",
        y="PaymentMethod",
        orientation="h",
        title="Churn Rate by Payment Method",
        labels={"ChurnRate": "Churn Rate", "PaymentMethod": "Payment Method"},
        color="ChurnRate",
        color_continuous_scale="Reds",
        text=pay_churn["ChurnRate"].apply(lambda x: f"{x:.1%}"),
    )
    fig_pay.update_traces(textposition="outside")
    fig_pay.update_layout(
        template="plotly_white",
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_pay, use_container_width=True)

st.divider()

# ── Correlation Heatmap ───────────────────────────────────────────────────────
st.header("Correlation Heatmap (Encoded Features)")

st.caption(
    "Pearson correlation between encoded numeric and categorical features. "
    "Values close to +1 or −1 indicate strong linear relationships."
)

corr = df_enc.corr(numeric_only=True).round(2)

fig_heat = go.Figure(
    go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 8},
        hoverongaps=False,
    )
)
fig_heat.update_layout(
    title="Feature Correlation Matrix",
    template="plotly_white",
    height=600,
    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
    yaxis=dict(tickfont=dict(size=9)),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("- **Home** — Dataset & model summary")
    st.markdown("- **EDA** — Exploratory data analysis")
    st.markdown("- **Predict** — Prediction, SHAP & recommendations")
    st.divider()
    st.markdown(f"Dataset: **{len(df):,} customers**")
    st.markdown(f"Churn rate: **{(df['Churn'] == 'Yes').mean():.1%}**")
