"""
pages/02_Analytics.py — Prediction, SHAP Explanation & Retention Recommendations
Combines: input form → prediction → feature importance → SHAP waterfall → recommendations.
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

from core.data import (
    CATEGORICAL_COLS,
    load_raw_data,
    preprocess,
    get_X_y,
    split_data,
)
from core.explainer import (
    compute_shap_values,
    customer_shap_summary,
    get_explainer,
    global_feature_importance,
)
from core.model import load_model, model_exists, predict, risk_tier
from core.recommender import generate_recommendations

st.set_page_config(
    page_title="Predict & Explain · Telco Churn",
    page_icon="🎯",
    layout="wide",
)

# ── Data & model ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load():
    df_raw = load_raw_data()
    df_enc, encoders, feature_names = preprocess(df_raw)
    X, y = get_X_y(df_enc)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return df_raw, df_enc, X, y, X_train, X_test, y_train, y_test, feature_names, encoders


@st.cache_resource(show_spinner=False)
def _get_model_explainer():
    """Load model and build SHAP explainer together (cached as a resource)."""
    m = load_model()
    exp = get_explainer(m)
    return m, exp


@st.cache_data(show_spinner=False)
def _global_shap(X_sample: pd.DataFrame) -> np.ndarray:
    """Compute global SHAP values on a test sample (cached by DataFrame content)."""
    _, exp = _get_model_explainer()
    return compute_shap_values(exp, X_sample)


if not model_exists():
    st.warning("Model not trained yet. Go to the Home page first.")
    st.stop()

try:
    (
        df_raw, df_enc, X, y,
        X_train, X_test, y_train, y_test,
        feature_names, encoders,
    ) = _load()
    model, explainer = _get_model_explainer()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Pre-compute global SHAP on 500-row test sample
SAMPLE_N = min(500, len(X_test))
X_shap_sample = X_test.sample(SAMPLE_N, random_state=42)

with st.spinner("Computing SHAP values for global importance chart…"):
    global_shap_vals = _global_shap(X_shap_sample)

importance_df = global_feature_importance(global_shap_vals, feature_names)

# ── Page ──────────────────────────────────────────────────────────────────────
st.title("Prediction & Explanation")
st.caption("Enter customer attributes to predict churn probability, view SHAP explanations, and get retention recommendations.")
st.divider()

# ── 1. Global Feature Importance ─────────────────────────────────────────────
st.header("1. Feature Importance (Global SHAP)")
st.caption(
    "Mean absolute SHAP value per feature across a 500-customer test sample. "
    "Higher values indicate greater influence on the model's predictions."
)

top_n = st.slider("Number of features to display", 5, len(feature_names), 15)
df_top = importance_df.head(top_n).sort_values("importance", ascending=True)

fig_importance = px.bar(
    df_top,
    x="importance",
    y="feature",
    orientation="h",
    title=f"Top {top_n} Features by Mean |SHAP Value|",
    labels={"importance": "Mean |SHAP Value|", "feature": "Feature"},
    color="importance",
    color_continuous_scale="Blues",
)
fig_importance.update_layout(
    template="plotly_white",
    showlegend=False,
    coloraxis_showscale=False,
    height=max(300, top_n * 28),
)
st.plotly_chart(fig_importance, use_container_width=True)

st.divider()

# ── 2. Prediction Form ────────────────────────────────────────────────────────
st.header("2. Customer Churn Prediction")
st.write("Fill in the customer profile below and click **Predict**.")

with st.form("churn_form"):
    col1, col2, col3, col4 = st.columns(4)

    gender          = col1.selectbox("Gender", ["Male", "Female"])
    senior          = col1.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = col1.selectbox("Partner", ["Yes", "No"])
    dependents      = col1.selectbox("Dependents", ["Yes", "No"])

    tenure          = col2.slider("Tenure (months)", 0, 72, 12)
    phone_service   = col2.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines  = col2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_svc    = col2.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    online_security = col3.selectbox("Online Security",   ["Yes", "No", "No internet service"])
    online_backup   = col3.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
    device_protect  = col3.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support    = col3.selectbox("Tech Support",      ["Yes", "No", "No internet service"])
    streaming_tv    = col3.selectbox("Streaming TV",      ["Yes", "No", "No internet service"])
    streaming_movies= col3.selectbox("Streaming Movies",  ["Yes", "No", "No internet service"])

    contract        = col4.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless       = col4.selectbox("Paperless Billing", ["Yes", "No"])
    payment         = col4.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly         = col4.number_input("Monthly Charges ($)", 10.0, 200.0, 65.0, 0.5)
    total           = col4.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly * tenure), 1.0)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

# ── Prediction result ─────────────────────────────────────────────────────────
if submitted:
    # Build input row
    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_svc,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protect,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": "No",  # placeholder
    }

    # Encode using training encoders
    df_input = pd.DataFrame([raw_input])
    for col in CATEGORICAL_COLS:
        if col in df_input.columns:
            le = encoders[col]
            val = str(df_input[col].iloc[0])
            df_input[col] = le.transform([val])[0] if val in le.classes_ else 0

    df_input["TotalCharges"] = pd.to_numeric(df_input["TotalCharges"], errors="coerce").fillna(0)
    X_input = df_input[feature_names]

    # Predict
    proba, label = predict(model, X_input)
    p = float(proba[0])
    tier = risk_tier(p)

    # Compute SHAP for this customer
    cust_shap = compute_shap_values(explainer, X_input)
    cust_summary = customer_shap_summary(cust_shap[0], X_input.iloc[0], feature_names, top_n=12)

    # Recommendations
    rec_result = generate_recommendations(cust_shap[0], feature_names, p)

    # ── Result header ─────────────────────────────────────────────────────────
    st.divider()
    st.header("3. Prediction Result")

    res_col, detail_col = st.columns([1, 2])

    with res_col:
        churn_label = "Will Churn" if label[0] == 1 else "Will Not Churn"
        tier_color  = {"High": "red", "Medium": "orange", "Low": "green"}.get(tier, "gray")

        st.metric("Churn Probability", f"{p:.1%}")
        st.metric("Prediction", churn_label)
        st.write(f"**Risk Tier:** :{tier_color}[{tier}]")

    with detail_col:
        st.write("**Top factors for this customer:**")
        top5 = cust_summary.head(5)
        for _, row in top5.iterrows():
            direction = "increases" if row["shap_value"] > 0 else "decreases"
            st.write(
                f"- **{row['feature']}** = `{row['feature_value']:.2f}` "
                f"— {direction} churn risk (SHAP: `{row['shap_value']:+.4f}`)"
            )

    # ── SHAP Waterfall ────────────────────────────────────────────────────────
    st.divider()
    st.header("4. SHAP Explanation (Customer Level)")
    st.caption(
        "Positive SHAP value → pushes toward churn. "
        "Negative SHAP value → pushes toward retention."
    )

    cust_sorted = cust_summary.sort_values("shap_value")
    bar_colors = ["#d62728" if v > 0 else "#1f77b4" for v in cust_sorted["shap_value"]]

    fig_wf = go.Figure(
        go.Bar(
            x=cust_sorted["shap_value"],
            y=cust_sorted["feature"],
            orientation="h",
            marker_color=bar_colors,
            text=cust_sorted["shap_value"].apply(lambda v: f"{v:+.4f}"),
            textposition="outside",
        )
    )
    fig_wf.add_vline(x=0, line_color="black", line_width=1)
    fig_wf.update_layout(
        title=f"SHAP Values — Churn Probability: {p:.1%}",
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        template="plotly_white",
        height=420,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    st.divider()
    st.header("5. Retention Recommendations")

    if tier == "Low":
        st.success("This customer has a low churn risk. Standard engagement is sufficient.")
    else:
        severity = "error" if tier == "High" else "warning"
        getattr(st, severity)(
            f"**Risk Level: {tier}** — Churn probability: {p:.1%}. "
            "Recommended retention actions:"
        )

    for i, rec in enumerate(rec_result["recommendations"], 1):
        st.write(f"**{i}.** {rec}")

    st.divider()
    st.write("**Top churn drivers identified:**", ", ".join(rec_result.get("top_factors", [])))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("- **Home** — Dataset & model summary")
    st.markdown("- **EDA** — Exploratory data analysis")
    st.markdown("- **Predict** — Prediction, SHAP & recommendations")
    st.divider()
    if "last_prediction" in st.session_state:
        lp = st.session_state["last_prediction"]
        st.markdown(f"**Last prediction:** {lp['proba']:.1%} ({lp['tier']} risk)")
