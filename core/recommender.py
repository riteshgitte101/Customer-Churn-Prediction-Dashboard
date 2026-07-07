"""
core/recommender.py
Maps a customer's churn probability and top SHAP drivers to
actionable, personalised retention recommendations.
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Recommendation rule library
# Each key maps to a feature name substring (case-insensitive match).
# Two tiers: "high" (≥0.65) and "medium" (0.35–0.65).
# ──────────────────────────────────────────────
_RULES: dict[str, dict[str, str]] = {
    "Contract": {
        "high": (
            "🔒 Lock in loyalty — offer a 12–24 month contract with a "
            "10–15 % discount off the current monthly rate. Include a "
            "'price-lock' guarantee to reduce switching anxiety."
        ),
        "medium": (
            "📋 Present a 12-month contract option bundled with one "
            "premium add-on (e.g., TechSupport or Online Security) at "
            "no extra cost for the first three months."
        ),
    },
    "tenure": {
        "high": (
            "👋 New customer at risk — assign a dedicated onboarding "
            "specialist. Schedule a 30-day check-in call to surface "
            "pain points before they escalate."
        ),
        "medium": (
            "📞 Schedule a proactive customer-success call. Celebrate "
            "milestones (e.g., '1 year with us') with a small loyalty "
            "reward."
        ),
    },
    "MonthlyCharges": {
        "high": (
            "💰 High bill risk — conduct an immediate plan audit. "
            "Propose a right-sized bundle or introduce a loyalty pricing "
            "tier (≥10 % savings) to remove the cost objection."
        ),
        "medium": (
            "🧾 Offer a personalised billing review to ensure the "
            "customer is on the most competitive plan for their usage."
        ),
    },
    "InternetService": {
        "high": (
            "🌐 Service quality concern — upgrade internet tier or "
            "offer a 30-day Fiber speed-boost trial at no cost. "
            "Pair with a service-level guarantee."
        ),
        "medium": (
            "📡 Provide a free service quality assessment and share "
            "an improvement roadmap for the customer's area."
        ),
    },
    "TechSupport": {
        "high": (
            "🛠️ Unresolved tech friction — enrol the customer in a "
            "complimentary TechSupport plan for 3 months and assign a "
            "named support engineer."
        ),
        "medium": (
            "💬 Proactively offer a tech-support consultation to "
            "resolve any open or latent issues before they trigger churn."
        ),
    },
    "OnlineSecurity": {
        "high": (
            "🔐 Security gap — provide a free 3-month Online Security "
            "subscription. Reinforce perceived value with an email "
            "highlighting protection milestones."
        ),
        "medium": (
            "🛡️ Send a security-awareness outreach with an offer to "
            "trial Online Security for one month at no charge."
        ),
    },
    "PaymentMethod": {
        "high": (
            "💳 Friction in payment flow — incentivise automatic payment "
            "setup with a one-time $10 bill credit. Reduce failed-payment "
            "churn risk proactively."
        ),
        "medium": (
            "📨 Send a payment-simplification guide and a small incentive "
            "(e.g., waived convenience fee) for switching to auto-pay."
        ),
    },
    "PaperlessBilling": {
        "high": (
            "📧 Enable digital engagement — offer a $5/month discount "
            "for paperless billing activation. Reduces billing confusion "
            "and increases customer touchpoints."
        ),
        "medium": (
            "📬 Send a paperless billing enrolment prompt highlighting "
            "the environmental benefit and instant access to statements."
        ),
    },
    "SeniorCitizen": {
        "high": (
            "🤝 Senior customer segment — offer a dedicated senior "
            "support line with simplified billing and a loyalty price cap. "
            "Consider an annual in-home service check."
        ),
        "medium": (
            "📖 Provide simplified account management resources and "
            "ensure the customer is aware of all senior benefit programs."
        ),
    },
    "MultipleLines": {
        "high": (
            "📱 Multi-line risk — offer a family/group plan discount "
            "to reduce per-line cost and increase household stickiness."
        ),
        "medium": (
            "📲 Highlight multi-line bundle savings and upsell to a "
            "shared-data family plan."
        ),
    },
}

_DEFAULTS = [
    "🎯 Schedule a personalised retention call with a senior account manager within 48 hours.",
    "🎁 Offer a loyalty reward — a free month of service or a $25 account credit.",
    "📊 Send a tailored impact report showing all the value the customer has received to date.",
    "⭐ Invite the customer to an exclusive loyalty programme with early access to new features.",
]


# ──────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────
def generate_recommendations(
    shap_row: np.ndarray,
    feature_names: list,
    churn_proba: float,
    top_n: int = 5,
) -> dict:
    """
    Generate personalised retention recommendations for one customer.

    Parameters
    ----------
    shap_row       : SHAP values for this customer (1-D, length = n_features)
    feature_names  : list of feature names matching shap_row
    churn_proba    : predicted churn probability [0, 1]
    top_n          : number of recommendations to return

    Returns
    -------
    dict with keys:
        risk_level        : "High" | "Medium" | "Low"
        churn_probability : float
        top_factors       : list of top feature names driving churn
        recommendations   : list of recommendation strings
    """
    # Risk tier
    if churn_proba >= 0.65:
        risk_level = "High"
    elif churn_proba >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if risk_level == "Low":
        return {
            "risk_level": risk_level,
            "churn_probability": churn_proba,
            "top_factors": [],
            "recommendations": [
                "✅ Customer appears healthy — continue standard engagement.",
                "📈 Consider an upsell opportunity (add-ons, upgrades) "
                "given high satisfaction probability.",
            ],
        }

    tier_key = risk_level.lower()  # "high" | "medium"

    # Rank features by absolute SHAP value
    importance = pd.Series(np.abs(shap_row), index=feature_names).sort_values(ascending=False)
    top_features = importance.head(10).index.tolist()

    # Match features to rules
    recs: list[str] = []
    matched_keys: set[str] = set()

    for feat in top_features:
        for rule_key, rule_tiers in _RULES.items():
            if rule_key.lower() in feat.lower() and rule_key not in matched_keys:
                rec = rule_tiers.get(tier_key, "")
                if rec:
                    recs.append(rec)
                    matched_keys.add(rule_key)
                break
        if len(recs) >= top_n:
            break

    # Pad with default recommendations if needed
    default_idx = 0
    while len(recs) < top_n:
        recs.append(_DEFAULTS[default_idx % len(_DEFAULTS)])
        default_idx += 1

    return {
        "risk_level": risk_level,
        "churn_probability": churn_proba,
        "top_factors": top_features[:5],
        "recommendations": recs[:top_n],
    }


# ──────────────────────────────────────────────
# Batch helper
# ──────────────────────────────────────────────
def batch_recommendations(
    shap_values: np.ndarray,
    feature_names: list,
    probas: np.ndarray,
) -> list[dict]:
    """Generate recommendations for every customer in a batch."""
    return [
        generate_recommendations(shap_values[i], feature_names, float(probas[i]))
        for i in range(len(probas))
    ]
