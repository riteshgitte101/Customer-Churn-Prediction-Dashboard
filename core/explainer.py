"""
core/explainer.py
SHAP TreeExplainer wrapper for XGBoost churn model.
"""

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


# ──────────────────────────────────────────────
# Explainer factory
# ──────────────────────────────────────────────
def get_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer from a fitted XGBoost model."""
    return shap.TreeExplainer(model)


# ──────────────────────────────────────────────
# SHAP value computation
# ──────────────────────────────────────────────
def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for a feature matrix.

    Returns shape (n_samples, n_features) — values for the positive (churn) class.
    """
    sv = explainer.shap_values(X)
    # For binary XGBoost, shap_values returns a single 2-D array
    if isinstance(sv, list):
        return sv[1]  # positive class
    return sv


# ──────────────────────────────────────────────
# Global feature importance
# ──────────────────────────────────────────────
def global_feature_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Mean absolute SHAP values per feature — global importance ranking.

    Returns a DataFrame sorted by importance descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame(
        {"feature": feature_names, "importance": mean_abs}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# Per-customer SHAP summary
# ──────────────────────────────────────────────
def customer_shap_summary(
    shap_row: np.ndarray,
    feature_values_row: pd.Series,
    feature_names: list,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return a sorted DataFrame of SHAP values for a single customer.
    Positive SHAP → pushes toward churn; Negative → pushes toward retention.
    """
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_row,
            "feature_value": feature_values_row.values,
        }
    )
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df
