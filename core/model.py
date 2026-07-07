"""
core/model.py
XGBoost model: training, evaluation, persistence, and inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val) -> XGBClassifier:
    """Train an XGBoost classifier with early stopping."""
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate_model(model: XGBClassifier, X_test, y_test) -> dict:
    """Return a dict of classification and business metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": cm,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "y_test": np.array(y_test),
        "y_proba": y_proba,
    }


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────
def save_model(model: XGBClassifier, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH) -> XGBClassifier:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at {path}. Run the app homepage to auto-train."
        )
    return joblib.load(path)


def model_exists(path: str = MODEL_PATH) -> bool:
    return os.path.exists(path)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def predict(model: XGBClassifier, X: pd.DataFrame):
    """
    Returns
    -------
    proba  : np.ndarray  — churn probability scores [0, 1]
    labels : np.ndarray  — binary predictions (0 = stay, 1 = churn)
    """
    proba = model.predict_proba(X)[:, 1]
    labels = (proba >= 0.50).astype(int)
    return proba, labels


def risk_tier(proba: float) -> str:
    """Assign a human-readable risk tier from a churn probability."""
    if proba >= 0.65:
        return "High"
    elif proba >= 0.35:
        return "Medium"
    else:
        return "Low"
