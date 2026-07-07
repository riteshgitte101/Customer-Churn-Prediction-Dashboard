"""
core/data.py
Data loading and preprocessing — IBM Telco Customer Churn dataset.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(_ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERICAL_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"
ID_COL = "customerID"


def load_raw_data() -> pd.DataFrame:
    """Load the IBM Telco Customer Churn CSV from disk."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at:\n  {DATA_PATH}\n\n"
            "Please place WA_Fn-UseC_-Telco-Customer-Churn.csv in the data/ folder."
        )
    return pd.read_csv(DATA_PATH)


def preprocess(df: pd.DataFrame):
    """
    Clean and encode the raw dataframe.

    Returns
    -------
    df_encoded   : fully numeric DataFrame (no customerID, target encoded 0/1)
    encoders     : dict of fitted LabelEncoders keyed by column name
    feature_names: list of feature column names
    """
    df = df.copy()

    # Drop ID column
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # TotalCharges has whitespace in the real dataset for some rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode binary target
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    # Label-encode all categoricals
    encoders: dict = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    feature_names = [c for c in df.columns if c != TARGET_COL]
    return df, encoders, feature_names


def get_X_y(df_encoded: pd.DataFrame):
    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]
    return X, y


def split_data(X, y, test_size: float = 0.20, random_state: int = 42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
