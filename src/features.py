"""
features.py
-----------
Full feature engineering + preprocessing pipeline for Olympics ML project.

This module provides:
  - Target engineering (binary + multiclass)
  - Feature engineering (BMI, AgeGroup, HomeAdvantage, EventLength)
  - Data validation
  - Missing value handling
  - Numeric & categorical feature separation
  - Preprocessing pipeline (StandardScaler + OneHotEncoder)
  - Final output for model training: X, y_binary, y_multi, preprocessor
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

# ───────────────────────────────────────────────────────────────
# Logging setup
# ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────
# 1) Data validation
# ───────────────────────────────────────────────────────────────
def validate_columns(df: pd.DataFrame, required_cols: List[str]):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Column validation passed.")


# ───────────────────────────────────────────────────────────────
# 2) Target columns
# ───────────────────────────────────────────────────────────────
def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding target columns...")

    df["medal_binary"] = (df["Medal"] != "No Medal").astype(int)

    mapping = {"No Medal": 0, "Bronze": 1, "Silver": 2, "Gold": 3}
    df["medal_class"] = df["Medal"].map(mapping)

    return df


# ───────────────────────────────────────────────────────────────
# 3) Feature Engineering
# ───────────────────────────────────────────────────────────────
def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying feature engineering...")

    # BMI (if missing)
    if "BMI" not in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2

    # Age groups
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 18, 25, 30, 40, 100],
        labels=["U18", "18-25", "25-30", "30-40", "40+"]
    )

    # Home advantage
    df["HomeAdvantage"] = (df["Team"] == df["NOC"]).astype(int)

    # Event name length
    df["EventLength"] = df["Event"].astype(str).str.len()

    return df


# ─────────────────