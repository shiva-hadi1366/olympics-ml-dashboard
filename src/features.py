"""
features.py
-----------
Feature engineering + target creation for Olympics ML project.

This module:
  - Adds binary and multiclass target columns
  - Selects ML-ready feature columns
  - Splits numeric and categorical features
  - Returns X, y_binary, y_multi for model training
"""

import pandas as pd
from typing import Tuple


# ───────────────────────────────────────────────────────────────
# 1) Add target columns
# ───────────────────────────────────────────────────────────────
def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds target columns for ML models based on cleaned Olympics dataset.

    Outputs:
    - medal_binary : 1 if athlete won any medal, else 0
    - medal_class  : 0 = No Medal, 1 = Bronze, 2 = Silver, 3 = Gold
    """

    # Binary target
    df["medal_binary"] = (df["Medal"] != "No Medal").astype(int)

    # Multi-class target
    mapping = {
        "No Medal": 0,
        "Bronze": 1,
        "Silver": 2,
        "Gold": 3
    }
    df["medal_class"] = df["Medal"].map(mapping)

    return df


# ───────────────────────────────────────────────────────────────
# 2) Select feature columns
# ───────────────────────────────────────────────────────────────
def select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    """
    Selects ML-ready feature columns and returns:
      - X (feature dataframe)
      - numeric_features
      - categorical_features
    """

    numeric_features = ["Age", "Height", "Weight", "BMI", "Year"]
    categorical_features = [
        "Sex", "Team", "NOC", "Region",
        "Season", "City", "Sport", "Event"
    ]

    feature_cols = numeric_features + categorical_features

    X = df[feature_cols]

    return X, numeric_features, categorical_features


# ───────────────────────────────────────────────────────────────
# 3) Main function used by train.py
# ───────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    """
    Full feature engineering pipeline:
      1. Add target columns
      2. Select feature columns
      3. Return X, y_binary, y_multi, numeric, categorical
    """

    df = add_target_columns(df)

    X, numeric, categorical = select_feature_columns(df)

    y_binary = df["medal_binary"]
    y_multi = df["medal_class"]

    return X, y_binary, y_multi, numeric, categorical


# ───────────────────────────────────────────────────────────────
# Debug run
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("features.py test run")
    sample = pd.DataFrame({
        "Medal": ["Gold", "No Medal", "Silver"],
        "Age": [23, 30, 27],
        "Height": [180, 175, 190],
        "Weight": [75, 80, 85],
        "BMI": [23.1, 26.1, 23.5],
        "Year": [2012, 2016, 2020],
        "Sex": ["M", "F", "M"],
        "Team": ["USA", "GER", "FRA"],
        "NOC": ["USA", "GER", "FRA"],
        "Region": ["Americas", "Europe", "Europe"],
        "Season": ["Summer", "Summer", "Summer"],
        "City": ["London", "Rio", "Tokyo"],
        "Sport": ["Swimming", "Athletics", "Judo"],
        "Event": ["100m", "Marathon", "Heavyweight"]
    })

    X, yb, ym, num, cat = build_features(sample)
    print(X.head())
    print("Binary:", yb.tolist())
    print("Multi:", ym.tolist())
    print("Numeric features:", num)
    print("Categorical features:", cat)
    