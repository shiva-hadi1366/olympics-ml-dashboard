"""
features.py
-----------
Feature Engineering und Zielvariablen für das Olympics ML Projekt.

Dieses Modul erstellt:
  - Zielvariablen (binär + Multiklassen)
  - Feature Engineering (BMI, AgeGroup, HomeAdvantage, EventLength)
  - Aufteilung in numerische und kategoriale Features
  - Finale Ausgabe für das Training: X, y_binary, y_multi, numeric_features, categorical_features
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Spaltenvalidierung
# ---------------------------------------------------------------
def validate_columns(df: pd.DataFrame, required_cols: List[str]):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")
    logger.info("Spaltenvalidierung erfolgreich.")


# ---------------------------------------------------------------
# Zielvariablen hinzufügen
# ---------------------------------------------------------------
def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Erstelle Zielvariablen...")

    df["medal_binary"] = (df["Medal"] != "No Medal").astype(int)

    mapping = {"No Medal": 0, "Bronze": 1, "Silver": 2, "Gold": 3}
    df["medal_class"] = df["Medal"].map(mapping)

    return df


# ---------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------
def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Wende Feature Engineering an...")

    # BMI
    if "BMI" not in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2

    # Altersgruppen
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 18, 25, 30, 40, 100],
        labels=["U18", "18-25", "25-30", "30-40", "40+"]
    )

    # Heimvorteil
    df["HomeAdvantage"] = (df["Team"] == df["NOC"]).astype(int)

    # Länge des Event-Namens
    df["EventLength"] = df["Event"].astype(str).str.len()

    return df


# ---------------------------------------------------------------
# Hauptfunktion: Build Features
# ---------------------------------------------------------------
def build_features(df: pd.DataFrame):
    """Erstellt alle Features und Zielvariablen für das Modelltraining."""

    logger.info("Validiere Spalten...")

    required = ["Age", "Height", "Weight", "Team", "NOC", "Event", "Medal"]
    validate_columns(df, required)

    logger.info("Füge Zielvariablen hinzu...")
    df = add_target_columns(df)

    logger.info("Führe Feature Engineering durch...")
    df = add_feature_engineering(df)

    # Zielvariablen
    y_binary = df["medal_binary"]
    y_multi = df["medal_class"]

    # Feature-Matrix
    feature_cols = [
        "Age", "Height", "Weight", "BMI",
        "AgeGroup", "HomeAdvantage", "EventLength",
        "Team", "NOC", "Event"
    ]

    X = df[feature_cols]

    # Numerische und kategoriale Features
    numeric_features = ["Age", "Height", "Weight", "BMI", "EventLength"]
    categorical_features = ["AgeGroup", "HomeAdvantage", "Team", "NOC", "Event"]

    logger.info("Feature Engineering abgeschlossen.")

    return X, y_binary, y_multi, numeric_features, categorical_features
