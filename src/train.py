"""
train.py
--------
Trains two ML models for the Olympics dashboard:
  - Binary model: Predicts if an athlete wins any medal
  - Multi-class model: Predicts medal type (Gold/Silver/Bronze/None)
"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from features import build_features


# ───────────────────────────────────────────────────────────────
# Load cleaned dataset
# ───────────────────────────────────────────────────────────────
def load_data():
    print("📥 Loading cleaned CSV...")

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    CLEAN_PATH = os.path.join(BASE_DIR, "data", "cleaned", "athletes_clean.csv")

    print(f"📄 Reading file: {CLEAN_PATH}")

    df = pd.read_csv(CLEAN_PATH)
    print(f"✔ Loaded {len(df)} rows")
    return df


# ───────────────────────────────────────────────────────────────
# Train a model
# ───────────────────────────────────────────────────────────────
def train_model(X, y, numeric_features, categorical_features):
    print("⚙ Building preprocessing pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=500))
        ]
    )

    print("🚀 Training model...")
    model.fit(X, y)

    return model


# ───────────────────────────────────────────────────────────────
# Main training pipeline
# ───────────────────────────────────────────────────────────────
def main():
    print("📥 Loading data...")
    df = load_data()

    print("🧱 Building features...")
    X, y_binary, y_multi, numeric_features, categorical_features = build_features(df)

    # Split
    X_train, X_test, yb_train, yb_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    _, _, ym_train, ym_test = train_test_split(X, y_multi, test_size=0.2, random_state=42)

    # Train binary model
    print("\n🏅 Training binary medal model...")
    binary_model = train_model(X_train, yb_train, numeric_features, categorical_features)

    # Evaluate binary model
    yb_pred = binary_model.predict(X_test)
    print("\n📊 Binary Model Accuracy:", accuracy_score(yb_test, yb_pred))
    print(classification_report(yb_test, yb_pred))

    # Train multi-class model
    print("\n🥇 Training multi-class medal model...")
    multi_model = train_model(X_train, ym_train, numeric_features, categorical_features)

    # Evaluate multi-class model
    ym_pred = multi_model.predict(X_test)
    print("\n📊 Multi-Class Model Accuracy:", accuracy_score(ym_test, ym_pred))
    print(classification_report(ym_test, ym_pred))

    # Save models
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(binary_model, os.path.join(MODEL_DIR, "binary_model.pkl"))
    joblib.dump(multi_model, os.path.join(MODEL_DIR, "multi_model.pkl"))

    print("\n💾 Models saved successfully!")
    print(f"  → {MODEL_DIR}/binary_model.pkl")
    print(f"  → {MODEL_DIR}/multi_model.pkl")


# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
