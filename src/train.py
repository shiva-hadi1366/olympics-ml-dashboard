"""
train.py
--------
Trainiert zwei Machine-Learning-Modelle für das Olympics-Dashboard:
  - Binäres Modell: Vorhersage, ob ein Athlet eine Medaille gewinnt
  - Multiklassen-Modell: Vorhersage des Medaillentyps (Gold/Silber/Bronze/None)
"""

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import build_features


# ---------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------
def load_data():
    """Lädt die bereinigte CSV-Datei."""
    print("📥 Lade bereinigte CSV-Datei...")

    base_dir = os.path.dirname(os.path.dirname(__file__))
    clean_path = os.path.join(base_dir, "data", "cleaned", "athletes_clean.csv")

    print(f"📄 Datei wird gelesen: {clean_path}")
    df = pd.read_csv(clean_path)

    print(f"✔ {len(df)} Zeilen geladen")
    return df


# ---------------------------------------------------------------
# Modelltraining
# ---------------------------------------------------------------
def train_model(X, y, numeric_features, categorical_features):
    """Erstellt Pipeline und trainiert ein Modell."""
    print("⚙ Erstelle Preprocessing-Pipeline...")

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

    print("🚀 Trainiere Modell...")
    model.fit(X, y)

    return model


# ---------------------------------------------------------------
# Feature Importance speichern
# ---------------------------------------------------------------
def save_feature_importance(model, feature_names, output_path="results/feature_importance.png"):
    """Speichert die Feature Importance für ein LogisticRegression-Modell."""

    print("📊 Speichere Feature Importance...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Preprocessor extrahieren
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    # Numerische Features
    num_features = pre.transformers_[0][2]

    # Kategoriale Features → OneHotEncoder Spaltennamen
    cat_encoder = pre.transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(pre.transformers_[1][2])

    # Gesamte Featureliste
    all_features = list(num_features) + list(cat_features)

    # Koeffizienten extrahieren
    importances = clf.coef_[0]

    # DataFrame erstellen
    fi = pd.DataFrame({
        "Feature": all_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=fi.head(20), palette="viridis")
    plt.title("Feature Importance – Binäres Modell (Top 20)")
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    print(f"✔ Feature Importance gespeichert unter: {output_path}")



# ---------------------------------------------------------------
# Confusion Matrix speichern
# ---------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, output_path="results/confusion_matrix.png"):
    """Speichert die Confusion Matrix als Grafik."""
    print("📊 Speichere Confusion Matrix...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix – Binäres Modell")
    plt.xlabel("Vorhersage")
    plt.ylabel("Wahrheit")
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    print(f"✔ Confusion Matrix gespeichert unter: {output_path}")


# ---------------------------------------------------------------
# Hauptpipeline
# ---------------------------------------------------------------
def main():
    print("📥 Lade Daten...")
    df = load_data()

    print("🧱 Erstelle Features...")
    X, y_binary, y_multi, numeric_features, categorical_features = build_features(df)

    # Split
    X_train, X_test, yb_train, yb_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    _, _, ym_train, ym_test = train_test_split(X, y_multi, test_size=0.2, random_state=42)

    # Binäres Modell
    print("\n🏅 Trainiere binäres Modell...")
    binary_model = train_model(X_train, yb_train, numeric_features, categorical_features)

    yb_pred = binary_model.predict(X_test)
    print("\n📊 Genauigkeit (Binary):", accuracy_score(yb_test, yb_pred))
    print(classification_report(yb_test, yb_pred))

    # Feature Importance speichern
    save_feature_importance(binary_model, X.columns)

    # Confusion Matrix speichern
    save_confusion_matrix(yb_test, yb_pred)

    # Multiklassen-Modell
    print("\n🥇 Trainiere Multiklassen-Modell...")
    multi_model = train_model(X_train, ym_train, numeric_features, categorical_features)

    ym_pred = multi_model.predict(X_test)
    print("\n📊 Genauigkeit (Multi-Class):", accuracy_score(ym_test, ym_pred))
    print(classification_report(ym_test, ym_pred))

    # Modelle speichern
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(binary_model, os.path.join(model_dir, "binary_model.pkl"))
    joblib.dump(multi_model, os.path.join(model_dir, "multi_model.pkl"))

    print("\n💾 Modelle erfolgreich gespeichert!")
    print(f"  → {model_dir}/binary_model.pkl")
    print(f"  → {model_dir}/multi_model.pkl")


# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
