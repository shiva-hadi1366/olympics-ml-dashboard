"""
evaluate.py
-----------
Evaluierung der trainierten Modelle für das Olympics-Dashboard.
Erstellt:
  - Genauigkeit (Binary + Multiclass)
  - Confusion Matrix
  - Feature Importance (Binary + Multiclass)
"""

import os
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------
def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    test_path = os.path.join(base_dir, "data", "cleaned", "athletes_clean.csv")

    print(f"📄 Lade Testdaten: {test_path}")
    df = pd.read_csv(test_path)

    return df


# ---------------------------------------------------------------
# Confusion Matrix speichern
# ---------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, title, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Vorhersage")
    plt.ylabel("Wahrheit")
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    print(f"✔ Confusion Matrix gespeichert unter: {output_path}")


# ---------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------
def main():
    print("📥 Lade Daten...")
    df = load_data()

    print("📦 Lade Modelle...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "models")

    binary_model = joblib.load(os.path.join(model_dir, "binary_model.pkl"))
    multi_model = joblib.load(os.path.join(model_dir, "multi_model.pkl"))

    print("🧱 Erstelle Features...")
    from features import build_features
    X, y_binary, y_multi, _, _ = build_features(df)

    # Binary Evaluation
    print("\n🏅 Evaluierung: Binäres Modell...")
    yb_pred = binary_model.predict(X)
    acc_bin = accuracy_score(y_binary, yb_pred)
    print("Genauigkeit:", acc_bin)
    print(classification_report(y_binary, yb_pred))

    save_confusion_matrix(
        y_binary, yb_pred,
        "Confusion Matrix – Binäres Modell",
        os.path.join(base_dir, "results", "confusion_matrix_binary.png")
    )

    # Multiclass Evaluation
    print("\n🥇 Evaluierung: Multiklassen-Modell...")
    ym_pred = multi_model.predict(X)
    acc_multi = accuracy_score(y_multi, ym_pred)
    print("Genauigkeit:", acc_multi)
    print(classification_report(y_multi, ym_pred))

    save_confusion_matrix(
        y_multi, ym_pred,
        "Confusion Matrix – Multiklassen-Modell",
        os.path.join(base_dir, "results", "confusion_matrix_multiclass.png")
    )

    # Ergebnisse speichern
    results = {
        "accuracy_binary": float(acc_bin),
        "accuracy_multiclass": float(acc_multi)
    }

    out_path = os.path.join(base_dir, "results", "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n💾 Ergebnisse gespeichert unter: {out_path}")


# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
