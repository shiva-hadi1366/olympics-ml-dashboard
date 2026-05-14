import pandas as pd
import joblib
import json
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_BINARY_PATH = os.path.join(BASE, "models", "binary_model.pkl")
MODEL_MULTI_PATH = os.path.join(BASE, "models", "multi_model.pkl")
TEST_DATA_PATH = os.path.join(BASE, "data", "processed", "test.csv")
RESULTS_DIR = os.path.join(BASE, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
CONF_MATRIX_BINARY = os.path.join(RESULTS_DIR, "confusion_matrix_binary.png")
CONF_MATRIX_MULTI = os.path.join(RESULTS_DIR, "confusion_matrix_multiclass.png")

def load_data():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop("Medal", axis=1)
    y = df["Medal"]
    return X, y

def evaluate_binary(model, X, y):
    y_binary = (y != "No Medal").astype(int)
    preds = model.predict(X)
    preds_binary = (preds != "No Medal").astype(int)

    acc = accuracy_score(y_binary, preds_binary)
    cm = confusion_matrix(y_binary, preds_binary)
    report = classification_report(y_binary, preds_binary, output_dict=True)

    return acc, cm, report

def evaluate_multiclass(model, X, y):
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    preds = model.predict(X)

    acc = accuracy_score(y_encoded, preds)
    cm = confusion_matrix(y_encoded, preds)
    report = classification_report(y_encoded, preds, target_names=le.classes_, output_dict=True)

    return acc, cm, report, le.classes_

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    print("📥 Loading test data...")
    X, y = load_data()

    print("📦 Loading models...")
    model_binary = joblib.load(MODEL_BINARY_PATH)
    model_multi = joblib.load(MODEL_MULTI_PATH)

    print("⚙️ Evaluating binary model...")
    acc_binary, cm_binary, report_binary = evaluate_binary(model_binary, X, y)

    print("⚙️ Evaluating multiclass model...")
    acc_multi, cm_multi, report_multi, labels_multi = evaluate_multiclass(model_multi, X, y)

    print("💾 Saving results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    save_confusion_matrix(cm_binary, ["No Medal", "Medal"], CONF_MATRIX_BINARY)
    save_confusion_matrix(cm_multi, labels_multi, CONF_MATRIX_MULTI)

    results = {
        "accuracy_binary": acc_binary,
        "accuracy_multiclass": acc_multi,
        "classification_report_binary": report_binary,
        "classification_report_multiclass": report_multi,
        "labels_multiclass": list(labels_multi)
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Done! Results saved in /results")

if __name__ == "__main__":
    main()
