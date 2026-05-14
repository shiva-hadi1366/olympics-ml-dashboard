# 🏅 Olympics ML Dashboard

Ein vollständiges End-to-End-Machine-Learning-Projekt zur Vorhersage olympischer Medaillenergebnisse – inklusive Datenpipeline, Modelltraining, Evaluation und einem interaktiven Streamlit-Dashboard.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-Pipeline-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
</p>

---

## 🎯 Projektüberblick

Dieses Projekt entwickelt ein Machine-Learning-Modell zur Vorhersage von olympischen Medaillenergebnissen basierend auf historischen Athletendaten.  
Es umfasst:

- Datenbereinigung & Feature Engineering  
- Binäre Klassifikation (Medaille / keine Medaille)  
- Mehrklassen-Klassifikation (Gold / Silber / Bronze / None)  
- Modelltraining & Evaluation  
- Interaktives Streamlit-Dashboard zur Visualisierung und Vorhersage  

Das Projekt eignet sich ideal für Data-Science-Portfolios und produktionsnahe ML-Demonstrationen.

---

## 📊 Demo (Vorschau)

> **Hinweis:** Hier wird später ein echtes GIF oder Screenshot des Dashboards eingefügt.

<p align="center">
  <img src="https://raw.githubusercontent.com/placeholder/demo.gif" width="650" />
</p>

---

## 📂 Projektstruktur

```text
olympics-ml-dashboard/
│
├── data/                   # Roh- und bereinigte Datensätze
├── notebooks/              # EDA, Experimente, Modelltests
├── src/
│   ├── features.py         # Feature Engineering
│   ├── train.py            # Trainingspipeline
│   ├── evaluate.py         # Evaluationsskripte
│   └── utils.py            # Hilfsfunktionen
│
├── dashboard/
│   └── app.py              # Streamlit-Dashboard
│
├── models/                 # Gespeicherte Modelle (.pkl)
├── results/                # Confusion Matrices, Feature Importance, Plots
├── requirements.txt        # Abhängigkeiten
└── README.md               # Dokumentation
