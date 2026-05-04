# 🏅 Olympics ML Dashboard

End-to-end Machine-Learning-Projekt zur Vorhersage olympischer Medaillenergebnisse – inklusive Streamlit-Dashboard zur interaktiven Exploration und Visualisierung.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-Pipeline-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
</p>

---

## 🎯 Projektüberblick

Dieses Repository enthält einen vollständigen ML-Workflow zur Vorhersage olympischer Medaillenergebnisse:

- Datenbereinigung und Vorverarbeitung
- Feature Engineering
- Modelltraining (binär und mehrklassig)
- Evaluation und Visualisierung
- Streamlit-Dashboard für eine interaktive Analyse

---

## 📊 Demo (Vorschau)

> TODO: Ersetze den Platzhalter durch ein echtes GIF oder einen Screenshot der App.

<p align="center">
  <img src="https://raw.githubusercontent.com/placeholder/demo.gif" width="650" />
</p>

---

## 📂 Projektstruktur

```text
olympics-ml-dashboard/
│
├── data/                   # Roh- und verarbeitete Datensätze
├── notebooks/              # EDA, Experimente, Modellentwicklung
├── src/
│   ├── features.py         # Feature Engineering
│   ├── train.py            # Trainings-Pipeline
│   ├── evaluate.py         # Evaluationsskripte
│   └── utils.py            # Hilfsfunktionen
│
├── dashboard/
│   └── app.py              # Streamlit-Dashboard
│
├── requirements.txt        # Abhängigkeiten
└── README.md               # Dokumentation
```

---

## 🧠 Machine-Learning-Pipeline

- ✅ Datenbereinigung
- ✅ Feature Engineering
- ✅ Binäre Klassifikation (Medaille / keine Medaille)
- ✅ Mehrklassen-Klassifikation (Gold / Silber / Bronze)
- ✅ Evaluation & Visualisierung

---

## 🚀 Ausführen (lokal)

### 1) Repository klonen

```bash
git clone https://github.com/shiva-hadi1366/olympics-ml-dashboard
cd olympics-ml-dashboard
```

### 2) Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 3) Modell trainieren und evaluieren

```bash
python src/train.py
python src/evaluate.py
```

### 4) Dashboard starten

```bash
streamlit run dashboard/app.py
```

---

## 📈 Ergebnisse (wird ergänzt)

| Metrik                | Wert |
|----------------------|------|
| Accuracy (binär)     | —    |
| Accuracy (mehrklassig) | —  |
| Wichtigste Features  | —    |

---

## 🛠 Tech-Stack

- Python
- Pandas / NumPy
- Scikit-Learn
- Streamlit
- Matplotlib / Seaborn
- Git & GitHub

---

## 🔮 Nächste Schritte

- Hyperparameter-Tuning
- Modell-Erklärbarkeit (z. B. SHAP)
- Deployment (z. B. Streamlit Cloud)
- API-Endpunkt für Vorhersagen

---

## 👤 Autor

**Mohammadhadi Shiva**  
Data-Science-Trainee — Deutschland  
GitHub: https://github.com/shiva-hadi1366
