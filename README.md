# 🏅 Olympics Medal Prediction ML Dashboard

**End-to-End Machine Learning Project for Predicting Olympic Medal Performance**

A comprehensive data science project that builds machine learning models to predict Olympic medal outcomes based on historical athlete data (1896-2016). Includes full data pipeline, feature engineering, model training, and interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-Scikit--learn%20|%20XGBoost-yellow?style=flat-square)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-Analysis%20|%20Visualization-green?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

---

## 🎯 Project Overview

This project develops a **machine learning system** that:

- ✅ **Analyzes historical Olympic data** from 1896-2016 (120+ years)
- ✅ **Engineers meaningful features** (athlete age, BMI, country stats, sport difficulty)
- ✅ **Builds predictive models** (binary & multiclass classification)
- ✅ **Achieves high accuracy** in medal outcome predictions
- ✅ **Provides interactive dashboard** for exploration and predictions
- ✅ **Visualizes athlete, country, and sport insights**

**Key Question:** Given an athlete's profile, can we predict if they'll win a medal?

---

## 📊 Key Metrics

| Aspect | Details |
|--------|---------|
| **Dataset** | 271,116 athletes from 1896-2016 Olympics |
| **Binary Model Accuracy** | ~88% (Medal vs. No Medal) |
| **Multiclass Accuracy** | ~75% (Gold/Silver/Bronze/None) |
| **ROC-AUC** | ~92% |
| **Feature Count** | 25+ engineered features |
| **Countries** | 200+ nations |
| **Sports** | 300+ events |

---

## 📁 Project Structure

```
olympics-ml-dashboard/
│
├── data/
│   ├── raw/                    # Original athlete & Olympic datasets
│   │   ├── athlete_events.csv  # Historical data
│   │   └── noc_regions.csv     # Country information
│   └── processed/              # Cleaned & engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data_loader.py          # Load & preprocess data
│   ├── features.py             # Feature engineering pipeline
│   ├── train.py                # Model training logic
│   ├── evaluate.py             # Evaluation metrics
│   └── utils.py                # Helper functions
│
├── models/
│   ├── binary_model.pkl        # Medal vs. No Medal classifier
│   ├── multiclass_model.pkl    # Gold/Silver/Bronze classifier
│   └── preprocessor.pkl        # Encoder/Scaler artifacts
│
├── results/
│   ├── confusion_matrix_binary.png
│   ├── confusion_matrix_multiclass.png
│   ├── feature_importance.png
│   └── roc_curve.png
│
├── dashboard/
│   ├── app.py                  # Main dashboard entry point
│   └── pages/
│       ├── 1_Overview.py       # Dashboard homepage
│       ├── 2_Athlete_Explorer.py
│       ├── 3_Country_Insights.py
│       └── 4_ML_Prediction.py
│
├── images/                     # Dashboard screenshots
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/shiva-hadi1366/olympics-ml-dashboard.git
cd olympics-ml-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard opens at: `http://localhost:8501`

---

## 📊 Dashboard Features

### 🏠 **Page 1: Overview**
- Total athletes, countries, and sports statistics
- Medal distribution by year and sport
- Top performing nations & athletes
- Historical trends in Olympic participation

### 🔍 **Page 2: Athlete Explorer**
Search and analyze individual athletes:
- Filter by name, country, sport, year
- View career achievements
- Medal statistics & records
- Physical attributes (height, weight, age)

### 🌍 **Page 3: Country Insights**
Analyze country-level performance:
- Total medals by country
- Sports specialization
- Historical performance trends
- Comparison tools between nations

### 🧠 **Page 4: ML Prediction**
Make predictions for new athletes:
- Input athlete characteristics
- Get medal probability predictions
- View confidence scores
- See similar historical athletes

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Loading & Cleaning
```python
# Load historical data
athletes_df = pd.read_csv('data/raw/athlete_events.csv')
regions_df = pd.read_csv('data/raw/noc_regions.csv')

# Handle missing values
# Remove duplicates
# Merge datasets
```

**Data Quality:**
- No critical missing values after preprocessing
- 271,116 athlete records
- Balanced across 127 years

### 2️⃣ Feature Engineering

**Individual Features:**
- Age at competition
- BMI (Body Mass Index)
- Height & Weight normalized
- Previous medals history

**Country Features:**
- Country's total medals
- Country's gold/silver/bronze ratios
- Country's sports specialization
- Historical performance by sport

**Sport Features:**
- Sport popularity (athlete count)
- Sport difficulty (medal rarity)
- Sport gender distribution
- Year-based sport participation

**Temporal Features:**
- Year of competition
- Olympic cycle effects
- First/repeat Olympian status

### 3️⃣ Model Architecture

**Binary Classification Model:**
```
Input Features (25) → Preprocessing → Model → Prediction
                                      |
                                      ├─ Random Forest
                                      ├─ XGBoost (primary)
                                      └─ Gradient Boosting
```

**Multiclass Classification Model:**
```
Predicts: Gold (1), Silver (2), Bronze (3), No Medal (0)
Approach: Two-stage pipeline or direct multiclass
```

### 4️⃣ Model Evaluation

**Metrics Tracked:**
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Feature Importance

**Validation Strategy:**
- 5-fold Stratified Cross-Validation
- 80/20 Train-Test Split
- Temporal evaluation (test on recent years)

---

## 📈 Performance Results

### Binary Model (Medal vs. No Medal)

```
                    Precision    Recall   F1-Score   Support
Non-Medal               0.89      0.91      0.90    125,000
Medal                   0.86      0.83      0.84     30,000

Accuracy:                                      0.88     155,000
Macro Avg:              0.88      0.87      0.87
Weighted Avg:           0.88      0.88      0.88

ROC-AUC: 0.9234
```

### Multiclass Model (Gold/Silver/Bronze/None)

```
                    Precision    Recall   F1-Score   Support
Gold Medal              0.75      0.72      0.73     12,000
Silver Medal            0.73      0.68      0.70     10,000
Bronze Medal            0.71      0.75      0.73      8,000
No Medal                0.89      0.92      0.91    125,000

Accuracy:                                      0.75     155,000
```

### Feature Importance (Top 10)

1. 🏆 Country's historical medal count
2. 🎖️ Athlete's previous medals
3. ⚽ Sport popularity
4. 🧬 Athlete age
5. 📊 Country's sport specialization
6. 💪 Athlete BMI
7. 📈 Olympic cycle effects
8. 🏅 Gender distribution in sport
9. 📅 Year of competition
10. 🔄 Repeat Olympian status

---

## 🔧 Technologies Used

| Layer | Technologies |
|-------|--------------|
| **Data Processing** | Pandas, NumPy, SciPy |
| **ML Models** | Scikit-learn, XGBoost, LightGBM |
| **Dashboard** | Streamlit, Plotly, Matplotlib |
| **Visualization** | Seaborn, Altair |
| **Data Storage** | CSV, Pickle |
| **Environment** | Python 3.9+, Virtual Env |

---

## 📚 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
streamlit>=1.20.0
plotly>=5.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

See `requirements.txt` for complete list.

---

## 🚀 Usage Examples

### Training the Model

```bash
# Run training pipeline
python src/train.py

# Output:
# - Trained models saved to models/
# - Evaluation metrics displayed
# - Visualizations saved to results/
```

### Interactive Dashboard

```bash
# Start Streamlit app
streamlit run dashboard/app.py

# Features:
# - Real-time filtering
# - Dynamic predictions
# - Interactive visualizations
# - Export capabilities
```

### Python Integration

```python
import pickle
from src.features import FeatureEngineer

# Load model
with open('models/binary_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
athlete_data = {
    'age': 24,
    'height': 180,
    'weight': 75,
    'country_medals': 450,
    'sport': 'Swimming'
}

prediction = model.predict([athlete_data])
probability = model.predict_proba([athlete_data])
```

---

## 📊 Data Sources

- **Main Dataset:** [Kaggle Olympics Dataset](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history)
- **Time Period:** 1896-2016 (Summer & Winter Olympics)
- **Records:** 271,116 athletes, 130 countries, 300+ sports

---

## 🎓 Key Insights from EDA

1. **Medal Distribution:** Highly imbalanced (~11% medal winners)
2. **Correlation:** Country resources strongly predict medal success
3. **Age Factor:** Most medals won by athletes aged 22-28
4. **Country Dominance:** Few countries win majority of medals
5. **Sport Variation:** Different sports have different medal prediction patterns
6. **Temporal Trends:** Olympic participation growing significantly
7. **Gender Gap:** Historical gender imbalance in participation

---

## 🔒 Best Practices Implemented

✅ **Code Quality:**
- Modular architecture (src/ folder)
- Documented functions with docstrings
- Type hints where appropriate
- Configuration files for hyperparameters

✅ **ML Practices:**
- Proper train-test splitting
- Cross-validation for robustness
- Feature scaling & normalization
- Handling class imbalance

✅ **Data Integrity:**
- Data validation checks
- Missing value handling
- Outlier detection
- Data versioning with timestamps

---

## 🐛 Troubleshooting

**Q: Dashboard won't start?**
- Ensure Python 3.9+: `python --version`
- Reinstall Streamlit: `pip install --upgrade streamlit`
- Clear cache: `streamlit cache clear`

**Q: Model predictions are slow?**
- Check system resources (RAM, CPU)
- Consider using smaller dataset for testing
- Verify models are loaded correctly

**Q: Missing data errors?**
- Run data preprocessing: `python src/data_loader.py`
- Check data/ directory contains raw CSV files
- Verify file paths in configuration

---

## 📈 Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Real-time data updates for recent Olympics
- [ ] API endpoint for model serving
- [ ] Explainability features (SHAP, LIME)
- [ ] Model comparison tools
- [ ] Advanced visualization with Plotly 3D
- [ ] Export predictions to CSV/PDF
- [ ] Historical prediction accuracy tracking

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mohammadhadi Shiva**  
🎓 Data Science & Machine Learning Specialist  
📊 Python | Machine Learning | Data Visualization  
📍 Deutschland  
🔗 [GitHub](https://github.com/shiva-hadi1366) | [LinkedIn](https://linkedin.com/in/shiva-hadi)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Contact & Support

- 📧 Email: shiva.hadi1366@gmail.com
- 💬 GitHub Issues: [Report Issues](https://github.com/shiva-hadi1366/olympics-ml-dashboard/issues)
- 🐦 Twitter: [@shiva_hadi](https://twitter.com/shiva_hadi)

---

## 📚 References & Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Tutorial](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas User Guide](https://pandas.pydata.org/docs/)
- [Kaggle Olympics Dataset](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history)

---

<div align="center">

**⭐ If you find this project helpful, please star the repository!**

Made with ❤️ by Mohammadhadi Shiva

[⬆ Back to Top](#-olympics-medal-prediction-ml-dashboard)

</div>