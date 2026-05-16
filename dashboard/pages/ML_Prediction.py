import streamlit as st
import pandas as pd
import joblib
import os

st.title("🤖 ML Prediction – Medaillen Vorhersage")

# ----------------------------------------------------
# 🔧 Modelle sicher laden (mit absolutem Pfad)
# ----------------------------------------------------
@st.cache_resource
def load_models():
    # Ordnerstruktur automatisch erkennen
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    binary_path = os.path.join(MODEL_DIR, "binary_model.pkl")
    multi_path = os.path.join(MODEL_DIR, "multi_model.pkl")

    if not os.path.exists(binary_path):
        st.error(f"❌ Binary Model nicht gefunden: {binary_path}")
        st.stop()

    if not os.path.exists(multi_path):
        st.error(f"❌ Multi Model nicht gefunden: {multi_path}")
        st.stop()

    # GANZ WICHTIG:
    # Modelle sind komplette Pipelines → kein Encoding nötig!
    binary = joblib.load(binary_path)
    multi = joblib.load(multi_path)

    return binary, multi


binary_model, multi_model = load_models()

# ----------------------------------------------------
# 📝 Eingabeformular
# ----------------------------------------------------
st.subheader("📝 Eingabe für Vorhersage")

age = st.number_input("Alter", 10, 60, 23)
height = st.number_input("Größe (cm)", 120, 220, 180)
weight = st.number_input("Gewicht (kg)", 40, 150, 75)
year = st.selectbox("Jahr", [2000, 2004, 2008, 2012, 2016, 2020])
sex = st.selectbox("Geschlecht", ["M", "F"])
team = st.text_input("Team", "USA")
noc = st.text_input("NOC", "USA")
region = st.text_input("Region", "Americas")
season = st.selectbox("Saison", ["Summer", "Winter"])
city = st.text_input("Stadt", "Rio")
sport = st.text_input("Sport", "Swimming")
event = st.text_input("Event", "100m Freestyle")

# ----------------------------------------------------
# 🔮 Vorhersage
# ----------------------------------------------------
if st.button("🔮 Vorhersage starten"):

    # BMI berechnen
    bmi = weight / ((height / 100) ** 2)

    # Eingabedaten in DataFrame
    df = pd.DataFrame([{
        "Age": age,
        "Height": height,
        "Weight": weight,
        "BMI": bmi,
        "Year": year,
        "Sex": sex,
        "Team": team,
        "NOC": noc,
        "Region": region,
        "Season": season,
        "City": city,
        "Sport": sport,
        "Event": event
    }])

    try:
        # Modelle sind Pipelines → direkt predict()
        binary_pred = binary_model.predict(df)[0]
        multi_pred = multi_model.predict(df)[0]

        st.success(f"🏅 Medaille? {'Ja' if binary_pred == 1 else 'Nein'}")

        medal_map = {0: "Keine", 1: "Bronze", 2: "Silber", 3: "Gold"}
        st.info(f"🥇 Medaillentyp: {medal_map[multi_pred]}")

    except Exception as e:
        st.error(f"❌ Fehler bei der Vorhersage: {e}")
