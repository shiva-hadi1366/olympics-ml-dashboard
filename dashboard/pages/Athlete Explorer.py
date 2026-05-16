import streamlit as st
import pandas as pd

st.title("🏃 Athlete Explorer – Athleten durchsuchen")
@st.cache_data
def load_data():
    return pd.read_csv("../data/cleaned/athletes_clean.csv")

df = load_data()

# -------------------------------
# Filterbereich
# -------------------------------
st.sidebar.header("🔍 Filter")

name_filter = st.sidebar.text_input("Name enthält:")
country_filter = st.sidebar.selectbox("Land (NOC):", ["Alle"] + sorted(df["NOC"].unique()))
sport_filter = st.sidebar.selectbox("Sportart:", ["Alle"] + sorted(df["Sport"].unique()))

filtered_df = df.copy()

if name_filter:
    filtered_df = filtered_df[filtered_df["Name"].str.contains(name_filter, case=False, na=False)]

if country_filter != "Alle":
    filtered_df = filtered_df[filtered_df["NOC"] == country_filter]

if sport_filter != "Alle":
    filtered_df = filtered_df[filtered_df["Sport"] == sport_filter]

st.subheader("📋 Gefundene Athleten")
st.write(f"{len(filtered_df)} Athleten gefunden")

st.dataframe(filtered_df[["Name", "Sex", "Age", "Team", "NOC", "Sport", "Event", "Medal"]].head(50))

# -------------------------------
# Detailansicht eines Athleten
# -------------------------------
st.subheader("👤 Athletenprofil")

athlete_names = filtered_df["Name"].unique()

if len(athlete_names) > 0:
    selected_athlete = st.selectbox("Athlet auswählen:", athlete_names)

    athlete = filtered_df[filtered_df["Name"] == selected_athlete].iloc[0]

    st.write(f"### {athlete['Name']}")
    st.write(f"**Team:** {athlete['Team']} ({athlete['NOC']})")
    st.write(f"**Sport:** {athlete['Sport']}")
    st.write(f"**Event:** {athlete['Event']}")
    st.write(f"**Alter:** {athlete['Age']}")
    st.write(f"**Medaille:** {athlete['Medal']}")
else:
    st.info("Keine Athleten gefunden.")
