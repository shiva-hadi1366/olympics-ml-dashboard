import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Seiteneinstellungen
# -------------------------------
st.set_page_config(
    page_title="Olympics ML Dashboard",
    layout="wide"
)

# -------------------------------
# Daten laden
# -------------------------------
@st.cache_data
def load_data():
    # CSV ohne Dateiendung im raw-Ordner
    df = pd.read_csv("../data/raw/athlete_events.zip")
    return df

df = load_data()

# -------------------------------
# Titel
# -------------------------------
st.title("🏅 Olympics Machine Learning Dashboard")
st.write("Dieses Dashboard zeigt eine erste Übersicht über die olympischen Datensätze und dient als Grundlage für weitere ML‑Analysen.")

st.subheader("📊 Beispielhafte Datenansicht")
st.dataframe(df.head())

# -------------------------------
# Einfache Statistik
# -------------------------------
st.subheader("📈 Grundlegende Statistiken")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Anzahl der Athleten", df["ID"].nunique())

with col2:
    st.metric("Anzahl der Nationen", df["NOC"].nunique())

with col3:
    st.metric("Anzahl der Sportarten", df["Sport"].nunique())

# -------------------------------
# Visualisierung: Verteilung der Altersgruppen
# -------------------------------
st.subheader("📉 Altersverteilung der Athleten")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Age"].dropna(), bins=30, kde=True, ax=ax)
ax.set_xlabel("Alter")
ax.set_ylabel("Anzahl")
ax.set_title("Verteilung des Alters")
st.pyplot(fig)

# -------------------------------
# Visualisierung: Top 10 Länder nach Medaillen
# -------------------------------
st.subheader("🥇 Top 10 Länder nach Anzahl der Medaillen")

medals = df[df["Medal"].notna()]
top_countries = medals["NOC"].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax2)
ax2.set_xlabel("Anzahl der Medaillen")
ax2.set_ylabel("Land")
ax2.set_title("Top 10 Länder nach Medaillen")
st.pyplot(fig2)

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.caption("Erstellt von Mohammadhadi Shiva – Data Science Projekt (Streamlit Dashboard)")
