import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🌍 Country Insights – Länderanalyse")

@st.cache_data
def load_data():
    return pd.read_csv("../data/cleaned/athletes_clean.csv")


df = load_data()

# -------------------------------
# Auswahl eines Landes
# -------------------------------
st.subheader("🇨🇳 Land auswählen")

countries = sorted(df["NOC"].unique())
selected_country = st.selectbox("Wähle ein Land (NOC):", countries)

country_df = df[df["NOC"] == selected_country]

st.write(f"### 📌 Statistiken für {selected_country}")

col1, col2, col3 = st.columns(3)
col1.metric("Athleten", country_df["ID"].nunique())
col2.metric("Sportarten", country_df["Sport"].nunique())
col3.metric("Medaillen", (country_df["Medal"] != "No Medal").sum())

# -------------------------------
# Medaillenverteilung
# -------------------------------
st.subheader("🥇 Medaillenverteilung")

medal_counts = country_df["Medal"].value_counts()

fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x=medal_counts.index, y=medal_counts.values, ax=ax)
ax.set_title(f"Medaillenverteilung – {selected_country}")
ax.set_xlabel("Medaille")
ax.set_ylabel("Anzahl")
st.pyplot(fig)

# -------------------------------
# Top-Sportarten des Landes
# -------------------------------
st.subheader("🏆 Top 10 Sportarten nach Medaillen")

medals_only = country_df[country_df["Medal"] != "No Medal"]
top_sports = medals_only["Sport"].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(6,4))
sns.barplot(y=top_sports.index, x=top_sports.values, ax=ax2)
ax2.set_title(f"Top 10 Sportarten – {selected_country}")
ax2.set_xlabel("Medaillen")
ax2.set_ylabel("Sportart")
st.pyplot(fig2)

# -------------------------------
# Altersverteilung
# -------------------------------
st.subheader("📉 Altersverteilung der Athleten")

fig3, ax3 = plt.subplots(figsize=(6,4))
sns.histplot(country_df["Age"].dropna(), bins=20, kde=True, ax=ax3)
ax3.set_title(f"Altersverteilung – {selected_country}")
ax3.set_xlabel("Alter")
ax3.set_ylabel("Anzahl")
st.pyplot(fig3)
