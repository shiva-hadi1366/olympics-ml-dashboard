import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import os

st.title("📊 Overview – Olympische Daten")

# -------------------------------
# Daten laden (ZIP lesen)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ZIP_PATH = os.path.join(BASE_DIR, "data", "cleaned", "athletes_clean.zip")

with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open("athletes_clean.csv") as f:
        df = pd.read_csv(f)

# -------------------------------
# Beispielhafte Datenansicht
# -------------------------------
st.subheader("🔍 Beispielhafte Datenansicht")
st.dataframe(df.head())

col1, col2, col3 = st.columns(3)
col1.metric("Anzahl Athleten", df["ID"].nunique())
col2.metric("Anzahl Nationen", df["NOC"].nunique())
col3.metric("Anzahl Sportarten", df["Sport"].nunique())

st.subheader("📉 Altersverteilung")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df["Age"].dropna(), bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("🥇 Top 10 Länder nach Medaillen")
medals = df[df["Medal"] != "No Medal"]
top_countries = medals["NOC"].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(8,4))
sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax2)
st.pyplot(fig2)
