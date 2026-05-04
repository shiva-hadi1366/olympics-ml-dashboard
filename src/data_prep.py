"""
data_prep.py
------------
Cleans and preprocesses the Olympics athlete_events dataset.

Steps:
  1. Load raw CSVs
  2. Merge with NOC regions
  3. Drop duplicate rows
  4. Fill missing Medal values with 'No Medal'
  5. Impute Age / Height / Weight with per-Sport medians
  6. Add helper columns (has_medal, bmi)
  7. Save cleaned CSV to data/cleaned/
"""

import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
CLEAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cleaned"))
os.makedirs(CLEAN_DIR, exist_ok=True)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSVs."""
    athletes = pd.read_csv(os.path.join(ROOT, "athlete_events.csv"))
    noc = pd.read_csv(os.path.join(ROOT, "noc_regions.csv"))
    return athletes, noc


def merge_regions(df: pd.DataFrame, noc: pd.DataFrame) -> pd.DataFrame:
    """Merge athlete data with NOC region table."""
    noc = noc.rename(columns={"region": "Region", "notes": "NOC_Notes"})
    df = df.merge(noc[["NOC", "Region"]], on="NOC", how="left")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[duplicates]  removed {before - len(df)} rows  →  {len(df)} remaining")
    return df


def clean_medal(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN medals with 'No Medal' and add a binary flag."""
    df["Medal"] = df["Medal"].fillna("No Medal")
    df["has_medal"] = (df["Medal"] != "No Medal").astype(int)
    return df


def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute Age, Height, Weight with per-Sport medians.
    Falls back to overall median for sports with no valid values.
    """
    for col in ["Age", "Height", "Weight"]:
        before = df[col].isna().sum()
        sport_medians = df.groupby("Sport")[col].transform("median")
        overall_median = df[col].median()
        df[col] = df[col].fillna(sport_medians).fillna(overall_median)
        after = df[col].isna().sum()
        print(f"[impute]  {col:8s}  filled {before - after}  missing  →  {after} remaining")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns useful for analysis/ML."""
    # BMI (Body Mass Index)
    df["BMI"] = (df["Weight"] / ((df["Height"] / 100) ** 2)).round(2)
    return df


def save_cleaned(df: pd.DataFrame, filename: str = "athletes_clean.csv") -> str:
    """Save cleaned dataframe to the cleaned data folder."""
    path = os.path.join(CLEAN_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[saved]  {path}  ({len(df)} rows × {len(df.columns)} cols)")
    return path


def run_pipeline() -> pd.DataFrame:
    """Execute the full cleaning pipeline and return the cleaned DataFrame."""
    print("── Loading raw data ──────────────────────────────────────────────────")
    athletes, noc = load_raw()
    print(f"  athletes: {athletes.shape}   noc: {noc.shape}")

    print("\n── Merging NOC regions ───────────────────────────────────────────────")
    df = merge_regions(athletes, noc)
    print(f"  merged shape: {df.shape}")

    print("\n── Removing duplicates ───────────────────────────────────────────────")
    df = drop_duplicates(df)

    print("\n── Cleaning Medal column ─────────────────────────────────────────────")
    df = clean_medal(df)

    print("\n── Imputing numeric columns ──────────────────────────────────────────")
    df = impute_numeric(df)

    print("\n── Adding derived features ───────────────────────────────────────────")
    df = add_features(df)

    print("\n── Saving cleaned data ───────────────────────────────────────────────")
    save_cleaned(df)

    print("\n── Summary ───────────────────────────────────────────────────────────")
    print(df.dtypes)
    print("\nMissing values after cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df


if __name__ == "__main__":
    run_pipeline()

