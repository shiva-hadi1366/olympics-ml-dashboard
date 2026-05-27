import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def load_data():
    athletes = pd.read_csv(os.path.join(RAW_DIR, 'athlete_events.csv'))
    regions = pd.read_csv(os.path.join(RAW_DIR, 'noc_regions.csv'))
    return athletes, regions


def preprocess(athletes, regions):
    df = athletes.merge(regions[['NOC', 'region']], on='NOC', how='left')
    df['Medal'].fillna('None', inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Height'].fillna(df['Height'].median(), inplace=True)
    df['Weight'].fillna(df['Weight'].median(), inplace=True)
    df['Has_Medal'] = (df['Medal'] != 'None').astype(int)
    return df


def save_processed(df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DIR, 'olympics_clean.csv'), index=False)
    print("Saved: olympics_clean.csv")


if __name__ == '__main__':
    athletes, regions = load_data()
    df = preprocess(athletes, regions)
    save_processed(df)
    print(df.shape)
    print(df.head())
