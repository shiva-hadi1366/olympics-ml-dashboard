import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple numerical features."""
    df["athletes_per_event"] = df["athletes"] / df["events"]
    df["gdp_per_capita"] = df["gdp"] / df["population"]
    return df


def add_medal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add medal-related engineered features."""
    df["total_medals"] = df["gold"] + df["silver"] + df["bronze"]
    df["medals_per_million"] = df["total_medals"] / (df["population"] / 1_000_000)
    df["gold_ratio"] = df["gold"] / df["total_medals"].replace(0, 1)
    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary and multi-class target labels."""
    df["medal_binary"] = (df["total_medals"] > 0).astype(int)

    def classify(row):
        if row["gold"] > 0:
            return "Gold"
        elif row["silver"] > 0:
            return "Silver"
        elif row["bronze"] > 0:
            return "Bronze"
        return "None"

    df["medal_class"] = df.apply(classify, axis=1)
    return df

def add_growth_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate medal growth compared to previous Olympic cycle."""
    df = df.sort_values(["country", "year"])
    df["prev_total_medals"] = df.groupby("country")["total_medals"].shift(1)
    df["medal_growth"] = df["total_medals"] - df["prev_total_medals"]
    df["medal_growth"].fillna(0, inplace=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_basic_features(df)
    df = add_medal_features(df)
    df = add_growth_feature(df)
    df = add_target_columns(df)
    return df
