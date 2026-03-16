"""
src/features.py
---------------
Feature engineering pipeline for Walmart Sales Forecasting.
Merges all three source datasets, engineers temporal/lag/rolling
features, and outputs a clean DataFrame ready for model training.

Usage:
    from src.features import build_features
    df = build_features(train_path, store_path, feature_path)
"""

import pandas as pd
import numpy as np
from pathlib import Path



# Walmart-defined holiday weeks
HOLIDAY_WEEKS = {
    "super_bowl":    ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
    "labor_day":     ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
    "thanksgiving":  ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
    "christmas":     ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
}

LAG_WEEKS = [1, 2, 4, 8, 12, 52]       # lag features to generate
ROLL_WINDOWS = [4, 8, 12]              # rolling window sizes (weeks)
MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]



def load_raw_data(
    train_path: str | Path,
    store_path: str | Path,
    feature_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw CSVs and parse dates."""
    train_df   = pd.read_csv(train_path,   parse_dates=["Date"])
    store_df   = pd.read_csv(store_path)
    feature_df = pd.read_csv(feature_path, parse_dates=["Date"])
    return train_df, store_df, feature_df



def merge_datasets(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join train → stores → features.
    The IsHoliday column from features is dropped to keep only the
    train version (ground truth).
    """
    df = train_df.merge(store_df, on="Store", how="left")
    df = df.merge(
        feature_df.drop(columns=["IsHoliday"]),
        on=["Store", "Date"],
        how="left",
    )
    return df



def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar components from the Date column."""
    df = df.copy()
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"]   = df["Date"].dt.dayofyear
    df["Quarter"]     = df["Date"].dt.quarter

    # Cyclical encoding for week-of-year (captures periodicity)
    df["Week_sin"]    = np.sin(2 * np.pi * df["Week"] / 52)
    df["Week_cos"]    = np.cos(2 * np.pi * df["Week"] / 52)
    df["Month_sin"]   = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]   = np.cos(2 * np.pi * df["Month"] / 12)

    return df



def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary columns for each named Walmart holiday week,
    plus a count of how many markdown promotions are active
    during that week.
    """
    df = df.copy()
    for holiday, dates in HOLIDAY_WEEKS.items():
        dt_dates = pd.to_datetime(dates)
        df[f"Is_{holiday}"] = df["Date"].isin(dt_dates).astype(int)

    # Numeric IsHoliday → integer (easier for tree models)
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    return df



def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the five MarkDown columns:
      - fill NaN with 0 (no promotion active)
      - total markdown spend
      - number of active markdowns
      - flag: any markdown active
    """
    df = df.copy()
    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)

    df["MarkDown_Total"]  = df[MARKDOWN_COLS].sum(axis=1)
    df["MarkDown_Count"]  = (df[MARKDOWN_COLS] > 0).sum(axis=1)
    df["MarkDown_Active"] = (df["MarkDown_Total"] > 0).astype(int)

    return df


def add_store_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode store Type; keep Size as-is."""
    df = df.copy()
    type_dummies = pd.get_dummies(df["Type"], prefix="StoreType", drop_first=False)
    df = pd.concat([df, type_dummies], axis=1)
    df.drop(columns=["Type"], inplace=True)
    return df



def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Store, Dept) group, create lag-k weekly sales features.
    Requires the DataFrame to be sorted by Store, Dept, Date first.
    """
    df = df.copy().sort_values(["Store", "Dept", "Date"])
    grp = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    for lag in LAG_WEEKS:
        df[f"Sales_Lag_{lag}w"] = grp.shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean and std of Weekly_Sales over several windows.
    Uses a minimum of 1 period so early rows aren't all NaN.
    """
    df = df.copy().sort_values(["Store", "Dept", "Date"])

    for window in ROLL_WINDOWS:
        df[f"Sales_Roll_Mean_{window}w"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"Sales_Roll_Std_{window}w"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=1).std())
        )

    return df



def add_dept_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical mean & std of Weekly_Sales per (Store, Dept).
    These are global stats so they must be computed on training data
    only and then merged in — pass fit_stats=None on first call to
    compute them, or pass a pre-computed dict to apply to test data.
    """
    df = df.copy()
    stats = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
          .agg(Dept_Mean_Sales="mean", Dept_Std_Sales="std")
          .reset_index()
    )
    df = df.merge(stats, on=["Store", "Dept"], how="left")
    return df



def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple median imputation for CPI and Unemployment (585 missing each).
    Lag/rolling NaNs at the start of each series are left as-is —
    tree models handle them natively; for linear models use further imputation.
    """
    df = df.copy()
    for col in ["CPI", "Unemployment"]:
        df[col] = df[col].fillna(df[col].median())
    return df


def build_features(
    train_path: str | Path,
    store_path: str | Path,
    feature_path: str | Path,
    drop_date: bool = True,
) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.

    Parameters
    ----------
    train_path, store_path, feature_path : paths to raw CSVs
    drop_date : whether to drop the Date column in the final output
                (set False if you need it for time-based CV splits)

    Returns
    -------
    pd.DataFrame with all engineered features, sorted by Store/Dept/Date.
    """

    train_df, store_df, feature_df = load_raw_data(train_path, store_path, feature_path)

    df = merge_datasets(train_df, store_df, feature_df)

    df = add_temporal_features(df)
    df = add_holiday_features(df)
    df = add_markdown_features(df)
    df = add_store_features(df)
    df = add_dept_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    df = impute_missing(df)

    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    if drop_date:
        df.drop(columns=["Date"], inplace=True)

    return df


if __name__ == "__main__":
    import sys

    DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")

    df = build_features(
        train_path   = DATA_DIR / "train.csv",
        store_path   = DATA_DIR / "stores.csv",
        feature_path = DATA_DIR / "features.csv",
        drop_date    = False,   # keep Date visible in smoke test
    )

    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):\n{df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nSample:\n{df.head(3).to_string()}")