"""Feature engineering for demand forecasting.

Builds ML-ready features on top of the enriched dataset.
All lag/rolling computations are per-store to avoid data leakage.
"""

import logging

import numpy as np
import pandas as pd

from src.config import FEATURES_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    target: str = "sales",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged sales values per store.

    Each lag_N feature = sales from N days ago for the same store.
    Rows where the lag is unavailable (beginning of series) get NaN.
    """
    if lags is None:
        lags = [7, 14, 28]

    for lag in lags:
        col = f"{target}_lag_{lag}"
        df[col] = df.groupby("store_id")[target].shift(lag)

    logger.info("Added lag features: %s", lags)
    return df


# ---------------------------------------------------------------------------
# Rolling / expanding features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    target: str = "sales",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std per store.

    The window is shifted by 1 so the current day is excluded
    (prevents target leakage).
    """
    if windows is None:
        windows = [7, 14, 28]

    grouped = df.groupby("store_id")[target]

    for w in windows:
        shifted = grouped.shift(1)
        rolling = shifted.rolling(window=w, min_periods=w)
        df[f"{target}_rolling_mean_{w}"] = rolling.mean()
        if w == 7:
            df[f"{target}_rolling_std_{w}"] = rolling.std()

    logger.info("Added rolling features: windows=%s", windows)
    return df


def add_ewm_features(
    df: pd.DataFrame,
    target: str = "sales",
    span: int = 7,
) -> pd.DataFrame:
    """Add exponentially weighted mean per store (shifted by 1)."""
    shifted = df.groupby("store_id")[target].shift(1)
    df[f"{target}_ewm_{span}"] = shifted.ewm(span=span, min_periods=span).mean()

    logger.info("Added EWM feature: span=%d", span)
    return df


# ---------------------------------------------------------------------------
# Trend features
# ---------------------------------------------------------------------------

def add_trend_features(
    df: pd.DataFrame,
    target: str = "sales",
) -> pd.DataFrame:
    """Add sales momentum: difference between short and long rolling mean."""
    if f"{target}_rolling_mean_7" in df.columns and f"{target}_rolling_mean_28" in df.columns:
        df[f"{target}_trend"] = (
            df[f"{target}_rolling_mean_7"] - df[f"{target}_rolling_mean_28"]
        )
    return df


# ---------------------------------------------------------------------------
# Promo features
# ---------------------------------------------------------------------------

def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add promo duration and promo-related features."""
    promo_groups = df.groupby("store_id")["promo"]
    df["promo_duration"] = promo_groups.transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    )
    df.loc[df["promo"] == 0, "promo_duration"] = 0

    if "promo2" in df.columns:
        df["promo2"] = df["promo2"].fillna(0).astype(int)

    logger.info("Added promo features")
    return df


# ---------------------------------------------------------------------------
# Competition features
# ---------------------------------------------------------------------------

def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add competition-related features."""
    if "competition_distance" in df.columns:
        median_dist = df["competition_distance"].median()
        df["competition_distance"] = df["competition_distance"].fillna(median_dist)
        df["competition_distance_log"] = np.log1p(df["competition_distance"])

    if "competition_open_since_year" in df.columns and "competition_open_since_month" in df.columns:
        comp_year = df["competition_open_since_year"].fillna(0).astype(int)
        comp_month = df["competition_open_since_month"].fillna(0).astype(int)

        df["competition_open_months"] = (
            (df["date"].dt.year - comp_year) * 12
            + (df["date"].dt.month - comp_month)
        )
        df.loc[comp_year == 0, "competition_open_months"] = 0
        df["competition_open_months"] = df["competition_open_months"].clip(lower=0)

    logger.info("Added competition features")
    return df


# ---------------------------------------------------------------------------
# Store encoding
# ---------------------------------------------------------------------------

def add_store_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode store_type and assortment."""
    type_map = {"a": 1, "b": 2, "c": 3, "d": 4}
    assort_map = {"a": 1, "b": 2, "c": 3}

    if "store_type" in df.columns:
        df["store_type_enc"] = df["store_type"].str.lower().map(type_map).fillna(0).astype(int)
    if "assortment" in df.columns:
        df["assortment_enc"] = df["assortment"].str.lower().map(assort_map).fillna(0).astype(int)

    logger.info("Added store encoding")
    return df


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature interactions that capture combined effects."""
    if "promo" in df.columns and "is_weekend" in df.columns:
        df["promo_x_weekend"] = df["promo"] * df["is_weekend"]

    if "promo" in df.columns and "is_holiday" in df.columns:
        df["promo_x_holiday"] = df["promo"] * df["is_holiday"]

    if "is_holiday" in df.columns and "store_type_enc" in df.columns:
        df["holiday_x_store_type"] = df["is_holiday"] * df["store_type_enc"]

    if "day_of_week" in df.columns and "promo" in df.columns:
        df["dow_x_promo"] = df["day_of_week"] * df["promo"]

    logger.info("Added interaction features")
    return df


# ---------------------------------------------------------------------------
# Holiday proximity (directional)
# ---------------------------------------------------------------------------

def add_holiday_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Split days_to_holiday into before/after components."""
    if "days_to_holiday" not in df.columns or "is_holiday" not in df.columns:
        return df

    df["days_before_holiday"] = df["days_to_holiday"]
    df["days_after_holiday"] = df["days_to_holiday"]

    if "holiday_name" in df.columns:
        is_before = df["holiday_name"] == ""
        df.loc[~is_before, "days_before_holiday"] = 0
    return df


# ---------------------------------------------------------------------------
# Store-level aggregates (fit on training data only)
# ---------------------------------------------------------------------------

def add_store_aggregates(
    df: pd.DataFrame,
    train_end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add per-store historical aggregates.

    If train_end_date is provided, aggregates are computed only on data
    before that date to prevent leakage during backtesting.
    """
    if train_end_date is not None:
        ref = df[df["date"] <= pd.Timestamp(train_end_date)]
    else:
        ref = df

    store_stats = ref.groupby("store_id")["sales"].agg(
        store_mean_sales="mean",
        store_median_sales="median",
        store_std_sales="std",
    ).reset_index()

    original_cols = set(df.columns)
    df = df.merge(store_stats, on="store_id", how="left")

    for col in ["store_mean_sales", "store_median_sales", "store_std_sales"]:
        if col in original_cols:
            df = df.drop(columns=[col])
            df = df.merge(store_stats[["store_id", col]], on="store_id", how="left")

    logger.info("Added store-level aggregates")
    return df


# ---------------------------------------------------------------------------
# Day-of-week per store profile
# ---------------------------------------------------------------------------

def add_dow_profile(
    df: pd.DataFrame,
    train_end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add day-of-week sales profile per store (mean sales ratio)."""
    if train_end_date is not None:
        ref = df[df["date"] <= pd.Timestamp(train_end_date)]
    else:
        ref = df

    global_dow_mean = ref.groupby("day_of_week")["sales"].mean()

    store_dow_mean = ref.groupby(["store_id", "day_of_week"])["sales"].mean().reset_index()
    store_dow_mean = store_dow_mean.rename(columns={"sales": "store_dow_mean_sales"})

    df = df.merge(store_dow_mean, on=["store_id", "day_of_week"], how="left")

    store_mean = ref.groupby("store_id")["sales"].mean()
    df["store_dow_ratio"] = df["store_dow_mean_sales"] / df["store_id"].map(store_mean)
    df["store_dow_ratio"] = df["store_dow_ratio"].fillna(1.0)

    logger.info("Added day-of-week store profile")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    train_end_date: str | pd.Timestamp | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Parameters
    ----------
    df : DataFrame with enriched data (output of run_enrichment).
    train_end_date : If set, store aggregates use only data before this date.
    drop_na : Drop rows where lag features produced NaN (first 28 days per store).
    """
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ewm_features(df)
    df = add_trend_features(df)
    df = add_promo_features(df)
    df = add_competition_features(df)
    df = add_store_encoding(df)
    df = add_interaction_features(df)
    df = add_holiday_proximity(df)
    df = add_store_aggregates(df, train_end_date=train_end_date)
    df = add_dow_profile(df, train_end_date=train_end_date)

    if drop_na:
        before = len(df)
        df = df.dropna(subset=[c for c in df.columns if "lag_" in c or "rolling_" in c or "ewm_" in c])
        df = df.reset_index(drop=True)
        logger.info("Dropped %d rows with NaN from lag/rolling warm-up", before - len(df))

    logger.info("Feature engineering complete: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_path = PROCESSED_DIR / "rossmann_enriched.parquet"
    output_path = FEATURES_DIR / "rossmann_features.parquet"

    logger.info("Loading enriched data from %s", input_path)
    df = pd.read_parquet(input_path)

    df = build_features(df)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved features to %s", output_path)
