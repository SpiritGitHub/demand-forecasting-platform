"""Enrich sales data with weather, holidays, and calendar features."""

import logging
from datetime import date

import holidays
import pandas as pd
import requests

from src.config import EXTERNAL_DIR

logger = logging.getLogger(__name__)

GERMAN_STATES = {
    "HB,NI": "NI",
    "BE": "BE",
    "BW": "BW",
    "BY": "BY",
    "HE": "HE",
    "HH": "HH",
    "NW": "NW",
    "RP": "RP",
    "SH": "SH",
    "SL": "SL",
    "SN": "SN",
    "ST": "ST",
    "TH": "TH",
}


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add German holiday flags and distance to nearest holiday."""
    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1)
    de_holidays = holidays.Germany(years=years)

    holiday_dates = sorted(de_holidays.keys())

    df["is_holiday"] = df["date"].dt.date.isin(de_holidays).astype(int)
    df["holiday_name"] = df["date"].dt.date.map(
        lambda d: de_holidays.get(d, "")
    )

    holiday_series = pd.Series(holiday_dates)

    def days_to_nearest_holiday(d: date) -> int:
        idx = holiday_series.searchsorted(d)
        candidates = []
        if idx < len(holiday_series):
            candidates.append(abs((holiday_series.iloc[idx] - d).days))
        if idx > 0:
            candidates.append(abs((d - holiday_series.iloc[idx - 1]).days))
        return min(candidates) if candidates else 999

    df["days_to_holiday"] = df["date"].dt.date.map(days_to_nearest_holiday)

    logger.info("Added holiday features")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features."""
    dt = df["date"].dt
    df["month"] = dt.month
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["quarter"] = dt.quarter
    df["year"] = dt.year
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.is_month_start.astype(int)
    df["is_month_end"] = dt.is_month_end.astype(int)
    df["day_of_month"] = dt.day

    logger.info("Added calendar features")
    return df


def fetch_weather_data(
    latitude: float = 51.5,
    longitude: float = 10.5,
    start_date: str = "2013-01-01",
    end_date: str = "2015-07-31",
) -> pd.DataFrame:
    """Fetch historical weather from Open-Meteo API (free, no key)."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EXTERNAL_DIR / "weather_germany.parquet"

    if cache_path.exists():
        logger.info("Loading cached weather data")
        return pd.read_parquet(cache_path)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "Europe/Berlin",
    }

    logger.info("Fetching weather data from Open-Meteo...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    weather = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temperature_mean": data["daily"]["temperature_2m_mean"],
        "temperature_max": data["daily"]["temperature_2m_max"],
        "temperature_min": data["daily"]["temperature_2m_min"],
        "precipitation": data["daily"]["precipitation_sum"],
        "wind_speed": data["daily"]["wind_speed_10m_max"],
    })

    weather.to_parquet(cache_path, index=False)
    logger.info("Weather data: %d days", len(weather))
    return weather


def enrich_with_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather data into the sales DataFrame."""
    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")

    weather = fetch_weather_data(start_date=start, end_date=end)
    df = df.merge(weather, on="date", how="left")

    df["is_rainy"] = (df["precipitation"].fillna(0) > 1.0).astype(int)

    logger.info("Merged weather data")
    return df


def run_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Run all enrichment steps."""
    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = enrich_with_weather(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import RAW_DIR
    df = pd.read_parquet(RAW_DIR / "rossmann_clean.parquet")
    enriched = run_enrichment(df)
    enriched.to_parquet(RAW_DIR.parent / "processed" / "rossmann_enriched.parquet", index=False)
    logger.info("Enrichment complete: %d rows, %d columns", len(enriched), len(enriched.columns))
