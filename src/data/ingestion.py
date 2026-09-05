"""Data ingestion: download Rossmann dataset and load into PostgreSQL."""

import logging
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL, RAW_DIR

logger = logging.getLogger(__name__)


def download_rossmann(output_dir: Path = RAW_DIR) -> Path:
    """Download Rossmann Store Sales dataset from Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "rossmann-store-sales.zip"

    if (output_dir / "train.csv").exists():
        logger.info("Dataset already downloaded, skipping.")
        return output_dir

    logger.info("Downloading Rossmann dataset from Kaggle...")
    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "rossmann-store-sales",
            "-p", str(output_dir),
        ],
        check=True,
    )

    if zip_path.exists():
        logger.info("Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink()

    logger.info("Download complete: %s", output_dir)
    return output_dir


def load_raw_data(data_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files into DataFrames."""
    train = pd.read_csv(
        data_dir / "train.csv",
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
        low_memory=False,
    )
    store = pd.read_csv(data_dir / "store.csv")
    logger.info("Loaded train: %s rows, store: %s rows", len(train), len(store))
    return train, store


def clean_data(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Merge and clean raw data."""
    df = train.merge(store, on="Store", how="left")

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={
        "store": "store_id",
        "dayofweek": "day_of_week",
        "stateholiday": "state_holiday",
        "schoolholiday": "school_holiday",
        "competitiondistance": "competition_distance",
        "competitionopensincemonth": "competition_open_since_month",
        "competitionopensinceyear": "competition_open_since_year",
        "storetype": "store_type",
        "promo2sinceweek": "promo2_since_week",
        "promo2sinceyear": "promo2_since_year",
        "promointerval": "promo_interval",
    })

    df = df[df["is_open"] == 1].copy()
    df = df[df["sales"] > 0].copy()
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)

    logger.info("Cleaned data: %s rows", len(df))
    return df


def insert_stores(store: pd.DataFrame, engine) -> None:
    """Insert store data into PostgreSQL."""
    store_clean = store.copy()
    store_clean.columns = [c.lower() for c in store_clean.columns]
    store_clean = store_clean.rename(columns={
        "store": "store_id",
        "storetype": "store_type",
        "competitiondistance": "competition_distance",
        "competitionopensincemonth": "competition_open_since_month",
        "competitionopensinceyear": "competition_open_since_year",
        "promo2sinceweek": "promo2_since_week",
        "promo2sinceyear": "promo2_since_year",
        "promointerval": "promo_interval",
    })

    cols = [
        "store_id", "store_type", "assortment", "competition_distance",
        "competition_open_since_month", "competition_open_since_year",
        "promo2", "promo2_since_week", "promo2_since_year", "promo_interval",
    ]
    store_clean = store_clean[cols]
    store_clean.to_sql("stores", engine, if_exists="replace", index=False)
    logger.info("Inserted %d stores into DB", len(store_clean))


def insert_sales(df: pd.DataFrame, engine) -> None:
    """Insert sales data into PostgreSQL."""
    cols = [
        "store_id", "date", "day_of_week", "sales", "customers",
        "is_open", "promo", "state_holiday", "school_holiday",
    ]
    sales_df = df[cols].copy()
    sales_df.to_sql("sales", engine, if_exists="replace", index=False)
    logger.info("Inserted %d sales rows into DB", len(sales_df))


def run_ingestion() -> pd.DataFrame:
    """Full ingestion pipeline: download, clean, insert into DB."""
    download_rossmann()
    train, store = load_raw_data()
    df = clean_data(train, store)

    engine = create_engine(DATABASE_URL)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS forecasts"))
        conn.execute(text("DROP TABLE IF EXISTS weather"))
        conn.execute(text("DROP TABLE IF EXISTS sales"))
        conn.execute(text("DROP TABLE IF EXISTS stores"))

    insert_stores(store, engine)
    insert_sales(df, engine)

    df.to_parquet(RAW_DIR / "rossmann_clean.parquet", index=False)
    logger.info("Saved clean parquet to %s", RAW_DIR / "rossmann_clean.parquet")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion()
