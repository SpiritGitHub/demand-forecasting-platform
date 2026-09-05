"""Data validation using Pandera schemas."""

import logging

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

logger = logging.getLogger(__name__)

raw_sales_schema = DataFrameSchema(
    columns={
        "store_id": Column(int, Check.gt(0)),
        "date": Column("datetime64[ns]"),
        "sales": Column(int, Check.ge(0)),
        "customers": Column(int, Check.ge(0), nullable=True),
        "is_open": Column(int, Check.isin([0, 1])),
        "promo": Column(int, Check.isin([0, 1])),
        "day_of_week": Column(int, Check.in_range(1, 7)),
    },
    coerce=True,
)

enriched_schema = DataFrameSchema(
    columns={
        "store_id": Column(int, Check.gt(0)),
        "date": Column("datetime64[ns]"),
        "sales": Column(int, Check.gt(0)),
        "month": Column(int, Check.in_range(1, 12)),
        "day_of_week": Column(int, Check.in_range(1, 7)),
        "is_weekend": Column(int, Check.isin([0, 1])),
        "is_holiday": Column(int, Check.isin([0, 1])),
    },
    coerce=True,
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw sales data."""
    validated = raw_sales_schema.validate(df, lazy=True)
    logger.info("Raw data validation passed: %d rows", len(validated))
    return validated


def validate_enriched(df: pd.DataFrame) -> pd.DataFrame:
    """Validate enriched data."""
    validated = enriched_schema.validate(df, lazy=True)
    logger.info("Enriched data validation passed: %d rows", len(validated))
    return validated


def check_date_continuity(df: pd.DataFrame) -> dict:
    """Check for gaps in date sequences per store."""
    issues = {}
    for store_id, group in df.groupby("store_id"):
        dates = group["date"].sort_values()
        expected = pd.date_range(dates.min(), dates.max(), freq="D")
        missing = set(expected) - set(dates)
        if missing:
            issues[store_id] = sorted(missing)

    if issues:
        total_gaps = sum(len(v) for v in issues.values())
        logger.warning(
            "Date gaps found in %d stores (%d total missing days)",
            len(issues), total_gaps,
        )
    else:
        logger.info("No date gaps found")

    return issues


def run_validation(df: pd.DataFrame, stage: str = "raw") -> pd.DataFrame:
    """Run validation for the specified stage."""
    if stage == "raw":
        df = validate_raw(df)
    elif stage == "enriched":
        df = validate_enriched(df)

    gaps = check_date_continuity(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import RAW_DIR
    df = pd.read_parquet(RAW_DIR / "rossmann_clean.parquet")
    run_validation(df, stage="raw")
