"""Phase 5: Replenishment engine — safety stock, reorder point, stockout risk."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.config import FEATURES_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

SERVICE_LEVEL_Z = {
    0.90: 1.282,
    0.95: 1.645,
    0.99: 2.326,
}


def compute_safety_stock(
    demand_std: float,
    lead_time_days: int,
    service_level: float = 0.95,
    forecast_error_std: float = 0.0,
) -> float:
    z = SERVICE_LEVEL_Z.get(service_level)
    if z is None:
        z = stats.norm.ppf(service_level)
    total_std = np.sqrt(lead_time_days * demand_std**2 + forecast_error_std**2)
    return z * total_std


def compute_reorder_point(
    avg_daily_demand: float,
    lead_time_days: int,
    safety_stock: float,
) -> float:
    return avg_daily_demand * lead_time_days + safety_stock


def compute_order_quantity(
    avg_daily_demand: float,
    review_period_days: int,
    lead_time_days: int,
    safety_stock: float,
    current_stock: float,
) -> float:
    target = avg_daily_demand * (lead_time_days + review_period_days) + safety_stock
    return max(0.0, target - current_stock)


def compute_stockout_risk(
    current_stock: float,
    avg_daily_demand: float,
    demand_std: float,
    horizon_days: int,
) -> float:
    if demand_std <= 0 or horizon_days <= 0:
        return 0.0
    expected = avg_daily_demand * horizon_days
    std = demand_std * np.sqrt(horizon_days)
    if std == 0:
        return 1.0 if expected > current_stock else 0.0
    return float(1.0 - stats.norm.cdf(current_stock, loc=expected, scale=std))


def build_store_demand_summary(
    df: pd.DataFrame,
    target_col: str = "sales",
    horizon_days: int = 30,
) -> pd.DataFrame:
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=horizon_days)
    recent = df[df["date"] > cutoff]

    summary = (
        recent.groupby("store_id")[target_col]
        .agg(avg_daily_demand="mean", demand_std="std")
        .reset_index()
    )
    summary["demand_std"] = summary["demand_std"].fillna(0)
    return summary


def generate_replenishment_plan(
    store_demand: pd.DataFrame,
    lead_time_days: int = 3,
    review_period_days: int = 7,
    service_level: float = 0.95,
    current_stock_col: str = "current_stock",
) -> pd.DataFrame:
    df = store_demand.copy()
    if current_stock_col not in df.columns:
        df[current_stock_col] = 0.0

    df["safety_stock"] = df.apply(
        lambda r: compute_safety_stock(r["demand_std"], lead_time_days, service_level),
        axis=1,
    )
    df["reorder_point"] = df.apply(
        lambda r: compute_reorder_point(
            r["avg_daily_demand"], lead_time_days, r["safety_stock"]
        ),
        axis=1,
    )
    df["order_quantity"] = df.apply(
        lambda r: compute_order_quantity(
            r["avg_daily_demand"],
            review_period_days,
            lead_time_days,
            r["safety_stock"],
            r[current_stock_col],
        ),
        axis=1,
    )
    df["stockout_risk"] = df.apply(
        lambda r: compute_stockout_risk(
            r[current_stock_col],
            r["avg_daily_demand"],
            r["demand_std"],
            lead_time_days,
        ),
        axis=1,
    )
    df["days_of_stock"] = df.apply(
        lambda r: r[current_stock_col] / r["avg_daily_demand"]
        if r["avg_daily_demand"] > 0
        else np.inf,
        axis=1,
    )
    df["alert"] = df["stockout_risk"].apply(
        lambda r: "CRITICAL" if r > 0.5 else "WARNING" if r > 0.2 else "OK"
    )

    return df.round(
        {
            "safety_stock": 0,
            "reorder_point": 0,
            "order_quantity": 0,
            "stockout_risk": 4,
            "days_of_stock": 1,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
    )

    logger.info("Loading features for demand summary...")
    df = pd.read_parquet(FEATURES_DIR / "rossmann_features.parquet")
    df = df[(df["open"] == 1) & (df["sales"] > 0)]

    store_demand = build_store_demand_summary(df, horizon_days=30)
    plan = generate_replenishment_plan(
        store_demand,
        lead_time_days=3,
        review_period_days=7,
        service_level=0.95,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "replenishment_plan.csv"
    plan.to_csv(output, index=False)

    critical = len(plan[plan["alert"] == "CRITICAL"])
    warning = len(plan[plan["alert"] == "WARNING"])
    ok = len(plan) - critical - warning
    logger.info("Replenishment plan: %d stores", len(plan))
    logger.info("  CRITICAL: %d | WARNING: %d | OK: %d", critical, warning, ok)
    logger.info("Saved to %s", output)
