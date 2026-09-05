"""Phase 3 training pipeline: 7 models + walk-forward backtesting + MLflow."""

import logging

import mlflow
import pandas as pd

from src.config import FEATURES_DIR, MLFLOW_TRACKING_URI, RESULTS_DIR
from src.evaluation.backtesting import WalkForwardSplitter, run_backtesting
from src.models.naive import SeasonalNaive
from src.models.linear import RidgeForecaster
from src.models.random_forest import RandomForestForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.lightgbm_model import LightGBMForecaster

logger = logging.getLogger(__name__)

TARGET = "sales"

FEATURE_COLS = [
    "open", "promo", "school_holiday",
    "day_of_week", "month", "week_of_year",
    "is_weekend", "is_month_start", "is_month_end", "is_holiday",
    "days_to_holiday",
    "temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max",
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_mean_14", "sales_rolling_mean_28",
    "sales_rolling_std_7", "sales_ewm_7", "sales_trend",
    "promo_duration",
    "competition_distance_log", "competition_open_months",
    "store_type_enc", "assortment_enc",
    "promo_x_weekend", "promo_x_holiday", "holiday_x_store_type", "dow_x_promo",
    "days_before_holiday", "days_after_holiday",
    "store_mean_sales", "store_median_sales", "store_std_sales",
    "store_dow_mean_sales", "store_dow_ratio",
]


def get_available_features(df, candidates):
    return [c for c in candidates if c in df.columns]


def build_model_list():
    models = [
        SeasonalNaive(),
        RidgeForecaster(alpha=1.0),
        RandomForestForecaster(n_estimators=200, max_depth=15),
        XGBoostForecaster(n_estimators=300, max_depth=8, learning_rate=0.05),
        LightGBMForecaster(n_estimators=300, max_depth=8, learning_rate=0.05),
    ]

    try:
        from src.models.prophet_model import ProphetForecaster
        models.append(ProphetForecaster())
    except ImportError:
        logger.warning("Prophet not available, skipping")

    try:
        from src.models.lstm import LSTMForecaster
        models.append(LSTMForecaster(seq_len=14, hidden_size=64, epochs=20))
    except ImportError:
        logger.warning("PyTorch not available, skipping LSTM")

    return models


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    input_path = FEATURES_DIR / "rossmann_features.parquet"
    logger.info("Loading features from %s", input_path)
    df = pd.read_parquet(input_path)

    df = df[(df["open"] == 1) & (df["sales"] > 0)].copy()
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
    logger.info("Dataset: %d rows, %d columns", len(df), len(df.columns))

    feature_cols = get_available_features(df, FEATURE_COLS)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    models = build_model_list()
    logger.info("Models: %s", [m.name for m in models])

    splitter = WalkForwardSplitter(n_splits=3, test_horizon_days=14, gap_days=1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("demand-forecasting-phase3")

    with mlflow.start_run(run_name="walk-forward-backtesting"):
        mlflow.log_param("n_splits", splitter.n_splits)
        mlflow.log_param("test_horizon_days", splitter.test_horizon_days)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_models", len(models))

        results = run_backtesting(df, models, splitter, feature_cols, TARGET)

        summary = results.groupby("model")[["mae", "rmse", "wape", "bias"]].mean()
        for model_name, row in summary.iterrows():
            for metric, value in row.items():
                if pd.notna(value):
                    mlflow.log_metric(f"{model_name}_{metric}", value)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / "backtesting_results.csv"
        results.to_csv(output_path, index=False)
        mlflow.log_artifact(str(output_path))

        logger.info("\n=== Backtesting Summary (mean across folds) ===")
        logger.info("\n%s", summary.round(2).to_string())
        logger.info("Results saved to %s", output_path)

    return results


if __name__ == "__main__":
    main()
