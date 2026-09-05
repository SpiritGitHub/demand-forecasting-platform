"""Walk-forward temporal backtesting."""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """Time-series cross-validation stepping backward from the dataset end.

    Each fold: train = all data before cutoff, test = next horizon days.
    Folds are returned in chronological order (fold 0 has the least training data).
    """

    def __init__(self, n_splits: int = 3, test_horizon_days: int = 14, gap_days: int = 1):
        self.n_splits = n_splits
        self.test_horizon_days = test_horizon_days
        self.gap_days = gap_days

    def split(self, df: pd.DataFrame, date_col: str = "date"):
        dates = df[date_col]
        max_date = dates.max()
        min_date = dates.min()

        total_test = self.n_splits * self.test_horizon_days
        first_test_start = max_date - timedelta(days=total_test - 1)

        if first_test_start <= min_date + timedelta(days=90):
            raise ValueError(
                f"Not enough data for {self.n_splits} folds of "
                f"{self.test_horizon_days} days. Reduce n_splits or test_horizon_days."
            )

        splits = []
        for i in range(self.n_splits):
            test_start = first_test_start + timedelta(days=i * self.test_horizon_days)
            test_end = test_start + timedelta(days=self.test_horizon_days - 1)
            train_end = test_start - timedelta(days=self.gap_days)

            train_mask = dates <= train_end
            test_mask = (dates >= test_start) & (dates <= test_end)

            train_idx = df[train_mask].index
            test_idx = df[test_mask].index

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            splits.append((train_idx, test_idx, train_end))
            logger.info(
                "Fold %d: train [%s -> %s] (%d rows)  test [%s -> %s] (%d rows)",
                i,
                min_date.date(),
                train_end.date(),
                len(train_idx),
                test_start.date(),
                test_end.date(),
                len(test_idx),
            )

        return splits


def run_backtesting(
    df: pd.DataFrame,
    models: list,
    splitter: WalkForwardSplitter,
    feature_cols: list[str],
    target_col: str = "sales",
) -> pd.DataFrame:
    """Run walk-forward backtesting for a list of models.

    Returns a DataFrame with per-fold, per-model metrics.
    """
    splits = splitter.split(df)
    results = []

    for fold_idx, (train_idx, test_idx, train_end) in enumerate(splits):
        df_train = df.loc[train_idx].copy()
        df_test = df.loc[test_idx].copy()

        for model in models:
            logger.info("Fold %d | %s ...", fold_idx, model.name)

            try:
                model.fit(df_train, feature_cols, target_col)
                y_pred = model.predict(df_test, feature_cols)
                y_true = df_test[target_col].values

                y_pred = np.clip(np.nan_to_num(y_pred, nan=0.0), 0, None)

                metrics = compute_all_metrics(y_true, y_pred)
                metrics.update(
                    model=model.name,
                    fold=fold_idx,
                    train_end=str(train_end.date()),
                    n_train=len(df_train),
                    n_test=len(df_test),
                )
                results.append(metrics)
                logger.info(
                    "  -> MAE=%.1f  RMSE=%.1f  WAPE=%.3f  Bias=%.3f",
                    metrics["mae"],
                    metrics["rmse"],
                    metrics["wape"],
                    metrics["bias"],
                )

            except Exception as e:
                logger.error("  -> FAILED: %s", e)
                results.append({
                    "model": model.name,
                    "fold": fold_idx,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "wape": np.nan,
                    "bias": np.nan,
                    "train_end": str(train_end.date()),
                    "n_train": len(df_train),
                    "n_test": len(df_test),
                    "error": str(e),
                })

    return pd.DataFrame(results)
