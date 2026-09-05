"""Prophet forecaster (aggregate + per-store scaling)."""

import logging

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


class ProphetForecaster(BaseForecaster):
    name = "prophet"

    def __init__(self, yearly: bool = True, weekly: bool = True):
        if not HAS_PROPHET:
            raise ImportError("prophet is not installed: pip install prophet")
        self.yearly = yearly
        self.weekly = weekly

    def fit(self, df_train, feature_cols, target_col):
        daily = df_train.groupby("date").agg(
            y=(target_col, "mean"),
            promo=("promo", "mean"),
        ).reset_index().rename(columns={"date": "ds"})

        global_mean = daily["y"].mean()
        self.store_mult_ = (
            df_train.groupby("store_id")[target_col].mean() / global_mean
        )
        self.global_mean_ = global_mean

        self.model_ = Prophet(
            yearly_seasonality=self.yearly,
            weekly_seasonality=self.weekly,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        self.model_.add_regressor("promo")

        import logging as _log
        _log.getLogger("prophet").setLevel(_log.WARNING)
        _log.getLogger("cmdstanpy").setLevel(_log.WARNING)

        self.model_.fit(daily)
        return self

    def predict(self, df_test, feature_cols):
        daily = df_test.groupby("date").agg(
            promo=("promo", "mean"),
        ).reset_index().rename(columns={"date": "ds"})

        forecast = self.model_.predict(daily)
        yhat = forecast.set_index("ds")["yhat"]

        preds = (
            df_test["date"].map(yhat)
            * df_test["store_id"].map(self.store_mult_)
        )
        preds = preds.fillna(self.global_mean_)
        return preds.clip(lower=0).values

    def get_params(self):
        return {"yearly": self.yearly, "weekly": self.weekly}
