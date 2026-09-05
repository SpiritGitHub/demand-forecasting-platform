"""Seasonal Naive baseline: predict = same day last week."""

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster


class SeasonalNaive(BaseForecaster):
    name = "seasonal_naive"

    def fit(self, df_train, feature_cols, target_col):
        self.target_col_ = target_col
        self.store_means_ = df_train.groupby("store_id")[target_col].mean()
        return self

    def predict(self, df_test, feature_cols):
        if "sales_lag_7" in df_test.columns:
            preds = df_test["sales_lag_7"].copy()
        else:
            preds = pd.Series(np.nan, index=df_test.index)

        mask = preds.isna()
        if mask.any():
            preds[mask] = df_test.loc[mask, "store_id"].map(self.store_means_)

        return preds.fillna(self.store_means_.mean()).values

    def get_params(self):
        return {"method": "repeat_last_week"}
