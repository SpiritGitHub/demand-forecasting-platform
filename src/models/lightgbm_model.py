"""LightGBM forecaster."""

import numpy as np
import lightgbm as lgb

from src.models.base import BaseForecaster


class LightGBMForecaster(BaseForecaster):
    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        num_leaves: int = 63,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.num_leaves = num_leaves

    def fit(self, df_train, feature_cols, target_col):
        X = df_train[feature_cols].fillna(0).values
        y = df_train[target_col].values

        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            num_leaves=self.num_leaves,
            random_state=42,
            verbosity=-1,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, df_test, feature_cols):
        X = df_test[feature_cols].fillna(0).values
        return self.model_.predict(X)

    def get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
        }
