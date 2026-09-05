"""XGBoost forecaster."""

import numpy as np
import xgboost as xgb

from src.models.base import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def fit(self, df_train, feature_cols, target_col):
        X = df_train[feature_cols].fillna(0).values
        y = df_train[target_col].values

        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method="hist",
            random_state=42,
            verbosity=0,
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
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }
