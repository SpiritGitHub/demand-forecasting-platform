"""Random Forest forecaster."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseForecaster


class RandomForestForecaster(BaseForecaster):
    name = "random_forest"

    def __init__(self, n_estimators: int = 200, max_depth: int = 15, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs

    def fit(self, df_train, feature_cols, target_col):
        X = df_train[feature_cols].fillna(0).values
        y = df_train[target_col].values

        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=42,
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
        }
