"""Ridge regression forecaster."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseForecaster


class RidgeForecaster(BaseForecaster):
    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, df_train, feature_cols, target_col):
        X = df_train[feature_cols].fillna(0).values
        y = df_train[target_col].values

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, df_test, feature_cols):
        X = df_test[feature_cols].fillna(0).values
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def get_params(self):
        return {"alpha": self.alpha}
