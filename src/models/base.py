"""Abstract base class for all forecasting models."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(
        self,
        df_train: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(
        self,
        df_test: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        ...

    def get_params(self) -> dict:
        return {}
