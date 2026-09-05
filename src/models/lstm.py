"""LSTM forecaster with per-store sequence creation."""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _LSTMNet(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMForecaster(BaseForecaster):
    name = "lstm"

    def __init__(
        self,
        seq_len: int = 14,
        hidden_size: int = 64,
        n_layers: int = 2,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 512,
        max_train_samples: int = 200_000,
    ):
        if not HAS_TORCH:
            raise ImportError("torch is not installed: pip install torch")
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_samples = max_train_samples

    def _make_sequences(self, values, targets=None):
        X, y = [], []
        for t in range(self.seq_len, len(values)):
            X.append(values[t - self.seq_len : t])
            if targets is not None:
                y.append(targets[t])
        if not X:
            return np.empty((0, self.seq_len, values.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32) if targets is not None else None
        return X_arr, y_arr

    def fit(self, df_train, feature_cols, target_col):
        self.scaler_ = StandardScaler()
        self.y_mean_ = df_train[target_col].mean()
        self.y_std_ = max(df_train[target_col].std(), 1e-8)

        all_X, all_y = [], []
        for _, group in df_train.groupby("store_id"):
            group = group.sort_values("date")
            X_vals = group[feature_cols].fillna(0).values
            y_vals = group[target_col].values
            X_seq, y_seq = self._make_sequences(X_vals, y_vals)
            if len(X_seq) > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)

        X_all = np.concatenate(all_X)
        y_all = np.concatenate(all_y)

        if len(X_all) > self.max_train_samples:
            idx = np.random.default_rng(42).choice(len(X_all), self.max_train_samples, replace=False)
            X_all, y_all = X_all[idx], y_all[idx]

        n_samples, seq_len, n_feat = X_all.shape
        X_flat = X_all.reshape(-1, n_feat)
        X_flat = self.scaler_.fit_transform(X_flat)
        X_all = X_flat.reshape(n_samples, seq_len, n_feat)

        y_all = (y_all - self.y_mean_) / self.y_std_

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = device
        self.model_ = _LSTMNet(n_feat, self.hidden_size, self.n_layers).to(device)

        dataset = TensorDataset(
            torch.from_numpy(X_all),
            torch.from_numpy(y_all).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                pred = self.model_(X_b)
                loss = loss_fn(pred, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(X_b)
            if (epoch + 1) % 5 == 0:
                logger.info("LSTM epoch %d/%d  loss=%.4f", epoch + 1, self.epochs, total_loss / len(dataset))

        self.train_tail_ = (
            df_train.groupby("store_id")
            .apply(lambda g: g.sort_values("date").tail(self.seq_len), include_groups=False)
            .reset_index(level=0)
        )
        return self

    def predict(self, df_test, feature_cols):
        self.model_.eval()
        preds = np.full(len(df_test), self.y_mean_)

        for store_id, test_group in df_test.groupby("store_id"):
            tail = self.train_tail_[self.train_tail_["store_id"] == store_id]
            combined = pd.concat([tail, test_group]).sort_values("date")
            X_vals = combined[feature_cols].fillna(0).values

            X_flat = self.scaler_.transform(X_vals)
            X_scaled = X_flat

            test_start = len(tail)
            store_preds = []
            for t in range(test_start, len(combined)):
                if t >= self.seq_len:
                    seq = X_scaled[t - self.seq_len : t][np.newaxis]
                    with torch.no_grad():
                        p = self.model_(torch.from_numpy(seq.astype(np.float32)).to(self.device_))
                    store_preds.append(p.item() * self.y_std_ + self.y_mean_)
                else:
                    store_preds.append(self.y_mean_)

            for i, idx in enumerate(test_group.index):
                loc = df_test.index.get_loc(idx)
                if i < len(store_preds):
                    preds[loc] = store_preds[i]

        return np.clip(preds, 0, None)

    def get_params(self):
        return {
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "epochs": self.epochs,
            "lr": self.lr,
        }
