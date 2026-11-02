"""Leakage-aware splitting and preprocessing."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .manual_ml import ManualScaler, manual_train_test_split


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    strategy: str = "stratified"
    scaler: ManualScaler | None = None
    train_time_values: np.ndarray | None = None

    def sizes(self):
        return {
            "train": len(self.y_train),
            "val": len(self.y_val),
            "test": len(self.y_test),
        }

    def fraud_counts(self):
        return {
            "train": int(np.sum(self.y_train == 1)),
            "val": int(np.sum(self.y_val == 1)),
            "test": int(np.sum(self.y_test == 1)),
        }


def prepare_xy(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)
    return X, y


def _chronological_boundaries(n, val_size, test_size, time_values):
    """Calculate chronological boundaries without splitting equal timestamps."""
    train_end = int(n * (1.0 - val_size - test_size))
    val_end = int(n * (1.0 - test_size))

    if not (0 < train_end < val_end < n):
        raise ValueError(
            "Dataset is too small for the requested chronological split ratios."
        )
    # Move a boundary forward to keep identical timestamps together.
    if time_values is not None:
        while train_end < n and time_values[train_end] == time_values[train_end - 1]:
            train_end += 1

        while val_end < n and time_values[val_end] == time_values[val_end - 1]:
            val_end += 1

    if not (0 < train_end < val_end < n):
        raise ValueError(
            "Chronological split produced an empty partition. "
            "Check dataset size, split ratios, and timestamp groups."
        )

    return train_end, val_end


def split_data(
    X,
    y,
    strategy="stratified",
    val_size=0.2,
    test_size=0.2,
    seed=42,
    time_values=None,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if X.ndim != 2 or len(X) != len(y) or not len(y):
        raise ValueError("X and y must be non-empty and have matching row counts.")

    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("Validation and test sizes must be positive and sum to less than one.")

    if strategy not in ("stratified", "random", "chronological"):
        raise ValueError("strategy must be 'stratified', 'random', or 'chronological'.")

    if strategy == "chronological":
        if time_values is None:
            raise ValueError(
                "time_values are required for chronological splitting."
            )

        time_values = np.asarray(time_values)
        if time_values.ndim != 1 or len(time_values) != len(y):
            raise ValueError("time_values must be one-dimensional and match the row count.")

        if pd.isna(time_values).any():
            raise ValueError("time_values cannot contain missing values.")

        if np.any(time_values[1:] < time_values[:-1]):
            raise ValueError(
                "Rows must be sorted by ascending time before chronological splitting."
            )

        train_end, val_end = _chronological_boundaries(
            len(y), val_size, test_size, time_values
        )

        train_time_values = time_values[:train_end].copy()

        X_train, X_val, X_test = (
            X[:train_end],
            X[train_end:val_end],
            X[val_end:],
        )
        y_train, y_val, y_test = (
            y[:train_end],
            y[train_end:val_end],
            y[val_end:],
        )

    else:
        X_tv, X_test, y_tv, y_test = manual_train_test_split(
            X, y, test_size=test_size, random_state=seed, strategy=strategy
        )
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = manual_train_test_split(
            X_tv,
            y_tv,
            test_size=val_ratio,
            random_state=seed + 1,
            strategy=strategy,
        )

    if strategy != "chronological":
        train_time_values = None

    # Fit preprocessing only on training data to prevent leakage.
    scaler = ManualScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return Splits(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        strategy=strategy,
        scaler=scaler,
        train_time_values=train_time_values,
    )


def imbalance_summary(y):
    y = np.asarray(y)
    fraud = int(np.sum(y == 1))
    normal = int(len(y) - fraud)
    return {
        "fraud": fraud,
        "normal": normal,
        "imbalance_ratio": normal / max(fraud, 1),
    }