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

    def sizes(self):
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}

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


def split_data(X, y, strategy="stratified", val_size=0.2, test_size=0.2, seed=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or len(X) != len(y) or not len(y):
        raise ValueError("X and y must be non-empty and have matching row counts.")
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("Validation and test sizes must be positive and sum to less than one.")
    if strategy not in ("stratified", "random"):
        raise ValueError("strategy must be 'stratified' or 'random'.")

    X_tv, X_test, y_tv, y_test = manual_train_test_split(
        X, y, test_size=test_size, random_state=seed, strategy=strategy
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = manual_train_test_split(
        X_tv, y_tv, test_size=val_ratio, random_state=seed + 1, strategy=strategy
    )

    scaler = ManualScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return Splits(X_train, X_val, X_test, y_train, y_val, y_test,
                  strategy=strategy, scaler=scaler)


def imbalance_summary(y):
    y = np.asarray(y)
    fraud = int(np.sum(y == 1))
    normal = int(len(y) - fraud)
    return {"fraud": fraud, "normal": normal,
            "imbalance_ratio": normal / max(fraud, 1)}
