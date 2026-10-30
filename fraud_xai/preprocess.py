# fraud_xai/preprocess.py
import pandas as pd
import numpy as np
from .manual_ml import ManualScaler, manual_train_test_split
from dataclasses import dataclass

@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    strategy: str = "stratified"

    def sizes(self):
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}

    def fraud_counts(self):
        return {
            "train": int(np.sum(self.y_train == 1)),
            "val": int(np.sum(self.y_val == 1)),
            "test": int(np.sum(self.y_test == 1))
        }

def prepare_xy(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]).values
    y = df[target].values
    return X, y

def split_data(X, y, strategy="stratified", val_size=0.2, test_size=0.2, seed=42):
    # Manual split (val + test)
    X_tv, X_test, y_tv, y_test = manual_train_test_split(X, y, test_size=test_size, random_state=seed)
    
    # Manual split (train + val)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = manual_train_test_split(X_tv, y_tv, test_size=val_ratio, random_state=seed)
    
    # Manual Scaling
    scaler = ManualScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test, strategy=strategy)

def imbalance_summary(y):
    fraud = np.sum(y == 1)
    normal = len(y) - fraud
    return {"fraud": int(fraud), "normal": int(normal), "imbalance_ratio": normal / (fraud + 1e-8)}