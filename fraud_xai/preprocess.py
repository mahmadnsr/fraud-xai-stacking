"""Leakage-safe preprocessing: splitting plus learned transforms that live inside sklearn Pipelines."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import ValidationError


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    strategy: str

    def sizes(self) -> dict:
        return {k: int(len(getattr(self, f"y_{k}"))) for k in ("train", "val", "test")}

    def fraud_counts(self) -> dict:
        return {k: int(getattr(self, f"y_{k}").sum()) for k in ("train", "val", "test")}


def prepare_xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Drop exact duplicate rows (so identical records cannot land in both train and test) and separate X / y."""
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame.drop(columns=[target]), frame[target].astype(int)


def imbalance_summary(y: pd.Series) -> dict:
    pos, neg = int(y.sum()), int((y == 0).sum())
    return {"legitimate": neg, "fraud": pos, "fraud_rate": pos / max(len(y), 1), "imbalance_ratio": neg / max(pos, 1),
            "scale_pos_weight": neg / max(pos, 1)}


def split_data(X: pd.DataFrame, y: pd.Series, strategy: str = "stratified", val_size: float = 0.2,
               test_size: float = 0.2, seed: int = 42) -> Splits:
    """Train / validation / test split.

    stratified    - random split preserving the fraud rate in every part (default for i.i.d. data).
    chronological - requires a 'Time' column; oldest rows train, newest rows test. Prefer this when the
                    deployment scenario is "predict tomorrow's fraud from yesterday's data".
    The test set is only used for the final report; thresholds are chosen on the validation set.
    """
    if not 0 < val_size < 0.5 or not 0 < test_size < 0.5:
        raise ValidationError("val_size and test_size must each be between 0 and 0.5.")
    if strategy == "stratified":
        X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_size / (1 - test_size), stratify=y_tmp, random_state=seed)
    elif strategy == "chronological":
        if "Time" not in X.columns:
            raise ValidationError("Chronological splitting requires a 'Time' column.")
        order = X["Time"].argsort(kind="stable").to_numpy()
        X, y = X.iloc[order], y.iloc[order]
        n = len(X)
        a, b = int(n * (1 - val_size - test_size)), int(n * (1 - test_size))
        X_train, X_val, X_test = X.iloc[:a], X.iloc[a:b], X.iloc[b:]
        y_train, y_val, y_test = y.iloc[:a], y.iloc[a:b], y.iloc[b:]
    else:
        raise ValidationError("Unknown split strategy.")
    for name, part in (("train", y_train), ("validation", y_val), ("test", y_test)):
        if part.nunique() < 2:
            raise ValidationError(f"The {name} split contains only one class; use a larger dataset or another split strategy.")
    s = Splits(X_train, X_val, X_test, y_train, y_val, y_test, strategy)
    assert_disjoint(s)
    return s


def assert_disjoint(s: Splits) -> None:
    """Hard leakage guard: no row index may appear in more than one split."""
    tr, va, te = set(s.X_train.index), set(s.X_val.index), set(s.X_test.index)
    if tr & va or tr & te or va & te:
        raise AssertionError("Data leakage: splits overlap.")


def make_preprocessor(scale: bool) -> Pipeline:
    """Median imputation (+ standard scaling for linear models). Fitted on training folds only because it
    is embedded in each model Pipeline and therefore re-fitted inside every cross-validation fold."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")
