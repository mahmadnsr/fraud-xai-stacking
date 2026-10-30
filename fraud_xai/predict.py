"""Prediction with schema checks, probability validation and risk categories."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .persist import LoadedModel
from .schema import ValidationError, validate_feature_frame

RISK_LABELS = ("Low", "Medium", "High")


def risk_category(p: float, threshold: float) -> str:
    """Low: below half the decision threshold. Medium: below the threshold. High: at/above it."""
    return RISK_LABELS[2] if p >= threshold else RISK_LABELS[1] if p >= threshold / 2 else RISK_LABELS[0]


def predict_frame(model: LoadedModel, model_name: str, X: pd.DataFrame, threshold: float, max_rows: int = 100_000) -> pd.DataFrame:
    """Score a frame. Columns must match the training schema exactly (order is enforced here)."""
    if model_name not in model.models:
        raise ValidationError("Unknown model.")
    if not 0.0 < threshold < 1.0:
        raise ValidationError("Threshold must be strictly between 0 and 1.")
    X = validate_feature_frame(X, model.feature_names, max_rows)
    p = model.models[model_name].predict_proba(X)[:, 1]
    if p.shape != (len(X),) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise RuntimeError("Model returned invalid probabilities.")
    return pd.DataFrame({"fraud_probability": p, "predicted_class": (p >= threshold).astype(int),
                         "risk_category": [risk_category(v, threshold) for v in p]}, index=X.index)
