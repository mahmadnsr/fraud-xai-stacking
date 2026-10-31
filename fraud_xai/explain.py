"""Reproducible Monte Carlo approximation of Shapley feature attributions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_explanation_inputs(model, X_instance, X_background):
    instance = np.asarray(X_instance, dtype=float)
    background = np.asarray(X_background, dtype=float)

    if instance.ndim != 1:
        raise ValueError("X_instance must be a one-dimensional feature vector.")
    if background.ndim != 2 or len(background) == 0:
        raise ValueError("X_background must be a non-empty 2D array.")
    if background.shape[1] != len(instance):
        raise ValueError("Instance and background feature counts must match.")
    if not np.isfinite(instance).all() or not np.isfinite(background).all():
        raise ValueError("Explanation inputs must contain finite values.")
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must implement predict_proba.")

    return instance, background


def manual_shap_values(
    model, X_instance, X_background, n_samples=20, random_state=42
):
    """
    Estimate feature Shapley values using random feature permutations.

    Each permutation starts from a sampled background row and replaces
    features in permutation order. Contributions are averaged over samples.
    This is an approximation, not an exact SHAP implementation.
    """
    instance, background = _validate_explanation_inputs(
        model, X_instance, X_background
    )
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")

    rng = np.random.default_rng(random_state)
    n_features = len(instance)
    contributions = np.zeros(n_features, dtype=float)

    base_predictions = np.asarray(
        model.predict_proba(background), dtype=float
    )
    if (
        base_predictions.ndim != 2
        or base_predictions.shape[0] != len(background)
        or base_predictions.shape[1] < 2
        or not np.isfinite(base_predictions[:, 1]).all()
    ):
        raise ValueError("Model returned invalid background probabilities.")

    base_value = float(np.mean(base_predictions[:, 1]))

    for _ in range(n_samples):
        bg = background[int(rng.integers(len(background)))].copy()
        permutation = rng.permutation(n_features)
        hybrid = bg.copy()

        previous = float(model.predict_proba(hybrid.reshape(1, -1))[0, 1])

        for feature in permutation:
            hybrid[feature] = instance[feature]
            current = float(
                model.predict_proba(hybrid.reshape(1, -1))[0, 1]
            )
            contributions[feature] += current - previous
            previous = current

    contributions /= n_samples
    return contributions, base_value


def get_feature_importance(
    model,
    X_background,
    feature_names,
    n_samples=10,
    max_rows=50,
    random_state=42,
):
    """
    Global importance = mean absolute Monte Carlo Shapley approximation.

    Returns a DataFrame with `feature` and `importance` columns.
    """
    X = np.asarray(X_background, dtype=float)
    names = list(feature_names)

    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X_background must be a non-empty 2D array.")
    if X.shape[1] != len(names):
        raise ValueError("Feature names must match the input feature count.")
    if not np.isfinite(X).all():
        raise ValueError("Explanation data must contain finite values.")
    if not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer.")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")

    rng = np.random.default_rng(random_state)
    sample_size = min(max_rows, len(X))
    sample_indices = rng.choice(len(X), sample_size, replace=False)

    global_importance = np.zeros(X.shape[1], dtype=float)

    for index in sample_indices:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        values, _ = manual_shap_values(
            model,
            X[index],
            X,
            n_samples=n_samples,
            random_state=seed,
        )
        global_importance += np.abs(values)

    global_importance /= sample_size
    result = pd.DataFrame({
        "feature": names,
        "importance": global_importance,
    })
    return result.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
