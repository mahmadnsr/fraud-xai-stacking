import numpy as np

from fraud_xai.security import hash_password, verify_password
from fraud_xai.data import make_synthetic
from fraud_xai.evaluate import compute_metrics
from fraud_xai.preprocess import split_data


def test_password_hashing():
    password = "TestPassword@123"
    hashed = hash_password(password)
    assert verify_password(password, hashed)
    assert not verify_password("WrongPassword@123", hashed)


def test_synthetic_data_generation():
    frame = make_synthetic(n_rows=500, fraud_rate=0.05, seed=42)
    assert len(frame) == 500
    assert "Class" in frame.columns
    assert set(frame["Class"].unique()).issubset({0, 1})
    assert frame["Class"].sum() > 0


def test_metrics_return_valid_values():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    result = compute_metrics(y, probabilities)
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"):
        assert key in result
        assert np.isfinite(result[key])


def test_split_shapes_and_disjoint_sizes():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    y = np.array([0] * 180 + [1] * 20)

    splits = split_data(X, y, seed=42)

    assert len(splits.X_train) == len(splits.y_train)
    assert len(splits.X_val) == len(splits.y_val)
    assert len(splits.X_test) == len(splits.y_test)
    assert sum(splits.sizes().values()) == len(y)


def test_manual_boosters_are_distinct_and_predict_probabilities():
    from fraud_xai.manual_ml import ManualXGBoost, ManualLightGBM
    from fraud_xai.models import TrainConfig, build_base_models

    rng = np.random.default_rng(17)
    X = rng.normal(size=(240, 6))
    signal = X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2]
    y = (signal > np.quantile(signal, 0.78)).astype(int)

    xgb = ManualXGBoost(
        n_estimators=4, max_depth=2, scale_pos_weight=2.0,
        random_state=17
    ).fit(X, y)
    lgbm = ManualLightGBM(
        n_estimators=4, max_depth=3, max_leaves=5, max_bins=8,
        scale_pos_weight=2.0, random_state=17
    ).fit(X, y)

    px = xgb.predict_proba(X)
    pl = lgbm.predict_proba(X)

    assert px.shape == (len(y), 2)
    assert pl.shape == (len(y), 2)
    assert np.isfinite(px).all() and np.isfinite(pl).all()
    assert np.allclose(px.sum(axis=1), 1.0)
    assert np.allclose(pl.sum(axis=1), 1.0)
    assert type(xgb) is not type(lgbm)
    assert len(xgb.trees_) == 4
    assert len(lgbm.trees_) == 4

    factory = build_base_models(
        TrainConfig(seed=17, n_estimators=2), scale_pos_weight=2.0
    )
    assert isinstance(factory["xgboost"], ManualXGBoost)
    assert isinstance(factory["lightgbm"], ManualLightGBM)
    assert factory["xgboost"].scale_pos_weight == 2.0
    assert factory["lightgbm"].scale_pos_weight == 2.0


def test_manual_boosters_reject_invalid_inputs():
    import pytest
    from fraud_xai.manual_ml import ManualXGBoost, ManualLightGBM

    with pytest.raises(ValueError):
        ManualXGBoost(scale_pos_weight=0)
    with pytest.raises(ValueError):
        ManualLightGBM(max_bins=1)
    with pytest.raises(ValueError):
        ManualXGBoost().fit(np.ones((10, 3)), np.zeros(10))


def test_monte_carlo_shapley_is_reproducible_and_additive():
    from fraud_xai.explain import manual_shap_values, get_feature_importance

    class SimpleProbabilityModel:
        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            p = 1.0 / (1.0 + np.exp(-np.clip(X[:, 0] + 2 * X[:, 1], -30, 30)))
            return np.column_stack((1 - p, p))

    model = SimpleProbabilityModel()
    background = np.zeros((20, 2))
    instance = np.array([1.0, 1.0])

    values1, base1 = manual_shap_values(
        model, instance, background, n_samples=100, random_state=123
    )
    values2, base2 = manual_shap_values(
        model, instance, background, n_samples=100, random_state=123
    )

    prediction = model.predict_proba(instance.reshape(1, -1))[0, 1]
    assert np.allclose(values1, values2)
    assert base1 == base2
    assert abs(base1 + values1.sum() - prediction) < 1e-10

    importance1 = get_feature_importance(
        model, background, ["a", "b"], n_samples=5,
        max_rows=10, random_state=99
    )
    importance2 = get_feature_importance(
        model, background, ["a", "b"], n_samples=5,
        max_rows=10, random_state=99
    )
    assert list(importance1.columns) == ["feature", "importance"]
    assert np.allclose(importance1["importance"], importance2["importance"])
