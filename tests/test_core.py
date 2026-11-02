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



def test_chronological_split_preserves_time_order():
    n = 200
    time_values = np.arange(n)
    X = np.column_stack([time_values, time_values * 2])
    y = np.array([0, 1] * 100)

    splits = split_data(
        X,
        y,
        strategy="chronological",
        val_size=0.2,
        test_size=0.2,
        time_values=time_values,
    )

    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == n
    assert splits.strategy == "chronological"

    # Recover original ordering from the first feature.
    train_times = splits.X_train[:, 0]
    val_times = splits.X_val[:, 0]
    test_times = splits.X_test[:, 0]

    assert train_times.max() < val_times.min()
    assert val_times.max() < test_times.min()


def test_chronological_split_fits_scaler_on_training_data_only():
    n = 100
    time_values = np.arange(n)
    X = np.column_stack([
        np.arange(n, dtype=float),
        np.ones(n, dtype=float),
    ])
    X[-20:, 0] = 10000.0
    y = np.array([0, 1] * 50)

    splits = split_data(
        X,
        y,
        strategy="chronological",
        val_size=0.2,
        test_size=0.2,
        time_values=time_values,
    )

    # The training partition contains rows 0 through 59.
    assert np.isclose(splits.scaler.mean_[0], np.mean(np.arange(60)))
    assert splits.scaler.mean_[0] < 10000


def test_chronological_split_rejects_unsorted_time():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0, 1] * 10)
    time_values = np.arange(20)
    time_values[10], time_values[11] = time_values[11], time_values[10]

    try:
        split_data(
            X,
            y,
            strategy="chronological",
            time_values=time_values,
        )
    except ValueError as exc:
        assert "sorted by ascending time" in str(exc)
    else:
        raise AssertionError("Unsorted timestamps should be rejected.")



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


def test_chronological_stacking_never_trains_on_future_rows():
    from fraud_xai.manual_ml import ManualStackingClassifier

    class RecordingBaseModel:
        events = []

        def fit(self, X, y):
            self.events.append(("fit", float(np.max(X[:, 0])))
            )
            return self

        def predict_proba(self, X):
            self.events.append(("predict", float(np.min(X[:, 0])))
            )
            p = 1.0 / (1.0 + np.exp(-np.clip(X[:, 0], -30, 30)))
            return np.column_stack((1.0 - p, p))

    class SimpleMetaModel:
        def fit(self, X, y):
            assert X.shape[0] == len(y)
            assert X.shape[1] == 1
            assert len(np.unique(y)) == 2
            self.fitted = True
            return self

        def predict_proba(self, X):
            p = np.full(len(X), 0.5)
            return np.column_stack((1.0 - p, p))

    RecordingBaseModel.events = []
    X = np.arange(120, dtype=float).reshape(-1, 1)
    y = np.array([0, 1] * 60)
    # Two consecutive rows share each timestamp.
    # OOF boundaries must keep each pair together.
    time_values = np.repeat(np.arange(60, dtype=float), 2)

    model = ManualStackingClassifier(
        base_models=[RecordingBaseModel()],
        meta_model=SimpleMetaModel(),
        cv=3,
        split_strategy="chronological",
    )
    model.fit(X, y, time_values=time_values)

    # Every OOF training window ends after a complete timestamp pair;
    # validation begins at the first row of a later timestamp pair.
    oof_events = RecordingBaseModel.events[:-1]  # Exclude final full-data fit.
    assert len(oof_events) == 2 * 3
    for i in range(0, len(oof_events), 2):
        fit_event, predict_event = oof_events[i:i + 2]
        assert fit_event[0] == "fit"
        assert predict_event[0] == "predict"
        assert int(fit_event[1]) % 2 == 1
        assert int(predict_event[1]) % 2 == 0
        assert fit_event[1] < predict_event[1]

    events = RecordingBaseModel.events

    # Three OOF folds: each training window must end before its
    # corresponding validation block begins.
    for i in range(3):
        fit_event = events[2 * i]
        predict_event = events[2 * i + 1]

        assert fit_event[0] == "fit"
        assert predict_event[0] == "predict"
        assert fit_event[1] < predict_event[1]

    # The final event is the expected refit on the complete training set.
    assert events[-1] == ("fit", 119.0)
    assert model.meta_model.fitted


def test_chronological_oof_uses_fold_specific_class_weights():
    from fraud_xai.manual_ml import ManualStackingClassifier

    class RecordingWeightedModel:
        events = []

        def __init__(self, scale_pos_weight=99.0):
            self.scale_pos_weight = scale_pos_weight

        def fit(self, X, y):
            self.events.append((
                len(y),
                float(self.scale_pos_weight),
                int(np.sum(y == 0)),
                int(np.sum(y == 1)),
            ))
            return self

        def predict_proba(self, X):
            p = np.full(len(X), 0.5)
            return np.column_stack((1.0 - p, p))

    class SimpleMetaModel:
        def fit(self, X, y):
            assert len(np.unique(y)) == 2
            return self

        def predict_proba(self, X):
            p = np.full(len(X), 0.5)
            return np.column_stack((1.0 - p, p))

    X = np.arange(120, dtype=float).reshape(-1, 1)
    y = np.zeros(120, dtype=int)
    y[[1, 4, 8, 15, 25, 35, 50, 70, 90, 110]] = 1
    time_values = np.repeat(np.arange(60, dtype=float), 2)

    RecordingWeightedModel.events = []
    model = ManualStackingClassifier(
        base_models=[RecordingWeightedModel()],
        meta_model=SimpleMetaModel(),
        cv=3,
        split_strategy="chronological",
    )
    model.fit(X, y, time_values=time_values)

    expected_sizes = [30, 60, 90]
    for event, expected_size in zip(
        RecordingWeightedModel.events[:3], expected_sizes
    ):
        n_rows, weight, negatives, positives = event
        assert n_rows == expected_size
        assert positives > 0 and negatives > 0
        assert np.isclose(weight, negatives / positives)

    final_event = RecordingWeightedModel.events[-1]
    assert final_event[0] == len(y)
    assert np.isclose(
        final_event[1],
        np.sum(y == 0) / np.sum(y == 1),
    )
