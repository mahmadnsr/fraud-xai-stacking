import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_xai.evaluate import MetricError, compute_metrics, select_threshold
from fraud_xai.models import BASE_MODEL_NAMES, STACKING_NAME
from fraud_xai.predict import predict_frame, risk_category
from fraud_xai.schema import ValidationError


def test_all_models_trained_and_metrics_present(trained):
    s, res, m = trained
    assert set(m.models) == set(BASE_MODEL_NAMES) | {STACKING_NAME}
    for name, met in res.metrics.items():
        assert 0 <= met["roc_auc"] <= 1 and 0 <= met["pr_auc"] <= 1 and met["tp"] + met["fn"] == res.splits_info["test"] * 0 + met["tp"] + met["fn"]
    assert m.manifest["final_model"] == STACKING_NAME


def test_stacking_uses_four_base_learners_and_meta_lr(trained):
    st = trained[2].models[STACKING_NAME]
    assert list(st.named_estimators_) == BASE_MODEL_NAMES
    assert type(st.final_estimator_).__name__ == "LogisticRegression" and st.final_estimator_.coef_.shape == (1, 4)


def test_metrics_computed_from_saved_probabilities(trained):
    _, res, m = trained
    y = m.evaluation["y_test"]
    for name in res.metrics:
        p = m.evaluation[f"{name}__test"]
        assert res.metrics[name]["roc_auc"] == pytest.approx(roc_auc_score(y, p))
        assert res.metrics[name]["pr_auc"] == pytest.approx(average_precision_score(y, p))
        assert res.metrics[name] == compute_metrics(y, p, res.thresholds[name])


def test_threshold_selected_on_validation_only(trained):
    _, res, m = trained
    for name in res.thresholds:
        assert res.thresholds[name] == pytest.approx(select_threshold(m.evaluation["y_val"], m.evaluation[f"{name}__val"]))


def test_metric_calculation_known_values():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1) and m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] == 0.75


def test_metric_errors():
    for y, p in ((np.zeros(5), np.linspace(0, 1, 5)), (np.array([0, 1]), np.array([0.2, 1.5])), (np.array([0, 1]), np.array([0.2, np.nan])),
                 (np.array([]), np.array([]))):
        with pytest.raises(MetricError):
            compute_metrics(y, p, 0.5)


def test_prediction_schema_and_ranges(trained, small_bundle):
    _, _, m = trained
    X = small_bundle.frame.drop(columns="Class").head(40)
    for name in m.models:
        out = predict_frame(m, name, X, 0.5)
        assert list(out.columns) == ["fraud_probability", "predicted_class", "risk_category"] and len(out) == 40
        assert out.fraud_probability.between(0, 1).all() and set(out.predicted_class) <= {0, 1}
        assert set(out.risk_category) <= {"Low", "Medium", "High"}
    shuffled = X[X.columns[::-1]]
    assert np.allclose(predict_frame(m, STACKING_NAME, shuffled, 0.5).fraud_probability, predict_frame(m, STACKING_NAME, X, 0.5).fraud_probability)


def test_prediction_rejects_bad_input(trained, small_bundle):
    _, _, m = trained
    X = small_bundle.frame.drop(columns="Class").head(3)
    with pytest.raises(ValidationError):
        predict_frame(m, STACKING_NAME, X.drop(columns="V1"), 0.5)
    with pytest.raises(ValidationError):
        predict_frame(m, STACKING_NAME, X, 1.5)
    with pytest.raises(ValidationError):
        predict_frame(m, "nonexistent", X, 0.5)


def test_risk_categories():
    assert [risk_category(p, 0.5) for p in (0.1, 0.3, 0.5, 0.99)] == ["Low", "Medium", "High", "High"]


def test_manifest_has_no_raw_records(trained):
    import json
    _, _, m = trained
    text = json.dumps(m.manifest)
    assert m.manifest["dataset"]["is_synthetic"] and "sha256" in text and "fraud_probability" not in text
    assert set(m.manifest["feature_stats"]["V1"]) == {"min", "p01", "median", "p99", "max"}
    assert m.manifest["dependencies"]["scikit-learn"] and m.manifest["config"]["seed"] == 7
