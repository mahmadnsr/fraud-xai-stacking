import numpy as np
import pandas as pd
import pytest

from fraud_xai.data import make_synthetic
from fraud_xai.models import TrainConfig, build_base_models, build_stacking
from fraud_xai.preprocess import Splits, assert_disjoint, make_preprocessor, prepare_xy, split_data
from fraud_xai.schema import ValidationError


@pytest.fixture(scope="module")
def xy():
    return prepare_xy(make_synthetic(2000, 0.05, seed=5), "Class")


def test_stratified_split_proportions_and_disjoint(xy):
    X, y = xy
    s = split_data(X, y, "stratified", 0.2, 0.2, 1)
    assert sum(s.sizes().values()) == len(X)
    for part in (s.y_train, s.y_val, s.y_test):
        assert abs(part.mean() - y.mean()) < 0.01
    assert_disjoint(s)
    bad = Splits(s.X_train, s.X_train, s.X_test, s.y_train, s.y_train, s.y_test, "x")
    with pytest.raises(AssertionError):
        assert_disjoint(bad)


def test_no_duplicate_rows_across_splits():
    df = make_synthetic(1500, 0.05, seed=2)
    df = pd.concat([df, df.sample(300, random_state=1)])            # inject exact duplicates
    X, y = prepare_xy(df, "Class")
    s = split_data(X, y, "stratified", 0.2, 0.2, 1)
    tr = set(map(tuple, s.X_train.round(6).to_numpy())); te = set(map(tuple, s.X_test.round(6).to_numpy()))
    assert not tr & te


def test_chronological_split_orders_time(xy):
    X, y = xy
    s = split_data(X.sample(frac=1, random_state=0), y, "chronological", 0.2, 0.2, 1)
    assert s.X_train.Time.max() <= s.X_val.Time.min() and s.X_val.Time.max() <= s.X_test.Time.min()
    with pytest.raises(ValidationError):
        split_data(X.drop(columns="Time"), y, "chronological")
    with pytest.raises(ValidationError):
        split_data(X, y, "bogus")


def test_preprocessor_fit_on_train_only(xy):
    X, y = xy
    s = split_data(X, y, "stratified", 0.2, 0.2, 1)
    Xn = s.X_train.copy(); Xn.iloc[:50, 3] = np.nan
    pre = make_preprocessor(True).fit(Xn)
    assert pre.named_steps["impute"].statistics_[3] == pytest.approx(Xn.iloc[:, 3].median())
    assert pre.named_steps["impute"].statistics_[3] != pytest.approx(X.iloc[:, 3].median())
    Xt = pre.transform(s.X_test)                                     # inference reuses fitted state
    assert list(Xt.columns) == list(X.columns) and Xt.shape == s.X_test.shape


def test_scaling_only_for_logistic_regression():
    base = build_base_models(TrainConfig(), 10.0)
    assert "scale" in base["logistic_regression"]["prep"].named_steps
    for n in ("random_forest", "xgboost", "lightgbm"):
        assert "scale" not in base[n]["prep"].named_steps


def test_stack_structure():
    st = build_stacking(TrainConfig(cv_folds=4), 10.0)
    assert [n for n, _ in st.estimators] == ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    assert type(st.final_estimator).__name__ == "LogisticRegression"
    assert st.stack_method == "predict_proba" and st.passthrough is False and st.cv.n_splits == 4 and st.cv.shuffle
