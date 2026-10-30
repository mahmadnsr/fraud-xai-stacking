import numpy as np
import pandas as pd
import pytest

from fraud_xai.explain import (ExplainError, describe_local, explain_ensemble, explain_tree, global_importance,
                               local_contributions, plot_local, plot_summary, positive_class_shap)
from fraud_xai.models import STACKING_NAME, TREE_MODELS


@pytest.mark.parametrize("name", TREE_MODELS)
def test_tree_shap_shapes_names_and_additivity(trained, name):
    _, _, m = trained
    _, ex = m.explain_sample()
    X = ex.head(15)
    e = explain_tree(m.models, name, X)
    assert e.values.shape == X.shape and e.feature_names == m.feature_names and np.isfinite(e.values).all()
    if name != "random_forest":   # boosting: base + sum(shap) == raw margin (log-odds)
        margin = m.models[name][-1].predict(m.models[name][:-1].transform(X), output_margin=True) if name == "xgboost" else \
            m.models[name][-1].predict(m.models[name][:-1].transform(X), raw_score=True)
        assert np.allclose(e.base_value + e.values.sum(axis=1), margin, atol=1e-3)
    else:                          # RF: base + sum(shap) == predicted probability
        p = m.models[name].predict_proba(X)[:, 1]
        assert np.allclose(e.base_value + e.values.sum(axis=1), p, atol=1e-3)
    assert global_importance(e).index[0] in m.feature_names
    assert len(describe_local(e, 0)) == 5 and len(local_contributions(e, 0)) == 10


def test_ensemble_kernel_shap(trained):
    _, _, m = trained
    bg, ex = m.explain_sample()
    e = explain_ensemble(m.models, ex.head(3), bg.head(15), nsamples=100)
    assert e.values.shape == (3, len(m.feature_names)) and e.output_space == "probability" and "Kernel" in e.method
    import matplotlib.pyplot as plt
    plt.close(plot_summary(e)); plt.close(plot_local(e, 0))


def test_tree_explainer_rejected_for_non_tree(trained):
    with pytest.raises(ExplainError):
        explain_tree(trained[2].models, "logistic_regression", pd.DataFrame())
    with pytest.raises(ExplainError):
        explain_tree(trained[2].models, STACKING_NAME, pd.DataFrame())


def test_positive_class_shap_normalisation():
    a = np.random.rand(4, 3)
    assert positive_class_shap([a * 0, a], 3).shape == (4, 3)
    assert np.array_equal(positive_class_shap(np.stack([a * 0, a], axis=-1), 3), a)
    with pytest.raises(ExplainError):
        positive_class_shap(a, 5)
