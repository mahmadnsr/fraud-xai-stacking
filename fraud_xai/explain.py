"""SHAP explanations.

* Random Forest / XGBoost / LightGBM  -> shap.TreeExplainer on the fitted classifier inside each pipeline.
* Stacking ensemble                   -> shap.KernelExplainer (model-agnostic) on the ensemble's predict_proba.
  A TreeExplainer for one base learner does NOT explain the stacked model, because the meta-learner mixes four
  models. KernelExplainer explains the full pipeline output (fraud probability) but is approximate and slow, so it
  is computed on a small sample with a small background set.
SHAP values describe how the MODEL's output depends on inputs; they do not establish causes of fraud.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .models import DISPLAY, STACKING_NAME, TREE_MODELS


class ExplainError(Exception):
    """SHAP output failed validation."""


@dataclass
class Explanation:
    values: np.ndarray            # (n_rows, n_features) contributions toward the fraud class
    base_value: float
    data: pd.DataFrame            # feature values in the same row/column order
    feature_names: list[str]
    model_name: str
    output_space: str             # "probability" or "log-odds"
    method: str


def positive_class_shap(values, n_features: int) -> np.ndarray:
    """Normalise the shapes returned by different SHAP/model versions to (n_rows, n_features) for class 1."""
    if isinstance(values, list):                       # older API: [class0, class1]
        values = values[-1]
    arr = np.asarray(values)
    if arr.ndim == 3:                                  # (rows, features, classes)
        arr = arr[:, :, -1]
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ExplainError(f"Unexpected SHAP output shape {arr.shape} for {n_features} features.")
    return arr


def _base(ev, n_classes_axis: bool = True) -> float:
    b = np.asarray(ev).ravel()
    return float(b[-1])


def _validate(values: np.ndarray, X: pd.DataFrame) -> None:
    if values.shape != X.shape:
        raise ExplainError("SHAP values do not match the input shape.")
    if not np.isfinite(values).all():
        raise ExplainError("SHAP values contain non-finite numbers.")


def explain_tree(models: dict, name: str, X: pd.DataFrame) -> Explanation:
    if name not in TREE_MODELS:
        raise ExplainError("Tree explanations are available for Random Forest, XGBoost and LightGBM only.")
    pipe = models[name]
    Xt = pipe[:-1].transform(X)                        # same fitted preprocessing used at inference
    explainer = shap.TreeExplainer(pipe[-1])
    values = positive_class_shap(explainer.shap_values(Xt), Xt.shape[1])
    _validate(values, Xt)
    space = "probability" if name == "random_forest" else "log-odds"
    return Explanation(values, _base(explainer.expected_value), Xt.reset_index(drop=True), list(Xt.columns), DISPLAY[name], space,
                       "TreeExplainer")


def explain_ensemble(models: dict, X: pd.DataFrame, background: pd.DataFrame, nsamples: int = 200) -> Explanation:
    """Model-agnostic KernelExplainer on the stacking ensemble's fraud probability."""
    stack, cols = models[STACKING_NAME], list(X.columns)

    def f(arr: np.ndarray) -> np.ndarray:
        return stack.predict_proba(pd.DataFrame(arr, columns=cols))[:, 1]

    explainer = shap.KernelExplainer(f, background[cols].to_numpy())
    values = positive_class_shap(explainer.shap_values(X[cols].to_numpy(), nsamples=nsamples, silent=True), len(cols))
    _validate(values, X)
    # Local accuracy check: base + sum(contributions) must reproduce the model output.
    recon = float(_base(explainer.expected_value)) + values.sum(axis=1)
    if not np.allclose(recon, f(X[cols].to_numpy()), atol=1e-3):
        raise ExplainError("SHAP additivity check failed for the ensemble.")
    return Explanation(values, _base(explainer.expected_value), X.reset_index(drop=True), cols, DISPLAY[STACKING_NAME],
                       "probability", f"KernelExplainer (nsamples={nsamples}, background={len(background)} rows)")


def global_importance(e: Explanation) -> pd.Series:
    return pd.Series(np.abs(e.values).mean(axis=0), index=e.feature_names).sort_values(ascending=False)


def plot_summary(e: Explanation, bar: bool = False, max_display: int = 15):
    plt.figure(figsize=(7, 5))
    shap.summary_plot(e.values, e.data, plot_type="bar" if bar else "dot", max_display=max_display, show=False)
    plt.title(f"{e.model_name}: SHAP {'mean |value|' if bar else 'summary'} ({e.output_space})", fontsize=10)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def local_contributions(e: Explanation, row: int, top: int = 10) -> pd.DataFrame:
    vals, data = e.values[row], e.data.iloc[row]
    df = pd.DataFrame({"feature": e.feature_names, "value": data.to_numpy(), "contribution": vals})
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def plot_local(e: Explanation, row: int, top: int = 10):
    df = local_contributions(e, row, top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(df) + 1.2))
    ax.barh([f"{f} = {v:.3g}" for f, v in zip(df["feature"], df["value"])], df["contribution"],
            color=["#d62728" if c > 0 else "#1f77b4" for c in df["contribution"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel=f"Contribution to model output ({e.output_space}); red raises fraud score", title=f"{e.model_name}: local explanation")
    fig.tight_layout()
    return fig


def describe_local(e: Explanation, row: int, top: int = 5) -> list[str]:
    """Plain-language sentences. Wording is deliberately about the model, not about real-world causes."""
    unit = "fraud probability" if e.output_space == "probability" else "fraud log-odds"
    out = []
    for r in local_contributions(e, row, top).itertuples():
        verb = "pushed the model's score UP" if r.contribution > 0 else "pushed the model's score DOWN"
        out.append(f"{r.feature} = {r.value:.4g} {verb} by {abs(r.contribution):.4f} ({unit}).")
    return out
