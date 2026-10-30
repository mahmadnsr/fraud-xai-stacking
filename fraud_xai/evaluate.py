"""Metric computation (always from real predictions) and evaluation plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from .models import DISPLAY


class MetricError(Exception):
    """Metrics cannot be computed (e.g. a single class in y_true)."""


def _check(y_true, proba) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y_true), np.asarray(proba, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise MetricError("y_true and probabilities must be non-empty 1-D arrays of the same length.")
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise MetricError("Probabilities must be finite values in [0, 1].")
    if len(np.unique(y)) < 2:
        raise MetricError("Metrics need both classes in the evaluation labels.")
    return y, p


def select_threshold(y_val, p_val) -> float:
    """Threshold maximising F1 on the VALIDATION set. The test set is never passed to this function."""
    y, p = _check(y_val, p_val)
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y, p = _check(y_true, proba)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)),  # misleading under heavy imbalance
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def comparison_table(metrics: dict[str, dict]):
    import pandas as pd
    rows = [{"Model": DISPLAY.get(k, k), "Threshold": m["threshold"], "Precision": m["precision"], "Recall": m["recall"],
             "F1": m["f1"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"], "Accuracy*": m["accuracy"],
             "FP": m["fp"], "FN": m["fn"]} for k, m in metrics.items()]
    return pd.DataFrame(rows).set_index("Model")


def plot_confusion(m: dict, title: str):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
           xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    return fig


def plot_roc(y_true, probas: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{DISPLAY.get(name, name)} (AUC={roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_pr(y_true, probas: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=f"{DISPLAY.get(name, name)} (AP={average_precision_score(y_true, p):.3f})")
    ax.axhline(float(np.mean(y_true)), color="k", ls="--", lw=0.8, label="No-skill")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_metric_bars(metrics: dict[str, dict]):
    names = list(metrics)
    keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.16
    for i, k in enumerate(keys):
        ax.bar(np.arange(len(names)) + i * w, [metrics[n][k] for n in names], w, label=k)
    ax.set_xticks(np.arange(len(names)) + 2 * w, [DISPLAY.get(n, n) for n in names], rotation=15, fontsize=8)
    ax.set(ylim=(0, 1.05), title="Test-set metrics")
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout()
    return fig
