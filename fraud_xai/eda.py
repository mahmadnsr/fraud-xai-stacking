"""Exploratory data analysis plots (computed from the validated dataset in memory)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_balance(counts: dict):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Legitimate", "Fraud"], [counts[0], counts[1]], color=["#1f77b4", "#d62728"])
    ax.set_yscale("log")
    ax.set(ylabel="Records (log scale)", title="Class distribution")
    for i, v in enumerate([counts[0], counts[1]]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_distribution(frame: pd.DataFrame, feature: str, target: str):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    data = frame[feature].dropna()
    lo, hi = data.quantile([0.005, 0.995])
    bins = np.linspace(lo, hi, 40) if hi > lo else 20
    for cls, color, name in ((0, "#1f77b4", "Legitimate"), (1, "#d62728", "Fraud")):
        ax.hist(frame.loc[frame[target] == cls, feature].dropna().clip(lo, hi), bins=bins, alpha=0.6, density=True, color=color, label=name)
    ax.set(xlabel=feature, ylabel="Density", title=f"{feature} by class (0.5-99.5 percentile range)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_correlations(frame: pd.DataFrame, target: str, max_features: int = 20):
    corr_t = frame.drop(columns=[target]).corrwith(frame[target]).abs().sort_values(ascending=False)
    cols = list(corr_t.head(max_features).index) + [target]
    corr = frame[cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cols)), cols, fontsize=6)
    ax.set_title(f"Correlation of the {max_features} features most correlated with {target}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def plot_amount_by_class(frame: pd.DataFrame, target: str, col: str = "Amount"):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.boxplot([np.log1p(frame.loc[frame[target] == c, col].clip(lower=0).dropna()) for c in (0, 1)], tick_labels=["Legitimate", "Fraud"])
    ax.set(ylabel=f"log(1 + {col})", title=f"{col} by class")
    fig.tight_layout()
    return fig


def plot_fraud_rate_over_time(frame: pd.DataFrame, target: str, col: str = "Time", bins: int = 24):
    grp = pd.cut(frame[col], bins=bins)
    rate = frame.groupby(grp, observed=True)[target].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(range(len(rate)), rate.to_numpy() * 100, marker="o", ms=3)
    ax.set(xlabel=f"{col} bin (chronological)", ylabel="Fraud rate (%)", title="Fraud rate over time")
    fig.tight_layout()
    return fig
