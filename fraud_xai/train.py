"""End-to-end training: validation -> split -> four base models -> stacking -> evaluation -> signed artifacts."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import Settings
from .data import DatasetBundle
from .evaluate import compute_metrics, select_threshold
from .models import BASE_MODEL_NAMES, DISPLAY, STACKING_NAME, TrainConfig, build_base_models, build_stacking, describe_params
from .persist import save_artifacts
from .preprocess import Splits, imbalance_summary, prepare_xy, split_data

log = logging.getLogger("fraud_xai.train")
Progress = Callable[[str], None]
EXPLAIN_ROWS, BACKGROUND_ROWS = 200, 100

# Safeguard: Manual SHAP is slow, so we evaluate it on a subset of the test set
SHAP_EVAL_ROWS = 300

@dataclass
class TrainResult:
    version: str
    metrics: dict
    metrics_at_half: dict
    thresholds: dict
    fit_seconds: dict
    splits_info: dict


def train_all(bundle: DatasetBundle, cfg: TrainConfig, settings: Settings, progress: Progress | None = None) -> TrainResult:
    say = progress or (lambda m: None)
    X, y = prepare_xy(bundle.frame, bundle.target)
    imb = imbalance_summary(y)
    say(f"Prepared {len(X):,} records ({imb['fraud']:,} fraud, imbalance ratio {imb['imbalance_ratio']:.1f}:1)")



    time_values = None
    if cfg.split_strategy == "chronological":
        if "Time" not in bundle.frame.columns:
            raise ValueError(
                "Chronological splitting requires a 'Time' column in the dataset."
            )
        time_values = bundle.frame["Time"].to_numpy()

    s: Splits = split_data(
        X,
        y,
        cfg.split_strategy,
        cfg.val_size,
        cfg.test_size,
        cfg.seed,
        time_values=time_values,
    )
    say(f"Split ({s.strategy}): {s.sizes()} fraud per split: {s.fraud_counts()}")
    spw = float((s.y_train == 0).sum() / max((s.y_train == 1).sum(), 1))

    models, fit_seconds = {}, {}
    base = build_base_models(cfg, spw)

    # 1. Train Base Models
    for name in BASE_MODEL_NAMES:
        say(f"Training {DISPLAY[name]} ...")
        t = time.perf_counter()
        try:
            models[name] = base[name].fit(s.X_train, s.y_train)
        except Exception as exc:
            log.exception("training failed for %s", name)
            raise RuntimeError(f"Training failed for {DISPLAY[name]} ({type(exc).__name__}).") from exc
        fit_seconds[name] = time.perf_counter() - t

    # 2. Train Standard Stacking (Baseline)
    stack_method_label = (f"{cfg.cv_folds} timestamp-group-aware expanding-window chronological OOF blocks" if cfg.split_strategy == "chronological" else f"{cfg.cv_folds}-fold stratified OOF")
    say(f"Training standard stacking ensemble ({stack_method_label}, Logistic Regression meta-learner) ...")
    t = time.perf_counter()
    try:
        stack_model = build_stacking(cfg, spw)
        if cfg.split_strategy == "chronological":
            models[STACKING_NAME] = stack_model.fit(
                s.X_train, s.y_train, time_values=s.train_time_values
            )
        else:
            models[STACKING_NAME] = stack_model.fit(s.X_train, s.y_train)
    except Exception as exc:
        log.exception("stacking training failed")
        raise RuntimeError(f"Training failed for the stacking ensemble ({type(exc).__name__}).") from exc
    fit_seconds[STACKING_NAME] = time.perf_counter() - t

    # SHAP-weighted soft voting is excluded from the validated training
    # pipeline until its weighting hypothesis has been experimentally tested.

    # 4. Evaluation Setup
    say("Evaluating on validation (threshold selection) and held-out test data ...")
    ev, thresholds, metrics, metrics_half = {"y_val": s.y_val, "y_test": s.y_test}, {}, {}, {}

    # Every retained model uses identical validation and held-out test rows.
    for name, model in models.items():
        pv = model.predict_proba(s.X_val)[:, 1]
        pt = model.predict_proba(s.X_test)[:, 1]

        ev[f"{name}__val"], ev[f"{name}__test"] = pv, pt
        thresholds[name] = select_threshold(s.y_val, pv)
        metrics[name] = compute_metrics(s.y_test, pt, thresholds[name])
        metrics_half[name] = compute_metrics(s.y_test, pt, 0.5)

    # 5. Artifact Generation
    rng = np.random.default_rng(cfg.seed)
    feature_names = list(bundle.frame.drop(columns=[bundle.target]).columns)

    bg_idx = rng.choice(len(s.X_train), min(BACKGROUND_ROWS, len(s.X_train)), replace=False)
    bg = pd.DataFrame(s.X_train[bg_idx], columns=feature_names).assign(__part="background")

    ex_idx = rng.permutation(len(s.X_test))[:EXPLAIN_ROWS]
    ex = pd.DataFrame(s.X_test[ex_idx], columns=feature_names).assign(__part="explain")

    explain_frame = pd.concat([bg, ex], ignore_index=True)

    desc = bundle.frame.drop(columns=[bundle.target]).describe(percentiles=[0.01, 0.5, 0.99]).T

    manifest = {
        "project": "Financial Fraud Detection using XAI and Stacking Ensembles",
        "dataset": bundle.provenance(), "target": bundle.target, "final_model": STACKING_NAME,
        "feature_schema": {"features": feature_names, "dtype": "float64"},
        "feature_stats": {c: {"min": float(desc.loc[c, "min"]), "p01": float(desc.loc[c, "1%"]), "median": float(desc.loc[c, "50%"]),
                              "p99": float(desc.loc[c, "99%"]), "max": float(desc.loc[c, "max"])} for c in feature_names},
        "preprocessing": {"imputer": "median", "scaler": "ManualScaler (fit on training data)",
                          "dedup": "no deduplication; exact duplicate rows preserved and reported",
                          "imbalance": (
                              "positive-class weighting for Logistic Regression and "
                              "second-order boosters; Random Forest uses bootstrap "
                              "sampling without class weighting; no global resampling"
                          )},
        "config": cfg.as_dict(), "split_sizes": s.sizes(), "split_fraud_counts": s.fraud_counts(), "imbalance": imb,
        "model_params": {n: describe_params(m) for n, m in models.items() if n not in [STACKING_NAME, "shap_weighted_stack"]},
        "stacking": {
            "base": BASE_MODEL_NAMES,
            "standard_meta_learner": "ManualLogisticRegression",
            "meta_scale_pos_weight": cfg.meta_scale_pos_weight,
            "stack_method": "predict_proba",
            "cv": (f"Timestamp-group-aware expanding-window chronological OOF: warm-up block plus {cfg.cv_folds} sequential validation blocks; equal timestamps kept together at fold boundaries" if cfg.split_strategy == "chronological" else f"Manual stratified {cfg.cv_folds}-fold OOF split (seed={cfg.seed})"),
            "passthrough": False
        },
        "thresholds": thresholds, "threshold_selection": "max F1 on validation set",
        "metrics_test": metrics, "metrics_test_at_0.5": metrics_half,
        "fit_seconds": {k: round(v, 2) for k, v in fit_seconds.items()},
        "note": "Academic decision-support model. All retained models are evaluated on the same full validation and held-out test sets. Synthetic data is for pipeline testing only."
    }

    version = save_artifacts(settings, models, ev, explain_frame, manifest, preprocessor=s.scaler)
    say(f"Saved signed artifacts as {version}")

    return TrainResult(version, metrics, metrics_half, thresholds, fit_seconds, s.sizes())
