"""Dependency-free binary classification metrics."""
import numpy as np


def _validate_inputs(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=float).reshape(-1)

    if len(y_true) == 0 or len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same non-zero length.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("Prediction scores must be finite.")
    if np.any((y_score < 0) | (y_score > 1)):
        raise ValueError("Prediction probabilities must be between 0 and 1.")

    return y_true.astype(int), y_score


def manual_roc_auc(y_true, y_score):
    """ROC-AUC using grouped tied scores and trapezoidal integration."""
    y_true, y_score = _validate_inputs(y_true, y_score)
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]

    # Evaluate ROC points only after complete tied-score groups.
    group_ends = np.r_[np.where(np.diff(scores) != 0)[0], len(scores) - 1]
    tp = np.cumsum(labels)[group_ends]
    fp = (group_ends + 1) - tp

    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]

    return float(np.trapezoid(tpr, fpr))


def manual_pr_auc(y_true, y_score):
    """Trapezoidal area under the precision-recall curve."""
    y_true, y_score = _validate_inputs(y_true, y_score)
    positives = np.sum(y_true == 1)

    if positives == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]

    group_ends = np.r_[np.where(np.diff(scores) != 0)[0], len(scores) - 1]
    tp = np.cumsum(labels)[group_ends]
    fp = (group_ends + 1) - tp

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    return float(np.trapezoid(precision, recall))


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_true, y_pred_proba = _validate_inputs(y_true, y_pred_proba)

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(manual_roc_auc(y_true, y_pred_proba), 4),
        "pr_auc": round(manual_pr_auc(y_true, y_pred_proba), 4),
        "mcc": round(float(mcc), 4),
    }


def select_threshold(y_true, y_pred_proba):
    """Select the threshold with maximum validation F1."""
    y_true, y_pred_proba = _validate_inputs(y_true, y_pred_proba)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Threshold selection requires both classes.")

    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_threshold = -1.0, 0.5

    for threshold in thresholds:
        result = compute_metrics(y_true, y_pred_proba, threshold)
        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_threshold = float(threshold)

    return best_threshold
