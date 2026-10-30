# fraud_xai/evaluate.py
import numpy as np
from .manual_ml import manual_metrics

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    return manual_metrics(y_true, y_pred_proba, threshold)

def select_threshold(y_true, y_pred_proba):
    # Simple threshold search to find max F1
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        m = compute_metrics(y_true, y_pred_proba, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = t
    return best_thresh