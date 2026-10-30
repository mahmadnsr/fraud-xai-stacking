# fraud_xai/models.py
from dataclasses import dataclass, asdict
from .manual_ml import (
    ManualLogisticRegression, 
    ManualRandomForest, 
    ManualGradientBoosting, 
    ManualStackingClassifier
)

BASE_MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
STACKING_NAME = "stacking_ensemble"
TREE_MODELS = ["random_forest", "xgboost", "lightgbm"]
DISPLAY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost Proxy",
           "lightgbm": "LightGBM Proxy", STACKING_NAME: "Stacking Ensemble"}

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    val_size: float = 0.2
    test_size: float = 0.2
    cv_folds: int = 3       # Fast training ke liye 3 folds
    split_strategy: str = "stratified"
    n_estimators: int = 5   # Fast training ke liye 5 trees

    def as_dict(self) -> dict:
        return asdict(self)

def build_base_models(cfg: TrainConfig, scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": ManualLogisticRegression(epochs=100),
        "random_forest": ManualRandomForest(n_estimators=cfg.n_estimators, max_depth=3),
        "xgboost": ManualGradientBoosting(n_estimators=cfg.n_estimators, max_depth=2),
        "lightgbm": ManualGradientBoosting(n_estimators=cfg.n_estimators, max_depth=2)
    }

def build_stacking(cfg: TrainConfig, scale_pos_weight: float) -> ManualStackingClassifier:
    base = build_base_models(cfg, scale_pos_weight)
    return ManualStackingClassifier(
        base_models=list(base.values()),
        meta_model=ManualLogisticRegression(epochs=100),
        cv=cfg.cv_folds
    )

def describe_params(model) -> dict:
    return {"class": type(model).__name__}