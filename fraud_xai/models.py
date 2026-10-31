# fraud_xai/models.py
from dataclasses import dataclass, asdict
from .manual_ml import (
    ManualLogisticRegression,
    ManualRandomForest,
    ManualXGBoost,
    ManualLightGBM,
    ManualStackingClassifier,
)

BASE_MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
STACKING_NAME = "stacking_ensemble"
TREE_MODELS = ["random_forest", "xgboost", "lightgbm"]
DISPLAY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "Manual XGBoost-inspired",
           "lightgbm": "Manual LightGBM-inspired", STACKING_NAME: "Stacking Ensemble"}

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
        "logistic_regression": ManualLogisticRegression(
            epochs=100, scale_pos_weight=scale_pos_weight
        ),
        "random_forest": ManualRandomForest(
            n_estimators=cfg.n_estimators, max_depth=3, random_state=cfg.seed
        ),
        "xgboost": ManualXGBoost(
            n_estimators=cfg.n_estimators, max_depth=3,
            learning_rate=0.05, reg_lambda=1.0, gamma=0.0,
            scale_pos_weight=scale_pos_weight, random_state=cfg.seed
        ),
        "lightgbm": ManualLightGBM(
            n_estimators=cfg.n_estimators, max_depth=4,
            max_leaves=7, max_bins=16, learning_rate=0.05,
            reg_lambda=1.0, scale_pos_weight=scale_pos_weight,
            random_state=cfg.seed
        ),
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