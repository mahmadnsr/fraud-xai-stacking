"""The four base learners and the stacking ensemble (Logistic Regression meta-learner)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .preprocess import make_preprocessor

BASE_MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
STACKING_NAME = "stacking_ensemble"
TREE_MODELS = ["random_forest", "xgboost", "lightgbm"]
DISPLAY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
           "lightgbm": "LightGBM", STACKING_NAME: "Stacking Ensemble"}


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    val_size: float = 0.2
    test_size: float = 0.2
    cv_folds: int = 5
    split_strategy: str = "stratified"
    n_estimators: int = 200

    def as_dict(self) -> dict:
        return asdict(self)


def build_base_models(cfg: TrainConfig, scale_pos_weight: float) -> dict[str, Pipeline]:
    """Each model is a Pipeline(preprocessor -> classifier) so inference reuses the exact fitted preprocessing.
    Class imbalance is handled with class weights (no resampling, so validation/test data stay untouched)."""
    n, s = cfg.n_estimators, cfg.seed
    clfs = {
        "logistic_regression": (True, LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s)),
        "random_forest": (False, RandomForestClassifier(n_estimators=n, min_samples_leaf=2, class_weight="balanced_subsample",
                                                        n_jobs=1, random_state=s)),
        "xgboost": (False, XGBClassifier(n_estimators=n, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                         scale_pos_weight=scale_pos_weight, eval_metric="logloss", tree_method="hist",
                                         n_jobs=1, random_state=s)),
        "lightgbm": (False, LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                                           n_jobs=1, random_state=s, verbose=-1)),
    }
    return {name: Pipeline([("prep", make_preprocessor(scale)), ("clf", clf)]) for name, (scale, clf) in clfs.items()}


def build_stacking(cfg: TrainConfig, scale_pos_weight: float) -> StackingClassifier:
    """Stacking: the meta-learner is trained on out-of-fold predicted probabilities (StackingClassifier uses
    cross_val_predict internally), so it never sees in-sample base predictions. Base learners are then refit on
    the full training set for inference. Preprocessing is inside each base pipeline, hence refit per fold."""
    base = build_base_models(cfg, scale_pos_weight)
    return StackingClassifier(
        estimators=[(name, base[name]) for name in BASE_MODEL_NAMES],
        final_estimator=LogisticRegression(max_iter=2000, random_state=cfg.seed),
        cv=StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed),
        stack_method="predict_proba", passthrough=False, n_jobs=1)


def describe_params(model) -> dict:
    """Short, JSON-safe hyper-parameter summary for the manifest."""
    clf = model[-1] if isinstance(model, Pipeline) else model
    keep = ("n_estimators", "max_depth", "learning_rate", "num_leaves", "class_weight", "scale_pos_weight", "C",
            "max_iter", "min_samples_leaf", "subsample", "colsample_bytree", "random_state")
    p = clf.get_params()
    return {"class": type(clf).__name__, **{k: p[k] for k in keep if k in p and isinstance(p[k], (int, float, str, type(None)))}}
