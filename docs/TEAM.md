# Team Work Allocation and Shared Interface Contract
These are **task allocations**, not claims that members have completed them; the code in this repository is a shared baseline for the team to review, own and extend.

| Member | Allocation | Primary modules |
|---|---|---|
| **Saikat** | Literature review; data cleaning & preprocessing; Logistic Regression; base-model prediction preparation; precision/recall evaluation; dashboard preprocessing integration; end-to-end testing | `schema.py`, `preprocess.py`, LR in `models.py`, `predict.py`, `tests/test_schema.py`, `test_preprocess.py`, `test_smoke.py` |
| **Shaurya** | Ensemble-learning literature review; EDA; Random Forest; base-model comparison; SHAP summary visualisations; confusion matrices & F1 analysis; dashboard evaluation visuals; final report | `eda.py`, RF in `models.py`, `evaluate.py`, `explain.plot_summary`, `docs/PROJECT_REPORT.md` |
| **Sunny** | Dataset identification/selection; tree & boosting feature prep; XGBoost; stacking architecture; SHAP importance for RF/XGB/LGBM; ROC-AUC and base-vs-ensemble comparison; model/ensemble integration into dashboard backend; GitHub organisation & documentation | `data.py`, XGBoost + `build_stacking` in `models.py`, `explain.explain_tree`, `persist.py`, `README.md`, `ui.page_eval` |
| **Ahmad** | Objectives & problem statement; data pipeline and split strategy; LightGBM; LR meta-learner & complete stacking training; SHAP for the final ensemble; consolidation of results; meta-learner/explainability dashboard integration; slides & demo | `train.py`, `split_data`, LightGBM in `models.py`, `explain.explain_ensemble`, `ui.page_explain`, presentation |

## Interface contract (what each module guarantees)
* Data: `DatasetBundle.frame` = float64 features + int `Class`; `report` is a `DatasetReport`.
* Models: every model is an sklearn `Pipeline([("prep", ...), ("clf", ...)])` (stacking is a `StackingClassifier` of those) exposing `predict_proba(X: DataFrame) -> (n, 2)`; column 1 = fraud probability; feature order = `manifest["feature_schema"]["features"]`.
* Evaluation arrays: `evaluation.npz` holds `y_val`, `y_test`, and `<model>__val` / `<model>__test` probabilities.
* Explanations: `Explanation.values` has shape `(n_rows, n_features)` for the fraud class; `output_space` says probability or log-odds.
* Errors: user-correctable problems raise `ValidationError`; untrusted artifacts raise `ArtifactError`; configuration raises `ConfigError`; the UI shows these messages and hides everything else behind `safe_error`.
