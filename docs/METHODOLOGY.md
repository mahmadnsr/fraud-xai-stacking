# Methodology

## Imbalance handling
Class weights (`balanced`, `balanced_subsample`) and XGBoost `scale_pos_weight` (computed from the **training** split). No SMOTE/resampling is used, so validation and test sets keep the true class distribution.

## Leakage controls (risk -> safeguard)
| Risk | Safeguard | Test |
|---|---|---|
| Identical rows in train and test | exact duplicates dropped before splitting | `test_no_duplicate_rows_across_splits` |
| Index overlap between splits | `assert_disjoint` runs after every split | `test_stratified_split_proportions_and_disjoint` |
| Imputer/scaler learning from test data | transforms live inside Pipelines; fitted on training folds only | `test_preprocessor_fit_on_train_only` |
| Meta-learner trained on in-sample base predictions | scikit-learn stacking uses out-of-fold predictions | `test_stacking_uses_four_base_learners_and_meta_lr`, `test_stack_structure` |
| Threshold tuned on test data | `select_threshold` receives validation data only | `test_threshold_selected_on_validation_only` |
| Repeated tuning against the test set | no tuning loop exists; test set is evaluated once per run | design + README policy |
Note: the validation split is used for the threshold only. Hyper-parameters are fixed defaults; if you tune them, do so with cross-validation on the training split.

## Random vs chronological split
Fraud patterns drift. When a `Time` column exists, a chronological split (past -> future) is the more realistic estimate of deployment performance and is available (`--split chronological`). Stratified random splitting is the default because it guarantees fraud cases in every split and suits i.i.d. benchmarks, but it can be optimistic.

## Evaluation
Precision, recall, F1, confusion matrix at the validation-selected threshold; ROC-AUC and PR-AUC from probabilities; accuracy shown with a warning. PR-AUC is the more informative threshold-free metric for rare fraud. Recall = share of fraud caught (missed fraud = loss); precision = share of alerts that are real (false alerts = customer friction and analyst workload). Both are reported; neither alone is sufficient. Test sets contain few frauds, so metrics have wide uncertainty - differences of a few points between models are not evidence of superiority.

## SHAP strategy
* Random Forest / XGBoost / LightGBM: `shap.TreeExplainer` on the fitted classifier, applied to the output of the pipeline's fitted preprocessing. Units: RF = probability, XGBoost/LightGBM = log-odds.
* Stacking ensemble: `shap.KernelExplainer` on `predict_proba` of the whole ensemble (model-agnostic; 25-row background; 200 samples/row). Approximate and slower, hence computed on 20 rows. Local accuracy (base + sum = output) is checked on every call.
* A base model's TreeExplainer is **not** presented as an explanation of the ensemble; the UI labels the method and units.
* Validation: output shape == input shape, finite values, feature names preserved, additivity (tests in `tests/test_explain.py`).
* Limits: SHAP explains model behaviour, not causes of fraud; correlated features share credit; in the public dataset V1-V28 are PCA components, so explanations are not human-meaningful business reasons.
