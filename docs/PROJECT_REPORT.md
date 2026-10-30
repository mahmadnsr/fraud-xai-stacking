# Project Report Material

## Abstract
Financial fraud is rare, costly and adversarial. This project builds an end-to-end, locally runnable fraud-detection system that trains four complementary classifiers (Logistic Regression, Random Forest, XGBoost, LightGBM), fuses them with a stacking ensemble whose Logistic Regression meta-learner is trained on out-of-fold probabilities, and explains predictions with SHAP. Evaluation uses held-out data with precision, recall, F1, ROC-AUC and PR-AUC suited to imbalance. A Streamlit dashboard provides EDA, training, evaluation, prediction and explanation, with authentication, AES-256-GCM encryption at rest and integrity-signed model artifacts.

## Problem statement
Detect potentially fraudulent card transactions in a highly imbalanced dataset while (1) avoiding evaluation leakage, (2) keeping false alarms manageable, and (3) giving analysts understandable reasons for each model score, without exposing sensitive financial data.

## Objectives and scope
In scope: four base models, stacking, SHAP, leakage-safe evaluation, dashboard, secure local deployment, tests, documentation. Out of scope: real-time streaming, bank integration, concept-drift monitoring, privacy-preserving (encrypted) inference, regulatory compliance certification.

## Literature review outline (verify each reference before citing; read the originals)
1. Stacked generalisation - D. H. Wolpert, *Neural Networks* 5(2), 1992.
2. Random Forests - L. Breiman, *Machine Learning* 45, 2001.
3. XGBoost: A Scalable Tree Boosting System - T. Chen & C. Guestrin, KDD 2016.
4. LightGBM: A Highly Efficient Gradient Boosting Decision Tree - G. Ke et al., NeurIPS 2017.
5. A Unified Approach to Interpreting Model Predictions (SHAP) - S. Lundberg & S.-I. Lee, NeurIPS 2017.
6. Consistent individualized feature attribution for tree ensembles (TreeSHAP) - Lundberg, Erion, Lee, arXiv:1802.03888, 2018.
7. The Precision-Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets - T. Saito & M. Rehmsmeier, *PLoS ONE* 10(3), 2015.
8. Calibrating Probability with Undersampling for Unbalanced Classification - A. Dal Pozzolo et al., IEEE CIDM 2015 (the dataset's research group).
9. Credit card fraud detection: a realistic modeling and a novel learning strategy - Dal Pozzolo et al., IEEE TNNLS 29(8), 2018.
Suggested themes: imbalance techniques, ensemble learning, explainability in finance, concept drift, evaluation pitfalls.

## Dataset description and preprocessing
Intended: ULB/Worldline Credit Card Fraud Detection (two days of European card transactions, 492 frauds / 284,807; `V1-V28` PCA components, `Time`, `Amount`, `Class`). Preprocessing: validation, sensitive-column exclusion, median imputation, scaling for Logistic Regression, exact-duplicate removal, stratified or chronological split. Demo mode: synthetic data with the same layout (clearly labelled; not real).

## Testing methodology and results
Automated tests (pytest) use small **synthetic fixtures** for speed and are never used as project results. Categories: schema/upload validation, malformed CSV, leakage and preprocessing, model/stacking structure, probability ranges, metric calculation (known values + recomputation from saved probabilities), SHAP shape/additivity, AES-GCM round trips, tamper/invalid-key/invalid-payload rejection, authentication/authorisation/rate limiting, artifact integrity (including a malicious pickle that must not execute), configuration errors, CLI end-to-end and Streamlit `AppTest` boot.
Result recorded on 2026-10-27 (Python 3.12.3): **61 passed, 0 failed** (`python -m pytest`, ~7 s). Manual `AppTest` walkthrough of all seven dashboard pages, in-app training (and no retraining on rerun), login failure/success and role enforcement, plus a real `streamlit run` health check, were also executed. `pip-audit` reported no known vulnerabilities after upgrading `cryptography`.
Not covered: real-dataset run, browser-level (Selenium) UI tests, load/performance testing, TLS proxy deployment, Windows/macOS, Python 3.11.

## Limitations and future scope
Synthetic demo only until a real dataset is supplied; small number of frauds in test splits means high metric variance; threshold chosen by F1 ignores asymmetric business costs (add cost-sensitive thresholding); no drift monitoring or scheduled retraining; KernelExplainer is slow/approximate; PCA features limit interpretability; no calibration step; future work: SHAP on larger samples, cost-sensitive learning, time-aware cross-validation, KMS integration, OIDC/SSO login, privacy-preserving inference.

## Final report outline
1 Introduction · 2 Problem statement & objectives · 3 Literature review · 4 Dataset & EDA · 5 Methodology (preprocessing, imbalance, base models, stacking, SHAP) · 6 System design (architecture, DFD, UML) · 7 Security design & threat model · 8 Implementation · 9 Testing · 10 Results & discussion (real data; FP/FN analysis) · 11 Limitations · 12 Conclusion & future work · References · Appendix (config, commands, manifest sample).

## Presentation outline (12-15 slides)
Title/team · Problem & motivation · Why imbalance breaks accuracy · Dataset · Architecture · Base models · Stacking (OOF diagram) · Leakage controls · SHAP (global + local, ensemble strategy) · Results with honest caveats · Live demo (train -> predict -> explain) · Security (what is and is not protected; no E2EE) · Limitations & future work · Q&A.
