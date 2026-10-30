# Financial Fraud Detection Using Explainable AI (XAI) and Stacking Ensemble Methods

Final-year B.Tech (CSBS) project. A runnable, locally hosted system that trains **Logistic Regression, Random Forest, XGBoost and
LightGBM**, combines them in a **stacking ensemble with a Logistic Regression meta-learner**, evaluates everything on held-out data,
and explains predictions with **SHAP**, behind a Streamlit dashboard with authentication, AES-256-GCM encryption at rest and
signed model artifacts.

> **Academic decision-support system - not a bank's production fraud-monitoring infrastructure.** Model output is an estimated
> fraud score, never a verified finding of fraud. Demo mode uses **synthetic** data; its metrics say nothing about real-world performance.

## Supported environment
Developed and tested on **Python 3.12.3 / Linux x86-64** with the exact versions in `requirements.txt`
(numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, xgboost 3.4.1, lightgbm 4.7.0, shap 0.54.1, streamlit 1.66.1, cryptography 50.0.2).
Python 3.11 was the brief's suggestion, but it was **not** tested; 3.12 is the supported version. Other versions may work but are unverified.

## Setup
```bash
git clone <your-repo-url> fraud-xai && cd fraud-xai
python3.12 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/init_env.py            # creates .env (mode 600) with fresh random keys + first admin user (password prompt)
python -m fraud_xai.cli check-config  # validates configuration
```
`init_env.py --dev-no-auth` creates a local-only config with login disabled (refused when `APP_ENV=production`).
Add more users with `python scripts/create_user.py NAME analyst` and merge the printed JSON into `AUTH_USERS_JSON`.

## Run
```bash
# 1) Train (reproducible; synthetic demo data)
python -m fraud_xai.cli train --source sample --seed 42
# 1b) Train on a real dataset you obtained legitimately (file must be in data/)
python -m fraud_xai.cli train --source csv --csv creditcard.csv --split chronological
# 2) Verify artifacts, list versions
python -m fraud_xai.cli list && python -m fraud_xai.cli verify <version>
# 3) Dashboard (binds to 127.0.0.1 by default)
streamlit run app.py
# 4) Tests
python -m pytest
# 5) Dependency vulnerability check
pip-audit -r requirements.txt
```
You can also load data and train from the dashboard (**admin** role).

## Real dataset
The public **Credit Card Fraud Detection** dataset (ULB / Worldline, ~284,807 transactions, 492 frauds; columns `Time`, `V1-V28`, `Amount`,
`Class`) is available from Kaggle (`mlg-ulb/creditcardfraud`) under the Database Contents License (DbCL). Download it yourself, accept its terms,
and place `creditcard.csv` in `data/` (git-ignored). This repository does **not** ship it and no download was performed during development.
Any other CSV works if it has numeric features and a binary `Class` column (name configurable). With `Time` present, prefer `--split chronological`.

## What is implemented (see `docs/` for details)
| Area | Module |
|---|---|
| Config + startup validation | `fraud_xai/config.py` |
| CSV parsing, schema validation, sensitive-column exclusion | `schema.py`, `data.py` |
| Splits, leakage guards, preprocessing pipelines | `preprocess.py` |
| 4 base models + stacking (OOF, LR meta-learner) | `models.py`, `train.py` |
| Metrics, thresholds (validation only), plots | `evaluate.py` |
| SHAP (Tree + Kernel for ensemble) | `explain.py` |
| Prediction + risk categories | `predict.py` |
| AES-256-GCM, scrypt, HMAC | `security.py`, `auth.py` |
| Signed manifests, integrity-checked loading | `persist.py` |
| Redacting logs, safe errors | `logging_utils.py` |
| Dashboard | `app.py`, `ui.py`, `eda.py` |

## Documentation
`docs/ARCHITECTURE.md` (diagrams) · `docs/METHODOLOGY.md` (stacking, SHAP, leakage) · `docs/SECURITY.md` (threat model, TLS, E2EE) ·
`docs/PROJECT_REPORT.md` (abstract, literature, report + slide outline, testing) · `docs/TEAM.md` (work allocation + interface contract)

## Publishing to GitHub safely
1. `git status` - confirm `.env`, `data/*`, `models/*`, `logs/` are **not** listed (they are in `.gitignore`).
2. `git add . && git commit -m "Initial commit"`; `git log --stat | grep -iE "\.env|\.csv|joblib"` should print nothing.
3. If a key or password was ever committed, treat it as compromised: delete `.env`, generate new keys, and rewrite history (e.g. `git filter-repo`) - deleting in a later commit is not enough.
4. Do not publish datasets whose license forbids redistribution. Use a **private** repo until you have checked.
5. Enable GitHub Dependabot / secret scanning; run `pip-audit` before releases.

## Retraining and versions
Each training run writes an immutable `models/vYYYYMMDDTHHMMSSZ/` with `models.joblib`, `evaluation.npz`, an encrypted explanation sample, a `manifest.json`
(dataset hash, schema, seeds, config, thresholds, dependency versions, test metrics) and an HMAC `manifest.sig`. Old versions are kept for rollback;
select the active version on *Security & Status*. Retrain when data drifts or dependency versions change (artifacts record the versions they were built with).
Never choose hyper-parameters by repeatedly looking at the **test** metrics; the test set is for the final report only.
