"""Streamlit dashboard. Every control is wired to real computation; nothing here fabricates results."""
from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd
import streamlit as st

from . import eda
from .auth import LoginLimiter, authenticate, can
from .config import ConfigError, load_settings
from .data import (DatasetBundle, list_local_csvs, load_local_csv, load_synthetic, load_uploaded_csv)
from .evaluate import (MetricError, comparison_table, plot_confusion, plot_metric_bars, plot_pr, plot_roc)
from .explain import (ExplainError, describe_local, explain_ensemble, explain_tree, global_importance, plot_local, plot_summary)
from .logging_utils import safe_error, setup_logging
from .models import DISPLAY, STACKING_NAME, TREE_MODELS, TrainConfig
from .persist import ArtifactError, list_versions, load_version, verify_version
from .predict import predict_frame
from .schema import ValidationError, validate_feature_frame
from .security import CryptoError, decrypt, encrypt
from .train import train_all

IDLE_TIMEOUT_S = 30 * 60
PAGES = ["Home", "Dataset & EDA", "Model Training", "Model Evaluation", "Fraud Prediction", "Explainability", "Security & Status"]
DISCLAIMER = ("Model output is an **estimated fraud score**, not a verified finding of fraud. False positives block legitimate "
              "customers; false negatives let fraud through. Use as decision support alongside human review.")


@st.cache_resource
def _limiter() -> LoginLimiter:
    return LoginLimiter()


@st.cache_resource(show_spinner="Verifying and loading model artifacts ...")
def _cached_model(_settings, version: str):
    return load_version(_settings, version)


def _login(settings) -> str | None:
    if settings.auth_disabled:
        st.sidebar.warning("Authentication is DISABLED (development mode only).")
        return "admin"
    now = time.time()
    if st.session_state.get("role") and now - st.session_state.get("last_seen", 0) > IDLE_TIMEOUT_S:
        st.session_state.clear()
        st.warning("Session expired after inactivity. Please sign in again.")
    if st.session_state.get("role"):
        st.session_state.last_seen = now
        return st.session_state.role
    st.title("Sign in")
    with st.form("login"):
        user, pw = st.text_input("Username"), st.text_input("Password", type="password")
        go = st.form_submit_button("Sign in")
    if go:
        try:
            role = authenticate(settings.auth_users, user, pw, _limiter())
        except PermissionError as exc:
            st.error(str(exc))
            return None
        if role:
            st.session_state.update(role=role, user=user, last_seen=now)
            st.rerun()
        st.error("Invalid username or password.")
    return None


def _current_model(settings):
    versions = list_versions(settings.models_dir)
    if not versions:
        return None
    version = st.session_state.get("model_version") or versions[0]
    if version not in versions:
        version = versions[0]
    return _cached_model(settings, version)


def run() -> None:
    st.set_page_config(page_title="Fraud Detection - XAI & Stacking", page_icon="🛡️", layout="wide")
    try:
        settings = load_settings()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    logger = setup_logging(settings.log_level, settings.logs_dir)
    role = _login(settings)
    if not role:
        st.stop()
    st.sidebar.title("🛡️ Fraud Detection")
    st.sidebar.caption("XAI + Stacking Ensemble · academic decision-support demo")
    if not settings.auth_disabled and st.sidebar.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    page = st.sidebar.radio("Navigate", PAGES)
    try:
        model = _current_model(settings)
    except (ArtifactError, CryptoError) as exc:
        model = None
        st.sidebar.error(f"Model artifacts rejected: {exc}")
    except Exception as exc:  # unexpected: generic message to user, details to log
        model = None
        st.sidebar.error(safe_error(logger, exc, "Loading the model"))
    if model is not None:
        st.sidebar.success(f"Model {model.version}")
        if model.manifest["dataset"]["is_synthetic"]:
            st.sidebar.warning("Trained on SYNTHETIC demo data. Metrics are not real-world results.")
    ctx = dict(settings=settings, role=role, model=model, logger=logger)
    {"Home": page_home, "Dataset & EDA": page_eda, "Model Training": page_train, "Model Evaluation": page_eval,
     "Fraud Prediction": page_predict, "Explainability": page_explain, "Security & Status": page_status}[page](**ctx)


def _need_model(model) -> bool:
    if model is None:
        st.info("No trained model is available yet. An admin can train one on the **Model Training** page.")
        return False
    return True


def _fig(fig) -> None:
    import matplotlib.pyplot as plt
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------- pages
def page_home(settings, role, model, logger):
    st.title("Financial Fraud Detection using Explainable AI and Stacking Ensembles")
    st.markdown(
        "Detects potentially fraudulent card transactions with **four base learners** (Logistic Regression, Random Forest, XGBoost, "
        "LightGBM) combined by a **stacking ensemble** whose **Logistic Regression meta-learner** is trained on out-of-fold "
        "probabilities. **SHAP** explains individual and global model behaviour.")
    st.code("CSV/sample -> schema validation -> dedup -> stratified/chronological split -> Pipelines(impute[,scale]) -> "
            "4 base models -> 5-fold OOF stacking (LR meta) -> validation threshold -> held-out test metrics -> "
            "signed artifacts -> prediction + SHAP", language="text")
    st.subheader("Dataset and model")
    if model:
        d = model.manifest["dataset"]
        st.write(f"**Dataset:** {d['label']} · {d['n_rows']:,} rows · fraud rate {d['fraud_rate']:.2%} · SHA-256 `{d['sha256'][:12]}…`")
        st.write(f"**Model version:** {model.version} · trained {model.manifest['created_utc']}")
    else:
        st.write("No model trained yet.")
    st.subheader("Important limitations")
    st.markdown("- Academic project, **not** a bank-grade fraud-monitoring system.\n- Demo mode uses **synthetic** data; real results require a legitimately obtained dataset (see README).\n"
                "- SHAP shows what drives the *model's score*, not why fraud occurred.\n- Public datasets age; real fraud patterns drift, so models need monitoring and retraining.\n- "
                "Server-side inference needs plaintext features; this is not end-to-end encrypted (see docs/SECURITY.md).")
    st.info(DISCLAIMER)


def page_eda(settings, role, model, logger):
    st.title("Dataset & Exploratory Data Analysis")
    src = st.radio("Data source", ["Synthetic demo data", "Local CSV in data/", "Upload CSV"], horizontal=True)
    target = st.text_input("Target column (0 = legitimate, 1 = fraud)", "Class", max_chars=64)
    bundle = None
    try:
        if src == "Synthetic demo data":
            rows = st.slider("Rows", 2_000, 20_000, 10_000, 1_000)
            if st.button("Generate synthetic demo data"):
                st.session_state.bundle = load_synthetic(rows)
        elif src == "Local CSV in data/":
            files = list_local_csvs(settings.data_dir)
            name = st.selectbox("File", files) if files else None
            if not files:
                st.info("No CSV files in data/. Place e.g. creditcard.csv there (see README for the legitimate source).")
            if name and st.button("Load and validate"):
                st.session_state.bundle = load_local_csv(name, settings.data_dir, settings.max_upload_bytes, settings.max_rows, target)
        else:
            up = st.file_uploader("CSV file (validated by content, kept in memory only)", type=["csv"])
            if up is not None and st.button("Validate uploaded file"):
                st.session_state.bundle = load_uploaded_csv(up.getvalue(), settings.max_upload_bytes, settings.max_rows, target)
    except ValidationError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(safe_error(logger, exc, "Dataset loading"))
    bundle: DatasetBundle | None = st.session_state.get("bundle")
    if not bundle:
        st.info("Load a dataset to see the analysis.")
        return
    r = bundle.report
    if bundle.is_synthetic:
        st.warning("This is SYNTHETIC demonstration data, not real transactions.")
    st.success(f"Validated: {bundle.label}")
    c = st.columns(5)
    c[0].metric("Rows", f"{r.n_rows:,}"); c[1].metric("Features", r.n_features); c[2].metric("Fraud", f"{r.class_counts[1]:,}")
    c[3].metric("Fraud rate", f"{r.fraud_rate:.3%}"); c[4].metric("Duplicates", r.duplicate_rows)
    for w in r.warnings:
        st.warning(w)
    if r.excluded_columns:
        st.write("Excluded sensitive-looking columns:", ", ".join(r.excluded_columns))
    left, right = st.columns(2)
    with left:
        _fig(eda.plot_class_balance(r.class_counts))
        st.caption(f"Imbalance ratio ≈ {r.class_counts[0] / max(r.class_counts[1], 1):.0f}:1. Accuracy is misleading here; use recall, precision and PR-AUC.")
    with right:
        st.subheader("Missing values")
        st.write(f"{r.missing_cells:,} missing cells.")
        if r.missing_by_column:
            st.dataframe(pd.Series(r.missing_by_column, name="missing").to_frame())
    feat = st.selectbox("Feature distribution", r.feature_names)
    _fig(eda.plot_feature_distribution(bundle.frame, feat, bundle.target))
    _fig(eda.plot_correlations(bundle.frame, bundle.target))
    cols = st.columns(2)
    if "Amount" in r.feature_names:
        with cols[0]:
            _fig(eda.plot_amount_by_class(bundle.frame, bundle.target))
    if "Time" in r.feature_names:
        with cols[1]:
            _fig(eda.plot_fraud_rate_over_time(bundle.frame, bundle.target))
        st.caption("A 'Time' column exists: a chronological split (train on past, test on future) better mimics deployment than a random split.")
    st.dataframe(bundle.frame.describe().T, width="stretch")


def page_train(settings, role, model, logger):
    st.title("Model Training")
    if flash := st.session_state.pop("flash", None):
        st.success(flash)
    if not can(role, "train"):
        st.warning("Only administrators can train models.")
        return
    bundle = st.session_state.get("bundle")
    if not bundle:
        st.info("Load and validate a dataset on the **Dataset & EDA** page first.")
        return
    st.write(f"Dataset: **{bundle.label}** ({bundle.report.n_rows:,} rows)")
    c = st.columns(4)
    split = c[0].selectbox("Split", ["stratified", "chronological"])
    folds = c[1].number_input("Stacking CV folds", 3, 10, 5)
    n_est = c[2].number_input("Trees/boosting rounds", 20, 600, 200, step=20)
    seed = c[3].number_input("Random seed", 0, 10_000, 42)
    if split == "chronological" and "Time" not in bundle.report.feature_names:
        st.error("Chronological split needs a 'Time' column.")
        return
    if st.session_state.get("training"):
        st.warning("Training is already running.")
        return
    if st.button("Train four base models + stacking ensemble", type="primary"):
        st.session_state.training = True  # guard against duplicate runs from reruns
        try:
            cfg = TrainConfig(seed=int(seed), cv_folds=int(folds), split_strategy=split, n_estimators=int(n_est))
            with st.status("Training (real progress messages from the pipeline)", expanded=True) as status:
                res = train_all(bundle, cfg, settings, progress=lambda m: st.write(m))
                status.update(label=f"Finished: {res.version}", state="complete")
            st.session_state.model_version = res.version
            st.session_state.last_train = res
            st.session_state.flash = f"Training complete: {res.version}. Open **Model Evaluation**."
            st.session_state.training = False
            st.rerun()  # refresh the sidebar so it shows the new model
        except (ValidationError, RuntimeError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(safe_error(logger, exc, "Training"))
        finally:
            st.session_state.training = False
    res = st.session_state.get("last_train")
    if res:
        st.write("Fit time (s):", {DISPLAY[k]: round(v, 1) for k, v in res.fit_seconds.items()})


def page_eval(settings, role, model, logger):
    st.title("Model Evaluation (held-out test set)")
    if not _need_model(model):
        return
    m = model.manifest
    st.caption(f"Version {model.version} · test rows {m['split_sizes']['test']:,} · thresholds chosen on the validation set only.")
    try:
        st.dataframe(comparison_table(m["metrics_test"]).style.format("{:.3f}", subset=["Threshold", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Accuracy*"]),
                     width="stretch")
        st.caption("*Accuracy is potentially misleading for imbalanced data and is shown for completeness only.")
        _fig(plot_metric_bars(m["metrics_test"]))
        y, probas = model.evaluation["y_test"], {k: model.evaluation[f"{k}__test"] for k in m["metrics_test"]}
        c = st.columns(2)
        with c[0]:
            _fig(plot_roc(y, probas))
        with c[1]:
            _fig(plot_pr(y, probas))
        name = st.selectbox("Confusion matrix", list(m["metrics_test"]), format_func=DISPLAY.get, index=len(m["metrics_test"]) - 1)
        mm = m["metrics_test"][name]
        c = st.columns([1, 2])
        with c[0]:
            _fig(plot_confusion(mm, DISPLAY[name]))
        with c[1]:
            st.markdown(f"**False positives: {mm['fp']}** - legitimate transactions flagged (customer friction, review cost).\n\n"
                        f"**False negatives: {mm['fn']}** - fraud that slipped through (direct financial loss).\n\n"
                        "Recall measures how much fraud is caught; precision measures how many alerts are real. Raising the threshold "
                        "trades recall for precision, so both must be reported.")
        st.subheader("Feature importance comparison (impurity/gain-based, model-internal)")
        imp = {DISPLAY[n]: pd.Series(model.models[n][-1].feature_importances_, index=model.feature_names) for n in TREE_MODELS}
        imp = pd.DataFrame({k: v / v.sum() for k, v in imp.items()})
        st.bar_chart(imp.loc[imp.mean(axis=1).sort_values(ascending=False).head(12).index])
        st.caption("Built-in importances are not SHAP values; see the Explainability page for SHAP.")
        with st.expander("Metrics at the default 0.5 threshold"):
            st.dataframe(comparison_table(m["metrics_test_at_0.5"]), width="stretch")
    except (MetricError, KeyError, ValueError) as exc:
        st.error(f"Metrics could not be displayed: {exc}")


def page_predict(settings, role, model, logger):
    st.title("Fraud Prediction")
    st.info(DISCLAIMER)
    if not _need_model(model):
        return
    names = list(model.models)
    name = st.selectbox("Model", names, index=names.index(STACKING_NAME), format_func=DISPLAY.get)
    thr = st.slider("Decision threshold", 0.01, 0.99, float(np.clip(model.manifest["thresholds"][name], 0.01, 0.99)), 0.01,
                    help="Default = the F1-optimal threshold found on validation data. Lower = more alerts (fewer misses, more false alarms).")
    tab1, tab2 = st.tabs(["Single transaction", "Batch CSV"])
    feats, stats = model.feature_names, model.manifest["feature_stats"]
    with tab1:
        _, ex = model.explain_sample()
        pick = st.selectbox("Start from", ["Median values"] + [f"Held-out sample row {i}" for i in range(min(25, len(ex)))])
        base = {f: stats[f]["median"] for f in feats} if pick == "Median values" else ex.iloc[int(pick.split()[-1])].to_dict()
        with st.form("single"):
            cols, vals = st.columns(4), {}
            for i, f in enumerate(feats):
                vals[f] = cols[i % 4].number_input(f, value=float(base[f]), format="%.6g", key=f"in_{pick}_{f}")
            go = st.form_submit_button("Predict")
        if go:
            try:
                row = pd.DataFrame([vals])
                out = predict_frame(model, name, row, thr)
                st.session_state.last_input = row
                p = float(out.fraud_probability.iloc[0])
                c = st.columns(3)
                c[0].metric("Estimated fraud probability", f"{p:.2%}"); c[1].metric("Predicted class", "Flagged" if out.predicted_class.iloc[0] else "Not flagged")
                c[2].metric("Risk category", out.risk_category.iloc[0])
                st.caption("A flag is a model estimate for review, not confirmation of fraud. See **Explainability** for the drivers of this score.")
                if (row.iloc[0] < pd.Series({f: stats[f]["min"] for f in feats})).any() or (row.iloc[0] > pd.Series({f: stats[f]["max"] for f in feats})).any():
                    st.warning("Some inputs lie outside the range seen in training; the prediction may be unreliable.")
            except ValidationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(safe_error(logger, exc, "Prediction"))
    with tab2:
        up = st.file_uploader("CSV with exactly the model's feature columns", type=["csv"], key="batch")
        if up is not None and st.button("Score file"):
            try:
                from .schema import parse_csv_bytes
                df = parse_csv_bytes(up.getvalue(), settings.max_upload_bytes, min(settings.max_rows, 50_000))
                X = validate_feature_frame(df, feats, 50_000, target=model.manifest["target"])
                out = predict_frame(model, name, X, thr)
                st.session_state.batch_result = out
            except ValidationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(safe_error(logger, exc, "Batch prediction"))
        out = st.session_state.get("batch_result")
        if out is not None:
            st.write(f"{len(out):,} rows scored; {int(out.predicted_class.sum()):,} flagged at threshold {thr:.2f}.")
            st.dataframe(out.head(500), width="stretch")
            st.download_button("Download scores (CSV)", out.to_csv(index_label="row").encode(), "fraud_scores.csv", "text/csv")


def _explanation(model, name: str, X: pd.DataFrame, bg: pd.DataFrame):
    key = ("expl", model.version, name, len(X), int(pd.util.hash_pandas_object(X).sum() % 10**9))
    if key not in st.session_state:
        st.session_state[key] = explain_ensemble(model.models, X, bg, 200) if name == STACKING_NAME else explain_tree(model.models, name, X)
    return st.session_state[key]


def page_explain(settings, role, model, logger):
    st.title("Explainability (SHAP)")
    if not _need_model(model):
        return
    st.info("SHAP attributes the *model's output* to input features. It does not prove causation or why a transaction is fraudulent. "
            "Tree models use TreeExplainer; the **stacking ensemble uses the model-agnostic KernelExplainer** (approximate, computed on a small sample), "
            "because a base model's explainer alone does not explain the combined model.")
    name = st.selectbox("Model to explain", TREE_MODELS + [STACKING_NAME], index=3, format_func=DISPLAY.get)
    bg, ex = model.explain_sample()
    n = 20 if name == STACKING_NAME else 100
    try:
        with st.spinner("Computing SHAP values ..."):
            e = _explanation(model, name, ex.head(n)[model.feature_names], bg[model.feature_names].head(25))
        st.caption(f"{e.method}; {len(e.data)} held-out rows; output in {e.output_space} units.")
        c = st.columns(2)
        with c[0]:
            _fig(plot_summary(e))
        with c[1]:
            _fig(plot_summary(e, bar=True))
        st.dataframe(global_importance(e).head(10).rename("mean |SHAP|").to_frame(), width="stretch")
        st.subheader("Local explanation")
        options = ["Held-out sample row"] + (["Last transaction entered on the Prediction page"] if "last_input" in st.session_state else [])
        src = st.radio("Explain", options, horizontal=True)
        if src == "Held-out sample row":
            row = st.number_input("Row", 0, len(e.data) - 1, 0)
            le = e
        else:
            li = st.session_state.last_input[model.feature_names]
            le, row = (explain_ensemble(model.models, li, bg[model.feature_names].head(25)) if name == STACKING_NAME
                       else explain_tree(model.models, name, li)), 0
        _fig(plot_local(le, int(row)))
        st.markdown("**In words (about the model, not real-world causes):**")
        for s in describe_local(le, int(row)):
            st.markdown(f"- {s}")
    except (ExplainError, ValidationError) as exc:
        st.error(f"Explanation could not be produced: {exc}")
    except Exception as exc:
        st.error(safe_error(logger, exc, "Explanation"))


def page_status(settings, role, model, logger):
    st.title("Security & System Status")
    st.subheader("Configuration checks (values are never shown)")
    st.table(pd.DataFrame({"Check": ["Encryption key valid (32 bytes)", "Signing key valid & distinct", "Authentication enabled",
                                     "Environment", "Upload limit", "Row limit"],
                           "Status": ["OK", "OK", "No (dev only)" if settings.auth_disabled else f"Yes ({len(settings.auth_users)} user(s))",
                                      settings.app_env, f"{settings.max_upload_mb} MB", f"{settings.max_rows:,}"]}))
    try:  # live self-test of the AES-GCM path
        ok = decrypt(encrypt(b"selftest", settings.encryption_key), settings.encryption_key) == b"selftest"
    except CryptoError:
        ok = False
    st.write("AES-256-GCM self-test:", "✅ passed" if ok else "❌ failed")
    st.subheader("Model artifacts")
    versions = list_versions(settings.models_dir)
    rows = []
    for v in versions[:10]:
        try:
            verify_version(settings, v)
            rows.append({"Version": v, "Integrity": "verified (signature + SHA-256)"})
        except ArtifactError as exc:
            rows.append({"Version": v, "Integrity": f"REJECTED: {exc}"})
    st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame({"Version": []}), width="stretch")
    if len(versions) > 1 and can(role, "status"):
        st.session_state.model_version = st.selectbox("Active version", versions, index=versions.index(model.version) if model else 0)
    st.subheader("Dataset validation")
    b = st.session_state.get("bundle")
    st.write(f"{b.label}: validated, {b.report.n_rows:,} rows." if b else "No dataset loaded in this session.")
    st.subheader("What this deployment does and does not guarantee")
    st.markdown("- Encryption **at rest** (AES-256-GCM) protects the stored explanation sample; model files are integrity-signed (HMAC-SHA256).\n"
                "- Encryption **in transit** requires HTTPS at a reverse proxy; Streamlit itself serves HTTP (see docs/SECURITY.md).\n"
                "- This is **not end-to-end encrypted**: the server must see plaintext features to score them.\n"
                "- No claim of immunity to attacks is made; see the threat model for residual risks.")
