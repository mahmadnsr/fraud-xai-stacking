"""Primary Streamlit dashboard for the manual fraud-detection pipeline."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .config import load_settings
from .data import load_synthetic
from .train import train_all
from .models import TrainConfig, DISPLAY
from .preprocess import prepare_xy, split_data
from .persist import list_versions, load_version
from .predict import predict_frame
from .explain import get_feature_importance, manual_shap_values


st.set_page_config(
    page_title="Financial Fraud Detection | Explainable AI",
    page_icon="🛡️",
    layout="wide",
)


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #080e1b;
        --panel: #101a2c;
        --panel2: #14223a;
        --line: rgba(148, 176, 220, .15);
        --text: #edf4ff;
        --muted: #91a4c2;
        --blue: #58a6ff;
        --cyan: #4de0d0;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(35, 95, 170, .20), transparent 35%),
            radial-gradient(ellipse at 90% 12%, rgba(24, 160, 170, .10), transparent 30%),
            #080e1b;
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 14, 27, .72);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101b30 0%, #0b1322 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #a9bad4;
    }

    [data-testid="stSidebar"] label {
        font-weight: 600;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.1rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.035em;
        color: #f3f7ff !important;
    }

    h2 {
        margin-top: 1.2rem !important;
        font-size: 1.65rem !important;
    }

    h3 {
        font-size: 1.2rem !important;
    }

    p, li, label, [data-testid="stCaptionContainer"] {
        color: #b6c5dc;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem 1.5rem;
        margin: 0 0 1.8rem;
        border: 1px solid rgba(104, 166, 255, .28);
        border-radius: 24px;
        background:
            radial-gradient(circle at 82% 20%, rgba(60, 190, 220, .18), transparent 28%),
            linear-gradient(115deg, #142744 0%, #102039 48%, #102d40 100%);
        box-shadow: 0 18px 55px rgba(0, 0, 0, .23);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: -75px;
        bottom: -150px;
        border: 1px solid rgba(100, 190, 255, .25);
        border-radius: 50%;
        box-shadow: 0 0 0 25px rgba(100,190,255,.025),
                    0 0 0 50px rgba(100,190,255,.025);
    }

    .hero-topline {
        color: #77d9e7;
        font-size: .69rem;
        font-weight: 800;
        letter-spacing: .18em;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .hero-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4de0d0;
        box-shadow: 0 0 12px #4de0d0;
    }

    .hero-tag {
        border: 1px solid rgba(120, 190, 255, .35);
        color: #b8d9ff;
        border-radius: 20px;
        padding: 4px 9px;
        font-size: .58rem;
        letter-spacing: .1em;
    }

    .hero-title {
        margin-top: 1.2rem;
        color: #f5f8ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.2rem, 5vw, 3.65rem);
        line-height: 1.02;
        font-weight: 700;
        letter-spacing: -.055em;
    }

    .hero-title span {
        color: #72c9ff;
    }

    .hero-subtitle {
        margin-top: .9rem;
        color: #b2c6e1;
        font-size: 1rem;
    }

    .hero-footer {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 24px;
        margin-top: 1.7rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(180, 210, 255, .15);
        color: #91b6d7;
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: .12em;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(23, 39, 64, .96), rgba(15, 27, 46, .96));
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 18px 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,.13);
    }

    [data-testid="stMetricLabel"] {
        color: #9db2d0 !important;
        font-size: .78rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f2f7ff !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 17px !important;
    }

    .stButton > button {
        min-height: 2.8rem;
        border: 1px solid rgba(94, 169, 255, .4);
        border-radius: 11px;
        color: #f4f8ff;
        background: linear-gradient(105deg, #2366b5, #168b9a);
        font-weight: 700;
        transition: all .18s ease;
        box-shadow: 0 7px 18px rgba(20, 100, 170, .17);
    }

    .stButton > button:hover {
        border-color: #79d9ff;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(30, 140, 200, .25);
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 13px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 13px;
        border: 1px solid var(--line);
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        border-radius: 10px;
        background: #111e32;
        border-color: #2a3d5b;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stImage"] {
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 8px;
        background: rgba(15, 27, 46, .55);
    }

    hr {
        border-color: var(--line);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 3rem; }
        .hero { padding: 1.4rem; border-radius: 17px; }
        .hero-footer { gap: 8px 14px; }
    }
    </style>
""", unsafe_allow_html=True)



def _metric_table(metrics):
    rows = []
    for name, values in metrics.items():
        row = {"Model": DISPLAY.get(name, name)}
        for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"):
            value = values.get(key)
            row[key.upper().replace("_", "-")] = (
                round(float(value), 4)
                if value is not None and np.isfinite(value)
                else "N/A"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _get_loaded(settings, version):
    return load_version(settings, version)


def run():
    settings = load_settings()

    st.markdown("""
    <div class="hero">
        <div class="hero-topline">
            <span class="hero-dot"></span>
            FRAUD INTELLIGENCE PLATFORM
            <span class="hero-tag">RESEARCH EDITION</span>
        </div>
        <div class="hero-title">Financial Fraud<br><span>Detection System</span></div>
        <div class="hero-subtitle">
            Explainable AI · Manual ML Algorithms · Stacking Ensemble
        </div>
        <div class="hero-footer">
            <span>MODEL-DRIVEN ANALYTICS</span>
            <span>MONTE CARLO SHAPLEY EXPLANATIONS</span>
            <span>ARTIFACT-INTEGRITY VERIFIED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select section",
            ["Overview", "Dataset", "Train Models", "Evaluation",
             "Transaction Analysis", "Explainability", "Security & Artifacts"],
        )
        st.divider()
        st.caption(f"Environment: {settings.app_env}")
        st.caption(f"Authentication disabled: {settings.auth_disabled}")

    # Keep the loaded dataset and trained version available across reruns.
    if "fraud_bundle" not in st.session_state:
        st.session_state["fraud_bundle"] = None
    if "fraud_version" not in st.session_state:
        st.session_state["fraud_version"] = None

    if page == "Overview":
        st.subheader("Project overview")
        st.write(
            "This research prototype compares manually implemented Logistic "
            "Regression, Random Forest, XGBoost-inspired and LightGBM-inspired "
            "boosters, and a stacking ensemble."
        )
        st.info(
            "The booster implementations are original NumPy-based "
            "implementations inspired by the respective algorithms; they are "
            "not the upstream XGBoost or LightGBM packages."
        )
        versions = list_versions(settings.models_dir)
        c1, c2, c3 = st.columns(3)
        c1.metric("Saved model versions", len(versions))
        c2.metric("Active session dataset", "Loaded" if st.session_state["fraud_bundle"] else "Not loaded")
        c3.metric("Active session model", st.session_state["fraud_version"] or "None")
        st.markdown("**Workflow**")
        st.write("1. Load a dataset.  2. Train and persist models.  3. Evaluate saved results.")
        st.write("4. Inspect a transaction and calculate feature attributions.")

    elif page == "Dataset":
        st.subheader("Dataset")
        st.write(
            "Load the built-in synthetic dataset. It is intended for pipeline "
            "development and reproducible experiments, not as a substitute "
            "for a real-world financial transaction dataset."
        )
        n_rows = st.number_input(
            "Number of rows", min_value=500, max_value=100000,
            value=5000, step=500,
        )
        fraud_rate = st.slider(
            "Synthetic fraud rate", min_value=0.01, max_value=0.20,
            value=0.03, step=0.01,
        )
        seed = st.number_input("Dataset random seed", min_value=0, value=42)
        if st.button("Generate / Load Synthetic Dataset", type="primary"):
            with st.spinner("Generating dataset..."):
                st.session_state["fraud_bundle"] = load_synthetic(
                    n_rows=int(n_rows),
                    fraud_rate=float(fraud_rate),
                    seed=int(seed),
                )
                st.session_state["fraud_version"] = None
            st.success("Synthetic dataset loaded.")

        bundle = st.session_state["fraud_bundle"]
        if bundle is not None:
            df = bundle.frame
            target = bundle.target
            counts = df[target].value_counts().sort_index()
            left, right = st.columns(2)
            left.metric("Rows", f"{len(df):,}")
            right.metric("Features", len(df.columns) - 1)
            st.write("**Class distribution**")
            st.bar_chart(counts)
            st.write("**Data preview**")
            st.dataframe(df.head(20), use_container_width=True)
            st.write("**Dataset summary**")
            st.dataframe(df.describe().T, use_container_width=True)

    elif page == "Train Models":
        st.subheader("Train models")
        bundle = st.session_state["fraud_bundle"]
        if bundle is None:
            st.warning("Load a dataset in the Dataset section first.")
        else:
            c1, c2, c3 = st.columns(3)
            n_estimators = c1.number_input(
                "Trees / boosting rounds", min_value=3, max_value=100,
                value=25, step=1,
            )
            cv_folds = c2.number_input(
                "Stacking folds", min_value=2, max_value=5, value=3,
            )
            seed = c3.number_input("Training seed", min_value=0, value=42)

            st.caption(
                "Training uses the project's manual implementations and "
                "stratified out-of-fold stacking."
            )
            if st.button("Train and Save All Models", type="primary"):
                cfg = TrainConfig(
                    seed=int(seed),
                    cv_folds=int(cv_folds),
                    n_estimators=int(n_estimators),
                )
                progress_area = st.empty()
                try:
                    with st.spinner("Training models and creating signed artifacts..."):
                        result = train_all(
                            bundle,
                            cfg,
                            settings,
                            progress=lambda message: progress_area.write(message),
                        )
                        # Always reopen the persisted artifact; downstream pages
                        # therefore use exactly the models that were saved.
                        loaded = load_version(settings, result.version)
                    st.session_state["fraud_version"] = result.version
                    st.session_state["fraud_metrics"] = result.metrics
                    X, y = prepare_xy(bundle.frame, bundle.target)
                    time_values = None
                    if cfg.split_strategy == "chronological":
                        if "Time" not in bundle.frame.columns:
                            raise ValueError(
                                "Chronological splitting requires a 'Time' column in the dataset."
                            )
                        time_values = bundle.frame["Time"].to_numpy()

                    split = split_data(
                        X,
                        y,
                        cfg.split_strategy,
                        cfg.val_size,
                        cfg.test_size,
                        cfg.seed,
                        time_values=time_values,
                    )
                    st.session_state["fraud_X_test"] = split.X_test
                    st.session_state["fraud_y_test"] = split.y_test
                    st.session_state["fraud_feature_names"] = list(loaded.feature_names)
                    st.success(f"Training complete. Saved version: {result.version}")
                    st.write("Persisted models:", ", ".join(loaded.models.keys()))
                except Exception as exc:
                    st.error(f"Training failed: {type(exc).__name__}: {exc}")
                    st.exception(exc)

    elif page == "Evaluation":
        st.subheader("Model evaluation")
        version_list = list_versions(settings.models_dir)

        if not version_list:
            st.warning("No saved model versions are available. Train models first.")
        else:
            default_version = st.session_state["fraud_version"]
            default_index = (
                version_list.index(default_version)
                if default_version in version_list else 0
            )
            version = st.selectbox(
                "Saved model version", version_list, index=default_index
            )

            try:
                loaded = _get_loaded(settings, version)
                evaluation = loaded.manifest.get("metrics_test", {})
                metrics_half = loaded.manifest.get("metrics_test_at_0.5", {})
                thresholds = loaded.manifest.get("thresholds", {})

                if evaluation:
                    st.markdown("**Held-out test performance**")
                    st.caption(
                        "Metrics use thresholds selected by maximizing F1 "
                        "on the validation set."
                    )
                    metric_frame = _metric_table(evaluation)
                    st.dataframe(metric_frame, use_container_width=True)

                    if "F1" in metric_frame.columns:
                        st.markdown("**F1 comparison**")
                        st.bar_chart(
                            metric_frame.set_index("Model")[["F1"]]
                        )

                    st.markdown("**Validation-selected decision thresholds**")
                    threshold_frame = pd.DataFrame([
                        {
                            "Model": DISPLAY.get(name, name),
                            "Threshold": round(float(value), 4),
                        }
                        for name, value in thresholds.items()
                    ])
                    st.dataframe(
                        threshold_frame, use_container_width=True
                    )

                    if metrics_half:
                        with st.expander("Compare with fixed 0.50 threshold"):
                            st.dataframe(
                                _metric_table(metrics_half),
                                use_container_width=True,
                            )
                else:
                    st.warning(
                        "This artifact does not contain saved test metrics."
                    )

                st.caption(f"Artifact version: {loaded.version}")

            except Exception as exc:
                st.error(f"Could not load verified artifact: {exc}")

    elif page == "Transaction Analysis":
        st.subheader("Transaction analysis")
        version_list = list_versions(settings.models_dir)
        if not version_list:
            st.warning("Train a model first.")
        else:
            default_version = st.session_state["fraud_version"]
            default_index = version_list.index(default_version) if default_version in version_list else 0
            version = st.selectbox("Model artifact", version_list, index=default_index)
            try:
                loaded = _get_loaded(settings, version)
                model_names = list(loaded.models.keys())
                model_name = st.selectbox(
                    "Model", model_names,
                    index=model_names.index("stacking") if "stacking" in model_names else 0,
                )
                X_test = st.session_state.get("fraud_X_test")
                session_version = st.session_state.get("fraud_version")
                if X_test is None or session_version != version:
                    st.info(
                        "Transaction-level analysis requires test rows from the "
                        "same training run as the selected artifact. Select the "
                        "current session's artifact or load the dataset and train "
                        "again in this session."
                    )
                else:
                    row_index = st.number_input(
                        "Test transaction row", min_value=0,
                        max_value=max(0, len(X_test) - 1), value=0, step=1,
                    )
                    saved_threshold = float(
                        loaded.manifest.get("thresholds", {}).get(model_name, 0.50)
                    )
                    threshold = st.slider(
                        "Fraud decision threshold",
                        min_value=0.01,
                        max_value=0.99,
                        value=min(0.99, max(0.01, saved_threshold)),
                        step=0.01,
                        help=(
                            "Defaults to the threshold selected on this artifact's "
                            "validation set. You can adjust it for interactive analysis."
                        ),
                    )
                    sample = pd.DataFrame(X_test[[int(row_index)]], columns=loaded.feature_names)
                    st.write("**Input transaction features**")
                    st.dataframe(sample, use_container_width=True)
                    if st.button("Run saved model prediction", type="primary"):
                        result = predict_frame(
                            loaded, model_name, sample,
                            threshold=float(threshold),
                        )
                        st.write("**Actual model output**")
                        st.dataframe(result, use_container_width=True)
                        st.caption(
                            "Prediction is generated by the selected saved model "
                            "and its persisted preprocessing pipeline."
                        )
            except Exception as exc:
                st.error(f"Inference failed: {type(exc).__name__}: {exc}")
                st.exception(exc)

    elif page == "Explainability":
        st.subheader("Explainability")
        st.write(
            "Global importance is the mean absolute value of Monte Carlo "
            "Shapley approximations. Local explanations estimate feature "
            "contributions for one transaction. These are approximations, "
            "not exact SHAP TreeExplainer results."
        )
        version_list = list_versions(settings.models_dir)
        X_test = st.session_state.get("fraud_X_test")
        feature_names = st.session_state.get("fraud_feature_names")
        if not version_list:
            st.warning("Train a model first.")
        elif (
            X_test is None
            or feature_names is None
            or st.session_state.get("fraud_version") not in version_list
        ):
            st.info("Load a dataset and train in this session to explain test transactions.")
        else:
            default_version = st.session_state["fraud_version"]
            default_index = version_list.index(default_version) if default_version in version_list else 0
            version = st.selectbox("Artifact to explain", version_list, index=default_index)
            try:
                loaded = _get_loaded(settings, version)
                if st.session_state.get("fraud_version") != version:
                    st.info(
                        "To avoid mixing artifacts and test rows, explanation "
                        "is enabled only for the model version trained in this session."
                    )
                    st.caption(f"Selected artifact: {loaded.version}")
                    st.stop()

                model_names = list(loaded.models.keys())
                model_name = st.selectbox("Model to explain", model_names)
                row_index = st.number_input(
                    "Transaction row", min_value=0,
                    max_value=max(0, len(X_test) - 1), value=0, step=1,
                )
                if st.button("Calculate Shapley approximations", type="primary"):
                    model = loaded.models[model_name]
                    X_scaled = loaded.preprocessor.transform(X_test)
                    X_scaled = np.asarray(X_scaled, dtype=float)
                    names = list(feature_names)
                    importance = get_feature_importance(
                        model, X_scaled, names,
                        n_samples=5, max_rows=20, random_state=42,
                    )
                    st.write("**Global feature importance**")
                    st.dataframe(importance, use_container_width=True)
                    fig, ax = plt.subplots(figsize=(9, 6))
                    top = importance.head(15).sort_values("importance")
                    ax.barh(top["feature"], top["importance"])
                    ax.set_xlabel("Mean absolute Shapley approximation")
                    ax.set_title("Global feature importance")
                    fig.tight_layout()
                    st.pyplot(fig)
                    local_values, base_value = manual_shap_values(
                        model,
                        X_scaled[int(row_index)],
                        X_scaled,
                        n_samples=50,
                        random_state=42,
                    )
                    local = pd.DataFrame({
                        "Feature": names,
                        "Shapley approximation": local_values,
                    })
                    local["Absolute contribution"] = local["Shapley approximation"].abs()
                    local = local.sort_values(
                        "Absolute contribution", ascending=False
                    )
                    st.write("**Local feature contributions**")
                    st.metric("Baseline fraud probability", f"{base_value:.4f}")
                    st.dataframe(local, use_container_width=True)
                    st.caption(
                        "Positive contributions raise the model's fraud probability; "
                        "negative contributions lower it, relative to the sampled baseline."
                    )
            except Exception as exc:
                st.error(f"Explanation failed: {type(exc).__name__}: {exc}")
                st.exception(exc)

    elif page == "Security & Artifacts":
        st.subheader("Security and artifact status")
        st.write(f"**Application environment:** {settings.app_env}")
        st.write(f"**Authentication disabled:** {settings.auth_disabled}")
        st.write(
            "**Artifact integrity:** Persisted model versions are verified "
            "using the project's signed-artifact verification before loading."
        )
        st.write(
            "**Prediction preprocessing:** The saved preprocessing object is "
            "loaded with the model artifact and applied during inference."
        )
        st.write(
            "**Important:** Artifact integrity signing is not the same as "
            "AES-256 encryption. This dashboard does not claim that model "
            "files are AES-256 encrypted."
        )
        versions = list_versions(settings.models_dir)
        if versions:
            st.write("**Available verified model versions**")
            for version in versions:
                st.code(version)
            selected = st.selectbox("Verify and load version", versions)
            if st.button("Verify artifact"):
                try:
                    loaded = load_version(settings, selected)
                    st.success(
                        f"Artifact verified and loaded: {loaded.version}"
                    )
                    st.write("Model entries:", list(loaded.models.keys()))
                except Exception as exc:
                    st.error(f"Artifact verification/load failed: {exc}")
        else:
            st.info("No saved model artifacts found.")


if __name__ == "__main__":
    run()
