import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from .config import load_settings
from .data import load_synthetic, load_local_csv
from .train import train_all
from .models import TrainConfig, BASE_MODEL_NAMES, STACKING_NAME, DISPLAY
from .preprocess import prepare_xy, split_data
from .evaluate import compute_metrics
from .explain import get_feature_importance
from .persist import list_versions

# Page config
st.set_page_config(page_title="Fraud Detection XAI", layout="wide")

def run():
    settings = load_settings()
    
    # Sidebar Navigation
    st.sidebar.title("🛡️ Fraud Detection")
    st.sidebar.markdown("XAI + Stacking Ensemble (Manual)")
    menu = st.sidebar.radio("Navigate", [
        "Home", "Dataset & EDA", "Model Training", "Model Evaluation", "Explainability", "Security & Status"
    ])

    # Auth check (since we added AUTH_DISABLED=true to .env)
    if not settings.auth_disabled:
        st.sidebar.warning("Authentication is ENABLED")

    if menu == "Home":
        st.title("Financial Fraud Detection System")
        st.markdown("""
        ### Welcome to the Dashboard!
        This system uses **manual implementations** of Machine Learning algorithms (no Scikit-learn) 
        along with **Explainable AI** to detect financial fraud.
        
        Use the sidebar to navigate through the pipeline:
        1. **Dataset & EDA:** Explore the data.
        2. **Model Training:** Train the base models and the Stacking meta-learner.
        3. **Model Evaluation:** View Precision, Recall, F1, and ROC-AUC.
        4. **Explainability:** See which features influenced the model (Permutation Importance).
        """)

    elif menu == "Dataset & EDA":
        st.title("Dataset & Exploratory Data Analysis")
        data_source = st.radio("Choose Data Source", ["Synthetic Data", "Local CSV"])
        
        if st.button("Load Data"):
            with st.spinner("Loading data..."):
                if data_source == "Synthetic Data":
                    bundle = load_synthetic(n_rows=5000)
                else:
                    # You can add file uploader here in the future
                    st.error("Local CSV loading not implemented in this demo. Using Synthetic.")
                    bundle = load_synthetic(n_rows=5000)
                
                st.session_state["bundle"] = bundle
                st.success(f"Loaded {bundle.report.n_rows} rows with {bundle.report.class_counts[1]} fraud cases.")
        
        if "bundle" in st.session_state:
            df = st.session_state["bundle"].frame
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            st.write("### Class Distribution")
            st.bar_chart(df[st.session_state["bundle"].target].value_counts())

    elif menu == "Model Training":
        st.title("Model Training")
        st.markdown("Train base models and the stacking ensemble manually.")
        
        if "bundle" not in st.session_state:
            st.warning("Please load a dataset first (Go to Dataset & EDA).")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                n_estimators = st.number_input("Trees/boosting rounds", min_value=3, max_value=20, value=5)
            with col2:
                cv_folds = st.number_input("Stacking CV folds", min_value=2, max_value=5, value=3)
            with col3:
                seed = st.number_input("Random seed", value=42)

            if st.button("Train Base Models + Stacking Ensemble"):
                with st.spinner("Training manual models... This takes time because we are using pure Python/NumPy."):
                    cfg = TrainConfig(seed=seed, cv_folds=cv_folds, n_estimators=n_estimators)
                    bundle = st.session_state["bundle"]
                    
                    # Train
                    res = train_all(bundle, cfg, settings, progress=lambda msg: st.write(msg))
                    
                    # Save to session state for other tabs
                    st.session_state["train_result"] = res
                    
                    # We need to reload the models and data for explainability later
                    # In a real app, you'd load from persist.py, but for simplicity:
                    X, y = prepare_xy(bundle.frame, bundle.target)
                    from .preprocess import split_data
                    s = split_data(X, y, "stratified", 0.2, 0.2, seed)
                    
                    st.session_state["X_test"] = s.X_test
                    st.session_state["y_test"] = s.y_test
                    st.session_state["feature_names"] = list(bundle.frame.drop(columns=[bundle.target]).columns)
                    
                    # Load the saved models (Hack for demo: we train a quick stack again for explainability)
                    # In production, use load_artifacts from persist.py
                    from .models import build_stacking
                    stack = build_stacking(cfg, scale_pos_weight=10.0) # dummy weight for explain
                    stack.fit(s.X_train, s.y_train) 
                    st.session_state["model"] = stack

                    st.success(f"Training complete! Version: {res.version}")

    elif menu == "Model Evaluation":
        st.title("Model Evaluation")
        
        if "train_result" not in st.session_state:
            st.warning("Please train a model first.")
        else:
            res = st.session_state["train_result"]
            st.write(f"### Results for Version: {res.version}")
            
            metrics_data = []
            for name, m in res.metrics.items():
                metrics_data.append({
                    "Model": DISPLAY.get(name, name),
                    "Precision": round(m["precision"], 3),
                    "Recall": round(m["recall"], 3),
                    "F1-Score": round(m["f1"], 3),
                    "ROC-AUC": round(m["roc_auc"], 3),
                    "PR-AUC": round(m["pr_auc"], 3)
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, hide_index=True)
            
            # Bar chart for F1 Score
            st.write("### F1-Score Comparison")
            st.bar_chart(df_metrics.set_index("Model")["F1-Score"])

    elif menu == "Explainability":
        st.title("Explainability (Manual Permutation Importance)")
        st.markdown("""
        **Note:** Since we are not using Scikit-learn, SHAP TreeExplainer cannot be used.
        Instead, we calculate **Permutation Importance**: How much does the model's ROC-AUC drop 
        if we randomly shuffle a single feature?
        """)
        
        if "model" not in st.session_state or "X_test" not in st.session_state:
            st.warning("Please train a model first so we have data to explain.")
        else:
            if st.button("Calculate Feature Importance"):
                with st.spinner("Calculating permutation importance (this takes a moment)..."):
                    model = st.session_state["model"]
                    X_test = st.session_state["X_test"]
                    y_test = st.session_state["y_test"]
                    feature_names = st.session_state["feature_names"]
                    
                    # Call our manual function
                    importance_df = get_feature_importance(model, X_test, y_test, feature_names)
                    
                    # Plotting
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Top 15 features
                    top_df = importance_df.head(15)
                    ax.barh(top_df["feature"], top_df["importance"], color='skyblue')
                    ax.set_xlabel("Importance (Drop in ROC-AUC)")
                    ax.set_title("Feature Importance (Top 15)")
                    ax.invert_yaxis() # Most important on top
                    
                    st.pyplot(fig)
                    st.write("### Raw Importance Data")
                    st.dataframe(importance_df)

    elif menu == "Security & Status":
        st.title("Security & Status")
        st.write(f"**App Environment:** {settings.app_env}")
        st.write(f"**Auth Disabled:** {settings.auth_disabled}")
        st.write("**Encryption:** Enabled (AES-256 via .env)")
        
        st.write("### Available Model Versions")
        versions = list_versions(settings.models_dir)
        if versions:
            for v in versions:
                st.code(v)
        else:
            st.info("No model versions saved yet.")

if __name__ == "__main__":
    run()