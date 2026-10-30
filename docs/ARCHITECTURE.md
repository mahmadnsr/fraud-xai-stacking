# System Architecture

## Layers
Streamlit UI (`ui.py`) -> auth/RBAC (`auth.py`) -> services (`data`, `train`, `predict`, `explain`) -> persistence with integrity checks (`persist.py`) -> encrypted/signed files on local disk.

## Data-flow diagram
```mermaid
flowchart LR
  U[User / Analyst] -->|CSV upload or sample| V[Schema validation<br/>size, content, numeric, target]
  V -->|validated frame in memory| P[Dedup + stratified/chronological split]
  P --> T[Train 4 base Pipelines]
  P --> S[Stacking: 5-fold OOF -> LR meta-learner]
  T --> E[Validation: threshold<br/>Test: final metrics]
  S --> E
  E --> A[(Signed artifacts<br/>models.joblib, manifest + HMAC)]
  P -->|small sample| X[(AES-256-GCM encrypted explain sample)]
  A --> I[Inference: verify -> load -> predict_proba]
  I --> D[Dashboard: score, risk]
  A --> H[SHAP: Tree / Kernel]
  X --> H --> D
```

## UML use-case diagram
```mermaid
flowchart LR
  An((Analyst)) --- UC1([View EDA / evaluation])
  An --- UC2([Predict single / batch])
  An --- UC3([Inspect SHAP explanation])
  An --- UC4([View security status])
  Ad((Admin)) --- UC5([Load dataset])
  Ad --- UC6([Train models + ensemble])
  Ad --- UC7([Select model version])
  Ad -. inherits .-> An
  UC6 -. include .-> UC8([Validate dataset])
  UC2 -. include .-> UC9([Verify artifacts])
```

## UML component diagram
```mermaid
flowchart TB
  subgraph UI[Presentation]
    app[app.py] --> ui[ui.py] --> eda[eda.py]
  end
  subgraph Core[Domain]
    schema[schema.py] --> data[data.py] --> preprocess[preprocess.py] --> models[models.py] --> train[train.py]
    train --> evaluate[evaluate.py]
    predict[predict.py]
    explain[explain.py]
  end
  subgraph Sec[Security & infrastructure]
    config[config.py]; auth[auth.py]; security[security.py]; persist[persist.py]; log[logging_utils.py]
  end
  ui --> auth & data & train & predict & explain & evaluate
  train --> persist --> security
  config --> security
```

## Training and inference workflow
1. Validate (size, UTF-8, header, names, numeric, finite, binary target, class minimums, sensitive columns excluded).
2. Remove exact duplicate rows, then split (stratified 60/20/20, or chronological when `Time` exists).
3. Fit four `Pipeline(preprocessor -> classifier)` objects on the **training split only**; fit `StackingClassifier` on the same training split.
4. Predict probabilities on validation (choose F1-optimal threshold per model) and test (report metrics).
5. Save artifacts atomically, sign manifest. Inference loads only after signature + SHA-256 verification and uses the stored pipelines (identical preprocessing).

## Stacking architecture
Level-0: Logistic Regression (scaled), Random Forest, XGBoost, LightGBM - each a Pipeline with median imputation.
Level-1: `LogisticRegression` on the 4 out-of-fold fraud probabilities (`StackingClassifier(cv=StratifiedKFold(5), stack_method="predict_proba", passthrough=False)`).
After the meta-learner is fitted, base learners are refit on the whole training split for inference (scikit-learn behaviour).
