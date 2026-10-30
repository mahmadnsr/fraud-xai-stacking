# fraud_xai/explain.py
import numpy as np
import pandas as pd
from .manual_ml import manual_feature_importance

# Note: SHAP library use kar sakte hain kyunki wo sklearn nahi hai, par manual importance bhi de rahe hain
def get_feature_importance(model, X, y, feature_names):
    importances = manual_feature_importance(model, X, y)
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)