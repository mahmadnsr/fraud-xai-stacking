# fraud_xai/manual_ml.py
import numpy as np
import pandas as pd
import copy

# ==========================================
# 1. MANUAL PREPROCESSING & METRICS
# ==========================================

class ManualScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
    def transform(self, X):
        return (X - self.mean) / self.std
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def manual_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_len = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_len], indices[test_len:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def manual_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # ROC-AUC Calculation (Manual)
    desc_score_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    
    tpr = tps / (tps[-1] + 1e-8)
    fpr = fps / (fps[-1] + 1e-8)
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2.0
    
    # PR-AUC Calculation (Manual)
    precision_curve = tps / (tps + fps + 1e-8)
    recall_curve = tps / (tps[-1] + 1e-8)
    pr_auc = np.sum((recall_curve[1:] - recall_curve[:-1]) * (precision_curve[1:] + precision_curve[:-1])) / 2.0
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "roc_auc": abs(auc), 
        "pr_auc": abs(pr_auc)
    }
# ==========================================
# 2. MANUAL MODELS
# ==========================================

class ManualLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Class imbalance handle karne ke liye weights
        pos_weight = (n_samples - np.sum(y)) / (np.sum(y) + 1e-8)
        sample_weights = np.where(y == 1, pos_weight, 1.0)

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = 1 / (1 + np.exp(-np.clip(linear_model, -250, 250)))
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y) * sample_weights)
            db = (1 / n_samples) * np.sum((y_pred - y) * sample_weights)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        prob = 1 / (1 + np.exp(-np.clip(linear_model, -250, 250)))
        return np.vstack([1 - prob, prob]).T # 2D array return karta hai

class ManualDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini -= p**2
        return gini

    def _split(self, X, y):
        # Continuous targets (residuals) ke liye Variance (MSE) use karega
        best_var, best_idx, best_thresh = float('inf'), None, None
        
        # Sirf 10 random features check karo (Speed badhane ke liye)
        features = np.random.choice(X.shape[1], min(10, X.shape[1]), replace=False)
        
        for idx in features:
            thresh = np.median(X[:, idx]) 
            left_mask = X[:, idx] <= thresh
            right_mask = ~left_mask
            
            if len(y[left_mask]) == 0 or len(y[right_mask]) == 0: 
                continue
            
            # Variance / MSE Calculation
            left_var = np.var(y[left_mask]) if len(y[left_mask]) > 0 else 0
            right_var = np.var(y[right_mask]) if len(y[right_mask]) > 0 else 0
            var = (len(y[left_mask])/len(y)) * left_var + (len(y[right_mask])/len(y)) * right_var
            
            if var < best_var:
                best_var, best_idx, best_thresh = var, idx, thresh
                
        return best_idx, best_thresh
        # Continuous targets (residuals) ke liye Variance use karega
        best_var, best_idx, best_thresh = float('inf'), None, None
        
        features = np.random.choice(X.shape[1], min(10, X.shape[1]), replace=False)
        
        for idx in features:
            thresh = np.median(X[:, idx]) 
            left_mask = X[:, idx] <= thresh
            right_mask = ~left_mask
            
            if len(y[left_mask]) == 0 or len(y[right_mask]) == 0: 
                continue
            
            # Variance / MSE Calculation
            left_var = np.var(y[left_mask]) if len(y[left_mask]) > 0 else 0
            right_var = np.var(y[right_mask]) if len(y[right_mask]) > 0 else 0
            var = (len(y[left_mask])/len(y)) * left_var + (len(y[right_mask])/len(y)) * right_var
            
            if var < best_var:
                best_var, best_idx, best_thresh = var, idx, thresh
                
        return best_idx, best_thresh
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return np.mean(y)
        
        idx, thresh = self._split(X, y)
        if idx is None: return np.mean(y)
        
        left_mask = X[:, idx] <= thresh
        right_mask = ~left_mask
        
        return {
            'idx': idx, 'thresh': thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict): return tree
        if x[tree['idx']] <= tree['thresh']: return self._predict_single(x, tree['left'])
        return self._predict_single(x, tree['right'])

    def predict_proba(self, X):
        prob = np.array([self._predict_single(x, self.tree) for x in X])
        prob = np.clip(prob, 0.001, 0.999)
        return np.vstack([1 - prob, prob]).T

class ManualRandomForest:
    def __init__(self, n_estimators=5, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = len(X)
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            tree = ManualDecisionTree(max_depth=self.max_depth)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(preds, axis=0)

class ManualGradientBoosting:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=2):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        self.trees = []
        self.base_pred = np.log(np.mean(y) / (1 - np.mean(y) + 1e-8))
        current_pred = np.full(len(y), self.base_pred)
        
        for _ in range(self.n_estimators):
            p = 1 / (1 + np.exp(-np.clip(current_pred, -250, 250)))
            residuals = y - p
            
            tree = ManualDecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            update = tree.predict_proba(X)[:, 1]
            current_pred += self.lr * update
        return self

    def predict_proba(self, X):
        current_pred = np.full(len(X), self.base_pred)
        for tree in self.trees:
            current_pred += self.lr * tree.predict_proba(X)[:, 1]
        prob = 1 / (1 + np.exp(-np.clip(current_pred, -250, 250)))
        return np.vstack([1 - prob, prob]).T

class ManualStackingClassifier:
    def __init__(self, base_models, meta_model, cv=3):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv

    def fit(self, X, y):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.cv)
        
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i in range(self.cv):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(self.cv) if j != i])
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            
            for j, model in enumerate(self.base_models):
                m = copy.deepcopy(model)
                m.fit(X_train, y_train)
                meta_features[val_idx, j] = m.predict_proba(X_val)[:, 1]

        self.meta_model.fit(meta_features, y)
        
        for model in self.base_models:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        meta_features = np.zeros((len(X), len(self.base_models)))
        for j, model in enumerate(self.base_models):
            meta_features[:, j] = model.predict_proba(X)[:, 1]
        return self.meta_model.predict_proba(meta_features)

# ==========================================
# 3. MANUAL SHAP (Permutation Importance)
# ==========================================
def manual_feature_importance(model, X, y):
    base_pred = model.predict_proba(X)[:, 1]
    base_score = manual_metrics(y, base_pred)['roc_auc']
    importances = []
    
    for col in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, col])
        perm_pred = model.predict_proba(X_permuted)[:, 1]
        perm_score = manual_metrics(y, perm_pred)['roc_auc']
        importances.append(base_score - perm_score)
        
    return np.array(importances)