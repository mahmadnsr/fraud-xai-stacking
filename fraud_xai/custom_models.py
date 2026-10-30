# fraud_xai/custom_models.py
import numpy as np

# 1. Manual Logistic Regression (Using Gradient Descent)
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, class_weight='balanced'):
        self.lr = learning_rate
        self.epochs = epochs
        self.class_weight = class_weight
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Handle class imbalance
        if self.class_weight == 'balanced':
            weights = np.where(y == 1, len(y) / (2 * np.sum(y == 1)), len(y) / (2 * np.sum(y == 0)))
        else:
            weights = np.ones(n_samples)

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = 1 / (1 + np.exp(-linear_model)) # Sigmoid
            
            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y) * weights)
            db = (1 / n_samples) * np.sum((y_predicted - y) * weights)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        prob = 1 / (1 + np.exp(-linear_model))
        return np.vstack([1 - prob, prob]).T

# 2. Manual Decision Tree (Simplified for classification)
class CustomDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini -= p ** 2
        return gini

    def _split(self, X, y):
        best_gini, best_idx, best_thresh = 1.0, None, None
        for idx in range(X.shape[1]):
            thresholds = np.unique(X[:, idx])
            for thresh in thresholds:
                left_mask = X[:, idx] <= thresh
                right_mask = ~left_mask
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gini = (len(y[left_mask]) / len(y)) * self._gini(y[left_mask]) + \
                       (len(y[right_mask]) / len(y)) * self._gini(y[right_mask])
                if gini < best_gini:
                    best_gini, best_idx, best_thresh = gini, idx, thresh
        return best_idx, best_thresh

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return np.bincount(y.astype(int)).argmax()
        
        idx, thresh = self._split(X, y)
        if idx is None:
            return np.bincount(y.astype(int)).argmax()
        
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
        if not isinstance(tree, dict):
            return tree
        if x[tree['idx']] <= tree['thresh']:
            return self._predict_single(x, tree['left'])
        return self._predict_single(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def predict_proba(self, X):
        preds = self.predict(X)
        return np.vstack([1 - preds, preds]).T

# 3. Manual Stacking Ensemble (Ahmad's Task)
class CustomStackingClassifier:
    def __init__(self, base_models, meta_model, cv=3):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv

    def fit(self, X, y):
        # Split data into K folds
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.cv)
        
        # Out-of-fold predictions for meta-features
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, (train_idx, val_idx) in enumerate([(np.concatenate([folds[j] for j in range(self.cv) if j != i]), folds[i]) for i in range(self.cv)]):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            
            for j, model in enumerate(self.base_models):
                # Clone and train base model on fold
                import copy
                m = copy.deepcopy(model)
                m.fit(X_train, y_train)
                meta_features[val_idx, j] = m.predict_proba(X_val)[:, 1]

        # Train meta-learner on the generated meta-features
        self.meta_model.fit(meta_features, y)
        
        # Refit base models on full data
        for model in self.base_models:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        meta_features = np.zeros((len(X), len(self.base_models)))
        for j, model in enumerate(self.base_models):
            meta_features[:, j] = model.predict_proba(X)[:, 1]
        return self.meta_model.predict_proba(meta_features)