"""Dependency-free, deterministic implementations of core ML algorithms."""
from __future__ import annotations

import copy
import numpy as np


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _validate_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or not len(y):
        raise ValueError("X must be a non-empty 2D array aligned with y.")
    # Missing feature values are imputed by ManualScaler after splitting.
    # Infinite values and invalid targets are never allowed.
    if np.isinf(X).any():
        raise ValueError("Feature data cannot contain infinity.")
    if not np.isfinite(y).all():
        raise ValueError("Target data must contain only finite values.")
    return X, y


class ManualScaler:
    """Median imputation followed by standardisation, fitted on training data only."""
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X):
            raise ValueError("X must be a non-empty 2D array.")
        if np.isinf(X).any():
            raise ValueError("Infinite feature values are not supported.")
        with np.errstate(all="ignore"):
            self.median_ = np.nanmedian(X, axis=0)
        if not np.isfinite(self.median_).all():
            raise ValueError("A feature is entirely missing in the training data.")
        filled = np.where(np.isnan(X), self.median_, X)
        self.mean_ = np.mean(filled, axis=0)
        self.scale_ = np.std(filled, axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "median_"):
            raise RuntimeError("Scaler has not been fitted.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match fitted scaler.")
        if np.isinf(X).any():
            raise ValueError("Infinite feature values are not supported.")
        X = np.where(np.isnan(X), self.median_, X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _stratified_indices(y, test_size, rng):
    y = np.asarray(y)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    train_parts, test_parts = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        if len(idx) < 2:
            raise ValueError("Each class needs at least two rows for stratified splitting.")
        n_test = min(len(idx) - 1, max(1, int(round(len(idx) * test_size))))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def manual_train_test_split(X, y, test_size=0.2, random_state=42, strategy="stratified"):
    X, y = _validate_xy(X, y)
    rng = np.random.default_rng(random_state)
    if strategy == "stratified":
        train_idx, test_idx = _stratified_indices(y, test_size, rng)
    elif strategy == "random":
        idx = rng.permutation(len(y))
        n_test = max(1, min(len(y) - 1, int(round(len(y) * test_size))))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
    else:
        raise ValueError("strategy must be 'stratified' or 'random'.")
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def manual_metrics(y_true, y_pred_proba, threshold=0.5):
    from .evaluate import compute_metrics
    return compute_metrics(y_true, y_pred_proba, threshold)


class ManualLogisticRegression:
    def __init__(self, learning_rate=0.05, epochs=300, l2=1e-4,
                 scale_pos_weight=None):
        if scale_pos_weight is not None and (
            not np.isfinite(scale_pos_weight) or scale_pos_weight <= 0
        ):
            raise ValueError("scale_pos_weight must be positive and finite.")
        self.lr, self.epochs, self.l2 = learning_rate, epochs, l2
        self.scale_pos_weight = scale_pos_weight

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        if not set(np.unique(y)).issubset({0.0, 1.0}) or len(np.unique(y)) != 2:
            raise ValueError("Logistic regression requires both binary classes.")
        n, p = X.shape
        self.weights = np.zeros(p)
        prior = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.bias = float(np.log(prior / (1 - prior)))
        pos_weight = (
            self.scale_pos_weight
            if self.scale_pos_weight is not None
            else (n - y.sum()) / max(y.sum(), 1.0)
        )
        sw = np.where(y == 1, pos_weight, 1.0)
        sw = sw / np.mean(sw)
        for _ in range(self.epochs):
            prob = _sigmoid(X @ self.weights + self.bias)
            error = (prob - y) * sw
            self.weights -= self.lr * ((X.T @ error) / n + self.l2 * self.weights)
            self.bias -= self.lr * np.mean(error)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("Feature count does not match fitted logistic regression.")
        p = _sigmoid(X @ self.weights + self.bias)
        return np.column_stack((1 - p, p))


class ManualDecisionTree:
    """Binary Gini classification tree or squared-error regression tree."""
    def __init__(self, max_depth=3, min_samples_split=4, max_features=None,
                 criterion="gini", random_state=42, max_thresholds=16):
        if criterion not in ("gini", "mse"):
            raise ValueError("criterion must be 'gini' or 'mse'.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.max_thresholds = max_thresholds

    def _impurity(self, y):
        if self.criterion == "mse":
            return float(np.var(y))
        p = float(np.mean(y))
        return 2.0 * p * (1.0 - p)

    def _best_split(self, X, y, rng):
        n, p = X.shape
        count = p if self.max_features is None else min(p, self.max_features)
        features = rng.choice(p, count, replace=False)
        parent = self._impurity(y)
        best_gain, best = 0.0, None

        for j in features:
            values = np.unique(X[:, j])
            if len(values) < 2:
                continue
            if len(values) > self.max_thresholds + 1:
                qs = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
                thresholds = np.unique(np.quantile(values, qs))
            else:
                thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left = X[:, j] <= threshold
                nl = int(left.sum())
                nr = n - nl
                if nl < 1 or nr < 1:
                    continue
                child = (nl / n) * self._impurity(y[left]) + (nr / n) * self._impurity(y[~left])
                gain = parent - child
                if gain > best_gain + 1e-12:
                    best_gain, best = gain, (int(j), float(threshold))
        return best

    def _build(self, X, y, depth, rng):
        if self.criterion == "gini":
            value = float(np.mean(y))
            pure = value <= 0 or value >= 1
        else:
            value = float(np.mean(y))
            pure = float(np.var(y)) <= 1e-14
        if depth >= self.max_depth or len(y) < self.min_samples_split or pure:
            return value
        split = self._best_split(X, y, rng)
        if split is None:
            return value
        j, threshold = split
        left = X[:, j] <= threshold
        return {
            "feature": j, "threshold": threshold,
            "left": self._build(X[left], y[left], depth + 1, rng),
            "right": self._build(X[~left], y[~left], depth + 1, rng),
        }

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        if self.criterion == "gini" and not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("Gini classification tree requires binary labels.")
        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build(X, y, 0, np.random.default_rng(self.random_state))
        return self

    def _predict_one(self, row, node):
        while isinstance(node, dict):
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return float(node)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match fitted tree.")
        return np.array([self._predict_one(row, self.tree_) for row in X])

    def predict_proba(self, X):
        if self.criterion != "gini":
            raise RuntimeError("Regression tree does not provide class probabilities.")
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack((1 - p, p))


class ManualRandomForest:
    def __init__(self, n_estimators=25, max_depth=5, random_state=42, max_features="sqrt"):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.random_state, self.max_features = random_state, max_features

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("Random forest requires binary labels.")
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        p = X.shape[1]
        mf = max(1, int(np.sqrt(p))) if self.max_features == "sqrt" else int(self.max_features)
        for i in range(self.n_estimators):
            idx = rng.integers(0, len(y), size=len(y))
            tree = ManualDecisionTree(
                max_depth=self.max_depth, min_samples_split=4,
                max_features=mf, criterion="gini",
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        self.n_features_in_ = p
        return self

    def predict_proba(self, X):
        if not self.trees:
            raise RuntimeError("Random forest has not been fitted.")
        return np.mean([tree.predict_proba(X) for tree in self.trees], axis=0)


class ManualGradientBoosting:
    """Gradient boosting for binary logistic loss using regression trees on pseudo-residuals."""
    def __init__(self, n_estimators=25, learning_rate=0.05, max_depth=2, random_state=42):
        self.n_estimators, self.lr = n_estimators, learning_rate
        self.max_depth, self.random_state = max_depth, random_state

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        if not set(np.unique(y)).issubset({0.0, 1.0}) or len(np.unique(y)) != 2:
            raise ValueError("Gradient boosting requires both binary classes.")
        prior = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.base_pred = float(np.log(prior / (1 - prior)))
        raw = np.full(len(y), self.base_pred)
        self.trees = []
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            prob = _sigmoid(raw)
            residual = y - prob
            tree = ManualDecisionTree(
                max_depth=self.max_depth, min_samples_split=4,
                criterion="mse", random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X, residual)
            update = tree.predict(X)
            raw += self.lr * update
            self.trees.append(tree)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match fitted gradient booster.")
        raw = np.full(len(X), self.base_pred)
        for tree in self.trees:
            raw += self.lr * tree.predict(X)
        p = _sigmoid(raw)
        return np.column_stack((1 - p, p))




class _ManualNewtonBooster:
    """Shared second-order logistic boosting primitives; no external ML library."""

    def __init__(
        self, n_estimators=10, learning_rate=0.05, max_depth=3,
        reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
        min_samples_leaf=2, scale_pos_weight=1.0, random_state=42
    ):
        if n_estimators < 1 or learning_rate <= 0 or max_depth < 1:
            raise ValueError("Invalid boosting configuration.")
        if reg_lambda < 0 or gamma < 0 or min_child_weight < 0:
            raise ValueError("Regularisation parameters cannot be negative.")
        if min_samples_leaf < 1 or scale_pos_weight <= 0:
            raise ValueError("Invalid leaf size or class weight.")
        self.n_estimators = int(n_estimators)
        self.lr = float(learning_rate)
        self.max_depth = int(max_depth)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.min_samples_leaf = int(min_samples_leaf)
        self.scale_pos_weight = float(scale_pos_weight)
        self.random_state = int(random_state)

    def _leaf_value(self, grad, hess, indices):
        g = float(np.sum(grad[indices]))
        h = float(np.sum(hess[indices]))
        return -g / (h + self.reg_lambda)

    def _gain(self, grad, hess, left, right):
        gl, gr = float(grad[left].sum()), float(grad[right].sum())
        hl, hr = float(hess[left].sum()), float(hess[right].sum())
        if (
            len(left) < self.min_samples_leaf
            or len(right) < self.min_samples_leaf
            or hl < self.min_child_weight
            or hr < self.min_child_weight
        ):
            return None
        gp, hp = gl + gr, hl + hr
        gain = (
            gl * gl / (hl + self.reg_lambda)
            + gr * gr / (hr + self.reg_lambda)
            - gp * gp / (hp + self.reg_lambda)
            - self.gamma
        )
        return float(gain) if np.isfinite(gain) else None

    @staticmethod
    def _predict_tree_one(row, node):
        while "value" not in node:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]

    def _tree_predict(self, X, tree):
        return np.fromiter(
            (self._predict_tree_one(row, tree) for row in X),
            dtype=float, count=len(X)
        )

    def _initialise(self, X, y):
        X, y = _validate_xy(X, y)
        if not set(np.unique(y)).issubset({0.0, 1.0}) or len(np.unique(y)) != 2:
            raise ValueError("Boosting requires both binary classes.")
        self.n_features_in_ = X.shape[1]
        weights = np.where(y == 1, self.scale_pos_weight, 1.0)
        prior = np.sum(weights * y) / np.sum(weights)
        prior = float(np.clip(prior, 1e-8, 1 - 1e-8))
        self.base_pred_ = float(np.log(prior / (1 - prior)))
        raw = np.full(len(y), self.base_pred_, dtype=float)
        self.trees_ = []
        return X, y, weights, raw

    def _fit_boosting(self, X, y, tree_builder):
        X, y, weights, raw = self._initialise(X, y)
        for _ in range(self.n_estimators):
            prob = _sigmoid(raw)
            grad = (prob - y) * weights
            hess = np.maximum(prob * (1.0 - prob) * weights, 1e-8)
            tree = tree_builder(X, grad, hess)
            raw += self.lr * self._tree_predict(X, tree)
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "trees_"):
            raise RuntimeError("Booster has not been fitted.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count does not match fitted booster.")
        if not np.isfinite(X).all():
            raise ValueError("Prediction features must be finite.")
        raw = np.full(len(X), self.base_pred_, dtype=float)
        for tree in self.trees_:
            raw += self.lr * self._tree_predict(X, tree)
        p = _sigmoid(raw)
        return np.column_stack((1.0 - p, p))


class ManualXGBoost(_ManualNewtonBooster):
    """
    XGBoost-inspired manual binary booster.
    Uses logistic gradients/Hessians, L2-regularised split gain,
    gamma pruning and recursive depth-wise tree construction.
    """

    def __init__(self, n_estimators=10, learning_rate=0.05, max_depth=3,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
                 min_samples_leaf=2, scale_pos_weight=1.0,
                 max_thresholds=16, random_state=42):
        super().__init__(
            n_estimators, learning_rate, max_depth, reg_lambda, gamma,
            min_child_weight, min_samples_leaf, scale_pos_weight, random_state
        )
        if max_thresholds < 2:
            raise ValueError("max_thresholds must be at least 2.")
        self.max_thresholds = int(max_thresholds)

    def _thresholds(self, values):
        unique = np.unique(values)
        if len(unique) < 2:
            return np.empty(0)
        if len(unique) > self.max_thresholds + 1:
            qs = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
            cuts = np.unique(np.quantile(unique, qs))
        else:
            cuts = (unique[:-1] + unique[1:]) / 2.0
        return cuts

    def _best_split(self, X, grad, hess, indices):
        best_gain, best = self.gamma, None
        for feature in range(X.shape[1]):
            values = X[indices, feature]
            for threshold in self._thresholds(values):
                mask = values <= threshold
                left, right = indices[mask], indices[~mask]
                gain = self._gain(grad, hess, left, right)
                if gain is not None and gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (feature, float(threshold), left, right)
        return best_gain, best

    def _build_depthwise(self, X, grad, hess, indices, depth):
        leaf = {"value": self._leaf_value(grad, hess, indices)}
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return leaf
        _, split = self._best_split(X, grad, hess, indices)
        if split is None:
            return leaf
        feature, threshold, left, right = split
        return {
            "feature": feature, "threshold": threshold,
            "left": self._build_depthwise(X, grad, hess, left, depth + 1),
            "right": self._build_depthwise(X, grad, hess, right, depth + 1),
        }

    def fit(self, X, y):
        def builder(features, grad, hess):
            return self._build_depthwise(
                features, grad, hess, np.arange(len(grad)), 0
            )
        return self._fit_boosting(X, y, builder)


class ManualLightGBM(_ManualNewtonBooster):
    """
    LightGBM-inspired manual binary booster.
    Uses quantile-derived histogram bin thresholds and grows the leaf
    with the largest regularised second-order gain until max_leaves.
    """

    def __init__(self, n_estimators=10, learning_rate=0.05, max_depth=4,
                 max_leaves=7, max_bins=16, reg_lambda=1.0, gamma=0.0,
                 min_child_weight=1.0, min_samples_leaf=2,
                 scale_pos_weight=1.0, random_state=42):
        super().__init__(
            n_estimators, learning_rate, max_depth, reg_lambda, gamma,
            min_child_weight, min_samples_leaf, scale_pos_weight, random_state
        )
        if max_leaves < 2 or max_bins < 2:
            raise ValueError("max_leaves and max_bins must be at least 2.")
        self.max_leaves = int(max_leaves)
        self.max_bins = int(max_bins)

    def _build_leafwise(self, X, grad, hess):
        # Histogram cut points are estimated once per tree, then reused
        # by every leaf, rather than searching all raw unique values.
        cuts = []
        quantiles = np.linspace(0, 1, self.max_bins + 1)[1:-1]
        for j in range(X.shape[1]):
            values = np.unique(X[:, j])
            if len(values) < 2:
                cuts.append(np.empty(0))
            else:
                cuts.append(np.unique(np.quantile(values, quantiles)))

        root_idx = np.arange(len(grad))
        root = {
            "indices": root_idx,
            "depth": 0,
            "value": self._leaf_value(grad, hess, root_idx),
        }

        def best_for_leaf(node):
            idx = node["indices"]
            if node["depth"] >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
                return None
            best_gain, best = self.gamma, None
            for feature, thresholds in enumerate(cuts):
                vals = X[idx, feature]
                for threshold in thresholds:
                    mask = vals <= threshold
                    left, right = idx[mask], idx[~mask]
                    gain = self._gain(grad, hess, left, right)
                    if gain is not None and gain > best_gain + 1e-12:
                        best_gain = gain
                        best = (feature, float(threshold), left, right)
            if best is None:
                return None
            return best_gain, best

        leaves = [root]
        candidates = {id(root): best_for_leaf(root)}

        while len(leaves) < self.max_leaves:
            eligible = [
                (candidate[0], node, candidate[1])
                for node in leaves
                if (candidate := candidates.get(id(node))) is not None
            ]
            if not eligible:
                break
            _, node, split = max(eligible, key=lambda item: item[0])
            feature, threshold, left_idx, right_idx = split
            left = {
                "indices": left_idx, "depth": node["depth"] + 1,
                "value": self._leaf_value(grad, hess, left_idx),
            }
            right = {
                "indices": right_idx, "depth": node["depth"] + 1,
                "value": self._leaf_value(grad, hess, right_idx),
            }
            node.clear()
            node.update({
                "feature": feature, "threshold": threshold,
                "left": left, "right": right,
            })
            leaves.remove(node)
            candidates.pop(id(node), None)
            leaves.extend((left, right))
            candidates[id(left)] = best_for_leaf(left)
            candidates[id(right)] = best_for_leaf(right)

        # Remove training row indices from terminal nodes before persistence.
        for leaf in leaves:
            leaf.pop("indices", None)
            leaf.pop("depth", None)
        return root

    def fit(self, X, y):
        return self._fit_boosting(X, y, self._build_leafwise)


class ManualStackingClassifier:
    def __init__(
        self, base_models, meta_model, cv=3, random_state=42,
        split_strategy="stratified"
    ):
        self.base_models, self.meta_model = base_models, meta_model
        self.cv, self.random_state = cv, random_state
        self.split_strategy = split_strategy

    @staticmethod
    def _set_fold_class_weight(model, y_train):
        """Calculate class weight using only this OOF fold's training labels."""
        if hasattr(model, "scale_pos_weight"):
            positives = int(np.sum(y_train == 1))
            negatives = int(np.sum(y_train == 0))
            if positives == 0 or negatives == 0:
                raise ValueError("Each OOF training fold must contain both classes.")
            model.scale_pos_weight = float(negatives / positives)
        return model

    def fit(self, X, y, time_values=None):
        X, y = _validate_xy(X, y)
        if self.cv < 2:
            raise ValueError("Stacking requires at least two folds.")

        if self.split_strategy == "chronological":
            if time_values is None:
                raise ValueError(
                    "time_values are required for timestamp-safe chronological stacking."
                )

            time_values = np.asarray(time_values)
            if time_values.ndim != 1 or len(time_values) != len(y):
                raise ValueError("time_values must be one-dimensional and match y.")

            try:
                time_values = time_values.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Chronological timestamps must be numeric.") from exc

            if not np.all(np.isfinite(time_values)):
                raise ValueError("time_values cannot contain missing or non-finite values.")

            if np.any(time_values[1:] < time_values[:-1]):
                raise ValueError("Chronological stacking requires ascending timestamps.")

            # Group equal timestamps before dividing into OOF blocks.
            group_starts = np.flatnonzero(
                np.r_[True, time_values[1:] != time_values[:-1]]
            )
            group_ends = np.r_[group_starts[1:], len(y)]

            if len(group_starts) < self.cv + 1:
                raise ValueError(
                    "Not enough distinct timestamp groups for chronological "
                    "stacking; reduce cv_folds or provide more training data."
                )

            group_chunks = np.array_split(
                np.arange(len(group_starts)), self.cv + 1
            )
            blocks = [
                np.arange(group_starts[g[0]], group_ends[g[-1]])
                for g in group_chunks
            ]

            if len(blocks[0]) == 0 or len(np.unique(y[blocks[0]])) < 2:
                raise ValueError(
                    "Chronological stacking warm-up block must contain "
                    "both classes. Increase training data or reduce cv_folds."
                )

            meta_parts, target_parts = [], []

            for k in range(1, len(blocks)):
                val_idx = blocks[k]
                if len(val_idx) == 0:
                    continue

                train_idx = np.arange(val_idx[0])
                if len(np.unique(y[train_idx])) < 2:
                    raise ValueError(
                        f"Chronological stacking fold {k} has fewer than "
                        "two classes in its historical training window."
                    )

                fold_meta = np.zeros(
                    (len(val_idx), len(self.base_models)), dtype=float
                )
                for j, model in enumerate(self.base_models):
                    fitted = copy.deepcopy(model)
                    self._set_fold_class_weight(fitted, y[train_idx])
                    fitted.fit(X[train_idx], y[train_idx])
                    fold_meta[:, j] = fitted.predict_proba(X[val_idx])[:, 1]

                meta_parts.append(fold_meta)
                target_parts.append(y[val_idx])

            if not meta_parts:
                raise ValueError("No chronological OOF predictions were generated.")

            meta = np.vstack(meta_parts)
            meta_y = np.concatenate(target_parts)

            if len(np.unique(meta_y)) < 2:
                raise ValueError(
                    "Chronological OOF meta-data must contain both classes. "
                    "Adjust cv_folds or the chronological training window."
                )

            self.meta_model.fit(meta, meta_y)

        elif self.split_strategy == "stratified":
            rng = np.random.default_rng(self.random_state)
            class_folds = [[] for _ in range(self.cv)]

            for cls in np.unique(y):
                idx = np.flatnonzero(y == cls)
                if len(idx) < self.cv:
                    raise ValueError(
                        "Each class must have at least cv rows for stratified stacking."
                    )
                rng.shuffle(idx)
                for k, part in enumerate(np.array_split(idx, self.cv)):
                    class_folds[k].extend(part.tolist())

            folds = [np.asarray(f, dtype=int) for f in class_folds]
            meta = np.zeros((len(y), len(self.base_models)), dtype=float)

            for k, val_idx in enumerate(folds):
                train_idx = np.concatenate(
                    [folds[j] for j in range(self.cv) if j != k]
                )
                for j, model in enumerate(self.base_models):
                    fitted = copy.deepcopy(model)
                    self._set_fold_class_weight(fitted, y[train_idx])
                    fitted.fit(X[train_idx], y[train_idx])
                    meta[val_idx, j] = fitted.predict_proba(X[val_idx])[:, 1]

            self.meta_model.fit(meta, y)

        else:
            raise ValueError(
                "split_strategy must be 'stratified' or 'chronological'."
            )

        self.fitted_base_models_ = []
        for model in self.base_models:
            fitted = copy.deepcopy(model)
            self._set_fold_class_weight(fitted, y)
            fitted.fit(X, y)
            self.fitted_base_models_.append(fitted)

        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        meta = np.column_stack([m.predict_proba(X)[:, 1] for m in self.fitted_base_models_])
        return self.meta_model.predict_proba(meta)


def manual_feature_importance(model, X, y, random_state=42):
    """Permutation importance; this is not SHAP."""
    X, y = _validate_xy(X, y)
    from .evaluate import compute_metrics
    base = compute_metrics(y, model.predict_proba(X)[:, 1])["roc_auc"]
    rng = np.random.default_rng(random_state)
    result = []
    for j in range(X.shape[1]):
        permuted = X.copy()
        permuted[:, j] = rng.permutation(permuted[:, j])
        score = compute_metrics(y, model.predict_proba(permuted)[:, 1])["roc_auc"]
        result.append(base - score if np.isfinite(base) and np.isfinite(score) else 0.0)
    return np.asarray(result)
