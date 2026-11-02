import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_xai.evaluate import manual_roc_auc, manual_pr_auc

VERSIONS = {
    "weighted_stacking": "v20261102T093038Z",
    "unweighted_stacking": "v20261102T093426Z",
    "lightgbm": "v20261102T093038Z",
}

N_BOOTSTRAP = 2000
RANDOM_SEED = 2026
CONFIDENCE = 0.95

METRIC_NAMES = [
    "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"
]


def load_run(version):
    folder = Path("models") / version
    manifest = json.loads((folder / "manifest.json").read_text())
    data = np.load(folder / "evaluation.npz")
    return manifest, data


def get_model_key(name):
    return {
        "weighted_stacking": "stacking_ensemble",
        "unweighted_stacking": "stacking_ensemble",
        "lightgbm": "lightgbm",
    }[name]


def full_precision_metrics(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)

    tp = int(np.sum((y == 1) & (predictions == 1)))
    tn = int(np.sum((y == 0) & (predictions == 0)))
    fp = int(np.sum((y == 0) & (predictions == 1)))
    fn = int(np.sum((y == 1) & (predictions == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = (
        (tp * tn - fp * fn) / denominator
        if denominator else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(manual_roc_auc(y, scores)),
        "pr_auc": float(manual_pr_auc(y, scores)),
        "mcc": float(mcc),
    }


def percentile_interval(values):
    values = np.asarray(values, dtype=float)
    alpha = 1.0 - CONFIDENCE

    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def main():
    loaded = {
        name: load_run(version)
        for name, version in VERSIONS.items()
    }

    y = loaded["weighted_stacking"][1]["y_test"].astype(int)

    for name, (_, data) in loaded.items():
        if not np.array_equal(y, data["y_test"].astype(int)):
            raise ValueError(f"Test labels differ for {name}")

    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError("Both binary classes must be present.")

    class_indices = {
        cls: np.flatnonzero(y == cls)
        for cls in (0, 1)
    }

    model_data = {}

    for name, (manifest, data) in loaded.items():
        key = get_model_key(name)
        model_data[name] = {
            "scores": np.asarray(data[key + "__test"], dtype=float),
            "threshold": float(manifest["thresholds"][key]),
        }

    # Original held-out test-set point estimates.
    point_estimates = {
        name: full_precision_metrics(
            y, info["scores"], info["threshold"]
        )
        for name, info in model_data.items()
    }

    rng = np.random.default_rng(RANDOM_SEED)

    samples = {
        name: {metric: [] for metric in METRIC_NAMES}
        for name in model_data
    }

    comparisons = [
        ("weighted_stacking", "unweighted_stacking"),
        ("weighted_stacking", "lightgbm"),
    ]

    differences = {
        f"{a}_minus_{b}": {metric: [] for metric in METRIC_NAMES}
        for a, b in comparisons
    }

    for _ in range(N_BOOTSTRAP):
        # Preserve observed class counts; use identical sampled indices
        # for every model to maintain paired comparisons.
        sampled = np.concatenate([
            rng.choice(
                class_indices[0],
                size=len(class_indices[0]),
                replace=True,
            ),
            rng.choice(
                class_indices[1],
                size=len(class_indices[1]),
                replace=True,
            ),
        ])

        rng.shuffle(sampled)
        current = {}

        for name, info in model_data.items():
            result = full_precision_metrics(
                y[sampled],
                info["scores"][sampled],
                info["threshold"],
            )

            current[name] = result

            for metric in METRIC_NAMES:
                samples[name][metric].append(result[metric])

        for a, b in comparisons:
            label = f"{a}_minus_{b}"

            for metric in METRIC_NAMES:
                differences[label][metric].append(
                    current[a][metric] - current[b][metric]
                )

    # Model-level estimates and confidence intervals.
    rows = []

    for name, metric_values in samples.items():
        for metric, values in metric_values.items():
            low, high = percentile_interval(values)

            rows.append({
                "model": name,
                "metric": metric,
                "point_estimate": point_estimates[name][metric],
                "bootstrap_median": float(np.median(values)),
                "ci_95_low": low,
                "ci_95_high": high,
                "bootstrap_repetitions": N_BOOTSTRAP,
            })

    # Paired difference estimates and confidence intervals.
    diff_rows = []

    for comparison, metric_values in differences.items():
        a, b = comparison.split("_minus_", 1)

        for metric, values in metric_values.items():
            low, high = percentile_interval(values)
            point_difference = (
                point_estimates[a][metric] - point_estimates[b][metric]
            )

            diff_rows.append({
                "comparison": comparison,
                "metric": metric,
                "point_difference": point_difference,
                "bootstrap_median_difference": float(np.median(values)),
                "ci_95_low": low,
                "ci_95_high": high,
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "bootstrap_repetitions": N_BOOTSTRAP,
            })

    output_dir = Path("research_results")
    output_dir.mkdir(exist_ok=True)

    model_df = pd.DataFrame(rows)
    diff_df = pd.DataFrame(diff_rows)

    model_path = output_dir / "bootstrap_confidence_intervals.csv"
    diff_path = output_dir / "paired_bootstrap_differences.csv"

    model_df.to_csv(model_path, index=False)
    diff_df.to_csv(diff_path, index=False)

    print(f"Bootstrap repetitions: {N_BOOTSTRAP}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Confidence level: {CONFIDENCE:.0%}")
    print(f"Test observations: {len(y)}")
    print(f"Test fraud observations: {int(y.sum())}")
    print("Method: paired, class-stratified percentile bootstrap")
    print("Thresholds: fixed from validation; no test tuning")

    print("\nOriginal test-set point estimates and 95% CIs:")
    print(model_df.to_string(index=False))

    print("\nPaired differences and 95% CIs:")
    print(diff_df.to_string(index=False))

    print("\nSaved:")
    print(model_path)
    print(diff_path)


if __name__ == "__main__":
    main()
