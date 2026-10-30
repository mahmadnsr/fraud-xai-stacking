"""Dataset loading. Synthetic demo data is always labelled as synthetic and never presented as real."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from .config import DEFAULT_TARGET
from .schema import DatasetReport, parse_csv_bytes, resolve_local_csv, validate_training_frame

SOURCE_SYNTHETIC = "synthetic_demo"
SOURCE_LOCAL = "local_csv"
SOURCE_UPLOAD = "uploaded_csv"


@dataclass
class DatasetBundle:
    frame: pd.DataFrame          # validated features + target (kept in memory only)
    target: str
    source: str
    label: str
    sha256: str
    report: DatasetReport

    @property
    def is_synthetic(self) -> bool:
        return self.source == SOURCE_SYNTHETIC

    def provenance(self) -> dict:
        """Metadata only - never raw records."""
        return {"source": self.source, "label": self.label, "sha256": self.sha256, "n_rows": self.report.n_rows,
                "n_features": self.report.n_features, "class_counts": self.report.class_counts,
                "fraud_rate": self.report.fraud_rate, "excluded_columns": self.report.excluded_columns,
                "is_synthetic": self.is_synthetic}


def make_synthetic(n_rows: int = 10_000, fraud_rate: float = 0.03, seed: int = 42) -> pd.DataFrame:
    """Synthetic imbalanced data with the Credit-Card-Fraud column layout (Time, V1..V28, Amount, Class).

    This is NOT real transaction data and results on it say nothing about real-world performance.
    """
    X, y = make_classification(n_samples=n_rows, n_features=28, n_informative=10, n_redundant=4, n_clusters_per_class=2,
                               weights=[1 - fraud_rate, fraud_rate], class_sep=1.2, flip_y=0.005, random_state=seed)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n_rows)).round(0))
    df["Amount"] = np.round(rng.lognormal(3.5, 1.2, n_rows) * (1 + 1.5 * y * rng.random(n_rows)), 2)
    df[DEFAULT_TARGET] = y
    return df


def _bundle(df: pd.DataFrame, raw_hash: str, source: str, label: str, target: str, min_per_class: int) -> DatasetBundle:
    clean, report = validate_training_frame(df, target, min_per_class)
    return DatasetBundle(clean, target, source, label, raw_hash, report)


def load_synthetic(n_rows: int = 10_000, fraud_rate: float = 0.03, seed: int = 42) -> DatasetBundle:
    df = make_synthetic(n_rows, fraud_rate, seed)
    digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    return _bundle(df, digest, SOURCE_SYNTHETIC, f"SYNTHETIC demo data (seed={seed})", DEFAULT_TARGET, 10)


def load_local_csv(name: str, data_dir: Path, max_bytes: int, max_rows: int, target: str = DEFAULT_TARGET) -> DatasetBundle:
    path = resolve_local_csv(name, data_dir)
    raw = path.read_bytes() if path.stat().st_size <= max_bytes else b"x" * (max_bytes + 1)  # size check in parser
    df = parse_csv_bytes(raw, max_bytes, max_rows)
    return _bundle(df, hashlib.sha256(raw).hexdigest(), SOURCE_LOCAL, f"Local CSV: {path.name}", target, 10)


def load_uploaded_csv(raw: bytes, max_bytes: int, max_rows: int, target: str = DEFAULT_TARGET) -> DatasetBundle:
    df = parse_csv_bytes(raw, max_bytes, max_rows)
    return _bundle(df, hashlib.sha256(raw).hexdigest(), SOURCE_UPLOAD, "Uploaded CSV (held in memory only)", target, 10)


def list_local_csvs(data_dir: Path) -> list[str]:
    from .schema import LOCAL_FILE_RE
    return sorted(p.name for p in data_dir.glob("*.csv") if p.is_file() and LOCAL_FILE_RE.match(p.name))
