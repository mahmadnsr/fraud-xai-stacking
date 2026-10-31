"""Dataset loading. Synthetic demo data is always labelled as synthetic and never presented as real."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Generate deterministic, clearly labelled synthetic binary fraud data.

    This is artificial data for pipeline testing, not a representation of
    real credit-card transactions or real-world fraud behaviour.
    """
    if not isinstance(n_rows, int) or n_rows < 20:
        raise ValueError("n_rows must be an integer >= 20.")
    if not 0.0 < fraud_rate < 1.0:
        raise ValueError("fraud_rate must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)

    n_fraud = int(round(n_rows * fraud_rate))
    n_fraud = max(10, n_fraud)
    n_fraud = min(n_rows - 10, n_fraud)

    y = np.zeros(n_rows, dtype=np.int64)
    y[:n_fraud] = 1
    rng.shuffle(y)

    # Ten informative latent features with class-dependent distributions.
    latent = rng.normal(0.0, 1.0, size=(n_rows, 10))
    latent[:, 0] += y * 1.8
    latent[:, 1] -= y * 1.4
    latent[:, 2] += y * 1.1
    latent[:, 3] += y * 0.8

    # Fourteen additional noisy/redundant features.
    noise = rng.normal(0.0, 1.0, size=(n_rows, 14))
    redundant = np.column_stack([
        latent[:, 0] * 0.7 + rng.normal(0, 0.35, n_rows),
        latent[:, 1] * 0.6 + rng.normal(0, 0.4, n_rows),
        latent[:, 2] * 0.5 + rng.normal(0, 0.5, n_rows),
        latent[:, 3] * 0.5 + rng.normal(0, 0.5, n_rows),
    ])

    X = np.column_stack([latent, noise, redundant])
    X = X[:, :28]

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n_rows)).round(0))
    df["Amount"] = np.round(
        rng.lognormal(3.5, 1.2, n_rows) * (1.0 + 1.5 * y * rng.random(n_rows)),
        2,
    )
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
