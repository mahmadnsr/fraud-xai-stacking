"""CSV parsing and schema validation. All error messages are safe to show to end users."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_ .\-]{0,63}$")
LOCAL_FILE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,100}\.csv$")
# Columns that look like direct identifiers or credentials are never used as model features.
SENSITIVE_RE = re.compile(
    r"(card.?(number|no|num)|\bpan\b|cvv|cvc|password|passwd|\bssn\b|account.?(number|no|num)|\biban\b|"
    r"customer.?(name|id)|first.?name|last.?name|full.?name|e.?mail|phone|address)", re.I)


class ValidationError(Exception):
    """User-correctable input problem with a safe message."""


@dataclass
class DatasetReport:
    n_rows: int
    n_features: int
    feature_names: list[str]
    excluded_columns: list[str]
    missing_cells: int
    missing_by_column: dict[str, int]
    duplicate_rows: int
    class_counts: dict[int, int]
    fraud_rate: float
    warnings: list[str] = field(default_factory=list)


def parse_csv_bytes(raw: bytes, max_bytes: int, max_rows: int) -> pd.DataFrame:
    """Parse untrusted CSV bytes. Content is inspected; filenames and MIME types are never trusted."""
    if not raw:
        raise ValidationError("The file is empty.")
    if len(raw) > max_bytes:
        raise ValidationError(f"The file exceeds the {max_bytes // (1024 * 1024)} MB size limit.")
    if b"\x00" in raw[:65536]:
        raise ValidationError("The file does not look like a text CSV.")
    try:
        header = next(csv.reader(io.StringIO(raw[:65536].decode("utf-8-sig", errors="strict").splitlines()[0])))
    except (UnicodeDecodeError, StopIteration, IndexError, csv.Error):
        raise ValidationError("The file must be UTF-8 encoded text with a header row.") from None
    if len({h.strip() for h in header}) != len(header):
        raise ValidationError("Duplicate column names are not allowed.")   # pandas would silently rename them
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", nrows=max_rows + 1, on_bad_lines="error",
                         skipinitialspace=True, low_memory=False)
    except UnicodeDecodeError:
        raise ValidationError("The file must be UTF-8 encoded text.") from None
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        raise ValidationError("The file is not a well-formed CSV.") from None
    if not isinstance(df.index, pd.RangeIndex):  # rows with more fields than the header are silently turned into an index
        raise ValidationError("The file is not a well-formed CSV (rows have more fields than the header).")
    if len(df) > max_rows:
        raise ValidationError(f"The file has more than the allowed {max_rows:,} rows.")
    if df.empty or df.shape[1] < 2:
        raise ValidationError("The CSV must contain a header row, at least two columns and at least one data row.")
    df.columns = [str(c).strip() for c in df.columns]
    bad = [c for c in df.columns if not COLUMN_NAME_RE.match(c)]
    if bad:
        raise ValidationError("Column names may only contain letters, digits, spaces, '_', '.' and '-' (max 64 characters).")
    if df.columns.duplicated().any():
        raise ValidationError("Duplicate column names are not allowed.")
    return df


def resolve_local_csv(name: str, data_dir: Path) -> Path:
    """Resolve a CSV that must live directly inside data_dir (no path traversal, no symlink escape)."""
    if not LOCAL_FILE_RE.match(name or ""):
        raise ValidationError("Invalid file name. Use a plain *.csv file name located in the data/ folder.")
    base = data_dir.resolve()
    path = (base / name).resolve()
    if path.parent != base:
        raise ValidationError("The file must be located directly inside the data/ folder.")
    if not path.is_file():
        raise ValidationError("File not found in the data/ folder.")
    return path


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = {}
    for c in cols:
        orig = df[c]
        conv = pd.to_numeric(orig, errors="coerce")
        if (conv.isna() & orig.notna()).any():
            raise ValidationError(f"Column '{c}' contains non-numeric values. Only numeric feature columns are supported.")
        out[c] = conv.astype("float64")
    res = pd.DataFrame(out, index=df.index)
    if not np.isfinite(res.to_numpy()[~np.isnan(res.to_numpy())]).all():
        raise ValidationError("Feature values must be finite numbers (no infinity).")
    return res


def validate_training_frame(df: pd.DataFrame, target: str, min_per_class: int = 10) -> tuple[pd.DataFrame, DatasetReport]:
    """Return (clean_frame, report). Clean frame has float64 features plus an int target column."""
    if target not in df.columns:
        raise ValidationError(f"Target column '{target}' was not found.")
    y = pd.to_numeric(df[target], errors="coerce")
    if y.isna().any() or not set(y.unique()) <= {0, 1}:
        raise ValidationError(f"Target column '{target}' must contain only 0 (legitimate) and 1 (fraud).")
    counts = y.astype(int).value_counts().to_dict()
    counts = {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0))}
    if min(counts.values()) < min_per_class:
        raise ValidationError(f"Each class needs at least {min_per_class} records (found {counts[0]} legitimate / {counts[1]} fraud).")
    candidates = [c for c in df.columns if c != target]
    excluded = [c for c in candidates if SENSITIVE_RE.search(c)]
    features = [c for c in candidates if c not in excluded]
    if len(features) < 2:
        raise ValidationError("At least two non-sensitive numeric feature columns are required.")
    X = _to_numeric(df, features)
    if X.isna().all().any():
        raise ValidationError("A feature column is entirely empty.")
    clean = X.copy()
    clean[target] = y.astype(int).to_numpy()
    warnings = []
    if excluded:
        warnings.append(f"Excluded {len(excluded)} column(s) that look like personal identifiers or credentials.")
    const = [c for c in features if X[c].nunique(dropna=True) <= 1]
    if const:
        warnings.append(f"{len(const)} constant column(s) carry no information: {', '.join(const[:5])}")
    miss = X.isna().sum()
    report = DatasetReport(
        n_rows=len(clean), n_features=len(features), feature_names=features, excluded_columns=excluded,
        missing_cells=int(miss.sum()), missing_by_column={c: int(v) for c, v in miss.items() if v > 0},
        duplicate_rows=int(clean.duplicated().sum()), class_counts=counts,
        fraud_rate=counts[1] / len(clean), warnings=warnings)
    if report.duplicate_rows:
        warnings.append(
            f"{report.duplicate_rows} exact duplicate row(s) found. "
            "They are preserved; duplicate-aware validation is recommended "
            "because identical rows may occur across data splits."
        )
    return clean, report


def validate_feature_frame(df: pd.DataFrame, feature_names: list[str], max_rows: int, target: str | None = None) -> pd.DataFrame:
    """Validate inference input: exact expected feature columns, numeric, finite, bounded size."""
    if len(df) > max_rows:
        raise ValidationError(f"Too many rows (limit {max_rows:,}).")
    if len(df) == 0:
        raise ValidationError("No rows to score.")
    cols = list(df.columns)
    ignorable = {target} if target else set()
    missing = [c for c in feature_names if c not in cols]
    extra = [c for c in cols if c not in feature_names and c not in ignorable]
    if missing:
        raise ValidationError(f"Missing {len(missing)} required feature column(s), e.g. {', '.join(missing[:3])}.")
    if extra:
        raise ValidationError(f"Unexpected {len(extra)} column(s) that are not part of the model schema.")
    X = _to_numeric(df, feature_names)
    return X[feature_names]
