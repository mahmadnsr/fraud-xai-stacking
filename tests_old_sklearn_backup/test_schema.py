import numpy as np
import pandas as pd
import pytest

from fraud_xai.data import load_local_csv, load_uploaded_csv, make_synthetic
from fraud_xai.schema import ValidationError, parse_csv_bytes, resolve_local_csv, validate_feature_frame, validate_training_frame


def csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


@pytest.fixture()
def df():
    return make_synthetic(600, 0.1, seed=1)


def test_valid_dataset(df):
    b = load_uploaded_csv(csv(df), 10**7, 10**5)
    assert b.report.n_features == 30 and b.report.class_counts[1] > 10 and not b.is_synthetic


def test_size_and_row_limits(df):
    with pytest.raises(ValidationError, match="size"):
        parse_csv_bytes(csv(df), 100, 10**5)
    with pytest.raises(ValidationError, match="rows"):
        parse_csv_bytes(csv(df), 10**7, 50)


@pytest.mark.parametrize("raw", [b"", b"\x00\x01\x02binary", b"\xff\xfe\xfa\xfb", b"a,b\n1,2,3,4,5\n", b"onlyheader\n"])
def test_malformed_files(raw):
    with pytest.raises(ValidationError):
        parse_csv_bytes(raw, 10**6, 1000)


def test_bad_column_names(df):
    df = df.rename(columns={"V1": "=cmd|' /C calc'!A0"})
    with pytest.raises(ValidationError, match="Column names"):
        parse_csv_bytes(csv(df), 10**7, 10**5)
    with pytest.raises(ValidationError, match="Duplicate"):
        parse_csv_bytes(b"a,a,Class\n1,2,0\n", 10**6, 10)


def test_nonnumeric_inf_and_target(df):
    bad = df.copy().astype({"V1": object}); bad.loc[0, "V1"] = "abc"
    with pytest.raises(ValidationError, match="non-numeric"):
        validate_training_frame(bad, "Class")
    inf = df.copy(); inf.loc[0, "V2"] = np.inf
    with pytest.raises(ValidationError, match="finite"):
        validate_training_frame(inf, "Class")
    with pytest.raises(ValidationError, match="not found"):
        validate_training_frame(df.drop(columns="Class"), "Class")
    multi = df.copy(); multi.loc[0, "Class"] = 2
    with pytest.raises(ValidationError, match="0 .* and 1"):
        validate_training_frame(multi, "Class")
    one = df.copy(); one["Class"] = 0
    with pytest.raises(ValidationError, match="at least"):
        validate_training_frame(one, "Class")


def test_missing_values_reported_and_duplicates(df):
    d = df.copy(); d.loc[:4, "V3"] = np.nan; d = pd.concat([d, d.iloc[[10]]])
    clean, rep = validate_training_frame(d, "Class")
    assert rep.missing_cells == 5 and rep.missing_by_column == {"V3": 5} and rep.duplicate_rows >= 1


def test_sensitive_columns_excluded(df):
    d = df.copy(); d["card_number"] = 4111111111111111; d["Customer Name"] = 1.0
    clean, rep = validate_training_frame(d, "Class")
    assert set(rep.excluded_columns) == {"card_number", "Customer Name"} and "card_number" not in clean.columns


def test_local_csv_path_safety(tmp_path, df):
    data = tmp_path / "data"; data.mkdir()
    (data / "ok.csv").write_bytes(csv(df))
    (tmp_path / "secret.csv").write_text("x")
    for evil in ("../secret.csv", "/etc/passwd", "..\\x.csv", "ok.csv/../../secret.csv", "ok.txt", "sub/ok.csv", ""):
        with pytest.raises(ValidationError):
            resolve_local_csv(evil, data)
    (data / "link.csv").symlink_to(tmp_path / "secret.csv")
    with pytest.raises(ValidationError):
        resolve_local_csv("link.csv", data)
    assert load_local_csv("ok.csv", data, 10**7, 10**5).report.n_rows == 600


def test_feature_frame_validation():
    feats = ["a", "b"]
    ok = validate_feature_frame(pd.DataFrame({"b": [1, 2], "a": [3, 4]}), feats, 10)
    assert list(ok.columns) == feats
    for bad in (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1], "b": [2], "c": [3]}), pd.DataFrame({"a": ["x"], "b": [1]}),
                pd.DataFrame({"a": [np.inf], "b": [1]}), pd.DataFrame({"a": [], "b": []})):
        with pytest.raises(ValidationError):
            validate_feature_frame(bad, feats, 10)
    with pytest.raises(ValidationError):
        validate_feature_frame(pd.DataFrame({"a": range(20), "b": range(20)}), feats, 10)
    validate_feature_frame(pd.DataFrame({"a": [1], "b": [2], "Class": [0]}), feats, 10, target="Class")
