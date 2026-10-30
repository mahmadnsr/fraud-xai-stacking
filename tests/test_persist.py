import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from fraud_xai.persist import ArtifactError, list_versions, load_latest, load_version, verify_version
from fraud_xai.security import generate_key
import base64


def copy_version(trained, tmp_path):
    s, res, _ = trained
    dst = tmp_path / "models"
    shutil.copytree(s.models_dir / res.version, dst / res.version)
    return replace(s, models_dir=dst), res.version, dst / res.version


def test_integrity_ok_and_explain_sample_roundtrip(trained):
    s, res, m = trained
    verify_version(s, res.version)
    bg, ex = m.explain_sample()
    assert len(bg) > 0 and len(ex) > 0 and list(bg.columns) == m.feature_names


def test_encrypted_sample_is_not_plaintext_and_needs_right_key(trained):
    s, res, m = trained
    raw = (s.models_dir / res.version / "explain_sample.enc").read_bytes()
    assert b"Time,V1,V2" not in raw and b"Amount,__part" not in raw
    m2 = replace(m, _enc_key=base64.urlsafe_b64decode(generate_key()))
    with pytest.raises(ArtifactError):
        m2.explain_sample()


@pytest.mark.parametrize("victim", ["models.joblib", "evaluation.npz", "explain_sample.enc"])
def test_tampered_artifact_rejected_before_loading(trained, tmp_path, victim):
    s, v, d = copy_version(trained, tmp_path)
    with (d / victim).open("ab") as fh:
        fh.write(b"X")
    with pytest.raises(ArtifactError, match="Integrity"):
        load_version(s, v)


def test_replaced_model_with_malicious_pickle_not_loaded(trained, tmp_path):
    import pickle
    s, v, d = copy_version(trained, tmp_path)
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (Path.write_text, (marker, "x"))
    (d / "models.joblib").write_bytes(pickle.dumps(Evil()))
    with pytest.raises(ArtifactError):
        load_version(s, v)
    assert not marker.exists()


def test_tampered_manifest_and_signature_rejected(trained, tmp_path):
    s, v, d = copy_version(trained, tmp_path)
    mf = json.loads((d / "manifest.json").read_text())
    mf["metrics_test"]["stacking_ensemble"]["recall"] = 1.0
    (d / "manifest.json").write_text(json.dumps(mf))
    with pytest.raises(ArtifactError, match="signature"):
        verify_version(s, v)
    s2, v2, d2 = copy_version(trained, tmp_path / "b")
    (d2 / "manifest.sig").write_text("0" * 64)
    with pytest.raises(ArtifactError):
        verify_version(s2, v2)


def test_wrong_signing_key_rejected(trained):
    s, res, _ = trained
    other = replace(s, signing_key=base64.urlsafe_b64decode(generate_key()))
    with pytest.raises(ArtifactError):
        verify_version(other, res.version)


def test_invalid_versions_and_missing_artifacts(trained, tmp_path):
    s, _, _ = trained
    for bad in ("../x", "v1", "", "v20260101T000000Z/../..", None):
        with pytest.raises(ArtifactError):
            verify_version(s, bad)
    with pytest.raises(ArtifactError):
        verify_version(s, "v20200101T000000Z")
    empty = replace(s, models_dir=tmp_path / "nothing")
    assert list_versions(empty.models_dir) == []
    with pytest.raises(ArtifactError, match="No trained model"):
        load_latest(empty)
    s2, v, d = copy_version(trained, tmp_path)
    (d / "manifest.json").unlink()
    with pytest.raises(ArtifactError):
        verify_version(s2, v)
