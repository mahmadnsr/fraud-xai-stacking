"""End-to-end smoke tests: CLI train -> verify -> predict -> explain, and the Streamlit app boot."""
from pathlib import Path

import pytest

from fraud_xai import cli
from fraud_xai.config import ConfigError, load_settings
from fraud_xai.logging_utils import redact
from fraud_xai.persist import load_latest
from fraud_xai.predict import predict_frame
from fraud_xai.explain import explain_ensemble
from tests.conftest import make_env

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    env = make_env(tmp_path)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import fraud_xai.config as c
    monkeypatch.setattr(c, "MODELS_DIR", tmp_path / "models"); monkeypatch.setattr(c, "LOGS_DIR", tmp_path / "logs")
    from dataclasses import replace
    real = c.load_settings
    monkeypatch.setattr(cli, "load_settings", lambda: replace(real(use_dotenv=False), models_dir=tmp_path / "models", logs_dir=tmp_path / "logs"))
    assert cli.main(["check-config"]) == 0
    assert cli.main(["train", "--source", "sample", "--sample-rows", "2500", "--n-estimators", "15", "--cv-folds", "3"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "stacking_ensemble" in out
    version = cli.list_versions(tmp_path / "models")[0]
    assert cli.main(["verify", version]) == 0
    s = cli.load_settings()
    m = load_latest(s)
    bg, ex = m.explain_sample()
    pred = predict_frame(m, "stacking_ensemble", ex.head(5), 0.5)
    assert len(pred) == 5
    assert explain_ensemble(m.models, ex.head(2), bg.head(10), 60).values.shape[0] == 2


def test_cli_reports_errors_cleanly(tmp_path, monkeypatch, capsys):
    for k in ("ENCRYPTION_KEY", "MODEL_SIGNING_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings({"AUTH_DISABLED": "true"}, use_dotenv=False))
    assert cli.main(["check-config"]) == 1
    assert "ENCRYPTION_KEY" in capsys.readouterr().err


def test_log_redaction():
    t = redact("card 4111111111111111 mail a.b@x.com password=hunter2 KEY: abc")
    assert "4111" not in t and "a.b@x.com" not in t and "hunter2" not in t and "abc" not in t


def test_streamlit_app_boots_and_requires_model(tmp_path, monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    for k, v in make_env(tmp_path).items():
        monkeypatch.setenv(k, v)
    import fraud_xai.config as c
    monkeypatch.setattr(c, "MODELS_DIR", tmp_path / "m"); monkeypatch.setattr(c, "LOGS_DIR", tmp_path / "l")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Financial Fraud Detection" in t.value for t in at.title)


def test_streamlit_config_error_page(monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    for k in ("ENCRYPTION_KEY", "MODEL_SIGNING_KEY", "AUTH_USERS_JSON"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AUTH_DISABLED", "true")
    monkeypatch.setattr("fraud_xai.config.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("Configuration error" in e.value for e in at.error)
