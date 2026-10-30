import json
from pathlib import Path

import pytest

from fraud_xai.auth import LoginLimiter
from fraud_xai.config import load_settings
from fraud_xai.data import load_synthetic
from fraud_xai.models import TrainConfig
from fraud_xai.persist import load_version
from fraud_xai.security import generate_key, hash_password
from fraud_xai.train import train_all

FAST = TrainConfig(seed=7, cv_folds=3, n_estimators=20)


def make_env(tmp_path: Path, **extra) -> dict:
    env = {"APP_ENV": "development", "ENCRYPTION_KEY": generate_key(), "MODEL_SIGNING_KEY": generate_key(), "AUTH_DISABLED": "true"}
    env.update(extra)
    return env


def settings_for(tmp_path: Path, **extra):
    s = load_settings(make_env(tmp_path, **extra), use_dotenv=False)
    from dataclasses import replace
    return replace(s, models_dir=tmp_path / "models", data_dir=tmp_path / "data", logs_dir=tmp_path / "logs")


@pytest.fixture()
def settings(tmp_path):
    (tmp_path / "data").mkdir()
    return settings_for(tmp_path)


@pytest.fixture(scope="session")
def small_bundle():
    """SYNTHETIC test fixture - never used as a project result."""
    return load_synthetic(3000, 0.05, seed=3)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, small_bundle):
    tmp = tmp_path_factory.mktemp("trained")
    (tmp / "data").mkdir()
    s = settings_for(tmp)
    res = train_all(small_bundle, FAST, s)
    return s, res, load_version(s, res.version)
