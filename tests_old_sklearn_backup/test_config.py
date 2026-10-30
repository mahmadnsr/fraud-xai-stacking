import pytest

from fraud_xai.config import ConfigError, load_settings
from fraud_xai.security import generate_key, hash_password
from tests.conftest import make_env


def test_valid_dev_config(tmp_path):
    s = load_settings(make_env(tmp_path), use_dotenv=False)
    assert len(s.encryption_key) == 32 and "ENCRYPTION" not in repr(s) and "key" not in repr(s).lower().replace("max", "")


def test_missing_and_invalid_keys(tmp_path):
    env = make_env(tmp_path)
    for name, bad in (("ENCRYPTION_KEY", None), ("ENCRYPTION_KEY", "not-base64!!"), ("MODEL_SIGNING_KEY", "AAAA")):
        e = dict(env)
        e.pop(name, None)
        if bad:
            e[name] = bad
        with pytest.raises(ConfigError) as ei:
            load_settings(e, use_dotenv=False)
        assert env["ENCRYPTION_KEY"] not in str(ei.value)


def test_same_key_rejected(tmp_path):
    env = make_env(tmp_path)
    env["MODEL_SIGNING_KEY"] = env["ENCRYPTION_KEY"]
    with pytest.raises(ConfigError):
        load_settings(env, use_dotenv=False)


def test_auth_rules(tmp_path):
    env = make_env(tmp_path)
    prod = dict(env, APP_ENV="production")
    with pytest.raises(ConfigError):
        load_settings(prod, use_dotenv=False)          # AUTH_DISABLED not allowed in production
    nouser = {k: v for k, v in env.items() if k != "AUTH_DISABLED"}
    with pytest.raises(ConfigError):
        load_settings(nouser, use_dotenv=False)        # no users and auth not explicitly disabled
    import json
    ok = dict(nouser, AUTH_USERS_JSON=json.dumps({"a": {"hash": hash_password("long-password!"), "role": "admin"}}), APP_ENV="production")
    assert load_settings(ok, use_dotenv=False).auth_users["a"]["role"] == "admin"
    with pytest.raises(ConfigError):
        load_settings(dict(nouser, AUTH_USERS_JSON='{"a":{"hash":"plain","role":"root"}}'), use_dotenv=False)


def test_numeric_limits(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(make_env(tmp_path, MAX_UPLOAD_MB="abc"), use_dotenv=False)
    with pytest.raises(ConfigError):
        load_settings(make_env(tmp_path, MAX_UPLOAD_MB="100000"), use_dotenv=False)
