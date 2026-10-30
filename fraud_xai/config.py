"""Configuration and startup validation. Secrets are read from the environment only."""
from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_TARGET = "Class"
VALID_ROLES = ("admin", "analyst")


class ConfigError(Exception):
    """Raised for invalid configuration. Messages never contain secret values."""


@dataclass(frozen=True)
class Settings:
    app_env: str
    encryption_key: bytes = field(repr=False)
    signing_key: bytes = field(repr=False)
    auth_users: dict = field(repr=False)
    auth_disabled: bool
    max_upload_mb: int
    max_rows: int
    log_level: str
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _decode_key(name: str, value: str | None) -> bytes:
    if not value:
        raise ConfigError(f"{name} is not set. Run `python scripts/init_env.py` to create a local .env file.")
    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii"))
    except (binascii.Error, UnicodeEncodeError, ValueError):
        raise ConfigError(f"{name} is not valid urlsafe base64.") from None
    if len(raw) != 32:
        raise ConfigError(f"{name} must decode to exactly 32 bytes.")
    return raw


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    try:
        val = int(env.get(name, default))
    except ValueError:
        raise ConfigError(f"{name} must be an integer.") from None
    if not lo <= val <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}.")
    return val


def _parse_users(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        users = json.loads(raw)
    except json.JSONDecodeError:
        raise ConfigError("AUTH_USERS_JSON is not valid JSON.") from None
    if not isinstance(users, dict):
        raise ConfigError("AUTH_USERS_JSON must be a JSON object.")
    for name, spec in users.items():
        if not isinstance(spec, dict) or spec.get("role") not in VALID_ROLES or not str(spec.get("hash", "")).startswith("scrypt$"):
            raise ConfigError(f"AUTH_USERS_JSON entry for '{name}' needs a scrypt 'hash' and role in {VALID_ROLES}.")
    return users


def load_settings(env: Mapping[str, str] | None = None, *, use_dotenv: bool = True) -> Settings:
    """Load and validate settings. Pass `env` explicitly in tests."""
    if env is None:
        if use_dotenv:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
        env = os.environ
    app_env = env.get("APP_ENV", "development").lower()
    if app_env not in ("development", "production"):
        raise ConfigError("APP_ENV must be 'development' or 'production'.")
    users = _parse_users(env.get("AUTH_USERS_JSON"))
    auth_disabled = env.get("AUTH_DISABLED", "false").lower() == "true"
    if auth_disabled and app_env == "production":
        raise ConfigError("AUTH_DISABLED=true is not allowed when APP_ENV=production.")
    if not auth_disabled and not users:
        raise ConfigError("No users configured. Set AUTH_USERS_JSON (see scripts/create_user.py) or, for local development only, AUTH_DISABLED=true.")
    enc, sig = _decode_key("ENCRYPTION_KEY", env.get("ENCRYPTION_KEY")), _decode_key("MODEL_SIGNING_KEY", env.get("MODEL_SIGNING_KEY"))
    if enc == sig:
        raise ConfigError("ENCRYPTION_KEY and MODEL_SIGNING_KEY must be different keys.")
    level = env.get("LOG_LEVEL", "INFO").upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ConfigError("LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR.")
    return Settings(
        app_env=app_env, encryption_key=enc, signing_key=sig, auth_users=users, auth_disabled=auth_disabled,
        max_upload_mb=_int(env, "MAX_UPLOAD_MB", 50, 1, 500), max_rows=_int(env, "MAX_ROWS", 400_000, 100, 5_000_000),
        log_level=level,
    )
