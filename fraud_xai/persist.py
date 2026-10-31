"""Model persistence with a signed manifest. Artifacts are verified BEFORE joblib deserialisation.

joblib/pickle can execute arbitrary code when loading, so we only load files whose SHA-256 digests are listed in a
manifest carrying a valid HMAC signature made with MODEL_SIGNING_KEY (kept outside the models directory).
"""
from __future__ import annotations

import io
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Settings
from .security import CryptoError, decrypt, encrypt, sha256_file, sign_bytes, verify_signature

VERSION_RE = re.compile(r"^v\d{8}T\d{6}Z(_\d{1,3})?$")
ARTIFACTS = ["models.joblib", "evaluation.npz", "explain_sample.enc"]
MANIFEST_SCHEMA = 2


class ArtifactError(Exception):
    """Missing, unsafe or tampered model artifacts."""


@dataclass
class LoadedModel:
    version: str
    models: dict
    manifest: dict
    evaluation: dict
    _explain_path: Path
    _enc_key: bytes
    preprocessor: object

    @property
    def feature_names(self) -> list[str]:
        return self.manifest["feature_schema"]["features"]

    def explain_sample(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Decrypt (in memory only) the stored background + explanation rows."""
        try:
            raw = decrypt(self._explain_path.read_bytes(), self._enc_key, aad=self.version.encode())
        except CryptoError:
            raise ArtifactError("The encrypted explanation sample could not be authenticated (wrong key or tampering).") from None
        df = pd.read_csv(io.BytesIO(raw))
        part = df.pop("__part")
        return df[part == "background"].reset_index(drop=True), df[part == "explain"].reset_index(drop=True)


def dependency_versions() -> dict:
    """Return versions without importing optional or forbidden ML libraries."""
    from importlib.metadata import PackageNotFoundError, version

    def installed(distribution: str) -> str:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return "not-installed"

    return {
        "python": sys.version.split()[0],
        "numpy": installed("numpy"),
        "pandas": installed("pandas"),
        "joblib": installed("joblib"),
        "streamlit": installed("streamlit"),
        "cryptography": installed("cryptography"),
    }


def _canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


def save_artifacts(settings: Settings, models: dict, evaluation: dict, explain_frame: pd.DataFrame, manifest_body: dict, *, preprocessor) -> str:
    """Write a new immutable version directory atomically and return its version id."""
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    version, n = stamp, 0
    while (settings.models_dir / version).exists():
        n += 1
        version = f"{stamp}_{n}"
    tmp = settings.models_dir / f".tmp_{version}"
    tmp.mkdir(parents=True)
    try:
        joblib.dump({"models": models, "preprocessor": preprocessor}, tmp / "models.joblib", compress=3)
        np.savez_compressed(tmp / "evaluation.npz", **evaluation)
        (tmp / "explain_sample.enc").write_bytes(encrypt(explain_frame.to_csv(index=False).encode(), settings.encryption_key,
                                                         aad=version.encode()))
        manifest = {**manifest_body, "manifest_schema": MANIFEST_SCHEMA, "version": version,
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "dependencies": dependency_versions(), "artifacts": {f: sha256_file(tmp / f) for f in ARTIFACTS}}
        body = _canonical(manifest)
        (tmp / "manifest.json").write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode())
        (tmp / "manifest.sig").write_text(sign_bytes(body, settings.signing_key))
        tmp.rename(settings.models_dir / version)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return version


def list_versions(models_dir: Path) -> list[str]:
    if not models_dir.is_dir():
        return []
    return sorted((p.name for p in models_dir.iterdir() if p.is_dir() and VERSION_RE.match(p.name)), reverse=True)


def verify_version(settings: Settings, version: str) -> dict:
    """Verify signature and file digests; return the manifest. Raises ArtifactError on any problem."""
    if not VERSION_RE.match(version or ""):
        raise ArtifactError("Invalid model version identifier.")
    d = settings.models_dir / version
    if not d.is_dir():
        raise ArtifactError("Model version not found.")
    try:
        manifest = json.loads((d / "manifest.json").read_bytes())
        sig = (d / "manifest.sig").read_text()
    except (OSError, json.JSONDecodeError):
        raise ArtifactError("Manifest or signature is missing or unreadable.") from None
    if not verify_signature(_canonical(manifest), sig, settings.signing_key):
        raise ArtifactError("Manifest signature is invalid: the artifact set cannot be trusted.")
    if manifest.get("manifest_schema") != MANIFEST_SCHEMA:
        raise ArtifactError("Unsupported manifest schema.")
    if manifest.get("version") != version or set(manifest.get("artifacts", {})) != set(ARTIFACTS):
        raise ArtifactError("Manifest does not match the artifact set.")
    for fname, digest in manifest["artifacts"].items():
        p = d / fname
        if not p.is_file() or sha256_file(p) != digest:
            raise ArtifactError(f"Integrity check failed for {fname}.")
    return manifest


def load_version(settings: Settings, version: str) -> LoadedModel:
    manifest = verify_version(settings, version)
    d = settings.models_dir / version
    payload = joblib.load(d / "models.joblib")  # safe only because the file was verified above
    if not isinstance(payload, dict) or set(payload) != {"models", "preprocessor"}:
        raise ArtifactError("Invalid model artifact payload.")
    models = payload["models"]
    preprocessor = payload["preprocessor"]
    if preprocessor is None:
        raise ArtifactError("Saved preprocessing pipeline is missing.")
    with np.load(d / "evaluation.npz", allow_pickle=False) as z:
        evaluation = {k: z[k] for k in z.files}
    return LoadedModel(version, models, manifest, evaluation, d / "explain_sample.enc", settings.encryption_key, preprocessor)


def load_latest(settings: Settings) -> LoadedModel:
    versions = list_versions(settings.models_dir)
    if not versions:
        raise ArtifactError("No trained model found. Train a model first.")
    return load_version(settings, versions[0])
