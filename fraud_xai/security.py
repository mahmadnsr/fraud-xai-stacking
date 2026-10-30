"""Cryptographic helpers built on maintained primitives only (cryptography, hashlib, hmac)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"FXE1"          # format/version marker, authenticated as part of the AAD
NONCE_LEN = 12           # 96-bit random nonce, the size recommended for AES-GCM
KEY_LEN = 32             # AES-256


class CryptoError(Exception):
    """Generic cryptographic failure (invalid key, malformed or tampered payload)."""


def generate_key() -> str:
    """Return a fresh random 256-bit key as urlsafe base64 text."""
    return base64.urlsafe_b64encode(os.urandom(KEY_LEN)).decode("ascii")


def _check_key(key: bytes) -> None:
    if not isinstance(key, (bytes, bytearray)) or len(key) != KEY_LEN:
        raise CryptoError("Invalid encryption key: expected 32 bytes.")


def encrypt(plaintext: bytes, key: bytes, aad: bytes = b"") -> bytes:
    """AES-256-GCM. Output = MAGIC | nonce | ciphertext+tag. A fresh random nonce is used per call."""
    _check_key(key)
    nonce = os.urandom(NONCE_LEN)
    return MAGIC + nonce + AESGCM(bytes(key)).encrypt(nonce, plaintext, MAGIC + aad)


def decrypt(payload: bytes, key: bytes, aad: bytes = b"") -> bytes:
    """Authenticate and decrypt. Any failure raises CryptoError without revealing why."""
    _check_key(key)
    if not isinstance(payload, (bytes, bytearray)) or len(payload) < len(MAGIC) + NONCE_LEN + 16 or payload[:4] != MAGIC:
        raise CryptoError("Invalid encrypted payload.")
    nonce = payload[4:4 + NONCE_LEN]
    try:
        return AESGCM(bytes(key)).decrypt(nonce, bytes(payload[4 + NONCE_LEN:]), MAGIC + aad)
    except InvalidTag:
        raise CryptoError("Decryption failed: wrong key or tampered data.") from None


def reencrypt(payload: bytes, old_key: bytes, new_key: bytes, aad: bytes = b"") -> bytes:
    """Key rotation helper: decrypt with the old key and encrypt with the new key."""
    return encrypt(decrypt(payload, old_key, aad), new_key, aad)


# ---- passwords (scrypt from the standard library) ----
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}


def hash_password(password: str) -> str:
    if not password or len(password) < 10:
        raise ValueError("Password must be at least 10 characters.")
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return "scrypt${n}${r}${p}${}${}".format(base64.b64encode(salt).decode(), base64.b64encode(dk).decode(), **_SCRYPT)


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt_b64, dk_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(dk_b64)
        dk = hashlib.scrypt(password.encode(), salt=salt, dklen=len(expected), n=int(n), r=int(r), p=int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, expected)


# ---- integrity ----
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sign_bytes(data: bytes, key: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def verify_signature(data: bytes, signature: str, key: bytes) -> bool:
    return hmac.compare_digest(sign_bytes(data, key), signature.strip())
