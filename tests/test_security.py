import pytest

from fraud_xai.auth import LoginLimiter, authenticate, can
from fraud_xai.security import (CryptoError, decrypt, encrypt, generate_key, hash_password, reencrypt, sign_bytes,
                                verify_password, verify_signature)
import base64

KEY = base64.urlsafe_b64decode(generate_key())


def test_roundtrip_and_unique_nonces():
    a, b = encrypt(b"hello", KEY), encrypt(b"hello", KEY)
    assert decrypt(a, KEY) == b"hello" == decrypt(b, KEY)
    assert a[4:16] != b[4:16] and a != b


def test_tampered_ciphertext_rejected():
    ct = bytearray(encrypt(b"secret", KEY))
    for pos in (0, 6, len(ct) - 1):
        bad = bytearray(ct)
        bad[pos] ^= 1
        with pytest.raises(CryptoError):
            decrypt(bytes(bad), KEY)


def test_wrong_key_and_bad_keys_and_payloads():
    ct = encrypt(b"x", KEY)
    with pytest.raises(CryptoError):
        decrypt(ct, base64.urlsafe_b64decode(generate_key()))
    for bad_key in (b"short", b"", None):
        with pytest.raises(CryptoError):
            encrypt(b"x", bad_key)
    for bad in (b"", b"garbage", b"FXE1" + b"\x00" * 5, None):
        with pytest.raises(CryptoError):
            decrypt(bad, KEY)


def test_aad_binding_and_rotation():
    ct = encrypt(b"data", KEY, aad=b"v1")
    with pytest.raises(CryptoError):
        decrypt(ct, KEY, aad=b"v2")
    new = base64.urlsafe_b64decode(generate_key())
    rotated = reencrypt(ct, KEY, new, aad=b"v1")
    assert decrypt(rotated, new, aad=b"v1") == b"data"
    with pytest.raises(CryptoError):
        decrypt(rotated, KEY, aad=b"v1")


def test_password_hashing():
    h = hash_password("correct horse battery")
    assert "correct" not in h and verify_password("correct horse battery", h)
    assert not verify_password("wrong password!!", h) and not verify_password("x", "garbage")
    with pytest.raises(ValueError):
        hash_password("short")


def test_hmac_signatures():
    sig = sign_bytes(b"manifest", KEY)
    assert verify_signature(b"manifest", sig, KEY) and not verify_signature(b"manifesT", sig, KEY)


def test_authentication_roles_and_rate_limit():
    users = {"alice": {"hash": hash_password("a-long-password"), "role": "analyst"}}
    lim = LoginLimiter(max_attempts=3, window=1000)
    assert authenticate(users, "alice", "a-long-password", lim) == "analyst"
    assert authenticate(users, "alice", "nope-nope-nope", lim) is None
    assert authenticate(users, "ghost", "whatever-long", lim) is None
    for _ in range(2):
        authenticate(users, "alice", "bad-bad-bad-bad", lim)
    with pytest.raises(PermissionError):
        authenticate(users, "alice", "a-long-password", lim)
    assert can("admin", "train") and not can("analyst", "train") and can("analyst", "predict") and not can(None, "view")
