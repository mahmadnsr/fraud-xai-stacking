"""Authentication (scrypt password check), role-based authorization and login rate limiting."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from .security import hash_password, verify_password

PERMISSIONS = {
    "admin": {"view", "predict", "explain", "train", "status"},
    "analyst": {"view", "predict", "explain", "status"},
}
_DUMMY_HASH = hash_password("dummy-password-for-timing")


class LoginLimiter:
    """In-process limiter: `max_attempts` failures per username within `window` seconds locks further tries."""

    def __init__(self, max_attempts: int = 5, window: float = 300.0, clock=time.monotonic):
        self.max_attempts, self.window, self._clock = max_attempts, window, clock
        self._fails: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def _prune(self, key: str) -> deque:
        q, now = self._fails[key], self._clock()
        while q and now - q[0] > self.window:
            q.popleft()
        return q

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._prune(key)) >= self.max_attempts

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._prune(key).append(self._clock())

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


def authenticate(users: dict, username: str, password: str, limiter: LoginLimiter) -> str | None:
    """Return the user's role on success, otherwise None. Raises PermissionError when rate-limited."""
    key = (username or "").strip().lower()[:64]
    if limiter.blocked(key):
        raise PermissionError("Too many failed attempts. Try again later.")
    spec = users.get(username)
    # Always run one scrypt verification so unknown users and wrong passwords take similar time.
    ok = verify_password(password or "", spec["hash"] if spec else _DUMMY_HASH)
    if spec and ok:
        limiter.reset(key)
        return spec["role"]
    limiter.record_failure(key)
    return None


def can(role: str | None, action: str) -> bool:
    return action in PERMISSIONS.get(role or "", set())
