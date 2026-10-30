"""Logging with redaction and safe user-facing error messages."""
from __future__ import annotations

import logging
import re
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

_PATTERNS = [
    (re.compile(r"\b\d{12,19}\b"), "[REDACTED-NUMBER]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[REDACTED-EMAIL]"),
    (re.compile(r"(?i)(key|password|secret|token)\s*[=:]\s*\S+"), r"\1=[REDACTED]"),
]


def redact(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # also redacts formatted tracebacks
        return redact(super().format(record))


def setup_logging(level: str, logs_dir: Path) -> logging.Logger:
    logger = logging.getLogger("fraud_xai")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(logs_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def safe_error(logger: logging.Logger, exc: BaseException, context: str) -> str:
    """Log full (redacted) diagnostics to the file; return a generic message with a reference id."""
    ref = uuid.uuid4().hex[:8]
    logger.error("[%s] %s: %s", ref, context, type(exc).__name__, exc_info=exc)
    return f"{context} failed (reference {ref}). See the application log for details."
