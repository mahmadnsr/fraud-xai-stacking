# Security Documentation and Threat Model

## Assets
Transaction data (in memory, encrypted explain sample); model artifacts and manifests; `.env` secrets (encryption key, signing key, password hashes); dashboard access; logs.

## Trust boundaries and assumptions
Browser <-> Streamlit server (network boundary) · uploaded file <-> parser (untrusted input) · `models/` directory <-> loader (writable by anyone with disk access, so untrusted until verified) · `.env`/environment <-> process (trusted; protected by OS permissions).
Assumptions: the host OS is not compromised; `.env` permissions are respected; a reverse proxy provides TLS in any networked deployment; users with the `admin` role are trusted.

## Controls actually implemented
| Threat | Control | Where / test |
|---|---|---|
| Malicious or malformed CSV, oversized files, resource exhaustion | size cap, row cap, NUL/UTF-8 checks, strict column-name regex (blocks formula-injection names), duplicate-header rejection, numeric-only/finite checks | `schema.py`, `tests/test_schema.py` |
| Path traversal / symlink escape | local files only by plain name in `data/`, resolved path must stay in `data/` | `resolve_local_csv` + tests |
| Model artifact tampering / malicious pickle | HMAC-SHA256 signed manifest + per-file SHA-256 verified **before** `joblib.load`; no user model uploads; version-id regex | `persist.py`, `tests/test_persist.py` (incl. malicious pickle not executed) |
| Data-at-rest exposure | AES-256-GCM (maintained `cryptography` lib), random 96-bit nonce per message, version bound as AAD, tamper/wrong-key rejection | `security.py`, tests |
| Key leakage | keys only from environment; `repr=False`; never logged; `.env` created mode 600 and git-ignored; log redaction | `config.py`, `logging_utils.py` |
| Unauthorised access | scrypt password hashes, roles (`admin`, `analyst`), per-user failed-login lockout, 30-min idle timeout, auth cannot be disabled in production | `auth.py`, `tests/test_security.py`, `test_config.py` |
| Information disclosure | generic errors with a reference id; detail only in redacted log file; `showErrorDetails=none`; no raw records in manifests | `logging_utils.safe_error` |
| Cross-site issues | Streamlit XSRF protection on; CORS not widened; text rendered via Streamlit (no `unsafe_allow_html`) | `.streamlit/config.toml` |
| Dependency compromise | exact pins; `pip-audit` (clean at time of writing; `cryptography` was upgraded after it flagged advisories) | `requirements.txt` |

## Encryption in transit (NOT provided by the app)
Streamlit serves plain HTTP on `127.0.0.1`. For any network deployment put it behind a TLS-terminating reverse proxy (nginx/Caddy/cloud load balancer) with a valid certificate, HSTS, WebSocket upgrade enabled, request-size and rate limits, and keep Streamlit bound to localhost. Never disable certificate verification. Localhost development is **not** equivalent to secure deployment.

## Transport vs storage vs end-to-end encryption
* Transport (TLS): protects data between browser and proxy/server; the server sees plaintext.
* Storage (AES-GCM, implemented for the explain sample): protects files at rest; the app holds the key to use them.
* End-to-end: only sender and intended recipient can read data. **This system is not E2EE.** Server-side fraud scoring requires plaintext features, so browser-to-server E2EE would not protect data from the very server that must decrypt it. Privacy-preserving inference (homomorphic encryption, secure enclaves, on-device scoring) is future work and not implemented. Uploaded CSVs are held in memory only and not written to disk.

## Key management
Generate: `scripts/init_env.py` (`os.urandom(32)`). Store: `.env`/secret manager, separate from `models/`. Rotate: `security.reencrypt` re-encrypts payloads under a new key; rotating `MODEL_SIGNING_KEY` requires retraining or re-signing manifests. Recovery: there is none without the key - back it up securely; losing it makes the encrypted sample unreadable (retrain to regenerate). Revoke: replace the key, re-encrypt/retrain, delete old artifacts. Production should use a managed KMS. Random 96-bit nonces are safe up to ~2^32 messages per key; this app writes one per training run.

## Residual risks (honest list)
* An attacker who obtains both `.env` and `models/` can forge artifacts and read the sample.
* Login rate limiting is in-process and per-username; add proxy-level IP rate limiting for real deployments. Sessions are Streamlit-managed (no custom cookie handling); there is no MFA.
* Streamlit file-size limit is enforced after upload begins; the proxy should cap request bodies.
* Model abuse (querying to infer decision boundaries / adversarial evasion) is not mitigated beyond authentication and batch-size limits.
* `joblib` remains a pickle format; integrity checks protect against replacement but not against a compromised signing key.
* Denial of service by authorised users (large training jobs) is limited only by row/size caps.
* Synthetic demo data is not sensitive, but real data would still be held in process memory during training.
No claim of immunity to attack is made.
