"""Create a local .env with fresh random keys (never printed). Optionally create the first admin user.

Usage:  python scripts/init_env.py [--admin-user NAME] [--password-env VAR]   (password prompted if VAR not given)
"""
import argparse, getpass, json, os, stat, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fraud_xai.security import generate_key, hash_password

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--admin-user", default="admin")
ap.add_argument("--password-env", help="name of an environment variable holding the admin password")
ap.add_argument("--dev-no-auth", action="store_true", help="local development only: disable login")
a = ap.parse_args()
target = ROOT / ".env"
if target.exists():
    sys.exit(".env already exists; refusing to overwrite. Delete it first if you really want new keys (old encrypted data becomes unreadable).")
lines = ["APP_ENV=development", f"ENCRYPTION_KEY={generate_key()}", f"MODEL_SIGNING_KEY={generate_key()}",
         "MAX_UPLOAD_MB=50", "MAX_ROWS=400000", "LOG_LEVEL=INFO"]
if a.dev_no_auth:
    lines.append("AUTH_DISABLED=true")
else:
    pw = os.environ.get(a.password_env or "") or getpass.getpass("Admin password (min 10 chars): ")
    lines.append("AUTH_USERS_JSON='" + json.dumps({a.admin_user: {"hash": hash_password(pw), "role": "admin"}}) + "'")
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
with os.fdopen(fd, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"Created {target} (permissions 600). Keys were not printed. Back it up securely; it is git-ignored.")
