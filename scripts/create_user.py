"""Print an AUTH_USERS_JSON entry for a new user (password is prompted, never echoed).

Usage: python scripts/create_user.py NAME {admin|analyst}   then merge the printed JSON into AUTH_USERS_JSON in .env
"""
import getpass, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fraud_xai.security import hash_password

if len(sys.argv) != 3 or sys.argv[2] not in ("admin", "analyst"):
    sys.exit(__doc__)
pw = getpass.getpass("Password (min 10 chars): ")
print(json.dumps({sys.argv[1]: {"hash": hash_password(pw), "role": sys.argv[2]}}))
