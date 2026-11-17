"""Build identity for verifying that a running service matches local source."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "pondlite").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
