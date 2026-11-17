#!/usr/bin/env python3
"""Fail closed when required PondLite production configuration is absent."""

import os
import sys
from pathlib import Path


REQUIRED_AUTH_VARS = (
    "PONDLITE_USERNAME",
    "PONDLITE_PASSWORD",
    "PONDLITE_SECRET_KEY",
)


def main() -> int:
    errors = []

    if os.environ.get("PONDLITE_ENV") != "production":
        errors.append("PONDLITE_ENV must be exactly 'production'")

    for name in REQUIRED_AUTH_VARS:
        if not os.environ.get(name, "").strip():
            errors.append(f"{name} must be set to a non-empty value")

    root = Path(__file__).resolve().parent.parent
    for relative_path in ("pondlite/web.py", "pondlite/serve.py", "requirements.txt"):
        if not (root / relative_path).is_file():
            errors.append(f"required application file is missing: {relative_path}")

    if errors:
        print("Production configuration check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Production configuration check passed; required auth values are set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
