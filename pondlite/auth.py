"""Shared environment-based authentication and production configuration."""

import hmac
import os


def auth_enabled() -> bool:
    return bool(os.environ.get("PONDLITE_USERNAME") or os.environ.get("PONDLITE_PASSWORD"))


def valid_credentials(username: str, password: str) -> bool:
    expected_user = os.environ.get("PONDLITE_USERNAME", "")
    expected_password = os.environ.get("PONDLITE_PASSWORD", "")
    user_ok = hmac.compare_digest(username or "", expected_user)
    password_ok = hmac.compare_digest(password or "", expected_password)
    return bool(expected_user and expected_password and user_ok and password_ok)


def require_production_config() -> None:
    if os.environ.get("PONDLITE_ENV", "development").lower() != "production":
        return
    required = ("PONDLITE_USERNAME", "PONDLITE_PASSWORD", "PONDLITE_SECRET_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing required production settings: " + ", ".join(missing))
