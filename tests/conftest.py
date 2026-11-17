import json

import pytest

from pondlite import store


@pytest.fixture(autouse=True)
def clear_production_environment(monkeypatch):
    """Tests must not inherit deployment credentials or startup behavior."""
    for key in (
        "PONDLITE_ENV", "PONDLITE_USERNAME", "PONDLITE_PASSWORD",
        "PONDLITE_SECRET_KEY", "PONDLITE_RUN_DIR", "PONDLITE_SECURE_COOKIES",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    """Give each test an empty registry and isolated project root."""
    root = tmp_path / "project"
    (root / "raw").mkdir(parents=True)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "registry.db")
    return root


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
