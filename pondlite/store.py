"""PondLite: raw data -> trained data -> served model, locally."""

import json
import os
import re
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "pondlite.db"


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS datasets(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, version INTEGER, dir TEXT,
            n_records INTEGER, n_holdout INTEGER, created_at TEXT)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, version INTEGER, dataset_dir TEXT, run_dir TEXT,
            backend TEXT, epochs INTEGER, status TEXT,
            final_loss REAL, eval_loss REAL, perplexity REAL,
            created_at TEXT, updated_at TEXT)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS evals(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_dir TEXT, eval_loss REAL, perplexity REAL,
            samples_scored INTEGER, created_at TEXT)"""
    )
    return conn


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def run_dir(root: Path, name: str, version: int) -> Path:
    return root / "runs" / name / f"v{version}"


def dataset_dir(root: Path, name: str, version: int) -> Path:
    return root / "data" / name / f"v{version}"


def validate_name(name: str) -> str:
    """Return a safe dataset/run name or reject path-like input."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("name must be 1-64 characters: letters, numbers, underscores, or hyphens")
    return name


def _versions(base: Path):
    if not base.exists():
        return []
    return sorted(
        int(p.name[1:]) for p in base.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    )


def next_version(root: Path, name: str, kind: str) -> int:
    """kind: 'data' or 'runs'"""
    if kind not in {"data", "runs"}:
        raise ValueError("kind must be 'data' or 'runs'")
    name = validate_name(name)
    versions = _versions(root / kind / name)
    return versions[-1] + 1 if versions else 1


def latest_dataset_dir(root: Path, name: str) -> Path:
    """Resolve the latest existing dataset version."""
    name = validate_name(name)
    versions = _versions(root / "data" / name)
    if not versions:
        raise FileNotFoundError(f"no datasets found for {name!r}")
    return dataset_dir(root, name, versions[-1])


def resolve_run_dir(root: Path, value: str) -> Path:
    """Resolve a completed run, refusing paths outside the project's runs directory."""
    runs_root = (root / "runs").resolve()
    candidate = Path(value)
    candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        candidate.relative_to(runs_root)
    except ValueError:
        raise ValueError("run_dir must be inside the project's runs directory")
    if candidate == runs_root or not candidate.is_dir():
        raise ValueError("run_dir is not a run directory")
    if not (candidate / "metrics.json").is_file() or not (candidate / "adapter").is_dir():
        raise ValueError("run_dir must contain metrics.json and an adapter directory")
    return candidate


def dump_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
