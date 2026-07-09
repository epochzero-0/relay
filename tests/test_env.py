"""Tests for the stdlib .env parser (relay/env.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from relay.env import load_env


def _write_env(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_parses_key_value_pairs(tmp_path, monkeypatch):
    monkeypatch.delenv("PB_FOO", raising=False)
    monkeypatch.delenv("PB_BAR", raising=False)
    path = _write_env(tmp_path, "PB_FOO=hello\nPB_BAR=123\n")

    result = load_env(path)

    assert result == {"PB_FOO": "hello", "PB_BAR": "123"}
    assert os.environ["PB_FOO"] == "hello"
    assert os.environ["PB_BAR"] == "123"


def test_skips_comments_and_blank_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("PB_FOO", raising=False)
    path = _write_env(
        tmp_path,
        "# a comment\n"
        "\n"
        "   \n"
        "PB_FOO=hello\n"
        "# PB_IGNORED=nope\n",
    )

    result = load_env(path)

    assert result == {"PB_FOO": "hello"}
    assert "PB_IGNORED" not in os.environ


def test_strips_surrounding_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("PB_DOUBLE", raising=False)
    monkeypatch.delenv("PB_SINGLE", raising=False)
    path = _write_env(
        tmp_path,
        'PB_DOUBLE="hello world"\n'
        "PB_SINGLE='hello single'\n",
    )

    result = load_env(path)

    assert result == {"PB_DOUBLE": "hello world", "PB_SINGLE": "hello single"}
    assert os.environ["PB_DOUBLE"] == "hello world"
    assert os.environ["PB_SINGLE"] == "hello single"


def test_handles_export_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("PB_EXPORTED", raising=False)
    path = _write_env(tmp_path, "export PB_EXPORTED=value\n")

    result = load_env(path)

    assert result == {"PB_EXPORTED": "value"}
    assert os.environ["PB_EXPORTED"] == "value"


def test_does_not_override_existing_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("PB_PRESET", "already-set")
    path = _write_env(tmp_path, "PB_PRESET=from-file\n")

    result = load_env(path)

    # the parsed dict still reports what was in the file...
    assert result["PB_PRESET"] == "from-file"
    # ...but os.environ keeps the value that was already there.
    assert os.environ["PB_PRESET"] == "already-set"


def test_missing_file_is_a_noop(tmp_path):
    missing = tmp_path / "does_not_exist.env"
    assert not missing.exists()

    result = load_env(missing)

    assert result == {}
