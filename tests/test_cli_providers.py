"""Tests for real-provider wiring in the CLI (registry + dotenv + `run`).

Driven entirely through `relay.cli.main` with patched sys.argv -- never a
subprocess. These must pass regardless of which optional SDKs are actually
installed in the test environment (anthropic and google are genuinely
absent here; openai happens to be present), so the SDK-availability check is
monkeypatched directly where determinism is required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from relay.cli import main


def _run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["relay"] + argv)
    return main()


@pytest.fixture
def input_csv(tmp_path: Path) -> Path:
    path = tmp_path / "input.csv"
    path.write_text(
        "order_id,text\n"
        "rec_0001,hello world\n"
        "rec_0002,another row\n",
        encoding="utf-8",
    )
    return path


def _base_run_args(input_csv: Path, tmp_path: Path, provider: str) -> list[str]:
    return [
        "run",
        "--input", str(input_csv),
        "--provider", provider,
        "--output-dir", str(tmp_path / "outputs"),
        "--tracker", str(tmp_path / "outputs" / "tracker.json"),
    ]


def test_anthropic_sdk_absent_exits_2(monkeypatch, input_csv, tmp_path):
    # anthropic is genuinely not installed in this environment; assert that
    # directly so the test documents its own precondition rather than only
    # relying on the monkeypatch below.
    assert importlib.util.find_spec("anthropic") is None

    exit_code = _run_cli(
        monkeypatch, _base_run_args(input_csv, tmp_path, "anthropic")
    )
    assert exit_code == 2


def test_anthropic_sdk_absent_message_mentions_extra(monkeypatch, input_csv, tmp_path, capsys):
    exit_code = _run_cli(
        monkeypatch, _base_run_args(input_csv, tmp_path, "anthropic")
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "relay[anthropic]" in out


def test_sdk_absent_path_is_deterministic_via_find_spec_patch(monkeypatch, input_csv, tmp_path, capsys):
    # Force the "SDK missing" branch for openai regardless of whether the
    # real package happens to be installed, by patching find_spec itself.
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, *args, **kwargs):
        if name == "openai":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    exit_code = _run_cli(
        monkeypatch, _base_run_args(input_csv, tmp_path, "openai")
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "relay[openai]" in out


def test_openai_missing_api_key_exits_2_naming_env_var(monkeypatch, input_csv, tmp_path, capsys):
    # Make find_spec report the SDK as present (regardless of what's really
    # installed) so we reach the api-key check deterministically.
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, *args, **kwargs):
        if name == "openai":
            return object()  # any non-None sentinel counts as "found".
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = _run_cli(
        monkeypatch, _base_run_args(input_csv, tmp_path, "openai")
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "OPENAI_API_KEY" in out


def test_google_missing_api_key_mentions_both_env_vars(monkeypatch, input_csv, tmp_path, capsys):
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name, *args, **kwargs):
        if name == "google.genai":
            return object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    exit_code = _run_cli(
        monkeypatch, _base_run_args(input_csv, tmp_path, "google")
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "GOOGLE_API_KEY" in out
    assert "GEMINI_API_KEY" in out


def test_mock_provider_still_works_unaffected(monkeypatch, input_csv, tmp_path):
    # Sanity check: wiring real providers through the registry must not
    # disturb the existing mock path.
    exit_code = _run_cli(
        monkeypatch,
        _base_run_args(input_csv, tmp_path, "mock") + ["--poll-interval", "0"],
    )
    assert exit_code == 0


def test_unknown_provider_rejected_by_argparse(monkeypatch, input_csv, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, _base_run_args(input_csv, tmp_path, "not-a-provider"))
    assert excinfo.value.code == 2
