"""Tests for `relay demo`: the self-contained fault-tolerance narrative.

All runs go through Path(tmp_path) via an explicit --output-dir (never
deleted by the demo, regardless of --keep) so nothing touches the real
filesystem outside the pytest sandbox. Golden numbers (seed=7 chaos
scenario: passes=5, resent_items=92, coverage 100%) are pinned in
tests/test_convergence.py; the demo runs the identical scenario over its own
in-memory 300-record dataset, so the same numbers are asserted here via the
CLI's printed narrative.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from relay.cli import main


def _run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["relay"] + argv)
    return main()


def test_demo_exits_zero_and_prints_golden_numbers(monkeypatch, tmp_path, capsys):
    exit_code = _run_cli(
        monkeypatch, ["demo", "--seed", "7", "--output-dir", str(tmp_path)]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    # Act 1's golden resent_items count and full coverage, plus the closing
    # "passed" banner -- stable tokens pinned to the seed=7 chaos scenario.
    assert "resent_items=92" in out
    assert "coverage=100.0%" in out
    assert "DEMO PASSED" in out


def test_demo_explicit_output_dir_keeps_full_run_csv(monkeypatch, tmp_path, capsys):
    exit_code = _run_cli(
        monkeypatch,
        ["demo", "--seed", "7", "--output-dir", str(tmp_path), "--keep"],
    )

    assert exit_code == 0
    csv_path = tmp_path / "run1.csv"
    assert csv_path.exists()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 300
    assert {row["order_id"] for row in rows} == {
        f"rec_{i:04d}" for i in range(1, 301)
    }


def test_demo_return_value_is_a_clean_zero_exit_code(monkeypatch, tmp_path):
    exit_code = _run_cli(
        monkeypatch, ["demo", "--output-dir", str(tmp_path)]
    )

    # Matches the run/status subcommands' convention: func(args) returns an
    # int, and main() forwards it unchanged (no exception, no None).
    assert exit_code == 0
    assert isinstance(exit_code, int)
