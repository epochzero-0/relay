"""Tests for the offline cost estimation utility and `polybatch cost` CLI.

Everything here is fully offline and instant: no network, no API keys, only
tmp_path CSV files and in-memory math. ASCII-only per project convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from polybatch.cli import main
from polybatch.cost import (
    PRICES,
    CostEstimate,
    estimate_cost,
    estimate_cost_for_records,
    estimate_tokens,
    format_estimate,
)


def test_estimate_tokens_matches_len_over_4():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 0
    assert estimate_tokens("abcd") == 1
    text = "x" * 401
    assert estimate_tokens(text) == len(text) // 4


def test_estimate_cost_math_hand_computed():
    model = next(iter(PRICES))
    price = PRICES[model]
    prompts = ["a" * 40, "b" * 80]  # 10 tokens + 20 tokens = 30 input tokens
    max_tokens = 5

    est = estimate_cost(prompts, model, max_tokens)

    assert est.model == model
    assert est.n_items == 2
    assert est.input_tokens == 10 + 20
    assert est.output_tokens == 2 * max_tokens

    expected_input_cost = est.input_tokens * price.input_per_mtok / 1e6
    expected_output_cost = est.output_tokens * price.output_per_mtok / 1e6
    assert est.input_cost == pytest.approx(expected_input_cost)
    assert est.output_cost == pytest.approx(expected_output_cost)
    assert est.total_cost == pytest.approx(est.input_cost + est.output_cost)

    expected_per_1k = est.total_cost / est.n_items * 1000
    assert est.per_1k_items_cost == pytest.approx(expected_per_1k)


def test_estimate_cost_zero_items_no_division_error():
    model = next(iter(PRICES))
    est = estimate_cost([], model, 64)
    assert est.n_items == 0
    assert est.input_tokens == 0
    assert est.output_tokens == 0
    assert est.per_1k_items_cost == 0


def test_estimate_cost_unknown_model_raises_value_error():
    with pytest.raises(ValueError):
        estimate_cost(["hello"], "not-a-real-model", 64)


def test_estimate_cost_unknown_model_lists_available_models():
    with pytest.raises(ValueError) as excinfo:
        estimate_cost(["hello"], "not-a-real-model", 64)
    message = str(excinfo.value)
    for model_name in PRICES:
        assert model_name in message


def test_estimate_cost_for_records_uses_default_task():
    from polybatch.core.models import Record

    records = [
        Record(order_id="rec_0001", text="hello"),
        Record(order_id="rec_0002", text="world"),
    ]
    model = next(iter(PRICES))
    est = estimate_cost_for_records(records, model, max_tokens=16)
    assert isinstance(est, CostEstimate)
    assert est.n_items == 2
    assert est.output_tokens == 2 * 16
    assert est.input_tokens > 0


def test_format_estimate_is_ascii_and_has_model_and_total():
    model = next(iter(PRICES))
    est = estimate_cost(["some sample prompt text " * 3], model, 32)
    report = format_estimate(est)

    # ASCII-only: this must not raise UnicodeEncodeError.
    report.encode("ascii")

    assert model in report
    assert "TOTAL" in report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["polybatch"] + argv)
    return main()


@pytest.fixture
def input_csv(tmp_path: Path) -> Path:
    path = tmp_path / "input.csv"
    path.write_text(
        "order_id,text\n"
        "rec_0001,hello world\n"
        "rec_0002,another row of sample text\n"
        "rec_0003,a third row\n",
        encoding="utf-8",
    )
    return path


def test_cli_cost_success_returns_0(monkeypatch, input_csv, capsys):
    model = next(iter(PRICES))
    exit_code = _run_cli(
        monkeypatch,
        [
            "cost",
            "--input", str(input_csv),
            "--model", model,
            "--max-tokens", "16",
        ],
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert model in out
    assert "TOTAL" in out


def test_cli_cost_unknown_model_returns_2(monkeypatch, input_csv, capsys):
    exit_code = _run_cli(
        monkeypatch,
        [
            "cost",
            "--input", str(input_csv),
            "--model", "not-a-real-model",
        ],
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "not-a-real-model" in out


def test_cli_cost_missing_file_returns_2(monkeypatch, tmp_path, capsys):
    model = next(iter(PRICES))
    missing = tmp_path / "does_not_exist.csv"
    exit_code = _run_cli(
        monkeypatch,
        [
            "cost",
            "--input", str(missing),
            "--model", model,
        ],
    )
    out = capsys.readouterr().out
    assert exit_code == 2
    assert "not found" in out


def test_cli_cost_limit_items(monkeypatch, input_csv, capsys):
    model = next(iter(PRICES))
    exit_code = _run_cli(
        monkeypatch,
        [
            "cost",
            "--input", str(input_csv),
            "--model", model,
            "--limit-items", "1",
        ],
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "1" in out
