"""Tests for polybatch.core.parsing: parse_result_line, extract_order_id,
ParseFailure, and parse_batch_results.
"""

from __future__ import annotations

from polybatch.core.models import BatchResult
from polybatch.core.parsing import (
    ParseFailure,
    extract_order_id,
    parse_batch_results,
    parse_result_line,
)


def test_parse_result_line_id_and_values():
    result = parse_result_line("rec_0001,7,3", n_fields=2)
    assert result == {"order_id": "rec_0001", "values": [7.0, 3.0]}


def test_parse_result_line_bare_values_with_order_id_arg():
    result = parse_result_line("7,3", n_fields=2, order_id="rec_0002")
    assert result == {"order_id": "rec_0002", "values": [7.0, 3.0]}


def test_parse_result_line_bare_values_without_order_id_arg_fails():
    # Bare-values shape only matches when a caller-supplied order_id is given.
    assert parse_result_line("7,3", n_fields=2) is None


def test_parse_result_line_skips_header_line():
    text = "order_id,score1,score2\nrec_0003,4,9"
    result = parse_result_line(text, n_fields=2)
    assert result == {"order_id": "rec_0003", "values": [4.0, 9.0]}


def test_parse_result_line_header_case_insensitive():
    text = "Order_Id,score1,score2\nrec_0004,1,2"
    result = parse_result_line(text, n_fields=2)
    assert result == {"order_id": "rec_0004", "values": [1.0, 2.0]}


def test_parse_result_line_skips_commentary_and_finds_match():
    text = "Sure! Here is the rating:\nrec_0005,8,6"
    result = parse_result_line(text, n_fields=2)
    assert result == {"order_id": "rec_0005", "values": [8.0, 6.0]}


def test_parse_result_line_garbage_returns_none():
    assert parse_result_line("not a valid line at all", n_fields=2) is None


def test_parse_result_line_wrong_field_count_returns_none():
    assert parse_result_line("rec_0001,7,3,9", n_fields=2) is None


def test_parse_result_line_non_numeric_values_returns_none():
    assert parse_result_line("rec_0001,seven,three", n_fields=2) is None


def test_parse_result_line_blank_lines_skipped():
    text = "\n\nrec_0006,5,5"
    result = parse_result_line(text, n_fields=2)
    assert result == {"order_id": "rec_0006", "values": [5.0, 5.0]}


def test_extract_order_id_happy_path():
    assert extract_order_id("run_1_item_rec_0001") == "rec_0001"


def test_extract_order_id_with_underscores_in_order_id():
    assert extract_order_id("run_1_item_rec_0001_extra") == "rec_0001_extra"


def test_extract_order_id_no_item_marker_falls_back_to_whole_string():
    assert extract_order_id("no-marker-here") == "no-marker-here"


def test_parse_failure_truncates_raw_to_200_chars():
    long_text = "x" * 500
    failure = ParseFailure(order_id="rec_0001", error="Parse failure", raw=long_text)
    assert len(failure.raw) == 200
    assert failure.raw == "x" * 200


def test_parse_failure_short_raw_not_truncated():
    failure = ParseFailure(order_id="rec_0001", error="Parse failure", raw="short")
    assert failure.raw == "short"


def test_parse_batch_results_splits_rows_and_failures():
    results = [
        BatchResult(custom_id="run_1_item_rec_0001", ok=True, text="rec_0001,1,2"),
        BatchResult(custom_id="run_1_item_rec_0002", ok=False, error="simulated item error"),
        BatchResult(custom_id="run_1_item_rec_0003", ok=True, text="garbage unparseable"),
    ]
    rows, failures = parse_batch_results(results, n_fields=2)

    assert rows == [{"order_id": "rec_0001", "values": [1.0, 2.0]}]
    assert len(failures) == 2

    item_error = next(f for f in failures if f.order_id == "rec_0002")
    assert item_error.error == "simulated item error"

    parse_failure = next(f for f in failures if f.order_id == "rec_0003")
    assert parse_failure.error == "Parse failure"
    assert len(parse_failure.raw) <= 200


def test_parse_batch_results_empty_input():
    rows, failures = parse_batch_results([], n_fields=2)
    assert rows == []
    assert failures == []


def test_parse_batch_results_ok_false_with_no_error_text_uses_unknown():
    results = [BatchResult(custom_id="run_1_item_rec_0009", ok=False, error=None)]
    rows, failures = parse_batch_results(results, n_fields=2)
    assert rows == []
    assert failures[0].error == "Unknown"
