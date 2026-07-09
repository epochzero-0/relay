"""Tests for relay.core.coverage: present_ids and missing_ids."""

from __future__ import annotations

import csv

from relay.core.coverage import missing_ids, present_ids


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_present_ids_missing_file_returns_empty_set(tmp_path):
    assert present_ids(tmp_path / "does_not_exist.csv") == set()


def test_present_ids_csv_without_order_id_column_returns_empty_set(tmp_path):
    path = tmp_path / "run1.csv"
    _write_csv(path, ["id", "v1"], [{"id": "a", "v1": "1"}])
    assert present_ids(path) == set()


def test_present_ids_returns_ids_present(tmp_path):
    path = tmp_path / "run1.csv"
    _write_csv(
        path,
        ["order_id", "v1", "v2"],
        [
            {"order_id": "rec_0001", "v1": "1", "v2": "2"},
            {"order_id": "rec_0002", "v1": "3", "v2": "4"},
        ],
    )
    assert present_ids(path) == {"rec_0001", "rec_0002"}


def test_present_ids_empty_csv_with_header_only(tmp_path):
    path = tmp_path / "run1.csv"
    _write_csv(path, ["order_id", "v1"], [])
    assert present_ids(path) == set()


def test_missing_ids_preserves_expected_order(tmp_path):
    path = tmp_path / "run1.csv"
    _write_csv(path, ["order_id", "v1"], [{"order_id": "b", "v1": "1"}])
    expected = ["c", "a", "b", "d"]
    assert missing_ids(expected, path) == ["c", "a", "d"]


def test_missing_ids_all_present_returns_empty_list(tmp_path):
    path = tmp_path / "run1.csv"
    _write_csv(
        path,
        ["order_id", "v1"],
        [{"order_id": "a", "v1": "1"}, {"order_id": "b", "v1": "2"}],
    )
    assert missing_ids(["a", "b"], path) == []


def test_missing_ids_no_file_means_everything_missing(tmp_path):
    path = tmp_path / "does_not_exist.csv"
    expected = ["a", "b", "c"]
    assert missing_ids(expected, path) == expected
