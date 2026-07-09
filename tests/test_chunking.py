"""Tests for polybatch.core.chunking: split_requests and merge_rows."""

from __future__ import annotations

from polybatch.core.chunking import merge_rows, split_requests
from polybatch.core.models import ProviderLimits, Request


def _requests(n: int, max_tokens: int = 64) -> list[Request]:
    return [
        Request(custom_id=f"c{i}", order_id=f"o{i}", prompt=f"p{i}", max_tokens=max_tokens)
        for i in range(n)
    ]


def test_split_respects_max_items_per_batch():
    reqs = _requests(25)
    limits = ProviderLimits(max_items_per_batch=10)
    chunks = split_requests(reqs, limits)
    assert [len(c) for c in chunks] == [10, 10, 5]


def test_split_preserves_order_across_chunks():
    reqs = _requests(23)
    limits = ProviderLimits(max_items_per_batch=7)
    chunks = split_requests(reqs, limits)
    flattened = [r for chunk in chunks for r in chunk]
    assert flattened == reqs


def test_split_single_chunk_when_no_limits():
    reqs = _requests(50)
    limits = ProviderLimits()
    chunks = split_requests(reqs, limits)
    assert len(chunks) == 1
    assert chunks[0] == reqs


def test_split_empty_requests_returns_empty_list():
    limits = ProviderLimits(max_items_per_batch=10)
    assert split_requests([], limits) == []


def test_split_no_request_is_split_item_stays_whole():
    reqs = _requests(3)
    limits = ProviderLimits(max_items_per_batch=1)
    chunks = split_requests(reqs, limits)
    assert [len(c) for c in chunks] == [1, 1, 1]


def test_split_token_limit_path_with_custom_estimator():
    # Custom estimator: cost = 1 token per request, so max_tokens_per_batch=3
    # should behave like an item-count limit of 3.
    reqs = _requests(7)
    limits = ProviderLimits(max_tokens_per_batch=3)
    chunks = split_requests(reqs, limits, estimate_tokens=lambda r: 1)
    assert [len(c) for c in chunks] == [3, 3, 1]


def test_split_token_limit_never_splits_a_single_oversized_request():
    # A single request whose own cost exceeds the token budget still gets its
    # own chunk (the "current" guard only closes a chunk when non-empty).
    reqs = _requests(2)
    limits = ProviderLimits(max_tokens_per_batch=5)
    chunks = split_requests(reqs, limits, estimate_tokens=lambda r: 100)
    assert [len(c) for c in chunks] == [1, 1]


def test_split_combined_item_and_token_limits():
    reqs = _requests(6)
    limits = ProviderLimits(max_items_per_batch=4, max_tokens_per_batch=10)
    # cost 4/item -> token limit closes chunk at 2 items well before the item
    # limit of 4 is reached.
    chunks = split_requests(reqs, limits, estimate_tokens=lambda r: 4)
    assert [len(c) for c in chunks] == [2, 2, 2]


def test_merge_rows_keeps_last_on_duplicate_id():
    existing = [{"order_id": "a", "v1": "1"}, {"order_id": "b", "v1": "2"}]
    new = [{"order_id": "a", "v1": "99"}]
    merged = merge_rows(existing, new)
    assert merged == [{"order_id": "a", "v1": "99"}, {"order_id": "b", "v1": "2"}]


def test_merge_rows_preserves_first_seen_order():
    existing = [{"order_id": "b", "v1": "1"}, {"order_id": "a", "v1": "2"}]
    new = [{"order_id": "b", "v1": "3"}, {"order_id": "c", "v1": "4"}]
    merged = merge_rows(existing, new)
    assert [r["order_id"] for r in merged] == ["b", "a", "c"]
    assert merged[0]["v1"] == "3"  # updated in place, not moved to the end


def test_merge_rows_new_id_appends():
    existing = [{"order_id": "a", "v1": "1"}]
    new = [{"order_id": "z", "v1": "9"}]
    merged = merge_rows(existing, new)
    assert [r["order_id"] for r in merged] == ["a", "z"]


def test_merge_rows_custom_id_field():
    existing = [{"pk": "1", "x": "a"}]
    new = [{"pk": "1", "x": "b"}, {"pk": "2", "x": "c"}]
    merged = merge_rows(existing, new, id_field="pk")
    assert merged == [{"pk": "1", "x": "b"}, {"pk": "2", "x": "c"}]


def test_merge_rows_empty_inputs():
    assert merge_rows([], []) == []
    assert merge_rows([{"order_id": "a"}], []) == [{"order_id": "a"}]
    assert merge_rows([], [{"order_id": "a"}]) == [{"order_id": "a"}]
