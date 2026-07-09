"""Light determinism/contract tests for relay.providers.mock.MockProvider."""

from __future__ import annotations

import pytest

from relay.core.models import ProviderLimits, Request
from relay.providers.mock import MockBatchTooLargeError, MockProvider


def _requests(n: int) -> list[Request]:
    return [
        Request(custom_id=f"run_1_item_rec_{i:04d}", order_id=f"rec_{i:04d}", prompt=f"p{i}")
        for i in range(n)
    ]


def test_same_seed_yields_identical_fetch_results():
    reqs = _requests(5)

    p1 = MockProvider(seed=42, polls_to_complete=1)
    job1 = p1.submit(reqs)
    p1.poll(job1)
    results1 = list(p1.fetch_results(job1))

    p2 = MockProvider(seed=42, polls_to_complete=1)
    job2 = p2.submit(reqs)
    p2.poll(job2)
    results2 = list(p2.fetch_results(job2))

    assert results1 == results2


def test_fetch_before_terminal_raises_runtime_error():
    reqs = _requests(3)
    provider = MockProvider(seed=0, polls_to_complete=3)
    job_id = provider.submit(reqs)
    provider.poll(job_id)  # first poll: validating, not terminal

    with pytest.raises(RuntimeError):
        list(provider.fetch_results(job_id))


def test_submit_over_limit_raises_batch_too_large():
    provider = MockProvider(seed=0, limits=ProviderLimits(max_items_per_batch=2))
    reqs = _requests(3)
    with pytest.raises(MockBatchTooLargeError):
        provider.submit(reqs)


def test_fetch_results_idempotent_across_repeated_calls():
    reqs = _requests(4)
    provider = MockProvider(seed=1, polls_to_complete=1, error_rate=0.5)
    job_id = provider.submit(reqs)
    provider.poll(job_id)

    first = list(provider.fetch_results(job_id))
    second = list(provider.fetch_results(job_id))
    assert first == second


def test_expired_job_fetch_raises_runtime_error():
    reqs = _requests(3)
    provider = MockProvider(seed=0, polls_to_complete=1, expire_rate=1.0)
    job_id = provider.submit(reqs)
    status = provider.poll(job_id)
    assert status.state == "expired"

    with pytest.raises(RuntimeError):
        list(provider.fetch_results(job_id))


def test_poll_lifecycle_reaches_ended_after_polls_to_complete():
    reqs = _requests(2)
    provider = MockProvider(seed=0, polls_to_complete=3)
    job_id = provider.submit(reqs)

    s1 = provider.poll(job_id)
    assert s1.state == "validating"
    s2 = provider.poll(job_id)
    assert s2.state == "in_progress"
    s3 = provider.poll(job_id)
    assert s3.state == "ended"
