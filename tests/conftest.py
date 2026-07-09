"""Shared pytest fixtures for the relay test suite.

Everything here runs fully offline and instantly: no real sleeps, no network,
no writes outside tmp_path. The orchestrator factory always zeroes every
timing knob (poll_interval, submit_delay, backoff_base) and silences logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from relay.core.models import Job, Record, TaskSpec
from relay.core.orchestrator import Orchestrator
from relay.providers.mock import MockProvider

#: A tiny, deterministic task template independent of DEFAULT_TASK so tests
#: do not depend on the CLI's default wording. n_fields=2 matches
#: MockProvider's two-score fake output ("{order_id},{d1},{d2}").
TEST_TASK = TaskSpec(
    prompt_template="order_id: {order_id}\ntext: {text}",
    n_fields=2,
    max_tokens=16,
)


def make_records(n: int, start: int = 1) -> list[Record]:
    """Build n Records with stable, zero-padded ids like 'rec_0001'."""
    return [
        Record(order_id=f"rec_{i:04d}", text=f"sample text {i}")
        for i in range(start, start + n)
    ]


@pytest.fixture
def records_factory():
    """Return the make_records helper itself, for tests that need custom n."""
    return make_records


@pytest.fixture
def make_job(tmp_path: Path):
    """Factory fixture: make_job(records, run_id=1, task=TEST_TASK) -> Job.

    output_dir and tracker_path both live under tmp_path so nothing escapes
    the pytest sandbox.
    """

    def _make_job(records, run_id: int = 1, task: TaskSpec = TEST_TASK) -> Job:
        output_dir = tmp_path / f"outputs_{run_id}"
        tracker_path = output_dir / "tracker.json"
        return Job(
            run_id=run_id,
            records=tuple(records),
            task=task,
            output_dir=output_dir,
            tracker_path=tracker_path,
        )

    return _make_job


class CountingProvider:
    """Wraps a real Provider, counting submit() calls and items submitted.

    Delegates every method unchanged so it is a drop-in Provider substitute;
    only submit() is instrumented.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.name = inner.name
        self.limits = inner.limits
        self.submit_calls = 0
        self.items_submitted = 0

    def submit(self, requests):
        self.submit_calls += 1
        self.items_submitted += len(requests)
        return self._inner.submit(requests)

    def poll(self, job_id):
        return self._inner.poll(job_id)

    def fetch_results(self, job_id):
        return self._inner.fetch_results(job_id)


@pytest.fixture
def counting_provider():
    """Factory fixture: counting_provider(**mock_kwargs) -> CountingProvider."""

    def _make(**mock_kwargs) -> CountingProvider:
        provider = MockProvider(**mock_kwargs)
        return CountingProvider(provider)

    return _make


@pytest.fixture
def orchestrator_factory():
    """Factory fixture: orchestrator_factory(provider, **overrides) -> Orchestrator.

    All timing knobs default to 0 and logging is silenced so tests run
    instantly and quietly; override any kwarg (e.g. max_passes) as needed.
    """

    def _make(provider, **overrides) -> Orchestrator:
        kwargs = dict(
            poll_interval=0,
            submit_delay=0,
            backoff_base=0,
            log=lambda *a, **k: None,
        )
        kwargs.update(overrides)
        return Orchestrator(provider, **kwargs)

    return _make
