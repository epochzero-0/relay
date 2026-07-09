"""In-memory mock provider for local development and fault-tolerance testing.

MockProvider implements the Provider protocol with zero network calls. It has
tunable failure knobs so Phase 2 can exercise the orchestrator's recovery
paths deterministically (same seed -> same behavior):

  polls_to_complete   how many poll() calls before a job reaches "ended".
  error_rate          fraction of items returned as errored BatchResults
                      (simulates per-item model/API errors).
  drop_rate           fraction of items silently OMITTED from fetch_results
                      (simulates a partial batch; the coverage re-send loop
                      in Phase 2 is what detects and re-sends these).
  submit_failure_rate probability that submit() raises MockTransientError
                      (simulates a 429 that the submit-retry loop handles).
  expire_rate         probability that a job's terminal state is "expired"
                      instead of "ended" (simulates a batch that timed out;
                      the coverage re-send loop re-submits its items).

Batches larger than limits.max_items_per_batch raise MockBatchTooLargeError
(simulates a token/queue rejection at submit time).

Error, drop, and expiry membership is decided ONCE at submit time from the
seeded RNG, so repeated poll()/fetch_results() calls on the same job yield
identical output.
"""

from __future__ import annotations

import random
from typing import Iterator

from polybatch.core.models import BatchResult, JobStatus, ProviderLimits, Request
from polybatch.providers.base import BatchTooLargeError, TransientSubmitError


class MockTransientError(TransientSubmitError):
    """Simulated transient failure at submit time (e.g. HTTP 429)."""


class MockBatchTooLargeError(BatchTooLargeError):
    """Raised when a submitted batch exceeds the provider's item limit."""


class _Item:
    """Internal per-request record of the decision made at submit time."""

    __slots__ = ("request", "outcome")

    def __init__(self, request: Request, outcome: str) -> None:
        self.request = request
        self.outcome = outcome  # one of: "ok", "error", "drop"


class _Job:
    """Internal state for a single submitted mock batch."""

    __slots__ = ("items", "polls", "expired")

    def __init__(self, items: list[_Item], expired: bool = False) -> None:
        self.items = items
        self.polls = 0
        self.expired = expired


class MockProvider:
    """A fully in-memory Provider implementation.

    Uses a private random.Random(seed) instance for all decisions so it never
    perturbs (or is perturbed by) the global RNG. See module docstring for the
    failure knobs, which Phase 2 exercises.
    """

    def __init__(
        self,
        seed: int = 0,
        polls_to_complete: int = 3,
        error_rate: float = 0.0,
        drop_rate: float = 0.0,
        submit_failure_rate: float = 0.0,
        expire_rate: float = 0.0,
        limits: ProviderLimits | None = None,
    ) -> None:
        self.name: str = "mock"
        self.limits: ProviderLimits = (
            limits if limits is not None else ProviderLimits(max_items_per_batch=100)
        )
        self.polls_to_complete = polls_to_complete
        self.error_rate = error_rate
        self.drop_rate = drop_rate
        self.submit_failure_rate = submit_failure_rate
        self.expire_rate = expire_rate
        self._rng = random.Random(seed)
        self._seed = seed
        self._jobs: dict[str, _Job] = {}
        self._counter = 0

    def submit(self, requests: list[Request]) -> str:
        """Validate limits, maybe fail, else register a job and return its id."""
        limit = self.limits.max_items_per_batch
        if limit is not None and len(requests) > limit:
            raise MockBatchTooLargeError(
                f"batch of {len(requests)} exceeds max_items_per_batch={limit}"
            )

        # Simulated 429 -- decided before the job is registered so a retry
        # submits fresh. The draw is skipped when the knob is off so enabling
        # one failure knob never perturbs the RNG sequence of the others.
        if self.submit_failure_rate > 0.0 and self._rng.random() < self.submit_failure_rate:
            raise MockTransientError("simulated transient submit failure (429)")

        # Roll once for whole-job expiry, after the transient roll but before
        # the per-item fate rolls. The draw is skipped entirely when the knob
        # is off, so enabling expiry never perturbs the existing (seed,
        # request) behavior of runs that leave expire_rate == 0.
        expired = self.expire_rate > 0.0 and self._rng.random() < self.expire_rate

        # Decide each item's fate now, so fetch_results is stable across calls.
        items: list[_Item] = []
        for req in requests:
            roll = self._rng.random()
            if roll < self.error_rate:
                outcome = "error"
            elif roll < self.error_rate + self.drop_rate:
                outcome = "drop"
            else:
                outcome = "ok"
            items.append(_Item(req, outcome))

        self._counter += 1
        job_id = f"mock_batch_{self._counter:04d}"
        self._jobs[job_id] = _Job(items, expired=expired)
        return job_id

    def poll(self, job_id: str) -> JobStatus:
        """Advance the job's poll counter and report a plausible status."""
        job = self._jobs[job_id]  # KeyError on unknown job id, by contract.
        job.polls += 1
        total = len(job.items)

        if job.polls >= self.polls_to_complete:
            if job.expired:
                # Expired terminal: no results are ever fetchable.
                return JobStatus(
                    state="expired", succeeded=0, errored=0, processing=0
                )
            succeeded = sum(1 for it in job.items if it.outcome == "ok")
            errored = sum(1 for it in job.items if it.outcome == "error")
            # Dropped items count as neither succeeded nor errored -- they are
            # simply missing from a partial batch.
            return JobStatus(
                state="ended",
                succeeded=succeeded,
                errored=errored,
                processing=0,
            )

        if job.polls == 1:
            return JobStatus(state="validating", processing=total)

        # in_progress: report a monotonically increasing done count.
        remaining_polls = max(self.polls_to_complete - job.polls, 1)
        done = total - min(total, total // (remaining_polls + 1))
        done = min(done, total)
        return JobStatus(
            state="in_progress",
            succeeded=done,
            processing=total - done,
        )

    def fetch_results(self, job_id: str) -> Iterator[BatchResult]:
        """Yield results for a terminal job; dropped items are omitted."""
        job = self._jobs[job_id]  # KeyError on unknown job id.
        if job.polls < self.polls_to_complete:
            raise RuntimeError(
                f"fetch_results called on non-terminal job {job_id!r}"
            )
        if job.expired:
            # Expired jobs have no results; the orchestrator only fetches
            # "ended" terminals, so reaching here is a contract violation.
            raise RuntimeError(
                f"fetch_results called on expired job {job_id!r}"
            )

        for item in job.items:
            req = item.request
            if item.outcome == "drop":
                continue  # partial batch: silently omit.
            if item.outcome == "error":
                yield BatchResult(
                    custom_id=req.custom_id,
                    ok=False,
                    error="simulated item error",
                )
                continue
            d1, d2 = self._fake_scores(req.order_id)
            yield BatchResult(
                custom_id=req.custom_id,
                ok=True,
                text=f"{req.order_id},{d1},{d2}",
            )

    def _fake_scores(self, order_id: str) -> tuple[int, int]:
        """Deterministic 0-10 score pair derived from order_id and seed.

        Stable across runs and across repeated fetch_results calls for the
        same seed. Uses a fresh local RNG so it does not touch submit/poll
        ordering of self._rng.
        """
        local = random.Random(f"{self._seed}:{order_id}")
        return local.randint(0, 10), local.randint(0, 10)
