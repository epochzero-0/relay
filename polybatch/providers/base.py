"""Provider protocol: the contract every backend (mock, Claude, GPT, ...) implements.

A Provider is a thin adapter over one vendor's batch API. polybatch's
orchestrator drives every provider through the same lifecycle so that
fault-tolerance logic (chunking, polling, coverage re-send) lives in one place.

Lifecycle contract (per chunk of Requests):
  1. submit(requests) -> job_id
       Called AT MOST ONCE per chunk. Returns an opaque job id that the
       orchestrator persists to its tracker for crash recovery. May raise on
       transient failure (retry) or on a batch that exceeds provider limits.
  2. poll(job_id) -> JobStatus
       Called REPEATEDLY (with backoff) until JobStatus.is_terminal is True.
       Must tolerate being called many times; must not mutate results.
  3. fetch_results(job_id) -> Iterator[BatchResult]
       Called ONLY AFTER poll reports a terminal state. May be called more
       than once (e.g. retry after a crash mid-download) and MUST yield the
       same items each time for a given job. Partial batches are allowed:
       some input custom_ids may simply be absent, which the orchestrator's
       coverage check detects and re-sends.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from polybatch.core.models import BatchResult, JobStatus, ProviderLimits, Request


# ----- error taxonomy --------------------------------------------------------
# Providers signal *why* a submit failed by the exception type they raise, so
# the orchestrator can decide whether (and how) to retry. Concrete providers
# raise these or subclasses of them from submit().


class TransientSubmitError(RuntimeError):
    """A retryable submit failure (rate limit / HTTP 429 / flaky network).

    The orchestrator re-submits the same chunk, with exponential backoff, up
    to its retry ceiling. The chunk should be submittable unchanged.
    """


class BatchTooLargeError(ValueError):
    """A submit rejected because the batch exceeds a provider size limit.

    NOT retryable as-is: re-submitting the identical chunk would fail again.
    The orchestrator instead shrinks its effective chunk size and lets the
    coverage re-send loop re-chunk the skipped items at the smaller size.
    """


@runtime_checkable
class Provider(Protocol):
    """Structural contract for a batch-inference backend.

    Implementations may be classes with attributes or properties; only the
    shapes below are required.
    """

    #: Short, stable identifier for the provider (e.g. "mock", "claude").
    name: str

    #: Per-batch limits used by the chunker to size sub-batches safely.
    limits: ProviderLimits

    def submit(self, requests: list[Request]) -> str:
        """Submit one chunk of requests. Returns an opaque job_id.

        Called at most once per chunk. Implementations should raise
        TransientSubmitError (or a subclass) on retryable failures so the
        orchestrator can retry with backoff, and BatchTooLargeError (or a
        subclass) when the batch exceeds self.limits so the orchestrator can
        shrink its chunk size instead of retrying the same batch.
        """
        ...

    def poll(self, job_id: str) -> JobStatus:
        """Return the current status of a submitted job.

        Called repeatedly until the returned JobStatus.is_terminal is True.
        """
        ...

    def fetch_results(self, job_id: str) -> Iterator[BatchResult]:
        """Yield results for a terminal job.

        Only valid after poll has reported a terminal state. Must be
        deterministic across repeated calls for the same job.
        """
        ...
