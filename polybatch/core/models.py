"""Core data models shared across polybatch.

These are deliberately small, transport-agnostic value objects. Providers
translate them to/from their own wire formats; the orchestrator only ever
sees these types.

The linchpin of fault tolerance is the stable identifier chain:
  Record.order_id  ->  Request.custom_id (e.g. "run_1_item_rec_0001")
Async batch results come back out of order (and sometimes partial), so every
result is matched back to its input via custom_id / order_id rather than by
position.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    """A single input row loaded from the source dataset.

    order_id is stable and unique per record; it is the anchor that every
    downstream request and result is keyed against.
    """

    order_id: str
    text: str


@dataclass(frozen=True)
class Request:
    """One provider request derived from a Record.

    custom_id embeds both the run and the order_id (e.g.
    "run_{run}_item_{order_id}") so out-of-order async results can be matched
    back to their originating Record.
    """

    custom_id: str
    order_id: str
    prompt: str
    max_tokens: int = 64


@dataclass(frozen=True)
class ProviderLimits:
    """Per-batch ceilings advertised by a provider.

    None means "no limit enforced by polybatch" for that dimension.
    """

    max_items_per_batch: int | None = None
    max_tokens_per_batch: int | None = None


@dataclass(frozen=True)
class JobStatus:
    """Snapshot of a submitted batch job's progress.

    state is a provider-normalized string. Once state is in TERMINAL_STATES
    the job will not change further and results may be fetched.
    """

    #: States after which a job is final and fetch_results is allowed.
    TERMINAL_STATES = frozenset({"ended", "failed", "expired", "cancelled"})

    state: str
    succeeded: int = 0
    errored: int = 0
    processing: int = 0

    @property
    def is_terminal(self) -> bool:
        """True once the job has reached a state that will not change."""
        return self.state in self.TERMINAL_STATES


@dataclass(frozen=True)
class BatchResult:
    """One result item fetched from a terminal job.

    ok=True carries text; ok=False carries error. custom_id ties the result
    back to the originating Request (and thus its Record.order_id).
    """

    custom_id: str
    ok: bool
    text: str | None = None
    error: str | None = None
