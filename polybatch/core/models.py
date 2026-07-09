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
from pathlib import Path


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


@dataclass(frozen=True)
class TaskSpec:
    """What to ask the model, independent of any provider or dataset.

    prompt_template is a str.format template with two placeholders,
    {order_id} and {text}, filled per Record when building Requests. n_fields
    is how many numeric values the parser should expect back per item.
    """

    prompt_template: str
    n_fields: int
    system: str | None = None
    max_tokens: int = 64


@dataclass(frozen=True)
class Job:
    """One end-to-end unit of work: a run over a fixed set of records.

    records is a tuple so the dataclass stays frozen and hashable-friendly.
    output_dir receives the run CSV and failures JSON; tracker_path is the
    per-run JSON state file that makes the run resumable.
    """

    run_id: int
    records: tuple[Record, ...]
    task: TaskSpec
    output_dir: Path
    tracker_path: Path


#: A generic, domain-neutral scoring task used as the CLI default. It asks the
#: model to score the input text on two 0-10 integer dimensions and reply on a
#: single comma-separated line, matching n_fields=2.
DEFAULT_TASK = TaskSpec(
    prompt_template=(
        "Score the following text on two dimensions, each an integer from 0 to "
        "10:\n"
        "  1. quality  - how clear and well-formed the text is\n"
        "  2. interest - how novel or engaging the text is\n\n"
        "Reply with exactly one line and nothing else, in the form:\n"
        "  order_id,quality,interest\n\n"
        "order_id: {order_id}\n"
        "text: {text}"
    ),
    n_fields=2,
    system="You are a careful evaluator. Reply as: order_id,score1,score2",
    max_tokens=64,
)
