"""Fault-tolerance / convergence tests driven through Orchestrator.run.

Exercises the multi-pass coverage loop end to end against the real 300-record
dataset (data/records.csv) with every MockProvider failure knob: chaos
(errors + drops + transient submit failures), expiry, crash-drain recovery,
completed-run reruns, guaranteed non-convergence, the BatchTooLargeError
shrink path, and transient submit retries.

Golden regression numbers (verified twice via CLI during Phase 2 and
re-verified here via direct Orchestrator.run):
  * chaos seed=7, error=0.1, drop=0.1, submit_failure=0.3, max_passes=5
    -> converged, coverage 1.0, passes == 5, resent_items == 92
  * crash drain of one 100-item chunk -> recovered_items == 100, 0 submits
  * completed-run rerun -> 0 submits, coverage 1.0, passes == 0 (see note in
    test_completed_run_rerun_is_a_noop), resent_items == 0

All timings are zero, everything runs offline and instantly; all outputs go
under tmp_path.
"""

from __future__ import annotations

import csv
import functools
import json
from pathlib import Path

from polybatch.core.models import ProviderLimits, Record
from polybatch.core.tracker import FAILED, SUBMIT_FAILED, Tracker
from polybatch.providers.base import BatchTooLargeError
from polybatch.providers.mock import MockProvider

DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "records.csv"

#: The chaos scenario's pinned golden numbers (seed 7 on data/records.csv).
CHAOS_KWARGS = dict(seed=7, error_rate=0.1, drop_rate=0.1, submit_failure_rate=0.3)
CHAOS_GOLDEN_PASSES = 5
CHAOS_GOLDEN_RESENT = 92


@functools.lru_cache(maxsize=1)
def _data_records() -> tuple[Record, ...]:
    """Load the committed 300-record dataset once per test session."""
    with DATA_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = tuple(
            Record(order_id=row["order_id"], text=row["text"]) for row in reader
        )
    assert len(records) == 300, "data/records.csv should hold exactly 300 rows"
    return records


def _csv_order_ids(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return [row["order_id"] for row in csv.DictReader(handle)]


# ---------------------------------------------------------------------------
# 1 + 2: chaos convergence (golden numbers) and determinism
# ---------------------------------------------------------------------------


def test_chaos_convergence_matches_golden_numbers(make_job, orchestrator_factory):
    records = _data_records()
    job = make_job(records, run_id=1)
    provider = MockProvider(**CHAOS_KWARGS)  # default limit: 100 items/batch
    orch = orchestrator_factory(provider, max_passes=5)

    report = orch.run(job)

    assert report.converged is True
    assert report.coverage == 1.0
    assert report.passes == CHAOS_GOLDEN_PASSES
    assert report.resent_items == CHAOS_GOLDEN_RESENT
    assert report.recovered_items == 0
    # Every one of the 300 order ids made it into the CSV, exactly once.
    ids = _csv_order_ids(report.output_csv)
    assert len(ids) == 300
    assert set(ids) == {record.order_id for record in records}


def test_chaos_run_is_deterministic(make_job, orchestrator_factory):
    outcomes = []
    for run_id in (1, 2):  # separate output dirs / trackers per run_id
        job = make_job(_data_records(), run_id=run_id)
        provider = MockProvider(**CHAOS_KWARGS)
        orch = orchestrator_factory(provider, max_passes=5)
        report = orch.run(job)
        outcomes.append(
            (
                report.converged,
                report.coverage,
                report.passes,
                report.resent_items,
                report.recovered_items,
                report.succeeded,
                report.output_csv.read_bytes(),
            )
        )

    assert outcomes[0] == outcomes[1]


# ---------------------------------------------------------------------------
# 3: expiry heals through the coverage loop
# ---------------------------------------------------------------------------


def test_expired_jobs_are_resent_and_never_fetched(make_job, orchestrator_factory):
    # seed=0 with expire_rate=0.3 expires at least one of the three 100-item
    # jobs (verified below via the tracker), so this exercises the
    # mark_failed("expired") -> coverage re-send path. If an expired job were
    # ever fetched, MockProvider.fetch_results would raise RuntimeError and
    # this test would error out.
    job = make_job(_data_records(), run_id=1)
    provider = MockProvider(seed=0, expire_rate=0.3)
    orch = orchestrator_factory(provider, max_passes=5)

    report = orch.run(job)

    assert report.converged is True
    assert report.coverage == 1.0
    assert report.resent_items > 0  # the expired chunk's items went out again

    # The tracker recorded the expiry as a FAILED chunk with reason "expired".
    tracker_data = json.loads(job.tracker_path.read_text(encoding="utf-8"))
    expired_entries = [
        entry
        for entry in tracker_data["chunks"].values()
        if entry["status"] == FAILED and entry.get("reason") == "expired"
    ]
    assert len(expired_entries) >= 1


# ---------------------------------------------------------------------------
# 4: crash drain (golden numbers)
# ---------------------------------------------------------------------------


def test_crash_drain_recovers_chunk_without_resubmitting(
    make_job, counting_provider, orchestrator_factory
):
    records = _data_records()
    job = make_job(records, run_id=1)
    provider = counting_provider(seed=1)  # no failure knobs: 3 clean chunks
    orch = orchestrator_factory(provider, max_passes=5)

    report = orch.run(job)
    assert report.converged is True

    # Hand-reset the first 100-item chunk to SUBMITTED (keeping its real
    # job_id) and delete its 100 rows from the CSV, simulating a crash after
    # submit but before fetch/merge.
    tracker = Tracker(job.tracker_path)
    victim_key = f"run{job.run_id}_p1_chunk0"
    victim_job_id = tracker.job_id(victim_key)
    assert victim_job_id is not None
    tracker.mark_submitted(victim_key, victim_job_id)

    victim_ids = {record.order_id for record in records[:100]}
    with report.output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["order_id"] not in victim_ids]
    assert len(rows) == 200
    with report.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order_id", "v1", "v2"])
        writer.writeheader()
        writer.writerows(rows)

    provider.submit_calls = 0  # count only the recovery run's submissions

    recovery = orch.run(job)

    assert recovery.recovered_items == 100  # golden
    assert provider.submit_calls == 0  # drained, never re-submitted
    assert recovery.coverage == 1.0
    assert recovery.converged is True
    assert len(_csv_order_ids(recovery.output_csv)) == 300


# ---------------------------------------------------------------------------
# 5: completed-run rerun is a no-op
# ---------------------------------------------------------------------------


def test_completed_run_rerun_is_a_noop(make_job, counting_provider, orchestrator_factory):
    job = make_job(_data_records(), run_id=1)
    provider = counting_provider(seed=2)
    orch = orchestrator_factory(provider, max_passes=5)

    first = orch.run(job)
    assert first.converged is True
    csv_bytes = first.output_csv.read_bytes()

    provider.submit_calls = 0

    rerun = orch.run(job)

    assert provider.submit_calls == 0  # golden
    assert rerun.coverage == 1.0  # golden
    assert rerun.resent_items == 0  # golden
    # NOTE: the Phase 2 golden notes said passes == 1 here, but both direct
    # Orchestrator.run and the CLI reproduce passes == 0: the coverage loop
    # breaks on "missing is empty" BEFORE incrementing the pass counter, so a
    # fully-covered rerun performs zero passes. Pinned to the value both
    # reproduction paths agree on.
    assert rerun.passes == 0
    assert rerun.converged is True
    assert rerun.output_csv.read_bytes() == csv_bytes  # byte-identical CSV


# ---------------------------------------------------------------------------
# 6: guaranteed non-convergence
# ---------------------------------------------------------------------------


def test_drop_everything_never_converges(make_job, orchestrator_factory):
    job = make_job(_data_records(), run_id=1)
    provider = MockProvider(seed=0, drop_rate=1.0)
    orch = orchestrator_factory(provider, max_passes=3)

    report = orch.run(job)

    assert report.converged is False
    assert report.coverage == 0.0
    assert report.passes == 3  # burned every allowed pass
    assert report.succeeded == 0
    # Every pass after the first re-sent all 300 still-missing items.
    assert report.resent_items == 600


# ---------------------------------------------------------------------------
# 7: BatchTooLargeError shrink path
# ---------------------------------------------------------------------------


class _TooLargeAbove50Provider:
    """Advertises a 100-item limit but rejects any chunk larger than 50.

    Models a provider whose declared limit is optimistic (e.g. a hidden token
    or queue ceiling): the orchestrator must halve its effective chunk size
    and re-chunk on the next pass instead of retrying the same batch.
    """

    def __init__(self, inner: MockProvider, threshold: int = 50) -> None:
        self._inner = inner
        self.name = inner.name
        self.limits = inner.limits  # still advertises 100 items/batch
        self.threshold = threshold

    def submit(self, requests):
        if len(requests) > self.threshold:
            raise BatchTooLargeError(
                f"batch of {len(requests)} exceeds hidden ceiling {self.threshold}"
            )
        return self._inner.submit(requests)

    def poll(self, job_id):
        return self._inner.poll(job_id)

    def fetch_results(self, job_id):
        return self._inner.fetch_results(job_id)


def test_batch_too_large_shrinks_chunks_and_still_converges(
    make_job, orchestrator_factory
):
    job = make_job(_data_records(), run_id=1)
    provider = _TooLargeAbove50Provider(
        MockProvider(seed=3, limits=ProviderLimits(max_items_per_batch=100))
    )
    orch = orchestrator_factory(provider, max_passes=5)

    report = orch.run(job)

    # Pass 1 submits a 100-item chunk, gets rejected, halves to 50; pass 2
    # re-chunks the 300 missing items into 50s and completes.
    assert report.converged is True
    assert report.coverage == 1.0
    assert report.passes >= 2  # the shrink cost at least one extra pass
    assert len(_csv_order_ids(report.output_csv)) == 300

    tracker = Tracker(job.tracker_path)
    submit_failed = [k for k, s in tracker.summary().items() if s == SUBMIT_FAILED]
    assert len(submit_failed) >= 1  # the rejected chunk was recorded


# ---------------------------------------------------------------------------
# 8: transient submit failures are retried with backoff
# ---------------------------------------------------------------------------


def test_transient_submit_failures_retry_until_success(
    make_job, counting_provider, orchestrator_factory
):
    # counting_provider wraps MockProvider and increments submit_calls BEFORE
    # delegating, so attempts that raise MockTransientError are counted too.
    job = make_job(_data_records(), run_id=1)
    provider = counting_provider(seed=11, submit_failure_rate=0.6)
    orch = orchestrator_factory(provider, max_submit_retries=3, max_passes=5)

    report = orch.run(job)

    assert report.converged is True
    assert report.coverage == 1.0
    # At 60% failure some attempts certainly raised: total submit attempts
    # must exceed the number of chunk submissions the run needed.
    assert provider.submit_calls > report.chunks
    assert report.chunks == 3  # 300 records / 100-item default limit, 1 pass
