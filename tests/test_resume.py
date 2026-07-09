"""Behavioral resume/rerun tests, driven through Orchestrator.run.

Covers: fresh convergence, idempotent rerun (no double submit, byte-identical
CSV), crash-drain recovery of in-flight work left SUBMITTED by a "crashed"
process, and the overall no-double-submit invariant across a resume.
"""

from __future__ import annotations

import csv

from polybatch.core.models import ProviderLimits
from polybatch.core.tracker import Tracker, DONE

from .conftest import make_records


def _csv_row_count(csv_path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def test_fresh_run_converges_with_full_coverage(make_job, counting_provider, orchestrator_factory):
    records = make_records(30)
    job = make_job(records, run_id=1)
    provider = counting_provider(seed=0, limits=ProviderLimits(max_items_per_batch=10))
    orch = orchestrator_factory(provider)

    report = orch.run(job)

    assert report.converged is True
    assert report.coverage == 1.0
    assert _csv_row_count(report.output_csv) == 30


def test_rerun_with_same_tracker_and_output_does_not_resubmit(
    make_job, counting_provider, orchestrator_factory
):
    records = make_records(30)
    job = make_job(records, run_id=1)
    provider = counting_provider(seed=0, limits=ProviderLimits(max_items_per_batch=10))
    orch = orchestrator_factory(provider)

    first_report = orch.run(job)
    csv_bytes_before = first_report.output_csv.read_bytes()
    total_items_after_first_run = provider.items_submitted

    provider.submit_calls = 0  # measure only the second run's submissions

    second_report = orch.run(job)

    assert provider.submit_calls == 0
    assert second_report.converged is True
    assert second_report.coverage == 1.0
    csv_bytes_after = second_report.output_csv.read_bytes()
    assert csv_bytes_after == csv_bytes_before

    # No double submit across the resume: total items ever submitted equals
    # the number of records, not more (the rerun contributed nothing new).
    assert total_items_after_first_run == len(records)
    assert provider.items_submitted == len(records)


def test_crash_drain_recovers_in_flight_chunk_without_resubmitting(
    make_job, counting_provider, orchestrator_factory
):
    # 12 records, limit 10 -> two chunks: [0:10] and [10:12]. We will "crash"
    # after chunk1 (the 2-item chunk) was submitted but never completed.
    records = make_records(12)
    job = make_job(records, run_id=1)
    provider = counting_provider(seed=0, limits=ProviderLimits(max_items_per_batch=10))
    orch = orchestrator_factory(provider)

    report = orch.run(job)
    assert report.converged is True
    assert _csv_row_count(report.output_csv) == 12

    # Find the second chunk's tracker key (the 2-item one) and its real job id.
    tracker = Tracker(job.tracker_path)
    prefix = f"run{job.run_id}_"
    chunk_keys = sorted(
        key for key, status in tracker.summary().items() if status == DONE
    )
    assert len(chunk_keys) == 2
    victim_key = chunk_keys[1]  # p1_chunk1, the 2-item chunk
    victim_job_id = tracker.job_id(victim_key)
    assert victim_job_id is not None

    victim_order_ids = {records[10].order_id, records[11].order_id}

    # Simulate a crash: the chunk was submitted (real job id preserved) but
    # never marked done, and its rows never made it into the CSV.
    tracker.mark_submitted(victim_key, victim_job_id)

    rows = list(csv.DictReader(report.output_csv.open("r", newline="", encoding="utf-8")))
    remaining_rows = [r for r in rows if r["order_id"] not in victim_order_ids]
    assert len(remaining_rows) == 10
    with report.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order_id", "v1", "v2"])
        writer.writeheader()
        writer.writerows(remaining_rows)

    provider.submit_calls = 0  # measure only the recovery run's submissions

    recovery_report = orch.run(job)

    assert recovery_report.recovered_items > 0
    assert provider.submit_calls == 0
    assert recovery_report.coverage == 1.0
    assert recovery_report.converged is True
    assert _csv_row_count(recovery_report.output_csv) == 12
