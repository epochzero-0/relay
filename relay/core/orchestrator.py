"""Orchestrator: the multi-pass, fault-tolerant run loop.

run(job) drives one run to full coverage:
    step 0 (resume drain) -> coverage loop over passes:
        compute missing ids -> build requests for only the missing records ->
        chunk -> submit (backoff retry) -> poll -> fetch -> parse ->
        merge to CSV -> mark. Repeat until the CSV covers every expected id
        or max_passes is exhausted.

This is a port of the legacy submit/poll/download loop plus the legacy
fix_missing script, generalized to any Provider and unified into one loop:
coverage (missing_ids over the run CSV) drives re-sends, so dropped items,
expired batches, and failed submits all heal on a later pass.

Fault-tolerance design:
  * Step 0 drains in-flight jobs left SUBMITTED by a crashed process, polling
    and fetching them BEFORE any coverage computation so their work is never
    re-submitted (no double spend).
  * Submit retries use exponential backoff (backoff_base * 2**(attempt-1),
    capped at backoff_cap). BatchTooLargeError is NOT retried: the orchestrator
    halves its effective chunk size and lets the next pass re-chunk smaller.
  * Pass numbers are allocated from the tracker (next_pass), so chunk keys
    never collide across crashes/resumes.

All sleeping happens only through poll_interval / submit_delay / backoff_base,
so tests can pass 0 for all three and run instantly.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from relay.core.chunking import merge_rows, split_requests
from relay.core.coverage import missing_ids, present_ids
from relay.core.models import Job, ProviderLimits, Record, Request
from relay.core.parsing import ParseFailure, parse_batch_results
from relay.core.tracker import Tracker
from relay.providers.base import BatchTooLargeError, Provider

#: Error label the parser uses for unparseable (but non-errored) results. Used
#: to split failures into parse failures vs. provider-side item errors.
_PARSE_FAILURE = "Parse failure"


@dataclass
class RunReport:
    """Summary of a single run, returned by Orchestrator.run."""

    run_id: int
    total_records: int
    chunks: int
    skipped_chunks: int
    resumed_chunks: int
    submitted_chunks: int
    succeeded: int
    parse_failures: int
    item_errors: int
    output_csv: Path
    coverage: float
    # ----- Phase 2 additions -----
    passes: int
    resent_items: int
    recovered_items: int
    converged: bool


@dataclass
class _ChunkOutcome:
    """Counts produced by completing one job (drained or freshly submitted)."""

    ok: int = 0
    parse_failures: int = 0
    item_errors: int = 0
    state: str = ""


class Orchestrator:
    """Drives a Provider through the chunk lifecycle for one Job at a time.

    Holds no module-level or cross-run state; every run is self-contained and
    resumable purely from the Job's tracker file. The one piece of per-run
    mutable state (the shrinking effective item limit) lives on the stack of
    run(), not on the instance.
    """

    def __init__(
        self,
        provider: Provider,
        poll_interval: float = 1.0,
        submit_delay: float = 0.0,
        max_submit_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        max_passes: int = 5,
        log: Callable[[str], None] = print,
    ) -> None:
        self.provider = provider
        self.poll_interval = poll_interval
        self.submit_delay = submit_delay
        self.max_submit_retries = max_submit_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.max_passes = max_passes
        self.log = log

    # ----- public API --------------------------------------------------

    def run(self, job: Job) -> RunReport:
        """Execute the job to full coverage (or max_passes) and report."""
        job.output_dir.mkdir(parents=True, exist_ok=True)
        tracker = Tracker(job.tracker_path)

        expected = [record.order_id for record in job.records]
        by_id = {record.order_id: record for record in job.records}
        output_csv = job.output_dir / f"run{job.run_id}.csv"
        prefix = f"run{job.run_id}_"

        # Effective per-batch item ceiling. Starts at the provider's declared
        # limit (or None) and is halved whenever a batch is rejected as too
        # large, for the rest of this run.
        effective_max_items = self.provider.limits.max_items_per_batch

        self.log(f"run {job.run_id}: {len(expected)} records, up to {self.max_passes} passes")

        totals = _ChunkOutcome()
        submitted_chunks = 0
        total_chunks = 0
        recovered_items = 0
        resent_items = 0

        # ---- step 0: resume drain (crash recovery, before coverage) ----
        drain_keys = tracker.submitted_keys(prefix)
        resumed_chunks = len(drain_keys)
        for key in drain_keys:
            job_id = tracker.job_id(key)
            if job_id is None:  # defensive; submitted_keys already filters this.
                continue
            self.log(f"[drain {key}] resuming in-flight job {job_id}")
            outcome = self._complete_job(
                job, key, job_id, expected_count=None,
                output_csv=output_csv, tracker=tracker,
            )
            self._accumulate(totals, outcome)
            recovered_items += outcome.ok

        # ---- coverage loop ----
        passes = 0
        for invocation_ordinal in range(self.max_passes):
            missing = missing_ids(expected, output_csv)
            if not missing:
                break

            passes += 1
            n = tracker.next_pass(prefix)
            missing_records = [by_id[order_id] for order_id in missing]
            requests = self._build_requests(job, missing_records)
            limits = ProviderLimits(
                max_items_per_batch=effective_max_items,
                max_tokens_per_batch=self.provider.limits.max_tokens_per_batch,
            )
            chunks = split_requests(requests, limits)
            self.log(
                f"[pass {n}] {len(missing)} missing -> {len(chunks)} chunks"
            )

            # Submit every chunk first so batches process server-side in
            # parallel; poll/fetch afterwards. (key, chunk size, job_id) per
            # successfully submitted chunk.
            in_flight: list[tuple[str, int, str]] = []
            for i, chunk in enumerate(chunks):
                key = f"run{job.run_id}_p{n}_chunk{i}"
                total_chunks += 1
                # decide() on fresh pass keys is always "submit"; called for
                # uniformity with the tracker's state machine.
                tracker.decide(key)

                job_id, too_large = self._submit_with_retry(
                    key, chunk, tracker
                )
                if too_large:
                    # Shrink and stop submitting this pass; the next pass
                    # re-chunks the still-missing items at the smaller size.
                    base = effective_max_items if effective_max_items else len(chunk)
                    effective_max_items = max(1, base // 2)
                    self.log(
                        f"[{key}] batch too large; "
                        f"max_items_per_batch -> {effective_max_items}"
                    )
                    break
                if job_id is None:
                    continue  # submit_failed recorded; next pass retries.

                submitted_chunks += 1
                if invocation_ordinal > 0:
                    resent_items += len(chunk)
                in_flight.append((key, len(chunk), job_id))

            for key, chunk_size, job_id in in_flight:
                outcome = self._complete_job(
                    job, key, job_id, expected_count=chunk_size,
                    output_csv=output_csv, tracker=tracker,
                )
                self._accumulate(totals, outcome)

        # ---- final coverage, computed from the CSV (fixes the rerun wart) ----
        present = present_ids(output_csv)
        total = len(expected)
        covered = sum(1 for order_id in expected if order_id in present)
        coverage = (covered / total) if total else 0.0
        converged = covered == total

        report = RunReport(
            run_id=job.run_id,
            total_records=total,
            chunks=total_chunks,
            skipped_chunks=0,
            resumed_chunks=resumed_chunks,
            submitted_chunks=submitted_chunks,
            succeeded=totals.ok,
            parse_failures=totals.parse_failures,
            item_errors=totals.item_errors,
            output_csv=output_csv,
            coverage=coverage,
            passes=passes,
            resent_items=resent_items,
            recovered_items=recovered_items,
            converged=converged,
        )
        self._log_summary(report)
        return report

    # ----- internals ---------------------------------------------------

    def _build_requests(
        self, job: Job, records: list[Record]
    ) -> list[Request]:
        """Turn each Record into a provider Request via the task template."""
        task = job.task
        requests: list[Request] = []
        for record in records:
            prompt = task.prompt_template.format(
                order_id=record.order_id, text=record.text
            )
            requests.append(
                Request(
                    custom_id=f"run_{job.run_id}_item_{record.order_id}",
                    order_id=record.order_id,
                    prompt=prompt,
                    max_tokens=task.max_tokens,
                )
            )
        return requests

    def _submit_with_retry(
        self, key: str, chunk: list[Request], tracker: Tracker
    ) -> tuple[str | None, bool]:
        """Submit one chunk with bounded, exponential-backoff retry.

        Returns (job_id, too_large):
          * (job_id, False) on success (mark_submitted recorded).
          * (None, True) if the batch was rejected as too large -- NOT retried;
            mark_submit_failed recorded and the caller shrinks its chunk size.
          * (None, False) on any other error after all retries are exhausted
            (mark_submit_failed recorded). Never raises past here, so one bad
            chunk cannot crash the whole run.
        """
        last_error = "unknown"
        for attempt in range(1, self.max_submit_retries + 1):
            try:
                job_id = self.provider.submit(chunk)
            except BatchTooLargeError as exc:
                tracker.mark_submit_failed(key, str(exc))
                self.log(f"[{key}] submit rejected (too large): {exc}")
                return (None, True)
            except Exception as exc:  # noqa: BLE001 - bounded retry by design.
                last_error = str(exc)
                self.log(f"[{key}] submit attempt {attempt} failed: {exc}")
                if attempt < self.max_submit_retries:
                    time.sleep(self._backoff(attempt))
                continue
            tracker.mark_submitted(key, job_id)
            self.log(f"[{key}] submitted job {job_id}")
            time.sleep(self.submit_delay)
            return (job_id, False)

        tracker.mark_submit_failed(key, last_error)
        self.log(f"[{key}] submit failed after {self.max_submit_retries} tries")
        return (None, False)

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff wait for a 1-indexed retry attempt."""
        return min(self.backoff_cap, self.backoff_base * 2 ** (attempt - 1))

    def _complete_job(
        self,
        job: Job,
        key: str,
        job_id: str,
        expected_count: int | None,
        output_csv: Path,
        tracker: Tracker,
    ) -> _ChunkOutcome:
        """Poll a job to terminal, then fetch/parse/merge and mark the tracker.

        expected_count is the chunk's item count for freshly-submitted chunks;
        None for drained jobs (whose original size is not recorded), in which
        case the count of returned results is used as the expectation.
        """
        state = self._poll_to_terminal(key, job_id)
        if state != "ended":
            tracker.mark_failed(key, state)
            self.log(f"[{key}] terminal state {state} -- marked failed")
            return _ChunkOutcome(state=state)

        rows, failures = self._download(job_id, job.task.n_fields)
        self._write_rows(output_csv, rows, job.task.n_fields)
        self._append_failures(job, failures)

        ok = len(rows)
        n_parse = sum(1 for f in failures if f.error == _PARSE_FAILURE)
        n_item = len(failures) - n_parse
        expected = expected_count if expected_count is not None else ok + len(failures)

        if ok < expected:
            tracker.mark_partial_failed(key, succeeded=ok, expected=expected)
            self.log(
                f"[{key}] partial: {ok}/{expected} ok, "
                f"{n_parse} parse failures, {n_item} item errors"
            )
        else:
            tracker.mark_done(key, succeeded=ok, failed=len(failures))
            self.log(
                f"[{key}] done: {ok} ok, "
                f"{n_parse} parse failures, {n_item} item errors"
            )
        return _ChunkOutcome(
            ok=ok, parse_failures=n_parse, item_errors=n_item, state=state
        )

    @staticmethod
    def _accumulate(totals: _ChunkOutcome, outcome: _ChunkOutcome) -> None:
        """Fold a chunk's counts into the run totals."""
        totals.ok += outcome.ok
        totals.parse_failures += outcome.parse_failures
        totals.item_errors += outcome.item_errors

    def _poll_to_terminal(self, label: str, job_id: str) -> str:
        """Poll a job until it reaches a terminal state; return that state.

        Transient poll exceptions are logged and retried (legacy behavior).
        """
        while True:
            try:
                status = self.provider.poll(job_id)
            except Exception as exc:  # noqa: BLE001 - transient; keep polling.
                self.log(f"[{label}] poll error (retrying): {exc}")
                time.sleep(self.poll_interval)
                continue

            done = status.succeeded + status.errored
            total = done + status.processing
            self.log(f"[{label}] {status.state} {done}/{total}")

            if status.is_terminal:
                return status.state
            time.sleep(self.poll_interval)

    def _download(
        self, job_id: str, n_fields: int
    ) -> tuple[list[dict], list[ParseFailure]]:
        """Fetch and parse results for a terminal job."""
        results = self.provider.fetch_results(job_id)
        return parse_batch_results(results, n_fields)

    # ----- output ------------------------------------------------------

    def _write_rows(
        self, output_csv: Path, rows: list[dict], n_fields: int
    ) -> None:
        """Merge parsed rows into the run CSV, keeping it idempotent.

        Rows arrive as {"order_id", "values": [...]} and are flattened to the
        columns order_id,v1..v{n_fields}. Any existing CSV is read and merged
        (dedupe by order_id, keep last) so reruns and re-sends do not
        duplicate rows.
        """
        columns = ["order_id"] + [f"v{j}" for j in range(1, n_fields + 1)]
        new_flat = [self._flatten(row, n_fields) for row in rows]

        existing: list[dict] = []
        if output_csv.exists():
            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))

        merged = merge_rows(existing, new_flat, id_field="order_id")

        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(merged)

    @staticmethod
    def _flatten(row: dict, n_fields: int) -> dict:
        """Convert {"order_id", "values"} into flat order_id,v1..vn columns."""
        flat = {"order_id": row["order_id"]}
        for j, value in enumerate(row["values"], start=1):
            flat[f"v{j}"] = f"{value:g}"
        return flat

    def _append_failures(self, job: Job, failures: list[ParseFailure]) -> None:
        """Append failure records (as dicts) to the run's failures JSON."""
        if not failures:
            return
        path = job.output_dir / f"run{job.run_id}_failures.json"
        existing: list[dict] = []
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.extend(asdict(f) for f in failures)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def _log_summary(self, report: RunReport) -> None:
        """Emit a short ASCII summary block at the end of a run."""
        lines = [
            "----- run summary -----",
            f"run id           : {report.run_id}",
            f"records          : {report.total_records}",
            f"passes           : {report.passes}",
            f"chunks           : {report.chunks}",
            f"  resumed        : {report.resumed_chunks}",
            f"  submitted      : {report.submitted_chunks}",
            f"succeeded        : {report.succeeded}",
            f"recovered        : {report.recovered_items}",
            f"resent           : {report.resent_items}",
            f"parse failures   : {report.parse_failures}",
            f"item errors      : {report.item_errors}",
            f"coverage         : {report.coverage * 100:.1f}%",
            f"converged        : {report.converged}",
            f"output           : {report.output_csv}",
            "-----------------------",
        ]
        for line in lines:
            self.log(line)
