"""Orchestrator: the happy-path run loop.

run(job) drives one run end to end:
    build requests -> chunk -> (skip | resume | submit) -> poll -> fetch ->
    parse -> merge to CSV -> mark done.

This is a port of the legacy submit/poll/download loop, generalized to any
Provider and to multiple chunks, with per-chunk crash recovery via the Tracker.

Phase 1 is the happy path only: a bounded retry on submit, but NO failure
injection, NO coverage-based partial_failed enforcement, and NO exponential
backoff or re-send loop. Those are Phase 2. The seams are left clean: chunks
that partially fail or error still get recorded, and merge_rows keeps CSV
writes idempotent so a future re-send folds in cleanly.

All sleeping happens only through poll_interval / submit_delay, so tests can
pass 0 and run instantly.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from polybatch.core.chunking import merge_rows, split_requests
from polybatch.core.models import Job, Request
from polybatch.core.parsing import ParseFailure, parse_batch_results
from polybatch.core.tracker import Tracker
from polybatch.providers.base import Provider

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


class Orchestrator:
    """Drives a Provider through the chunk lifecycle for one Job at a time.

    Holds no module-level or cross-run state; every run is self-contained and
    resumable purely from the Job's tracker file.
    """

    def __init__(
        self,
        provider: Provider,
        poll_interval: float = 1.0,
        submit_delay: float = 0.0,
        max_submit_retries: int = 3,
        log: Callable[[str], None] = print,
    ) -> None:
        self.provider = provider
        self.poll_interval = poll_interval
        self.submit_delay = submit_delay
        self.max_submit_retries = max_submit_retries
        self.log = log

    # ----- public API --------------------------------------------------

    def run(self, job: Job) -> RunReport:
        """Execute the job end to end and return a RunReport."""
        job.output_dir.mkdir(parents=True, exist_ok=True)
        tracker = Tracker(job.tracker_path)

        requests = self._build_requests(job)
        chunks = split_requests(requests, self.provider.limits)
        output_csv = job.output_dir / f"run{job.run_id}.csv"

        self.log(
            f"run {job.run_id}: {len(requests)} records -> {len(chunks)} chunks"
        )

        skipped = 0
        resumed = 0
        submitted = 0
        # (chunk index, tracker key, job_id) for every chunk we must poll.
        in_flight: list[tuple[int, str, str]] = []

        # ---- phase 1: decide + submit each chunk ----
        for i, chunk in enumerate(chunks):
            key = f"run{job.run_id}_chunk{i}"
            action, saved_job_id = tracker.decide(key)

            if action == "skip":
                skipped += 1
                self.log(f"[chunk {i}] skip (already done)")
                continue

            if action == "resume" and saved_job_id is not None:
                resumed += 1
                self.log(f"[chunk {i}] resuming job {saved_job_id}")
                in_flight.append((i, key, saved_job_id))
                continue

            job_id = self._submit_with_retry(i, key, chunk, tracker)
            if job_id is None:
                continue  # submit_failed already recorded; do not crash the run.
            submitted += 1
            in_flight.append((i, key, job_id))

        # ---- phase 2: poll + download each in-flight chunk ----
        succeeded = 0
        parse_failures = 0
        item_errors = 0

        for i, key, job_id in in_flight:
            state = self._poll_to_terminal(i, job_id)
            if state != "ended":
                tracker.mark_failed(key, state)
                self.log(f"[chunk {i}] terminal state {state} -- marked failed")
                continue

            rows, failures = self._download(job_id, job.task.n_fields)
            self._write_rows(output_csv, rows, job.task.n_fields)
            self._append_failures(job, failures)

            n_parse = sum(1 for f in failures if f.error == _PARSE_FAILURE)
            n_item = len(failures) - n_parse
            succeeded += len(rows)
            parse_failures += n_parse
            item_errors += n_item

            tracker.mark_done(key, succeeded=len(rows), failed=len(failures))
            self.log(
                f"[chunk {i}] done: {len(rows)} ok, "
                f"{n_parse} parse failures, {n_item} item errors"
            )

        total = len(job.records)
        coverage = (succeeded / total) if total else 0.0
        report = RunReport(
            run_id=job.run_id,
            total_records=total,
            chunks=len(chunks),
            skipped_chunks=skipped,
            resumed_chunks=resumed,
            submitted_chunks=submitted,
            succeeded=succeeded,
            parse_failures=parse_failures,
            item_errors=item_errors,
            output_csv=output_csv,
            coverage=coverage,
        )
        self._log_summary(report)
        return report

    # ----- internals ---------------------------------------------------

    def _build_requests(self, job: Job) -> list[Request]:
        """Turn each Record into a provider Request via the task template."""
        task = job.task
        requests: list[Request] = []
        for record in job.records:
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
        self, i: int, key: str, chunk: list[Request], tracker: Tracker
    ) -> str | None:
        """Submit one chunk with a bounded, fixed-delay retry.

        Returns the job id on success (and records mark_submitted), or None on
        exhaustion (and records mark_submit_failed). Never raises past here, so
        one bad chunk cannot crash the whole run.
        """
        last_error = "unknown"
        for attempt in range(1, self.max_submit_retries + 1):
            try:
                job_id = self.provider.submit(chunk)
            except Exception as exc:  # noqa: BLE001 - bounded retry by design.
                last_error = str(exc)
                self.log(f"[chunk {i}] submit attempt {attempt} failed: {exc}")
                if attempt < self.max_submit_retries:
                    time.sleep(self.submit_delay)
                continue
            tracker.mark_submitted(key, job_id)
            self.log(f"[chunk {i}] submitted job {job_id}")
            time.sleep(self.submit_delay)
            return job_id

        tracker.mark_submit_failed(key, last_error)
        self.log(f"[chunk {i}] submit failed after {self.max_submit_retries} tries")
        return None

    def _poll_to_terminal(self, i: int, job_id: str) -> str:
        """Poll a job until it reaches a terminal state; return that state.

        Transient poll exceptions are logged and retried (legacy behavior).
        """
        while True:
            try:
                status = self.provider.poll(job_id)
            except Exception as exc:  # noqa: BLE001 - transient; keep polling.
                self.log(f"[chunk {i}] poll error (retrying): {exc}")
                time.sleep(self.poll_interval)
                continue

            done = status.succeeded + status.errored
            total = done + status.processing
            self.log(f"[chunk {i}] {status.state} {done}/{total}")

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
        (dedupe by order_id, keep last) so reruns and future re-sends do not
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
            f"chunks           : {report.chunks}",
            f"  skipped        : {report.skipped_chunks}",
            f"  resumed        : {report.resumed_chunks}",
            f"  submitted      : {report.submitted_chunks}",
            f"succeeded        : {report.succeeded}",
            f"parse failures   : {report.parse_failures}",
            f"item errors      : {report.item_errors}",
            f"coverage         : {report.coverage * 100:.1f}%",
            f"output           : {report.output_csv}",
            "-----------------------",
        ]
        for line in lines:
            self.log(line)
