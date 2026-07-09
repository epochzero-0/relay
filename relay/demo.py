"""Scripted end-to-end fault-tolerance demo against the mock provider.

Self-contained and deterministic: generates 300 synthetic records in memory
(no repo checkout needed), then runs the pinned "golden" chaos scenario
(seed=7, 10% item errors, 10% dropped items, 30% transient submit failures)
through the Orchestrator twice against the same tracker/output:

  Act 1 -- a fresh chaos run. Despite the injected faults, the multi-pass
           coverage loop re-sends only what's missing, pass after pass,
           until every record is covered.
  Act 2 -- an immediate rerun of the same job. Coverage is already 100%, so
           this should submit nothing at all (a no-op rerun).

Exits 0 with a closing summary if both acts behave as expected; exits 1 with
an ASCII failure banner otherwise. ASCII-only output, stdlib only.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable

from relay.core.models import DEFAULT_TASK, Job, Record
from relay.core.orchestrator import Orchestrator
from relay.providers.mock import MockProvider

#: Golden chaos scenario knobs (pinned; see tests/test_convergence.py).
CHAOS_KWARGS = dict(error_rate=0.1, drop_rate=0.1, submit_failure_rate=0.3)

RECORD_COUNT = 300


def _make_demo_records(n: int = RECORD_COUNT) -> list[Record]:
    """Deterministic, seed-independent synthetic records rec_0001..rec_NNNN."""
    return [
        Record(order_id=f"rec_{i:04d}", text=f"demo record number {i}")
        for i in range(1, n + 1)
    ]


def _per_pass_log(base_log: Callable[[str], None]) -> Callable[[str], None]:
    """Wrap a log function so only the orchestrator's per-pass lines pass.

    The orchestrator's raw log is very chatty (per-chunk submit/poll/done
    lines); the demo narrative only wants the "[pass N] ... missing -> ...
    chunks" boundary lines so the whole run fits on one screen. This still
    forwards the orchestrator's own log output (not a re-derived summary),
    just filtered to the per-pass granularity the narrative calls for.
    """

    def _log(line: str) -> None:
        if line.startswith("[pass "):
            base_log(line)

    return _log


def run_demo(
    seed: int = 7,
    output_dir: Path | None = None,
    keep: bool = False,
    log: Callable[[str], None] = print,
) -> int:
    """Run the two-act fault-tolerance narrative; return a process exit code.

    output_dir=None uses a fresh tempfile.mkdtemp directory, removed at the
    end unless keep=True. An explicit output_dir is never removed.
    """
    explicit_output_dir = output_dir is not None
    if output_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="relay_demo_"))
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("relay demo -- fault-tolerance narrative")
    log("=" * 60)
    log(f"seed={seed}  records={RECORD_COUNT}  output_dir={out_dir}")
    log("")
    log("Act 1: chaos run")
    log("  injecting: 10% item errors, 10% dropped items,")
    log("             30% transient submit failures (simulated 429s)")
    log("  watch the orchestrator re-send only what's missing, pass by pass")
    log("")

    records = _make_demo_records()
    provider = MockProvider(seed=seed, **CHAOS_KWARGS)
    job = Job(
        run_id=1,
        records=tuple(records),
        task=DEFAULT_TASK,
        output_dir=out_dir,
        tracker_path=out_dir / "tracker.json",
    )
    orchestrator = Orchestrator(
        provider,
        poll_interval=0,
        submit_delay=0,
        backoff_base=0,
        log=_per_pass_log(log),
    )
    report1 = orchestrator.run(job)

    log("")
    log(
        f"Act 1 result: passes={report1.passes} resent_items={report1.resent_items} "
        f"coverage={report1.coverage * 100:.1f}% converged={report1.converged}"
    )
    log("")
    log("Act 2: idempotent rerun (same tracker, same output CSV)")
    log("  coverage is already 100% -- expect 0 resubmits")
    log("")

    report2 = orchestrator.run(job)

    log("")
    log(
        f"Act 2 result: submitted_chunks={report2.submitted_chunks} "
        f"resent_items={report2.resent_items} coverage={report2.coverage * 100:.1f}%"
    )
    log("")

    act1_ok = report1.converged and report1.coverage == 1.0
    act2_ok = report2.submitted_chunks == 0 and report2.resent_items == 0
    success = act1_ok and act2_ok

    if not success:
        log("X" * 60)
        log("X  DEMO FAILED -- fault-tolerance invariant violated       X")
        log("X" * 60)
        if not explicit_output_dir and not keep:
            shutil.rmtree(out_dir, ignore_errors=True)
        return 1

    log("=" * 60)
    log("DEMO PASSED -- fault tolerance verified end to end")
    if explicit_output_dir or keep:
        log(f"output CSV: {report1.output_csv}")
    else:
        shutil.rmtree(out_dir, ignore_errors=True)
        log("temp output dir cleaned up (pass --keep to preserve it)")
    log("=" * 60)
    return 0
