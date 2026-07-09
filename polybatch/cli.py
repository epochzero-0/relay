"""polybatch CLI: run / status / demo.

  polybatch run     load records, run the orchestrator against a provider.
  polybatch status  print each chunk's tracked state from a tracker file.
  polybatch demo    placeholder (the scripted demo arrives in Phase 3).

Only the mock provider is wired up in Phase 1; real providers arrive in
Phase 4.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from polybatch.core.models import DEFAULT_TASK, Job, Record
from polybatch.core.orchestrator import Orchestrator
from polybatch.core.tracker import Tracker
from polybatch.providers.mock import MockProvider


def _load_records(path: Path, limit: int | None = None) -> list[Record]:
    """Load records from a CSV with an order_id,text header."""
    records: list[Record] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "order_id" not in reader.fieldnames:
            raise ValueError(
                f"input CSV must have an 'order_id,text' header: {path}"
            )
        for row in reader:
            records.append(
                Record(order_id=row["order_id"], text=row.get("text", ""))
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def _cmd_run(args: argparse.Namespace) -> int:
    if args.provider != "mock":
        print(
            f"unsupported provider {args.provider!r}: real providers arrive "
            f"in Phase 4 (only 'mock' is supported now)"
        )
        return 2

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input CSV not found: {input_path}")
        return 2

    records = _load_records(input_path, limit=args.limit_items)
    if not records:
        print(f"no records loaded from {input_path}")
        return 2

    provider = MockProvider(seed=args.seed)
    job = Job(
        run_id=args.run_id,
        records=tuple(records),
        task=DEFAULT_TASK,
        output_dir=Path(args.output_dir),
        tracker_path=Path(args.tracker),
    )
    orchestrator = Orchestrator(provider, poll_interval=args.poll_interval)
    orchestrator.run(job)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    tracker_path = Path(args.tracker)
    if not tracker_path.exists():
        print(f"no tracker file at {tracker_path} (nothing submitted yet)")
        return 0

    tracker = Tracker(tracker_path)
    summary = tracker.summary()
    if not summary:
        print("tracker is empty (no chunks recorded)")
        return 0

    key_width = max(len("chunk"), max(len(k) for k in summary))
    status_width = max(len("status"), max(len(v) for v in summary.values()))
    header = f"{'chunk':<{key_width}}  {'status':<{status_width}}  job_id"
    print(header)
    print("-" * len(header))
    for key in summary:
        status = summary[key]
        job_id = tracker.job_id(key) or "-"
        print(f"{key:<{key_width}}  {status:<{status_width}}  {job_id}")
    return 0


def _cmd_demo(_args: argparse.Namespace) -> int:
    print("demo arrives in Phase 3")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polybatch",
        description="Fault-tolerant batch-inference orchestrator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a batch job against a provider")
    run_p.add_argument("--input", required=True, help="input CSV (order_id,text)")
    run_p.add_argument("--run-id", type=int, default=1, help="run identifier")
    run_p.add_argument("--provider", default="mock", help="provider name")
    run_p.add_argument("--output-dir", default="outputs", help="output directory")
    run_p.add_argument(
        "--tracker", default="outputs/tracker.json", help="tracker JSON path"
    )
    run_p.add_argument(
        "--poll-interval", type=float, default=0.5, help="seconds between polls"
    )
    run_p.add_argument("--seed", type=int, default=0, help="mock provider seed")
    run_p.add_argument(
        "--limit-items", type=int, default=None, help="cap records loaded"
    )
    run_p.set_defaults(func=_cmd_run)

    status_p = sub.add_parser("status", help="show tracked chunk states")
    status_p.add_argument(
        "--tracker", default="outputs/tracker.json", help="tracker JSON path"
    )
    status_p.set_defaults(func=_cmd_status)

    demo_p = sub.add_parser("demo", help="scripted demo (Phase 3)")
    demo_p.set_defaults(func=_cmd_demo)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
