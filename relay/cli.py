"""relay CLI: run / status / demo.

  relay run     load records, run the orchestrator against a provider.
  relay status  print each chunk's tracked state from a tracker file.
  relay demo    self-contained, deterministic fault-tolerance narrative.

Real providers (openai/anthropic/google) are resolved through the provider
registry; each requires its optional SDK extra to be installed and its API
key to be set (via a real env var or a ``.env`` file loaded at the start of
``run``).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path

from relay.core.models import DEFAULT_TASK, Job, Record
from relay.core.orchestrator import Orchestrator
from relay.core.tracker import Tracker
from relay.cost import estimate_cost_for_records, format_estimate
from relay.demo import run_demo
from relay.env import load_env
from relay.providers.mock import MockProvider
from relay.providers.registry import get_provider_class, provider_names


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


def _sdk_available(module_name: str) -> bool:
    """True if ``module_name`` is importable.

    Wraps importlib.util.find_spec: for a dotted name whose parent package is
    not installed (e.g. "google.genai" with no "google" package at all),
    find_spec raises ModuleNotFoundError instead of returning None.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _resolve_api_key(provider_cls: type) -> str | None:
    """Look up the API key env var for a provider class.

    The google provider also honors GEMINI_API_KEY as a fallback when
    GOOGLE_API_KEY is unset.
    """
    key = os.environ.get(provider_cls.api_key_env)
    if key:
        return key
    if provider_cls.registry_name == "google":
        return os.environ.get("GEMINI_API_KEY") or None
    return None


def _cmd_run(args: argparse.Namespace) -> int:
    load_env()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input CSV not found: {input_path}")
        return 2

    records = _load_records(input_path, limit=args.limit_items)
    if not records:
        print(f"no records loaded from {input_path}")
        return 2

    try:
        provider_cls = get_provider_class(args.provider)
    except ValueError as exc:
        print(str(exc))
        return 2

    if args.provider == "mock":
        provider = MockProvider(
            seed=args.seed,
            error_rate=args.error_rate,
            drop_rate=args.drop_rate,
            submit_failure_rate=args.submit_failure_rate,
            expire_rate=args.expire_rate,
        )
    else:
        if not _sdk_available(provider_cls.sdk_module):
            print(
                f"the {provider_cls.sdk_module!r} package is required for "
                f"--provider {args.provider}; install it with: "
                f"pip install relay[{provider_cls.install_extra}]"
            )
            return 2

        api_key = _resolve_api_key(provider_cls)
        if not api_key:
            env_name = provider_cls.api_key_env
            if provider_cls.registry_name == "google":
                print(
                    f"missing API key: set {env_name} (or GEMINI_API_KEY) "
                    f"for --provider {args.provider}"
                )
            else:
                print(
                    f"missing API key: set {env_name} for --provider "
                    f"{args.provider}"
                )
            return 2

        provider_kwargs: dict = {
            "api_key": api_key,
            "system": DEFAULT_TASK.system,
        }
        if args.model is not None:
            provider_kwargs["model"] = args.model
        provider = provider_cls(**provider_kwargs)

    job = Job(
        run_id=args.run_id,
        records=tuple(records),
        task=DEFAULT_TASK,
        output_dir=Path(args.output_dir),
        tracker_path=Path(args.tracker),
    )
    orchestrator = Orchestrator(
        provider,
        poll_interval=args.poll_interval,
        backoff_base=args.backoff_base,
        max_passes=args.max_passes,
    )
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


def _cmd_cost(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input CSV not found: {input_path}")
        return 2

    records = _load_records(input_path, limit=args.limit_items)
    if not records:
        print(f"no records loaded from {input_path}")
        return 2

    try:
        estimate = estimate_cost_for_records(
            records, args.model, args.max_tokens
        )
    except ValueError as exc:
        print(str(exc))
        return 2

    print(format_estimate(estimate))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir) if args.output_dir else None
    return run_demo(seed=args.seed, output_dir=output_dir, keep=args.keep)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relay",
        description="Fault-tolerant batch-inference orchestrator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a batch job against a provider")
    run_p.add_argument("--input", required=True, help="input CSV (order_id,text)")
    run_p.add_argument("--run-id", type=int, default=1, help="run identifier")
    run_p.add_argument(
        "--provider", default="mock", choices=provider_names(),
        help="provider name",
    )
    run_p.add_argument(
        "--model", default=None,
        help="model name override for real providers (default: adapter's own default)",
    )
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
    run_p.add_argument(
        "--error-rate", type=float, default=0.0,
        help="mock: fraction of items returned as errors",
    )
    run_p.add_argument(
        "--drop-rate", type=float, default=0.0,
        help="mock: fraction of items dropped (partial batch)",
    )
    run_p.add_argument(
        "--submit-failure-rate", type=float, default=0.0,
        help="mock: probability submit raises a transient error",
    )
    run_p.add_argument(
        "--expire-rate", type=float, default=0.0,
        help="mock: probability a job's terminal state is expired",
    )
    run_p.add_argument(
        "--max-passes", type=int, default=5,
        help="max coverage re-send passes per run",
    )
    run_p.add_argument(
        "--backoff-base", type=float, default=0.5,
        help="base seconds for exponential submit-retry backoff",
    )
    run_p.set_defaults(func=_cmd_run)

    status_p = sub.add_parser("status", help="show tracked chunk states")
    status_p.add_argument(
        "--tracker", default="outputs/tracker.json", help="tracker JSON path"
    )
    status_p.set_defaults(func=_cmd_status)

    cost_p = sub.add_parser(
        "cost", help="offline token/cost estimate for a batch job (no network)"
    )
    cost_p.add_argument("--input", required=True, help="input CSV (order_id,text)")
    cost_p.add_argument("--model", required=True, help="model name to price against")
    cost_p.add_argument(
        "--max-tokens", type=int, default=64,
        help="assumed output token budget per item (default: 64)",
    )
    cost_p.add_argument(
        "--limit-items", type=int, default=None, help="cap records loaded"
    )
    cost_p.set_defaults(func=_cmd_cost)

    demo_p = sub.add_parser(
        "demo", help="scripted, self-contained fault-tolerance narrative"
    )
    demo_p.add_argument(
        "--seed", type=int, default=7, help="mock provider seed"
    )
    demo_p.add_argument(
        "--output-dir", default=None,
        help="output directory (default: a fresh temp dir, cleaned up unless --keep)",
    )
    demo_p.add_argument(
        "--keep", action="store_true",
        help="keep the (temp) output directory instead of deleting it",
    )
    demo_p.set_defaults(func=_cmd_demo)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
