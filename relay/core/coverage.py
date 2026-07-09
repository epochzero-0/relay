"""Coverage helpers: which expected order ids actually made it into the CSV.

Async batch results come back partial -- some items are dropped, some batches
expire, some submits fail. "Coverage" is the answer to a single question: of
the order ids we expected, which are present in the run's output CSV and which
are still missing? The orchestrator's re-send loop drives itself off exactly
this, re-chunking the missing ids until the CSV is complete.

This is a port of the legacy fix script's find_missing (set difference of
expected ids against the ids present in run{N}.csv), generalized to any CSV
with an "order_id" column and kept order-preserving so re-sends stay stable.

Stdlib only; imports nothing from relay.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def present_ids(csv_path: Path) -> set[str]:
    """Return the set of order_id values present in a run CSV.

    Returns an empty set if the file does not exist (nothing submitted yet) or
    has no "order_id" column.
    """
    path = Path(csv_path)
    if not path.exists():
        return set()

    present: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "order_id" not in reader.fieldnames:
            return set()
        for row in reader:
            order_id = row.get("order_id")
            if order_id:
                present.add(order_id)
    return present


def missing_ids(expected: Iterable[str], csv_path: Path) -> list[str]:
    """Return expected ids not yet present in the CSV, in expected's order.

    Order is preserved (unlike the legacy sorted set difference) so re-sends
    chunk the missing items deterministically and in a human-readable order.
    """
    present = present_ids(csv_path)
    return [order_id for order_id in expected if order_id not in present]
