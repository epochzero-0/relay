"""Shared, tolerant result parser plus failure capture.

This deduplicates the three near-identical parse_rating_line copies scattered
across the legacy scripts into one generic function. Models often wrap their
answer in commentary ("Sure! Here is the rating:\n rec_1,7,3"), so the parser
scans every line and accepts the first one that matches the expected shape.

Two accepted shapes per line (n_fields = number of numeric values expected):
    order_id,v1,...,vn   (n_fields + 1 parts; model echoed the order_id)
    v1,...,vn            (n_fields parts; uses the caller-supplied order_id)

Header-ish lines (starting with "order_id", case-insensitive) and blank lines
are skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from relay.core.models import BatchResult


@dataclass
class ParseFailure:
    """A single item that could not be turned into a usable result row.

    raw holds a truncated snippet of the offending model text, for debugging.
    """

    order_id: str
    error: str
    raw: str = ""

    def __post_init__(self) -> None:
        if len(self.raw) > 200:
            self.raw = self.raw[:200]


def parse_result_line(
    text: str, n_fields: int, order_id: str | None = None
) -> dict | None:
    """Scan text for the first line that yields n_fields numeric values.

    Returns {"order_id": str, "values": list[float]} on success, else None.
    The bare-values shape only matches when order_id is supplied as a fallback.
    """
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("order_id"):
            continue
        parts = [p.strip() for p in line.split(",")]

        # Preferred: model echoed the order_id as the first field.
        if len(parts) == n_fields + 1:
            try:
                values = [float(p) for p in parts[1:]]
            except ValueError:
                continue
            return {"order_id": parts[0], "values": values}

        # Fallback: bare values only, attach the caller-supplied order_id.
        if len(parts) == n_fields and order_id is not None:
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue
            return {"order_id": order_id, "values": values}

    return None


def extract_order_id(custom_id: str) -> str:
    """Recover the order_id embedded in a custom_id.

    Our scheme is "run_{run}_item_{order_id}", so we match after "item_".
    Falls back to the whole custom_id if the pattern is absent.
    """
    match = re.search(r"item_(.+)$", custom_id)
    return match.group(1) if match else custom_id


def parse_batch_results(
    results: Iterable[BatchResult], n_fields: int
) -> tuple[list[dict], list[ParseFailure]]:
    """Split a stream of BatchResults into parsed rows and parse failures.

    Errored results become ParseFailures carrying the provider error. Succeeded
    results are parsed; unparseable text becomes a "Parse failure". Each parsed
    row is {"order_id": str, "values": list[float]}.
    """
    rows: list[dict] = []
    failures: list[ParseFailure] = []

    for result in results:
        order_id = extract_order_id(result.custom_id)

        if not result.ok:
            failures.append(
                ParseFailure(order_id=order_id, error=result.error or "Unknown")
            )
            continue

        text = result.text or ""
        parsed = parse_result_line(text, n_fields, order_id=order_id)
        if parsed is None:
            failures.append(
                ParseFailure(order_id=order_id, error="Parse failure", raw=text)
            )
            continue

        rows.append(parsed)

    return rows, failures
