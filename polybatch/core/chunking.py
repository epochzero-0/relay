"""Split requests into provider-safe chunks and merge result rows.

Chunking respects the provider's advertised per-batch limits (item count and,
optionally, an estimated token budget). A single request is never split across
chunks; input order is always preserved so results stay easy to reason about.

merge_rows implements the legacy "concat then drop_duplicates(keep='last')"
behavior used when re-sent results are folded back into an existing CSV. It is
wired up now so the Phase 2 re-send loop can rely on it.
"""

from __future__ import annotations

from typing import Callable

from polybatch.core.models import ProviderLimits, Request


def _default_estimate_tokens(request: Request) -> int:
    """Rough token estimate: ~4 chars per token of prompt, plus output budget."""
    return len(request.prompt) // 4 + request.max_tokens


def split_requests(
    requests: list[Request],
    limits: ProviderLimits,
    estimate_tokens: Callable[[Request], int] | None = None,
) -> list[list[Request]]:
    """Greedily pack requests into chunks that fit the provider's limits.

    A new chunk is started when adding the next request would exceed either
    max_items_per_batch or (if set) max_tokens_per_batch. Both limits being
    None yields a single chunk. Order is preserved and no request is split.
    """
    if not requests:
        return []

    max_items = limits.max_items_per_batch
    max_tokens = limits.max_tokens_per_batch

    if max_items is None and max_tokens is None:
        return [list(requests)]

    estimate = estimate_tokens or _default_estimate_tokens

    chunks: list[list[Request]] = []
    current: list[Request] = []
    current_tokens = 0

    for request in requests:
        cost = estimate(request) if max_tokens is not None else 0

        would_exceed_items = max_items is not None and len(current) >= max_items
        would_exceed_tokens = (
            max_tokens is not None
            and current
            and current_tokens + cost > max_tokens
        )

        if current and (would_exceed_items or would_exceed_tokens):
            chunks.append(current)
            current = []
            current_tokens = 0

        current.append(request)
        current_tokens += cost

    if current:
        chunks.append(current)

    return chunks


def merge_rows(
    existing: list[dict], new: list[dict], id_field: str = "order_id"
) -> list[dict]:
    """Merge two lists of row dicts, deduped by id_field keeping the LAST row.

    First-seen order of ids is preserved (a later duplicate updates the row in
    place rather than moving it to the end). Rows missing id_field are kept as
    is, in order.
    """
    order: list[object] = []
    by_id: dict[object, dict] = {}
    anonymous: list[dict] = []

    for row in list(existing) + list(new):
        if id_field not in row:
            anonymous.append(row)
            continue
        key = row[id_field]
        if key not in by_id:
            order.append(key)
        by_id[key] = row

    merged = [by_id[key] for key in order]
    merged.extend(anonymous)
    return merged
