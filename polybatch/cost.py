"""Offline cost estimation utility.

Port of the legacy ``check-batches/check_claude_cost.py`` live token-tally
script into a self-contained, OFFLINE estimator. polybatch must never make
real/billable API calls, so this module never touches the network or reads an
API key: it estimates input/output tokens from the input CSV, a model name,
and a max_tokens budget, then multiplies by an editable per-model price
table.

The token-count heuristic (``len(text) // 4``) intentionally matches the
default estimator used by ``polybatch.core.chunking.split_requests`` so the
cost estimate and the chunk-sizing estimate agree with each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from polybatch.core.models import Record, TaskSpec

# ---------------------------------------------------------------------------
# EDITABLE PRICE TABLE
#
# Prices are BATCH-tier (i.e. already discounted ~50% off standard synchronous
# pricing, mirroring the legacy script's "batch pricing" comment), quoted in
# USD per 1,000,000 ("MTok") tokens. These figures are ILLUSTRATIVE
# PLACEHOLDERS dated 2026-07 -- they are not guaranteed to reflect any
# provider's real current pricing. Edit this table to keep it current; the
# exact dollar values are not load-bearing for correctness, only the
# structure and the arithmetic that consumes them are.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """Batch-tier price for one model, in USD per 1,000,000 tokens."""

    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, ModelPrice] = {
    # Anthropic
    "claude-opus-4-5": ModelPrice(input_per_mtok=7.50, output_per_mtok=37.50),
    "claude-sonnet-4-5": ModelPrice(input_per_mtok=1.50, output_per_mtok=7.50),
    "claude-3-5-haiku": ModelPrice(input_per_mtok=0.40, output_per_mtok=2.00),
    # OpenAI
    "gpt-4o": ModelPrice(input_per_mtok=1.25, output_per_mtok=5.00),
    "gpt-4o-mini": ModelPrice(input_per_mtok=0.075, output_per_mtok=0.30),
    # Google
    "gemini-2.5-pro": ModelPrice(input_per_mtok=0.625, output_per_mtok=5.00),
    "gemini-2.5-flash": ModelPrice(input_per_mtok=0.15, output_per_mtok=0.30),
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token.

    Single source of truth for the heuristic, matching
    ``polybatch.core.chunking._default_estimate_tokens``.
    """
    return len(text) // 4


@dataclass(frozen=True)
class CostEstimate:
    """Result of estimating tokens and cost for a batch of prompts."""

    model: str
    n_items: int
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    per_1k_items_cost: float


def estimate_cost(prompts: list[str], model: str, max_tokens: int) -> CostEstimate:
    """Estimate tokens and cost for a list of already-built prompt strings.

    input_tokens is the sum of estimate_tokens(prompt) over all prompts.
    output_tokens is n_items * max_tokens (the requested output budget, since
    without a real API call there is no actual output to measure).
    Raises ValueError if ``model`` is not in PRICES, listing the available
    model keys.
    """
    if model not in PRICES:
        available = ", ".join(sorted(PRICES))
        raise ValueError(
            f"unknown model {model!r}; available models: {available}"
        )

    price = PRICES[model]
    n_items = len(prompts)
    input_tokens = sum(estimate_tokens(p) for p in prompts)
    output_tokens = n_items * max_tokens

    input_cost = input_tokens * price.input_per_mtok / 1e6
    output_cost = output_tokens * price.output_per_mtok / 1e6
    total_cost = input_cost + output_cost
    per_1k_items_cost = (total_cost / n_items * 1000) if n_items else 0.0

    return CostEstimate(
        model=model,
        n_items=n_items,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        per_1k_items_cost=per_1k_items_cost,
    )


def estimate_cost_for_records(
    records: list[Record],
    model: str,
    max_tokens: int,
    task: TaskSpec | None = None,
) -> CostEstimate:
    """Convenience wrapper: build prompts from records via task.prompt_template.

    Mirrors how the orchestrator builds each prompt:
    ``task.prompt_template.format(order_id=record.order_id, text=record.text)``.
    """
    from polybatch.core.models import DEFAULT_TASK

    task = task or DEFAULT_TASK
    prompts = [
        task.prompt_template.format(order_id=r.order_id, text=r.text)
        for r in records
    ]
    return estimate_cost(prompts, model, max_tokens)


def format_estimate(est: CostEstimate) -> str:
    """Render a CostEstimate as an ASCII-only aligned report block."""
    lines: list[str] = []
    width = 60
    lines.append(f"Cost estimate -- model: {est.model}")
    lines.append("=" * width)
    lines.append(f"   Items                 : {est.n_items:>12,}")
    lines.append(
        f"   Input tokens          : {est.input_tokens:>12,}   ${est.input_cost:>8.4f}"
    )
    lines.append(
        f"   Output tokens         : {est.output_tokens:>12,}   ${est.output_cost:>8.4f}"
    )
    lines.append("-" * width)
    total_tokens = est.input_tokens + est.output_tokens
    lines.append(
        f"   TOTAL                 : {total_tokens:>12,}   ${est.total_cost:>8.4f}"
    )
    lines.append("")
    lines.append(
        f"   Per-1k-items average  : ${est.per_1k_items_cost:.4f} / 1k items"
    )
    return "\n".join(lines)
