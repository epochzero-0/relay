"""Anthropic Message Batches adapter (extra: ``pip install relay[anthropic]``).

Implements the :class:`~relay.providers.base.Provider` protocol against
Anthropic's Message Batches API:

  submit()         ``messages.batches.create(requests=[...])`` where each item
                   carries a ``custom_id`` and per-message ``params``.
  poll()           ``messages.batches.retrieve`` -> normalized JobStatus. The
                   only terminal ``processing_status`` is ``"ended"``.
  fetch_results()  stream ``messages.batches.results(id)``; each entry has a
                   ``custom_id`` and a ``result`` whose ``.type`` is one of
                   succeeded / errored / canceled / expired.

The ``anthropic`` SDK is imported lazily inside the methods so importing this
module never requires the SDK. Identity flows through ``custom_id`` (echoed
back on every result entry), so ``fetch_results`` is crash-safe.

Ported from the shape of ``legacy/submit-batches/submit_claude_batches.py`` and
``legacy/check-batches/check_claude_cost.py``.
"""

from __future__ import annotations

from typing import Iterator

from relay.core.models import BatchResult, JobStatus, ProviderLimits, Request
from relay.providers.base import BatchTooLargeError, TransientSubmitError

#: Default model. EDIT THIS to whatever Claude model you want to batch against.
DEFAULT_MODEL = "claude-3-5-haiku-latest"

#: Anthropic accepts up to 100,000 requests (or 256 MB) per Message Batch.
DEFAULT_MAX_ITEMS = 100_000

#: Registry metadata (consumed by the provider registry / CLI in checkpoint 4).
REGISTRY_NAME = "anthropic"
SDK_MODULE = "anthropic"
INSTALL_EXTRA = "anthropic"
API_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicBatchProvider:
    """Provider backed by the Anthropic Message Batches API."""

    registry_name = REGISTRY_NAME
    sdk_module = SDK_MODULE
    install_extra = INSTALL_EXTRA
    api_key_env = API_KEY_ENV

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        limits: ProviderLimits | None = None,
    ) -> None:
        self.name = REGISTRY_NAME
        self.model = model
        self.system = system
        self.temperature = temperature
        self.limits = (
            limits if limits is not None
            else ProviderLimits(max_items_per_batch=DEFAULT_MAX_ITEMS)
        )
        self._api_key = api_key
        self._client_obj = None

    # ----- SDK plumbing (lazy) -----------------------------------------

    def _sdk(self):
        try:
            import anthropic  # noqa: PLC0415 - lazy on purpose (optional extra).
        except ImportError as exc:  # pragma: no cover - trivial guard.
            raise ImportError(
                "the 'anthropic' package is required for --provider anthropic; "
                "install it with: pip install relay[anthropic]"
            ) from exc
        return anthropic

    def _client(self):
        if self._client_obj is None:
            anthropic = self._sdk()
            self._client_obj = anthropic.Anthropic(api_key=self._api_key)
        return self._client_obj

    def _map_submit_error(self, exc: Exception) -> Exception:
        """Translate an SDK exception into relay's submit taxonomy."""
        anthropic = self._sdk()
        transient = (
            getattr(anthropic, "RateLimitError", ()),
            getattr(anthropic, "APIConnectionError", ()),
            getattr(anthropic, "APITimeoutError", ()),
            getattr(anthropic, "InternalServerError", ()),
        )
        transient = tuple(t for t in transient if isinstance(t, type))
        if transient and isinstance(exc, transient):
            return TransientSubmitError(str(exc))
        bad_request = getattr(anthropic, "BadRequestError", None)
        if isinstance(bad_request, type) and isinstance(exc, bad_request):
            msg = str(exc).lower()
            if any(w in msg for w in ("too large", "limit", "exceed", "token")):
                return BatchTooLargeError(str(exc))
        return exc

    # ----- Provider protocol -------------------------------------------

    def submit(self, requests: list[Request]) -> str:
        """Create a Message Batch; return its id."""
        limit = self.limits.max_items_per_batch
        if limit is not None and len(requests) > limit:
            raise BatchTooLargeError(
                f"batch of {len(requests)} exceeds max_items_per_batch={limit}"
            )
        client = self._client()
        payload = [self._build_request(req) for req in requests]
        try:
            batch = client.messages.batches.create(requests=payload)
        except (BatchTooLargeError, TransientSubmitError):
            raise
        except Exception as exc:  # noqa: BLE001 - remap to taxonomy, then raise.
            raise self._map_submit_error(exc) from exc
        return batch.id

    def poll(self, job_id: str) -> JobStatus:
        """Retrieve the batch and normalize its status."""
        client = self._client()
        batch = client.messages.batches.retrieve(job_id)
        status = getattr(batch, "processing_status", None)
        counts = getattr(batch, "request_counts", None)
        succeeded = getattr(counts, "succeeded", 0) or 0
        errored = getattr(counts, "errored", 0) or 0
        canceled = getattr(counts, "canceled", 0) or 0
        expired = getattr(counts, "expired", 0) or 0
        processing = getattr(counts, "processing", 0) or 0

        # Anthropic's only terminal processing_status is "ended". "in_progress"
        # and "canceling" are non-terminal. canceled/expired requests are folded
        # into the errored count so the orchestrator's coverage loop re-sends
        # any items that did not succeed.
        state = "ended" if status == "ended" else (status or "in_progress")
        return JobStatus(
            state=state,
            succeeded=succeeded,
            errored=errored + canceled + expired,
            processing=processing,
        )

    def fetch_results(self, job_id: str) -> Iterator[BatchResult]:
        """Stream the batch results, yielding one BatchResult per entry."""
        client = self._client()
        for entry in client.messages.batches.results(job_id):
            custom_id = getattr(entry, "custom_id", "") or ""
            if not custom_id:
                continue
            result = getattr(entry, "result", None)
            rtype = getattr(result, "type", None)
            if rtype == "succeeded":
                text = self._extract_text(result)
                yield BatchResult(custom_id=custom_id, ok=True, text=text)
            elif rtype == "errored":
                err = getattr(result, "error", None)
                yield BatchResult(
                    custom_id=custom_id, ok=False, error=str(err) if err else "errored"
                )
            else:
                # canceled / expired / anything unexpected -> non-ok.
                yield BatchResult(
                    custom_id=custom_id, ok=False, error=str(rtype) if rtype else "unknown"
                )

    # ----- helpers ------------------------------------------------------

    def _build_request(self, req: Request) -> dict:
        """Build one Message Batch request dict for a single Request."""
        params: dict = {
            "model": self.model,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if self.system:
            params["system"] = self.system
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return {"custom_id": req.custom_id, "params": params}

    @staticmethod
    def _extract_text(result) -> str:
        """Concatenate the ``.text`` of every text content block in a message."""
        message = getattr(result, "message", None)
        content = getattr(message, "content", None) or []
        text = ""
        for block in content:
            block_text = getattr(block, "text", None)
            if block_text:
                text += block_text
        return text
