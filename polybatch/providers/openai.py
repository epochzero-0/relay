"""OpenAI Batch API adapter (optional extra: ``pip install polybatch[openai]``).

Implements the :class:`~polybatch.providers.base.Provider` protocol against
OpenAI's asynchronous Batch API (``/v1/batches`` with a JSONL input file):

  submit()         build a JSONL of chat-completion requests, upload it with
                   ``files.create(purpose="batch")``, then ``batches.create``.
  poll()           ``batches.retrieve`` -> normalized JobStatus.
  fetch_results()  ``batches.retrieve`` for the output/error file ids, download
                   them with ``files.content``, parse the JSONL back to
                   BatchResult keyed by custom_id.

The ``openai`` SDK is imported lazily *inside* the methods so importing this
module never requires the SDK; the stdlib-only default (mock) path stays
import-safe. All per-request identity flows through ``custom_id`` (which OpenAI
echoes back verbatim in the result file), so ``fetch_results`` is crash-safe: a
fresh adapter can fetch a job submitted by a previous process and still map
every result back to its order_id.

Ported from the shape of ``legacy/submit-batches/submit_gpt_batches.py``.
"""

from __future__ import annotations

import json
from typing import Iterator

from polybatch.core.models import BatchResult, JobStatus, ProviderLimits, Request
from polybatch.providers.base import BatchTooLargeError, TransientSubmitError

#: Default model. EDIT THIS to whatever chat model you want to batch against.
DEFAULT_MODEL = "gpt-4o-mini"

#: OpenAI's hard ceiling is 50,000 requests per batch input file. The binding
#: constraint in practice is the per-tier enqueued-token limit, which polybatch
#: cannot see; if you hit token rejections, lower this so chunks stay small.
DEFAULT_MAX_ITEMS = 50_000

#: Registry metadata (consumed by the provider registry / CLI in checkpoint 4).
REGISTRY_NAME = "openai"
SDK_MODULE = "openai"
INSTALL_EXTRA = "openai"
API_KEY_ENV = "OPENAI_API_KEY"


class OpenAIBatchProvider:
    """Provider backed by the OpenAI Batch API. See module docstring."""

    #: Registry-facing metadata (also available as module constants above).
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
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        limits: ProviderLimits | None = None,
    ) -> None:
        self.name = REGISTRY_NAME
        self.model = model
        self.system = system
        self.temperature = temperature
        self.endpoint = endpoint
        self.completion_window = completion_window
        self.limits = (
            limits if limits is not None
            else ProviderLimits(max_items_per_batch=DEFAULT_MAX_ITEMS)
        )
        self._api_key = api_key
        self._client_obj = None  # lazily constructed SDK client

    # ----- SDK plumbing (lazy) -----------------------------------------

    def _sdk(self):
        """Import the openai SDK, or raise a clear, install-hint error."""
        try:
            import openai  # noqa: PLC0415 - lazy on purpose (optional extra).
        except ImportError as exc:  # pragma: no cover - trivial guard.
            raise ImportError(
                "the 'openai' package is required for --provider openai; "
                "install it with: pip install polybatch[openai]"
            ) from exc
        return openai

    def _client(self):
        if self._client_obj is None:
            openai = self._sdk()
            self._client_obj = openai.OpenAI(api_key=self._api_key)
        return self._client_obj

    def _map_submit_error(self, exc: Exception) -> Exception:
        """Translate an SDK exception into polybatch's submit taxonomy.

        Rate limits / connection / timeout / 5xx -> TransientSubmitError so the
        orchestrator retries with backoff. A 400 that reads like a size/token
        rejection -> BatchTooLargeError so the orchestrator shrinks the chunk.
        Anything else is returned unchanged (re-raised by the caller).
        """
        openai = self._sdk()
        transient = (
            getattr(openai, "RateLimitError", ()),
            getattr(openai, "APIConnectionError", ()),
            getattr(openai, "APITimeoutError", ()),
            getattr(openai, "InternalServerError", ()),
        )
        transient = tuple(t for t in transient if isinstance(t, type))
        if transient and isinstance(exc, transient):
            return TransientSubmitError(str(exc))
        bad_request = getattr(openai, "BadRequestError", None)
        if isinstance(bad_request, type) and isinstance(exc, bad_request):
            msg = str(exc).lower()
            if any(w in msg for w in ("too large", "limit", "exceed", "token")):
                return BatchTooLargeError(str(exc))
        return exc

    # ----- Provider protocol -------------------------------------------

    def submit(self, requests: list[Request]) -> str:
        """Upload a JSONL batch file and create the batch; return its id."""
        limit = self.limits.max_items_per_batch
        if limit is not None and len(requests) > limit:
            raise BatchTooLargeError(
                f"batch of {len(requests)} exceeds max_items_per_batch={limit}"
            )
        client = self._client()
        payload = self._build_jsonl(requests).encode("utf-8")
        try:
            uploaded = client.files.create(
                file=("polybatch_batch.jsonl", payload),
                purpose="batch",
            )
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint=self.endpoint,
                completion_window=self.completion_window,
            )
        except (BatchTooLargeError, TransientSubmitError):
            raise
        except Exception as exc:  # noqa: BLE001 - remap to taxonomy, then raise.
            raise self._map_submit_error(exc) from exc
        return batch.id

    def poll(self, job_id: str) -> JobStatus:
        """Retrieve the batch and normalize its status."""
        client = self._client()
        batch = client.batches.retrieve(job_id)
        status = batch.status
        counts = getattr(batch, "request_counts", None)
        total = getattr(counts, "total", 0) or 0
        completed = getattr(counts, "completed", 0) or 0
        failed = getattr(counts, "failed", 0) or 0

        # Map OpenAI's status vocabulary onto polybatch's terminal states.
        # completed -> "ended"; failed/expired/cancelled map straight through;
        # everything else (validating/in_progress/finalizing/cancelling) is a
        # non-terminal state name (deliberately NOT in JobStatus.TERMINAL_STATES).
        if status == "completed":
            state = "ended"
        elif status in ("failed", "expired", "cancelled"):
            state = status
        else:
            state = status  # non-terminal (validating/in_progress/finalizing/...)

        return JobStatus(
            state=state,
            succeeded=completed,
            errored=failed,
            processing=max(total - completed - failed, 0),
        )

    def fetch_results(self, job_id: str) -> Iterator[BatchResult]:
        """Download the output (and error) files and yield BatchResults."""
        client = self._client()
        batch = client.batches.retrieve(job_id)
        output_file_id = getattr(batch, "output_file_id", None)
        error_file_id = getattr(batch, "error_file_id", None)

        if output_file_id:
            for line in self._download_lines(client, output_file_id):
                result = self._parse_output_line(line, errored=False)
                if result is not None:
                    yield result
        if error_file_id:
            for line in self._download_lines(client, error_file_id):
                result = self._parse_output_line(line, errored=True)
                if result is not None:
                    yield result

    # ----- helpers ------------------------------------------------------

    def _build_jsonl(self, requests: list[Request]) -> str:
        """Render requests as OpenAI batch JSONL (one task object per line)."""
        lines: list[str] = []
        for req in requests:
            messages: list[dict] = []
            if self.system:
                messages.append({"role": "system", "content": self.system})
            messages.append({"role": "user", "content": req.prompt})
            body: dict = {
                "model": self.model,
                "messages": messages,
                "max_tokens": req.max_tokens,
            }
            if self.temperature is not None:
                body["temperature"] = self.temperature
            task = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": self.endpoint,
                "body": body,
            }
            lines.append(json.dumps(task))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _download_lines(client, file_id: str) -> list[str]:
        """Return the non-empty text lines of a downloaded batch file."""
        content = client.files.content(file_id)
        text = getattr(content, "text", None)
        if text is None:  # some SDK versions return raw bytes.
            raw = content.read() if hasattr(content, "read") else content
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return [ln for ln in text.splitlines() if ln.strip()]

    @staticmethod
    def _parse_output_line(line: str, *, errored: bool) -> BatchResult | None:
        """Parse one JSONL result line into a BatchResult (or None if junk)."""
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return None
        custom_id = entry.get("custom_id") or ""
        if not custom_id:
            return None
        response = entry.get("response") or {}
        status_code = response.get("status_code", 0)
        if errored or (status_code and status_code != 200):
            err = entry.get("error")
            detail = err if isinstance(err, str) else json.dumps(err) if err else f"HTTP {status_code}"
            return BatchResult(custom_id=custom_id, ok=False, error=detail)
        body = response.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            return BatchResult(custom_id=custom_id, ok=False, error="no choices in response")
        text = (choices[0].get("message") or {}).get("content", "")
        return BatchResult(custom_id=custom_id, ok=True, text=text)
