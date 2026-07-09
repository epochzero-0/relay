"""Google Gemini Batch adapter (extra: ``pip install polybatch[google]``).

Implements the :class:`~polybatch.providers.base.Provider` protocol against the
Google Gen AI SDK's batch API (``google-genai``, ``from google import genai``).

RISK NOTE (read before trusting this against a live key)
--------------------------------------------------------
This is the one adapter with **no legacy reference script for crash-safe
keying** and it was written without an API key to test against. The legacy
``submit_gemini_batches.py`` used *inlined* requests whose responses come back
positionally in ``job.dest.inlined_responses`` with **no per-item id** — that
cannot satisfy polybatch's contract that ``fetch_results(job_id)`` maps every
result to its ``custom_id`` after a crash/resume (a fresh adapter has lost the
original ordering).

So this adapter instead uses the **file-based** batch flow: it uploads a JSONL
file where every line carries a ``"key"`` (= the Request.custom_id) alongside
its ``"request"`` body, and reads results from the output file, where each line
echoes that ``"key"``. That makes fetch crash-safe. The exact JSONL field
casing (camelCase ``generationConfig`` / ``maxOutputTokens`` per the REST batch
format) and the ``job.dest.file_name`` download path are the parts most likely
to need a small tweak once tested against a real key. Parsing is written
defensively (``.get`` chains, broad attribute fallbacks) to degrade to a clean
per-item failure rather than crash the run.

  submit()         write JSONL (key + request per line), upload via files, then
                   ``batches.create(model=, src=<uploaded file name>)``.
  poll()           ``batches.get(name=)`` -> map ``JOB_STATE_*`` to JobStatus.
  fetch_results()  download the result file, parse each ``{"key","response"}``
                   line back to a BatchResult.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator

from polybatch.core.models import BatchResult, JobStatus, ProviderLimits, Request
from polybatch.providers.base import BatchTooLargeError, TransientSubmitError

#: Default model. EDIT THIS to whatever Gemini model you want to batch against.
DEFAULT_MODEL = "gemini-2.5-flash"

#: Conservative default item ceiling; adjust to your quota. Gemini's documented
#: batch limits are large but quota/token-bound, which polybatch cannot see.
DEFAULT_MAX_ITEMS = 50_000

#: Registry metadata (consumed by the provider registry / CLI in checkpoint 4).
#: The SDK also honours GEMINI_API_KEY; the registry may fall back to it.
REGISTRY_NAME = "google"
SDK_MODULE = "google.genai"
INSTALL_EXTRA = "google"
API_KEY_ENV = "GOOGLE_API_KEY"

_TERMINAL_SUCCESS = frozenset({"JOB_STATE_SUCCEEDED", "SUCCEEDED"})
_TERMINAL_FAIL_MARKERS = ("FAIL", "CANCEL", "EXPIRE")


class GoogleBatchProvider:
    """Provider backed by the Gemini file-based batch API. See module docstring."""

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
            from google import genai  # noqa: PLC0415 - lazy (optional extra).
        except ImportError as exc:  # pragma: no cover - trivial guard.
            raise ImportError(
                "the 'google-genai' package is required for --provider google; "
                "install it with: pip install polybatch[google]"
            ) from exc
        return genai

    def _client(self):
        if self._client_obj is None:
            genai = self._sdk()
            if self._api_key is not None:
                self._client_obj = genai.Client(api_key=self._api_key)
            else:
                # Let the SDK read GOOGLE_API_KEY / GEMINI_API_KEY from env.
                self._client_obj = genai.Client()
        return self._client_obj

    def _map_submit_error(self, exc: Exception) -> Exception:
        """Translate a google-genai exception into polybatch's taxonomy."""
        try:
            from google.genai import errors  # noqa: PLC0415
        except ImportError:  # pragma: no cover - errors module always ships.
            errors = None
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        msg = str(exc).lower()
        if code == 429 or "rate limit" in msg or "resource_exhausted" in msg:
            return TransientSubmitError(str(exc))
        if errors is not None:
            server = getattr(errors, "ServerError", None)
            if isinstance(server, type) and isinstance(exc, server):
                return TransientSubmitError(str(exc))
            client_err = getattr(errors, "ClientError", None)
            if isinstance(client_err, type) and isinstance(exc, client_err):
                if any(w in msg for w in ("too large", "exceed", "size", "token")):
                    return BatchTooLargeError(str(exc))
        if isinstance(code, int) and code >= 500:
            return TransientSubmitError(str(exc))
        return exc

    # ----- Provider protocol -------------------------------------------

    def submit(self, requests: list[Request]) -> str:
        """Upload a keyed JSONL batch file and create the batch; return its name."""
        limit = self.limits.max_items_per_batch
        if limit is not None and len(requests) > limit:
            raise BatchTooLargeError(
                f"batch of {len(requests)} exceeds max_items_per_batch={limit}"
            )
        client = self._client()
        payload = self._build_jsonl(requests)

        tmp = Path(tempfile.mkdtemp(prefix="polybatch_google_")) / "batch.jsonl"
        try:
            tmp.write_text(payload, encoding="utf-8")
            try:
                uploaded = client.files.upload(
                    file=str(tmp),
                    config={"mime_type": "application/jsonl"},
                )
                src = getattr(uploaded, "name", uploaded)
                job = client.batches.create(model=self.model, src=src)
            except (BatchTooLargeError, TransientSubmitError):
                raise
            except Exception as exc:  # noqa: BLE001 - remap to taxonomy.
                raise self._map_submit_error(exc) from exc
        finally:
            try:
                tmp.unlink()
                tmp.parent.rmdir()
            except OSError:  # pragma: no cover - best-effort cleanup.
                pass
        return job.name

    def poll(self, job_id: str) -> JobStatus:
        """Fetch the batch job and map its JOB_STATE_* onto a JobStatus."""
        client = self._client()
        job = client.batches.get(name=job_id)
        state_obj = getattr(job, "state", None)
        raw = getattr(state_obj, "name", None) or str(state_obj)

        if raw in _TERMINAL_SUCCESS:
            state = "ended"
        elif any(marker in raw for marker in _TERMINAL_FAIL_MARKERS):
            # JOB_STATE_FAILED / CANCELLED / EXPIRED -> a terminal, non-"ended"
            # state so the orchestrator marks the chunk failed and re-sends.
            state = "failed"
        else:
            state = "running"  # PENDING / RUNNING / etc. -> non-terminal.
        return JobStatus(state=state)

    def fetch_results(self, job_id: str) -> Iterator[BatchResult]:
        """Download the result file and yield one BatchResult per keyed line."""
        client = self._client()
        job = client.batches.get(name=job_id)
        dest = getattr(job, "dest", None)

        file_name = getattr(dest, "file_name", None) if dest else None
        if file_name:
            raw = client.files.download(file=file_name)
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            for line in text.splitlines():
                if not line.strip():
                    continue
                result = self._parse_result_line(line)
                if result is not None:
                    yield result
            return

        # Fallback: inlined responses (same-process only; no keys -> best effort).
        inlined = getattr(dest, "inlined_responses", None) if dest else None
        if inlined:
            raise RuntimeError(
                "google batch returned inlined_responses without keys; "
                "polybatch requires the file-based (keyed) result path"
            )

    # ----- helpers ------------------------------------------------------

    def _build_jsonl(self, requests: list[Request]) -> str:
        """Render requests as keyed Gemini batch JSONL (one object per line)."""
        lines: list[str] = []
        for req in requests:
            body: dict = {
                "contents": [
                    {"role": "user", "parts": [{"text": req.prompt}]}
                ],
                "generationConfig": {"maxOutputTokens": req.max_tokens},
            }
            if self.temperature is not None:
                body["generationConfig"]["temperature"] = self.temperature
            if self.system:
                body["systemInstruction"] = {"parts": [{"text": self.system}]}
            lines.append(json.dumps({"key": req.custom_id, "request": body}))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _parse_result_line(line: str) -> BatchResult | None:
        """Parse one keyed JSONL result line into a BatchResult (or None)."""
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return None
        custom_id = entry.get("key") or entry.get("custom_id") or ""
        if not custom_id:
            return None
        if entry.get("error"):
            return BatchResult(
                custom_id=custom_id, ok=False, error=json.dumps(entry["error"])
            )
        response = entry.get("response") or {}
        candidates = response.get("candidates") or []
        if not candidates:
            return BatchResult(custom_id=custom_id, ok=False, error="no candidates")
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text:
            return BatchResult(custom_id=custom_id, ok=False, error="empty response text")
        return BatchResult(custom_id=custom_id, ok=True, text=text)
