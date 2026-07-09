"""JSON tracker state machine for resumable runs.

Ported from the legacy per-experiment tracker. The tracker records, per chunk,
what has happened so far so that an interrupted run can resume without
re-submitting or re-downloading completed work. Every transition saves
immediately (atomic replace) so a crash at any point leaves a consistent file.

State lifecycle for a chunk key:
    not_submitted (implicit; key absent)
        -> submitted(job_id)
            -> done
            -> partial_failed / failed
        -> submit_failed

decide() collapses that state into the next action the orchestrator should
take: skip, resume, or submit.

This module is intentionally dependency-free (stdlib only, no other polybatch
imports) so it can be reused and tested in isolation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Chunk states. NOT_SUBMITTED is implicit (the key is simply absent).
NOT_SUBMITTED = "not_submitted"
SUBMITTED = "submitted"
DONE = "done"
PARTIAL_FAILED = "partial_failed"
FAILED = "failed"
SUBMIT_FAILED = "submit_failed"

#: On-disk schema version, in case the layout changes in a later phase.
VERSION = 1


class Tracker:
    """Per-run JSON state file with crash-safe, save-on-every-transition writes.

    The in-memory shape mirrors the file:
        {"version": 1, "chunks": {key: {"status": str, "job_id": str | None,
                                        ...extra}}}
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._chunks: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._chunks = dict(data.get("chunks", {}))

    # ----- persistence -------------------------------------------------

    def save(self) -> None:
        """Atomically write the current state to disk.

        Writes to a sibling .tmp file then os.replace() over the target, so a
        reader never observes a half-written file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        payload = {"version": VERSION, "chunks": self._chunks}
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ----- reads -------------------------------------------------------

    def status(self, key: str) -> str:
        """Return the chunk's status, or NOT_SUBMITTED if it is unknown."""
        entry = self._chunks.get(key)
        if entry is None:
            return NOT_SUBMITTED
        return str(entry.get("status", NOT_SUBMITTED))

    def job_id(self, key: str) -> str | None:
        """Return the saved job id for a chunk, or None if there is none."""
        entry = self._chunks.get(key)
        if entry is None:
            return None
        return entry.get("job_id")

    def summary(self) -> dict[str, str]:
        """Return a key -> status mapping for every known chunk."""
        return {key: self.status(key) for key in self._chunks}

    # ----- transitions -------------------------------------------------

    def _set(self, key: str, status: str, **extra: Any) -> None:
        entry: dict[str, Any] = {"status": status, "job_id": None}
        entry.update(extra)
        self._chunks[key] = entry
        self.save()

    def mark_submitted(self, key: str, job_id: str) -> None:
        """Record that a chunk was submitted and got an opaque job id."""
        self._set(key, SUBMITTED, job_id=job_id)

    def mark_done(self, key: str, *, succeeded: int, failed: int) -> None:
        """Record a completed chunk with its success/failure counts."""
        entry = self._chunks.get(key, {})
        self._set(
            key,
            DONE,
            job_id=entry.get("job_id"),
            succeeded=succeeded,
            failed=failed,
        )

    def mark_partial_failed(self, key: str, *, succeeded: int, expected: int) -> None:
        """Record a chunk that returned fewer results than expected.

        Used by the Phase 2 coverage re-send loop; decide() treats it as
        submit for now.
        """
        entry = self._chunks.get(key, {})
        self._set(
            key,
            PARTIAL_FAILED,
            job_id=entry.get("job_id"),
            succeeded=succeeded,
            expected=expected,
        )

    def mark_failed(self, key: str, reason: str) -> None:
        """Record that a chunk's job reached a non-ended terminal state."""
        entry = self._chunks.get(key, {})
        self._set(key, FAILED, job_id=entry.get("job_id"), reason=reason)

    def mark_submit_failed(self, key: str, reason: str) -> None:
        """Record that submission of a chunk failed after all retries."""
        self._set(key, SUBMIT_FAILED, job_id=None, reason=reason)

    def reset(self, key: str) -> None:
        """Forget a chunk entirely (back to the implicit not_submitted state)."""
        if key in self._chunks:
            del self._chunks[key]
            self.save()

    # ----- resume decision ---------------------------------------------

    def decide(self, key: str) -> tuple[str, str | None]:
        """Collapse a chunk's stored state into the next action to take.

        Returns:
            ("skip", None)        the chunk is done; nothing to do.
            ("resume", job_id)    already submitted with a live job id; poll it.
            ("submit", None)      submit fresh. Covers not_submitted plus the
                                  Phase 2 re-send states (partial_failed,
                                  failed, submit_failed), which are simply
                                  re-submitted for now.
        """
        state = self.status(key)
        if state == DONE:
            return ("skip", None)
        if state == SUBMITTED:
            jid = self.job_id(key)
            if jid:
                return ("resume", jid)
            return ("submit", None)
        return ("submit", None)
