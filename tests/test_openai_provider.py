"""OpenAI batch adapter tests against a fully faked ``openai`` SDK.

No network, no real SDK: a fake ``openai`` module is installed into
``sys.modules`` so these tests RUN even though the real package may or may not
be present. They exercise request JSONL building, status normalization, result
parsing (output + error files), and submit-error -> taxonomy mapping.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from polybatch.core.models import ProviderLimits, Request
from polybatch.providers.base import BatchTooLargeError, TransientSubmitError
from polybatch.providers.openai import OpenAIBatchProvider


# ----- fake openai SDK -------------------------------------------------------


class _FakeErr(Exception):
    pass


def _make_fake_openai(*, batch=None, output_text="", error_text=""):
    """Build a fake ``openai`` module object with tunable behavior.

    Set ``mod._raise`` to an exception instance (built from this same module's
    exception classes) to make ``batches.create`` raise it.
    """
    mod = types.ModuleType("openai")

    # exception hierarchy the adapter maps against
    for name in ("RateLimitError", "APIConnectionError", "APITimeoutError",
                 "InternalServerError", "BadRequestError"):
        setattr(mod, name, type(name, (_FakeErr,), {}))

    captured: dict = {}
    mod._raise = None

    class _Files:
        def create(self, *, file, purpose):
            captured["file"] = file
            captured["purpose"] = purpose
            return types.SimpleNamespace(id="file-abc")

        def content(self, file_id):
            text = output_text if file_id == "out-1" else error_text
            return types.SimpleNamespace(text=text)

    class _Batches:
        def create(self, **kwargs):
            captured["batch_create"] = kwargs
            if mod._raise is not None:
                raise mod._raise
            return types.SimpleNamespace(id="batch-1")

        def retrieve(self, job_id):
            return batch

    class _Client:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.files = _Files()
            self.batches = _Batches()

    mod.OpenAI = _Client
    mod._captured = captured
    return mod


@pytest.fixture
def fake_openai(monkeypatch):
    def _install(**kwargs):
        mod = _make_fake_openai(**kwargs)
        monkeypatch.setitem(sys.modules, "openai", mod)
        return mod
    return _install


def _reqs(n=2):
    return [
        Request(custom_id=f"run_1_item_rec_{i:04d}", order_id=f"rec_{i:04d}",
                prompt=f"prompt {i}", max_tokens=16)
        for i in range(1, n + 1)
    ]


# ----- submit / request building --------------------------------------------


def test_submit_builds_keyed_jsonl_and_returns_id(fake_openai):
    mod = fake_openai(batch=None)
    provider = OpenAIBatchProvider(model="gpt-4o-mini", system="be terse",
                                   temperature=0.5)
    job_id = provider.submit(_reqs(2))

    assert job_id == "batch-1"
    assert mod._captured["purpose"] == "batch"
    name, payload = mod._captured["file"]
    assert name.endswith(".jsonl")
    lines = [json.loads(ln) for ln in payload.decode("utf-8").splitlines() if ln.strip()]
    assert [ln["custom_id"] for ln in lines] == [
        "run_1_item_rec_0001", "run_1_item_rec_0002"]
    body = lines[0]["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["max_tokens"] == 16
    assert body["temperature"] == 0.5
    assert body["messages"][0] == {"role": "system", "content": "be terse"}
    assert body["messages"][1] == {"role": "user", "content": "prompt 1"}
    assert lines[0]["url"] == "/v1/chat/completions"


def test_submit_omits_system_and_temperature_when_unset(fake_openai):
    mod = fake_openai(batch=None)
    provider = OpenAIBatchProvider()
    provider.submit(_reqs(1))
    _, payload = mod._captured["file"]
    body = json.loads(payload.decode("utf-8").splitlines()[0])["body"]
    assert all(m["role"] != "system" for m in body["messages"])
    assert "temperature" not in body


def test_submit_rejects_oversize_batch_locally(fake_openai):
    fake_openai(batch=None)
    provider = OpenAIBatchProvider(limits=ProviderLimits(max_items_per_batch=1))
    with pytest.raises(BatchTooLargeError):
        provider.submit(_reqs(2))


# ----- poll ------------------------------------------------------------------


def _batch(status, total=2, completed=2, failed=0, out="out-1", err=None):
    return types.SimpleNamespace(
        status=status,
        request_counts=types.SimpleNamespace(total=total, completed=completed,
                                             failed=failed),
        output_file_id=out,
        error_file_id=err,
    )


@pytest.mark.parametrize("status,expected,terminal", [
    ("completed", "ended", True),
    ("failed", "failed", True),
    ("expired", "expired", True),
    ("cancelled", "cancelled", True),
    ("in_progress", "in_progress", False),
    ("validating", "validating", False),
    ("finalizing", "finalizing", False),
])
def test_poll_maps_status(fake_openai, status, expected, terminal):
    fake_openai(batch=_batch(status, total=5, completed=3, failed=1))
    provider = OpenAIBatchProvider()
    st = provider.poll("batch-1")
    assert st.state == expected
    assert st.is_terminal is terminal
    assert st.succeeded == 3
    assert st.errored == 1
    assert st.processing == 1


# ----- fetch_results ---------------------------------------------------------


def test_fetch_parses_output_and_error_files(fake_openai):
    good = {"custom_id": "run_1_item_rec_0001",
            "response": {"status_code": 200,
                         "body": {"choices": [{"message": {"content": "rec_0001,7,3"}}]}}}
    bad_http = {"custom_id": "run_1_item_rec_0002",
                "response": {"status_code": 500, "body": {}}}
    err_line = {"custom_id": "run_1_item_rec_0003",
                "error": "model overloaded"}
    fake_openai(
        batch=_batch("completed", out="out-1", err="err-1"),
        output_text="\n".join(json.dumps(x) for x in (good, bad_http)) + "\n",
        error_text=json.dumps(err_line) + "\n",
    )
    provider = OpenAIBatchProvider()
    results = list(provider.fetch_results("batch-1"))
    by_id = {r.custom_id: r for r in results}

    assert by_id["run_1_item_rec_0001"].ok is True
    assert by_id["run_1_item_rec_0001"].text == "rec_0001,7,3"
    assert by_id["run_1_item_rec_0002"].ok is False
    assert "500" in by_id["run_1_item_rec_0002"].error
    assert by_id["run_1_item_rec_0003"].ok is False
    assert "overloaded" in by_id["run_1_item_rec_0003"].error


def test_fetch_handles_missing_output_file(fake_openai):
    fake_openai(batch=_batch("completed", out=None, err=None))
    provider = OpenAIBatchProvider()
    assert list(provider.fetch_results("batch-1")) == []


# ----- error mapping ---------------------------------------------------------


def test_submit_maps_rate_limit_to_transient(fake_openai):
    mod = fake_openai(batch=None)
    mod._raise = mod.RateLimitError("429 slow down")
    provider = OpenAIBatchProvider()
    with pytest.raises(TransientSubmitError):
        provider.submit(_reqs(1))


def test_submit_maps_bad_request_size_to_too_large(fake_openai):
    mod = fake_openai(batch=None)
    mod._raise = mod.BadRequestError("batch exceeds enqueued token limit")
    provider = OpenAIBatchProvider()
    with pytest.raises(BatchTooLargeError):
        provider.submit(_reqs(1))
