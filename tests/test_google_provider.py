"""Google Gemini batch adapter tests against a fully faked ``google.genai`` SDK.

The real ``google-genai`` package is not present in the test env and must never
be required. A fake ``google`` namespace with a ``genai`` submodule and a
``google.genai.errors`` module is installed into ``sys.modules``. Covers keyed
JSONL building, JOB_STATE_* normalization, keyed result-file parsing, and
submit-error -> taxonomy mapping.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from polybatch.core.models import ProviderLimits, Request
from polybatch.providers.base import BatchTooLargeError, TransientSubmitError
from polybatch.providers.google import GoogleBatchProvider


def _make_fake_google(*, job=None, result_bytes=b""):
    """Install a fake google/genai/errors trio; return (google_mod, state)."""
    errors_mod = types.ModuleType("google.genai.errors")

    class APIError(Exception):
        def __init__(self, message="", code=None):
            super().__init__(message)
            self.code = code

    class ClientError(APIError):
        pass

    class ServerError(APIError):
        pass

    errors_mod.APIError = APIError
    errors_mod.ClientError = ClientError
    errors_mod.ServerError = ServerError

    genai_mod = types.ModuleType("google.genai")
    genai_mod.errors = errors_mod
    captured: dict = {"raise": None}

    class _Files:
        def upload(self, *, file, config=None):
            captured["upload_config"] = config
            with open(file, "r", encoding="utf-8") as handle:
                captured["payload"] = handle.read()
            return types.SimpleNamespace(name="files/abc123")

        def download(self, *, file):
            captured["download_file"] = file
            return result_bytes

    class _Batches:
        def create(self, *, model, src):
            captured["create"] = {"model": model, "src": src}
            if captured["raise"] is not None:
                raise captured["raise"]
            return types.SimpleNamespace(name="batches/xyz")

        def get(self, *, name):
            return job

    class _Client:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.files = _Files()
            self.batches = _Batches()

    genai_mod.Client = _Client

    google_mod = types.ModuleType("google")
    google_mod.genai = genai_mod

    google_mod._captured = captured
    google_mod._errors = errors_mod
    return google_mod


@pytest.fixture
def fake_google(monkeypatch):
    def _install(**kwargs):
        google_mod = _make_fake_google(**kwargs)
        monkeypatch.setitem(sys.modules, "google", google_mod)
        monkeypatch.setitem(sys.modules, "google.genai", google_mod.genai)
        monkeypatch.setitem(sys.modules, "google.genai.errors", google_mod.genai.errors)
        return google_mod
    return _install


def _reqs(n=2):
    return [
        Request(custom_id=f"run_1_item_rec_{i:04d}", order_id=f"rec_{i:04d}",
                prompt=f"prompt {i}", max_tokens=24)
        for i in range(1, n + 1)
    ]


def _job(state_name, *, file_name="files/out", inlined=None):
    dest = types.SimpleNamespace(file_name=file_name, inlined_responses=inlined)
    return types.SimpleNamespace(state=types.SimpleNamespace(name=state_name),
                                 dest=dest)


def test_submit_builds_keyed_jsonl_and_returns_name(fake_google):
    mod = fake_google(job=None)
    provider = GoogleBatchProvider(model="gemini-x", system="be brief",
                                   temperature=0.2)
    name = provider.submit(_reqs(2))
    assert name == "batches/xyz"
    assert mod._captured["create"]["model"] == "gemini-x"
    assert mod._captured["create"]["src"] == "files/abc123"

    lines = [json.loads(ln) for ln in mod._captured["payload"].splitlines() if ln.strip()]
    assert [ln["key"] for ln in lines] == [
        "run_1_item_rec_0001", "run_1_item_rec_0002"]
    body = lines[0]["request"]
    assert body["contents"][0]["parts"][0]["text"] == "prompt 1"
    assert body["contents"][0]["role"] == "user"
    assert body["generationConfig"]["maxOutputTokens"] == 24
    assert body["generationConfig"]["temperature"] == 0.2
    assert body["systemInstruction"]["parts"][0]["text"] == "be brief"


def test_submit_omits_system_and_temperature_when_unset(fake_google):
    mod = fake_google(job=None)
    GoogleBatchProvider().submit(_reqs(1))
    body = json.loads(mod._captured["payload"].splitlines()[0])["request"]
    assert "systemInstruction" not in body
    assert "temperature" not in body["generationConfig"]


def test_submit_rejects_oversize_batch_locally(fake_google):
    fake_google(job=None)
    provider = GoogleBatchProvider(limits=ProviderLimits(max_items_per_batch=1))
    with pytest.raises(BatchTooLargeError):
        provider.submit(_reqs(2))


@pytest.mark.parametrize("state_name,expected,terminal", [
    ("JOB_STATE_SUCCEEDED", "ended", True),
    ("JOB_STATE_FAILED", "failed", True),
    ("JOB_STATE_CANCELLED", "failed", True),
    ("JOB_STATE_EXPIRED", "failed", True),
    ("JOB_STATE_RUNNING", "running", False),
    ("JOB_STATE_PENDING", "running", False),
])
def test_poll_maps_job_state(fake_google, state_name, expected, terminal):
    fake_google(job=_job(state_name))
    st = GoogleBatchProvider().poll("batches/xyz")
    assert st.state == expected
    assert st.is_terminal is terminal


def test_fetch_parses_keyed_result_file(fake_google):
    good = {"key": "run_1_item_rec_0001",
            "response": {"candidates": [
                {"content": {"parts": [{"text": "rec_0001,4,8"}]}}]}}
    empty = {"key": "run_1_item_rec_0002",
             "response": {"candidates": [{"content": {"parts": [{"text": ""}]}}]}}
    errored = {"key": "run_1_item_rec_0003",
               "error": {"code": 500, "message": "internal"}}
    payload = ("\n".join(json.dumps(x) for x in (good, empty, errored)) + "\n").encode("utf-8")
    fake_google(job=_job("JOB_STATE_SUCCEEDED"), result_bytes=payload)
    out = {r.custom_id: r for r in GoogleBatchProvider().fetch_results("batches/xyz")}

    assert out["run_1_item_rec_0001"].ok is True
    assert out["run_1_item_rec_0001"].text == "rec_0001,4,8"
    assert out["run_1_item_rec_0002"].ok is False  # empty text -> failure
    assert out["run_1_item_rec_0003"].ok is False
    assert "internal" in out["run_1_item_rec_0003"].error


def test_fetch_rejects_inlined_without_keys(fake_google):
    job = types.SimpleNamespace(
        state=types.SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
        dest=types.SimpleNamespace(file_name=None, inlined_responses=[object()]),
    )
    fake_google(job=job)
    with pytest.raises(RuntimeError):
        list(GoogleBatchProvider().fetch_results("batches/xyz"))


def test_submit_maps_rate_limit_to_transient(fake_google):
    mod = fake_google(job=None)
    mod._captured["raise"] = mod._errors.ClientError("rate limited", code=429)
    with pytest.raises(TransientSubmitError):
        GoogleBatchProvider().submit(_reqs(1))


def test_submit_maps_server_error_to_transient(fake_google):
    mod = fake_google(job=None)
    mod._captured["raise"] = mod._errors.ServerError("boom", code=500)
    with pytest.raises(TransientSubmitError):
        GoogleBatchProvider().submit(_reqs(1))


def test_submit_maps_size_client_error_to_too_large(fake_google):
    mod = fake_google(job=None)
    mod._captured["raise"] = mod._errors.ClientError("payload size exceeds limit", code=400)
    with pytest.raises(BatchTooLargeError):
        GoogleBatchProvider().submit(_reqs(1))
