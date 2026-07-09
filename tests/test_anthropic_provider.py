"""Anthropic Message Batches adapter tests against a fully faked SDK.

A fake ``anthropic`` module is installed into ``sys.modules`` (the real package
is not present in the test env and must never be required). Covers request
building, status normalization, result streaming (succeeded / errored /
expired), and submit-error -> taxonomy mapping.
"""

from __future__ import annotations

import sys
import types

import pytest

from polybatch.core.models import ProviderLimits, Request
from polybatch.providers.base import BatchTooLargeError, TransientSubmitError
from polybatch.providers.anthropic import AnthropicBatchProvider


class _FakeErr(Exception):
    pass


def _entry(custom_id, rtype, text=None, error=None):
    message = None
    if text is not None:
        blocks = [types.SimpleNamespace(text=text)]
        message = types.SimpleNamespace(content=blocks)
    result = types.SimpleNamespace(type=rtype, message=message, error=error)
    return types.SimpleNamespace(custom_id=custom_id, result=result)


def _make_fake_anthropic(*, batch=None, results=()):
    mod = types.ModuleType("anthropic")
    for name in ("RateLimitError", "APIConnectionError", "APITimeoutError",
                 "InternalServerError", "BadRequestError"):
        setattr(mod, name, type(name, (_FakeErr,), {}))

    captured: dict = {}
    mod._raise = None

    class _Batches:
        def create(self, *, requests):
            captured["requests"] = requests
            if mod._raise is not None:
                raise mod._raise
            return types.SimpleNamespace(id="msgbatch_1")

        def retrieve(self, job_id):
            return batch

        def results(self, job_id):
            return iter(results)

    class _Client:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.messages = types.SimpleNamespace(batches=_Batches())

    mod.Anthropic = _Client
    mod._captured = captured
    return mod


@pytest.fixture
def fake_anthropic(monkeypatch):
    def _install(**kwargs):
        mod = _make_fake_anthropic(**kwargs)
        monkeypatch.setitem(sys.modules, "anthropic", mod)
        return mod
    return _install


def _reqs(n=2):
    return [
        Request(custom_id=f"run_1_item_rec_{i:04d}", order_id=f"rec_{i:04d}",
                prompt=f"prompt {i}", max_tokens=32)
        for i in range(1, n + 1)
    ]


def test_submit_builds_requests_with_custom_id_and_params(fake_anthropic):
    mod = fake_anthropic(batch=None)
    provider = AnthropicBatchProvider(model="claude-x", system="rate it",
                                      temperature=0.0)
    job_id = provider.submit(_reqs(2))
    assert job_id == "msgbatch_1"
    reqs = mod._captured["requests"]
    assert [r["custom_id"] for r in reqs] == [
        "run_1_item_rec_0001", "run_1_item_rec_0002"]
    params = reqs[0]["params"]
    assert params["model"] == "claude-x"
    assert params["max_tokens"] == 32
    assert params["system"] == "rate it"
    assert params["temperature"] == 0.0
    assert params["messages"] == [{"role": "user", "content": "prompt 1"}]


def test_submit_omits_system_and_temperature_when_unset(fake_anthropic):
    mod = fake_anthropic(batch=None)
    AnthropicBatchProvider().submit(_reqs(1))
    params = mod._captured["requests"][0]["params"]
    assert "system" not in params
    assert "temperature" not in params


def test_submit_rejects_oversize_batch_locally(fake_anthropic):
    fake_anthropic(batch=None)
    provider = AnthropicBatchProvider(limits=ProviderLimits(max_items_per_batch=1))
    with pytest.raises(BatchTooLargeError):
        provider.submit(_reqs(2))


def _batch(status, succeeded=2, errored=0, canceled=0, expired=0, processing=0):
    return types.SimpleNamespace(
        processing_status=status,
        request_counts=types.SimpleNamespace(
            succeeded=succeeded, errored=errored, canceled=canceled,
            expired=expired, processing=processing),
    )


@pytest.mark.parametrize("status,expected,terminal", [
    ("ended", "ended", True),
    ("in_progress", "in_progress", False),
    ("canceling", "canceling", False),
])
def test_poll_maps_status(fake_anthropic, status, expected, terminal):
    fake_anthropic(batch=_batch(status, succeeded=3, errored=1, canceled=1,
                                expired=2, processing=4))
    st = AnthropicBatchProvider().poll("msgbatch_1")
    assert st.state == expected
    assert st.is_terminal is terminal
    assert st.succeeded == 3
    # errored folds in canceled + expired
    assert st.errored == 1 + 1 + 2
    assert st.processing == 4


def test_fetch_streams_results(fake_anthropic):
    results = (
        _entry("run_1_item_rec_0001", "succeeded", text="rec_0001,5,5"),
        _entry("run_1_item_rec_0002", "errored", error="overloaded_error"),
        _entry("run_1_item_rec_0003", "expired"),
    )
    fake_anthropic(batch=None, results=results)
    out = {r.custom_id: r for r in AnthropicBatchProvider().fetch_results("msgbatch_1")}
    assert out["run_1_item_rec_0001"].ok is True
    assert out["run_1_item_rec_0001"].text == "rec_0001,5,5"
    assert out["run_1_item_rec_0002"].ok is False
    assert "overloaded" in out["run_1_item_rec_0002"].error
    assert out["run_1_item_rec_0003"].ok is False
    assert "expired" in out["run_1_item_rec_0003"].error


def test_fetch_concatenates_multiple_text_blocks(fake_anthropic):
    result = types.SimpleNamespace(
        type="succeeded",
        message=types.SimpleNamespace(content=[
            types.SimpleNamespace(text="rec_0001,"),
            types.SimpleNamespace(text="9,2"),
        ]),
        error=None,
    )
    entry = types.SimpleNamespace(custom_id="run_1_item_rec_0001", result=result)
    fake_anthropic(batch=None, results=(entry,))
    (r,) = list(AnthropicBatchProvider().fetch_results("msgbatch_1"))
    assert r.text == "rec_0001,9,2"


def test_submit_maps_rate_limit_to_transient(fake_anthropic):
    mod = fake_anthropic(batch=None)
    mod._raise = mod.RateLimitError("429")
    with pytest.raises(TransientSubmitError):
        AnthropicBatchProvider().submit(_reqs(1))


def test_submit_maps_bad_request_size_to_too_large(fake_anthropic):
    mod = fake_anthropic(batch=None)
    mod._raise = mod.BadRequestError("request too large: token limit exceeded")
    with pytest.raises(BatchTooLargeError):
        AnthropicBatchProvider().submit(_reqs(1))
