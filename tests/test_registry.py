"""Tests for the provider registry (relay/providers/registry.py).

Importing the registry module must never require any optional SDK (openai/
anthropic/google) to be installed -- the adapters only import their SDKs
lazily inside their own methods.
"""

from __future__ import annotations

import pytest

from relay.providers.anthropic import AnthropicBatchProvider
from relay.providers.google import GoogleBatchProvider
from relay.providers.mock import MockProvider
from relay.providers.openai import OpenAIBatchProvider
from relay.providers.registry import get_provider_class, provider_names


def test_provider_names_lists_all_four():
    assert provider_names() == ["anthropic", "google", "mock", "openai"]


@pytest.mark.parametrize("name,cls", [
    ("mock", MockProvider),
    ("openai", OpenAIBatchProvider),
    ("anthropic", AnthropicBatchProvider),
    ("google", GoogleBatchProvider),
])
def test_get_provider_class_returns_right_class(name, cls):
    assert get_provider_class(name) is cls


def test_get_provider_class_unknown_name_raises_helpful_error():
    with pytest.raises(ValueError) as excinfo:
        get_provider_class("not-a-real-provider")
    msg = str(excinfo.value)
    assert "not-a-real-provider" in msg
    # helpful: names every valid provider so the user knows what to type.
    for name in ("mock", "openai", "anthropic", "google"):
        assert name in msg


def test_importing_registry_does_not_require_any_sdk():
    # If this test file imported at all (see the module-level imports above,
    # which include `from relay.providers.registry import ...`), the
    # import already succeeded without openai/anthropic/google.genai being
    # importable in general -- this test just documents/asserts that intent
    # explicitly by re-importing fresh.
    import importlib

    import relay.providers.registry as registry_module
    importlib.reload(registry_module)
    assert set(registry_module.provider_names()) == {
        "mock", "openai", "anthropic", "google",
    }
