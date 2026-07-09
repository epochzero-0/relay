"""Provider registry: name -> provider class lookup for the CLI.

Importing this module must stay stdlib-safe: each adapter class it imports
here (openai/anthropic/google) only imports its SDK lazily inside its own
methods, so simply importing the classes below never requires any of the
optional SDKs to be installed.
"""

from __future__ import annotations

from relay.providers.anthropic import AnthropicBatchProvider
from relay.providers.google import GoogleBatchProvider
from relay.providers.mock import MockProvider
from relay.providers.openai import OpenAIBatchProvider

#: name -> provider class, for every provider the CLI knows about.
#: MockProvider has no registry_name class attribute (its "mock" .name is set
#: in __init__), so it is keyed in literally here.
PROVIDERS: dict[str, type] = {
    "mock": MockProvider,
    OpenAIBatchProvider.registry_name: OpenAIBatchProvider,
    AnthropicBatchProvider.registry_name: AnthropicBatchProvider,
    GoogleBatchProvider.registry_name: GoogleBatchProvider,
}


def provider_names() -> list[str]:
    """Return the sorted list of valid provider names."""
    return sorted(PROVIDERS)


def get_provider_class(name: str) -> type:
    """Return the provider class registered under ``name``.

    Raises ValueError naming the valid choices when ``name`` is unknown.
    """
    try:
        return PROVIDERS[name]
    except KeyError:
        valid = ", ".join(provider_names())
        raise ValueError(
            f"unknown provider {name!r}; valid providers are: {valid}"
        ) from None
