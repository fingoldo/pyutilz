"""An unset max_tokens is omitted, never replaced by the catalogue ceiling.

The OpenRouter route refuses the deepseek catalogue ceiling (943718) with a 404 on every call, so a request that
carried it never got an answer. Only a number the caller chose may be sent as max_tokens.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from pyutilz.llm.openrouter_provider import OpenRouterProvider


class _CapturedError(Exception):
    """Raised by the fit stub so generate stops before any network call."""


def _provider() -> OpenRouterProvider:
    mock_settings = MagicMock()
    mock_settings.openrouter_api_key = None
    with patch("pyutilz.llm.openrouter_provider.get_llm_settings", return_value=mock_settings):
        return OpenRouterProvider(api_key="test-key", model="deepseek/deepseek-v4.1-flash", max_concurrent=10)  # pragma: allowlist secret


def _budget_generate_would_send(max_tokens: int) -> object:
    provider = _provider()
    seen: dict[str, object] = {}

    def capture(budget, prompt, system=None):
        seen["budget"] = budget
        raise _CapturedError

    with ExitStack() as stack:
        stack.enter_context(patch.object(OpenRouterProvider, "max_output_tokens", new_callable=PropertyMock, return_value=943718))
        stack.enter_context(patch.object(OpenRouterProvider, "_async_prepare", new=AsyncMock(return_value=None)))
        stack.enter_context(patch.object(provider, "fit_max_tokens_to_context", side_effect=capture))
        with pytest.raises(_CapturedError):
            asyncio.run(provider.generate("hello", max_tokens=max_tokens))
    return seen["budget"]


def test_an_unset_max_tokens_does_not_take_the_catalogue_ceiling() -> None:
    assert _budget_generate_would_send(0) is None


def test_a_number_the_caller_chose_is_sent_as_given() -> None:
    assert _budget_generate_would_send(4000) == 4000
