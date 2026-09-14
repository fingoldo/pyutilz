"""The ceiling a model is given must never be a number nobody chose.

``OpenRouterProvider.max_output_tokens`` reads ``top_provider.max_completion_tokens`` from
the models catalogue. The catalogue is fetched lazily and deliberately refuses to fetch from
the event-loop thread -- a blocking httpx call there would stall every in-flight coroutine
-- so a read taken before anything warms it returns nothing.

That read used to become ``_default_max_tokens``, 8192, against models whose real figure is
131072. Measured 2026-09-14, the third pipeline run lost to it: a corpus wave sized its budget
from this property BEFORE calling generate, got 8192 for a model serving 65536, and its first
article came back cut mid-JSON at 20,194 characters, billed in full. An unknown ceiling is now
the context window -- real and generous -- and never a small round number.

A separate file rather than another class in ``test_llm_openrouter.py``: that one is at its
LOC budget with no slack, and this is its own concern.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from pyutilz.llm.openrouter_provider import OpenRouterProvider
import pyutilz.llm.openrouter_provider as openrouter_module


def _provider(**overrides) -> OpenRouterProvider:
    """Same construction path as tests/test_llm_openrouter.py::_provider."""
    mock_settings = MagicMock()
    mock_settings.openrouter_api_key = None
    init_kwargs = {"api_key": "test-key", "model": "openai/gpt-4o-mini", "max_concurrent": 10}  # pragma: allowlist secret
    init_kwargs.update(overrides)
    with patch("pyutilz.llm.openrouter_provider.get_llm_settings", return_value=mock_settings):
        return OpenRouterProvider(**init_kwargs)


@pytest.fixture(autouse=True)
def _reset_caches():
    openrouter_module._MODELS_CATALOGUE = None
    openrouter_module._HEALTH_CACHE.clear()
    yield
    openrouter_module._MODELS_CATALOGUE = None
    openrouter_module._HEALTH_CACHE.clear()


class TestCeilingFallbackIsAnnounced:
    """2026-09-06: a silent fallback to _default_max_tokens cost two full pipeline runs.

    ``max_output_tokens`` reads ``top_provider.max_completion_tokens`` from the catalogue.
    The catalogue is fetched lazily and DELIBERATELY refuses to fetch from the event-loop
    thread, so a read before any warm returns nothing and the ceiling silently becomes 8192
    -- against models whose real limit is 131072. Downstream that shows up only as a
    truncated response, with nothing anywhere naming the cause.

    The fallback itself is correct behaviour, so this warns rather than raising - but it is no
    longer a SMALL fallback, because "the catalogue could not be read" and "the catalogue lists no
    cap" are both statements that the real ceiling is unknown, not that it is 8192.
    """

    @pytest.fixture(autouse=True)
    def _forget_previous_warnings(self):
        OpenRouterProvider._warned_default_ceiling.clear()
        yield
        OpenRouterProvider._warned_default_ceiling.clear()

    def test_a_missing_catalogue_entry_is_reported(self, caplog):
        provider = _provider(model="some/unlisted-model")

        with patch(
            "pyutilz.llm.openrouter_provider._provider._resolve_model_limits",
            return_value=(None, None),
        ), caplog.at_level(logging.WARNING):
            ceiling = provider.max_output_tokens

        assert ceiling == provider.context_window
        assert ceiling > OpenRouterProvider._default_max_tokens, "an unknown ceiling must be generous, never 8192"
        assert any("some/unlisted-model" in r.getMessage() for r in caplog.records)
        assert any("UNKNOWN rather than small" in r.getMessage() for r in caplog.records)

    def test_it_says_so_only_once_per_model(self, caplog):
        """A per-read warning would drown the log on the hot path."""
        provider = _provider(model="some/unlisted-model")

        with patch(
            "pyutilz.llm.openrouter_provider._provider._resolve_model_limits",
            return_value=(None, None),
        ), caplog.at_level(logging.WARNING):
            provider.max_output_tokens
            provider.max_output_tokens
            provider.max_output_tokens

        assert len([r for r in caplog.records if "unlisted-model" in r.getMessage()]) == 1

    def test_a_known_ceiling_is_returned_silently(self, caplog):
        """Negative control: the ordinary path must stay quiet."""
        provider = _provider(model="openai/gpt-4o-mini")

        with patch(
            "pyutilz.llm.openrouter_provider._provider._resolve_model_limits",
            return_value=(1_000_000, 131_072),
        ), caplog.at_level(logging.WARNING):
            ceiling = provider.max_output_tokens

        assert ceiling == 131_072
        assert not [r for r in caplog.records if "UNKNOWN" in r.getMessage()]


class TestAnUnreadCatalogueIsNotASmallCeiling:
    """The defect itself: a budget sized from this property before anything warmed the catalogue.

    `_fetch_models_catalogue` returns `{}` rather than blocking the event loop, so a caller that
    computes `max_tokens` inside async - which is every caller that sizes a budget before calling
    `generate` - used to receive 8192 and truncate every long answer it paid for.
    """

    @pytest.fixture(autouse=True)
    def _forget_previous_warnings(self):
        OpenRouterProvider._warned_default_ceiling.clear()
        yield
        OpenRouterProvider._warned_default_ceiling.clear()

    def test_a_cold_catalogue_read_from_the_event_loop_gives_the_context_window(self, caplog):
        import asyncio

        provider = _provider(model="deepseek/deepseek-v3.2")

        async def read_before_any_warm() -> tuple[int, int]:
            # Both reads inside the cold moment: once asyncio.run returns, an off-loop read is
            # allowed to fetch for real, and comparing across that boundary compares two catalogues.
            openrouter_module._MODELS_CATALOGUE = None
            return provider.max_output_tokens, provider.context_window

        with caplog.at_level(logging.WARNING):
            ceiling, window = asyncio.run(read_before_any_warm())

        assert ceiling == window == OpenRouterProvider._default_context_window
        assert ceiling != 8192
        assert any("has not been read yet" in r.getMessage() for r in caplog.records)

    def test_a_catalogue_that_answered_wins_over_any_fallback(self):
        """Negative control: the real number must still be preferred wherever one exists."""
        provider = _provider(model="deepseek/deepseek-v3.2")

        with patch(
            "pyutilz.llm.openrouter_provider._provider._resolve_model_limits",
            return_value=(163_840, 65_536),
        ):
            assert provider.max_output_tokens == 65_536
