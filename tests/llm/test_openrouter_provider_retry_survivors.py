"""Bounded routing-404 retry pins for OpenRouterProvider.generate (mutation survivors)."""

from __future__ import annotations

import asyncio

import pytest

from pyutilz.llm.exceptions import LLMProviderError
from pyutilz.llm.openai_compat import OpenAICompatibleProvider
from pyutilz.llm.openrouter_provider import OpenRouterProvider

ROUTING_404 = "API error 404: No endpoints found for this model"


@pytest.fixture
def base(monkeypatch):
    """Script the base generate(): each call pops the next outcome; exceptions are raised."""
    state = {"outcomes": [], "calls": [], "sleeps": []}

    async def fake_generate(self, *args, **kwargs):
        state["calls"].append((args, kwargs))
        out = state["outcomes"].pop(0)
        if isinstance(out, BaseException):
            raise out
        return out

    async def fake_sleep(seconds):
        state["sleeps"].append(seconds)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate", fake_generate)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return state


def _retrying(max_attempts: int = 3, pause: float = 7.5) -> OpenRouterProvider:
    return OpenRouterProvider(api_key="k", retry_routing_404=True, routing_404_max_attempts=max_attempts, routing_404_pause_sec=pause)


def test_without_the_opt_in_a_routing_404_is_raised_at_once(base):
    base["outcomes"] = [LLMProviderError(ROUTING_404), "never reached"]
    with pytest.raises(LLMProviderError):
        asyncio.run(OpenRouterProvider(api_key="k").generate("hi"))
    assert len(base["calls"]) == 1
    assert base["sleeps"] == []


def test_a_routing_404_then_success_returns_the_success_after_one_pause(base):
    base["outcomes"] = [LLMProviderError(ROUTING_404), "ok"]
    assert asyncio.run(_retrying().generate("hi")) == "ok"
    assert len(base["calls"]) == 2
    assert base["sleeps"] == [7.5]


def test_every_attempt_failing_makes_exactly_the_budget_of_calls_and_pauses_between_them(base):
    base["outcomes"] = [LLMProviderError(ROUTING_404) for _ in range(3)]
    with pytest.raises(LLMProviderError):
        asyncio.run(_retrying(max_attempts=3).generate("hi"))
    assert len(base["calls"]) == 3
    assert base["sleeps"] == [7.5, 7.5]


def test_a_single_attempt_budget_makes_one_call_and_never_pauses(base):
    base["outcomes"] = [LLMProviderError(ROUTING_404)]
    with pytest.raises(LLMProviderError):
        asyncio.run(_retrying(max_attempts=1).generate("hi"))
    assert len(base["calls"]) == 1
    assert base["sleeps"] == []


def test_a_non_routing_error_is_not_retried(base):
    base["outcomes"] = [LLMProviderError("API error 500: upstream exploded"), "never reached"]
    with pytest.raises(LLMProviderError):
        asyncio.run(_retrying().generate("hi"))
    assert len(base["calls"]) == 1
    assert base["sleeps"] == []


def test_a_routing_405_is_retried_like_a_404(base):
    base["outcomes"] = [LLMProviderError("API error 405: Method not allowed"), "ok"]
    assert asyncio.run(_retrying().generate("hi")) == "ok"
    assert len(base["calls"]) == 2


def test_routing_wording_without_the_status_prefix_is_not_retried(base):
    base["outcomes"] = [LLMProviderError("upstream said: No endpoints found"), "never reached"]
    with pytest.raises(LLMProviderError):
        asyncio.run(_retrying().generate("hi"))
    assert len(base["calls"]) == 1


@pytest.mark.parametrize(
    "message",
    ["API error 404: Method not allowed", "API error 404: resource not found", "API error 405: No endpoints found"],
)
def test_each_routing_wording_on_a_404_or_405_is_retried(base, message):
    base["outcomes"] = [LLMProviderError(message), "ok"]
    assert asyncio.run(_retrying().generate("hi")) == "ok"
    assert len(base["calls"]) == 2


@pytest.mark.parametrize(
    "message",
    ["API error 500: No endpoints found", "API error 404: rate limited", "API error 404: notfound", "No endpoints found"],
)
def test_a_message_missing_a_routing_status_or_wording_is_not_retried(base, message):
    base["outcomes"] = [LLMProviderError(message), "never reached"]
    with pytest.raises(LLMProviderError):
        asyncio.run(_retrying().generate("hi"))
    assert len(base["calls"]) == 1


def test_a_retry_is_announced_with_the_attempt_number_and_pause_before_sleeping(base, caplog):
    base["outcomes"] = [LLMProviderError(ROUTING_404), "ok"]
    caplog.set_level("WARNING")
    asyncio.run(_retrying().generate("hi"))
    assert caplog.records[-1].getMessage() == (
        "[OR routing 404/405] openai/gpt-4o-mini attempt 1/3 - sleeping 7.5s before retry: API error 404: No endpoints found for this model"
    )


def test_the_retry_log_carries_the_first_120_characters_of_a_long_error(base, caplog):
    long_message = ROUTING_404 + " " + "x" * 200
    base["outcomes"] = [LLMProviderError(long_message), "ok"]
    caplog.set_level("WARNING")
    asyncio.run(_retrying().generate("hi"))
    assert caplog.records[-1].getMessage().endswith(": " + long_message[:120])
    assert not caplog.records[-1].getMessage().endswith(long_message[:121])


def test_the_keyword_arguments_reach_the_base_call_as_keywords(base):
    base["outcomes"] = ["ok"]
    asyncio.run(_retrying().generate("hi", temperature=0.5))
    assert base["calls"] == [(("hi",), {"temperature": 0.5})]
