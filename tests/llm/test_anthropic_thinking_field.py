"""The Anthropic thinking budget: one rule, reachable without a provider instance.

Anthropic takes a token budget where everything else here takes an effort string, and the budget is carved out of
``max_tokens``: a budget at or above it leaves no room for the answer and the API rejects the request. A caller that
drives the SDK directly needs that arithmetic too, so it is a free function and the provider method delegates to it.
These tests pin the delegation, so the two cannot drift apart into two spellings of one rule.
"""

from __future__ import annotations

import pytest

from pyutilz.llm.anthropic_provider import MIN_THINKING_BUDGET, THINKING_BUDGETS, AnthropicProvider, anthropic_thinking_field


class TestTheBudgetItAsksFor:
    @pytest.mark.parametrize(("effort", "expected"), sorted(THINKING_BUDGETS.items()))
    def test_each_effort_maps_to_its_budget(self, effort: str, expected: int) -> None:
        assert anthropic_thinking_field(effort, 64000) == {"type": "enabled", "budget_tokens": expected}

    def test_true_means_the_provider_default(self) -> None:
        assert anthropic_thinking_field(True, 64000) == {"type": "enabled", "budget_tokens": THINKING_BUDGETS["medium"]}

    @pytest.mark.parametrize("off", [False, "off"])
    def test_reasoning_off_sends_no_field(self, off: bool | str) -> None:
        assert anthropic_thinking_field(off, 64000) is None

    def test_an_unknown_effort_leaves_thinking_off_rather_than_guessing(self) -> None:
        """Substituting a budget the caller did not ask for bills them either way while their setting looks applied."""
        assert anthropic_thinking_field("enormous", 64000) is None


class TestTheRoomItLeavesForTheAnswer:
    def test_the_budget_is_capped_by_what_max_tokens_can_spare(self) -> None:
        assert anthropic_thinking_field("high", 3072) == {"type": "enabled", "budget_tokens": 2048}

    def test_too_small_an_allowance_leaves_thinking_off(self) -> None:
        """Below twice the minimum there is no split that leaves room for both the reasoning and the answer."""
        assert anthropic_thinking_field("minimal", 2 * MIN_THINKING_BUDGET - 1) is None

    def test_exactly_twice_the_minimum_still_fits(self) -> None:
        assert anthropic_thinking_field("minimal", 2 * MIN_THINKING_BUDGET) == {"type": "enabled", "budget_tokens": MIN_THINKING_BUDGET}


class TestTheProviderDelegates:
    @pytest.mark.parametrize(("thinking", "max_tokens"), [("medium", 64000), ("high", 3072), (False, 64000), ("enormous", 64000)])
    def test_the_method_answers_exactly_as_the_function_does(self, thinking: bool | str, max_tokens: int) -> None:
        provider = AnthropicProvider.__new__(AnthropicProvider)  # no API key, no network: the rule reads no instance state
        provider.model = "claude-opus-4-20250514"

        assert provider._thinking_request_field(thinking, max_tokens) == anthropic_thinking_field(thinking, max_tokens, model=provider.model)

    def test_the_class_constants_are_the_module_ones(self) -> None:
        """Two copies of the table would drift; the class keeps names, not values."""
        assert AnthropicProvider._THINKING_BUDGETS is THINKING_BUDGETS
        assert AnthropicProvider._MIN_THINKING_BUDGET == MIN_THINKING_BUDGET
