from __future__ import annotations
from typing import Any
import pytest


class TestReasoningBudgetInTimeout:
    class _T:
        model_name = "m"
        max_output_tokens = 1000

        def _get_timeout(self, model: str) -> float:
            return 240.0

    @pytest.fixture
    def t(self) -> Any:
        from pyutilz.llm._timeouts import DerivedTimeoutMixin

        cls = type("T", (self._T, DerivedTimeoutMixin), {})
        return cls()

    def test_separate_reasoning_budget_raises_the_timeout(self, t):
        assert t._timeout_for({"max_tokens": 500}) == 240.0
        assert t._timeout_for({"max_tokens": 500, "reasoning": {"max_tokens": 12_000}}) == pytest.approx(12_500 / 30)
        assert t._timeout_for({"max_tokens": 500, "thinking": {"type": "enabled", "budget_tokens": 15_000}}) == pytest.approx(15_500 / 30)

    def test_reasoning_budget_without_max_tokens(self, t):
        assert t._timeout_for({"reasoning": {"max_tokens": 30_000}}) == pytest.approx(1000.0)

    def test_reasoning_is_still_capped(self, t):
        assert t._timeout_for({"max_tokens": 500, "reasoning": {"max_tokens": 10_000_000}}) == t._max_derived_timeout_s

    @pytest.mark.parametrize("block", [{"effort": "high"}, {"max_tokens": 0}, {"max_tokens": True}, "high", None])
    def test_non_budget_reasoning_is_ignored(self, t, block):
        assert t._timeout_for({"max_tokens": 500, "reasoning": block}) == 240.0
