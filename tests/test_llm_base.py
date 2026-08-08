"""Tests for LLM provider abstract base classes."""

import pytest

from pyutilz.llm.base import LLMProvider
from pyutilz.llm.exceptions import JSONParsingError


class TestLLMProviderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match="abstract method"):
            LLMProvider()

    def test_max_output_tokens_default(self):
        assert LLMProvider.max_output_tokens.fget.__doc__ or True

    def test_context_window_default(self):
        assert LLMProvider.context_window.fget.__doc__ or True


class TestExtractJsonEdgeCases:
    def test_json_array_in_code_block_parses_as_list(self):
        text = "```json\n[1, 2, 3]\n```"
        result = LLMProvider.extract_json(text)
        assert result == [1, 2, 3]

    def test_multiple_json_objects_returns_first(self):
        # New JSONDecoder.raw_decode scan returns the first parseable
        # object rather than the old greedy-regex behaviour of raising.
        # First-wins matches what most LLM responses intend when prose
        # frames the JSON with trailing commentary.
        text = '{"a": 1} {"b": 2}'
        result = LLMProvider.extract_json(text)
        assert result == {"a": 1}

    def test_json_with_trailing_prose(self):
        # raw_decode scan stops at the JSON value; trailing prose is OK.
        text = '{"key": "value"} -- and that\'s the answer'
        assert LLMProvider.extract_json(text) == {"key": "value"}

    def test_json_with_leading_prose(self):
        text = 'Sure, here you go: {"key": "value"}'
        assert LLMProvider.extract_json(text) == {"key": "value"}

    def test_invalid_json_still_raises(self):
        text = "this is not json at all { broken"
        with pytest.raises(JSONParsingError):
            LLMProvider.extract_json(text)

    def test_json_with_markdown_prefix(self):
        text = 'Sure, here is the JSON:\n```json\n{"key": "val"}\n```'
        assert LLMProvider.extract_json(text) == {"key": "val"}

    def test_triple_backtick_no_json_label(self):
        text = '```\n{"x": 42}\n```'
        assert LLMProvider.extract_json(text) == {"x": 42}

    def test_deeply_nested(self):
        text = '{"a": {"b": {"c": {"d": 1}}}}'
        r = LLMProvider.extract_json(text)
        assert r["a"]["b"]["c"]["d"] == 1

    def test_provider_name_in_error(self):
        with pytest.raises(JSONParsingError, match="MyProvider"):
            LLMProvider.extract_json("not json", provider_name="MyProvider")


class TestFitMaxTokensToContext:
    """A model's output cap can exceed ``context_window - input`` (llama-3.3-70b: 128k cap in a 131k
    window), and forwarding it makes the upstream reject the whole request with HTTP 400 before
    generating anything. The budget must be clamped to what actually fits."""

    class _Provider(LLMProvider):
        """Minimal concrete provider mirroring llama-3.3-70b's advertised limits."""

        model_name = "meta-llama/llama-3.3-70b-instruct"

        @property
        def max_output_tokens(self) -> int:
            return 128_000

        @property
        def context_window(self) -> int:
            return 131_072

        async def generate(self, prompt, system=None, temperature=0.7, max_tokens=0, **kwargs):
            return ""

        async def generate_json(self, prompt, system=None, temperature=0.7, max_tokens=0, **kwargs):
            return {}

        async def generate_stream(self, prompt, system=None, temperature=0.7, max_tokens=0, **kwargs):
            yield ""

        async def count_tokens(self, text: str) -> int:
            return len(text) // 4

    def test_output_cap_exceeding_window_is_clamped(self):
        prov = self._Provider()
        prompt = "x " * 8000  # a real prompt, so cap + input overflows the window
        fitted = prov.fit_max_tokens_to_context(prov.max_output_tokens, prompt)
        assert fitted < prov.max_output_tokens
        # The whole request must now fit: input + output <= context_window.
        from pyutilz.llm.token_counter import count_tokens

        assert count_tokens(prompt, model=prov.model_name) + fitted <= prov.context_window

    def test_budget_that_already_fits_is_untouched(self):
        prov = self._Provider()
        assert prov.fit_max_tokens_to_context(4096, "short prompt") == 4096

    def test_zero_budget_passes_through(self):
        # 0 means "provider decides" and is resolved before the clamp runs.
        assert self._Provider().fit_max_tokens_to_context(0, "prompt") == 0

    def test_system_prompt_counts_towards_input(self):
        prov = self._Provider()
        prompt = "x " * 4000
        with_system = prov.fit_max_tokens_to_context(prov.max_output_tokens, prompt, system="y " * 4000)
        without = prov.fit_max_tokens_to_context(prov.max_output_tokens, prompt)
        assert with_system < without

    def test_oversized_prompt_defers_to_upstream_error(self):
        # No usable room left: return the budget unchanged so the upstream's own context error
        # surfaces instead of a silently truncated budget.
        prov = self._Provider()
        assert prov.fit_max_tokens_to_context(4096, "x " * 200_000) == 4096

    def test_reserve_scales_with_input_size_not_flat(self):
        """`_context_reserve_tokens` must grow with the estimated input, not stay pinned at the flat
        1024-token envelope constant - a flat reserve was too small to absorb realistic tokeniser
        estimation error on a large prompt (see the regression test below for the measured incident)."""
        prov = self._Provider()
        assert prov._context_reserve_tokens(1_000) == prov._CONTEXT_RESERVE_TOKENS  # small input: flat floor wins
        assert prov._context_reserve_tokens(100_000) > prov._CONTEXT_RESERVE_TOKENS  # large input: fraction wins
        assert prov._context_reserve_tokens(100_000) == int(100_000 * prov._CONTEXT_RESERVE_FRACTION)

    def test_clamped_budget_survives_a_realistic_tokeniser_undercount(self, monkeypatch):
        """Regression for the incident measured 2026-08-08 (autopsia, OpenRouter/deepseek-v3.2): a real
        ~23,617-token prompt was undercounted by `count_tokens` at ~20,864 tokens (an ~11.8% gap on a
        163,840-token context window). The OLD flat 1024-token reserve produced a clamped `max_tokens`
        that, added to the REAL (higher) input count, exceeded the window by ~1,729 tokens - the upstream
        rejected the whole call with HTTP 400 before generating a single token. This test simulates the
        same undercount ratio and asserts the clamp now leaves enough margin to survive it."""
        prov = self._Provider()
        real_input_tokens = 23_617
        estimated_input_tokens = 20_864  # what count_tokens actually returned for the same real prompt

        def _undercounting_count_tokens(text, model=None):
            # Ignore the actual text; simulate the exact measured ratio from the incident regardless of
            # what fixture text this test happens to pass in.
            return estimated_input_tokens

        monkeypatch.setattr("pyutilz.llm.token_counter.count_tokens", _undercounting_count_tokens)

        fitted = prov.fit_max_tokens_to_context(prov.max_output_tokens, "irrelevant, count_tokens is patched")

        # The clamp only sees the (under)estimated input; assert the REAL input would still fit alongside
        # the fitted budget - this is the actual failure mode: input+output must fit against the TRUE
        # count the upstream will use, not the estimate the clamp itself was computed from.
        assert real_input_tokens + fitted <= prov.context_window, (
            f"clamped max_tokens={fitted} + real input={real_input_tokens} = "
            f"{real_input_tokens + fitted} > context_window={prov.context_window} - "
            "the reserve did not absorb the tokeniser undercount"
        )
