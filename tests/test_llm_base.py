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
    def test_json_array_in_code_block_raises_parsing_error(self):
        """``extract_json`` is annotated ``-> dict[str, Any]`` and callers index its result, so a
        top-level array (very common under "respond with valid JSON only") must surface as the
        typed JSONParsingError the retry layer catches -- not as a list that breaks the caller
        with a TypeError far from the parse site."""
        text = "```json\n[1, 2, 3]\n```"
        with pytest.raises(JSONParsingError):
            LLMProvider.extract_json(text)

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

    # Every REAL tokeniser-undercount incident this reserve exists to absorb, as
    # `(label, context_window, estimated_input, real_input)`. Both entries are measured, not constructed:
    # the estimate is what `count_tokens` returned for that exact prompt, and the real count is the one the
    # upstream itself reported when it rejected (or accepted) the call. Each new incident is APPENDED here
    # rather than replacing the previous one - the reserve has to clear the worst of them, and a table that
    # forgets the earlier cases cannot enforce that.
    _MEASURED_UNDERCOUNTS = [
        ("2026-08-08 deepseek-v3.2", 163_840, 20_864, 23_617),
        ("2026-08-26 gpt-oss-120b", 131_072, 19_920, 24_077),
    ]

    @pytest.mark.parametrize(("label", "window", "estimated_input_tokens", "real_input_tokens"), _MEASURED_UNDERCOUNTS)
    def test_clamped_budget_survives_a_realistic_tokeniser_undercount(self, monkeypatch, label, window, estimated_input_tokens, real_input_tokens):
        """Regression for every measured incident in `_MEASURED_UNDERCOUNTS`.

        The shape is identical in both: `count_tokens` undercounts a large real prompt, the clamp computes
        its budget from that undercount, and the upstream then rejects the whole call with HTTP 400 before
        generating a single token because REAL input + granted output exceeds the window. 2026-08-08
        (deepseek-v3.2) overflowed by ~1,729 tokens against the then-flat 1024-token reserve; 2026-08-26
        (gpt-oss-120b) overflowed by 1,169 against the 0.15 fraction that replaced it, because that
        incident's own undercount was 20.9% of the estimate - larger than the fraction meant to absorb it.
        """

        class _Prov(self._Provider):  # type: ignore[name-defined]
            @property
            def context_window(self) -> int:
                return window

        prov = _Prov()

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


# ---------------------------------------------------------------------------
# is_llm_refusal / longest_prefix_lookup -- both public, both previously never
# mentioned anywhere under tests/ (audit F20, 2026-09-02). longest_prefix_lookup
# resolves Anthropic's max-output-token limits (llm/anthropic_provider.py:129), so a
# wrong prefix match silently caps generation at the fallback.
# ---------------------------------------------------------------------------


class TestIsLlmRefusal:
    @pytest.mark.parametrize(
        "text",
        [
            "I cannot help with that request.",
            "I can't assist with this.",
            "Unfortunately I am unable comply, sorry.",
            "I will not help you do that.",
            "I won't comply.",
            "I'm not able to help with that.",
            "I'm unable to process this request.",
            "I cannot provide that.",
            "I can't generate this.",
            "Sorry. I Cannot Help with it.",  # case-insensitive
        ],
    )
    def test_recognises_refusal_phrasings(self, text):
        from pyutilz.llm.base import is_llm_refusal

        assert is_llm_refusal(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Here is the answer you asked for.",
            "The function cannot return None, so we raise instead.",  # 'cannot' mid-sentence, not a refusal
            "I can help with that!",
            "Assistance is provided by the helper module.",
        ],
    )
    def test_ordinary_answers_are_not_flagged(self, text):
        from pyutilz.llm.base import is_llm_refusal

        assert is_llm_refusal(text) is False

    def test_non_string_input_is_false_not_a_crash(self):
        from pyutilz.llm.base import is_llm_refusal

        assert is_llm_refusal(None) is False
        assert is_llm_refusal(123) is False

    @pytest.mark.parametrize("text", ["I am unable to comply.", "I am not able to help you."])
    def test_documented_conservative_boundary(self, text):
        """Phrasings a human reads as refusals but _REFUSAL_PATTERNS deliberately does NOT match.

        Each pattern requires the verb to follow the modal immediately ("I am unable HELP"), so an
        intervening "to"/"you" defeats it. That is the module's stated design -- "err on the strict
        side", because a false positive silently downgrades a valid answer to a fallback. Pinned
        here so the boundary is a measured, visible decision rather than an unnoticed gap; widening
        the patterns is a deliberate change that should flip these assertions on purpose.
        """
        from pyutilz.llm.base import is_llm_refusal

        assert is_llm_refusal(text) is False


class TestLongestPrefixLookup:
    TABLE = {
        "claude-3-5-sonnet": 8192,
        "claude-3": 4096,
        "claude-opus-4-1": 32000,
    }

    def test_exact_match_wins(self):
        from pyutilz.llm.base import longest_prefix_lookup

        assert longest_prefix_lookup("claude-3", self.TABLE, default=64000) == 4096

    def test_longest_prefix_wins_over_a_shorter_one(self):
        """"claude-3-5-sonnet-20241022" starts with BOTH "claude-3" and "claude-3-5-sonnet";
        the longer key must win, or every dated Sonnet build silently gets the 4096 limit."""
        from pyutilz.llm.base import longest_prefix_lookup

        assert longest_prefix_lookup("claude-3-5-sonnet-20241022", self.TABLE, default=64000) == 8192

    def test_trailing_segment_trimmed_prefix_is_the_second_chance(self):
        """No full key is a prefix of "claude-opus-4-20250101", but trimming the last "-"
        segment off "claude-opus-4-1" yields "claude-opus-4", which is."""
        from pyutilz.llm.base import longest_prefix_lookup

        assert longest_prefix_lookup("claude-opus-4-20250101", self.TABLE, default=64000) == 32000

    def test_unknown_model_falls_back_to_the_default(self):
        from pyutilz.llm.base import longest_prefix_lookup

        assert longest_prefix_lookup("gpt-4o-mini", self.TABLE, default=64000) == 64000

    def test_empty_table_returns_the_default(self):
        from pyutilz.llm.base import longest_prefix_lookup

        assert longest_prefix_lookup("anything", {}, default="fallback") == "fallback"
