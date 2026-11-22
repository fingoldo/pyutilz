"""Mutation-survivor pins for openai_compat: the buffered path, usage and cost accounting, and the provider knobs."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from tenacity import stop_after_attempt, wait_none
from pyutilz.llm.openai_compat import OpenAICompatibleProvider
from tests.openai_compat_support import _TestProvider, _make_provider, _mock_response
httpx = pytest.importorskip("httpx")


httpx = pytest.importorskip("httpx")


class TestPricingFallbackSurvivors:
    """The default pricing record keeps input and output in their own fields, and a missing cached-input or cache-write
    rate falls back to the uncached input rate, while a published one is used (openai_compat.py 263-277)."""

    def test_the_default_record_keeps_input_and_output_apart(self):
        from pyutilz.llm._pricing import Pricing

        record = _make_provider()._resolve_pricing("m")
        assert record == Pricing(input=1.0, output=2.0, cache_hit=None)

    def test_a_published_cache_hit_rate_is_used(self):
        from pyutilz.llm._pricing import Pricing

        p = _make_provider()
        p._resolve_pricing = lambda model: Pricing(input=1.0, output=2.0, cache_hit=0.25)
        assert OpenAICompatibleProvider._cache_hit_cost_per_1m(p, "m") == 0.25

    def test_a_missing_cache_hit_rate_falls_back_to_the_input_rate(self):
        from pyutilz.llm._pricing import Pricing

        p = _make_provider()
        p._resolve_pricing = lambda model: Pricing(input=1.0, output=2.0, cache_hit=None)
        assert OpenAICompatibleProvider._cache_hit_cost_per_1m(p, "m") == 1.0

    def test_a_published_cache_write_rate_is_used_and_a_missing_one_falls_back(self):
        from pyutilz.llm._pricing import Pricing

        p = _make_provider()
        p._resolve_pricing = lambda model: Pricing(input=1.0, output=2.0, cache_write=3.0)
        assert p._cache_write_cost_per_1m("m") == 3.0
        p._resolve_pricing = lambda model: Pricing(input=1.0, output=2.0, cache_write=None)
        assert p._cache_write_cost_per_1m("m") == 1.0

class TestUsageAccountingSurvivors:
    """Each usage block adds its prompt, completion, cache-hit, reasoning and cache-write tokens to the totals, read from
    the DeepSeek key or the nested OpenAI/xAI key, and a null field counts as zero (openai_compat.py 587-599)."""

    def test_plain_counts_add_to_the_totals(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "completion_tokens": 4})
        assert (p.total_prompt_tokens, p.total_completion_tokens, p.total_cache_hit_tokens, p.total_reasoning_tokens) == (10, 4, 0, 0)

    def test_the_per_call_usage_dict_sums_its_blocks_under_each_key(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "completion_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 3}, "prompt_cache_hit_tokens": 6})
        p._record_usage({"prompt_tokens": 5, "completion_tokens": 2})
        assert p._last_usage == {"input_tokens": 15, "output_tokens": 6, "reasoning_tokens": 3, "cached_tokens": 6}

    def test_the_usage_log_line_states_prompt_cache_completion_and_reasoning_in_order(self, caplog):
        import logging

        p = _make_provider()
        with caplog.at_level(logging.INFO):
            p._record_usage({"prompt_tokens": 10, "completion_tokens": 4, "prompt_cache_hit_tokens": 6, "completion_tokens_details": {"reasoning_tokens": 3}})
        assert "10 prompt (6 cached) + 4 completion (3 reasoning) | cumulative: 10 in, 4 out" in caplog.text

    def test_totals_accumulate_across_calls(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "completion_tokens": 4, "prompt_tokens_details": {"cache_write_tokens": 3}})
        p._record_usage({"prompt_tokens": 5, "completion_tokens": 2, "prompt_tokens_details": {"cache_write_tokens": 1}})
        assert (p.total_prompt_tokens, p.total_completion_tokens, p.total_cache_write_tokens) == (15, 6, 4)

    def test_the_deepseek_cache_key_is_read(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "prompt_cache_hit_tokens": 6})
        assert p.total_cache_hit_tokens == 6

    def test_the_nested_openai_cache_key_is_read(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 5}})
        assert p.total_cache_hit_tokens == 5

    def test_reasoning_tokens_are_read_from_the_completion_details(self):
        p = _make_provider()
        p._record_usage({"completion_tokens": 9, "completion_tokens_details": {"reasoning_tokens": 7}})
        assert p.total_reasoning_tokens == 7

    def test_cache_writes_are_read_from_either_nesting(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens_details": {"cache_write_tokens": 3}})
        p._record_usage({"input_tokens_details": {"cache_write_tokens": 2}})
        assert p.total_cache_write_tokens == 5

    def test_null_fields_count_as_zero(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": None, "completion_tokens": None, "prompt_cache_hit_tokens": None, "completion_tokens_details": {"reasoning_tokens": None}})
        assert (p.total_prompt_tokens, p.total_completion_tokens, p.total_cache_hit_tokens, p.total_reasoning_tokens) == (0, 0, 0, 0)

class TestGenerateDefaultsAndOrderSurvivors:
    """A buffered generate call: its defaults reach the body as documented, the options reach the body builder in order,
    the per-call state is reset and the async preparation awaited (openai_compat.py 656, 694-714)."""

    @pytest.mark.asyncio
    async def test_a_plain_generate_call_sends_the_documented_defaults(self):
        p = _make_provider()
        p._post_and_unwrap = AsyncMock(return_value="ok")
        assert await p.generate("q") == "ok"
        body = p._post_and_unwrap.call_args.args[0]
        assert body["temperature"] == 0.7
        assert "max_tokens" not in body
        assert "response_format" not in body

    @pytest.mark.asyncio
    async def test_the_options_reach_the_body_builder_in_order(self):
        p = _make_provider()
        spy = MagicMock(return_value={"model": "m", "messages": []})
        p._request_body = spy
        p._post_and_unwrap = AsyncMock(return_value="ok")
        await p.generate("q", "sys", 0.2, 50, False, None, None, ["img"], [{"t": 1}], "auto", {"x": 1})
        args = spy.call_args.args
        assert args[0] == "q" and args[1] == "sys" and args[2] == 0.2
        assert args[6] == ["img"]
        assert args[7] == [{"t": 1}]
        assert args[8] == "auto"
        assert args[9] == {"x": 1}

    @pytest.mark.asyncio
    async def test_a_generate_call_awaits_the_preparation_and_starts_clean(self):
        p = _make_provider()
        p._async_prepare = AsyncMock()
        p._post_and_unwrap = AsyncMock(return_value="ok")
        p.last_json_mode_fallback = True
        await p.generate("q")
        p._async_prepare.assert_awaited_once()
        assert p.last_json_mode_fallback is False

class TestBufferedFinishReasonSurvivors:
    """A buffered completion ending with finish_reason 'error' raises with its partial text; a tool-call-only completion
    returns empty text with the tool calls kept; a missing finish reason reads as 'unknown' (openai_compat.py 837-851)."""

    @pytest.mark.asyncio
    async def test_an_error_finish_raises_with_the_partial_text(self):
        from pyutilz.llm.exceptions import LLMStreamInterruptedError

        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body={"choices": [{"message": {"content": "partial"}, "finish_reason": "error"}]}))
        # Retryable, so the decorator would back off for real: one attempt, no wait.
        original_stop, original_wait = p.generate.retry.stop, p.generate.retry.wait
        p.generate.retry.stop = stop_after_attempt(1)
        p.generate.retry.wait = wait_none()
        try:
            with pytest.raises(LLMStreamInterruptedError, match=r"TestProvider response ended with finish_reason='error'") as exc:
                await p.generate("q")
        finally:
            p.generate.retry.stop = original_stop
            p.generate.retry.wait = original_wait
        assert exc.value.partial_text == "partial"

    @pytest.mark.asyncio
    async def test_a_tool_call_only_completion_returns_empty_text_and_keeps_the_calls(self):
        p = _make_provider()
        p._client = AsyncMock()
        call = {"id": "t1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        p._client.post = AsyncMock(return_value=_mock_response(body={"choices": [{"message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}]}))
        assert await p.generate("q") == ""
        assert p.last_tool_calls == [call]

    @pytest.mark.asyncio
    async def test_a_completion_without_a_finish_reason_reads_as_unknown(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body={"choices": [{"message": {"content": "ok"}}]}))
        assert await p.generate("q") == "ok"
        assert p._last_finish_reason == "unknown"

class TestCostWithNonUnitRatesSurvivors:
    """With rates other than 1, the estimate and the session cost multiply each token count by its own rate, at exact
    values (openai_compat.py 933-947). Rates: input 4.0, output 8.0, cache-hit 0.1 (test provider), cache-write 2.5."""

    @staticmethod
    def _provider():
        class _Priced(_TestProvider):
            def _input_cost_per_1m(self, model):
                return 4.0

            def _output_cost_per_1m(self, model):
                return 8.0

            def _cache_write_cost_per_1m(self, model):
                return 2.5

        return _Priced(api_key="test-key", model="test-model")  # pragma: allowlist secret

    def test_the_estimate_prices_input_and_output_at_their_rates(self):
        assert self._provider().estimate_cost(1_000_000, 500_000) == pytest.approx(4.0 + 4.0, rel=1e-9)

    def test_the_session_cost_prices_each_token_class_at_its_rate(self):
        p = self._provider()
        p.total_prompt_tokens = 1_000_000
        p.total_cache_hit_tokens = 400_000
        p.total_cache_write_tokens = 100_000
        p.total_completion_tokens = 500_000
        cost = p.get_session_cost()
        assert cost["input_cost_usd"] == pytest.approx(0.5 * 4.0 + 0.4 * 0.1 + 0.1 * 2.5, rel=1e-9)
        assert cost["output_cost_usd"] == pytest.approx(0.5 * 8.0, rel=1e-9)

class TestLongTierEstimateSurvivors:
    """An estimate for a prompt at the long-context tier prices the call with price_call(pricing, input, output) and sums
    the parts (openai_compat.py 932)."""

    def test_the_long_tier_estimate_passes_pricing_then_tokens_in_order(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod
        from pyutilz.llm._pricing import Pricing

        class _Tier:
            def applies(self, n):
                return n >= 10

        class _LongTier(_TestProvider):
            def _long_context_tier(self, model):
                return _Tier()

        seen = []

        def _price_call(pricing, input_tokens, output_tokens):
            seen.append((pricing, input_tokens, output_tokens))
            return (1.25, 2.5)

        monkeypatch.setattr(openai_compat_mod, "price_call", _price_call)
        p = _LongTier(api_key="test-key", model="test-model")  # pragma: allowlist secret
        assert p.estimate_cost(20, 7) == pytest.approx(3.75, rel=1e-9)
        assert seen == [(Pricing(input=1.0, output=2.0, cache_hit=None), 20, 7)]

class TestLongContextSurchargeSurvivors:
    """A call whose prompt reaches the long-context tier adds that tier's surcharge to a running total and counts it;
    a short prompt adds nothing (openai_compat.py 907-925)."""

    def test_a_long_prompt_adds_the_surcharge_and_counts_the_call(self, monkeypatch):
        import pyutilz.llm.openai_compat as openai_compat_mod

        class _Tier:
            def applies(self, n):
                return n >= 10

        class _LongTier(_TestProvider):
            def _long_context_tier(self, model):
                return _Tier()

        seen = []

        def _surcharge(pricing, prompt, billed, cache_read, cache_write):
            seen.append((prompt, billed, cache_read, cache_write))
            return (1.5, 2.5)

        monkeypatch.setattr(openai_compat_mod, "long_context_surcharge", _surcharge)
        p = _LongTier(api_key="test-key", model="test-model")  # pragma: allowlist secret
        p._add_long_context_surcharge(20, 5, 3, 2)
        p._add_long_context_surcharge(20, 5, 3, 2)
        assert seen == [(20, 5, 3, 2), (20, 5, 3, 2)]
        assert p._long_context_surcharge_usd == pytest.approx(8.0)
        assert p._long_context_calls == 2

    def test_a_short_prompt_adds_no_surcharge(self):
        p = _make_provider()
        p._add_long_context_surcharge(20, 5, 3, 2)
        assert p._long_context_calls == 0

class TestSessionCostArithmeticSurvivors:
    """The session cost splits prompt tokens into cache-hit, cache-write and miss, prices each at its own rate, prices
    the billed output, and adds any long-context surcharge (openai_compat.py 936-968). Test rates: input 1.0,
    cache-hit 0.1 (from the test provider), output 2.0; cache-write falls back to input."""

    def test_the_session_cost_prices_each_token_class_at_its_own_rate(self):
        p = _make_provider()
        p.total_prompt_tokens = 1_000_000
        p.total_cache_hit_tokens = 400_000
        p.total_cache_write_tokens = 100_000
        p.total_completion_tokens = 500_000
        p.total_reasoning_tokens = 0
        cost = p.get_session_cost()
        assert cost["cache_miss_tokens"] == 500_000
        assert cost["prompt_tokens"] == 1_000_000
        assert cost["cache_hit_tokens"] == 400_000
        assert cost["completion_tokens"] == 500_000
        assert cost["reasoning_tokens"] == 0
        assert cost["calls"] == 0
        assert cost["input_cost_usd"] == pytest.approx(0.5 * 1.0 + 0.4 * 0.1 + 0.1 * 1.0, rel=1e-9)
        assert cost["output_cost_usd"] == pytest.approx(0.5 * 2.0, rel=1e-9)
        assert cost["total_cost_usd"] == pytest.approx(0.64 + 1.0, rel=1e-9)

    def test_the_surcharge_is_added_to_the_total_and_reported(self):
        p = _make_provider()
        p.total_prompt_tokens = 1_000
        p._long_context_surcharge_usd = 0.25
        p._long_context_calls = 3
        cost = p.get_session_cost()
        assert cost["long_context_surcharge_usd"] == 0.25
        assert cost["long_context_calls"] == 3
        assert cost["total_cost_usd"] == pytest.approx(cost["input_cost_usd"] + cost["output_cost_usd"] + 0.25, rel=1e-9)

class TestGenerateJsonDefaultsSurvivors:
    """generate_json forwards its documented defaults: temperature 0.3, no token cap, and JSON mode on (openai_compat.py
    886-892)."""

    @pytest.mark.asyncio
    async def test_generate_json_forwards_its_defaults(self):
        p = _make_provider()
        spy = AsyncMock(return_value={})
        p._generate_json_via = spy
        await p.generate_json("q")
        args = spy.call_args.args
        assert args[2] == 0.3
        assert args[3] == 0
        assert spy.call_args.kwargs["json_mode"] is True

class TestFreshProviderAndCapSurvivors:
    """A fresh provider reports zero cost and no cache writes, and an explicit output cap is sent as max_tokens
    (openai_compat.py 698, 941)."""

    def test_a_fresh_provider_reports_zero_input_cost_and_no_cache_misses(self):
        cost = _make_provider().get_session_cost()
        assert cost["cache_miss_tokens"] == 0
        assert cost["input_cost_usd"] == 0.0
        assert cost["long_context_calls"] == 0

    @pytest.mark.asyncio
    async def test_an_explicit_output_cap_is_sent_as_max_tokens(self):
        p = _make_provider()
        p._post_and_unwrap = AsyncMock(return_value="ok")
        await p.generate("q", max_tokens=50)
        assert p._post_and_unwrap.call_args.args[0]["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_a_one_token_cap_is_sent_as_max_tokens(self):
        p = _make_provider()
        p._post_and_unwrap = AsyncMock(return_value="ok")
        await p.generate("q", max_tokens=1)
        assert p._post_and_unwrap.call_args.args[0]["max_tokens"] == 1

class TestUsageHookAndSpecialStatusSurvivors:
    """Recording usage calls the provider's usage hook and the long-context surcharge for a prompt at the tier; a buffered
    call hands its error response to the provider's special-status handler first (openai_compat.py 625, 627, 783)."""

    def test_recording_usage_calls_the_provider_hook_with_the_block(self):
        p = _make_provider()
        hook = MagicMock()
        p._track_provider_specific_usage = hook
        usage = {"prompt_tokens": 3, "completion_tokens": 1}
        p._record_usage(usage)
        hook.assert_called_once_with(usage)

    def test_recording_usage_applies_the_surcharge_for_a_long_prompt(self):
        class _Tier:
            def applies(self, n):
                return n >= 10

        class _LongTier(_TestProvider):
            def _long_context_tier(self, model):
                return _Tier()

        p = _LongTier(api_key="test-key", model="test-model")  # pragma: allowlist secret
        p._add_long_context_surcharge = MagicMock()
        p._record_usage({"prompt_tokens": 20, "completion_tokens": 5})
        p._add_long_context_surcharge.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_buffered_special_status_is_raised_by_the_provider_handler(self):
        from pyutilz.llm.exceptions import LLMProviderError
        from tests.openai_compat_support import _TestProvider

        class _Special(_TestProvider):
            def _handle_special_status(self, resp):
                if resp.status_code == 402:
                    raise LLMProviderError("credits exhausted")

        p = _Special(api_key="test-key", model="test-model")  # pragma: allowlist secret
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(status_code=402, body={"error": {"message": "pay up"}}))
        # 402 is retryable: one attempt and no wait, so a missed handler fails at once instead of backing off.
        original_stop, original_wait = p.generate.retry.stop, p.generate.retry.wait
        p.generate.retry.stop = stop_after_attempt(1)
        p.generate.retry.wait = wait_none()
        try:
            with pytest.raises(LLMProviderError, match="credits exhausted"):
                await p.generate("q")
        finally:
            p.generate.retry.stop = original_stop
            p.generate.retry.wait = original_wait

class TestSurchargeCallArgumentsSurvivors:
    """Recording usage hands the surcharge its prompt tokens, billed output, cache-hit and cache-write counts, in that order
    (openai_compat.py 627)."""

    def test_the_surcharge_receives_prompt_billed_output_cache_hit_and_cache_write_in_order(self):
        p = _make_provider()
        spy = MagicMock()
        p._add_long_context_surcharge = spy
        p._record_usage({"prompt_tokens": 20, "completion_tokens": 5, "prompt_cache_hit_tokens": 6, "completion_tokens_details": {"reasoning_tokens": 3}, "prompt_tokens_details": {"cache_write_tokens": 2}})
        spy.assert_called_once_with(20, 5, 6, 2)

    def test_the_surcharge_is_the_last_call_after_the_usage_hook(self):
        p = _make_provider()
        p._add_long_context_surcharge = MagicMock()
        p._track_provider_specific_usage = MagicMock()
        usage = {"prompt_tokens": 20, "completion_tokens": 5}
        p._record_usage(usage)
        p._track_provider_specific_usage.assert_called_once_with(usage)
        p._add_long_context_surcharge.assert_called_once()

class TestCacheWriteDefaultSurvivors:
    """A usage block that reports no cache writes adds none (openai_compat.py 602)."""

    def test_a_usage_block_without_cache_writes_adds_none(self):
        p = _make_provider()
        p._record_usage({"prompt_tokens": 1})
        assert p.total_cache_write_tokens == 0

class TestGenerateJsonThinkingForwardingSurvivors:
    """generate_json forwards a thinking setting only when one is given (openai_compat.py 905)."""

    @pytest.mark.asyncio
    async def test_no_thinking_setting_is_not_forwarded(self):
        p = _make_provider()
        spy = AsyncMock(return_value={})
        p._generate_json_via = spy
        await p.generate_json("q")
        assert "thinking" not in spy.call_args.kwargs

    @pytest.mark.asyncio
    async def test_a_thinking_setting_is_forwarded_as_given(self):
        p = _make_provider()
        spy = AsyncMock(return_value={})
        p._generate_json_via = spy
        await p.generate_json("q", thinking=False)
        assert spy.call_args.kwargs["thinking"] is False

class TestErrorRequestLabelSurvivors:
    """An in-body error reports the response's own request, and a response without one reports a POST to
    /chat/completions (openai_compat.py 839-841)."""

    @pytest.mark.asyncio
    async def test_an_in_body_error_reports_the_responses_own_request(self):
        p = _make_provider()
        p._client = AsyncMock()
        p._client.post = AsyncMock(return_value=_mock_response(body={"error": {"code": 400, "message": "bad"}}))
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await p.generate("q")
        assert str(exc.value.request.url) == "https://test.example.com/chat/completions"
        assert exc.value.request.method == "POST"

class TestClientConfigurationSurvivors:
    """The shared client carries the bearer token and JSON content type, a 10 s connect timeout, and the per-call
    flags are reset at the start of each call (openai_compat.py 94-122, 103, 115)."""

    def test_the_client_sends_the_bearer_token_and_a_json_content_type(self):
        p = _make_provider()
        assert p._client.headers["Authorization"] == "Bearer test-key"
        assert p._client.headers["Content-Type"] == "application/json"

    def test_the_connect_timeout_is_ten_seconds(self):
        assert _make_provider()._client.timeout.connect == 10.0

    def test_a_reset_clears_the_schema_applied_flag_and_the_json_mode_fallback(self):
        p = _make_provider()
        p._last_json_schema_applied = True
        p.last_json_mode_fallback = True
        p._reset_per_call_state()
        assert p._last_json_schema_applied is False
        assert p.last_json_mode_fallback is False

class TestUnparseableRateLimitHeadersSurvivors:
    """Headers that cannot be read as a mapping are logged with the reason and leave the previous snapshot in place
    (openai_compat.py 216)."""

    def test_unreadable_headers_are_logged_and_keep_the_previous_snapshot(self, caplog):
        import logging

        p = _make_provider()
        p._capture_rate_limit_headers({"x-ratelimit-remaining-requests": "7"})
        with caplog.at_level(logging.WARNING):
            p._capture_rate_limit_headers(42)
        assert "Could not parse response headers for rate-limit capture:" in caplog.text
        assert p.last_rate_limits == {"x-ratelimit-remaining-requests": "7"}

class TestRateLimitSnapshotSurvivors:
    """The captured snapshot keeps only the x-ratelimit-/ratelimit- headers, lower-cased, replaces the previous one
    on every response, and check_account_limits reports it with short keys (openai_compat.py 203-242)."""

    def test_only_the_ratelimit_headers_are_kept_and_lower_cased(self):
        p = _make_provider()
        p._capture_rate_limit_headers({"X-RateLimit-Remaining-Requests": "7", "ratelimit-reset": "9", "x-request-id": "abc", "Content-Type": "json"})
        assert p.last_rate_limits == {"x-ratelimit-remaining-requests": "7", "ratelimit-reset": "9"}

    def test_a_response_without_rate_limit_headers_clears_the_previous_snapshot(self):
        p = _make_provider()
        p._capture_rate_limit_headers({"x-ratelimit-remaining-requests": "7"})
        p._capture_rate_limit_headers({"x-request-id": "abc"})
        assert p.last_rate_limits == {}

    def test_account_limits_report_the_raw_headers_and_short_keys(self):
        import asyncio

        p = _make_provider()
        p._capture_rate_limit_headers({"x-ratelimit-remaining-requests": "0", "ratelimit-reset-tokens": "30"})
        out = asyncio.run(p.check_account_limits())
        assert out == {
            "raw": {"x-ratelimit-remaining-requests": "0", "ratelimit-reset-tokens": "30"},
            "remaining_requests": "0",
            "reset_tokens": "30",
        }

    def test_account_limits_without_a_snapshot_name_the_provider_and_the_fix(self):
        import asyncio

        p = _make_provider()
        with pytest.raises(NotImplementedError) as exc:
            asyncio.run(p.check_account_limits())
        assert str(exc.value) == (
            "TestProvider: no rate-limit snapshot captured yet -- issue at least one generate() call first; the headers are "
            "captured automatically. Standalone introspection endpoints are not exposed for regular keys on this provider."
        )
