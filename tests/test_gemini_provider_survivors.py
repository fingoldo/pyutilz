"""Mutation-survivor pins for gemini_provider: each class names the survivors of one code path it kills."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pyutilz.llm.gemini_provider as gemini_provider_module
from pyutilz.llm.exceptions import LLMSafetyBlockError
from pyutilz.llm.gemini_provider import GeminiProvider


def _make_provider(**overrides):
    # GENAI_AVAILABLE is a module-level flag set once at import time from whether the real
    # google-genai package was importable -- False on any CI leg where it's not installed
    # (e.g. an unsupported Python version). Patched True here so these tests exercise
    # GeminiProvider's own logic via a mocked `genai` client regardless of whether the real
    # optional dependency is present in this environment.
    mock_settings = MagicMock()
    mock_settings.gemini_api_key = None
    with patch("pyutilz.llm.gemini_provider.get_llm_settings", return_value=mock_settings), patch("pyutilz.llm.gemini_provider.GENAI_AVAILABLE", True), patch(
        "pyutilz.llm.gemini_provider.genai"
    ) as mock_genai:
        mock_genai.Client.return_value = MagicMock()
        p = GeminiProvider(api_key="test-key", **overrides)  # pragma: allowlist secret -- test placeholder, not a real credential
    return p


class TestRaiseIfBlockedSurvivors:
    """The safety check on a response: a prompt blocked before any candidate raises with its reason and details, a
    response without a block reason passes, and each blocking finish reason (or one containing it) raises with the finish
    reason in its details, while an ordinary finish passes (gemini_provider.py 406-432)."""

    def test_a_prompt_blocked_before_any_candidate_raises_with_its_reason_and_details(self):
        from types import SimpleNamespace

        p = _make_provider()
        p.last_safety_ratings = ["r1"]
        p._last_finish_reason = "PROMPT"
        feedback = SimpleNamespace(block_reason="SAFETY", block_reason_message="unsafe prompt")
        response = SimpleNamespace(candidates=[], prompt_feedback=feedback)
        with pytest.raises(LLMSafetyBlockError, match=r"Gemini blocked the prompt \(block_reason=SAFETY\)") as exc:
            p._raise_if_blocked(response, "")
        assert exc.value.details == {
            "finish_reason": "PROMPT",
            "block_reason": "SAFETY",
            "block_reason_message": "unsafe prompt",
            "safety_ratings": ["r1"],
        }

    def test_a_response_with_no_candidates_and_no_block_reason_passes(self):
        from types import SimpleNamespace

        p = _make_provider()
        response = SimpleNamespace(candidates=[], prompt_feedback=SimpleNamespace(block_reason=None))
        assert p._raise_if_blocked(response, "") is None

    @pytest.mark.parametrize("marker", ["SAFETY", "BLOCK", "PROHIBITED_CONTENT", "RECITATION", "SPII"])
    def test_each_blocking_finish_reason_raises_with_the_finish_reason_in_its_details(self, marker):
        from types import SimpleNamespace

        p = _make_provider()
        p._last_finish_reason = marker
        p.last_safety_ratings = ["r2"]
        response = SimpleNamespace(candidates=[SimpleNamespace()])
        with pytest.raises(LLMSafetyBlockError, match=rf"Gemini blocked response by safety filter \(finish_reason={marker}\)") as exc:
            p._raise_if_blocked(response, marker)
        assert exc.value.details == {"finish_reason": marker, "safety_ratings": ["r2"]}

    def test_a_finish_reason_that_only_contains_a_blocking_word_still_raises(self):
        from types import SimpleNamespace

        p = _make_provider()
        p._last_finish_reason = "BLOCKED_BY_SAFETY_FILTER"
        response = SimpleNamespace(candidates=[SimpleNamespace()])
        with pytest.raises(LLMSafetyBlockError):
            p._raise_if_blocked(response, "BLOCKED_BY_SAFETY_FILTER")

    def test_an_ordinary_finish_reason_passes(self):
        from types import SimpleNamespace

        p = _make_provider()
        response = SimpleNamespace(candidates=[SimpleNamespace()])
        assert p._raise_if_blocked(response, "STOP") is None

class TestSessionCostArithmeticSurvivors:
    """get_session_cost splits the prompt into cache hits and misses (hits capped at the prompt size), prices each part,
    adds reasoning to the output, and reports the surcharge and its call count (gemini_provider.py 322-351)."""

    def _priced(self, **tokens):
        p = _make_provider()
        p.model_name = "m"
        p._get_pricing = lambda: (2.0, 8.0)
        p._CACHE_HIT_COST = {"m": 0.5}
        p.total_prompt_tokens = tokens.get("prompt", 1_000_000)
        p.total_cached_content_tokens = tokens.get("cached", 400_000)
        p.total_completion_tokens = tokens.get("completion", 500_000)
        p.total_reasoning_tokens = tokens.get("reasoning", 0)
        p._call_count = tokens.get("calls", 3)
        p._long_context_surcharge_usd = tokens.get("surcharge", 0.0)
        p._long_context_calls = tokens.get("long_calls", 0)
        return p

    def test_cache_hits_are_priced_at_the_cache_rate_and_the_rest_at_the_input_rate(self):
        summary = self._priced().get_session_cost()
        assert summary["cache_hit_tokens"] == 400_000
        assert summary["cache_miss_tokens"] == 600_000
        assert summary["input_cost_usd"] == pytest.approx(0.6 * 2.0 + 0.4 * 0.5)
        assert summary["output_cost_usd"] == pytest.approx(0.5 * 8.0)
        assert summary["total_cost_usd"] == pytest.approx(0.6 * 2.0 + 0.4 * 0.5 + 0.5 * 8.0)

    def test_cached_tokens_never_exceed_the_prompt(self):
        summary = self._priced(cached=2_000_000).get_session_cost()
        assert summary["cache_hit_tokens"] == 1_000_000
        assert summary["cache_miss_tokens"] == 0
        assert summary["input_cost_usd"] == pytest.approx(1.0 * 0.5)

    def test_reasoning_tokens_are_billed_as_output(self):
        summary = self._priced(reasoning=250_000).get_session_cost()
        assert summary["output_cost_usd"] == pytest.approx((500_000 + 250_000) / 1_000_000 * 8.0)

    def test_the_long_context_surcharge_is_added_to_the_total_and_both_counts_are_reported(self):
        summary = self._priced(surcharge=0.25, calls=7, long_calls=2).get_session_cost()
        assert summary["long_context_surcharge_usd"] == pytest.approx(0.25)
        assert summary["total_cost_usd"] == pytest.approx(0.6 * 2.0 + 0.4 * 0.5 + 0.5 * 8.0 + 0.25)
        assert summary["calls"] == 7
        assert summary["long_context_calls"] == 2

class TestUsageRecordingSurvivors:
    """_record_gemini_usage folds one response's usage into the last-call record and the running totals, counts the call,
    passes input, cached and output-plus-reasoning on to the surcharge, and does nothing without usage metadata
    (gemini_provider.py 384-404)."""

    def test_one_response_is_recorded_as_the_last_call_and_added_to_the_totals(self):
        from types import SimpleNamespace

        p = _make_provider()
        recorded: list = []
        p._add_long_context_surcharge = lambda *args: recorded.append(args)
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=50, thoughts_token_count=10, cached_content_token_count=40)
        p._record_gemini_usage(SimpleNamespace(usage_metadata=usage))
        assert p._last_usage == {"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 10, "cached_content_token_count": 40}
        assert p.last_cached_content_tokens == 40
        assert p.total_cached_content_tokens == 40
        assert p._call_count == 1
        assert p.total_prompt_tokens == 100
        assert p.total_completion_tokens == 50
        assert p.total_reasoning_tokens == 10
        assert recorded == [(100, 40, 60)]

    def test_totals_accumulate_across_responses(self):
        from types import SimpleNamespace

        p = _make_provider()
        p._add_long_context_surcharge = lambda *args: None
        p._record_gemini_usage(
            SimpleNamespace(
                usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50, thoughts_token_count=10, cached_content_token_count=40)
            )
        )
        p._record_gemini_usage(
            SimpleNamespace(
                usage_metadata=SimpleNamespace(prompt_token_count=30, candidates_token_count=5, thoughts_token_count=0, cached_content_token_count=5)
            )
        )
        assert p.last_cached_content_tokens == 5
        assert p.total_cached_content_tokens == 45
        assert p._call_count == 2
        assert p.total_prompt_tokens == 130
        assert p.total_completion_tokens == 55

    def test_a_response_without_usage_metadata_changes_nothing(self):
        from types import SimpleNamespace

        p = _make_provider()
        before = (p._call_count, p.total_prompt_tokens, p._last_usage if hasattr(p, "_last_usage") else None)
        p._record_gemini_usage(SimpleNamespace(usage_metadata=None))
        p._record_gemini_usage(SimpleNamespace())
        assert (p._call_count, p.total_prompt_tokens, p._last_usage if hasattr(p, "_last_usage") else None) == before

    def test_missing_counts_are_recorded_as_zero(self):
        from types import SimpleNamespace

        p = _make_provider()
        p._add_long_context_surcharge = lambda *args: None
        p._record_gemini_usage(SimpleNamespace(usage_metadata=SimpleNamespace()))
        assert p._last_usage == {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_content_token_count": 0}
        assert p._call_count == 1

class TestEstimateCostThresholdSurvivors:
    """estimate_cost prices at the base rate up to and including the long-context threshold, switches to the tier above it,
    and a model without a tier always uses the base rate (gemini_provider.py 286-291)."""

    def test_a_model_without_a_long_context_tier_is_priced_at_the_base_rate(self):
        p = _make_provider()
        p.model_name = "gemini-2.5-flash"
        in_rate, out_rate = p._get_pricing()
        assert p.estimate_cost(1_000_000, 1_000_000) == pytest.approx(in_rate + out_rate)

    def test_the_threshold_itself_is_still_priced_at_the_base_rate(self):
        p = _make_provider()
        p.model_name = "gemini-2.5-pro"
        in_rate, _out_rate = p._get_pricing()
        assert p.estimate_cost(200_000, 0) == pytest.approx(200_000 / 1_000_000 * in_rate)

    def test_one_token_above_the_threshold_is_priced_at_the_tier(self):
        from pyutilz.llm.gemini_provider import price_call

        p = _make_provider()
        p.model_name = "gemini-2.5-pro"
        in_rate, _out_rate = p._get_pricing()
        base = 200_001 / 1_000_000 * in_rate
        assert p.estimate_cost(200_001, 0) == pytest.approx(sum(price_call(p._current_pricing(), 200_001, 0)))
        assert p.estimate_cost(200_001, 0) != pytest.approx(base)

class TestBatchExceptionClassificationSurvivors:
    """A safety-block exception is tagged for batch callers; any other exception, including a base-class exception that is
    not a safety block, is not tagged (gemini_provider.py 597-604)."""

    def test_a_safety_block_is_tagged_as_a_safety_block(self):
        from pyutilz.llm.exceptions import LLMSafetyBlockError

        p = _make_provider()
        assert p._classify_batch_exception(LLMSafetyBlockError("blocked")) == {"error_type": "safety_block"}

    def test_an_ordinary_exception_is_not_tagged(self):
        p = _make_provider()
        assert p._classify_batch_exception(ValueError("boom")) is None
        assert p._classify_batch_exception(RuntimeError("boom")) is None

class TestGenerateConfigSurvivors:
    """_build_generate_config sends only the options the caller set: sampling, output cap, system text, JSON constraints,
    candidate count, cached content and thinking, and leaves each out when it is absent (gemini_provider.py 353-382)."""

    def _build(self, monkeypatch, **overrides):
        from pyutilz.llm import gemini_provider as gp

        monkeypatch.setattr(gp.types, "GenerateContentConfig", lambda **kw: dict(kw))
        monkeypatch.setattr(gp.types, "ThinkingConfig", lambda **kw: {"tc": kw})
        monkeypatch.setattr(gp, "gemini_thinking_config", lambda thinking, model: None)
        p = _make_provider()
        p._candidate_count = overrides.pop("candidate_count", 1)
        p._cached_content = overrides.pop("cached_content", None)
        args = dict(max_tokens=0, temperature=0.7, system=None, json_mode=False, json_schema=None, thinking=None)
        args.update(overrides)
        return p._build_generate_config(**args)

    def test_a_default_call_sends_only_temperature_and_empty_system_and_mime(self, monkeypatch):
        cfg = self._build(monkeypatch)
        assert cfg == {"temperature": 0.7, "system_instruction": None, "response_mime_type": None}

    def test_an_absent_temperature_is_left_to_the_model(self, monkeypatch):
        assert "temperature" not in self._build(monkeypatch, temperature=None)

    def test_a_positive_output_cap_is_sent_and_a_zero_or_absent_one_is_not(self, monkeypatch):
        assert self._build(monkeypatch, max_tokens=128)["max_output_tokens"] == 128
        assert "max_output_tokens" not in self._build(monkeypatch, max_tokens=0)
        assert "max_output_tokens" not in self._build(monkeypatch, max_tokens=None)

    def test_a_system_text_is_sent_as_the_system_instruction(self, monkeypatch):
        assert self._build(monkeypatch, system="be brief")["system_instruction"] == "be brief"
        assert self._build(monkeypatch, system="")["system_instruction"] is None

    def test_json_mode_or_a_schema_asks_for_json_output(self, monkeypatch):
        assert self._build(monkeypatch, json_mode=True)["response_mime_type"] == "application/json"
        assert self._build(monkeypatch, json_schema={"type": "object"})["response_mime_type"] == "application/json"

    def test_a_named_schema_sends_its_inner_schema(self, monkeypatch):
        inner = {"type": "object", "properties": {"a": {"type": "string"}}}
        cfg = self._build(monkeypatch, json_schema={"name": "Out", "schema": inner})
        assert cfg["response_json_schema"] == inner

    def test_more_than_one_candidate_is_sent_and_one_is_left_to_the_default(self, monkeypatch):
        assert self._build(monkeypatch, candidate_count=3)["candidate_count"] == 3
        assert "candidate_count" not in self._build(monkeypatch, candidate_count=1)

    def test_cached_content_is_referenced_only_when_set(self, monkeypatch):
        assert self._build(monkeypatch, cached_content="cache/1")["cached_content"] == "cache/1"
        assert "cached_content" not in self._build(monkeypatch)

    def test_thinking_settings_become_a_thinking_config_only_when_given(self, monkeypatch):
        from pyutilz.llm import gemini_provider as gp

        monkeypatch.setattr(gp.types, "GenerateContentConfig", lambda **kw: dict(kw))
        monkeypatch.setattr(gp.types, "ThinkingConfig", lambda **kw: {"tc": kw})
        monkeypatch.setattr(gp, "gemini_thinking_config", lambda thinking, model: {"thinking_budget": 512})
        p = _make_provider()
        p._candidate_count = 1
        p._cached_content = None
        cfg = p._build_generate_config(max_tokens=0, temperature=0.7, system=None, json_mode=False, json_schema=None, thinking=True)
        assert cfg["thinking_config"] == {"tc": {"thinking_budget": 512}}

class TestSessionCostDefaultsAndKeysSurvivors:
    """get_session_cost with exact tolerance on the per-million prices, the keys of its token counts, and the defaults used
    when the surcharge or the long-context call count was never recorded (gemini_provider.py 329-349)."""

    def _priced(self):
        p = _make_provider()
        p.model_name = "m"
        p._get_pricing = lambda: (2.0, 8.0)
        p._CACHE_HIT_COST = {"m": 0.5}
        p.total_prompt_tokens = 1_000_000
        p.total_cached_content_tokens = 400_000
        p.total_completion_tokens = 500_000
        p.total_reasoning_tokens = 250_000
        p._call_count = 3
        return p

    def test_the_per_million_arithmetic_is_exact(self):
        summary = self._priced().get_session_cost()
        assert summary["input_cost_usd"] == pytest.approx(0.6 * 2.0 + 0.4 * 0.5, rel=1e-9)
        assert summary["output_cost_usd"] == pytest.approx(0.75 * 8.0, rel=1e-9)

    def test_the_token_counts_are_reported_under_their_names(self):
        summary = self._priced().get_session_cost()
        assert summary["prompt_tokens"] == 1_000_000
        assert summary["completion_tokens"] == 500_000
        assert summary["reasoning_tokens"] == 250_000

    def test_a_missing_surcharge_and_call_count_default_to_zero(self):
        p = self._priced()
        summary = p.get_session_cost()
        assert summary["long_context_surcharge_usd"] == 0.0
        assert summary["long_context_calls"] == 0

class _ResponseWithText:
    """A response whose text property either returns a value or raises, as the SDK's does for a blocked candidate."""

    def __init__(self, candidates, text=None, text_error=None):
        self.candidates = candidates
        self._text = text
        self._text_error = text_error

    @property
    def text(self):
        if self._text_error is not None:
            raise self._text_error
        return self._text

class TestGenerateOutcomeSurvivors:
    """generate turns the response into text or a typed error: a truncation keeps the partial text and logs a failed salvage,
    empty or unreadable text raises a safety error with its details, and a response without candidates records 'unknown'
    (gemini_provider.py 435-545). The per-call finish reason is read inside the coroutine, where it is set."""

    def _run(self, p, response):
        import asyncio
        from unittest.mock import AsyncMock

        p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=response))))

        async def _go():
            return await p.generate("hi"), p._last_finish_reason

        return asyncio.run(_go())

    def _candidates(self, finish="STOP"):
        return [SimpleNamespace(finish_reason=finish)]

    def test_a_plain_answer_is_returned_and_its_finish_reason_recorded(self):
        p = _make_provider()
        text, finish = self._run(p, _ResponseWithText(self._candidates("STOP"), text="answer"))
        assert text == "answer"
        assert finish == "STOP"

    def test_a_response_without_candidates_records_unknown_and_still_returns_text(self):
        p = _make_provider()
        text, finish = self._run(p, _ResponseWithText([], text="answer"))
        assert text == "answer"
        assert finish == "unknown"

    def test_a_truncated_answer_raises_with_its_partial_text(self):
        from pyutilz.llm.exceptions import LLMTruncationError

        p = _make_provider()
        with pytest.raises(LLMTruncationError, match=r"Gemini response truncated by max_tokens \(finish_reason=MAX_TOKENS\)") as exc:
            self._run(p, _ResponseWithText(self._candidates("MAX_TOKENS"), text="partial"))
        assert exc.value.partial_text == "partial"
        assert exc.value.finish_reason == "MAX_TOKENS"

    def test_a_truncated_answer_whose_text_cannot_be_read_is_logged_and_gives_empty_partial(self, caplog):
        import logging

        from pyutilz.llm.exceptions import LLMTruncationError

        p = _make_provider()
        with caplog.at_level(logging.WARNING):
            with pytest.raises(LLMTruncationError) as exc:
                self._run(p, _ResponseWithText(self._candidates("MAX_TOKENS"), text_error=ValueError("no text")))
        assert exc.value.partial_text == ""
        assert "Could not read partial text off a truncated Gemini response; reporting it as empty: no text" in caplog.text

    def test_an_empty_answer_raises_a_safety_block_with_its_details(self):
        from pyutilz.llm.exceptions import LLMSafetyBlockError

        p = _make_provider()
        with pytest.raises(LLMSafetyBlockError, match=r"Gemini returned empty text \(likely safety block\)") as exc:
            self._run(p, _ResponseWithText(self._candidates("STOP"), text=""))
        assert exc.value.details == {"finish_reason": "STOP", "safety_ratings": []}

    def test_an_unreadable_answer_raises_a_safety_block_naming_the_cause(self):
        from pyutilz.llm.exceptions import LLMSafetyBlockError

        p = _make_provider()
        with pytest.raises(LLMSafetyBlockError, match=r"Gemini response has no text \(likely safety block\): no text here") as exc:
            self._run(p, _ResponseWithText(self._candidates("STOP"), text_error=ValueError("no text here")))
        assert exc.value.details["finish_reason"] == "STOP"

    def test_an_unreadable_answer_carries_the_safety_ratings_of_its_candidate(self):
        from pyutilz.llm.exceptions import LLMSafetyBlockError

        p = _make_provider()
        ratings = [SimpleNamespace(category="HARASSMENT", probability="HIGH", blocked=True)]
        candidates = [SimpleNamespace(finish_reason="STOP", safety_ratings=ratings)]
        with pytest.raises(LLMSafetyBlockError) as exc:
            self._run(p, _ResponseWithText(candidates, text_error=ValueError("no text here")))
        assert exc.value.details["safety_ratings"] == [{"category": "HARASSMENT", "probability": "HIGH", "blocked": True}]

class TestWithoutGenaiInstalled:
    """With google.genai absent the module still imports: GENAI_AVAILABLE is False and no error is ever classed as
    retryable (gemini_provider.py 36, 73). Checked in a fresh interpreter, because the flag is set at import time."""

    def test_the_module_imports_without_genai_and_reports_it_unavailable(self):
        import subprocess
        import sys
        from pathlib import Path

        src = Path(gemini_provider_module.__file__).resolve().parents[2]
        code = (
            "import sys\n"
            "sys.modules['google.genai'] = None\n"
            "from pyutilz.llm import gemini_provider as g\n"
            "print(g.GENAI_AVAILABLE)\n"
            "print(g._is_retryable_genai_error(RuntimeError('429')))\n"
        )
        env = {**__import__("os").environ, "PYTHONPATH": str(src)}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["False", "False"]

class TestGenerateCallArgumentsSurvivors:
    """generate passes its options to the config builder in order, sends the prompt as the contents when there are no
    images, records 'unknown' for a candidate with no finish reason, and captures the candidate's safety ratings
    (gemini_provider.py 435-492)."""

    def _call(self, monkeypatch, response=None, **kwargs):
        import asyncio
        from unittest.mock import AsyncMock

        p = _make_provider()
        recorded: list = []

        def _build(*args):
            recorded.append(args)
            return "CFG"

        monkeypatch.setattr(p, "_build_generate_config", _build)
        monkeypatch.setattr(type(p), "fit_max_tokens_to_context", lambda self, max_tokens, prompt, system=None: max_tokens)
        resp = response or _ResponseWithText([SimpleNamespace(finish_reason="STOP")], text="ok")
        mock = AsyncMock(return_value=resp)
        p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock)))

        async def _go():
            out = await p.generate(**kwargs)
            return out, p.last_safety_ratings, p._last_finish_reason

        out = asyncio.run(_go())
        return out, recorded, mock, p

    def test_the_config_builder_receives_the_options_in_their_order(self, monkeypatch):
        _out, recorded, _mock, _p = self._call(
            monkeypatch,
            prompt="hi",
            system="s",
            temperature=0.2,
            max_tokens=50,
            json_mode=True,
            thinking="high",
            json_schema={"type": "object"},
        )
        assert recorded == [(50, 0.2, "s", True, {"type": "object"}, "high")]

    def test_a_zero_output_cap_is_passed_on_as_no_budget(self, monkeypatch):
        _out, recorded, _mock, _p = self._call(monkeypatch, prompt="hi", max_tokens=0)
        assert recorded[0][0] is None

    def test_the_prompt_is_sent_as_plain_contents_with_the_built_config(self, monkeypatch):
        _out, _recorded, mock, p = self._call(monkeypatch, prompt="hi")
        mock.assert_awaited_once()
        kwargs = mock.await_args.kwargs
        assert kwargs["contents"] == "hi"
        assert kwargs["config"] == "CFG"
        assert kwargs["model"] == p.model_name

    def test_a_candidate_without_a_finish_reason_is_recorded_as_unknown(self, monkeypatch):
        response = _ResponseWithText([SimpleNamespace()], text="ok")
        (text, ratings, finish), _recorded, _mock, _p = self._call(monkeypatch, response=response, prompt="hi")
        assert text == "ok"
        assert finish == "unknown"

    def test_the_candidate_safety_ratings_are_captured(self, monkeypatch):
        candidate = SimpleNamespace(
            finish_reason="STOP",
            safety_ratings=[SimpleNamespace(category="HARM_CATEGORY_HATE_SPEECH", probability="LOW", blocked=False)],
        )
        response = _ResponseWithText([candidate], text="ok")
        (text, ratings, _finish), _recorded, _mock, _p = self._call(monkeypatch, response=response, prompt="hi")
        assert ratings == [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "LOW", "blocked": False}]

class TestCountTokensAndUnsupportedAccountSurvivors:
    """count_tokens reports the API's total, and falls back to the local token counter with a debug note when the API gives
    no total or fails; the two account methods always raise with their explanations (gemini_provider.py 606-641)."""

    def _provider_with_count(self, result=None, error=None):
        from unittest.mock import AsyncMock

        p = _make_provider()
        mock = AsyncMock(return_value=result, side_effect=error)
        p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(count_tokens=mock)))
        return p

    def test_the_api_total_is_reported_as_an_integer(self):
        import asyncio

        p = self._provider_with_count(result=SimpleNamespace(total_tokens="42"))
        assert asyncio.run(p.count_tokens("hi")) == 42

    def test_a_missing_total_falls_back_to_the_local_counter_with_a_debug_note(self, monkeypatch, caplog):
        import asyncio
        import logging

        from pyutilz.llm import token_counter

        monkeypatch.setattr(token_counter, "count_tokens", lambda text: 7)
        p = self._provider_with_count(result=SimpleNamespace(total_tokens=None))
        with caplog.at_level(logging.DEBUG):
            assert asyncio.run(p.count_tokens("hi")) == 7
        assert "Gemini count_tokens API failed (Gemini count_tokens returned no total_tokens); falling back to tiktoken." in caplog.text

    def test_an_api_failure_falls_back_to_the_local_counter_and_names_the_error(self, monkeypatch, caplog):
        import asyncio
        import logging

        from pyutilz.llm import token_counter

        monkeypatch.setattr(token_counter, "count_tokens", lambda text: 9)
        p = self._provider_with_count(error=RuntimeError("network down"))
        with caplog.at_level(logging.DEBUG):
            assert asyncio.run(p.count_tokens("hi")) == 9
        assert "Gemini count_tokens API failed (network down)" in caplog.text

    def test_the_account_balance_is_refused_with_the_billing_explanation(self):
        import asyncio

        p = _make_provider()
        with pytest.raises(NotImplementedError) as exc:
            asyncio.run(p.get_account_credits())
        assert str(exc.value) == (
            "Gemini billing is GCP-based — there is no balance endpoint "
            "reachable with a Gemini API key. Check console.cloud.google.com/billing "
            "or use the Cloud Billing API with separate GCP credentials."
        )

    def test_the_account_limits_are_refused_with_the_quota_explanation(self):
        import asyncio

        p = _make_provider()
        with pytest.raises(NotImplementedError) as exc:
            asyncio.run(p.check_account_limits())
        assert str(exc.value).startswith("Gemini does not expose per-key rate limits via API. ")
        assert "console.cloud.google.com/iam-admin/quotas" in str(exc.value)

class TestSchemaAndCandidateCaptureSurvivors:
    """A bare schema is sent whole and a wrapper with only a strict flag sends its inner schema; the candidate's function
    calls, grounding, citations and all candidates are captured, and a failure to read them is logged at debug
    (gemini_provider.py 368-371, 575-606)."""

    def _config_for(self, monkeypatch, json_schema):
        from pyutilz.llm import gemini_provider as gp

        monkeypatch.setattr(gp.types, "GenerateContentConfig", lambda **kw: dict(kw))
        monkeypatch.setattr(gp, "gemini_thinking_config", lambda thinking, model: None)
        p = _make_provider()
        p._candidate_count = 1
        p._cached_content = None
        return p._build_generate_config(0, 0.7, None, False, json_schema, None)

    def test_a_bare_schema_is_sent_whole(self, monkeypatch):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        assert self._config_for(monkeypatch, schema)["response_json_schema"] == schema

    def test_a_schema_with_a_schema_key_but_no_wrapper_flags_is_sent_whole(self, monkeypatch):
        bare = {"schema": {"type": "string"}}
        assert self._config_for(monkeypatch, bare)["response_json_schema"] == bare

    def test_two_candidates_are_requested_in_the_config(self, monkeypatch):
        from pyutilz.llm import gemini_provider as gp

        monkeypatch.setattr(gp.types, "GenerateContentConfig", lambda **kw: dict(kw))
        monkeypatch.setattr(gp, "gemini_thinking_config", lambda thinking, model: None)
        p = _make_provider(candidate_count=2)
        p._cached_content = None
        assert p._build_generate_config(0, 0.7, None, False, None, None)["candidate_count"] == 2

    def test_a_wrapper_with_only_a_strict_flag_sends_its_inner_schema(self, monkeypatch):
        inner = {"type": "object"}
        cfg = self._config_for(monkeypatch, {"schema": inner, "strict": True})
        assert cfg["response_json_schema"] == inner

    def test_function_calls_grounding_and_all_candidates_are_captured(self, monkeypatch):
        p = _make_provider()
        grounding = SimpleNamespace(web="g")
        citation = SimpleNamespace(cite="c")
        call = SimpleNamespace(name="lookup", args={"q": "x"})
        candidate = SimpleNamespace(
            finish_reason="STOP",
            grounding_metadata=grounding,
            citation_metadata=citation,
            content=SimpleNamespace(parts=[SimpleNamespace(function_call=call), SimpleNamespace(function_call=None)]),
        )
        candidates = [candidate, SimpleNamespace()]
        p._capture_candidate_metadata(SimpleNamespace(candidates=candidates))
        assert p.last_grounding_metadata is grounding
        assert p.last_citation_metadata is citation
        assert p.last_function_calls == [{"name": "lookup", "args": {"q": "x"}}]
        assert p.last_all_candidates == candidates

    def test_a_failure_to_read_the_candidates_is_logged_at_debug(self, caplog):
        import logging

        class _Broken:
            @property
            def candidates(self):
                raise ValueError("unreadable")

        p = _make_provider()
        with caplog.at_level(logging.DEBUG):
            p._capture_candidate_metadata(_Broken())
        assert "Gemini candidate metadata capture failed: unreadable" in caplog.text

class TestGenerateJsonForwardingSurvivors:
    """generate_json forwards its options to the JSON path in a fixed order: the defaults are a 0.3 temperature and no cap,
    images are sent only when non-empty, thinking whenever it is set (including False), and the schema when given
    (gemini_provider.py 293-320)."""

    def _record(self, monkeypatch, **kwargs):
        import asyncio

        p = _make_provider()
        calls: list = []

        async def _via(prompt, system, temperature, max_tokens, json_mode=False, **extra):
            calls.append((prompt, system, temperature, max_tokens, json_mode, extra))
            return {"ok": True}

        monkeypatch.setattr(p, "_generate_json_via", _via)
        out = asyncio.run(p.generate_json("hi", **kwargs))
        assert out == {"ok": True}
        return calls[0]

    def test_the_defaults_are_a_point_three_temperature_and_no_cap_and_json_mode(self, monkeypatch):
        prompt, system, temperature, max_tokens, json_mode, extra = self._record(monkeypatch)
        assert (prompt, system, temperature, max_tokens, json_mode, extra) == ("hi", None, 0.3, 0, True, {})

    def test_an_empty_image_list_is_not_forwarded(self, monkeypatch):
        *_rest, extra = self._record(monkeypatch, images=[])
        assert extra == {}

    def test_images_thinking_and_schema_are_forwarded_under_their_names(self, monkeypatch):
        schema = {"type": "object"}
        *_rest, extra = self._record(monkeypatch, images=["data:x"], thinking="high", json_schema=schema)
        assert extra == {"images": ["data:x"], "thinking": "high", "json_schema": schema}

    def test_a_thinking_setting_of_false_is_still_forwarded(self, monkeypatch):
        *_rest, extra = self._record(monkeypatch, thinking=False)
        assert extra == {"thinking": False}

    def test_the_sampling_arguments_keep_their_positions(self, monkeypatch):
        prompt, system, temperature, max_tokens, *_rest = self._record(monkeypatch, system="be brief", temperature=0.9)
        assert (prompt, system, temperature, max_tokens) == ("hi", "be brief", 0.9, 0)

class TestModelFamilyLimitsSurvivors:
    """Each model family reports its exact context window and output ceiling, a model outside the families reports the
    default window and no ceiling, and the first matching prefix wins (gemini_provider.py 214-233)."""

    @pytest.mark.parametrize(
        ("model", "window", "output"),
        [
            ("gemini-2.5-flash", 1_048_576, 65536),
            ("gemini-2.0-flash", 1_048_576, 8192),
            ("gemini-3-pro-preview", 1_048_576, 65536),
        ],
    )
    def test_each_family_reports_its_window_and_output_ceiling(self, model, window, output):
        p = _make_provider()
        p.model_name = model
        assert p.context_window == window
        assert p.max_output_tokens == output

    def test_a_model_outside_the_families_reports_the_default_window_and_no_ceiling(self):
        p = _make_provider()
        p.model_name = "gemini-1.5-pro"
        assert p.context_window == 1_048_576
        assert p.max_output_tokens is None

    def test_the_first_matching_prefix_wins_and_the_default_is_used_when_none_matches(self):
        p = _make_provider()
        table = (("ab", 1), ("a", 2))
        p.model_name = "abc"
        assert p._family_limit(table, 9) == 1
        p.model_name = "ax"
        assert p._family_limit(table, 9) == 2
        p.model_name = "z"
        assert p._family_limit(table, 9) == 9

class TestLongContextSurchargeAccumulationSurvivors:
    """The surcharge is charged only above the threshold on a model with a tier, starts from zero and accumulates across
    calls, and counts each charged call (gemini_provider.py 277-284)."""

    def _provider(self, monkeypatch, model="gemini-2.5-pro"):
        from pyutilz.llm import gemini_provider as gp

        monkeypatch.setattr(gp, "long_context_surcharge", lambda pricing, prompt, output, cache_read=0: (0.5, 0.25))
        p = _make_provider()
        p.model_name = model
        return p

    def test_the_first_charged_call_starts_from_zero_and_counts_one(self, monkeypatch):
        p = self._provider(monkeypatch)
        p._add_long_context_surcharge(250_000, 0, 10)
        assert p._long_context_surcharge_usd == pytest.approx(0.75, rel=1e-9)
        assert p._long_context_calls == 1

    def test_later_charged_calls_accumulate(self, monkeypatch):
        p = self._provider(monkeypatch)
        p._add_long_context_surcharge(250_000, 0, 10)
        p._add_long_context_surcharge(300_000, 0, 10)
        assert p._long_context_surcharge_usd == pytest.approx(1.5, rel=1e-9)
        assert p._long_context_calls == 2

    def test_a_call_at_or_below_the_threshold_is_not_charged(self, monkeypatch):
        p = self._provider(monkeypatch)
        p._add_long_context_surcharge(200_000, 0, 10)
        assert p._long_context_surcharge_usd == 0.0
        assert p._long_context_calls == 0

    def test_a_model_without_a_tier_is_never_charged(self, monkeypatch):
        p = self._provider(monkeypatch, model="gemini-2.5-flash")
        p._add_long_context_surcharge(250_000, 0, 10)
        assert p._long_context_surcharge_usd == 0.0

class TestSafetyRatingDefaultsSurvivors:
    """A safety rating that carries no blocked flag is recorded as not blocked (gemini_provider.py 571)."""

    def test_a_rating_without_a_blocked_flag_is_recorded_as_not_blocked(self):
        import asyncio
        from unittest.mock import AsyncMock

        p = _make_provider()
        candidate = SimpleNamespace(finish_reason="STOP", safety_ratings=[SimpleNamespace(category="HARASSMENT", probability="LOW")])
        response = _ResponseWithText([candidate], text="ok")
        p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=response))))

        async def _go():
            await p.generate("hi")
            return p.last_safety_ratings

        assert asyncio.run(_go()) == [{"category": "HARASSMENT", "probability": "LOW", "blocked": False}]

class TestGenerateDefaultsAndBudgetSurvivors:
    """generate's own defaults reach the config builder unchanged (temperature 0.7, no cap, no JSON mode), a cap of one token
    is a real cap, and the budget is fitted with the prompt and system in that order (gemini_provider.py 435-470)."""

    def _run(self, monkeypatch, **kwargs):
        import asyncio
        from unittest.mock import AsyncMock

        p = _make_provider()
        built: list = []
        fitted: list = []

        def _build(*args):
            built.append(args)
            return "CFG"

        def _fit(self, max_tokens, prompt, system=None):
            fitted.append((max_tokens, prompt, system))
            return max_tokens

        monkeypatch.setattr(p, "_build_generate_config", _build)
        monkeypatch.setattr(type(p), "fit_max_tokens_to_context", _fit)
        resp = _ResponseWithText([SimpleNamespace(finish_reason="STOP")], text="ok")
        p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=resp))))
        asyncio.run(p.generate(**kwargs))
        return built, fitted

    def test_the_defaults_reach_the_config_builder_unchanged(self, monkeypatch):
        built, _fitted = self._run(monkeypatch, prompt="hi")
        assert built == [(None, 0.7, None, False, None, None)]

    def test_a_cap_of_one_token_is_kept(self, monkeypatch):
        built, _fitted = self._run(monkeypatch, prompt="hi", max_tokens=1)
        assert built[0][0] == 1

    def test_the_budget_is_fitted_with_the_prompt_then_the_system(self, monkeypatch):
        _built, fitted = self._run(monkeypatch, prompt="hi", system="s", max_tokens=5)
        assert fitted == [(5, "hi", "s")]

class TestProviderIdentityAndTimeoutSurvivors:
    """The provider names itself Gemini in its messages, and every request is bounded by its ten-minute timeout, since a
    thinking model can take minutes (gemini_provider.py 94, 155)."""

    def test_the_provider_is_named_gemini(self):
        assert GeminiProvider._provider_name == "Gemini"

    def test_a_request_is_bounded_by_ten_minutes(self):
        assert GeminiProvider._request_timeout_seconds == 600.0

class TestReasoningTotalsAndCallResetSurvivors:
    """Reasoning tokens accumulate across calls, and each generate call starts from a clean per-call state, so the last
    call's usage does not leak into the next call (gemini_provider.py 399, 468)."""

    def _generate(self, p, response):
        import asyncio
        from unittest.mock import AsyncMock

        p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=response))))

        async def _go():
            await p.generate("hi")
            return p._last_usage, p.total_reasoning_tokens

        return asyncio.run(_go())

    def _usage_response(self, thoughts):
        usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=1, thoughts_token_count=thoughts, cached_content_token_count=0)
        return SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")], text="ok", usage_metadata=usage)

    def test_a_failed_call_does_not_keep_the_safety_ratings_of_the_previous_call(self):
        import asyncio

        p = _make_provider()
        rated = SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason="STOP", safety_ratings=[SimpleNamespace(category="HARASSMENT", probability="LOW", blocked=False)])],
            text="ok",
            usage_metadata=None,
        )
        p.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(side_effect=[rated, RuntimeError("transport down")]))))

        async def _go():
            # PerCallAttr state is context-local: both calls and the reads happen inside one coroutine.
            await p.generate("hi")
            after_ok = list(p.last_safety_ratings)
            with pytest.raises(RuntimeError, match="transport down"):
                await p.generate("hi")
            return after_ok, list(p.last_safety_ratings)

        after_ok, after_failed = asyncio.run(_go())
        assert after_ok == [{"category": "HARASSMENT", "probability": "LOW", "blocked": False}]
        assert after_failed == []

    def test_reasoning_tokens_accumulate_across_calls(self):
        p = _make_provider()
        self._generate(p, self._usage_response(thoughts=10))
        _last, total = self._generate(p, self._usage_response(thoughts=4))
        assert total == 14

    def test_a_call_without_usage_does_not_inherit_the_previous_call_usage(self):
        p = _make_provider()
        self._generate(p, self._usage_response(thoughts=10))
        no_usage = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")], text="ok", usage_metadata=None)
        last, _total = self._generate(p, no_usage)
        assert last["reasoning_tokens"] == 0
        assert last["input_tokens"] == 0
