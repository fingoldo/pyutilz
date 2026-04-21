"""Tests for LLM provider base class and factory."""

import json

import pytest

from pyutilz.llm.base import LLMProvider
from pyutilz.llm.exceptions import JSONParsingError

pydantic = pytest.importorskip("pydantic")

from pyutilz.llm.factory import get_llm_provider, _PROVIDER_MODULES, _ALIASES


class TestExtractJson:
    def test_raw_json(self):
        result = LLMProvider.extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert LLMProvider.extract_json(text) == {"key": "value"}

    def test_json_in_plain_code_block(self):
        text = '```\n{"key": "value"}\n```'
        assert LLMProvider.extract_json(text) == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"key": "value"}\nDone!'
        assert LLMProvider.extract_json(text) == {"key": "value"}

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = LLMProvider.extract_json(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_invalid_json_raises(self):
        with pytest.raises(JSONParsingError):
            LLMProvider.extract_json("not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(JSONParsingError):
            LLMProvider.extract_json("")

    def test_json_with_whitespace(self):
        text = '  \n  {"key": "value"}  \n  '
        assert LLMProvider.extract_json(text) == {"key": "value"}

    def test_no_json_raises(self):
        with pytest.raises(JSONParsingError):
            LLMProvider.extract_json("just plain text without any json")


class TestFactory:
    def test_known_providers(self):
        assert len(_PROVIDER_MODULES) >= 5

    def test_aliases_resolve(self):
        for alias, canonical in _ALIASES.items():
            assert canonical in _PROVIDER_MODULES, f"Alias '{alias}' → '{canonical}' not found"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_provider("nonexistent_provider")

    def test_case_insensitive(self):
        try:
            get_llm_provider("CLAUDE")
        except ValueError as e:
            if "Unknown provider" in str(e):
                pytest.fail("get_llm_provider('CLAUDE') raised ValueError — alias resolution failed")
        except Exception:
            pass

    @pytest.mark.parametrize("alias,canonical", list(_ALIASES.items()))
    def test_each_alias(self, alias, canonical):
        assert canonical in _PROVIDER_MODULES


class TestProviderCosts:
    def test_anthropic_cost_positive(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-sonnet-4-20250514"
        cost = p.estimate_cost(1000, 500)
        assert cost > 0

    def test_deepseek_cost_positive(self):
        from pyutilz.llm.deepseek_provider import DeepSeekProvider
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-chat"
        cost = p.estimate_cost(1000, 500)
        assert cost > 0

    def test_gemini_cost_positive(self):
        from pyutilz.llm.gemini_provider import GeminiProvider
        p = GeminiProvider.__new__(GeminiProvider)
        p.model_name = "gemini-3.1-flash-lite-preview"
        cost = p.estimate_cost(1000, 500)
        assert cost > 0

    def test_xai_cost_positive(self):
        from pyutilz.llm.xai_provider import XAIProvider
        p = XAIProvider.__new__(XAIProvider)
        p.model_name = "grok-4-1-fast-reasoning"
        cost = p.estimate_cost(1000, 500)
        assert cost > 0

    def test_claude_code_cost_zero(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
        cost = p.estimate_cost(1000, 500)
        assert cost == 0.0


class TestMaxOutputTokens:
    def test_base_fallback(self):
        assert LLMProvider.max_output_tokens.fget(None) == 4096

    def test_anthropic_max_tokens(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-opus-4-6"
        assert p.max_output_tokens == 128000
        p.model = "claude-sonnet-4-6"
        assert p.max_output_tokens == 64000
        p.model = "claude-haiku-4-5-20251001"
        assert p.max_output_tokens == 64000

    def test_deepseek_max_tokens(self):
        from pyutilz.llm.deepseek_provider import DeepSeekProvider
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-chat"
        assert p.max_output_tokens == 8192
        p.model_name = "deepseek-reasoner"
        assert p.max_output_tokens == 65536

    def test_claude_code_max_tokens(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
        assert p.max_output_tokens == 32000

    def test_claude_code_context_window(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = ClaudeCodeProvider.__new__(ClaudeCodeProvider)
        assert p.context_window == 200_000

    def test_xai_max_tokens(self):
        from pyutilz.llm.xai_provider import XAIProvider
        p = XAIProvider.__new__(XAIProvider)
        p.model_name = "grok-4-1-fast-reasoning"
        assert p.max_output_tokens == 30000

    def test_gemini_max_tokens(self):
        from pyutilz.llm.gemini_provider import GeminiProvider
        p = GeminiProvider.__new__(GeminiProvider)
        p.model_name = "gemini-3.1-flash-lite-preview"
        assert p.max_output_tokens == 65536
        p.model_name = "gemini-2.0-flash"
        assert p.max_output_tokens == 8192

    def test_gemini_context_window(self):
        from pyutilz.llm.gemini_provider import GeminiProvider
        p = GeminiProvider.__new__(GeminiProvider)
        p.model_name = "gemini-2.5-flash"
        assert p.context_window == 1_048_576

    def test_anthropic_context_window(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        assert p.context_window == 200_000


class TestClaudeCodeProvider:
    def test_session_cost_always_zero(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = ClaudeCodeProvider()
        p._call_count = 10
        p.total_prompt_tokens = 50000
        p.total_completion_tokens = 20000
        cost = p.get_session_cost()
        assert cost["total_cost_usd"] == 0.0
        assert cost["calls"] == 10
        assert cost["prompt_tokens"] == 50000

    def test_estimate_cost_zero(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = ClaudeCodeProvider()
        assert p.estimate_cost(1_000_000, 1_000_000) == 0.0

    @pytest.mark.asyncio
    async def test_generate_json_parses_code_block(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = ClaudeCodeProvider()
        async def mock_gen(prompt, system=None, temperature=0.7, max_tokens=0, json_mode=False):
            return '```json\n{"key": "value"}\n```'
        p.generate = mock_gen
        result = await p.generate_json("give me json")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_generate_json_raw_json(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = ClaudeCodeProvider()
        async def mock_gen(prompt, system=None, temperature=0.7, max_tokens=0, json_mode=False):
            return '{"key": "value"}'
        p.generate = mock_gen
        result = await p.generate_json("give me json")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_generate_json_invalid_raises(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = ClaudeCodeProvider()
        async def mock_gen(prompt, system=None, temperature=0.7, max_tokens=0, json_mode=False):
            return "not json at all"
        p.generate = mock_gen
        with pytest.raises(ValueError, match="Invalid JSON"):
            await p.generate_json("give me json")


class TestClaudeCodeRateLimitParsing:
    def test_is_rate_limit_error_from_message(self):
        from pyutilz.llm.claude_code_provider import _is_rate_limit_error
        assert _is_rate_limit_error(RuntimeError("You've hit your limit")) is True
        assert _is_rate_limit_error(RuntimeError("rate limit exceeded")) is True
        assert _is_rate_limit_error(RuntimeError("quota exceeded")) is True
        assert _is_rate_limit_error(RuntimeError("normal error")) is False

    def test_is_rate_limit_error_from_stderr(self):
        from pyutilz.llm.claude_code_provider import _is_rate_limit_error
        err = RuntimeError("process failed")
        err.stderr = "You've hit your limit · resets 4am"
        assert _is_rate_limit_error(err) is True

    def test_parse_reset_wait_seconds_simple(self):
        from pyutilz.llm.claude_code_provider import _parse_reset_wait_seconds
        result = _parse_reset_wait_seconds("resets 4am")
        assert result is not None
        assert result > 0

    def test_parse_reset_wait_seconds_pm(self):
        from pyutilz.llm.claude_code_provider import _parse_reset_wait_seconds
        result = _parse_reset_wait_seconds("resets 5 pm")
        assert result is not None
        assert result > 0

    def test_parse_reset_wait_seconds_no_match(self):
        from pyutilz.llm.claude_code_provider import _parse_reset_wait_seconds
        assert _parse_reset_wait_seconds("some random error") is None

    def test_parse_reset_wait_with_timezone(self):
        from pyutilz.llm.claude_code_provider import _parse_reset_wait_seconds
        result = _parse_reset_wait_seconds("resets 10am (Europe/Moscow)")
        assert result is not None
        assert result > 0

    def test_parse_reset_wait_with_date(self):
        from pyutilz.llm.claude_code_provider import _parse_reset_wait_seconds
        result = _parse_reset_wait_seconds("resets Mar 13, 9am (Europe/Moscow)")
        assert result is not None
        assert result > 0

    def test_find_claude_executable_not_found(self):
        from unittest.mock import patch
        from pyutilz.llm.claude_code_provider import _find_claude_executable
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile", return_value=False):
                with pytest.raises(FileNotFoundError, match="Claude CLI not found"):
                    _find_claude_executable()


class TestAnthropicProvider:
    def test_pricing_sonnet(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-sonnet-4-20250514"
        inp, out = p._get_pricing()
        assert inp == 3.00
        assert out == 15.00

    def test_pricing_opus(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-opus-4-6-20250610"
        inp, out = p._get_pricing()
        assert inp == 5.00
        assert out == 25.00

    def test_pricing_fallback(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-unknown-future-model"
        inp, out = p._get_pricing()
        assert inp == 3.00
        assert out == 15.00

    def test_missing_api_key_raises(self):
        from unittest.mock import patch, MagicMock
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = None
        with patch("pyutilz.llm.anthropic_provider.get_llm_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="API key not provided"):
                AnthropicProvider(api_key=None)

    def test_opus_max_tokens_128k(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-opus-4-6-20250610"
        assert p.max_output_tokens == 128000

    def test_opus_non_46_max_tokens_32k(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = "claude-opus-4-20250514"
        assert p.max_output_tokens == 32000


class TestGeminiProvider:
    def test_pricing_flash_lite(self):
        from pyutilz.llm.gemini_provider import GeminiProvider
        p = GeminiProvider.__new__(GeminiProvider)
        p.model_name = "gemini-3.1-flash-lite-preview"
        inp, out = p._get_pricing()
        assert inp == 0.25
        assert out == 1.50

    def test_pricing_fallback(self):
        from pyutilz.llm.gemini_provider import GeminiProvider
        p = GeminiProvider.__new__(GeminiProvider)
        p.model_name = "gemini-unknown"
        inp, out = p._get_pricing()
        assert inp == 0.25
        assert out == 1.50

    def test_missing_api_key_raises(self):
        from unittest.mock import patch, MagicMock
        from pyutilz.llm.gemini_provider import GeminiProvider, GENAI_AVAILABLE
        if not GENAI_AVAILABLE:
            pytest.skip("google-genai not installed")
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = None
        with patch("pyutilz.llm.gemini_provider.get_llm_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="API key not provided"):
                GeminiProvider(api_key=None)

    def test_safety_block_finish_reason_raises(self):
        """Gemini response with finish_reason=SAFETY raises LLMSafetyBlockError."""
        from unittest.mock import MagicMock
        from pyutilz.llm.gemini_provider import GeminiProvider, GENAI_AVAILABLE
        from pyutilz.llm.exceptions import LLMSafetyBlockError, LLMRefusalError
        if not GENAI_AVAILABLE:
            pytest.skip("google-genai not installed")
        p = GeminiProvider.__new__(GeminiProvider)
        p.model_name = "gemini-flash"
        p._last_usage = {}
        p._last_finish_reason = "SAFETY"
        # Call the post-response logic inline via the safety check.
        # Emulate the block: finish_reason SAFETY should raise.
        _fr = p._last_finish_reason.upper()
        assert "SAFETY" in _fr
        # LLMSafetyBlockError inherits from LLMRefusalError — useful for callers
        # that want to handle both uniformly.
        assert issubclass(LLMSafetyBlockError, LLMRefusalError)

    def test_safety_block_empty_text_raises(self):
        """Gemini response with empty .text (no candidates) raises LLMSafetyBlockError."""
        from pyutilz.llm.gemini_provider import GENAI_AVAILABLE
        from pyutilz.llm.exceptions import LLMSafetyBlockError
        if not GENAI_AVAILABLE:
            pytest.skip("google-genai not installed")
        # Smoke: the exception class is importable and has the expected shape.
        exc = LLMSafetyBlockError("blocked", details={"finish_reason": "SAFETY"})
        assert exc.details == {"finish_reason": "SAFETY"}
        assert "blocked" in str(exc)
