"""Tests for unified ``get_account_credits`` / ``check_account_limits`` API.

Covers:
  - Base class default raises NotImplementedError for both methods
  - DeepSeek's real /user/balance implementation (mocked HTTP)
  - Per-provider stub messages name the right console URL (helps users)

OpenRouter has its own dedicated test file; this one focuses on the
shared interface and providers WITHOUT a public balance endpoint.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("pydantic")

from pyutilz.llm.base import LLMProvider


class _DummyProvider(LLMProvider):
    """Minimal concrete provider that doesn't override the new methods —
    proves the BASE class default is in effect."""

    async def generate(self, prompt, system=None, temperature=0.7, max_tokens=0):
        return ""

    async def generate_json(self, prompt, system=None, temperature=0.3, max_tokens=0):
        return {}

    async def generate_batch(self, requests):
        if False:
            yield {}

    def estimate_cost(self, input_tokens, output_tokens):
        return 0.0

    async def count_tokens(self, text):
        return 0


class TestBaseDefault:
    @pytest.mark.asyncio
    async def test_get_account_credits_raises_not_implemented(self):
        p = _DummyProvider()
        with pytest.raises(NotImplementedError, match="balance"):
            await p.get_account_credits()

    @pytest.mark.asyncio
    async def test_check_account_limits_raises_not_implemented(self):
        p = _DummyProvider()
        with pytest.raises(NotImplementedError, match="limits"):
            await p.check_account_limits()

    @pytest.mark.asyncio
    async def test_default_message_names_class(self):
        # Helpful debugging signal — the error should tell the user WHICH
        # provider they're hitting the limitation on, not a generic
        # "not implemented".
        p = _DummyProvider()
        with pytest.raises(NotImplementedError, match="_DummyProvider"):
            await p.get_account_credits()


class TestDeepSeekBalance:
    """DeepSeek's /user/balance is one of the few real implementations."""

    def _make(self):
        from pyutilz.llm.deepseek_provider import DeepSeekProvider
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.model_name = "deepseek-v4-flash"
        return p

    @pytest.mark.asyncio
    async def test_balance_normalizes_usd_entry(self):
        p = self._make()
        body = {
            "is_available": True,
            "balance_infos": [
                {
                    "currency": "USD",
                    "total_balance": "12.34",
                    "granted_balance": "5.00",
                    "topped_up_balance": "7.34",
                },
                {
                    "currency": "CNY",
                    "total_balance": "0.00",
                    "granted_balance": "0.00",
                    "topped_up_balance": "0.00",
                },
            ],
        }
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://api.deepseek.com/user/balance"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.get_account_credits()

        p._client.get.assert_awaited_once_with("/user/balance")
        assert out["balance_usd"] == pytest.approx(12.34)
        assert out["total_granted"] == pytest.approx(5.00)
        assert out["total_topped_up"] == pytest.approx(7.34)
        assert out["currency"] == "USD"
        assert out["is_available"] is True
        assert out["raw"] == body

    @pytest.mark.asyncio
    async def test_balance_picks_first_when_no_usd(self):
        # Some accounts may be CNY-only — pick the first entry rather than
        # silently returning None.
        p = self._make()
        body = {
            "is_available": True,
            "balance_infos": [
                {
                    "currency": "CNY",
                    "total_balance": "100.0",
                    "granted_balance": "10.0",
                    "topped_up_balance": "90.0",
                },
            ],
        }
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://api.deepseek.com/user/balance"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.get_account_credits()
        assert out["balance_usd"] == pytest.approx(100.0)  # named for the field, value is CNY
        assert out["currency"] == "CNY"

    @pytest.mark.asyncio
    async def test_balance_handles_empty_infos(self):
        p = self._make()
        body = {"is_available": False, "balance_infos": []}
        resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("GET", "https://api.deepseek.com/user/balance"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        out = await p.get_account_credits()
        assert out["balance_usd"] is None
        assert out["currency"] is None
        assert out["is_available"] is False

    @pytest.mark.asyncio
    async def test_balance_propagates_http_error(self):
        p = self._make()
        resp = httpx.Response(
            status_code=401,
            json={"error": "invalid key"},
            request=httpx.Request("GET", "https://api.deepseek.com/user/balance"),
        )
        p._client = AsyncMock()
        p._client.get = AsyncMock(return_value=resp)

        with pytest.raises(httpx.HTTPStatusError):
            await p.get_account_credits()

    @pytest.mark.asyncio
    async def test_check_account_limits_default_not_implemented(self):
        # DeepSeek doesn't have a rate-limits endpoint — keep base default.
        p = self._make()
        with pytest.raises(NotImplementedError):
            await p.check_account_limits()


class TestProviderStubMessages:
    """Each provider's NotImplementedError must point the user somewhere
    actionable (console URL or the right header to look at). Without this,
    the error would just be a dead-end."""

    def _make(self, cls):
        return cls.__new__(cls)

    @pytest.mark.asyncio
    async def test_anthropic_credits_points_to_console(self):
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = self._make(AnthropicProvider)
        with pytest.raises(NotImplementedError, match="console.anthropic.com"):
            await p.get_account_credits()

    @pytest.mark.asyncio
    async def test_anthropic_limits_no_snapshot_yet_hint(self):
        # Anthropic's check_account_limits returns the captured headers
        # snapshot from the LAST call. Before any call has been made,
        # NotImplementedError fires with a hint to issue a request first.
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = self._make(AnthropicProvider)
        with pytest.raises(NotImplementedError, match="generate"):
            await p.check_account_limits()

    @pytest.mark.asyncio
    async def test_anthropic_limits_returns_captured_headers(self):
        # After capture, returns structured dict.
        from pyutilz.llm.anthropic_provider import AnthropicProvider
        p = self._make(AnthropicProvider)
        p.last_rate_limits = {
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-requests-remaining": "47",
            "anthropic-ratelimit-tokens-limit": "100000",
        }
        p.last_organization_id = "org_test"
        out = await p.check_account_limits()
        assert out["requests_limit"] == "50"
        assert out["requests_remaining"] == "47"
        assert out["tokens_limit"] == "100000"
        assert out["organization_id"] == "org_test"
        assert "raw" in out

    @pytest.mark.asyncio
    async def test_openai_credits_points_to_platform(self):
        from pyutilz.llm.openai_provider import OpenAIProvider
        p = self._make(OpenAIProvider)
        with pytest.raises(NotImplementedError, match="platform.openai.com"):
            await p.get_account_credits()

    @pytest.mark.asyncio
    async def test_openai_limits_mentions_response_headers(self):
        from pyutilz.llm.openai_provider import OpenAIProvider
        p = self._make(OpenAIProvider)
        with pytest.raises(NotImplementedError, match="x-ratelimit"):
            await p.check_account_limits()

    @pytest.mark.asyncio
    async def test_xai_credits_points_to_console(self):
        from pyutilz.llm.xai_provider import XAIProvider
        p = self._make(XAIProvider)
        with pytest.raises(NotImplementedError, match="console.x.ai"):
            await p.get_account_credits()

    @pytest.mark.asyncio
    async def test_gemini_credits_mentions_gcp(self):
        gemini_module = pytest.importorskip("pyutilz.llm.gemini_provider")
        p = self._make(gemini_module.GeminiProvider)
        with pytest.raises(NotImplementedError, match="GCP"):
            await p.get_account_credits()

    @pytest.mark.asyncio
    async def test_claude_code_credits_explains_subscription(self):
        from pyutilz.llm.claude_code_provider import ClaudeCodeProvider
        p = self._make(ClaudeCodeProvider)
        with pytest.raises(NotImplementedError, match="subscription"):
            await p.get_account_credits()


class TestUniformInterface:
    """Every provider exposes both methods (even if NotImplementedError) —
    callers can write `await provider.get_account_credits()` polymorphically
    without checking provider type first."""

    def test_both_methods_exist_on_every_provider(self):
        from pyutilz.llm.factory import _PROVIDER_MODULES
        import importlib
        missing: list[str] = []
        for name, (mod_path, cls_name) in _PROVIDER_MODULES.items():
            try:
                mod = importlib.import_module(mod_path)
            except ImportError:
                continue
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            for method in ("get_account_credits", "check_account_limits"):
                if not callable(getattr(cls, method, None)):
                    missing.append(f"{name}: missing {method}")
        assert not missing, missing
