from __future__ import annotations
import asyncio
from pyutilz.core import jsonlib
from typing import Any
import pytest
from pyutilz.llm import _progress
from pyutilz.llm.base import LLMProvider


class _Provider(LLMProvider):
    """Minimal concrete provider: records every generate() call's kwargs."""

    _PRICING = {"known-model": (1.0, 2.0), "fam-model": (3.0, 4.0)}

    def __init__(self, max_concurrent: int = 1, model: str = "known-model") -> None:
        self._max_concurrent = max_concurrent
        self.model_name = model
        self.calls: list[dict[str, Any]] = []

    async def generate(self, prompt: str, system: str | None = None, temperature: float | None = 0.7, max_tokens: int = 0, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        async with self.semaphore:
            _progress.note_stream_progress(None, "x" * 10)
            await asyncio.sleep(0.01)
        return prompt

    async def count_tokens(self, text: str) -> int:
        return len(text)


def _or(**kwargs: Any) -> Any:
    pytest.importorskip("httpx")
    from pyutilz.llm.openrouter_provider import OpenRouterProvider

    return OpenRouterProvider(api_key="k", model=kwargs.pop("model", "openai/gpt-4o-mini"), **kwargs)


class TestRouteFingerprint:
    def test_same_config_same_key_and_is_sha256_hex(self):
        a, b = _or(provider_order=("a", "b")), _or(provider_order=("a", "b"))
        fp = a.route_fingerprint()
        assert fp == b.route_fingerprint()
        assert len(fp) == 64 and int(fp, 16) >= 0

    def test_tuple_and_list_digest_equal(self):
        assert _or(provider_quantizations=("fp8", "bf16")).route_fingerprint() == _or(provider_quantizations=["fp8", "bf16"]).route_fingerprint()

    def test_mapping_and_pair_tuple_digest_equal(self):
        assert _or(provider_max_price={"prompt": 1, "completion": 2}).route_fingerprint() == _or(provider_max_price=(("prompt", 1), ("completion", 2))).route_fingerprint()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model": "anthropic/claude-x"},
            {"provider_quantizations": ("fp4",)},
            {"provider_order": ("b", "a")},
            {"provider_only": ("a",)},
            {"provider_ignore": ("a",)},
            {"provider_sort": "price"},
            {"provider_allow_fallbacks": False},
            {"provider_require_parameters": True},
            {"provider_data_collection": "deny"},
            {"provider_zdr": True},
            {"models_fallback": ("x/y",)},
            {"enable_web_search": True},
            {"transforms": ("middle-out",)},
            {"system_cache_control": True},
            {"anthropic_top_level_cache": True},
            {"extra_body": {"seed": 1}},
            {"pdf_engine": "native"},
        ],
    )
    def test_each_routing_knob_changes_the_key(self, kwargs):
        assert _or(**kwargs).route_fingerprint() != _or().route_fingerprint()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"api_key": "other"},  # pragma: allowlist secret -- a dummy key: the test is that credentials never reach the digest
            {"max_concurrent": 3},
            {"app_name": "x"},
            {"site_url": "https://x"},
            {"retry_routing_404": True},
        ],
    )
    def test_non_routing_kwargs_do_not_change_the_key(self, kwargs):
        from pyutilz.llm.openrouter_provider import OpenRouterProvider

        base_kwargs: dict[str, Any] = {"api_key": "k", "model": "openai/gpt-4o-mini"}
        base_kwargs.update(kwargs)
        assert OpenRouterProvider(**base_kwargs).route_fingerprint() == _or().route_fingerprint()

    def test_api_key_never_enters_the_payload(self):
        p = _or()
        dummy_key = "sk-secret"  # pragma: allowlist secret -- a dummy key, asserted absent from the route payload
        assert dummy_key not in jsonlib.dumps(_or(**{}).__class__(api_key=dummy_key)._route_payload(), default=str)
        assert "api_key" not in p._route_payload()

    def test_reassigned_model_is_honoured(self):
        p = _or()
        before = p.route_fingerprint()
        p.model_name = "anthropic/claude-x"
        assert p.route_fingerprint() != before

    def test_every_ctor_kwarg_is_classified(self):
        import inspect

        from pyutilz.llm.openrouter_provider import OpenRouterProvider
        from pyutilz.llm.openrouter_provider._route import _ROUTE_FINGERPRINT_ATTRS, _ROUTE_FINGERPRINT_EXCLUDED

        params = {n for n in inspect.signature(OpenRouterProvider.__init__).parameters if n != "self"}
        included, excluded = set(_ROUTE_FINGERPRINT_ATTRS), set(_ROUTE_FINGERPRINT_EXCLUDED)
        assert not included & excluded
        assert params == included | excluded, f"unclassified: {sorted(params - included - excluded)}; stale: {sorted((included | excluded) - params)}"

    def test_base_fingerprint_digests_provider_and_model(self):
        a, b = _Provider(model="m1"), _Provider(model="m2")
        assert a.route_fingerprint() == _Provider(model="m1").route_fingerprint()
        assert a.route_fingerprint() != b.route_fingerprint()
