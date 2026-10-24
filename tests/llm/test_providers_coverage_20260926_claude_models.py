from __future__ import annotations
import logging
import pytest
from pyutilz.llm._claude_models import claude_model_spec, resolve_claude_model
from pyutilz.llm.anthropic_provider import AnthropicProvider


class TestProv2Prov3ModelTable:
    @pytest.mark.parametrize(
        ("model", "price", "max_out", "ctx"),
        [
            ("claude-fable-5-1", (10.0, 50.0), 128_000, 1_000_000),
            ("claude-opus-5-5", (4.0, 20.0), 128_000, 1_000_000),
            ("claude-sonnet-5", (2.0, 10.0), 128_000, 1_000_000),
            ("claude-sonnet-4-6", (3.0, 15.0), 128_000, 1_000_000),
            ("claude-haiku-4-5-20251001", (1.0, 5.0), 64_000, 200_000),
            ("claude-opus-4-5-20251101", (5.0, 25.0), 64_000, 200_000),
            ("claude-3-haiku-20240307", (0.25, 1.25), 4_096, 200_000),
            ("claude-3-5-haiku-20241022", (0.80, 4.0), 8_192, 200_000),
        ],
    )
    def test_real_ids_resolve(self, model: str, price: tuple, max_out: int, ctx: int) -> None:
        p = AnthropicProvider.__new__(AnthropicProvider)
        p.model = model
        assert p._get_pricing() == price
        assert p.max_output_tokens == max_out
        assert p.context_window == ctx

    def test_a_future_model_is_not_priced_from_a_legacy_prefix(self, caplog: pytest.LogCaptureFixture) -> None:
        assert resolve_claude_model("claude-opus-4-9") is None, "claude-opus-4 is a prefix of it, but not the same model"
        with caplog.at_level(logging.WARNING):
            spec = claude_model_spec("claude-opus-4-9-20270101")
        assert (spec.input_per_1m, spec.output_per_1m) == (4.0, 20.0)
        assert any("not in the Claude model table" in r.getMessage() for r in caplog.records)

    def test_platform_ids_resolve(self) -> None:
        assert resolve_claude_model("anthropic.claude-opus-5") == resolve_claude_model("claude-opus-5")
        assert resolve_claude_model("claude-opus-4-5@20251101") == resolve_claude_model("claude-opus-4-5")
        assert resolve_claude_model("anthropic.claude-opus-5")[0] == "claude-opus-5"  # type: ignore[index]


class TestProv9DefaultModel:
    def test_default_is_a_current_model(self) -> None:
        assert AnthropicProvider(api_key="sk-fake").model == "claude-sonnet-5"  # pragma: allowlist secret -- dummy key
        assert resolve_claude_model("claude-sonnet-5") is not None
