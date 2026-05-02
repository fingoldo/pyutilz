"""Pin LLMProvider.supports_json_mode() across all providers.

Catches the silent-drift bug pattern where a JSON-mode kwarg is
plumbed through to providers that don't actually accept it (or
vice versa: a provider claims support but the underlying model
doesn't). The base default is False — strict opt-in, every provider
that supports response_format / response_mime_type / equivalent
must explicitly say so.
"""

from __future__ import annotations

import pytest

from pyutilz.llm.base import LLMProvider


def test_base_default_is_false():
    """If a new provider subclass forgets to override, callers get
    False — safe default that triggers fallback parsing rather than
    crashing on an unsupported kwarg."""
    # Build a minimal subclass that satisfies the abstract interface
    # only enough to be instantiable.
    class _Stub(LLMProvider):
        async def generate(self, prompt, system=None, temperature=0.7, max_tokens=0):
            return ""
    assert _Stub().supports_json_mode() is False


@pytest.mark.parametrize("module_path,class_name,expected", [
    ("pyutilz.llm.openai_compat", "OpenAICompatibleProvider", True),
    ("pyutilz.llm.gemini_provider", "GeminiProvider", True),
    ("pyutilz.llm.anthropic_provider", "AnthropicProvider", False),
    ("pyutilz.llm.claude_code_provider", "ClaudeCodeProvider", False),
])
def test_provider_class_declares_own_override(module_path, class_name, expected):
    """Each provider must declare its own ``supports_json_mode`` (so
    silent inheritance from base doesn't cause drift)."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    # Must have its OWN definition, not just inherited.
    assert "supports_json_mode" in cls.__dict__, (
        f"{class_name} must declare its own supports_json_mode "
        f"(currently inheriting base default)"
    )
    # Method should return the documented expected value when called
    # on an instance — but instantiating each provider needs API keys
    # / SDK setup. Inspect the function bytecode default instead.
    fn = cls.__dict__["supports_json_mode"]
    # Find the literal True/False the function returns (covers the
    # simple ``return True`` / ``return False`` overrides).
    import dis
    return_consts = []
    for ins in dis.get_instructions(fn):
        if ins.opname == "RETURN_VALUE" and return_consts:
            break
        if ins.opname == "LOAD_CONST" and isinstance(ins.argval, bool):
            return_consts.append(ins.argval)
    if return_consts:
        # Last bool LOAD_CONST before RETURN should match expected for
        # the simple overrides. (OpenRouterProvider's override has
        # branching logic — skip the bytecode check for it.)
        assert expected in return_consts, (
            f"{class_name}.supports_json_mode bytecode doesn't include "
            f"expected literal {expected}; constants seen: {return_consts}"
        )


def test_openrouter_provider_declares_own_override():
    """OpenRouterProvider has model-aware logic; verify the method is
    overridden but skip the static-bool check."""
    from pyutilz.llm.openrouter_provider import OpenRouterProvider
    assert "supports_json_mode" in OpenRouterProvider.__dict__
