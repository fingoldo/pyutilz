"""LLM provider implementations."""

from pyutilz.llm.exceptions import (
    LLMProviderError,
    ClaudeCodeToolUseError,
    JSONParsingError,
    LLMRefusalError,
    LLMSafetyBlockError,
    LLMTruncationError,
    LLMUnparseableResponseError,
)

# Lazy-import map: attribute name → (module_path, object_name)
_LAZY_IMPORTS = {
    "LLMProvider": ("pyutilz.llm.base", "LLMProvider"),
    "count_tokens": ("pyutilz.llm.token_counter", "count_tokens"),
    "list_openrouter_models": ("pyutilz.llm.openrouter_provider", "list_openrouter_models"),
    "clear_openrouter_caches": ("pyutilz.llm.openrouter_provider", "clear_openrouter_caches"),
    "get_llm_provider": ("pyutilz.llm.factory", "get_llm_provider"),
}


def _provider_lazy_imports() -> dict:
    """Return ``{ClassName: (module_path, class_name)}`` derived from the factory's provider table.

    Derived rather than hand-copied so a provider added to ``factory._PROVIDER_MODULES`` is
    automatically reachable from this facade -- the hand-maintained duplicate silently dropped
    ``OpenAIProvider`` for as long as it existed. Imported lazily (inside the call, not at module
    top) because ``factory`` pulls in pydantic settings + the base provider, which this package's
    whole lazy-import design exists to avoid paying for at ``import pyutilz.llm``.
    """
    from pyutilz.llm.factory import _PROVIDER_MODULES

    return {cls_name: (mod_path, cls_name) for mod_path, cls_name, _key_attr in _PROVIDER_MODULES.values()}


def __getattr__(name: str):
    """Lazy-import so missing optional deps don't break the package."""
    import importlib

    if name in _LAZY_IMPORTS:
        mod_path, obj_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, obj_name)
    if name.endswith("Provider"):
        entry = _provider_lazy_imports().get(name)
        if entry is not None:
            mod_path, obj_name = entry
            return getattr(importlib.import_module(mod_path), obj_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Re-exported because its consumer is a DOWNSTREAM project: the runaway-decoder check is a public API of
# this package, and a helper whose only in-repo reference is its test reads as dead to the meta-test.
from pyutilz.llm.degeneracy import DegeneracyReport, DegeneracyThresholds, degeneracy_report

__all__ = [
    "DegeneracyReport",
    "DegeneracyThresholds",
    "degeneracy_report",
    "LLMProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "DeepSeekProvider",
    "XAIProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "list_openrouter_models",
    "clear_openrouter_caches",
    "ClaudeCodeProvider",
    "get_llm_provider",
    "LLMProviderError",
    "ClaudeCodeToolUseError",
    "JSONParsingError",
    "LLMRefusalError",
    "LLMSafetyBlockError",
    "LLMTruncationError",
    "LLMUnparseableResponseError",
    "count_tokens",
]
