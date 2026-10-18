"""LLM provider implementations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._claude_models import claude_model_listed_price
    from ._progress import track_stream_progress
    from .config import clear_llm_settings_cache
    from .openai_tokens import num_tokens_from_messages, num_tokens_from_string
    from .generation_metadata import fetch_generation_metadata
    from .openrouter_speech import OpenRouterSpeech, SpeechAccounting, SpeechError, SpeechInputError, SpeechRequest, SpeechResult

# Resolved on first attribute access by __getattr__ (lazy: the defining modules may need optional extras).
_REEXPORTS = {
    "claude_model_listed_price": "_claude_models",
    "track_stream_progress": "_progress",
    "clear_llm_settings_cache": "config",
    "num_tokens_from_messages": "openai_tokens",
    "num_tokens_from_string": "openai_tokens",
    "OpenRouterSpeech": "openrouter_speech",
    "SpeechRequest": "openrouter_speech",
    "SpeechResult": "openrouter_speech",
    "SpeechAccounting": "openrouter_speech",
    "SpeechError": "openrouter_speech",
    "SpeechInputError": "openrouter_speech",
    "fetch_generation_metadata": "generation_metadata",
}


from pyutilz.llm.exceptions import (
    LLMProviderError,
    ClaudeCodeToolUseError,
    JSONParsingError,
    LLMRefusalError,
    LLMSafetyBlockError,
    LLMTruncationError,
    LLMStreamInterruptedError,
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
    if name in _REEXPORTS:
        import importlib

        value = getattr(importlib.import_module(f".{_REEXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value

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
from pyutilz.llm.degeneracy import DegeneracyReport, DegeneracyThresholds, degeneracy_report, repetition_loop

__all__ = [
    "DegeneracyReport",
    "DegeneracyThresholds",
    "degeneracy_report",
    "repetition_loop",
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
    "LLMStreamInterruptedError",
    "LLMUnparseableResponseError",
    "count_tokens",
    "claude_model_listed_price",
    "track_stream_progress",
    "clear_llm_settings_cache",
    "num_tokens_from_messages",
    "num_tokens_from_string",
]
