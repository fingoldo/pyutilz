"""LLM provider implementations."""

from pyutilz.llm.base import LLMProvider
from pyutilz.llm.factory import get_llm_provider
from pyutilz.llm.exceptions import LLMProviderError, JSONParsingError
from pyutilz.llm.token_counter import count_tokens


def __getattr__(name: str):
    """Lazy-import provider classes so missing optional deps don't break the package."""
    _providers = {
        "AnthropicProvider": "pyutilz.llm.anthropic_provider",
        "GeminiProvider": "pyutilz.llm.gemini_provider",
        "DeepSeekProvider": "pyutilz.llm.deepseek_provider",
        "XAIProvider": "pyutilz.llm.xai_provider",
        "ClaudeCodeProvider": "pyutilz.llm.claude_code_provider",
    }
    if name in _providers:
        import importlib
        mod = importlib.import_module(_providers[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LLMProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "DeepSeekProvider",
    "XAIProvider",
    "ClaudeCodeProvider",
    "get_llm_provider",
    "LLMProviderError",
    "JSONParsingError",
    "count_tokens",
]
