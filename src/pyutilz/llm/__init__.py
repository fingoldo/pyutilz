"""LLM provider implementations."""

from pyutilz.llm.exceptions import LLMProviderError, JSONParsingError

# Lazy-import map: attribute name → (module_path, object_name)
_LAZY_IMPORTS = {
    "LLMProvider": ("pyutilz.llm.base", "LLMProvider"),
    "count_tokens": ("pyutilz.llm.token_counter", "count_tokens"),
    "AnthropicProvider": ("pyutilz.llm.anthropic_provider", "AnthropicProvider"),
    "GeminiProvider": ("pyutilz.llm.gemini_provider", "GeminiProvider"),
    "DeepSeekProvider": ("pyutilz.llm.deepseek_provider", "DeepSeekProvider"),
    "XAIProvider": ("pyutilz.llm.xai_provider", "XAIProvider"),
    "OpenRouterProvider": ("pyutilz.llm.openrouter_provider", "OpenRouterProvider"),
    "list_openrouter_models": ("pyutilz.llm.openrouter_provider", "list_openrouter_models"),
    "ClaudeCodeProvider": ("pyutilz.llm.claude_code_provider", "ClaudeCodeProvider"),
    "get_llm_provider": ("pyutilz.llm.factory", "get_llm_provider"),
}


def __getattr__(name: str):
    """Lazy-import so missing optional deps don't break the package."""
    if name in _LAZY_IMPORTS:
        import importlib
        mod_path, obj_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, obj_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LLMProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "DeepSeekProvider",
    "XAIProvider",
    "OpenRouterProvider",
    "list_openrouter_models",
    "ClaudeCodeProvider",
    "get_llm_provider",
    "LLMProviderError",
    "JSONParsingError",
    "count_tokens",
]
