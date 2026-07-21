"""Approximate token counting using tiktoken.

Falls back to len(text) // 4 if tiktoken is unavailable. NOT exact for non-OpenAI models (Claude
in particular has its own BPE that diverges from any tiktoken encoding for a meaningful fraction
of typical text -- see AnthropicProvider.count_tokens(), which calls Anthropic's real
messages.count_tokens API instead of this module for that reason).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import tiktoken

    # get_encoding() is not just an import-time check: it fetches the cl100k_base BPE merge
    # file over the network on first use (tiktoken caches it locally afterwards), so it can fail
    # for the same reasons any network call can -- DNS/proxy/CDN issues, or interpreter-specific
    # bugs surfaced through the request (e.g. a Python 3.8 stdlib SSLContext.verify_mode property
    # recursion hit via urllib3/requests -- not a pyutilz bug, but still must not crash this
    # module's import). None of those are ImportError, so catch broadly here and degrade to the
    # documented fallback exactly as when tiktoken itself is absent.
    _DEFAULT_ENCODING = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except Exception as e:
    _DEFAULT_ENCODING = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False
    logger.debug("tiktoken unavailable (%s: %s) -- using len//4 fallback for token counting", type(e).__name__, e)

_encoding_cache: dict[str, "tiktoken.Encoding"] = {}  # tiktoken may be unimported; forward-ref string annotation only


def _encoding_for_model(model: Optional[str]):
    """Resolve the correct tiktoken encoding for ``model`` (e.g. gpt-4o/o1 -> o200k_base, not
    cl100k_base -- verified against the installed tiktoken package), falling back to cl100k_base
    for unrecognized/legacy model names. Cached per model name (encoding objects are expensive to
    construct)."""
    if model is None:
        return _DEFAULT_ENCODING
    if model in _encoding_cache:
        return _encoding_cache[model]
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = _DEFAULT_ENCODING
    _encoding_cache[model] = enc
    return enc


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Count tokens in text using tiktoken, resolving the encoding for ``model`` when given
    (e.g. gpt-4o/o1 -> o200k_base) rather than always using cl100k_base.

    Falls back to len(text) // 4 if tiktoken is not available. This is an APPROXIMATION for any
    model tiktoken doesn't have a mapping for (notably Claude models -- diverges from Claude's
    real BPE for a meaningful fraction of typical text); it is not labeled "accurate" for that
    reason. Callers needing exact Claude counts should use a real Anthropic client's
    messages.count_tokens API instead (see AnthropicProvider.count_tokens()).
    """
    if _HAS_TIKTOKEN:
        enc = _encoding_for_model(model)
        return len(enc.encode(text))
    return len(text) // 4
