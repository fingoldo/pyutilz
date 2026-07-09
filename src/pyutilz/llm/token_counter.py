"""Accurate token counting using tiktoken.

Falls back to len(text) // 4 if tiktoken is unavailable.
"""

from __future__ import annotations

import logging

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
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except Exception as e:
    _ENCODING = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False
    logger.debug("tiktoken unavailable (%s: %s) -- using len//4 fallback for token counting", type(e).__name__, e)


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken cl100k_base encoding.

    Falls back to len(text) // 4 if tiktoken is not available.
    cl100k_base is the encoding used by GPT-4, Claude, and most modern LLMs.
    """
    if _HAS_TIKTOKEN:
        return len(_ENCODING.encode(text))
    return len(text) // 4
