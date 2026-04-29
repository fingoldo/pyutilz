"""Accurate token counting using tiktoken.

Falls back to len(text) // 4 if tiktoken is unavailable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _ENCODING = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False
    logger.debug("tiktoken not installed -- using len//4 fallback for token counting")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken cl100k_base encoding.

    Falls back to len(text) // 4 if tiktoken is not available.
    cl100k_base is the encoding used by GPT-4, Claude, and most modern LLMs.
    """
    if _HAS_TIKTOKEN:
        return len(_ENCODING.encode(text))
    return len(text) // 4
