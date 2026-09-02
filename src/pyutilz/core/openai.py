"""Deprecated alias for :mod:`pyutilz.llm.openai_tokens`.

The token-counting helpers moved to the ``llm`` domain, where the canonical tiktoken-backed
implementation they delegate to already lived. This shim keeps ``from pyutilz.core.openai import
num_tokens_from_string`` (and the ``pyutilz.openai`` alias) resolving.

Resolution is lazy (PEP 562) rather than a top-level re-export ON PURPOSE: a top-level
``from pyutilz.llm.openai_tokens import ...`` would recreate the very ``core -> llm`` dependency
edge the move removed, so ``core`` would still not be a leaf layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # names below are resolved at runtime by __getattr__; imported here only so
    # static tooling (and `__all__`) can see that they exist without creating a real import edge.
    from pyutilz.llm.openai_tokens import num_tokens_from_messages, num_tokens_from_string

__all__ = ["num_tokens_from_string", "num_tokens_from_messages"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from pyutilz.llm import openai_tokens

        value = getattr(openai_tokens, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
