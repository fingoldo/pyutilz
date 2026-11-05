"""Unknown-model fallback pin for llm.openai_tokens.num_tokens_from_messages (mutation survivor)."""

from __future__ import annotations

import pytest

from pyutilz.llm import openai_tokens


def _expected_count(messages: list[dict], encoding) -> int:
    """Chat-message token count computed independently of the module under test."""
    total = 0
    for message in messages:
        total += 4
        for key, value in message.items():
            total += len(encoding.encode(value if isinstance(value, str) else str(value)))
            if key == "name":
                total -= 1
    return total + 2


def test_an_unknown_model_falls_back_to_the_cl100k_base_encoding():
    tiktoken = pytest.importorskip("tiktoken")
    messages = [{"role": "user", "content": "The quick brown fox jumps over the lazy dog."}]
    expected = _expected_count(messages, tiktoken.get_encoding("cl100k_base"))
    assert openai_tokens.num_tokens_from_messages(messages, model="definitely-not-a-real-model-xyz") == expected
