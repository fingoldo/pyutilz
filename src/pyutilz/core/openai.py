"""Token-counting helpers for OpenAI-family chat models.

Thin wrappers over tiktoken. ``num_tokens_from_string`` delegates single-string
counting to :func:`pyutilz.llm.token_counter.count_tokens` (the canonical,
tiktoken-backed, len//4-fallback implementation) when the requested encoding is
the default cl100k_base, avoiding a duplicate tokenizer code path.

``num_tokens_from_messages`` retains the legacy per-message overhead accounting
for the ``gpt-3.5-turbo-0613`` message framing (``<im_start>``/``<im_end>``
sentinels). Newer chat models use a slightly different framing; treat the result
as an estimate for anything past the 0613 snapshot.
"""

from __future__ import annotations

try:
    import tiktoken
except ImportError:
    tiktoken = None


def _require_tiktoken() -> None:
    if tiktoken is None:
        raise ImportError("tiktoken is required: pip install pyutilz[nlp]")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Return the number of tokens in a text string for ``encoding_name``.

    For the default ``cl100k_base`` encoding this delegates to
    :func:`pyutilz.llm.token_counter.count_tokens`, which caches the encoding
    and degrades to a ``len//4`` estimate when tiktoken is unavailable. Other
    encodings are counted directly (and require tiktoken).
    """
    if encoding_name == "cl100k_base":
        from pyutilz.llm.token_counter import count_tokens

        return count_tokens(string)
    _require_tiktoken()
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def num_tokens_from_messages(
    messages: list[dict], model: str = "gpt-3.5-turbo-0613"
) -> int:
    """Return the number of tokens used by a list of chat messages.

    Uses the ``gpt-3.5-turbo-0613`` message-framing overhead (4 tokens per
    message, 2 priming tokens per reply). Newer models may deviate slightly;
    treat the result as an estimate for anything past that snapshot.

    Non-string ``value`` entries (e.g. a structured tool-call payload) are
    coerced to ``str`` before encoding so a non-str content field does not
    crash ``encoding.encode``.
    """
    _require_tiktoken()
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            if not isinstance(value, str):
                value = str(value)
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant

    return num_tokens
