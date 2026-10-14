"""orjson-backed JSON helpers: the single JSON surface for pyutilz modules (stdlib ``json`` is banned in src/).

Output differs from stdlib ``json``: compact separators, ``indent`` is always 2 spaces, non-ASCII is not escaped and
NaN/Infinity serialize as ``null``. Decoding accepts bytes or str; ``NaN``/``Infinity`` literals are rejected.
"""

from typing import Any, Callable, Optional, Union

import orjson

__all__ = ["JSONDecodeError", "loads", "dumps", "dumps_bytes"]

JSONDecodeError = orjson.JSONDecodeError


def loads(data: Union[bytes, bytearray, memoryview, str]) -> Any:
    """Parse JSON from bytes or str; raises :class:`JSONDecodeError` (a ``ValueError``) on malformed input."""
    return orjson.loads(data)


def dumps_bytes(
    obj: Any,
    *,
    indent: bool = False,
    sort_keys: bool = False,
    non_str_keys: bool = False,
    default: Optional[Callable[[Any], Any]] = None,
) -> bytes:
    """Serialize ``obj`` to UTF-8 JSON bytes. ``indent`` selects 2-space indentation, ``non_str_keys`` coerces int/float keys."""
    option = 0
    if indent:
        option |= orjson.OPT_INDENT_2
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    if non_str_keys:
        option |= orjson.OPT_NON_STR_KEYS
    dumped: bytes = orjson.dumps(obj, default=default, option=option)
    return dumped


def dumps(
    obj: Any,
    *,
    indent: bool = False,
    sort_keys: bool = False,
    non_str_keys: bool = False,
    default: Optional[Callable[[Any], Any]] = None,
) -> str:
    """Serialize ``obj`` to a JSON str (see :func:`dumps_bytes` for the options)."""
    return dumps_bytes(obj, indent=indent, sort_keys=sort_keys, non_str_keys=non_str_keys, default=default).decode("utf-8")
