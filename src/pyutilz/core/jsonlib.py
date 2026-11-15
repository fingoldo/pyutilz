"""JSON helpers: the single JSON surface for pyutilz modules (stdlib ``json`` is used only here).

Three backends, the first that imports wins: orjson, then msgspec, then the stdlib ``json`` module. The packaging
installs orjson on Python below 3.13 and msgspec from 3.13 on (orjson ships no free-threaded wheel, and a dependency
marker cannot tell a free-threaded build apart). An orjson that is installed by hand on any version is used too.

All three give the same output: compact separators, ``indent`` is always 2 spaces, non-ASCII is not escaped and
NaN/Infinity serialize as ``null``. Decoding accepts bytes or str; ``NaN``/``Infinity`` literals are rejected.
"""

import json as _json
import math
from typing import Any, Callable, Optional, Type, Union

__all__ = ["BACKEND", "JSONDecodeError", "loads", "dumps", "dumps_bytes"]

_orjson: Any = None
try:
    import orjson as _orjson_module

    _orjson = _orjson_module
except ImportError:
    pass

_msgspec: Any = None
if _orjson is None:
    try:
        import msgspec as _msgspec_module

        _msgspec = _msgspec_module
    except ImportError:
        pass

BACKEND: str = "orjson" if _orjson is not None else ("msgspec" if _msgspec is not None else "stdlib")
if _orjson is not None:
    JSONDecodeError: Type[ValueError] = _orjson.JSONDecodeError
elif _msgspec is not None:
    JSONDecodeError = _msgspec.DecodeError  # a ValueError subclass, so callers catching ValueError keep working
else:
    JSONDecodeError = _json.JSONDecodeError


def _reject_constant(name: str) -> Any:
    raise JSONDecodeError(f"{name} is not valid JSON", name, 0)


def _without_non_finite(obj: Any) -> Any:
    """A copy of ``obj`` with NaN and infinities replaced by None (what orjson and msgspec do natively)."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _without_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_without_non_finite(v) for v in obj]
    return obj


def _sorted_mappings(obj: Any) -> Any:
    """A copy of ``obj`` whose mappings have their keys in sorted order (msgspec has no sort option)."""
    if isinstance(obj, dict):
        return {k: _sorted_mappings(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_sorted_mappings(v) for v in obj]
    return obj


def _stdlib_dumps(obj: Any, default: Optional[Callable[[Any], Any]], sort_keys: bool, indent: bool) -> str:
    kwargs: dict[str, Any] = {
        "default": default,
        "sort_keys": sort_keys,
        "ensure_ascii": False,
        "allow_nan": False,
        "separators": (",", ":"),
    }
    if indent:
        kwargs["indent"] = 2
        kwargs["separators"] = (",", ": ")
    # allow_nan=False refuses NaN/Infinity, so they are replaced with null up front, as the other backends do natively.
    return _json.dumps(_without_non_finite(obj), **kwargs)


def loads(data: Union[bytes, bytearray, memoryview, str]) -> Any:
    """Parse JSON from bytes or str; raises :class:`JSONDecodeError` (a ``ValueError``) on malformed input."""
    if _orjson is not None:
        return _orjson.loads(data)
    if _msgspec is not None:
        return _msgspec.json.decode(bytes(data) if isinstance(data, (bytearray, memoryview)) else data)
    if isinstance(data, memoryview):
        data = data.tobytes()
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).decode("utf-8")
    return _json.loads(data, parse_constant=_reject_constant)


def dumps_bytes(
    obj: Any,
    *,
    indent: bool = False,
    sort_keys: bool = False,
    non_str_keys: bool = False,
    default: Optional[Callable[[Any], Any]] = None,
) -> bytes:
    """Serialize ``obj`` to UTF-8 JSON bytes. ``indent`` selects 2-space indentation, ``non_str_keys`` coerces int/float keys."""
    if _orjson is not None:
        option = 0
        if indent:
            option |= _orjson.OPT_INDENT_2
        if sort_keys:
            option |= _orjson.OPT_SORT_KEYS
        if non_str_keys:
            option |= _orjson.OPT_NON_STR_KEYS
        dumped: bytes = _orjson.dumps(obj, default=default, option=option)
        return dumped
    if _msgspec is not None:
        # msgspec coerces int/float/bool/None mapping keys to strings, as the stdlib does; ``non_str_keys`` is implied.
        body = _sorted_mappings(obj) if sort_keys else obj
        encoded: bytes = _msgspec.json.encode(body, enc_hook=default)
        return bytes(_msgspec.json.format(encoded, indent=2)) if indent else encoded
    # The stdlib coerces int/float/bool/None mapping keys whatever ``non_str_keys`` says, where orjson refuses them without it.
    return _stdlib_dumps(obj, default, sort_keys, indent).encode("utf-8")


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
