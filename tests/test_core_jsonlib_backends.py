"""The three ``pyutilz.core.jsonlib`` backends (orjson, msgspec, stdlib) must give the same answers.

On a development machine orjson is installed, so the msgspec and stdlib branches never run in the ordinary suite.
Each test here executes a fresh copy of the module with the other backends hidden from the import system.
"""

import importlib.util
import math
import sys

import pytest

from pyutilz.core import jsonlib as _primary

# Which top-level modules each backend hides. msgspec is hidden in the orjson-free run too, so the stdlib branch is the
# one that runs when neither C backend imports.
_HIDDEN = {
    "orjson": (),
    "msgspec": ("orjson",),
    "stdlib": ("orjson", "msgspec"),
}
_NEEDS = {"orjson": "orjson", "msgspec": "msgspec", "stdlib": None}


@pytest.fixture(params=["orjson", "msgspec", "stdlib"])
def jl(request, monkeypatch):
    needed = _NEEDS[request.param]
    if needed is not None:
        pytest.importorskip(needed)
    for name in _HIDDEN[request.param]:
        monkeypatch.setitem(sys.modules, name, None)
    # A fresh copy of the file, not registered under the package name: reloading the real module in place would rebind
    # the objects that other test modules already imported from it (see test_no_module_reload.py).
    spec = importlib.util.spec_from_file_location("_jsonlib_backend_probe", _primary.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The hidden set decides the backend: a mismatch means the test would silently check the wrong branch.
    assert module.BACKEND == request.param
    yield module


def test_compact_output(jl):
    assert jl.dumps({"a": 1, "b": [2]}) == '{"a":1,"b":[2]}'


def test_indent_is_two_spaces(jl):
    assert jl.dumps({"a": 1}, indent=True) == '{\n  "a": 1\n}'


def test_nested_sort_keys(jl):
    assert jl.dumps({"b": {"d": 1, "c": 2}, "a": 1}, sort_keys=True) == '{"a":1,"b":{"c":2,"d":1}}'


def test_nan_and_infinity_become_null(jl):
    assert jl.dumps([math.nan, math.inf, -math.inf]) == "[null,null,null]"


def test_nan_nested_in_a_mapping_becomes_null(jl):
    assert jl.dumps({"a": {"b": math.nan, "c": "kept"}}) == '{"a":{"b":null,"c":"kept"}}'


def test_sort_keys_reaches_mappings_inside_lists(jl):
    assert jl.dumps({"b": [{"d": 1, "c": 2}]}, sort_keys=True) == '{"b":[{"c":2,"d":1}]}'


def test_nul_is_escaped_and_non_ascii_is_not(jl):
    assert jl.dumps({"a": "x\x00y"}) == '{"a":"x\\u0000y"}'
    assert jl.dumps({"k": "é"}) == '{"k":"é"}'


def test_non_str_keys_are_coerced_with_the_flag(jl):
    assert jl.dumps({1: "x"}, non_str_keys=True) == '{"1":"x"}'


def test_default_hook_serializes_unknown_types(jl):
    class Point:
        def __init__(self, x: int) -> None:
            self.x = x

    assert jl.dumps([Point(3)], default=lambda o: {"x": o.x}) == '[{"x":3}]'


def test_dumps_bytes_returns_utf8(jl):
    assert jl.dumps_bytes({"a": 1}) == b'{"a":1}'
    assert jl.dumps_bytes({"a": 1}, indent=True) == b'{\n  "a": 1\n}'


def test_loads_accepts_bytes_str_bytearray_and_memoryview(jl):
    assert jl.loads(b'{"a": [1, 2]}') == {"a": [1, 2]}
    assert jl.loads('{"a": [1, 2]}') == {"a": [1, 2]}
    assert jl.loads(bytearray(b"[1]")) == [1]
    assert jl.loads(memoryview(b"[2]")) == [2]


def test_loads_rejects_malformed_input_and_nan_as_value_error(jl):
    with pytest.raises(jl.JSONDecodeError):
        jl.loads("{not json")
    with pytest.raises(jl.JSONDecodeError):
        jl.loads("NaN")
    assert issubclass(jl.JSONDecodeError, ValueError)


def test_a_lone_surrogate_is_refused(jl):
    with pytest.raises((TypeError, UnicodeError)):
        jl.dumps("\ud800")
