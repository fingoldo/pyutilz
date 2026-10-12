"""Behaviour of the orjson-backed helpers in ``pyutilz.core.jsonlib``."""

import math

import pytest

from pyutilz.core import jsonlib


def test_loads_accepts_bytes_and_str():
    assert jsonlib.loads(b'{"a": [1, 2]}') == {"a": [1, 2]}
    assert jsonlib.loads('{"a": [1, 2]}') == {"a": [1, 2]}


def test_loads_rejects_malformed_input_as_value_error():
    with pytest.raises(jsonlib.JSONDecodeError):
        jsonlib.loads("{not json")
    assert issubclass(jsonlib.JSONDecodeError, ValueError)


def test_loads_rejects_nan_and_infinity_literals():
    with pytest.raises(jsonlib.JSONDecodeError):
        jsonlib.loads("NaN")
    with pytest.raises(jsonlib.JSONDecodeError):
        jsonlib.loads("Infinity")


def test_dumps_is_compact_by_default():
    assert jsonlib.dumps({"a": 1, "b": [2]}) == '{"a":1,"b":[2]}'


def test_dumps_indent_uses_two_spaces():
    assert jsonlib.dumps({"a": 1}, indent=True) == '{\n  "a": 1\n}'


def test_dumps_sort_keys_orders_mapping_keys():
    assert jsonlib.dumps({"b": 1, "a": 2}, sort_keys=True) == '{"a":2,"b":1}'


def test_dumps_non_str_keys_coerces_int_keys():
    assert jsonlib.dumps({1: "x"}, non_str_keys=True) == '{"1":"x"}'


def test_dumps_default_hook_serializes_unknown_types():
    class Point:
        def __init__(self, x: int) -> None:
            self.x = x

    assert jsonlib.dumps([Point(3)], default=lambda o: {"x": o.x}) == '[{"x":3}]'


def test_dumps_nan_becomes_null():
    assert jsonlib.dumps([math.nan]) == "[null]"


def test_dumps_bytes_returns_utf8_bytes_with_same_options():
    assert jsonlib.dumps_bytes({"k": "é"}) == '{"k":"é"}'.encode()
    assert jsonlib.dumps_bytes({"a": 1}, indent=True) == b'{\n  "a": 1\n}'
