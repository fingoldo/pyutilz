"""Regression tests for pyutilz.text.strings.jsonutils."""

import logging

from pyutilz.text.strings.jsonutils import jsonize_atrtributes, get_jsonlist_property, get_jsonlist_properties


class ObjWithBrokenProperty:
    """Object whose 'broken' property raises, with normal attributes before/after it."""

    before_attr = "before_value"

    @property
    def broken(self) -> str:
        raise RuntimeError("lazy property failed on purpose")

    after_attr = "after_value"


def test_jsonize_atrtributes_continues_past_broken_attribute(caplog):
    obj = ObjWithBrokenProperty()

    with caplog.at_level(logging.ERROR):
        res = jsonize_atrtributes(obj=obj)

    assert isinstance(res, dict)

    # attribute defined after the broken one must still be present -> loop did not abort
    assert res.get("after_attr") == "after_value"
    assert res.get("before_attr") == "before_value"

    # the broken attribute itself must not silently appear as if it succeeded
    assert "broken" not in res

    # failure must be logged, not silently swallowed
    assert any("broken" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# get_jsonlist_property — dict input must honor return_indices (bug: dict
# branch used to always return a bare scalar, ignoring return_indices).
# ---------------------------------------------------------------------------


def test_get_jsonlist_property_dict_return_indices_true():
    result = get_jsonlist_property({"id": 5, "name": "John"}, "id", return_indices=True)
    assert result == (5, None)


def test_get_jsonlist_property_dict_return_indices_false():
    # default/False behavior must remain a bare scalar, unchanged
    result = get_jsonlist_property({"id": 5, "name": "John"}, "id")
    assert result == 5


# ---------------------------------------------------------------------------
# get_jsonlist_properties — must accept a return_indices param; default stays
# True (tuple return) for backward compat with existing positional-unpacking
# callers (see tests/test_strings_extra.py::test_get_jsonlist_properties_basic).
# ---------------------------------------------------------------------------


def test_get_jsonlist_properties_default_still_returns_tuple():
    data = [{"a": 1, "b": 2, "c": 3}]
    result, indices = get_jsonlist_properties(data, ["a", "c"])
    assert result == [{"a": 1, "c": 3}]
    assert indices == [0]


def test_get_jsonlist_properties_return_indices_false_gives_bare_list():
    data = [{"a": 1, "b": 2, "c": 3}]
    result = get_jsonlist_properties(data, ["a", "c"], return_indices=False)
    assert result == [{"a": 1, "c": 3}]
    assert not isinstance(result, tuple)
