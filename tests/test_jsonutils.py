"""Behaviour of ``pyutilz.text.strings.jsonutils``: objects are jsonized into plain dicts and lists (with recursion and
attribute-skip rules), attributes are removed, kept or extracted from JSON-like structures, defaults and empty values are
dropped, non-finite floats and NUL characters are normalized for postgres, and list-of-dict properties are gathered with
their indices. The psycopg2 adapter is checked only through its parsed ``adapted`` value, so no database is needed."""

from __future__ import annotations

import json
from datetime import date, datetime
from xml.etree.ElementTree import Element

from pyutilz.text.strings import jsonutils


class _Plain:
    def __init__(self):
        self.name = "  padded  "
        self.count = 3
        self._hidden = "never"


class TestJsonSerial:
    def test_dates_are_written_as_iso_strings(self):
        assert jsonutils.json_serial(date(2026, 1, 2)) == "2026-01-02"
        assert jsonutils.json_serial(datetime(2026, 1, 2, 3, 4)) == "2026-01-02T03:04:00"

    def test_other_types_are_refused(self):
        try:
            jsonutils.json_serial(object())
        except TypeError as e:
            assert "not serializable" in str(e)
        else:
            raise AssertionError("expected TypeError")


class TestSubElem:
    def test_the_child_is_appended_with_its_text_and_attributes(self):
        root = Element("root")
        child = jsonutils.sub_elem(root, "item", text="hello", attribs={"id": "7"})
        assert root.find("item") is child
        assert child.text == "hello"
        assert child.get("id") == "7"

    def test_an_empty_text_leaves_the_element_without_text(self):
        root = Element("root")
        child = jsonutils.sub_elem(root, "item", text="")
        assert child.text is None


class TestJsonizeAttributes:
    def test_strings_are_stripped_by_default(self):
        assert jsonutils.jsonize_atrtributes("  x  ") == "x"

    def test_strings_are_kept_as_they_are_when_strip_is_off(self):
        assert jsonutils.jsonize_atrtributes("  x  ", strip=False) == "  x  "

    def test_numbers_pass_through(self):
        assert jsonutils.jsonize_atrtributes(4.5) == 4.5

    def test_a_plain_dict_is_jsonized_value_by_value(self):
        assert jsonutils.jsonize_atrtributes({"a": " b ", "c": [1, " d "]}) == {"a": "b", "c": [1, "d"]}

    def test_sequences_become_lists(self):
        assert jsonutils.jsonize_atrtributes((1, " two ")) == [1, "two"]

    def test_an_object_gives_its_public_attributes_and_hides_private_ones(self):
        out = jsonutils.jsonize_atrtributes(_Plain())
        assert out == {"name": "padded", "count": 3}

    def test_excluded_attributes_are_left_out(self):
        assert jsonutils.jsonize_atrtributes(_Plain(), exclude=["count"]) == {"name": "padded"}

    def test_past_the_recursion_limit_containers_are_returned_untouched(self):
        nested = {"a": {"b": " c "}}
        assert jsonutils.jsonize_atrtributes(nested, recursion_level=1, max_recursion_level=0) is nested


class TestRemoveAndLeaveAttributes:
    def test_remove_drops_only_the_named_keys(self):
        obj = {"a": 1, "b": 2, "c": 3}
        jsonutils.remove_json_attributes(obj, ["a", "missing"])
        assert obj == {"b": 2, "c": 3}

    def test_remove_on_none_does_nothing(self):
        assert jsonutils.remove_json_attributes(None, ["a"]) is None

    def test_leave_keeps_only_the_named_keys(self):
        obj = {"a": 1, "b": 2, "c": 3}
        jsonutils.leave_json_attributes(obj, ["a", "c"])
        assert obj == {"a": 1, "c": 3}

    def test_leave_on_none_does_nothing(self):
        assert jsonutils.leave_json_attributes(None, ["a"]) is None


class TestExtractJsonAttribute:
    def test_a_dict_of_dicts_maps_each_value_to_the_attribute(self):
        obj = {"cat": {"uid": "1", "label": "Dev"}, "sub": [{"uid": "2", "label": "QA"}]}
        assert jsonutils.extract_json_attribute(obj, "label") == {"cat": "Dev", "sub": ["QA"]}

    def test_a_list_of_dicts_maps_each_element_to_the_first_present_attribute(self):
        items = [{"name": None, "title": "T"}, {"other": 1}]
        assert jsonutils.extract_json_attribute(items, ["name", "title"]) == ["T", {"other": 1}]

    def test_strings_inside_a_dict_are_kept(self):
        assert jsonutils.extract_json_attribute({"k": "v"}, "label") == {"k": "v"}

    def test_a_non_container_gives_an_empty_dict(self):
        assert jsonutils.extract_json_attribute("nope", "label") == {}


class TestRemoveEmptyAndDefaults:
    def test_empty_values_are_removed_and_others_kept(self):
        obj = {"a": [], "b": "", "c": {}, "d": [1], "e": 0}
        jsonutils.remove_json_empty_attributes(obj, ["a", "b", "c", "d", "e", "missing"])
        assert obj == {"d": [1], "e": 0}

    def test_a_value_without_len_is_kept(self):
        obj = {"n": None}
        jsonutils.remove_json_empty_attributes(obj, ["n"])
        assert obj == {"n": None}

    def test_attributes_equal_to_their_default_are_removed(self):
        obj = {"a": 1, "b": 5}
        jsonutils.remove_json_defaults(obj, {"a": 1, "b": 2})
        assert obj == {"b": 5}

    def test_no_defaults_or_no_object_changes_nothing(self):
        obj = {"a": 1}
        jsonutils.remove_json_defaults(obj, None)
        jsonutils.remove_json_defaults(None, {"a": 1})
        assert obj == {"a": 1}


class TestNormalizeForPg:
    def test_non_finite_floats_become_none(self):
        assert jsonutils._normalize_for_pg_json([1.0, float("nan"), float("inf")]) == [1.0, None, None]

    def test_nul_characters_are_stripped_from_strings(self):
        assert jsonutils._normalize_for_pg_json("a\x00b") == "ab"

    def test_lone_surrogates_become_the_replacement_character(self):
        assert jsonutils._normalize_for_pg_json("a\ud800b") == "a�b"

    def test_tuples_become_lists_and_dict_string_keys_are_normalized(self):
        assert jsonutils._normalize_for_pg_json({"k\x00": (1, " x ")}) == {"k": [1, " x "]}

    def test_a_literal_backslash_sequence_is_not_mangled(self):
        assert jsonutils._normalize_for_pg_json("\\u0000") == "\\u0000"


class TestJsonPgDumps:
    def test_the_adapted_value_matches_the_input(self):
        adapter = jsonutils.json_pg_dumps({"a": [1, 2.5], "b": "x"})
        assert adapter.adapted == {"a": [1, 2.5], "b": "x"}

    def test_the_dumps_hook_returns_the_prebuilt_text(self):
        adapter = jsonutils.json_pg_dumps({"a": 1})
        assert json.loads(adapter.dumps()) == {"a": 1}

    def test_nul_characters_are_removed_from_the_stored_document(self):
        adapter = jsonutils.json_pg_dumps({"a": "x\x00y"})
        assert adapter.adapted == {"a": "xy"}

    def test_a_literal_escape_text_survives_intact(self):
        adapter = jsonutils.json_pg_dumps({"path": "\\u0000"})
        assert adapter.adapted == {"path": "\\u0000"}

    def test_non_finite_floats_are_stored_as_null(self):
        adapter = jsonutils.json_pg_dumps({"x": float("nan")})
        assert adapter.adapted == {"x": None}

    def test_sort_keys_orders_the_serialized_text(self):
        adapter = jsonutils.json_pg_dumps({"b": 1, "a": 2}, sort_keys=True)
        assert adapter.dumps().index('"a"') < adapter.dumps().index('"b"')


class TestJsonListProperty:
    def test_a_single_property_is_gathered_with_its_indices(self):
        data = [{"id": 4}, {"name": "x"}, {"id": 12}]
        assert jsonutils.get_jsonlist_property(data, "id", return_indices=True) == ([4, 12], [0, 2])

    def test_a_single_property_without_indices_is_a_bare_list(self):
        assert jsonutils.get_jsonlist_property([{"id": 4}, {"id": 12}], "id") == [4, 12]

    def test_a_dict_has_no_positional_indices(self):
        assert jsonutils.get_jsonlist_property({"id": 4}, "id", return_indices=True) == (4, None)

    def test_several_properties_keep_only_the_present_ones(self):
        data = [{"id": 1, "name": "a"}, {"other": 2}, {"name": "b"}]
        out, indices = jsonutils.get_jsonlist_properties(data, ["id", "name"])
        assert out == [{"id": 1, "name": "a"}, {"name": "b"}]
        assert indices == [0, 2]

    def test_elements_with_none_of_the_properties_are_dropped(self):
        assert jsonutils.get_jsonlist_properties([{"other": 1}], ["id"], return_indices=False) == []
