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


class TestRecursionBoundaries:
    def test_a_container_exactly_at_the_limit_is_still_jsonized(self):
        assert jsonutils.jsonize_atrtributes({"a": " x "}, recursion_level=1, max_recursion_level=1) == {"a": "x"}

    def test_a_dict_nested_past_the_limit_is_returned_untouched(self):
        nested = {"a": {"b": {"c": " x "}}}
        assert jsonutils.jsonize_atrtributes(nested, max_recursion_level=1) == {"a": {"b": {"c": " x "}}}

    def test_a_list_nested_past_the_limit_is_returned_untouched(self):
        assert jsonutils.jsonize_atrtributes([[{"c": " x "}]], max_recursion_level=1) == [[{"c": " x "}]]


class TestSkipFunctions:
    def test_builtin_functions_are_dropped_when_skipping(self):
        class _Holder:
            helper = len

        assert "helper" not in jsonutils.jsonize_atrtributes(_Holder(), skip_functions=True)

    def test_builtin_functions_are_kept_when_not_skipping(self):
        class _Holder:
            helper = len

        assert "helper" in jsonutils.jsonize_atrtributes(_Holder(), skip_functions=False)


class TestFastSerializationPath:
    def test_a_document_without_nul_skips_the_slow_normalizing_path(self, monkeypatch):
        def _slow(*args, **kwargs):
            raise AssertionError("the slow path must not run for a NUL-free document")

        # The fast path serializes with jsonlib.dumps too, so the detector is the normalizer, which only the slow path calls.
        monkeypatch.setattr(jsonutils, "_normalize_for_pg_json", _slow)
        assert jsonutils.json_pg_dumps({"a": "b"}).adapted == {"a": "b"}


class TestDepthIncrementIsOne:
    def test_a_dict_one_level_below_the_limit_is_still_jsonized(self):
        assert jsonutils.jsonize_atrtributes({"a": {"b": " x "}}, max_recursion_level=1) == {"a": {"b": "x"}}

    def test_a_list_element_one_level_below_the_limit_is_still_jsonized(self):
        assert jsonutils.jsonize_atrtributes([{"b": " x "}], max_recursion_level=1) == [{"b": "x"}]


class TestAttributeLoopKeepsGoing:
    def test_a_failing_attribute_does_not_stop_the_remaining_attributes(self):
        class _Mixed:
            @property
            def a_bad(self):
                raise RuntimeError("boom")

            @property
            def b_good(self):
                return "kept"

        assert jsonutils.jsonize_atrtributes(_Mixed()) == {"b_good": "kept"}


class TestSkippedFunctionDoesNotEndTheLoop:
    def test_a_builtin_before_a_plain_attribute_is_skipped_and_the_plain_one_kept(self):
        class _Holder:
            a_fn = len
            b_val = 3

        assert jsonutils.jsonize_atrtributes(_Holder(), skip_functions=True) == {"b_val": 3}


class TestJsonutilsSweepSurvivors:
    """The survivors of the jsonutils sweep: the strip and skip-functions arguments carried into nested mappings and
    sequences, the default skip of builtins, the debug notes for an attribute that raises or has no length, the warning
    of remove_json_defaults with its truncated object text, the NUL removal, the default sort order, and the verbose and
    indices defaults of the list helpers (jsonutils.py 95-435)."""

    def test_nested_strings_keep_their_spaces_when_strip_is_off(self):
        from pyutilz.text.strings import jsonutils

        assert jsonutils.jsonize_atrtributes({"a": " x "}, strip=False) == {"a": " x "}
        assert jsonutils.jsonize_atrtributes([" x "], strip=False) == [" x "]

    def test_strings_are_stripped_by_default_at_the_top_level_too(self):
        from pyutilz.text.strings import jsonutils

        assert jsonutils.jsonize_atrtributes(" x ") == "x"
        assert jsonutils.jsonize_atrtributes(" x ", strip=False) == " x "

    def test_builtin_functions_are_dropped_by_default(self):
        from pyutilz.text.strings import jsonutils

        class _Holder:
            helper = len
            kept = 1

        result = jsonutils.jsonize_atrtributes(_Holder())
        assert "helper" not in result
        assert result["kept"] == 1

    def test_an_attribute_that_raises_is_skipped_with_a_debug_note(self, caplog):
        import logging

        from pyutilz.text.strings import jsonutils

        class _Holder:
            @property
            def bad(self):
                raise ValueError("boom")

            ok = 1

        with caplog.at_level(logging.DEBUG):
            result = jsonutils.jsonize_atrtributes(_Holder())
        assert result == {"ok": 1}
        assert "Failed to jsonize attribute 'bad' of" in caplog.text
        assert "boom" in caplog.text

    def test_a_value_without_a_length_is_kept_with_a_debug_note(self, caplog):
        import logging

        from pyutilz.text.strings import jsonutils

        with caplog.at_level(logging.DEBUG):
            jsonutils.remove_json_empty_attributes({"n": 5, "e": []}, ["n", "e"])
        assert "Attribute n has no len(), skipping empty-check:" in caplog.text

    def test_defaults_are_removed_silently_by_default(self, caplog):
        import logging

        from pyutilz.text.strings import jsonutils

        obj = {"a": 1, "b": 2}
        with caplog.at_level(logging.WARNING):
            jsonutils.remove_json_defaults(obj, {"a": 1})
        assert obj == {"b": 2}
        assert caplog.text == ""

    def test_a_non_default_value_is_warned_about_with_its_truncated_object(self, caplog):
        import logging

        from pyutilz.text.strings import jsonutils

        obj = {"aaaaaaaaaa": 1, "bb": 2}
        with caplog.at_level(logging.WARNING):
            jsonutils.remove_json_defaults(obj, {"aaaaaaaaaa": 2}, warn_if_not_default=True, obj_id="X")
        assert obj == {"aaaaaaaaaa": 1, "bb": 2}
        assert "aaaaaaaaaa field not equals 2 in object {'aaaaaaaaaa': 1, 'b X" in caplog.text

    def test_a_nul_character_is_removed_from_a_string_value(self):
        from pyutilz.text.strings.jsonutils import _normalize_for_pg_json

        assert _normalize_for_pg_json("a\x00b") == "ab"

    def test_json_keys_are_not_sorted_by_default(self):
        from pyutilz.text.strings.jsonutils import json_pg_dumps

        assert json_pg_dumps({"b": 1, "a": 2})._raw == '{"b":1,"a":2}'

    def test_a_missing_attribute_is_not_warned_about_by_default(self, caplog):
        import logging

        from pyutilz.text.strings.jsonutils import get_jsonlist_property

        with caplog.at_level(logging.WARNING):
            assert get_jsonlist_property([{"a": 1}, {"b": 2}], "a") == [1]
        assert "No a field" not in caplog.text

    def test_a_missing_attribute_is_warned_about_when_asked(self, caplog):
        import logging

        from pyutilz.text.strings.jsonutils import get_jsonlist_property

        with caplog.at_level(logging.WARNING):
            assert get_jsonlist_property([{"a": 1}, {"b": 2}], "a", verbose=True) == [1]
        assert "No a field for element {'b': 2}" in caplog.text

    def test_the_multi_property_helper_returns_indices_by_default(self):
        from pyutilz.text.strings.jsonutils import get_jsonlist_properties

        result = get_jsonlist_properties([{"id": 1, "n": "a"}], ["id"])
        assert isinstance(result, tuple)

    def test_the_multi_property_helper_warns_about_a_missing_property_when_verbose(self, caplog):
        import logging

        from pyutilz.text.strings.jsonutils import get_jsonlist_properties

        with caplog.at_level(logging.WARNING):
            get_jsonlist_properties([{"id": 1}, {"n": "x"}], ["id"], verbose=True)
        assert "No id field for element {'n': 'x'}" in caplog.text


class TestJsonutilsRecheckSurvivors:
    """What the jsonutils re-check still found: the recursion level passed down and its default when None, the debug text
    of an attribute that raises (the object is the holder, not the error), the default of warn_if_not_default, and
    non-string keys serialized with a NUL in another value (jsonutils.py 95, 105, 124, 248, 378)."""

    def test_each_level_of_nesting_is_one_deeper_than_the_last(self):
        from pyutilz.text.strings import jsonutils

        data = {"a": {"b": {"c": 1}}}
        result = jsonutils.jsonize_atrtributes(data, max_recursion_level=1)
        assert result["a"] == {"b": data["a"]["b"]}
        assert result["a"]["b"] is data["a"]["b"]

    def test_a_missing_starting_level_counts_as_zero(self):
        from pyutilz.text.strings import jsonutils

        class _In:
            x = 1

        assert jsonutils.jsonize_atrtributes(_In(), recursion_level=None, max_recursion_level=0) == {"x": 1}

    def test_a_failing_attribute_is_named_with_its_holder_not_the_error(self, caplog):
        import logging

        from pyutilz.text.strings import jsonutils

        class _Holder:
            @property
            def bad(self):
                raise ValueError("boom")

        with caplog.at_level(logging.DEBUG):
            jsonutils.jsonize_atrtributes(_Holder())
        assert "'bad' of <" in caplog.text

    def test_a_non_default_value_is_not_warned_about_unless_asked(self, caplog):
        import logging

        from pyutilz.text.strings import jsonutils

        obj = {"a": 1}
        with caplog.at_level(logging.WARNING):
            jsonutils.remove_json_defaults(obj, {"a": 2})
        assert obj == {"a": 1}
        assert caplog.text == ""

    def test_non_string_keys_are_serialized_and_a_nul_in_a_value_is_removed(self):
        from pyutilz.text.strings.jsonutils import json_pg_dumps

        assert json_pg_dumps({1: "x", "a": "\x00"})._raw == '{"1":"x","a":""}'


class TestPreSerializedAdapterSurvivors:
    """The psycopg2 adapter: it is its own ISQLQuote adapter and offers nothing for other protocols, a quoted ASCII
    literal comes out verbatim, and a non-ASCII byte in the text shows as the replacement character in str()
    (jsonutils.py 329-348)."""

    def test_the_adapter_answers_only_for_the_sql_quote_protocol(self):
        from psycopg2.extensions import ISQLQuote

        obj = jsonutils.json_pg_dumps({"a": "b"})
        assert obj.__conform__(ISQLQuote) is obj
        assert obj.__conform__(str) is None

    def test_an_ascii_document_is_quoted_verbatim(self):
        obj = jsonutils.json_pg_dumps({"a": "b"})
        assert obj.getquoted() == b'\'{"a":"b"}\''

    def test_a_non_ascii_byte_shows_as_the_replacement_character_in_str(self):
        obj = jsonutils.json_pg_dumps({"a": "\u00e9"})
        assert str(obj) == '\'{"a":"\ufffd"}\''


class TestDepthLimitIdentity:
    """A container within the depth limit is rebuilt, one past it is returned as the same object, and a missing starting
    level counts as zero (jsonutils.py 95, 124)."""

    def test_a_level_within_the_limit_is_rebuilt_and_the_next_one_is_kept_as_is(self):
        data = {"a": {"b": {"c": 1}}}
        result = jsonutils.jsonize_atrtributes(data, max_recursion_level=1)
        assert result["a"] is not data["a"]
        assert result["a"]["b"] is data["a"]["b"]

    def test_a_missing_starting_level_is_rebuilt_at_a_zero_limit(self):
        data = {"x": 1}
        result = jsonutils.jsonize_atrtributes(data, recursion_level=None, max_recursion_level=0)
        assert result == {"x": 1}
        assert result is not data


class TestObjectAttributeDepth:
    """An object's attribute that is a mapping is one level below the object, so it is rebuilt at the limit of one
    (jsonutils.py 95)."""

    def test_a_mapping_held_by_an_attribute_is_rebuilt_one_level_below_the_object(self):
        class _Holder:
            pass

        holder = _Holder()
        holder.inner = {"k": {"j": 1}}
        result = jsonutils.jsonize_atrtributes(holder, max_recursion_level=1)
        assert result["inner"] is not holder.inner
        assert result["inner"]["k"] is holder.inner["k"]
