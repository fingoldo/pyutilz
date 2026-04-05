"""Extra tests for strings.py — covers functions not in test_strings.py."""

import math
import json
import pytest
from datetime import datetime, date
from collections import Counter
from xml.etree.ElementTree import Element


# ---------------------------------------------------------------------------
# camel_case_split
# ---------------------------------------------------------------------------

from pyutilz.strings import camel_case_split


@pytest.mark.parametrize("inp,expected", [
    ("CamelCase", ["Camel", "Case"]),
    ("HTMLParser", ["HTMLParser"]),  # all-upper prefix not split
    ("a", ["a"]),
    ("alreadylower", ["alreadylower"]),
])
def test_camel_case_split(inp, expected):
    assert camel_case_split(inp) == expected


# ---------------------------------------------------------------------------
# entropy helpers
# ---------------------------------------------------------------------------

from pyutilz.strings import get_entropy_stats, entropy, entropy_rate, compute_entropy_stats


def test_get_entropy_stats_basic():
    cond, stats = get_entropy_stats(iter("abcabc"), model_order=1)
    assert sum(stats.values()) > 0


def test_get_entropy_stats_empty():
    cond, stats = get_entropy_stats(iter(""), model_order=1)
    assert len(stats) == 0


def test_entropy_uniform():
    c = Counter({"a": 10, "b": 10})
    e = entropy(c, 20)
    assert abs(e - 1.0) < 0.01  # log2(2) = 1


def test_entropy_single():
    c = Counter({"a": 5})
    assert entropy(c, 5) == 0.0  # -1*log2(1) = 0


def test_entropy_rate_basic():
    cond, stats = get_entropy_stats(iter("ababababab"), model_order=1)
    er = entropy_rate(cond, stats)
    assert 0 <= er <= math.log2(26)


def test_compute_entropy_stats_short():
    raw, rate = compute_entropy_stats("aaaa", order=0)
    assert raw is not None and rate is not None


def test_compute_entropy_stats_empty():
    raw, rate = compute_entropy_stats("", order=0)
    assert raw is None and rate is None


# ---------------------------------------------------------------------------
# naive_entropy_rate (needs numpy)
# ---------------------------------------------------------------------------

from pyutilz.strings import naive_entropy_rate


def test_naive_entropy_rate_uniform():
    e = naive_entropy_rate("ab" * 50)
    assert abs(e - 1.0) < 0.05


def test_naive_entropy_rate_single_char():
    e = naive_entropy_rate("aaaa")
    assert e == 0.0


# ---------------------------------------------------------------------------
# stringify_dict
# ---------------------------------------------------------------------------

from pyutilz.strings import stringify_dict


def test_stringify_dict_basic():
    assert stringify_dict({"a": 1, "b": 2}) == "a=1,b=2"


def test_stringify_dict_custom_sep():
    assert stringify_dict({"x": "y"}, sep="|") == "x=y"


def test_stringify_dict_empty():
    assert stringify_dict({}) == ""


# ---------------------------------------------------------------------------
# json_serial
# ---------------------------------------------------------------------------

from pyutilz.strings import json_serial


def test_json_serial_datetime():
    dt = datetime(2024, 1, 15, 12, 30, 0)
    assert json_serial(dt) == "2024-01-15T12:30:00"


def test_json_serial_date():
    d = date(2024, 1, 15)
    assert json_serial(d) == "2024-01-15"


def test_json_serial_unsupported():
    with pytest.raises(TypeError):
        json_serial(set())


# ---------------------------------------------------------------------------
# sub_elem
# ---------------------------------------------------------------------------

from pyutilz.strings import sub_elem


def test_sub_elem_basic():
    root = Element("root")
    child = sub_elem(root, "child", text="hello")
    assert child.tag == "child"
    assert child.text == "hello"


def test_sub_elem_no_text():
    root = Element("root")
    child = sub_elem(root, "empty")
    assert child.text is None


# ---------------------------------------------------------------------------
# jsonize_atrtributes
# ---------------------------------------------------------------------------

from pyutilz.strings import jsonize_atrtributes


def test_jsonize_string():
    assert jsonize_atrtributes("  hello  ") == "hello"


def test_jsonize_number():
    assert jsonize_atrtributes(42) == 42


def test_jsonize_dict():
    result = jsonize_atrtributes({"a": " x ", "b": 5})
    assert result == {"a": "x", "b": 5}


def test_jsonize_list():
    result = jsonize_atrtributes([" a ", 1])
    assert result == ["a", 1]


def test_jsonize_max_recursion():
    result = jsonize_atrtributes({"a": {"b": " c "}}, max_recursion_level=0)
    assert result == {"a": {"b": " c "}}


# ---------------------------------------------------------------------------
# remove_json_attributes / leave_json_attributes
# ---------------------------------------------------------------------------

from pyutilz.strings import remove_json_attributes, leave_json_attributes


def test_remove_json_attributes_none():
    remove_json_attributes(None, ["a"])  # should not raise


def test_leave_json_attributes_none():
    leave_json_attributes(None, ["a"])  # should not raise


# ---------------------------------------------------------------------------
# extract_json_attribute
# ---------------------------------------------------------------------------

from pyutilz.strings import extract_json_attribute


def test_extract_json_attribute_nested():
    data = {"cat": {"uid": "1", "name": "A"}, "sub": {"uid": "2", "name": "B"}}
    result = extract_json_attribute(data, "name")
    assert result == {"cat": "A", "sub": "B"}


def test_extract_json_attribute_list_of_dicts():
    data = [{"name": "x"}, {"name": "y"}]
    assert extract_json_attribute(data, "name") == ["x", "y"]


def test_extract_json_attribute_missing_key():
    data = [{"a": 1}, {"name": "z"}]
    result = extract_json_attribute(data, "name")
    assert "z" in result


# ---------------------------------------------------------------------------
# remove_json_empty_attributes
# ---------------------------------------------------------------------------

from pyutilz.strings import remove_json_empty_attributes


def test_remove_json_empty_attributes_removes():
    d = {"a": [], "b": "hi"}
    remove_json_empty_attributes(d, ["a", "b"])
    assert "a" not in d
    assert "b" in d


def test_remove_json_empty_attributes_keeps_nonempty():
    d = {"a": [1]}
    remove_json_empty_attributes(d, ["a"])
    assert "a" in d


# ---------------------------------------------------------------------------
# remove_json_defaults
# ---------------------------------------------------------------------------

from pyutilz.strings import remove_json_defaults


def test_remove_json_defaults_removes_matching():
    d = {"status": "ok", "count": 0}
    remove_json_defaults(d, attr_values={"status": "ok"})
    assert "status" not in d
    assert "count" in d


def test_remove_json_defaults_none_input():
    remove_json_defaults(None, attr_values={"a": 1})  # no crash


# ---------------------------------------------------------------------------
# get_jsonlist_property / get_jsonlist_properties
# ---------------------------------------------------------------------------

from pyutilz.strings import get_jsonlist_property, get_jsonlist_properties


def test_get_jsonlist_property_dict_input():
    assert get_jsonlist_property({"id": 5}, "id") == 5


def test_get_jsonlist_property_return_indices():
    data = [{"x": 1}, {"y": 2}, {"x": 3}]
    vals, idxs = get_jsonlist_property(data, "x", return_indices=True)
    assert vals == [1, 3]
    assert idxs == [0, 2]


def test_get_jsonlist_properties_basic():
    data = [{"a": 1, "b": 2, "c": 3}]
    result, indices = get_jsonlist_properties(data, ["a", "c"])
    assert result == [{"a": 1, "c": 3}]


# ---------------------------------------------------------------------------
# find_between
# ---------------------------------------------------------------------------

from pyutilz.strings import find_between


def test_find_between_empty_start():
    assert find_between("hello]world", "", "]") == "hello"


def test_find_between_empty_end():
    assert find_between("[hello world", "[", "") == "hello world"


def test_find_between_none_input():
    assert find_between("", "[", "]") is None


# ---------------------------------------------------------------------------
# parse_tokens
# ---------------------------------------------------------------------------

from pyutilz.strings import parse_tokens


def test_parse_tokens_chess_clocks():
    text = "1. e4 { [%clk 0:03:00] } e5 { [%clk 0:02:59] }"
    assert parse_tokens(text) == ["0:03:00", "0:02:59"]


def test_parse_tokens_empty():
    assert parse_tokens("no tokens here") == []


# ---------------------------------------------------------------------------
# underscorize_variable
# ---------------------------------------------------------------------------

from pyutilz.strings import underscorize_variable


@pytest.mark.parametrize("inp,expected", [
    ("ProdLangName", "prod_lang_name"),
    ("already_lower", "already_lower"),
    ("A", "a"),
    ("ABCDef", "abcdef"),
])
def test_underscorize_variable(inp, expected):
    assert underscorize_variable(inp) == expected


# ---------------------------------------------------------------------------
# strip_characters / strip_doubled_characters
# ---------------------------------------------------------------------------

from pyutilz.strings import strip_characters, strip_doubled_characters


def test_strip_characters_empty_list():
    assert strip_characters("hello", []) == "hello"


def test_strip_doubled_characters_basic():
    assert strip_doubled_characters("aaa,,bb", [","]) == "aaa,bb"


def test_strip_doubled_characters_spaces():
    assert "  " not in strip_doubled_characters("a   b", [" "])


# ---------------------------------------------------------------------------
# rpad
# ---------------------------------------------------------------------------

from pyutilz.strings import rpad


def test_rpad_shorter():
    assert rpad("ab", 5, ".") == "ab..."


def test_rpad_exact():
    assert rpad("abcde", 5, ".") == "abcde"


# ---------------------------------------------------------------------------
# fix_duplicate_tokens
# ---------------------------------------------------------------------------

from pyutilz.strings import fix_duplicate_tokens


def test_fix_duplicate_tokens_double_spaces():
    assert "  " not in fix_duplicate_tokens("a  b  c")


def test_fix_duplicate_tokens_quadruple_dots():
    assert "...." not in fix_duplicate_tokens("a....b")


# ---------------------------------------------------------------------------
# unescape_html
# ---------------------------------------------------------------------------

from pyutilz.strings import unescape_html


def test_unescape_html_entities():
    assert unescape_html("&lt;b&gt;") == "<b>"


def test_unescape_html_ampersand():
    assert unescape_html("&amp;") == "&"


# ---------------------------------------------------------------------------
# fix_quotations
# ---------------------------------------------------------------------------

from pyutilz.strings import fix_quotations


def test_fix_quotations_curly():
    assert fix_quotations("\u201Chello\u201D") == "'hello'"


def test_fix_quotations_none():
    assert fix_quotations(None) is None


# ---------------------------------------------------------------------------
# fix_spaces
# ---------------------------------------------------------------------------

from pyutilz.strings import fix_spaces


def test_fix_spaces_removes_space_before_comma():
    assert fix_spaces("hello , world") == "hello, world"


def test_fix_spaces_none():
    assert fix_spaces(None) is None


# ---------------------------------------------------------------------------
# fix_broken_sentences
# ---------------------------------------------------------------------------

from pyutilz.strings import fix_broken_sentences


def test_fix_broken_sentences_newline_lowercase():
    result = fix_broken_sentences("Hello world\ncontinuation here")
    assert isinstance(result, str)
    assert "\n" not in result or result.endswith(".")


def test_fix_broken_sentences_none():
    assert fix_broken_sentences(None) is None


# ---------------------------------------------------------------------------
# merge_punctuation_signs
# ---------------------------------------------------------------------------

from pyutilz.strings import merge_punctuation_signs


def test_merge_punctuation_basic():
    result = merge_punctuation_signs("ok!!")
    assert isinstance(result, list)
    assert "!!" in result


def test_merge_punctuation_no_merge():
    result = merge_punctuation_signs("abc")
    assert result == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# ensure_space_after_comma
# ---------------------------------------------------------------------------

from pyutilz.strings import ensure_space_after_comma


def test_ensure_space_after_comma_inserts():
    assert ensure_space_after_comma("a,b") == "a, b"


def test_ensure_space_after_comma_dot_fix():
    assert ensure_space_after_comma("end,.") == "end."


# ---------------------------------------------------------------------------
# suffixize (requires inflect — skip if missing)
# ---------------------------------------------------------------------------

from pyutilz.strings import suffixize


@pytest.mark.skipif(True, reason="Requires inflect module")
def test_suffixize_plural():
    assert suffixize("job", 2) == "jobs"


# ---------------------------------------------------------------------------
# shorten_path
# ---------------------------------------------------------------------------

from pyutilz.strings import shorten_path


def test_shorten_path_prefix():
    assert shorten_path("/a/b/c/file.txt", prefix="/a/b/c/", prefix_replacement="~/") == "~/file.txt"


def test_shorten_path_default():
    assert shorten_path("DEFAULT", prefix="", default="DEFAULT", default_replacement="<none>") == "<none>"


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------

from pyutilz.strings import slugify


def test_slugify_removes_special():
    assert slugify("Hello, World!") == "Hello-World"


def test_slugify_lowercase():
    assert slugify("ABC", lowercase=True) == "abc"


def test_slugify_no_unicode():
    result = slugify("Ünïcödé", allow_unicode=False)
    assert all(ord(c) < 128 for c in result)


# ---------------------------------------------------------------------------
# get_hash
# ---------------------------------------------------------------------------

from pyutilz.strings import get_hash


def test_get_hash_base64():
    h = get_hash("test", algo="sha256", base=64)
    assert isinstance(h, str) and len(h) > 0


def test_get_hash_hex():
    h = get_hash("test", algo="md5", base=None)
    assert all(c in "0123456789abcdef" for c in h)


def test_get_hash_binary():
    h = get_hash("test", algo="md5", base=None, return_binary=True)
    assert isinstance(h, bytes)


def test_get_hash_bytes_input():
    h = get_hash(b"raw bytes", algo="sha256")
    assert isinstance(h, str)


# ---------------------------------------------------------------------------
# json_pg_dumps (needs psycopg2)
# ---------------------------------------------------------------------------

from pyutilz.strings import json_pg_dumps


def test_json_pg_dumps_basic():
    result = json_pg_dumps({"a": 1})
    assert result is not None
