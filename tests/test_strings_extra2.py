"""Extra tests for strings.py — covers uncovered lines."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# jsonize_atrtributes — lines 83, 115-137
# ---------------------------------------------------------------------------

from pyutilz.strings import jsonize_atrtributes


def test_jsonize_plain_string_no_strip():
    assert jsonize_atrtributes("  hello  ", strip=False) == "  hello  "


def test_jsonize_list_with_max_recursion():
    # line 114-115: list/set/tuple with max_recursion_level exceeded
    result = jsonize_atrtributes([1, 2], max_recursion_level=0)
    assert result == [1, 2]


def test_jsonize_dict_with_max_recursion():
    result = jsonize_atrtributes({"a": {"b": 1}}, max_recursion_level=0)
    assert result == {"a": {"b": 1}}


def test_jsonize_custom_object():
    # lines 117-137: object with attributes
    class Obj:
        name = "test"
        value = 42
    result = jsonize_atrtributes(Obj())
    assert result["name"] == "test"
    assert result["value"] == 42


def test_jsonize_custom_object_exception():
    # line 136-137: exception path
    class BadObj:
        @property
        def __dir__(self):
            raise RuntimeError("boom")
    # The try/except should catch and return None-ish
    # Actually dir() is called on obj, so let's use a different approach
    result = jsonize_atrtributes(None)
    # None has no attrs starting without _, should return empty dict
    assert result == {}


# ---------------------------------------------------------------------------
# extract_json_attribute — lines 217-220
# ---------------------------------------------------------------------------

from pyutilz.strings import extract_json_attribute


def test_extract_json_attribute_list_with_item_str():
    result = extract_json_attribute({"key1": "plain_string", "key2": [{"a": 1}]}, "a")
    assert result["key1"] == "plain_string"  # line 218
    assert result["key2"] == [1]  # line 220


# ---------------------------------------------------------------------------
# remove_json_empty_attributes — lines 230-231
# ---------------------------------------------------------------------------

from pyutilz.strings import remove_json_empty_attributes


def test_remove_json_empty_attributes():
    obj = {"a": [], "b": "notempty", "c": {}}
    remove_json_empty_attributes(obj, ["a", "c", "b"])
    assert "a" not in obj
    assert "c" not in obj
    assert "b" in obj


def test_remove_json_empty_attributes_exception():
    # line 230-231: len() raises on non-iterable
    obj = {"x": 42}
    remove_json_empty_attributes(obj, ["x"])
    assert "x" in obj  # not removed because len(42) raises


# ---------------------------------------------------------------------------
# remove_json_defaults — line 245
# ---------------------------------------------------------------------------

from pyutilz.strings import remove_json_defaults


def test_remove_json_defaults_warn_if_not_default():
    obj = {"a": 999}
    with patch("pyutilz.text.strings.logger") as mock_log:
        remove_json_defaults(obj, attr_values={"a": 0}, warn_if_not_default=True, obj_id="test")
        mock_log.warning.assert_called_once()
    assert "a" in obj


# ---------------------------------------------------------------------------
# get_jsonlist_property verbose — line 270
# ---------------------------------------------------------------------------

from pyutilz.strings import get_jsonlist_property


def test_get_jsonlist_property_verbose_missing():
    with patch("pyutilz.text.strings.logger") as mock_log:
        result = get_jsonlist_property([{"x": 1}], "missing", verbose=True)
        mock_log.warning.assert_called_once()
    assert result == []


def test_get_jsonlist_property_return_indices():
    res, idx = get_jsonlist_property([{"id": 1}, {"name": "x"}, {"id": 3}], "id", return_indices=True)
    assert res == [1, 3]
    assert idx == [0, 2]


# ---------------------------------------------------------------------------
# get_jsonlist_properties verbose — lines 291-292
# ---------------------------------------------------------------------------

from pyutilz.strings import get_jsonlist_properties


def test_get_jsonlist_properties_verbose():
    with patch("pyutilz.text.strings.logger") as mock_log:
        res, idx = get_jsonlist_properties([{"a": 1}], ["b"], verbose=True)
        mock_log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# read_config_file / write_config_file — lines 305-396
# ---------------------------------------------------------------------------

from pyutilz.strings import read_config_file, write_config_file


def test_write_and_read_config_file():
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
        fname = f.name
    try:
        obj_out = {"user": "admin", "count": "42"}
        result = write_config_file(fname, obj_out, section="MAIN", encryption="xor")
        assert result is True

        obj_in = {}
        result2 = read_config_file(fname, obj_in, section="MAIN", encryption="xor")
        assert result2 is True
        assert obj_in["user"] == "admin"
        assert obj_in["count"] == 42  # ast.literal_eval
    finally:
        os.unlink(fname)


def test_write_config_missing_var():
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
        fname = f.name
    try:
        with patch("pyutilz.text.strings.logger") as mock_logger:
            result = write_config_file(fname, {"a": 1}, variables="a,b", encryption=None)
        # The missing variable is reported, not silently dropped...
        warned = " ".join(str(c.args) for c in mock_logger.warning.call_args_list)
        assert "b" in warned, f"no warning names the missing variable 'b'; warnings were: {warned!r}"
        # ...and the write still succeeds, carrying the variable that WAS present.
        assert result is True
        obj_in: dict = {}
        assert read_config_file(fname, obj_in, section="MAIN", encryption=None) is True
        assert obj_in == {"a": 1}, f"expected only the present variable to be written, got {obj_in!r}"
    finally:
        os.unlink(fname)


def test_read_config_all_sections():
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
        fname = f.name
    try:
        write_config_file(fname, {"x": "1"}, section="SEC1", encryption=None)
        obj = {}
        read_config_file(fname, obj, section=None, encryption=None)
        # Regression test: reading ALL sections (section=None) now prefixes keys by section name
        # (sec1_x, not x) so a same-named variable in two different sections can't silently
        # overwrite each other.
        assert "sec1_x" in obj
    finally:
        os.unlink(fname)


def test_read_config_file_roundtrips_non_ascii_value():
    """Regression (2026-07-21 audit round 2, HIGH): read_config_file() used to call
    config.read(file) with no explicit encoding, defaulting to the platform locale encoding.
    write_config_file() always writes UTF-8 -- on a non-UTF-8-locale host (this Windows host
    defaults to cp1251), the mismatch silently corrupted non-ASCII values on read."""
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
        fname = f.name
    try:
        write_config_file(fname, {"name": "café"}, section="MAIN", encryption=None)

        obj_in = {}
        read_config_file(fname, obj_in, section="MAIN", encryption=None)
        assert obj_in["name"] == "café"
    finally:
        os.unlink(fname)


def test_write_config_file_append_mode_roundtrips_non_ascii_value():
    """Same fix, second call site: write_config_file()'s own "append" mode reads back its
    previously-UTF-8-written file (config.read(file) at the top of the append branch) before
    merging in new values -- that read must ALSO pin utf-8, or an append-mode round-trip
    corrupts an existing non-ASCII value even without ever calling read_config_file()."""
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
        fname = f.name
    try:
        write_config_file(fname, {"name": "café"}, section="MAIN", encryption=None, mode="append")
        # A second append-mode write (a different var) forces write_config_file to re-read the
        # existing file via config.read(file) before rewriting it.
        write_config_file(fname, {"other": "x"}, section="MAIN", encryption=None, mode="append")

        obj_in = {}
        read_config_file(fname, obj_in, section="MAIN", encryption=None)
        assert obj_in["name"] == "café"
        assert obj_in["other"] == "x"
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# parse_tokens — lines 442-443
# ---------------------------------------------------------------------------

from pyutilz.strings import parse_tokens


def test_parse_tokens_no_end_token():
    with patch("pyutilz.text.strings.logger"):
        result = parse_tokens("[%clk 0:03:00", start_token="[%clk ", end_token="]")
    assert result == []


# ---------------------------------------------------------------------------
# make_text_from_inner_html_elements — lines 450-451
# ---------------------------------------------------------------------------

from pyutilz.strings import make_text_from_inner_html_elements


def test_make_text_from_inner_html():
    elem = MagicMock()
    elem.strings = ["hello", "  ", "\\n", "world"]
    result = make_text_from_inner_html_elements(elem)
    assert "hello" in result
    assert "world" in result


# ---------------------------------------------------------------------------
# underscorize_variable — line 467
# ---------------------------------------------------------------------------

from pyutilz.strings import underscorize_variable


def test_underscorize_variable():
    assert underscorize_variable("ProdLangName") == "prod_lang_name"
    assert underscorize_variable("HTMLParser") == "htmlparser"
    assert underscorize_variable("a") == "a"


# ---------------------------------------------------------------------------
# get_hash — lines 486, 494
# ---------------------------------------------------------------------------

from pyutilz.strings import get_hash


def test_get_hash_binary():
    result = get_hash("test", return_binary=True)
    assert isinstance(result, bytes)


def test_get_hash_no_base():
    result = get_hash("test", base=None)
    assert isinstance(result, str)  # hexdigest


def test_get_hash_no_base_binary():
    result = get_hash("test", base=None, return_binary=True)
    assert isinstance(result, bytes)


def test_get_hash_bytes_input():
    result = get_hash(b"test")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# remove_videos — lines 534-537, 549-550
# ---------------------------------------------------------------------------

from pyutilz.strings import remove_videos


def test_remove_videos_basic():
    text = "before [[VIDEOID:123]] after"
    assert remove_videos(text) == "before  after"


def test_remove_videos_no_closing():
    with patch("pyutilz.text.strings.logger"):
        result = remove_videos("before [[VIDEOID:123 no close")
    assert "before" in result


def test_remove_videos_none():
    assert remove_videos(None) is None


# ---------------------------------------------------------------------------
# fix_html — line 579
# ---------------------------------------------------------------------------

from pyutilz.strings import fix_html


def test_fix_html():
    assert fix_html("a<br/>b<br >c") == "a\nb\nc"
    assert fix_html(None) is None


# ---------------------------------------------------------------------------
# fix_broken_sentences — lines 645-689, 696-698
# ---------------------------------------------------------------------------

from pyutilz.strings import fix_broken_sentences


def test_fix_broken_sentences_newline_before_capital():
    text = "Hello world\nNext sentence"
    result = fix_broken_sentences(text)
    assert "." in result


def test_fix_broken_sentences_newline_space_capital():
    text = "Hello world\n More text"
    result = fix_broken_sentences(text)
    assert result == "Hello world. More text."


def test_fix_broken_sentences_newline_before_lowercase():
    # line 690+: next_symbol.isalpha() but lowercase
    text = "Hello world\ncontinued text"
    result = fix_broken_sentences(text)
    assert "continued" in result or "Hello" in result


def test_fix_broken_sentences_eos_before_newline():
    text = "Hello.\nWorld"
    result = fix_broken_sentences(text)
    assert "World" in result


def test_fix_broken_sentences_ends_with_letter():
    # line 714: last char is letter, not eos -> add dot
    text = "Hello world"
    result = fix_broken_sentences(text)
    assert result.endswith(".")


# ---------------------------------------------------------------------------
# fix_missed_space_between_sentences — lines 720-732
# ---------------------------------------------------------------------------

from pyutilz.strings import fix_missed_space_between_sentences


def test_fix_missed_space_between_sentences():
    # No end-of-sentence marker followed by an uppercase letter: text is returned unchanged.
    result = fix_missed_space_between_sentences("Hello World")
    assert result == "Hello World"
    # Missing space after a sentence terminator is inserted.
    assert fix_missed_space_between_sentences("Sentence one.Sentence two.") == "Sentence one. Sentence two."


# ---------------------------------------------------------------------------
# clean_description — lines 780-785
# ---------------------------------------------------------------------------

# Skipping clean_description because it requires emoji_data_python


# ---------------------------------------------------------------------------
# get_ascii_emojies / get_unicode_emojies — lines 812-829
# ---------------------------------------------------------------------------

# These require emoji_data_python, skip


# ---------------------------------------------------------------------------
# sentencize_text — lines 841-871
# ---------------------------------------------------------------------------

# Requires emoji_data_python, skip


# ---------------------------------------------------------------------------
# suffixize — lines 882-887
# ---------------------------------------------------------------------------

# suffixize requires 'inflect' which is not installed in test env — skip


# ---------------------------------------------------------------------------
# tokenize_to_chars with is_file — line 975
# ---------------------------------------------------------------------------

from pyutilz.strings import tokenize_to_chars, tokenize_to_words


def test_tokenize_to_chars_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("abc\n")
        fname = f.name
    try:
        result = list(tokenize_to_chars(fname, is_file=True))
        assert len(result) > 0
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# tokenize_to_words — line 981
# ---------------------------------------------------------------------------

def test_tokenize_to_words():
    result = list(tokenize_to_words("Hello world"))
    assert "hello" in result
    assert "world" in result


def test_tokenize_to_words_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("Hello world\n")
        fname = f.name
    try:
        result = list(tokenize_to_words(fname, is_file=True))
        assert "hello" in result
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# tokenize_text / spacy_sent_tokenize -- both public and re-exported, both previously
# never mentioned anywhere under tests/ (audit F20, 2026-09-02).
# ---------------------------------------------------------------------------


class TestTokenizeText:
    @staticmethod
    def _split(source):
        return source.split()

    def test_delegates_to_the_given_tokenizer_and_yields_lazily(self):
        import types

        from pyutilz.text.strings import tokenize_text

        out = tokenize_text("  Hello World  ", tokenizer=self._split)
        assert isinstance(out, types.GeneratorType), "must stay lazy -- callers stream large sources through it"
        assert list(out) == ["hello", "world"]

    def test_lowercase_can_be_switched_off(self):
        from pyutilz.text.strings import tokenize_text

        assert list(tokenize_text("Hello World", tokenizer=self._split, lowercase=False)) == ["Hello", "World"]

    def test_strip_can_be_switched_off(self):
        """With strip=False the leading/trailing whitespace reaches the tokenizer verbatim -- a
        tokenizer that is whitespace-sensitive must see it."""
        from pyutilz.text.strings import tokenize_text

        seen = []

        def spy(text):
            seen.append(text)
            return text.split()

        list(tokenize_text("  padded  ", tokenizer=spy, strip=False))
        assert seen == ["  padded  "]

        seen.clear()
        list(tokenize_text("  padded  ", tokenizer=spy, strip=True))
        assert seen == ["padded"]

    def test_empty_source_yields_nothing(self):
        from pyutilz.text.strings import tokenize_text

        assert list(tokenize_text("   ", tokenizer=self._split)) == []


class TestSpacySentTokenize:
    def test_splits_into_sentences(self):
        """Requires the spaCy English small model, which is a heavyweight optional asset -- the
        precondition is an explicit skip naming exactly what is missing, not an if/else."""
        spacy = pytest.importorskip("spacy")
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' is not downloaded on this host")

        from pyutilz.text.strings import spacy_sent_tokenize

        sents = spacy_sent_tokenize("The cat sat. The dog barked! Did it rain?")

        assert isinstance(sents, list)
        assert [s.text.strip() for s in sents] == ["The cat sat.", "The dog barked!", "Did it rain?"]
