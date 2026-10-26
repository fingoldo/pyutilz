"""Behaviour of the two ``basics`` modules: ``pyutilz.text.strings.basics`` (slicing, padding, slugging,
hashing, path shortening) and ``pyutilz.text.similarity.basics`` (edit-distance and contiguous-block
similarity). The expected values are the ones the functions' own docstrings state."""

from __future__ import annotations

import pytest

from pyutilz.text.similarity.basics import contigous_strings_similarity, levenshtein_strings_similarity
from pyutilz.text.strings.basics import (
    camel_case_split,
    find_between,
    get_hash,
    parse_tokens,
    rpad,
    shorten_path,
    slugify,
    strip_characters,
    strip_doubled_characters,
)


class TestFindBetween:
    def test_returns_the_text_between_the_markers(self):
        assert find_between("a[x]b", "[", "]") == "x"

    def test_missing_start_marker_returns_none(self):
        assert find_between("abc", "[", "]") is None


class TestStripping:
    def test_strip_characters_removes_every_listed_character(self):
        assert strip_characters("a.b,c", [".", ","]) == "abc"

    def test_strip_doubled_characters_collapses_runs_to_one(self):
        assert strip_doubled_characters("a  b   c", " ") == "a b c"


class TestPadding:
    def test_rpad_fills_to_the_width_with_the_default_symbol(self):
        assert rpad("ab", 5) == "ab..."

    def test_rpad_never_truncates_a_longer_string(self):
        assert rpad("ab", 1) == "ab"

    def test_rpad_uses_the_given_symbol(self):
        assert rpad("ab", 4, "-") == "ab--"


class TestSlugify:
    def test_spaces_and_punctuation_become_hyphens(self):
        assert slugify("Hello, World!") == "Hello-World"

    def test_lowercase_flag_lowercases(self):
        assert slugify("AB", lowercase=True) == "ab"

    def test_non_ascii_is_transliterated_when_unicode_is_disallowed(self):
        assert slugify("Ünïcode Test", allow_unicode=False) == "Unicode-Test"


class TestCamelCaseSplit:
    def test_splits_at_lower_to_upper_boundaries(self):
        assert camel_case_split("camelCaseString") == ["camel", "Case", "String"]

    def test_an_acronym_run_is_kept_whole(self):
        assert camel_case_split("HTTPServer") == ["HTTPServer"]


class TestHashing:
    def test_md5_hex_digest_of_a_string(self):
        assert get_hash("abc", "md5", 16) == "900150983CD24FB0D6963F7D28E17F72"  # pragma: allowlist secret

    def test_the_default_base_is_64(self):
        assert get_hash("abc") == "kAFQmDzST7DWlj99KOF/cg=="  # pragma: allowlist secret


class TestShortenPath:
    def test_prefix_is_replaced(self, tmp_path):
        prefix = str(tmp_path)
        assert shorten_path(prefix + "/x/y", prefix=prefix, prefix_replacement="~") == "~/x/y"


class TestParseTokens:
    def test_clock_annotation_is_extracted(self):
        assert parse_tokens("[%clk 0:01:00]x") == ["0:01:00"]


class TestSimilarity:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("", "", 1.0),
            ("MeasureOIS21", "MeasureOIS18", 0.8333333333333334),
        ],
    )
    def test_levenshtein_similarity(self, a, b, expected):
        assert levenshtein_strings_similarity(a, b) == pytest.approx(expected)

    def test_contiguous_similarity_of_identical_strings_is_full(self):
        assert contigous_strings_similarity("MeosureOIS21qwe", "MeosureOIS21qwe") == (1.0, "MeosureOIS21qwe")

    def test_contiguous_similarity_of_empty_strings(self):
        assert contigous_strings_similarity("", "") == (1.0, "")

    def test_contiguous_similarity_keeps_the_common_left_and_right_blocks(self):
        sim, root = contigous_strings_similarity("MeosureOIS21qwe", "MeasureOIS18qwe")
        assert sim == pytest.approx(1 / 3)
        assert root == "Meqwe"
