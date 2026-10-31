"""Behaviour of the two ``basics`` modules: ``pyutilz.text.strings.basics`` (slicing, padding, slugging,
hashing, path shortening) and ``pyutilz.text.similarity.basics`` (edit-distance and contiguous-block
similarity). The expected values are the ones the functions' own docstrings state."""

from __future__ import annotations

import pytest

from pyutilz.text.similarity.basics import contigous_strings_similarity, levenshtein_strings_similarity, normalize_sentence
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

    def test_a_common_left_block_alone_is_the_root(self):
        # "abc" is shared on the left, "X" and "Y" differ on the right: 3 of 4 characters.
        assert contigous_strings_similarity("abcX", "abcY") == (0.75, "abc")

    def test_a_common_right_block_alone_is_the_root(self):
        assert contigous_strings_similarity("Xabc", "Yabc") == (0.75, "abc")

    def test_a_prefix_of_the_longer_string_is_the_whole_shorter_string(self):
        # "ab" is a prefix of "abc": the shorter string covers the whole match, 2 of 3 characters.
        sim, root = contigous_strings_similarity("ab", "abc")
        assert sim == pytest.approx(2 / 3)
        assert root == "ab"

    def test_left_and_right_blocks_are_joined_into_the_root(self):
        # shared "A" on the left and "B" on the right, 2 of 5 characters, root "AB".
        assert contigous_strings_similarity("AXYZB", "AQQB") == (pytest.approx(0.4), "AB")

    def test_the_longer_string_sets_the_denominator(self):
        assert contigous_strings_similarity("a", "abcd")[0] == pytest.approx(0.25)


class TestLevenshteinBounds:
    def test_a_string_against_empty_has_zero_similarity(self):
        assert levenshtein_strings_similarity("abc", "") == 0.0
        assert levenshtein_strings_similarity("", "a") == 0.0


class TestNormalizeSentence:
    def test_default_symbols_are_replaced_by_the_placeholder(self):
        assert normalize_sentence("A.B,C") == ["A", "B", "C"]

    def test_the_text_is_uppercased(self):
        assert normalize_sentence("abc") == ["ABC"]

    def test_a_custom_placeholder_splits_the_tokens(self):
        assert normalize_sentence("A-B", placeholder="_") == ["A", "B"]

    def test_repeated_tokens_are_kept_once_in_first_seen_order(self):
        assert normalize_sentence("x x y") == ["X", "Y"]

    def test_an_empty_sentence_has_no_tokens(self):
        assert normalize_sentence("") == []

    def test_bsports_is_accepted_and_ignored(self):
        assert normalize_sentence("A", True) == ["A"]

    def test_an_abbreviation_in_the_middle_is_removed(self):
        assert normalize_sentence("ALPHA FC BETA", abbreviations=["FC"]) == ["ALPHA", "BETA"]

    def test_a_leading_abbreviation_is_removed_with_its_separator(self):
        assert normalize_sentence("FC SPORTS", abbreviations=["FC"]) == ["SPORTS"]

    def test_a_trailing_abbreviation_is_removed_with_its_separator(self):
        assert normalize_sentence("ALPHA FC", abbreviations=["FC"]) == ["ALPHA"]

    def test_an_abbreviation_inside_a_word_is_not_removed(self):
        assert normalize_sentence("AFC BETA", abbreviations=["FC"]) == ["AFC", "BETA"]

    def test_a_word_starting_with_an_abbreviation_is_not_cut(self):
        assert normalize_sentence("FCBAYERN", abbreviations=["FC"]) == ["FCBAYERN"]

    def test_a_word_ending_with_an_abbreviation_is_not_cut(self):
        assert normalize_sentence("BAYERNFC", abbreviations=["FC"]) == ["BAYERNFC"]

    def test_a_space_then_an_abbreviation_glued_to_a_word_is_not_cut(self):
        assert normalize_sentence("ALPHA FCX", abbreviations=["FC"]) == ["ALPHA", "FCX"]


class TestSimilarityBoundaries:
    def test_an_empty_string_against_a_nonempty_one_shares_nothing(self):
        assert contigous_strings_similarity("", "abc") == (0.0, "")
        assert contigous_strings_similarity("abc", "") == (0.0, "abc")

    def test_a_shorter_common_prefix_keeps_the_longer_string_as_root_when_fully_matched(self):
        # "ab" is a prefix of "abcd": 2 of 4 characters match and the root is the whole first string.
        assert contigous_strings_similarity("abcd", "ab") == (0.5, "abcd")
