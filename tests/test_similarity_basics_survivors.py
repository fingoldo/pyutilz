"""Boundary and abbreviation pins for pyutilz.text.similarity.basics (mutation survivors)."""

from __future__ import annotations

import pytest

from pyutilz.text.similarity.basics import (
    contigous_strings_similarity,
    levenshtein_strings_similarity,
    normalize_sentence,
)


class TestLevenshteinSimilarity:
    @pytest.mark.parametrize(("a", "b"), [("MeasureOIS21", "MeasureOIS18"), ("abc", "xyz1"), ("", "abc")])
    def test_the_distance_and_the_normaliser_are_symmetric(self, a, b):
        assert levenshtein_strings_similarity(a, b) == levenshtein_strings_similarity(b, a)

    def test_a_known_pair_scores_as_documented(self):
        assert levenshtein_strings_similarity("MeasureOIS21", "MeasureOIS18") == pytest.approx(0.8333333333333334)


class TestContiguousSimilarity:
    def test_an_empty_string_against_a_non_empty_one_scores_zero(self):
        assert contigous_strings_similarity("", "abc") == (0.0, "")

    def test_two_empty_strings_are_identical(self):
        assert contigous_strings_similarity("", "") == (1.0, "")

    def test_identical_strings_keep_their_full_root(self):
        assert contigous_strings_similarity("abc", "abc") == (1.0, "abc")

    def test_a_shared_prefix_and_suffix_covering_the_whole_word_keeps_the_root_as_written(self):
        # The prefix "ab" and the suffix "c" together do not cover "abxc" in full, so the root is rebuilt
        # from them and the mismatch is only in the middle; the root must still be the longer string.
        assert contigous_strings_similarity("abxc", "abc") == (0.75, "abxc")

    def test_a_shared_prefix_only_gives_the_prefix_as_root(self):
        assert contigous_strings_similarity("ab", "ac") == (0.5, "a")

    def test_a_shared_suffix_only_gives_the_suffix_as_root(self):
        assert contigous_strings_similarity("xc", "yc") == (0.5, "c")

    def test_the_score_is_the_larger_shared_block_not_the_smaller(self):
        # With no shared prefix the shared suffix sets the score; a min() in place of max() scores zero here.
        score, _ = contigous_strings_similarity("xc", "yc")
        assert score == 0.5


class TestNormalizeSentenceAbbreviations:
    def test_a_term_between_words_is_removed(self):
        assert normalize_sentence("BAYERN FC MUNICH", abbreviations=["FC"]) == ["BAYERN", "MUNICH"]

    def test_a_term_at_the_start_is_removed(self):
        assert normalize_sentence("FC BAYERN", abbreviations=["FC"]) == ["BAYERN"]

    def test_a_term_at_the_end_is_removed(self):
        assert normalize_sentence("BAYERN FC", abbreviations=["FC"]) == ["BAYERN"]

    def test_a_term_inside_a_word_is_kept(self):
        assert normalize_sentence("ABFC MUNICH", abbreviations=["FC"]) == ["ABFC", "MUNICH"]

    def test_a_word_starting_with_the_term_is_kept_at_the_start(self):
        assert normalize_sentence("FCX MUNICH", abbreviations=["FC"]) == ["FCX", "MUNICH"]

    def test_a_word_ending_with_the_term_is_kept_at_the_end(self):
        assert normalize_sentence("XFC", abbreviations=["FC"]) == ["XFC"]

    def test_a_word_beginning_with_the_term_is_not_cut_in_the_middle(self):
        assert normalize_sentence("BAYERN FCX", abbreviations=["FC"]) == ["BAYERN", "FCX"]

    def test_a_multi_word_term_matches_after_doubled_spaces_are_collapsed(self):
        assert normalize_sentence("BAYERN SPORTING  CLUB", abbreviations=["SPORTING CLUB"]) == ["BAYERN"]

    def test_bsports_is_accepted_and_ignored(self):
        assert normalize_sentence("A", bSports=True) == normalize_sentence("A", bSports=False)
