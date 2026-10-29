"""Behaviour of ``pyutilz.text.similarity.sentences``: identical phrases score 1.0, an empty side gives no score, the gliding
window checks the suffix as well as the prefix (so ``MADRID`` matches inside ``REALMADRID``), an unrelated pair scores
low, and the opt-in coverage gate turns a partial match into no score while leaving the default score alone."""

from __future__ import annotations

import pytest

from pyutilz.text.similarity import sentences as sim


class TestSentenceSimilarity:
    def test_identical_phrases_score_one(self):
        assert sim.sentences_similarity(["HELLO", "WORLD"], ["HELLO", "WORLD"]) == pytest.approx(1.0)

    def test_an_empty_side_gives_no_score(self):
        assert sim.sentences_similarity([], ["TEST"]) is None

    def test_unrelated_words_score_low(self):
        assert sim.sentences_similarity(["HELLO"], ["WORLD"]) < 0.5

    def test_a_word_inside_a_longer_word_scores_above_an_unrelated_pair(self):
        assert sim.sentences_similarity(["REALMADRID"], ["MADRID"]) > sim.sentences_similarity(["REALMADRID"], ["WORLD"])

    def test_only_one_of_two_words_matched_scores_partially(self):
        assert sim.sentences_similarity(["ALPHA", "BETA"], ["ALPHA"]) == pytest.approx(0.5)


class TestGlidingDistance:
    def test_a_suffix_window_is_tested(self):
        assert sim._best_gliding_distance("MADRID", "REALMADRID", 6, 10) == 0

    def test_a_prefix_window_is_tested(self):
        assert sim._best_gliding_distance("REAL", "REALMADRID", 4, 10) == 0


class TestCoverageGate:
    def test_a_partial_match_is_refused_when_two_words_must_match(self):
        assert sim.sentences_similarity(["ALPHA", "BETA"], ["ALPHA"], required_matched_words=2) is None

    def test_the_same_partial_match_scores_without_the_gate(self):
        assert sim.sentences_similarity(["ALPHA", "BETA"], ["ALPHA"]) == pytest.approx(0.5)


class TestWordPairSimilarity:
    def test_an_empty_word_scores_zero(self):
        assert sim._word_pair_similarity("", "abc", 1) == 0.0

    def test_identical_words_score_one(self):
        assert sim._word_pair_similarity("abc", "abc", 1) == 1.0

    def test_a_word_at_exactly_the_length_threshold_is_still_scored(self):
        assert sim._word_pair_similarity("abc", "abz", 3) == pytest.approx(2 / 3)

    def test_a_word_below_the_length_threshold_scores_zero(self):
        assert sim._word_pair_similarity("ab", "abz", 3) == 0.0

    def test_a_shared_prefix_earns_the_prefix_floor_scaled_by_its_length(self):
        assert sim._word_pair_similarity("abc", "abcdef", 1) == pytest.approx(0.9 + 0.1 * 3 / 6)

    def test_a_single_shared_character_does_not_earn_the_prefix_floor(self):
        assert sim._word_pair_similarity("a", "ab", 1) == pytest.approx(0.75)

    def test_equal_length_words_use_the_edit_distance_alone(self):
        assert sim._word_pair_similarity("abcd", "abce", 1) == pytest.approx(0.75)

    def test_a_short_word_inside_a_longer_one_is_scored_by_the_window_and_the_edit_distance(self):
        assert sim._word_pair_similarity("ab", "xxab", 1) == pytest.approx(0.75)
