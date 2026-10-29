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


class TestGreedyMatch:
    def test_two_pairs_are_picked_best_first_and_each_is_recorded_on_both_sides(self):
        res, matched_a, matched_b = sim._greedy_match([[0.9, 0.1], [0.2, 0.8]], 2, 2, 2)
        assert res == pytest.approx(1.7)
        assert matched_a == pytest.approx([0.9, 0.8])
        assert matched_b == pytest.approx([0.9, 0.8])

    def test_a_single_pick_leaves_the_other_words_unmatched(self):
        res, matched_a, matched_b = sim._greedy_match([[0.9, 0.1], [0.2, 0.8]], 2, 2, 1)
        assert res == pytest.approx(0.9)
        assert matched_a == pytest.approx([0.9, 0.0])
        assert matched_b == pytest.approx([0.9, 0.0])

    def test_a_row_is_not_skipped_after_its_neighbour_is_taken(self):
        res, _matched_a, _matched_b = sim._greedy_match([[0.9, 0.1], [0.2, 0.8]], 2, 2, 2)
        assert res == pytest.approx(1.7)

    def test_a_tie_goes_to_the_later_column(self):
        _res, _matched_a, matched_b = sim._greedy_match([[0.5, 0.5], [0.0, 0.0]], 2, 2, 1)
        assert matched_b == pytest.approx([0.0, 0.5])

    def test_an_all_zero_matrix_scores_zero(self):
        res, matched_a, matched_b = sim._greedy_match([[0.0, 0.0], [0.0, 0.0]], 2, 2, 2)
        assert res == 0.0
        assert matched_a == [0.0, 0.0]
        assert matched_b == [0.0, 0.0]


class TestWordCoverageChecker:
    def test_a_sentence_of_only_stopwords_passes_trivially(self):
        assert sim._check_word_coverage(["the"], [0.0], 0.5, 1.0, 5, {"the"}) is True

    def test_a_match_exactly_at_the_similarity_floor_counts(self):
        assert sim._check_word_coverage(["a"], [0.5], 0.5, None, 1, set()) is True

    def test_the_absolute_count_is_inclusive(self):
        assert sim._check_word_coverage(["a", "b"], [0.9, 0.0], 0.5, None, 1, set()) is True
        assert sim._check_word_coverage(["a", "b"], [0.9, 0.0], 0.5, None, 2, set()) is False

    def test_the_fraction_is_inclusive(self):
        assert sim._check_word_coverage(["a", "b"], [0.9, 0.0], 0.5, 0.5, None, set()) is True
        assert sim._check_word_coverage(["a", "b"], [0.9, 0.0], 0.5, 0.6, None, set()) is False

    def test_both_requirements_must_hold(self):
        assert sim._check_word_coverage(["a", "b"], [0.9, 0.9], 0.5, 0.5, 3, set()) is False

    def test_stopwords_leave_the_denominator(self):
        assert sim._check_word_coverage(["the", "cat"], [0.0, 0.9], 0.5, 1.0, None, {"the"}) is True


class TestCoverageGateSides:
    def test_max_checks_the_longer_sentence_and_min_the_shorter(self):
        a, b = ["a"], ["x", "y"]
        matched_a, matched_b = [0.9], [0.0, 0.0]
        assert sim._coverage_gate_passes(a, b, matched_a, matched_b, "max", 0.5, None, 1, None) is False
        assert sim._coverage_gate_passes(a, b, matched_a, matched_b, "min", 0.5, None, 1, None) is True

    def test_both_requires_each_side(self):
        a, b = ["a"], ["x", "y"]
        matched_a, matched_b = [0.9], [0.0, 0.0]
        assert sim._coverage_gate_passes(a, b, matched_a, matched_b, "both", 0.5, None, 1, None) is False

    def test_no_requirement_always_passes(self):
        assert sim._coverage_gate_passes(["a"], ["b"], [0.0], [0.0], "both", 0.5, None, None, None) is True
