"""Coverage-gate, oversize-warning and empty-word pins for text.similarity.sentences (mutation survivors)."""

from __future__ import annotations

import logging

import pytest

from pyutilz.text.similarity import sentences as ss


class TestEmptyWords:
    @pytest.mark.parametrize("threshold", [0, 1])
    def test_an_empty_word_against_a_non_empty_one_scores_zero(self, threshold):
        assert ss._word_pair_similarity("", "ABC", threshold) == 0.0
        assert ss._word_pair_similarity("ABC", "", threshold) == 0.0


class TestCoverageGateSides:
    # "THE" is a stop word, so the stripped side is empty and passes any coverage requirement; "X" is
    # not, and matches nothing, so the side that keeps it fails.

    def test_the_default_max_side_checks_the_longer_side_only(self):
        # Equal lengths: the default side is A, whose only content word is absent -> pass.
        assert ss.sentences_similarity(["THE"], ["X"], required_coverage=1.0, stop_words=["THE"]) == 0.0

    def test_the_max_side_ignores_the_min_side_when_it_would_fail(self):
        assert ss.sentences_similarity(["THE"], ["X"], required_coverage=1.0, stop_words=["THE"], coverage_side="max") == 0.0

    def test_the_min_side_checks_the_shorter_side_and_fails_on_its_content(self):
        assert ss.sentences_similarity(["THE"], ["X"], required_coverage=1.0, stop_words=["THE"], coverage_side="min") is None

    def test_the_min_side_passes_when_the_shorter_side_is_all_stop_words(self):
        assert ss.sentences_similarity(["X"], ["THE"], required_coverage=1.0, stop_words=["THE"], coverage_side="min") == 0.0

    def test_the_max_side_fails_on_its_own_content(self):
        assert ss.sentences_similarity(["X"], ["THE"], required_coverage=1.0, stop_words=["THE"], coverage_side="max") is None

    def test_both_fails_when_the_min_side_fails_even_if_max_passes(self):
        assert ss.sentences_similarity(["THE"], ["X"], required_coverage=1.0, stop_words=["THE"], coverage_side="both") is None

    def test_both_passes_when_each_side_passes(self):
        # A sentence made only of stop words is kept unstripped, so THE matches THE exactly.
        assert ss.sentences_similarity(["THE"], ["THE"], required_coverage=1.0, stop_words=["THE"], coverage_side="both") == 1.0

    def test_an_invalid_coverage_side_is_rejected_when_the_gate_is_active(self):
        with pytest.raises(ValueError):
            ss.sentences_similarity(["A"], ["A"], required_coverage=1.0, coverage_side="bogus")

    def test_the_default_side_is_max_when_the_caller_passes_none(self):
        # Default side is max: A ("X") is the checked side and fails, so the result is None.
        assert ss.sentences_similarity(["X"], ["THE"], required_coverage=1.0, stop_words=["THE"]) is None


class TestCoverageRequirements:
    # ALPHA matches ALPHA exactly (1.0); BETA against ZZZZ scores 0.0.

    def test_required_matched_words_alone_counts_covered_words(self):
        assert ss.sentences_similarity(["ALPHA", "BETA"], ["ALPHA", "ZZZZ"], required_matched_words=1) == 0.5

    def test_required_matched_words_above_the_covered_count_fails(self):
        assert ss.sentences_similarity(["ALPHA", "BETA"], ["ALPHA", "ZZZZ"], required_matched_words=2) is None

    def test_required_coverage_is_a_fraction_of_the_content_words(self):
        assert ss.sentences_similarity(["ALPHA", "BETA"], ["ALPHA", "ZZZZ"], required_coverage=0.5) == 0.5

    def test_required_coverage_above_the_covered_fraction_fails_at_the_default_threshold(self):
        assert ss.sentences_similarity(["ALPHA", "BETA"], ["ALPHA", "ZZZZ"], required_coverage=1.0) is None

    def test_a_low_min_word_similarity_counts_an_unmatched_word_as_covered(self):
        # BETA is unmatched (the shorter side has only ALPHA), so its matched similarity is the sentinel
        # 0.0 -- below 0.05, so it is not covered; the sentinel must stay below the threshold.
        assert ss.sentences_similarity(["ALPHA", "BETA"], ["ALPHA"], required_coverage=1.0, min_word_similarity=0.05) is None

    def test_an_unmatched_word_on_the_longer_b_side_is_not_covered_at_a_low_threshold(self):
        assert ss.sentences_similarity(["ALPHA"], ["ALPHA", "BETA"], required_coverage=1.0, min_word_similarity=0.05) is None

    def test_the_min_side_fails_when_its_shorter_side_is_not_fully_covered(self):
        assert ss.sentences_similarity(["ALPHA", "BETA"], ["ALPHA", "ZZZZ"], required_coverage=1.0, coverage_side="min") is None

    def test_the_min_side_passes_a_required_matched_word_count_it_meets(self):
        assert ss.sentences_similarity(["ALPHA", "BETA"], ["ALPHA", "ZZZZ"], required_matched_words=1, coverage_side="min") == 0.5


class TestOversizeWarning:
    def test_a_sentence_over_the_threshold_on_side_a_warns_with_both_counts_in_order(self, caplog):
        caplog.set_level(logging.WARNING)
        ss.sentences_similarity(["A"] * 51, ["B"])
        assert "sentences_similarity called with N_a=51, N_b=1 tokens, exceeding the safe threshold of 50" in caplog.text

    def test_a_sentence_over_the_threshold_on_side_b_warns_with_counts_in_order(self, caplog):
        caplog.set_level(logging.WARNING)
        ss.sentences_similarity(["A"], ["B"] * 51)
        assert "sentences_similarity called with N_a=1, N_b=51 tokens" in caplog.text

    def test_a_sentence_at_exactly_the_threshold_does_not_warn_on_side_a(self, caplog):
        caplog.set_level(logging.WARNING)
        ss.sentences_similarity(["A"] * 50, ["B"])
        assert "exceeding the safe threshold" not in caplog.text

    def test_a_sentence_at_exactly_the_threshold_does_not_warn_on_side_b(self, caplog):
        caplog.set_level(logging.WARNING)
        ss.sentences_similarity(["A"], ["B"] * 50)
        assert "exceeding the safe threshold" not in caplog.text

    def test_either_side_over_the_threshold_is_enough_to_warn(self, caplog):
        caplog.set_level(logging.WARNING)
        ss.sentences_similarity(["A"] * 51, ["B"])
        assert "exceeding the safe threshold" in caplog.text

    def test_the_warning_explains_the_quadratic_slowdown(self, caplog):
        caplog.set_level(logging.WARNING)
        ss.sentences_similarity(["A"] * 51, ["B"])
        assert "expect quadratic-ish slowdown on long inputs." in caplog.text
