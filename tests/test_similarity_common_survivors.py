"""Logger name, validation message, coverage boundary and NFC pins for text.similarity._common (mutation survivors)."""

from __future__ import annotations

import re

import pytest

from pyutilz.text.similarity import _common as common


class TestLoggerName:
    def test_the_logger_keeps_the_historical_flat_module_channel(self):
        assert common.logger.name == "pyutilz.text.similarity"


class TestCoverageSideValidation:
    def test_an_unknown_side_is_rejected_with_the_full_message(self):
        expected = re.escape("coverage_side must be one of ('min', 'max', 'both'), got 'bogus'")
        with pytest.raises(ValueError, match=expected):
            common._validate_coverage_side("bogus")

    @pytest.mark.parametrize("side", ["min", "max", "both"])
    def test_each_valid_side_is_accepted(self, side):
        assert common._validate_coverage_side(side) is None


class TestCoverageBoundary:
    def test_a_matched_similarity_exactly_at_the_threshold_counts_as_covered(self):
        # 0.5 >= 0.5 is covered; with a strict > the single covered word would not count.
        assert common._check_word_coverage(["A", "B"], [0.5, 0.0], 0.5, None, 1, set()) is True

    def test_a_matched_similarity_just_below_the_threshold_is_not_covered(self):
        assert common._check_word_coverage(["A", "B"], [0.4999, 0.0], 0.5, None, 1, set()) is False


class TestNfcNormalisation:
    def test_a_decomposed_string_is_composed(self):
        assert common._nfc_words(["café"]) == ["café"]

    def test_a_non_string_entry_passes_through_unchanged(self):
        assert common._nfc_words(["café", 5]) == ["café", 5]
