"""Behaviour of ``pyutilz.core.pythonlib.numerics``: float parsing that keeps a decimal comma a failure rather
than a 10x misreading, the integer digit census, and the digit-level measures taken from the decimal rendering.
Expected values follow the functions' own docstrings."""

from __future__ import annotations

import pytest

from pyutilz.core.pythonlib.numerics import (
    count_trailing_zeros,
    float_distinct_digits_percent,
    integer_digits,
    is_float,
    to_float,
)


class TestFloatParsing:
    @pytest.mark.parametrize("text", ["1.5", "-2", "+3.25", "1,234", "1,234.56"])
    def test_plain_and_genuine_thousands_forms_are_floats(self, text):
        assert is_float(text) is True

    @pytest.mark.parametrize("text", ["abc", "1,5", ""])
    def test_non_numbers_and_decimal_commas_are_not_floats(self, text):
        assert is_float(text) is False

    def test_none_is_not_a_float(self):
        assert is_float(None) is False

    def test_genuine_thousands_separators_are_stripped(self):
        assert to_float("1,234.5") == 1234.5

    def test_a_decimal_comma_raises_instead_of_reading_fifteen(self):
        with pytest.raises(ValueError):
            to_float("1,5")


class TestIntegerDigits:
    def test_count_and_distinct_digits(self):
        count, distinct = integer_digits(1224)
        assert count == 4
        assert distinct == {1, 2, 4}

    def test_a_single_digit(self):
        assert integer_digits(7) == (1, {7})


class TestDigitMeasures:
    def test_distinct_digit_share_matches_the_docstring_examples(self):
        assert float_distinct_digits_percent(11.882, precision=3) == pytest.approx(0.6)
        assert float_distinct_digits_percent(25.05063, precision=5) == pytest.approx(5 / 7)

    def test_trailing_zeros_of_the_fractional_part(self):
        assert count_trailing_zeros(100.0, precision=5) == 5

    def test_round_number_at_zero_precision_has_no_fractional_zeros(self):
        assert count_trailing_zeros(100.0, precision=0) == 0

    def test_a_single_trailing_zero_after_the_fractional_digits(self):
        assert count_trailing_zeros(1.30e-6, precision=8) == 1
