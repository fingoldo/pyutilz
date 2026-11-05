"""Decimal-separator stop for count_trailing_zeros in core.pythonlib.numerics (mutation survivor)."""

from __future__ import annotations

from pyutilz.core.pythonlib.numerics import count_trailing_zeros


def test_the_decimal_point_stops_the_count_before_the_integer_part():
    # "100.0" has one fractional zero; the zeros of the integer part must not be counted.
    assert count_trailing_zeros(100.0, precision=1) == 1


def test_a_round_integer_counts_only_its_fractional_zeros():
    assert count_trailing_zeros(100.0, precision=5) == 5
