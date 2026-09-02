"""Numeric parsing and digit-level inspection helpers.

Split out of the historical flat ``pyutilz.core.pythonlib`` module; re-exported from the
package ``__init__`` to preserve the public import surface.
"""

from ._common import Tuple, njit

# ----------------------------------------------------------------------------------------------------------------------------
# Numerics
# ----------------------------------------------------------------------------------------------------------------------------


def is_float(string):
    """Checks if `string` (with thousands-separator commas stripped) can be parsed as a float."""
    try:
        float(str(string).replace(",", ""))
        return True
    except ValueError:
        return False


def to_float(string):
    """Parses `string` as a float, stripping thousands-separator commas first."""
    return float(str(string).replace(",", ""))


@njit()
def integer_digits(n: int) -> Tuple[int, set]:
    """Counts the digits of a non-negative integer `n` and collects the set of distinct digit values it contains.

    Returns:
        Tuple of (total digit count, set of distinct digits).
    """
    # Function to count digits in an integer
    digits = set()
    ntotal = 0
    while n > 0:
        ntotal += 1
        digit = n % 10
        digits.add(digit)
        n //= 10
    return ntotal, digits


# @njit()
def float_distinct_digits_percent(number: float, precision: int = 5) -> float:
    """
    >>>float_distinct_digits_percent(11.882, precision=3)
    0.6
    """
    # Extract the integer and fractional parts of the float
    int_part = int(number)
    frac_part = round(abs(number - int_part), precision)  # rounding needed for cases like 11.882
    if precision:
        frac_part = min(frac_part, 1 - 10**-precision)  # a rounding carry (e.g. 0.99999996 -> 1.0) must not push the fractional-digit count past `precision`

    # Initialize a set to store unique digits
    unique_digits = set()
    ntotal = 0

    # Count digits in the integer part
    nsubtotal, digits = integer_digits(abs(int_part))
    unique_digits.update(digits)
    ntotal += nsubtotal

    # Count digits in the fractional part (up to a certain precision)
    if precision:
        frac_digits = int(frac_part * (10**precision))  # Adjust precision as needed
        nsubtotal, digits = integer_digits(frac_digits)
        unique_digits.update(digits)
        ntotal += nsubtotal

    # Count the number of unique digits
    return len(unique_digits) / ntotal if ntotal > 0 else 1.0


def count_trailing_zeros(number: float, precision: int = 5) -> int:
    """Count the trailing zeros of ``number``'s FRACTIONAL part, formatted to ``precision`` digits.

    >>> count_trailing_zeros(1.30e-6, precision=8)
    1
    >>> count_trailing_zeros(100.0, precision=5)
    5
    """
    # Convert the float to a string
    num_str = format(number, f".{precision}f")
    nzeros = 0
    for char in num_str[::-1]:
        if char in ",.e+-":
            # The decimal separator ends the fractional part: continuing past it would count the
            # integer part's zeros too, so every round number was mis-classified.
            break
        if char == "0":
            nzeros += 1
        else:
            break

    return nzeros
