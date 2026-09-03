"""Numeric parsing and digit-level inspection helpers.

Split out of the historical flat ``pyutilz.core.pythonlib`` module; re-exported from the
package ``__init__`` to preserve the public import surface.
"""

import re

from ._common import Any, Tuple, njit

# A comma is a THOUSANDS separator only where it actually separates groups of three digits
# ("1,234", "1,234.56"). Stripping commas unconditionally silently reinterprets a decimal comma --
# "1,5" (1.5 in most of Europe) became 15.0, a 10x error with no warning -- so anything that is not
# this shape keeps its commas and simply fails to parse as a float.
_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?$")


def _strip_thousands_separators(text: str) -> str:
    """Return ``text`` with commas removed IF they occupy genuine thousands positions, else unchanged."""
    return text.replace(",", "") if _THOUSANDS_RE.match(text) else text

# ----------------------------------------------------------------------------------------------------------------------------
# Numerics
# ----------------------------------------------------------------------------------------------------------------------------


def is_float(string: Any) -> bool:
    """Checks if `string` (with GENUINE thousands-separator commas stripped) can be parsed as a float.

    A comma that is not a thousands separator (e.g. the decimal comma of ``"1,5"``) is left in
    place, so such a value reports ``False`` rather than being silently read as ``15.0``.
    """
    try:
        float(_strip_thousands_separators(str(string)))
        return True
    except ValueError:
        return False


def to_float(string: Any) -> float:
    """Parses `string` as a float, stripping GENUINE thousands-separator commas first.

    Raises ``ValueError`` for a decimal comma such as ``"1,5"`` instead of returning ``15.0``.
    """
    return float(_strip_thousands_separators(str(string)))


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
    """Share of DISTINCT decimal digits among the digits ``number`` shows at ``precision``.

    Both counts are taken from the decimal rendering, not from arithmetic on the binary float:
    ``int(frac_part * 10**precision)`` truncated (``0.05063 * 100000`` is ``5062.999...`` -> 5062)
    and ``integer_digits`` could not see a fractional part's LEADING zeros at all (``0.005`` at
    precision 3 counted as one digit), so numerator and denominator were both wrong for 12.8% of
    random 5-decimal values.

    >>> float_distinct_digits_percent(11.882, precision=3)
    0.6
    >>> float_distinct_digits_percent(25.05063, precision=5)
    0.7142857142857143
    """
    digits_str = format(abs(float(number)), f".{max(precision, 0)}f").replace(".", "")
    ntotal = len(digits_str)
    return len(set(digits_str)) / ntotal if ntotal > 0 else 1.0


def count_trailing_zeros(number: float, precision: int = 5) -> int:
    """Count the trailing zeros of ``number``'s FRACTIONAL part, formatted to ``precision`` digits.

    >>> count_trailing_zeros(1.30e-6, precision=8)
    1
    >>> count_trailing_zeros(100.0, precision=5)
    5
    >>> count_trailing_zeros(100.0, precision=0)
    0
    """
    if precision <= 0:
        # ``format(x, ".0f")`` has no decimal separator at all, so the separator ``break`` below
        # never fires and the INTEGER part's zeros get counted as fractional ones.
        return 0
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
