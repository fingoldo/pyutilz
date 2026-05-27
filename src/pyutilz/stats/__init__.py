"""Statistical tests + utilities reusable across projects.

Submodules:
* ``normality`` -- D'Agostino K^2 + Anderson-Darling normality tests
  (numba-jitted) + combined verdict helper.
"""
from . import normality  # noqa: F401

__all__ = ["normality"]
