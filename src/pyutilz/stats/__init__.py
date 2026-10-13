"""Statistical tests + utilities reusable across projects.

Submodules:
* ``normality`` -- D'Agostino K^2 + Anderson-Darling normality tests
  (numba-jitted) + combined verdict helper.
"""

from .normality import normality_verdict, phi_cdf

from . import normality

__all__ = ["normality", "normality_verdict", "phi_cdf"]
