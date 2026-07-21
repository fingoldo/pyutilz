"""Numba-jitted statistical normality tests.

Two formal hypothesis tests for whether a 1-D sample comes from a
Normal distribution:

* ``dagostino_k2`` -- D'Agostino & Pearson K^2 omnibus moment-based
  test. Returns (K2, p, Z_skew, Z_kurt). Matches
  ``scipy.stats.normaltest`` to floating-point on n >= 20.
* ``anderson_darling_normal`` -- Anderson-Darling A* statistic for
  Normal with mean + sigma estimated from data (Stephens 1986
  finite-sample correction). Returns (A*, p) with p approximated
  via the Stephens piecewise polynomial.
* ``normality_verdict`` -- run both, classify (reject_normal=True/False),
  emit human-readable verdict + raw stats.

Both kernels are ``@njit(cache=True, fastmath=True)`` so n=1M runs in
single-digit milliseconds on a recent CPU.

Why this exists: heuristic moment-cutoff diagnostics ("|skew|<0.3 and
|excess_kurt|<0.5 => Normal") miss bimodal / asymmetric / fat-tailed
distributions whose moments happen to land inside the cutoff band.
Both formal tests catch those (D'Agostino via the standardised moment
transforms; A-D via tail-weight integration of the empirical CDF).

Reference:
* D'Agostino & Pearson (1973), "Tests for departure from normality"
* Anscombe & Glynn (1983), "Distribution of the kurtosis statistic"
* Stephens (1986), "Tests based on EDF statistics", in D'Agostino &
  Stephens (eds) "Goodness-of-fit techniques"
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

    def njit(*args: Any, **kwargs: Any) -> Callable[..., Any]:  # type: ignore[no-redef]
        """No-op stand-in for ``numba.njit`` when numba isn't installed: returns the function unchanged.

        Supports both bare-decorator (``@njit``) and call-with-kwargs (``@njit(cache=True, ...)``) forms.
        """
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]  # type: ignore[no-any-return]

        def deco(f: Callable[..., Any]) -> Callable[..., Any]:
            """Return ``f`` unchanged (no-op JIT decorator)."""
            return f
        return deco


# =============================================================================
# Standard-normal CDF helpers (numba-friendly).
# =============================================================================

@njit(cache=True, fastmath=True)
def phi_cdf(z: float) -> float:
    """Standard Normal CDF Phi(z) = 0.5 * erfc(-z / sqrt(2))."""
    return 0.5 * math.erfc(-z * 0.7071067811865475)


@njit(cache=True, fastmath=True)
def log_phi_cdf(z: float) -> float:
    """Numerically stable log(Phi(z)).

    For z < -5 the naive ``log(0.5 * erfc(-z/sqrt(2)))`` underflows to
    -inf when erfc loses all bits. Use the Mill's-ratio asymptote
    log(Phi(z)) ~ -z^2/2 - log(-z) - 0.5*log(2*pi) for z << 0 (matches
    Abramowitz & Stegun 26.2.12 truncated to leading term, accurate to
    O(1/z^2) for z < -5).
    """
    if z < -5.0:
        return -0.5 * z * z - math.log(-z) - 0.9189385332046727  # 0.5 ln(2 pi)
    p = 0.5 * math.erfc(-z * 0.7071067811865475)
    if p < 1e-300:
        return -690.0
    return math.log(p)


# =============================================================================
# D'Agostino's K^2 test (joint skew + kurt -> chi^2(2))
# =============================================================================

@njit(cache=True, fastmath=True)
def dagostino_k2(x: np.ndarray) -> tuple:
    """D'Agostino & Pearson K^2 omnibus test for normality.

    Returns (K2_statistic, p_value, Z_skew, Z_kurt). Under H0 (normal),
    K2 ~ chi^2(2); p = exp(-K2/2) is the closed form for chi^2(2)
    upper tail. Each of Z_skew and Z_kurt is a transformed moment
    standardised to approximate standard Normal under H0, so |Z| > 1.96
    marks one rejection at alpha=0.05.

    n >= 20 required for the finite-sample transforms; returns NaN
    stat/p on smaller samples.
    """
    n = x.size
    if n < 20:
        return (np.nan, np.nan, np.nan, np.nan)

    # Moments
    mu = 0.0
    for i in range(n):
        mu += x[i]
    mu /= n

    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for i in range(n):
        d = x[i] - mu
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2
    m2 /= n
    m3 /= n
    m4 /= n

    if m2 <= 0.0:
        return (0.0, 1.0, 0.0, 0.0)

    sqrt_m2 = math.sqrt(m2)
    g1 = m3 / (sqrt_m2 * sqrt_m2 * sqrt_m2)
    g2 = m4 / (m2 * m2) - 3.0

    n_f = float(n)

    # ---- Z_skew via D'Agostino (1970) -----------------------------------
    Y = g1 * math.sqrt(((n_f + 1.0) * (n_f + 3.0)) / (6.0 * (n_f - 2.0)))
    beta2 = 3.0 * (n_f * n_f + 27.0 * n_f - 70.0) * (n_f + 1.0) * (n_f + 3.0) / ((n_f - 2.0) * (n_f + 5.0) * (n_f + 7.0) * (n_f + 9.0))
    W2 = -1.0 + math.sqrt(2.0 * (beta2 - 1.0))
    # scipy uses ``delta = 1/sqrt(0.5 * log(W2))`` -- W2 not sqrt(W2).
    delta = 1.0 / math.sqrt(0.5 * math.log(W2))
    alpha_skew = math.sqrt(2.0 / (W2 - 1.0))
    arg = Y / alpha_skew
    Z1 = delta * math.log(arg + math.sqrt(arg * arg + 1.0))

    # ---- Z_kurt via Anscombe-Glynn (1983) -------------------------------
    Eg2 = -6.0 / (n_f + 1.0)
    Vg2 = (24.0 * n_f * (n_f - 2.0) * (n_f - 3.0)) / ((n_f + 1.0) * (n_f + 1.0) * (n_f + 3.0) * (n_f + 5.0))
    if Vg2 <= 0.0:
        Z2 = 0.0
    else:
        X = (g2 - Eg2) / math.sqrt(Vg2)
        sqrt_b1_g2 = (
            (6.0 * (n_f * n_f - 5.0 * n_f + 2.0))
            / ((n_f + 7.0) * (n_f + 9.0))
            * math.sqrt((6.0 * (n_f + 3.0) * (n_f + 5.0)) / (n_f * (n_f - 2.0) * (n_f - 3.0)))
        )
        if sqrt_b1_g2 <= 0.0:
            Z2 = X
        else:
            A = 6.0 + (8.0 / sqrt_b1_g2) * (2.0 / sqrt_b1_g2 + math.sqrt(1.0 + 4.0 / (sqrt_b1_g2 * sqrt_b1_g2)))
            term = (1.0 - 2.0 / A) / (1.0 + X * math.sqrt(2.0 / (A - 4.0)))
            if term <= 0.0:
                Z2 = 0.0
            else:
                cube_root = math.copysign(math.fabs(term) ** (1.0 / 3.0), term)
                Z2 = (1.0 - 2.0 / (9.0 * A) - cube_root) / math.sqrt(2.0 / (9.0 * A))

    K2 = Z1 * Z1 + Z2 * Z2
    # chi^2(2) upper-tail in closed form: F(k) = 1 - exp(-k/2); p = exp(-K2/2).
    p = math.exp(-K2 * 0.5)
    return (K2, p, Z1, Z2)


# =============================================================================
# Anderson-Darling test for Normal
# =============================================================================

@njit(cache=True, fastmath=True)
def anderson_darling_normal(x: np.ndarray) -> tuple:
    """Anderson-Darling A^2 statistic for Normal, with the standard
    finite-sample correction A*^2 = A^2 * (1 + 0.75/n + 2.25/n^2)
    (Stephens 1986). Returns (A2_star, p_value_approx).

    Rejection table (Stephens 1974 / D'Agostino & Stephens 1986 case 3,
    mu and sigma both estimated):
        alpha 0.10:  A* > 0.631
        alpha 0.05:  A* > 0.752
        alpha 0.025: A* > 0.873
        alpha 0.01:  A* > 1.035

    p approximation (Stephens 1986, piecewise polynomial). Calibrated
    up to A* ~ 5; beyond that we saturate p=0 since rejection is
    certain at any sensible alpha.
    """
    n = x.size
    if n < 8:
        return (np.nan, np.nan)

    mu = 0.0
    for i in range(n):
        mu += x[i]
    mu /= n
    s2 = 0.0
    for i in range(n):
        d = x[i] - mu
        s2 += d * d
    s2 /= n - 1.0
    if s2 <= 0.0:
        return (np.nan, np.nan)
    sd = math.sqrt(s2)

    z = np.empty(n, dtype=np.float64)
    for i in range(n):
        z[i] = (x[i] - mu) / sd
    z.sort()

    # A^2 = -n - (1/n) sum_{i=1..n} (2i-1) * [ ln Phi(z_i) + ln(1 - Phi(z_{n-i+1})) ]
    # log(1 - Phi(z)) = log(Phi(-z)); reuse the stable log_phi helper.
    s = 0.0
    for i in range(n):
        log_lo = log_phi_cdf(z[i])
        log_hi_complement = log_phi_cdf(-z[n - 1 - i])
        s += (2.0 * (i + 1) - 1.0) * (log_lo + log_hi_complement)
    A2 = -float(n) - s / float(n)
    A2_star = A2 * (1.0 + 0.75 / n + 2.25 / (n * n))

    if A2_star <= 0.200:
        p = 1.0 - math.exp(-13.436 + 101.14 * A2_star - 223.73 * A2_star * A2_star)
    elif A2_star <= 0.340:
        p = 1.0 - math.exp(-8.318 + 42.796 * A2_star - 59.938 * A2_star * A2_star)
    elif A2_star <= 0.600:
        p = math.exp(0.9177 - 4.279 * A2_star - 1.38 * A2_star * A2_star)
    elif A2_star <= 5.0:
        p = math.exp(1.2937 - 5.709 * A2_star + 0.0186 * A2_star * A2_star)
    else:
        p = 0.0
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    return (A2_star, p)


# =============================================================================
# Combined verdict
# =============================================================================

def normality_verdict(
    sample: np.ndarray,
    *,
    alpha: float = 0.05,
    max_n_ad: int = 100_000,
    seed: int = 0,
) -> dict:
    """Run D'Agostino K^2 + Anderson-Darling on ``sample`` and classify.

    Anderson-Darling is O(n log n) and stable on n=100k; for larger
    samples we subsample (controlled by ``max_n_ad``). D'Agostino K^2
    is O(n) and always runs on the full array.

    Returns dict with the raw stats + p-values + a boolean
    ``reject_normal`` that fires when EITHER test rejects H0 at alpha.

    Why fail-fast (OR) instead of requiring both to fail: residual-
    distribution audits exist to surface non-Normality. K^2 catches
    moment-based deviations (skew + kurt anomalies); A-D catches
    tail-shape deviations (extreme percentile mis-fit). Requiring both
    to fail covers only the worst-case overlap and would let many
    real non-Normal distributions through.
    """
    r = np.ascontiguousarray(sample, dtype=np.float64)
    r = r[np.isfinite(r)]
    n_total = r.size
    if n_total < 20:
        return {
            "n": n_total,
            "n_total": n_total,
            "k2_stat": float("nan"),
            "k2_p": float("nan"),
            "k2_z_skew": float("nan"),
            "k2_z_kurt": float("nan"),
            "ad_stat": float("nan"),
            "ad_p": float("nan"),
            "reject_normal": False,
            "alpha": alpha,
            "verdict": "too-few-samples",
        }

    if float(np.var(r)) <= 0.0:
        # A constant sample is a degenerate distribution, not a Normal one. dagostino_k2/
        # anderson_darling_normal both short-circuit to a "can't reject" result here (k2_p=1.0,
        # ad_p=nan), which would otherwise report "consistent with Normal" for e.g. a broken
        # model that always predicts the same value -- exactly the kind of pipeline bug this
        # audit exists to surface, not mask.
        return {
            "n": n_total,
            "n_total": n_total,
            "k2_stat": 0.0,
            "k2_p": 1.0,
            "k2_z_skew": 0.0,
            "k2_z_kurt": 0.0,
            "ad_stat": float("nan"),
            "ad_p": float("nan"),
            "reject_normal": False,
            "alpha": alpha,
            "verdict": "degenerate (zero variance)",
        }

    k2_stat, k2_p, z_skew, z_kurt = dagostino_k2(r)

    if n_total > max_n_ad:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_n_ad, replace=False)
        ad_input = r[idx]
    else:
        ad_input = r
    ad_stat, ad_p = anderson_darling_normal(ad_input)

    reject_k2 = bool(np.isfinite(k2_p) and k2_p < alpha)
    reject_ad = bool(np.isfinite(ad_p) and ad_p < alpha)
    reject_normal = reject_k2 or reject_ad

    if reject_normal:
        why = []
        if reject_k2:
            why.append(f"D'Agostino K2 p={k2_p:.4g} (<{alpha})")
        if reject_ad:
            why.append(f"Anderson-Darling A*={ad_stat:.3f} p={ad_p:.4g} (<{alpha})")
        verdict = "non-Gaussian (" + "; ".join(why) + ")"
    else:
        verdict = f"consistent with Normal " f"(K2 p={k2_p:.3g}, AD A*={ad_stat:.3f} p={ad_p:.3g})"

    return {
        "n": int(ad_input.size),
        "n_total": int(n_total),
        "k2_stat": float(k2_stat),
        "k2_p": float(k2_p),
        "k2_z_skew": float(z_skew),
        "k2_z_kurt": float(z_kurt),
        "ad_stat": float(ad_stat),
        "ad_p": float(ad_p),
        "reject_normal": reject_normal,
        "alpha": alpha,
        "verdict": verdict,
    }


__all__ = [
    "phi_cdf",
    "log_phi_cdf",
    "dagostino_k2",
    "anderson_darling_normal",
    "normality_verdict",
]
