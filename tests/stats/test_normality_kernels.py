"""Kernel-level coverage for pyutilz.stats.normality.

Companion to ``tests/stats/test_normality.py`` (the statistical CALIBRATION suite: false-reject
rates on Normal samples, rejection of Laplace / Student-t(5) / contaminated mixtures). This file
covers the numerical kernels instead -- ``phi_cdf`` / ``log_phi_cdf`` tail accuracy, the exact
scipy.stats parity of ``dagostino_k2``, and ``normality_verdict``'s branch structure
(short-circuits, degenerate input, non-finite dropping, deterministic subsampling).

Lives in ``tests/stats/`` per TESTING.md's "Test layout" rule (tests for ``stats/*`` sources go in
this folder). It previously sat at flat-root as a second ``test_normality.py`` -- the only
duplicate test-file basename in the repo -- under a docstring claiming the module was "previously
untested -- no test_normality.py existed at all", which was never true: the calibration suite next
to this file predates it. That claim is what caused the duplicated effort; it is deleted here."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sp_stats

from pyutilz.stats.normality import (
    anderson_darling_normal,
    dagostino_k2,
    log_phi_cdf,
    normality_verdict,
    phi_cdf,
)


class TestPhiCdf:
    def test_matches_scipy_norm_cdf(self):
        for z in (-3.0, -1.0, 0.0, 0.5, 2.5):
            assert phi_cdf(z) == pytest.approx(sp_stats.norm.cdf(z), abs=1e-12)

    def test_zero_is_one_half(self):
        assert phi_cdf(0.0) == pytest.approx(0.5)


class TestLogPhiCdf:
    def test_matches_log_of_phi_cdf_near_zero(self):
        for z in (-2.0, -1.0, 0.0, 1.0, 2.0):
            assert log_phi_cdf(z) == pytest.approx(math.log(phi_cdf(z)), abs=1e-9)

    def test_deep_left_tail_matches_scipy_logcdf(self):
        """log_phi_cdf uses a truncated leading-term asymptote for z < -5 (O(1/z^2) error per
        the docstring), so it converges toward scipy's exact logcdf but isn't bit-identical --
        a loose relative tolerance confirms it's in the right ballpark, not exact."""
        for z in (-6.0, -10.0, -20.0):
            assert log_phi_cdf(z) == pytest.approx(sp_stats.norm.logcdf(z), rel=2e-2)

    def test_deep_left_tail_stays_finite(self):
        """Naive log(0.5 * erfc(...)) underflows to -inf around z < -37; the Mill's-ratio
        asymptote used below z < -5 must keep this finite well past that point."""
        assert math.isfinite(log_phi_cdf(-50.0))


class TestDagostinoK2:
    def test_matches_scipy_normaltest_on_normal_sample(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        k2, p, z_skew, z_kurt = dagostino_k2(x)
        ref_k2, ref_p = sp_stats.normaltest(x)
        assert k2 == pytest.approx(ref_k2, rel=1e-6)
        assert p == pytest.approx(ref_p, rel=1e-4, abs=1e-6)

    def test_matches_scipy_normaltest_on_skewed_sample(self):
        rng = np.random.default_rng(1)
        x = rng.exponential(size=3000)
        k2, p, _, _ = dagostino_k2(x)
        ref_k2, ref_p = sp_stats.normaltest(x)
        assert k2 == pytest.approx(ref_k2, rel=1e-6)
        assert p == pytest.approx(ref_p, rel=1e-4, abs=1e-6)

    def test_below_minimum_n_returns_nan(self):
        x = np.arange(19, dtype=np.float64)
        result = dagostino_k2(x)
        assert len(result) == 4, "dagostino_k2 always returns (k2, p, skew, kurt)"
        assert all(math.isnan(v) for v in result)

    def test_constant_sample_returns_zero_stat_and_p_one(self):
        x = np.full(50, 3.0)
        k2, p, z_skew, z_kurt = dagostino_k2(x)
        assert k2 == 0.0
        assert p == 1.0
        assert z_skew == 0.0
        assert z_kurt == 0.0


class TestAndersonDarlingNormal:
    def test_normal_sample_gives_high_p(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=2000)
        a2, p = anderson_darling_normal(x)
        assert a2 < 0.752  # alpha=0.05 rejection threshold from the module docstring
        assert p > 0.05

    def test_heavily_skewed_sample_gives_low_p(self):
        rng = np.random.default_rng(3)
        x = rng.exponential(size=2000)
        a2, p = anderson_darling_normal(x)
        assert a2 > 1.035  # alpha=0.01 rejection threshold
        assert p < 0.01

    def test_below_minimum_n_returns_nan(self):
        x = np.arange(7, dtype=np.float64)
        a2, p = anderson_darling_normal(x)
        assert math.isnan(a2)
        assert math.isnan(p)

    def test_zero_variance_returns_nan(self):
        x = np.full(20, 5.0)
        a2, p = anderson_darling_normal(x)
        assert math.isnan(a2)
        assert math.isnan(p)

    def test_p_value_bounded_in_unit_interval_across_scales(self):
        """A*_star piecewise polynomial approximation is only calibrated up to ~5 -- values must
        still be clamped into [0, 1] for a badly non-Normal sample that pushes A* past that."""
        rng = np.random.default_rng(4)
        x = np.concatenate([rng.normal(size=500), rng.normal(loc=50, scale=0.01, size=500)])
        a2, p = anderson_darling_normal(x)
        assert 0.0 <= p <= 1.0
        assert a2 > 5.0


class TestNormalityVerdict:
    def test_normal_sample_not_rejected(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=5000)
        result = normality_verdict(x)
        assert result["reject_normal"] is False
        assert "consistent with Normal" in result["verdict"]
        assert result["n_total"] == 5000

    def test_skewed_sample_rejected(self):
        rng = np.random.default_rng(6)
        x = rng.exponential(size=5000)
        result = normality_verdict(x)
        assert result["reject_normal"] is True
        assert "non-Gaussian" in result["verdict"]

    def test_too_few_samples_short_circuits(self):
        # Anderson-Darling is valid from n>=8, so only below that is the sample genuinely untestable.
        result = normality_verdict(np.arange(7, dtype=np.float64))
        assert result["verdict"] == "too-few-samples"
        assert result["reject_normal"] is False
        assert math.isnan(result["k2_stat"])
        assert math.isnan(result["ad_stat"])

    def test_between_8_and_20_runs_anderson_darling_only(self):
        # D'Agostino K2 needs n>=20 and stays NaN here, but reporting "too-few-samples" for the whole
        # 8..19 range made a strongly non-Normal small group indistinguishable from an untested one.
        result = normality_verdict(np.arange(10, dtype=np.float64))
        assert result["verdict"].startswith("AD-only (n<20)")
        assert math.isnan(result["k2_stat"])
        assert math.isfinite(result["ad_stat"])
        assert math.isfinite(result["ad_p"])

    def test_degenerate_zero_variance_short_circuits(self):
        result = normality_verdict(np.full(100, 7.0))
        assert result["verdict"] == "degenerate (zero variance)"
        assert result["reject_normal"] is False
        assert result["k2_p"] == 1.0

    def test_non_finite_values_dropped_before_testing(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=5000)
        x_with_nans = np.concatenate([x, [np.nan, np.inf, -np.inf]])
        result = normality_verdict(x_with_nans)
        assert result["n_total"] == 5000

    def test_large_sample_subsampled_for_anderson_darling(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=5000)
        result = normality_verdict(x, max_n_ad=1000, seed=42)
        assert result["n"] == 1000
        assert result["n_total"] == 5000

    def test_subsampling_is_deterministic_given_seed(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=5000)
        r1 = normality_verdict(x, max_n_ad=500, seed=123)
        r2 = normality_verdict(x, max_n_ad=500, seed=123)
        assert r1["ad_stat"] == r2["ad_stat"]
        assert r1["ad_p"] == r2["ad_p"]

    def test_rejection_reason_names_both_tests_when_both_fail(self):
        rng = np.random.default_rng(10)
        x = rng.exponential(size=5000)
        result = normality_verdict(x, alpha=0.5)
        assert "D'Agostino K2" in result["verdict"]
        assert "Anderson-Darling" in result["verdict"]
