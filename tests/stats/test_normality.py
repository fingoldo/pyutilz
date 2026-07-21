"""Tests for ``pyutilz.stats.normality``.

Calibration:
* On Normal samples, K2 and AD p-values must be > 0.05 most of the time
  (alpha=0.05 means at most ~5% false rejections; allow 15-20% slack).
* On Laplace / Student-t(5) / mixture-contaminated samples, p < 0.05.

scipy parity: stats agree with scipy implementations to within
floating-point tolerance on identical input.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_dagostino_matches_scipy_on_normal_sample() -> None:
    pytest.importorskip("scipy")
    from scipy import stats as sps
    from pyutilz.stats.normality import dagostino_k2

    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    K2, p, _, _ = dagostino_k2(x)
    sp_stat, sp_p = sps.normaltest(x)
    assert K2 == pytest.approx(sp_stat, rel=1e-6, abs=1e-6)
    assert p == pytest.approx(sp_p, rel=1e-6, abs=1e-6)


def test_dagostino_rejects_laplace() -> None:
    from pyutilz.stats.normality import dagostino_k2

    rng = np.random.default_rng(1)
    x = rng.laplace(0.0, 1.0, size=5000)
    _, p, _, _ = dagostino_k2(x)
    assert p < 0.01, f"K2 p={p} should reject Normal for Laplace"


def test_dagostino_rejects_student_t_df5() -> None:
    from pyutilz.stats.normality import dagostino_k2

    rng = np.random.default_rng(2)
    x = rng.standard_t(5, size=5000)
    _, p, _, _ = dagostino_k2(x)
    assert p < 0.01


def test_dagostino_accepts_normal() -> None:
    from pyutilz.stats.normality import dagostino_k2

    rng = np.random.default_rng(7)
    n_runs = 50
    rejected = 0
    for _ in range(n_runs):
        x = rng.standard_normal(1000)
        _, p, _, _ = dagostino_k2(x)
        if p < 0.05:
            rejected += 1
    assert rejected / n_runs < 0.15, f"false-positive rate = {rejected/n_runs}"


def test_anderson_darling_rejects_mixture_contamination() -> None:
    from pyutilz.stats.normality import anderson_darling_normal

    rng = np.random.default_rng(3)
    x = rng.standard_normal(5000)
    x[:50] = rng.normal(0.0, 10.0, 50)
    _, p = anderson_darling_normal(x)
    assert p < 0.05, f"AD p={p} should reject Normal for contaminated sample"


def test_anderson_darling_accepts_normal() -> None:
    from pyutilz.stats.normality import anderson_darling_normal

    rng = np.random.default_rng(4)
    n_runs = 30
    rejected = 0
    for _ in range(n_runs):
        x = rng.standard_normal(500)
        _, p = anderson_darling_normal(x)
        if p < 0.05:
            rejected += 1
    assert rejected / n_runs < 0.20


def test_normality_verdict_marks_bimodal_hump() -> None:
    """Mild skew + bimodal hump: legacy moment cutoffs miss, formal
    tests must catch. The user's bug case was skew=-0.17 / kurt=-0.12
    with visible asymmetric tails; this synthetic version reproduces
    the same character (moments inside heuristic band, distribution
    clearly non-Normal)."""
    from pyutilz.stats.normality import normality_verdict

    rng = np.random.default_rng(5)
    n = 4000
    n1 = int(0.7 * n)
    x = np.concatenate([
        rng.normal(-200.0, 600.0, n1),
        rng.normal(800.0, 400.0, n - n1),
    ])
    out = normality_verdict(x)
    assert out["reject_normal"] is True, f"verdict={out['verdict']}"


def test_normality_verdict_no_reject_on_clean_normal() -> None:
    from pyutilz.stats.normality import normality_verdict

    rng = np.random.default_rng(6)
    x = rng.standard_normal(3000) * 50.0
    out = normality_verdict(x)
    assert out["reject_normal"] is False, f"verdict={out['verdict']}"


def test_normality_verdict_too_few_samples() -> None:
    from pyutilz.stats.normality import normality_verdict

    out = normality_verdict(np.array([1.0, 2.0, 3.0]))
    assert out["verdict"] == "too-few-samples"
    assert out["reject_normal"] is False


def test_normality_verdict_constant_sample_is_degenerate_not_normal() -> None:
    """Test-gap coverage (2026-07-21 audit round 2, MEDIUM): a zero-variance (constant) sample
    must be flagged "degenerate", not misreported as "consistent with Normal" -- the whole point
    of this early-return per its own inline comment (catches e.g. a broken model that always
    predicts the same value)."""
    from pyutilz.stats.normality import normality_verdict

    out = normality_verdict(np.full(50, 3.0))
    assert out["verdict"] == "degenerate (zero variance)"
    assert out["reject_normal"] is False


def test_normality_verdict_subsamples_without_replacement_when_over_max_n_ad(monkeypatch) -> None:
    """Test-gap coverage (2026-07-21 audit round 2, MEDIUM): the max_n_ad subsampling branch was
    never exercised. Asserts BOTH the reported sizes AND that the underlying rng.choice() call
    actually used replace=False -- asserting only `out["n"] == max_n_ad` would not catch a
    mutation flipping replace=False to True, since the sample size stays the same either way."""
    from pyutilz.stats import normality as normality_module

    # Built BEFORE patching np.random.default_rng below -- that patch is process-global (np.random
    # is the same module object everywhere), so generating the input sample after patching would
    # route through the spy too.
    x = np.random.default_rng(1).standard_normal(500)

    captured_kwargs: dict = {}
    real_rng = np.random.default_rng(0)

    class _SpyRng:
        def choice(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return real_rng.choice(*args, **kwargs)

    monkeypatch.setattr(normality_module.np.random, "default_rng", lambda seed: _SpyRng())

    out = normality_module.normality_verdict(x, max_n_ad=50)

    assert out["n"] == 50
    assert out["n_total"] == 500
    assert captured_kwargs.get("replace") is False
    assert captured_kwargs.get("size") == 50
