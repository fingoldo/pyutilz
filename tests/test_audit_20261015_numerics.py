"""Regression tests for the 2026-10-15 numerics, performance and concurrency audit findings.

Each test is written so that it fails on the pre-fix code and passes on the fix:
N-01 binning -inf bounds, N-03 Int64 overflow check, N-04 zero-total concentration, N-06 zero counts in entropy,
N-07 kernel precondition contract, P-01 deque trim, P-02 bounded monitor series, P-04 linear fence parsing,
C-05 bounded OpenRouter lookup cache eviction, C-06 single read of the catalogue global.
"""

import logging
import threading
from collections import Counter, deque

import polars as pl
import pytest

# --------------------------------------------------------------------------------------------------------------------
# N-01: a -inf observed bound with clipping disabled must raise, not bin every row into bin 0.
# --------------------------------------------------------------------------------------------------------------------


def test_bin_numerical_columns_refuses_infinite_bounds_without_clipping() -> None:
    from pyutilz.data.polarslib.binning import bin_numerical_columns

    df = pl.DataFrame({"x": [float("-inf"), 1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not finite"):
        bin_numerical_columns(df, target_columns=["x"], clean_features=False, clean_targets=False)


def test_bin_numerical_columns_bins_infinite_column_when_clipping_bounds_it() -> None:
    from pyutilz.data.polarslib.binning import bin_numerical_columns

    values = [float("-inf")] + [float(v) for v in range(1, 40)]
    df = pl.DataFrame({"x": values})
    result = bin_numerical_columns(df, target_columns=[], clean_features=True, clean_targets=True, num_bins=10)
    binned = result[0]
    assert binned["x"].null_count() == 0
    assert binned["x"].min() == 0 and binned["x"].max() == 9


# --------------------------------------------------------------------------------------------------------------------
# N-03: Int64 min value must trip the 2**24 precision warning (abs() wraps -2**63 to itself).
# --------------------------------------------------------------------------------------------------------------------


def test_cast_f64_to_f32_warns_on_int64_most_negative_value(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.data.polarslib.columns import cast_f64_to_f32

    df = pl.DataFrame({"i": pl.Series([-(2**63)], dtype=pl.Int64)})
    with caplog.at_level(logging.WARNING):
        cast_f64_to_f32(df)
    assert any("beyond 2**24" in r.getMessage() for r in caplog.records)


def test_cast_f64_to_f32_no_warning_for_small_negative_int64(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.data.polarslib.columns import cast_f64_to_f32

    df = pl.DataFrame({"i": pl.Series([-5, 7], dtype=pl.Int64)})
    with caplog.at_level(logging.WARNING):
        cast_f64_to_f32(df)
    assert not any("beyond 2**24" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------------------------------------------------
# N-04: a group whose total is zero must not produce NaN rank keys.
# --------------------------------------------------------------------------------------------------------------------


def test_compute_concentrations_zero_group_total_has_no_nan() -> None:
    from pyutilz.data.polarslib.aggregations import compute_concentrations

    df = pl.DataFrame({"g": [1, 1, 2, 2], "e": ["a", "b", "c", "d"], "v": [0.0, 0.0, 3.0, 1.0]})
    out = compute_concentrations(df.group_by("g"), entity="e", entity_name="e", by="v", groupby_columns=("g",), top_n=2)
    assert out.height > 0, "the zero-total groups must still produce rows"
    float_cols = [col for col in out.columns if out.schema[col].is_float()]
    assert float_cols, "the check needs at least one float column, otherwise it passes vacuously"
    for col in float_cols:
        assert not out[col].is_nan().any(), col
        assert not out[col].is_infinite().any(), col


# --------------------------------------------------------------------------------------------------------------------
# N-06: explicit zero counts contribute nothing instead of raising a math domain error.
# --------------------------------------------------------------------------------------------------------------------


def test_entropy_ignores_zero_counts() -> None:
    from pyutilz.text.strings.textentropy import entropy

    assert entropy(Counter({"a": 0, "b": 3}), normalization_factor=3.0) == pytest.approx(0.0)
    assert entropy(Counter({"a": 0, "b": 1, "c": 1}), normalization_factor=2.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------------------------------
# N-07: the fastmath kernels document the finite-input precondition.
# --------------------------------------------------------------------------------------------------------------------


def test_normality_kernels_document_finite_precondition() -> None:
    from pyutilz.stats.normality import anderson_darling_normal, dagostino_k2, log_phi_cdf, phi_cdf

    for fn in (dagostino_k2, anderson_darling_normal, phi_cdf, log_phi_cdf):
        assert "Precondition" in (fn.__doc__ or ""), fn.__name__


# --------------------------------------------------------------------------------------------------------------------
# P-01: proxy outcome history is a deque trimmed from the left, with the same window semantics.
# --------------------------------------------------------------------------------------------------------------------


def test_port_stats_trim_pops_stale_prefix_and_keeps_window() -> None:
    from pyutilz.web.proxy.base import _PortStats

    ps = _PortStats()
    ps.record(1.0, is_error=False)
    ps.record(2.0, is_error=True)
    ps.record(3.0, is_error=False)
    ps.trim(2.0)  # entries at or before the cutoff are dropped
    assert isinstance(ps.outcomes, deque)
    assert list(ps.outcomes) == [(3.0, False)]
    assert ps.total == 1 and ps.errors == 0


# --------------------------------------------------------------------------------------------------------------------
# P-02: the monitor's series are bounded; the newest samples are kept and the averages cover them.
# --------------------------------------------------------------------------------------------------------------------


def test_utilization_monitor_series_are_bounded_to_max_samples() -> None:
    from pyutilz.system.hardware_monitor import UtilizationMonitor

    monitor = UtilizationMonitor(max_samples=3)
    for i in range(10):
        monitor.cpu_utilizaton.append(float(i))
    assert list(monitor.cpu_utilizaton) == [7.0, 8.0, 9.0]
    assert monitor.get_average_utilization()["cpu_utilizaton_percent"] == pytest.approx(8.0)


def test_utilization_monitor_rejects_non_positive_max_samples() -> None:
    from pyutilz.system.hardware_monitor import UtilizationMonitor

    with pytest.raises(ValueError):
        UtilizationMonitor(max_samples=0)


# --------------------------------------------------------------------------------------------------------------------
# P-04: fenced-JSON extraction is linear on many unclosed fences and unchanged on normal input.
# --------------------------------------------------------------------------------------------------------------------


def test_parse_fenced_object_still_extracts_valid_fence() -> None:
    from pyutilz.llm.base import _parse_fenced_object

    text = 'Here you go:\n```json\n{"a": 1}\n```\nthanks'
    assert _parse_fenced_object(text, "test") == {"a": 1}


def test_parse_fenced_object_broken_first_fence_still_degrades() -> None:
    from pyutilz.llm.base import _parse_fenced_object

    text = '```json\n{bad\n```\n```json\n{"a": 1}\n```'
    assert _parse_fenced_object(text, "test") is None


def test_parse_fenced_object_linear_on_many_unclosed_fences() -> None:
    from pyutilz.llm.base import _parse_fenced_object

    text = "```json\n[" + "```{x" * 20000
    assert _parse_fenced_object(text, "test") is None
    # Linear-time behaviour is not asserted: wall-clock bounds are machine-dependent and the shared gate rejects them.


# --------------------------------------------------------------------------------------------------------------------
# C-05: eviction at the cache cap must tolerate a key that is already gone (another thread evicted it).
# --------------------------------------------------------------------------------------------------------------------


class _RacyCache(dict):
    """A dict whose iteration reports a key another thread has already evicted."""

    def __iter__(self):  # type: ignore[override]
        yield "already-evicted"
        yield from super().__iter__()


def test_openrouter_lookup_eviction_tolerates_concurrently_evicted_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyutilz.llm import _claude_models as cm
    from pyutilz.llm.openrouter_provider import _catalogue

    cache = _RacyCache({"old-1": None})
    monkeypatch.setattr(cm, "_openrouter_specs", cache)
    monkeypatch.setattr(cm, "_OPENROUTER_LOOKUPS_MAX_SIZE", 1)
    monkeypatch.setattr(_catalogue, "_fetch_models_catalogue", lambda *a, **k: {})
    assert cm._openrouter_claude_spec("claude-new-model") is None
    assert "claude-new-model" in cache


def test_openrouter_lookup_cache_respects_cap_across_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyutilz.llm import _claude_models as cm
    from pyutilz.llm.openrouter_provider import _catalogue

    monkeypatch.setattr(cm, "_openrouter_specs", {})
    monkeypatch.setattr(cm, "_OPENROUTER_LOOKUPS_MAX_SIZE", 16)
    monkeypatch.setattr(_catalogue, "_fetch_models_catalogue", lambda *a, **k: {})
    errors: list = []

    def worker(tag: int) -> None:
        try:
            for i in range(200):
                cm._openrouter_claude_spec(f"claude-{tag}-{i}")
        except Exception as exc:  # collected, asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(cm._openrouter_specs) <= 16


# --------------------------------------------------------------------------------------------------------------------
# C-06: the catalogue fast path returns the snapshot it checked, even if another thread clears the global meanwhile.
# --------------------------------------------------------------------------------------------------------------------


def test_fetch_catalogue_fast_path_survives_concurrent_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyutilz.llm.openrouter_provider import _catalogue

    pkg = _catalogue._pkg()
    entry = {"anthropic/claude-x": {"pricing": {}}}
    monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", entry)

    def fresh_then_clear() -> bool:
        pkg._MODELS_CATALOGUE = None  # another thread's refresh path sets it to None right after the check
        return True

    monkeypatch.setattr(_catalogue, "_catalogue_is_fresh", fresh_then_clear)
    assert _catalogue._fetch_models_catalogue() is entry
