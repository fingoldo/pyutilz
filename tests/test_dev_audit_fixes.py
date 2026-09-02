"""Regression tests for the 2026-09-02 core/dev/system domain audit -- dev/ findings."""

from __future__ import annotations

import logging

import pytest

from pyutilz.dev.benchmarking import benchmark_algos_by_runtime, sweep_backend_crossover
from pyutilz.dev.logginglib import _log_filename, _message, log_loaded_rows, logged

# -------------------- F03: log_loaded_rows default --------------------


def test_log_loaded_rows_works_with_its_documented_default():
    """results_log=None is the documented default and must not raise KeyError."""
    # lang="ru" avoids the optional `inflect` dependency the English template pulls in.
    log_loaded_rows([1, 2, 3], source="t", lang="ru")


def test_log_loaded_rows_records_into_provided_log():
    results_log = {"results": {}}
    log_loaded_rows([1, 2, 3], source="t", results_log=results_log, lang="ru")
    assert results_log["results"]["loaded"]["db_table"]["t"] == {"rows": 3}


def test_log_loaded_rows_seeds_results_key_when_absent():
    results_log: dict = {}
    log_loaded_rows([1], source="t", results_log=results_log, lang="ru")
    assert results_log["results"]["loaded"]["db_table"]["t"] == {"rows": 1}


# -------------------- F21: _message on a whitespace-only name --------------------


def test_message_tolerates_whitespace_only_activity_name():
    _message("   ")  # must not raise IndexError


def test_message_still_logs_a_real_name(caplog):
    with caplog.at_level("INFO"):
        _message("loading data")
    assert any("loading data..." in r.getMessage() for r in caplog.records)


# -------------------- F33: init_logging filename derivation --------------------


@pytest.mark.parametrize(
    "caller, expected",
    [
        ("run.pytest_smoke.py", "run.pytest_smoke.log"),
        ("test.python_shim.py", "test.python_shim.log"),
        ("plain.py", "plain.log"),
    ],
)
def test_log_filename_replaces_only_the_trailing_extension(caller, expected):
    assert _log_filename(caller) == expected


# -------------------- F10: @logged must log a failing call --------------------


def test_logged_finalizes_and_records_the_error_on_a_raising_call():
    captured: dict = {}

    @logged(include_node_ip=False)
    def boom(results_log=None):
        captured["log"] = results_log
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        boom()

    log = captured["log"]
    assert "ValueError: kaboom" == log["results"]["error"]
    assert log["results"]["timing"].get("finished_at") is not None


# -------------------- F25: benchmark log matches the returned estimator ------


def test_benchmark_log_reports_the_min_over_reps(caplog):
    import time

    calls = {"n": 0}

    def slowest_last():
        calls["n"] += 1
        if calls["n"] == 3:  # make the LAST repetition clearly the slowest
            time.sleep(0.05)
        return 1

    with caplog.at_level(logging.INFO):
        _, durations = benchmark_algos_by_runtime([slowest_last], n_reps=3, verbose=2)
    lines = [r.getMessage() for r in caplog.records if "Benchmark timings" in r.getMessage()]
    assert lines
    assert f"{durations[0]:.3f}" in lines[0], (lines[0], durations)


# -------------------- F04: reference failure disables the equivalence gate ----


def test_sweep_skips_a_size_whose_reference_variant_raises(caplog):
    def bad_ref(n):
        raise MemoryError("no room")

    def divergent(n):
        return 12345.0

    with caplog.at_level(logging.WARNING):
        regions = sweep_backend_crossover(
            {"ref": bad_ref, "other": divergent},
            sizes=[8],
            make_inputs=lambda n: (n,),
            primary_axis="n",
            reference="ref",
            repeats=1,
        )
    assert regions == [], regions
    assert any("reference variant" in r.getMessage() for r in caplog.records)
