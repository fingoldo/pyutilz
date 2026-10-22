"""Regression tests for the 2026-10-15 silent-failure, security and numerics audit findings (S-01..S-14, P-03, N-02).

Each test pins the behaviour a fix changed, so it fails on the pre-fix code.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from py_ci_shared.code_audit import register_scanner
from py_ci_shared.code_audit.broad_except import scan_broad_except_swallows
from py_ci_shared.code_audit.registry import _SCANNERS, run_all, run_all_checked


def _write(root: Path, name: str, text: str) -> None:
    (root / name).write_text(text, encoding="utf-8")


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ---- S-01: a crashed scanner must be visible to the caller of run_all_checked ----------------------


def _boom_scanner(root: Path, exclude_dirs: Any = None) -> list:
    raise RuntimeError("pathological file")


def test_s01_crashed_scanner_is_reported_by_run_all_checked(tmp_path: Path) -> None:
    _write(tmp_path, "bad.py", "def f(items=[]):\n    items.append(1)\n")
    register_scanner("boom_probe_s01", _boom_scanner, allow_override=True)
    try:
        findings, failed = run_all_checked(tmp_path, checks=["boom_probe_s01", "mutable_default"], parallel=False)
    finally:
        _SCANNERS.pop("boom_probe_s01", None)
    assert failed == {"boom_probe_s01": "RuntimeError: pathological file"}
    assert [f.check for f in findings] == ["mutable_default"]


def test_s01_clean_run_reports_no_failed_scanner(tmp_path: Path) -> None:
    _write(tmp_path, "ok.py", "x = 1\n")
    findings, failed = run_all_checked(tmp_path, checks=["mutable_default", "bare_except"], parallel=False)
    assert findings == []
    assert failed == {}


def test_s01_run_all_keeps_its_list_return_for_existing_callers(tmp_path: Path) -> None:
    _write(tmp_path, "ok.py", "x = 1\n")
    register_scanner("boom_probe_s01b", _boom_scanner, allow_override=True)
    try:
        assert run_all(tmp_path, checks=["boom_probe_s01b"], parallel=False) == []
    finally:
        _SCANNERS.pop("boom_probe_s01b", None)


# ---- S-14: a debug-only handler that hands a default back is a silent swallow ---------------------


def test_s14_debug_only_handler_returning_default_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "bad.py",
        "import logging\nlogger = logging.getLogger(__name__)\n\n"
        "def f():\n    try:\n        probe()\n    except Exception as e:\n        logger.debug('probe failed: %s', e)\n        return False\n",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert len(findings) == 1
    assert findings[0].line == 7


def test_s14_debug_only_handler_assigning_default_is_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "bad.py",
        "import logging\nlogger = logging.getLogger(__name__)\n\n"
        "def f(out):\n    try:\n        probe()\n    except Exception as e:\n        logger.debug('probe failed: %s', e)\n        out['k'] = None\n",
    )
    assert len(scan_broad_except_swallows(tmp_path)) == 1


def test_s14_debug_only_handler_returning_structured_error_is_not_flagged(tmp_path: Path) -> None:
    """A structured error return is an escalation, so the debug-only exemption still applies."""
    _write(
        tmp_path,
        "ok.py",
        "import logging\nlogger = logging.getLogger(__name__)\n\n"
        "def f():\n    try:\n        probe()\n    except Exception as e:\n        logger.debug('probe failed: %s', e)\n        return {'ok': False, 'error': str(e)}\n",
    )
    assert scan_broad_except_swallows(tmp_path) == []


# ---- S-02: ensure_installed keeps its logged-not-raised contract, at WARNING -----------------------


def test_s02_failed_install_is_warned_and_not_raised(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.core.pythonlib import packages

    class _FailingSubprocess:
        @staticmethod
        def check_call(cmd: list) -> None:
            raise OSError("pip exploded")

    monkeypatch.setattr(packages, "subprocess", _FailingSubprocess)
    caplog.set_level(logging.DEBUG)
    packages.ensure_installed("pyutilz_no_such_package_s02")  # must not raise
    warned = _warnings(caplog)
    assert any("pyutilz_no_such_package_s02" in m and "still not installed" in m for m in warned), warned


# ---- S-03 / S-08 / P-03: io_ops --------------------------------------------------------------------


def _pickle_frames(folder: Path, n: int = 2) -> None:
    for i in range(n):
        pd.DataFrame({"a": [i, i + 1]}).to_pickle(folder / f"part_{i}.pckl")


def test_s03_concat_failure_is_raised_not_reported_as_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from pyutilz.data.pandaslib import io_ops

    _pickle_frames(tmp_path)

    def _disk_full(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(io_ops._facade, "concat_and_flush_df_list", _disk_full)
    with pytest.raises(OSError, match="disk full"):
        io_ops.read_stats_from_multiple_files(joint_file_name=str(tmp_path / "joint"), folder=str(tmp_path), template="*.pckl")


def test_s03_no_input_still_returns_none(tmp_path: Path) -> None:
    from pyutilz.data.pandaslib import io_ops

    assert io_ops.read_stats_from_multiple_files(joint_file_name=str(tmp_path / "joint"), folder=str(tmp_path), template="*.pckl") is None


def test_s08_undeletable_source_is_warned_with_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.data.pandaslib import io_ops

    _pickle_frames(tmp_path, n=3)
    real_remove = os.remove

    def _refuse_one(path: str) -> None:
        if path.endswith("part_1.pckl"):
            raise PermissionError("in use")
        real_remove(path)

    monkeypatch.setattr(io_ops.os, "remove", _refuse_one)
    caplog.set_level(logging.DEBUG)
    res = io_ops.read_stats_from_multiple_files(joint_file_name=str(tmp_path / "joint"), folder=str(tmp_path), template="*.pckl", delete_after=True)
    assert res is not None and len(res) == 6
    assert any("1 source file(s) could not be removed" in m for m in _warnings(caplog)), _warnings(caplog)


def test_p03_default_path_skips_deep_memory_accounting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from pyutilz.data.pandaslib import io_ops

    _pickle_frames(tmp_path)
    deep_flags: list[bool] = []

    def _record(df: pd.DataFrame, deep: bool = False) -> int:
        deep_flags.append(deep)
        return 0

    monkeypatch.setattr(io_ops, "get_df_memory_consumption", _record)
    io_ops.read_stats_from_multiple_files(joint_file_name=str(tmp_path / "joint"), folder=str(tmp_path), template="*.pckl", optimize=False)
    assert deep_flags and not any(deep_flags)


# ---- S-04: check_cpu_flag reports a real probe failure at WARNING ------------------------------------


def test_s04_cpuinfo_probe_failure_is_warned(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.core.pythonlib import hardware

    fake = types.ModuleType("cpuinfo")

    def _broken() -> dict:
        raise RuntimeError("cpuid trap")

    fake.get_cpu_info = _broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cpuinfo", fake)
    caplog.set_level(logging.DEBUG)
    assert hardware.check_cpu_flag("avx2") is False
    assert any("avx2" in m for m in _warnings(caplog)), _warnings(caplog)


def test_s04_missing_cpuinfo_stays_debug(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.core.pythonlib import hardware

    monkeypatch.setitem(sys.modules, "cpuinfo", None)  # import raises ImportError
    caplog.set_level(logging.DEBUG)
    assert hardware.check_cpu_flag("avx2") is False
    assert not _warnings(caplog)


# ---- S-05: a candidate that raises in the mean ranking is warned about ------------------------------


def test_s05_mean_ranking_warns_the_crashing_candidate(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.dev import benchmarking

    def _boom() -> None:
        raise ValueError("kernel crashed")

    caplog.set_level(logging.DEBUG)
    out = benchmarking._rank_mean({"bad": _boom, "ok": lambda: None}, ["bad", "ok"], 1, False)
    assert out["bad"] == float("inf")
    assert any("bad" in m and "kernel crashed" in m for m in _warnings(caplog)), _warnings(caplog)


# ---- S-06: a failed category conversion is warned about even with verbose=False ---------------------


def test_s06_category_failure_warned_without_verbose(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.data.pandaslib.dtypes import _promote_column_to_category

    df = pd.DataFrame({"c": [[1], [2]]})  # lists are unhashable, so nunique raises
    new_dtypes: dict = {}
    caplog.set_level(logging.DEBUG)
    _promote_column_to_category(df, "c", "object", new_dtypes, 5, False, verbose=False)
    assert "c" not in new_dtypes
    assert any("leaving its dtype unchanged" in m for m in _warnings(caplog)), _warnings(caplog)


# ---- S-07 / X-05: load_file does not turn an OS error into FileNotFoundError ------------------------


def test_s07_existence_check_error_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from pyutilz.core.pythonlib import filesystem

    def _denied(path: str) -> bool:
        raise PermissionError("denied")

    monkeypatch.setattr(filesystem, "exists", _denied)
    with pytest.raises(PermissionError):
        filesystem.load_file(str(tmp_path / "model.joblib"))


def test_s07_missing_file_still_raises_file_not_found(tmp_path: Path) -> None:
    from pyutilz.core.pythonlib import filesystem

    with pytest.raises(FileNotFoundError):
        filesystem.load_file(str(tmp_path / "absent.joblib"))


# ---- S-09: a failed stats fetch is not the same record as "stats not requested" ---------------------


class _StatsBrokenProvider:
    model_name = "fake/model"
    last_generation_id = "gen-1"
    last_upstream_provider = "Upstream"
    _last_usage: dict = {}
    _last_finish_reason = "stop"
    last_actual_cost_usd = None

    async def generate(self, prompt: str, system: Any = None, temperature: float = 0.7, max_tokens: int = 0) -> str:
        return "answer"

    async def fetch_generation_stats(self, generation_id: str) -> dict:
        raise RuntimeError("upstream timeout")


def test_s09_failed_stats_fetch_is_marked(tmp_path: Path) -> None:
    from pyutilz.dev.attempt_archive import DirectoryContentStore, JsonlAttemptSink, archive_provider

    store = DirectoryContentStore(tmp_path / "t")
    sink = JsonlAttemptSink(tmp_path / "a.jsonl")
    wrapped = archive_provider(_StatsBrokenProvider(), store, sink, fetch_stats=True)
    asyncio.run(wrapped.generate("p"))
    attempt = sink.read()[0]
    assert attempt["stats_fetch_failed"] is True
    assert attempt["latency_ms"] is None


def test_s09_stats_not_requested_leave_marker_unset(tmp_path: Path) -> None:
    from pyutilz.dev.attempt_archive import DirectoryContentStore, JsonlAttemptSink, archive_provider

    store = DirectoryContentStore(tmp_path / "t")
    sink = JsonlAttemptSink(tmp_path / "a.jsonl")
    wrapped = archive_provider(_StatsBrokenProvider(), store, sink, fetch_stats=False)
    asyncio.run(wrapped.generate("p"))
    assert sink.read()[0]["stats_fetch_failed"] is None


# ---- S-11: an unexpected liveness-probe failure is warned, and still never steals -------------------


def test_s11_unexpected_probe_failure_warned_and_assumed_alive(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.performance.kernel_tuning.cache import cache_base

    def _broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("probe broke")

    if os.name == "nt":
        monkeypatch.setattr(cache_base, "_kernel32", _broken)
    else:
        monkeypatch.setattr(cache_base.os, "kill", _broken)
    caplog.set_level(logging.DEBUG)
    assert cache_base._pid_alive(os.getpid() if os.name != "nt" else 4242) is True
    assert any("probe broke" in m for m in _warnings(caplog)), _warnings(caplog)


# ---- S-12: a column skipped by the constant screen is reported ---------------------------------------


def test_s12_unhashable_column_skip_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.data.pandaslib import frames

    df = pd.DataFrame({"const": [1, 1], "lst": [[1], [2]]})
    caplog.set_level(logging.DEBUG)
    assert frames.get_suspiciously_constant_columns(df) == ["const"]
    assert any("could not check 1 column(s)" in m and "lst" in m for m in _warnings(caplog)), _warnings(caplog)


# ---- S-13: a persistent download error ends in an exception -------------------------------------------


def test_s13_persistent_download_error_gives_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("boto3")
    from unittest.mock import MagicMock

    from pyutilz.cloud import cloud as cloud_mod

    monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", "mybucket")
    sleep_calls: list = []
    monkeypatch.setattr(cloud_mod, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(cloud_mod, "s3_file_exists", lambda key, bucket: True)
    fake_s3 = MagicMock()
    fake_s3.meta.client.download_file.side_effect = OSError("access denied")
    monkeypatch.setattr(cloud_mod, "s3", fake_s3)
    monkeypatch.setattr(cloud_mod, "exists", lambda path: False)

    with pytest.raises(RuntimeError, match="Giving up"):
        cloud_mod.get_from_s3_or_cache(str(tmp_path / "model.bin"), "model.bin", str(tmp_path))
    assert fake_s3.meta.client.download_file.call_count == cloud_mod._MAX_DOWNLOAD_ATTEMPTS
    assert len(sleep_calls) == cloud_mod._MAX_DOWNLOAD_ATTEMPTS - 1


# ---- N-02: every int -> float32 downcast warns past 2**24 --------------------------------------------


def test_n02_pandas_int64_beyond_mantissa_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.data.pandaslib.dtypes import ensure_dataframe_float32_convertability

    df = pd.DataFrame({"id": pd.Series([16777217], dtype="int64")})
    caplog.set_level(logging.DEBUG)
    out = ensure_dataframe_float32_convertability(df)
    assert out["id"].iloc[0] == 16777216.0  # the lossy contract itself is unchanged
    assert any("'id'" in m or "['id']" in m for m in _warnings(caplog)), _warnings(caplog)


def test_n02_small_ints_not_warned(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.data.pandaslib.dtypes import ensure_dataframe_float32_convertability

    df = pd.DataFrame({"id": pd.Series([1, 2, 16777216], dtype="int64")})
    caplog.set_level(logging.DEBUG)
    ensure_dataframe_float32_convertability(df)
    assert not any("2**24" in m for m in _warnings(caplog))


def test_n02_polars_int64_beyond_mantissa_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    pl = pytest.importorskip("polars")
    from pyutilz.data.pandaslib.dtypes import ensure_dataframe_float32_convertability

    df = pl.DataFrame({"id": pl.Series([16777217], dtype=pl.Int64)})
    caplog.set_level(logging.DEBUG)
    out = ensure_dataframe_float32_convertability(df)
    assert out.schema["id"] == pl.Float32
    assert any("2**24" in m for m in _warnings(caplog)), _warnings(caplog)
