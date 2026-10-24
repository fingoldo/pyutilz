from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any
import pandas as pd
import pytest


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


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
