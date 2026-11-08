"""Resave-decision pins for data.pandaslib.io_ops._optimize_and_maybe_resave (mutation survivors, first group)."""

from __future__ import annotations

import pandas as pd
import pytest

from pyutilz.data.pandaslib import io_ops


class _RecordingLogger:
    def __init__(self):
        self.infos: list = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        pass


@pytest.fixture
def recording_logger(monkeypatch):
    fake = _RecordingLogger()
    monkeypatch.setattr(io_ops, "logger", fake)
    return fake


def _sizes(monkeypatch, new_gb):
    monkeypatch.setattr(io_ops, "optimize_dtypes", lambda df: df)
    monkeypatch.setattr(io_ops, "get_df_memory_consumption", lambda df, deep: new_gb * 1024**3)


def _run(tmp_path, *, old_size, new_gb, save=True, pct=0.1, absolute=1.0, name="data.parquet", ext="csv", monkeypatch=None):
    _sizes(monkeypatch, new_gb)
    target = tmp_path / name
    df = pd.DataFrame({"x": [1, 2]})
    out = io_ops._optimize_and_maybe_resave(
        df,
        str(target),
        old_size,
        "to_csv",
        ext,
        save,
        pct,
        absolute,
    )
    return out, tmp_path / (target.stem + "." + ext)


class TestResaveDecision:
    def test_a_relative_shrink_past_the_percentage_triggers_a_resave(self, tmp_path, monkeypatch):
        _, written = _run(tmp_path, old_size=10.0, new_gb=8.0, pct=0.1, absolute=100.0, monkeypatch=monkeypatch)
        assert written.exists()

    def test_a_shrink_below_the_percentage_and_the_absolute_gain_does_not_resave(self, tmp_path, monkeypatch):
        _, written = _run(tmp_path, old_size=10.0, new_gb=9.5, pct=0.1, absolute=1.0, monkeypatch=monkeypatch)
        assert not written.exists()

    def test_an_absolute_gain_exactly_at_the_threshold_triggers_a_resave(self, tmp_path, monkeypatch):
        _, written = _run(tmp_path, old_size=10.0, new_gb=9.0, pct=0.5, absolute=1.0, monkeypatch=monkeypatch)
        assert written.exists()

    def test_an_absolute_gain_below_the_threshold_does_not_resave(self, tmp_path, monkeypatch):
        _, written = _run(tmp_path, old_size=10.0, new_gb=9.5, pct=0.5, absolute=1.0, monkeypatch=monkeypatch)
        assert not written.exists()

    def test_a_percentage_exactly_at_the_limit_triggers_a_resave(self, tmp_path, monkeypatch):
        _, written = _run(tmp_path, old_size=10.0, new_gb=9.0, pct=0.1, absolute=100.0, monkeypatch=monkeypatch)
        assert written.exists()

    def test_nothing_is_written_when_saving_is_off(self, tmp_path, monkeypatch):
        _, written = _run(tmp_path, old_size=10.0, new_gb=1.0, save=False, monkeypatch=monkeypatch)
        assert not written.exists()

    def test_the_resaved_file_takes_the_new_extension_and_the_frame_is_returned(self, tmp_path, monkeypatch):
        out, written = _run(tmp_path, old_size=10.0, new_gb=1.0, name="sales.parquet", ext="csv", monkeypatch=monkeypatch)
        assert written == tmp_path / "sales.csv"
        assert written.exists()
        assert list(out["x"]) == [1, 2]


class TestResaveLogging:
    def test_the_optimized_size_is_logged_in_gigabytes(self, tmp_path, monkeypatch, recording_logger):
        _run(tmp_path, old_size=10.0, new_gb=3.0, monkeypatch=monkeypatch)
        assert ("After optimization, %s got size %.1f Gb", str(tmp_path / "data.parquet"), 3.0) in recording_logger.infos

    def test_a_resave_is_announced_with_the_source_name(self, tmp_path, monkeypatch, recording_logger):
        _run(tmp_path, old_size=10.0, new_gb=1.0, monkeypatch=monkeypatch)
        assert ("Re-saving file %s due to lower size", str(tmp_path / "data.parquet")) in recording_logger.infos

    def test_no_resave_announcement_when_the_size_did_not_shrink_enough(self, tmp_path, monkeypatch, recording_logger):
        _run(tmp_path, old_size=10.0, new_gb=9.5, pct=0.5, absolute=5.0, monkeypatch=monkeypatch)
        assert not any(call[0] == "Re-saving file %s due to lower size" for call in recording_logger.infos)


class TestSizeMeasurementAndNaming:
    def test_the_optimized_size_is_measured_with_deep_accounting(self, tmp_path, monkeypatch):
        seen: list = []

        def fake_size(df, deep):
            seen.append(deep)
            return 1024**3

        monkeypatch.setattr(io_ops, "optimize_dtypes", lambda df: df)
        monkeypatch.setattr(io_ops, "get_df_memory_consumption", fake_size)
        io_ops._optimize_and_maybe_resave(pd.DataFrame({"x": [1]}), str(tmp_path / "a.parquet"), 2.0, "to_csv", "csv", False, 0.1, 1.0)
        assert seen == [True]

    def test_a_name_with_several_dots_keeps_every_part_before_the_extension(self, tmp_path, monkeypatch):
        _sizes(monkeypatch, 1.0)
        target = tmp_path / "sales.v2.parquet"
        io_ops._optimize_and_maybe_resave(pd.DataFrame({"x": [1]}), str(target), 10.0, "to_csv", "csv", True, 0.1, 1.0)
        assert (tmp_path / "sales.v2.csv").exists()
        assert not (tmp_path / "salesv2.csv").exists()
