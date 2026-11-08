"""Resave-decision pins for data.pandaslib.io_ops._optimize_and_maybe_resave (mutation survivors, first group)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from pyutilz.data.pandaslib import io_ops


class _RecordingLogger:
    def __init__(self):
        self.infos: list = []
        self.warnings: list = []

    def info(self, *args):
        self.infos.append(args)

    def warning(self, *args):
        self.warnings.append(args)


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


class TestSentinelColumn:
    def test_the_new_column_holds_the_source_file_name(self):
        df = pd.DataFrame({"a": [1]})
        io_ops._add_sentinel_column(df, os.path.join("data", "file.csv"), "src", None)
        assert list(df["src"]) == ["file.csv"]

    def test_a_path_contributes_only_its_last_part(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        io_ops._add_sentinel_column(df, str(tmp_path / "file.csv"), "src", None)
        assert list(df["src"]) == ["file.csv"]

    def test_a_colliding_name_gets_a_numeric_suffix_and_a_warning(self, recording_logger):
        df = pd.DataFrame({"src": [0]})
        io_ops._add_sentinel_column(df, "f.csv", "src", None)
        assert "src1" in df.columns
        assert list(df["src1"]) == ["f.csv"]
        assert recording_logger.warnings and recording_logger.warnings[0][0].startswith("Sentinel field src was already in the frame f.csv")

    def test_repeated_collisions_keep_counting_up(self):
        df = pd.DataFrame({"src": [0], "src1": [0]})
        io_ops._add_sentinel_column(df, "f.csv", "src", None)
        assert "src11" in df.columns

    def test_a_sentinel_function_is_applied_to_the_file_name(self):
        df = pd.DataFrame({"a": [1]})
        io_ops._add_sentinel_column(df, os.path.join("data", "file.csv"), "src", lambda name: name.upper())
        assert list(df["src"]) == ["FILE.CSV"]


class TestCollectFrames:
    @pytest.fixture
    def two_csvs(self, tmp_path):
        (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
        (tmp_path / "b.csv").write_text("x\n2\n", encoding="utf-8")
        return tmp_path

    def _collect(self, folder, **kwargs):
        defaults = dict(
            template="*.csv",
            max_files=None,
            exclude=None,
            read_fcn="read_csv",
            optimize=False,
            save_on_successful_optimization=False,
            min_size_improvement_percent=0.1,
            min_size_improvement=1.0,
            write_fcn="to_csv",
            write_extension="csv",
            sentinel_field=None,
            sentinel_fcn=None,
        )
        defaults.update(kwargs)
        return io_ops._collect_frames(str(folder), **defaults)

    def test_every_matching_file_is_read_and_listed(self, two_csvs):
        lst, fnames = self._collect(two_csvs)
        assert len(lst) == 2
        assert sorted(fnames) == sorted(str(two_csvs / n) for n in ("a.csv", "b.csv"))

    def test_an_excluded_name_is_skipped_entirely(self, two_csvs):
        lst, fnames = self._collect(two_csvs, exclude="b")
        assert [os.path.basename(f) for f in fnames] == ["a.csv"]
        assert len(lst) == 1

    def test_a_file_cap_stops_reading_at_the_limit(self, two_csvs):
        lst, _ = self._collect(two_csvs, max_files=1)
        assert len(lst) == 1

    def test_no_file_cap_reads_all_of_them(self, two_csvs):
        lst, _ = self._collect(two_csvs, max_files=None)
        assert len(lst) == 2

    def test_the_read_function_is_looked_up_on_pandas(self, two_csvs):
        lst, _ = self._collect(two_csvs, read_fcn="read_csv")
        assert all(isinstance(frame, pd.DataFrame) for frame in lst)

    def test_a_sentinel_column_is_added_per_file(self, two_csvs):
        lst, _ = self._collect(two_csvs, sentinel_field="src")
        names = sorted(frame["src"].iloc[0] for frame in lst)
        assert names == ["a.csv", "b.csv"]

    def test_no_sentinel_column_without_a_sentinel_field(self, two_csvs):
        lst, _ = self._collect(two_csvs)
        assert all("src" not in frame.columns for frame in lst)

    def test_the_merge_is_logged_with_the_file_rows_and_size(self, two_csvs, monkeypatch, recording_logger):
        monkeypatch.setattr(io_ops, "get_df_memory_consumption", lambda df, deep: 3 * 1024**3)
        self._collect(two_csvs, max_files=1)
        assert ("Merging %s with %s rows of size %.1f Gb", str(two_csvs / "a.csv"), 1, 3.0) in recording_logger.infos or (
            "Merging %s with %s rows of size %.1f Gb",
            str(two_csvs / "b.csv"),
            1,
            3.0,
        ) in recording_logger.infos

    def test_shallow_size_is_measured_without_optimization(self, two_csvs, monkeypatch):
        seen: list = []
        monkeypatch.setattr(io_ops, "get_df_memory_consumption", lambda df, deep: seen.append(deep) or 1024**3)
        self._collect(two_csvs, max_files=1, optimize=False)
        assert seen == [False]

    def test_deep_size_is_measured_with_optimization(self, two_csvs, monkeypatch):
        seen: list = []
        monkeypatch.setattr(io_ops, "get_df_memory_consumption", lambda df, deep: seen.append(deep) or 1024**3)
        monkeypatch.setattr(io_ops, "optimize_dtypes", lambda df: df)
        self._collect(two_csvs, max_files=1, optimize=True)
        assert seen[0] is True

    def test_optimization_receives_its_arguments_in_order(self, two_csvs, monkeypatch):
        calls: list = []

        def spy(tmp_df, filename, old_size, write_fcn, write_extension, save, pct, absolute):
            calls.append((filename, old_size, write_fcn, write_extension, save, pct, absolute))
            return tmp_df

        monkeypatch.setattr(io_ops, "_optimize_and_maybe_resave", spy)
        monkeypatch.setattr(io_ops, "get_df_memory_consumption", lambda df, deep: 1024**3)
        self._collect(
            two_csvs,
            max_files=1,
            optimize=True,
            save_on_successful_optimization=True,
            min_size_improvement_percent=0.25,
            min_size_improvement=2.0,
            write_fcn="to_parquet",
            write_extension="parquet",
        )
        filename, old_size, write_fcn, write_extension, save, pct, absolute = calls[0]
        assert write_fcn == "to_parquet"
        assert write_extension == "parquet"
        assert save is True
        assert pct == 0.25
        assert absolute == 2.0
        assert old_size == 1.0

    def test_optimization_is_skipped_when_disabled(self, two_csvs, monkeypatch):
        calls: list = []
        monkeypatch.setattr(io_ops, "_optimize_and_maybe_resave", lambda *a, **k: calls.append(1) or a[0])
        self._collect(two_csvs, max_files=1, optimize=False)
        assert calls == []


class TestExcludeAndArgumentOrder:
    def test_an_excluded_file_does_not_stop_the_later_files(self, tmp_path):
        (tmp_path / "a_skip.csv").write_text("x\n1\n", encoding="utf-8")
        (tmp_path / "b.csv").write_text("x\n2\n", encoding="utf-8")
        (tmp_path / "c.csv").write_text("x\n3\n", encoding="utf-8")
        _, fnames = io_ops._collect_frames(str(tmp_path), "*.csv", None, "skip", "read_csv", False, False, 0.1, 1.0, "to_csv", "csv", None, None)
        assert sorted(os.path.basename(f) for f in fnames) == ["b.csv", "c.csv"]

    def test_the_optimizer_receives_the_frame_first_and_the_path_second(self, tmp_path, monkeypatch):
        (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
        seen: list = []

        def spy(tmp_df, filename, *rest):
            seen.append((type(tmp_df).__name__, filename))
            return tmp_df

        monkeypatch.setattr(io_ops, "_optimize_and_maybe_resave", spy)
        monkeypatch.setattr(io_ops, "get_df_memory_consumption", lambda df, deep: 1024**3)
        io_ops._collect_frames(str(tmp_path), "*.csv", None, None, "read_csv", True, False, 0.1, 1.0, "to_csv", "csv", None, None)
        assert seen == [("DataFrame", str(tmp_path / "a.csv"))]


class TestFlushAndCleanup:
    def test_the_frames_are_concatenated_into_the_joint_file_with_the_given_options(self, monkeypatch):
        calls: list = []

        def spy(lst, file_name, write_fcn, write_extension, set_index):
            calls.append((len(lst), file_name, write_fcn, write_extension, set_index))
            return pd.DataFrame({"x": [1, 2]})

        monkeypatch.setattr(io_ops._facade, "concat_and_flush_df_list", spy)
        io_ops._flush_and_cleanup([pd.DataFrame({"x": [1]})], [], "joint.pckl", "to_pickle", "pckl", "key", False)
        assert calls == [(1, "joint.pckl", "to_pickle", "pckl", "key")]

    def test_the_merged_size_is_logged_when_a_result_exists(self, monkeypatch, recording_logger):
        monkeypatch.setattr(io_ops._facade, "concat_and_flush_df_list", lambda *a, **k: pd.DataFrame({"x": [1, 2, 3]}))
        io_ops._flush_and_cleanup([], [], "joint", "to_csv", "csv", None, False)
        assert ("Final df size (%s rows)", 3) in recording_logger.infos

    def test_nothing_is_logged_when_no_result_comes_back(self, monkeypatch, recording_logger):
        monkeypatch.setattr(io_ops._facade, "concat_and_flush_df_list", lambda *a, **k: None)
        out = io_ops._flush_and_cleanup([], [], "joint", "to_csv", "csv", None, False)
        assert out is None
        assert recording_logger.infos == []

    def test_the_source_files_are_removed_when_asked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io_ops._facade, "concat_and_flush_df_list", lambda *a, **k: pd.DataFrame({"x": [1]}))
        src = tmp_path / "src.csv"
        src.write_text("x\n1\n", encoding="utf-8")
        io_ops._flush_and_cleanup([], [str(src)], "joint", "to_csv", "csv", None, True)
        assert not src.exists()

    def test_the_source_files_are_kept_when_not_asked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io_ops._facade, "concat_and_flush_df_list", lambda *a, **k: pd.DataFrame({"x": [1]}))
        src = tmp_path / "src.csv"
        src.write_text("x\n1\n", encoding="utf-8")
        io_ops._flush_and_cleanup([], [str(src)], "joint", "to_csv", "csv", None, False)
        assert src.exists()

    def test_a_source_that_cannot_be_removed_is_warned_about_with_the_count(self, monkeypatch, recording_logger):
        monkeypatch.setattr(io_ops._facade, "concat_and_flush_df_list", lambda *a, **k: pd.DataFrame({"x": [1]}))
        monkeypatch.setattr(io_ops, "_try_remove_source", lambda filename: False)
        io_ops._flush_and_cleanup([], ["left.csv", "other.csv"], "joint", "to_csv", "csv", None, True)
        assert recording_logger.warnings == [
            (
                "%d source file(s) could not be removed after the concat and are still on disk, e.g. %s",
                2,
                "left.csv",
            )
        ]

    def test_no_warning_when_every_source_is_removed(self, monkeypatch, recording_logger):
        monkeypatch.setattr(io_ops._facade, "concat_and_flush_df_list", lambda *a, **k: pd.DataFrame({"x": [1]}))
        monkeypatch.setattr(io_ops, "_try_remove_source", lambda filename: True)
        io_ops._flush_and_cleanup([], ["a.csv"], "joint", "to_csv", "csv", None, True)
        assert recording_logger.warnings == []

    def test_removal_is_not_attempted_when_not_asked(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(io_ops._facade, "concat_and_flush_df_list", lambda *a, **k: pd.DataFrame({"x": [1]}))
        monkeypatch.setattr(io_ops, "_try_remove_source", lambda filename: calls.append(filename) or True)
        io_ops._flush_and_cleanup([], ["a.csv"], "joint", "to_csv", "csv", None, False)
        assert calls == []


class TestReadStatsDefaults:
    def test_the_documented_defaults_are_pinned(self):
        import inspect

        params = inspect.signature(io_ops.read_stats_from_multiple_files).parameters
        expected = {
            "joint_file_name": "joint_features",
            "folder": "features",
            "max_files": 250,
            "template": "*.pckl",
            "exclude": None,
            "read_fcn": "read_pickle",
            "write_fcn": "to_pickle",
            "write_extension": "pckl",
            "delete_after": False,
            "sentinel_field": None,
            "sentinel_fcn": None,
            "set_index": None,
            "optimize": False,
            "save_on_successful_optimization": False,
            "min_size_improvement_percent": 0.05,
            "min_size_improvement": 5.0,
        }
        for name, value in expected.items():
            assert params[name].default == value, name

    def test_a_call_with_defaults_merges_the_pickles_into_the_joint_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        features = tmp_path / "features"
        features.mkdir()
        pd.DataFrame({"x": [1]}).to_pickle(features / "a.pckl")
        pd.DataFrame({"x": [2]}).to_pickle(features / "b.pckl")
        out = io_ops.read_stats_from_multiple_files()
        assert sorted(out["x"]) == [1, 2]
        assert (tmp_path / "joint_features.pckl").exists()
        assert (features / "a.pckl").exists()


class TestReadStatsCollectCall:
    def test_the_collection_receives_its_arguments_in_declared_order(self, monkeypatch):
        calls: list = []

        def spy(*args):
            calls.append(args)
            return [], []

        monkeypatch.setattr(io_ops, "_collect_frames", spy)
        io_ops.read_stats_from_multiple_files(
            joint_file_name="j",
            folder="f",
            max_files=7,
            template="*.x",
            exclude="skip",
            read_fcn="read_csv",
            write_fcn="to_csv",
            write_extension="csv",
            sentinel_field="src",
            sentinel_fcn=None,
            optimize=True,
            save_on_successful_optimization=False,
            min_size_improvement_percent=0.2,
            min_size_improvement=3.0,
        )
        assert calls == [("f", "*.x", 7, "skip", "read_csv", True, False, 0.2, 3.0, "to_csv", "csv", "src", None)]


class TestReadStatsBranches:
    def test_no_matching_files_returns_none_without_merging(self, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setattr(io_ops, "_flush_and_cleanup", lambda *a: calls.append(a) or None)
        assert io_ops.read_stats_from_multiple_files(folder=str(tmp_path), template="*.nothing") is None
        assert calls == []

    def test_a_single_matching_file_is_merged(self, tmp_path):
        pd.DataFrame({"x": [5]}).to_pickle(tmp_path / "only.pckl")
        out = io_ops.read_stats_from_multiple_files(joint_file_name=str(tmp_path / "joint"), folder=str(tmp_path))
        assert out is not None
        assert list(out["x"]) == [5]

    def test_the_flush_receives_the_index_and_delete_flag_in_order(self, tmp_path, monkeypatch):
        pd.DataFrame({"x": [1]}).to_pickle(tmp_path / "a.pckl")
        seen: list = []

        def spy(lst, fnames, joint, write_fcn, write_extension, set_index, delete_after):
            seen.append((len(lst), joint, write_fcn, write_extension, set_index, delete_after))
            return None

        monkeypatch.setattr(io_ops, "_flush_and_cleanup", spy)
        io_ops.read_stats_from_multiple_files(joint_file_name="joint", folder=str(tmp_path), set_index="key", delete_after=True)
        assert seen == [(1, "joint", "to_pickle", "pckl", "key", True)]


class TestLoadTailAndRemoval:
    @pytest.fixture
    def pickled(self, tmp_path):
        path = tmp_path / "rows.pckl"
        pd.DataFrame({"x": [1, 2, 3]}).to_pickle(path)
        return str(path)

    def test_a_tail_of_one_keeps_only_the_last_row(self, pickled):
        assert list(io_ops.load_df(pickled, 1)["x"]) == [3]

    def test_a_tail_of_two_keeps_the_last_two_rows(self, pickled):
        assert list(io_ops.load_df(pickled, 2)["x"]) == [2, 3]

    def test_a_zero_or_missing_tail_keeps_every_row(self, pickled):
        assert list(io_ops.load_df(pickled, 0)["x"]) == [1, 2, 3]
        assert list(io_ops.load_df(pickled, None)["x"]) == [1, 2, 3]

    def test_a_failed_removal_returns_false_and_warns_with_the_file_and_error(self, monkeypatch, recording_logger):
        error = OSError("locked")

        def refuse(path):
            raise error

        monkeypatch.setattr(io_ops.os, "remove", refuse)
        assert io_ops._try_remove_source("x.csv") is False
        assert recording_logger.warnings == [("Failed to remove source file %s after concat: %s", "x.csv", error)]

    def test_a_successful_removal_returns_true_and_is_quiet(self, tmp_path, recording_logger):
        path = tmp_path / "gone.csv"
        path.write_text("x\n1\n", encoding="utf-8")
        assert io_ops._try_remove_source(str(path)) is True
        assert not path.exists()
        assert recording_logger.warnings == []


class TestParquetRowLimit:
    def test_a_row_limit_returns_that_many_rows(self, tmp_path):
        pytest.importorskip("pyarrow")
        path = tmp_path / "data.parquet"
        pd.DataFrame({"x": [1, 2, 3]}).to_parquet(path)
        assert len(io_ops.read_parquet_with_pyarrow(str(path), 2)) == 2

    def test_no_row_limit_returns_every_row(self, tmp_path):
        pytest.importorskip("pyarrow")
        path = tmp_path / "data.parquet"
        pd.DataFrame({"x": [1, 2, 3]}).to_parquet(path)
        assert len(io_ops.read_parquet_with_pyarrow(str(path), 0)) == 3
