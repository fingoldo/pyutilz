"""Behaviour of ``pyutilz.data.pandaslib.io_ops``: ``load_df`` reads a pickled frame and keeps its last ``tail`` rows when
``tail`` is a positive int (a zero, negative or missing ``tail`` keeps every row), and ``_try_remove_source`` deletes a
file and reports success, or reports failure without raising when the file cannot be removed. Everything runs on files
under ``tmp_path``."""

from __future__ import annotations

import pandas as pd
import pytest

from pyutilz.data.pandaslib import io_ops


@pytest.fixture
def pickled_frame(tmp_path):
    path = tmp_path / "frame.pkl"
    pd.DataFrame({"v": [1, 2, 3, 4, 5]}).to_pickle(path)
    return str(path)


class TestLoadDf:
    def test_a_positive_tail_keeps_only_the_last_rows(self, pickled_frame):
        assert io_ops.load_df(pickled_frame, 2)["v"].tolist() == [4, 5]

    def test_a_tail_longer_than_the_frame_keeps_everything(self, pickled_frame):
        assert io_ops.load_df(pickled_frame, 99)["v"].tolist() == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("tail", [0, -1, None])
    def test_a_non_positive_or_missing_tail_keeps_every_row(self, pickled_frame, tail):
        assert io_ops.load_df(pickled_frame, tail)["v"].tolist() == [1, 2, 3, 4, 5]

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            io_ops.load_df(str(tmp_path / "absent.pkl"), 1)


class TestTryRemoveSource:
    def test_an_existing_file_is_removed_and_reported(self, tmp_path):
        path = tmp_path / "part.txt"
        path.write_text("x", encoding="utf-8")
        assert io_ops._try_remove_source(str(path)) is True
        assert not path.exists()

    def test_a_missing_file_is_reported_without_raising(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            assert io_ops._try_remove_source(str(tmp_path / "gone.txt")) is False
        assert any("Failed to remove source file" in rec.getMessage() for rec in caplog.records)


class TestConcatAndFlush:
    def test_an_empty_list_writes_nothing_and_returns_none(self, tmp_path):
        lst = []
        assert io_ops.concat_and_flush_df_list(lst, str(tmp_path / "out")) is None
        assert list(tmp_path.iterdir()) == []

    def test_the_frames_are_joined_and_the_list_is_cleared(self, tmp_path):
        lst = [pd.DataFrame({"v": [1]}), pd.DataFrame({"v": [2]})]
        joined = io_ops.concat_and_flush_df_list(lst, str(tmp_path / "out"))
        assert joined["v"].tolist() == [1, 2]
        assert lst == []

    def test_the_default_write_is_a_pickle_with_the_pckl_extension(self, tmp_path):
        io_ops.concat_and_flush_df_list([pd.DataFrame({"v": [1]})], str(tmp_path / "out"))
        assert (tmp_path / "out.pckl").exists()
        assert pd.read_pickle(tmp_path / "out.pckl")["v"].tolist() == [1]

    def test_a_custom_extension_is_used_in_the_file_name(self, tmp_path):
        io_ops.concat_and_flush_df_list([pd.DataFrame({"v": [1]})], str(tmp_path / "out"), write_extension="dat")
        assert (tmp_path / "out.dat").exists()

    def test_the_index_is_reset_by_default_and_kept_when_asked(self, tmp_path):
        frame = pd.DataFrame({"v": [1, 2]}, index=["x", "y"])
        kept = io_ops.concat_and_flush_df_list([frame.copy()], str(tmp_path / "kept"), ignore_index=False)
        reset = io_ops.concat_and_flush_df_list([frame.copy()], str(tmp_path / "reset"))
        assert list(kept.index) == ["x", "y"]
        assert list(reset.index) == [0, 1]

    def test_set_index_promotes_the_column_before_writing(self, tmp_path):
        frame = pd.DataFrame({"key": ["a", "b"], "v": [1, 2]})
        io_ops.concat_and_flush_df_list([frame], str(tmp_path / "out"), set_index="key")
        written = pd.read_pickle(tmp_path / "out.pckl")
        assert written.index.name == "key"

    def test_csv_with_a_header_overwrites_the_file(self, tmp_path):
        io_ops.concat_and_flush_df_list([pd.DataFrame({"v": [1]})], str(tmp_path / "out"), to_csv=True)
        assert (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()[0] == ",v"

    def test_csv_with_selected_columns_appends_without_a_header(self, tmp_path):
        io_ops.concat_and_flush_df_list([pd.DataFrame({"v": [1], "w": [9]})], str(tmp_path / "out"), to_csv=True)
        io_ops.concat_and_flush_df_list([pd.DataFrame({"v": [2], "w": [8]})], str(tmp_path / "out"), to_csv=True, csv_cols=["v"])
        lines = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()
        assert lines[-1] == "0,2"
        assert len(lines) == 3


class TestLoadLogging:
    def test_loading_is_logged_with_the_path(self, tmp_path, caplog):
        path = tmp_path / "frame.pkl"
        pd.DataFrame({"v": [1]}).to_pickle(path)
        with caplog.at_level("INFO"):
            io_ops.load_df(str(path), None)
        assert any(rec.getMessage() == f"Loading data from file {path}..." for rec in caplog.records)

    def test_a_tail_is_logged_with_its_size(self, tmp_path, caplog):
        path = tmp_path / "frame.pkl"
        pd.DataFrame({"v": [1, 2, 3]}).to_pickle(path)
        with caplog.at_level("INFO"):
            io_ops.load_df(str(path), 2)
        assert any(rec.getMessage() == "Limiting to last 2 responses" for rec in caplog.records)

    def test_a_zero_tail_is_not_logged_as_a_limit(self, tmp_path, caplog):
        path = tmp_path / "frame.pkl"
        pd.DataFrame({"v": [1, 2, 3]}).to_pickle(path)
        with caplog.at_level("INFO"):
            io_ops.load_df(str(path), 0)
        assert not any("Limiting to last" in rec.getMessage() for rec in caplog.records)
