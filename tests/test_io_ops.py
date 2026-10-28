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
