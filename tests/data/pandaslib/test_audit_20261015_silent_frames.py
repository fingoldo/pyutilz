from __future__ import annotations
import logging
import pandas as pd
import pytest


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_s12_unhashable_column_skip_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.data.pandaslib import frames

    df = pd.DataFrame({"const": [1, 1], "lst": [[1], [2]]})
    caplog.set_level(logging.DEBUG)
    assert frames.get_suspiciously_constant_columns(df) == ["const"]
    assert any("could not check 1 column(s)" in m and "lst" in m for m in _warnings(caplog)), _warnings(caplog)
