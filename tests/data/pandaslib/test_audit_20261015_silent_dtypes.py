from __future__ import annotations
import logging
import pandas as pd
import pytest


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_s06_category_failure_warned_without_verbose(caplog: pytest.LogCaptureFixture) -> None:
    from pyutilz.data.pandaslib.dtypes import _promote_column_to_category

    df = pd.DataFrame({"c": [[1], [2]]})  # lists are unhashable, so nunique raises
    new_dtypes: dict = {}
    caplog.set_level(logging.DEBUG)
    _promote_column_to_category(df, "c", "object", new_dtypes, 5, False, verbose=False)
    assert "c" not in new_dtypes
    assert any("leaving its dtype unchanged" in m for m in _warnings(caplog)), _warnings(caplog)


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
