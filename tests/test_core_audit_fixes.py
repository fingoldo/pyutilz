"""Regression tests for the 2026-09-02 core/dev/system domain audit -- core/ findings."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest

from pyutilz.core.matrix import CsrIndPtrConstructor, CsrRowColConstructor
from pyutilz.core.pythonlib import check_cpu_flag, count_trailing_zeros, load_file, show_methods
from pyutilz.core.serialization import serialize, str_to_class, unserialize

# -------------------- F08: count_trailing_zeros --------------------


@pytest.mark.parametrize(
    "number, precision, expected",
    [
        (1.30e-6, 8, 1),
        (100.0, 5, 5),
        (1000, 3, 3),
        (1.25, 4, 2),
        (0.0, 2, 2),
    ],
)
def test_count_trailing_zeros_ignores_integer_part(number, precision, expected):
    """The scan must stop at the decimal separator instead of counting the integer part's zeros."""
    assert count_trailing_zeros(number, precision=precision) == expected


# -------------------- F09: str_to_class bracket slice --------------------


class _Probe:
    """Records the single positional argument it was constructed with."""

    def __init__(self, init_data):
        self.init_data = init_data


def test_str_to_class_passes_balanced_parentheses():
    """The documented contract includes BOTH parentheses; the slice used to drop the closing one."""
    obj = str_to_class(f"{__name__}._Probe(abc)")
    assert obj.init_data == "(abc)"


def test_str_to_class_without_parentheses_still_works():
    obj = str_to_class(f"{__name__}._Probe")
    assert obj.init_data == ""


# -------------------- F26: load_file --------------------


def test_load_file_unknown_extension_raises(tmp_path):
    """A present file with an unhandled extension must not silently produce None."""
    p = tmp_path / "model.pkl"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_file(str(p))


def test_load_file_pckl_raw_object_roundtrip(tmp_path):
    """unpickle_to_pd=False is the documented way to get the raw object back."""
    from pyutilz.core.safe_pickle import safe_dump

    p = tmp_path / "obj.pckl"
    safe_dump({"a": 1}, str(p))
    assert load_file(str(p), unpickle_to_pd=False) == {"a": 1}


# -------------------- F27: unserialize None-conflation --------------------


def test_unserialize_none_roundtrip_is_distinguishable_from_missing_file(tmp_path):
    p = tmp_path / "state.pkl"
    assert serialize(None, str(p)) is True
    assert unserialize(str(p)) is None  # a genuinely stored None
    with pytest.raises(FileNotFoundError):
        unserialize(str(p) + ".nope")


def test_unserialize_corrupt_payload_raises(tmp_path):
    p = tmp_path / "corrupt.pkl"
    p.write_bytes(b"not a pickle at all")
    with pytest.raises(Exception) as exc:
        unserialize(str(p))
    assert not isinstance(exc.value, FileNotFoundError)


# -------------------- F32: flatten_keys_to_set preview --------------------


def test_flatten_keys_to_set_logs_a_truncated_preview(caplog):
    from pyutilz.core.pythonlib import flatten_keys_to_set

    class _Opaque:
        def __str__(self):
            return "A" * 50 + "B" * 50

    with caplog.at_level("INFO"):
        flatten_keys_to_set(_Opaque(), verbose=True)
    messages = [r.getMessage() for r in caplog.records if "Skipping object" in r.getMessage()]
    assert messages, "expected a skip log line"
    assert "B" not in messages[0], messages[0]


# -------------------- F34/F35: docstring vs behavior --------------------


def test_ensure_valid_filename_docstring_examples_execute():
    """The examples used to call a nonexistent `fix_filename`, so doctest collection failed."""
    import doctest

    import pyutilz.core.pythonlib as pythonlib

    results = doctest.run_docstring_examples(pythonlib.ensure_valid_filename, {"ensure_valid_filename": pythonlib.ensure_valid_filename}, verbose=False)
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(verbose=False)
    for test in finder.find(pythonlib.ensure_valid_filename, globs={"ensure_valid_filename": pythonlib.ensure_valid_filename}):
        runner.run(test)
    assert runner.failures == 0


class _WithInnerDunder:
    def fit__transform(self):  # noqa: D102
        return None


def test_show_methods_keeps_names_with_an_inner_double_underscore():
    """ "non-dunder" means leading AND trailing, not "contains __ anywhere"."""
    names = show_methods(_WithInnerDunder())
    assert "fit__transform" in names
    assert "__init__" not in names


# -------------------- F38: check_cpu_flag --------------------


def test_check_cpu_flag_missing_flags_key_warns(monkeypatch, caplog):
    """A py-cpuinfo shape change must not be reported as "flag unsupported" in silence."""
    import cpuinfo

    monkeypatch.setattr(cpuinfo, "get_cpu_info", lambda: {})
    with caplog.at_level("WARNING"):
        assert check_cpu_flag("avx2") is False
    assert any("no 'flags' key" in r.getMessage() for r in caplog.records)


@pytest.mark.slow  # >3s measured (pytest --durations, 2026-09-02)
def test_check_cpu_flag_present_flag_is_reported():
    import cpuinfo

    info = cpuinfo.get_cpu_info()
    flags = info.get("flags") or []
    if not flags:
        pytest.skip("py-cpuinfo reports no flags on this host")
    assert check_cpu_flag(flags[0]) is True


# -------------------- F29: matrix builders --------------------


def test_csr_indptr_constructor_is_reusable_after_clear_source():
    c = CsrIndPtrConstructor()
    c.add_element(1, 0)
    c.add_row()
    m = c.build_matrix(dtype=np.int32, clear_source=True)
    assert m.toarray().tolist() == [[1]]
    c.add_element(5, 0)  # must not raise AttributeError
    c.add_row()
    assert c.build_matrix(dtype=np.int32).toarray().tolist() == [[5]]


def test_csr_indptr_constructor_closes_a_pending_row():
    """Elements added since the last add_row() used to be silently dropped."""
    c = CsrIndPtrConstructor()
    c.add_element(1, 0)
    c.add_row()
    c.add_element(2, 1)  # no trailing add_row()
    m = c.build_matrix(dtype=np.int32)
    assert m.toarray().tolist() == [[1, 0], [0, 2]]


def test_csr_rowcol_constructor_is_reusable_after_clear_source():
    c = CsrRowColConstructor()
    c.add_element(3, row=0, col=0)
    assert c.build_matrix(dtype=np.int32, clear_source=True).toarray().tolist() == [[3]]
    c.add_element(4, row=0, col=0)
    assert c.build_matrix(dtype=np.int32).toarray().tolist() == [[4]]
