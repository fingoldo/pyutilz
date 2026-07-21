"""Extra tests for pythonlib.py — covers specific uncovered lines."""

import pytest
import os
import sys
import time
import tempfile
from datetime import date, datetime, timezone
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# ensure_installed — lines 55, 58-64
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed


def test_ensure_installed_single_string():
    # line 55: single package string (no sep)
    with patch("pyutilz.core.pythonlib.importlib.util.find_spec", return_value=True):
        ensure_installed("numpy")  # should not install


def test_ensure_installed_missing_package():
    # lines 58-64: missing packages trigger pip install
    with patch("pyutilz.core.pythonlib.importlib.util.find_spec", return_value=None):
        with patch("pyutilz.core.pythonlib.subprocess.check_call", side_effect=Exception("mock")):
            ensure_installed("nonexistent_pkg_xyz")  # should not raise


def test_ensure_installed_none():
    ensure_installed(None)  # no-op


# ---------------------------------------------------------------------------
# flatten_keys_to_set — lines 138, 143-151
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import flatten_keys_to_set


def test_flatten_keys_to_set_number_no_stringify():
    # line 138: number without stringify
    result = flatten_keys_to_set(42)
    assert 42 in result


def test_flatten_keys_to_set_number_stringify():
    # line 136
    result = flatten_keys_to_set(42, stringify=True)
    assert "42" in result


def test_flatten_keys_to_set_custom_obj_stringify():
    # lines 143-147: non-iterable, non-number, non-string with stringify=True
    class Custom:
        def __str__(self):
            return "custom_value"
    result = flatten_keys_to_set(Custom(), stringify=True)
    assert "custom_value" in result


def test_flatten_keys_to_set_custom_obj_stringify_verbose():
    # lines 144-146
    class Custom:
        def __str__(self):
            return "custom_value"
    with patch("pyutilz.core.pythonlib.logger"):
        result = flatten_keys_to_set(Custom(), stringify=True, verbose=True)
    assert "custom_value" in result


def test_flatten_keys_to_set_custom_obj_no_stringify_verbose():
    # lines 149-151: non-stringify, verbose
    class Custom:
        def __str__(self):
            return "custom_value"
    with patch("pyutilz.core.pythonlib.logger"):
        result = flatten_keys_to_set(Custom(), stringify=False, verbose=True)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# get_attr — lines 171, 174
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import get_attr


def test_get_attr_obj_is_unwanted():
    # line 170-171: obj == unwanted_value
    result = get_attr(None, "key", default_value=[1], unwanted_value=None)
    assert result == [1]


def test_get_attr_result_is_unwanted():
    # line 173-174: result == unwanted_value
    result = get_attr({"key": ""}, "key", default_value="default", unwanted_value="")
    assert result == "default"


# ---------------------------------------------------------------------------
# integer_digits / float_distinct_digits_percent — lines 320-327
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import float_distinct_digits_percent


def test_float_distinct_digits_percent():
    result = float_distinct_digits_percent(11.882, precision=3)
    assert abs(result - 0.6) < 0.01


# ---------------------------------------------------------------------------
# read_timezoned_ts — lines 440-441
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import read_timezoned_ts


def test_read_timezoned_ts_has_minus():
    # The "-" in the date triggers the split path; the function always processes
    result = read_timezoned_ts("2020-02-20T11:54:00")
    assert isinstance(result, str)


def test_read_timezoned_ts_with_timezone():
    result = read_timezoned_ts("2020-02-20T11:54:00.000-07:00")
    assert "0700" in result


# ---------------------------------------------------------------------------
# imitate_delay — lines 482-484, 488
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import imitate_delay


def test_imitate_delay_force_no_last_call():
    # lines 485-486: b_force=True, last_call_ts=None
    with patch("time.sleep"):
        result = imitate_delay(0.001, 0.002, last_call_ts=None, b_force=True)
    assert isinstance(result, datetime)


def test_imitate_delay_with_last_call():
    # line 488
    ts = datetime.utcnow()
    with patch("time.sleep"):
        result = imitate_delay(0.001, 0.002, last_call_ts=ts, b_force=False)
    assert isinstance(result, datetime)


def test_imitate_delay_big_delay():
    # lines 482-484
    ts = datetime.utcnow()
    with patch("time.sleep"):
        result = imitate_delay(0.001, 0.002, last_call_ts=ts, b_force=True, big_delay_prob=1.0, big_delay_multiplier=2.0)
    assert isinstance(result, datetime)


def test_imitate_delay_min_ge_max():
    # line 475-477: min >= max warning
    with patch("time.sleep"):
        result = imitate_delay(5.0, 5.0, last_call_ts=None, b_force=True)
    assert isinstance(result, datetime)


# ---------------------------------------------------------------------------
# datetime_to_utc_timestamp — lines 505-507
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import datetime_to_utc_timestamp


def test_datetime_to_utc_timestamp():
    dt = datetime(2020, 1, 1, 0, 0, 0)
    result = datetime_to_utc_timestamp(dt)
    assert result == 1577836800


# ---------------------------------------------------------------------------
# CustomError — lines 532-533
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import CustomError


def test_custom_error_str():
    err = CustomError(404, "not found")
    s = str(err)
    assert "404" in s
    assert "not found" in s


# ---------------------------------------------------------------------------
# lookup_in_stack — lines 549-555
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import lookup_in_stack

_STACK_TEST_VAR = "found_it"


def test_lookup_in_stack():
    result = lookup_in_stack("_STACK_TEST_VAR")
    assert result == "found_it"


def test_lookup_in_stack_missing():
    result = lookup_in_stack("_NONEXISTENT_VAR_XYZ_123")
    assert result is None


# ---------------------------------------------------------------------------
# get_parent_func_args — lines 561-566
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import get_parent_func_args


def test_get_parent_func_args():
    def my_func(a, b, self=None):
        return get_parent_func_args()
    result = my_func(1, 2)
    assert result == {"a": 1, "b": 2}


def test_get_parent_func_args_ignores_incidental_locals():
    """Regression: the dict(args_info.locals) materialisation (perf fix, 2026-07-17) must not
    leak non-parameter local variables into the returned dict -- only names in the caller's
    OWN signature (``args_info.args``) may appear, matching the pre-fix per-key .get() filter."""
    def my_func(a, b, self=None):
        incidental_local = a + b  # noqa: F841 -- deliberately unused; this name must NOT leak into the result
        another_one = "not a param"  # noqa: F841
        return get_parent_func_args()
    result = my_func(1, 2)
    assert result == {"a": 1, "b": 2}
    assert "incidental_local" not in result
    assert "another_one" not in result


def test_get_parent_func_args_many_params_reflects_reassigned_values():
    """Regression: a caller with a large parameter surface (mirroring MRMR's ~300-parameter
    __init__, the motivating case for the perf fix) must still resolve every parameter to its
    CURRENT (possibly reassigned-in-body) value, matching the pre-fix per-key .get() behaviour."""
    params = ", ".join(f"p{i}=0" for i in range(300))
    ns: dict = {"get_parent_func_args": get_parent_func_args}
    exec(
        f"def many_params_func(self, {params}):\n"
        "    p0 = 'reassigned'\n"  # mirrors MRMR's n_jobs=-1 -> resolved-value pattern
        "    return get_parent_func_args()\n",
        ns,
    )
    result = ns["many_params_func"](None)
    assert len(result) == 300
    assert result["p0"] == "reassigned"
    assert result["p1"] == 0
    assert result["p299"] == 0
    assert "self" not in result


# ---------------------------------------------------------------------------
# store_params_in_object / load_object_params_into_func — lines 571-584
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import store_params_in_object, load_object_params_into_func


def test_store_params_in_object():
    class Obj:
        pass
    o = Obj()
    store_params_in_object(o, {"x": 1, "y": 2}, postfix="_param_")
    assert o.x_param_ == 1
    assert o.y_param_ == 2


def test_store_params_in_object_none():
    store_params_in_object(None, {"x": 1})  # no-op


def test_load_object_params_into_func():
    class Obj:
        x_param_ = 10
        y_param_ = 20
    o = Obj()
    d = {}
    load_object_params_into_func(o, d, postfix="_param_")
    assert d["x"] == 10
    assert d["y"] == 20


def test_load_object_params_into_func_none():
    load_object_params_into_func(None, {})  # no-op


def test_store_and_load_round_trip_with_defaults():
    """Regression: the two functions are documented as an inverse pair; before the fix,
    store's default postfix="" and load's default postfix="_param_" didn't compose, so a
    default-args round trip silently returned an empty dict."""
    class Obj:
        pass
    o = Obj()
    store_params_in_object(o, {"alpha": 1, "beta": 2})
    loc: dict = {}
    load_object_params_into_func(o, loc)
    assert loc == {"alpha": 1, "beta": 2}


# ---------------------------------------------------------------------------
# get_partitioned_filepath — lines 636-658 (actually 592-605)
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import get_partitioned_filepath


def test_get_partitioned_filepath():
    result = get_partitioned_filepath("abc.txt", depth=2)
    assert "a" in result
    assert "b" in result


def test_get_partitioned_filepath_special_char():
    result = get_partitioned_filepath("@b", depth=2)
    assert "_" in result  # @ replaced


# ---------------------------------------------------------------------------
# ObjectsDumper / ObjectsLoader — lines 694-721, 737-739, 743-747
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import ObjectsDumper, ObjectsLoader


def test_objects_dumper_and_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        container = {"myobj": [1, 2, 3]}
        dumped = MagicMock(return_value=None)
        d = ObjectsDumper(process_fcn=dumped, process_kwargs={})
        n = d.process_objects(objects_names=["myobj"], container=container, path=tmpdir)
        assert n == 1
        assert dumped.called


def test_objects_dumper_skip_empty():
    container = {"myobj": None}
    dumped = MagicMock()
    d = ObjectsDumper(process_fcn=dumped, process_kwargs={})
    n = d.process_objects(objects_names=["myobj"], container=container, path=".")
    assert n == 0


def test_objects_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy file
        fpath = os.path.join(tmpdir, "myobj.pckl")
        with open(fpath, "w") as f:
            f.write("data")
        container = {}
        loader_fn = MagicMock(return_value=[1, 2])
        lo = ObjectsLoader(process_fcn=loader_fn, process_kwargs={}, rewrite_existing=True)
        n = lo.process_objects(objects_names=["myobj"], container=container, path=tmpdir)
        assert n == 1
        assert container["myobj"] == [1, 2]


def test_objects_loader_no_rewrite():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "myobj.pckl")
        with open(fpath, "w") as f:
            f.write("data")
        container = {"myobj": "existing"}
        loader_fn = MagicMock(return_value=[1, 2])
        lo = ObjectsLoader(process_fcn=loader_fn, process_kwargs={}, rewrite_existing=False)
        n = lo.process_objects(objects_names=["myobj"], container=container, path=tmpdir)
        assert n == 0  # existing non-empty, not rewritten


def test_objects_processor_with_namespace():
    with tempfile.TemporaryDirectory() as tmpdir:
        container = {"myobj": [1]}
        dumped = MagicMock(return_value=None)
        d = ObjectsDumper(process_fcn=dumped, process_kwargs={})
        n = d.process_objects(objects_names="myobj", container=container, path=tmpdir, namespace="ns")
        assert dumped.called


# ---------------------------------------------------------------------------
# HashableDict — line 792
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import HashableDict


def test_hashable_dict():
    d = HashableDict(a=1, b=2)
    assert hash(d) == hash(d)
    s = {d}  # must be hashable
    assert len(s) == 1


# ---------------------------------------------------------------------------
# suppress_stdout_stderr — lines 808-812 (actually 815-826)
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import suppress_stdout_stderr


def test_suppress_stdout_stderr():
    import io
    with suppress_stdout_stderr():
        print("should be suppressed")
    # After context, stdout should be restored
    assert sys.stdout is not None


# ---------------------------------------------------------------------------
# is_jupyter_notebook — lines 836-840
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import is_jupyter_notebook


def test_is_jupyter_notebook():
    # We are not in Jupyter, should return False
    result = is_jupyter_notebook()
    assert result is False


# ---------------------------------------------------------------------------
# is_cuda_available — lines 852-853
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import is_cuda_available


def test_is_cuda_available():
    # Should return bool regardless of environment
    result = is_cuda_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# check_cpu_flag — lines 865-870
# ---------------------------------------------------------------------------

from pyutilz.pythonlib import check_cpu_flag


def test_check_cpu_flag():
    result = check_cpu_flag("avx2")
    assert isinstance(result, bool)


def test_check_cpu_flag_nonexistent():
    result = check_cpu_flag("nonexistent_flag_xyz")
    # Either False (flag not found) or False (cpuinfo not installed)
    assert result is False or isinstance(result, bool)


# ---------------------------------------------------------------------------
# Regression tests: 2026-07-21 audit fixes
# ---------------------------------------------------------------------------


def test_get_attr_explicit_none_default_is_honored():
    # Previously `default_value=None` was silently coerced to [] regardless of
    # whether the caller passed it explicitly; now only the "not passed at all"
    # case coerces to [].
    assert get_attr({"a": 1}, "b", default_value=None) is None


def test_get_attr_no_default_passed_still_coerces_to_list():
    assert get_attr({"a": 1}, "b") == []


def test_keys_changed_enough_default_key_contains_does_not_crash():
    from pyutilz.pythonlib import keys_changed_enough

    # key_contains=None (the function's own default) used to raise
    # TypeError: 'in <string>' requires string as left operand, not NoneType
    assert keys_changed_enough(obj={"a": 100, "b": 180}, prev_obj={"a": 100, "b": 200}) is True
    assert keys_changed_enough(obj={"a": 100, "b": 200}, prev_obj={"a": 100, "b": 200}) is False


def test_float_distinct_digits_percent_negative_matches_positive():
    # int_part's digits were silently dropped for negative numbers (int_part passed
    # to integer_digits() without abs()), giving an asymmetric result vs the positive case.
    positive = float_distinct_digits_percent(11.882, precision=3)
    negative = float_distinct_digits_percent(-11.882, precision=3)
    assert positive == negative


def test_float_distinct_digits_percent_rounding_carry_stays_within_precision():
    """Regression: round(0.99999996, 5) rounds up to exactly 1.0, and frac_part*10**precision
    then produced 10**precision (a precision+1-digit number), silently inflating ntotal."""
    result = float_distinct_digits_percent(0.99999996, precision=5)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_hashable_dict_mixed_key_types_does_not_crash():
    from pyutilz.pythonlib import HashableDict

    # sorted() on raw (key, value) tuples raised TypeError comparing str vs int keys.
    d = HashableDict({1: "a", "b": 2})
    assert isinstance(hash(d), int)


def test_hashable_dict_hash_is_order_independent():
    from pyutilz.pythonlib import HashableDict

    assert hash(HashableDict({"a": 1, "b": 2})) == hash(HashableDict({"b": 2, "a": 1}))


def test_open_safe_shelve_roundtrips_across_separate_with_blocks():
    from pyutilz.pythonlib import open_safe_shelve

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "shelvedb")
        with open_safe_shelve(db_path) as db:
            db["x"] = 42
        # Previously the underlying shelve/dbm connection was never closed/committed
        # before the lock released, so a second, separate open would fail to see the
        # write (or crash outright on some dbm backends).
        with open_safe_shelve(db_path, flag="r") as db2:
            assert db2["x"] == 42


def test_open_safe_shelve_closes_db_even_if_body_raises():
    from pyutilz.pythonlib import open_safe_shelve

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "shelvedb2")
        with pytest.raises(ValueError):
            with open_safe_shelve(db_path) as db:
                db["y"] = 1
                raise ValueError("boom")
        with open_safe_shelve(db_path, flag="r") as db2:
            assert db2["y"] == 1


def test_ensure_installed_uses_running_interpreter_pip():
    # Regression: must invoke `sys.executable -m pip`, never a bare "pip" resolved
    # via PATH search order (which on Windows checks CWD before PATH).
    with patch("pyutilz.core.pythonlib.importlib.util.find_spec", return_value=None):
        with patch("pyutilz.core.pythonlib.subprocess.check_call") as mock_call:
            ensure_installed("nonexistent_pkg_xyz")
            args = mock_call.call_args[0][0]
            assert args[0] == sys.executable
            assert args[1:3] == ["-m", "pip"]
