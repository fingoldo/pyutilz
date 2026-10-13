"""Continuation of test_domain_core_dev_system_audit_20260903: split out to keep each test module under the 1k-LOC gate."""
import logging
import os
import sys
import threading
import types
from pathlib import Path
import numpy as np
import pytest

def test_f43_distinct_digits_match_the_decimal_rendering():
    """int(frac_part * 10**precision) truncated (0.05063*100000 = 5062.999... -> 5062) and
    integer_digits could not see leading zeros -- 12.8% of random 5-decimal values disagreed with
    the string ground truth."""
    from pyutilz.core.pythonlib import float_distinct_digits_percent

    assert float_distinct_digits_percent(25.05063, precision=5) == pytest.approx(5 / 7)
    assert float_distinct_digits_percent(0.005, precision=3) == pytest.approx(2 / 4)

    rng = np.random.default_rng(0)
    for value in rng.uniform(0, 100, 500):
        rendered = format(abs(value), ".5f").replace(".", "")
        assert float_distinct_digits_percent(value, precision=5) == pytest.approx(len(set(rendered)) / len(rendered))


def test_f108_precision_zero_counts_no_trailing_zeros():
    """format(x, ".0f") has no decimal separator, so the separator `break` never fired and the
    INTEGER part's zeros were counted as fractional ones."""
    from pyutilz.core.pythonlib import count_trailing_zeros

    assert count_trailing_zeros(100.0, precision=0) == 0
    assert count_trailing_zeros(1000, precision=0) == 0


def test_f114_decimal_comma_is_not_read_as_a_thousands_separator():
    """Stripping commas unconditionally turned "1,5" into 15.0 -- a 10x error with no warning."""
    from pyutilz.core.pythonlib import is_float, to_float

    assert is_float("1,5") is False
    with pytest.raises(ValueError):
        to_float("1,5")
    assert to_float("1,000") == 1000.0
    assert to_float("1,234.5") == 1234.5


# ---------------------------------------------------------------------------
# F44 / F45 / F49 / F116 -- objects
# ---------------------------------------------------------------------------


def test_f44_recursive_calls_forward_every_flag():
    """Lines 72/88 forwarded only dict_merge_symbol, so stringify/verbose/max_chars were dead for
    any nested input -- {"a": [object()]} silently produced an empty set."""
    from pyutilz.core.pythonlib import flatten_keys_to_set

    assert flatten_keys_to_set({"a": [object()]}, stringify=True) != set()


def test_f45_string_values_keep_their_key():
    """str is Iterable, so a string VALUE recursed and the key was lost: {"a": "b"} -> {'b'}, making
    {"host": "prod"} and {"region": "prod"} compare equal."""
    from pyutilz.core.pythonlib import flatten_keys_to_set

    assert flatten_keys_to_set({"a": "b"}) == {"a:b"}


def test_f49_numpy_array_values_do_not_raise():
    """`==` against a numpy array raises "truth value of an array ... is ambiguous"; the sentinel is
    None, so identity is the right comparison."""
    from pyutilz.core.pythonlib import get_attr

    out = get_attr({"a": np.array([1, 2])}, "a")
    np.testing.assert_array_equal(out, np.array([1, 2]))


def test_f116_a_vanished_key_counts_as_a_change():
    """obj.get(key) yielded None, is_float(None) is False, and the key was skipped with no log -- so
    a change-triggered alert never fired for a metric that disappeared."""
    from pyutilz.core.pythonlib import keys_changed_enough

    assert keys_changed_enough(obj={}, prev_obj={"b": 100}) is True


# ---------------------------------------------------------------------------
# F46 -- read_timezoned_ts
# ---------------------------------------------------------------------------


def test_f46_timestamp_without_an_offset_is_returned_unchanged():
    """Searching for "+" then "-" ANYWHERE matched the date's own hyphens, so an offset-less
    timestamp had every colon stripped out of its time, unparseable and unlogged."""
    from pyutilz.core.pythonlib import read_timezoned_ts

    assert read_timezoned_ts("2020-02-20T11:54:00") == "2020-02-20T11:54:00"
    assert read_timezoned_ts("2020-02-20T11:54:00.000-07:00") == "2020-02-20T11:54:00.000-0700"
    assert read_timezoned_ts("2020-02-20T11:54:00.000+05:30") == "2020-02-20T11:54:00.000+0530"


# ---------------------------------------------------------------------------
# F47 / F48 / F149 -- filesystem
# ---------------------------------------------------------------------------


def test_f47_overlapping_suppression_blocks_do_not_close_stdout():
    """The saved-stream snapshot is per-invocation over process-GLOBAL state: the inner block's exit
    reinstalled the outer block's already-closed devnull, and every later print raised."""
    from pyutilz.core.pythonlib import suppress_stdout_stderr

    real_stdout = sys.stdout
    outer = suppress_stdout_stderr()
    inner = suppress_stdout_stderr()
    outer.__enter__()
    inner.__enter__()
    inner.__exit__(None, None, None)
    outer.__exit__(None, None, None)
    assert sys.stdout is real_stdout
    assert not sys.stdout.closed


def test_f48_empty_name_does_not_produce_an_absolute_prefix():
    """`sep.join([]) + sep` is a bare separator, which os.path.join treats as absolute -- everything
    before it is discarded and the file lands at the filesystem root."""
    from pyutilz.core.pythonlib import get_partitioned_filepath

    assert get_partitioned_filepath("") == ""
    joined = os.path.join("C:" + os.sep + "data", get_partitioned_filepath("") + "x.pckl")
    assert joined == os.path.join("C:" + os.sep + "data", "x.pckl")


def test_f149_pandas_is_imported_lazily(monkeypatch, tmp_path: Path):
    """A top-level `import pandas` made load_file("m.joblib") fail outright wherever joblib is
    installed but pandas is not, though only the .pckl branch uses it."""
    import joblib

    from pyutilz.core.pythonlib import load_file

    path = tmp_path / "m.joblib"
    joblib.dump({"a": 1}, path)

    real_import = __import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("pandas is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_pandas)
    assert load_file(str(path)) == {"a": 1}


# ---------------------------------------------------------------------------
# F50 / F115 -- serialize
# ---------------------------------------------------------------------------


def test_f50_named_temporary_file_is_accepted():
    """tempfile._TemporaryFileWrapper is not an io.IOBase subclass, and the TypeError raised for it
    sat inside the function's own try -- swallowed, and reported as a plain None."""
    import tempfile

    from pyutilz.core.serialization import serialize, unserialize

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        assert serialize({"a": 1}, fh) is True
        name = fh.name
    try:
        assert unserialize(name) == {"a": 1}
    finally:
        os.unlink(name)


def test_f50_a_non_file_object_raises_typeerror():
    from pyutilz.core.serialization import serialize

    with pytest.raises(TypeError):
        serialize({"a": 1}, 42)


def test_f115_failures_are_raised_not_returned_as_none():
    """unserialize was deliberately reframed to raise (2026-09-02 F27), so the two halves of the
    round trip had opposite error contracts and a caller could store the None as the payload."""
    from pyutilz.core.serialization import serialize

    with pytest.raises(TypeError, match="cannot pickle"):
        serialize(threading.Lock(), None)


# ---------------------------------------------------------------------------
# F51 / F52 -- safe_pickle sidecar verification
# ---------------------------------------------------------------------------


def test_f51_non_hex_sidecar_returns_false_instead_of_raising_typeerror(tmp_path: Path):
    """hmac.compare_digest rejects non-ASCII str with TypeError, escaping a function documented to
    return only True/False -- and DiskCache.get does not catch TypeError either."""
    from pyutilz.core.safe_pickle import verify_sidecar

    payload = tmp_path / "p.pkl"
    payload.write_bytes(b"data")
    (tmp_path / "p.pkl.sha256").write_text("\u00e9" * 64, encoding="utf-8")
    assert verify_sidecar(str(payload)) is False


def test_f52_orphaned_sidecar_returns_false_instead_of_filenotfounderror(tmp_path: Path):
    """DiskCache._evict_if_needed produces exactly this state: payload unlinked, sidecar left."""
    from pyutilz.core.safe_pickle import verify_sidecar

    (tmp_path / "gone.pkl.sha256").write_text("0" * 64, encoding="utf-8")
    assert verify_sidecar(str(tmp_path / "gone.pkl")) is False


# ---------------------------------------------------------------------------
# F54 / F110 / F133 -- filemaker
# ---------------------------------------------------------------------------


def test_f54_init_raises_when_no_token_can_be_obtained(monkeypatch):
    """init() discarded the token and returned normally, so web.connect kept the Basic-auth headers
    and every later post_filemaker_record 401'd with a misleading ValueError."""
    from pyutilz.core import filemaker as fm

    monkeypatch.setattr(fm, "get_session_token", lambda **kw: None)
    monkeypatch.setattr(fm, "web", types.SimpleNamespace(connect=lambda **kw: None, get_url=lambda *a, **kw: None))
    with pytest.raises(RuntimeError):
        fm.init("https://fm.example.com", "u", "p")


def test_f133_simplify_types_does_not_mutate_its_argument():
    from pyutilz.core.filemaker import simplify_types

    obj = {"a": [1, 2], "b": None, "c": 3}
    out = simplify_types(obj)
    assert obj == {"a": [1, 2], "b": None, "c": 3}
    assert out == {"a": "1,2", "c": 3}


# ---------------------------------------------------------------------------
# F55 / F113 / F145 / F146 -- dashlib
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# F56 / F109 -- benchmarking
# ---------------------------------------------------------------------------


def test_f56_fractional_axis_values_are_not_truncated():
    """int() collapsed density 0.25 and 0.5 to 0, so the kernel_tuning_cache matcher could not tell
    the measured cells apart and every real density fell to the catch-all entry."""
    from pyutilz.dev.benchmarking import _preserve_axis_value

    assert _preserve_axis_value(0.25) == 0.25
    assert _preserve_axis_value(4.0) == 4
    assert isinstance(_preserve_axis_value(4.0), int)


def test_f109_zero_reps_raises_instead_of_returning_the_sentinel():
    """The min-tracking sentinel (1e20) was returned as a real timing when the repetition loop never
    ran, and a dispatcher persisted both backends at 1e20 s."""
    from pyutilz.dev.benchmarking import benchmark_algos_by_runtime

    with pytest.raises(ValueError):
        benchmark_algos_by_runtime([lambda: None], n_reps=0)


# ---------------------------------------------------------------------------
# F57 / F144 -- logginglib
# ---------------------------------------------------------------------------


def test_f57_a_failing_finalizer_does_not_replace_the_real_exception(monkeypatch):
    """An exception raised from a `finally` SUPERSEDES whatever was propagating, so a transient DB
    outage hid every real error behind "DB DOWN"."""
    from pyutilz.dev import logginglib

    def boom(*args, **kwargs):
        raise RuntimeError("DB DOWN")

    monkeypatch.setattr(logginglib, "finalize_function_log", boom)

    @logginglib.logged()
    def failing(results_log=None):
        raise ValueError("REAL ERROR")

    with pytest.raises(ValueError, match="REAL ERROR"):
        failing()


def test_f57_a_failing_finalizer_does_not_discard_the_return_value(monkeypatch):
    from pyutilz.dev import logginglib

    def boom(*args, **kwargs):
        raise RuntimeError("DB DOWN")

    monkeypatch.setattr(logginglib, "finalize_function_log", boom)

    @logginglib.logged()
    def ok(results_log=None):
        return 42

    assert ok() == 42


def test_f144_external_ip_is_actually_resolved(monkeypatch):
    """EXTERNAL_IP was declared and read but never assigned, so include_node_ip=True (the default)
    recorded {"ip": None} in every log row ever written."""
    from pyutilz.dev import logginglib
    from pyutilz.web import web as web_pkg  # noqa: F401 -- import surface check only

    monkeypatch.setattr(logginglib, "EXTERNAL_IP", None)
    monkeypatch.setattr(logginglib, "_EXTERNAL_IP_RESOLVED", False)
    monkeypatch.setattr("pyutilz.web.get_external_ip", lambda *a, **kw: "203.0.113.7")
    assert logginglib.get_node_external_ip() == "203.0.113.7"


# ---------------------------------------------------------------------------
# F106 / F129 -- resilience
# ---------------------------------------------------------------------------


def test_f106_get_recent_zero_returns_nothing():
    """`queue[-0:]` is a WHOLE-list slice, so get_recent(0) returned the entire queue."""
    from pyutilz.system.resilience import DeadLetterQueue

    dlq = DeadLetterQueue()
    for i in range(3):
        dlq.add(operation=f"op{i}", args={}, error=Exception("x"))
    assert dlq.get_recent(0) == []
    assert dlq.get_recent(-1) == []
    assert len(dlq.get_recent(2)) == 2


def test_f129_success_threshold_is_named_for_what_it_does():
    """`half_open_max_calls` never capped admissions -- it is only a consecutive-success count."""
    from pyutilz.system.resilience import CircuitBreaker

    cb = CircuitBreaker("t", half_open_successes_to_close=2)
    assert cb.half_open_successes_to_close == 2
    legacy = CircuitBreaker("t", half_open_max_calls=5)
    assert legacy.half_open_successes_to_close == 5


# ---------------------------------------------------------------------------
# F111 -- safe_repr truncation notice
# ---------------------------------------------------------------------------


def test_f111_truncation_notice_reports_the_real_dropped_count(caplog):
    """The code keeps 2*half characters but reported `len - max_size` as dropped, under-reporting by
    one for an odd max_arg_size."""
    from pyutilz.system.monitoring import log_duration

    @log_duration(threshold=-1, max_arg_size=11)
    def probe(payload):
        return payload

    with caplog.at_level(logging.INFO, logger="pyutilz.system.monitoring"):
        probe("y" * 40)
    # repr() is 42 chars; half = 11 // 2 = 5, so 10 are KEPT and 32 dropped -- `len - max_arg_size`
    # reported 31.
    assert "truncated 32 chars" in caplog.text, caplog.text


# ---------------------------------------------------------------------------
# F117 -- ensure_installed on a dotted name
# ---------------------------------------------------------------------------


def test_f117_a_dotted_missing_name_does_not_abort_the_whole_call(monkeypatch):
    """find_spec raises ModuleNotFoundError for a dotted name whose parent is missing, outside the
    try -- so NONE of the earlier packages installed, against the documented "logged, not raised"."""
    from pyutilz.core.pythonlib import packages as pkg

    attempted = []
    monkeypatch.setattr(pkg.subprocess, "check_call", lambda argv, *a, **kw: attempted.append(argv[-1]))
    pkg.ensure_installed(["nosuch_top_level_pkg", "nosuch.child"])
    assert "nosuch.child" in attempted


# ---------------------------------------------------------------------------
# F118 -- is_cuda_available's handler
# ---------------------------------------------------------------------------


def test_f118_non_import_probe_failures_are_warned_not_hidden(monkeypatch, caplog):
    """`except (ImportError, Exception)` read narrower than it was, and the result is memoized for
    the process -- one transient probe failure pinned False, logged at DEBUG only."""
    from pyutilz.core.pythonlib import hardware

    hardware.is_cuda_available.cache_clear()
    monkeypatch.setattr(hardware, "_ensure_cuda_home_from_pip", lambda: (_ for _ in ()).throw(RuntimeError("driver hiccup")))
    try:
        with caplog.at_level(logging.WARNING):
            assert hardware.is_cuda_available() is False
        assert "driver hiccup" in caplog.text
    finally:
        hardware.is_cuda_available.cache_clear()


# ---------------------------------------------------------------------------
# F123 -- ensure_idle_devices bounded wait
# ---------------------------------------------------------------------------


def test_f123_wait_is_bounded_when_max_wait_seconds_is_given(monkeypatch):
    """`while True` with no bound and DEBUG-only messages made a permanently busy device look like
    a hang."""
    from pyutilz.system.system import misc

    import psutil

    monkeypatch.setattr(psutil, "cpu_percent", lambda percpu=False: 100.0)
    assert misc.ensure_idle_devices(duration_seconds=1, max_wait_seconds=0) is False


# ---------------------------------------------------------------------------
# F131 -- occupancy when shared memory does not bind
# ---------------------------------------------------------------------------


def test_f131_shared_memory_limit_is_only_required_when_it_binds():
    """With bytes_per_thread == 0 shared memory cannot bind, yet a device reporting
    max_shared_mem_per_sm: 0 bailed to a single warp instead of the width the other limits allow."""
    from pyutilz.system.gpu_dispatch import occupancy_aware_block_size

    caps = {
        "warp_size": 32,
        "max_shared_mem_per_sm": 0,
        "max_shared_mem_per_block": 0,
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 16,
        "max_threads_per_block": 1024,
        "reserved_shared_mem_per_block": 0,
    }
    width, shared = occupancy_aware_block_size(0, caps=caps)
    assert width == 1024
    assert shared == 0


# ---------------------------------------------------------------------------
# F132 -- load_object_params_into_func returns what it collected
# ---------------------------------------------------------------------------


def test_f132_collected_params_are_returned():
    """Writes to an optimized frame's locals() snapshot never reach fast locals, in every CPython
    version, so the target parameter stayed None with no signal that nothing happened."""
    from pyutilz.core.pythonlib import load_object_params_into_func, store_params_in_object

    class _Holder:
        pass

    holder = _Holder()
    store_params_in_object(holder, {"alpha": 3, "beta": "x"})
    assert load_object_params_into_func(holder, {}) == {"alpha": 3, "beta": "x"}
    assert load_object_params_into_func(None, {}) == {}


# ---------------------------------------------------------------------------
# F134 / F141 -- image
# ---------------------------------------------------------------------------


def test_f134_skip_unknown_exif_tags_is_the_new_name(tmp_path: Path):
    """The flag never had anything to do with EMPTY exif; turning it off injects raw integer keys
    and undecoded bytes into a dict the function promises is JSON-serializable."""
    import inspect

    from pyutilz.core.image import get_image_properties

    params = inspect.signature(get_image_properties).parameters
    assert "skip_unknown_exif_tags" in params
    assert "skip_empty_exif" in params  # deprecated alias kept


def test_f141_supplied_filesize_is_honoured_and_zero_is_kept(tmp_path: Path):
    """A caller-supplied filesize was silently overwritten whenever img was a path, and `if filesize:`
    dropped a genuine 0."""
    from PIL import Image

    from pyutilz.core.image import get_image_properties

    path = tmp_path / "i.png"
    Image.new("RGB", (2, 2)).save(path)
    _, props = get_image_properties(str(path), filesize=123)
    assert props["filesize"] == 123
    _, props0 = get_image_properties(str(path), filesize=0)
    assert props0["filesize"] == 0


# ---------------------------------------------------------------------------
# F143 -- show_biggest_session_objects with an unavailable reading
# ---------------------------------------------------------------------------


def test_f143_none_memory_reading_is_not_percent_formatted(monkeypatch, caplog):
    """get_own_memory_usage() documents returning None when the probe fails; "%.2f" then raised
    TypeError out of a diagnostic helper."""
    from pyutilz.system.system import memory

    monkeypatch.setattr(memory, "get_own_memory_usage", lambda *a, **kw: None)
    with caplog.at_level(logging.INFO, logger=memory.logger.name):
        memory.show_biggest_session_objects({"a": 1})

    messages = [r.getMessage() for r in caplog.records]
    assert any("Own process RAM usage: unavailable" in m for m in messages), messages
    assert not any("None" in m for m in messages), messages


# ---------------------------------------------------------------------------
# F137 -- setup_polars_config reports honestly
# ---------------------------------------------------------------------------


def test_f137_late_polars_config_is_reported_as_ineffective(capsys):
    """polars reads POLARS_MAX_THREADS once, at import; setting it afterwards printed
    "Using N polars threads" while polars kept its default pool."""
    import polars  # noqa: F401 -- the point of the test is that polars IS imported

    from pyutilz.dev.notebook_init import setup_polars_config

    setup_polars_config()
    assert "will NOT apply" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# F121 -- IPython magics via run_line_magic
# ---------------------------------------------------------------------------


def test_f121_magics_go_through_run_line_magic():
    """InteractiveShell.magic() was REMOVED in IPython 9.0 and each call site only appended to a
    `failed` list, so init_notebook still printed success with autoreload silently off."""
    from pyutilz.dev.notebook_init import _run_magic

    calls = []

    class _Shell:
        def run_line_magic(self, name, rest):
            calls.append((name, rest))

        def magic(self, line):  # pragma: no cover - must not be reached
            raise AssertionError("magic() is removed in IPython 9.0")

    _run_magic(_Shell(), "load_ext autoreload")
    assert calls == [("load_ext", "autoreload")]


# ---------------------------------------------------------------------------
# F148 -- doctest collection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "pyutilz.core.pythonlib.numerics",
        "pyutilz.core.pythonlib.objects",
        "pyutilz.core.pythonlib.datetimes",
        "pyutilz.core.pythonlib.filesystem",
        "pyutilz.system.parallel",
        "pyutilz.system.system.misc",
    ],
)
def test_f148_module_doctests_collect_and_pass(module_name):
    """A `>>>foo(...)` with no space is a doctest COLLECTION error that aborts the whole module, so
    every valid doctest in it -- including the ones the 2026-09-02 F34 added -- never ran."""
    import doctest
    import importlib

    results = doctest.testmod(importlib.import_module(module_name), verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
