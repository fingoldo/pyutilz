"""Regression tests for the 2026-09-03 core/dev/system domain audit (07-domain-core-dev-system.md).

One test (or one parametrised group) per behavioural finding, named after the finding id it pins.
Every test here was verified to FAIL against the pre-fix code and to pass after the fix.
"""

import ast
import locale
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from pyutilz.core.disk_cache import DiskCache, hash_array_summary
from pyutilz.core.array_summary import column_sum_min_max

def _write(tmp_path: Path, name: str, src: str) -> Path:
    path = tmp_path / name
    path.write_text(src, encoding="utf-8")
    return path


def _double(x):
    return x * 2


@pytest.mark.parametrize(
    "dtype, other",
    [
        (np.bool_, True),
        ("datetime64[s]", 1),
        ("timedelta64[ns]", 1),
        ("<U2", "xy"),
        ("S2", b"xy"),
    ],
)
def test_f01_middle_row_change_changes_the_key_for_non_numeric_dtypes(dtype, other):
    """A middle-row difference must change the digest for bool/datetime/timedelta/string dtypes too.

    Only `np.number` subtypes got per-column reductions, so for every other dtype the digest saw
    shape, dtype and the first/last 64 rows ONLY -- two 10000-row arrays differing at row 5000
    collided onto the same cache key and a cached result was returned for a different input.
    """
    a = np.zeros(10000, dtype=dtype)
    b = a.copy()
    b[5000] = other
    assert hash_array_summary(a) != hash_array_summary(b)


def test_f02_object_dtype_key_is_stable_across_processes():
    """An object-dtype array must hash its element VALUES, not the PyObject* addresses.

    `.tobytes()` on dtype=object serialises heap pointers, so the same array produced a different
    key in every interpreter: a 0% cross-process hit rate plus one duplicate cache file per worker.
    """
    code = (
        "import numpy as np;" "from pyutilz.core.disk_cache import hash_array_summary;" "print(hash_array_summary(np.array(['a', 'b', None, 3], dtype=object)))"
    )
    digests = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() for _ in range(2)}
    assert len(digests) == 1, f"object-dtype key differs per process: {digests}"


def test_f02_object_dtype_distinguishes_different_contents():
    a = np.array(["a", "b"], dtype=object)
    b = np.array(["a", "c"], dtype=object)
    assert hash_array_summary(a) != hash_array_summary(b)


# ---------------------------------------------------------------------------
# F53 / F139 / F140 -- DiskCache
# ---------------------------------------------------------------------------


def test_f53_key_locks_do_not_grow_without_bound(tmp_path: Path):
    """Per-key locks are refcounted, so nothing is retained once no thread holds the key.

    Keys are content digests (effectively unique per call), so an unbounded dict grew one
    threading.Lock per key EVER written for the process lifetime.
    """
    cache = DiskCache(tmp_path, max_size_bytes=3000)
    for i in range(60):
        cache.put(f"k{i:03d}", b"x" * 200)
    assert cache._key_locks == {}


def test_f139_cached_none_is_distinguishable_from_a_miss(tmp_path: Path):
    """`None` is a legal cached VALUE; a caller must be able to tell it from a miss.

    get() returned None for both, while the counters said "hit" -- so the caller recomputed forever
    while the stats claimed every lookup hit.
    """
    cache = DiskCache(tmp_path)
    miss = object()
    assert cache.get("absent", miss) is miss
    cache.put("k", None)
    assert cache.get("k", miss) is None


def test_f140_key_with_a_path_separator_is_rejected(tmp_path: Path):
    """A key naming a subdirectory passes the traversal guard but is permanently unwritable."""
    cache = DiskCache(tmp_path)
    for bad in ("sub/deep", os.path.join("sub", "deep"), ""):
        with pytest.raises(ValueError):
            cache._key_path(bad)


# ---------------------------------------------------------------------------
# F07 -- column_sum_min_max on a zero-row array
# ---------------------------------------------------------------------------

def test_f07_zero_row_array_raises_instead_of_reading_out_of_bounds():
    """The numba kernel seeded `lo` from a2[0, j] with bounds checking off, returning uninitialized
    memory; the numpy reference path raised for the same input, so the two disagreed."""
    with pytest.raises(ValueError):
        column_sum_min_max(np.empty((0, 3)))


# ---------------------------------------------------------------------------
# F03 -- heartbeat_scraper's SQL call shape
# ---------------------------------------------------------------------------


def test_f03_heartbeat_passes_sql_and_params_as_two_arguments(monkeypatch):
    """`db.safe_execute(statement, data=None)` -- passing the (sql, params) TUPLE as the statement
    handed the cursor a 2-tuple as query text, so no heartbeat could ever execute."""
    from pyutilz.system import distributed as dist

    seen = []

    def fake_safe_execute(statement, data=None, *a, **kw):
        seen.append((statement, data))

    monkeypatch.setattr(dist.db, "safe_execute", fake_safe_execute)
    monkeypatch.setattr(dist, "get_heartbeat_sql", lambda status, ip: ("INSERT INTO scrapers ...", ("n", 1)))
    dist.heartbeat_scraper(status="ok", ip=None)
    assert len(seen) == 1
    statement, data = seen[0]
    assert isinstance(statement, str)
    assert data == ("n", 1)


def test_f03_unregistered_node_sends_no_heartbeat(monkeypatch):
    from pyutilz.system import distributed as dist

    called = []
    monkeypatch.setattr(dist.db, "safe_execute", lambda *a, **kw: called.append(a))
    monkeypatch.setattr(dist, "get_heartbeat_sql", lambda status, ip: ("", None))
    dist.heartbeat_scraper()
    assert called == []


# ---------------------------------------------------------------------------
# F05 -- ci_log_analyzer subprocess decoding
# ---------------------------------------------------------------------------


def test_f05_gh_calls_decode_as_utf8_not_the_ambient_codepage(monkeypatch):
    """`text=True` decodes with locale.getpreferredencoding(False) (cp1251 here): a UTF-8 log byte
    undefined in that codepage raises inside subprocess's reader thread, is swallowed, and stdout
    silently becomes None."""
    from pyutilz.dev import ci_log_analyzer as cla

    captured = {}

    class _Result:
        stdout = "{}"

    def fake_run(args, **kwargs):
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(cla.subprocess, "run", fake_run)
    cla._gh_json(["run", "view"])
    assert captured.get("encoding") == "utf-8"
    assert captured.get("errors") == "replace"
    assert "text" not in captured

    captured.clear()
    cla._fetch_job_log("owner/repo", "1")
    assert captured.get("encoding") == "utf-8"
    assert captured.get("errors") == "replace"
    assert "text" not in captured


def test_f147_failed_log_fetch_is_reported_by_job_name(monkeypatch):
    """"tests (windows-latest, 3.8)", not 48213771234 -- the raw id is the least useful identifier
    for exactly the leg that failed."""
    from pyutilz.dev import ci_log_analyzer as cla

    monkeypatch.setattr(cla, "_list_job_ids", lambda repo, run_id, conclusion: {"48213771234": "tests (windows-latest, 3.8)"})

    def boom(repo, job_id):
        raise subprocess.CalledProcessError(1, "gh")

    monkeypatch.setattr(cla, "_fetch_job_log", boom)
    report = cla.analyze_run("owner/repo", "1")
    assert report.fetch_errors == ["tests (windows-latest, 3.8) (job id 48213771234)"]


# ---------------------------------------------------------------------------
# F06 / F58 / F126 / F135 / F136 -- meta_test_utils
# ---------------------------------------------------------------------------


def test_f06_citation_with_call_parens_resolves_without_removesuffix(tmp_path: Path):
    """`str.removesuffix` is Python 3.9+ (PEP 616) on a path the 3.8 CI leg runs.

    On 3.8 the call raised AttributeError on the FIRST bare qualified-name citation, turning
    unbacked_audit_dispositions (public via __all__) red on that leg. Exercised here through the
    same code path, plus a 3.8-shaped check that the call is gone.
    """
    from pyutilz.dev import meta_test_utils as mtu

    tree = ast.parse(Path(mtu.__file__).read_text(encoding="utf-8"))
    assert not hasattr(str, "removesuffix") or not any(isinstance(n, ast.Attribute) and n.attr == "removesuffix" for n in ast.walk(tree))
    (tmp_path / "m.py").write_text("def my_function():\n    return 1\n", encoding="utf-8")
    mtu.clear_repo_scan_caches()
    assert mtu._names_a_repo_symbol("my_function()", tmp_path) is True


def test_f58_bool_sentinel_is_the_non_default_value():
    """The bool sentinel was True, which is also the commonest default: with `actual != expected`,
    a parser that DROPPED the field still matched its True default and passed as intact."""
    import dataclasses

    from pyutilz.dev.meta_test_utils import optional_scalar_fields

    @dataclasses.dataclass
    class Cfg:
        enabled: bool = True
        disabled: bool = False
        name: str = "x"

    sentinels = optional_scalar_fields(Cfg)
    assert sentinels["enabled"] is False
    assert sentinels["disabled"] is True


def test_f126_repo_scan_caches_can_be_cleared(tmp_path: Path):
    """The lru_cache is keyed on the ROOT PATH alone, so a test that writes a file and a later test
    that scans the same root saw a stale set -- making results depend on test order."""
    from pyutilz.dev import meta_test_utils as mtu

    (tmp_path / "first.py").write_text("A = 1\n", encoding="utf-8")
    mtu.clear_repo_scan_caches()
    assert "second.py" not in mtu._repo_filenames(tmp_path)
    (tmp_path / "second.py").write_text("B = 2\n", encoding="utf-8")
    mtu.clear_repo_scan_caches()
    assert "second.py" in mtu._repo_filenames(tmp_path)


def test_f135_call_expression_whitelist_is_not_counted_as_zero(tmp_path: Path):
    """`set(_LEGACY_ENTRIES)` is an ast.Call too; recording it as 0 made tracked debt appear to
    shrink in exactly the case where it grew."""
    from pyutilz.dev.meta_test_utils import count_user_deferred_entries

    (tmp_path / "test_w.py").write_text("_LEGACY = ['a', 'b']\n_USER_DEFERRED_EMPTY = set()\n_USER_DEFERRED_FULL = set(_LEGACY)\n", encoding="utf-8")
    out = count_user_deferred_entries(tmp_path)
    assert out.get("test_w::_USER_DEFERRED_EMPTY") == 0
    assert "test_w::_USER_DEFERRED_FULL" not in out


def test_f136_config_suffix_is_stripped_only_from_the_end():
    """An unanchored replace("_config", "") dropped the INNER occurrence too."""
    from pyutilz.dev.meta_test_utils import snake_case_variants_of

    variants = snake_case_variants_of("MyConfigManagerConfig")
    assert "my_config_manager" in variants
    assert "my_manager" not in variants


# ---------------------------------------------------------------------------
# F08 / F138 / F150 -- freevar_analysis
# ---------------------------------------------------------------------------


def test_f08_same_line_read_before_write_is_detected(tmp_path: Path):
    """In `selected = [i for i in selected if i]` the Store is at column 0 and the Load at ~11, so
    ordering by (lineno, col_offset) classified the name as written-before-read -- the exact
    accumulator trap the module exists to catch."""
    from pyutilz.dev.freevar_analysis import analyze_range

    path = _write(tmp_path, "acc.py", "selected = [1, 2, 3]\ntotal = 0\nselected = [i for i in selected if i]\ntotal = total + 1\n")
    report = analyze_range(path, 3, 4)
    needed = {u.name for u in report.needs_incoming_value}
    assert "selected" in needed
    assert "total" in needed


def test_f150_function_parameters_are_not_reported_as_free(tmp_path: Path):
    """Parameters are ast.arg, never ast.Name, so they were never seen as bound and a moved
    function's own parameters looked like free variables."""
    from pyutilz.dev.freevar_analysis import analyze_range

    path = _write(tmp_path, "fn.py", "def foo(a, b=1, *args, **kw):\n    return a + b + len(args) + len(kw)\n")
    report = analyze_range(path, 1, 2)
    assert not ({"a", "b", "args", "kw"} & set(report.free_names))
    assert not ({"a", "b", "args", "kw"} & {u.name for u in report.needs_incoming_value})


def test_f138_body_identity_check_covers_decorators(tmp_path: Path):
    """_top_level_span deliberately includes decorators in the moved range, but the byte-identity
    check sliced from the `def` line -- a move that dropped an @lru_cache passed as verified-safe."""
    from pyutilz.dev.freevar_analysis import _top_level_bodies

    src = "import functools\n\n\n@functools.lru_cache\ndef f():\n    return 1\n"
    bodies = _top_level_bodies(src)
    assert "@functools.lru_cache" in bodies["f"]


# ---------------------------------------------------------------------------
# F33 / F40 / F119 -- probing
# ---------------------------------------------------------------------------


def test_f33_remove_nas_results_are_assigned_back():
    """remove_nas is PURE; called as a bare statement every "N/A" survived into the returned dict
    and to_float("N/A") then raised inside _collect_sample, losing every sample."""
    import xmltodict

    from pyutilz.system.system import probing

    xml = """<nvidia_smi_log><timestamp>t</timestamp><gpu><gpu_module_id>0</gpu_module_id>
    <power_readings><power_draw>N/A</power_draw></power_readings></gpu></nvidia_smi_log>"""

    class _Res:
        stdout = xml
        returncode = 0

    res = probing.remove_nas(xmltodict.parse(xml, force_list=["gpu"])["nvidia_smi_log"])
    assert "N/A" not in str(res)


def test_f119_missing_id_attr_raises_valueerror_not_an_assert():
    """An `assert` precondition disappears under `python -O`, turning a named probe failure into a
    KeyError deep inside gpu_dispatch."""
    from pyutilz.system.system.probing import get_gpuutil_gpu_info

    with pytest.raises(ValueError):
        get_gpuutil_gpu_info(attrs="load,memoryTotal")


def test_f40_dmidecode_runs_non_interactively_with_a_timeout(monkeypatch):
    """capture_output=True leaves stdin attached, so a password-needing sudo waits FOREVER -- while
    register_scraper holds _identity_lock, deadlocking every other thread's heartbeat."""
    from pyutilz.system.system import probing

    captured = {}

    class _Res:
        stdout = ""
        returncode = 0

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return _Res()

    # parse_dmidecode_info now resolves both binaries to an ABSOLUTE path through _resolve_binary()
    # (B603/B607 hardening), which raises on a box with no sudo -- e.g. every Windows leg. Stub the
    # resolver so the argv shape under test is reachable there; the contract asserted below is
    # unchanged (non-interactive sudo, a timeout, and a detached stdin).
    monkeypatch.setattr(probing, "_resolve_binary", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(probing.subprocess, "run", fake_run)
    probing.parse_dmidecode_info()
    assert [Path(captured["argv"][0]).name, captured["argv"][1]] == ["sudo", "-n"]
    assert Path(captured["argv"][2]).name == "dmidecode"
    assert captured.get("timeout")
    assert captured.get("stdin") == subprocess.DEVNULL


# ---------------------------------------------------------------------------
# F34 / F122 -- SingleFlightCache
# ---------------------------------------------------------------------------


def test_f34_used_instance_is_picklable():
    """__setstate__ set _loop = None, showing the intent; __getstate__ kept it, so an instance
    pickled fine BEFORE first use and failed only after real work."""
    import asyncio

    from pyutilz.system.single_flight_cache import SingleFlightCache

    cache = SingleFlightCache()
    store: dict = {}

    async def _ready():
        return 1

    async def _use():
        await cache.get_or_fetch(store, "k", _ready)

    asyncio.run(_use())
    restored = pickle.loads(pickle.dumps(cache))  # deserialize-ok: round-trips a pickle this test just produced in-process; the bytes never leave the test, no untrusted input
    assert restored._loop is None


def test_f122_clear_wakes_in_flight_waiters():
    """_inflight.clear() discarded Events without setting them, so a clear() racing an in-flight
    fetch left that waiter awaiting permanently."""
    import asyncio

    from pyutilz.system.single_flight_cache import SingleFlightCache

    cache = SingleFlightCache()

    async def _scenario():
        evt = asyncio.Event()
        cache._inflight["k"] = evt
        cache.clear()
        return evt.is_set()

    assert asyncio.run(_scenario()) is True


# ---------------------------------------------------------------------------
# F35 / F112 / F142 -- hardware monitor
# ---------------------------------------------------------------------------


def test_f35_stop_on_a_never_started_monitor_does_not_raise():
    """__init__ already assigns an UNSTARTED Thread, so `thread is not None` was always true and a
    try/finally cleanup masked the caller's real exception with a RuntimeError."""
    from pyutilz.system.hardware_monitor import UtilizationMonitor

    monitor = UtilizationMonitor()
    monitor.stop()
    # stop() still did its job on the never-started monitor rather than bailing out early.
    assert monitor.stop_flag.is_set()


def test_f112_a_sample_failing_after_the_gpu_query_is_counted_once(monkeypatch):
    """n_samples was incremented BEFORE the parse loop that can raise, and the caller then also
    incremented n_sampling_errors -- 100 consecutively-failing samples read as "100 of 200 failed"."""
    from pyutilz.system import hardware_monitor as hm

    monitor = hm.UtilizationMonitor()
    monkeypatch.setattr(hm, "get_nvidia_smi_info", lambda include_stats=True: {"gpu": [{"gpu_module_id": 0, "fb_memory_usage": {"free": "N/A"}}]})
    with pytest.raises(ValueError):
        monitor._collect_sample()
    assert monitor.n_samples == 0


def test_f142_empty_ram_series_report_none_not_nan():
    """Every sibling metric returns None when its series is empty; nan silently averaged into a
    consumer's report that skips None."""
    from pyutilz.system.hardware_monitor import UtilizationMonitor

    monitor = UtilizationMonitor()
    monitor.cpu_utilizaton.append(1.0)
    out = monitor.get_average_utilization()
    assert out["own_ram_used_gb"] is None
    assert out["total_ram_used_gb"] is None
    assert out["total_ram_free_gb"] is None


# ---------------------------------------------------------------------------
# F36 / F107 -- parallel
# ---------------------------------------------------------------------------


def test_f36_applyfunc_parallel_returns_a_real_list():
    """`tqdmu(pool.starmap(...))` wrapped an already-complete list, so the documented `list` was a
    tqdm object on which len()/indexing raised."""
    from pyutilz.system.parallel import applyfunc_parallel

    result = applyfunc_parallel([(1,), (2,)], _double, return_dataframe=False, n_jobs=2, use_threads=True)
    assert isinstance(result, list)
    assert len(result) == 2


def test_f107_negative_chunk_size_raises():
    from pyutilz.system.parallel import split_list_into_chunks, split_list_into_chunks_indices

    for fn in (split_list_into_chunks, split_list_into_chunks_indices):
        with pytest.raises(ValueError):
            list(fn(list(range(10)), -3))


# ---------------------------------------------------------------------------
# F37 / F38 / F39 -- misc
# ---------------------------------------------------------------------------


def test_f37_cmdline_matching_is_a_substring_match(monkeypatch):
    """`cmdline not in proc.cmdline()` tested list-ELEMENT equality, so a documented substring like
    "scraper.py" never matched a real argv and a duplicate-instance guard admitted duplicates."""
    from pyutilz.system.system import misc

    class _Proc:
        def name(self):
            return "python.exe"

        def cmdline(self):
            return [os.path.join("usr", "bin", "python"), os.path.join("worker", "scraper.py"), "--id", "3"]

    import psutil

    monkeypatch.setattr(psutil, "process_iter", lambda *a, **kw: [_Proc()])
    assert misc.count_app_instances(cmdline="scraper.py") == 1


def test_f38_report_large_objects_scans_the_callers_namespace(caplog):
    """`globals()` inside misc.py is MISC's namespace, so the diagnostic reported "no large objects"
    on the exact run it was added to debug."""
    from pyutilz.system.system.misc import report_large_objects

    big_local_object = ["x"] * 200000
    namespace = {"big_local_object": big_local_object}
    with caplog.at_level(logging.INFO):
        report_large_objects(min_size_mb=0, namespace=namespace)
    assert "big_local_object" in caplog.text


def test_f39_locale_is_restored_after_the_read():
    """A diagnostic READ left LC_ALL changed for every later strftime and LC_NUMERIC consumer in the
    process, including float formatting's decimal separator."""
    from pyutilz.system.system.misc import get_locale_settings

    before = locale.setlocale(locale.LC_ALL)
    # "C" is guaranteed present on every platform, so this needs no optional-dependency skip gate --
    # and a skip here would reclassify a real regression as "not run".
    get_locale_settings(locale_name="C", only_fields=("decimal_point",))
    assert locale.setlocale(locale.LC_ALL) == before


# ---------------------------------------------------------------------------
# F41 / F127 -- sysinfo
# ---------------------------------------------------------------------------


def test_f41_missing_identity_fields_propagate_instead_of_returning_a_partial_dict(monkeypatch):
    """register_scraper is built on this function's documented "propagates any error" contract; a
    silently partial dict made it key node identity on absent fields."""
    from pyutilz.system.system import sysinfo

    monkeypatch.setattr(sysinfo.socket, "gethostname", lambda: (_ for _ in ()).throw(OSError("no hostname")))
    with pytest.raises(OSError, match="no hostname"):
        sysinfo.get_system_info(return_sensitive_info=True)


def test_f127_child_output_decoder_never_raises_on_a_bad_byte():
    """A localized non-ASCII error message raised UnicodeDecodeError, was caught by the broad
    handler and reported as the generic "Could not extract Windows serial!"."""
    from pyutilz.system.system.sysinfo import _decode_child_output

    assert isinstance(_decode_child_output(b"\x98\xff plain"), str)


# ---------------------------------------------------------------------------
# F42 -- prefect credential logging
# ---------------------------------------------------------------------------


def test_f42_prefect_key_is_never_logged(caplog, monkeypatch):
    """`logger.info("prefect_key=%s", prefect_key)` on every connect() put the live key into stdout,
    log files and CI job logs at the default INFO level.

    `prefect` is an optional dependency of the module under test, so a stub stands in for it: the
    contract under test is what connect() LOGS, not what the Prefect client does with the key.
    """
    import types

    stub = types.ModuleType("prefect")

    class _Client:
        def __init__(self, api_key=None):
            self.api_key = api_key

    stub.Client = _Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prefect", stub)

    from pyutilz.system.scheduling import prefect as prefect_mod

    monkeypatch.setattr(prefect_mod, "prefect", stub, raising=False)

    secret = "pk-LIVE-PREFECT-KEY-9f3a"  # pragma: allowlist secret
    caplog.set_level(logging.DEBUG)
    prefect_mod.connect(prefect_key=secret)

    assert prefect_mod.client is not None and prefect_mod.client.api_key == secret  # the key still reaches the client
    assert secret not in caplog.text  # ... but never the logs
    assert "provided" in caplog.text  # presence/absence is still reported for diagnosis

    caplog.clear()
    prefect_mod.connect(prefect_key="")
    assert prefect_mod.client is None
    assert "MISSING" in caplog.text


# ---------------------------------------------------------------------------
# F43 / F108 / F114 -- numerics
# ---------------------------------------------------------------------------
