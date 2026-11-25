"""``hw_fingerprint`` distinguishes a failed or opted-out GPU probe from a genuinely GPU-less host and keys the hardware it will really use."""
from __future__ import annotations

from pyutilz.core import jsonlib
import sys
import types
from typing import Any, Callable, Dict

import pytest

from pyutilz.performance.kernel_tuning import cache as ktc
from pyutilz.performance.kernel_tuning.cache import cache_base as cb

_SUMMARY = {"name": "TestCard 9000", "cc_major": 8, "cc_minor": 9}
_PROV = {"cuda_driver_version": 12040, "cuda_runtime_version": 12020, "gpu_summary": {"total_vram_gb": 8192.0, "total_cuda_cores": 4608, "clock_khz": 2_520_000}}


class _Env:
    """Mutable description of the fake machine plus helpers to resolve its fingerprint afresh."""

    def __init__(self, path: Any) -> None:
        """Start from a healthy one-GPU host with 4 numba threads."""
        self.path = path
        self.summary: Any = dict(_SUMMARY)
        self.device_count: Any = 1
        self.device_id = 0
        self.threads = 4
        self.prov: Dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _PROV.items()}
        self.summary_calls = 0

    def fp(self) -> str:
        """Fingerprint of a brand-new process on this machine."""
        ktc.hw_fingerprint.cache_clear()
        cb._gpu_summary_cached.cache_clear()
        return ktc.hw_fingerprint()

    def persisted(self) -> Dict[str, Any]:
        """Entries stored on disk (``{}`` when no file was written)."""
        if not self.path.exists():
            return {}
        return jsonlib.loads(self.path.read_text(encoding="utf-8"))["entries"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated fingerprint cache directory and a fake machine behind every probe the fingerprint consults."""
    for var in ("PYUTILZ_HW_FP_REFRESH", "PYUTILZ_DISABLE_GPU", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cb, "_GPU_OPT_OUT_PREDICATES", [])
    state = _Env(tmp_path / cb._HW_FP_DISK_FILENAME)

    def summary_probe(_device_id: int = 0) -> Any:
        """Fake ``gpu_capability_summary``: the configured summary, ``None``, or a raised failure."""
        state.summary_calls += 1
        if isinstance(state.summary, Exception):
            raise state.summary
        return state.summary

    def device_count() -> int:
        """Fake device-count query; a configured exception is raised."""
        if isinstance(state.device_count, Exception):
            raise state.device_count
        return int(state.device_count)

    monkeypatch.setattr(ktc, "_cpu_model_slug", lambda: "testcpu")
    monkeypatch.setattr(cb, "_cpu_identity_suffix", lambda: "p4l8_f28")
    monkeypatch.setattr(ktc, "gpu_capability_summary", summary_probe)
    monkeypatch.setattr(cb, "_gpu_device_count", device_count)
    monkeypatch.setattr(cb, "_current_device_id", lambda: state.device_id)
    monkeypatch.setattr(cb, "_numba_threads", lambda: state.threads)
    monkeypatch.setattr(cb, "_build_provenance", lambda: state.prov)
    ktc.hw_fingerprint.cache_clear()
    cb._gpu_summary_cached.cache_clear()
    yield state
    ktc.hw_fingerprint.cache_clear()
    cb._gpu_summary_cached.cache_clear()


def _gpu_part(fp: str) -> str:
    """Fingerprint without its trailing thread component."""
    return fp.rsplit("_t", 1)[0]


def test_gpu_key_carries_cpu_cores_clock_device_cuda_cores_vram_clock_driver_runtime_and_threads(env):
    """The key names the CPU core counts and rated clock, the device index, the CUDA core count, the VRAM in GiB, the rated GPU clock, the driver and runtime versions and the numba thread count."""
    fp = env.fp()
    assert fp.startswith("cpu_testcpu_p4l8_f28_gpu_testcard-9000_cc8.9_d0_cu4608_vram8g_f25_drv12040_rt12020_nb")
    assert fp.endswith("_t4")


def test_only_the_hardware_part_is_persisted_not_the_thread_count(env):
    """The persisted entry is the hardware key; the thread count is appended per process so it never goes stale."""
    fp = env.fp()
    entry = env.persisted()["default"]["fingerprint"]
    assert entry == _gpu_part(fp)
    assert "_t4" not in entry


def test_different_numba_thread_counts_give_different_keys_sharing_one_persisted_hardware_part(env):
    """A tuning measured with 2 threads is not reused at 16, while the expensive hardware probe still runs once."""
    env.threads = 2
    fp_two = env.fp()
    env.summary = RuntimeError("must come from disk")
    env.threads = 16
    fp_sixteen = env.fp()
    assert fp_two != fp_sixteen
    assert fp_two.endswith("_t2") and fp_sixteen.endswith("_t16")
    assert _gpu_part(fp_two) == _gpu_part(fp_sixteen)


def test_device_index_vram_cuda_cores_clock_and_driver_each_change_the_key(env, monkeypatch):
    """Routing to another device, a different VRAM size, core count or rated clock, or a driver upgrade yields a new key (a re-tune)."""
    monkeypatch.setenv("PYUTILZ_HW_FP_REFRESH", "1")
    base = env.fp()
    env.device_id = 1
    assert env.fp() != base
    env.device_id = 0
    env.prov["gpu_summary"]["total_vram_gb"] = 24576.0
    assert env.fp() != base
    env.prov["gpu_summary"]["total_vram_gb"] = 8192.0
    env.prov["gpu_summary"]["total_cuda_cores"] = 3584
    assert "_cu3584_" in env.fp() and env.fp() != base
    env.prov["gpu_summary"]["total_cuda_cores"] = 4608
    env.prov["gpu_summary"]["clock_khz"] = 1_800_000
    assert "_f18_" in env.fp() and env.fp() != base
    env.prov["gpu_summary"]["clock_khz"] = 2_520_000
    env.prov["cuda_driver_version"] = 12050
    env_fp = env.fp()
    assert env_fp != base and "_drv12050_" in env_fp


def test_vram_of_12_and_16_gib_no_longer_share_a_key_and_a_clock_inside_a_bin_does_not_split_it(env, monkeypatch):
    """The power-of-two class lumped 12 and 16 GB into ``vram16g``; the key now rounds to the GiB. The rated clock is rounded to 100 MHz, so 2.52 and 2.54 GHz are one key."""
    monkeypatch.setenv("PYUTILZ_HW_FP_REFRESH", "1")
    env.prov["gpu_summary"]["total_vram_gb"] = 12288.0
    twelve = env.fp()
    env.prov["gpu_summary"]["total_vram_gb"] = 16384.0
    assert twelve != env.fp() and "_vram12g_" in twelve
    env.prov["gpu_summary"]["clock_khz"] = 2_540_000
    assert env.fp().count("_f25_") == 1


def test_identity_helpers_round_and_report_unknown_fields():
    """VRAM to the nearest GiB for MiB and GiB inputs, the clock to 100 MHz, junk and zero to the unknown token."""
    assert cb._vram_gib(8192.0) == "vram8g" and cb._vram_gib(8.0) == "vram8g" and cb._vram_gib(11264.0) == "vram11g" and cb._vram_gib(4095.0) == "vram4g"
    assert cb._vram_gib(None) == "vramx" and cb._vram_gib(0) == "vramx"
    assert cb._rounded_clock(1_620_000) == "f16" and cb._rounded_clock(1_649_000) == "f16" and cb._rounded_clock(1_651_000) == "f17"
    assert cb._rounded_clock(None) == "fx" and cb._rounded_clock(0) == "fx"


def test_cpu_identity_reads_physical_and_logical_cores_and_the_rated_maximum_clock(monkeypatch):
    """``p<physical>l<logical>_f<100 MHz>`` from psutil (the maximum, not the current clock); a missing clock falls back to cpuinfo, then to ``x``."""
    freq = types.SimpleNamespace(current=800.0, max=2808.0)
    fake = types.SimpleNamespace(cpu_count=lambda logical=True: 8 if logical else 4, cpu_freq=lambda: freq)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    cb._cpu_identity_suffix.cache_clear()
    assert cb._cpu_identity_suffix() == "p4l8_f28"
    freq.max = 0.0
    monkeypatch.setitem(sys.modules, "cpuinfo", types.SimpleNamespace(get_cpu_info=lambda: {"hz_advertised": (3_400_000_000, 0)}))
    cb._cpu_identity_suffix.cache_clear()
    assert cb._cpu_identity_suffix() == "p4l8_f34"
    monkeypatch.setitem(sys.modules, "cpuinfo", types.SimpleNamespace(get_cpu_info=lambda: {}))
    cb._cpu_identity_suffix.cache_clear()
    assert cb._cpu_identity_suffix() == "p4l8_fx"
    cb._cpu_identity_suffix.cache_clear()


def test_vram_class_rounds_up_to_a_power_of_two_gib_for_both_mib_and_gib_inputs():
    """GPUtil reports MiB while the summary key is named GB; both scales land in the same class and junk maps to ``vramx``."""
    assert cb._vram_class(8192.0) == "vram8g"
    assert cb._vram_class(8.0) == "vram8g"
    assert cb._vram_class(6144.0) == "vram8g"
    assert cb._vram_class(4096.0) == "vram4g"
    assert cb._vram_class(None) == "vramx"
    assert cb._vram_class(0) == "vramx"


def test_failed_probe_is_keyed_unknown_and_never_persisted(env):
    """A raising probe yields a non-persistent ``gpu-unknown`` key instead of caching ``no-gpu``."""
    env.summary = RuntimeError("driver busy")
    fp = env.fp()
    assert "_gpu-unknown_" in fp and "no-gpu" not in fp
    assert env.persisted() == {}


def test_a_gpu_run_after_a_failed_probe_resolves_the_gpu_key(env):
    """The failure left no trace on disk, so the next process with a working GPU keys the GPU hardware."""
    env.summary = RuntimeError("device busy")
    env.fp()
    env.summary = dict(_SUMMARY)
    fp = env.fp()
    assert "_gpu_testcard-9000_cc8.9_" in fp and "no-gpu" not in fp


def test_summary_none_while_devices_exist_is_unknown_not_no_gpu(env):
    """The capability probe returning nothing on a host the stack says has a device is an unknown state."""
    env.summary = None
    env.device_count = 1
    fp = env.fp()
    assert "_gpu-unknown_" in fp
    assert env.persisted() == {}


def test_device_count_query_failure_is_unknown(env):
    """If even the device count cannot be read the state is unknown and nothing is persisted."""
    env.summary = None
    env.device_count = OSError("cudart missing")
    fp = env.fp()
    assert "_gpu-unknown_" in fp
    assert env.persisted() == {}


def test_genuine_gpu_less_host_persists_no_gpu_and_rereads_it(env):
    """Zero devices reported without error is a definite answer: persisted, then served from disk without probing."""
    env.summary = None
    env.device_count = 0
    fp = env.fp()
    assert "_no-gpu_" in fp
    assert env.persisted()["default"]["fingerprint"] == _gpu_part(fp)
    calls = env.summary_calls
    ktc.hw_fingerprint.cache_clear()
    assert ktc.hw_fingerprint() == fp
    assert env.summary_calls == calls


def test_legacy_schema_no_gpu_file_is_not_read(env):
    """A ``no-gpu`` fingerprint persisted by the old schema (possibly from a failed probe) is ignored and the GPU is probed afresh."""
    env.path.write_text(jsonlib.dumps({"schema_version": 1, "fingerprint": "cpu_testcpu_no-gpu", "ts_utc": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
    fp = env.fp()
    assert "_gpu_testcard-9000_" in fp and "no-gpu" not in fp
    assert env.persisted()["default"]["fingerprint"] == _gpu_part(fp)


@pytest.mark.parametrize("value", ["", "  ", "-1", "NoDevFiles"])
def test_cuda_visible_devices_opt_out_is_keyed_no_gpu_without_probing_or_writing(env, monkeypatch, value):
    """An opted-out run keys the CPU-only hardware in memory and neither probes the GPU nor touches the file."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    fp = env.fp()
    assert "_no-gpu_" in fp and fp.endswith("_t4")
    assert env.summary_calls == 0
    assert env.persisted() == {}


def test_opted_out_run_neither_reads_nor_overwrites_the_gpu_entry(env, monkeypatch):
    """A GPU entry persisted earlier is not served to an opted-out run and survives it unchanged."""
    gpu_fp = env.fp()
    before = env.persisted()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert "_no-gpu_" in env.fp()
    assert env.persisted() == before
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert env.fp() == gpu_fp


def test_pyutilz_disable_gpu_env_is_an_opt_out(env, monkeypatch):
    """``PYUTILZ_DISABLE_GPU=1`` behaves like an empty ``CUDA_VISIBLE_DEVICES``."""
    monkeypatch.setenv("PYUTILZ_DISABLE_GPU", "1")
    assert "_no-gpu_" in env.fp()
    assert env.persisted() == {}


def test_registered_opt_out_predicate_is_honoured_and_a_raising_one_is_ignored(env):
    """A project can declare its own opt-out; a predicate that raises does not opt the run out."""
    flags = {"out": True}

    def project_flag() -> bool:
        """Project-level opt-out switch."""
        return flags["out"]

    def broken() -> bool:
        """Predicate that fails."""
        raise RuntimeError("boom")

    ktc.register_gpu_opt_out(broken)
    ktc.register_gpu_opt_out(project_flag)
    ktc.register_gpu_opt_out(project_flag)
    assert cb._GPU_OPT_OUT_PREDICATES.count(project_flag) == 1
    assert "_no-gpu_" in env.fp()
    assert env.persisted() == {}
    flags["out"] = False
    assert "_gpu_testcard-9000_" in env.fp()


def test_persisted_entries_are_keyed_by_the_visible_device_selector(env, monkeypatch):
    """An entry recorded while device 0 was visible is not reused by a run that sees only device 1."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env.fp()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    env.device_id = 1
    fp = env.fp()
    assert "_d1_" in fp
    assert set(env.persisted()) == {"0", "1"}


def test_entry_recorded_under_other_package_versions_is_discarded(env):
    """A numba / cupy upgrade invalidates the persisted hardware part."""
    env.fp()
    data = jsonlib.loads(env.path.read_text(encoding="utf-8"))
    data["entries"]["default"]["stamp"] = "nb0.0.0_cupy0.0.0"
    data["entries"]["default"]["fingerprint"] = "STALE-SENTINEL"
    env.path.write_text(jsonlib.dumps(data), encoding="utf-8")
    assert "STALE-SENTINEL" not in env.fp()


def test_expired_entry_is_discarded(env):
    """An entry older than the freshness window is re-probed."""
    env.fp()
    data = jsonlib.loads(env.path.read_text(encoding="utf-8"))
    data["entries"]["default"]["ts"] = 1.0
    data["entries"]["default"]["fingerprint"] = "STALE-SENTINEL"
    env.path.write_text(jsonlib.dumps(data), encoding="utf-8")
    assert "STALE-SENTINEL" not in env.fp()


def _fake_cupy(count: Callable[[], int], error_type: type) -> types.ModuleType:
    """Stand-in ``cupy`` module whose ``getDeviceCount`` is ``count``."""
    runtime = types.SimpleNamespace(getDeviceCount=count, CUDARuntimeError=error_type)
    mod = types.ModuleType("cupy")
    mod.cuda = types.SimpleNamespace(runtime=runtime)  # type: ignore[attr-defined]
    return mod


class _FakeRuntimeError(Exception):
    """Mimics ``cupy.cuda.runtime.CUDARuntimeError``."""

    def __init__(self, status: int) -> None:
        """Remember the CUDA status code."""
        super().__init__(status)
        self.status = status


@pytest.mark.parametrize("status, expected", [(100, 0), (35, 0)])
def test_device_count_maps_no_device_statuses_to_zero(monkeypatch, status, expected):
    """cudaErrorNoDevice / cudaErrorInsufficientDriver are a definite "no GPU", not a probe failure."""

    def count() -> int:
        """Raise the configured CUDA status."""
        raise _FakeRuntimeError(status)

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(count, _FakeRuntimeError))
    assert cb._gpu_device_count() == expected


def test_device_count_propagates_other_runtime_errors(monkeypatch):
    """A runtime error that is not a no-device status stays an error so the caller classifies the state as unknown."""

    def count() -> int:
        """Raise an unrelated CUDA status."""
        raise _FakeRuntimeError(2)

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(count, _FakeRuntimeError))
    with pytest.raises(_FakeRuntimeError):
        cb._gpu_device_count()


def test_device_count_reports_devices_when_the_stack_sees_them(monkeypatch):
    """A successful query returns the real count."""
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(lambda: 2, _FakeRuntimeError))
    assert cb._gpu_device_count() == 2


# ---- the entries of the pre-2026-11-25 key are adopted once ------------------------------------------------------------------------------------------------------------------------------------


def _write_legacy_entry(root, legacy_fp: str, rel: str, record: Dict[str, Any]) -> None:
    """One tuning-entry file under the legacy key directory."""
    path = root / legacy_fp / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonlib.dumps(record), encoding="utf-8")


def test_the_legacy_key_is_the_old_composition(env):
    """CPU model string, GPU name and capability, device index, power-of-two VRAM class, driver, runtime, library versions and threads: none of the new components."""
    ktc.legacy_hw_fingerprint.cache_clear()
    legacy = ktc.legacy_hw_fingerprint()
    assert legacy.startswith("cpu_testcpu_gpu_testcard-9000_cc8.9_d0_vram8g_drv12040_rt12020_nb") and legacy.endswith("_t4")
    assert "_p4l8_" not in legacy and "_cu4608_" not in legacy


def test_entries_of_the_legacy_directory_are_adopted_once_with_the_new_fingerprint(env, tmp_path):
    """An existing host keeps its tuning: every ``.json`` entry is copied to the new key directory with the new fingerprint inside, a marker names the source, and a second call copies nothing."""
    ktc.legacy_hw_fingerprint.cache_clear()
    legacy = ktc.legacy_hw_fingerprint()
    new = env.fp()
    assert legacy != new
    record = {"hw_fingerprint": legacy, "kernel_name": "k", "best": "cpu"}
    _write_legacy_entry(tmp_path, legacy, "k-1234/abc.0.1.2.json", record)
    _write_legacy_entry(tmp_path, legacy, "k-1234/abc.0.1.3.json", {**record, "best": "gpu"})
    (tmp_path / legacy / "k-1234" / "notes.txt").write_text("not an entry", encoding="utf-8")
    cb._ADOPTED_DIRS.clear()
    new_dir = ktc.host_cache_dir()
    assert new_dir.endswith(new)
    copied = sorted(p.name for p in (tmp_path / new / "k-1234").iterdir())
    assert copied == ["abc.0.1.2.json", "abc.0.1.3.json"]
    entry = jsonlib.loads((tmp_path / new / "k-1234" / "abc.0.1.3.json").read_text(encoding="utf-8"))
    assert entry["hw_fingerprint"] == new and entry["adopted_from_fingerprint"] == legacy and entry["best"] == "gpu"
    marker = jsonlib.loads((tmp_path / new / cb._ADOPTION_MARKER).read_text(encoding="utf-8"))
    assert marker["adopted_from"] == legacy and marker["files"] == 2
    assert (tmp_path / legacy / "k-1234" / "abc.0.1.2.json").exists()  # nothing is removed from the old directory
    cb._ADOPTED_DIRS.clear()
    (tmp_path / legacy / "k-1234" / "late.0.1.4.json").write_text(jsonlib.dumps(record), encoding="utf-8")
    ktc.host_cache_dir()
    assert not (tmp_path / new / "k-1234" / "late.0.1.4.json").exists()  # the marker stops a second copy


def test_an_entry_already_in_the_new_directory_is_kept_and_a_host_without_a_legacy_directory_adopts_nothing(env, tmp_path):
    """The new directory wins over the copy; with no legacy directory the marker is written and nothing is created."""
    import shutil

    ktc.legacy_hw_fingerprint.cache_clear()
    legacy = ktc.legacy_hw_fingerprint()
    new = env.fp()
    _write_legacy_entry(tmp_path, legacy, "k-1/e.json", {"hw_fingerprint": legacy, "best": "old"})
    _write_legacy_entry(tmp_path, new, "k-1/e.json", {"hw_fingerprint": new, "best": "new"})
    cb._ADOPTED_DIRS.clear()
    ktc.host_cache_dir()
    assert jsonlib.loads((tmp_path / new / "k-1" / "e.json").read_text(encoding="utf-8"))["best"] == "new"
    shutil.rmtree(tmp_path / legacy)
    (tmp_path / new / cb._ADOPTION_MARKER).unlink()
    cb._ADOPTED_DIRS.clear()
    assert cb._adopt_legacy_entries(str(tmp_path / new)) == 0
    assert (tmp_path / new / cb._ADOPTION_MARKER).exists()


def test_a_host_whose_gpu_probe_failed_adopts_nothing(env, tmp_path):
    """The unknown-GPU key is never persisted and never receives the entries of another key."""
    env.summary = RuntimeError("probe failed")
    ktc.legacy_hw_fingerprint.cache_clear()
    fp = env.fp()
    assert "gpu-unknown" in fp
    legacy = ktc.legacy_hw_fingerprint()
    _write_legacy_entry(tmp_path, legacy, "k-1/e.json", {"hw_fingerprint": legacy})
    cb._ADOPTED_DIRS.clear()
    ktc.host_cache_dir()
    assert not (tmp_path / fp / "k-1").exists()
