"""Tests for hardware detection functions migrated from ml_perf_test."""

import pytest
import platform


class TestUtilityFunctions:
    """Test utility helper functions."""

    def test_remove_nas(self):
        """Test remove_nas() recursively removes N/A values."""
        from pyutilz.system.system import remove_nas

        obj = {
            "a": 1,
            "b": "N/A",
            "c": {"d": "N/A", "e": "5.5", "f": "text"},
            "g": ["N/A", 1, "2.3"],
        }
        result = remove_nas(obj)
        assert "b" not in result
        assert "d" not in result["c"]
        assert result["c"]["e"] == 5.5  # Converted to float
        assert result["c"]["f"] == "text"  # String kept as is
        assert "N/A" not in result["g"]
        assert 2.3 in result["g"]  # Converted to float

    def test_dict_to_tuple(self):
        """Test dict_to_tuple() creates sorted tuple."""
        from pyutilz.system.system import dict_to_tuple

        d = {"z": 3, "a": 1, "m": 2}
        result = dict_to_tuple(d)
        assert result == (("a", 1), ("m", 2), ("z", 3))


class TestCPUDetection:
    """Test CPU detection functions."""

    def test_get_cpu_info(self):
        """get_cpu_info() returns a populated, filtered CPU dict.

        The availability precondition (py-cpuinfo installed) is an explicit importorskip; the
        structure is then asserted UNCONDITIONALLY. The previous ``if cpu_info: ... else:
        print("[WARN] ...")`` shape passed green on exactly the regression that matters most --
        the probe returning None/{} because an exception was swallowed inside it.
        """
        pytest.importorskip("cpuinfo")
        from pyutilz.system.system import get_cpu_info

        cpu_info = get_cpu_info()

        assert isinstance(cpu_info, dict)
        assert cpu_info, "get_cpu_info() returned an empty dict -- the probe found nothing"
        assert "arch" in cpu_info, f"expected an 'arch' key, got {sorted(cpu_info)}"
        # Filtered-out keys: version strings, hz variants and count are stripped on purpose.
        for dropped in ("python_version", "cpuinfo_version", "hz_advertised", "hz_actual", "count"):
            assert dropped not in cpu_info, f"{dropped!r} should have been filtered out"
        # `flags`, when present, is flattened from a list to a sorted space-joined string.
        if "flags" in cpu_info:
            assert isinstance(cpu_info["flags"], str)
            flags = cpu_info["flags"].split(" ")
            assert flags == sorted(flags)

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_get_wmi_cpuinfo(self):
        """get_wmi_cpuinfo() returns a summarised, non-empty Win32_Processor list.

        ``wmi`` is the availability precondition (importorskip); everything else is asserted
        unconditionally, so a probe that starts returning None fails instead of printing a
        warning and passing.
        """
        pytest.importorskip("wmi")
        from pyutilz.system.system import get_wmi_cpuinfo

        cpu_info = get_wmi_cpuinfo()

        assert isinstance(cpu_info, list)
        assert len(cpu_info) > 0
        assert "Count" in cpu_info[0]
        assert isinstance(cpu_info[0]["Count"], int) and cpu_info[0]["Count"] >= 1
        assert "Name" in cpu_info[0], f"summarize_devices output shape changed: {sorted(cpu_info[0])}"
        # Excluded per-instance/volatile/identifying properties must not leak into the summary.
        for excluded in ("SerialNumber", "ProcessorId", "LoadPercentage", "DeviceID", "Status", "AssetTag", "Description"):
            assert excluded not in cpu_info[0], f"{excluded!r} should have been excluded by get_wmi_cpuinfo"

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
    def test_get_lscpu_info(self):
        """get_lscpu_info() parses `lscpu` into a non-empty, type-converted dict.

        The precondition is the `lscpu` binary being on PATH (named explicitly in the skip);
        the parse result is then asserted unconditionally rather than warned about.
        """
        import shutil

        if shutil.which("lscpu") is None:
            pytest.skip("the `lscpu` binary is not installed on this host")

        from pyutilz.system.system import get_lscpu_info

        lscpu_info = get_lscpu_info()

        assert isinstance(lscpu_info, dict)
        assert lscpu_info, "lscpu is installed but get_lscpu_info() parsed nothing out of it"
        assert "Architecture" in lscpu_info
        assert isinstance(lscpu_info["Architecture"], str)
        # Numeric fields are converted, not left as strings (the documented behaviour).
        if "CPU(s)" in lscpu_info:
            assert isinstance(lscpu_info["CPU(s)"], (int, float))

    def test_get_nix_cpu_sockets_number_parses_real_output(self, monkeypatch):
        """get_nix_cpu_sockets_number() must parse a well-formed `lscpu` `Socket(s):` line."""
        from pyutilz.system.system.probing import get_nix_cpu_sockets_number

        # lscpu is now resolved to an absolute path via resolve_binary() before being spawned
        # (binary-planting fix), which is a separate concern from the output parsing under test
        # here -- and lscpu does not exist on the Windows box this suite also runs on.
        monkeypatch.setattr("pyutilz.system.system.probing._resolve_binary", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(
            "pyutilz.system.system.probing.subprocess.check_output",
            lambda *a, **kw: b"Architecture:  x86_64\nSocket(s):     2\n",
        )
        assert get_nix_cpu_sockets_number() == 2

    def test_get_nix_cpu_sockets_number_falls_back_to_1_on_unparseable_match(self, monkeypatch):
        """Regression test: when `lscpu` succeeds and the ``Socket(s):`` line matches the
        regex but the captured text isn't int-parseable, the function must still return the
        documented fallback of 1 -- it previously left `num_sockets` overwritten to the raw
        `re.findall()` match list, so a caller doing arithmetic on the "int" got a confusing
        `TypeError` far from the real cause.
        """
        from pyutilz.system.system.probing import get_nix_cpu_sockets_number

        # lscpu is now resolved to an absolute path via resolve_binary() before being spawned
        # (binary-planting fix), which is a separate concern from the output parsing under test
        # here -- and lscpu does not exist on the Windows box this suite also runs on.
        monkeypatch.setattr("pyutilz.system.system.probing._resolve_binary", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(
            "pyutilz.system.system.probing.subprocess.check_output",
            lambda *a, **kw: b"Architecture:  x86_64\nSocket(s):     N/A\n",
        )
        result = get_nix_cpu_sockets_number()
        assert result == 1
        assert isinstance(result, int)


class TestGPUDetection:
    """Test GPU detection functions."""

    @pytest.mark.gpu
    def test_get_nvidia_smi_info(self):
        """get_nvidia_smi_info() returns a populated GPU dict on a host that HAS nvidia-smi.

        Precondition made explicit (the `nvidia-smi` binary on PATH) instead of the old
        ``if gpu_info: ... else: print("[WARN] nvidia-smi not available")``, which reported a
        probe returning nothing as a pass on a machine that genuinely has an NVIDIA GPU.
        """
        import shutil

        if shutil.which("nvidia-smi") is None:
            pytest.skip("the `nvidia-smi` binary is not installed on this host")

        from pyutilz.system.system import get_nvidia_smi_info

        gpu_info = get_nvidia_smi_info(include_stats=False)

        assert isinstance(gpu_info, dict)
        assert gpu_info, "nvidia-smi is installed but get_nvidia_smi_info() parsed nothing out of it"
        assert "gpu" in gpu_info or "driver_version" in gpu_info, f"unexpected keys: {sorted(gpu_info)}"
        if "gpu" in gpu_info:
            assert isinstance(gpu_info["gpu"], list) and gpu_info["gpu"]

    @pytest.mark.gpu
    def test_get_cuda_gpu_details(self):
        """get_cuda_gpu_details() returns per-device CUDA capabilities on a real CUDA host.

        Unconditional assertions after an explicit binary precondition -- the old form asserted
        nothing at all whenever ``get_nvidia_smi_info`` came back empty, i.e. exactly when the
        upstream probe had broken.
        """
        import shutil

        if shutil.which("nvidia-smi") is None:
            pytest.skip("the `nvidia-smi` binary is not installed on this host")

        pytest.importorskip("numba.cuda")

        from pyutilz.system.system import get_cuda_gpu_details, get_nvidia_smi_info

        gpu_info = get_nvidia_smi_info(include_stats=False)
        assert gpu_info, "nvidia-smi is installed but returned no GPU info to derive CUDA details from"

        cuda_details = get_cuda_gpu_details(gpu_info)

        assert isinstance(cuda_details, dict)
        for device_id, caps in cuda_details.items():
            assert device_id is not None
            assert isinstance(caps, dict) and caps, f"device {device_id!r} yielded no CUDA capabilities"
            assert "COMPUTE_CAPABILITY_MAJOR" in caps

    def test_get_gpuutil_gpu_info(self):
        """Test get_gpuutil_gpu_info() returns GPU stats."""
        from pyutilz.system.system import get_gpuutil_gpu_info

        gpu_stats = get_gpuutil_gpu_info()
        assert isinstance(gpu_stats, list)
        if gpu_stats:
            assert "id" in gpu_stats[0]
            print(f"[OK] GPUtil: {len(gpu_stats)} GPU(s) found")
        else:
            print("[WARN] No GPUs detected by GPUtil")

    def test_get_gpuutil_gpu_info_skips_one_bad_gpu_instead_of_truncating(self, monkeypatch):
        """A per-GPU attribute-read failure must not silently drop every GPU after it.

        Regression test: get_gpuutil_gpu_info() previously wrapped the WHOLE enumeration
        loop in a single try/except, so one bad getattr() partway through returned whatever
        partial list had been built so far -- indistinguishable from "these are genuinely
        all the GPUs" for a caller like compute_total_gpus_ram() that sums the result.
        """
        import sys
        import types
        from pyutilz.system.system import get_gpuutil_gpu_info

        good0 = types.SimpleNamespace(id=0, name="GPU0", memoryTotal=8192, memoryFree=4096, load=0.5, driver="1.0", temperature=60, uuid="a")
        bad = types.SimpleNamespace(id=1, name="GPU1")  # missing memoryTotal -- getattr() raises AttributeError
        good2 = types.SimpleNamespace(id=2, name="GPU2", memoryTotal=8192, memoryFree=4096, load=0.5, driver="1.0", temperature=60, uuid="c")

        class _FakeGPUtil:
            @staticmethod
            def getGPUs():
                return [good0, bad, good2]

        monkeypatch.setitem(sys.modules, "GPUtil", _FakeGPUtil)
        devices = get_gpuutil_gpu_info()

        assert [d["id"] for d in devices] == [0, 2]  # the bad GPU is skipped, not a crash or a truncated-after-it list


class TestPowerAndLargePages:
    """Test power management and large pages functions."""

    def test_check_large_pages_support(self):
        """Test check_large_pages_support() cross-platform."""
        from pyutilz.system.system import check_large_pages_support

        result = check_large_pages_support()
        assert isinstance(result, (bool, type(None)))
        if result is not None:
            print(f"[OK] Large pages support: {result}")
        else:
            print("[WARN] Large pages check not supported on this OS")

    def test_get_power_plan(self):
        """get_power_plan() returns None or a dict/list -- and always the right platform probe.

        A desktop/VM legitimately has no power-plan data, so "None" is a valid ANSWER here, not
        an excuse to assert nothing: the shape is pinned unconditionally, and a second,
        deterministic half asserts the platform dispatch itself (which the old
        ``if power_plan: ... else: print("[WARN] ...")`` never touched).
        """
        from pyutilz.system.system import get_power_plan

        power_plan = get_power_plan()
        assert power_plan is None or isinstance(power_plan, (dict, list))
        if isinstance(power_plan, dict):
            assert power_plan, "a returned power-plan dict must not be empty"

    @pytest.mark.parametrize(
        "system_name,expected_delegate",
        [("Windows", "get_windows_power_plan"), ("Darwin", "get_macos_power_plan"), ("Linux", "get_linux_power_plan")],
    )
    def test_get_power_plan_dispatches_per_platform(self, monkeypatch, system_name, expected_delegate):
        """Each platform is routed to its own probe, and the delegate's value is returned as-is."""
        from pyutilz.system.system import probing as probing_mod

        sentinel = {"plan": f"from-{expected_delegate}"}
        called = []
        for name in ("get_windows_power_plan", "get_macos_power_plan", "get_linux_power_plan"):
            monkeypatch.setattr(probing_mod, name, lambda n=name: called.append(n) or (sentinel if n == expected_delegate else {"plan": "WRONG"}))
        monkeypatch.setattr(probing_mod.platform, "system", lambda: system_name)

        assert probing_mod.get_power_plan() == sentinel
        assert called == [expected_delegate]

    def test_get_battery_info(self):
        """get_battery_info() returns None (no battery) or a dict carrying the documented keys.

        Both outcomes are real answers on different hardware, so both are asserted -- unlike the
        old ``if battery_info: ... else: print("[WARN] No battery detected")``, which passed
        even when a laptop's probe silently started returning None.
        """
        from pyutilz.system.system import get_battery_info

        battery_info = get_battery_info()

        assert battery_info is None or isinstance(battery_info, dict)
        if battery_info is not None:
            assert {"percent", "secsleft", "power_plugged"} <= set(battery_info)
            assert 0 <= battery_info["percent"] <= 100
            assert isinstance(battery_info["power_plugged"], (bool, type(None)))

    def test_get_battery_info_converts_the_psutil_namedtuple(self, monkeypatch):
        """The psutil namedtuple is converted to a plain dict (deterministic, hardware-free)."""
        from collections import namedtuple

        from pyutilz.system.system import probing as probing_mod

        sbattery = namedtuple("sbattery", ["percent", "secsleft", "power_plugged"])
        monkeypatch.setattr(probing_mod.psutil, "sensors_battery", lambda: sbattery(percent=77.0, secsleft=3600, power_plugged=False))

        assert probing_mod.get_battery_info() == {"percent": 77.0, "secsleft": 3600, "power_plugged": False}

    def test_get_battery_info_returns_none_without_a_battery(self, monkeypatch):
        from pyutilz.system.system import probing as probing_mod

        monkeypatch.setattr(probing_mod.psutil, "sensors_battery", lambda: None)
        assert probing_mod.get_battery_info() is None


class TestOSAndSoftware:
    """Test OS and software info functions."""

    def test_get_os_info(self):
        """Test get_os_info() returns OS details."""
        from pyutilz.system.system import get_os_info

        os_info = get_os_info()
        assert isinstance(os_info, dict)
        assert "system" in os_info
        assert "platform" in os_info
        print(f"[OK] OS: {os_info['system']} - {os_info['platform']}")

    def test_get_python_info(self):
        """Test get_python_info() returns Python details."""
        from pyutilz.system.system import get_python_info

        python_info = get_python_info()
        assert isinstance(python_info, dict)
        assert "implementation" in python_info
        assert "version" in python_info
        print(f"[OK] Python: {python_info['implementation']} {python_info['version']}")


class TestSystemInfo:
    """Test get_system_info() integration."""

    def test_get_system_info_basic(self):
        """Test get_system_info() with only_stats=False (backward compatibility)."""
        from pyutilz.system.system import get_system_info

        info = get_system_info(only_stats=False)
        assert isinstance(info, dict)
        # Check backward compatibility fields for distributed.py
        assert "host_name" in info
        assert "os_machine_guid" in info
        assert "os_serial" in info
        print(f"[OK] System info (backward compat): host={info['host_name']}")

    @pytest.mark.slow  # >3s measured (pytest --durations, 2026-09-02)
    def test_get_system_info_hardware(self):
        """Test get_system_info() with return_hardware_info=True."""
        from pyutilz.system.system import get_system_info

        info = get_system_info(return_hardware_info=True)
        assert isinstance(info, dict)

        # Check new CPU fields
        if "cpu" in info:
            print(f"[OK] CPU info present: {info['cpu'].get('brand_raw', 'Unknown')[:50]}")

        # Check platform-specific CPU info
        if platform.system() == "Windows" and "cpu_wmi_info" in info:
            print(f"[OK] WMI CPU info present")
        elif platform.system() == "Linux" and "cpu_lscpu_info" in info:
            print(f"[OK] lscpu info present")

        # Check GPU info
        if "gpu_nvidia_smi_info" in info:
            print(f"[OK] NVIDIA GPU info present")

        # Check large pages
        if "large_pages_support" in info:
            print(f"[OK] Large pages support: {info['large_pages_support']}")

    def test_get_system_info_hdd_reports_best_disk(self):
        """get_max_singledisk_free_space_gb() computes best_disk / usage_percent
        alongside the already-surfaced aggregate free/total space, but
        get_system_info() used to discard them (found 2026-07-09 deptry/vulture
        triage). Confirm both now land in the returned dict."""
        from unittest.mock import patch
        from pyutilz.system.system import get_system_info

        with patch(
            "pyutilz.system.system.sysinfo.get_max_singledisk_free_space_gb",
            return_value=(120.5, 42.0, "/dev/sda1", 500.0, 200.0),
        ):
            info = get_system_info(return_hdd_info=True)

        assert info["hdd_max_singledisk_free_space_gb"] == 120.5
        assert info["hdd_best_disk"] == "/dev/sda1"
        assert info["hdd_best_disk_usage_percent"] == 42.0
        assert info["hdd_total_space_gb"] == 500.0
        assert info["hdd_free_space_gb"] == 200.0

    def test_get_system_info_os(self):
        """Test get_system_info() with return_os_info=True."""
        from pyutilz.system.system import get_system_info

        info = get_system_info(return_os_info=True, return_sensitive_info=False)
        assert isinstance(info, dict)

        # Check OS fields
        if "system" in info:
            print(f"[OK] OS system: {info['system']}")

        # Check power plan
        if "power_plan" in info:
            print(f"[OK] Power plan: {info['power_plan']}")

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_get_system_info_hardware_details_windows(self):
        """Test get_system_info() with return_hardware_details=True on Windows."""
        from pyutilz.system.system import get_system_info

        info = get_system_info(return_hardware_details=True)
        assert isinstance(info, dict)

        if "system_wmi_summary" in info:
            print(f"[OK] WMI system summary present")
            if "GPU" in info["system_wmi_summary"]:
                print(f"  - GPU info present")
            if "RAM" in info["system_wmi_summary"]:
                print(f"  - RAM info present")


class TestHardwareMonitor:
    """Test UtilizationMonitor class."""

    def test_utilization_monitor_import(self):
        """Test UtilizationMonitor can be imported."""
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        assert isinstance(UtilizationMonitor, type)
        assert hasattr(UtilizationMonitor, "__init__")
        print("[OK] UtilizationMonitor imported successfully")

    def test_utilization_monitor_init(self):
        """Test UtilizationMonitor initialization."""
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        monitor = UtilizationMonitor(sleep_interval_seconds=0.5, gpu_ids=[])
        assert monitor.sleep_interval_seconds == 0.5
        assert monitor.gpu_ids == []
        print("[OK] UtilizationMonitor initialized")

    @pytest.mark.slow
    def test_utilization_monitor_basic_usage(self):
        """Test UtilizationMonitor basic start/stop."""
        import time
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        monitor = UtilizationMonitor(sleep_interval_seconds=0.5, gpu_ids=[])
        monitor.start()
        time.sleep(2)  # Let it collect some samples
        monitor.stop()

        avg_util = monitor.get_average_utilization()
        assert isinstance(avg_util, dict)
        assert "cpu_utilizaton_percent" in avg_util
        print(f"[OK] UtilizationMonitor collected data: CPU {avg_util['cpu_utilizaton_percent']}%")

    def test_monitor_thread_is_daemon(self):
        """Regression (2026-07-21 audit round 2, HIGH): the worker thread was created without
        daemon=True. If the code between start()/stop() raises, stop() is skipped and the
        thread's infinite loop runs forever -- CPython refuses to exit until every non-daemon
        thread finishes, hanging the whole process at shutdown, not just leaking a thread."""
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        monitor = UtilizationMonitor(sleep_interval_seconds=0.5, gpu_ids=[])
        assert monitor.thread.daemon is True

    def test_context_manager_stops_monitor_even_on_exception(self):
        """Regression: `with UtilizationMonitor(...) as monitor:` must guarantee stop() runs
        (stop_flag set, thread joined) even when the wrapped code raises."""
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        monitor = UtilizationMonitor(sleep_interval_seconds=0.1, gpu_ids=[])
        with pytest.raises(ValueError, match="boom"):
            with monitor as m:
                assert m is monitor
                raise ValueError("boom")

        assert monitor.stop_flag.is_set()
        assert not monitor.thread.is_alive()

    def test_context_manager_normal_exit_stops_monitor(self):
        from pyutilz.system.hardware_monitor import UtilizationMonitor

        with UtilizationMonitor(sleep_interval_seconds=0.1, gpu_ids=[]) as monitor:
            pass

        assert monitor.stop_flag.is_set()
        assert not monitor.thread.is_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
