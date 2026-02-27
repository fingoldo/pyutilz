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
        """Test get_cpu_info() returns CPU information."""
        from pyutilz.system.system import get_cpu_info

        cpu_info = get_cpu_info()
        if cpu_info:
            assert isinstance(cpu_info, dict)
            # Should not include these filtered keys
            assert "python_version" not in cpu_info
            assert "cpuinfo_version" not in cpu_info
            print(f"[OK] CPU Info: {cpu_info.get('brand_raw', 'Unknown')}")
        else:
            print("[WARN] py-cpuinfo not available")

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_get_wmi_cpuinfo(self):
        """Test get_wmi_cpuinfo() on Windows."""
        from pyutilz.system.system import get_wmi_cpuinfo

        cpu_info = get_wmi_cpuinfo()
        if cpu_info:
            assert isinstance(cpu_info, list)
            assert len(cpu_info) > 0
            assert "Count" in cpu_info[0]
            print(f"[OK] WMI CPU Info: {cpu_info[0].get('Name', 'Unknown')}")
        else:
            print("[WARN] WMI not available")

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
    def test_get_lscpu_info(self):
        """Test get_lscpu_info() on Linux."""
        from pyutilz.system.system import get_lscpu_info

        lscpu_info = get_lscpu_info()
        if lscpu_info:
            assert isinstance(lscpu_info, dict)
            assert "Architecture" in lscpu_info
            print(f"[OK] lscpu Info: {lscpu_info.get('Model name', 'Unknown')}")
        else:
            print("[WARN] lscpu not available")


class TestGPUDetection:
    """Test GPU detection functions."""

    @pytest.mark.gpu
    def test_get_nvidia_smi_info(self):
        """Test get_nvidia_smi_info() returns GPU information."""
        from pyutilz.system.system import get_nvidia_smi_info

        gpu_info = get_nvidia_smi_info(include_stats=False)
        if gpu_info:
            assert isinstance(gpu_info, dict)
            assert "gpu" in gpu_info or "driver_version" in gpu_info
            if "gpu" in gpu_info:
                print(f"[OK] GPU Info: {len(gpu_info['gpu'])} GPU(s) detected")
        else:
            print("[WARN] nvidia-smi not available (no NVIDIA GPU or drivers)")

    @pytest.mark.gpu
    def test_get_cuda_gpu_details(self):
        """Test get_cuda_gpu_details() returns CUDA capabilities."""
        from pyutilz.system.system import get_nvidia_smi_info, get_cuda_gpu_details

        gpu_info = get_nvidia_smi_info(include_stats=False)
        if gpu_info:
            cuda_details = get_cuda_gpu_details(gpu_info)
            assert isinstance(cuda_details, dict)
            if cuda_details:
                print(f"[OK] CUDA Details: {len(cuda_details)} device(s) with CUDA capabilities")
        else:
            print("[WARN] No GPU info available for CUDA details test")

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
        """Test get_power_plan() cross-platform."""
        from pyutilz.system.system import get_power_plan

        power_plan = get_power_plan()
        if power_plan:
            assert isinstance(power_plan, (dict, list))
            print(f"[OK] Power plan: {power_plan}")
        else:
            print("[WARN] Power plan info not available")

    def test_get_battery_info(self):
        """Test get_battery_info()."""
        from pyutilz.system.system import get_battery_info

        battery_info = get_battery_info()
        if battery_info:
            assert isinstance(battery_info, dict)
            print(f"[OK] Battery: {battery_info.get('percent', 'N/A')}%")
        else:
            print("[WARN] No battery detected (desktop system)")


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

        assert UtilizationMonitor is not None
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
