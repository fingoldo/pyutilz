"""Tests targeting uncovered lines in system.py (extra3)."""

import platform
import pytest
from unittest.mock import patch, MagicMock, mock_open
import types


# ── parse_dmidecode_info (lines 757-832) ──

class TestParseDmidecodeInfo:
    SAMPLE_OUTPUT = (
        "Handle 0x0000, DMI type 0, 24 bytes\n"
        "BIOS Information\n"
        "\tVendor: American Megatrends\n"
        "\tVersion: 1.0\n"
        "\tRelease Date: 01/01/2020\n"
        "\tCharacteristics:\n"
        "\t\tPCI is supported\n"
        "\t\tBIOS is upgradeable\n"
        "\n"
        "Handle 0x0001, DMI type 1, 27 bytes\n"
        "System Information\n"
        "\tManufacturer: ASUS\n"
        "\tProduct Name: ROG\n"
        "\n"
        "Handle 0x0002, DMI type 9, 17 bytes\n"
        "System Boot Information\n"
        "\tStatus: No errors\n"
    )

    @patch("pyutilz.system.system.subprocess")
    def test_parses_bios_and_system(self, mock_sub):
        from pyutilz.system.system import parse_dmidecode_info
        mock_sub.run.return_value = MagicMock(stdout=self.SAMPLE_OUTPUT)
        result = parse_dmidecode_info()
        assert result is not None
        assert len(result) >= 2
        # Check BIOS section parsed
        items = {dict(r["item"]).get("ItemType"): dict(r["item"]) for r in result}
        assert "BIOS Information" in items
        assert items["BIOS Information"]["Vendor"] == "American Megatrends"

    @patch("pyutilz.system.system.subprocess")
    def test_features_collected(self, mock_sub):
        from pyutilz.system.system import parse_dmidecode_info
        mock_sub.run.return_value = MagicMock(stdout=self.SAMPLE_OUTPUT)
        result = parse_dmidecode_info()
        items = {dict(r["item"]).get("ItemType"): dict(r["item"]) for r in result}
        bios = items["BIOS Information"]
        assert "Characteristics" in bios
        assert "BIOS is upgradeable" in bios["Characteristics"]
        assert "PCI is supported" in bios["Characteristics"]

    @patch("pyutilz.system.system.subprocess")
    def test_sections_to_avoid_skipped(self, mock_sub):
        from pyutilz.system.system import parse_dmidecode_info
        mock_sub.run.return_value = MagicMock(stdout=self.SAMPLE_OUTPUT)
        result = parse_dmidecode_info()
        item_types = [dict(r["item"]).get("ItemType") for r in result]
        assert "System Boot Information" not in item_types

    @patch("pyutilz.system.system.subprocess")
    def test_skip_keys(self, mock_sub):
        from pyutilz.system.system import parse_dmidecode_info
        mock_sub.run.return_value = MagicMock(stdout=self.SAMPLE_OUTPUT)
        result = parse_dmidecode_info(skip_keys={"Vendor", "Version", "Release Date", "Serial Number", "Socket Designation", "Manufacturer", "Product Name"})
        items = {dict(r["item"]).get("ItemType"): dict(r["item"]) for r in result}
        bios = items.get("BIOS Information")
        if bios:
            assert "Vendor" not in bios

    @patch("pyutilz.system.system.subprocess")
    def test_skip_values(self, mock_sub):
        from pyutilz.system.system import parse_dmidecode_info
        output = (
            "Handle 0x0, DMI type 0, 24 bytes\n"
            "BIOS Information\n"
            "\tVendor: Not Provided\n"
            "\tVersion: 2.0\n"
            "\tSpeed: 3600\n"
        )
        mock_sub.run.return_value = MagicMock(stdout=output)
        result = parse_dmidecode_info()
        items = {dict(r["item"]).get("ItemType"): dict(r["item"]) for r in result}
        bios = items["BIOS Information"]
        assert "Vendor" not in bios
        # Version has "Version" in key so no float conversion
        assert bios["Version"] == "2.0"
        # Speed does get numeric conversion
        assert bios["Speed"] == 3600

    @patch("pyutilz.system.system.subprocess")
    def test_exception_returns_none(self, mock_sub):
        from pyutilz.system.system import parse_dmidecode_info
        mock_sub.run.side_effect = FileNotFoundError("no dmidecode")
        result = parse_dmidecode_info()
        assert result is None

    @patch("pyutilz.system.system.subprocess")
    def test_empty_output(self, mock_sub):
        from pyutilz.system.system import parse_dmidecode_info
        mock_sub.run.return_value = MagicMock(stdout="")
        result = parse_dmidecode_info()
        assert result == []

    @patch("pyutilz.system.system.subprocess")
    def test_numeric_value_int_conversion(self, mock_sub):
        from pyutilz.system.system import parse_dmidecode_info
        output = (
            "Handle 0x0, DMI type 4, 42 bytes\n"
            "Processor Information\n"
            "\tMax Speed: 3600\n"
            "\tCore Count: 8\n"
        )
        mock_sub.run.return_value = MagicMock(stdout=output)
        result = parse_dmidecode_info()
        items = {dict(r["item"]).get("ItemType"): dict(r["item"]) for r in result}
        proc = items["Processor Information"]
        # 3600.0.is_integer() -> int(3600)
        assert proc["Max Speed"] == 3600
        assert isinstance(proc["Max Speed"], int)

    @patch("pyutilz.system.system.subprocess")
    def test_handle_line_skipped(self, mock_sub):
        from pyutilz.system.system import parse_dmidecode_info
        output = (
            "Handle 0x0, DMI type 4, 42 bytes\n"
            "Processor Information\n"
            "\tL1 Cache Handle: 0x0049\n"
            "\tSpeed: 2400\n"
        )
        mock_sub.run.return_value = MagicMock(stdout=output)
        result = parse_dmidecode_info()
        items = {dict(r["item"]).get("ItemType"): dict(r["item"]) for r in result}
        proc = items["Processor Information"]
        assert "L1 Cache Handle" not in proc

    @patch("pyutilz.system.system.subprocess")
    def test_counter_dedup(self, mock_sub):
        """Two identical Memory Device sections get counted."""
        from pyutilz.system.system import parse_dmidecode_info
        output = (
            "Handle 0x0, DMI type 17, 40 bytes\n"
            "Memory Device\n"
            "\tSize: 8192 MB\n"
            "\n"
            "Handle 0x1, DMI type 17, 40 bytes\n"
            "Memory Device\n"
            "\tSize: 8192 MB\n"
        )
        mock_sub.run.return_value = MagicMock(stdout=output)
        result = parse_dmidecode_info()
        assert len(result) == 1
        assert result[0]["Count"] == 2

    @patch("pyutilz.system.system.subprocess")
    def test_features_flushed_on_new_section(self, mock_sub):
        """Features list is flushed when a new non-tab line starts."""
        from pyutilz.system.system import parse_dmidecode_info
        output = (
            "Handle 0x0, DMI type 0, 24 bytes\n"
            "BIOS Information\n"
            "\tFeatures:\n"
            "\t\tFeature A\n"
            "\t\tFeature B\n"
            "\tVendor: TestVendor\n"
        )
        mock_sub.run.return_value = MagicMock(stdout=output)
        result = parse_dmidecode_info()
        items = {dict(r["item"]).get("ItemType"): dict(r["item"]) for r in result}
        bios = items["BIOS Information"]
        assert bios["Features"] == ("Feature A", "Feature B")

    @patch("pyutilz.system.system.subprocess")
    def test_last_section_flushed(self, mock_sub):
        """Last section with features is flushed at EOF."""
        from pyutilz.system.system import parse_dmidecode_info
        # "BIOS" matches interesting_sections list item "BIOS"
        output = (
            "Handle 0x0, DMI type 0, 24 bytes\n"
            "BIOS Information\n"
            "\tVendor: Test\n"
            "\tCapabilities:\n"
            "\t\tCap1\n"
            "\t\tCap2\n"
        )
        mock_sub.run.return_value = MagicMock(stdout=output)
        result = parse_dmidecode_info()
        assert len(result) >= 1
        item = dict(result[0]["item"])
        assert item.get("Capabilities") == ("Cap1", "Cap2")


# ── summarize_system_info (lines 901-903) ──

class TestSummarizeSystemInfo:
    @patch("pyutilz.system.system.summarize_devices", return_value=[{"name": "test"}])
    def test_wmi_not_available(self, mock_sd):
        from pyutilz.system.system import summarize_system_info
        with patch.dict("sys.modules", {"wmi": None}):
            # Force ImportError
            with patch("builtins.__import__", side_effect=ImportError("no wmi")):
                result = summarize_system_info()
        assert result is None

    def test_wmi_exception(self):
        from pyutilz.system.system import summarize_system_info
        mock_wmi_module = MagicMock()
        mock_wmi_module.WMI.side_effect = Exception("WMI failed")
        with patch.dict("sys.modules", {"wmi": mock_wmi_module}):
            result = summarize_system_info()
        assert result is None


# ── get_system_info: Linux fallback both fail (lines 456-458) ──

class TestGetSystemInfoLinuxFallbackBothFail:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.subprocess")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={"system": "Linux"})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_both_machine_id_fail(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                                   mock_psutil, mock_sub, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Linux"
        mock_sub.check_output.side_effect = [Exception("no dbus"), Exception("no etc")]
        mock_socket.gethostname.return_value = "host"
        result = get_system_info(return_os_info=True, return_sensitive_info=True)
        assert "os_serial" in result


# ── get_system_info: Android serial (lines 459-466) ──

class TestGetSystemInfoAndroid:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.subprocess")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_android_serial_success(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                                     mock_psutil, mock_sub, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Android"
        mock_sub.check_output.return_value = b"ANDROID123\n"
        mock_socket.gethostname.return_value = "droid"
        result = get_system_info(return_os_info=True, return_sensitive_info=True)
        assert result["os_machine_guid"] == "ANDROID123"

    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.subprocess")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_android_serial_failure(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                                     mock_psutil, mock_sub, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Android"
        mock_sub.check_output.side_effect = Exception("no getprop")
        mock_socket.gethostname.return_value = "droid"
        result = get_system_info(return_os_info=True, return_sensitive_info=True)
        assert "os_serial" in result


# ── get_system_info: Mac UUID (lines 467-485) ──

class TestGetSystemInfoMac:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.subprocess")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_mac_uuid_success(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                               mock_psutil, mock_sub, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Mac"
        ioreg_proc = MagicMock()
        grep_proc = MagicMock()
        grep_proc.communicate.return_value = (b'"IOPlatformUUID" = "ABC-DEF-123"', b"")
        ioreg_proc.stdout = MagicMock()
        mock_sub.Popen.side_effect = [ioreg_proc, grep_proc]
        mock_sub.PIPE = -1
        mock_socket.gethostname.return_value = "machost"
        result = get_system_info(return_os_info=True, return_sensitive_info=True)
        assert result["os_machine_guid"] == "ABC-DEF-123"

    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.subprocess")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_mac_uuid_failure(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                               mock_psutil, mock_sub, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Mac"
        mock_sub.Popen.side_effect = Exception("no ioreg")
        mock_sub.PIPE = -1
        mock_socket.gethostname.return_value = "machost"
        result = get_system_info(return_os_info=True, return_sensitive_info=True)
        assert "os_serial" in result


# ── get_system_info: Windows serial failure (lines 439-440) ──

class TestGetSystemInfoWindowsSerialFail:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.subprocess")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_wmic_failure(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                           mock_psutil, mock_sub, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Windows"
        mock_sub.check_output.side_effect = Exception("wmic not found")
        mock_socket.gethostname.return_value = "winhost"
        result = get_system_info(return_os_info=True, return_sensitive_info=True)
        assert "os_machine_guid" in result  # falls back to uuid


# ── get_system_info: gpu_cuda_capabilities (line 533) ──

class TestGetSystemInfoGpuCuda:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.get_nvidia_smi_info", return_value={"gpu": [{"name": "GTX"}]})
    @patch("pyutilz.system.system.get_cuda_gpu_details", return_value={"0": {"cores": 128}})
    @patch("pyutilz.system.system.get_cpu_info", return_value=None)
    @patch("pyutilz.system.system.check_large_pages_support", return_value=None)
    def test_gpu_cuda_capabilities_included(self, mock_lp, mock_cpu, mock_cuda, mock_nv,
                                             mock_bat, mock_pp, mock_os_info, mock_psutil, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Linux"
        mock_plat.python_implementation.return_value = "CPython"
        mock_plat.python_version.return_value = "3.11"
        freq = MagicMock(current=2400, min=800, max=3600)
        ram = MagicMock(total=16 * 2**30, free=8 * 2**30)
        mock_psutil.cpu_freq.return_value = freq
        mock_psutil.virtual_memory.return_value = ram
        mock_psutil.cpu_count.side_effect = [4, 8]
        with patch("pyutilz.system.system.get_lscpu_info", return_value=None), \
             patch("pyutilz.system.system.get_linux_board_info", return_value=None), \
             patch("pyutilz.system.system.get_nix_cpu_sockets_number", return_value=1):
            result = get_system_info(return_hardware_info=True)
        assert result["gpu_cuda_capabilities"] == {"0": {"cores": 128}}


# ── get_system_info: gpu stats exception paths (lines 570-571, 576-577) ──

class TestGetSystemInfoGpuStatsExceptions:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    def test_gpu_stats_exceptions_handled(self, mock_bat, mock_pp, mock_os_info, mock_psutil, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Linux"
        mock_plat.python_implementation.return_value = "CPython"
        mock_plat.python_version.return_value = "3.11"
        freq = MagicMock(current=2400, min=800, max=3600)
        ram = MagicMock(total=16 * 2**30, free=8 * 2**30)
        mock_psutil.cpu_freq.return_value = freq
        mock_psutil.virtual_memory.return_value = ram

        # Mock numba to trigger the cuda branch
        mock_numba = MagicMock()
        mock_numba.cuda.is_available.return_value = True
        mock_cuda = MagicMock()

        with patch.dict("sys.modules", {"numba": mock_numba, "numba.cuda": mock_cuda}), \
             patch("pyutilz.system.system.subprocess") as mock_sub, \
             patch("pyutilz.system.system.get_nvidia_smi_info", side_effect=Exception("no smi")), \
             patch("pyutilz.system.system.get_gpuutil_gpu_info", side_effect=Exception("no gputil")):
            mock_sub.check_output.side_effect = Exception("no nvcc")
            result = get_system_info()
        assert "gpu_current_stats_nvidia_smi" not in result
        assert "gpu_current_stats" not in result


# ── get_system_info: outer exception (lines 613-615) ──

class TestGetSystemInfoOuterException:
    @patch("pyutilz.system.system.platform")
    def test_outer_exception_returns_partial(self, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.side_effect = Exception("boom")
        result = get_system_info()
        assert isinstance(result, dict)


# ── get_system_info: hardware_details linux devices exception (lines 610-611) ──

class TestGetSystemInfoDevicesException:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.parse_dmidecode_info", return_value=None)
    @patch("pyutilz.system.system.list_linux_devices", side_effect=Exception("lspci fail"))
    def test_devices_list_exception(self, mock_dev, mock_dmi, mock_bat, mock_pp, mock_os_info, mock_psutil, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Linux"
        mock_plat.python_implementation.return_value = "CPython"
        mock_plat.python_version.return_value = "3.11"
        result = get_system_info(return_hardware_details=True)
        assert "devices_list" not in result


# ── get_cpu_info import error (lines 631-633) ──

class TestGetCpuInfoImportError:
    def test_no_cpuinfo(self):
        from pyutilz.system.system import get_cpu_info
        with patch.dict("sys.modules", {"cpuinfo": None}):
            with patch("builtins.__import__", side_effect=ImportError("no cpuinfo")):
                result = get_cpu_info()
        assert result is None


# ── get_wmi_cpu_info exception (lines 668-670) ──

class TestGetWmiCpuinfoException:
    def test_wmi_exception(self):
        from pyutilz.system.system import get_wmi_cpuinfo
        mock_wmi_module = MagicMock()
        mock_wmi_module.WMI.side_effect = Exception("WMI fail")
        with patch.dict("sys.modules", {"wmi": mock_wmi_module}):
            result = get_wmi_cpuinfo()
        assert result is None


# ── show_biggest_session_objects (lines 1004-1006) ──

class TestShowBiggestSessionObjects:
    @patch("pyutilz.system.system.clean_ram")
    @patch("pyutilz.system.system.get_own_memory_usage", return_value=1.5)
    def test_getsizeof_exception(self, mock_mem, mock_clean):
        from pyutilz.system.system import show_biggest_session_objects

        class BadObj:
            def __sizeof__(self):
                raise TypeError("cannot size")

        session = {"bad": BadObj(), "good": "hello"}
        result = show_biggest_session_objects(session, N=5, min_size_bytes=1)
        # Should not crash; good object is in result
        assert result is not None


# ── check_large_pages_windows exception (lines 1072-1074) ──

@pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
class TestCheckLargePagesWindows:
    def test_exception_returns_false(self):
        from pyutilz.system.system import check_large_pages_windows
        import ctypes as real_ctypes
        mock_ct = MagicMock()
        mock_ct.windll.kernel32.GetLargePageMinimum.side_effect = OSError("no kernel32")
        mock_ct.c_size_t = real_ctypes.c_size_t
        with patch("ctypes.windll", mock_ct.windll):
            result = check_large_pages_windows()
        assert result is False


# ── check_large_pages_support Darwin (line 1106) ──

class TestCheckLargePagesSupport:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.check_huge_pages_macos", return_value=True)
    def test_darwin(self, mock_hp, mock_plat):
        from pyutilz.system.system import check_large_pages_support
        mock_plat.system.return_value = "Darwin"
        assert check_large_pages_support() is True


# ── get_windows_power_plan WMI active (lines 1195-1197) ──

class TestGetWindowsPowerPlanWMI:
    def test_active_plan(self):
        from pyutilz.system.system import get_windows_power_plan
        mock_wmi = MagicMock()
        plan = MagicMock()
        plan.IsActive = True
        plan.InstanceID = "GUID-123"
        plan.ElementName = "High Performance"
        mock_wmi_module = MagicMock()
        mock_wmi_module.WMI.return_value = mock_wmi
        mock_wmi.Win32_PowerPlan.return_value = [plan]
        with patch.dict("sys.modules", {"wmi": mock_wmi_module}):
            result = get_windows_power_plan()
        assert result == {"plan_guid": "GUID-123", "plan_name": "High Performance"}

    def test_no_active_plan(self):
        from pyutilz.system.system import get_windows_power_plan
        mock_wmi = MagicMock()
        plan = MagicMock()
        plan.IsActive = False
        mock_wmi_module = MagicMock()
        mock_wmi_module.WMI.return_value = mock_wmi
        mock_wmi.Win32_PowerPlan.return_value = [plan]
        with patch.dict("sys.modules", {"wmi": mock_wmi_module}):
            result = get_windows_power_plan()
        assert result is None


# ── get_power_plan Darwin (line 1213) ──

class TestGetPowerPlanDarwin:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.get_macos_power_plan", return_value={"plan": "auto"})
    def test_darwin_branch(self, mock_mac, mock_plat):
        from pyutilz.system.system import get_power_plan
        mock_plat.system.return_value = "Darwin"
        result = get_power_plan()
        assert result == {"plan": "auto"}


# ── get_nvidia_smi_info import error (lines 1277-1279) ──

class TestGetNvidiaSmiInfoImportError:
    def test_no_xmltodict(self):
        from pyutilz.system.system import get_nvidia_smi_info
        with patch.dict("sys.modules", {"xmltodict": None}):
            with patch("builtins.__import__", side_effect=ImportError("no xmltodict")):
                result = get_nvidia_smi_info()
        assert result is None


# ── get_nvidia_smi_info: power_readings stat removal (line 1352) ──

class TestGetNvidiaSmiInfoStatRemoval:
    @patch("pyutilz.system.system.subprocess")
    def test_stat_fields_removed_when_not_stats(self, mock_sub):
        from pyutilz.system.system import get_nvidia_smi_info
        mock_xmltodict = MagicMock()
        smi_data = {
            "nvidia_smi_log": {
                "driver_version": "535",
                "cuda_version": "12.2",
                "attached_gpus": "1",
                "gpu": [{
                    "@id": "00000000:01:00.0",
                    "product_name": "GTX 1050",
                    "fb_memory_usage": {"total": "4096 MiB", "reserved": "100", "used": "200", "free": "3796"},
                    "temperature": {"gpu_temp": "45 C", "memory_temp": "N/A"},
                    "power_readings": {"power_state": "P0", "power_draw": "75W"},
                    "gpu_power_readings": {"power_state": "P0", "power_draw": "75W"},
                    "utilization": {"gpu_util": "5%", "memory_util": "10%"},
                    "clocks": {"graphics_clock": "1500 MHz"},
                }],
            }
        }
        mock_xmltodict.parse.return_value = smi_data
        mock_run_result = MagicMock()
        mock_run_result.returncode = 0
        mock_run_result.stdout = "<xml/>"
        mock_sub.run.return_value = mock_run_result
        mock_sub.PIPE = -1

        with patch.dict("sys.modules", {"xmltodict": mock_xmltodict}):
            result = get_nvidia_smi_info(include_stats=False)
        assert result is not None
        gpu = result["gpu"][0]
        assert "reserved" not in gpu.get("fb_memory_usage", {})


# ── get_gpu_cuda_capabilities (lines 1384-1412) ──

class TestGetGpuCudaCapabilities:
    def test_success_with_cores(self):
        from pyutilz.system.system import get_gpu_cuda_capabilities
        mock_enums = MagicMock()
        mock_enums.__dir__ = lambda self: [
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
            "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT",
            "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
        ]
        mock_cuda = MagicMock()
        device = MagicMock()
        device.COMPUTE_CAPABILITY_MAJOR = 7
        device.COMPUTE_CAPABILITY_MINOR = 5
        device.MULTIPROCESSOR_COUNT = 20
        device.MAX_THREADS_PER_BLOCK = 1024
        mock_cuda.get_current_device.return_value = device

        with patch.dict("sys.modules", {
            "numba": MagicMock(),
            "numba.cuda": mock_cuda,
            "numba.cuda.cudadrv": MagicMock(),
            "numba.cuda.cudadrv.enums": mock_enums,
        }):
            # Need to also patch dir() on enums
            with patch("pyutilz.system.system.remove_json_attributes"), \
                 patch("pyutilz.system.system.remove_json_defaults"):
                result = get_gpu_cuda_capabilities(device_id=0)
        assert result is not None

    def test_import_error_returns_none(self):
        from pyutilz.system.system import get_gpu_cuda_capabilities
        with patch.dict("sys.modules", {"numba": None, "numba.cuda": None, "numba.cuda.cudadrv": None, "numba.cuda.cudadrv.enums": None}):
            with patch("builtins.__import__", side_effect=ImportError("no numba")):
                result = get_gpu_cuda_capabilities()
        assert result is None

    def test_cuda_exception_returns_none(self):
        from pyutilz.system.system import get_gpu_cuda_capabilities
        mock_enums = MagicMock()
        mock_cuda_mod = MagicMock()
        mock_cuda_mod.select_device.side_effect = Exception("CUDA error")

        with patch.dict("sys.modules", {
            "numba": MagicMock(),
            "numba.cuda": mock_cuda_mod,
            "numba.cuda.cudadrv": MagicMock(),
            "numba.cuda.cudadrv.enums": mock_enums,
        }):
            # The function does `from numba import cuda` then `cuda.select_device()`
            # We need to make sure the import resolves to our mock
            import numba as numba_ref
            numba_ref.cuda = mock_cuda_mod
            result = get_gpu_cuda_capabilities(device_id=0)
        assert result is None


# ── get_cuda_gpu_details (lines 1425, 1431-1432) ──

class TestGetCudaGpuDetails:
    def test_none_input(self):
        from pyutilz.system.system import get_cuda_gpu_details
        assert get_cuda_gpu_details(None) == {}

    @patch("pyutilz.system.system.get_gpu_cuda_capabilities", return_value={"cores": 128})
    def test_with_gpu_module_id(self, mock_caps):
        from pyutilz.system.system import get_cuda_gpu_details
        result = get_cuda_gpu_details({"gpu": [{"gpu_module_id": 0}]})
        assert 0 in result

    @patch("pyutilz.system.system.get_gpu_cuda_capabilities", return_value={"cores": 128})
    def test_no_module_id(self, mock_caps):
        from pyutilz.system.system import get_cuda_gpu_details
        result = get_cuda_gpu_details({"gpu": [{"name": "GTX"}]})
        assert result == {}


# ── get_gpuutil_gpu_info (lines 1453-1455, 1468-1469) ──

class TestGetGpuutilGpuInfo:
    def test_import_error(self):
        from pyutilz.system.system import get_gpuutil_gpu_info
        with patch.dict("sys.modules", {"GPUtil": None}):
            with patch("builtins.__import__", side_effect=Exception("no GPUtil")):
                result = get_gpuutil_gpu_info()
        assert result == []

    def test_getgpus_exception(self):
        from pyutilz.system.system import get_gpuutil_gpu_info
        mock_gputil = MagicMock()
        mock_gputil.getGPUs.side_effect = Exception("GPU error")
        with patch.dict("sys.modules", {"GPUtil": mock_gputil}):
            result = get_gpuutil_gpu_info()
        assert result == []


# ── list_linux_devices (lines 1550-1556) ──

class TestListLinuxDevices:
    @patch("pyutilz.system.system.subprocess")
    def test_success(self, mock_sub):
        from pyutilz.system.system import list_linux_devices
        devices = [
            {"slot": "00:00.0", "cls": "Host bridge", "progif": None, "driver": None,
             "kernel_modules": [], "numa_node": None, "iommu_group": None, "physical_slot": None}
        ]
        import json
        mock_sub.check_output.return_value = json.dumps(devices).encode("utf-8")
        result = list_linux_devices()
        assert isinstance(result, list)
        # defaults should be removed
        assert "progif" not in result[0]

    @patch("pyutilz.system.system.subprocess")
    def test_exception(self, mock_sub):
        from pyutilz.system.system import list_linux_devices
        mock_sub.check_output.side_effect = FileNotFoundError("no pylspci")
        result = list_linux_devices()
        assert result is None


# ── report_large_objects (lines 1659-1678) ──

class TestReportLargeObjects:
    @patch("pyutilz.system.system.asizeof")
    def test_no_big_objects(self, mock_asizeof):
        from pyutilz.system.system import report_large_objects
        mock_asizeof.asizeof.return_value = 100  # 100 bytes, way under 200MB
        report_large_objects(min_size_mb=200)

    @patch("pyutilz.system.system.asizeof")
    def test_with_big_objects(self, mock_asizeof):
        from pyutilz.system.system import report_large_objects
        mock_asizeof.asizeof.return_value = 300 * 1024 * 1024  # 300 MB
        report_large_objects(min_size_mb=200)

    @patch("pyutilz.system.system.asizeof")
    def test_with_memory_snapshot(self, mock_asizeof):
        from pyutilz.system.system import report_large_objects
        mock_asizeof.asizeof.return_value = 300 * 1024 * 1024
        mock_snapshot = MagicMock()
        import tracemalloc
        with patch("pyutilz.system.system.tracemalloc") as mock_tm:
            mock_tm.take_snapshot.return_value.compare_to.return_value = []
            report_large_objects(min_size_mb=200, initial_memory_snapshot=mock_snapshot)

    @patch("pyutilz.system.system.asizeof")
    def test_with_memory_snapshot_exception(self, mock_asizeof):
        from pyutilz.system.system import report_large_objects
        mock_asizeof.asizeof.return_value = 300 * 1024 * 1024
        mock_snapshot = MagicMock()
        with patch("pyutilz.system.system.tracemalloc") as mock_tm:
            mock_tm.take_snapshot.side_effect = Exception("tracemalloc fail")
            report_large_objects(min_size_mb=200, initial_memory_snapshot=mock_snapshot)


# ── ensure_idle_devices (lines 1731, 1734-1806) ──

class TestEnsureIdleDevices:
    @patch("pyutilz.system.system.psutil")
    def test_cpu_ram_exceeds_total(self, mock_psutil):
        from pyutilz.system.system import ensure_idle_devices
        mem = MagicMock()
        mem.total = 4 * (1024**3)  # 4GB
        mock_psutil.virtual_memory.return_value = mem
        result = ensure_idle_devices(min_cpu_free_ram_gb=8.0)
        assert result is False

    @patch("pyutilz.system.system.psutil")
    def test_gpu_ram_exceeds_total(self, mock_psutil):
        from pyutilz.system.system import ensure_idle_devices
        mem = MagicMock()
        mem.total = 16 * (1024**3)
        mock_psutil.virtual_memory.return_value = mem

        mock_gputil = MagicMock()
        gpu = MagicMock()
        gpu.id = 0
        gpu.memoryTotal = 2048  # 2GB
        mock_gputil.getGPUs.return_value = [gpu]

        with patch.dict("sys.modules", {"GPUtil": mock_gputil}):
            result = ensure_idle_devices(min_gpu_free_ram_gb=4.0)
        assert result is False

    @patch("pyutilz.system.system.psutil")
    def test_gpu_ids_filter(self, mock_psutil):
        from pyutilz.system.system import ensure_idle_devices
        mem = MagicMock()
        mem.total = 16 * (1024**3)
        mem.available = 12 * (1024**3)
        mock_psutil.virtual_memory.return_value = mem
        mock_psutil.cpu_percent.return_value = 5.0

        mock_gputil = MagicMock()
        gpu0 = MagicMock(id=0, memoryTotal=2048, memoryFree=1024, load=0.05)
        gpu1 = MagicMock(id=1, memoryTotal=8192, memoryFree=6144, load=0.05)
        mock_gputil.getGPUs.return_value = [gpu0, gpu1]

        with patch.dict("sys.modules", {"GPUtil": mock_gputil}), \
             patch("time.time", side_effect=[0, 0, 6]), \
             patch("time.sleep"):
            # Only check GPU 1 which has enough memory
            result = ensure_idle_devices(duration_seconds=5, min_gpu_free_ram_gb=4.0, gpu_ids=[1])
        assert result is True

    @patch("pyutilz.system.system.psutil")
    def test_gputil_not_available(self, mock_psutil):
        from pyutilz.system.system import ensure_idle_devices
        mem = MagicMock()
        mem.total = 16 * (1024**3)
        mem.available = 12 * (1024**3)
        mock_psutil.virtual_memory.return_value = mem
        mock_psutil.cpu_percent.return_value = 5.0

        with patch.dict("sys.modules", {"GPUtil": None}), \
             patch("builtins.__import__", side_effect=_selective_import_error("GPUtil")), \
             patch("time.time", side_effect=[0, 0, 6]), \
             patch("time.sleep"):
            result = ensure_idle_devices(duration_seconds=5, min_cpu_free_ram_gb=1.0)
        assert result is True

    @patch("pyutilz.system.system.psutil")
    def test_gpu_conditions_checked(self, mock_psutil):
        from pyutilz.system.system import ensure_idle_devices
        mem = MagicMock()
        mem.total = 16 * (1024**3)
        mem.available = 12 * (1024**3)
        mock_psutil.virtual_memory.return_value = mem
        mock_psutil.cpu_percent.return_value = 5.0

        mock_gputil = MagicMock()
        gpu = MagicMock()
        gpu.id = 0
        gpu.memoryTotal = 8192  # 8GB
        gpu.memoryFree = 6144   # 6GB
        gpu.load = 0.05         # 5%
        mock_gputil.getGPUs.return_value = [gpu]

        with patch.dict("sys.modules", {"GPUtil": mock_gputil}), \
             patch("time.time", side_effect=[0, 0, 6]), \
             patch("time.sleep"):
            result = ensure_idle_devices(duration_seconds=5)
        assert result is True

    @patch("pyutilz.system.system.psutil")
    def test_gpu_high_load(self, mock_psutil):
        from pyutilz.system.system import ensure_idle_devices
        mem = MagicMock()
        mem.total = 16 * (1024**3)
        mem.available = 12 * (1024**3)
        mock_psutil.virtual_memory.return_value = mem
        mock_psutil.cpu_percent.return_value = 5.0

        mock_gputil = MagicMock()
        gpu_busy = MagicMock(id=0, memoryTotal=8192, memoryFree=6144, load=0.90)
        gpu_idle = MagicMock(id=0, memoryTotal=8192, memoryFree=6144, load=0.05)
        mock_gputil.getGPUs.side_effect = [[gpu_busy], [gpu_idle], [gpu_idle]]

        with patch.dict("sys.modules", {"GPUtil": mock_gputil}), \
             patch("time.time", side_effect=[0, 100, 100, 106]), \
             patch("time.sleep"):
            result = ensure_idle_devices(duration_seconds=5)
        assert result is True

    @patch("pyutilz.system.system.psutil")
    def test_gpu_low_free_ram(self, mock_psutil):
        from pyutilz.system.system import ensure_idle_devices
        mem = MagicMock()
        mem.total = 16 * (1024**3)
        mem.available = 12 * (1024**3)
        mock_psutil.virtual_memory.return_value = mem
        mock_psutil.cpu_percent.return_value = 5.0

        mock_gputil = MagicMock()
        gpu_low = MagicMock(id=0, memoryTotal=8192, memoryFree=100, load=0.05)
        gpu_ok = MagicMock(id=0, memoryTotal=8192, memoryFree=6144, load=0.05)
        mock_gputil.getGPUs.side_effect = [[gpu_low], [gpu_ok], [gpu_ok]]

        with patch.dict("sys.modules", {"GPUtil": mock_gputil}), \
             patch("time.time", side_effect=[0, 100, 100, 106]), \
             patch("time.sleep"):
            result = ensure_idle_devices(duration_seconds=5)
        assert result is True

    def test_invalid_cpu_load_percent(self):
        from pyutilz.system.system import ensure_idle_devices
        with pytest.raises(ValueError):
            ensure_idle_devices(max_cpu_load_percent=150)

    def test_invalid_gpu_load_percent(self):
        from pyutilz.system.system import ensure_idle_devices
        with pytest.raises(ValueError):
            ensure_idle_devices(max_gpu_load_percent=-1)


# ── Helper for selective import errors ──

def _selective_import_error(blocked_module):
    """Return an __import__ replacement that blocks only the specified module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Mocked: {name}")
        return real_import(name, *args, **kwargs)
    return _import
