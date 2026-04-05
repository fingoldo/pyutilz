import pytest
from unittest.mock import patch, MagicMock, mock_open
import platform


# ── get_wmi_obj_as_dict (lines 270-282) ──

class TestGetWmiObjAsDict:
    def test_ensure_float_conversion(self):
        from pyutilz.system.system import get_wmi_obj_as_dict
        obj = MagicMock()
        obj.properties = ["speed"]
        obj.speed = "3.14"
        result = get_wmi_obj_as_dict(obj, ensure_float={"speed"})
        assert result["speed"] == 3.14

    def test_ensure_float_invalid(self):
        from pyutilz.system.system import get_wmi_obj_as_dict
        obj = MagicMock()
        obj.properties = ["speed"]
        obj.speed = "not_a_number"
        result = get_wmi_obj_as_dict(obj, ensure_float={"speed"})
        assert result["speed"] == "not_a_number"

    def test_ensure_int_conversion(self):
        from pyutilz.system.system import get_wmi_obj_as_dict
        obj = MagicMock()
        obj.properties = ["cores"]
        obj.cores = "8"
        result = get_wmi_obj_as_dict(obj, ensure_int={"cores"})
        assert result["cores"] == 8

    def test_ensure_int_invalid(self):
        from pyutilz.system.system import get_wmi_obj_as_dict
        obj = MagicMock()
        obj.properties = ["cores"]
        obj.cores = None
        result = get_wmi_obj_as_dict(obj, ensure_int={"cores"})
        assert result["cores"] is None

    def test_decode_dict(self):
        from pyutilz.system.system import get_wmi_obj_as_dict
        obj = MagicMock()
        obj.properties = ["status"]
        obj.status = 3
        result = get_wmi_obj_as_dict(obj, decode_dict={"status": lambda x: f"code_{x}"})
        assert result["status"] == "code_3"

    def test_exclude_pros(self):
        from pyutilz.system.system import get_wmi_obj_as_dict
        obj = MagicMock()
        obj.properties = ["name", "secret"]
        obj.name = "cpu"
        obj.secret = "hidden"
        result = get_wmi_obj_as_dict(obj, exclude_pros={"secret"})
        assert "secret" not in result
        assert result["name"] == "cpu"


# ── tqdmu (lines 347-351) ──

class TestTqdmu:
    @patch("pyutilz.system.system.run_from_ipython", return_value=True)
    @patch("pyutilz.system.system.tqdm")
    def test_ipython_notebook(self, mock_tqdm, mock_ip):
        from pyutilz.system.system import tqdmu
        mock_tqdm.tqdm_notebook.return_value = "notebook_bar"
        result = tqdmu([1, 2, 3])
        assert result == "notebook_bar"

    @patch("pyutilz.system.system.run_from_ipython", return_value=True)
    @patch("pyutilz.system.system.tqdm")
    def test_ipython_notebook_fallback(self, mock_tqdm, mock_ip):
        from pyutilz.system.system import tqdmu
        mock_tqdm.tqdm_notebook.side_effect = Exception("no notebook")
        mock_tqdm.tqdm.return_value = "plain_bar"
        result = tqdmu([1, 2, 3])
        assert result == "plain_bar"


# ── get_system_info sensitive info branches (lines 439-485, 491, 509-517, 545-549, 557-558, 570-577, 580-590, 603-615) ──

class TestGetSystemInfoSensitive:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.subprocess")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={"system": "Linux"})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_linux_machine_id(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                               mock_psutil, mock_sub, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Linux"
        mock_sub.check_output.return_value = b"abc123\n"
        mock_socket.gethostname.return_value = "myhost"
        result = get_system_info(return_os_info=True, return_sensitive_info=True)
        assert result["os_machine_guid"] == "abc123"
        assert result["host_name"] == "myhost"

    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.subprocess")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={"system": "Linux"})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_linux_fallback_etc_machine_id(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                                            mock_psutil, mock_sub, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Linux"
        mock_sub.check_output.side_effect = [Exception("no dbus"), b"fallback123\n"]
        mock_socket.gethostname.return_value = "host2"
        result = get_system_info(return_os_info=True, return_sensitive_info=True)
        assert result["os_machine_guid"] == "fallback123"

    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.subprocess")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_windows_serial(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                             mock_psutil, mock_sub, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Windows"
        mock_sub.check_output.return_value = b"UUID\nABCD-1234\n"
        mock_socket.gethostname.return_value = "winhost"
        result = get_system_info(return_os_info=True, return_sensitive_info=True)
        assert result["os_machine_guid"] == "ABCD-1234"
        assert result["os_serial"] == "ABCD-1234"

    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.socket")
    def test_usage_stats(self, mock_socket, mock_bat, mock_pp, mock_os_info,
                          mock_psutil, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Linux"
        mock_plat.python_implementation.return_value = "CPython"
        mock_plat.python_version.return_value = "3.11"
        freq = MagicMock(current=2400, min=800, max=3600)
        ram = MagicMock(total=16 * 2**30, free=8 * 2**30)
        mock_psutil.cpu_freq.return_value = freq
        mock_psutil.virtual_memory.return_value = ram
        mock_psutil.cpu_percent.side_effect = [50.0, [25.0, 75.0]]
        result = get_system_info(return_usage_stats=True)
        assert result["cpu_current_frequency_hz"] == 2400
        assert result["ram_free_gb"] == 8.0

    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.get_nvidia_smi_info", return_value=None)
    @patch("pyutilz.system.system.get_cpu_info", return_value={"brand": "Intel"})
    @patch("pyutilz.system.system.check_large_pages_support", return_value=True)
    @patch("pyutilz.system.system.socket")
    def test_hardware_info_linux(self, mock_socket, mock_lp, mock_cpu, mock_gpu,
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
        with patch("pyutilz.system.system.get_lscpu_info", return_value={"Model": "79"}), \
             patch("pyutilz.system.system.get_linux_board_info", return_value={"Vendor": "ASUSTeK"}), \
             patch("pyutilz.system.system.get_nix_cpu_sockets_number", return_value=1):
            result = get_system_info(return_hardware_info=True)
        assert result["cpu"] == {"brand": "Intel"}
        assert result["cpu_lscpu_info"] == {"Model": "79"}
        assert result["cpu_num_sockets"] == 1

    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.get_max_singledisk_free_space_gb", return_value=(100, 0.5, "C:", 500, 200))
    def test_hdd_info(self, mock_hdd, mock_bat, mock_pp, mock_os_info, mock_psutil, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Windows"
        mock_plat.python_implementation.return_value = "CPython"
        mock_plat.python_version.return_value = "3.11"
        result = get_system_info(return_hdd_info=True)
        assert result["hdd_free_space_gb"] == 200
        assert result["hdd_total_space_gb"] == 500

    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.psutil")
    @patch("pyutilz.system.system.get_os_info", return_value={})
    @patch("pyutilz.system.system.get_power_plan", return_value=None)
    @patch("pyutilz.system.system.get_battery_info", return_value=None)
    @patch("pyutilz.system.system.parse_dmidecode_info", return_value=[{"item": {"key": "val"}, "Count": 1}])
    @patch("pyutilz.system.system.list_linux_devices", return_value=[{"device": "eth0"}])
    def test_hardware_details_linux(self, mock_dev, mock_dmi, mock_bat, mock_pp, mock_os_info, mock_psutil, mock_plat):
        from pyutilz.system.system import get_system_info
        mock_plat.system.return_value = "Linux"
        mock_plat.python_implementation.return_value = "CPython"
        mock_plat.python_version.return_value = "3.11"
        result = get_system_info(return_hardware_details=True)
        assert "dmidecode_info" in result
        assert "devices_list" in result


# ── get_lscpu_info (lines 685-708) ──

class TestGetLscpuInfo:
    @patch("pyutilz.system.system.subprocess")
    def test_parses_lscpu_output(self, mock_sub):
        from pyutilz.system.system import get_lscpu_info
        mock_sub.check_output.return_value = "Architecture: x86_64\nCPU(s): 8\nCPU MHz: 2199.998\nFlags: avx sse\n"
        result = get_lscpu_info()
        assert result["Architecture"] == "x86_64"
        assert result["CPU(s)"] == 8
        assert result["CPU MHz"] == 2199.998
        assert result["Flags"] == "avx sse"

    @patch("pyutilz.system.system.subprocess")
    def test_lscpu_exception(self, mock_sub):
        from pyutilz.system.system import get_lscpu_info
        mock_sub.check_output.side_effect = FileNotFoundError("no lscpu")
        result = get_lscpu_info()
        assert result is None


# ── get_linux_board_info (lines 717-727) ──

class TestGetLinuxBoardInfo:
    def test_reads_board_files(self):
        from pyutilz.system.system import get_linux_board_info
        m = mock_open()
        m.side_effect = [
            mock_open(read_data="ASUSTeK\n")(),
            mock_open(read_data="ROG STRIX\n")(),
            mock_open(read_data="Rev 1.0\n")(),
        ]
        with patch("builtins.open", m):
            result = get_linux_board_info()
        assert result["Vendor"] == "ASUSTeK"
        assert result["Name"] == "ROG STRIX"

    def test_file_not_found(self):
        from pyutilz.system.system import get_linux_board_info
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = get_linux_board_info()
        assert result == {} or isinstance(result, dict)


# ── get_nix_cpu_sockets_number (lines 907-915) ──

class TestGetNixCpuSocketsNumber:
    @patch("pyutilz.system.system.subprocess")
    def test_parses_socket_count(self, mock_sub):
        from pyutilz.system.system import get_nix_cpu_sockets_number
        mock_sub.check_output.return_value = b"Socket(s): 2\nother stuff\n"
        result = get_nix_cpu_sockets_number()
        assert result == 2

    @patch("pyutilz.system.system.subprocess")
    def test_exception_returns_1(self, mock_sub):
        from pyutilz.system.system import get_nix_cpu_sockets_number
        mock_sub.check_output.side_effect = FileNotFoundError
        result = get_nix_cpu_sockets_number()
        assert result == 1


# ── get_own_memory_usage (lines 929-930) ──

class TestGetOwnMemoryUsage:
    @patch("pyutilz.system.system.psutil")
    def test_exception_returns_none(self, mock_psutil):
        from pyutilz.system.system import get_own_memory_usage
        mock_psutil.Process.side_effect = Exception("no process")
        result = get_own_memory_usage()
        assert result is None


# ── trim_windows_process_memory (lines 939-969) ──

class TestTrimWindowsProcessMemory:
    @patch("pyutilz.system.system.ctypes")
    def test_success(self, mock_ctypes):
        from pyutilz.system.system import trim_windows_process_memory
        mock_ctypes.sizeof.return_value = 8
        mock_ctypes.c_void_p = MagicMock()
        mock_ctypes.windll.kernel32.GetCurrentProcess.return_value = 42
        mock_ctypes.windll.kernel32.SetProcessWorkingSetSizeEx.return_value = 1
        result = trim_windows_process_memory()
        assert result is True

    @patch("pyutilz.system.system.ctypes")
    def test_failure(self, mock_ctypes):
        from pyutilz.system.system import trim_windows_process_memory
        mock_ctypes.sizeof.return_value = 4
        mock_ctypes.c_void_p = MagicMock()
        mock_ctypes.windll.kernel32.GetCurrentProcess.return_value = 42
        mock_ctypes.windll.kernel32.SetProcessWorkingSetSizeEx.return_value = 0
        mock_ctypes.windll.kernel32.GetLastError.return_value = 5
        result = trim_windows_process_memory()
        assert result is False

    @patch("pyutilz.system.system.ctypes")
    def test_with_explicit_pid(self, mock_ctypes):
        from pyutilz.system.system import trim_windows_process_memory
        mock_ctypes.sizeof.return_value = 8
        mock_ctypes.c_void_p = MagicMock()
        mock_ctypes.windll.kernel32.SetProcessWorkingSetSizeEx.return_value = 1
        result = trim_windows_process_memory(pid=1234)
        assert result is True


# ── clean_ram (lines 976-983) ──

class TestCleanRam:
    @patch("pyutilz.system.system.gc")
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.trim_windows_process_memory")
    def test_windows(self, mock_trim, mock_plat, mock_gc):
        from pyutilz.system.system import clean_ram
        mock_plat.system.return_value = "Windows"
        clean_ram()
        mock_gc.collect.assert_called_once()
        mock_trim.assert_called_once()

    @patch("pyutilz.system.system.gc")
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.ctypes")
    def test_linux(self, mock_ctypes, mock_plat, mock_gc):
        from pyutilz.system.system import clean_ram
        mock_plat.system.return_value = "Linux"
        clean_ram()
        mock_gc.collect.assert_called_once()

    @patch("pyutilz.system.system.gc")
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.ctypes")
    def test_linux_malloc_trim_fails(self, mock_ctypes, mock_plat, mock_gc):
        from pyutilz.system.system import clean_ram
        mock_plat.system.return_value = "Linux"
        mock_ctypes.CDLL.side_effect = OSError("no libc")
        clean_ram()
        mock_gc.collect.assert_called_once()


# ── show_biggest_session_objects (lines 992-1012) ──

class TestShowBiggestSessionObjects:
    @patch("pyutilz.system.system.clean_ram")
    @patch("pyutilz.system.system.get_own_memory_usage", return_value=1.5)
    def test_returns_dataframe(self, mock_mem, mock_clean):
        from pyutilz.system.system import show_biggest_session_objects
        import pandas as pd
        session = {"a": [1]*1000, "b": "hello"}
        result = show_biggest_session_objects(session, N=2, min_size_bytes=1)
        assert isinstance(result, pd.DataFrame) or isinstance(result, list)

    @patch("pyutilz.system.system.clean_ram")
    @patch("pyutilz.system.system.get_own_memory_usage", return_value=0.5)
    def test_empty_session(self, mock_mem, mock_clean):
        from pyutilz.system.system import show_biggest_session_objects
        result = show_biggest_session_objects({}, N=5, min_size_bytes=999999999)
        assert result == [] or (hasattr(result, '__len__') and len(result) == 0)


# ── check_huge_pages_linux (lines 1047-1055) ──

class TestCheckHugePagesLinux:
    def test_huge_pages_enabled(self):
        from pyutilz.system.system import check_huge_pages_linux
        data = "HugePages_Total:      10\nHugePages_Free:        5\n"
        with patch("builtins.open", mock_open(read_data=data)):
            assert check_huge_pages_linux() is True

    def test_huge_pages_disabled(self):
        from pyutilz.system.system import check_huge_pages_linux
        data = "HugePages_Total:       0\nHugePages_Free:        0\n"
        with patch("builtins.open", mock_open(read_data=data)):
            assert check_huge_pages_linux() is False

    def test_huge_pages_error(self):
        from pyutilz.system.system import check_huge_pages_linux
        with patch("builtins.open", side_effect=PermissionError):
            assert check_huge_pages_linux() is False


# ── check_large_pages_support (lines 1102, 1105-1109) ──

class TestCheckLargePagesSupport:
    @patch("pyutilz.system.system.platform")
    @patch("pyutilz.system.system.check_huge_pages_linux", return_value=True)
    def test_linux(self, mock_hp, mock_plat):
        from pyutilz.system.system import check_large_pages_support
        mock_plat.system.return_value = "Linux"
        assert check_large_pages_support() is True

    @patch("pyutilz.system.system.platform")
    def test_unsupported_os(self, mock_plat):
        from pyutilz.system.system import check_large_pages_support
        mock_plat.system.return_value = "FreeBSD"
        assert check_large_pages_support() is None


# ── get_linux_power_plan (lines 1123-1134) ──

class TestGetLinuxPowerPlan:
    @patch("os.listdir", return_value=["cpu0", "cpu1", "other"])
    def test_reads_governors(self, mock_ls):
        from pyutilz.system.system import get_linux_power_plan
        m = mock_open(read_data="performance\n")
        with patch("builtins.open", m):
            result = get_linux_power_plan()
        assert result == ["performance"]

    @patch("os.listdir", side_effect=FileNotFoundError)
    def test_error(self, mock_ls):
        from pyutilz.system.system import get_linux_power_plan
        result = get_linux_power_plan()
        assert result is None


# ── get_macos_power_plan (lines 1143-1150) ──

class TestGetMacosPowerPlan:
    @patch("pyutilz.system.system.subprocess")
    def test_returns_plan(self, mock_sub):
        from pyutilz.system.system import get_macos_power_plan
        mock_sub.run.return_value = MagicMock(stdout="Battery Power:\n displaysleep 10\n")
        result = get_macos_power_plan()
        assert "plan_full_name" in result

    @patch("pyutilz.system.system.subprocess")
    def test_empty_result(self, mock_sub):
        from pyutilz.system.system import get_macos_power_plan
        mock_sub.run.return_value = MagicMock(stdout="")
        result = get_macos_power_plan()
        assert result is None

    @patch("pyutilz.system.system.subprocess")
    def test_exception(self, mock_sub):
        from pyutilz.system.system import get_macos_power_plan
        mock_sub.run.side_effect = FileNotFoundError
        result = get_macos_power_plan()
        assert result is None


# ── get_windows_power_plan_cmd (lines 1166, 1174-1177) ──

class TestGetWindowsPowerPlanCmd:
    @patch("pyutilz.system.system.subprocess")
    def test_parses_powercfg(self, mock_sub):
        from pyutilz.system.system import get_windows_power_plan_cmd
        mock_sub.run.return_value = MagicMock(
            stdout="Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance)"
        )
        result = get_windows_power_plan_cmd()
        assert result["plan_guid"] == "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
        assert result["plan_name"] == "High performance"

    @patch("pyutilz.system.system.subprocess")
    def test_unparseable(self, mock_sub):
        from pyutilz.system.system import get_windows_power_plan_cmd
        mock_sub.run.return_value = MagicMock(stdout="some weird output")
        result = get_windows_power_plan_cmd()
        assert "plan_full_name" in result

    @patch("pyutilz.system.system.subprocess")
    def test_empty(self, mock_sub):
        from pyutilz.system.system import get_windows_power_plan_cmd
        mock_sub.run.return_value = MagicMock(stdout="")
        result = get_windows_power_plan_cmd()
        assert result is None

    @patch("pyutilz.system.system.subprocess")
    def test_exception(self, mock_sub):
        from pyutilz.system.system import get_windows_power_plan_cmd
        mock_sub.run.side_effect = OSError
        result = get_windows_power_plan_cmd()
        assert result is None


# ── get_windows_power_plan WMI (lines 1188-1197) ──

class TestGetWindowsPowerPlan:
    @patch("pyutilz.system.system.get_windows_power_plan_cmd", return_value={"plan_guid": "x", "plan_name": "y"})
    def test_wmi_import_error(self, mock_cmd):
        from pyutilz.system.system import get_windows_power_plan
        with patch.dict("sys.modules", {"wmi": None}):
            # when wmi import fails it falls back to cmd
            result = get_windows_power_plan()
        assert result is not None


# ── get_battery_info (lines 1277-1279) ──

class TestGetBatteryInfo:
    @patch("pyutilz.system.system.psutil")
    def test_no_battery(self, mock_psutil):
        from pyutilz.system.system import get_battery_info
        mock_psutil.sensors_battery.return_value = None
        result = get_battery_info()
        assert result is None

    @patch("pyutilz.system.system.psutil")
    def test_exception(self, mock_psutil):
        from pyutilz.system.system import get_battery_info
        mock_psutil.sensors_battery.side_effect = RuntimeError("no battery")
        result = get_battery_info()
        assert result is None


# ── get_nvidia_smi_info (lines 1291-1293, 1297-1298) ──

class TestGetNvidiaSmiInfo:
    @patch("pyutilz.system.system.subprocess")
    def test_not_found(self, mock_sub):
        from pyutilz.system.system import get_nvidia_smi_info
        mock_sub.run.side_effect = FileNotFoundError
        with patch.dict("sys.modules", {"xmltodict": MagicMock()}):
            result = get_nvidia_smi_info()
        assert result is None

    @patch("pyutilz.system.system.subprocess")
    def test_nonzero_returncode(self, mock_sub):
        from pyutilz.system.system import get_nvidia_smi_info
        mock_sub.run.return_value = MagicMock(returncode=1, stdout="err", stderr="fail")
        with patch.dict("sys.modules", {"xmltodict": MagicMock()}):
            result = get_nvidia_smi_info()
        assert result is None


# ── get_gpu_cuda_capabilities (lines 1377-1412) ──

class TestGetGpuCudaCapabilities:
    def test_import_error(self):
        from pyutilz.system.system import get_gpu_cuda_capabilities
        with patch.dict("sys.modules", {"numba": None, "numba.cuda": None, "numba.cuda.cudadrv": None, "numba.cuda.cudadrv.enums": None}):
            result = get_gpu_cuda_capabilities()
            assert result is None


# ── compute_total_gpus_ram (lines 1487-1491) ──

class TestComputeTotalGpusRam:
    def test_empty_list(self):
        from pyutilz.system.system import compute_total_gpus_ram
        result = compute_total_gpus_ram([])
        assert result == {"gpu_max_ram_total": 0, "gpus_ram_total": 0, "gpus_ram_free": 0}

    def test_multiple_gpus(self):
        from pyutilz.system.system import compute_total_gpus_ram
        gpus = [{"memoryTotal": 8, "memoryFree": 4}, {"memoryTotal": 16, "memoryFree": 10}]
        result = compute_total_gpus_ram(gpus)
        assert result["gpu_max_ram_total"] == 16
        assert result["gpus_ram_total"] == 24
        assert result["gpus_ram_free"] == 14


# ── get_max_singledisk_free_space_gb (lines 1515-1537) ──

class TestGetMaxSinglediskFreeSpaceGb:
    @patch("pyutilz.system.system.psutil")
    def test_basic(self, mock_psutil):
        from pyutilz.system.system import get_max_singledisk_free_space_gb
        disk = MagicMock(opts="rw", fstype="ntfs", mountpoint="C:\\")
        mock_psutil.disk_partitions.return_value = [disk]
        mock_psutil.disk_usage.return_value = MagicMock(total=500 * 2**30, free=200 * 2**30)
        max_free, pct, best, total, free = get_max_singledisk_free_space_gb()
        assert abs(max_free - 200.0) < 0.01
        assert abs(total - 500.0) < 0.01

    @patch("pyutilz.system.system.psutil")
    def test_required_filesystem_filter(self, mock_psutil):
        from pyutilz.system.system import get_max_singledisk_free_space_gb
        disk1 = MagicMock(opts="rw", fstype="ntfs", mountpoint="C:\\")
        disk2 = MagicMock(opts="rw", fstype="ext4", mountpoint="/data")
        mock_psutil.disk_partitions.return_value = [disk1, disk2]
        mock_psutil.disk_usage.return_value = MagicMock(total=100 * 2**30, free=50 * 2**30)
        max_free, _, _, _, _ = get_max_singledisk_free_space_gb(required_filesystem="ext4")
        assert abs(max_free - 50.0) < 0.01


# ── list_linux_devices (lines 1548-1559) ──

class TestListLinuxDevices:
    @patch("pyutilz.system.system.subprocess")
    def test_exception(self, mock_sub):
        from pyutilz.system.system import list_linux_devices
        mock_sub.check_output.side_effect = FileNotFoundError
        result = list_linux_devices()
        assert result is None


# ── count_app_instances (lines 1641-1647) ──

class TestCountAppInstances:
    @patch("psutil.process_iter")
    def test_with_cmdline_filter(self, mock_iter):
        from pyutilz.system.system import count_app_instances
        proc1 = MagicMock()
        proc1.name.return_value = "python"
        proc1.cmdline.return_value = ["python", "train.py"]
        proc2 = MagicMock()
        proc2.name.return_value = "python"
        proc2.cmdline.return_value = ["python", "serve.py"]
        mock_iter.return_value = [proc1, proc2]
        count = count_app_instances(processname="python", cmdline="train.py")
        assert count == 1

    @patch("psutil.process_iter")
    def test_cmdline_exception(self, mock_iter):
        from pyutilz.system.system import count_app_instances
        proc = MagicMock()
        proc.name.return_value = "python"
        proc.cmdline.side_effect = PermissionError
        mock_iter.return_value = [proc]
        count = count_app_instances(cmdline="something")
        assert count == 0


# ── ensure_idle_devices (lines 1710-1806) ──

class TestEnsureIdleDevices:
    def test_invalid_cpu_load_raises(self):
        from pyutilz.system.system import ensure_idle_devices
        with pytest.raises(ValueError):
            ensure_idle_devices(max_cpu_load_percent=150)

    def test_invalid_gpu_load_raises(self):
        from pyutilz.system.system import ensure_idle_devices
        with pytest.raises(ValueError):
            ensure_idle_devices(max_gpu_load_percent=-1)

    @patch("pyutilz.system.system.psutil")
    def test_cpu_ram_exceeds_total(self, mock_psutil):
        from pyutilz.system.system import ensure_idle_devices
        mock_psutil.virtual_memory.return_value = MagicMock(total=2 * 1024**3)
        result = ensure_idle_devices(min_cpu_free_ram_gb=100)
        assert result is False


# ── get_locale_settings (lines 1829-1834) ──

class TestGetLocaleSettings:
    def test_with_only_fields(self):
        from pyutilz.system.system import get_locale_settings
        result = get_locale_settings(locale_name="", only_fields=("decimal_point",))
        assert "decimal_point" in result
        assert "thousands_sep" not in result


# ── beep (lines 1843-1850) ──

class TestBeep:
    @patch.dict("sys.modules", {"winsound": MagicMock()})
    def test_beep_no_crash(self):
        from pyutilz.system.system import beep
        beep()

    def test_beep_no_winsound(self):
        from pyutilz.system.system import beep
        with patch.dict("sys.modules", {"winsound": None}):
            beep()  # should not raise


# ── get_wmi_cpuinfo / summarize_system_info WMI not available (lines 656-670, 843-845, 901-903) ──

class TestWmiFallbacks:
    def test_get_wmi_cpuinfo_no_wmi(self):
        from pyutilz.system.system import get_wmi_cpuinfo
        with patch.dict("sys.modules", {"wmi": None}):
            result = get_wmi_cpuinfo()
            assert result is None

    def test_summarize_system_info_no_wmi(self):
        from pyutilz.system.system import summarize_system_info
        with patch.dict("sys.modules", {"wmi": None}):
            result = summarize_system_info()
            assert result is None


# ── check_huge_pages_macos (lines 1083-1091) ──

class TestCheckHugePagesMacos:
    @patch("pyutilz.system.system.subprocess")
    def test_vm_stat_output(self, mock_sub):
        from pyutilz.system.system import check_huge_pages_macos
        mock_sub.check_output.return_value = b"Pages free: 12345\nPages active: 67890\n"
        result = check_huge_pages_macos()
        assert result is True

    @patch("pyutilz.system.system.subprocess")
    def test_vm_stat_error(self, mock_sub):
        from pyutilz.system.system import check_huge_pages_macos
        mock_sub.check_output.side_effect = FileNotFoundError
        result = check_huge_pages_macos()
        assert result is False
