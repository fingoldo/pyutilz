import pytest
from unittest.mock import patch, MagicMock
from collections import namedtuple

from pyutilz.system.system import (
    remove_nas,
    dict_to_tuple,
    decode_memory_type,
    decode_cpu_upgrade_method,
    get_os_info,
    get_battery_info,
    get_power_plan,
    get_python_info,
    get_utc_unix_timestamp,
    get_max_affordable_workers_count,
    run_from_ipython,
)


# -- remove_nas --

def test_remove_nas_removes_na_values():
    assert remove_nas({"a": "N/A", "b": "ok"}) == {"b": "ok"}


def test_remove_nas_converts_numeric_strings():
    assert remove_nas({"a": "3.14"}) == {"a": 3.14}


def test_remove_nas_nested_dict():
    assert remove_nas({"a": {"b": "N/A", "c": "1"}}) == {"a": {"c": 1.0}}


def test_remove_nas_list():
    assert remove_nas(["N/A", "2", "hi"]) == [2.0, "hi"]


def test_remove_nas_passthrough():
    assert remove_nas(42) == 42


# -- dict_to_tuple --

def test_dict_to_tuple_sorted():
    assert dict_to_tuple({"b": 2, "a": 1}) == (("a", 1), ("b", 2))


def test_dict_to_tuple_empty():
    assert dict_to_tuple({}) == ()


# -- decode_memory_type --

def test_decode_memory_type_known():
    assert decode_memory_type(26) == "DDR4"
    assert decode_memory_type(24) == "DDR3"


def test_decode_memory_type_unknown():
    assert decode_memory_type(999) == "Unknown"


# -- decode_cpu_upgrade_method --

def test_decode_cpu_upgrade_method_known():
    assert decode_cpu_upgrade_method(49) == "Socket AM4"
    assert decode_cpu_upgrade_method(50) == "Socket LGA1151"


def test_decode_cpu_upgrade_method_unknown():
    assert decode_cpu_upgrade_method(9999) == "Unknown"


# -- get_os_info --

@patch("pyutilz.system.system.platform")
def test_get_os_info_linux(mock_plat):
    mock_plat.system.return_value = "Linux"
    mock_plat.machine.return_value = "x86_64"
    mock_plat.version.return_value = "#1 SMP"
    mock_plat.platform.return_value = "Linux-5.15"
    mock_plat.architecture.return_value = ("64bit", "ELF")
    info = get_os_info()
    assert info["system"] == "Linux"
    assert info["architecture"] == "64bit, ELF"
    assert "edition" not in info


@patch("pyutilz.system.system.platform")
def test_get_os_info_windows(mock_plat):
    mock_plat.system.return_value = "Windows"
    mock_plat.machine.return_value = "AMD64"
    mock_plat.version.return_value = "10.0.19045"
    mock_plat.platform.return_value = "Windows-10"
    mock_plat.architecture.return_value = ("64bit", "WindowsPE")
    mock_plat.win32_edition.return_value = "Professional"
    info = get_os_info()
    assert info["edition"] == "Professional"


# -- get_battery_info --

@patch("pyutilz.system.system.psutil")
def test_get_battery_info_present(mock_psutil):
    BatteryTuple = namedtuple("sbattery", ["percent", "secsleft", "power_plugged"])
    mock_psutil.sensors_battery.return_value = BatteryTuple(75, 3600, True)
    result = get_battery_info()
    assert result["percent"] == 75
    assert result["power_plugged"] is True


@patch("pyutilz.system.system.psutil")
def test_get_battery_info_none(mock_psutil):
    mock_psutil.sensors_battery.return_value = None
    assert get_battery_info() is None


@patch("pyutilz.system.system.psutil")
def test_get_battery_info_exception(mock_psutil):
    mock_psutil.sensors_battery.side_effect = RuntimeError("no battery")
    assert get_battery_info() is None


# -- get_power_plan --

@patch("pyutilz.system.system.platform")
@patch("pyutilz.system.system.get_windows_power_plan", return_value={"plan": "High"})
def test_get_power_plan_windows(mock_wp, mock_plat):
    mock_plat.system.return_value = "Windows"
    assert get_power_plan() == {"plan": "High"}


@patch("pyutilz.system.system.platform")
@patch("pyutilz.system.system.get_linux_power_plan", return_value={"governor": "performance"})
def test_get_power_plan_linux(mock_lp, mock_plat):
    mock_plat.system.return_value = "Linux"
    assert get_power_plan() == {"governor": "performance"}


# -- get_python_info --

def test_get_python_info_keys():
    info = get_python_info()
    assert "implementation" in info
    assert "version" in info
    assert "sys_version" in info


# -- get_utc_unix_timestamp --

def test_get_utc_unix_timestamp_is_int():
    ts = get_utc_unix_timestamp()
    assert isinstance(ts, int)
    assert ts > 1_000_000_000


# -- get_max_affordable_workers_count --

@patch("psutil.cpu_count", return_value=8)
def test_max_workers_normal(mock_cpu):
    assert get_max_affordable_workers_count() == 7


@patch("psutil.cpu_count", return_value=1)
def test_max_workers_minimum(mock_cpu):
    assert get_max_affordable_workers_count() == 1


# -- run_from_ipython --

def test_run_from_ipython_false():
    assert run_from_ipython() is False
