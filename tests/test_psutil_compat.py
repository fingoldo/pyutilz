"""Behavioural tests for :mod:`pyutilz.system.psutil_compat`.

The module is a graceful-degradation shim over psutil's platform-gated functions. Its contract is
narrow and worth pinning exactly: an absent capability must become ``None`` rather than an
``AttributeError``, and an unavailable clock must become ``None`` rather than ``0.0`` -- the silent
corruption the module's own docstring calls out. Platform simulation is done the way
``has_psutil_function``'s docstring prescribes: by deleting attributes from the psutil module for
the duration of one test (``monkeypatch.delattr``), never by mutating ``sys.modules`` or reloading.
"""

from types import SimpleNamespace

import pytest

psutil = pytest.importorskip("psutil")

from pyutilz.system import psutil_compat as pc


@pytest.fixture(autouse=True)
def _fresh_absence_log(monkeypatch):
    """Isolate the once-per-process absence log so tests don't depend on execution order."""
    monkeypatch.setattr(pc, "_ABSENCE_LOGGED", set())


def _fake_psutil(**attrs):
    return SimpleNamespace(**attrs)


# ------------------------------------------------------------------------------------------------
# has_psutil_function
# ------------------------------------------------------------------------------------------------


def test_has_psutil_function_true_when_attribute_present():
    assert pc.has_psutil_function("cpu_count") is True


def test_has_psutil_function_false_when_attribute_deleted(monkeypatch):
    """Simulate a platform whose psutil backend never defined cpu_freq."""
    monkeypatch.delattr(psutil, "cpu_freq", raising=False)
    assert pc.has_psutil_function("cpu_freq") is False


def test_has_psutil_function_is_not_cached_across_calls(monkeypatch):
    """The docstring promises re-evaluation per call; a cached snapshot would defeat simulation."""
    assert pc.has_psutil_function("cpu_times") is True
    monkeypatch.delattr(psutil, "cpu_times", raising=False)
    assert pc.has_psutil_function("cpu_times") is False
    monkeypatch.undo()
    assert pc.has_psutil_function("cpu_times") is True


def test_has_psutil_function_honours_injected_module():
    assert pc.has_psutil_function("cpu_freq", _fake_psutil(cpu_freq=lambda **kw: None)) is True
    assert pc.has_psutil_function("cpu_freq", _fake_psutil()) is False


# ------------------------------------------------------------------------------------------------
# missing_psutil_functions
# ------------------------------------------------------------------------------------------------


def test_missing_psutil_functions_reports_only_absent_names():
    present = _fake_psutil(cpu_freq=1, sensors_temperatures=1, sensors_fans=1, sensors_battery=1)
    assert pc.missing_psutil_functions(present) == ()
    assert pc.missing_psutil_functions(_fake_psutil()) == pc.OPTIONAL_PSUTIL_FUNCTIONS


def test_missing_psutil_functions_is_a_subset_in_declared_order():
    partial = _fake_psutil(sensors_temperatures=1, sensors_battery=1)
    missing = pc.missing_psutil_functions(partial)
    assert missing == ("cpu_freq", "sensors_fans")
    assert list(missing) == [n for n in pc.OPTIONAL_PSUTIL_FUNCTIONS if n in missing]


def test_missing_psutil_functions_on_real_psutil_with_one_attribute_deleted(monkeypatch):
    monkeypatch.delattr(psutil, "cpu_freq", raising=False)
    assert "cpu_freq" in pc.missing_psutil_functions()


# ------------------------------------------------------------------------------------------------
# get_cpu_freq
# ------------------------------------------------------------------------------------------------


def test_get_cpu_freq_returns_none_instead_of_raising_when_absent(monkeypatch):
    monkeypatch.delattr(psutil, "cpu_freq", raising=False)
    assert pc.get_cpu_freq() is None


def test_get_cpu_freq_passes_percpu_through():
    seen = {}

    def cpu_freq(percpu=False):
        seen["percpu"] = percpu
        return "sentinel"

    assert pc.get_cpu_freq(percpu=True, psutil_module=_fake_psutil(cpu_freq=cpu_freq)) == "sentinel"
    assert seen == {"percpu": True}


def test_get_cpu_freq_returns_none_when_psutil_reports_none():
    assert pc.get_cpu_freq(psutil_module=_fake_psutil(cpu_freq=lambda percpu=False: None)) is None


def test_get_cpu_freq_does_not_swallow_an_exception_from_psutil():
    """Absence is handled; a psutil call that *fails* is a real fault and must stay visible.

    The module docstring is explicit that a missing capability must not be "swallowed by a broad
    ``except``" -- so an error from an existing cpu_freq propagates rather than being disguised as
    the same ``None`` that means "not measurable here".
    """

    def boom(percpu=False):
        raise OSError("counter read failed")

    with pytest.raises(OSError):
        pc.get_cpu_freq(psutil_module=_fake_psutil(cpu_freq=boom))


def test_get_cpu_freq_logs_absence_only_once(caplog):
    absent = _fake_psutil()
    with caplog.at_level("INFO", logger=pc.logger.name):
        pc.get_cpu_freq(psutil_module=absent)
        pc.get_cpu_freq(psutil_module=absent)
    assert len([r for r in caplog.records if "cpu_freq" in r.getMessage()]) == 1


# ------------------------------------------------------------------------------------------------
# get_cpu_freq_current_mhz
# ------------------------------------------------------------------------------------------------


def test_current_mhz_returns_float_value():
    freq = SimpleNamespace(current=2400.0, min=800.0, max=3600.0)
    got = pc.get_cpu_freq_current_mhz(psutil_module=_fake_psutil(cpu_freq=lambda percpu=False: freq))
    assert got == pytest.approx(2400.0)
    assert isinstance(got, float)


def test_current_mhz_is_none_not_zero_when_platform_lacks_cpu_freq(monkeypatch):
    monkeypatch.delattr(psutil, "cpu_freq", raising=False)
    got = pc.get_cpu_freq_current_mhz()
    assert got is None
    assert got != 0.0


def test_current_mhz_is_none_not_zero_when_psutil_returns_none():
    got = pc.get_cpu_freq_current_mhz(psutil_module=_fake_psutil(cpu_freq=lambda percpu=False: None))
    assert got is None


def test_current_mhz_is_none_when_current_field_is_none():
    freq = SimpleNamespace(current=None)
    assert pc.get_cpu_freq_current_mhz(psutil_module=_fake_psutil(cpu_freq=lambda percpu=False: freq)) is None


def test_current_mhz_preserves_a_genuine_zero_reading():
    """0.0 reported by psutil is a measurement, not an absence -- it must survive as 0.0, and the
    caller must still be able to tell it apart from the ``None`` that means "unavailable"."""
    freq = SimpleNamespace(current=0.0)
    got = pc.get_cpu_freq_current_mhz(psutil_module=_fake_psutil(cpu_freq=lambda percpu=False: freq))
    assert got == 0.0
    assert got is not None


def test_current_mhz_coerces_an_integer_reading_to_float():
    freq = SimpleNamespace(current=2400)
    got = pc.get_cpu_freq_current_mhz(psutil_module=_fake_psutil(cpu_freq=lambda percpu=False: freq))
    assert isinstance(got, float)
    assert got == 2400.0
