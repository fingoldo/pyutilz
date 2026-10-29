"""Behaviour of ``pyutilz.core.pythonlib.hardware``: outside a notebook kernel ``is_jupyter_notebook`` is False, a CPU flag
that is not present is reported as absent, every probe returns a plain bool, and ``is_cuda_available`` gives the same answer
on repeated calls because it is memoized for the process. Probes that need a real device are only checked for their type."""

from __future__ import annotations

from pyutilz.core.pythonlib import hardware


class TestNotebookDetection:
    def test_a_plain_python_process_is_not_a_notebook(self):
        assert hardware.is_jupyter_notebook() is False


class TestCpuFlags:
    def test_an_unknown_flag_is_absent(self):
        assert hardware.check_cpu_flag("zzz_no_such_flag") is False

    def test_the_default_probe_returns_a_bool(self):
        assert isinstance(hardware.check_cpu_flag("avx2"), bool)


class TestCudaProbe:
    def test_the_probe_returns_a_bool(self):
        assert isinstance(hardware.is_cuda_available(), bool)

    def test_repeated_calls_agree(self):
        assert hardware.is_cuda_available() == hardware.is_cuda_available()
