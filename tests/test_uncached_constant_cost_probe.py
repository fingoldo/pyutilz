"""Unit tests for the ``uncached_constant_cost_probe`` scanner.

Every defective fixture is a reduction of a real 2026-09-02 performance finding, written to a tmp
tree rather than by breaking the shipped source, and paired with the cached shape that must be
silent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from pyutilz.dev.code_audit import scan_uncached_constant_cost_probe

# 10/F02 [High]: gpu_capability_summary re-shells out to nvidia-smi on every call (64 ms per
# dispatch decision).
_NVIDIA_SMI_DEFECTIVE = '''
import subprocess


def gpu_capability_summary():
    """Query the device's compute capability."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv"], capture_output=True)
    return out.stdout.decode()
'''

_NVIDIA_SMI_FIXED = '''
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def gpu_capability_summary():
    """Query the device's compute capability, once per process."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv"], capture_output=True)
    return out.stdout.decode()
'''

# 10/F04 [High]: is_cuda_available() re-probes numba on every call (18-24 us per dispatch decision).
_CUDA_PROBE_DEFECTIVE = '''
import importlib


def is_cuda_available():
    try:
        cuda = importlib.import_module("numba.cuda")
    except ImportError:
        return False
    return cuda.is_available()
'''

# 10/F09: cache_dir() makedirs(exist_ok=True) on every call for a directory that already exists.
_CACHE_DIR_DEFECTIVE = '''
import os


def cache_dir(root="/var/cache"):
    os.makedirs(root, exist_ok=True)
    return root
'''

# The module-level-memo form of the same fix must be recognised as cached.
_CACHE_DIR_MEMOIZED = '''
import os

_CACHE_DIR = None


def cache_dir(root="/var/cache"):
    global _CACHE_DIR
    if _CACHE_DIR is None:
        os.makedirs(root, exist_ok=True)
        _CACHE_DIR = root
    return _CACHE_DIR
'''

# 10/F11: _pid_alive constructs a fresh ctypes.WinDLL("kernel32") on every call. The scanner flags
# the function even though the pid check itself genuinely must re-run -- documented in the scanner's
# own docstring as exactly why it warns rather than blocks.
_PID_ALIVE_DEFECTIVE = '''
import ctypes


def _pid_alive(pid=0):
    kernel32 = ctypes.WinDLL("kernel32")
    handle = kernel32.OpenProcess(0x1000, False, pid)
    return bool(handle)
'''

# A function whose answer varies with a REQUIRED argument is out of scope by construction.
_REQUIRED_ARG = '''
import subprocess


def run_tool(tool_name):
    return subprocess.run([tool_name], capture_output=True)
'''


def _scan(tmp_path: Path, name: str, source: str):
    (tmp_path / name).write_text(textwrap.dedent(source), encoding="utf-8")
    return scan_uncached_constant_cost_probe(tmp_path)


def test_flags_subprocess_probe_repeated_per_call(tmp_path: Path) -> None:
    findings = _scan(tmp_path, "gpu.py", _NVIDIA_SMI_DEFECTIVE)
    assert len(findings) == 1
    assert findings[0].check == "uncached_constant_cost_probe"
    assert findings[0].severity == "P2"
    assert "gpu_capability_summary()" in findings[0].detail
    assert "subprocess.run" in findings[0].detail


def test_lru_cache_clears_the_finding(tmp_path: Path) -> None:
    assert _scan(tmp_path, "gpu.py", _NVIDIA_SMI_FIXED) == []


def test_flags_capability_probe_via_import_module(tmp_path: Path) -> None:
    findings = _scan(tmp_path, "dispatch.py", _CUDA_PROBE_DEFECTIVE)
    assert len(findings) == 1
    assert "is_cuda_available()" in findings[0].detail


def test_flags_makedirs_on_every_call(tmp_path: Path) -> None:
    findings = _scan(tmp_path, "paths.py", _CACHE_DIR_DEFECTIVE)
    assert len(findings) == 1
    assert "os.makedirs" in findings[0].detail


def test_module_level_memo_counts_as_cached(tmp_path: Path) -> None:
    assert _scan(tmp_path, "paths.py", _CACHE_DIR_MEMOIZED) == []


def test_flags_ctypes_handle_rebuilt_per_call(tmp_path: Path) -> None:
    findings = _scan(tmp_path, "cache_base.py", _PID_ALIVE_DEFECTIVE)
    assert len(findings) == 1
    assert "ctypes.WinDLL" in findings[0].detail


def test_function_with_a_required_argument_is_out_of_scope(tmp_path: Path) -> None:
    assert _scan(tmp_path, "tools.py", _REQUIRED_ARG) == []


def test_one_finding_per_function(tmp_path: Path) -> None:
    """Two probes in one body are one triage item, not two."""
    source = '''
    import os
    import subprocess


    def bootstrap():
        os.makedirs("/tmp/x", exist_ok=True)
        return subprocess.run(["true"])
    '''
    assert len(_scan(tmp_path, "boot.py", source)) == 1


def test_scanner_is_opt_in_only() -> None:
    """Warn-only: it must not join the default sweep, whose findings block via the baseline test."""
    from pyutilz.dev.code_audit.registry import OPT_IN_ONLY, get_scanners

    assert "uncached_constant_cost_probe" in get_scanners()
    assert "uncached_constant_cost_probe" in OPT_IN_ONLY
