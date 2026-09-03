"""Scanner tests for unpicklable_resource_state, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_unpicklable_resource_state,
)

from ._helpers import _write

# ---- unpicklable_resource_state ------------------------------------------


def test_unpicklable_resource_state_lock_without_getstate_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import threading

class Cache:
    def __init__(self):
        self._lock = threading.Lock()
        self._mem = {}
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.check == "unpicklable_resource_state"
    assert f.severity == "P2"
    assert "Cache" in f.detail
    assert "_lock" in f.detail


def test_unpicklable_resource_state_bare_rlock_import_flagged(tmp_path: Path):
    """A directly-imported ``RLock`` (not ``threading.RLock()``) must match on the constructor
    name alone, not require the ``threading.`` prefix."""
    _write(tmp_path, "bad.py", """
from threading import RLock

class Guarded:
    def __init__(self):
        self._lock = RLock()
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings


def test_unpicklable_resource_state_open_file_handle_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
class LogWriter:
    def __init__(self, path):
        self._fh = open(path, "w")
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings
    assert "_fh" in findings[0].detail


def test_unpicklable_resource_state_with_getstate_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import threading

class Cache:
    def __init__(self):
        self._lock = threading.Lock()
        self._mem = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert findings == [], f"class with __getstate__ must not be flagged; got {findings}"


def test_unpicklable_resource_state_plain_attribute_not_flagged(tmp_path: Path):
    """A class whose __init__ only assigns plain data (no lock/thread/file) must not be flagged."""
    _write(tmp_path, "ok.py", """
class Config:
    def __init__(self, name):
        self.name = name
        self.values = {}
        self.count = 0
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert findings == []


def test_unpicklable_resource_state_no_init_not_flagged(tmp_path: Path):
    """A class with no __init__ at all must not crash the scanner or be flagged."""
    _write(tmp_path, "ok.py", """
class Bare:
    x = 1
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert findings == []


def test_unpicklable_resource_state_thread_ctor_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import threading

class Worker:
    def __init__(self):
        self._thread = threading.Thread(target=self._run)

    def _run(self):
        pass
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings
    assert "_thread" in findings[0].detail


def test_unpicklable_resource_state_dotted_cuda_stream_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import torch

class GpuBuffer:
    def __init__(self):
        self._stream = torch.cuda.Stream()
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings


# ---- F68: unpicklable resource state needs an unambiguous constructor ---------------


def test_unpicklable_resource_state_ignores_a_domain_pool(tmp_path: Path):
    _write(tmp_path, "a.py", """
import catboost


class C:
    def __init__(self, X, y):
        self.train_pool = catboost.Pool(X, y)
""")
    assert scan_unpicklable_resource_state(tmp_path) == []


def test_unpicklable_resource_state_still_flags_a_multiprocessing_pool(tmp_path: Path):
    _write(tmp_path, "a.py", """
import multiprocessing


class C:
    def __init__(self):
        self.p = multiprocessing.Pool(4)
""")
    assert len(scan_unpicklable_resource_state(tmp_path)) == 1


def test_unpicklable_resource_state_follows_a_from_import(tmp_path: Path):
    _write(tmp_path, "a.py", """
from threading import Event


class C:
    def __init__(self):
        self.ev = Event()
""")
    assert len(scan_unpicklable_resource_state(tmp_path)) == 1
