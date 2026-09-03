"""Scanner tests for per_call_state_on_shared_instance, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.per_call_state_on_shared_instance import scan_per_call_state_on_shared_instance

from ._helpers import _write

# ---- F103/F104/F105: per-call state on a shared instance ----------------------------


def test_per_call_state_needs_a_module_level_registry(tmp_path: Path):
    _write(tmp_path, "a.py", """
def labels():
    labels = ["Worker", "Other"]
    return labels


class Worker:
    async def run(self):
        self.last_usage = 1
""")
    assert scan_per_call_state_on_shared_instance(tmp_path) == []


def test_per_call_state_does_not_take_blocking_io_for_a_lock(tmp_path: Path):
    _write(tmp_path, "a.py", """
_PROVIDERS = ["Worker"]


class Worker:
    async def run(self):
        with self.blocking_io:
            self.last_usage = 1
""")
    assert len(scan_per_call_state_on_shared_instance(tmp_path)) == 1


def test_per_call_state_accepts_a_real_lock(tmp_path: Path):
    _write(tmp_path, "a.py", """
_PROVIDERS = ["Worker"]


class Worker:
    async def run(self):
        with self._lock:
            self.last_usage = 1
""")
    assert scan_per_call_state_on_shared_instance(tmp_path) == []


def test_per_call_state_ignores_an_annotation_only_attribute(tmp_path: Path):
    _write(tmp_path, "a.py", """
_PROVIDERS = ["Worker"]


class Worker:
    async def run(self):
        self.last_usage: int
""")
    assert scan_per_call_state_on_shared_instance(tmp_path) == []
