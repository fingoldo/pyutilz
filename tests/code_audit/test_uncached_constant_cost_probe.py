"""Scanner tests for uncached_constant_cost_probe, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.uncached_constant_cost_probe import scan_uncached_constant_cost_probe

from ._helpers import _write

# ---- F89/F90/F180/F181/F182/F188: uncached constant-cost probe ----------------------


def test_uncached_constant_cost_probe_sees_the_path_mkdir_spelling(tmp_path: Path):
    _write(tmp_path, "a.py", """
from pathlib import Path


def ensure(p=Path("x")):
    p.mkdir(parents=True, exist_ok=True)
""")
    assert len(scan_uncached_constant_cost_probe(tmp_path)) == 1


def test_uncached_constant_cost_probe_reports_a_nested_def_once(tmp_path: Path):
    _write(tmp_path, "b.py", """
import subprocess


def outer():
    def inner():
        return subprocess.run(["x"])

    return inner
""")
    assert len(scan_uncached_constant_cost_probe(tmp_path)) == 1


def test_uncached_constant_cost_probe_matches_decorators_structurally(tmp_path: Path):
    _write(tmp_path, "c.py", """
import subprocess


@app.route("/cache")
def probe():
    return subprocess.run(["x"])
""")
    assert len(scan_uncached_constant_cost_probe(tmp_path)) == 1


def test_uncached_constant_cost_probe_ignores_a_local_function_named_run(tmp_path: Path):
    _write(tmp_path, "d.py", """
def run():
    return 1


def probe():
    return run()
""")
    assert scan_uncached_constant_cost_probe(tmp_path) == []


def test_uncached_constant_cost_probe_needs_the_global_to_be_written(tmp_path: Path):
    _write(tmp_path, "g.py", """
import subprocess

counter = 0


def probe():
    global counter
    return subprocess.run(["x"])
""")
    assert len(scan_uncached_constant_cost_probe(tmp_path)) == 1


def test_uncached_constant_cost_probe_accepts_a_hand_rolled_memo(tmp_path: Path):
    _write(tmp_path, "h.py", """
import subprocess

_memo = None


def probe():
    global _memo
    if _memo is None:
        _memo = subprocess.run(["x"])
    return _memo
""")
    assert scan_uncached_constant_cost_probe(tmp_path) == []


def test_uncached_constant_cost_probe_still_honours_lru_cache(tmp_path: Path):
    _write(tmp_path, "i.py", """
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def probe():
    return subprocess.run(["x"])
""")
    assert scan_uncached_constant_cost_probe(tmp_path) == []
