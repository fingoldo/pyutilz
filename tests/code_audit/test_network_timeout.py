"""Scanner tests for network_timeout, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_missing_network_timeout,
)

from ._helpers import _write

# ---- missing_network_timeout -------------------------------------------


def test_missing_network_timeout_flags_bare_get(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import requests

def f():
    return requests.get("http://example.com")
""")
    findings = scan_missing_network_timeout(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "missing_network_timeout"


def test_missing_network_timeout_with_timeout_kwarg_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import requests

def f():
    return requests.get("http://example.com", timeout=5)
""")
    findings = scan_missing_network_timeout(tmp_path)
    assert findings == []


# ---- F194: a directly imported urlopen ----------------------------------------------


def test_network_timeout_sees_a_directly_imported_urlopen(tmp_path: Path):
    _write(tmp_path, "a.py", """
from urllib.request import urlopen

def f(u):
    return urlopen(u).read()
""")
    assert len(scan_missing_network_timeout(tmp_path)) == 1


def test_network_timeout_ignores_a_local_function_named_get(tmp_path: Path):
    _write(tmp_path, "a.py", """
def get(u):
    return u

def f(u):
    return get(u)
""")
    assert scan_missing_network_timeout(tmp_path) == []
