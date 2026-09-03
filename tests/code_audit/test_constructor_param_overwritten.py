"""Scanner tests for constructor_param_overwritten, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.constructor_param_overwritten import scan_constructor_param_overwritten

from ._helpers import _write

# ---- constructor_param_overwritten ---------------------------------------


def test_constructor_param_overwritten_follows_one_call_hop(tmp_path: Path):
    """The worked example assigns through a second method: `_refresh_rate` reads config and calls
    `update_rate(rate)`, which does the assignment. Requiring both in one statement missed it."""
    _write(tmp_path, "bucket.py", """
class TokenBucket:
    def __init__(self, rate):
        self._rate = rate

    def update_rate(self, rate):
        self._rate = rate

    def _refresh_rate(self):
        self.update_rate(cfg().get("traffic", "max_rps", 10.0, float))
""")
    findings = scan_constructor_param_overwritten(tmp_path)
    assert len(findings) == 1
    assert "_refresh_rate" in findings[0].detail
    assert "update_rate" in findings[0].detail


def test_constructor_param_overwritten_ignores_a_stable_attribute(tmp_path: Path):
    _write(tmp_path, "bucket.py", """
class Plain:
    def __init__(self, rate):
        self._rate = rate

    def use(self):
        return self._rate * 2
""")
    assert scan_constructor_param_overwritten(tmp_path) == []


def test_constructor_param_overwritten_ignores_a_reassignment_not_from_config(tmp_path: Path):
    """Reassigning from an argument is ordinary mutation, not the deployment overriding a test."""
    _write(tmp_path, "bucket.py", """
class Plain:
    def __init__(self, rate):
        self._rate = rate

    def set_rate(self, rate):
        self._rate = rate
""")
    assert scan_constructor_param_overwritten(tmp_path) == []
