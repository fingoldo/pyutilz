"""Scanner tests for lazy_log_assertion, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.lazy_log_assertion import scan_lazy_log_assertion

from ._helpers import _write

# ---- lazy_log_assertion --------------------------------------------------


def test_lazy_log_assertion_flags_a_formatted_expectation(tmp_path: Path):
    _write(tmp_path, "prod.py", """
def go(label, r, t):
    log.warning("%s: reached only %d/%d", label, r, t)
""")
    _write(tmp_path, "test_x.py", """
def test_shortfall_is_warned():
    assert "reached only 0/3" in str(log.warning.call_args)
""")
    findings = scan_lazy_log_assertion(tmp_path)
    assert len(findings) == 1
    assert "reached only 0/3" in findings[0].detail


def test_lazy_log_assertion_ignores_an_fstring_rendering(tmp_path: Path):
    """f-strings format EAGERLY, so the values DO reach the record and the assertion can match.
    Both of this rule's first hits on a real repository were this."""
    _write(tmp_path, "prod.py", """
def go(n):
    log.info(f"Found {n} on-disk checkpoint(s) -- these will resume")
""")
    _write(tmp_path, "test_x.py", """
def test_inventory_is_logged():
    assert any("Found 2 on-disk checkpoint" in str(c) for c in log.info.call_args_list)
""")
    assert scan_lazy_log_assertion(tmp_path) == []


def test_lazy_log_assertion_ignores_a_bare_value(tmp_path: Path):
    """`"j1"` is an id the test supplied, with no message text around it -- production logs it
    through an f-string, so it really is in args[0]."""
    _write(tmp_path, "prod.py", """
def go(jid, e):
    log.warning(f"Reconcile sample failed for {jid}: {e}")
""")
    _write(tmp_path, "test_x.py", """
def test_error_logged():
    assert "j1" in log.warning.call_args[0][0]
""")
    assert scan_lazy_log_assertion(tmp_path) == []


def test_lazy_log_assertion_ignores_a_format_that_carries_its_own_digit(tmp_path: Path):
    _write(tmp_path, "prod.py", """
def go(host):
    log.warning("HTTP 429 from %s", host)
""")
    _write(tmp_path, "test_x.py", """
def test_rate_limit_logged():
    assert "HTTP 429 from" in str(log.warning.call_args)
""")
    assert scan_lazy_log_assertion(tmp_path) == []
