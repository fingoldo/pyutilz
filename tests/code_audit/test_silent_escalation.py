"""Scanner tests for silent_escalation, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_log_only_except,
)

from ._helpers import _write

# ---- log_only_except -------------------------------------------------------


def test_log_only_except_flags_when_convention_used(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import logging
logger = logging.getLogger(__name__)

def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
""")
    findings = scan_log_only_except(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "log_only_except"
    assert findings[0].severity == "P2"


def test_log_only_except_escalated_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
        result.validation_errors.append(f"write_failed: {e}")
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_reraise_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
        raise
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_no_convention_in_file_is_clean(tmp_path: Path):
    """The escalation convention (validation_errors / errors / etc) isn't
    used anywhere in the file, so silence here is a design choice, not a
    detected gap."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def save():
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_no_log_call_not_double_flagged(tmp_path: Path):
    """No log call at all is scan_broad_except_swallows' territory, not this scanner's."""
    _write(tmp_path, "ok.py", """
def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception:
        pass
""")
    assert scan_log_only_except(tmp_path) == []


# ---- log_only_except: alternate escalation conventions (2026-07 FP fix) ----


def test_log_only_except_error_counter_increment_is_clean(tmp_path: Path):
    """``stats["errors"] += 1`` / ``total_errors += len(batch)`` is a
    legitimate escalation convention this scanner didn't originally
    recognise -- the file's OWN naming (``validation_errors`` elsewhere)
    triggers the file-level scope gate, but the actual handler escalates
    via a differently-shaped counter."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def process(items):
    validation_errors = []
    stats = {"errors": 0}
    for item in items:
        try:
            do_thing(item)
        except Exception as e:
            logger.warning("failed: %s", e)
            stats["errors"] += 1
    return stats
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_return_false_sentinel_is_clean(tmp_path: Path):
    """A Phase0-style ``return False`` on failure is a caller-visible
    escalation contract even though nothing gets appended to a list."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def run_test(errors):
    try:
        do_check()
        return True
    except Exception as e:
        logger.warning("check failed: %s", e)
        return False
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_return_error_dict_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def run(errors):
    try:
        return {"result": do_thing()}
    except Exception as e:
        logger.warning("failed: %s", e)
        return {"error": str(e)}
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_warn_method_call_is_clean(tmp_path: Path):
    """``results.warn(...)`` -- a distinct object-method escalation
    convention -- is recognised regardless of the base object's name."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def run(errors, results):
    try:
        do_thing()
    except Exception as e:
        logger.warning("failed: %s", e)
        results.warn(f"skipped: {e}")
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_local_error_var_assignment_is_clean(tmp_path: Path):
    """Stashing the failure into a local ``error_message``-named variable
    (persisted after the loop) is a real escalation path even without an
    immediate append/return."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def run(errors):
    error_message = None
    try:
        do_thing()
    except Exception as e:
        logger.warning("failed: %s", e)
        error_message = str(e)
    return error_message
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_no_escalation_at_all_still_flagged(tmp_path: Path):
    """None of the recognised escalation conventions apply -- must still
    be flagged (the fix must not become a blanket exemption)."""
    _write(tmp_path, "bad.py", """
import logging
logger = logging.getLogger(__name__)

def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
""")
    findings = scan_log_only_except(tmp_path)
    assert findings, "handler with no escalation path at all must still be flagged"
