"""Scanner tests for retry_loops, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_retry_loops,
)

from ._helpers import _write

# ---- retry_loop ----------------------------------------------------------


def test_retry_loop_busy_loop_flagged_p1(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def connect():
    while True:
        try:
            return do_connect()
        except ConnectionError:
            continue
""")
    findings = scan_retry_loops(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "busy_retry_loop"
    assert findings[0].severity == "P1"


def test_retry_loop_with_sleep_and_break_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import time

def connect():
    while True:
        try:
            result = do_connect()
            break
        except ConnectionError:
            time.sleep(1)
    return result
""")
    findings = scan_retry_loops(tmp_path)
    assert findings == []


def test_retry_loop_sleep_backed_no_break_flagged_low(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import time

def connect():
    while True:
        try:
            return do_connect()
        except ConnectionError:
            time.sleep(1)
""")
    findings = scan_retry_loops(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "unbounded_retry_loop"
    assert findings[0].severity == "Low"


def test_retry_loop_bounded_via_raise_not_break_is_clean(tmp_path: Path):
    """Regression (2026-07-22, false positive found in the wild in
    llm/claude_code_provider.py): a `while True:` retry loop that bounds itself by raising
    once an attempt counter is exceeded (checked BEFORE the loop's own try/except, so nothing
    inside the SAME loop catches it) is just as bounded as one using `break` -- the scanner
    used to only recognize `break`, flagging every raise-bounded retry loop as unbounded."""
    _write(tmp_path, "ok.py", """
import time

def connect(max_attempts=5):
    attempt = 0
    while True:
        attempt += 1
        if attempt > max_attempts:
            raise RuntimeError("exceeded max attempts")
        try:
            return do_connect()
        except ConnectionError:
            time.sleep(1)
""")
    findings = scan_retry_loops(tmp_path)
    assert findings == []


# ---- F60: retry loops, a break belonging to a NESTED loop ---------------------------


def test_retry_loops_ignores_a_break_in_a_nested_loop(tmp_path: Path):
    _write(tmp_path, "a.py", """
import time


def go():
    while True:
        try:
            for x in range(3):
                if x:
                    break
            work()
        except Exception:
            time.sleep(1)
""")
    assert len(scan_retry_loops(tmp_path)) == 1


def test_retry_loops_accepts_the_loops_own_break(tmp_path: Path):
    _write(tmp_path, "a.py", """
import time


def go():
    while True:
        try:
            work()
            break
        except Exception:
            time.sleep(1)
""")
    assert scan_retry_loops(tmp_path) == []
