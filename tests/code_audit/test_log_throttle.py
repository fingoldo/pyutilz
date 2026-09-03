"""Scanner tests for log_throttle, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_unthrottled_hot_loop_log,
)

from ._helpers import _write

# ---- unthrottled_hot_loop_log ---------------------------------------------


def test_unthrottled_hot_loop_log_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def scan(items, log):
    for item in items:
        if item.bad:
            log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "unthrottled_hot_loop_log"
    assert findings[0].severity == "P2"


def test_unthrottled_hot_loop_log_throttled_guard_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def scan(items, log):
    for item in items:
        if item.bad:
            if _log_throttle("key"):
                log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_modulo_guard_is_clean(tmp_path: Path):
    _write(tmp_path, "ok2.py", """
def scan(items, log):
    for i, item in enumerate(items):
        if i % 100 == 0:
            log.warning("progress %s", i)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_outside_loop_is_clean(tmp_path: Path):
    _write(tmp_path, "ok3.py", """
def scan(item, log):
    if item.bad:
        log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_debug_call_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok4.py", """
def scan(items, log):
    for item in items:
        log.debug("processing %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_while_loop_flagged(tmp_path: Path):
    _write(tmp_path, "bad2.py", """
def scan(get_next, log):
    while True:
        item = get_next()
        if item.bad:
            log.error("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1


def test_unthrottled_hot_loop_log_while_true_bounded_by_internal_retry_break_is_clean(tmp_path: Path):
    """Real false-positive pattern found dogfooding this scanner on pyutilz's own source
    (2026-08-04, ``dev/logginglib.py``'s ``debugged()`` decorator): a `while True:` retry loop
    whose only exit is `if not interactive or attempts >= max_retries: raise` inside the except
    block is EXACTLY as bounded as the already-recognized `while attempts < max_retries:` idiom --
    the bound just lives in an internal break/raise condition instead of the loop's own
    (constant, uninformative) test. Must stay clean; the sibling test above
    (`test_unthrottled_hot_loop_log_while_loop_flagged`) confirms a genuinely-unbounded
    `while True:` loop with no such internal bound is still flagged."""
    _write(tmp_path, "retry.py", """
def call_with_retry(func, log, max_retries=3):
    attempts = 0
    while True:
        try:
            return func()
        except Exception as e:
            log.exception(e)
            attempts += 1
            if attempts >= max_retries:
                raise
""")
    assert scan_unthrottled_hot_loop_log(tmp_path) == []


def test_unthrottled_hot_loop_log_else_branch_flagged(tmp_path: Path):
    """An unguarded log call in the `else` of an if/else, inside a loop, must still be flagged --
    only the `if`'s own throttle-guarded body is exempt, not its sibling `else`."""
    _write(tmp_path, "bad3.py", """
def scan(items, log):
    for item in items:
        if item.ok:
            pass
        else:
            log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1


def test_unthrottled_hot_loop_log_attribute_receiver_and_throttle_call_covered(tmp_path: Path):
    """Both the log receiver AND the throttle-check call are attribute access
    (self.log.warning(...), self.limiter.should_throttle(...)) -- exercises the Attribute
    branches of _call_name/_is_log_call, not just the bare-Name ones."""
    _write(tmp_path, "ok5.py", """
class Scanner:
    def scan(self, items):
        for item in items:
            if self.limiter.should_throttle(item):
                self.log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_non_log_named_receiver_not_flagged(tmp_path: Path):
    """A `.warning(...)` call on a receiver whose name doesn't end in log/logger (e.g. a
    warnings-module-shaped object) is out of scope for this scanner -- not every `.warning(...)`
    call is a logger call."""
    _write(tmp_path, "ok6.py", """
def scan(items, notifier):
    for item in items:
        notifier.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_receiver_from_a_call_not_flagged(tmp_path: Path):
    """A `.warning(...)` call whose receiver is itself a Call (e.g. `get_logger().warning(...)`)
    can't be name-matched by this scanner's simple Name/Attribute receiver check -- exercises the
    receiver_name-stays-None fallthrough."""
    _write(tmp_path, "ok7.py", """
def scan(items):
    for item in items:
        get_logger().warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_skips_file_with_syntax_error(tmp_path: Path):
    """A file with a syntax error must be skipped, not crash the whole scan -- and a sibling
    valid file in the same directory must still be scanned normally."""
    _write(tmp_path, "broken.py", "def f(:\n    pass\n")
    _write(tmp_path, "bad4.py", """
def scan(items, log):
    for item in items:
        log.error("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1
    assert findings[0].file == "bad4.py"


def test_unthrottled_hot_loop_log_guard_call_via_subscript_not_a_throttle_hint(tmp_path: Path):
    """The guard's Call func is neither a bare Name nor an Attribute (e.g. a subscripted
    dispatch-table lookup) -- can't name-match it as a throttle hint, so the log call inside
    stays flagged (exercises _call_name's final None fallthrough)."""
    _write(tmp_path, "bad5.py", """
def scan(items, log, checks):
    for item in items:
        if checks["ok"](item):
            log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1
