"""Scanner tests for locals_globals_output, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_locals_globals_as_output,
)

from ._helpers import _write

# ---- locals_globals_as_output ------------------------------------------


def test_locals_globals_as_output_kwarg_flagged_p1(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def connect(session):
    session.apply(object=locals())
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"


def test_locals_globals_as_output_never_passed_to_a_call_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def debug_dump():
    snapshot = locals()
    return snapshot
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert findings == []


def test_locals_globals_as_output_read_only_builtin_consumer_is_clean(tmp_path: Path):
    """Regression (2026-07-22, false positive found in the wild in text/strings/__init__.py's
    __dir__()): passing globals()/locals() to a builtin that only ever READS its argument
    (set/list/dict/sorted/len/etc.) is never the "callee writes into it expecting write-back"
    bug this scanner targets."""
    _write(tmp_path, "ok.py", """
def __dir__():
    return sorted(set(globals()))
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert findings == []


def test_locals_globals_as_output_still_flags_positional_to_user_function(tmp_path: Path):
    """The read-only-builtin exclusion must not blind the scanner to the real bug shape:
    locals()/globals() passed positionally to a user-defined (non-builtin) function."""
    _write(tmp_path, "bad.py", """
def connect():
    read_config_file(path, locals())
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "Low"


def test_locals_globals_as_output_skips_unparseable_file(tmp_path: Path):
    """A file with a syntax error must be skipped (via _safe_parse returning None), not raise."""
    _write(tmp_path, "broken.py", """
def connect(:
    session.apply(object=locals())
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert findings == []
