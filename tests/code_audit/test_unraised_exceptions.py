"""Scanner tests for unraised_exceptions, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_unraised_exceptions,
)

from ._helpers import _write

# ---- unraised_exception_class ---------------------------------------------


def test_unraised_exception_class_never_raised_flagged(tmp_path: Path):
    _write(tmp_path, "exc.py", """
class LLMTruncationError(Exception):
    pass
""")
    findings = scan_unraised_exceptions(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P2"


def test_unraised_exception_class_raised_in_different_file_is_clean(tmp_path: Path):
    _write(tmp_path, "exc.py", """
class RetryableError(Exception):
    pass
""")
    _write(tmp_path, "use.py", """
from exc import RetryableError

def f():
    raise RetryableError("boom")
""")
    findings = scan_unraised_exceptions(tmp_path)
    assert findings == []


def test_unraised_exception_class_base_covered_by_raised_subclass_is_clean(tmp_path: Path):
    """2026-08-03 glossum audit: a base class's error-signaling contract fires
    whenever ANY in-tree subclass is raised -- `except BaseError:` still
    catches a raised `SpecificError(BaseError)`. Flagging the never-directly-
    raised ancestor here was a false positive (`GlossumError`/`ProviderError`
    in glossum's exceptions.py, never raised directly, only via subclasses
    `JobLockConflictError`/`LLMProviderError` respectively)."""
    _write(tmp_path, "exc.py", """
class BaseError(Exception):
    pass

class SpecificError(BaseError):
    pass
""")
    _write(tmp_path, "use.py", """
from exc import SpecificError

def f():
    raise SpecificError("boom")
""")
    findings = scan_unraised_exceptions(tmp_path)
    assert findings == []


def test_unraised_exception_class_base_without_raised_subclass_still_flagged(tmp_path: Path):
    """The base-class exemption above must not blanket-suppress an unrelated
    sibling base whose OWN subclasses are also never raised."""
    _write(tmp_path, "exc.py", """
class BaseError(Exception):
    pass

class SpecificError(BaseError):
    pass

class OtherBaseError(Exception):
    pass

class OtherSpecificError(OtherBaseError):
    pass
""")
    _write(tmp_path, "use.py", """
from exc import SpecificError

def f():
    raise SpecificError("boom")
""")
    findings = scan_unraised_exceptions(tmp_path)
    flagged = {f.snippet for f in findings}
    assert any("OtherBaseError" in s for s in flagged)
    assert any("OtherSpecificError" in s for s in flagged)
    assert not any("BaseError" in s and "OtherBaseError" not in s for s in flagged)
    assert not any("SpecificError" in s and "OtherSpecificError" not in s for s in flagged)


# ---- F192: same-named exception classes in different files -------------------------


def test_unraised_exceptions_keys_classes_per_file(tmp_path: Path):
    _write(tmp_path, "e1.py", "class DupError(Exception):\n    pass\n")
    _write(tmp_path, "e2.py", "class DupError(Exception):\n    pass\n")
    findings = scan_unraised_exceptions(tmp_path)
    assert sorted(f.file for f in findings) == ["e1.py", "e2.py"]


def test_unraised_exceptions_accepts_a_raised_class(tmp_path: Path):
    _write(tmp_path, "e1.py", """
class DupError(Exception):
    pass


def go():
    raise DupError()
""")
    assert scan_unraised_exceptions(tmp_path) == []
