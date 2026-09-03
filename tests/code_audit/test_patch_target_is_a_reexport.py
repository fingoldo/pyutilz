"""Scanner tests for patch_target_is_a_reexport, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.patch_target_is_a_reexport import scan_patch_target_is_a_reexport

from ._helpers import _write

# ---- patch_target_is_a_reexport ------------------------------------------

_REEXPORT_IMPL = """
def fetch(key):
    return key


def run(key):
    return fetch(key)
"""

_REEXPORT_TEST = """
from unittest.mock import patch


def test_run():
    with patch("facade.fetch", return_value="mocked"):
        assert facade.run("k") == "mocked"
"""


def test_patch_target_is_a_reexport_flags_the_canonical_case(tmp_path: Path):
    """The facade re-exports `fetch`; the caller in `_impl` calls it directly.

    Verified by construction rather than by reasoning: run as written, `run("k")` returns "k" and
    not the mock's "mocked" -- the patch rebinds a name nothing reads at call time.
    """
    _write(tmp_path, "_impl.py", _REEXPORT_IMPL)
    _write(tmp_path, "facade.py", "from _impl import fetch, run")
    _write(tmp_path, "test_facade.py", _REEXPORT_TEST)
    findings = scan_patch_target_is_a_reexport(tmp_path)
    assert len(findings) == 1, findings
    assert "_impl.fetch" in findings[0].detail


def test_patch_target_is_a_reexport_accepts_a_call_through_the_facade(tmp_path: Path):
    """The definer calls its own function back THROUGH the facade, so the patch reaches it.

    This looks circular and is exactly what the scraper codebase does: `_load_tracked_active_from_db`
    is defined in `rescan_active_scans`, re-exported by `rescan_active_jobs`, and called from the
    definer as `_facade._load_tracked_active_from_db(db)`. Counting that attribute call as a direct
    one reported fifteen correct tests as vacuous, so this is the case that pins the distinction.
    """
    _write(
        tmp_path,
        "_impl.py",
        """
import facade


def fetch(key):
    return key


def run(key):
    return facade.fetch(key)
""",
    )
    _write(tmp_path, "facade.py", "from _impl import fetch, run")
    _write(tmp_path, "test_facade.py", _REEXPORT_TEST)
    assert scan_patch_target_is_a_reexport(tmp_path) == []


def test_patch_target_is_a_reexport_accepts_an_optional_accelerator(tmp_path: Path):
    """A name imported in a `try` and DEFINED in the fallback is not a re-export.

    The optional-accelerator shape: import the fast implementation if it is there, otherwise define
    a pure-Python version under the same name. The module owns the name either way, so patching it
    is the ordinary correct case.
    """
    _write(tmp_path, "_impl.py", _REEXPORT_IMPL)
    # `_fast` has to EXIST and call `fetch` itself, or the rule stops for want of a module and the
    # condition under test is never reached -- the first version of this test proved nothing.
    _write(tmp_path, "_fast.py", _REEXPORT_IMPL)
    _write(
        tmp_path,
        "facade.py",
        """
from _impl import run

try:
    from _fast import fetch
except ImportError:

    def fetch(key):
        return key
""",
    )
    _write(tmp_path, "test_facade.py", _REEXPORT_TEST)
    assert scan_patch_target_is_a_reexport(tmp_path) == []


def test_patch_target_is_a_reexport_accepts_a_defined_name(tmp_path: Path):
    """Patching a name the module DEFINES is the ordinary, correct case."""
    _write(
        tmp_path,
        "facade.py",
        """
def fetch(key):
    return key


def run(key):
    return fetch(key)
""",
    )
    _write(tmp_path, "test_facade.py", _REEXPORT_TEST)
    assert scan_patch_target_is_a_reexport(tmp_path) == []


def test_patch_target_is_a_reexport_accepts_a_facade_that_uses_the_name(tmp_path: Path):
    """If the facade calls it too, the patch lands on something real, and which call the test
    means is not decidable from here."""
    _write(tmp_path, "_impl.py", _REEXPORT_IMPL)
    _write(
        tmp_path,
        "facade.py",
        """
from _impl import fetch, run


def warm(key):
    return fetch(key)
""",
    )
    _write(tmp_path, "test_facade.py", _REEXPORT_TEST)
    assert scan_patch_target_is_a_reexport(tmp_path) == []


def test_patch_target_is_a_reexport_only_reads_test_files(tmp_path: Path):
    """Production code that patches something is doing it deliberately, not asserting on it."""
    _write(tmp_path, "_impl.py", _REEXPORT_IMPL)
    _write(tmp_path, "facade.py", "from _impl import fetch, run")
    _write(
        tmp_path,
        "harness.py",
        """
from unittest.mock import patch


def dry_run():
    with patch("facade.fetch", return_value="mocked"):
        return facade.run("k")
""",
    )
    assert scan_patch_target_is_a_reexport(tmp_path) == []


# ---- F20/F169: patch target is a re-export ------------------------------------------


def _reexport_package(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir(exist_ok=True)
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/_impl.py", """
def fetch():
    return 1


def run():
    return fetch()
""")
    _write(tmp_path, "pkg/facade.py", "from ._impl import fetch\n")


def test_patch_target_is_a_reexport_resolves_a_relative_import_in_a_plain_module(tmp_path: Path):
    _reexport_package(tmp_path)
    _write(tmp_path, "test_a.py", """
from unittest.mock import patch

def test_x():
    with patch("pkg.facade.fetch"):
        pass
""")
    assert len(scan_patch_target_is_a_reexport(tmp_path)) == 1


def test_patch_target_is_a_reexport_matches_patch_object(tmp_path: Path):
    _reexport_package(tmp_path)
    _write(tmp_path, "test_a.py", """
from unittest.mock import patch
from pkg import facade

def test_x():
    with patch.object(facade, "fetch"):
        pass
""")
    assert len(scan_patch_target_is_a_reexport(tmp_path)) == 1


def test_patch_target_is_a_reexport_stays_silent_when_the_facade_calls_it(tmp_path: Path):
    (tmp_path / "pkg").mkdir(exist_ok=True)
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/_impl.py", "def fetch():\n    return 1\n")
    _write(tmp_path, "pkg/facade.py", """
from ._impl import fetch


def go():
    return fetch()
""")
    _write(tmp_path, "test_a.py", """
from unittest.mock import patch

def test_x():
    with patch("pkg.facade.fetch"):
        pass
""")
    assert scan_patch_target_is_a_reexport(tmp_path) == []
