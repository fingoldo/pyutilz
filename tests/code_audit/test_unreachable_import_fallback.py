"""Scanner tests for unreachable_import_fallback, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.unreachable_import_fallback import scan_unreachable_import_fallback

from ._helpers import _write

# ---- unreachable_import_fallback -----------------------------------------


def test_unreachable_import_fallback_flags_a_dead_guard(tmp_path: Path):
    """The handler cannot run, and its comment advertises a degradation path that never has."""
    _write(
        tmp_path,
        "mod.py",
        """
import struct

def parse(b):
    try:
        import struct
    except ImportError:
        return None
    return struct.unpack("<I", b)
""",
    )
    findings = scan_unreachable_import_fallback(tmp_path)
    assert len(findings) == 1
    assert "struct" in findings[0].detail


def test_unreachable_import_fallback_allows_an_optional_submodule(tmp_path: Path):
    """`import pkg.optional` can fail on a missing dependency where `import pkg` cannot. Comparing
    only the top-level package reported thirteen honest optional-dependency guards as dead."""
    _write(
        tmp_path,
        "mod.py",
        """
import pkg

def go():
    try:
        import pkg.optional
    except ImportError:
        return None
    return pkg.optional
""",
    )
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_allows_a_genuinely_optional_import(tmp_path: Path):
    _write(
        tmp_path,
        "mod.py",
        """
def go():
    try:
        import orjson
    except ImportError:
        import json as orjson
    return orjson
""",
    )
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_allows_the_type_checking_idiom(tmp_path: Path):
    """A TYPE_CHECKING-only import is not an import at runtime, so a try/except
    ImportError beside it is reachable, not dead.

    This is the standard shape for an optional dependency: import it under
    TYPE_CHECKING so a function can carry a real return annotation, and guard the
    real import so the package staying absent is handled. Flagging it told the
    author to delete a handler their code demonstrably needs.
    """
    _write(tmp_path, "mod.py", '''
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tiktoken

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODING = None
''')
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_allows_the_qualified_type_checking_form(tmp_path: Path):
    _write(tmp_path, "mod.py", '''
import typing

if typing.TYPE_CHECKING:
    import tiktoken

try:
    import tiktoken
except ImportError:
    tiktoken = None
''')
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_still_flags_a_real_dead_guard_alongside_type_checking(tmp_path: Path):
    """Non-vacuousness. The exemption is for imports INSIDE the TYPE_CHECKING
    block; a genuinely unconditional runtime import elsewhere in the same file
    still makes the handler dead, and a file that happens to use TYPE_CHECKING
    must not become exempt wholesale."""
    _write(tmp_path, "mod.py", '''
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections

import tiktoken

try:
    import tiktoken
except ImportError:
    tiktoken = None
''')
    assert len(scan_unreachable_import_fallback(tmp_path)) == 1


def test_unreachable_import_fallback_ignores_a_branch_import(tmp_path: Path):
    """An import inside ANY branch is conditional, so it cannot make a handler dead.

    ``if TYPE_CHECKING or X:`` may well execute at runtime -- but it may equally not, so the
    ``except ImportError`` below it is reachable and must not be reported as dead code. The rule
    only claims a handler is unreachable when the module is imported at the module body's top
    level, which is the only place an import is genuinely unconditional.
    """
    _write(tmp_path, "mod.py", '''
from typing import TYPE_CHECKING

SOMETHING = True

if TYPE_CHECKING or SOMETHING:
    import tiktoken

try:
    import tiktoken
except ImportError:
    tiktoken = None
''')
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_ignores_an_import_guarded_everywhere(tmp_path: Path):
    """If EVERY import of the module is itself inside a try, none of them is certain."""
    _write(
        tmp_path,
        "mod.py",
        """
try:
    import cupy
except ImportError:
    cupy = None

def go():
    try:
        import cupy
    except ImportError:
        return None
    return cupy
""",
    )
    assert scan_unreachable_import_fallback(tmp_path) == []


# ---- F15/F93/F185/F186: unreachable import fallback ---------------------------------


def test_unreachable_import_fallback_ignores_a_function_local_import(tmp_path: Path):
    _write(tmp_path, "a.py", """
def load():
    import numpy

    return numpy


try:
    import numpy
except ImportError:
    numpy = None
""")
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_ignores_a_platform_conditional_import(tmp_path: Path):
    _write(tmp_path, "a.py", """
import sys

if sys.platform == "win32":
    import winreg
else:
    winreg = None

try:
    import winreg
except ImportError:
    winreg = None
""")
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_walks_the_whole_try_body(tmp_path: Path):
    _write(tmp_path, "a.py", """
import numpy

try:
    if True:
        import numpy
except ImportError:
    numpy = None
""")
    assert len(scan_unreachable_import_fallback(tmp_path)) == 1


def test_unreachable_import_fallback_resolves_a_relative_import(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/util.py", "x = 1\n")
    _write(tmp_path, "pkg/m.py", """
from . import util

try:
    from . import util
except ImportError:
    util = None
""")
    assert len(scan_unreachable_import_fallback(tmp_path)) == 1
