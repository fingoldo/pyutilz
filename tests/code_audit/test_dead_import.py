"""Scanner tests for dead_import, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_possibly_dead_import,
)

from ._helpers import _write

# ---- possibly_dead_import --------------------------------------------------


def test_possibly_dead_import_flagged(tmp_path: Path):
    _write(tmp_path, "mod.py", """
import os
""")
    findings = scan_possibly_dead_import(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "possibly_dead_import"
    assert findings[0].severity == "Low"


def test_possibly_dead_import_bare_name_usage_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", """
import os

def f():
    return os.getcwd()
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_from_import_usage_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", """
from pathlib import Path

def f():
    return Path(".")
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_aliased_usage_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", """
import numpy as np

def f():
    return np.array([1, 2, 3])
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_facade_reexport_suppressed_by_corpus_attribute_access(tmp_path: Path):
    """The exact confirmed-real bug class this scanner exists for: `helper` is imported into
    `facade.py` purely to be re-exported, unused within facade.py itself, but consumed elsewhere
    as `facade.helper` -- must NOT be flagged."""
    _write(tmp_path, "facade.py", """
from _impl import helper
""")
    _write(tmp_path, "test_facade.py", """
import facade

def test_it():
    facade.helper()
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_dunder_all_reexport_is_clean(tmp_path: Path):
    _write(tmp_path, "facade.py", """
from _impl import helper

__all__ = ["helper"]
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_underscore_alias_skipped(tmp_path: Path):
    """`import x as _` is a conventional "explicitly discard" marker, not a name meant to be
    referenced -- must not be flagged as a dead import."""
    _write(tmp_path, "mod.py", """
import os as _
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_from_import_underscore_alias_skipped(tmp_path: Path):
    _write(tmp_path, "mod.py", """
from os import path as _
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_star_import_skipped(tmp_path: Path):
    """A star import can't be usage-checked by name -- must not crash or be flagged."""
    _write(tmp_path, "mod.py", """
from os import *
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_future_annotations_skipped(tmp_path: Path):
    """`from __future__ import annotations` is a compiler directive, never referenced as a
    name by design -- must never be flagged."""
    _write(tmp_path, "mod.py", """
from __future__ import annotations
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_noqa_line_skipped(tmp_path: Path):
    """A line already carrying `# noqa` has already been reviewed and explicitly exempted --
    re-flagging it is pure noise."""
    _write(tmp_path, "mod.py", """
import os  # noqa: F401
""")
    assert scan_possibly_dead_import(tmp_path) == []
    _write(tmp_path, "mod2.py", """
from os import path  # noqa: F401
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_multiline_block_reports_each_dead_name_separately(tmp_path: Path):
    """Real bug found dogfooding this scanner on pyutilz's own source (2026-08-04): a multi-line
    `from x import (a, b, c)` block used to report every dead name at the SAME `node.lineno` (the
    opening line of the statement), and the baseline-diff harness keys findings on exactly
    `(check, file, line)` -- so two independently-dead names in the same block collided onto one
    key and one silently masked the other. Confirmed in the wild: `database/db/__init__.py`'s
    multi-line `sql_helpers` import had THREE independently-unused names (`nu`,
    `MakeSetExcludedClause`, `update_if_now`) but only one finding ever surfaced. Each alias must
    be reported at its OWN line so distinct dead names never collide onto one key."""
    _write(tmp_path, "mod.py", """
from helper_module import (
    used_name,
    dead_one,
    dead_two,
)

def f():
    return used_name()
""")
    findings = scan_possibly_dead_import(tmp_path)
    flagged_names = {f.detail.split("binds '")[1].split("'")[0] for f in findings}
    flagged_lines = {f.line for f in findings}
    assert flagged_names == {"dead_one", "dead_two"}, findings
    assert len(flagged_lines) == 2, findings  # each dead name got its OWN line, not a shared one


def test_alias_own_lineno_fallback_when_ast_alias_lacks_lineno():
    """``ast.alias`` only gained ``lineno``/``col_offset`` in Python 3.10 (bpo-39235) -- on 3.8/3.9
    (this package's own supported floor), the getattr(alias, "lineno", node.lineno) fallback used
    to collapse every alias in a multi-line block back onto node.lineno, silently reproducing the
    exact collision test_possibly_dead_import_multiline_block_reports_each_dead_name_separately
    exists to prevent (confirmed failing in CI on Python 3.8, 2026-08-04). This test exercises the
    fallback path directly (source-text line scan) regardless of which Python actually runs it,
    by stripping ``lineno`` off a real parsed alias before calling the helper -- so the fallback's
    correctness doesn't depend on which interpreter happens to run the test suite."""
    import ast

    from pyutilz.dev.code_audit.dead_import import _alias_own_lineno

    src = "from helper_module import (\n    foo,\n    foo_bar,\n)\n"
    tree = ast.parse(src)
    src_lines = src.splitlines()
    node = tree.body[0]
    assert isinstance(node, ast.ImportFrom)

    class _Py38Alias:
        """Mimics ast.alias on Python <3.10: no lineno/col_offset attributes at all."""

        def __init__(self, real: ast.alias) -> None:
            self.name = real.name
            self.asname = real.asname

    claimed: set[int] = set()
    linenos = [_alias_own_lineno(_Py38Alias(alias), node, src_lines, claimed) for alias in node.names]  # type: ignore[arg-type]

    # Each alias gets its own distinct line, in source order -- including the substring-collision
    # case (foo vs foo_bar) where a naive `name in line` check would wrongly match "foo"'s line
    # against "foo_bar"'s text too.
    assert linenos == [2, 3]
    assert len(set(linenos)) == 2


def test_alias_own_lineno_fallback_skips_comment_line_repeating_the_name():
    """A same-block why-comment documenting an otherwise-flagged import routinely repeats the
    bound name in prose (this project's own convention: "consumed via `from x import foo`"),
    which would satisfy the fallback's name-pattern match on the COMMENT line, several lines
    before the scan ever reaches the real import line -- misattributing the finding. Confirmed
    live in CI on Python 3.9 (2026-08-27): mlframe's discretization/__init__.py and
    hermite_fe/__init__.py both carry exactly this comment style, and the fallback reported
    findings on the comment lines instead of the actual import lines."""
    import ast

    from pyutilz.dev.code_audit.dead_import import _alias_own_lineno

    src = "from helper_module import (\n    # consumed via `from x import foo` by tests/test_x.py\n    foo,\n)\n"
    tree = ast.parse(src)
    src_lines = src.splitlines()
    node = tree.body[0]
    assert isinstance(node, ast.ImportFrom)

    class _Py38Alias:
        def __init__(self, real: ast.alias) -> None:
            self.name = real.name
            self.asname = real.asname

    claimed: set[int] = set()
    lineno = _alias_own_lineno(_Py38Alias(node.names[0]), node, src_lines, claimed)  # type: ignore[arg-type]

    assert lineno == 3  # the real `foo,` line, not the line-2 comment mentioning "foo"


def test_possibly_dead_import_skips_file_with_syntax_error(tmp_path: Path):
    _write(tmp_path, "broken.py", "def f(:\n    pass\n")
    _write(tmp_path, "mod.py", """
import os
""")
    findings = scan_possibly_dead_import(tmp_path)
    assert len(findings) == 1
    assert findings[0].file == "mod.py"


def test_possibly_dead_import_no_imports_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", """
def f():
    return 1
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_relative_import_with_no_module_skipped(tmp_path: Path):
    """`from . import x` (ImportFrom with module=None) is a relative package import -- skipped
    rather than crashing on the None module attribute."""
    _write(tmp_path, "mod.py", """
from . import helper
""")
    findings = scan_possibly_dead_import(tmp_path)
    assert findings == []
