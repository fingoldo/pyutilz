"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- non-ASCII console/log output -----------------------------------------

_LOG_METHOD_NAMES = {"info", "warning", "error", "critical", "debug", "exception", "log"}


def _is_console_call(call: ast.Call) -> bool:
    """True for a ``print(...)`` call or a logging-method call (``logger.info`` etc.)."""
    func = call.func
    if isinstance(func, ast.Name) and func.id == "print":
        return True
    if isinstance(func, ast.Attribute) and func.attr in _LOG_METHOD_NAMES:
        return True
    return False


def _first_str_arg_value(call: ast.Call) -> Optional[str]:
    """If the first positional arg is a string literal (plain or f-string), return it."""
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    if isinstance(first, ast.JoinedStr):
        parts = [v.value for v in first.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        return "".join(parts) if parts else None
    return None


def _has_non_ascii(s: str) -> bool:
    """True if ``s`` contains any character outside the printable-ASCII range."""
    return any(ord(c) > 127 for c in s)


def _has_stdio_utf8_reconfigure(source: str) -> bool:
    """True if ``source`` calls ``sys.stdout.reconfigure(encoding=...)`` /
    ``sys.stderr.reconfigure(encoding=...)`` anywhere in the module -- the standard idiom for
    fixing exactly the failure mode this scanner exists to catch (Windows' cp1251/cp1252 default
    console encoding). A file that already forces UTF-8 stdio at its own entry point cannot hit
    the ``UnicodeEncodeError`` the scanner warns about, so flagging its non-ASCII literals is a
    false positive. A plain substring check (not an AST walk restricted to module-top-level) is
    deliberate: the reconfigure call is just as effective inside an ``if __name__ ==
    "__main__":`` guard or a ``main()`` function that runs before any print/log call, which is
    the common placement in this codebase's scripts.
    """
    return "reconfigure(encoding" in source and (".stdout.reconfigure(encoding" in source or ".stderr.reconfigure(encoding" in source)


def _ancestor_package_has_stdio_utf8_reconfigure(py: Path, root: Path) -> bool:
    """True if any ``__init__.py`` from ``py``'s own package up through ``root`` reconfigures
    stdio to UTF-8. A package ``__init__.py`` runs on the FIRST ``import <package>`` regardless
    of entry point (CLI, an API server under uvicorn, a test runner, a script that merely
    imports the package) -- so a reconfigure call there protects every module beneath it, the
    same way it protects a script's own top-level call, just triggered on import instead of on
    ``__main__``. Checked once per package directory (not every ancestor __init__.py needs its
    own separate call; the outermost one that runs first is what matters), stopping at the
    first ``__init__.py`` found reconfiguring, or at ``root`` if none do."""
    current = py.parent
    root = root.resolve()
    while True:
        init_file = current / "__init__.py"
        if init_file.is_file():
            try:
                init_src = init_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                init_src = ""
            if _has_stdio_utf8_reconfigure(init_src):
                return True
        if current.resolve() == root or current.parent == current:
            return False
        current = current.parent


def scan_console_unicode(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``print(...)`` / ``logger.X(...)`` calls whose first string-literal argument contains a
    non-ASCII character.

    On Windows, default stdout/stderr encoding is ``cp1251``/``cp1252``; printing a fancy Unicode
    arrow, checkmark, or any non-Latin character crashes with ``UnicodeEncodeError`` -- silently
    fine on Linux/macOS dev machines, guaranteed broken on Windows. Skips files that already call
    ``sys.stdout.reconfigure(encoding=...)`` / ``sys.stderr.reconfigure(encoding=...)`` themselves
    (see :func:`_has_stdio_utf8_reconfigure`) -- the established fix for this exact failure mode,
    confirmed as this codebase's own convention across dozens of scripts -- OR whose enclosing
    PACKAGE already does so in an ``__init__.py`` up the directory chain (see
    :func:`_ancestor_package_has_stdio_utf8_reconfigure`): a library module imported as part of a
    package is exactly as protected as a script with its own top-level call, just via the
    package's own import-time guard instead of a per-file one.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        if _has_stdio_utf8_reconfigure("\n".join(src_lines)):
            continue
        if _ancestor_package_has_stdio_utf8_reconfigure(py, root):
            continue
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_console_call(node):
                continue
            literal = _first_str_arg_value(node)
            if not literal or not _has_non_ascii(literal):
                continue
            findings.append(Finding(
                check="console_unicode",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    "print()/logger.*() call emits a non-ASCII string literal; crashes with "
                    "UnicodeEncodeError on Windows cp1251/cp1252 stdout. Replace fancy characters "
                    "(arrows, checkmarks, em-dash) with ASCII equivalents."
                ),
            ))
    return findings
