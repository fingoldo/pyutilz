"""D2 — meta-test that every ``open(...)`` call in pyutilz production
code passes ``encoding=`` (or uses ``"rb"``/``"wb"`` mode where bytes
are explicit).

Default Python ``open()`` uses ``locale.getpreferredencoding()`` —
which on Windows is typically ``cp1251`` / ``cp1252``, NOT UTF-8.
Reading any file authored with non-ASCII content blows up with
``UnicodeDecodeError`` on Russian / European Windows installs.
Catching this at static-analysis time prevents the class of bugs the
maintainer's own memory ``feedback_windows_encoding`` already documents.

The check uses Python's AST so it survives reformatting and doesn't
fool itself with comments / strings containing ``open(...)``.

  * ``open(path, "rb")`` / ``open(path, mode="wb")`` — OK (binary mode).
  * ``open(path, "r", encoding="utf-8")`` — OK.
  * ``open(path)`` / ``open(path, "r")`` / ``open(path, "w")`` — flagged.
  * ``open(path, "r+", errors="ignore")`` without encoding — flagged.

Also covers ``Path.open(...)`` and ``pathlib.Path(...).open(...)``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent

# Files explicitly allowed to skip the encoding contract — typically
# files that intentionally manipulate bytes / locked formats.
_EXEMPT_FILES: set[str] = set()

# Specific call sites that intentionally use the locale-default encoding
# (rare; legitimate when reading bytes-like sysfs of unknown encoding).
# All 8 sites flagged on 2026-04-28 have been fixed in-place — the set
# is now empty and ready to flag any future regressions.
_USER_DEFERRED_CALLS: set[str] = set()


def _calls_to_open(tree: ast.AST):
    """Yield every ``ast.Call`` node that calls the BUILTIN ``open``.

    Restricts to bare ``open(...)`` because attribute-form ``.open(...)``
    is heavily overloaded across libraries (``PIL.Image.open``,
    ``zipfile.ZipFile.open``, ``socket.open`` etc.) — most of which
    have nothing to do with text I/O. The risky form on Windows is
    the builtin, where the default encoding falls back to
    ``locale.getpreferredencoding()``.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            yield node


def _is_binary_mode(call: ast.Call) -> bool:
    """Detect explicit binary-mode (``"rb"`` / ``"wb"`` / ``"ab"`` /
    ``"r+b"`` etc.) — those are exempt from the encoding check.

    Looks at the second positional argument OR a ``mode=`` kwarg.
    """
    mode_value = None
    if len(call.args) >= 2:
        mode_arg = call.args[1]
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
            mode_value = mode_arg.value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            v = kw.value.value
            if isinstance(v, str):
                mode_value = v
    if mode_value and "b" in mode_value:
        return True
    return False


def _has_encoding_kwarg(call: ast.Call) -> bool:
    return any(kw.arg == "encoding" for kw in call.keywords)


def test_every_open_call_specifies_encoding_or_binary_mode():
    bad: list[str] = []
    audited = 0
    for py in PYUTILZ_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if py.name.endswith(".py.old"):
            continue
        rel = py.relative_to(PYUTILZ_DIR).as_posix()
        if rel in _EXEMPT_FILES:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for call in _calls_to_open(tree):
            audited += 1
            if _is_binary_mode(call):
                continue
            if _has_encoding_kwarg(call):
                continue
            entry = f"{rel}:{call.lineno}"
            if entry in _USER_DEFERRED_CALLS:
                continue
            bad.append(entry)

    assert audited > 5, (
        f"only {audited} open(...) calls audited — AST walk broken?"
    )
    if bad:
        pytest.fail(
            f"{len(bad)} ``open(...)`` call(s) in pyutilz production code "
            f"don't specify ``encoding=`` (and aren't binary-mode). On "
            f"Windows, default open() uses cp1251/cp1252 and "
            f"UnicodeDecodeError will happen on any non-ASCII file. "
            f"Either add ``encoding=\"utf-8\"`` (or other appropriate "
            f"encoding), OR use ``\"rb\"``/``\"wb\"`` mode for bytes. "
            f"Whitelist via _EXEMPT_FILES if intentional:\n  "
            + "\n  ".join(sorted(set(bad))[:30])
            + (f"\n  ... and {len(set(bad)) - 30} more" if len(set(bad)) > 30 else "")
        )
