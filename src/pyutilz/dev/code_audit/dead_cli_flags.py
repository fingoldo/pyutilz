"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- dead argparse flags ---------------------------------------------------


def _flag_dest_name(call: ast.Call) -> str | None:
    """Derive the ``argparse.Namespace`` attribute name a
    ``parser.add_argument(...)`` call binds to.

    Priority: an explicit ``dest="..."`` keyword wins. Otherwise the
    first long-form flag string (``"--foo-bar"``) is used, falling back
    to the first positional flag string of any form. Returns ``None``
    when no derivable name exists (e.g. the flag list is empty or every
    arg is a non-constant expression).
    """
    for kw in call.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    flag_strings = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not flag_strings:
        return None
    long_forms = [f for f in flag_strings if f.startswith("--")]
    chosen = long_forms[0] if long_forms else flag_strings[0]
    return chosen.lstrip("-").replace("-", "_")


def _is_add_argument_call(node: ast.AST) -> bool:
    """True if ``node`` is an ``argparse.ArgumentParser``-style ``<something>.add_argument(...)``
    call -- deliberately excludes calls with ZERO keyword arguments, since Selenium's
    ``ChromeOptions``/``FirefoxOptions`` (and similar browser-automation libraries) expose an
    UNRELATED ``add_argument(flag_string)`` method with the identical name that appends a raw
    command-line flag to a list passed to an external browser BINARY -- it has no ``dest=``/
    ``action=``/``type=``/etc. concept at all, so ``args.<name>`` is never expected to exist in
    THIS codebase's own Python source (the flag is consumed by Chrome/Firefox itself, not parsed
    into a namespace here). Confirmed false positive found in the wild (2026-07-22):
    ``web/browser.py``'s ``options.add_argument("--no-sandbox")`` and 3 siblings were flagged as
    "dead" despite doing exactly what they're meant to -- configure the launched browser process.
    Real argparse usage in this codebase (``dev/code_audit/cli.py``) always carries at least one
    argparse-specific keyword (``type=``, ``help=``, etc.); a zero-keyword-argument call is the
    reliable, checked-in-the-wild signal that distinguishes the two libraries' identically-named
    methods.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
        return False
    return bool(node.keywords)


def scan_dead_cli_flags(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``argparse`` flags declared via ``add_argument`` whose bound
    attribute (``args.<name>``) is never referenced anywhere in the
    scanned tree.

    A flag that's parsed but never read is either dead code (safe to
    remove) or -- the more dangerous case -- a flag the AUTHOR believes
    controls behavior but which silently does nothing (e.g. ``--resume``
    parsed with a default, but the query it's meant to gate never checks
    ``args.resume``). Either way the CLI's documented behavior and its
    actual behavior have diverged.

    The reference-check scans the WHOLE tree (not just the defining
    file), so a flag consumed in a different module than where it's
    declared is not flagged. This trades a little precision for
    resilience against multi-file CLI designs; a name collision with an
    unrelated ``.name`` attribute access elsewhere only produces a
    (safe) false negative, never a false positive from THAT mechanism.

    Known blind spot: the corpus is a regex over literal ``.attrname``
    dot-access text, plus ``getattr(x, "name")`` / ``vars(x)["name"]``
    shapes. A flag read through some other dynamic-access spelling this
    scanner doesn't recognize will still be misreported as dead (a real,
    if narrow, false-positive source).

    Severity: P2 (a CLI flag silently not doing what its help text
    promises is a real behavior-vs-documentation gap, but not a crash).
    """
    findings: list[Finding] = []
    py_files = list(_iter_py_files(root, exclude_dirs))

    # Build corpus of `.attr` usages across the whole tree once, up front.
    used_attrs: set[str] = set()
    for py in py_files:
        try:
            src = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        used_attrs.update(re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\b", src))
        # getattr(args, "name") / vars(args)["name"] dynamic-access shapes.
        used_attrs.update(re.findall(r"""getattr\s*\([^,]+,\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""", src))
        used_attrs.update(re.findall(r"""vars\s*\([^)]*\)\s*\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]""", src))

    for py in py_files:
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_add_argument_call(node):
                continue
            name = _flag_dest_name(node)
            if not name or name == "help":
                continue
            if name in used_attrs:
                continue
            findings.append(Finding(
                check="dead_cli_flag",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"CLI flag binds to `.{name}` but that attribute is never "
                    f"referenced anywhere in the scanned tree -- either dead "
                    f"code (remove it) or a flag that silently does nothing "
                    f"(wire it into the behavior it's meant to gate)."
                ),
            ))
    return findings
