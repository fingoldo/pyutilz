"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- locals()/globals() passed as a mutable output parameter --------------
#
# Found independently TWICE in the 2026-07-21 audit, both with real operational impact:
# cloud.py:connect_to_s3() called read_config_file(..., object=globals()) expecting the callee's
# writes to populate its own local aws_access_key_id/aws_secret_access_key variables -- but in
# CPython, locals()/globals() inside an ordinary function scope is a DISCONNECTED SNAPSHOT dict;
# writes to it never reach the real variable slot. boto3.Session() ended up built from the
# original None/None values every time (Critical -- credentials silently never applied).
# scheduling/prefect.py:connect(prefect_key=None) had the identical bug via locals().
#
# This AST shape is narrow and almost never legitimate as an OUTPUT parameter (locals()/globals()
# passed to introspection/debugging calls -- e.g. `logger.debug(locals())`, `eval(expr, globals())`
# -- is common and fine; the bug is specifically "pass locals()/globals() as a kwarg/positional
# arg the callee is expected to WRITE INTO", which this scanner can't perfectly distinguish
# without semantic analysis of the callee, so it flags every non-first-arg call-site of
# locals()/globals() passed into ANOTHER function call, at Low severity for the ones that might
# be legitimate introspection and P1 when passed via a kwarg literally named "object"/"out"/
# "output"/"target" (the vocabulary an output-parameter API tends to use).


_OUTPUT_LIKE_KWARG_NAMES = frozenset({"object", "out", "output", "target", "dest", "destination", "sink"})

# audit-2026-07-22 false-positive fix: a builtin that only ever READS its argument (never writes
# into it expecting the caller to observe a mutation) is never the "callee writes into it,
# expecting write-back" bug this scanner targets -- e.g. `sorted(set(globals()))` in a __dir__()
# implementation is a completely ordinary read-only introspection idiom, not the output-parameter
# bug shape.
_READ_ONLY_BUILTIN_CONSUMERS = frozenset({"set", "list", "dict", "tuple", "frozenset", "sorted", "len", "str", "repr", "print", "iter", "any", "all", "min", "max", "sum"})


def _is_locals_or_globals_call(node: ast.AST) -> bool:
    """True if ``node`` is a call to the bare builtin ``locals()``/``globals()`` (no args)."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("locals", "globals") and not node.args and not node.keywords


def scan_locals_globals_as_output(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``locals()``/``globals()`` passed as an argument to ANOTHER function call -- in
    CPython, ``locals()``/``globals()`` inside an ordinary function scope returns a disconnected
    snapshot dict; a callee's writes into it never reach the real variable/module-global slot the
    caller presumably expected to be updated.

    Severity: P1 when passed via a kwarg whose name suggests an output/mutation contract
    (``object=``, ``out=``, ``target=``, etc. -- the exact shape of both real bugs found in the
    2026-07-21 audit); Low otherwise (could be legitimate introspection/debug logging).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in ("locals", "globals"):
                continue  # this Call IS the locals()/globals() call itself, not a use of it
            if isinstance(node.func, ast.Name) and node.func.id in _READ_ONLY_BUILTIN_CONSUMERS:
                continue  # read-only consumer -- never the write-back bug shape this scanner targets
            for kw in node.keywords:
                if kw.arg is not None and _is_locals_or_globals_call(kw.value):
                    severity = "P1" if kw.arg in _OUTPUT_LIKE_KWARG_NAMES else "Low"
                    which = "locals" if isinstance(kw.value, ast.Call) and kw.value.func.id == "locals" else "globals"  # type: ignore[attr-defined]
                    findings.append(Finding(
                        check="locals_globals_as_output",
                        severity=severity,
                        file=rel,
                        line=kw.value.lineno,
                        snippet=_line_text(src_lines, kw.value.lineno),
                        detail=(
                            f"`{kw.arg}={which}()` passed as an argument -- {which}() inside an "
                            "ordinary function scope is a DISCONNECTED SNAPSHOT dict in CPython; "
                            "if the callee is expected to write into it and have those writes "
                            "reach the real variable/global slot, they never will. Pass a real "
                            "dict, capture the callee's return value, or read the resolved value "
                            "back out of a local dict after the call."
                        ),
                    ))
            for arg in node.args:
                if _is_locals_or_globals_call(arg):
                    which = arg.func.id  # type: ignore[attr-defined]
                    findings.append(Finding(
                        check="locals_globals_as_output",
                        severity="Low",
                        file=rel,
                        line=arg.lineno,
                        snippet=_line_text(src_lines, arg.lineno),
                        detail=(
                            f"`{which}()` passed as a positional argument to another call -- if "
                            "the callee is expected to write into it and have those writes reach "
                            f"the real variable/global slot, they never will ({which}() is a "
                            "disconnected snapshot dict inside an ordinary function scope)."
                        ),
                    ))
    return findings
