"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- in-place numpy mutation of an uncopied pandas .to_numpy() result --------

# ``np.<attr>(target, ...)`` calls whose first positional arg is mutated IN PLACE.
_INPLACE_MUTATORS = frozenset({"fill_diagonal", "copyto"})


def _is_to_numpy_call_without_copy(call: ast.Call) -> bool:
    """True for ``<anything>.to_numpy(...)`` with no ``copy=True`` keyword argument.

    Under pandas Copy-on-Write (default from pandas 3.0, opt-in in 2.x via
    ``pd.options.mode.copy_on_write``), ``DataFrame.to_numpy()`` / ``Series.to_numpy()`` can return
    a read-only view of the frame's backing storage. Passing that straight into an in-place numpy
    mutator raises ``ValueError: underlying array is read-only`` -- environment-dependent (silent
    on a non-CoW pandas install, real on a CoW one), so it is easy to ship unnoticed.
    """
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "to_numpy"):
        return False
    for kw in call.keywords:
        if kw.arg == "copy":
            return not (isinstance(kw.value, ast.Constant) and kw.value.value is True)
    return True


def _is_np_inplace_mutator_call(call: ast.Call) -> "str | None":
    """Return the mutated arg's Name id if ``call`` is ``np.fill_diagonal(X, ...)`` / ``np.copyto(X, ...)``, else None."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr in _INPLACE_MUTATORS):
        return None
    if not (isinstance(func.value, ast.Name) and func.value.id in ("np", "numpy")):
        return None
    if not call.args:
        return None
    first = call.args[0]
    return first.id if isinstance(first, ast.Name) else None


def scan_readonly_to_numpy_mutation(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``np.fill_diagonal(X, ...)`` / ``np.copyto(X, ...)`` where ``X`` was assigned, earlier
    in the same function, from an uncopied ``<pandas obj>.to_numpy()`` call.

    Real bug this generalizes (2026-07-31, mlframe): ``dataset_diagnostics`` fed
    ``df.corr().to_numpy()`` straight into ``np.fill_diagonal(C, 0.0)``; under a CoW-enabled pandas
    install ``to_numpy()`` returned a read-only array and the mutation raised at runtime, while
    passing silently on a non-CoW install -- invisible without the exact same pandas config. Fix:
    ``.to_numpy(copy=True)``.

    False-negative-biased by design (matches this package's other heuristic scanners): only
    same-function, same-name tracking (no cross-function/attribute flow), only the two mutators
    listed above, and only a direct ``Name = <call>.to_numpy()`` assignment shape.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            risky_names: set[str] = set()
            for node in ast.walk(func_node):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and _is_to_numpy_call_without_copy(node.value):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            risky_names.add(target.id)
            for node in ast.walk(func_node):
                if not isinstance(node, ast.Call):
                    continue
                target_name = _is_np_inplace_mutator_call(node)
                if target_name is None or target_name not in risky_names:
                    continue
                findings.append(Finding(
                    check="readonly_to_numpy_mutation",
                    severity="P2",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"`{target_name}` came from an uncopied `.to_numpy()` and is passed to an in-place "
                        f"numpy mutator; under pandas Copy-on-Write this raises `ValueError: underlying "
                        f"array is read-only`. Use `.to_numpy(copy=True)`."
                    ),
                ))
    return findings
