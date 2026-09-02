"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- additive-epsilon denominators ---------------------------------------
#
# Class of bug confirmed eight times across four subsystems in the 2026-09-01 mlframe audit. `x / (d + 1e-12)`
# reads as a division guard and is not one, for two reasons that are both invisible at the call site:
#
# * The pad is harmless only while the denominator's natural scale stays far ABOVE it -- an assumption about the
#   DATA, not about the code. It fails in ordinary regimes: an exponentially-decayed weighted variance
#   compounding every row, a band energy in SQUARED units on 1e-3-amplitude log-returns, a per-category variance
#   of 1e-6 whose square is exactly 1e-12. Measured consequences in that audit: a slope reported as exactly 0.0
#   against a true 1.0, a spectral ratio wrong by 237x purely from rescaling the same signal, a skew of 2.0 read
#   as 0.25 (which flips a basis-routing branch).
# * It does not guard the division, it REPLACES the answer. Every input yields a plausible finite number, so
#   there is no NaN, no warning, and nothing downstream can distinguish a real ratio from a padded one.
#
# The sanctioned form is an explicit degeneracy branch, with the threshold RELATIVE to the denominator's own
# scale whenever that scale can shrink, so the degenerate value is a deliberate choice a reader can see.
#
# Only genuinely epsilon-sized float literals count: `x / (n + 1)` (Laplace count smoothing) and `x / (d + 0.5)`
# are ordinary arithmetic and must never be flagged.

DEFAULT_EPSILON_MAX = 1e-3


def _is_epsilon_constant(node: ast.AST, epsilon_max: float) -> bool:
    """True for a positive float literal small enough to be a degeneracy pad rather than real arithmetic."""
    return isinstance(node, ast.Constant) and isinstance(node.value, float) and 0.0 < node.value <= epsilon_max


def _is_epsilon_sum(node: ast.AST, epsilon_max: float) -> bool:
    """True for ``<expr> + <epsilon>`` or ``<epsilon> + <expr>``."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return False
    return _is_epsilon_constant(node.right, epsilon_max) or _is_epsilon_constant(node.left, epsilon_max)


def _epsilon_padded_names(scope: ast.AST, epsilon_max: float) -> set[str]:
    """Names bound in ``scope`` to an epsilon-padded sum -- ``denom = var + 1e-12`` followed by ``x / denom``."""
    names: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name) and _is_epsilon_sum(node.value, epsilon_max):
            names.add(node.targets[0].id)
    return names


def scan_additive_epsilon_denominator(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    *,
    epsilon_max: float = DEFAULT_EPSILON_MAX,
) -> list[Finding]:
    """Flag a division whose denominator is guarded by ADDING a small constant to it.

    Detects ``<x> / (<expr> + <eps>)`` in either operand order, and the two-statement form where a name is bound
    to such a sum and then divided by. ``eps`` counts only when it is a positive float literal at or below
    ``epsilon_max`` (default 1e-3), so integer and half-unit smoothing terms are left alone.

    Severity: P1. The division silently returns a wrong-but-plausible finite value for every input in the regime
    where the pad is comparable to the true denominator.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        functions: list[ast.AST] = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        # A module with no functions still has module-level divisions worth scanning, so fall back to the
        # whole tree as a single scope. Written as an explicit emptiness test rather than `... or [tree]`,
        # which `default_via_or` rightly flags: `or` cannot distinguish 'no functions' from any other falsy
        # value the comprehension might one day produce.
        scopes: list[ast.AST] = functions if functions else [tree]
        seen: set[int] = set()
        for scope in scopes:
            padded = _epsilon_padded_names(scope, epsilon_max)
            for node in ast.walk(scope):
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div) or node.lineno in seen:
                    continue
                rhs = node.right
                if isinstance(rhs, ast.Name) and rhs.id in padded:
                    where = f"the name `{rhs.id}`, bound above to an epsilon-padded sum,"
                elif _is_epsilon_sum(rhs, epsilon_max):
                    where = "an inline `+ <epsilon>` pad"
                else:
                    continue
                seen.add(node.lineno)
                findings.append(
                    Finding(
                        check="additive_epsilon_denominator",
                        severity="P1",
                        file=rel,
                        line=node.lineno,
                        snippet=_line_text(src_lines, node.lineno),
                        detail=(
                            f"denominator carries {where}. Adding a constant does not guard the division -- it "
                            f"REPLACES the answer with a plausible finite value whenever the true denominator is "
                            f"comparable to the pad, with no NaN and no warning to reveal it. Use an explicit "
                            f"degeneracy branch (`x / d if d > threshold else <documented value>`, or "
                            f"`np.where(d > 0, x / np.where(d > 0, d, 1.0), fill)`), and make the threshold "
                            f"RELATIVE to the denominator's own scale when that scale can shrink."
                        ),
                    )
                )
    return findings
