"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# LLMProvider methods whose ``max_tokens`` kwarg defaults to 0 ("provider max") --
# see pyutilz.llm.base.LLMProvider.generate/generate_json.
_CAPPED_METHOD_NAMES: frozenset[str] = frozenset({"generate", "generate_json", "generate_batch"})


def _is_zero_literal(node: ast.expr) -> bool:
    """True if ``node`` is the bare integer literal ``0``.

    Not ``== 0``: that is also true of ``False`` and ``0.0``, and ``max_tokens=False`` is a
    different mistake with a different fix than an explicit zero cap.
    """
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool) and node.value == 0


def _provider_var_names(tree: ast.AST) -> set[str]:
    """Every simple name assigned directly from a ``get_llm_provider(...)`` call,
    anywhere in the module (function-local or module-level assignment)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "get_llm_provider"):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def scan_llm_call_missing_max_tokens_cap(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a call to ``.generate(...)``/``.generate_json(...)``/
    ``.generate_batch(...)`` on a variable assigned from
    ``pyutilz.llm.factory.get_llm_provider(...)``, with no explicit,
    non-zero ``max_tokens`` argument.

    ``LLMProvider.generate``/``generate_json`` default ``max_tokens`` to
    ``0``, which every built-in provider treats as "use the provider's own
    maximum" -- an unbounded, provider-decided output length. A call site
    that never passes an explicit cap (or passes a bare ``0``) inherits
    whatever the provider's current maximum happens to be, with no local
    guardrail against an unexpectedly long (and unexpectedly expensive, or
    unexpectedly slow) response. A sibling call site elsewhere in the same
    codebase applying an explicit, reasoned cap (e.g. a classifier/judge
    call capped at 2000 tokens) while a peer generation call site has none
    is exactly the asymmetry this scanner catches.

    Deliberately narrow: only tracks the SIMPLE case (a bare-name variable
    assigned directly from ``get_llm_provider(...)``, then called as
    ``var.generate(...)`` in the same module) -- a provider obtained via a
    more indirect route (returned from a helper function, stored on
    ``self`` in ``__init__`` and called via ``self.provider.generate(...)``)
    is not tracked, since that requires cross-function data-flow analysis
    this scanner doesn't do.

    Severity: P2 (unbounded cost/latency exposure per call, not a crash).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        provider_vars = _provider_var_names(tree)
        if not provider_vars:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in _CAPPED_METHOD_NAMES):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id in provider_vars):
                continue
            max_tokens_kw = next((kw for kw in node.keywords if kw.arg == "max_tokens"), None)
            if max_tokens_kw is not None and not _is_zero_literal(max_tokens_kw.value):
                continue  # an explicit, non-zero cap is passed -- fine
            findings.append(Finding(
                check="llm_call_missing_max_tokens_cap",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"{func.value.id}.{func.attr}(...) has no explicit, non-zero max_tokens -- this "
                    "defaults to 0 ('use provider max'), an unbounded-cost/latency output length. Pass an "
                    "explicit, reasoned cap (as a sibling call site elsewhere in the codebase likely already does)."
                ),
            ))
    return findings
