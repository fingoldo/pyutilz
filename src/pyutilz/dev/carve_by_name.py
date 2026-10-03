"""Carve named module-level functions, plus everything they need, out of an oversized module into a new sibling.

Complements :func:`pyutilz.dev.freevar_analysis.split_out_module`, which moves one CONTIGUOUS run of definitions: here the
functions are selected by NAME PATTERN (the ``_fit_step3_*`` helpers an extraction tool scattered through a module), and the
module-level definitions they read are pulled along transitively, so the sibling never has to import the module it came
from (no import cycle) and a monkeypatch of a moved name has exactly one place to land.

The source keeps ``from .<sibling> import (name, ...)  # noqa: F401`` in place of the moved definitions, so existing
importers keep resolving. Both outputs are parsed before anything is written, and the sibling is written first: a failure
leaves the source untouched instead of a source that has already lost its definitions.

CLI: ``python -m pyutilz.dev.carve_by_name SOURCE SIBLING NAME_REGEX [--dry-run]`` (``SIBLING`` is a module name or a path).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

__all__ = ["carve_by_name"]

_Def = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign]


def _span(node: _Def) -> Tuple[int, int]:
    """0-based start line (decorators included) and exclusive end line of a definition."""
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
    return start, int(node.end_lineno or node.lineno)


def _loaded_names(nodes: Sequence[ast.AST]) -> Set[str]:
    """Names the given nodes read (``Load`` context)."""
    return {x.id for n in nodes for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


def _top_level_definitions(tree: ast.Module) -> Dict[str, _Def]:
    """Module-level functions, classes and simple (annotated) assignments by name."""
    defs: Dict[str, _Def] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            defs[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            defs[node.target.id] = node
    return defs


def carve_by_name(source: Union[str, Path], sibling: Union[str, Path], name_regex: str, *, apply: bool = True) -> List[str]:
    """Move the module-level functions whose names fully match ``name_regex`` (and what they need) into ``sibling``.

    ``sibling`` is a module name (``"_fit_steps"``, placed beside ``source``) or a path to the new file. Returns the moved
    names in source order; with ``apply=False`` nothing is written. Refuses when nothing matches, when the sibling already
    exists, or when either output fails to parse.
    """
    path = Path(source)
    sib = Path(sibling)
    sib_path = sib if sib.suffix == ".py" else path.parent / f"{sib.name}.py"
    sib_module = sib_path.stem
    if sib_path.exists():
        raise FileExistsError(f"{sib_path} already exists; refusing to overwrite it")

    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    tree = ast.parse(text)
    lines = text.split("\n")
    defs = _top_level_definitions(tree)

    rx = re.compile(name_regex)
    moved = {name for name, node in defs.items() if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and rx.fullmatch(name)}
    if not moved:
        raise ValueError(f"no module-level function in {path.name} fully matches {name_regex!r}")
    changed = True
    while changed:
        changed = False
        for name in _loaded_names([defs[m] for m in moved]):
            if name in defs and name not in moved:
                moved.add(name)
                changed = True
    order = sorted(moved, key=lambda nm: defs[nm].lineno)

    first_def = min(_span(n)[0] for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))) + 1
    doc_end = tree.body[0].end_lineno if ast.get_docstring(tree) else 0
    header = "\n".join(lines[doc_end : first_def - 1])
    bodies = ["\n".join(lines[slice(*_span(defs[nm]))]) for nm in order]
    sib_src = f'"""Helpers carved out of ``{path.stem}`` to keep that module under its size budget."""\n' + header.rstrip("\n") + "\n\n\n" + "\n\n\n".join(bodies) + "\n"

    for nm in sorted(moved, key=lambda nm: -defs[nm].lineno):
        a, b = _span(defs[nm])
        while b < len(lines) and lines[b].strip() == "":
            b += 1
        del lines[a:b]
    remaining = "\n".join(lines)
    rest_tree = ast.parse(remaining)
    anchor = min(_span(n)[0] for n in rest_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))) + 1
    block = f"from .{sib_module} import (  # noqa: F401  -- carved helpers\n" + "".join(f"    {nm},\n" for nm in order) + ")\n\n"
    out = remaining.split("\n")
    out[anchor - 1 : anchor - 1] = block.split("\n")
    new_source = "\n".join(out)
    ast.parse(new_source)
    ast.parse(sib_src)

    if apply:
        sib_path.write_bytes(sib_src.replace("\n", newline).encode("utf-8"))
        path.write_bytes(new_source.replace("\n", newline).encode("utf-8"))
    return order


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source")
    parser.add_argument("sibling", help="new module name or path")
    parser.add_argument("name_regex", help="full-match pattern for the functions to move")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    moved = carve_by_name(args.source, args.sibling, args.name_regex, apply=not args.dry_run)
    sys.stdout.write(f"{'would move' if args.dry_run else 'moved'} {len(moved)} definitions: {', '.join(moved)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
