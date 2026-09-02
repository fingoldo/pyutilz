"""One idiom for a re-export package's submodule reaching back into its own package.

Several packages here are the result of a monolith split: the `__init__` imports its submodules and
re-exports their names, so a submodule importing the package back is a genuine cycle. Two spellings
were in use, neither documenting the other:

* `import <parent> as _facade` + attribute access at call time (`data/pandaslib`), and the same idea
  behind an object in `text/strings/_logproxy.py`;
* `from <parent> import <name>` -- which only ever "worked" by accident of import order.

The second form is the trap: plain `import x` binds the partially-initialised `sys.modules` entry and
defers the lookup, while `from x import name` needs the name to already exist and raises
``ImportError: cannot import name ... (most likely due to a circular import)``. It also snapshots the
value, so a name the facade is expected to be able to replace (a monkeypatched `HAS_IPYTHON`, a
mutable module global) stops being patchable. This test makes the choice mechanical rather than
folklore.
"""

import ast
from pathlib import Path

import pyutilz

_SRC = Path(pyutilz.__file__).resolve().parent


def _package_of(path: Path) -> str:
    """Dotted name of the package the module at ``path`` lives in."""
    rel = path.relative_to(_SRC.parent)
    parts = list(rel.parts[:-1])
    return ".".join(parts)


def _iter_modules():
    for path in _SRC.rglob("*.py"):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        yield path


def test_no_submodule_from_imports_its_own_package_at_top_level():
    offenders = []
    for path in _iter_modules():
        own_package = _package_of(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - a syntax error is another test's job
            offenders.append(f"{path}: unparseable ({exc})")
            continue
        package_dir = path.parent
        for node in tree.body:  # module level only: a deferred import inside a function is fine
            if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == own_package):
                continue
            # Importing a SIBLING SUBMODULE this way is safe and common: the name resolves through
            # sys.modules rather than through the half-built parent's namespace, and a module object
            # cannot be the patchable value the rule is about.
            values = [a.name for a in node.names if not (package_dir / f"{a.name}.py").exists() and not (package_dir / a.name / "__init__.py").exists()]
            if values:
                offenders.append(f"{path.relative_to(_SRC).as_posix()}:{node.lineno}: from {own_package} import {', '.join(values)}")
    assert not offenders, (
        "a submodule must reach its own package via `import <parent> as _facade` + attribute access, "
        "never a top-level from-import (see the idiom comment in data/pandaslib/frames.py):\n  " + "\n  ".join(offenders)
    )


def test_the_facade_idiom_is_actually_in_use_where_it_is_documented():
    """Guards the test above against passing by vacuity if the packages it describes go away."""
    documented = [
        _SRC / "data" / "pandaslib" / "frames.py",
        _SRC / "data" / "pandaslib" / "io_ops.py",
        _SRC / "data" / "pandaslib" / "benchmarks.py",
    ]
    for path in documented:
        assert path.exists(), path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = {a.asname for node in tree.body if isinstance(node, ast.Import) for a in node.names}
        assert "_facade" in aliases, f"{path.name} no longer uses the documented `import <parent> as _facade` idiom"
