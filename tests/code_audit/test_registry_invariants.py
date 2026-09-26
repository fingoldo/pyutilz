"""Scanner tests for registry_invariants, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path
import pytest

from pyutilz.dev.code_audit import (
    Finding,
    run_all,
    scan_mutable_defaults,
)

from ._helpers import _scanner_function, _write


def test_registry_and_facade_are_in_bijection():
    """F10/F207 regression: registered scanners and the package's public ``scan_*`` surface must be
    the SAME set. Before the fix two scanners were exported but unregistered (so run_all/--check
    could never reach them) and 21 were registered but not importable from the package."""
    import pyutilz.dev.code_audit as facade

    registered = {_scanner_function(fn).__name__ for fn in facade.get_scanners().values()}
    exported = {name for name in facade.__all__ if name.startswith("scan_")}
    assert registered - exported == set(), "registered but not exported"
    assert exported - registered == set(), "exported but not registered"
    for name in exported:
        assert callable(getattr(facade, name, None)), name


def test_every_emitted_check_id_is_selectable():
    """F199 regression: an id printed in ``Finding.check`` must be runnable via ``--check``/``checks=``.
    Nine emitted ids (e.g. ``unbounded_retry_loop``) resolved to nothing, so a reader handed one
    could not re-run it and a baseline keyed on it could not be mapped back to a scanner."""
    import ast

    from pyutilz.dev.code_audit import registry as _registry

    pkg = Path(_registry.__file__).parent
    emitted = set()
    for module in sorted(pkg.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Finding"):
                continue
            for kw in node.keywords:
                if kw.arg == "check" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    emitted.add(kw.value.value)
    assert emitted, "no literal Finding(check=...) ids found - the harvest broke, not the invariant"
    unresolvable = sorted(cid for cid in emitted if _registry.resolve_check(cid) is None)
    assert unresolvable == []


def test_finding_rejects_a_severity_outside_the_vocabulary():
    """F210 regression: the P0/P1/P2/Low contract was documented in a docstring and enforced nowhere,
    which is how a ``severity="Medium"`` made one scanner's entire output invisible (F09)."""
    from pyutilz.dev.code_audit._base import SEVERITIES

    for severity in SEVERITIES:
        assert Finding(check="c", severity=severity, file="f.py", line=1, snippet="", detail="d").severity == severity
    with pytest.raises(ValueError):
        Finding(check="c", severity="Medium", file="f.py", line=1, snippet="", detail="d")


def test_no_scanner_emits_a_severity_outside_the_vocabulary(tmp_path: Path):
    """F09 regression: running the whole registry over a corpus that trips many rules must not
    produce a severity the CLI's filter cannot see."""
    from pyutilz.dev.code_audit._base import SEVERITIES

    _write(tmp_path, "bad.py", """
class NeverRaisedError(Exception):
    pass

def g(items=[]):
    items.append(1)
    try:
        pass
    except:
        pass
""")
    findings = run_all(tmp_path, parallel=False)
    assert findings
    assert {f.severity for f in findings} <= set(SEVERITIES)


def test_cli_renders_and_gates_on_an_unknown_severity(tmp_path: Path, capsys, monkeypatch):
    """F200 regression: ``sev_order.get(f.severity, 99) <= cutoff`` deleted an unrecognised severity
    at EVERY --min-severity setting and the exit code ignored it too."""
    from pyutilz.dev.code_audit import cli as _cli
    from pyutilz.dev.code_audit import main as cli_main

    stray = Finding.__new__(Finding)
    object.__setattr__(stray, "check", "stray")
    object.__setattr__(stray, "severity", "Medium")
    object.__setattr__(stray, "file", "x.py")
    object.__setattr__(stray, "line", 1)
    object.__setattr__(stray, "snippet", "x = 1")
    object.__setattr__(stray, "detail", "d")
    monkeypatch.setattr(_cli, "run_all", lambda *a, **k: [stray])

    assert cli_main([str(tmp_path), "--min-severity", "P0"]) == 1
    assert "stray" in capsys.readouterr().out


def test_run_all_parallel_runs_a_runtime_registered_scanner(tmp_path: Path):
    """F04 regression: the parallel path (the default) looked every scanner up by name in the
    WORKER's registry, which holds only the built-ins, so any scanner added through the public
    register_scanner() died with KeyError and took the whole run's output with it."""
    import functools

    from pyutilz.dev.code_audit import register_scanner
    from pyutilz.dev.code_audit.registry import _SCANNERS

    _write(tmp_path, "bad.py", "def f(items=[]):\n    items.append(1)\n")
    name = "runtime_registered_probe"
    register_scanner(name, functools.partial(scan_mutable_defaults), allow_override=True)
    try:
        checks = [name, "mutable_default", "bare_except", "nan_equality", "console_unicode"]
        findings = run_all(tmp_path, checks=checks, parallel=True)
    finally:
        _SCANNERS.pop(name, None)
    assert any(f.check == "mutable_default" for f in findings)


def test_run_all_parallel_keeps_an_unpicklable_runtime_scanner(tmp_path: Path):
    """A locally-defined scanner cannot cross a process boundary at all; it must still run, in-process."""
    from pyutilz.dev.code_audit import register_scanner
    from pyutilz.dev.code_audit.registry import _SCANNERS

    _write(tmp_path, "ok.py", "x = 1\n")
    name = "unpicklable_probe"
    marker = Finding(check="unpicklable_probe", severity="Low", file="ok.py", line=1, snippet="x = 1", detail="d")
    register_scanner(name, lambda root, exclude_dirs=None: [marker], allow_override=True)
    try:
        checks = [name, "mutable_default", "bare_except", "nan_equality", "console_unicode"]
        findings = run_all(tmp_path, checks=checks, parallel=True)
    finally:
        _SCANNERS.pop(name, None)
    assert marker in findings


def test_run_all_survives_one_failing_scanner(tmp_path: Path):
    """F216 regression: an exception in one scanner aborted pool.map (and the sequential loop),
    discarding every other scanner's findings."""
    from pyutilz.dev.code_audit import register_scanner
    from pyutilz.dev.code_audit.registry import _SCANNERS

    _write(tmp_path, "bad.py", "def f(items=[]):\n    items.append(1)\n")

    def _boom(root, exclude_dirs=None):
        raise RuntimeError("pathological file")

    register_scanner("boom_probe", _boom, allow_override=True)
    try:
        findings = run_all(tmp_path, checks=["boom_probe", "mutable_default"], parallel=False)
    finally:
        _SCANNERS.pop("boom_probe", None)
    assert [f.check for f in findings] == ["mutable_default"]


def test_snippet_is_correct_after_a_form_feed(tmp_path: Path):
    """F61 regression: str.splitlines() also breaks on the form feed a Python file may use as a
    section separator, which the tokenizer does not count -- every snippet after one was the
    following line's text while Finding.line stayed right."""
    _write(tmp_path, "ff.py", "x = 1\n\x0c\ndef f():\n    try:\n        pass\n    except:\n        pass\n")
    findings = [f for f in run_all(tmp_path, checks=["bare_except"], parallel=False)]
    assert findings
    assert findings[0].line == 6
    assert findings[0].snippet == "except:"


def test_parse_cache_is_bounded_and_clearable(tmp_path: Path):
    """F208 regression: an unbounded process-global retained one full AST per version of every file
    ever parsed, with no eviction and no way to release it."""
    from pyutilz.dev.code_audit import _base

    _base.clear_parse_cache()
    assert len(_base._PARSE_CACHE) == 0
    monkey = _base._PARSE_CACHE_MAX_ENTRIES
    _base._PARSE_CACHE_MAX_ENTRIES = 3
    try:
        for i in range(10):
            f = tmp_path / f"m{i}.py"
            f.write_text(f"x = {i}\n", encoding="utf-8")
            _base._safe_parse(f)
        assert len(_base._PARSE_CACHE) <= 3
    finally:
        _base._PARSE_CACHE_MAX_ENTRIES = monkey
        _base.clear_parse_cache()


def test_iter_py_files_never_resolves_a_path(tmp_path: Path, monkeypatch):
    """F209 regression: Path.resolve() ran on the CONSTANT root once per candidate file, a
    syscall-bound call repeated ~111k times on a 1500-file tree with 74 scanners.

    The walk is now a pruned ``os.walk``, so it needs NO resolve() at all: every path it yields is
    below ``root`` by construction, which is the whole reason the resolve() was there. Zero is the
    invariant to pin, not one -- a reintroduced per-file resolve() would fail this either way."""
    from pyutilz.dev.code_audit import _base

    for i in range(8):
        (tmp_path / f"m{i}.py").write_text("x = 1\n", encoding="utf-8")

    calls = {"n": 0}
    real = Path.resolve

    def counting(self, *a, **kw):
        calls["n"] += 1
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "resolve", counting)
    files = list(_base._iter_py_files(tmp_path, _base._DEFAULT_EXCLUDE_DIRS))
    assert len(files) == 8
    assert calls["n"] == 0


def test_module_sql_constants_sees_class_body_constants():
    """F215 regression: only tree.body was scanned, so a repository keeping its SQL in a
    ``class Queries:`` got no findings from any constant-resolving scanner."""
    import ast

    from pyutilz.dev.code_audit._base import _module_sql_constants, _sql_text

    tree = ast.parse('class Queries:\n    SELECT_ALL = "SELECT * FROM t"\n\nq = Queries.SELECT_ALL\n')
    constants = _module_sql_constants(tree)
    assert constants["Queries.SELECT_ALL"] == "SELECT * FROM t"
    assert constants["SELECT_ALL"] == "SELECT * FROM t"
    attribute = tree.body[-1].value
    assert _sql_text(attribute, constants) == "SELECT * FROM t"


def test_every_shared_gate_counterpart_is_a_registered_scanner_whose_module_names_its_gate():
    """Each scanner with a py-ci-shared gate for the same class says so in its module docstring, and says it the same
    way the table does: a renamed gate, or a scanner dropped from the table, makes the two disagree here."""
    import ast
    import sys

    from pyutilz.dev.code_audit import get_scanners
    from pyutilz.dev.code_audit.registry import SHARED_GATE_COUNTERPARTS

    scanners = get_scanners()
    assert len(SHARED_GATE_COUNTERPARTS) >= 18
    for name, gate in SHARED_GATE_COUNTERPARTS.items():
        assert name in scanners, f"{name} is in SHARED_GATE_COUNTERPARTS but is not a registered scanner"
        assert gate.startswith("py_ci_shared.") and gate.count(".") == 1, gate
        module = sys.modules[_scanner_function(scanners[name]).__module__]
        doc = ast.get_docstring(ast.parse(Path(module.__file__).read_text(encoding="utf-8"))) or ""
        assert (
            f"Shared-gate counterpart: {gate} " in doc or f"Shared-gate counterpart for {name}: {gate} " in doc
        ), f"{module.__name__}'s docstring does not name {gate} as the counterpart of {name}"


def test_every_module_that_names_a_shared_gate_is_in_the_table():
    """The reverse direction: a scanner module whose docstring names a gate must be listed, so the table is complete."""
    import re

    import pyutilz.dev.code_audit as pkg
    from pyutilz.dev.code_audit.registry import SHARED_GATE_COUNTERPARTS

    gates = set(SHARED_GATE_COUNTERPARTS.values())
    named = set()
    for path in Path(pkg.__file__).parent.glob("*.py"):
        named.update(re.findall(r"Shared-gate counterpart(?: for \w+)?: (py_ci_shared\.\w+)", path.read_text(encoding="utf-8")))
    assert named, "no scanner module names a shared gate; the docstring pattern changed"
    assert named - gates == set(), "a module names a gate the table does not list"
