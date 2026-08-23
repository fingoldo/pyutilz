"""Unit tests for ``scan_reexport_patch_target``.

Positive: a test patching a name on a module that only RE-EXPORTS it. Negative: the two shapes that
must never be flagged - the patched module defines the name itself, and the patched module re-exports
it but also calls it (a real caller, where the stub genuinely changes that module's behaviour).
"""

from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_reexport_patch_target


def _tree(tmp_path: Path, *, facade: str, test_body: str) -> tuple[Path, Path]:
    """A package root holding ``definer``/``facade`` plus a tests dir holding one test file."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "definer.py").write_text("def target_fn():\n    return 1\n", encoding="utf-8")
    (pkg / "facade.py").write_text(facade, encoding="utf-8")
    tests = tmp_path / "t"
    tests.mkdir()
    (tests / "test_it.py").write_text(test_body, encoding="utf-8")
    return tests, tmp_path


def test_patching_a_pure_facade_is_flagged(tmp_path: Path):
    tests, root = _tree(
        tmp_path,
        facade="from pkg.definer import target_fn\n",
        test_body="import pkg.facade as facade\n\n\ndef test_x(monkeypatch):\n    monkeypatch.setattr(facade, 'target_fn', lambda: 2)\n",
    )
    findings = scan_reexport_patch_target(tests, package_roots=(root,))
    assert len(findings) == 1, findings
    assert findings[0].check == "reexport_patch_target"
    assert "pkg.definer" in findings[0].detail, "the detail must name where the caller's own binding comes from"


def test_patching_the_module_that_defines_the_name_is_not_flagged(tmp_path: Path):
    """The unambiguous case: there is only one binding, so the patch cannot miss."""
    tests, root = _tree(
        tmp_path,
        facade="def target_fn():\n    return 1\n",
        test_body="import pkg.facade as facade\n\n\ndef test_x(monkeypatch):\n    monkeypatch.setattr(facade, 'target_fn', lambda: 2)\n",
    )
    assert scan_reexport_patch_target(tests, package_roots=(root,)) == []


def test_a_reexporter_that_also_calls_the_name_is_not_flagged(tmp_path: Path):
    """The precision filter. Patching a module that re-exports AND calls really does change what that
    module does, so it is a legitimate target - flagging it would bury the real finding in noise."""
    tests, root = _tree(
        tmp_path,
        facade="from pkg.definer import target_fn\n\n\ndef use_it():\n    return target_fn() + 1\n",
        test_body="import pkg.facade as facade\n\n\ndef test_x(monkeypatch):\n    monkeypatch.setattr(facade, 'target_fn', lambda: 2)\n",
    )
    assert scan_reexport_patch_target(tests, package_roots=(root,)) == []


def test_patch_object_form_is_covered_too(tmp_path: Path):
    """``mock.patch.object(mod, "name")`` is the same hazard through a different idiom."""
    tests, root = _tree(
        tmp_path,
        facade="from pkg.definer import target_fn\n",
        test_body="from unittest.mock import patch\nimport pkg.facade as facade\n\n\ndef test_x():\n    with patch.object(facade, 'target_fn', lambda: 2):\n        pass\n",
    )
    assert len(scan_reexport_patch_target(tests, package_roots=(root,))) == 1


def test_an_unresolvable_module_is_skipped_rather_than_guessed(tmp_path: Path):
    """No source file for the patched module means no evidence either way - silence, not a report."""
    tests = tmp_path / "t"
    tests.mkdir()
    (tests / "test_it.py").write_text(
        "import not_a_real_package.somewhere as m\n\n\ndef test_x(monkeypatch):\n    monkeypatch.setattr(m, 'thing', 1)\n",
        encoding="utf-8",
    )
    assert scan_reexport_patch_target(tests, package_roots=(tmp_path,)) == []
