"""Meta-test: package facades and exception hierarchies stay consistent with what ships.

Four rules, each one a defect class that reached a release because two in-repo declarations
disagreed and nothing compared them:

(a) **Facade reachability.** Every real subpackage directory under ``src/pyutilz/`` must be an
    attribute of the imported ``pyutilz`` module, and must itself declare ``__all__``. The
    subpackage list used to be a hand-typed literal that omitted ``stats`` and ``performance``,
    so ``pyutilz.stats`` raised AttributeError while ``from pyutilz import stats`` worked.

(b) **Registry-vs-facade parity.** Every class named in an in-repo registry must appear in its
    package's ``__all__``. ``OpenAIProvider`` was missing from ``pyutilz.llm``'s public surface
    while every sibling provider was exported, because the facade's provider list was a hand-kept
    copy of the factory's.

(c) **Exception rooting.** Every class in a package's ``exceptions.py`` must transitively reach
    that module's single in-module root, so ``except <DomainRoot>`` really is catch-all for the
    domain. Two of six LLM exception types once bypassed ``LLMProviderError``, and a caller's
    ``except LLMProviderError`` silently did not catch them.

(d) **Exported but never raised.** An exception class defined and exported but raised nowhere is
    either dead or a promise the code never keeps (``LLMTruncationError`` was fully specified and
    unraised for a whole wave). Domain roots are exempt - a root exists to be caught, and is
    normally raised only through its subclasses - and ``_RAISED_BY_DOWNSTREAM_ONLY`` names the
    genuine exceptions to the rule.

Pure AST plus one import of ``pyutilz`` (which the meta-suite already performs), so the whole
module runs in well under a second and touches no network or GPU.
"""

from __future__ import annotations

import ast
import importlib
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pytest

import pyutilz

SRC_DIR = Path(pyutilz.__file__).resolve().parent

# Exception classes deliberately defined and exported here for a DOWNSTREAM project to raise.
# Empty today; an entry must name the consumer, not merely record that the class is unraised.
_RAISED_BY_DOWNSTREAM_ONLY: frozenset[str] = frozenset()

# Registries whose values name a class that the package facade must export.
# (facade module, registry module, registry attribute, position of the class name in each value tuple)
_CLASS_REGISTRIES: tuple[tuple[str, str, str, int], ...] = (("pyutilz.llm", "pyutilz.llm.factory", "_PROVIDER_MODULES", 1),)


def _subpackage_dirs() -> list[str]:
    """Directory names directly under ``src/pyutilz/`` that are real packages."""
    return sorted(d.name for d in SRC_DIR.iterdir() if d.is_dir() and (d / "__init__.py").exists() and not d.name.startswith(("_", ".")))


def _class_defs(path: Path) -> dict[str, list[str]]:
    """Map each top-level class in ``path`` to the base names it lists, unqualified."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = [b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "") for b in node.bases]
    return out


def _exception_modules() -> list[Path]:
    return sorted(SRC_DIR.glob("*/exceptions.py"))


@lru_cache(maxsize=1)
def _raised_names() -> frozenset[str]:
    """Every name appearing as ``raise <Name>(...)`` or ``raise <Name>`` anywhere under ``src/pyutilz/``."""
    raised: set[str] = set()
    for py in SRC_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if isinstance(target, ast.Name):
                    raised.add(target.id)
                elif isinstance(target, ast.Attribute):
                    raised.add(target.attr)
    return frozenset(raised)


# --- rule kernels ----------------------------------------------------------------------------
# The four rules live here as pure functions over already-extracted facts, so each one can be
# exercised against a RECONSTRUCTED defective input (see the module's proof fixture) without
# breaking the real tree to see it fire.


def unreachable_subpackages(names: Iterable[str], facade: object) -> list[str]:
    """Subpackage names that are not attributes of ``facade`` (rule a)."""
    return sorted(name for name in names if not hasattr(facade, name))


def missing_from_facade(registered: Iterable[str], exported: Iterable[str]) -> list[str]:
    """Registered class names absent from a facade's ``__all__`` (rule b)."""
    return sorted(set(registered) - set(exported))


def domain_roots(classes: Mapping[str, Sequence[str]]) -> list[str]:
    """Classes in an exceptions module with no base defined in that same module (rule c)."""
    return sorted(name for name, bases in classes.items() if not any(b in classes for b in bases))


def orphan_exceptions(classes: Mapping[str, Sequence[str]], root: str) -> list[str]:
    """Classes that do NOT transitively subclass ``root`` (rule c)."""
    orphans = []
    for name in classes:
        seen: set[str] = set()
        stack = [name]
        while stack:
            current = stack.pop()
            if current == root or current in seen:
                break
            seen.add(current)
            stack.extend(b for b in classes.get(current, ()) if b in classes)
        else:
            orphans.append(name)
    return sorted(orphans)


def unraised_exports(classes: Mapping[str, Sequence[str]], exported: Iterable[str], raised: Iterable[str], allowed: Iterable[str] = ()) -> list[str]:
    """Exported exception classes that nothing raises, excluding domain roots and ``allowed`` (rule d)."""
    roots = set(domain_roots(classes))
    exported, raised, allowed = set(exported), set(raised), set(allowed)
    return sorted(name for name in classes if name in exported and name not in roots and name not in raised and name not in allowed)


# --- (a) facade reachability ---------------------------------------------------------------


def test_every_subpackage_is_reachable_from_the_package_facade() -> None:
    unreachable = unreachable_subpackages(_subpackage_dirs(), pyutilz)
    assert not unreachable, (
        "Subpackage(s) shipped under src/pyutilz/ but invisible as attributes of `pyutilz` "
        f"(so `pyutilz.<name>` raises AttributeError): {unreachable}. Add them to _SUBPACKAGES in pyutilz/__init__.py."
    )


def test_every_subpackage_declares_all() -> None:
    """A subpackage with no ``__all__`` binds no submodule for a facade consumer to find, which is how
    ``pyutilz.performance`` shipped as an empty namespace."""
    missing = [name for name in _subpackage_dirs() if not hasattr(importlib.import_module(f"pyutilz.{name}"), "__all__")]
    assert not missing, f"Subpackage(s) with no __all__ (nothing declared public, no submodule bound): {missing}"


# --- (b) registry-vs-facade parity ---------------------------------------------------------


@pytest.mark.parametrize(("facade_name", "registry_module", "registry_attr", "class_index"), _CLASS_REGISTRIES)
def test_registered_classes_are_exported_by_their_facade(facade_name: str, registry_module: str, registry_attr: str, class_index: int) -> None:
    # importorskip, not a try/except: a missing optional dep skips, while a registry that IMPORTS
    # but no longer carries the attribute must fail rather than be reclassified as an environment gap.
    module = pytest.importorskip(registry_module)
    registry = getattr(module, registry_attr)

    facade = importlib.import_module(facade_name)
    exported = set(getattr(facade, "__all__", ()))
    missing = missing_from_facade((value[class_index] for value in registry.values()), exported)
    assert not missing, f"{registry_module}.{registry_attr} registers {missing}, absent from {facade_name}.__all__ -- the public surface disagrees with the registry."


def test_registered_classes_resolve_through_the_facade() -> None:
    """Being listed in ``__all__`` is not the same as being importable: the lazy ``__getattr__`` must
    actually produce the class, or ``from pyutilz.llm import X`` fails at runtime with a green gate."""
    try:
        from pyutilz.llm.factory import _PROVIDER_MODULES
    except ImportError as exc:
        pytest.skip(f"pyutilz.llm.factory not importable here: {exc}")

    import pyutilz.llm as llm

    unresolvable = []
    for _module_path, class_name, _key_attr in _PROVIDER_MODULES.values():
        try:
            getattr(llm, class_name)
        except (AttributeError, ImportError) as exc:
            unresolvable.append(f"{class_name}: {type(exc).__name__}: {exc}")
    assert not unresolvable, "Provider classes listed by the factory but not resolvable on the pyutilz.llm facade:\n  " + "\n  ".join(unresolvable)


# --- (c) exception rooting -----------------------------------------------------------------


@pytest.mark.parametrize("module_path", _exception_modules(), ids=lambda p: p.parent.name)
def test_every_domain_exception_reaches_its_domain_root(module_path: Path) -> None:
    classes = _class_defs(module_path)
    roots = domain_roots(classes)
    assert len(roots) == 1, f"{module_path.parent.name}/exceptions.py must declare exactly ONE domain root (a class with no in-module base); found {roots}."
    root = roots[0]

    orphans = orphan_exceptions(classes, root)
    assert not orphans, (
        f"{module_path.parent.name}/exceptions.py: {orphans} do not transitively subclass {root}, "
        f"so `except {root}` cannot catch every error from this domain."
    )


# --- (d) exported but never raised ----------------------------------------------------------


@pytest.mark.parametrize("module_path", _exception_modules(), ids=lambda p: p.parent.name)
def test_every_exported_exception_is_raised_somewhere(module_path: Path) -> None:
    classes = _class_defs(module_path)
    package = importlib.import_module(f"pyutilz.{module_path.parent.name}")
    # Roots are exempt: a domain root exists to be CAUGHT, and is normally reached via a subclass.
    unraised = unraised_exports(classes, getattr(package, "__all__", ()), _raised_names(), _RAISED_BY_DOWNSTREAM_ONLY)
    assert not unraised, (
        f"{module_path.parent.name}/exceptions.py exports {unraised} but nothing under src/pyutilz/ ever raises them -- "
        "either the raise site was lost, or add the class to _RAISED_BY_DOWNSTREAM_ONLY naming the consumer that raises it."
    )


# --- rule proofs against reconstructed defects -----------------------------------------------
# Each case below is the 2026-09-02 finding as it actually looked, fed to the SAME rule kernel the
# tests above use, so a future edit that quietly defeats a rule fails here instead of passing
# silently on a tree that happens to be clean.


def test_rule_a_catches_a_facade_missing_a_shipped_subpackage() -> None:
    """`pyutilz.stats` and `pyutilz.performance` shipped but raised AttributeError on the facade."""
    import types

    names = ["core", "data", "llm", "performance", "stats"]
    broken = types.SimpleNamespace(core=1, data=1, llm=1)
    assert unreachable_subpackages(names, broken) == ["performance", "stats"]
    assert unreachable_subpackages(names, types.SimpleNamespace(**{n: 1 for n in names})) == []


def test_rule_b_catches_a_registered_provider_missing_from_the_facade() -> None:
    """`OpenAIProvider` was registered by the factory and absent from `pyutilz.llm.__all__`."""
    registered = ["AnthropicProvider", "OpenAIProvider", "OpenRouterProvider"]
    assert missing_from_facade(registered, [n for n in registered if n != "OpenAIProvider"]) == ["OpenAIProvider"]
    assert missing_from_facade(registered, registered) == []


def test_rule_c_catches_exceptions_that_bypass_the_domain_root() -> None:
    """Two of six LLM exception types subclassed only ValueError, so `except LLMProviderError` missed them."""
    defective = {
        "LLMProviderError": ["Exception"],
        "JSONParsingError": ["ValueError"],
        "LLMRefusalError": ["LLMProviderError"],
        "LLMSafetyBlockError": ["LLMRefusalError"],
        "LLMTruncationError": ["ValueError"],
    }
    assert domain_roots(defective) == ["JSONParsingError", "LLMProviderError", "LLMTruncationError"]
    assert orphan_exceptions(defective, "LLMProviderError") == ["JSONParsingError", "LLMTruncationError"]

    fixed = dict(defective, JSONParsingError=["LLMProviderError", "ValueError"], LLMTruncationError=["LLMProviderError", "ValueError"])
    assert domain_roots(fixed) == ["LLMProviderError"]
    assert orphan_exceptions(fixed, "LLMProviderError") == []


def test_rule_d_catches_an_exported_exception_nothing_raises() -> None:
    """`LLMTruncationError` was fully specified, exported, and raised nowhere for a whole wave."""
    classes = {"LLMProviderError": ["Exception"], "LLMRefusalError": ["LLMProviderError"], "LLMTruncationError": ["LLMProviderError"]}
    exported = list(classes)
    assert unraised_exports(classes, exported, ["LLMRefusalError"]) == ["LLMTruncationError"]
    assert unraised_exports(classes, exported, ["LLMRefusalError", "LLMTruncationError"]) == []
    # The domain root itself is exempt: it exists to be caught, not raised directly.
    assert unraised_exports(classes, exported, ["LLMRefusalError", "LLMTruncationError"], allowed=()) == []
