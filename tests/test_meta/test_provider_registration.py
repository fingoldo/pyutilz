"""PT-1 — meta-test for ``pyutilz.llm.factory``.

Every canonical provider name in ``_PROVIDER_MODULES`` must point to an
importable module exposing the named class. Every alias in ``_ALIASES``
must resolve to a real canonical entry. The factory's runtime check
(``ValueError: Unknown provider``) catches user typos but NOT these
config-vs-code drifts; that's what this test exists for.

Catches the failure mode where a provider module is renamed / deleted
but the factory map isn't updated — first user request crashes with
``ImportError`` deep inside ``importlib.import_module``.
"""

from __future__ import annotations

import importlib

import pytest

# llm.factory transitively imports pyutilz.llm.config which requires
# pydantic. CI's default install matrix omits [llm] — gate the module.
pytest.importorskip("pydantic")

from pyutilz.llm import factory as factory_module


def test_every_canonical_provider_module_imports():
    """Every value in ``_PROVIDER_MODULES`` is a real (module_path, class_name)
    pair: the module imports cleanly AND exposes the named class.
    """
    failures: list[str] = []
    for name, (mod_path, cls_name) in factory_module._PROVIDER_MODULES.items():
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as e:
            failures.append(f"{name!r} → {mod_path!r}: ImportError({e})")
            continue
        if not hasattr(mod, cls_name):
            failures.append(
                f"{name!r} → {mod_path}::{cls_name}: module imported but has "
                f"no {cls_name!r} attribute"
            )

    if failures:
        pytest.fail(
            f"{len(failures)} _PROVIDER_MODULES entry(ies) broken — "
            f"first user request via get_llm_provider() will crash:\n  "
            + "\n  ".join(failures)
        )


def test_every_alias_resolves_to_a_canonical_provider():
    """Every key in ``_ALIASES`` maps to a key actually present in
    ``_PROVIDER_MODULES`` — so ``get_llm_provider("claude")`` doesn't
    silently fall through to the "Unknown provider" error path.
    """
    canonicals = set(factory_module._PROVIDER_MODULES.keys())
    bad: list[str] = []
    for alias, canonical in factory_module._ALIASES.items():
        if canonical not in canonicals:
            bad.append(f"{alias!r} → {canonical!r} (not in _PROVIDER_MODULES)")
    if bad:
        pytest.fail(
            f"{len(bad)} alias(es) point to non-existent canonical names:\n  "
            + "\n  ".join(bad)
        )


def test_aliases_dont_collide_with_canonical_names():
    """An alias key must NOT also be a canonical name — otherwise the
    ``_ALIASES.get(name, name)`` resolution path silently overrides the
    canonical lookup.
    """
    overlap = set(factory_module._ALIASES) & set(factory_module._PROVIDER_MODULES)
    if overlap:
        pytest.fail(
            f"{len(overlap)} key(s) appear in BOTH _ALIASES and "
            f"_PROVIDER_MODULES — alias resolution shadows the canonical "
            f"path: {sorted(overlap)}"
        )
