"""PT-2 — meta-test for ``pyutilz._MODULE_ALIASES``.

pyutilz exposes 24+ backward-compat module aliases (``pyutilz.pythonlib``
→ ``pyutilz.core.pythonlib``, ``pyutilz.pandaslib`` →
``pyutilz.data.pandaslib``, etc.) so legacy code that imports the old
flat module paths keeps working after the package was reorganised into
sub-packages.

Downstream code does ``from pyutilz.pandaslib import X`` — if the real
target is renamed / deleted, the alias silently breaks at import time
on the user's machine. This test catches the drift in CI:

  (1) Every value in ``_MODULE_ALIASES`` is a real importable module.
  (2) ``import pyutilz.<alias>`` (the proxy form) returns a working
      module that exposes at least one public symbol from the real
      target.
  (3) No alias key collides with an actual sub-package name (would
      shadow it).
"""

from __future__ import annotations

import importlib
import sys

import pytest

import pyutilz


def test_every_alias_target_imports():
    """The right-hand side of every alias maps to a module that imports
    cleanly.

    Distinguishes "pyutilz module missing" (real failure) from "transitive
    optional third-party dep missing" (skip — CI's default install matrix
    intentionally omits ``filelock``/``jellyfish``/``flask``/``IPython``;
    those alias targets only matter when the user opts in).
    """
    failures: list[str] = []
    for alias, real_path in pyutilz._MODULE_ALIASES.items():
        try:
            importlib.import_module(real_path)
        except ModuleNotFoundError as e:
            missing = (e.name or "")
            if missing.startswith("pyutilz"):
                # The pyutilz target module itself is gone — real drift.
                failures.append(f"{alias!r} → {real_path!r}: {e}")
            # else: transitive optional dep missing — silently skip.
        except ImportError as e:
            failures.append(f"{alias!r} → {real_path!r}: ImportError({e})")
    if failures:
        pytest.fail(
            f"{len(failures)} alias target(s) fail to import — downstream "
            f"``from pyutilz.<alias> import X`` will crash:\n  "
            + "\n  ".join(failures)
        )


def test_every_alias_proxy_resolves_a_public_symbol():
    """The proxy module installed at ``pyutilz.<alias>`` returns a real
    object when a public symbol is accessed.

    We don't enumerate the target's full surface — just ensure SOMETHING
    public resolves through the proxy. (Walking every public name would
    needlessly tax the lazy-import system at test time.)
    """
    failures: list[str] = []
    for alias, real_path in pyutilz._MODULE_ALIASES.items():
        try:
            real_mod = importlib.import_module(real_path)
        except ImportError:
            continue  # Already reported by the prior test.
        # Find any public attribute in the real module.
        public = next(
            (n for n in dir(real_mod) if not n.startswith("_")),
            None,
        )
        if public is None:
            # Empty modules are OK — no symbols to alias-test.
            continue
        try:
            proxy = importlib.import_module(f"pyutilz.{alias}")
        except ImportError as e:
            failures.append(
                f"pyutilz.{alias} (proxy) failed to import: {e}"
            )
            continue
        # Trigger __getattr__ on the proxy. Resolution to ``None`` is OK
        # — many modules expose lazily-initialised constants that are
        # ``None`` until first use (``logginglib.EXTERNAL_IP``).
        try:
            getattr(proxy, public)
        except AttributeError as e:
            failures.append(
                f"pyutilz.{alias}.{public}: proxy returned AttributeError "
                f"({e}) but real target {real_path!r} does have {public!r}"
            )
    if failures:
        pytest.fail(
            f"{len(failures)} alias proxy resolution failure(s):\n  "
            + "\n  ".join(failures)
        )


def test_alias_keys_dont_collide_with_subpackages():
    """An alias key must not collide with a real sub-package — else the
    alias silently shadows the canonical sub-package import path.

    Acknowledged exception in pyutilz/__init__.py: aliases are NOT
    created for names that conflict with sub-packages (``system``,
    ``web``, ``cloud``). The test asserts that comment-as-policy.
    """
    sub_packages = {"core", "data", "database", "web", "cloud", "text",
                    "system", "dev", "llm"}
    overlap = set(pyutilz._MODULE_ALIASES) & sub_packages
    if overlap:
        pytest.fail(
            f"{len(overlap)} alias key(s) collide with real sub-package "
            f"names: {sorted(overlap)} — would shadow ``import pyutilz.<X>`` "
            f"on direct attribute access."
        )


def test_alias_targets_live_under_pyutilz_namespace():
    """Every alias must point to a module *inside* ``pyutilz.`` — not
    accidentally to a third-party module that happens to share a name."""
    bad: list[str] = []
    for alias, real_path in pyutilz._MODULE_ALIASES.items():
        if not real_path.startswith("pyutilz."):
            bad.append(f"{alias!r} → {real_path!r}")
    if bad:
        pytest.fail(
            f"{len(bad)} alias target(s) point outside the pyutilz "
            f"namespace:\n  " + "\n  ".join(bad)
        )
