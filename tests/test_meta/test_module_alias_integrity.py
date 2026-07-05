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
            missing = e.name or ""
            if missing.startswith("pyutilz"):
                # The pyutilz target module itself is gone — real drift.
                failures.append(f"{alias!r} → {real_path!r}: {e}")
            # else: transitive optional dep missing — silently skip.
        except ImportError as e:
            failures.append(f"{alias!r} → {real_path!r}: ImportError({e})")
    if failures:
        pytest.fail(
            f"{len(failures)} alias target(s) fail to import — downstream " f"``from pyutilz.<alias> import X`` will crash:\n  " + "\n  ".join(failures)
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
            failures.append(f"pyutilz.{alias} (proxy) failed to import: {e}")
            continue
        # Trigger __getattr__ on the proxy. Resolution to ``None`` is OK
        # — many modules expose lazily-initialised constants that are
        # ``None`` until first use (``logginglib.EXTERNAL_IP``).
        try:
            getattr(proxy, public)
        except AttributeError as e:
            failures.append(f"pyutilz.{alias}.{public}: proxy returned AttributeError " f"({e}) but real target {real_path!r} does have {public!r}")
    if failures:
        pytest.fail(f"{len(failures)} alias proxy resolution failure(s):\n  " + "\n  ".join(failures))


def test_alias_keys_dont_collide_with_subpackages():
    """An alias key must not collide with a real sub-package — else the
    alias silently shadows the canonical sub-package import path.

    Acknowledged exception in pyutilz/__init__.py: aliases are NOT
    created for names that conflict with sub-packages (``system``,
    ``web``, ``cloud``). The test asserts that comment-as-policy.
    """
    sub_packages = {"core", "data", "database", "web", "cloud", "text", "system", "dev", "llm"}
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
        pytest.fail(f"{len(bad)} alias target(s) point outside the pyutilz " f"namespace:\n  " + "\n  ".join(bad))


def test_lazy_proxy_does_not_shadow_real_toplevel_packages():
    """A lazy alias proxy must replace ITSELF in ``sys.modules`` under its OWN
    fully-qualified key (``pyutilz.<alias>``) on first attribute access -- NEVER
    under the bare leaf name.

    Regression: ``_create_lazy_module`` built the proxy with ``__name__`` set to the
    bare leaf (``tokenizers``), so the first attribute access wrote
    ``sys.modules['tokenizers'] = pyutilz.text.tokenizers`` and GLOBALLY shadowed the
    unrelated HuggingFace top-level ``tokenizers`` package -- which broke
    ``transformers`` (``from tokenizers import Encoding``) and every SHAP TreeExplainer
    downstream. The alias whose leaf collides with an installed top-level package is the
    canonical tripwire.
    """
    import importlib.util

    def _is_pyutilz_file(mod) -> bool:
        path = str(getattr(mod, "__file__", "") or "").replace("\\", "/")
        return "/pyutilz/" in path

    # Alias keys whose bare leaf is ALSO a real installed top-level package (the collision case).
    risky = {}
    for alias, real_path in pyutilz._MODULE_ALIASES.items():
        leaf = real_path.rsplit(".", 1)[-1]
        if leaf != alias:
            continue
        try:
            spec = importlib.util.find_spec(alias)
        except (ImportError, ValueError):
            spec = None
        if spec is not None and "pyutilz" not in (spec.origin or ""):
            risky[alias] = real_path
    if not risky:
        pytest.skip("no alias leaf collides with an installed top-level package on this host")

    for alias, real_path in risky.items():
        proxy_key = f"pyutilz.{alias}"
        proxy = sys.modules.get(proxy_key) or importlib.import_module(proxy_key)
        # Force the proxy's __getattr__ to fire (mirrors transformers touching the module),
        # which is what triggered the bad ``sys.modules[<leaf>] = pyutilz_module`` write.
        real_mod = importlib.import_module(real_path)
        public = next((c for c in dir(real_mod) if not c.startswith("_")), None)
        if public is not None:
            try:
                getattr(proxy, public)
            except Exception:
                pass
        # The bare top-level key must NOT now point at a pyutilz module.
        shadow = sys.modules.get(alias)
        assert shadow is None or not _is_pyutilz_file(shadow), (
            f"alias {alias!r} shadowed the bare top-level ``{alias}`` in sys.modules with the "
            f"pyutilz proxy ({getattr(shadow, '__file__', '?')}); it must cache under {proxy_key!r}."
        )
        # And the genuine top-level package must still import to its real (non-pyutilz) location.
        real_toplevel = importlib.import_module(alias)
        assert not _is_pyutilz_file(real_toplevel), (
            f"``import {alias}`` resolved to the pyutilz module {getattr(real_toplevel,'__file__','?')} " f"instead of the real top-level package."
        )
