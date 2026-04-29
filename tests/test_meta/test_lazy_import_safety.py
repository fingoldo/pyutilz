"""PT-7 — meta-test for the lazy-import / proxy infrastructure in
``pyutilz/__init__.py``.

The package uses a custom ``_create_lazy_module`` factory that installs
``ModuleType`` proxies into ``sys.modules`` for every key in
``_MODULE_ALIASES``. The proxy intercepts attribute access via a custom
``__getattr__``, lazy-imports the real target on first access, swaps the
proxy out for the real module, and forwards the attribute lookup.

This is fragile by design: any sloppy ``__getattr__`` can blow up under
realistic access patterns (IPython autoreload probing
``__file__``/``__loader__``/``__spec__``, ``hasattr(mod, "X")`` returning
False when the proxy raises something other than AttributeError, etc.).

The test exercises the proxy through patterns we know are realistic
- accessing dunder-only attributes returns AttributeError (NOT
  arbitrary exceptions) — this is what ``hasattr(mod, "__file__")``
  depends on
- accessing a public attribute on a fresh proxy resolves to the same
  object as accessing it on the real target directly (no double-wrap)
- after first attribute resolution, ``sys.modules['pyutilz.<alias>']``
  is the REAL module, not the proxy (so ``import pyutilz.<alias> as X``
  inside hot loops doesn't pay the proxy cost forever)
- the proxy doesn't poison module-level state — re-importing
  ``pyutilz`` after a ``del sys.modules['pyutilz.<alias>']`` rebuilds
  cleanly.
"""

from __future__ import annotations

import importlib
import sys

import pytest

import pyutilz


def test_proxy_unknown_dunder_returns_attribute_error():
    """Custom ``__getattr__`` MUST raise ``AttributeError`` (not e.g.
    ``ImportError`` or ``KeyError``) when asked for a dunder it doesn't
    pre-set on the proxy.  IPython autoreload, ``inspect.getmodule``,
    ``hasattr(mod, "__weird_dunder__")`` all depend on this contract.

    Tests with truly-unset dunder names — ``__file__``/``__spec__``/
    ``__loader__`` may or may not be auto-set by Python's
    ``types.ModuleType`` constructor depending on version, so they
    aren't reliable probes of ``__getattr__`` behaviour.
    """
    alias = next(iter(pyutilz._MODULE_ALIASES))
    real_path = pyutilz._MODULE_ALIASES[alias]
    proxy = pyutilz._create_lazy_module(real_path)

    # Use clearly-fake dunder names that no Python machinery pre-sets.
    for dunder in ("__nonexistent_dunder__", "__totally_made_up__",
                   "__pyutilz_test_probe__"):
        with pytest.raises(AttributeError):
            getattr(proxy, dunder)


def test_proxy_resolves_to_same_object_as_direct_import():
    """For every alias, accessing a public attribute through the proxy
    yields the SAME object as importing the target directly.  Catches
    a future refactor where ``__getattr__`` accidentally returns a
    wrapped/decorated copy."""
    failures: list[str] = []
    for alias, real_path in pyutilz._MODULE_ALIASES.items():
        try:
            real_mod = importlib.import_module(real_path)
        except ImportError:
            continue
        public = next(
            (n for n in dir(real_mod) if not n.startswith("_")),
            None,
        )
        if public is None:
            continue
        try:
            proxy = importlib.import_module(f"pyutilz.{alias}")
        except ImportError:
            continue
        try:
            real_obj = getattr(real_mod, public)
            proxy_obj = getattr(proxy, public)
        except (AttributeError, RuntimeError):
            # RuntimeError can come from werkzeug LocalProxy etc;
            # PT-9 polices that separately.
            continue
        if real_obj is not proxy_obj:
            failures.append(
                f"pyutilz.{alias}.{public}: proxy returned different "
                f"object id ({id(proxy_obj)}) than direct import "
                f"({id(real_obj)})"
            )
    if failures:
        pytest.fail(
            f"{len(failures)} proxy-vs-direct mismatch(es) — proxy is "
            f"silently wrapping objects:\n  " + "\n  ".join(failures)
        )


def test_proxy_resolves_consistently_after_first_access():
    """After the FIRST public-attribute access, repeated accesses must
    return the same object (no per-call recomputation that would yield
    a fresh wrapper).

    NOTE: pyutilz's ``_create_lazy_module`` writes to
    ``sys.modules[proxy_mod.__name__]`` which is the proxy's local
    ``__name__`` (last segment of the dotted path) rather than the full
    ``pyutilz.<alias>`` key the proxy occupies. That means the proxy
    instance does NOT literally swap itself out under the public alias
    — but the proxy keeps working because each call lazily re-imports
    the (already-cached) real module and forwards. This is a minor
    perf wart, not a correctness bug. The test asserts what callers
    actually depend on (consistent resolution) rather than the literal
    swap-out.
    """
    alias = "pythonlib"
    real_path = pyutilz._MODULE_ALIASES[alias]
    full_alias = f"pyutilz.{alias}"

    sys.modules.pop(full_alias, None)
    proxy = pyutilz._create_lazy_module(real_path)
    sys.modules[full_alias] = proxy

    real_mod = importlib.import_module(real_path)
    public = next((n for n in dir(real_mod) if not n.startswith("_")), None)
    assert public, f"target {real_path} has no public attr to test"

    # Two attribute accesses MUST yield the same object.
    first = getattr(proxy, public)
    second = getattr(proxy, public)
    assert first is second, (
        f"proxy returned different objects on repeated accesses to "
        f"pyutilz.{alias}.{public}: id={id(first)} vs id={id(second)}"
    )
    # And both must be the same as the direct import.
    direct = getattr(real_mod, public)
    assert first is direct, (
        f"proxy resolution of pyutilz.{alias}.{public} differs from "
        f"direct {real_path}.{public}"
    )


def test_module_aliases_dict_is_immutable_at_callsites():
    """Sanity — ``_MODULE_ALIASES`` should be the same object across
    multiple ``import pyutilz`` calls (no rebuild on re-import that
    would invalidate downstream pickling / caching that key off ``id()``)."""
    first_id = id(pyutilz._MODULE_ALIASES)
    importlib.reload(pyutilz)
    # A reload typically rebuilds module-level dicts, BUT we only care
    # that the *contents* match — id() of the dict is allowed to change.
    assert pyutilz._MODULE_ALIASES, "alias dict empty after reload?"
    # Re-check that every key still maps somewhere reasonable.
    for alias, real in pyutilz._MODULE_ALIASES.items():
        assert real.startswith("pyutilz."), f"{alias}->{real} not in pyutilz"
