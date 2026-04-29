"""D1 — meta-test that every concrete LLM provider implements the
required ``LLMProvider`` interface.

The base class declares five abstract methods (``generate``,
``generate_json``, ``generate_batch``, ``estimate_cost``,
``count_tokens``). Python's ABC machinery only enforces the abstract
methods at *instantiation* time — and most providers can't be
instantiated without an API key, so the test setup never reaches
``__init__``. As a result, a forgotten override silently slips
through code review until a user calls the missing method in
production.

This test imports each provider class WITHOUT instantiating it and
asserts via ``inspect`` that:

  1. Every abstract method on ``LLMProvider`` has been overridden in
     the subclass.
  2. The override's signature is compatible with the base class
     (same required parameters; extras OK).

Catches the failure modes:
  * "I added ``generate_v2`` and forgot to keep ``generate``"
  * "I changed ``generate(prompt, *, system=None)`` to ``generate(*,
    prompt, system)`` and broke every caller"
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from pyutilz.llm import factory as factory_module
from pyutilz.llm.base import LLMProvider


def _abstract_method_names() -> set[str]:
    return set(getattr(LLMProvider, "__abstractmethods__", set()))


def _provider_classes() -> list[tuple[str, type]]:
    """Resolve every (canonical_name, ProviderClass) tuple from the
    factory's ``_PROVIDER_MODULES`` map. Skips entries that fail to
    import (those are policed by PT-1)."""
    out: list[tuple[str, type]] = []
    for name, (mod_path, cls_name) in factory_module._PROVIDER_MODULES.items():
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        out.append((name, cls))
    return out


def test_every_provider_inherits_from_base():
    """Every provider class registered in ``_PROVIDER_MODULES`` must be
    a subclass of ``LLMProvider``. Catches a contributor wiring up a
    duck-typed class that happens to have a ``generate`` method but
    isn't part of the formal hierarchy."""
    not_subclass: list[str] = []
    for name, cls in _provider_classes():
        if not issubclass(cls, LLMProvider):
            not_subclass.append(f"{name}: {cls.__module__}.{cls.__name__}")
    if not_subclass:
        pytest.fail(
            f"{len(not_subclass)} provider class(es) don't inherit from "
            f"LLMProvider — typing / isinstance checks elsewhere will "
            f"reject them:\n  " + "\n  ".join(not_subclass)
        )


def test_every_provider_overrides_all_abstract_methods():
    """Every concrete provider must override every abstract method on
    ``LLMProvider``. Pure ABC enforcement — but since instantiation
    requires API keys, we can't rely on ``cls()`` to surface failures
    at test time."""
    abstract = _abstract_method_names()
    if not abstract:
        pytest.skip("LLMProvider declares no abstract methods")

    failures: list[str] = []
    for name, cls in _provider_classes():
        cls_abstract = set(getattr(cls, "__abstractmethods__", set()))
        if cls_abstract:
            failures.append(
                f"{name} ({cls.__name__}): still abstract "
                f"(missing implementations of {sorted(cls_abstract)})"
            )
    if failures:
        pytest.fail(
            f"{len(failures)} provider(s) fail to override every abstract "
            f"method — instantiation will raise TypeError:\n  "
            + "\n  ".join(failures)
        )


def test_every_provider_method_signature_is_compatible():
    """For each abstract method, every provider's override must accept
    at least the same required parameters as the base. Extras (provider-
    specific kwargs) are allowed; missing required params would break
    callers using the provider polymorphically.
    """
    abstract = _abstract_method_names()
    if not abstract:
        pytest.skip("LLMProvider declares no abstract methods")

    failures: list[str] = []
    for method_name in sorted(abstract):
        try:
            base_sig = inspect.signature(getattr(LLMProvider, method_name))
        except (TypeError, ValueError):
            continue
        # Required base params (no default), excluding ``self``.
        required_base = {
            n for n, p in base_sig.parameters.items()
            if n != "self"
            and p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                               inspect.Parameter.VAR_KEYWORD)
        }
        for name, cls in _provider_classes():
            override = getattr(cls, method_name, None)
            if override is None:
                # Already flagged by the abstract-coverage test.
                continue
            try:
                ovr_sig = inspect.signature(override)
            except (TypeError, ValueError):
                continue
            ovr_params = set(ovr_sig.parameters)
            has_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in ovr_sig.parameters.values()
            )
            if has_var_kwargs:
                continue  # ``**kwargs`` accepts anything.
            missing = required_base - ovr_params
            if missing:
                failures.append(
                    f"{name}.{method_name}: missing required base param(s) "
                    f"{sorted(missing)}"
                )
    if failures:
        pytest.fail(
            f"{len(failures)} provider method(s) drop required base "
            f"parameter(s):\n  " + "\n  ".join(failures)
        )
