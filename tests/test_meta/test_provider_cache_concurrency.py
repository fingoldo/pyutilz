"""D3 — meta-test that ``pyutilz.llm.factory.get_llm_provider`` is
thread-safe: N parallel callers asking for the same provider get the
same instance back, with no double-instantiation, no race on the cache,
and no exception escapes.

The factory uses a module-level dict + lock (``_provider_cache``,
``_provider_lock``) and the canonical "double-checked locking" pattern.
That pattern is correct in CPython BUT trivially breaks under:

  * a refactor that drops the second cache check inside the locked
    section,
  * a refactor that moves ``import``-time work (which holds the GIL
    differently) inside the locked section,
  * a refactor that swaps ``threading.Lock`` for an ``RLock``-leaking
    helper.

Hard to catch by code review; cheap to catch with a stress test.
Mocks the actual provider classes so no API keys / network are
required, and so we can count exact instantiations.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from pyutilz.llm import factory as factory_module


class _CountingFakeProvider:
    """Fake provider whose ``__init__`` increments a class-level
    counter — lets the test count exactly how many times the factory
    constructed an instance under contention."""

    instantiations = 0
    _lock = threading.Lock()

    def __init__(self, **kwargs):
        with type(self)._lock:
            type(self).instantiations += 1
        self.kwargs = kwargs


@pytest.fixture
def reset_factory_state():
    """Clear the factory's module-level cache + reset the fake counter
    so tests don't bleed into each other.  ``yield`` so the cleanup runs
    AFTER the test even if it errored."""
    factory_module._provider_cache.clear()
    _CountingFakeProvider.instantiations = 0
    yield
    factory_module._provider_cache.clear()


def _patch_provider_resolution(monkeypatch):
    """Make ``get_llm_provider("anthropic")`` resolve to the fake
    provider class without touching the real anthropic_provider module
    (which would require an API key)."""
    # Replace the canonical module mapping with one that resolves to
    # the fake provider for ``anthropic``.
    fake_modules = {"anthropic": ("pyutilz.llm.factory",
                                  "_CountingFakeProvider")}
    monkeypatch.setattr(factory_module, "_PROVIDER_MODULES", fake_modules)
    monkeypatch.setattr(factory_module, "_ALIASES", {})
    # Stash the fake class on the factory module so importlib finds it
    # via the module-path lookup.
    monkeypatch.setattr(factory_module, "_CountingFakeProvider",
                        _CountingFakeProvider, raising=False)


def test_concurrent_callers_share_one_instance(monkeypatch, reset_factory_state):
    """20 concurrent ``get_llm_provider("anthropic", model="x")`` calls
    must yield the SAME instance, instantiated exactly once."""
    _patch_provider_resolution(monkeypatch)

    n_callers = 20
    instances: list = []
    with ThreadPoolExecutor(max_workers=n_callers) as ex:
        futures = [
            ex.submit(factory_module.get_llm_provider,
                      provider_name="anthropic", model="test_model")
            for _ in range(n_callers)
        ]
        for f in as_completed(futures):
            instances.append(f.result())

    # All callers got the same instance (identity check, not equality).
    first = instances[0]
    assert all(inst is first for inst in instances), (
        f"got {len(set(id(i) for i in instances))} distinct instances "
        f"under contention — cache lookup not properly synchronised"
    )
    # And the constructor ran exactly once.
    assert _CountingFakeProvider.instantiations == 1, (
        f"provider instantiated "
        f"{_CountingFakeProvider.instantiations} time(s) for {n_callers} "
        f"identical-kwargs callers — cache double-check missing or "
        f"lock acquired AFTER the constructor"
    )


def test_distinct_kwargs_produce_distinct_instances(monkeypatch,
                                                    reset_factory_state):
    """Different kwargs MUST yield distinct instances — caching by
    ``(name, kwargs)`` key. Catches a refactor that accidentally
    coalesces ``(name=anthropic, model=A)`` with ``(name=anthropic,
    model=B)``."""
    _patch_provider_resolution(monkeypatch)

    a = factory_module.get_llm_provider(provider_name="anthropic",
                                        model="model_a")
    b = factory_module.get_llm_provider(provider_name="anthropic",
                                        model="model_b")
    assert a is not b, "distinct model kwargs returned the same instance"
    assert _CountingFakeProvider.instantiations == 2


def test_repeated_same_kwargs_dont_reinstantiate(monkeypatch,
                                                 reset_factory_state):
    """Sequential repeated calls with identical kwargs return the
    cached instance — no growth in instantiation count."""
    _patch_provider_resolution(monkeypatch)

    a = factory_module.get_llm_provider(provider_name="anthropic",
                                        model="model_x")
    b = factory_module.get_llm_provider(provider_name="anthropic",
                                        model="model_x")
    c = factory_module.get_llm_provider(provider_name="anthropic",
                                        model="model_x")
    assert a is b is c
    assert _CountingFakeProvider.instantiations == 1
