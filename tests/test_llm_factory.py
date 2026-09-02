"""Tests for pyutilz.llm.factory."""

import pytest
from unittest.mock import patch

pytest.importorskip("pydantic")

from pyutilz.llm.factory import (
    get_llm_provider,
    _provider_cache,
    _PROVIDER_MODULES,
    _ALIASES,
)


class TestAliases:
    @pytest.mark.parametrize("alias,canonical", [
        ("claude", "anthropic"),
        ("google", "gemini"),
        ("cc", "claude-code"),
        ("ds", "deepseek"),
        ("grok", "xai"),
    ])
    def test_alias_resolves(self, alias, canonical):
        assert _ALIASES[alias] == canonical

    def test_all_aliases_point_to_valid_constructors(self):
        for alias, canonical in _ALIASES.items():
            assert canonical in _PROVIDER_MODULES, f"Alias '{alias}' → '{canonical}' has no constructor"


class TestProviderModules:
    def test_has_at_least_five_providers(self):
        assert len(_PROVIDER_MODULES) >= 5

    def test_all_entries_are_module_class_tuples(self):
        for name, entry in _PROVIDER_MODULES.items():
            assert isinstance(entry, tuple) and len(entry) == 3, f"Entry for '{name}' should be (module_path, class_name, settings_api_key_attr)"
            mod_path, cls_name, key_attr = entry
            assert isinstance(mod_path, str) and isinstance(cls_name, str)
            # None means "this provider needs no API key". claude-code is the only such provider
            # (it drives the local Claude Code CLI/SDK, no key involved); for everyone else the
            # attribute must be a real, non-empty pyutilz Settings field name. The previous
            # `key_attr is None or isinstance(key_attr, str)` accepted BOTH shapes for every
            # provider, so a key-requiring provider silently losing its key attribute passed.
            if name == "claude-code":
                assert key_attr is None
            else:
                assert isinstance(key_attr, str) and key_attr, f"Provider '{name}' must name a settings API-key attribute"
                assert key_attr.endswith("_api_key"), f"Provider '{name}' key attr '{key_attr}' does not follow the <provider>_api_key convention"


class TestGetLlmProvider:
    def setup_method(self):
        _provider_cache.clear()

    def teardown_method(self):
        _provider_cache.clear()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_provider("nonexistent_provider_xyz")

    def test_unknown_provider_lists_available(self):
        with pytest.raises(ValueError, match="Available"):
            get_llm_provider("bad")

    def test_case_insensitive(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_provider("NONEXISTENT")

    @pytest.mark.parametrize("name", ["claude-code", "cc", "CC"])
    def test_claude_code_variants_accepted(self, name):
        try:
            get_llm_provider(name)
        except ValueError as e:
            if "Unknown provider" in str(e):
                pytest.fail(f"Provider '{name}' should be recognized")

    def test_cache_returns_same_instance(self):
        """Two lookups of the same provider name return the identical cached instance.

        The construction calls are deliberately NOT wrapped in try/except+pytest.skip any more:
        that swallowed every genuine regression (an unhashable cache key, a broken constructor)
        as an "environment" skip, leaving the caching contract -- the entire point of this test
        -- unverified. Only the provider MODULE's importability is treated as an environment
        precondition, which is what importorskip is for.
        """
        mod_path, _cls_name, _key_attr = _PROVIDER_MODULES["claude-code"]
        pytest.importorskip(mod_path)

        p1 = get_llm_provider("claude-code")
        p2 = get_llm_provider("claude-code")
        assert p1 is p2
        assert _provider_cache  # the instance really came from the cache, not a fresh build each time

    def test_default_is_claude_code(self):
        import inspect
        sig = inspect.signature(get_llm_provider)
        assert sig.parameters["provider_name"].default == "claude-code"

    def test_unhashable_kwarg_bypasses_cache_but_is_tracked_for_cleanup(self):
        # Regression: a provider built with an unhashable kwarg bypasses the
        # instance cache; it must still be tracked (in _uncached_providers) so
        # the atexit closer shuts its HTTP client instead of leaking it.
        from pyutilz.llm import factory

        created = []

        class _FakeLeakProvider:
            def __init__(self, **kwargs):
                created.append(self)

        # Register a controllable fake provider resolvable via importlib.
        globals()["_FakeLeakProvider"] = _FakeLeakProvider
        with patch.dict(factory._PROVIDER_MODULES, {"faketest": ("tests.test_llm_factory", "_FakeLeakProvider", None)}):
            returned = get_llm_provider("faketest", extra_headers=["unhashable", "list"])

        assert returned is created[0]
        assert returned not in _provider_cache.values()  # cache was bypassed
        assert returned in list(factory._uncached_providers)  # but tracked for cleanup


class TestProviderCacheLRUEviction:
    """Regression (2026-07-21 audit round 2, MEDIUM): ``_provider_cache`` used to be a plain
    unbounded dict -- every distinct (provider, kwargs) combination a long-running process ever
    saw stayed cached (and its live resources open) forever. It's now an ``OrderedDict``-backed
    LRU bounded by ``_PROVIDER_CACHE_MAX_SIZE``, evicting the least-recently-used entry and
    scheduling its close."""

    def setup_method(self):
        from pyutilz.llm import factory
        factory._provider_cache.clear()
        factory._uncached_providers.clear()
        self._orig_max_size = factory._PROVIDER_CACHE_MAX_SIZE
        # Regression fix: _register_fake() and test_evicted_provider_scheduled_for_close_when_no_running_loop
        # below REASSIGN factory._PROVIDER_MODULES to a brand-new dict (not patch.dict, which
        # auto-restores) -- with no restore here, the leftover "faketest" entry stayed in the
        # module's real provider registry for the REST OF THE TEST SESSION, independently observed
        # as a hard-to-reproduce, order-dependent failure in both
        # test_llm_account_credits.py::test_both_methods_exist_on_every_provider ("faketest: missing
        # get_account_credits") and test_meta/test_provider_contract.py's own scan of
        # _PROVIDER_MODULES, purely depending on pytest-randomly's execution order.
        self._orig_provider_modules = dict(factory._PROVIDER_MODULES)

    def teardown_method(self):
        from pyutilz.llm import factory
        factory._provider_cache.clear()
        factory._uncached_providers.clear()
        factory._PROVIDER_CACHE_MAX_SIZE = self._orig_max_size
        factory._PROVIDER_MODULES = self._orig_provider_modules

    def _register_fake(self, factory):
        created = []

        class _FakeProvider:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(self)

        globals()["_FakeCacheLRUProvider"] = _FakeProvider
        factory._PROVIDER_MODULES = dict(factory._PROVIDER_MODULES, faketest=("tests.test_llm_factory", "_FakeCacheLRUProvider", None))
        return created

    def test_cache_bounded_by_max_size(self):
        from pyutilz.llm import factory
        factory._PROVIDER_CACHE_MAX_SIZE = 3
        created = self._register_fake(factory)

        for i in range(5):
            get_llm_provider("faketest", model=f"m{i}")

        assert len(created) == 5
        assert len(_provider_cache) == 3

    def test_lru_order_evicts_least_recently_used_not_oldest_created(self):
        """Touching an old entry (a cache hit) must protect it from eviction --
        a plain FIFO (not LRU) would evict it purely by insertion order."""
        from pyutilz.llm import factory
        factory._PROVIDER_CACHE_MAX_SIZE = 2
        self._register_fake(factory)

        first = get_llm_provider("faketest", model="m0")
        get_llm_provider("faketest", model="m1")
        # Touch "m0" again -- it becomes the most-recently-used, so the NEXT
        # insertion should evict "m1" (least-recently-used), not "m0".
        touched_again = get_llm_provider("faketest", model="m0")
        assert touched_again is first
        get_llm_provider("faketest", model="m2")

        cached_models = {kwargs.get("model") for (name, kwargs_tuple) in _provider_cache for kwargs in [dict(kwargs_tuple)]}
        assert "m0" in cached_models
        assert "m1" not in cached_models
        assert "m2" in cached_models

    def test_evicted_provider_scheduled_for_close_when_no_running_loop(self):
        """Without a running event loop, an evicted provider (that exposes ``_close``) must
        fall back to ``_uncached_providers`` so the atexit handler still closes it."""
        from pyutilz.llm import factory
        factory._PROVIDER_CACHE_MAX_SIZE = 1

        class _ClosableFakeProvider:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def _close(self):
                pass

        globals()["_ClosableFakeProvider"] = _ClosableFakeProvider
        factory._PROVIDER_MODULES = dict(factory._PROVIDER_MODULES, faketest=("tests.test_llm_factory", "_ClosableFakeProvider", None))

        first = get_llm_provider("faketest", model="m0")
        get_llm_provider("faketest", model="m1")  # evicts "m0" (no running loop in this sync test)

        assert first in list(factory._uncached_providers)
        assert len(_provider_cache) == 1
