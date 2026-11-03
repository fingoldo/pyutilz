"""Signature, default, boundary, guard and argument-order pins for list_openrouter_models (health survivors, third batch)."""

from __future__ import annotations

import inspect
import logging
import time

import pytest

import pyutilz.llm.openrouter_provider as pkg
from pyutilz.llm.openrouter_provider import _health as h


def _catalogue(rows: dict) -> None:
    pkg._MODELS_CATALOGUE = rows
    pkg._MODELS_CATALOGUE_FETCHED_AT = time.monotonic()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(pkg, "_MODELS_CATALOGUE", None)
    monkeypatch.setattr(pkg, "_MODELS_CATALOGUE_FETCHED_AT", float("inf"))
    monkeypatch.setattr(pkg, "_HEALTH_CACHE", {})
    monkeypatch.setattr(pkg, "_resolve_or_api_key", lambda explicit: None)


class TestSignature:
    def test_the_keyword_only_marker_follows_refresh(self):
        params = inspect.signature(h.list_openrouter_models).parameters
        assert params["refresh"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params["sort_by"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_a_positional_sort_by_is_refused(self):
        with pytest.raises(TypeError):
            h.list_openrouter_models(False, "name")

    def test_the_documented_defaults(self):
        params = inspect.signature(h.list_openrouter_models).parameters
        assert params["max_workers"].default == 8
        assert params["timeout"].default == 30.0
        assert params["health_ttl_seconds"].default == 300.0
        assert params["min_uptime"].default == 0.95


class TestPriceBoundary:
    def test_a_model_priced_exactly_at_the_input_cap_is_kept(self):
        _catalogue({"m": {"id": "m", "pricing": {"prompt": "0.000001", "completion": "0"}}})
        out = h.list_openrouter_models(max_input_per_1m=1.0, return_only_healthy=False)
        assert [r["id"] for r in out] == ["m"]

    def test_a_model_priced_exactly_at_the_output_cap_is_kept(self):
        _catalogue({"m": {"id": "m", "pricing": {"prompt": "0", "completion": "0.000002"}}})
        out = h.list_openrouter_models(max_output_per_1m=2.0, return_only_healthy=False)
        assert [r["id"] for r in out] == ["m"]


class TestHealthGuard:
    def test_without_a_key_the_health_filter_is_skipped_and_no_warning_is_emitted(self, monkeypatch, caplog):
        _catalogue({"m": {"id": "m", "pricing": {}}})
        caplog.set_level(logging.WARNING)
        monkeypatch.setattr(h, "_enrich_with_health", lambda *a, **k: pytest.fail("enrichment must not run without a key"))
        out = h.list_openrouter_models(return_only_healthy=False)
        assert [r["id"] for r in out] == ["m"]
        assert "needs an OpenRouter API key" not in caplog.text

    def test_with_return_only_healthy_and_no_key_the_warning_names_the_fix(self, caplog):
        _catalogue({"m": {"id": "m", "pricing": {}}})
        caplog.set_level(logging.WARNING)
        h.list_openrouter_models(return_only_healthy=True)
        assert caplog.records[-1].getMessage() == (
            "list_openrouter_models(return_only_healthy=True) needs an OpenRouter API key (env OPENROUTER_API_KEY or api_key=); "
            "skipping health filter -- returning Stage-1 results."
        )

    def test_with_a_key_the_enrichment_receives_its_arguments_in_order(self, monkeypatch):
        _catalogue({"m": {"id": "m", "pricing": {}}})
        monkeypatch.setattr(pkg, "_resolve_or_api_key", lambda explicit: "KEY")
        seen = []

        def fake(rows, key, min_uptime, max_workers, timeout, ttl):
            seen.append((len(rows), key, min_uptime, max_workers, timeout, ttl))
            return rows

        monkeypatch.setattr(h, "_enrich_with_health", fake)
        h.list_openrouter_models(min_uptime=0.5, max_workers=3, timeout=7.0, health_ttl_seconds=60.0)
        assert seen == [(1, "KEY", 0.5, 3, 7.0, 60.0)]

    def test_the_default_enrichment_arguments_are_the_documented_ones(self, monkeypatch):
        _catalogue({"m": {"id": "m", "pricing": {}}})
        monkeypatch.setattr(pkg, "_resolve_or_api_key", lambda explicit: "KEY")
        seen = []
        monkeypatch.setattr(h, "_enrich_with_health", lambda rows, key, min_uptime, max_workers, timeout, ttl: seen.append((max_workers, timeout, ttl)) or rows)
        h.list_openrouter_models()
        assert seen == [(8, 30.0, 300.0)]
