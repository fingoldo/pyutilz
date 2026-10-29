"""Behaviour of ``pyutilz.performance.kernel_tuning.cache.cache_class``: ``register_default_cache`` loads a project's default
tuning file into the shared default layer and degrades to no defaults for a missing or corrupt file, and
``_region_constraints_match`` applies each ``max``/``min``/``eq`` bound only to the axes the caller supplied, with a NaN
dimension failing a bound rather than passing it. The facade's default slot is restored after every test."""

from __future__ import annotations

import math

import pytest

from pyutilz.core import jsonlib
from pyutilz.performance.kernel_tuning import cache as ktc
from pyutilz.performance.kernel_tuning.cache import cache_class
from pyutilz.performance.kernel_tuning.cache._common import _facade


@pytest.fixture
def facade(monkeypatch):
    fac = _facade()
    monkeypatch.setattr(fac, "_DEFAULT_CACHE", None, raising=False)
    return fac


class TestRegisterDefaultCache:
    def test_a_valid_file_registers_its_kernels_as_the_default_layer(self, tmp_path, facade):
        path = tmp_path / "defaults.json"
        path.write_text(jsonlib.dumps({"schema_version": 4, "kernels": {"matmul": {"tuned_utc": "2026-01-01", "regions": []}}}), encoding="utf-8")
        assert cache_class.register_default_cache(str(path)) is True
        assert isinstance(facade._DEFAULT_CACHE, ktc.KernelTuningCache)
        assert "matmul" in facade._DEFAULT_CACHE._loaded["kernels"]

    def test_a_missing_file_registers_nothing_and_says_so(self, tmp_path, facade):
        assert cache_class.register_default_cache(str(tmp_path / "absent.json")) is False
        assert facade._DEFAULT_CACHE is None

    def test_a_corrupt_file_registers_nothing_without_raising(self, tmp_path, facade):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert cache_class.register_default_cache(str(path)) is False
        assert facade._DEFAULT_CACHE is None

    def test_a_file_without_a_schema_version_takes_the_current_one(self, tmp_path, facade):
        path = tmp_path / "defaults.json"
        path.write_text(jsonlib.dumps({"kernels": {}}), encoding="utf-8")
        assert cache_class.register_default_cache(str(path)) is True
        assert facade._DEFAULT_CACHE._loaded["schema_version"] == ktc.SCHEMA_VERSION


class TestRegionConstraints:
    def test_a_max_bound_is_inclusive(self):
        assert cache_class._region_constraints_match({"n": 8}, (("n", cache_class._OP_MAX, 8),)) is True
        assert cache_class._region_constraints_match({"n": 9}, (("n", cache_class._OP_MAX, 8),)) is False

    def test_a_min_bound_is_inclusive(self):
        assert cache_class._region_constraints_match({"n": 8}, (("n", cache_class._OP_MIN, 8),)) is True
        assert cache_class._region_constraints_match({"n": 7}, (("n", cache_class._OP_MIN, 8),)) is False

    def test_an_axis_the_caller_did_not_supply_is_unconstrained(self):
        assert cache_class._region_constraints_match({}, (("n", cache_class._OP_MAX, 1),)) is True

    def test_a_nan_dimension_fails_a_max_bound(self):
        assert cache_class._region_constraints_match({"n": math.nan}, (("n", cache_class._OP_MAX, 8),)) is False

    def test_every_constraint_must_hold(self):
        constraints = (("n", cache_class._OP_MAX, 8), ("m", cache_class._OP_MIN, 4))
        assert cache_class._region_constraints_match({"n": 8, "m": 3}, constraints) is False
        assert cache_class._region_constraints_match({"n": 8, "m": 4}, constraints) is True

    def test_an_equality_bound_compares_the_value(self):
        assert cache_class._region_constraints_match({"kind": "gpu"}, (("kind", cache_class._OP_EQ, "gpu"),)) is True
        assert cache_class._region_constraints_match({"kind": "cpu"}, (("kind", cache_class._OP_EQ, "gpu"),)) is False
