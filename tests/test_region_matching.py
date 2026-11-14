"""Behaviour of ``pyutilz.performance.kernel_tuning.cache.region_matching``: a region matches a dims dict when every
constraint it carries holds (``_max`` caps, ``_min`` floors, ``_eq`` exact values with tuples equal to lists), an
absent or None constraint is unconstrained, and a NaN dim matches no constrained region."""

from __future__ import annotations

import numpy as np

from pyutilz.performance.kernel_tuning.cache.region_matching import _region_match_reason, _region_matches


class TestConstraints:
    def test_a_region_without_constraints_matches_anything(self):
        assert _region_matches({"decision": "x"}, {"n": 10}) is True

    def test_the_cap_is_inclusive(self):
        assert _region_matches({"n_max": 10}, {"n": 10}) is True
        assert _region_matches({"n_max": 10}, {"n": 11}) is False

    def test_the_floor_is_inclusive(self):
        assert _region_matches({"n_min": 10}, {"n": 10}) is True
        assert _region_matches({"n_min": 10}, {"n": 9}) is False

    def test_an_exact_value_must_be_equal(self):
        assert _region_matches({"device_eq": "cpu"}, {"device": "cpu"}) is True
        assert _region_matches({"device_eq": "cpu"}, {"device": "gpu"}) is False

    def test_a_none_constraint_is_unconstrained(self):
        assert _region_matches({"n_max": None, "n_min": None}, {"n": 1e9}) is True

    def test_a_dim_without_a_constraint_key_is_ignored(self):
        assert _region_matches({"n_max": 10}, {"n": 5, "other": "anything"}) is True


class TestTuplesAndLists:
    def test_a_tuple_value_matches_its_list_form(self):
        assert _region_matches({"shape_eq": [2, 3]}, {"shape": (2, 3)}) is True

    def test_a_tuple_constraint_matches_a_list_dim(self):
        assert _region_matches({"shape_eq": (2, 3)}, {"shape": [2, 3]}) is True


class TestNaN:
    def test_a_nan_dim_matches_no_constrained_region(self):
        ok, reason = _region_match_reason({"x_max": 10}, {"x": float("nan")})
        assert ok is False
        assert "NaN" in reason

    def test_a_numpy_nan_is_treated_the_same_way(self):
        assert _region_matches({"x_min": 0}, {"x": np.float64("nan")}) is False

    def test_a_nan_dim_is_fine_when_the_region_does_not_constrain_it(self):
        assert _region_matches({"y_max": 10}, {"x": float("nan"), "y": 5}) is True


class TestReason:
    def test_the_reason_names_the_violated_cap(self):
        ok, reason = _region_match_reason({"n_max": 10}, {"n": 11})
        assert ok is False
        assert reason == "n=11 > n_max=10"

    def test_a_satisfied_region_reports_all_constraints_satisfied(self):
        assert _region_match_reason({"n_max": 10}, {"n": 3}) == (True, "all constraints satisfied")


class TestRegionMatchingConstantsAndReasons:
    """The suffix and op-code constants are exact, a NaN is recognised only for a float, and each refusal names the axis, the value and the bound in its exact text
    (region_matching 12-62)."""

    def test_the_axis_suffixes_are_exact(self):
        from pyutilz.performance.kernel_tuning.cache import region_matching as rm

        assert rm._AXIS_SUFFIXES == ("_max", "_min", "_eq")

    def test_the_op_codes_are_distinct_and_exact(self):
        from pyutilz.performance.kernel_tuning.cache import region_matching as rm

        assert (rm._OP_MAX, rm._OP_MIN, rm._OP_EQ) == (0, 1, 2)
        assert rm._SUFFIX_OPS == (("_max", 0), ("_min", 1), ("_eq", 2))

    def test_a_nan_is_recognised_only_for_a_float(self):
        import numpy as np

        from pyutilz.performance.kernel_tuning.cache import region_matching as rm

        assert rm._is_nan(float("nan")) is True
        assert rm._is_nan(np.float64("nan")) is True
        assert rm._is_nan(3) is False
        assert rm._is_nan("x") is False

    def test_each_refusal_names_the_axis_value_and_bound(self):
        from pyutilz.performance.kernel_tuning.cache import region_matching as rm

        assert rm._region_match_reason({"x_max": 5}, {"x": 6}) == (False, "x=6 > x_max=5")
        assert rm._region_match_reason({"x_min": 5}, {"x": 4}) == (False, "x=4 < x_min=5")
        assert rm._region_match_reason({"x_eq": "a"}, {"x": "b"}) == (False, "x='b' != x_eq='a'")
        assert rm._region_match_reason({"x_max": 5}, {"x": float("nan")}) == (False, "x is NaN")
