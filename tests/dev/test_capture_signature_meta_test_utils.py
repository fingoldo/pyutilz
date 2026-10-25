"""Regression tests for pyutilz.dev.meta_test_utils.capture_signature's default-value capture.

Regression (2026-07-21 audit round 2, HIGH): capture_signature() -- the function backing
test_api_stability.py's public-API drift detector, "the highest-value defensive test for a
backward-compat-oriented library" per that test's own docstring -- used to capture only
name:kind:has_default per parameter. Two functions ``def f(verbose=False)`` and
``def f(verbose=True)`` produced the byte-identical signature string, so a silent
default-value flip (a real, common backward-compat break) went completely undetected.
"""

from __future__ import annotations

from pyutilz.dev.meta_test_utils import capture_signature


class TestDefaultValueIsCaptured:
    def test_bool_default_flip_is_detected(self):
        def f_false(verbose=False):
            pass

        def f_true(verbose=True):
            pass

        assert capture_signature(f_false) != capture_signature(f_true)

    def test_int_default_flip_is_detected(self):
        def f_zero(retries=0):
            pass

        def f_five(retries=5):
            pass

        assert capture_signature(f_zero) != capture_signature(f_five)

    def test_str_default_flip_is_detected(self):
        def f_a(mode="a"):
            pass

        def f_b(mode="b"):
            pass

        assert capture_signature(f_a) != capture_signature(f_b)

    def test_none_default_is_captured(self):
        def f_none(x=None):
            pass

        def f_zero(x=0):
            pass

        assert capture_signature(f_none) != capture_signature(f_zero)

    def test_float_default_flip_is_detected(self):
        def f_low(threshold=0.5):
            pass

        def f_high(threshold=0.9):
            pass

        assert capture_signature(f_low) != capture_signature(f_high)

    def test_tuple_of_literals_default_flip_is_detected(self):
        def f_a(axes=(1, 2)):
            pass

        def f_b(axes=(3, 4)):
            pass

        assert capture_signature(f_a) != capture_signature(f_b)

    def test_identical_defaults_produce_identical_signatures(self):
        def f1(verbose=False):
            pass

        def f2(verbose=False):
            pass

        assert capture_signature(f1) == capture_signature(f2)

    def test_no_default_still_captured_as_before(self):
        def f(x):
            pass

        assert "x:any:0" in capture_signature(f)


class TestUnsafeDefaultsDoNotChurn:
    """Objects/sentinels that repr() unstably (memory address, etc.) must NOT be captured
    verbatim -- that would make the snapshot flap across runs even with an unchanged default."""

    def test_object_sentinel_default_does_not_leak_repr(self):
        sentinel = object()

        def f(x=sentinel):
            pass

        sig = capture_signature(f)
        assert "0x" not in sig  # no raw memory address leaked into the snapshot
        assert "<unstable-default>" in sig

    def test_two_distinct_object_sentinels_produce_identical_signatures(self):
        """Since both are "unstable" and collapse to the same placeholder, two DIFFERENT
        (but equally unstable) sentinel defaults must not spuriously look different."""

        def f1(x=object()):
            pass

        def f2(x=object()):
            pass

        assert capture_signature(f1) == capture_signature(f2)

    def test_mutable_list_default_does_not_leak_repr(self):
        def f(x=[1, 2, 3]):  # noqa: B006 -- deliberately testing a mutable-default edge case
            pass

        sig = capture_signature(f)
        assert "<unstable-default>" in sig
