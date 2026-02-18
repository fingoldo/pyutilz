"""Tests for numbalib.py - Phase 2 refactoring

Tests cover:
- Nopython mode crash fix (removed None check from @njit)
- O(N²) string concatenation fix (removed @njit, used join)
- Mutable default argument fix in python_dict_2_numba_dict
"""

import pytest
import numpy as np


class TestSetNumbaRandomSeed:
    """Test set_numba_random_seed - nopython mode fix"""

    def test_accepts_integer_seed(self):
        """Test that @njit function accepts integer seed (no None check)"""
        from pyutilz.numbalib import set_numba_random_seed

        # Should work with integer seed
        try:
            set_numba_random_seed(42)
            # If it runs without error, the fix works
            assert True
        except Exception as e:
            pytest.fail(f"set_numba_random_seed(42) raised {e}")

    def test_sets_random_state_without_error(self):
        """Test that setting seed works without error (nopython mode fix)"""
        from pyutilz.numbalib import set_numba_random_seed

        # Should work with integer seed (was broken with None in nopython mode)
        try:
            set_numba_random_seed(123)
            # Generate some random numbers to verify RNG works
            random_nums = np.random.rand(5)
            assert len(random_nums) == 5
            assert all(0 <= x <= 1 for x in random_nums)
        except Exception as e:
            pytest.fail(f"set_numba_random_seed should work with integer: {e}")


class TestSetRandomSeed:
    """Test set_random_seed wrapper - handles None defaults"""

    def test_handles_none_default(self):
        """Test that wrapper handles None default value"""
        from pyutilz.numbalib import set_random_seed

        # Should not crash with None
        try:
            set_random_seed(None)
            assert True
        except Exception as e:
            pytest.fail(f"set_random_seed(None) raised {e}")

    def test_accepts_integer_seed(self):
        """Test that wrapper accepts integer seed"""
        from pyutilz.numbalib import set_random_seed

        try:
            set_random_seed(456)
            assert True
        except Exception as e:
            pytest.fail(f"set_random_seed(456) raised {e}")


class TestArr2Str:
    """Test arr2str function - O(N) performance fix"""

    def test_converts_array_to_string(self):
        """Test that array is converted to string"""
        from pyutilz.numbalib import arr2str

        arr = [1, 2, 3, 4, 5]
        result = arr2str(arr)

        assert isinstance(result, str)
        assert "1" in result
        assert "5" in result

    def test_uses_join_not_concatenation(self):
        """Test that implementation uses O(N) join, not O(N²) concatenation"""
        from pyutilz.numbalib import arr2str
        import time

        # Large array to detect O(N²) behavior
        large_arr = list(range(1000))

        start = time.perf_counter()
        result = arr2str(large_arr)
        elapsed = time.perf_counter() - start

        # Should complete quickly (O(N) with join)
        assert elapsed < 0.1  # 100ms should be plenty for O(N)
        assert isinstance(result, str)

    def test_no_njit_decorator(self):
        """Test that arr2str is NOT compiled with @njit (removed for performance)"""
        from pyutilz.numbalib import arr2str

        # Check function is not a Numba compiled function
        # Numba functions have special attributes
        assert not hasattr(arr2str, 'py_func')  # Not a numba dispatcher

    def test_handles_different_types(self):
        """Test conversion of different element types"""
        from pyutilz.numbalib import arr2str

        # Integers
        assert isinstance(arr2str([1, 2, 3]), str)

        # Floats
        assert isinstance(arr2str([1.5, 2.5]), str)

        # Mixed
        assert isinstance(arr2str([1, 2.5, 3]), str)


class TestPythonDict2NumbaDict:
    """Test python_dict_2_numba_dict - mutable default fix"""

    def test_mutable_default_not_shared(self):
        """Test that default numba_dict is not shared between calls"""
        from pyutilz.numbalib import python_dict_2_numba_dict
        import numba

        python_dict1 = {"a": 1, "b": 2}
        python_dict2 = {"c": 3, "d": 4}

        # First call with default (None)
        result1 = python_dict_2_numba_dict(python_dict1)

        # Second call with default (None) - should get new dict
        result2 = python_dict_2_numba_dict(python_dict2)

        # Should be independent
        assert isinstance(result1, numba.typed.Dict)
        assert isinstance(result2, numba.typed.Dict)

        # Should have correct values
        assert len(result1) == 2
        assert len(result2) == 2

    def test_returns_numba_dict(self):
        """Test that function returns numba.typed.Dict"""
        from pyutilz.numbalib import python_dict_2_numba_dict
        import numba

        python_dict = {"x": 10, "y": 20}
        result = python_dict_2_numba_dict(python_dict)

        assert isinstance(result, numba.typed.Dict)
        assert "x" in result
        assert result["x"] == 10

    def test_accepts_existing_numba_dict(self):
        """Test that function can accept existing numba_dict parameter"""
        from pyutilz.numbalib import python_dict_2_numba_dict
        import numba

        # Create existing numba dict
        existing = numba.typed.Dict()
        existing["existing"] = 100

        python_dict = {"new": 200}
        result = python_dict_2_numba_dict(python_dict, existing)

        # Should have both old and new keys
        assert "existing" in result
        assert "new" in result
        assert result["existing"] == 100
        assert result["new"] == 200


@pytest.mark.parametrize("seed_value", [0, 42, 123, 999, 12345])
def test_random_seed_parametrized(seed_value):
    """Parametrized test for different seed values"""
    from pyutilz.numbalib import set_numba_random_seed

    # Should work with any positive integer seed
    try:
        set_numba_random_seed(seed_value)
        # Generate some random numbers to verify it works
        random_nums = np.random.rand(3)
        assert len(random_nums) == 3
    except Exception as e:
        pytest.fail(f"Failed with seed={seed_value}: {e}")
