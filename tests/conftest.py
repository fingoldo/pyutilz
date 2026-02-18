import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_df():
    """Simple test DataFrame"""
    return pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e']
    })


@pytest.fixture
def mixed_types_df():
    """DataFrame with various dtypes for optimization testing"""
    return pd.DataFrame({
        'large_int': np.array([1, 2, 3], dtype=np.int64),
        'small_int': [1, 2, 3],
        'float64_col': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'category': pd.Categorical(['a', 'b', 'a']),
        'object_col': ['x', 'y', 'z']
    })


@pytest.fixture
def temp_dir():
    """Temporary directory for file I/O tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def float_with_integers_df():
    """DataFrame with float columns that have no fractional part"""
    return pd.DataFrame({
        'float_int': [1.0, 2.0, 3.0, 4.0],
        'float_real': [1.1, 2.2, 3.3, 4.4],
        'float_with_nan': [1.0, 2.0, np.nan, 4.0]
    })


@pytest.fixture
def constant_columns_df():
    """DataFrame with some constant columns"""
    return pd.DataFrame({
        'const1': [1, 1, 1, 1],
        'const2': ['a', 'a', 'a', 'a'],
        'varying': [1, 2, 3, 4],
        'varying2': ['a', 'b', 'c', 'd']
    })
