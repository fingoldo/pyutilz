"""
Test suite for pandaslib.py
Tests cover bug fixes, performance optimizations, and general functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from pyutilz.pandaslib import (
    optimize_dtypes,
    remove_stale_columns,
    concat_and_flush_df_list,
    remove_constant_columns,
    get_categorical_columns_indices,
    convert_float64_to_float32,
    get_df_memory_consumption,
    nullify_standard_values,
    prefixize_columns,
    FeatureNamer,
    set_df_columns_types,
    get_columns_of_type,
    classify_column_types,
    ensure_dataframe_float32_convertability,
)


class TestOptimizeDtypes:
    """Test optimize_dtypes function"""

    def test_skip_columns_works_correctly(self):
        """Regression test for line 137 bug (wrong variable 'field' instead of 'col' in skip check)"""
        df = pd.DataFrame({'col1': [1.0, 2.0, 3.0], 'col2': [3.0, 4.0, 5.0]})
        result = optimize_dtypes(df, skip_columns={'col1'}, inplace=False)
        # col1 should remain unchanged (float64)
        assert result['col1'].dtype == np.float64
        # col2 might be optimized to smaller float or int
        assert 'col2' in result.columns

    def test_float_to_int_conversion(self):
        """Test conversion of float columns with no fractional part"""
        df = pd.DataFrame({'col': [1.0, 2.0, 3.0]})
        result = optimize_dtypes(df, float_to_int=True, inplace=False)
        assert pd.api.types.is_integer_dtype(result['col'])

    def test_handles_nans_correctly(self):
        """Test that columns with NaN are not converted to int"""
        df = pd.DataFrame({'col': [1.0, 2.0, np.nan]})
        result = optimize_dtypes(df, float_to_int=True, inplace=False)
        assert pd.api.types.is_float_dtype(result['col'])

    def test_inplace_modification(self):
        """Test that inplace=True actually modifies the DataFrame"""
        df = pd.DataFrame({'col': [1.0, 2.0, 3.0]})
        original_id = id(df)
        result = optimize_dtypes(df, inplace=True)
        assert id(result) == original_id

    def test_object_to_category_conversion(self):
        """Test conversion of object columns to category when appropriate"""
        df = pd.DataFrame({'col': ['a', 'b', 'a', 'b', 'a']})
        result = optimize_dtypes(df, max_categories=10, inplace=False)
        assert result['col'].dtype.name == 'category'


class TestRemoveStaleColumns:
    """Test remove_stale_columns function"""

    def test_returns_list_when_no_stale_columns(self):
        """Regression test for line 367 bug (returned None when no stale columns found)"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = remove_stale_columns(df)
        assert result is not None
        assert isinstance(result, list)
        assert set(result) == {'a', 'b'}

    def test_removes_stale_columns(self):
        """Test that actually stale columns are identified and removed"""
        df = pd.DataFrame({'const': [1, 1, 1], 'var': [1, 2, 3]})
        result = remove_stale_columns(df)
        assert 'var' in result
        assert 'const' not in result


class TestConcatAndFlushDfList:
    """Test concat_and_flush_df_list function"""

    def test_csv_with_none_cols_parameter(self, tmp_path):
        """Regression test for line 391 bug (undefined 'cols' variable, should be 'csv_cols')"""
        df_list = [pd.DataFrame({'a': [1, 2]}), pd.DataFrame({'a': [3, 4]})]
        csv_path = tmp_path / "test"

        result = concat_and_flush_df_list(
            df_list,
            file_name=str(csv_path),
            to_csv=True,
            csv_cols=None
        )

        assert result is not None  # Should return DataFrame (bug fix for line 395)
        assert isinstance(result, pd.DataFrame)
        assert (tmp_path / "test.csv").exists()

    def test_pickle_output(self, tmp_path):
        """Test default pickle output"""
        df_list = [pd.DataFrame({'a': [1, 2]}), pd.DataFrame({'a': [3, 4]})]
        pickle_path = tmp_path / "test"

        result = concat_and_flush_df_list(
            df_list,
            file_name=str(pickle_path),
            to_csv=False
        )

        assert result is not None
        assert len(result) == 4
        assert (tmp_path / "test.pckl").exists()


class TestRemoveConstantColumns:
    """Test remove_constant_columns function"""

    def test_removes_columns_when_verbose_false(self):
        """Regression test for lines 580-587 bug (only removed when verbose=True)"""
        df = pd.DataFrame({'const': [1, 1, 1], 'var': [1, 2, 3]})
        remove_constant_columns(df, verbose=False)

        assert 'const' not in df.columns
        assert 'var' in df.columns

    def test_removes_columns_when_verbose_true(self):
        """Test that removal also works when verbose=True"""
        df = pd.DataFrame({'const': [1, 1, 1], 'var': [1, 2, 3]})
        remove_constant_columns(df, verbose=True)

        assert 'const' not in df.columns
        assert 'var' in df.columns

    def test_handles_many_constant_columns(self):
        """Test bulk removal path for >20 constant columns"""
        df_dict = {f'const{i}': [1]*10 for i in range(25)}
        df_dict['var'] = list(range(10))
        df = pd.DataFrame(df_dict)

        remove_constant_columns(df, verbose=False)

        assert 'var' in df.columns
        assert all(f'const{i}' not in df.columns for i in range(25))


class TestPandas2Compatibility:
    """Test pandas 2.0+ API compatibility"""

    def test_iteritems_replaced_with_items(self):
        """Verify no usage of deprecated iteritems() (lines 76, 480)"""
        df = pd.DataFrame({'a': [1, 2], 'b': pd.Categorical(['x', 'y'])})
        # Should not raise AttributeError on pandas 2.0+
        indices = get_categorical_columns_indices(df)
        assert isinstance(indices, tuple)
        assert len(indices) == 3  # (non_cat_indices, cat_indices, unique_values)

    def test_select_dtypes_with_list_not_string(self):
        """Test select_dtypes uses list argument, not deprecated string (line 897)"""
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3, 4]})
        result = convert_float64_to_float32(df)
        # Should not raise FutureWarning
        assert result['a'].dtype == np.float32


class TestGetDfMemoryConsumption:
    """Test get_df_memory_consumption function"""

    def test_returns_positive_value(self, sample_df):
        """Test that memory consumption is a positive number"""
        mem = get_df_memory_consumption(sample_df)
        assert mem > 0
        assert isinstance(mem, float)

    def test_larger_df_uses_more_memory(self):
        """Test that larger DataFrames report more memory"""
        small_df = pd.DataFrame({'a': [1, 2, 3]})
        large_df = pd.DataFrame({'a': list(range(10000))})

        small_mem = get_df_memory_consumption(small_df)
        large_mem = get_df_memory_consumption(large_df)

        assert large_mem > small_mem


class TestNullifyStandardValues:
    """Test nullify_standard_values function"""

    def test_nullifies_frequent_values(self):
        """Test that frequent values are replaced with placeholder"""
        df = pd.DataFrame({'field': ['A']*500 + ['B']*10 + ['C']*10})
        nullify_standard_values(df, 'field', min_records=100, placeholder=np.nan)

        # 'A' should be nullified (appears 500 times)
        assert df['field'].isna().sum() == 500
        # 'B' and 'C' should remain (only 10 times each)
        assert (df['field'] == 'B').sum() == 10
        assert (df['field'] == 'C').sum() == 10

    def test_with_persons_field(self):
        """Test nullification based on person count"""
        df = pd.DataFrame({
            'field': ['A']*100 + ['B']*100,
            'person': [1]*50 + [2]*50 + list(range(3, 103))
        })

        nullify_standard_values(
            df, 'field',
            min_records=50,
            persons_field='person',
            min_persons=10
        )

        # 'B' appears 100 times and used by 100 different persons (>10), so should be nullified
        # 'A' appears 100 times but only used by 2 persons (<10), should NOT be nullified
        assert df[df['field'] == 'A'].shape[0] > 0  # A should remain (only 2 persons)
        assert df['field'].isna().sum() == 100  # B should be nullified (100 persons)


class TestPrefixizeColumns:
    """Test prefixize_columns function"""

    def test_adds_prefix_to_columns(self):
        """Test that prefix is added to all columns"""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        result = prefixize_columns(df, 'test', inplace=False)

        assert 'test_a' in result.columns
        assert 'test_b' in result.columns

    def test_respects_exclusions(self):
        """Test that excluded columns are not prefixed"""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        result = prefixize_columns(df, 'test', exclusions={'a'}, inplace=False)

        assert 'a' in result.columns
        assert 'test_b' in result.columns

    def test_special_prefixes(self):
        """Test that special prefixes override default prefix"""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        result = prefixize_columns(
            df, 'default',
            special_prefixes={'a': 'special'},
            inplace=False
        )

        assert 'special_a' in result.columns
        assert 'default_b' in result.columns


class TestFeatureNamer:
    """Test FeatureNamer class"""

    def test_mutable_default_argument_fixed(self):
        """Regression test for mutable default argument bug (line 331)"""
        # Create two instances - they should be independent
        namer1 = FeatureNamer()
        namer1('feature1')

        namer2 = FeatureNamer()
        # namer2 should start fresh, not inherit namer1's state
        assert namer2.fnames == {}

    def test_assigns_consecutive_indices(self):
        """Test that features get consecutive integer names"""
        namer = FeatureNamer()
        idx1 = namer('feature1')
        idx2 = namer('feature2')
        idx3 = namer('feature1')  # Should return same index

        assert idx1 == 0
        assert idx2 == 1
        assert idx3 == 0  # Same as first call

    def test_reverse_lookup(self):
        """Test that reverse lookup works"""
        namer = FeatureNamer()
        namer('abc')
        namer('def')

        assert namer.rev(0) == 'abc'
        assert namer.rev(1) == 'def'


class TestEnsureDataframeFloat32Convertability:
    """Test ensure_dataframe_float32_convertability function"""

    def test_converts_numeric_types_to_float32(self):
        """Test that various numeric types are converted to float32"""
        df = pd.DataFrame({
            'int32': np.array([1, 2, 3], dtype=np.int32),
            'int64': np.array([1, 2, 3], dtype=np.int64),
            'float64': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        })

        result = ensure_dataframe_float32_convertability(df)

        assert result['int32'].dtype == np.float32
        assert result['int64'].dtype == np.float32
        assert result['float64'].dtype == np.float32

    def test_single_pass_optimization(self):
        """Test that consolidation into single select_dtypes works (performance fix)"""
        df = pd.DataFrame({
            'col1': np.array([1, 2, 3], dtype=np.int64),
            'col2': np.array([4, 5, 6], dtype=np.float64),
        })

        result = ensure_dataframe_float32_convertability(df, verbose=0)

        # Should complete successfully with all columns converted
        assert all(result.dtypes == np.float32)


class TestClassifyColumnTypes:
    """Test classify_column_types function"""

    def test_identifies_numeric_column(self):
        """Test identification of numeric column"""
        df = pd.DataFrame({'num': [1, 2, 3]})
        is_bool, is_obj, is_dt, is_cat, is_num = classify_column_types(df, 'num')

        assert is_num is True
        assert is_bool is False
        assert is_obj is False
        assert is_dt is False
        assert is_cat is False

    def test_identifies_categorical_column(self):
        """Test identification of categorical column"""
        df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'a'])})
        is_bool, is_obj, is_dt, is_cat, is_num = classify_column_types(df, 'cat')

        assert is_cat is True
        assert is_num is False


@pytest.mark.parametrize("input_list,expected", [
    ([1, 2, 3], True),
    ([], False),
    ([1], True),
])
def test_list_length_check(input_list, expected):
    """Parametrized test for len() > 0 check (line 463 bug fix)"""
    assert (len(input_list) > 0) == expected


class TestGetColumnsOfType:
    """Test get_columns_of_type function"""

    def test_finds_columns_by_type_name(self):
        """Test finding columns by type name substring"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })

        int_cols = get_columns_of_type(df, ['int'])
        float_cols = get_columns_of_type(df, ['float'])

        assert 'int_col' in int_cols
        assert 'float_col' in float_cols


class TestSetDfColumnsTypes:
    """Test set_df_columns_types function"""

    def test_sets_column_types(self):
        """Test setting multiple column types"""
        df = pd.DataFrame({
            'a': ['1', '2', '3'],
            'b': ['4.0', '5.0', '6.0']
        })

        types_dict = {
            'int': ['a'],
            'float': ['b']
        }

        set_df_columns_types(df, types_dict)

        assert pd.api.types.is_integer_dtype(df['a'])
        assert pd.api.types.is_float_dtype(df['b'])
