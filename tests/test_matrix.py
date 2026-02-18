"""
Test suite for matrix.py
Tests cover sparse matrix utilities and CSR matrix constructors.
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix

from pyutilz.matrix import (
    CsrIndPtrConstructor,
    CsrRowColConstructor,
    get_sparse_memory_usage
)


class TestCsrIndPtrConstructor:
    """Test CsrIndPtrConstructor class for building CSR matrices"""

    def test_basic_construction(self):
        """Test basic CSR matrix construction"""
        constructor = CsrIndPtrConstructor()

        # Build a simple 3x3 matrix:
        # [1, 0, 2]
        # [0, 3, 0]
        # [4, 5, 0]

        # Row 0: elements at columns 0, 2
        constructor.add_element(1, 0)
        constructor.add_element(2, 2)
        constructor.add_row()

        # Row 1: element at column 1
        constructor.add_element(3, 1)
        constructor.add_row()

        # Row 2: elements at columns 0, 1
        constructor.add_element(4, 0)
        constructor.add_element(5, 1)
        constructor.add_row()

        # Build matrix
        matrix = constructor.build_matrix(dtype=np.int32)

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (3, 3)

        # Check values
        assert matrix[0, 0] == 1
        assert matrix[0, 2] == 2
        assert matrix[1, 1] == 3
        assert matrix[2, 0] == 4
        assert matrix[2, 1] == 5

        # Check zeros
        assert matrix[0, 1] == 0
        assert matrix[1, 0] == 0
        assert matrix[2, 2] == 0

    def test_empty_row(self):
        """Test matrix with empty row"""
        constructor = CsrIndPtrConstructor()

        # Row 0: has element
        constructor.add_element(1, 0)
        constructor.add_row()

        # Row 1: empty row
        constructor.add_row()

        # Row 2: has element
        constructor.add_element(2, 1)
        constructor.add_row()

        matrix = constructor.build_matrix(dtype=np.int32)

        assert matrix.shape == (3, 2)
        assert matrix[0, 0] == 1
        assert matrix[1, 0] == 0  # Empty row
        assert matrix[1, 1] == 0
        assert matrix[2, 1] == 2

    def test_float_dtype(self):
        """Test with float data type"""
        constructor = CsrIndPtrConstructor()

        constructor.add_element(1.5, 0)
        constructor.add_element(2.7, 1)
        constructor.add_row()

        matrix = constructor.build_matrix(dtype=np.float64)

        assert matrix.dtype == np.float64
        assert matrix[0, 0] == 1.5
        assert matrix[0, 1] == 2.7

    def test_single_row(self):
        """Test single row matrix"""
        constructor = CsrIndPtrConstructor()

        constructor.add_element(1, 0)
        constructor.add_element(2, 2)
        constructor.add_element(3, 5)
        constructor.add_row()

        matrix = constructor.build_matrix(dtype=np.int32)

        assert matrix.shape == (1, 6)
        assert matrix[0, 0] == 1
        assert matrix[0, 2] == 2
        assert matrix[0, 5] == 3


class TestCsrRowColConstructor:
    """Test CsrRowColConstructor class for building CSR matrices"""

    def test_basic_construction(self):
        """Test basic CSR matrix construction with row/col format"""
        constructor = CsrRowColConstructor()

        # Build a 3x3 matrix:
        # [1, 0, 2]
        # [0, 3, 0]
        # [4, 5, 0]

        constructor.add_element(1, row=0, col=0)
        constructor.add_element(2, row=0, col=2)
        constructor.add_element(3, row=1, col=1)
        constructor.add_element(4, row=2, col=0)
        constructor.add_element(5, row=2, col=1)

        matrix = constructor.build_matrix(dtype=np.int32)

        assert isinstance(matrix, csr_matrix)
        assert matrix[0, 0] == 1
        assert matrix[0, 2] == 2
        assert matrix[1, 1] == 3
        assert matrix[2, 0] == 4
        assert matrix[2, 1] == 5

    def test_unordered_elements(self):
        """Test adding elements in random order"""
        constructor = CsrRowColConstructor()

        # Add elements in non-sequential order
        constructor.add_element(5, row=2, col=1)
        constructor.add_element(1, row=0, col=0)
        constructor.add_element(3, row=1, col=1)
        constructor.add_element(2, row=0, col=2)

        matrix = constructor.build_matrix(dtype=np.int32)

        assert matrix[0, 0] == 1
        assert matrix[0, 2] == 2
        assert matrix[1, 1] == 3
        assert matrix[2, 1] == 5

    def test_duplicate_position(self):
        """Test adding multiple values to same position (should sum)"""
        constructor = CsrRowColConstructor()

        # Add to same position twice
        constructor.add_element(3, row=0, col=0)
        constructor.add_element(7, row=0, col=0)

        matrix = constructor.build_matrix(dtype=np.int32)

        # CSR matrix sums duplicate entries
        assert matrix[0, 0] == 10

    def test_large_sparse_matrix(self):
        """Test building larger sparse matrix"""
        constructor = CsrRowColConstructor()

        # Create 100x100 matrix with only diagonal elements
        for i in range(100):
            constructor.add_element(i + 1, row=i, col=i)

        matrix = constructor.build_matrix(dtype=np.int32)

        assert matrix.shape == (100, 100)

        # Check diagonal
        for i in range(100):
            assert matrix[i, i] == i + 1

        # Check sparsity
        assert matrix.nnz == 100  # Only 100 non-zero elements


class TestGetSparseMemoryUsage:
    """Test get_sparse_memory_usage function"""

    def test_csr_matrix_memory(self):
        """Test memory calculation for CSR matrix"""
        # Create small CSR matrix
        data = np.array([1, 2, 3, 4], dtype=np.int32)
        indices = np.array([0, 2, 1, 0], dtype=np.int32)
        indptr = np.array([0, 2, 3, 4], dtype=np.int32)

        matrix = csr_matrix((data, indices, indptr), shape=(3, 3))

        memory = get_sparse_memory_usage(matrix)

        # Should be sum of data, indices, and indptr arrays
        expected = data.nbytes + indices.nbytes + indptr.nbytes
        assert memory == expected
        assert memory > 0

    def test_coo_matrix_memory(self):
        """Test memory calculation for COO matrix"""
        row = np.array([0, 0, 1, 2], dtype=np.int32)
        col = np.array([0, 2, 1, 0], dtype=np.int32)
        data = np.array([1, 2, 3, 4], dtype=np.int32)

        matrix = coo_matrix((data, (row, col)), shape=(3, 3))

        memory = get_sparse_memory_usage(matrix)

        # Should be sum of data, row, and col arrays
        expected = data.nbytes + row.nbytes + col.nbytes
        assert memory == expected
        assert memory > 0

    def test_csc_matrix_memory(self):
        """Test with CSC matrix (should return -1 for unsupported type)"""
        data = np.array([1, 2, 3, 4], dtype=np.int32)
        indices = np.array([0, 1, 2, 0], dtype=np.int32)
        indptr = np.array([0, 2, 3, 4], dtype=np.int32)

        matrix = csc_matrix((data, indices, indptr), shape=(3, 3))

        memory = get_sparse_memory_usage(matrix)

        # CSC not explicitly supported, should return -1
        assert memory == -1

    def test_invalid_input(self):
        """Test with non-matrix input"""
        memory = get_sparse_memory_usage("not a matrix")
        assert memory == -1

        memory = get_sparse_memory_usage(None)
        assert memory == -1

        memory = get_sparse_memory_usage(42)
        assert memory == -1

    def test_dense_array_input(self):
        """Test with dense numpy array (should return -1)"""
        arr = np.array([[1, 2], [3, 4]])
        memory = get_sparse_memory_usage(arr)
        assert memory == -1

    def test_different_dtypes(self):
        """Test memory calculation with different data types"""
        # Float64 CSR matrix
        data_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.int32)
        indptr = np.array([0, 1, 2, 3], dtype=np.int32)

        matrix_f64 = csr_matrix((data_f64, indices, indptr))
        memory_f64 = get_sparse_memory_usage(matrix_f64)

        # Float32 CSR matrix (same structure)
        data_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        matrix_f32 = csr_matrix((data_f32, indices, indptr))
        memory_f32 = get_sparse_memory_usage(matrix_f32)

        # Float64 should use more memory than float32
        assert memory_f64 > memory_f32

    def test_empty_matrix_memory(self):
        """Test memory calculation for empty sparse matrix"""
        matrix = csr_matrix((0, 0), dtype=np.int32)
        memory = get_sparse_memory_usage(matrix)

        # Empty matrix still has some overhead
        assert memory >= 0


class TestEdgeCases:
    """Test edge cases"""

    def test_constructor_no_clear_source(self):
        """Test building matrix without clearing source data"""
        constructor = CsrRowColConstructor()

        constructor.add_element(1, row=0, col=0)
        constructor.add_element(2, row=0, col=1)

        # Build without clearing
        matrix = constructor.build_matrix(dtype=np.int32, clear_source=False)

        # Source data should still exist
        assert len(constructor.data) == 2
        assert len(constructor.rows) == 2
        assert len(constructor.cols) == 2

    def test_constructor_clear_source(self):
        """Test building matrix with clearing source data"""
        constructor = CsrRowColConstructor()

        constructor.add_element(1, row=0, col=0)
        constructor.add_element(2, row=0, col=1)

        # Build with clearing (default)
        matrix = constructor.build_matrix(dtype=np.int32, clear_source=True)

        # Source data should be deleted - accessing should raise AttributeError
        with pytest.raises(AttributeError):
            _ = constructor.data

    def test_very_sparse_matrix(self):
        """Test memory efficiency of very sparse matrix"""
        constructor = CsrRowColConstructor()

        # 1000x1000 matrix with only 10 elements
        for i in range(10):
            constructor.add_element(1, row=i * 100, col=i * 100)

        matrix = constructor.build_matrix(dtype=np.int32)

        memory = get_sparse_memory_usage(matrix)

        # Sparse should be much smaller than dense (1000*1000*4 bytes)
        dense_memory = 1000 * 1000 * 4
        assert memory < dense_memory / 100  # At least 100x smaller
