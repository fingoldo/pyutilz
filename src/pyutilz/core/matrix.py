"""Incremental builders and memory-usage helpers for scipy sparse (CSR/COO) matrices."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, List

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

class CsrIndPtrConstructor:
    """Incremental builder of CSR matrices from (indptr, indices, data) triples.

    Call add_element() for each non-zero of the current row, then add_row() to
    close the row; finally build_matrix() to materialise a scipy csr_matrix.
    """
    def __init__(self) -> None:
        self.indptr = [0]
        self.indices: List[Any] = []
        self.data: List[Any] = []
    def add_element(self, element, index: int) -> None:
        """Append a non-zero value and its column index to the current (still open) row."""
        self.indices.append(index)
        self.data.append(element)

    def add_row(self) -> None:
        """Close the current row by recording the current element count as the next indptr boundary."""
        self.indptr.append(len(self.indices))

    def build_matrix(self, dtype: np.dtype, clear_source: bool = True) -> csr_matrix:
        """Materialise the accumulated (data, indices, indptr) triple into a scipy csr_matrix.

        Args:
            dtype: dtype of the resulting matrix's data array.
            clear_source: if True, free the internal buffers after building.
        """
        matrix = csr_matrix((self.data, self.indices, self.indptr), dtype=dtype)
        if clear_source:
            del self.data, self.indices, self.indptr
        return matrix

class CsrRowColConstructor:
    """Incremental builder of CSR matrices from a COO (row, col, data) stream.

    Elements may be added in any order; duplicate (row, col) positions are summed
    by scipy on build_matrix().
    """
    def __init__(self) -> None:
        self.rows: List[Any] = []
        self.cols: List[Any] = []
        self.data: List[Any] = []
    def add_element(self, element, row: int, col: int) -> None:
        """Append a non-zero value at the given (row, col) coordinate, in any order."""
        self.rows.append(row)
        self.cols.append(col)
        self.data.append(element)

    def build_matrix(self, dtype: np.dtype, clear_source: bool = True) -> csr_matrix:
        """Materialise the accumulated (row, col, data) triples into a scipy csr_matrix.

        Duplicate (row, col) entries are summed by scipy during construction.

        Args:
            dtype: dtype of the resulting matrix's data array.
            clear_source: if True, free the internal buffers after building.
        """
        matrix = csr_matrix((self.data, (self.rows, self.cols)), dtype=dtype)
        if clear_source:
            del self.data, self.rows, self.cols
        return matrix

def get_sparse_memory_usage(mat: object) -> int:
    """
    Return mem usage of a csr, csc, or coo matrix.

    Raises ``TypeError`` for an unsupported ``mat`` type, matching ``get_df_memory_consumption``'s
    convention elsewhere in the package -- an int return space that also used ``-1`` as an
    "unsupported input" sentinel let callers silently under-count real memory usage when summing
    across mixed container types.
    """
    try:
        if isinstance(mat, (csr_matrix, csc_matrix)):
            # csr and csc share the same (data, indptr, indices) layout, just transposed semantics.
            return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
        elif isinstance(mat, coo_matrix):
            return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
        else:
            raise TypeError(f"Unsupported sparse matrix type: {type(mat)}")
    except AttributeError as e:
        logger.warning("get_sparse_memory_usage: AttributeError while reading memory-usage attributes of a %s instance", type(mat).__name__)
        raise TypeError(f"Unsupported sparse matrix type: {type(mat)}") from e
