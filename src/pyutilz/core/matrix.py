# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed
ensure_installed("numba scipy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

class CsrIndPtrConstructor:
    """Incremental builder of CSR matrices from (indptr, indices, data) triples.

    Call add_element() for each non-zero of the current row, then add_row() to
    close the row; finally build_matrix() to materialise a scipy csr_matrix.
    """
    def __init__(self) -> None:
        self.indptr = [0]
        self.indices = []
        self.data = []
    def add_element(self, element, index: int) -> None:
        self.indices.append(index)
        self.data.append(element)

    def add_row(self) -> None:
        self.indptr.append(len(self.indices))

    def build_matrix(self, dtype: np.dtype, clear_source: bool = True) -> csr_matrix:
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
        self.rows = []
        self.cols = []
        self.data = []
    def add_element(self, element, row: int, col: int) -> None:
        self.rows.append(row)
        self.cols.append(col)
        self.data.append(element)

    def build_matrix(self, dtype: np.dtype, clear_source: bool = True) -> csr_matrix:
        matrix = csr_matrix((self.data, (self.rows, self.cols)), dtype=dtype)
        if clear_source:
            del self.data, self.rows, self.cols
        return matrix

def get_sparse_memory_usage(mat:object)->int:
    """
    Return mem usage of a csr or csc matrix
    """
    try:
        if isinstance(mat, csr_matrix):
            return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
        elif isinstance(mat, coo_matrix):
            return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes
        else:
            return -1
    except AttributeError:
        return -1
