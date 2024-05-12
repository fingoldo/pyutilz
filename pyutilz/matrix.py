# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .python import ensure_installed
ensure_installed("numba scipy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from numba import njit
from numba.experimental import jitclass
from scipy.sparse import csr_matrix,coo_matrix,save_npz,load_npz

class CsrIndPtrConstructor:
    """
        Class for consecutive building of Csr matrices
    """    
    def __init__(self):
        self.indptr = [0]
        self.indices = []
        self.data = []    
    def add_element(self, element, index: int):
        self.indices.append(index)
        self.data.append(element)

    def add_row(self):
        self.indptr.append(len(self.indices))

    def build_matrix(self, dtype, clear_source:bool=True):
        try:
            matrix = csr_matrix((self.data, self.indices, self.indptr), dtype=dtype)
            if clear_source: del self.data, self.indices, self.indptr

            return matrix
        except Exception as e:
            raise (e)

class CsrRowColConstructor:
    """
        Class for consecutive building of Csr matrices
    """    
    def __init__(self):
        self.rows = []
        self.cols = []
        self.data = []    
    def add_element(self, element, row: int, col: int):
        self.rows.append(row)
        self.cols.append(col)
        self.data.append(element)

    def build_matrix(self, dtype, clear_source:bool=True):
        try:
            matrix = csr_matrix((self.data, (self.rows, self.cols)), dtype=dtype)
            if clear_source: 
                del self.data, self.rows, self.cols

            return matrix
        except Exception as e:
            raise (e)
            
def get_sparse_memory_usage(mat:object)->int:
    """
        Return mem usage of a csr or csc matrix
    """
    try:
        if type(mat) == csr_matrix:
            return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
        elif type(mat) == coo_matrix:
            return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes
        else:
            return -1
    except AttributeError:
        return -1