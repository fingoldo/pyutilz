import pytest


def test_f01_sparse_memory_usage_returns_a_real_int():
    """`get_sparse_memory_usage` returns a builtin int, not numpy's np.intp, which is what the removed ignore was hiding."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    from pyutilz.core.matrix import get_sparse_memory_usage

    mat = scipy_sparse.csr_matrix([[1, 0], [0, 2]])
    result = get_sparse_memory_usage(mat)
    assert type(result) is int
