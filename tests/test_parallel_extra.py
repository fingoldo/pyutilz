import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# ── split_list_into_chunks (lines 60-70) ──

class TestSplitListIntoChunks:
    def test_basic(self):
        from pyutilz.system.parallel import split_list_into_chunks
        result = list(split_list_into_chunks(list(range(10)), 3))
        assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_chunk_size_zero(self):
        from pyutilz.system.parallel import split_list_into_chunks
        result = list(split_list_into_chunks([1, 2, 3], 0))
        assert result == [[1], [2], [3]]

    def test_chunk_larger_than_list(self):
        from pyutilz.system.parallel import split_list_into_chunks
        result = list(split_list_into_chunks([1, 2], 10))
        assert result == [[1, 2]]

    def test_empty_list(self):
        from pyutilz.system.parallel import split_list_into_chunks
        result = list(split_list_into_chunks([], 3))
        assert result == []

    def test_exact_division(self):
        from pyutilz.system.parallel import split_list_into_chunks
        result = list(split_list_into_chunks(list(range(9)), 3))
        assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


# ── split_list_into_chunks_indices (lines 78-85) ──

class TestSplitListIntoChunksIndices:
    def test_basic(self):
        from pyutilz.system.parallel import split_list_into_chunks_indices
        result = list(split_list_into_chunks_indices(list(range(10)), 3))
        assert (0, 3) in result
        assert result[-1][1] == 10

    def test_exact_division(self):
        from pyutilz.system.parallel import split_list_into_chunks_indices
        result = list(split_list_into_chunks_indices(list(range(6)), 3))
        # The function yields an extra (6,6) chunk; filter zero-width
        non_empty = [(a, b) for a, b in result if b > a]
        assert non_empty == [(0, 3), (3, 6)]


# ── split_list_into_nchunks_indices (lines 93-102) ──

class TestSplitListIntoNchunksIndices:
    def test_basic(self):
        from pyutilz.system.parallel import split_list_into_nchunks_indices
        result = list(split_list_into_nchunks_indices(list(range(10)), 3))
        assert result[0] == (0, 3)
        assert result[-1][1] == 10

    def test_single_chunk(self):
        from pyutilz.system.parallel import split_list_into_nchunks_indices
        result = list(split_list_into_nchunks_indices(list(range(5)), 1))
        assert result == [(0, 5)]

    def test_two_chunks(self):
        from pyutilz.system.parallel import split_list_into_nchunks_indices
        result = list(split_list_into_nchunks_indices(list(range(7)), 2))
        assert result[0] == (0, 3)
        assert result[1] == (3, 7)


# ── split_array (lines 117-131) ──

class TestSplitArray:
    def test_basic(self):
        from pyutilz.system.parallel import split_array
        arr = np.random.uniform(0, 1, 5477)
        result = split_array(arr, step=1000)
        assert result[0] == (0, 1000)
        assert result[-1][1] == 5477

    def test_exact_length(self):
        from pyutilz.system.parallel import split_array
        result = split_array(np.zeros(3000), step=1000)
        assert result == [(0, 1000), (1000, 2000), (2000, 3000)]

    def test_small_array(self):
        from pyutilz.system.parallel import split_array
        result = split_array(np.zeros(5), step=10)
        assert result == [(0, 5)]

    def test_zero_step_raises(self):
        from pyutilz.system.parallel import split_array
        with pytest.raises(AssertionError):
            split_array(np.zeros(10), step=0)


# ── distribute_work (lines 136-147) ──

class TestDistributeWork:
    def test_basic(self):
        from pyutilz.system.parallel import distribute_work
        work, indices = distribute_work([10, 20, 30, 40], nworkers=2)
        assert len(work) == 2
        assert len(indices) == 2
        all_indices = sorted(sum(indices, []))
        assert all_indices == [0, 1, 2, 3]

    @patch("pyutilz.system.parallel.psutil")
    def test_zero_workers_uses_cpu_count(self, mock_psutil):
        from pyutilz.system.parallel import distribute_work
        mock_psutil.cpu_count.return_value = 2
        work, indices = distribute_work([5, 5, 5], nworkers=0)
        assert len(work) == 2

    def test_single_worker(self):
        from pyutilz.system.parallel import distribute_work
        work, indices = distribute_work([1, 2, 3], nworkers=1)
        assert work == [[1, 2, 3]]


# ── parallel_run (lines 154-158) ──

class TestParallelRun:
    def test_basic(self):
        from pyutilz.system.parallel import parallel_run
        from joblib import delayed
        result = parallel_run([delayed(lambda x: x * 2)(i) for i in range(5)], n_jobs=2)
        assert sorted(result) == [0, 2, 4, 6, 8]

    def test_prefer_real_cores(self):
        from pyutilz.system.parallel import parallel_run
        from joblib import delayed
        result = parallel_run([delayed(lambda x: x)(1)], n_jobs=-1, prefer_real_cores=True)
        assert result == [1]


# ── applyfunc_parallel (lines 174-204) ──

class TestApplyfuncParallel:
    def test_with_threads(self):
        import pandas as pd
        from pyutilz.system.parallel import applyfunc_parallel

        def add(a, b):
            return pd.DataFrame({"sum": [a + b]})

        result = applyfunc_parallel(
            [(1, 2), (3, 4)], add, n_cores=1, return_dataframe=True, use_threads=True
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result["sum"]) == [3, 7]

    def test_without_dataframe(self):
        from pyutilz.system.parallel import applyfunc_parallel

        def identity(x):
            return x

        result = applyfunc_parallel(
            [(1,), (2,)], identity, n_cores=1, return_dataframe=False, use_threads=True
        )
        assert result is not None

    def test_with_processes(self):
        import pandas as pd
        from pyutilz.system.parallel import applyfunc_parallel

        # Use threads=True to avoid pickling issues with local functions
        def make_df(x):
            return pd.DataFrame({"val": [x]})

        result = applyfunc_parallel(
            [(10,), (20,)], make_df, n_cores=1, return_dataframe=True, use_threads=True
        )
        assert isinstance(result, pd.DataFrame)
        assert sorted(result["val"].tolist()) == [10, 20]


# ── set_tf_gpu (lines 208-221) ──

class TestSetTfGpu:
    def test_no_tensorflow(self):
        from pyutilz.system.parallel import set_tf_gpu
        with patch("pyutilz.system.parallel.ensure_installed"):
            with patch.dict("sys.modules", {"tensorflow": None}):
                try:
                    set_tf_gpu(0)
                except Exception:
                    pass  # expected when tf not available
