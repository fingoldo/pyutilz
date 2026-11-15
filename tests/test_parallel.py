"""Tests for parallel.py - Phase 2 refactoring

Tests cover:
- Temp directory cleanup (atexit handler for mem_map_array)
- Resource management for memory-mapped arrays
"""

import pytest
import tempfile
import os
import numpy as np


class TestMemMapArray:
    """Test mem_map_array function - temp directory leak fix"""

    def test_temp_directory_tracked_for_cleanup(self):
        """Test that temporary directories are tracked in _TEMP_DIRS list"""
        try:
            from pyutilz.parallel import mem_map_array, _TEMP_DIRS
        except ImportError:
            pytest.skip("mem_map_array or _TEMP_DIRS not available")

        initial_count = len(_TEMP_DIRS)

        # Create a memory-mapped array
        test_array = np.array([1, 2, 3, 4, 5])
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
        temp_file.close()

        try:
            mem_map_array(test_array, temp_file.name, mmap_mode="r+")

            # Should have added temp directory to tracking list
            assert len(_TEMP_DIRS) >= initial_count, "Temp directory should be tracked for cleanup (resource leak fix)"
        finally:
            # Cleanup
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def test_atexit_cleanup_handler_actually_runs_on_interpreter_exit(self, tmp_path):
        """A directory tracked in ``_TEMP_DIRS`` is really gone after the process exits.

        Behavioural replacement for an ``inspect.getsource()`` scan for the literal strings
        ``"atexit.register"`` / ``"cleanup"``: that passed on a module that merely mentioned
        them (or registered a handler that cleaned nothing), and failed on a pure rename. The
        registration can only be observed end-to-end -- atexit exposes no public introspection
        API -- so this drives a real child interpreter that imports the module, tracks a
        directory, and exits normally; the parent then checks the directory is gone.
        """
        import subprocess
        import sys
        import textwrap

        victim = tmp_path / "tracked_temp_dir"
        victim.mkdir()
        (victim / "payload.bin").write_bytes(b"x" * 16)
        survivor = tmp_path / "untracked_temp_dir"
        survivor.mkdir()

        child = textwrap.dedent(
            f"""
            import pyutilz.parallel as p
            p._TEMP_DIRS.append(r{str(victim)!r})
            # NOT appended: r{str(survivor)!r} -- the handler must only remove what it tracks.
            """
        )
        proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=180)

        assert proc.returncode == 0, f"child interpreter failed: {proc.stderr[-2000:]}"
        assert not victim.exists(), "the atexit-registered handler did not remove the tracked temp directory"
        assert survivor.exists(), "the handler removed a directory that was never tracked in _TEMP_DIRS"

    def test_cleanup_function_removes_directories(self):
        """Test that cleanup function properly removes temp directories"""
        try:
            from pyutilz.parallel import _cleanup_temp_dirs, _TEMP_DIRS
        except ImportError:
            pytest.skip("Cleanup function not available")

        # Create a test temp directory
        test_temp_dir = tempfile.mkdtemp()
        _TEMP_DIRS.append(test_temp_dir)

        # Verify directory exists
        assert os.path.exists(test_temp_dir)

        # Call cleanup
        _cleanup_temp_dirs()

        # Directory should be removed
        assert not os.path.exists(test_temp_dir), "Cleanup function should remove temp directories"

    def test_cleanup_handles_missing_directories(self):
        """Test that cleanup doesn't crash on already-deleted directories"""
        try:
            from pyutilz.parallel import _cleanup_temp_dirs, _TEMP_DIRS
        except ImportError:
            pytest.skip("Cleanup function not available")

        # Add non-existent directory to cleanup list
        fake_dir = "/tmp/nonexistent_dir_12345"
        _TEMP_DIRS.append(fake_dir)

        # Should not raise exception
        try:
            _cleanup_temp_dirs()
        except Exception as e:
            pytest.fail(f"Cleanup should handle missing directories gracefully: {e}")


class TestMemoryMappedArrayOperations:
    """Test memory-mapped array functionality"""

    def test_mem_map_creates_readable_array(self):
        """Test that memory-mapped arrays can be read"""
        try:
            from pyutilz.parallel import mem_map_array
        except ImportError:
            pytest.skip("mem_map_array not available")

        test_array = np.array([10, 20, 30, 40, 50])
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
        temp_file.close()

        try:
            result = mem_map_array(test_array, temp_file.name, mmap_mode="r+")

            # Should be a numpy array
            assert isinstance(result, np.ndarray)

            # Should have correct values
            np.testing.assert_array_equal(result, test_array)
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def test_mem_map_different_dtypes(self):
        """Test memory-mapped arrays with different dtypes"""
        try:
            from pyutilz.parallel import mem_map_array
        except ImportError:
            pytest.skip("mem_map_array not available")

        dtypes = [np.int32, np.float32, np.float64]

        for dtype in dtypes:
            test_array = np.array([1, 2, 3], dtype=dtype)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
            temp_file.close()

            try:
                result = mem_map_array(test_array, temp_file.name, mmap_mode="r+")
                assert result.dtype == dtype
                np.testing.assert_array_equal(result, test_array)
            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)


class TestGpuConfiguration:
    """Test GPU selection configuration"""

    def test_gpu_selection_follows_the_caller_supplied_device_id(self, monkeypatch):
        """``cuda.select_device`` receives the caller's ``device_id``, never a hardcoded index.

        Rewritten twice over. (1) It was an ``inspect.getsource()`` scan whose only assertion
        sat under ``if "cuda.select_device" in source:`` -- and inside that, an ``if/else``
        whose BOTH branches were ``pass``. (2) The outer guard is false today anyway: no
        ``select_device`` call remains anywhere in ``pyutilz.parallel``; the two live call sites
        are ``system/system/probing.py`` (``get_gpu_cuda_capabilities``) and
        ``system/gpu_dispatch.py``, both taking ``device_id`` as a parameter. So the whole test
        had degraded into a no-op guarding a check for code that had moved. It now drives the
        real function with a stubbed numba.cuda and asserts the index that arrives at the
        driver is exactly the one the caller asked for -- which is what "not hardcoded to 3"
        means behaviourally, and which a later regression to a literal index would fail.
        """
        numba = pytest.importorskip("numba")
        pytest.importorskip("numba.cuda.cudadrv.enums")

        from pyutilz.system.system.probing import get_gpu_cuda_capabilities

        selected = []

        class _FakeDevice:
            def __getattr__(self, _name):
                return 0

        class _FakeCuda:
            @staticmethod
            def select_device(device_id):
                selected.append(device_id)

            @staticmethod
            def get_current_device():
                return _FakeDevice()

        monkeypatch.setattr(numba, "cuda", _FakeCuda)

        for requested in (0, 2, 7):
            selected.clear()
            get_gpu_cuda_capabilities(device_id=requested)
            assert selected == [requested], f"device_id={requested} must reach cuda.select_device, got {selected}"

        # And the default is device 0 -- the only index guaranteed to exist on any CUDA host
        # (a hardcoded 3 crashes every box with fewer than four GPUs, the original bug).
        selected.clear()
        get_gpu_cuda_capabilities()
        assert selected == [0]


@pytest.mark.parametrize("array_size", [10, 100, 1000])
def test_mem_map_different_sizes(array_size):
    """Parametrized test for different array sizes"""
    try:
        from pyutilz.parallel import mem_map_array
    except ImportError:
        pytest.skip("mem_map_array not available")

    test_array = np.arange(array_size)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
    temp_file.close()

    try:
        result = mem_map_array(test_array, temp_file.name, mmap_mode="r+")
        assert len(result) == array_size
        np.testing.assert_array_equal(result, test_array)
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def test_parallel_module_imports_successfully():
    """Test that parallel module can be imported without errors"""
    try:
        import pyutilz.parallel
    except ImportError as e:
        # Some dependencies might be missing, but import should not crash
        pytest.skip(f"parallel module dependencies not available: {e}")
    assert hasattr(pyutilz.parallel, "mem_map_array")
    assert hasattr(pyutilz.parallel, "applyfunc_parallel")


# ---------------------------------------------------------------------------
# applyfunc_parallel — return type genuinely depends on return_dataframe.
# Was annotated "-> list" but with the default return_dataframe=True it
# actually returns a pd.DataFrame (via pd.concat). Fixed the annotation to
# Union[list, pd.DataFrame]; these tests pin the ACTUAL runtime return types
# for both branches so the annotation can never silently drift from reality again.
# ---------------------------------------------------------------------------

import pandas as pd


def _echo_df(chunk):
    return pd.DataFrame({"x": chunk})


def _echo_list(chunk):
    return list(chunk)


def test_applyfunc_parallel_returns_dataframe_by_default():
    from pyutilz.system.parallel import applyfunc_parallel

    result = applyfunc_parallel(iterable=[([1, 2],), ([3, 4],)], func=_echo_df, n_jobs=1, use_threads=True)
    assert isinstance(result, pd.DataFrame)
    assert list(result["x"]) == [1, 2, 3, 4]


def test_applyfunc_parallel_returns_list_when_return_dataframe_false():
    from pyutilz.system.parallel import applyfunc_parallel

    result = applyfunc_parallel(iterable=[([1, 2],), ([3, 4],)], func=_echo_list, n_jobs=1, return_dataframe=False, use_threads=True)
    # not a DataFrame (that's the whole point of the annotation fix); tqdmu wraps the raw
    # pool.starmap() result in a tqdm progress iterator rather than returning a bare list,
    # but its contents are exactly the per-chunk results, list-like end to end.
    assert not isinstance(result, pd.DataFrame)
    assert list(result) == [[1, 2], [3, 4]]


class TestChunking:
    """The three chunking helpers in parallel.py: exact chunk bounds (including the short last chunk), the rejection of a
    non-positive size with its message, empty input, and the cleanup warning when a temp directory cannot be checked."""

    def test_split_into_chunks_gives_exact_chunks_with_a_short_last_one(self):
        from pyutilz.system.parallel import split_list_into_chunks

        assert list(split_list_into_chunks(list(range(10)), 3)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_split_into_chunks_keeps_an_exact_multiple_without_an_empty_tail(self):
        from pyutilz.system.parallel import split_list_into_chunks

        assert list(split_list_into_chunks([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_a_chunk_size_of_one_gives_singletons(self):
        from pyutilz.system.parallel import split_list_into_chunks

        assert list(split_list_into_chunks(["a", "b"], 1)) == [["a"], ["b"]]

    def test_a_chunk_size_larger_than_the_list_gives_one_chunk(self):
        from pyutilz.system.parallel import split_list_into_chunks

        assert list(split_list_into_chunks([1, 2, 3], 10)) == [[1, 2, 3]]

    def test_an_empty_list_gives_no_chunks(self):
        from pyutilz.system.parallel import split_list_into_chunks, split_list_into_chunks_indices

        assert list(split_list_into_chunks([], 3)) == []
        assert list(split_list_into_chunks_indices([], 3)) == []

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_chunk_size_is_refused_with_its_value_in_the_message(self, bad):
        from pyutilz.system.parallel import split_list_into_chunks

        with pytest.raises(ValueError, match=f"split_list_into_chunks: chunk_size must be >= 1, got {bad}"):
            list(split_list_into_chunks([1, 2], bad))

    def test_the_index_bounds_are_exact_and_the_last_right_bound_is_clipped(self):
        from pyutilz.system.parallel import split_list_into_chunks_indices

        assert list(split_list_into_chunks_indices(list(range(10)), 3)) == [(0, 3), (3, 6), (6, 9), (9, 10)]
        assert list(split_list_into_chunks_indices(list(range(4)), 4)) == [(0, 4)]
        assert list(split_list_into_chunks_indices(list(range(5)), 2)) == [(0, 2), (2, 4), (4, 5)]

    @pytest.mark.parametrize("bad", [0, -2])
    def test_the_index_helper_refuses_a_non_positive_size_with_its_value(self, bad):
        from pyutilz.system.parallel import split_list_into_chunks_indices

        with pytest.raises(ValueError, match=f"split_list_into_chunks_indices: chunk_size must be >= 1, got {bad}"):
            list(split_list_into_chunks_indices([1, 2], bad))

    def test_the_nchunks_helper_refuses_zero_with_its_value(self):
        from pyutilz.system.parallel import split_list_into_nchunks_indices

        with pytest.raises(ValueError, match="split_list_into_nchunks_indices: nchunks must be >= 1, got 0"):
            list(split_list_into_nchunks_indices([1, 2], 0))

    def test_more_chunks_than_items_are_capped_at_one_item_each(self):
        from pyutilz.system.parallel import split_list_into_nchunks_indices

        assert list(split_list_into_nchunks_indices(list(range(3)), 10)) == [(0, 1), (1, 2), (2, 3)]

    def test_nchunks_splits_as_evenly_as_it_can(self):
        from pyutilz.system.parallel import split_list_into_nchunks_indices

        assert list(split_list_into_nchunks_indices(list(range(10)), 3)) == [(0, 3), (3, 6), (6, 10)]

    def test_a_failed_temp_directory_check_is_logged_with_its_path_and_error(self, monkeypatch):
        from pyutilz.system import parallel

        warned: list = []
        monkeypatch.setattr(parallel, "logger", type("L", (), {"warning": lambda self, fmt, *a, **k: warned.append((fmt, a)), "info": lambda self, *a, **k: None, "debug": lambda self, *a, **k: None, "error": lambda self, *a, **k: None})())
        monkeypatch.setattr(parallel, "_TEMP_DIRS", ["/tmp/some-dir"])
        boom = OSError("permission denied")

        def _exists(path):
            raise boom

        monkeypatch.setattr(parallel.os.path, "exists", _exists)
        parallel._cleanup_temp_dirs()
        assert warned == [("Failed to cleanup temp directory %s: %s", ("/tmp/some-dir", boom))]


class _Recorder:
    """A stand-in for a pool or joblib dispatcher: remembers its constructor arguments and what it was asked to run."""

    def __init__(self, sink, **kwargs):
        self.kwargs = kwargs
        sink.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


class TestSplitArrayDistributeAndCpu:
    """split_array's bounds and refusal, distribute_work's largest-first assignment, its deprecated alias and argument rules,
    and cpu_count_physical's fallbacks (parallel.py 126-197)."""

    def test_split_array_gives_exact_bounds_and_refuses_a_non_positive_step(self):
        import numpy as np

        from pyutilz.system.parallel import split_array

        assert split_array(np.zeros(10), 3) == [(0, 3), (3, 6), (6, 9), (9, 10)]
        assert split_array(np.zeros(0), 3) == []
        with pytest.raises(ValueError, match="split_array: step must be >= 1, got 0"):
            split_array(np.zeros(4), 0)

    def test_distribute_work_is_largest_first_and_keeps_each_workers_items_in_input_order(self):
        from pyutilz.system.parallel import distribute_work

        values, indices = distribute_work([5, 4, 3, 3, 2], n_jobs=2)
        assert values == [[5, 3], [4, 3, 2]]
        assert indices == [[0, 3], [1, 2, 4]]

    def test_the_deprecated_nworkers_alias_warns_and_sets_the_worker_count(self):
        from pyutilz.system.parallel import distribute_work

        with pytest.warns(DeprecationWarning, match="nworkers"):
            values, _indices = distribute_work([1, 2, 3], nworkers=2)
        assert len(values) == 2

    def test_n_jobs_wins_over_the_deprecated_alias_when_both_are_given(self):
        from pyutilz.system.parallel import distribute_work

        with pytest.warns(DeprecationWarning):
            values, _indices = distribute_work([1, 2], n_jobs=3, nworkers=5)
        assert len(values) == 3

    def test_distribute_work_needs_a_worker_count(self):
        from pyutilz.system.parallel import distribute_work

        with pytest.raises(TypeError, match="missing required argument: 'n_jobs'"):
            distribute_work([1, 2])

    def test_a_non_positive_worker_count_means_the_physical_core_count(self, monkeypatch):
        from pyutilz.system import parallel

        monkeypatch.setattr(parallel, "cpu_count_physical", lambda: 2)
        values, _indices = parallel.distribute_work([1, 2, 3], n_jobs=0)
        assert len(values) == 2

    def test_cpu_count_physical_asks_psutil_for_physical_cores(self, monkeypatch):
        from pyutilz.system import parallel

        asked: list = []
        monkeypatch.setattr(parallel.psutil, "cpu_count", lambda logical=True: asked.append(logical) or 6)
        assert parallel.cpu_count_physical() == 6
        assert asked == [False]

    def test_cpu_count_physical_falls_back_when_psutil_cannot_tell(self, monkeypatch):
        from pyutilz.system import parallel

        monkeypatch.setattr(parallel.psutil, "cpu_count", lambda logical=True: None)
        assert parallel.cpu_count_physical() == 1
        assert parallel.cpu_count_physical(fallback=4) == 4
        monkeypatch.setattr(parallel.psutil, "cpu_count", lambda logical=True: 0)
        assert parallel.cpu_count_physical(fallback=3) == 3


class TestParallelDispatch:
    """parallel_run hands joblib the worker count, backend and limits it was given; applyfunc_parallel sizes and builds its
    pool, honours the deprecated alias, concatenates results, and handles empty input (parallel.py 198-273)."""

    def _joblib(self, monkeypatch):
        from pyutilz.system import parallel

        made: list = []

        class _FakeParallel(_Recorder):
            def __init__(self, **kwargs):
                super().__init__(made, **kwargs)

            def __call__(self, jobs):
                return [f"ran:{job}" for job in jobs]

        monkeypatch.setattr(parallel, "Parallel", _FakeParallel)
        return parallel, made

    def test_parallel_run_passes_its_worker_count_backend_and_limits_to_joblib(self, monkeypatch):
        parallel, made = self._joblib(monkeypatch)
        assert parallel.parallel_run(["a", "b"], n_jobs=2) == ["ran:a", "ran:b"]
        assert made[0].kwargs == {"n_jobs": 2, "backend": None, "max_nbytes": 50_000, "verbose": 0}

    def test_a_non_positive_worker_count_becomes_the_physical_cores_by_default(self, monkeypatch):
        parallel, made = self._joblib(monkeypatch)
        monkeypatch.setattr(parallel, "cpu_count_physical", lambda: 3)
        parallel.parallel_run(["a"], n_jobs=0)
        assert made[0].kwargs["n_jobs"] == 3

    def test_a_non_positive_worker_count_is_kept_when_real_cores_are_not_preferred(self, monkeypatch):
        parallel, made = self._joblib(monkeypatch)
        monkeypatch.setattr(parallel, "cpu_count_physical", lambda: 3)
        parallel.parallel_run(["a"], n_jobs=0, prefer_real_cores=False)
        assert made[0].kwargs["n_jobs"] == 0

    def test_a_dask_backend_is_entered_as_a_context_and_joblib_gets_no_backend(self, monkeypatch):
        parallel, made = self._joblib(monkeypatch)
        entered: list = []

        class _Context:
            def __init__(self, name):
                entered.append(name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(parallel, "parallel_backend", _Context)
        parallel.parallel_run(["a"], n_jobs=2, backend="dask", batch_size=5)
        assert entered == ["dask"]
        assert made[0].kwargs["backend"] is None
        assert made[0].kwargs["batch_size"] == 5

    def test_applyfunc_parallel_returns_an_empty_frame_or_list_for_no_work(self):
        from pyutilz.system.parallel import applyfunc_parallel

        assert applyfunc_parallel([], lambda x: x).empty
        assert applyfunc_parallel([], lambda x: x, return_dataframe=False) == []

    def test_applyfunc_parallel_concatenates_frames_with_a_fresh_index(self):
        import pandas as pd

        from pyutilz.system.parallel import applyfunc_parallel

        out = applyfunc_parallel([(1,), (2,)], lambda x: pd.DataFrame({"v": [x]}), n_jobs=2, use_threads=True)
        assert list(out["v"]) == [1, 2]
        assert list(out.index) == [0, 1]

    def test_applyfunc_parallel_returns_the_plain_list_when_asked(self):
        import pandas as pd

        from pyutilz.system.parallel import applyfunc_parallel

        out = applyfunc_parallel([(1,), (2,)], lambda x: pd.DataFrame({"v": [x]}), n_jobs=2, return_dataframe=False, use_threads=True)
        assert isinstance(out, list) and len(out) == 2

    def test_the_deprecated_n_cores_alias_warns_and_sets_the_pool_size(self, monkeypatch):
        from pyutilz.system import parallel

        made: list = []
        monkeypatch.setattr(parallel, "ThreadPool", lambda **kwargs: _Recorder(made, **kwargs))
        with pytest.warns(DeprecationWarning, match="n_cores"):
            parallel.applyfunc_parallel([(1,), (2,)], lambda x: x, n_cores=3, return_dataframe=False, use_threads=True)
        assert made[0].kwargs["processes"] == 3

    def test_the_default_pool_size_is_the_physical_count_capped_by_the_work(self, monkeypatch):
        from pyutilz.system import parallel

        made: list = []
        monkeypatch.setattr(parallel, "ThreadPool", lambda **kwargs: _Recorder(made, **kwargs))
        monkeypatch.setattr(parallel.psutil, "cpu_count", lambda logical=False: 8)
        parallel.applyfunc_parallel([(1,), (2,)], lambda x: x, return_dataframe=False, use_threads=True)
        assert made[0].kwargs["processes"] == 2

    def test_a_missing_core_count_means_one_worker(self, monkeypatch):
        from pyutilz.system import parallel

        made: list = []
        monkeypatch.setattr(parallel, "ThreadPool", lambda **kwargs: _Recorder(made, **kwargs))
        monkeypatch.setattr(parallel.psutil, "cpu_count", lambda logical=False: None)
        parallel.applyfunc_parallel([(1,), (2,)], lambda x: x, return_dataframe=False, use_threads=True)
        assert made[0].kwargs["processes"] == 1

    def test_the_process_pool_gets_its_initializer_and_arguments(self, monkeypatch):
        from pyutilz.system import parallel

        made: list = []
        monkeypatch.setattr(parallel, "Pool", lambda **kwargs: _Recorder(made, **kwargs))

        def _init(*args):
            return None

        out = parallel.applyfunc_parallel([(1,)], lambda x: x, n_jobs=1, return_dataframe=False, initializer=_init, initargs=("cfg",))
        assert out == [1]
        assert made[0].kwargs["initializer"] is _init
        assert made[0].kwargs["initargs"] == ("cfg",)
        assert made[0].kwargs["processes"] == 1


class TestTensorFlowGpuSelection:
    """set_tf_gpu restricts TensorFlow to one listed GPU and reports a bad index or a late failure instead of raising
    (parallel.py 275-291). TensorFlow is replaced by a stub module."""

    def _stub_tf(self, monkeypatch, gpus, fail=False):
        import sys
        import types

        set_calls: list = []

        def _set(device, kind):
            if fail:
                raise RuntimeError("GPUs already initialized")
            set_calls.append((device, kind))

        experimental = types.SimpleNamespace(
            list_physical_devices=lambda kind: list(gpus),
            set_visible_devices=_set,
            list_logical_devices=lambda kind: [f"logical-{g}" for g in gpus],
        )
        tf = types.SimpleNamespace(config=types.SimpleNamespace(experimental=experimental))
        monkeypatch.setitem(sys.modules, "tensorflow", tf)
        return set_calls

    def _recorder(self, monkeypatch):
        from pyutilz.system import parallel

        errors: list = []
        infos: list = []
        monkeypatch.setattr(
            parallel,
            "logger",
            type(
                "L",
                (),
                {
                    "error": lambda self, fmt, *a, **k: errors.append((fmt, a)),
                    "info": lambda self, fmt, *a, **k: infos.append((fmt, a)),
                    "warning": lambda self, *a, **k: None,
                    "debug": lambda self, *a, **k: None,
                },
            )(),
        )
        return parallel, errors, infos

    def test_with_no_gpu_nothing_is_set_and_nothing_is_reported(self, monkeypatch):
        parallel, errors, infos = self._recorder(monkeypatch)
        calls = self._stub_tf(monkeypatch, gpus=[])
        parallel.set_tf_gpu(0)
        assert calls == [] and errors == []

    @pytest.mark.parametrize("bad", [2, -1])
    def test_an_index_outside_the_gpus_is_reported_and_nothing_is_set(self, monkeypatch, bad):
        parallel, errors, infos = self._recorder(monkeypatch)
        calls = self._stub_tf(monkeypatch, gpus=["g0", "g1"])
        parallel.set_tf_gpu(bad)
        assert calls == []
        assert errors == [("Requested GPU index %s is out of range for %s available GPU(s)", (bad, 2))]

    def test_a_valid_index_makes_only_that_gpu_visible_and_reports_the_counts(self, monkeypatch):
        parallel, errors, infos = self._recorder(monkeypatch)
        calls = self._stub_tf(monkeypatch, gpus=["g0", "g1"])
        parallel.set_tf_gpu(1)
        assert calls == [("g1", "GPU")]
        assert errors == []
        assert infos == [("%s Physical GPUs, %s Logical GPU", (2, 2))]

    def test_a_failure_to_set_visibility_is_logged_not_raised(self, monkeypatch):
        parallel, errors, infos = self._recorder(monkeypatch)
        self._stub_tf(monkeypatch, gpus=["g0"], fail=True)
        parallel.set_tf_gpu(0)
        assert len(errors) == 1
        assert isinstance(errors[0][0], RuntimeError)


class TestMemMapArrayBehaviour:
    """mem_map_array maps the array read-only by default, keeps every call in one shared directory, and overwrites a
    name it has seen before (parallel.py 293-318)."""

    def test_the_mapping_is_a_read_only_memmap_with_the_same_values(self):
        import numpy as np

        from pyutilz.system import parallel

        data = np.arange(6, dtype=np.float64)
        mapped = parallel.mem_map_array(data, "gap_read_only")
        assert isinstance(mapped, np.memmap)
        assert np.array_equal(mapped, data)
        assert mapped.flags.writeable is False

    def test_every_call_shares_one_directory_and_a_reused_name_is_overwritten(self):
        import os

        import numpy as np

        from pyutilz.system import parallel

        first = parallel.mem_map_array(np.zeros(3), "gap_shared_a")
        folder = parallel._MEM_MAP_DIR
        second = parallel.mem_map_array(np.ones(3), "gap_shared_b")
        assert parallel._MEM_MAP_DIR == folder
        assert os.path.dirname(first.filename) == folder == os.path.dirname(second.filename)
        del first
        import gc

        gc.collect()
        again = parallel.mem_map_array(np.full(3, 7.0), "gap_shared_a")
        assert np.array_equal(again, np.full(3, 7.0))
