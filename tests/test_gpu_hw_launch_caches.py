"""The launch-geometry caches answer only for the object they were computed for, and stay bounded."""

from pyutilz.performance import gpu_hw_launch as gpu


class _Kernel:
    """A stand-in compiled kernel with the two attributes the caches read."""

    def __init__(self, regs, static=0):
        self.num_regs = regs
        self.shared_size_bytes = static


def _fresh_caches(monkeypatch):
    """Empty both caches for the test, so no answer from another test is replayed."""
    monkeypatch.setattr(gpu, "_KERNEL_RESOURCES", {})
    monkeypatch.setattr(gpu, "_BLOCK_SIZE_CACHE", {})


def test_kernel_resources_are_read_per_object(monkeypatch):
    _fresh_caches(monkeypatch)
    first = _Kernel(40, 1024)
    assert gpu._kernel_resources(first) == (40, 1024)
    assert gpu._kernel_resources(_Kernel(20)) == (20, 0)


def test_a_reused_address_does_not_answer_for_a_new_kernel(monkeypatch):
    """An entry under the same id() but for another object is not used: the new kernel reads its own resources."""
    _fresh_caches(monkeypatch)
    old = _Kernel(40, 1024)
    gpu._KERNEL_RESOURCES[id(_Kernel(0))] = (old, 40, 1024)  # a stale entry under a key the new object would hash to
    new = _Kernel(24)
    gpu._KERNEL_RESOURCES[id(new)] = (old, 40, 1024)
    assert gpu._kernel_resources(new) == (24, 0)


def test_kernel_resource_cache_is_bounded(monkeypatch):
    _fresh_caches(monkeypatch)
    monkeypatch.setattr(gpu, "_KERNEL_RESOURCES_MAX_SIZE", 2)
    kernels = [_Kernel(10 + i) for i in range(5)]
    for k in kernels:
        gpu._kernel_resources(k)
    assert len(gpu._KERNEL_RESOURCES) <= 2
