"""The digests `hash_array_summary` returns are a wire format, and nothing pinned them.

`_HASH_VERSION` exists because these hex strings are a compatibility contract: they are cache KEYS
written to disk by one run and looked up by the next. Change how any byte reaches the hash and every
existing entry becomes unreachable -- not corrupt, not an error, just a permanent miss that reads as
a cold cache. On the frames this library caches, that is hours of recomputation, and the only
symptom is that the cache "stopped helping".

The existing suite asserts the digest is DETERMINISTIC and that it CHANGES when the data changes.
Both are necessary and neither catches the failure above: a rewrite that alters every digest
consistently passes both. So the literals below are the missing half. They were captured at
`_HASH_VERSION = 4`, and if a change makes them fail the correct response is to decide deliberately:
bump `_HASH_VERSION` (which invalidates old entries loudly, by key prefix) and re-capture, or revert
the change. Editing the literals to match new output without bumping the version is the one response
that reproduces the bug this file exists to prevent.

THE DTYPE COVERAGE IS NOT DECORATION. `hash_array_summary` routes object, fixed-width string,
datetime64/timedelta64, structured, 0-d and empty arrays down five different branches, each with its
own reason. A digest test over `float64` alone would pin one branch and leave the four with actual
subtleties unguarded.

Written to py-ci-shared/WRITING_TESTS.md habits 1, 2 and 5.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pyutilz.core.disk_cache import _HASH_VERSION, _buffer, hash_array_summary


def _case(name: str) -> np.ndarray:
    """The array for *name*, built from a pinned seed so the digests below are reproducible."""
    rng = np.random.default_rng(0)
    builders = {
        "float64_2d": lambda: rng.random((50, 4)),
        "float64_F": lambda: np.asfortranarray(rng.random((50, 4))),
        "float64_strided": lambda: rng.random((50, 8))[:, ::2],
        "float32_1d": lambda: rng.random(50).astype(np.float32),
        "int64_2d": lambda: rng.integers(0, 100, (50, 3)),
        "int8": lambda: rng.integers(0, 10, (20, 3)).astype(np.int8),
        "bool": lambda: rng.random(40) > 0.5,
        "datetime64": lambda: np.array([1, 2, 3, 4, 5], dtype="datetime64[ns]"),
        "datetime64_2d": lambda: np.arange(20, dtype="int64").reshape(5, 4).view("datetime64[ns]"),
        "timedelta64": lambda: np.array([1, 2, 3], dtype="timedelta64[ns]"),
        "unicode": lambda: np.array(["alpha", "beta", "gamma"], dtype="U10"),
        "bytes": lambda: np.array([b"alpha", b"beta"], dtype="S10"),
        "structured": lambda: np.zeros(4, dtype=[("a", "i4"), ("b", "f8")]),
        "object": lambda: np.array([{"k": 1}, "s", 3], dtype=object),
        "scalar_0d": lambda: np.array(3.5),
        "scalar_0d_dt": lambda: np.array(np.datetime64("2026-01-01")),
        "empty": lambda: np.array([], dtype=np.float64),
        "empty_2d": lambda: np.zeros((0, 3)),
        "large_1d": lambda: rng.random(5000),
    }
    # The generator is consumed in insertion order when the digests are captured, so each case
    # rebuilds from a fresh seed and draws only its own arrays.
    ordered = list(builders)
    for earlier in ordered[: ordered.index(name)]:
        builders[earlier]()
    return builders[name]()


#: Captured at ``_HASH_VERSION = 4``. See the module docstring before changing any of these.
_EXPECTED_DIGESTS = {
    "float64_2d": "9c58e6dec6c35aad4d02c980c0035512",  # pragma: allowlist secret
    "float64_F": "fe0ad97c0172e43cc6452b1e6f7716a6",  # pragma: allowlist secret
    "float64_strided": "acb4f0ad3e29facc58c16071eb87ebf6",  # pragma: allowlist secret
    "float32_1d": "a860afd12baae914bfdb190db95ac7b7",  # pragma: allowlist secret
    "int64_2d": "8a34237262ecb435df73fa4751105ae0",  # pragma: allowlist secret
    "int8": "06828830e902f204a00784bf3f25f0d1",  # pragma: allowlist secret
    "bool": "efcaa255abdb66c15ebdb51b8d185765",  # pragma: allowlist secret
    "datetime64": "2710279116b52801a4fc986925972e8e",  # pragma: allowlist secret
    "datetime64_2d": "d6665c2adc43b87291cc705d1f8102c6",  # pragma: allowlist secret
    "timedelta64": "daa4c8efebb8e7a2a67779eb9094fae6",  # pragma: allowlist secret
    "unicode": "77468a6ac746ab30d1dbb52c146fa614",  # pragma: allowlist secret
    "bytes": "48d5c0c3def78c2ee7d640770f17b12b",  # pragma: allowlist secret
    "structured": "3820e737dd85baa3af8c0ca274968879",  # pragma: allowlist secret
    "object": "37d7f152d662487e6d89b9db9b27b31c",  # pragma: allowlist secret
    "scalar_0d": "9e79357aee80ab97d4efd8194ac23949",  # pragma: allowlist secret
    "scalar_0d_dt": "5e9c201747bb2722d9a3e75eb726414e",  # pragma: allowlist secret
    "empty": "ead9854ac24024a71d823494fff217b2",  # pragma: allowlist secret
    "empty_2d": "55c2cbf88b75389a3548866dac4c2dd9",  # pragma: allowlist secret
    "large_1d": "c6fb22459d2988bb39cc7c4e94a6bd63",  # pragma: allowlist secret
}


class TestTheDigestsAreTheOnesAlreadyOnDisk:
    """Kills any change that alters a cache key without bumping `_HASH_VERSION`."""

    def test_the_version_the_digests_were_captured_at_is_still_current(self):
        """The literals below only mean anything at the version they were taken at. A bump is the
        CORRECT way to change them, and it must invalidate this pin rather than pass silently."""
        assert _HASH_VERSION == 4, "digests below were captured at version 4 -- re-capture them with the bump"

    @pytest.mark.parametrize("name", sorted(_EXPECTED_DIGESTS))
    def test_the_digest_is_unchanged(self, name):
        assert hash_array_summary(_case(name)) == _EXPECTED_DIGESTS[name], (
            f"the cache key for {name} changed. Every entry written by an earlier run is now "
            "unreachable -- a permanent miss that reads as a cold cache, not as an error. "
            "Bump _HASH_VERSION and re-capture, or revert the change."
        )

    def test_every_branch_of_the_router_is_covered(self):
        """The five dtype branches carry the subtleties; a digest suite over float64 alone would
        pin the easy one and leave the rest unguarded."""
        kinds = {np.asarray(_case(name)).dtype.kind for name in _EXPECTED_DIGESTS}
        for kind in ("f", "i", "b", "M", "m", "U", "S", "V", "O"):
            assert kind in kinds, f"no pinned case exercises dtype kind {kind!r}"


class TestTheBufferHelperFeedsExactlyTheBytesTobytesWould:
    """`_buffer` replaced `.tobytes()` at every site; the digest depends on it byte for byte."""

    @pytest.mark.parametrize("name", sorted(_EXPECTED_DIGESTS))
    def test_it_matches_tobytes_for_every_dtype(self, name):
        arr = np.asarray(_case(name))
        if arr.dtype.kind == "O":
            pytest.skip("object arrays are hashed by value, never through the buffer")

        assert bytes(_buffer(arr)) == np.ascontiguousarray(arr).tobytes()

    def test_it_does_not_copy_a_contiguous_array(self):
        """The whole point. `.tobytes()` allocates a second full copy purely to feed the hash, and
        this is the cache-KEY computation -- the copy is paid on every lookup, including the hits."""
        arr = np.arange(1000, dtype=np.float64)

        assert np.shares_memory(np.frombuffer(_buffer(arr), dtype=np.uint8), arr)

    def test_it_works_on_the_dtypes_that_have_no_buffer_format(self):
        """The reason for the `uint8` view. `.data` on a datetime64 array raises `cannot include
        dtype 'M' in a buffer`, so the obvious copy-free form breaks on exactly the two dtypes this
        module goes out of its way to reduce correctly -- and a future simplification back to
        `.data` would fail here rather than in a caller's cache."""
        for arr in (np.array([1, 2, 3], dtype="datetime64[ns]"), np.array([1, 2, 3], dtype="timedelta64[ns]")):
            with pytest.raises(ValueError, match="buffer"):
                memoryview(np.ascontiguousarray(arr).data)

            assert bytes(_buffer(arr)) == arr.tobytes()

    def test_a_strided_view_is_serialised_in_c_order(self):
        """A column view of a bigger frame is not contiguous. Hashing its raw strided memory would
        make the key depend on the frame it was sliced from rather than on its own values."""
        frame = np.arange(24, dtype=np.float64).reshape(4, 6)
        view = frame[:, ::2]

        assert bytes(_buffer(view)) == np.ascontiguousarray(view).tobytes()
        assert hash_array_summary(view) == hash_array_summary(view.copy())

    def test_the_hash_accepts_the_buffer_directly(self):
        """`hashlib` takes any C-contiguous buffer, which is what makes the copy avoidable at all."""
        arr = np.arange(100, dtype=np.float64)

        assert hashlib.blake2b(_buffer(arr), digest_size=16).hexdigest() == hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()
