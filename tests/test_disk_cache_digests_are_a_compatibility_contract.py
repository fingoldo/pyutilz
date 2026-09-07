"""The digests `hash_array_summary` returns are a wire format, and nothing pinned them.

`_HASH_VERSION` exists because these hex strings are a compatibility contract: they are cache KEYS
written to disk by one run and looked up by the next. Change how any byte reaches the hash and every
existing entry becomes unreachable -- not corrupt, not an error, just a permanent miss that reads as
a cold cache. On the frames this library caches, that is hours of recomputation, and the only
symptom is that the cache "stopped helping".

The existing suite asserts the digest is DETERMINISTIC and that it CHANGES when the data changes.
Both are necessary and neither catches the failure above: a rewrite that alters every digest
consistently passes both. So the literals below are the missing half. They were captured at
`_HASH_VERSION = 3`, and if a change makes them fail the correct response is to decide deliberately:
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


#: Captured at ``_HASH_VERSION = 3``. See the module docstring before changing any of these.
_EXPECTED_DIGESTS = {
    "float64_2d": "09fffdfe495f20978206b48eb04476e4",  # pragma: allowlist secret
    "float64_F": "5e0b1024447653d3786ffad091747380",  # pragma: allowlist secret
    "float64_strided": "96d321ef3728b600e7e505770c9dc8a4",  # pragma: allowlist secret
    "float32_1d": "b7df7105755dcc5c6b7157bbf431549c",  # pragma: allowlist secret
    "int64_2d": "0bcca7fd844e677c81a6bc7d5323ec08",  # pragma: allowlist secret
    "int8": "24a92683ae9fbb800b62477c6a09876d",  # pragma: allowlist secret
    "bool": "8943eb5116c266aecfa6e1089675fe0c",  # pragma: allowlist secret
    "datetime64": "84574906d71de98f9019c1f35bd4c210",  # pragma: allowlist secret
    "datetime64_2d": "2197b9527ad9b35cfa76d45c5ccf249e",  # pragma: allowlist secret
    "timedelta64": "b4d54c409ce9b8ea48a2c33572de1fc2",  # pragma: allowlist secret
    "unicode": "09f9c8c18204c9cdd7f34a6c12646821",  # pragma: allowlist secret
    "bytes": "0c7ad3899f542cd3d4e9afb07fecdb5c",  # pragma: allowlist secret
    "structured": "41bd771690ee2076e262c1fb1ccb8749",  # pragma: allowlist secret
    "object": "e60e752c8ce28df08992e1f65bdbbd0a",  # pragma: allowlist secret
    "scalar_0d": "f4dda7a9265c8558c2d9e57a848a314b",  # pragma: allowlist secret
    "scalar_0d_dt": "8dedd768d232e986d4ac2cfb5f23d0fe",  # pragma: allowlist secret
    "empty": "ffa2c86b04ce89ce5697f0b76713aa10",  # pragma: allowlist secret
    "empty_2d": "662572b7e3695a80f3ffb8da6db66035",  # pragma: allowlist secret
    "large_1d": "f9d0eda4b5644da98e69c58879e880e4",  # pragma: allowlist secret
}


class TestTheDigestsAreTheOnesAlreadyOnDisk:
    """Kills any change that alters a cache key without bumping `_HASH_VERSION`."""

    def test_the_version_the_digests_were_captured_at_is_still_current(self):
        """The literals below only mean anything at the version they were taken at. A bump is the
        CORRECT way to change them, and it must invalidate this pin rather than pass silently."""
        assert _HASH_VERSION == 3, "digests below were captured at version 3 -- re-capture them with the bump"

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
