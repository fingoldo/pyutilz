"""Content-addressable disk cache for repeated heavy computations.

Fits a specific niche: a deterministic transform of (large numpy/pandas inputs, params) whose cost
dominates a single call, invoked repeatedly across hyperparam sweeps / ablations / incremental data
updates where the inputs recur exactly or near-exactly. Caching the result amortises the cost
across re-calls with zero correctness loss. NOT a replacement for ``functools.lru_cache``
(in-process, hash-keyed by identity-of-args) or ``joblib.Memory`` (persists pickled function
arguments, slow for large arrays) -- this is content-addressable, content-hashed, multi-process
safe, and built for numpy/pandas-shaped inputs specifically.

Ported from a downstream ML project's own shared cache module (used across two independent
consumers there). Design:

* Content-addressable, NOT path-keyed. Two processes writing the same key end up with
  bit-identical bytes (last-writer-wins via atomic rename), so no locking is needed across
  processes. The cache is safe under parallel workers.

* Hashing covers the full content. :func:`hash_array_summary` (name kept for compatibility) hashes
  shape, dtype and every byte of the array with a fixed-leaf blake2b tree, parallel across leaves.

* Atomic writes: write to ``tmp_<uuid>.pkl``, ``os.replace`` to the final path. ``os.replace`` is
  atomic on both POSIX and Windows (when source and destination share a filesystem); a crash
  mid-write leaves the orphan ``tmp_`` file but never a truncated final file.

* LRU eviction by file mtime when total size exceeds ``max_size_bytes``. Eviction is best-effort
  (no global lock) and only triggered on ``put``; worst case a parallel ``put`` race leaves the
  cache transiently over cap.

* Pickle protocol 5 + numpy buffer protocol for fast (de)serialisation of numpy arrays (zero-copy
  where possible). Every entry is verified against a ``pyutilz.core.safe_pickle`` sha256 sidecar on
  load -- fails CLOSED by default (a payload with no sidecar is refused, not silently unpickled);
  see that module's ``PYUTILZ_ALLOW_UNVERIFIED_PICKLE`` env var for the opt-out.

* Cache miss is silent (just return None and the caller computes). Cache hit logs at DEBUG so
  production logs stay quiet but ``-v`` shows the speedup.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
import struct
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np

from pyutilz.core.safe_pickle import PickleVerificationError, safe_load, write_sidecar


class _KeyLockEntry:
    """A per-key lock plus a count of callers currently holding/waiting on it."""

    __slots__ = ("lock", "refcount")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.refcount = 0

logger = logging.getLogger(__name__)

__all__ = [
    "DiskCache",
    "hash_array_summary",
    "hash_object",
    "compose_key",
]


# Former head/tail row count of the (retired, v<=3) summary hash. Kept only as the default of the ignored
# ``n_summary_rows`` parameter so existing callers keep working.
_DEFAULT_SUMMARY_ROWS = 64

# Default cache cap: ~1 GB.
_DEFAULT_MAX_SIZE_BYTES = 1_000_000_000

# Pickle protocol. 5 (PEP 574) supports out-of-band buffers for numpy arrays.
_PICKLE_PROTOCOL = 5

# Hash backend. blake2b is faster than sha256 and collision-resistant for the summary-bytes
# regime; outputs are truncated to 32 hex chars (128 bits) which is plenty for content addressing.
_HASH_DIGEST_BYTES = 16

# Version tag mixed into every array-summary digest. Bump whenever the SUMMARY RECIPE changes so
# pre-existing on-disk entries miss cleanly instead of silently colliding with a differently-derived
# key. This is a cache KEY, not a persisted value, so a one-time full miss is the correct migration.
#   v2 (2026-09-02): per-column sum/min/max moved to a fused numba pass (sequential accumulation
#   differs from numpy's pairwise summation in the last ulp) and the up-front whole-array
#   ``ascontiguousarray`` copy was dropped.
#   v3 (2026-09-03): non-numeric dtypes are no longer summarised by head/tail bytes alone --
#   bool/datetime64/timedelta64 get exact integer column reductions, string/void dtypes hash their
#   full buffer, and object dtype hashes element values instead of heap pointers.
#   v4 (2026-09-26): every non-object dtype hashes its FULL buffer (leaf-tree blake2b); the head/tail + sum/min/max
#   summary collided on middle-row swaps and on any change in a column containing a NaN.
_HASH_VERSION = 4


def _hasher() -> Any:
    """New blake2b hasher truncated to ``_HASH_DIGEST_BYTES``."""
    return hashlib.blake2b(digest_size=_HASH_DIGEST_BYTES)


def _buffer(arr: np.ndarray) -> memoryview:
    """The bytes of *arr* in C order, WITHOUT copying them.

    ``arr.tobytes()`` allocates a second full copy of the array purely to feed a hash. This is the
    cache-KEY computation, so that copy is paid on every lookup including the hits -- on the frames
    this library is used with, it can cost more than the work being cached.

    The digest is unchanged: ``tobytes()`` serialises in C order, which is exactly what
    ``ascontiguousarray`` guarantees, and every dtype in the suite is pinned to its pre-existing
    hex digest in the tests.

    The ``uint8`` view is not decoration. ``.data`` on a datetime64 or timedelta64 array raises
    ``ValueError: cannot include dtype 'M' in a buffer`` -- those dtypes have no buffer-protocol
    format -- so the plain copy-free form breaks on exactly the two dtypes this function goes out of
    its way to reduce correctly. Viewing the contiguous buffer as raw bytes first works for every
    dtype including 0-d, empty, structured and fixed-width string arrays.
    """
    # ``reshape(-1)``: a 2-D array gives a 2-D memoryview whose slices are ROWS, not bytes; the leaf hasher slices bytes.
    return np.ascontiguousarray(arr).view(np.uint8).reshape(-1).data


# Full-content hashing: the buffer is cut into fixed-size leaves, each leaf is blake2b-hashed (in a thread pool when there
# is more than one leaf; hashlib releases the GIL on large updates), and the leaf digests are hashed in order. The leaf
# size is a constant, never derived from the core count, so the digest is identical on every machine. Measured on 20M
# float64 (160 MB): one serial blake2b pass 0.83 s, the leaf tree about 0.2 s on 8 threads.
_LEAF_BYTES = 8 << 20
_MAX_HASH_THREADS = 8


def _feed_full_buffer(h: Any, arr: np.ndarray) -> None:
    """Feed a digest of EVERY byte of *arr* (C order) into *h*: single pass for one leaf, leaf tree otherwise.

    A leaf is a whole number of elements (``_LEAF_BYTES // itemsize``), so a strided input can be hashed leaf by leaf
    through ``arr.flat[i:j]`` (a C-order copy of just that leaf) instead of materialising a contiguous duplicate of the
    whole array, and still produce exactly the digest of its contiguous copy.
    """
    n_elems = int(arr.size)
    itemsize = int(arr.dtype.itemsize)
    h.update(struct.pack("<Q", n_elems * itemsize))
    leaf_elems = max(1, _LEAF_BYTES // max(1, itemsize))
    contiguous = bool(arr.flags["C_CONTIGUOUS"])
    if n_elems <= leaf_elems:
        h.update(_buffer(arr))
        return
    flat_buf = _buffer(arr) if contiguous else None

    def _leaf_digest(start: int) -> bytes:
        """blake2b digest of the leaf starting at element ``start`` (the last leaf may be shorter)."""
        stop = min(start + leaf_elems, n_elems)
        if flat_buf is not None:
            chunk = flat_buf[start * itemsize : stop * itemsize]
        else:
            chunk = _buffer(arr.flat[start:stop])
        return hashlib.blake2b(chunk, digest_size=_HASH_DIGEST_BYTES).digest()

    starts = range(0, n_elems, leaf_elems)
    n_workers = min(_MAX_HASH_THREADS, len(starts), os.cpu_count() or 1)
    if n_workers <= 1:
        digests = [_leaf_digest(st) for st in starts]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            digests = list(pool.map(_leaf_digest, starts))
    h.update(b"T")
    for d in digests:
        h.update(d)


def hash_array_summary(arr: np.ndarray, n_summary_rows: int = _DEFAULT_SUMMARY_ROWS) -> str:
    """Stable content hash of an ndarray over its FULL content.

    Hash inputs (in order, length-tagged so concatenation can't collide):
      1. ``_HASH_VERSION``
      2. shape tuple
      3. dtype.str (e.g. ``'<f8'``)
      4. every byte of the array in C order (object arrays: the ``repr`` of their element values)

    Until ``_HASH_VERSION`` 4 this was a summary (head/tail rows plus per-column sum/min/max), which gave the same key for
    two arrays differing only in the order of middle rows, or anywhere in a column containing a NaN (sum/min/max all
    NaN), so ``DiskCache`` returned results computed for different data. The summary pass already read every element,
    so it was never sub-O(N); the full hash costs a constant factor more and is exact.

    ``n_summary_rows`` is accepted for backward compatibility and ignored. Returns a 32-character hex string.
    """
    del n_summary_rows
    arr = np.asarray(arr)
    h = _hasher()
    h.update(struct.pack("<I", _HASH_VERSION))
    h.update(struct.pack("<I", len(arr.shape)))
    for dim in arr.shape:
        h.update(struct.pack("<q", int(dim)))
    dtype_bytes = arr.dtype.str.encode("ascii")
    h.update(struct.pack("<I", len(dtype_bytes)))
    h.update(dtype_bytes)
    if arr.size == 0:
        return str(h.hexdigest())
    if arr.dtype.kind == "O":
        # ``tobytes()`` on an object array serialises PyObject* ADDRESSES, which differ in every process -- hash the
        # element VALUES instead.
        h.update(repr(arr.tolist()).encode("utf-8", "backslashreplace"))
    else:
        _feed_full_buffer(h, arr)
    return str(h.hexdigest())


def hash_object(obj: Any) -> str:
    """Deterministic hash for hashable / JSON-able objects (params dicts, etc).

    Dict keys are sorted so hash order doesn't depend on dict-insertion order.
    """
    h = _hasher()
    _feed(h, obj)
    return str(h.hexdigest())


def _feed(h: Any, obj: Any) -> None:
    """Recursively feed an object into the hasher with a stable byte encoding."""
    if obj is None:
        h.update(b"N\0")
    elif isinstance(obj, bool):
        h.update(b"B" + (b"\x01" if obj else b"\x00"))
    elif isinstance(obj, int):
        h.update(b"I")
        # Variable-length signed encoding so arbitrary-size python ints hash without raising
        # OverflowError. Length-prefixed so concatenation cannot collide.
        n = int(obj)
        nbytes = max(1, (n.bit_length() + 8) // 8)
        h.update(struct.pack("<I", nbytes))
        h.update(n.to_bytes(nbytes, "little", signed=True))
    elif isinstance(obj, float):
        h.update(b"F")
        h.update(struct.pack("<d", obj))
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        h.update(b"b" + struct.pack("<Q", len(b)) + b)
    elif isinstance(obj, str):
        b = obj.encode("utf-8")
        h.update(b"s" + struct.pack("<Q", len(b)) + b)
    elif isinstance(obj, dict):
        h.update(b"D" + struct.pack("<Q", len(obj)))
        for k in sorted(obj, key=lambda x: repr(x)):
            _feed(h, k)
            _feed(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        tag = b"L" if isinstance(obj, list) else b"T"
        h.update(tag + struct.pack("<Q", len(obj)))
        for item in obj:
            _feed(h, item)
    elif isinstance(obj, (set, frozenset)):
        # frozenset is NOT a set subclass, so without it here a frozenset fell through to the
        # repr() last resort, whose iteration order varies with PYTHONHASHSEED -- a different
        # cache key in every process for the same content.
        h.update((b"S" if isinstance(obj, set) else b"Z") + struct.pack("<Q", len(obj)))
        for item in sorted(obj, key=lambda x: repr(x)):
            _feed(h, item)
    elif isinstance(obj, np.ndarray):
        # Defer to the array hasher so a nested array contributes a fixed-width digest, not its raw bytes.
        h.update(b"A")
        h.update(hash_array_summary(obj).encode("ascii"))
    elif hasattr(obj, "tolist"):
        # numpy scalars (np.int64, np.float32, ...) and 0-D arrays.
        _feed(h, obj.tolist())
    else:
        # Last resort: repr() the object. Not stable across Python runs for arbitrary types --
        # caller should only pass hashable primitives.
        h.update(b"R")
        h.update(repr(obj).encode("utf-8", errors="replace"))


def compose_key(*parts: str) -> str:
    """Compose multiple hash parts into one stable cache key.

    Joins with a length-prefix so parts of different cardinality can't collide via
    concatenation, then re-hashes so the final filename is fixed-width regardless of how many
    parts (or how long) went in.
    """
    if not parts:
        raise ValueError("compose_key requires at least one part")
    h = _hasher()
    for p in parts:
        b = str(p).encode("utf-8")
        h.update(struct.pack("<Q", len(b)))
        h.update(b)
    return str(h.hexdigest())


class DiskCache:
    """Content-addressable disk cache with LRU eviction.

    All entries live under ``cache_dir`` as files named ``<key>.pkl``. Hits use ``pickle.load``
    with protocol 5 buffer support; misses return ``None`` (the caller is responsible for the
    actual compute + ``put``).

    Eviction is best-effort: on ``put``, if the total directory size exceeds ``max_size_bytes``
    after the write, the oldest files (by mtime) are removed until the cap is met. Two parallel
    writers may transiently push the cache over cap; the next ``put`` from either reclaims.

    ``put`` for the SAME key is safe across threads in this process: the payload's ``os.replace``
    and its ``write_sidecar`` call are serialized per-key (see ``_key_locks``), so a thread's stale
    digest can never land after a later thread's payload replace (which would otherwise make
    ``get`` intermittently raise ``PickleVerificationError`` for an entry that was, in fact,
    written correctly by the last writer). Mirrors ``pyutilz.core.safe_pickle.safe_dump``'s
    per-path lock. Across processes the contract is unchanged: atomic rename means a partial write
    never wins.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ):
        """Create (or reuse) a disk cache rooted at ``cache_dir`` with an LRU cap of ``max_size_bytes``."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_bytes)
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.put_failures = 0
        # Per-key locks so a (payload replace, sidecar write) pair is atomic as a unit across
        # threads sharing this instance -- see the class docstring.
        self._key_locks: Dict[str, "_KeyLockEntry"] = {}
        self._key_locks_guard = threading.Lock()

    def __getstate__(self) -> Dict[str, Any]:
        """Excludes the unpicklable ``threading.Lock`` objects (``_key_locks`` holds per-key locks
        too); ``__setstate__`` rebuilds both fresh."""
        state = self.__dict__.copy()
        del state["_key_locks"]
        del state["_key_locks_guard"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._key_locks = {}
        self._key_locks_guard = threading.Lock()

    @contextlib.contextmanager
    def _get_key_lock(self, key: str) -> Iterator[None]:
        """Serialize (payload replace + sidecar write) for ``key`` across threads of this process.

        Reference-counted exactly like ``safe_pickle._get_path_lock``: cache keys are content
        digests, so they are effectively unique per call and a plain dict would grow one
        ``threading.Lock`` per key EVER written for the life of the process. A plain LRU bound is
        unsafe here (evicting an entry another thread is still blocked on would hand a later caller
        for the same key a different Lock), so entries are dropped only when nobody holds or waits
        on them -- keeping ``_key_locks`` proportional to keys currently in flight.
        """
        with self._key_locks_guard:
            entry = self._key_locks.get(key)
            if entry is None:
                entry = _KeyLockEntry()
                self._key_locks[key] = entry
            entry.refcount += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._key_locks_guard:
                entry.refcount -= 1
                if entry.refcount == 0:
                    del self._key_locks[key]

    def _key_path(self, key: str) -> Path:
        """File path under ``cache_dir`` for ``key``.

        ``key`` is meant to be a content-addressable hash (per the module docstring:
        "Content-addressable, NOT path-keyed"), but nothing enforces that here -- a key containing
        ``..``/path separators (e.g. ``"../evil"``) would resolve OUTSIDE ``cache_dir``, writing/
        reading the pickle (and its .sha256 sidecar) wherever the traversal pointed. Every internal
        caller already produces safe hex digests via ``hash_object``/``compose_key``/
        ``hash_array_summary``, but ``get``/``put`` accept an arbitrary ``str`` from any caller.
        """
        if not key or os.sep in key or (os.altsep and os.altsep in key) or "/" in key:
            # A key with a path separator survives the parents-traversal guard below ("sub/deep"
            # stays inside cache_dir) but names a file in a directory ``put`` never creates, so
            # every put fails and every get misses -- permanently, and silently. Reject it up front.
            raise ValueError(f"DiskCache: key {key!r} must not be empty or contain a path separator.")
        candidate = (self.cache_dir / f"{key}.pkl").resolve()
        cache_root = self.cache_dir.resolve()
        if cache_root not in candidate.parents:
            raise ValueError(f"DiskCache: key {key!r} resolves outside cache_dir ({candidate} not under {cache_root}); refusing.")
        return candidate

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value for ``key``, or ``default`` (``None``) on miss.

        ``None`` is a perfectly legal cached VALUE, so a bare ``None`` return cannot be read as a
        miss: ``put(k, None)`` followed by ``get(k)`` is a HIT (and is counted as one). Callers
        that need to tell the two apart must pass their own sentinel as ``default`` and compare
        with ``is``, e.g. ``MISS = object(); v = cache.get(k, MISS); if v is MISS: ...``.

        On hit, the file's mtime is touched (best-effort) so LRU eviction considers it
        recently-used. Touch failure is non-fatal.
        """
        path = self._key_path(key)
        if not path.exists():
            self.misses += 1
            return default
        try:
            # Fail CLOSED by default: a cache file with no .sha256 sidecar is refused so a payload
            # planted in the cache dir is never unpickled silently. See
            # pyutilz.core.safe_pickle's allow-unverified env var for the opt-in.
            value = safe_load(str(path))
        except PickleVerificationError as exc:
            # Sidecar digest mismatch -- payload bytes diverged from the hash recorded at
            # put-time. Treat as a corrupt cache entry (third-party tampering, mid-rename crash on
            # a non-atomic FS, or just a truncated copy) and drop both files so the next put
            # rebuilds cleanly.
            logger.debug("DiskCache: sidecar verification failed for %s: %s; removing", path, exc)
            self._drop_corrupt_entry(path)
            self.misses += 1
            return default
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            # Corrupt entry (e.g. mid-rename crash on a non-atomic FS). Drop the file so the next
            # put rebuilds cleanly.
            logger.debug("DiskCache: corrupt entry %s (%s); removing", path, exc)
            self._drop_corrupt_entry(path)
            self.misses += 1
            return default
        try:
            os.utime(path, None)
        except OSError:
            pass
        try:
            # The sidecar shares the entry's fate during eviction, so its mtime must track the
            # payload's; otherwise a frequently-read entry's sidecar looks like the oldest file in
            # the directory and gets evicted out from under its own payload.
            os.utime(str(path) + ".sha256", None)
        except OSError:
            pass
        self.hits += 1
        logger.debug("DiskCache: hit key=%s", key)
        return value

    @staticmethod
    def _drop_corrupt_entry(path: Path) -> None:
        """Best-effort removal of a corrupt cache entry's payload and sidecar."""
        try:
            path.unlink()
        except OSError:
            pass
        try:
            Path(str(path) + ".sha256").unlink()
        except OSError:
            pass

    def put(self, key: str, value: Any) -> None:
        """Atomically write ``value`` under ``key`` and LRU-evict if over cap.

        Write strategy:
          1. Pickle to a ``tmp_<uuid>.pkl`` file in the same directory.
          2. ``os.replace`` to the final ``<key>.pkl``. Atomic on POSIX and on Windows when source
             + destination share the filesystem.

        If the post-write directory size exceeds the cap, oldest entries (by mtime) are removed
        until back under cap. The just-written entry is protected from eviction in the same call.

        Never raises: a write failure (unpicklable value, full volume) is logged at WARNING and
        counted in ``put_failures``; the value is simply not cached.
        """
        path = self._key_path(key)
        tmp_name = f"tmp_{uuid.uuid4().hex}.pkl"
        tmp_path = self.cache_dir / tmp_name
        with self._get_key_lock(key):
            try:
                # 0o600 at creation (not open()'s default 0o666 & ~umask): os.replace carries the
                # mode onto the entry, and cache entries routinely hold API results / scraped pages.
                with os.fdopen(os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "wb") as f:
                    pickle.dump(value, f, protocol=_PICKLE_PROTOCOL)
                os.replace(tmp_path, path)
                try:
                    write_sidecar(str(path))
                except OSError as exc:
                    # A payload with no sidecar is refused (fail-closed) by the very next get(),
                    # so this is a real cache failure, not a cosmetic one.
                    self.put_failures += 1
                    logger.warning("DiskCache: sidecar write failed for %s: %s", path, exc)
            except Exception as exc:
                # Deliberately broad: an unpicklable payload raises TypeError (locks, sockets,
                # generators), AttributeError or RecursionError at least as often as
                # PicklingError, and any of those escaping here would strand the tmp_ file, which
                # neither eviction nor total_size() accounts for.
                self.put_failures += 1
                logger.warning("DiskCache: put failed for key=%s: %s", key, exc)
                return
            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        self._evict_if_needed(protect=path)

    def _evict_if_needed(self, protect: Optional[Path] = None) -> None:
        """LRU-evict by payload mtime until total size <= max_size_bytes.

        A cache entry is the indivisible pair ``(<key>.pkl, <key>.pkl.sha256)``: the sidecar is
        never an eviction candidate in its own right, its size counts towards the owning payload,
        and both files are unlinked together. A sidecar whose payload is already gone is an orphan
        that no lookup can ever revive, so it is swept unconditionally.
        """
        payload_sizes: Dict[Path, int] = {}
        sidecar_sizes: Dict[Path, int] = {}
        mtimes: Dict[Path, float] = {}
        try:
            with os.scandir(self.cache_dir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    if entry.name.startswith("tmp_"):
                        continue
                    try:
                        st = entry.stat()
                    except OSError:
                        continue
                    path = Path(entry.path)
                    if entry.name.endswith(".sha256"):
                        sidecar_sizes[Path(str(path)[: -len(".sha256")])] = st.st_size
                    else:
                        payload_sizes[path] = st.st_size
                        mtimes[path] = st.st_mtime
        except OSError:
            return

        total = sum(payload_sizes.values())
        for owner, size in sidecar_sizes.items():
            if owner in payload_sizes:
                total += size
                continue
            try:
                Path(str(owner) + ".sha256").unlink()
            except OSError:
                total += size

        if total <= self.max_size_bytes:
            return
        entries = sorted(payload_sizes, key=lambda p: mtimes[p])  # oldest first
        protect_resolved = protect.resolve() if protect is not None else None
        for fpath in entries:
            if total <= self.max_size_bytes:
                break
            if protect_resolved is not None and fpath.resolve() == protect_resolved:
                continue
            size = payload_sizes[fpath] + sidecar_sizes.get(fpath, 0)
            try:
                fpath.unlink()
            except OSError:
                continue
            try:
                Path(str(fpath) + ".sha256").unlink()
            except OSError:
                pass
            total -= size
            self.evictions += 1

    def clear(self) -> None:
        """Remove every entry in the cache directory.

        Does not touch the directory itself, so the instance stays valid for subsequent puts.
        """
        try:
            with os.scandir(self.cache_dir) as it:
                for entry in it:
                    if entry.is_file():
                        try:
                            os.unlink(entry.path)
                        except OSError:
                            pass
        except OSError:
            pass

    def total_size(self) -> int:
        """Sum of file sizes under cache_dir, ignoring tmp_ in-flight writes."""
        total = 0
        try:
            with os.scandir(self.cache_dir) as it:
                for entry in it:
                    if entry.is_file() and not entry.name.startswith("tmp_"):
                        try:
                            total += entry.stat().st_size
                        except OSError:
                            pass
        except OSError:
            pass
        return total
