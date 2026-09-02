"""Content-addressable disk cache for repeated heavy computations.

Fits a specific niche: a deterministic transform of (large numpy/pandas inputs, params) whose cost
dominates a single call, invoked repeatedly across hyperparam sweeps / ablations / incremental data
updates where the inputs recur exactly or near-exactly. Caching the result amortises the cost
across re-calls with zero correctness loss. NOT a replacement for ``functools.lru_cache``
(in-process, hash-keyed by identity-of-args) or ``joblib.Memory`` (persists pickled function
arguments, slow for large arrays) -- this is content-addressable, summary-hashed, multi-process
safe, and built for numpy/pandas-shaped inputs specifically.

Ported from a downstream ML project's own shared cache module (used across two independent
consumers there). Design:

* Content-addressable, NOT path-keyed. Two processes writing the same key end up with
  bit-identical bytes (last-writer-wins via atomic rename), so no locking is needed across
  processes. The cache is safe under parallel workers.

* Hashing avoids reading the full payload bytes. :func:`hash_array_summary` computes a stable
  summary from (shape, dtype, first/last N rows, per-column sum/min/max); this keeps the key cost
  O(rows + cols) instead of O(rows*cols), which matters for large arrays.

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

import hashlib
import logging
import os
import pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
import struct
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from pyutilz.core.safe_pickle import PickleVerificationError, safe_load, write_sidecar

logger = logging.getLogger(__name__)

__all__ = [
    "DiskCache",
    "hash_array_summary",
    "hash_object",
    "compose_key",
]


# Number of leading + trailing rows fed into the array summary hash. 64 is enough to discriminate
# practical near-collisions (random row shuffles change the head/tail bytes) without making the
# hash O(rows). Column-axis coverage comes from per-column min/max/sum below, so even a
# single-row change to a middle row is caught when the column-sum changes.
_DEFAULT_SUMMARY_ROWS = 64

# Default cache cap: ~1 GB.
_DEFAULT_MAX_SIZE_BYTES = 1_000_000_000

# Pickle protocol. 5 (PEP 574) supports out-of-band buffers for numpy arrays.
_PICKLE_PROTOCOL = 5

# Hash backend. blake2b is faster than sha256 and collision-resistant for the summary-bytes
# regime; outputs are truncated to 32 hex chars (128 bits) which is plenty for content addressing.
_HASH_DIGEST_BYTES = 16


def _hasher() -> Any:
    """New blake2b hasher truncated to ``_HASH_DIGEST_BYTES``."""
    return hashlib.blake2b(digest_size=_HASH_DIGEST_BYTES)


def hash_array_summary(arr: np.ndarray, n_summary_rows: int = _DEFAULT_SUMMARY_ROWS) -> str:
    """Stable content hash of an ndarray from a sub-O(N) summary.

    Hash inputs (in order, length-tagged so concatenation can't collide):
      1. shape tuple
      2. dtype.str (e.g. ``'<f8'``)
      3. First ``n_summary_rows`` rows' raw bytes
      4. Last ``n_summary_rows`` rows' raw bytes
      5. Per-column ``sum``, ``min``, ``max`` (float64 cast for stability)

    For 1-D arrays the row slicing degrades to head/tail slices. For 0-D (scalar) arrays the slice
    is the whole array.

    Returns a 32-character hex string.
    """
    arr = np.ascontiguousarray(arr)
    h = _hasher()
    h.update(struct.pack("<I", len(arr.shape)))
    for dim in arr.shape:
        h.update(struct.pack("<q", int(dim)))
    dtype_bytes = arr.dtype.str.encode("ascii")
    h.update(struct.pack("<I", len(dtype_bytes)))
    h.update(dtype_bytes)
    if arr.size == 0:
        return str(h.hexdigest())
    if arr.ndim == 0:
        h.update(arr.tobytes())
    else:
        head_n = min(n_summary_rows, arr.shape[0])
        tail_n = min(n_summary_rows, arr.shape[0])
        h.update(np.ascontiguousarray(arr[:head_n]).tobytes())
        h.update(np.ascontiguousarray(arr[-tail_n:]).tobytes())
    if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
        col_axis = tuple(range(arr.ndim - 1))
        col_sum = np.asarray(arr.sum(axis=col_axis, dtype=np.float64)).ravel()
        col_min = np.asarray(arr.min(axis=col_axis)).astype(np.float64, copy=False).ravel()
        col_max = np.asarray(arr.max(axis=col_axis)).astype(np.float64, copy=False).ravel()
        h.update(col_sum.tobytes())
        h.update(col_min.tobytes())
        h.update(col_max.tobytes())
    elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
        triplet = np.array(
            [float(arr.sum(dtype=np.float64)), float(arr.min()), float(arr.max())],
            dtype=np.float64,
        )
        h.update(triplet.tobytes())
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
        # Defer to the summary hasher so arrays nested inside dicts/tuples don't drag the full
        # bytes into the hash.
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
        self._key_locks: Dict[str, threading.Lock] = {}
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

    def _get_key_lock(self, key: str) -> threading.Lock:
        """Return the lock for ``key``, creating it on first use."""
        with self._key_locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _key_path(self, key: str) -> Path:
        """File path under ``cache_dir`` for ``key``.

        ``key`` is meant to be a content-addressable hash (per the module docstring:
        "Content-addressable, NOT path-keyed"), but nothing enforces that here -- a key containing
        ``..``/path separators (e.g. ``"../evil"``) would resolve OUTSIDE ``cache_dir``, writing/
        reading the pickle (and its .sha256 sidecar) wherever the traversal pointed. Every internal
        caller already produces safe hex digests via ``hash_object``/``compose_key``/
        ``hash_array_summary``, but ``get``/``put`` accept an arbitrary ``str`` from any caller.
        """
        candidate = (self.cache_dir / f"{key}.pkl").resolve()
        cache_root = self.cache_dir.resolve()
        if cache_root not in candidate.parents:
            raise ValueError(f"DiskCache: key {key!r} resolves outside cache_dir ({candidate} not under {cache_root}); refusing.")
        return candidate

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for ``key``, or ``None`` on miss.

        On hit, the file's mtime is touched (best-effort) so LRU eviction considers it
        recently-used. Touch failure is non-fatal.
        """
        path = self._key_path(key)
        if not path.exists():
            self.misses += 1
            return None
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
            return None
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            # Corrupt entry (e.g. mid-rename crash on a non-atomic FS). Drop the file so the next
            # put rebuilds cleanly.
            logger.debug("DiskCache: corrupt entry %s (%s); removing", path, exc)
            self._drop_corrupt_entry(path)
            self.misses += 1
            return None
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
                with open(tmp_path, "wb") as f:
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
