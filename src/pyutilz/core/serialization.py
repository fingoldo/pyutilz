"""Everything related to persisting objects into memory or disk."""

# ----------------------------------------------------------------------------------------------------------------------------
# Everything related to persisting objects into memory or disk!
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Callable, Optional, Union

import sys
import io
import errno
import itertools
import uuid
import pickle, zlib, os  # nosec B403 - pickle.dumps/loads here operate on caller-provided in-process objects/paths for serialize()/unserialize(); untrusted-path loading is additionally gated via the optional verify_sidecar path (see pyutilz.core.safe_pickle)
from pyutilz.system import system
from pyutilz.core.safe_pickle import PickleVerificationError


def str_to_class(classname: str) -> Any:
    """Instantiates a class given its fully-qualified dotted name, optionally with an inline constructor argument.

    `classname` may include a trailing `(...)` segment (e.g. "module.sub.ClassName(some_arg)"), whose
    contents (including the parentheses) are passed as the single positional argument to the class's
    constructor. If `classname` has no module prefix, the class is looked up in this module's namespace.
    """
    pos1 = classname.find("(")
    init_data = ""
    if pos1 >= 0:
        pos2 = classname.find(")", pos1)
        if pos2 >= 0:
            # Both are indexes OF the bracket characters, so the closing one must be included to
            # honour the documented "including the parentheses" contract -- an end-exclusive
            # slice handed the constructor an unbalanced "(arg".
            init_data = classname[pos1 : pos2 + 1]
            classname = classname[:pos1]
    opts = classname.split(".")
    if len(opts) > 1:
        the_module = ".".join(opts[:-1])
        the_class = opts[-1]
        return getattr(sys.modules[the_module], the_class)(init_data)
    else:
        return getattr(sys.modules[__name__], classname)(init_data)


def serialize(obj: Any, fname: Optional[Union[str, io.IOBase]] = None, compression: Optional[int] = 9) -> Union[bool, bytes, None]:
    """
    If fname is passed, data will be persisted onto disk and success code will be returned
    Otherwise, serialized representation of the obj in memory will be returned.
    """
    if compression is not None:
        assert isinstance(compression, int)  # nosec B101 - internal API-misuse guard on caller's zlib compression level arg, not a security boundary
        assert compression >= -1 and compression <= 9  # nosec B101 - validates zlib's own accepted compression-level range (-1..9), not a security boundary
    try:
        data = pickle.dumps(obj)
        if compression is not None:
            data = zlib.compress(data, compression)
        if fname is not None:
            if isinstance(fname, str):
                # Bug: this passed fname (the FILE path) to ensure_dir_exists, which os.makedirs()'d
                # a DIRECTORY at that exact path whenever it didn't already exist -- the subsequent
                # open(fname, "wb") then failed with PermissionError (can't open a directory for
                # writing). Every existing caller/test happened to pre-create fname via
                # tempfile.mkstemp() first, so os.path.exists(fname) was already True and the buggy
                # makedirs() call never fired. Pass the DIRECTORY (dirname), and only when non-empty
                # (a bare relative filename with no directory component needs no makedirs at all).
                dirname = os.path.dirname(fname)
                if dirname:
                    system.ensure_dir_exists(dirname)
                with open(fname, "wb") as f:
                    f.write(data)
            elif isinstance(fname, io.IOBase):
                fname.write(data)
            else:
                raise TypeError(f"Unsupported fname type for serialize: {type(fname)}")
            return True
        else:
            return data
    except Exception as e:
        logger.exception("serialize failed (fname=%r, type=%s): %s", fname, type(obj).__name__, e)
        return None


def unserialize(obj: Union[str, bytes, io.IOBase], compression: Optional[int] = 9, verify_sidecar: bool = False) -> Any:
    """
    If fname is passed, data will be read from disk.
    Otherwise, obj will be read from memory directl.
    Unpacked data will be returned.

    ``verify_sidecar`` (default False, preserving historical behaviour for existing callers --
    ``serialize()`` in this same module never writes a ``.sha256`` sidecar, so flipping this
    default would break every plain ``serialize()``/``unserialize()`` round-trip; use
    :mod:`pyutilz.core.safe_pickle`'s ``safe_dump``/``safe_load`` instead when you want
    verification on by default): when True AND ``obj`` is a file path, requires a matching
    ``<path>.sha256`` sidecar (see :mod:`pyutilz.core.safe_pickle`) before unpickling. Unlike
    every other failure mode in this function, a missing/mismatched sidecar is raised to the
    caller as :class:`pyutilz.core.safe_pickle.PickleVerificationError` -- it is NOT swallowed
    into the generic ``except Exception: logger.exception(...); return None`` path below, so
    callers can distinguish "tampered/corrupt" from "file not found" or any other I/O error.
    Write the sidecar for a trusted file with ``pyutilz.core.safe_pickle.write_sidecar(path)``.
    Has no effect when ``obj`` is already bytes/a file-like object (there's no "path" to check a
    sidecar against; that in-memory data was already produced within the same process/caller).

    Raises ``FileNotFoundError`` for a missing path, ``TypeError`` for an unsupported input type,
    and propagates any deserialization/I/O error. A ``None`` return therefore means exactly one
    thing: the stored object itself was ``None``.
    """
    if compression is not None:
        assert isinstance(compression, int)  # nosec B101 - internal API-misuse guard mirroring serialize()'s compression arg check, not a security boundary
        assert compression >= -1 and compression <= 9  # nosec B101 - validates zlib's own accepted compression-level range (-1..9), not a security boundary
    try:
        if isinstance(obj, str):
            if not os.path.isfile(obj):
                # None is a perfectly valid unpickled value, so it cannot double as the
                # error signal: a missing file must be distinguishable from a successful
                # round-trip of None.
                logger.error("File %s not found", obj)
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), obj)
            else:
                if verify_sidecar:
                    from pyutilz.core.safe_pickle import verify_sidecar as _verify_sidecar

                    if not _verify_sidecar(obj):
                        raise PickleVerificationError(
                            f"unserialize: refusing to unpickle {obj!r}; sha256 sidecar missing or mismatch. "
                            "Run pyutilz.core.safe_pickle.write_sidecar(path) on a trusted copy, or call "
                            "unserialize(..., verify_sidecar=False) to accept the historical unverified behaviour."
                        )
                with open(obj, "rb") as f:
                    obj = f.read()
        elif isinstance(obj, io.IOBase):
            obj = obj.read()

        if isinstance(obj, bytes):
            data = obj
            if compression is not None:
                try:
                    data = zlib.decompress(obj)
                except zlib.error:
                    # zlib raises "incorrect header check" / "incorrect data check" for uncompressed input:
                    # fall back to treating obj as raw (uncompressed) pickle bytes.
                    logger.warning("Data seems to be not compressed; reading as raw pickle")
                    data = obj
            data = pickle.loads(data)  # nosec B301 - opt-in sidecar verification above (verify_sidecar=True) covers the file-path case;
            # in-memory bytes/file-like input is the caller's own data, produced within the same process
            return data
        else:
            raise TypeError(f"unserialize: unexpected input data type {type(obj)}; expected str path, bytes or a file-like object")
    except PickleVerificationError:
        # Deliberately NOT swallowed into the blanket handler below: a tamper/corruption signal
        # from an opted-in sidecar check must stay distinguishable from "file not found" or any
        # other I/O error, so callers can `except PickleVerificationError` specifically.
        raise
    except Exception as e:
        # Genuine deserialization/I/O failures are re-raised rather than folded into a None
        # return, which a caller cannot tell apart from a legitimately-unpickled None.
        logger.exception(e)
        raise


# ----------------------------------------------------------------------------------------------------------------------------
# Atomic file writes
# ----------------------------------------------------------------------------------------------------------------------------

# Monotonic counter for the atomic-write temp-file name suffix. Combined with PID + an 8-byte uuid
# hex this gives a unique name without paying mkstemp's O_EXCL retry loop, which on Windows is
# roughly 30x slower than a direct os.open due to Defender / filesystem-filter-driver intercepts
# on the secure-name-generation probe.
_ATOMIC_WRITE_COUNTER = itertools.count(0)


def _atomic_write_counter() -> int:
    """Next value of the monotonic counter used (with PID + a uuid suffix) to name atomic-write
    temp files uniquely, avoiding ``mkstemp``'s much slower O_EXCL retry loop on Windows."""
    return next(_ATOMIC_WRITE_COUNTER)


def atomic_write_bytes(target_path: str, writer_fn: Callable[[Any], None], *, fsync: bool = False) -> None:
    """Atomically write to ``target_path`` via write-tmp-then-rename.

    A plain ``with open(target_path, "wb") as f: ...`` lets two parallel writers truncate each
    other mid-write -- a subsequent load then raises an opaque ``UnpicklingError``/``EOFError``,
    with no way to tell corruption from a legitimate version mismatch.

    This helper:
      1. Creates a temp file in the SAME directory (``os.replace`` across filesystems isn't
         atomic; same-FS is).
      2. Invokes ``writer_fn(fileobj)`` -- the caller owns the bytes.
      3. ``os.replace()`` atomically renames tmp -> target (works on both POSIX and Windows since
         Python 3.3; ``os.rename`` on Windows would fail when the target already exists).
      4. Cleans up the tmp file on any exception so a failed write doesn't leak a
         ``target.xyz.tmp`` file alongside the real one.

    The atomicity guarantee: a concurrent reader either sees the complete pre-write file or the
    complete post-write file, never a partial one. Concurrent writers still race (last writer
    wins), but neither produces corruption.

    ``fsync``: when True, calls ``f.flush()`` + ``os.fsync(fd)`` before the rename so the new
    contents survive a power loss BEFORE the OS page-cache commits. This is the dominant cost on
    Windows (``FlushFileBuffers`` blocks until the disk write cache is committed -- measured
    ~400ms per call on commodity SSDs even for ~1MB files), so it defaults to False: the atomic
    write-tmp-then-rename semantics still hold WITHOUT fsync (concurrent readers never see a
    partial file), only the post-rename DURABILITY window is shortened (the OS flushes the page
    cache to physical disk within a few seconds; a power loss in that window may leave a
    freshly-renamed file with its contents only on RAM-side pages). Pass ``fsync=True`` explicitly
    when writing state that must survive a crash immediately after the call returns.
    """
    target_dir = os.path.dirname(target_path) or "."
    _tmp_basename = f"{os.path.basename(target_path)}.tmp.{os.getpid()}.{_atomic_write_counter():d}.{uuid.uuid4().hex[:8]}"
    tmp_path = os.path.join(target_dir, _tmp_basename)
    fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
    # fd ownership: passed to os.fdopen -> the resulting BufferedWriter takes ownership and closes
    # on context exit. But if os.fdopen itself raises (rare: MemoryError during buffer alloc, or
    # an invalid-mode TypeError after a future refactor), the raw fd is never adopted and Python
    # leaks it -- under sustained write pressure this exhausts the process fd ceiling. Track
    # adoption via a flag and explicitly close in the except branch when adoption never happened.
    _fd_adopted = False
    try:
        with os.fdopen(fd, "wb") as f:
            _fd_adopted = True  # BufferedWriter now owns fd; on with-exit it closes.
            writer_fn(f)
            if fsync:
                # fsync inside the with-block: pickle.dump/dill.dump/numpy.save only flush their
                # own buffers, not the OS page cache. Without an explicit fsync, a power loss
                # between rename and writeback can publish a visible filename whose contents are
                # still dirty pages.
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        if not _fd_adopted:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
