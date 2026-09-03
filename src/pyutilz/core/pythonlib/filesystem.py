"""Filesystem-facing helpers: path/name sanitising, object dump/load, shelve, stream redirection.

Split out of the historical flat ``pyutilz.core.pythonlib`` module; re-exported from the
package ``__init__`` to preserve the public import surface.
"""

from ._common import (
    Callable,
    Iterable,
    Optional,
    SizedABC,
    Union,
    abspath,
    contextlib,
    contextmanager,
    errno,
    exists,
    inspect,
    join,
    joblib,
    logger,
    os,
    portalocker,
    re,
    shelve,
    sys,
)
from typing import Literal

# ----------------------------------------------------------------------------------------------------------------------------
# Filesystem
# ----------------------------------------------------------------------------------------------------------------------------


def get_partitioned_filepath(fname: str, depth: int = 2, replace_char="_"):
    """Returns optimized filepath derived from the first N (depth) chars of the filename, to view them manually in a more optimized form
    Instead of an enormously huge chunk of files in a single dir.
    Chars are lowercased and replaced with a safe symbol if needed, to stcik to the names allowed by a filesystem.
    """
    from os import sep

    if not fname:
        # ``sep.join([]) + sep`` is a bare separator, i.e. an ABSOLUTE prefix: os.path.join then
        # discards every component before it and the file lands at the filesystem root.
        return ""
    folders = []
    for char in fname[:depth].lower():
        if char.isalnum():
            folders.append(char)
        else:
            folders.append(replace_char)
    return sep.join(folders) + sep


def ensure_valid_filename(name: str, max_length: int = 255) -> str:
    """
    Replace invalid characters on Linux/Windows/MacOS with underscores.
    List from https://stackoverflow.com/a/31976060/819417
    Trailing spaces & periods are ignored on Windows.
    >>> ensure_valid_filename("  COM1  ")
    '_ COM1 _'
    >>> ensure_valid_filename("COM10")
    'COM10'
    >>> ensure_valid_filename("COM1,")
    'COM1,'
    >>> ensure_valid_filename("COM1.txt")
    '_.txt'
    >>> all('_' == ensure_valid_filename(chr(i)) for i in list(range(32)))
    True
    """
    return re.sub(
        r'[/\\:|<>"?*\0-\x1f]|^(AUX|COM[1-9]|CON|LPT[1-9]|NUL|PRN)(?![^.])|^\s|[\s.]$',
        "_",
        name[:max_length],
        flags=re.IGNORECASE,
    )


def load_file(fpath: str, unpickle_to_pd: bool = True, **kwargs):
    """
    Load plicked object, dataframe, Catboost model, based on file presence and name.
    """
    is_here = False
    fpath = abspath(fpath)
    try:
        is_here = exists(fpath)
    except Exception as e:  # nosec B110 - best-effort existence check; any OS-level error (e.g. invalid path chars) is treated as "not found" via the is_here default below
        logger.debug("Failed to check existence of %s: %s", fpath, e)
    if not is_here:
        logger.warning("File %s not found!", fpath)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fpath)
    else:
        if fpath.lower().endswith(".joblib"):
            return joblib.load(fpath)
        elif fpath.lower().endswith(".pckl"):
            if unpickle_to_pd:
                # Imported here, not at function entry: only this branch needs pandas, and a
                # top-level import made load_file("m.joblib") fail outright wherever joblib is
                # installed but pandas is not -- the same lazy-import reasoning the catboost
                # branch below already documents.
                import pandas as pd

                return pd.read_pickle(fpath)  # nosec B301 - fpath is caller-supplied (same trust level as the joblib.load branch above); this loader's whole purpose is deserializing a named local file
            from pyutilz.core.safe_pickle import safe_load

            return safe_load(fpath, **kwargs)
        elif fpath.lower().endswith(".bin"):
            # Lazy import: catboost is only needed for this branch, not for .joblib/.pckl
            # callers, who should not be forced to have it installed (found 2026-07-09 deptry
            # triage: the module-level import made catboost a hard dependency of this function).
            from catboost import CatBoostClassifier

            clf = CatBoostClassifier()
            return clf.load_model(fpath)
        else:
            # Falling off the end returned None for a file that exists but has an unknown
            # extension, which the caller only discovered far away as AttributeError on None.
            raise ValueError(f"load_file: unsupported extension for {fpath}")


class ObjectsAndFilesProcessor:
    """Container objects processing via interacting with the filesystem."""

    def process_objects(
        self,
        objects_names: Optional[Union[str, Iterable]] = None,
        container: Optional[dict] = None,
        names_sep: str = " ",
        path: str = "",
        namespace: str = "",
        namespace_sep: str = "_",
        file_extension: str = ".pckl",
        verbose: bool = True,
    ) -> int:
        """Walks over all сontainer's (dict-like) objects associated with files (one file per object), applies subclass-specific processing.

        If container is not specified, globals of the calling module are used.
        Objects names must be given exactly.

        Args:
            objects_names: list of object names to be populated. If it's a string, names_sep parameter is used to split it.
            If empty, all keys are processed.
            container: dictionary to be populated with objects names as keys, objects as values.
            names_sep: objects names separator.
            path: relative or absolute directory where the files are located.
            namespace: optional prefix added to files names.
            namespace_sep: separator of the namespace and file name.
            file_extension: default files extension to look for.
            verbose: whether to show or hide warnings in the log.

        Returns:
            number of successfully processed files.
        """
        nprocessed = 0
        fpath = abspath(path)

        # Split names if needed
        if isinstance(objects_names, str):
            objects_names = objects_names.split(sep=names_sep)

        # Get caller globals is no container specified
        if container is None:
            caller_globals = inspect.stack()[1][0].f_globals
            container = caller_globals

        if not objects_names:
            objects_names = container.keys()

        for obj_name in objects_names:

            file_name = f"{obj_name}{file_extension}"
            if namespace:
                file_name = namespace + namespace_sep + file_name
            file_name = join(fpath, file_name)

            if self._process_object(container=container, obj_name=obj_name, file_name=file_name, verbose=verbose):
                nprocessed += 1
            else:
                if verbose:
                    logger.warning("Skipped object %s.", obj_name)
        return nprocessed

    def _process_object(self, container: dict, obj_name: str, file_name: str, verbose: bool = True):
        """Processes a single object/file pair. Must be overridden in subclasses."""
        # This method should be overridden in the subclasses
        raise NotImplementedError


class ObjectsDumper(ObjectsAndFilesProcessor):
    """Dumps container to disk.

    Usage Example:
        >>> ObjectsDumper().process_objects("discovered_fields required_arguments nested_fields", path="vars")  # doctest: +SKIP
        3
    """

    def __init__(self, process_fcn: Callable = joblib.dump, process_kwargs: Optional[dict] = None, rewrite_existing: bool = True):
        if process_kwargs is None:
            process_kwargs = {"compress": 9}
        self.process_fcn = process_fcn
        self.process_kwargs = process_kwargs
        self.rewrite_existing = rewrite_existing

    def _process_object(self, container: dict, obj_name: str, file_name: str, verbose: bool = True):
        """Dumps `container[obj_name]` to `file_name` via `process_fcn`, if the object is truthy and (rewrite_existing or the file doesn't already exist)."""
        # Do not rewrite existing non-empty objects/keys, warn instead.
        obj = container.get(obj_name)
        if obj:
            if self.rewrite_existing or not exists(file_name):
                self.process_fcn(obj, file_name, **self.process_kwargs)
                return True


class ObjectsLoader(ObjectsAndFilesProcessor):
    """Populates container from disk.

    Usage Example:
        >>> ObjectsLoader(rewrite_existing=True).process_objects("discovered_fields required_arguments nested_fields", path="vars")  # doctest: +SKIP
        3
    """

    def __init__(self, process_fcn: Callable = joblib.load, process_kwargs: Optional[dict] = None, rewrite_existing: bool = False):
        if process_kwargs is None:
            process_kwargs = {}
        self.process_fcn = process_fcn
        self.process_kwargs = process_kwargs
        self.rewrite_existing = rewrite_existing

    def _process_object(self, container: dict, obj_name: str, file_name: str, verbose: bool = True):
        """Loads `file_name` via `process_fcn` into `container[obj_name]` if the file exists and the target is either absent, empty, or rewrite_existing is set."""
        if exists(file_name):
            if not self.rewrite_existing:
                # Do not rewrite existing non-empty objects/keys, warn instead.
                obj = container.get(obj_name)
                proceed = obj is None or (isinstance(obj, SizedABC) and len(obj) == 0)
            else:
                proceed = True

            if proceed:
                container[obj_name] = self.process_fcn(file_name, **self.process_kwargs)
                return True


def get_human_readable_set_size(set_size: int, rounding: int = 1) -> str:
    """Converts integer number of records into something human-redable

    >>> get_human_readable_set_size(100500)
    '100.5K'

    """
    for power, letter in ((9, "B"), (6, "M"), (3, "K")):
        if set_size >= 10**power:
            return str(round(set_size / 10**power, rounding)) + letter
    return str(round(set_size, rounding))


class HashableDict(dict):
    """A dict subclass that is hashable, based on its sorted (key, value) items. The dict must not be mutated while used as a hash key."""

    def __hash__(self):  # type: ignore[override]  # intentional: dict.__hash__ is None (unhashable); this recipe makes it hashable
        # Sort by (type name, str) rather than the raw key: sorted() on raw (key, value) tuples
        # raises TypeError for dicts with mixed-incomparable-type keys (e.g. str and int keys in
        # the same dict), since Python 3 no longer allows comparing across unrelated types.
        return hash(tuple(sorted(self.items(), key=lambda kv: (type(kv[0]).__name__, str(kv[0])))))


@contextlib.contextmanager
def open_safe_shelve(db_path: str, flag: Literal["r", "w", "c", "n"] = "c", protocol=None, writeback=False, timeout: int = 10):
    """Context manager opening `db_path` as a shelve db, guarded by a file lock to prevent concurrent access.

    On exit, closes/syncs the shelve database, then flushes the lock file and fsyncs it to disk
    before releasing the lock -- both run even if the caller's `with`-block body raises.
    """

    with portalocker.Lock(f"{db_path}.lock", "wb", timeout=timeout) as fh:
        db = shelve.open(db_path, flag=flag, protocol=protocol, writeback=writeback)  # nosec B301 - db_path is caller-supplied (this context manager's whole purpose is a locked local shelve db)
        try:
            yield db
        finally:
            # db.close() commits/syncs the underlying dbm backend (e.g. dbm.sqlite3 needs an
            # explicit close/commit before a subsequent open can see the writes; writeback=True
            # additionally never flushes its in-memory cache without this). Must happen before
            # the lock-file flush/fsync below and before the portalocker.Lock releases.
            db.close()
            fh.flush()
            os.fsync(fh.fileno())


@contextmanager
def suppress_stdout_stderr():
    """Context manager that redirects stdout and stderr to os.devnull for the duration of the block, restoring them afterwards."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            # Restore ONLY if this block's own devnull is still installed. The snapshot is
            # per-invocation but sys.stdout is process-global, so with two overlapping blocks
            # (threads, interleaved generators, ExitStack) the inner block's exit used to install
            # the outer block's already-CLOSED devnull as sys.stdout permanently -- every later
            # print raised "I/O operation on closed file".
            if sys.stdout is devnull:
                sys.stdout = old_stdout
            if sys.stderr is devnull:
                sys.stderr = old_stderr
