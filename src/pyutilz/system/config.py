"""Thread-safe, hot-reloadable TOML configuration.

Watches a TOML file via mtime comparison (no background thread).
On each ``get()`` call, checks whether the file has changed and reloads
if necessary.  Thread-safe via a single ``threading.Lock``.

Usage::

    from pyutilz.system.config import TomlLiveConfig

    cfg = TomlLiveConfig("config.toml")
    value = cfg.get("section", "key", default=42, type_=int)

Features:
- Mtime-based debounced reload (configurable interval, default 5s)
- Type-casting with fallback chain: TOML value → *default* → *defaults* dict
- Lists and dicts returned as-is (no casting)
- Parse errors keep previous config (never crashes a running pipeline)
- File deletion keeps previous config + logs warning
- Change logging (every key that changed)
- Pure stdlib + tomllib (no pydantic dependency)
"""

from __future__ import annotations

import logging
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class TomlLiveConfig:
    """Thread-safe, hot-reloadable TOML config with mtime-based refresh.

    Parameters
    ----------
    path : Path or str
        Path to the TOML file.
    check_interval : float
        Minimum seconds between mtime checks (default 5.0).
    defaults : dict or None
        Flat ``{section: {key: value}}`` dict used as last-resort fallback
        when a key is missing from both the file and the *default* argument.
    logger : logging.Logger or None
        Logger for warnings and change notifications.  Falls back to the
        module-level logger if not provided.
    """

    def __init__(
        self,
        path: Path | str,
        check_interval: float = 5.0,
        defaults: dict[str, dict[str, Any]] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._path = Path(path)
        self._check_interval = check_interval
        self._defaults: dict[str, dict[str, Any]] = defaults or {}
        self._log = logger or _log
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._mtime: float = 0.0
        self._last_check: float = 0.0
        self._reload()

    # ── Internal reload logic ──────────────────────────────────────────

    def _reload(self) -> None:
        """Read and parse the TOML file.

        On parse error or missing file, keeps previous data and logs a
        warning so that a typo never crashes a running pipeline.
        """
        try:
            raw = self._path.read_bytes()
            new_data = tomllib.loads(raw.decode())
        except FileNotFoundError:
            if self._data:
                self._log.warning(
                    "Config file %s not found — keeping previous values", self._path,
                )
            else:
                # First load with no file — start with empty config (defaults used).
                self._mtime = 0.0
            return
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            self._log.warning(
                "Config parse error (keeping previous values): %s", exc,
            )
            try:
                self._mtime = self._path.stat().st_mtime
            except OSError:
                pass
            return

        self._data = new_data
        try:
            self._mtime = self._path.stat().st_mtime
        except OSError:
            pass

    def _maybe_reload(self) -> None:
        """Check mtime and reload if the file changed (debounced)."""
        now = time.monotonic()
        if now - self._last_check < self._check_interval:
            return
        with self._lock:
            if now - self._last_check < self._check_interval:
                return  # double-check after lock
            self._last_check = now
            try:
                mt = self._path.stat().st_mtime
            except FileNotFoundError:
                if self._data:
                    self._log.warning(
                        "Config file %s disappeared — keeping previous values",
                        self._path,
                    )
                return
            if mt != self._mtime:
                self._log.info("Config file changed, reloading: %s", self._path)
                old_data = self._data
                self._reload()
                self._log_changes(old_data, self._data)

    # ── Public API ─────────────────────────────────────────────────────

    def get(
        self,
        section: str,
        key: str,
        default: Any = None,
        type_: type = int,
    ) -> Any:
        """Return a config value, falling back to *default* then *defaults*.

        Lists and dicts are returned as-is (no casting).
        If the TOML value cannot be cast to *type_*, logs a warning and
        returns the fallback instead of crashing.
        """
        self._maybe_reload()
        # Avoid allocating a throwaway {} on every section miss — use a
        # module-level empty sentinel instead.
        section_data = self._data.get(section)
        val = section_data.get(key) if section_data is not None else None
        if val is not None:
            if isinstance(val, (list, dict)):
                return val
            # Skip the type cast when the value already has the right type —
            # TOML parses floats/ints/bools natively so type_() is usually a
            # no-op coercion that still costs a Python function call.
            if type_ is not type(val) and not isinstance(val, type_):
                try:
                    return type_(val)
                except (TypeError, ValueError) as exc:
                    fallback = (
                        default
                        if default is not None
                        else (self._defaults.get(section) or {}).get(key, 0)
                    )
                    self._log.warning(
                        "config [%s].%s: bad value %r (%s), using %s",
                        section, key, val, exc, fallback,
                    )
                    return fallback
            return val
        if default is not None:
            return default
        section_defaults = self._defaults.get(section)
        return section_defaults.get(key) if section_defaults is not None else None

    def get_section(self, section: str) -> dict[str, Any]:
        """Return the entire section dict (triggers reload check once).

        Faster than N individual ``get()`` calls when reading several keys
        from the same section — ``_maybe_reload()`` is called only once and
        the section dict is looked up only once.  Returns an empty dict if
        the section is missing.
        """
        self._maybe_reload()
        return self._data.get(section) or self._defaults.get(section) or {}

    @property
    def data(self) -> dict[str, Any]:
        """Return the full config dict (triggers lazy reload check)."""
        self._maybe_reload()
        return self._data

    @data.setter
    def data(self, value: dict[str, Any]) -> None:
        """Replace config data programmatically (e.g. after validation)."""
        self._data = value

    # ── Change logging ─────────────────────────────────────────────────

    def _log_changes(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        """Log every parameter that changed between two config snapshots."""
        all_sections = set(old) | set(new)
        for section in sorted(all_sections):
            old_sec = old.get(section, {})
            new_sec = new.get(section, {})
            if not isinstance(old_sec, dict) or not isinstance(new_sec, dict):
                if old_sec != new_sec:
                    self._log.info("[config] [%s] changed: %r -> %r", section, old_sec, new_sec)
                continue
            all_keys = set(old_sec) | set(new_sec)
            for key in sorted(all_keys):
                old_val = old_sec.get(key)
                new_val = new_sec.get(key)
                if old_val != new_val:
                    self._log.info(
                        "[config] [%s].%s changed: %r -> %r",
                        section, key, old_val, new_val,
                    )
