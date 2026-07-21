# Configuration & Cross-Platform Robustness Audit

## Summary

I read the following files in full: `src/pyutilz/system/config.py`, `src/pyutilz/system/system/_common.py`, `src/pyutilz/system/system/fsutils.py`, `src/pyutilz/system/system/memory.py`, `src/pyutilz/system/system/misc.py`, `src/pyutilz/system/system/probing.py`, `src/pyutilz/system/system/sysinfo.py`, `src/pyutilz/system/parallel.py`, `src/pyutilz/system/gpu_dispatch.py`, `src/pyutilz/system/distributed.py` (grep pass), `src/pyutilz/core/filemaker.py`, `src/pyutilz/database/deltalakes.py`, `src/pyutilz/web/browser.py`, `src/pyutilz/dev/notebook_init.py`, `src/pyutilz/text/strings/configfiles.py`, `src/pyutilz/web/proxy/decodo.py`, `src/pyutilz/performance/kernel_tuning/cache/cache_base.py`, `src/pyutilz/performance/kernel_tuning/cache/cache_class.py`, plus a repo-wide grep for `os.environ`, `tempfile`, `os.getcwd`, `os.sep`, `.upper()/.lower()`. All three material findings below were empirically reproduced with standalone scripts (not just read and asserted). Findings: 3 HIGH, 1 MEDIUM.

## Findings

### [HIGH] `safe_delta_write` lock key is case-sensitive on a case-insensitive filesystem — src/pyutilz/database/deltalakes.py:82-84

- **Category**: cross-platform path handling / concurrency
- **Problem**: The per-table lock filename is derived via `hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()`. `os.path.abspath()` (via `ntpath` on Windows) normalizes slashes and relative components but **never changes case**. On Windows, the default filesystem (NTFS) is case-*insensitive* but case-*preserving* — `C:\Data\orders` and `c:\data\ORDERS` refer to the exact same file/table, but hash to two completely different lock filenames.
- **Failure scenario**: Two processes/threads on the same Windows host call `safe_delta_write("C:\\Data\\region-us\\orders", writer_a)` and `safe_delta_write("c:\\data\\REGION-US\\Orders", writer_b)` — a very ordinary occurrence when the path is assembled from different sources (a hardcoded literal in one module vs. an env-var-driven path built elsewhere, or a caller that happens to normalize case differently). Verified empirically:
  ```
  lock1: 3f05ac74ce7d758d28d854c7f101baa07ff96d2351f77dd8b203edeccc5f2a00
  lock2: e0ff141c2fc2ce8f47ec7475890add45b6941a1cdcd3b835daa250a5308f578c
  SAME LOCK FILE: False
  ```
  Both `FileLock` instances acquire "their" lock immediately (no contention observed), and both `delta_op_func()` calls proceed concurrently against the *same* underlying Delta table — exactly the race this function exists to prevent (concurrent commits to the same Delta log can corrupt/interleave transactions). This is a Windows-only failure mode: the same code is safe on Linux/macOS where `os.path.abspath` on a case-sensitive filesystem genuinely does distinguish different paths, but per CLAUDE.md this library explicitly targets Windows as a first-class dev/runtime platform.
- **Suggested fix**: Hash `os.path.normcase(os.path.abspath(path))` instead of the raw `abspath`. `os.path.normcase` is a no-op on POSIX and lower-cases (and backslash-normalizes) on Windows, so both spellings of the same table collapse to one lock file everywhere.

### [HIGH] `write_config_file`/`read_config_file` round-trip corrupts non-ASCII values on any non-UTF-8-locale host — src/pyutilz/text/strings/configfiles.py:32, 123

- **Category**: locale-dependent I/O / silent data corruption
- **Problem**: `write_config_file` always writes with an explicit encoding: `open(file, "w", encoding="utf-8")` (line 123). `read_config_file`, however, calls `config.read(file)` (line 32) with **no `encoding=` argument** — `configparser.ConfigParser.read(filenames, encoding=None)` then opens the file using Python's locale-dependent default text encoding (`locale.getpreferredencoding(False)`), which is UTF-8 only on hosts explicitly configured that way (e.g. `PYTHONUTF8=1`, or a UTF-8 system locale). On a non-English Windows install (or any POSIX host with a non-UTF-8 `LANG`), this default is something else (e.g. cp1251, cp1252, cp936).
- **Failure scenario**: Empirically reproduced on this Windows host (default locale encoding cp1251):
  ```python
  write_config_file("cfg.ini", {"name": "café"}, section="MAIN", encryption=None)
  out = {}
  read_config_file("cfg.ini", out, section="MAIN", encryption=None)
  # out["name"] == "caf" + U+0413("Г") + U+00A9("©")   -- NOT "café"  # codespell:ignore
  # MATCH: False
  ```
  The UTF-8 bytes for "é" (`0xC3 0xA9`) get silently re-decoded as cp1251, producing mojibake with **no exception and no warning** — the function returns `True` (success) both ways. On a strict-ASCII locale (`LANG=C`/`POSIX`) the same mismatch instead raises an uncaught `UnicodeDecodeError` deep inside `configparser`, crashing the caller. This is reachable through the module's own public, documented `encryption=None` parameter (used whenever a caller wants human-readable, non-base64 config values) — not an adversarial or default-only path.
- **Suggested fix**: Pass `encoding="utf-8"` explicitly to `config.read(file, encoding="utf-8")` in `read_config_file`, matching the writer.

### [HIGH] `TomlLiveConfig.get()` silently substitutes a hardcoded `0` for a malformed config value when no default is given — src/pyutilz/system/config.py:166-174

- **Category**: configuration fallback / fail-loud violation
- **Problem**:
  ```python
  try:
      return type_(val)
  except (TypeError, ValueError) as exc:
      fallback = default if default is not None else (self._defaults.get(section) or {}).get(key, 0)
      self._log.warning(...)
      return fallback
  ```
  When a TOML value can't be cast to the requested `type_` and the caller passed no `default` and the `defaults` dict has no matching entry, the method falls back to a hardcoded integer `0` — regardless of the requested `type_` (even for `type_=str` or `type_=float` call sites) and regardless of what `0` actually *means* for that particular setting.
- **Failure scenario**: Empirically reproduced:
  ```python
  # cfg.toml: [limits]\nmax_retries = "unlimited"
  cfg.get("limits", "max_retries", type_=int)
  # -> logs "bad value 'unlimited' ... using 0"
  # -> returns 0
  ```
  A single typo/bad value in a hot-reloaded TOML file (this class's entire purpose is to be *live*-reloaded in a running pipeline) silently turns e.g. `max_retries=0` into "retry loop body never executes," or a `min_free_gb_threshold=0` into "disk-space safety check always passes," or feeds `0` into a downstream divisor causing a confusing `ZeroDivisionError` far from the real mistake — all without the process failing at the point the bad config was actually read, only a `logger.warning` that's easy to miss in a busy log stream.
- **Suggested fix**: When `default is None` and the key isn't in `self._defaults`, either re-raise (fail loudly — a config file this class is designed to watch/reload should not silently coerce garbage into a number) or return `None` so the caller's own downstream logic has to explicitly handle "no valid value," instead of a type-blind, semantically-arbitrary `0`.

## Things done well

- `_pid_alive()` in `cache_base.py` correctly implements Windows liveness probing via `OpenProcess`/`GetExitCodeProcess` with `use_last_error=True` (documented fix for a real `ctypes.windll.kernel32` shared-DLL last-error bug) instead of naively trying `os.kill(pid, 0)`, which doesn't have POSIX signal semantics on Windows.
- `get_own_memory_usage()`/`clean_ram()` in `system/memory.py` correctly distinguish Windows `rss` (working-set, falsely near-zero after a working-set trim) from `private` (commit charge), with an empirically-documented before/after measurement — a real, well-verified Windows-specific gotcha many libraries get wrong.
- `get_linux_board_info()` in `probing.py` explicitly opens `/sys/devices/...` files with `encoding="utf-8"` specifically to avoid the locale-dependent default breaking under WSL — showing awareness of exactly the class of bug flagged in Finding 2 above, just not applied consistently across the whole codebase.
- `nvidia-smi`/`nvcc` subprocess calls use explicit `timeout=` and list-form `argv` (not a shell string), with a documented fix for the classic `["nvcc --version"]` vs `["nvcc", "--version"]` POSIX `shell=False` footgun.
- `_try_create_marker()` in `cache_class.py` handles the "hardlinks unsupported on this filesystem" case (relevant to some network/exotic filesystems) with a documented O_EXCL fallback rather than assuming `os.link` always works.

## Investigated, not an issue

- `find_chrome_executable()` (`web/browser.py:143-157`) builds candidate paths via `os.sep.join((item, "Google/Chrome/Application", "chrome.exe"))`, mixing `os.sep` (backslash on Windows) with hardcoded forward slashes in the sub-path. This produces mixed-separator strings like `C:\Program Files\Google/Chrome/Application\chrome.exe`, but Windows' file APIs (and thus `os.path.exists`/`os.access`) accept forward slashes transparently, so this doesn't actually break — confirmed no failure mode, just unusual style (should use `os.path.join` throughout).
- `get_locale_settings(locale_name="en_US.utf8")` (`system/misc.py`, docstring example) — suspected this POSIX-style locale name would raise `locale.Error` under `locale.setlocale` on Windows (which historically used its own locale-name syntax). Tested directly on this environment: it succeeds and returns the expected dict. Modern Windows (UCRT-based, Windows 10 1803+) accepts many POSIX-style locale aliases, so this is not reliably broken — lower-confidence area but not a confirmed bug.
- `is_local_path()` (`deltalakes.py:21-32`) correctly classifies UNC paths (`\\server\share\...`) and mapped drives as "local" via the empty-scheme fallback, which is intentional and consistent with the module's stated scope (local file locking) — flagged mentally as a possible "network share, cross-machine" gap, but the docstring's contract is specifically local-lock semantics, so it isn't a divergent-behavior bug, just an inherent limitation of `tempfile.gettempdir()`-based locking that's out of scope for this pass.
