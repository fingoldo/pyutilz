"""Proxy-pool primitives: building a ``requests``-style proxies dict, recording last-use/failure
timestamps, picking an eligible (not recently used/failed) port, and recognising a server-side
rotating gateway.

Named ``proxy_pool`` rather than ``proxies`` on purpose: the package exposes a mutable
``proxies`` global, and a submodule of that name would be bound over it as a package attribute
by ordinary import machinery.
"""

from datetime import datetime, timezone
from typing import Optional

from ._common import joblib_hash, logger

# See ipinfo.py for why the parent is reached through `import <parent> as _facade` rather than a
# from-import: cycle survival plus patchability of ``make_proxies_dict``/``sleep``/``random``.
import pyutilz.web.web as _facade


def set_proxy_last_use_time(last_used_dict: Optional[dict], proxies: Optional[dict]) -> None:
    """Record the current UTC time as the last-use timestamp for ``proxies`` (keyed by its joblib hash) in ``last_used_dict``, if it's a dict."""
    if isinstance(last_used_dict, dict):
        last_used_dict[joblib_hash(proxies)] = datetime.now(timezone.utc).replace(tzinfo=None)  # naive-UTC is this module's public dict-timestamp convention (see get_new_smartproxy/tests, which store/compare naive-UTC values); switching to aware would break subtraction against caller-supplied entries


def make_proxies_dict(proxy_user: Optional[str], proxy_pass: Optional[str], proxy_server: str, proxy_port: int, proxy_type: str = "https") -> dict:
    """Build a ``requests``-style ``{"http": ..., "https": ...}`` proxy URL dict from the given credentials/server/port/scheme."""
    if proxy_user and proxy_pass:
        proxy_url = "%s:%s@%s:%s" % (proxy_user, proxy_pass, proxy_server, proxy_port)
    else:
        proxy_url = "%s:%s" % (proxy_server, proxy_port)
    # return {"http": f"http://{proxy_url}", "https": f"https://{proxy_url}"}
    return {"http": f"{proxy_type}://{proxy_url}", "https": f"{proxy_type}://{proxy_url}"}


def get_new_smartproxy(
    proxy_user: Optional[str],
    proxy_pass: Optional[str],
    proxy_server: str,
    proxy_min_port: int = 20001,
    proxy_max_port: int = 37960,
    job_desc: str = "",
    last_used_dict: Optional[dict] = None,
    min_idle_interval_minutes: float = 0,
    failed_dict: Optional[dict] = None,
    min_failed_idle_interval_minutes: float = 60 * 24,
    warn_after_n_failures: int = 5,
    delay: int = 5,
    proxy_port: Optional[int] = None,
    proxy_type: str = "http",
    verbose=False,
    max_wait_seconds: Optional[float] = None,
) -> dict:
    """Pick a proxy (random port within ``[proxy_min_port, proxy_max_port]`` unless ``proxy_port`` is fixed) that hasn't been touched recently per ``last_used_dict``/``failed_dict``.

    Keeps re-rolling a random port and sleeping ``delay`` seconds every ``warn_after_n_failures``
    attempts until an eligible (not recently used/failed within ``min_idle_interval_minutes``)
    proxy dict is found, then returns it. Blocks indefinitely if ``max_wait_seconds`` is None
    (the default, preserving historical behaviour) and no proxy in the range is ever eligible --
    a real risk for a fixed/single-proxy pool (e.g. ``proxy_port`` fixed, or
    ``proxy_min_port == proxy_max_port``, the exact "server-side rotation" pattern
    :func:`is_rotating_proxy` recognizes) after one transient failure marks the sole candidate
    "touched" for up to ``min_failed_idle_interval_minutes`` (default 24h). Every
    ``warn_after_n_failures`` attempts is now logged unconditionally (not gated on ``verbose``)
    specifically so a stuck wait is diagnosable from logs alone; pass ``max_wait_seconds`` to
    raise :class:`TimeoutError` instead of blocking forever once that budget is exhausted.
    """
    if failed_dict is None:
        failed_dict = {}
    if last_used_dict is None:
        last_used_dict = {}
    n = 0
    now_time = datetime.now(timezone.utc).replace(tzinfo=None)  # must stay naive to subtract against caller-supplied last_used_dict/failed_dict entries, which follow this module's naive-UTC timestamp convention (see set_proxy_last_use_time)
    wait_started_at: Optional[datetime] = None
    # Captured once: rebinding the `proxy_port` PARAMETER inside the loop froze the first random
    # draw forever, so the "keeps re-rolling a random port" contract above never held.
    fixed_port = proxy_port
    while True:
        # ----------------------------------------------------------------------------------------------------------------------------
        # Get random port
        # ----------------------------------------------------------------------------------------------------------------------------
        if fixed_port is None:
            proxy_port = int(proxy_min_port) + int(  # nosec B311 - non-cryptographic random port pick within an allowed proxy port range, for load spreading, not security-sensitive
                _facade.random() * (int(proxy_max_port) - int(proxy_min_port))
            )
        else:
            proxy_port = fixed_port

        proxies = _facade.make_proxies_dict(proxy_user, proxy_pass, proxy_server, proxy_port, proxy_type)

        proxy_key = joblib_hash(proxies)
        # ----------------------------------------------------------------------------------------------------------------------------
        # Check if it's allowed for immediate use by the policies
        # ----------------------------------------------------------------------------------------------------------------------------
        b_time_to_check_now = True
        for dict_to_check, min_interval in ((failed_dict, min_failed_idle_interval_minutes), (last_used_dict, min_idle_interval_minutes)):
            if dict_to_check is not None:
                if proxy_key in dict_to_check:
                    # A failed proxy has its own (much longer) cooldown; comparing it against
                    # min_idle_interval_minutes (default 0) handed a just-blocked exit IP straight back.
                    if (now_time - dict_to_check[proxy_key]).total_seconds() / 60 < min_interval:
                        if verbose:
                            logger.info("Skipping proxy %s:%s, touched recently", proxy_server, proxy_port)
                        b_time_to_check_now = False
                        break

        if b_time_to_check_now:
            if verbose:
                logger.info("Got new proxy: %s:%s", proxy_server, proxy_port)
            return proxies
        else:
            n = n + 1
            if n > warn_after_n_failures:
                # Unconditional, not gated on verbose=True: every real call site in this module
                # invokes get_new_smartproxy() (directly or via set_proxy()) with the default
                # verbose=False, so this was previously the ONLY diagnostic signal available for
                # a stuck wait -- and it never fired, making an indefinite hang undiagnosable
                # from logs alone (the only other symptom being a silent sleep(delay) every ~5
                # attempts with no output at all).
                logger.info(
                    "Could not get an untouched proxy%s after %d attempts, sleeping %s sec.",
                    "" if job_desc == "" else " for " + job_desc,
                    n,
                    delay,
                )
                if max_wait_seconds is not None:
                    if wait_started_at is None:
                        wait_started_at = datetime.now(timezone.utc).replace(tzinfo=None)  # naive-UTC, matches now_time's convention above
                    elif (datetime.now(timezone.utc).replace(tzinfo=None) - wait_started_at).total_seconds() > max_wait_seconds:  # naive-UTC, matches now_time's convention above
                        raise TimeoutError(
                            f"get_new_smartproxy: no eligible proxy found within {max_wait_seconds}s "
                            f"(pool [{proxy_min_port}, {proxy_max_port}] may be exhausted or too small "
                            f"for min_idle_interval_minutes={min_idle_interval_minutes})"
                        )
                _facade.sleep(delay)
                n = 0


def is_rotating_proxy(proxy_server: dict) -> Optional[bool]:
    """Return True if ``proxy_server`` config identifies Smartproxy's DC gateway with a fixed (non-range) port, i.e. a server-side auto-rotating proxy; otherwise None."""
    # {"PROXY_HOST": "gate.dc.smartproxy.com","PROXY_MIN_PORT": 20001,"PROXY_MAX_PORT": 37960}
    if proxy_server.get("PROXY_HOST", "").lower() == "gate.dc.smartproxy.com":
        if proxy_server.get("PROXY_MIN_PORT") == 20000:
            if proxy_server.get("PROXY_MAX_PORT") == 20000:
                return True
    return None
