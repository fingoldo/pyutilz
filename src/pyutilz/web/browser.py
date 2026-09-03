"""Selenium/undetected_chromedriver browser automation helpers: driver startup, element lookup, login, and cookie extraction."""

# ***************************************************************************************************************************
# IMPORTS
# ***************************************************************************************************************************

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

import os
import json

# ----------------------------------------------------------------------------------------------------------------------------
# Typing
# ----------------------------------------------------------------------------------------------------------------------------

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

if TYPE_CHECKING:
    # Type-checking-only: selenium is an optional [web] extra and every runtime import of it in
    # this module is deliberately lazy (see the comment block below), so importing it for real
    # here would break that contract.
    # Chrome's driver, not the generic remote one: start_selenium() always builds either
    # webdriver.Chrome or undetected_chromedriver.Chrome (a subclass), and Chrome-only methods
    # such as execute_cdp_cmd() are used below.
    from selenium.webdriver.chrome.webdriver import WebDriver
    from selenium.webdriver.remote.webelement import WebElement

# ----------------------------------------------------------------------------------------------------------------------------
# Selenium connectivity
# ----------------------------------------------------------------------------------------------------------------------------
#
# This module's other selenium/undetected_chromedriver imports (start_selenium(), Keys) are all
# lazy, inside the functions that need them, so importing pyutilz.web.browser itself never forces
# a hard selenium dependency -- callers who never touch a browser shouldn't need it installed
# (selenium lives under pyutilz's optional [web] extra). The `By` import below used to be the one
# module-level exception, which broke that contract: any transitive import of pyutilz.web (e.g.
# pyutilz.system.distributed's `from pyutilz.web import web`, which pulls in the whole `web`
# package's __init__) raised ModuleNotFoundError for selenium even when the caller never touches
# a browser -- found 2026-07-09 via mlframe's CI failing ~1300 unrelated tests on exactly this
# chain. Moved to a lazy import inside each of the three functions that actually use it.

# ----------------------------------------------------------------------------------------------------------------------------
# Utilz
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core import pythonlib

from datetime import datetime, timezone
from time import sleep
from random import random

# ***************************************************************************************************************************
# INITS
# ***************************************************************************************************************************

last_session_updated_at: Optional[datetime] = None

version_main = None
login: Optional[str] = None
pwd: Optional[str] = None
# Annotated individually rather than as one six-way tuple unpack: without annotations mypy infers
# the type of each as exactly `None`, which makes every `if browser is not None:` guard in this
# module provably dead to the checker AND removes all checking of selenium/dict attribute access
# on them (a typo in a WebDriver method name would be invisible).
browser: Optional["WebDriver"] = None
headers: Optional[dict] = None
proxy_server: Optional[dict] = None
target: Optional[str] = None
home_page: Optional[str] = None
user_agent: Optional[str] = None
TheCookies: Optional[str] = None
data_dir: Optional[str] = None  # "chrome-data"
logout_signs = "Sign-In"
successful_login_signs = ()  # Define as empty tuple, to be overridden by user
login_input_name = "email"
password_input_name = "password"  # nosec B105 - this is the HTML form field NAME/selector used to locate the password input element, not a literal credential value (the actual secret is the module-level `pwd` variable, supplied by the caller)
use_real_useragent = True
# None = decide automatically (see _should_disable_sandbox): only a uid-0 process, where Chrome
# refuses to start otherwise, gets --no-sandbox. Set True/False via init() to force either way.
no_sandbox: Optional[bool] = None
undetectable = False
find_executable = False
use_subprocess = False
required_cookies: Tuple[Any, ...] = tuple()
fixed_cookies: Dict[str, Any] = {}
basic_headers = {"accept-encoding": "gzip,deflate", "accept-language": "en-US,en;q=0.9", "accept": "*/*"}
# dict(), not an alias: `headers = basic_headers` made browser.headers IS browser.basic_headers at
# import time and kept it so whenever LoginAndGetCookies(default_headers=False) is used or is never
# reached, so any caller mutating browser.headers silently rewrote the documented neutral
# basic_headers default for every other consumer in the process.
headers = dict(basic_headers)

def find_element_by_xpath(browser: Any, query: str) -> "WebElement":
    """Locates an element by XPath."""
    from selenium.webdriver.common.by import By

    # Regression fix: the previous `except Exception: browser.find_element_by_xpath(query)`
    # fallback called a method Selenium 4.0+ (this project's own declared floor, pyproject.toml
    # `selenium>=4.0`) removed entirely, not merely deprecated -- confirmed empirically against
    # selenium==4.39.0: WebDriver has no find_element_by_xpath attribute at all. The intended
    # "element genuinely not present" case (find_element() raising NoSuchElementException) was
    # masked by a confusing, unrelated AttributeError from calling a nonexistent method, instead
    # of the real, actionable exception propagating to the caller.
    return cast("WebElement", browser.find_element(By.XPATH, query))

def find_element_by_name(browser: Any, query: str) -> "WebElement":
    """Locates an element by its `name` attribute."""
    from selenium.webdriver.common.by import By

    return cast("WebElement", browser.find_element(By.NAME, query))

def find_element_by_tag_name(browser: Any, query: str) -> "WebElement":
    """Locates an element by its tag name."""
    from selenium.webdriver.common.by import By

    return cast("WebElement", browser.find_element(By.TAG_NAME, query))

def init(**params) -> None:
    """Sets module-level configuration variables (e.g. target, home_page, login, pwd) from keyword arguments."""

    globals().update(params)

def close_browser():
    """Closes the active Selenium browser instance if any, swallowing errors, and clears the module-level `browser` reference."""
    global browser
    try:
        if browser is not None:
            # .quit() (not .close()): Selenium's .close() only closes the current window/tab and
            # leaves the WebDriver session (and, in undetectable=True + use_subprocess=True mode,
            # its own OS subprocess) running. Every restart cycle that called .close() then
            # start_selenium() again spawned a brand-new chromedriver/undetected_chromedriver
            # process while the old one was left running (or zombied), accumulating memory and
            # process-table entries without bound over a long-running scraper's lifetime.
            browser.quit()
    except Exception as e:  # nosec B110 - best-effort cleanup on a browser handle that may already be dead/closed; the function unconditionally sets browser=None on the next line regardless
        logger.debug("Ignoring error while closing browser: %s", e)
    browser = None

def browser_get(path:str)->None:
    """Navigates the module-level `browser` to `path`, retrying once after a short sleep on transient loading-status errors."""
    if browser is None:
        raise ValueError("pyutilz.web.browser.browser is not initialized; call start_selenium() first")
    try:
        browser.get(path)
    except Exception as e:
        if ("cannot determine loading status" in str(e)) or ("unexpected command response" in str(e)):
            # Regression fix: this previously only slept and returned -- browser.get(path) was
            # never actually called a second time, despite the docstring's "retrying once"
            # claim. A caller relying on browser_get() succeeding (e.g. LoginAndGetCookies())
            # would proceed to inspect browser.title/attempt login while the browser was still
            # on the PREVIOUS page (navigation to `path` never completed), acting on stale
            # content instead of the page it thinks it just loaded.
            sleep(2)
            browser.get(path)
        else:
            raise(e)

def find_chrome_executable():
    """fix find_chrome_executable for x86 Windows"""
    candidates = set()
    for item in map(os.environ.get, ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")):
        if item:  # it happens to be None
            for subitem in (
                "Google/Chrome/Application",
                "Google/Chrome Beta/Application",
                "Google/Chrome Canary/Application",
            ):
                candidates.add(os.sep.join((item, subitem, "chrome.exe")))
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return os.path.normpath(candidate)
    return None

def _build_proxy_extension_js(proxy_server: dict) -> str:
    """Render the proxy-auth extension's background.js for ``proxy_server``.

    A module-level function, not an inline literal inside ``start_selenium()``: the escaping
    below is the security-relevant part of this file and needs a seam a test can reach without
    starting a browser."""
    return """
    var config = {
            mode: "fixed_servers",
            rules: {
            singleProxy: {
                scheme: "http",
                host: %s,
                port: %d
            },
            bypassList: ["localhost"]
            }
        };

    chrome.proxy.settings.set({value: config, scope: "regular"}, function() {});

    function callbackFn(details) {
        return {
            authCredentials: {
                username: %s,
                password: %s
            }
        };
    }

    chrome.webRequest.onAuthRequired.addListener(
                callbackFn,
                {urls: ["<all_urls>"]},
                ['blocking']
    );
    """ % (
        # json.dumps, not raw %s inside a JS string literal: these four values arrive from a
        # rotating-proxy vendor's API response, a config file or a DB row, and were spliced
        # unescaped into the source of an extension holding `webRequest`+`webRequestBlocking`
        # over `<all_urls>` -- the most privileged JS context in the browser. A password
        # containing a plain `"` or `\` (legal, and common in generated credentials) silently
        # broke the script so `onAuthRequired` never registered and every request went out
        # unauthenticated (unexplained 407s); a hostile one (`x"};alert(1);var y={"a":"`)
        # executed arbitrary JS able to read and rewrite every request of the authenticated
        # session. json.dumps emits a correctly escaped JS/JSON string literal for any input,
        # so it supplies its own quotes -- the `"%s"` wrappers are gone from the template
        # above. The port goes through int() + %d so a non-numeric port fails loudly here
        # instead of being spliced into parseInt().
        json.dumps(str(proxy_server["PROXY_HOST"])),
        int(proxy_server["PROXY_PORT"]),
        json.dumps(str(proxy_server["PROXY_USER"])),
        json.dumps(str(proxy_server["PROXY_PASS"])),
    )


def _should_disable_sandbox() -> bool:
    """Whether to pass Chrome ``--no-sandbox``.

    This library exists to point a browser at arbitrary third-party sites, so the renderer sandbox is
    the one thing standing between a renderer exploit (a hostile ad, a compromised CDN script) and
    code execution as the scraper's OS user -- the same user holding the S3 credentials, the DB DSN
    and the proxy passwords this module handles a few lines up. It used to be passed unconditionally,
    on every platform, with no way to turn it back on (the guard was a commented-out
    ``# if not undetectable:``), while the very next lines hardened the CDP surface -- inconsistent
    within one function.

    It is now passed only where Chrome genuinely refuses to start without it: running as uid 0, which
    is the containerised-as-root deployment the flag was really there for. Anyone else (every Windows
    and macOS caller, and any Linux caller running as a normal user) gets the sandbox back. An
    operator who needs it anyway sets the module-level ``no_sandbox`` via ``init(no_sandbox=True)`` --
    the correct container fix remains a non-root user plus a seccomp profile, not a disabled sandbox.
    """
    if no_sandbox is not None:
        return bool(no_sandbox)
    # os.geteuid is POSIX-only (absent on Windows), so presence-test the attribute rather than
    # branching on platform.system() -- Windows has no uid-0 process to detect in the first place.
    if hasattr(os, "geteuid"):
        return bool(os.geteuid() == 0)  # type: ignore[attr-defined]  # POSIX-only, guarded by the hasattr above
    return False


def start_selenium() -> "WebDriver":
    """Launches a Chrome Selenium webdriver (undetected or standard), applying module-level config (proxy, user agent, data dir), and stores it in the module-level `browser`."""
    import zipfile
    import tempfile
    import atexit

    global browser

    # if "PROGRAMFILES(X86)" not in os.environ: os.environ["PROGRAMFILES(X86)"] = ""

    logger.info("Starting Selenium for %s", target)
    kwargs: Dict[str, Any] = {}
    if undetectable:
        logger.info("Undetectable mode")
        try:
            import undetected_chromedriver as webdriver
            try:
                webdriver.install()
            except Exception as e:  # nosec B110 - undetected_chromedriver.install() is a best-effort driver-binary fetch; if it fails (e.g. already installed, offline), webdriver.Chrome() below will still attempt to use whatever driver is on PATH
                logger.debug("undetected_chromedriver install() failed, continuing: %s", e)
            options = webdriver.ChromeOptions()
            kwargs["version_main"] = version_main
            kwargs["use_subprocess"] = use_subprocess

            if find_executable:
                try:
                    webdriver.find_chrome_executable = find_chrome_executable
                except Exception as e:  # nosec B110 - optional monkeypatch of the driver's executable-finder only applied when find_executable=True; if the attribute assignment fails, the driver's own default lookup still applies
                    logger.debug("Could not patch find_chrome_executable, using driver default: %s", e)
        except Exception as e:
            logger.exception(e)
            from selenium import webdriver
            options = webdriver.ChromeOptions()
    else:
        logger.info("Standard mode")
        from selenium import webdriver

        options = webdriver.ChromeOptions()  # webdriver.chrome.options.Options()

    if user_agent:
        options.add_argument(f"--user-agent={user_agent}")

    if data_dir:
        options.add_argument(f"--user-data-dir={data_dir}")

    if proxy_server:
        # Truthiness, not len(): a present-but-None PROXY_PASS (the JSON way of saying
        # "unauthenticated proxy") made len() raise TypeError.
        if proxy_server.get("PROXY_PASS"):
            # Validated up front (matches the unauthenticated branch's guard below) -- bare
            # proxy_server["PROXY_HOST"]/["PROXY_PORT"]/["PROXY_USER"] indexing a few lines down
            # would otherwise raise an unguarded KeyError on a missing key instead of this clear
            # error naming exactly what's missing.
            missing = [k for k in ("PROXY_HOST", "PROXY_PORT", "PROXY_USER") if not proxy_server.get(k)]
            if missing:
                raise ValueError(f"start_selenium: proxy_server is missing {missing}: {proxy_server!r}")
            manifest_json = """
            {
                "version": "1.0.0",
                "manifest_version": 2,
                "name": "Chrome Proxy",
                "permissions": [
                    "proxy",
                    "tabs",
                    "unlimitedStorage",
                    "storage",
                    "<all_urls>",
                    "webRequest",
                    "webRequestBlocking"
                ],
                "background": {
                    "scripts": ["background.js"]
                },
                "minimum_chrome_version":"22.0.0"
            }
            """

            background_js = _build_proxy_extension_js(proxy_server)
            # Regression fix: previously written to a hardcoded "proxy_auth_plugin.zip" in the
            # process's current working directory, with no cleanup anywhere in this file -- the
            # plaintext proxy password (embedded in background.js above) sat on disk
            # indefinitely in a commonly backed-up / accidentally `git add .`-ed location,
            # readable by any other local process/user on a shared host, and two concurrent
            # scraping workers (this package explicitly supports parallelism) could race on the
            # same filename. tempfile.mkstemp gives an unpredictable path with restrictive
            # default permissions (further tightened via chmod below); atexit is a defensive
            # backstop in case the immediate cleanup below is ever skipped (e.g. an exception
            # between here and webdriver.Chrome() starting) -- the PRIMARY cleanup is the
            # explicit os.remove() right after Chrome has loaded the extension, a few lines down,
            # since relying on atexit alone would leave the file on disk for a long-running
            # scraper's entire process lifetime.
            plugin_fd, pluginfile = tempfile.mkstemp(suffix=".zip", prefix="pyutilz_proxy_auth_")
            os.close(plugin_fd)
            os.chmod(pluginfile, 0o600)
            def _remove_plugin_file(p: str = pluginfile) -> None:
                """Delete the temp proxy-auth extension zip at interpreter exit, if it still exists."""
                if os.path.exists(p):
                    os.remove(p)

            atexit.register(_remove_plugin_file)

            with zipfile.ZipFile(pluginfile, "w") as zp:
                zp.writestr("manifest.json", manifest_json)
                zp.writestr("background.js", background_js)
            options.add_extension(pluginfile)
        else:
            pluginfile = None
            if not proxy_server.get("PROXY_HOST") or not proxy_server.get("PROXY_PORT"):
                raise ValueError(f"start_selenium: proxy_server is missing PROXY_HOST/PROXY_PORT: {proxy_server!r}")
            options.add_argument(f"--proxy-server={proxy_server['PROXY_HOST']}:{proxy_server['PROXY_PORT']}")  # example: "localhost:8118"
    else:
        pluginfile = None

    if _should_disable_sandbox():
        options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # Regression fix: a fixed, well-known CDP debug port (9222) with no address restriction let
    # ANY other local process/user on a shared host connect to http://localhost:9222/json and
    # attach to this already-authenticated Selenium session (reading cookies, injecting JS),
    # and collided if two start_selenium() calls ran concurrently on one machine. port=0 lets
    # Chrome pick an OS-assigned ephemeral port (nothing in this codebase reads it back --
    # Selenium's own control channel is negotiated separately by the driver, not via this CDP
    # port); the explicit address bind additionally blocks any non-loopback interface.
    options.add_argument("--remote-debugging-port=0")
    options.add_argument("--remote-debugging-address=127.0.0.1")

    kwargs["options"] = options
    # if not data_dir:
    #    path = os.path.dirname(os.path.curdir)
    #    path=os.path.join(path, 'chromedriver')
    #    kwargs["path"]=path

    browser = webdriver.Chrome(**kwargs)

    if pluginfile is not None:
        # Primary cleanup (atexit above is only the defensive backstop): Chrome has now started
        # and read the extension zip during driver startup, so the temp file's job is done.
        try:
            os.remove(pluginfile)
        except OSError as e:  # nosec B110 - best-effort cleanup; the atexit-registered fallback above still covers this file if removal fails here (e.g. a transient Windows sharing violation)
            logger.debug("Could not remove temporary proxy-auth plugin file %s: %s", pluginfile, e)

    return browser


def ensure_session_is_valid(interval_minutes: Optional[int] = 10) -> None:
    """Re-runs LoginAndGetCookies() if the session was never updated or is older than `interval_minutes`."""
    global last_session_updated_at
    do_update = False
    if last_session_updated_at is None:
        do_update = True
    elif interval_minutes is None:
        # Regression fix: interval_minutes is declared Optional[int] specifically to let a
        # caller mean "always re-validate" -- previously `float >= None` raised TypeError the
        # moment a session already existed, an unhandled crash on a value the type annotation
        # explicitly invites.
        do_update = True
    else:
        now_time = datetime.now(timezone.utc)
        if (now_time - last_session_updated_at).total_seconds() / 60 >= interval_minutes:
            do_update = True
    if do_update:
        LoginAndGetCookies()
        last_session_updated_at = datetime.now(timezone.utc)


def LoginAndGetCookies(
    default_headers: bool = True,
    seconds_to_sleep_on_error: int = 60,
    restart_on_no_cookie=False,
    _restart_attempt: int = 0,
    _max_restart_attempts: int = 50,
) -> bool:
    """
    Ensures Selenium is started
    Logins, if not logged in already (detected by opening a home page)
    Extracts (or updates) desired cookies from Selenium browser instance into global headers dict.

    ``_restart_attempt``/``_max_restart_attempts`` are internal bookkeeping for the
    restart_on_no_cookie retry below -- not meant to be passed by external callers.
    """
    global browser, TheCookies, headers
    _session_refresh_attempts = 0
    while True:
        if browser is None:
            browser = start_selenium()
            break
        else:
            try:
                browser.refresh()
                browser.execute_cdp_cmd(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {"source": """
                    Object.defineProperty(navigator, 'webdriver', {
                      get: () => undefined
                    })
                  """},
                )
            except Exception as e:
                logger.exception(e)
                if "window was already closed" in str(e) or "window already closed" in str(e) or "chrome not reachable" in str(e):
                    logger.info("Restarting webdriver")
                    # close_browser(), not `browser = None`: on the "window was already closed"
                    # half of this condition the chromedriver (and, with use_subprocess=True,
                    # Chrome) process is typically still alive, so dropping the handle orphaned one
                    # per restart -- the same leak the sibling browser_get restart path below and
                    # the one at the end of this function already close by quitting first.
                    close_browser()
                else:
                    # Regression fix: any OTHER exception (e.g. InvalidSessionIdException, a
                    # stale-element/timeout error, a non-English driver message) previously fell
                    # through here with no break/sleep/re-raise, looping straight back to the top
                    # of `while True` and immediately retrying the identical failing call --
                    # an unbounded, zero-delay busy loop pinning a CPU core at 100% forever
                    # (with logger.exception re-logging the same traceback every iteration).
                    _session_refresh_attempts += 1
                    if _session_refresh_attempts >= 20:
                        raise RuntimeError(
                            f"LoginAndGetCookies: session-refresh failed {_session_refresh_attempts} times "
                            f"in a row with an unrecognized error, giving up: {e}"
                        ) from e
                    sleep(min(2**_session_refresh_attempts, 30) * random())  # nosec B311 - jittered backoff on an unrecognized session-refresh error, not security-sensitive
            else:
                break
    if home_page is None:
        raise ValueError("pyutilz.web.browser.home_page must be set (e.g. `browser.home_page = url`) before calling LoginAndGetCookies()")
    while True:
        try:
            browser_get(home_page)
            pythonlib.imitate_delay(min_delay_seconds=5, max_delay_seconds=10, b_force=True)
        except Exception as e:  # noqa: PERF203 -- per-attempt retry loop; the try/except IS the retry mechanism
            ste = str(e)
            if "not reachable" in ste or "no such window" in ste:
                logger.warning("Restarting Selenium instance")
                # On the "no such window" half the driver process is typically still alive;
                # overwriting the handle directly orphaned a chromedriver on every restart.
                close_browser()
                browser = start_selenium()
            else:
                logger.exception(e)
                return False
        else:
            break
    res = False

    # print(browser.title)
    from selenium.webdriver.common.keys import Keys

    Ret = Keys.RETURN

    if pythonlib.anyof_elements_in_string(("Cloudflare",), browser.title):
        logger.warning("Ddos or captcha protection on %s. Waiting for operator to solve it...", target)
        sleep(120)

    if pythonlib.anyof_elements_in_string(logout_signs, browser.title):
        pythonlib.imitate_delay(min_delay_seconds=2, max_delay_seconds=5, b_force=True)
        elem_login = None
        try:
            elem_login = find_element_by_name(browser, login_input_name)
            elem_login.send_keys(Keys.CONTROL, "a")
            elem_login.send_keys(Keys.DELETE)
            # `login`/`pwd` are module-level Optionals set by the caller; sending None types the literal
            # string "None" into the field, which fails the login with no diagnosis.
            elem_login.send_keys(login or "")
            pythonlib.imitate_delay(min_delay_seconds=2, max_delay_seconds=5, b_force=True)
            elem_login.send_keys(Ret)
        except Exception as e:  # nosec B110 - best-effort login-by-name-field attempt; the code below explicitly falls back to find_element_by_xpath, and if elem_login stays None it is logged and reported as an error a few lines down
            logger.debug("find_element_by_name login attempt failed, will try xpath fallback: %s", e)
        if elem_login is None:
            try:
                # `login` is a module-level global the caller sets via init(); str() keeps the xpath
                # well-formed (and the lookup merely failing) instead of raising AttributeError out of
                # the fallback when it was never configured.
                elem_login = find_element_by_xpath(browser, "//div[text()='" + str(login).lower() + "']")
            except Exception as e:  # nosec B110 - best-effort xpath fallback for locating the login element; if elem_login is still None afterward it is explicitly checked and logged as an error two lines below
                logger.debug("find_element_by_xpath login fallback failed: %s", e)
        if elem_login is None:
            logger.error("Could not login to %s: elem_login %s not located.", target, login_input_name)
            return False

        pythonlib.imitate_delay(min_delay_seconds=2, max_delay_seconds=5, b_force=True)
        elem_pwd = None
        try:
            elem_pwd = find_element_by_name(browser, password_input_name)
            elem_pwd.send_keys(Keys.CONTROL, "a")
            elem_pwd.send_keys(Keys.DELETE)
            elem_pwd.send_keys(pwd or "")
        except Exception as e:  # nosec B110 - best-effort password-field lookup/entry; if elem_pwd stays None it is explicitly checked and logged as an error two lines below
            logger.debug("find_element_by_name password attempt failed: %s", e)
        if elem_pwd is None:
            logger.error("Could not login to %s: elem_pwd %s not located.", target, password_input_name)
            return False

        pythonlib.imitate_delay(min_delay_seconds=0, max_delay_seconds=3, b_force=True)
        elem_pwd.send_keys(Ret)

        pythonlib.imitate_delay(min_delay_seconds=5, max_delay_seconds=15, b_force=True)

        title = browser.title
        if not pythonlib.anyof_elements_in_string(successful_login_signs, title):
            logger.critical("Can't login to %s,got page %s", target, title)
        else:
            logger.info("Logged in to %s", target)
            res = True
    else:
        if pythonlib.anyof_elements_in_string(successful_login_signs, browser.title):
            res = True

    if res:
        cookies_vals = fixed_cookies.copy()
        if len(required_cookies) > 0:
            for c in required_cookies:
                cook = browser.get_cookie(c)
                if cook is None:
                    logger.error(
                        "Unexpected: required cookie %s is missing when getting cookies from %s. Sleeping %s seconds...",
                        c,
                        target,
                        seconds_to_sleep_on_error,
                    )
                    sleep(seconds_to_sleep_on_error)
                    if restart_on_no_cookie:
                        # Regression fix: this recursive retry previously (a) called
                        # LoginAndGetCookies(default_headers=default_headers) only, silently
                        # dropping the caller's own seconds_to_sleep_on_error/restart_on_no_cookie
                        # overrides back to their defaults on every subsequent retry level, and
                        # (b) had no depth limit, so a persistently-failing cookie check would
                        # eventually hit Python's own opaque RecursionError after ~1000 frames
                        # (many hours later, given the seconds_to_sleep_on_error sleep each level)
                        # instead of failing with a clear, actionable error.
                        if _restart_attempt >= _max_restart_attempts:
                            raise RuntimeError(
                                f"LoginAndGetCookies: exceeded {_max_restart_attempts} restart attempts "
                                f"without obtaining required cookie {c!r} for {target}"
                            )
                        logger.warning("Trying to restart Selenium...")
                        try:
                            browser.quit()
                        except Exception as e:  # nosec B110 - best-effort cleanup before forcing a Selenium restart; the next line unconditionally sets browser=None regardless of whether quit() succeeded
                            logger.debug("Ignoring error while closing browser before restart: %s", e)
                        browser = None
                    return LoginAndGetCookies(
                        default_headers=default_headers,
                        seconds_to_sleep_on_error=seconds_to_sleep_on_error,
                        restart_on_no_cookie=restart_on_no_cookie,
                        _restart_attempt=_restart_attempt + 1,
                        _max_restart_attempts=_max_restart_attempts,
                    )
                else:
                    cook = cook.get("value")
                    cookies_vals[c] = cook

        TheCookies = ""
        for cookie, val in cookies_vals.items():
            TheCookies = TheCookies + cookie + "=" + str(val) + "; "

        if default_headers:
            # A copy, not the shared global: mutating basic_headers in place left the documented
            # neutral default header set permanently carrying the last session's cookie and
            # Bearer token, and leaked them to every other consumer of it.
            headers = dict(basic_headers)
            # Drop any authorization inherited from a previous login: without this, a session
            # that yields no oauth2_global_js_token keeps sending the PREVIOUS account's token.
            headers.pop("authorization", None)
            if use_real_useragent:
                headers["user-agent"] = browser.execute_script("return navigator.userAgent;")

            headers["cookie"] = TheCookies
            if "oauth2_global_js_token" in cookies_vals:
                headers["authorization"] = "Bearer " + str(cookies_vals["oauth2_global_js_token"])
    return res
