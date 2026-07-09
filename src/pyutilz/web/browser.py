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

# ----------------------------------------------------------------------------------------------------------------------------
# Typing
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Dict, Optional, Tuple

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

# ***************************************************************************************************************************
# INITS
# ***************************************************************************************************************************

last_session_updated_at = None

version_main = None
login, pwd = None, None
browser, headers, proxy_server, target, home_page, user_agent = None, None, None, None, None, None
TheCookies = None
data_dir = None  # "chrome-data"
logout_signs = "Sign-In"
successful_login_signs = ()  # Define as empty tuple, to be overridden by user
login_input_name = "email"
password_input_name = "password"  # nosec B105 - this is the HTML form field NAME/selector used to locate the password input element, not a literal credential value (the actual secret is the module-level `pwd` variable, supplied by the caller)
use_real_useragent = True
undetectable = False
find_executable = False
use_subprocess = False
required_cookies: Tuple[Any, ...] = tuple()
fixed_cookies: Dict[str, Any] = {}
basic_headers = {"accept-encoding": "gzip,deflate", "accept-language": "en-US,en;q=0.9", "accept": "*/*"}
headers = basic_headers

def find_element_by_xpath(browser:Any,query:str)->object:
    """Locates an element by XPath, falling back to the deprecated Selenium API for older driver versions."""
    from selenium.webdriver.common.by import By

    try:
        res = browser.find_element(By.XPATH, query)
    except Exception:
        res = browser.find_element_by_xpath(query)

    return res

def find_element_by_name(browser:Any,query:str)->object:
    """Locates an element by its `name` attribute, falling back to the deprecated Selenium API for older driver versions."""
    from selenium.webdriver.common.by import By

    try:
        res = browser.find_element(By.NAME, query)
    except Exception:
        res = browser.find_element_by_name(query)

    return res

def find_element_by_tag_name(browser:Any,query:str)->object:
    """Locates an element by its tag name, falling back to the deprecated Selenium API for older driver versions."""
    from selenium.webdriver.common.by import By

    try:
        res = browser.find_element(By.TAG_NAME, query)
    except Exception:
        res = browser.find_element_by_tag_name(query)

    return res

def init(**params) -> None:
    """Sets module-level configuration variables (e.g. target, home_page, login, pwd) from keyword arguments."""

    globals().update(params)

def close_browser():
    """Closes the active Selenium browser instance if any, swallowing errors, and clears the module-level `browser` reference."""
    global browser
    try:
        if browser is not None:
            browser.close()
    except Exception as e:  # nosec B110 - best-effort cleanup on a browser handle that may already be dead/closed; the function unconditionally sets browser=None on the next line regardless
        logger.debug("Ignoring error while closing browser: %s", e)
    browser = None

def browser_get(path:str)->None:
    """Navigates the module-level `browser` to `path`, retrying once after a short sleep on transient loading-status errors."""
    try:
        if browser is None:
            raise ValueError("pyutilz.web.browser.browser is not initialized; call start_selenium() first")
        browser.get(path)
    except Exception as e:
        if ("cannot determine loading status" in str(e)) or ("unexpected command response" in str(e)):
            # logger.warning(e)
            sleep(2)
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

def start_selenium() -> object:
    """Launches a Chrome Selenium webdriver (undetected or standard), applying module-level config (proxy, user agent, data dir), and stores it in the module-level `browser`."""
    import zipfile

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
        if len(proxy_server.get("PROXY_PASS", "")) > 0:
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

            background_js = """
            var config = {
                    mode: "fixed_servers",
                    rules: {
                    singleProxy: {
                        scheme: "http",
                        host: "%s",
                        port: parseInt(%s)
                    },
                    bypassList: ["localhost"]
                    }
                };

            chrome.proxy.settings.set({value: config, scope: "regular"}, function() {});

            function callbackFn(details) {
                return {
                    authCredentials: {
                        username: "%s",
                        password: "%s"
                    }
                };
            }

            chrome.webRequest.onAuthRequired.addListener(
                        callbackFn,
                        {urls: ["<all_urls>"]},
                        ['blocking']
            );
            """ % (proxy_server["PROXY_HOST"], proxy_server["PROXY_PORT"], proxy_server["PROXY_USER"], proxy_server["PROXY_PASS"])
            pluginfile = "proxy_auth_plugin.zip"

            with zipfile.ZipFile(pluginfile, "w") as zp:
                zp.writestr("manifest.json", manifest_json)
                zp.writestr("background.js", background_js)
            options.add_extension(pluginfile)
        else:
            options.add_argument(f"--proxy-server={proxy_server['PROXY_HOST']}:{proxy_server['PROXY_PORT']}")  # example: "localhost:8118"

    # if not undetectable:
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9222")

    kwargs["options"] = options
    # if not data_dir:
    #    path = os.path.dirname(os.path.curdir)
    #    path=os.path.join(path, 'chromedriver')
    #    kwargs["path"]=path

    browser = webdriver.Chrome(**kwargs)
    return browser


def ensure_session_is_valid(interval_minutes: Optional[int] = 10) -> None:
    """Re-runs LoginAndGetCookies() if the session was never updated or is older than `interval_minutes`."""
    global last_session_updated_at
    do_update = False
    if last_session_updated_at is None:
        do_update = True
    else:
        now_time = datetime.now(timezone.utc)
        if (now_time - last_session_updated_at).total_seconds() / 60 >= interval_minutes:
            do_update = True
    if do_update:
        LoginAndGetCookies()
        last_session_updated_at = datetime.now(timezone.utc)


def LoginAndGetCookies(default_headers: bool = True, seconds_to_sleep_on_error: int = 60, restart_on_no_cookie=False) -> bool:
    """
    Ensures Selenium is started
    Logins, if not logged in already (detected by opening a home page)
    Extracts (or updates) desired cookies from Selenium browser instance into global headers dict.
    """
    global browser, TheCookies, headers
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
                    browser = None
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
            elem_login.send_keys(login)
            pythonlib.imitate_delay(min_delay_seconds=2, max_delay_seconds=5, b_force=True)
            elem_login.send_keys(Ret)
        except Exception as e:  # nosec B110 - best-effort login-by-name-field attempt; the code below explicitly falls back to find_element_by_xpath, and if elem_login stays None it is logged and reported as an error a few lines down
            logger.debug("find_element_by_name login attempt failed, will try xpath fallback: %s", e)
        if elem_login is None:
            try:
                elem_login = find_element_by_xpath(browser, "//div[text()='" + login.lower() + "']")
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
            elem_pwd.send_keys(pwd)
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
                        logger.warning("Trying to restart Selenium...")
                        try:
                            browser.close()
                        except Exception as e:  # nosec B110 - best-effort cleanup before forcing a Selenium restart; the next line unconditionally sets browser=None regardless of whether close() succeeded
                            logger.debug("Ignoring error while closing browser before restart: %s", e)
                        browser = None
                    return LoginAndGetCookies(default_headers=default_headers)
                else:
                    cook = cook.get("value")
                    cookies_vals[c] = cook

        TheCookies = ""
        for cookie, val in cookies_vals.items():
            TheCookies = TheCookies + cookie + "=" + str(val) + "; "

        if default_headers:
            headers = basic_headers
            if use_real_useragent:
                headers["user-agent"] = browser.execute_script("return navigator.userAgent;")

            headers["cookie"] = TheCookies
            if "oauth2_global_js_token" in cookies_vals:
                headers["authorization"] = "Bearer " + str(cookies_vals["oauth2_global_js_token"])
    return res
