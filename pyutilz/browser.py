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

from .pythonlib import ensure_installed
ensure_installed("undetected_chromedriver selenium")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import os

# ----------------------------------------------------------------------------------------------------------------------------
# Typing
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

# ----------------------------------------------------------------------------------------------------------------------------
# Selenium connectivity
# ----------------------------------------------------------------------------------------------------------------------------

from selenium.webdriver.common.by import By

# ----------------------------------------------------------------------------------------------------------------------------
# Utilz
# ----------------------------------------------------------------------------------------------------------------------------

from . import pythonlib

from datetime import datetime, timedelta
from time import sleep
import time

# ***************************************************************************************************************************
# INITS
# ***************************************************************************************************************************

last_session_updated_at = None

version_main = None
login, pwd = None, None
browser, headers, proxy_server, target, home_page, user_agent = None, None, None, None, None, None
data_dir = None#"chrome-data"
logout_signs = "Sign-In"
login_input_name = "email"
password_input_name = "password"
use_real_useragent = True
undetectable = False
find_executable = False
use_subprocess = False
required_cookies = tuple()
fixed_cookies = {}
basic_headers = {"accept-encoding": "gzip,deflate", "accept-language": "en-US,en;q=0.9", "accept": "*/*"}
headers = basic_headers

def find_element_by_xpath(browser:object,query:str)->object:
    try:
        res=browser.find_element(By.XPATH, query)
    except:
        res=browser.find_element_by_xpath(query)
    
    return res
    
def find_element_by_name(browser:object,query:str)->object:
    try:
        res=browser.find_element(By.NAME, query)
    except:
        res=browser.find_element_by_name(query)
    
    return res
    
def find_element_by_tag_name(browser:object,query:str)->object:
    try:
        res=browser.find_element(By.TAG_NAME, query)
    except:
        res=browser.find_element_by_tag_name(query)
    
    return res
    
def init(**params) -> None:

    globals().update(params)

def close_browser():
    global browser
    try:
        browser.close()
    except Exception as e: pass
    browser=None

def browser_get(path:str)->None:
    try:
        browser.get(path)
    except Exception as e:
        if ('cannot determine loading status' in str(e)) or ('unexpected command response' in str(e)):
            #logger.warning(e)
            sleep(2)
        else:
            raise(e)

def find_chrome_executable():
    """fix find_chrome_executable for x86 Windows"""
    candidates = set()
    for item in map(
        os.environ.get, ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")
    ):
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
    import zipfile    
    import os
    
    global browser

    #if "PROGRAMFILES(X86)" not in os.environ: os.environ["PROGRAMFILES(X86)"] = ""
    
    logger.info(f"Starting Selenium for {target}")
    kwargs={}
    if undetectable:
        logger.info(f"Undetectable mode")
        try:
            import undetected_chromedriver as webdriver
            try:
                webdriver.install()
            except: pass
            options = webdriver.ChromeOptions()
            kwargs["version_main"]=version_main
            kwargs["use_subprocess"]=use_subprocess
            
            if find_executable:
                try:
                    webdriver.find_chrome_executable = find_chrome_executable
                except: pass
        except Exception as e:
            logger.exception(e)
            from selenium import webdriver
            options = webdriver.ChromeOptions()
    else:
        logger.info(f"Standard mode")
        from selenium import webdriver

        options = webdriver.ChromeOptions()#webdriver.chrome.options.Options()

    if user_agent:
        options.add_argument(f"--user-agent={user_agent}")
        
    if data_dir:
        options.add_argument(f"--user-data-dir={data_dir}")
        
    if proxy_server:
        if len(proxy_server.get('PROXY_PASS',''))>0:
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
            """ % (proxy_server['PROXY_HOST'], proxy_server['PROXY_PORT'], proxy_server['PROXY_USER'], proxy_server['PROXY_PASS'])    
            pluginfile = 'proxy_auth_plugin.zip'

            with zipfile.ZipFile(pluginfile, 'w') as zp:
                zp.writestr("manifest.json", manifest_json)
                zp.writestr("background.js", background_js)
            options.add_extension(pluginfile)
        else:
            options.add_argument(f"--proxy-server={proxy_server['PROXY_HOST']}:{proxy_server['PROXY_PORT']}")  # example: "localhost:8118"

    #if not undetectable:
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=9222")
    
    kwargs["options"]=options
    #if not data_dir:       
    #    path = os.path.dirname(os.path.curdir)
    #    path=os.path.join(path, 'chromedriver')
    #    kwargs["path"]=path
    
    browser = webdriver.Chrome(**kwargs)
    return browser


def ensure_session_is_valid(interval_minutes: Optional[int] = 10) -> None:
    global last_session_updated_at
    do_update = False
    if last_session_updated_at is None:
        do_update = True
    else:
        now_time = datetime.utcnow()
        if (now_time - last_session_updated_at).total_seconds() / 60 >= interval_minutes:
            do_update = True
    if do_update:
        LoginAndGetCookies()
        last_session_updated_at = datetime.utcnow()


def LoginAndGetCookies(default_headers:bool=True,seconds_to_sleep_on_error:int=60,restart_on_no_cookie=False) -> bool:
    global browser, TheCookies, headers
    """
        Ensures Selenium is started
        Logins, if not logged in already (detected by opening a home page)
        Extracts (or updates) desired cookies from Selenium browser instance into global headers dict.
    """
    while True:
        if browser is None:
            browser = start_selenium()
            break
        else:
            try:
                browser.refresh()        
                browser.execute_cdp_cmd(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {
                        "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                      get: () => undefined
                    })
                  """
                    },
                )
            except Exception as e:
                logger.exception(e)
                if 'window was already closed' in str(e) or  'window already closed' in str(e) or 'chrome not reachable' in str(e):
                    logger.info('Restarting webdriver')
                    browser=None
            else:
                break
    while True:
        try:
            browser_get(home_page)
            python.imitate_delay(min_delay_seconds=5, max_delay_seconds=10, b_force=True)
        except Exception as e:
            ste = str(e)
            if "not reachable" in ste or "no such window" in ste:
                logger.warning("Restarting Selenium instance")
                browser = start_selenium()
            else:
                logger.exception(e)
                return
        else:
            break
    res = False

    # print(browser.title)
    from selenium.webdriver.common.keys import Keys

    Ret = Keys.RETURN

    if python.anyof_elements_in_string(("Cloudflare",), browser.title):
        logger.warning(f"Ddos or captcha protection on {target}. Waiting for operator to solve it...")
        sleep(120)

    if python.anyof_elements_in_string(logout_signs, browser.title):
        python.imitate_delay(min_delay_seconds=2, max_delay_seconds=5, b_force=True)
        elem_login = None
        try:
            elem_login = find_element_by_name(browser, login_input_name)            
            elem_login.send_keys(Keys.CONTROL, 'a');elem_login.send_keys(Keys.DELETE)
            elem_login.send_keys(login)
            python.imitate_delay(min_delay_seconds=2, max_delay_seconds=5, b_force=True)
            elem_login.send_keys(Ret)
        except:
            pass
        if elem_login is None:
            try:
                elem_login = find_element_by_xpath(browser,"//div[text()='" + login.lower() + "']")
            except:
                pass
        if elem_login is None:
            logger.error(f"Could not login to {target}: elem_login {login_input_name} not located.")
            return

        python.imitate_delay(min_delay_seconds=2, max_delay_seconds=5, b_force=True)
        elem_pwd = None
        try:
            elem_pwd = find_element_by_name(browser, password_input_name)
            elem_pwd.send_keys(Keys.CONTROL, 'a');elem_pwd.send_keys(Keys.DELETE)
            elem_pwd.send_keys(pwd)
        except:
            pass
        if elem_pwd is None:
            logger.error(f"Could not login to {target}: elem_pwd {password_input_name} not located.")
            return

        python.imitate_delay(min_delay_seconds=0, max_delay_seconds=3, b_force=True)
        elem_pwd.send_keys(Ret)

        python.imitate_delay(min_delay_seconds=5, max_delay_seconds=15, b_force=True)

        title = browser.title
        if not python.anyof_elements_in_string(successful_login_signs, title):
            logger.critical(f"Can't login to {target},got page {title}")
        else:
            logger.info(f"Logged in to {target}")
            res = True
    else:
        if python.anyof_elements_in_string(successful_login_signs, browser.title):
            res = True

    if res:
        cookies_vals = fixed_cookies.copy()
        if len(required_cookies) > 0:
            for c in required_cookies:
                cook = browser.get_cookie(c)
                if cook is None:
                    logger.error(f"Unexpected: required cookie {c} is missing when getting cookies from {target}. Sleeping {seconds_to_sleep_on_error} seconds...")
                    sleep(seconds_to_sleep_on_error)
                    if restart_on_no_cookie:
                        logger.warning(f"Trying to restart Selenium...")
                        try:
                            browser.close()
                        except: pass
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