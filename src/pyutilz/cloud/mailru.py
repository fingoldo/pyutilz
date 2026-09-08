"""Read files out of a mail.ru Cloud PUBLIC folder, which is not the directory its URL looks like.

``https://cloud.mail.ru/public/<a>/<b>`` is a web page. Joining a file name onto it --
``.../public/<a>/<b>/data.tar.gz`` -- returns HTML rather than the file, so the obvious "treat the public
link as a base URL" approach downloads a page and then fails whatever checksum the caller applies. That is
at least a loud failure, but a confusing one, and it is the reason this module exists rather than a comment
somewhere saying "public links do not work like that".

The real object is served by a CDN host that mail.ru issues from a dispatcher endpoint and ROTATES, so it
cannot be hardcoded: resolve it once per process and build ``<host>/<weblink>/<name>``. The CDN also 403s a
request with no ``User-Agent``, which is why one is set - this reads a folder its owner deliberately
published, not anything gated.

Every request goes through `pyutilz.web.url_guard.urlopen_checked`, which validates the scheme on the URL
AND on each redirect hop - not ceremony here: a public download redirects once to a signed CDN URL, so the
hop is exactly where an unchecked `urlopen` would follow whatever it was handed.

Ported from a downstream project that needed a credential-free mirror for ~2.4 GB of build inputs: a public
cloud folder needs no account, no keys and no SDK on the machine restoring from it, which for a bootstrap
step is worth more than the niceties of a real object store.

    from pyutilz.cloud.mailru import list_public_folder, public_file_url, open_public_file

    for entry in list_public_folder("https://cloud.mail.ru/public/oxLS/4ZnDN8UJw"):
        print(entry["name"], entry["size"])
    with open_public_file("https://cloud.mail.ru/public/oxLS/4ZnDN8UJw", "manifest.json") as response:
        payload = response.read()
"""

from __future__ import annotations

import json
import logging
import urllib.request
from functools import lru_cache
from typing import Any, Dict, Iterator, List

from pyutilz.web.url_guard import urlopen_checked

logger = logging.getLogger(__name__)

PUBLIC_PREFIX = "cloud.mail.ru/public/"
DISPATCHER_URL = "https://cloud.mail.ru/api/v2/dispatcher"
FOLDER_API_URL = "https://cloud.mail.ru/api/v2/folder"

# Their CDN answers 403 to a request that sends no agent at all. Not an attempt to look like a browser
# beyond that: the folder is public and its owner published it for this.
USER_AGENT = "Mozilla/5.0"

# The folder listing is paged; this is the API's own maximum per call.
_PAGE_LIMIT = 500


def is_public_folder_url(url: str) -> bool:
    """Whether `url` is a mail.ru Cloud public-folder link, and so needs this module rather than a plain GET."""
    return PUBLIC_PREFIX in url


def weblink_of(folder_url: str) -> str:
    """The `<a>/<b>` weblink id inside a public-folder URL, which every API call below is keyed on."""
    if not is_public_folder_url(folder_url):
        raise ValueError(f"not a mail.ru public folder URL: {folder_url!r}")
    return folder_url.split(PUBLIC_PREFIX, 1)[1].strip("/")


def _get_json(url: str, timeout: int = 60) -> Dict[str, Any]:
    """One JSON GET with the agent set, since every endpoint here needs it."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen_checked(request, timeout=timeout) as response:
        return dict(json.loads(response.read()))


@lru_cache(maxsize=1)
def download_host() -> str:
    """The CDN host mail.ru is currently handing out for public-link downloads, without a trailing slash.

    Cached for the process: the value rotates between runs but not within one, and every file in a restore
    would otherwise pay a dispatcher round trip. Call `download_host.cache_clear()` if a long-lived process
    starts seeing failures that look like an expired host.
    """
    hosts = _get_json(DISPATCHER_URL).get("body", {}).get("weblink_get") or []
    if not hosts:
        raise RuntimeError("mail.ru dispatcher returned no weblink_get host - cannot resolve public-folder downloads")
    host = str(hosts[0]["url"]).rstrip("/")
    logger.debug("mail.ru public-download host resolved to %s", host)
    return host


def public_file_url(folder_url: str, name: str) -> str:
    """The URL a named file in a public folder actually lives at."""
    return f"{download_host()}/{weblink_of(folder_url)}/{name}"


def open_public_file(folder_url: str, name: str, timeout: int = 600) -> Any:
    """Open one file from a public folder for streaming. The caller closes it; the response redirects once.

    Returned as-is (an ``addinfourl``) so a caller can `shutil.copyfileobj` a multi-gigabyte object rather
    than reading it into memory.
    """
    request = urllib.request.Request(public_file_url(folder_url, name), headers={"User-Agent": USER_AGENT})
    return urlopen_checked(request, timeout=timeout)


def list_public_folder(folder_url: str, timeout: int = 60) -> List[Dict[str, Any]]:
    """Every entry in a public folder, paged through, as the API's own dicts (`name`, `size`, `type`, ...).

    Useful on its own for checking a mirror is complete before trusting it, which is the shape of question
    that otherwise gets answered by opening the link in a browser and counting.
    """
    return list(_iter_public_folder(folder_url, timeout))


def _iter_public_folder(folder_url: str, timeout: int) -> Iterator[Dict[str, Any]]:
    """The paging loop behind `list_public_folder`, kept separate so the public function returns a real list."""
    weblink = weblink_of(folder_url)
    offset = 0
    while True:
        body = _get_json(f"{FOLDER_API_URL}?weblink={weblink}&limit={_PAGE_LIMIT}&offset={offset}", timeout).get("body", {})
        page = body.get("list") or []
        if not page:
            return
        yield from page
        offset += len(page)
        counts = body.get("count") or {}
        if offset >= int(counts.get("files", 0)) + int(counts.get("folders", 0)):
            return
