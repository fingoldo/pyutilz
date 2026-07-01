# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed
ensure_installed("")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Optional, Tuple

from pyutilz.database import db
from pyutilz.system import system
from pyutilz.core import pythonlib
from pyutilz.web import web


class Container:
    def __init__(self,):
        self.node_id = None


pid = None
node_id = None
_container = Container()
m_app_name, m_scraper_name, m_version, m_ip = None, None, None, None


def register_scraper(scraper_name: str = None, version: str = None, app_name: str = None, ip: str = None) -> Optional[Any]:
    """Register the current process as a scraper node and emit a starting heartbeat.

    Resolves node identity from system info, upserts it into the ``nodes`` table,
    stores node stats, and sends an initial heartbeat.

    Args:
        scraper_name: logical name of the scraper.
        version: explicit version; when None, a content-hash-based version of the
            calling module is computed.
        app_name: application name; looked up from the call stack when None.
        ip: external IP; looked up from the stack / external service when None.

    Returns:
        The resolved ``node_id`` on success.

    Raises:
        Exception: propagates any error raised while gathering system info, so the
            caller learns the scraper did NOT register instead of failing silently.
    """
    global pid
    global _container, node_id, pid, m_app_name, m_scraper_name, m_version, m_ip

    import os
    import inspect
    from datetime import datetime

    pid = os.getpid()

    frame = None
    if version is None:
        # get content-based version of the calling file (hash)
        import hashlib
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        with open(module.__file__, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        version = f"{datetime.now().strftime('%Y.%m.%d')}.{file_hash}"
    if app_name is None:
        app_name = pythonlib.lookup_in_stack("app_name")

    if ip is None:
        ip = pythonlib.lookup_in_stack("ip")
    if ip is None:
        ip = web.get_external_ip()

    m_app_name, m_scraper_name, m_version, m_ip = app_name, scraper_name, version, ip

    if _container.node_id is None:
        try:
            info = system.get_system_info(only_stats=False)
        except Exception as e:
            # Fail loudly: without system info the scraper cannot register at all,
            # and silently returning None here left callers believing registration
            # succeeded. Re-raise so the failure is visible.
            logger.exception(e)
            raise
        else:
            fields = "host_name,os_machine_guid,os_serial"

            db.db_command("select", "nodes", where_fields=fields, returning="id", source=info, fetch_into=_container)
            if _container.node_id is None:
                db.db_command("insert", "nodes", set_fields=fields, returning="id", source=info, fetch_into=_container)
            if _container.node_id is None:
                return

            db.db_command(
                "insert",
                "nodes_info",
                set_fields=((set(info.keys()) - set(["host_name", "os_machine_guid", "os_serial"])) | set(["node"])),
                replace_values={"node": _container.node_id},
                returning="",
                source=info,
                jsonize=True,
            )
            logger.info("Registered as %s with node_id %s", m_scraper_name, _container.node_id)
            heartbeat_scraper(status="starting", ip=None)
            return _container.node_id


def get_heartbeat_sql(status: str = "ok", ip: str = None) -> Tuple[str, Optional[tuple]]:
    """Generate a parameterized heartbeat UPSERT SQL query.

    Args:
        status: scraper status to record (e.g. "ok", "starting").
        ip: IP to record; falls back to the module-level registered IP when None.

    Returns:
        A ``(sql, params)`` tuple for safe parameterized execution. When the node
        has not been registered yet, returns ``("", None)``.
    """
    if _container.node_id:
        sql = """
            INSERT INTO scrapers(node, pid, last_ping_at, version, name, status, ip, application)
            VALUES (%s, %s, (now() at time zone 'utc'), %s, %s, %s, %s, %s)
            ON CONFLICT(node, pid) DO UPDATE SET
                last_ping_at=excluded.last_ping_at,
                version=excluded.version,
                name=excluded.name,
                status=excluded.status,
                ip=excluded.ip,
                application=excluded.application
        """
        params = (
            _container.node_id,
            pid,
            m_version,
            m_scraper_name,
            status,
            ip if ip else m_ip,
            m_app_name
        )
        return (sql, params)
    else:
        return ("", None)


def heartbeat_scraper(status: str = "ok", ip: str = None) -> None:
    """Execute a heartbeat UPSERT for the currently registered scraper node.

    Args:
        status: scraper status to record.
        ip: IP to record; falls back to the registered IP when None.
    """
    db.safe_execute(get_heartbeat_sql(status, ip))
