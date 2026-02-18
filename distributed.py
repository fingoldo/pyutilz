# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed
ensure_installed("")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any

from . import db, system, pythonlib, web


class Container:
    def __init__(self,):
        self.node_id = None


pid = None
node_id = None
_container = Container()
m_app_name, m_scraper_name, m_version, m_ip = None, None, None, None


def register_scraper(scraper_name=None, version=None, app_name=None, ip=None):
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
            logger.exception(e)
        else:
            fields = "host_name,os_machine_guid,os_serial"

            nodes = db.db_command("select", "nodes", where_fields=fields, returning="id", source=info, fetch_into=_container)
            if _container.node_id is None:
                nodes = db.db_command("insert", "nodes", set_fields=fields, returning="id", source=info, fetch_into=_container)
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
            logger.info("Registered as %s with node_id %s" % (m_scraper_name, _container.node_id))
            heartbeat_scraper(status="starting", ip=None)
            return _container.node_id


def get_heartbeat_sql(status="ok", ip=None):
    """Generate parameterized heartbeat SQL query.

    Returns: tuple of (sql_query, parameters) for safe execution
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


def heartbeat_scraper(status="ok", ip=None):
    db.safe_execute(get_heartbeat_sql(status, ip))
