# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .python import ensure_installed
ensure_installed("")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from . import db, system, python, web


class Container:
    def __init__(self,):
        self.node_id = None


pid = None
node_id = None
self = Container()
m_app_name, m_scraper_name, m_version, m_ip = None, None, None, None


def register_scraper(scraper_name=None, version=None, app_name=None, ip=None):
    global pid
    global self, node_id, pid, m_app_name, m_scraper_name, m_version, m_ip

    import os
    import inspect
    from datetime import datetime

    pid = os.getpid()

    frame = None
    if version is None:
        # get modification date of the calling file
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        version = datetime.fromtimestamp(os.path.getmtime((module.__file__))).strftime("%Y.%m.%d")
    if app_name is None:
        app_name = python.lookup_in_stack("app_name")

    if ip is None:
        ip = python.lookup_in_stack("ip")
    if ip is None:
        ip = web.get_external_ip()

    m_app_name, m_scraper_name, m_version, m_ip = app_name, scraper_name, version, ip

    if self.node_id is None:
        try:
            info = system.get_system_info(only_stats=False)
        except Exception as e:
            logger.exception(e)
        else:
            fields = "host_name,os_machine_guid,os_serial"

            nodes = db.db_command("select", "nodes", where_fields=fields, returning="id", source=info, fetch_into=self)
            if self.node_id is None:
                nodes = db.db_command("insert", "nodes", set_fields=fields, returning="id", source=info, fetch_into=self)
            if self.node_id is None:
                return

            db.db_command(
                "insert",
                "nodes_info",
                set_fields=((set(info.keys()) - set(["host_name", "os_machine_guid", "os_serial"])) | set(["node"])),
                replace_values={"node": self.node_id},
                returning="",
                source=info,
                jsonize=True,
            )
            logger.info("Registered as %s with node_id %s" % (m_scraper_name, self.node_id))
            heartbeat_scraper(status="starting", ip=None)
            return self.node_id


def get_heartbeat_sql(status="ok", ip=None):
    if self.node_id:
        return (
            "insert into scrapers(node,pid,last_ping_at,version,name,status,ip,application) values ("
            + ",".join(
                [
                    str(self.node_id),
                    str(pid),
                    "(now() at time zone 'utc')",
                    db.nu(m_version),
                    db.nu(m_scraper_name),
                    db.nu(status),
                    db.nu((ip if ip else m_ip)),
                    db.nu(m_app_name),
                ]
            )
            + ") on conflict(node,pid) do update set last_ping_at=excluded.last_ping_at,version=excluded.version,name=excluded.name,status=excluded.status,ip=excluded.ip,application=excluded.application"
        )
    else:
        return ""


def heartbeat_scraper(status="ok", ip=None):
    db.safe_execute(get_heartbeat_sql(status, ip))