"""Regular-jobs (``regular_jobs`` table) scheduling helpers for ``pyutilz.database.db``.

A tiny cooperative scheduler on top of one Postgres table: create it, poll for a job that is due
(``for update skip locked``), heartbeat progress, and finalize the result.
"""

from ._common import sql, Json

# See the PROJECT IDIOM comment in connection.py.
import pyutilz.database.db as _facade


def regjobs_create_table(table_name: str = "regular_jobs"):
    """Creates the jobs table backing this scheduler: one row per job, holding its schedule
    (``interval_minutes``, ``singleton``, ``ping_timeout_minutes``) and its run state (owner node,
    start/ping/finish timestamps and the jsonb results)."""
    _facade.safe_execute(sql.SQL("""
create table {table_name} (
	name citext primary key,
	description text,
	interval_minutes int,
	singleton boolean default true, --do not start new instance if prev one is still running (unless ping has timed out)
	ping_timeout_minutes int,

	taken_by text, --node ip

	started_at timestamp without time zone,

	last_ping_at timestamp without time zone,
	last_result jsonb,

	finished_at timestamp without time zone,
	result jsonb

	)
    """).format(table_name=sql.Identifier(table_name)))


def regjobs_poll(job_name: str, taken_by: str, table_name: str = "regular_jobs"):
    """Claims the named job for ``taken_by`` if it is due, returning its name (a one-row result) or an
    empty result when it is not.

    A job is due when it never ran, or when its interval has elapsed since it last finished; for a
    singleton, a run that started but never finished only becomes claimable once its ping has timed
    out. The row is locked with ``for update skip locked``, so concurrent pollers never take the same
    job, and claiming stamps the owner and a fresh ``started_at``.
    """
    return _facade.safe_execute(
        sql.SQL("""
        with base as (select name from {table_name} where name=%(job_name)s
            and
                (
                    (started_at is null) --never started before
                        or
                    (
                        (finished_at is null or finished_at <(now() at time zone 'utc'-interval_minutes * interval '1 minute'))
                            and
                        (finished_at>started_at)
                            and
                        singleton=true
                    ) -- not finished long enough after successful finishing for a singleton
                        or
                    (
                        (finished_at is null or finished_at <(now() at time zone 'utc'-interval_minutes * interval '1 minute'))
                            and
                        (finished_at is null or finished_at<started_at)
                            and
                        (ping_timeout_minutes is not null and (last_ping_at is null or last_ping_at<(now() at time zone 'utc'-ping_timeout_minutes * interval '1 minute')))
                            and
                        singleton=true
                    ) -- not finished long enough WITHOUT successful finishing yet AND with a missed ping, for a singleton


                        or
                    (
                        (finished_at is null or finished_at <(now() at time zone 'utc'-interval_minutes * interval '1 minute'))
                            and
                        singleton=false
                    ) -- not finished long enough for NOT a singleton

                )
                limit 1 for update skip locked)

            update {table_name} rj set taken_by=%(taken_by)s,started_at=now() at time zone 'utc' from base where rj.name=base.name returning rj.name
    """).format(table_name=sql.Identifier(table_name)),
        {"job_name": job_name, "taken_by": taken_by},
    )


def _regjobs_update(job_name: str, result: dict, table_name: str, *, ts_column: str, result_column: str):
    """Shared UPDATE shape behind regjobs_progress/regjobs_finalize: stamp ``ts_column`` with the
    current UTC time and store ``result`` into ``result_column`` for the row named ``job_name``."""
    return _facade.safe_execute(
        sql.SQL("""
        update {table_name} set {ts_column}=now() at time zone 'utc',{result_column}=%(result)s where name=%(job_name)s
    """).format(table_name=sql.Identifier(table_name), ts_column=sql.Identifier(ts_column), result_column=sql.Identifier(result_column)),
        # psycopg2 cannot adapt a bare dict; Json() is the documented adapter for jsonb columns.
        {"job_name": job_name, "result": Json(result) if isinstance(result, (dict, list)) else result},
    )


def regjobs_progress(job_name: str, result: dict, table_name: str = "regular_jobs"):
    """Heartbeats a running job: refreshes ``last_ping_at`` and stores ``result`` as its interim
    ``last_result``, which is what keeps a singleton's claim from being judged timed out."""
    return _regjobs_update(job_name, result, table_name, ts_column="last_ping_at", result_column="last_result")


def regjobs_finalize(job_name: str, result: dict, table_name: str = "regular_jobs"):
    """Marks a job as completed: stamps ``finished_at`` and stores ``result`` as its final result, from
    which the next ``regjobs_poll`` measures the job's interval."""
    return _regjobs_update(job_name, result, table_name, ts_column="finished_at", result_column="result")
