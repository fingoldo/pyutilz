"""Property test: every SQL string the query builders can emit is parsed and checked, offline.

The SQL-builder family is this repo's highest-recurrence code surface -- it produced a High-or-worse
defect in three consecutive audit waves, and the 2026-09-02 wave's Critical was
``build_upsert_query`` emitting ``update <table> AS u set ... from changed_data as c`` with no WHERE
at all, which silently rewrites every row of the table. Hand-picked example queries kept missing
these because the defect always lived in a parameter COMBINATION nobody had written a test for, not
in a single flag.

So instead of examples, this sweeps a bounded cartesian product of the builders' own option space
and asserts structural invariants on whatever comes out, parsed with a real SQL parser (sqlglot,
postgres dialect):

1. **Parses.** Every statement is real SQL, and no statement degenerates into an INSERT with an
   empty column list or a SELECT with an empty projection -- the shape an unguarded history-table
   combination produced, which is invalid SQL that a lenient parser will still accept.
2. **No unbounded write.** Every ``UPDATE``/``DELETE`` anywhere in the tree carries a ``WHERE``.
   This is the Critical, stated directly.
3. **ON CONFLICT is targeted.** When the caller named ``conflict_fields``, the emitted
   ``ON CONFLICT`` names them -- a bare ``ON CONFLICT DO NOTHING`` matches ANY unique constraint on
   the table, swallowing violations the caller never asked to tolerate.
4. **Caller data stays parameterized.** Values reach the driver as bound parameters: the placeholder
   count matches the parameter count, and a hostile value never appears in the SQL text. Identifiers,
   which cannot be parameterized, must be rejected outright rather than interpolated.
5. **No opaque crash.** An unsupported combination raises a typed, named error (``ValueError`` /
   ``SQLValidationError`` / ``AssertionError``) or is refused with a logged error -- never a
   ``TypeError`` from three frames down that tells the caller nothing.

No database is touched: ``build_upsert_query`` is pure, and ``db_command``'s execution is captured at
the facade so only the string it built is examined.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Iterator, Optional

import pytest

sqlglot = pytest.importorskip("sqlglot", reason="sqlglot is a dev-only dependency (pyproject.toml [dev] extra, python>=3.9)")
from sqlglot import exp  # imported only after the importorskip above has passed

import pyutilz.database.db as db_facade
from pyutilz.database.db.execution import db_command
from pyutilz.database.db.upsert import build_upsert_query
from pyutilz.database.exceptions import SQLValidationError

DIALECT = "postgres"

# A builder is allowed to refuse a combination, but only by name: these say "this input is not
# supported" to the caller. Anything else (TypeError, AttributeError, KeyError, IndexError,
# UnboundLocalError) is an internal crash leaking out as the error message.
DECLARED_REFUSALS = (ValueError, SQLValidationError, AssertionError)

# Toy schema. Deliberately tiny: the combinatorics live in the flags, not in the column count.
TABLE = "orders"
HISTORY_TABLE = "orders_history"
FIELDS = ["id", "payload", "hash_col"]
CHECK_FIELD = "checked_at"
UPDATE_FIELD = "updated_at"

# A value that would be catastrophic if it were ever spliced into SQL text instead of bound.
HOSTILE_VALUE = "'); drop table orders; --"
HOSTILE_IDENTIFIERS = ["id; drop table orders", "id)", "1id", "", "id\n", "id --", "*"]


# --------------------------------------------------------------------------------------------------
# Statement normalization
# --------------------------------------------------------------------------------------------------


def split_statements(sql: str) -> list[str]:
    """Split ``sql`` on top-level semicolons, ignoring those inside parens or string literals."""
    statements: list[str] = []
    current: list[str] = []
    depth = 0
    in_string = False
    for ch in sql:
        if ch == "'":
            in_string = not in_string
        elif not in_string:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ";" and depth == 0:
                statements.append("".join(current))
                current = []
                continue
        current.append(ch)
    statements.append("".join(current))
    return [s.strip() for s in statements if s.strip()]


_TEMP_TABLE_WRAPPER = "create temp table "


def unwrap_temp_table(statement: str) -> str:
    """Strip a ``create temp table X on commit drop as ( ... )`` wrapper down to its inner query.

    sqlglot parses that Postgres-specific form as an opaque ``Command`` -- the whole body, including
    the UPDATE the WHERE-clause invariant exists to inspect, would be swallowed as a string. The
    builder emits this wrapper on its timestamp-update path, which is exactly the path that carried
    the Critical, so unwrapping it is not cosmetic.
    """
    lowered = statement.lower()
    if not lowered.startswith(_TEMP_TABLE_WRAPPER):
        return statement
    open_paren = statement.find("(")
    if open_paren < 0 or " as " not in lowered[:open_paren]:
        return statement
    inner = statement[open_paren + 1 :].rstrip()
    return inner[:-1] if inner.endswith(")") else inner


def parse_all(sql: str) -> list[exp.Expression]:
    """Every parsed statement in ``sql``, temp-table wrappers unwrapped."""
    trees: list[exp.Expression] = []
    for statement in split_statements(sql):
        trees.extend(t for t in sqlglot.parse(unwrap_temp_table(statement), dialect=DIALECT) if t is not None)
    return trees


# --------------------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------------------


def assert_sql_invariants(sql: str, context: str, *, expect_conflict_target: bool = False, param_count: Optional[int] = None) -> None:
    """Assert invariants 1-4 on one emitted query. ``context`` names the generating case."""
    try:
        trees = parse_all(sql)
    except sqlglot.errors.ParseError as e:
        raise AssertionError(f"{context}: emitted SQL does not parse ({e})\nSQL: {sql}") from e
    assert trees, f"{context}: builder emitted nothing parseable\nSQL: {sql}"

    for tree in trees:
        assert not isinstance(tree, exp.Command), f"{context}: statement did not parse as SQL, sqlglot fell back to a raw Command\nSQL: {sql}"

        for update in tree.find_all(exp.Update):
            assert update.args.get("where") is not None, f"{context}: UPDATE with no WHERE clause -- this rewrites EVERY row of the table\nSQL: {sql}"
        for delete in tree.find_all(exp.Delete):
            assert delete.args.get("where") is not None, f"{context}: DELETE with no WHERE clause -- this empties the table\nSQL: {sql}"

        for insert in tree.find_all(exp.Insert):
            schema = insert.this
            if isinstance(schema, exp.Schema):
                assert schema.expressions, f"{context}: INSERT with an empty column list -- invalid SQL\nSQL: {sql}"
        for select in tree.find_all(exp.Select):
            assert select.expressions, f"{context}: SELECT with an empty projection -- invalid SQL\nSQL: {sql}"

    if expect_conflict_target:
        conflicts = [oc for tree in trees for oc in tree.find_all(exp.OnConflict)]
        assert conflicts, f"{context}: conflict_fields were requested but no ON CONFLICT clause was emitted\nSQL: {sql}"
        for on_conflict in conflicts:
            keys = on_conflict.args.get("conflict_keys") or on_conflict.args.get("constraint")
            assert keys, f"{context}: bare ON CONFLICT with no target -- matches ANY unique constraint, not the requested one\nSQL: {sql}"

    if param_count is not None:
        assert sql.count("%s") == param_count, f"{context}: {sql.count('%s')} placeholder(s) for {param_count} bound parameter(s)\nSQL: {sql}"
    assert HOSTILE_VALUE not in sql, f"{context}: a caller-supplied VALUE was interpolated into the SQL text instead of being bound\nSQL: {sql}"


# --------------------------------------------------------------------------------------------------
# build_upsert_query
# --------------------------------------------------------------------------------------------------


def upsert_param_grid() -> Iterator[dict]:
    """The bounded cartesian product of ``build_upsert_query``'s option space over the toy schema.

    Each dimension carries its meaningfully-different states (absent / empty / populated), which is
    where the recurring defects lived -- every one of them needed two options set at once.
    """
    dimensions: dict[str, tuple] = {
        "conflict_fields": (None, ["id"], []),
        "fields_types": (None, {"payload": "jsonb"}),
        "skip_fields": (None, ["payload"]),
        "timestamp_check_fields": (None, [CHECK_FIELD]),
        "timestamp_update_fields": (None, [UPDATE_FIELD]),
        "on_conflict_update_fields": (None, ["payload"]),
        "custom_onconflict": (None, "on conflict (id) do nothing "),
        "history_table_name": (None, HISTORY_TABLE),
        "history_fields": (None, ["payload"]),
        "hash_fields": ("", ["hash_col"]),
    }
    names = list(dimensions)
    for combination in itertools.product(*dimensions.values()):
        yield dict(zip(names, combination))


def build_upsert_or_refusal(builder: Callable[..., str], kwargs: dict) -> Optional[str]:
    """``builder(**kwargs)``'s SQL, or None when it declared the combination unsupported.

    Re-raises anything that is not a declared refusal, so an opaque internal ``TypeError`` fails the
    sweep by name instead of being mistaken for validation (invariant 5).
    """
    try:
        return builder(fields_names=list(FIELDS), table_name=TABLE, **kwargs)
    except DECLARED_REFUSALS:
        return None


def check_upsert_grid(builder: Callable[..., str]) -> int:
    """Run every grid case through ``builder`` and its invariants; return how many built a query.

    Takes the builder as an argument so the same sweep can be pointed at an alternative
    implementation (a pre-fix copy, when demonstrating that this test actually catches the defects
    it was written for) without touching the shipped module.
    """
    built = 0
    for kwargs in upsert_param_grid():
        sql = build_upsert_or_refusal(builder, kwargs)
        if sql is None:
            continue
        built += 1
        expect_conflict_target = bool(kwargs["conflict_fields"]) and kwargs["custom_onconflict"] is None
        assert_sql_invariants(sql, context=f"build_upsert_query({kwargs})", expect_conflict_target=expect_conflict_target, param_count=1)
    return built


def test_build_upsert_query_option_space_emits_valid_bounded_sql() -> None:
    built = check_upsert_grid(build_upsert_query)
    # Guards against the sweep silently degenerating into "every combination was refused", which
    # would make the invariants above vacuous while the test still went green.
    assert built > 100, f"only {built} of the grid's combinations produced a query -- the sweep is no longer exercising the builder"


@pytest.mark.parametrize("bad_identifier", HOSTILE_IDENTIFIERS)
def test_build_upsert_query_rejects_hostile_identifiers(bad_identifier: str) -> None:
    """Identifiers cannot be bound as parameters, so the only safe handling is refusal.

    Checked on every identifier-shaped parameter, because the recurring failure has been ONE
    parameter left out of the validation loop while its siblings were covered.
    """
    for param in ("table_name", "conflict_fields", "on_conflict_update_fields", "history_table_name", "history_fields", "skip_fields", "hash_fields"):
        if param == "history_table_name" and bad_identifier == "":
            # "" is this parameter's documented "no history table" sentinel (same as None), not an
            # identifier that reaches the SQL text -- refusing it would be a behavior change, not a fix.
            continue
        kwargs: dict[str, Any] = {"fields_names": list(FIELDS), "table_name": TABLE}
        if param == "table_name":
            kwargs["table_name"] = bad_identifier
        elif param == "history_fields":
            kwargs["history_table_name"] = HISTORY_TABLE
            kwargs["history_fields"] = [bad_identifier]
        elif param == "history_table_name":
            kwargs["history_table_name"] = bad_identifier
            kwargs["history_fields"] = ["payload"]
        else:
            kwargs[param] = [bad_identifier]
        with pytest.raises(DECLARED_REFUSALS):
            sql = build_upsert_query(**kwargs)
            pytest.fail(f"{param}={bad_identifier!r} was accepted and reached the SQL text:\n{sql}")


def test_build_upsert_query_rejects_unlisted_field_types() -> None:
    """``fields_types`` values are spliced verbatim as a ``::cast``, so the type name is an
    identifier-grade input -- and its guard must not be an ``assert``, which ``python -O`` removes."""
    with pytest.raises(ValueError):
        build_upsert_query(fields_names=list(FIELDS), table_name=TABLE, fields_types={"payload": "text; drop table orders"})


# --------------------------------------------------------------------------------------------------
# db_command
# --------------------------------------------------------------------------------------------------


def db_command_param_grid() -> Iterator[dict]:
    """``db_command``'s option space. ``None`` / ``[]`` / ``""`` are included on every parameter
    that has ever produced a crash or a malformed query when left out."""
    dimensions: dict[str, tuple] = {
        "mode": ("select", "insert", "update", "delete"),
        "where_fields": (None, [], ["id"], "id,payload"),
        "set_fields": (None, [], ["payload"], "payload"),
        "returning": ("*", None, "", "id"),
        "jsonize": (True, False),
    }
    names = list(dimensions)
    for combination in itertools.product(*dimensions.values()):
        yield dict(zip(names, combination))


def check_db_command_grid(command: Callable[..., Any]) -> int:
    """Run every grid case through ``command`` with execution captured; return how many built SQL."""
    captured: dict = {}

    def capture(sql, params=None, *args, **kwargs):
        captured["sql"] = sql
        captured["params"] = params
        return []

    original = db_facade.safe_execute
    db_facade.safe_execute = capture
    built = 0
    try:
        for kwargs in db_command_param_grid():
            captured.clear()
            source = {"id": HOSTILE_VALUE, "payload": HOSTILE_VALUE, "checked_at": HOSTILE_VALUE}
            try:
                command(table_name=TABLE, source=source, **kwargs)
            except DECLARED_REFUSALS:
                continue
            if "sql" not in captured:
                # Refused with a logged error and an early return -- a named refusal, not a crash.
                continue
            built += 1
            params = captured["params"] or []
            assert_sql_invariants(captured["sql"], context=f"db_command({kwargs})", param_count=len(params))
    finally:
        db_facade.safe_execute = original
    return built


def test_db_command_option_space_emits_valid_parameterized_sql() -> None:
    built = check_db_command_grid(db_command)
    assert built > 20, f"only {built} of the grid's combinations produced a query -- the sweep is no longer exercising db_command"


@pytest.mark.parametrize("mode", ["insert", "update"])
def test_db_command_refuses_empty_set_fields_by_name(mode: str) -> None:
    """Both write modes build their column list from ``set_fields``; neither may crash on its absence.

    Surfaced by the sweep above: the ``insert`` half of this guard was added on its own, so
    ``mode="update", set_fields=None`` still died with ``TypeError: 'NoneType' object is not
    iterable`` two frames down, and ``set_fields=[]`` still built ``update t set  where id=%s``.
    """

    def capture(sql, params=None, *args, **kwargs):
        pytest.fail(f"db_command(mode={mode!r}) built a query from an empty set_fields:\n{sql}")

    original = db_facade.safe_execute
    db_facade.safe_execute = capture
    try:
        for set_fields in (None, []):
            assert db_command(mode=mode, table_name=TABLE, where_fields=["id"], set_fields=set_fields, source={"id": 1}) is None
    finally:
        db_facade.safe_execute = original


@pytest.mark.parametrize("bad_identifier", HOSTILE_IDENTIFIERS)
def test_db_command_rejects_hostile_identifiers(bad_identifier: str) -> None:
    def capture(sql, params=None, *args, **kwargs):
        pytest.fail(f"a hostile identifier reached execution:\n{sql}")

    original = db_facade.safe_execute
    db_facade.safe_execute = capture
    try:
        with pytest.raises(DECLARED_REFUSALS):
            db_command(mode="select", table_name=bad_identifier, where_fields=["id"], source={"id": 1})
        with pytest.raises(DECLARED_REFUSALS):
            db_command(mode="select", table_name=TABLE, where_fields=[bad_identifier], source={"id": 1})
    finally:
        db_facade.safe_execute = original
