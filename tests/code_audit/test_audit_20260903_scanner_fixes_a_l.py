"""Scanner tests for audit_20260903_scanner_fixes_a_l, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_bare_except,
    scan_credential_shaped_log_args,
    scan_import_cycles,
)
from pyutilz.dev.code_audit.constructor_param_overwritten import scan_constructor_param_overwritten
from pyutilz.dev.code_audit.asymmetric_except_siblings import scan_asymmetric_except_siblings
from pyutilz.dev.code_audit.effect_flag_outside_its_effect import scan_effect_flag_outside_its_effect
from pyutilz.dev.code_audit.guard_decidable_from_constants import scan_guard_decidable_from_constants
from pyutilz.dev.code_audit.count_then_fetch_same_table import scan_count_then_fetch_same_table
from pyutilz.dev.code_audit.accumulator_helper_bypassed import scan_accumulator_helper_bypassed
from pyutilz.dev.code_audit.column_no_write_path import scan_column_no_write_path
from pyutilz.dev.code_audit.comment_names_missing_symbol import (
    scan_comment_cites_absolute_line,
    scan_comment_names_missing_symbol,
)
from pyutilz.dev.code_audit.docstring_numbers_moved_to_config import scan_docstring_numbers_moved_to_config

from ._helpers import HELPER, _write

# ==== audit 2026-09-03 / 07-domain-core-dev-system: scanner logic fixes (a-l) ==================
#
# One positive (must flag) and one negative (must not) per repaired rule. Kept together at the end
# of the file rather than spread through it because the whole set lands in a single pass and each
# case names the finding it pins.


def test_bare_except_does_not_flag_an_immediate_bare_reraise(tmp_path: Path):
    """F66: the docstring promises a bare re-raise is not flagged; that exemption applied only to
    the `except BaseException:` spelling."""
    _write(tmp_path, "a.py", "try:\n    pass\nexcept:\n    raise\n")
    assert scan_bare_except(tmp_path) == []
    _write(tmp_path, "b.py", "try:\n    pass\nexcept:\n    pass\n")
    assert [f.line for f in scan_bare_except(tmp_path)] == [3]


def test_broad_except_accepts_a_handler_returning_a_structured_error(tmp_path: Path):
    """F65: `return {"ok": False, "error": str(e)}` IS the escalation the detail text asks for."""
    from pyutilz.dev.code_audit.broad_except import scan_broad_except_swallows

    _write(tmp_path, "a.py", 'def f():\n    try:\n        g()\n    except Exception as e:\n        return {"ok": False, "error": str(e)}\n')
    assert scan_broad_except_swallows(tmp_path) == []
    _write(tmp_path, "b.py", "def f():\n    try:\n        g()\n    except Exception:\n        return None\n")
    assert [f.line for f in scan_broad_except_swallows(tmp_path)] == [4]


def test_additive_epsilon_scans_module_scope_and_annotated_and_chained_bindings(tmp_path: Path):
    """F16 / F78 / F79: module scope was skipped whenever the file had a function; AnnAssign and
    chained targets were not recorded; two padded divisions on one line collapsed to one."""
    from pyutilz.dev.code_audit.additive_epsilon_denominator import scan_additive_epsilon_denominator

    _write(tmp_path, "m.py", "def f():\n    return 1\n\nRATIO = 5.0 / (SCALE + 1e-12)\n")
    assert [f.line for f in scan_additive_epsilon_denominator(tmp_path)] == [4]

    _write(tmp_path, "m.py", "def f(d, x):\n    denom: float = d + 1e-12\n    return x / denom\n")
    assert len(scan_additive_epsilon_denominator(tmp_path)) == 1

    _write(tmp_path, "m.py", "def f(d, x):\n    a = denom = d + 1e-12\n    return x / denom\n")
    assert len(scan_additive_epsilon_denominator(tmp_path)) == 1

    _write(tmp_path, "m.py", "def f(x, d, e):\n    return (x / (d + 1e-12), x / (e + 1e-12))\n")
    assert len(scan_additive_epsilon_denominator(tmp_path)) == 2

    _write(tmp_path, "m.py", "def f(x, d):\n    return x / d\n")
    assert scan_additive_epsilon_denominator(tmp_path) == []


def test_additive_epsilon_detail_has_no_stray_comma(tmp_path: Path):
    """F153: rendered as `...an epsilon-padded sum,. Adding a constant...`."""
    from pyutilz.dev.code_audit.additive_epsilon_denominator import scan_additive_epsilon_denominator

    _write(tmp_path, "m.py", "def f(d, x):\n    denom = d + 1e-12\n    return x / denom\n")
    assert ",." not in scan_additive_epsilon_denominator(tmp_path)[0].detail


def test_effect_flag_does_not_treat_an_integer_one_as_a_true_flag(tmp_path: Path):
    """F17 / F163: `1 == True` in Python, so a plain counter assignment was reported; and the
    annotated spelling of a real True flag was missed."""
    _write(tmp_path, "m.py", "def f(rows, counts):\n    if rows:\n        write_parquet(rows)\n    counts['rows'] = 1\n")
    assert scan_effect_flag_outside_its_effect(tmp_path) == []
    _write(tmp_path, "m.py", "def f(rows, counts):\n    if rows:\n        write_parquet(rows)\n    counts['rows'] = True\n")
    assert [f.line for f in scan_effect_flag_outside_its_effect(tmp_path)] == [4]
    _write(tmp_path, "m.py", "def f(rows, ok):\n    if rows:\n        write_parquet(rows)\n    ok['rows']: bool = True\n")
    assert [f.line for f in scan_effect_flag_outside_its_effect(tmp_path)] == [4]


def test_count_then_fetch_dedupes_and_reads_both_orders_and_outer_pagination(tmp_path: Path):
    """F70 / F71 / F162: one finding per site, either statement order, and a subquery LIMIT does
    not bound the outer statement."""
    _write(tmp_path, "n.py", "def outer(cur):\n    def inner(cur):\n        cur.execute('SELECT COUNT(*) FROM jobs')\n        cur.execute('SELECT id FROM jobs')\n    return inner\n")
    assert len(scan_count_then_fetch_same_table(tmp_path)) == 1

    _write(tmp_path, "n.py", "def f(cur):\n    cur.execute('SELECT id FROM jobs')\n    cur.execute('SELECT COUNT(*) FROM jobs')\n")
    assert len(scan_count_then_fetch_same_table(tmp_path)) == 1

    _write(tmp_path, "n.py", "def f(cur):\n    cur.execute('SELECT COUNT(*) FROM t')\n    cur.execute('SELECT * FROM (SELECT id FROM t LIMIT 10) x')\n")
    assert len(scan_count_then_fetch_same_table(tmp_path)) == 1

    _write(tmp_path, "n.py", "def f(cur):\n    cur.execute('SELECT COUNT(*) FROM t')\n    cur.execute('SELECT id FROM t LIMIT 10')\n")
    assert scan_count_then_fetch_same_table(tmp_path) == []


def test_async_primitive_reinit_tuple_defaults_and_import_spellings(tmp_path: Path):
    """F24 / F80 / F81: tuple-assigned `self.x`, default-argument primitives, and the
    `from asyncio import Lock` / `import asyncio as aio` spellings."""
    from pyutilz.dev.code_audit.async_primitive_reinit import scan_async_primitive_reinit_per_call

    # One directory per case: `_safe_parse` caches on (path, mtime, size), so rewriting one name
    # with same-sized content inside a single filesystem clock tick can be served the stale tree.
    def scan(case: str, source: str):
        """Write `source` as the only module of its own subdirectory, and scan just that."""
        directory = tmp_path / case
        directory.mkdir()
        (directory / "e.py").write_text(source, encoding="utf-8")
        return scan_async_primitive_reinit_per_call(directory)

    assert scan("tuple_init", "import asyncio\nclass C:\n    def __init__(self):\n        self.a, self.b = asyncio.Lock(), asyncio.Event()\n") == []
    assert scan("default_arg", "import asyncio\nasync def handler(x, lock=asyncio.Lock()):\n    return x\n") == []
    assert [f.line for f in scan("from_import", "from asyncio import Lock\nasync def f():\n    lk = Lock()\n    return lk\n")] == [3]
    assert [f.line for f in scan("module_alias", "import asyncio as aio\nasync def f():\n    lk = aio.Lock()\n    return lk\n")] == [3]


def test_accumulator_helper_bypassed_setup_names_match_whole_segments(tmp_path: Path):
    """F25: `load`/`copy`/`init` as substrings exempted `upload_batch`, `recopy` and friends."""
    _write(tmp_path, "stats_mixin.py", HELPER)
    for name in ("upload_batch", "download_page", "reload_rows", "payload_scan", "recopy"):
        _write(tmp_path, "parallel_mixin.py", "\nclass Parallel:\n    def %s(self, ids):\n        self.stats['total_paginated'] += len(ids)\n" % name)
        assert len(scan_accumulator_helper_bypassed(tmp_path)) == 1, name
    _write(tmp_path, "parallel_mixin.py", "\nclass Parallel:\n    def reset_stats(self, ids):\n        self.stats['total_paginated'] += len(ids)\n")
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_lock_exemption_needs_a_lock(tmp_path: Path):
    """F26: `block`, `clock` and `unblocked` all contain "lock" as a substring."""
    _write(tmp_path, "stats_mixin.py", HELPER)
    for ctx in ("self._block_reader", "blocking_section()", "unblocked_ctx", "clock_timer"):
        _write(tmp_path, "parallel_mixin.py", "\nclass Parallel:\n    def handle_dup(self, ids):\n        with %s:\n            self.stats['total_paginated'] += len(ids)\n" % ctx)
        assert len(scan_accumulator_helper_bypassed(tmp_path)) == 1, ctx
    _write(tmp_path, "parallel_mixin.py", "\nclass Parallel:\n    def handle_dup(self, ids):\n        with self._ids_lock:\n            self.stats['total_paginated'] += len(ids)\n")
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_owner_suppression_is_per_file(tmp_path: Path):
    """F151 / F152: a same-named helper elsewhere suppressed a real bypass, and the single-owner
    detail read `... own it -- they are what key ...`."""
    _write(tmp_path, "stats_mixin.py", HELPER)
    _write(tmp_path, "other.py", "\nclass Other:\n    def _inc_stat(self, ids):\n        self.stats['total_paginated'] += len(ids)\n")
    findings = scan_accumulator_helper_bypassed(tmp_path)
    assert [f.file for f in findings] == ["other.py"]
    assert "owns it -- it is what keys" in findings[0].detail


def test_assert_in_loop_detail_stays_per_site_without_ast_unparse(tmp_path: Path):
    """F62: on the 3.8 fallback both sweeps rendered `for <item> in <source>`, so one baseline
    entry silenced the whole file."""
    import ast as _ast

    from pyutilz.dev.code_audit.assert_in_loop import scan_assert_in_loop_reports_only_the_first

    _write(tmp_path, "t.py", "def test_x():\n    for r in load_rows():\n        assert r\n    for q in load_rows():\n        assert q\n")
    unparse = _ast.unparse
    del _ast.unparse
    try:
        details = [f.detail for f in scan_assert_in_loop_reports_only_the_first(tmp_path)]
    finally:
        _ast.unparse = unparse
    assert len(details) == 2
    assert len(set(details)) == 2


def test_getattr_literal_on_known_dataclass_honours_bases_and_methods(tmp_path: Path):
    """F64: inherited fields and the class's own methods were not in the field set."""
    from pyutilz.dev.code_audit.getattr_literal_on_known_dataclass import scan_getattr_literal_on_known_dataclass

    _write(tmp_path, "m.py", (
        "from dataclasses import dataclass\n\n@dataclass\nclass Base:\n    shared: int = 0\n\n"
        "@dataclass\nclass Child(Base):\n    own: int = 0\n\n    def helper(self):\n        return 1\n\n"
        'def use(c: Child):\n    return getattr(c, "shared", None), getattr(c, "helper", None)\n'
    ))
    assert scan_getattr_literal_on_known_dataclass(tmp_path) == []
    _write(tmp_path, "m.py", (
        "from dataclasses import dataclass\n\n@dataclass\nclass Child:\n    own: int = 0\n\n"
        'def use(c: Child):\n    return getattr(c, "missing", None)\n'
    ))
    assert len(scan_getattr_literal_on_known_dataclass(tmp_path)) == 1


def test_import_cycles_sees_package_init_relative_imports(tmp_path: Path):
    """F11 / F202: `pkg/__init__.py` resolved `from .a import f` one component too far up, and the
    reported file for a package member was `pkg.py`."""
    (tmp_path / "pkg").mkdir()
    _write(tmp_path, "pkg/__init__.py", "from .a import f\n")
    _write(tmp_path, "pkg/a.py", "from pkg import f\n")
    findings = scan_import_cycles(tmp_path)
    assert len(findings) == 1
    assert findings[0].file == "pkg/__init__.py"
    assert findings[0].snippet == "pkg -> pkg.a -> pkg"


def test_import_cycles_finds_a_cycle_under_a_source_tree_root(tmp_path: Path):
    """F12: the CLI documents `root` as a source-tree root (`./src`), where `root.name` matched no
    import target and the whole scan returned nothing."""
    (tmp_path / "pkg").mkdir()
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/a.py", "from pkg import b\n")
    _write(tmp_path, "pkg/b.py", "from pkg import a\n")
    assert [f.snippet for f in scan_import_cycles(tmp_path)] == ["pkg.a -> pkg.b -> pkg.a"]


def test_import_cycles_does_not_fabricate_one_from_a_from_dot_import(tmp_path: Path):
    """F213: `from . import other` used to add an edge to the base package."""
    (tmp_path / "pkg").mkdir()
    _write(tmp_path, "pkg/__init__.py", "from pkg.mod import a\n")
    _write(tmp_path, "pkg/mod.py", "from . import other\n")
    _write(tmp_path, "pkg/other.py", "b = 2\n")
    assert scan_import_cycles(tmp_path) == []


def test_column_no_write_path_line_survives_a_block_comment(tmp_path: Path):
    """F27 / F212: stripping `/*...*/` swallowed its newlines, and single-line DDL never matched."""
    _write(tmp_path, "m.py", "def r(cur):\n    cur.execute('SELECT payload FROM t')\n")
    (tmp_path / "sql").mkdir(exist_ok=True)
    (tmp_path / "sql" / "a.sql").write_text("/* a block\ncomment\nspanning lines */\nCREATE TABLE t (\n    payload text NOT NULL\n);\n", encoding="utf-8")
    findings = scan_column_no_write_path(tmp_path)
    assert [(f.line, f.snippet.strip()) for f in findings] == [(5, "payload text NOT NULL")]

    (tmp_path / "sql" / "a.sql").write_text("CREATE TABLE t (payload text NOT NULL);\n", encoding="utf-8")
    assert len(scan_column_no_write_path(tmp_path)) == 1

    _write(tmp_path, "m.py", "def r(cur):\n    cur.execute('SELECT payload FROM t')\n    cur.execute('INSERT INTO t (payload) VALUES (%s)')\n")
    assert scan_column_no_write_path(tmp_path) == []


def test_comment_citation_line_numbers_and_positions(tmp_path: Path):
    """F157 / F158 / F159: single-digit line numbers, every citation in one comment, and a
    docstring citation reported at its own line."""
    _write(tmp_path, "m.py", "# see line 7 of foo\nx = 1\n")
    assert len(scan_comment_cites_absolute_line(tmp_path)) == 1

    _write(tmp_path, "m.py", "# see line 42 and line 99\nx = 1\n")
    assert len(scan_comment_cites_absolute_line(tmp_path)) == 2

    _write(tmp_path, "m.py", '"""Doc first line.\n\nSee line 42 here.\n"""\nx = 1\n')
    assert [f.line for f in scan_comment_cites_absolute_line(tmp_path)] == [3]

    _write(tmp_path, "m.py", "# nothing cited here\nx = 1\n")
    assert scan_comment_cites_absolute_line(tmp_path) == []


def test_comment_names_missing_symbol_bails_out_on_an_empty_symbol_table(tmp_path: Path):
    """F156: a dead builtin allowlist was unioned into `known`, making it never empty."""
    _write(tmp_path, "m.py", "# cites `_gone()`\n")
    assert scan_comment_names_missing_symbol(tmp_path) == []
    _write(tmp_path, "m.py", "# cites `_gone()`\ndef f():\n    pass\n")
    assert len(scan_comment_names_missing_symbol(tmp_path)) == 1


def test_constructor_param_overwritten_positional_only_dedup_and_reachability(tmp_path: Path):
    """F102 / F160 / F161: PEP 570 params were invisible, sites were reported once per
    config-reading method, and a method merely LOGGING `self.config` counted."""
    _write(tmp_path, "a.py", 'class A:\n    def __init__(self, rate, /):\n        self._rate = rate\n\n    def refresh(self):\n        self._rate = cfg("rate")\n')
    assert len(scan_constructor_param_overwritten(tmp_path)) == 1

    _write(tmp_path, "a.py", (
        'class A:\n    def __init__(self, rate):\n        self._rate = rate\n\n'
        '    def refresh(self):\n        self._rate = cfg("rate")\n\n'
        '    def other(self):\n        return settings.get("x")\n\n'
        '    def third(self):\n        return config.get("y")\n'
    ))
    assert len(scan_constructor_param_overwritten(tmp_path)) == 1

    _write(tmp_path, "a.py", (
        'class A:\n    def __init__(self, rate):\n        self._rate = rate\n\n'
        '    def log_and_set(self, value):\n        log.info("cfg=%s", self.config)\n        self._rate = value\n'
    ))
    assert scan_constructor_param_overwritten(tmp_path) == []


def test_credential_logging_matches_snake_case_and_attribute_loggers(tmp_path: Path):
    """F29 / F198: a word-boundary regex never matched inside `db_password`, and only bare
    `logger`/`log`/`logging` receivers were recognised."""
    _write(tmp_path, "m.py", "def f(db_password, proxy_url, auth_token):\n    log.info('connecting %s %s %s', db_password, proxy_url, auth_token)\n")
    assert len(scan_credential_shaped_log_args(tmp_path)) == 1

    _write(tmp_path, "m.py", "class C:\n    def f(self, password):\n        self.logger.info('p %s', password)\n")
    assert len(scan_credential_shaped_log_args(tmp_path)) == 1

    _write(tmp_path, "m.py", "def f(bypass, count):\n    log.info('x %s %s', bypass, count)\n")
    assert scan_credential_shaped_log_args(tmp_path) == []


def test_docstring_numbers_ignores_prose_abbreviations_and_reads_environ_subscripts(tmp_path: Path):
    """F28 / F187: `e.g.`, `i.e.` and `run.py` discarded the whole line, and `os.environ["X"]` was
    not recognised as a configuration read."""
    template = 'def f():\n    """%s"""\n    return cfg("limit")\n'
    for doc in (
        "Prunes at a limit of 10 hits, 5 for rare sources.",
        "Prunes at a limit of 10 hits, e.g. rare sources.",
        "Prunes at a limit of 10 hits, i.e. aggressively.",
        "Prunes at a limit of 10 hits, per run.py invocation.",
    ):
        _write(tmp_path, "m.py", template % doc)
        assert len(scan_docstring_numbers_moved_to_config(tmp_path)) == 1, doc

    _write(tmp_path, "m.py", 'def f():\n    """Prunes at a limit of 10 hits, 5 for rare sources."""\n    return os.environ["PRUNE"]\n')
    assert len(scan_docstring_numbers_moved_to_config(tmp_path)) == 1

    _write(tmp_path, "m.py", template % "Prunes at a limit of MAX_HITS hits.")
    assert scan_docstring_numbers_moved_to_config(tmp_path) == []


def test_dead_wiring_does_not_seed_from_another_audited_file(tmp_path: Path):
    """F31: a callee merely NAMED in a sibling audited file was seeded live, so "called only by
    another dead public function is dead too" never held."""
    from pyutilz.dev.code_audit.dead_wiring import scan_dead_public_callables

    _write(tmp_path, "m1.py", "def dead_leaf():\n    return 1\n")
    _write(tmp_path, "m2.py", "def also_dead():\n    return dead_leaf()\n")
    assert sorted(f.file for f in scan_dead_public_callables(tmp_path)) == ["m1.py", "m2.py"]

    _write(tmp_path, "m1.py", "def alive():\n    return 1\n")
    _write(tmp_path, "m2.py", "HANDLERS = [alive]\n")
    assert scan_dead_public_callables(tmp_path) == []


def test_default_via_or_examines_a_chain_of_more_than_two_operands(tmp_path: Path):
    """F201: `arg or fallback or 5` was skipped outright."""
    from pyutilz.dev.code_audit.default_via_or import scan_default_via_or_trap

    _write(tmp_path, "m.py", "def f(arg, fallback):\n    x = arg or fallback or 5\n    return x\n")
    assert len(scan_default_via_or_trap(tmp_path)) == 2

    _write(tmp_path, "m.py", "def f(a, b):\n    if a or b:\n        return 1\n    return 0\n")
    assert scan_default_via_or_trap(tmp_path) == []


def test_field_text_agreement_rejects_a_cue_that_normalises_to_nothing():
    """F191: such a cue compiled to a double word boundary and matched every record."""
    from pyutilz.dev.code_audit.field_text_agreement import FieldTextRule, cues_in_text

    rule = FieldTextRule(name="x", field="f", text_fields=("t",), cues={"postmortem": ["_"]})
    assert cues_in_text(rule, "vital hanging") == {}
    rule = FieldTextRule(name="x", field="f", text_fields=("t",), cues={"postmortem": ["postmortem"]})
    assert cues_in_text(rule, "a post-mortem was held") == {"postmortem": "postmortem"}


def test_field_text_agreement_tiebreak_prefers_the_longest_cue():
    """F197: the alphabetically-first value won, so a rename changed the verdict."""
    from pyutilz.dev.code_audit.field_text_agreement import FieldTextRule, check_record

    rule = FieldTextRule(name="x", field="f", text_fields=("t",), cues={"alpha": ["one"], "zeta": ["a much longer phrase"]})
    verdict = check_record(rule, {"f": "", "t": "a much longer phrase and one"})
    assert verdict.supported == "zeta"
    assert verdict.alternatives == ("alpha",)


def test_guard_decidable_keyword_arguments_and_subscript_scope_and_ifexp_line(tmp_path: Path):
    """F72 / F164 / F165: any keyword name counted as an external write, a string-key store in an
    unrelated file suppressed the constant, and an `IfExp` reported the value's line."""
    _write(tmp_path, "m.py", "_ENABLED = False\n\ndef go():\n    helper(_ENABLED=1)\n    if _ENABLED:\n        recover()\n")
    assert [f.line for f in scan_guard_decidable_from_constants(tmp_path)] == [5]

    _write(tmp_path, "m.py", "_ENABLED = False\n\ndef go():\n    if _ENABLED:\n        recover()\n")
    _write(tmp_path, "other.py", "d = {}\nd['_ENABLED'] = 1\n")
    assert [(f.file, f.line) for f in scan_guard_decidable_from_constants(tmp_path)] == [("m.py", 4)]

    (tmp_path / "other.py").unlink()
    _write(tmp_path, "m.py", "_FLAG = False\n\ndef go():\n    return (\n        'a'\n        if _FLAG\n        else 'b'\n    )\n")
    assert [(f.line, f.snippet.strip()) for f in scan_guard_decidable_from_constants(tmp_path)] == [(6, "if _FLAG")]

    _write(tmp_path, "m.py", "_ENABLED = False\n\ndef go():\n    setattr(mod, '_ENABLED', 1)\n    if _ENABLED:\n        recover()\n")
    assert scan_guard_decidable_from_constants(tmp_path) == []


def test_hardcoded_test_path_judges_the_path_below_the_root(tmp_path: Path):
    """F94: a `"tests" in path.parts` test counted ancestors ABOVE the scan root."""
    from pyutilz.dev.code_audit.hardcoded_test_path import scan_hardcoded_absolute_path_in_test

    root = tmp_path / "tests" / "myproj"
    root.mkdir(parents=True)
    (root / "prod.py").write_text('DATA = "C:/Users/alice/data.csv"\n', encoding="utf-8")
    assert scan_hardcoded_absolute_path_in_test(root) == []
    (root / "test_x.py").write_text('DATA = "C:/Users/alice/data.csv"\n', encoding="utf-8")
    assert [f.file for f in scan_hardcoded_absolute_path_in_test(root)] == ["test_x.py"]


def test_lazy_log_assertion_reads_logger_log_formats_and_unittest_assertions(tmp_path: Path):
    """F73 / F166: `logger.log(LEVEL, fmt, ...)` had its LEVEL harvested as the format, and only
    the bare `assert` statement was scanned."""
    from pyutilz.dev.code_audit.lazy_log_assertion import scan_lazy_log_assertion

    _write(tmp_path, "prod.py", "import logging\ndef go(log, x):\n    log.log(logging.WARNING, 'Retried 3 times for %s', x)\n")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_a.py").write_text("def test_a(log):\n    assert 'Retried 3 times for' in str(log.log.call_args)\n", encoding="utf-8")
    assert scan_lazy_log_assertion(tmp_path) == []

    _write(tmp_path, "prod.py", "def go(log, n):\n    log.warning('reached only %s/3 items', n)\n")
    (tmp_path / "tests" / "test_a.py").write_text(
        "import unittest\nclass T(unittest.TestCase):\n    def test_b(self):\n" "        self.assertIn('reached only 0/3 items', str(log.warning.call_args))\n",
        encoding="utf-8",
    )
    assert len(scan_lazy_log_assertion(tmp_path)) == 1


def test_llm_max_tokens_cap_does_not_read_false_as_a_zero_literal(tmp_path: Path):
    """F193: an `== 0` test is true for `False`."""
    from pyutilz.dev.code_audit.llm_max_tokens_cap import scan_llm_call_missing_max_tokens_cap

    source = "from pyutilz.llm import get_llm_provider\n\ndef go():\n    p = get_llm_provider()\n    return p.generate('prompt', max_tokens=%s)\n"
    _write(tmp_path, "m.py", source % "False")
    assert scan_llm_call_missing_max_tokens_cap(tmp_path) == []
    _write(tmp_path, "m.py", source % "0")
    assert len(scan_llm_call_missing_max_tokens_cap(tmp_path)) == 1


def test_log_throttle_does_not_count_a_for_loops_own_iter_as_in_loop(tmp_path: Path):
    """F190: the iter expression evaluates once, not per iteration."""
    from pyutilz.dev.code_audit.log_throttle import scan_unthrottled_hot_loop_log

    _write(tmp_path, "m.py", "def f(items, logger):\n    for x in (logger.error('boom') or items):\n        pass\n")
    assert scan_unthrottled_hot_loop_log(tmp_path) == []
    _write(tmp_path, "m.py", "def f(items, logger):\n    for x in items:\n        logger.error('boom')\n")
    assert [f.line for f in scan_unthrottled_hot_loop_log(tmp_path)] == [3]


def test_asymmetric_except_siblings_needs_two_methods_and_a_compatible_inner_guard(tmp_path: Path):
    """F76 / F77 / F154 / F155: one method was reported as its own sibling, methods nested in a
    class-body `if` were invisible, the finding pointed at the `except` line, and any enclosing
    `try` counted as a guard."""
    pair = (
        "class Db:\n"
        "    def a(self):\n        try:\n            work()\n        except OSError:\n"
        "            try:\n                self.rollback()\n            except OSError:\n                pass\n\n"
        "    def b(self):\n        try:\n            work()\n        except OSError:\n            self.rollback()\n"
    )
    _write(tmp_path, "m.py", pair)
    findings = scan_asymmetric_except_siblings(tmp_path)
    assert [(f.line, f.snippet.strip()) for f in findings] == [(15, "self.rollback()")]

    nested = "class Db:\n    if True:\n" + "".join(("    " + line + "\n") if line.strip() else "\n" for line in pair.splitlines()[1:])
    _write(tmp_path, "m.py", nested)
    assert len(scan_asymmetric_except_siblings(tmp_path)) == 1

    _write(tmp_path, "m.py", (
        "class Db:\n"
        "    def run(self):\n        try:\n            work()\n        except OSError:\n"
        "            try:\n                self.rollback()\n            except OSError:\n                pass\n"
        "        try:\n            work()\n        except OSError:\n            self.rollback()\n"
    ))
    assert scan_asymmetric_except_siblings(tmp_path) == []

    incompatible = pair.replace(
        "        except OSError:\n            self.rollback()\n",
        "        except OSError:\n            try:\n                self.rollback()\n            except ValueError:\n                pass\n",
    )
    _write(tmp_path, "m.py", incompatible)
    assert len(scan_asymmetric_except_siblings(tmp_path)) == 1


def test_asymmetric_resource_guard_downgrades_an_in_memory_container(tmp_path: Path):
    """F195: P0 is the crash tier; a plain dict under a lock is a deliberate asymmetry."""
    from pyutilz.dev.code_audit.asymmetric_resource_guard import scan_asymmetric_resource_guard

    body = 'class C:\n    def a(self):\n        with self._lock:\n            self.%s.update({"k": 1})\n\n    def b(self):\n        self.%s.update({"j": 2})\n'
    _write(tmp_path, "m.py", body % ("_cache", "_cache"))
    assert [f.severity for f in scan_asymmetric_resource_guard(tmp_path)] == ["P2"]
    _write(tmp_path, "m.py", body % ("_db_conn", "_db_conn"))
    assert [f.severity for f in scan_asymmetric_resource_guard(tmp_path)] == ["P0"]


def test_console_unicode_scans_every_positional_argument(tmp_path: Path):
    """F205: only args[0] was inspected, though a later argument reaches the same console."""
    from pyutilz.dev.code_audit.console_unicode import scan_console_unicode

    _write(tmp_path, "m.py", "print('done', '\u2192')\n")
    assert len(scan_console_unicode(tmp_path)) == 1
    _write(tmp_path, "m.py", "logger.info('x %s', '\u2192')\n")
    assert len(scan_console_unicode(tmp_path)) == 1
    _write(tmp_path, "m.py", "print('ok', 'fine')\n")
    assert scan_console_unicode(tmp_path) == []


# =====================================================================================
# Regression tests for audit 2026-09-03 / 07-domain-core-dev-system, scanner modules m-z.
# Each test names the finding it pins; every one fails on the pre-fix scanner.
# =====================================================================================
