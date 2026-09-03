"""Scanner tests for uncurated_star_export, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_broad_except_swallows,
    scan_default_via_or_trap,
    scan_log_only_except,
    scan_possibly_dead_import,
    scan_uncurated_star_exports,
)

from ._helpers import _write

# ---- uncurated_star_export -------------------------------------------


def test_uncurated_star_export_flagged(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def public_helper():
    return 1
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from .sub import *
""",
    )
    findings = scan_uncurated_star_exports(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "uncurated_star_export"


def test_uncurated_star_export_with_init_all_not_flagged(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def public_helper():
    return 1
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from .sub import *

__all__ = ["public_helper"]
""",
    )
    findings = scan_uncurated_star_exports(tmp_path)
    assert findings == []


def test_uncurated_star_export_with_submodule_all_not_flagged(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def public_helper():
    return 1

__all__ = ["public_helper"]
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from .sub import *
""",
    )
    findings = scan_uncurated_star_exports(tmp_path)
    assert findings == []


def test_uncurated_star_export_absolute_import_not_flagged(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "__init__.py",
        """
from numpy import *
""",
    )
    findings = scan_uncurated_star_exports(tmp_path)
    assert findings == []


def test_broad_except_nosec_comment_on_except_line_skipped(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def probe():
    try:
        import cupy as cp
        return cp, True
    except Exception:  # nosec B110 - GPU probe is opportunistic, CPU fallback below
        return None, False
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == [], f"nosec-documented swallow must not be flagged; got {findings}"


def test_broad_except_opportunistic_keyword_in_handler_body_skipped(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def probe(n_full, n_sub):
    try:
        from ._gpu import fast_path
        return fast_path(n_full, n_sub)
    except Exception:
        # GPU path is opportunistic; any failure falls through to the host path below.
        pass
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == [], f"opportunistic-documented swallow must not be flagged; got {findings}"


def test_broad_except_best_effort_keyword_hyphenated_and_spaced_both_match(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def a():
    try:
        risky()
    except Exception:
        pass  # best-effort cleanup, safe to skip

def b():
    try:
        risky()
    except Exception:
        pass  # best effort cleanup, safe to skip
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == []


def test_broad_except_unrelated_nosec_elsewhere_in_function_does_not_exempt(tmp_path: Path):
    """The exemption window is the handler's own line + body span -- an unrelated nosec comment on
    a DIFFERENT, unrelated line elsewhere in the same function must not accidentally exempt a real,
    undocumented swallow."""
    _write(
        tmp_path,
        "bad.py",
        """
def f(rows):
    eval(rows)  # nosec B307 - trusted internal input, unrelated to the block below
    out = []
    try:
        out.append(transform(rows))
    except Exception:
        continue
    return out
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings, "an unrelated nosec comment elsewhere in the function must not suppress a real finding"


def test_broad_except_no_rationale_still_flagged(tmp_path: Path):
    """Sanity: a plain undocumented swallow with none of the rationale markers is still flagged --
    confirms the new exemption isn't accidentally matching everything."""
    _write(
        tmp_path,
        "bad.py",
        """
def process(rows):
    out = []
    for r in rows:
        try:
            out.append(transform(r))
        except Exception:
            continue
    return out
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings, "undocumented swallow must still be flagged"


def test_default_via_or_boolean_valued_return_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def overlaps(lo_a, hi_a, lo_b, hi_b):
    return not (hi_a < lo_b or hi_b < lo_a)
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == [], f"pure-boolean return must not be flagged; got {findings}"


def test_default_via_or_isinstance_or_isinstance_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def is_not_or_ne(op):
    return isinstance(op, int) or isinstance(op, float)
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_startswith_endswith_not_flagged(tmp_path: Path):
    """``str.startswith``/``str.endswith`` always return a real bool, never an arbitrary
    falsy value -- same class of false positive already fixed for ``.all()``/``.any()``."""
    _write(
        tmp_path,
        "ok.py",
        """
def is_diminutive(lemma):
    return lemma.endswith("chen") or lemma.endswith("lein")
""",
    )
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_looks_and_has_predicate_names_not_flagged(tmp_path: Path):
    """``_foo_looks_bar(...)`` / ``_foo_has_bar(...)`` follow the same predicate-shaped-name
    convention as ``is_*``, just spelled differently -- both sides here are ``-> bool``."""
    _write(
        tmp_path,
        "ok.py",
        """
def _loop_looks_bounded_retry(test):
    return True


def _loop_body_has_meaningful_sleep(stmts):
    return True


def check(test, stmts):
    return _loop_looks_bounded_retry(test) or _loop_body_has_meaningful_sleep(stmts)
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_underscore_prefixed_is_predicate_not_flagged(tmp_path: Path):
    """A private helper's leading underscore (``_is_known_immutable_scalar_annotation``) must not
    defeat the ``is_*`` predicate-name recognition -- module-privacy doesn't change the naming
    convention's meaning."""
    _write(
        tmp_path,
        "ok.py",
        """
def _is_known_immutable_scalar_annotation(x):
    return True


def check(a, b):
    return _is_known_immutable_scalar_annotation(a) or _is_known_immutable_scalar_annotation(b)
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_empty_tuple_default_not_flagged(tmp_path: Path):
    """``x or ()`` is the tuple-literal spelling of the same trivial-empty-container idiom already
    covered for ``[]``/``{}``/``set()`` -- empty in, empty out, no distinct value to clobber."""
    _write(
        tmp_path,
        "ok.py",
        """
def normalize(items):
    for g in items or ():
        pass
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_boolean_valued_assignment_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def check(a, b):
    ok = (a > 0) or (b > 0)
    return ok
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_non_boolean_return_still_flagged(tmp_path: Path):
    """Sanity: a genuine default-via-or trap in a return statement is still caught -- the new
    exemption only suppresses PURE-boolean operands, not arbitrary return-position ors."""
    _write(
        tmp_path,
        "bad.py",
        """
def get_count(x):
    return x.count or 5
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings, "a non-boolean-valued or-default in a return must still be flagged"


def test_possibly_dead_import_facade_reexport_consumed_via_from_import_elsewhere(tmp_path: Path):
    """A name re-exported by a package __init__.py, consumed elsewhere ONLY via
    `from package import name` (never as `package.name` attribute access), must not be flagged."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def do_thing():
    return 1
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from pkg.sub import do_thing
""",
    )
    consumer_dir = tmp_path / "consumer"
    consumer_dir.mkdir()
    _write(
        consumer_dir,
        "user.py",
        """
from pkg import do_thing

do_thing()
""",
    )
    findings = scan_possibly_dead_import(tmp_path)
    assert findings == [], f"facade re-export consumed via a downstream from-import must not be flagged; got {findings}"


def test_possibly_dead_import_facade_reexport_never_imported_anywhere_still_flagged(tmp_path: Path):
    """Sanity: a name imported into __init__.py but genuinely never consumed anywhere (no bare-name
    use, no attribute access, no downstream from-import) must still be flagged."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def do_thing():
    return 1
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from pkg.sub import do_thing
""",
    )
    findings = scan_possibly_dead_import(tmp_path)
    assert any(f.file.endswith("__init__.py") for f in findings), "a genuinely unconsumed re-export must still be flagged"


def test_log_only_except_nosec_documented_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def process(rows):
    validation_errors = []
    try:
        risky()
    except Exception as e:  # nosec B110 - opportunistic path, logging is sufficient here
        logger.warning("failed: %s", e)
""",
    )
    findings = scan_log_only_except(tmp_path)
    assert findings == [], f"nosec-documented log-only except must not be flagged; got {findings}"


# ---- F97: star export resolves ImportFrom.level -------------------------------------


def test_uncurated_star_exports_honours_the_import_level(tmp_path: Path):
    (tmp_path / "outer" / "inner").mkdir(parents=True)
    _write(tmp_path, "outer/__init__.py", "")
    _write(tmp_path, "outer/shared.py", "def a():\n    pass\n")
    _write(tmp_path, "outer/inner/__init__.py", "from ..shared import *\n")
    _write(tmp_path, "outer/inner/shared.py", '__all__ = ["b"]\n\n\ndef b():\n    pass\n')
    assert len(scan_uncurated_star_exports(tmp_path)) == 1


def test_uncurated_star_exports_accepts_a_curated_parent_module(tmp_path: Path):
    (tmp_path / "outer" / "inner").mkdir(parents=True)
    _write(tmp_path, "outer/__init__.py", "")
    _write(tmp_path, "outer/shared.py", '__all__ = ["a"]\n\n\ndef a():\n    pass\n')
    _write(tmp_path, "outer/inner/__init__.py", "from ..shared import *\n")
    _write(tmp_path, "outer/inner/shared.py", "def b():\n    pass\n")
    assert scan_uncurated_star_exports(tmp_path) == []
