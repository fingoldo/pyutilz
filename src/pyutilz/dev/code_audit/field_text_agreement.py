"""Back-compat shim: ``field_text_agreement`` now lives at ``pyutilz.dev.field_text_agreement``.

It is a runtime record checker, not an AST source scanner, so it is no longer a member of the
scanner package -- every other module here is a ``scan_*(root, exclude_dirs=...) -> list[Finding]``
over a source tree. This module re-exports it unchanged so
``pyutilz.dev.code_audit.field_text_agreement`` keeps resolving.
"""

from ..field_text_agreement import (  # noqa: F401  -- re-export shim
    AGREE,
    CONTRADICT,
    KIND_OPPOSED,
    KIND_UNFILLED,
    UNCHECKABLE,
    FieldTextReport,
    FieldTextRule,
    FieldTextVerdict,
    check_all,
    check_record,
    check_records,
    cues_in_text,
    normalise_text,
)
