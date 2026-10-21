"""(internal) getattr_unknown_attribute now lives in py_ci_shared.code_audit.getattr_unknown_attribute; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.getattr_unknown_attribute import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _add_class_body_names,
    _add_module_level_names,
    _class_attribute_names,
    _dynamic_attribute_classes,
    _function_receiver_ids,
    _in_tree_bound_names,
    _in_tree_class_names,
    _is_plain_in_tree_class,
    _module_receiver_names,
    _receivers_defined_in_tree,
    _self_receiver_ids,
    scan_getattr_unknown_attribute,
)

from py_ci_shared.code_audit import getattr_unknown_attribute as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
