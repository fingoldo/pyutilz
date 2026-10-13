"""Classified exceptions for the py-ci-shared ``uncalled_functions`` gate.

Each entry is a bare function name the gate must not count, with the one-line reason it is allowed to stay
without a production call site. Categories: deprecated aliases kept for backward compatibility, PEP 562 module
``__dir__`` hooks invoked by ``dir()``, helpers superseded by a live function that does the same work, and
``atexit`` handlers registered by decorator. Plain public utility API re-exported from a package facade is exempted by the gate itself
(``public_api_from_exports``), so it is not listed here (owner decision 2026-10-14).
"""

from typing import Dict

UNCALLED_FUNCTIONS_IGNORE: Dict[str, str] = {
    "__dir__": "PEP 562 module hook invoked by dir(), not by a call site in the code.",
    "EnsurePgTableExists": "Deprecated wrapper in database/db/legacy.py that warns and forwards to ensure_pg_table_exists in database/db/schema.py.",
    "GetIdByKeyFieldAndInsertIfNeeded": "Deprecated alias superseded by get_id_by_key_field_and_insert_if_needed.",
    "ReadTableIntoDic": "Deprecated alias superseded by read_table_into_dict.",
    "ReadTableIntoDicReversed": "Deprecated alias superseded by read_table_into_dict_reversed.",
    "MakeSetExcludedClause": "Deprecated alias superseded by make_set_excluded_clause in database/db/sql_helpers.py.",
    "remove_stale_columns": "Deprecated alias superseded by get_non_stale_columns in data/pandaslib/frames.py.",
    "_attribute_names_in_corpus": "Thin accessor superseded by _corpus_import_index, which the dead-import scanner calls once for both indexes.",
    "_from_import_names_by_file": "Thin accessor superseded by _corpus_import_index, which the dead-import scanner calls once for both indexes.",
    "_dataclass_fields": "Superseded: the getattr-literal scanner resolves members from _class_shapes directly, with no caller for this mapping.",
    "_root_of": "Superseded: tautological_guard pairs targets by full key prefix; root collapse was rejected as a false P1.",
    "_cleanup_temp_dirs": "atexit handler registered by the @atexit.register decorator; the uncalled finder does not see decorators.",
    "array_location": "Public residency check returning device or host, defined in performance.kernel_tuning; no in-repo caller; kernel-tuning hook.",
    "decode_cpu_upgrade_method": "Public WMI decoder mapping a CPU upgrade-method code to a socket type in system.system._common; exposed by the star import of system.system, which the facade scan does not read; no in-repo caller.",
    "decode_memory_type": "Public WMI decoder mapping a memory-type code to a readable name in system.system._common; exposed by the star import of system.system; no in-repo caller.",
    "show_biggest_session_objects": "Public diagnostic that logs own process RAM and the largest session objects in system.system.memory; exposed by the star import of system.system; no in-repo caller.",
    "show_tracemalloc_snapshot": "Public diagnostic that takes a tracemalloc snapshot and prints its top lines in system.system.memory; exposed by the star import of system.system; no in-repo caller.",
    "beep": "Public Windows audible-beep helper via winsound in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "count_app_instances": "Public counter of running processes matching a name or command-line substring in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "ensure_idle_devices": "Public pre-benchmark gate waiting until CPU and GPU devices are idle in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "get_libs_versions": "Public mapping of module names to their __version__ strings in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "get_locale_settings": "Public dict of locale parameters in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "get_max_affordable_workers_count": "Public count of physical CPU cores free for parallel workers in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "get_python_info": "Public Python implementation and version details in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "get_script_file": "Public base-filename helper defaulting to the calling module in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "get_utc_unix_timestamp": "Public current UTC Unix timestamp in whole seconds in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "report_large_objects": "Public logger of module-level globals above a deep-size threshold in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "tqdmu_lazy_start": "Public tqdm drop-in that starts its elapsed clock lazily in system.system.misc; exposed by the star import of system.system; no in-repo caller.",
    "compute_total_gpus_ram": "Public aggregate GPU RAM statistics over a list of GPU info dicts in system.system.probing; exposed by the star import of system.system; no in-repo caller.",
}
