"""Vulture whitelist: confirmed false positives / documented no-ops, not to be re-flagged.

Format follows vulture's own ``--make-whitelist`` convention: a bare name as an
expression statement counts as a "use" in vulture's model (an assignment does NOT --
it creates a new binding instead of referencing an existing one).

This file is never imported by production code. Pass it alongside the real target on
the vulture command line: ``vulture src/pyutilz scripts/vulture_whitelist.py``.
"""

# core/pandaslib/dtypes.py get_df_memory_consumption -- explicitly documented in its
# own docstring as a deprecated no-op kept for back-compat with positional/keyword callers.
max_cols

# text/similarity.py normalize_sentence -- legacy flag with no implemented behavior and
# no caller relying on it; left in the signature rather than removing public API surface.
bSports

# performance/kernel_tuning/code_versioning.py _normalized_source_by_code -- code_key
# participates in the lru_cache key (see _code_identity_key's docstring: content-aware
# invalidation on an in-place __code__ swap) but is never read inside the function body.
code_key

# system/hardware_monitor.py UtilizationMonitor.__exit__ -- standard context-manager
# protocol signature (exc_type, exc_val, exc_tb); never inspected since this cleanup path
# stops the monitor unconditionally regardless of how the with-block exited.
exc_type
exc_val
exc_tb
