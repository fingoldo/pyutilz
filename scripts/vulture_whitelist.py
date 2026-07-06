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
