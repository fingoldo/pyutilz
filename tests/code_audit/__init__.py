"""Unit tests for pyutilz.dev.code_audit AST scanners, one module per scanner family.

Each scanner gets a positive case (constructed snippet that MUST be flagged) and a
negative case (constructed snippet that MUST NOT be flagged). Tests use tmp_path so the
audit runs against a hermetic directory; no cross-test bleed.

Mirrors the production split in ``src/pyutilz/dev/code_audit/``: one module per scanner
family, so neither side of the feature is a single multi-thousand-line file.
"""
