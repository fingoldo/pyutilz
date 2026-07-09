"""Entry point for running the code_audit package as ``python -m pyutilz.dev.code_audit``."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
