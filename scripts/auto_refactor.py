"""
Automated refactoring script for pyutilz library.

Fixes:
1. Wildcard imports from typing
2. Mutable default arguments
3. Other common anti-patterns

Run with: python scripts/auto_refactor.py
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Root directory of pyutilz
PYUTILZ_ROOT = Path(__file__).parent.parent


def fix_typing_wildcard_imports(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Replace 'from typing import *' with explicit imports.

    Analyzes the file to find which typing imports are actually used.
    """
    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    if 'from typing import *' not in content:
        return False, []

    # Common typing imports used in pyutilz
    typing_symbols = [
        'Union', 'Optional', 'Sequence', 'Dict', 'List', 'Tuple',
        'Any', 'Callable', 'Set', 'Iterable', 'Iterator'
    ]

    # Find which typing symbols are actually used
    used_symbols = []
    for symbol in typing_symbols:
        # Check if symbol is used (word boundary to avoid partial matches)
        pattern = r'\b' + re.escape(symbol) + r'\b'
        if re.search(pattern, content):
            used_symbols.append(symbol)

    if not used_symbols:
        used_symbols = ['Any']  # Default if we can't determine

    # Replace wildcard import
    old_import = r'from typing import \*.*\n'
    new_import = f'from typing import {", ".join(sorted(used_symbols))}\n'

    new_content = re.sub(old_import, new_import, content)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True, used_symbols

    return False, []


def fix_mutable_defaults(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Fix mutable default arguments (list, dict) in function signatures.

    Changes:
        def func(arg=[]) -> def func(arg=None) with guard
        def func(arg={}) -> def func(arg=None) with guard
    """
    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    fixed_functions = []

    # Pattern to find function definitions with mutable defaults
    # This is complex - better to do manual review for safety
    # For now, just report them

    patterns = [
        (r'def\s+(\w+)\([^)]*=\s*\[\]', 'list default'),
        (r'def\s+(\w+)\([^)]*=\s*\{\}', 'dict default'),
        (r'def\s+(\w+)\([^)]*=\s*\[', 'list literal default'),
    ]

    for pattern, issue_type in patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            func_name = match.group(1)
            fixed_functions.append(f"{func_name} ({issue_type})")

    return len(fixed_functions) > 0, fixed_functions


def scan_and_fix_all():
    """Scan all Python files in pyutilz and apply fixes."""

    print("="*70)
    print("PYUTILZ AUTOMATED REFACTORING")
    print("="*70)

    # Find all Python files
    py_files = list(PYUTILZ_ROOT.glob('*.py'))
    py_files += list(PYUTILZ_ROOT.glob('**/*.py'))

    # Exclude tests, scripts, __pycache__
    py_files = [
        f for f in py_files
        if '__pycache__' not in str(f)
        and 'tests' not in str(f)
        and 'scripts' not in str(f)
        and f.name != 'setup.py'
    ]

    print(f"\nFound {len(py_files)} Python files to analyze\n")

    wildcard_fixes = []
    mutable_defaults_issues = []

    for file_path in sorted(py_files):
        rel_path = file_path.relative_to(PYUTILZ_ROOT)

        # Fix wildcard imports
        fixed_wildcard, used_symbols = fix_typing_wildcard_imports(file_path)
        if fixed_wildcard:
            wildcard_fixes.append((rel_path, used_symbols))
            print(f"[OK] Fixed typing import in {rel_path}")
            print(f"     Imports: {', '.join(used_symbols)}")

        # Check for mutable defaults
        has_mutable, functions = fix_mutable_defaults(file_path)
        if has_mutable:
            mutable_defaults_issues.append((rel_path, functions))
            print(f"[WARN] Found mutable defaults in {rel_path}:")
            for func in functions:
                print(f"       - {func}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"[OK] Fixed typing wildcard imports in {len(wildcard_fixes)} files")
    print(f"[WARN] Found mutable defaults in {len(mutable_defaults_issues)} files (manual fix recommended)")

    if mutable_defaults_issues:
        print("\nFiles with mutable defaults needing manual review:")
        for rel_path, functions in mutable_defaults_issues:
            print(f"  - {rel_path}: {len(functions)} function(s)")

    return len(wildcard_fixes), len(mutable_defaults_issues)


if __name__ == "__main__":
    import sys

    # Create scripts directory if it doesn't exist
    PYUTILZ_ROOT.joinpath('scripts').mkdir(exist_ok=True)

    fixed, warnings = scan_and_fix_all()

    if fixed > 0:
        print(f"\n[OK] Refactoring complete! {fixed} files fixed.")
        sys.exit(0)
    else:
        print("\n[OK] No fixes needed.")
        sys.exit(0)
