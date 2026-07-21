# Contributing to PyUtilz

First off, thank you for considering contributing to PyUtilz! It's people like you that make PyUtilz such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our commitment to fostering an open and welcoming environment. Be respectful and constructive.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Environment details** (Python version, OS, dependencies)
- **Code samples** if applicable
- **Error messages** with full traceback

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear title** describing the enhancement
- **Provide detailed description** of the proposed functionality
- **Explain why** this enhancement would be useful
- **Include examples** of how the feature would be used

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Commit your changes (see commit message guidelines)
7. Push to your fork
8. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

Note: some `[dev]` tooling requires a newer interpreter than the package's own 3.8 floor —
`black` needs Python >=3.10 and `py-ci-shared` (this project's Black-filtered-apply / mypy-cache
tooling) needs >=3.9. On 3.8/3.9, `pip install -e .[all,dev]` below succeeds but silently skips
those two packages (their `pyproject.toml` entries carry explicit `python_version` markers for
this reason); `black .` / the pre-commit hooks that depend on them just won't be installed. If
you'll be doing dev-tooling work (formatting, pre-commit), use Python >=3.10.

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/pyutilz.git
cd pyutilz

# Install in development mode with all extras
pip install -e .[all,dev]

# Install the pre-commit hooks (runs the meta-test suite + linters on every commit)
pre-commit install

# Verify installation
pytest
```

### Optional Dependencies

Install only what you need:

```bash
pip install -e .[pandas]      # For pandas development
pip install -e .[database]    # For database development
pip install -e .[web]         # For web scraping development
pip install -e .[dev]         # For development tools only
```

## Code Standards

### Style Guide

We use **Black** for code formatting and **Ruff** for linting — but NEVER run raw `black`/`black --fix`
repo-wide: this project excludes two Black behaviors (arg/collection-list explosion and blank-line
insertion) that stock Black has no flag for, via a shared filtered-apply script:

```bash
# Format code (line-length: 160) -- only the files you're touching, via py-ci-shared
python -m py_ci_shared.black_filtered_apply --config pyproject.toml --write <files>

# Check formatting (CI / dry-run, whole repo)
python -m py_ci_shared.black_filtered_apply --config pyproject.toml --check .

# Lint code (repo-wide read-only; never --fix beyond files you're actively editing)
ruff check .

# Fix auto-fixable issues -- ONLY for the specific files you just edited, and only
# genuinely mechanical/safe rules (e.g. F541 empty f-strings); never a repo-wide --fix,
# and never for rules that can change behavior or delete a re-export (F401)
ruff check --fix <files>
```

A project-wide reformat or `ruff check --fix` beyond files you're already editing requires
explicit maintainer approval first — see this project's own `CLAUDE.md` for the incident that
established this rule.

**Key style points:**
- Line length: **160 characters** (not 88!)
- Use type hints where practical
- Docstrings: Google style format
- Imports: sorted with isort-compatible ordering

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, List, Dict

def process_data(data: List[str], max_items: Optional[int] = None) -> Dict[str, int]:
    """Process data and return statistics."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_pandaslib.py

# Run specific test
pytest tests/test_dtypes_regression.py::test_optimize_dtypes_does_not_truncate_fractional_object_column

# Run with coverage
pytest --cov=. --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Writing Tests

- **Location**: `tests/test_<module_name>.py`
- **Naming**: `test_<function_name>` or `test_<feature>`
- **Coverage**: Aim for >80% coverage for new code. Note the measured percentage is against a
  shrunk denominator: `pyproject.toml`'s `[tool.coverage.run] omit` list excludes `scheduling/`,
  `cloud/`, `web/browser.py`, `dev/dashlib.py`, `dev/notebook_init.py`, and `text/tokenizers.py`
  entirely (heavy-IO/external-service code not exercised by the unit suite) — those files aren't
  measured at all, not measured-and-low. Don't add new modules to that omit list without a reason;
  prefer writing the tests instead.
- **Fixtures**: Use pytest fixtures for common setups

Example test:

```python
import pytest
from pyutilz.pandaslib import optimize_dtypes

def test_optimize_dtypes():
    """Test that optimize_dtypes reduces memory usage."""
    import pandas as pd

    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.0, 2.0, 3.0]
    })

    original_memory = df.memory_usage(deep=True).sum()
    df_optimized = optimize_dtypes(df)
    optimized_memory = df_optimized.memory_usage(deep=True).sum()

    assert optimized_memory <= original_memory
```

### Test Categories

Mark tests with appropriate markers:

```python
@pytest.mark.slow
def test_expensive_operation():
    pass

@pytest.mark.integration
def test_database_integration():
    pass

@pytest.mark.gpu
def test_gpu_operation():
    pass
```

Run specific categories:

```bash
pytest -m "not slow"        # Skip slow tests
pytest -m integration       # Only integration tests
```

## Security

### Reporting Vulnerabilities

**DO NOT** open public issues for security vulnerabilities.

Instead, email security concerns to the maintainers or use GitHub's private vulnerability reporting.

### Security Best Practices

- **SQL Injection**: Always use parameterized queries
- **Command Injection**: Never use `shell=True` in subprocess
- **Input Validation**: Validate all user inputs
- **Dependency Security**: Keep dependencies updated

## Commit Message Guidelines

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(pandaslib): add memory optimization for string columns

Implement automatic conversion of object dtype columns to category
when cardinality is low. Reduces memory by up to 80% for categorical data.

Closes #123
```

```
fix(db): prevent SQL injection in table name validation

Replace string formatting with proper parameterized queries.
Add validate_sql_identifier() calls before dynamic table names.
```

For a breaking change, describe it in the commit body (as above) and add a "BREAKING" entry
under CHANGELOG.md's "Unreleased" section (see "Pull Request Process" below) -- that CHANGELOG
entry, not a commit-message footer, is what this project actually treats as the source of truth
for release notes; there is no commit-msg hook or CI check that parses a `BREAKING CHANGE:`
footer, so don't rely on one to surface the change.

## Pull Request Process

1. **Update documentation** if adding features
2. **Add tests** for new functionality (maintain >80% coverage)
3. **Update CHANGELOG.md** under "Unreleased" section
4. **Run full test suite** and ensure all tests pass
5. **Run linters** (black, ruff) and fix issues
6. **Write clear PR description** explaining changes
7. **Link related issues** using "Closes #123" syntax

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Code formatted with black (via `py_ci_shared.black_filtered_apply`, not raw `black`)
- [ ] Linting passes (`ruff check .`)
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated
- [ ] Type hints added (where practical)
- [ ] Docstrings added/updated
- [ ] No security issues introduced

## Module Structure

When adding new modules:

```python
# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed
ensure_installed("package1 package2")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any
import package1

# ----------------------------------------------------------------------------------------------------------------------------
# Code
# ----------------------------------------------------------------------------------------------------------------------------

def your_function():
    """Your code here."""
    pass
```

## Performance Considerations

- **Benchmark**: Use `tests/benchmark_*.py` scripts to measure performance
- **Profile**: Use `cProfile` or `line_profiler` for bottlenecks
- **Document**: Note performance improvements in CHANGELOG.md
- **Verify**: Include benchmark results in PR description

## Questions?

Feel free to ask questions by:
- Opening a GitHub issue with the "question" label
- Starting a discussion in GitHub Discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- GitHub contributors page
- Release notes for significant contributions
- Documentation credits (for major features)

---

Thank you for contributing to PyUtilz! 🎉
