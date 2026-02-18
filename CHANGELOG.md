# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.90] - 2026-02-18

### Added
- Public GitHub release with full packaging infrastructure
- Comprehensive test suite (142 tests passing)
- CI/CD automation with GitHub Actions
- Quality badges (CI, coverage, Codacy, security)
- Modern packaging with pyproject.toml
- Professional README with documentation
- CHANGELOG for version tracking
- CONTRIBUTING guidelines for developers
- Code coverage measurement with pytest-cov
- Security scanning with bandit
- Code style enforcement with black (line-length: 160)
- Linting with ruff

### Fixed
- **SECURITY**: SQL injection vulnerabilities in db.py (6 locations)
- **SECURITY**: Command injection risks in system.py
- Broken imports in cloud.py, distributed.py, matrix.py (.python → .pythonlib)
- Resource leaks (tracemalloc snapshots, temporary directories)
- Import errors preventing module loading
- Multiple bare except clauses replaced with proper exception handling

### Changed
- All print() calls replaced with proper logging
- type() comparisons replaced with isinstance()
- Module structure improved for better maintainability
- Test coverage improved with additional test cases

### Performance
- pandaslib: optimize_dtypes 2x faster (verified benchmarks)
- pandaslib: nullify_standard_values 200x faster
- pandaslib: get_df_memory_consumption 15x faster
- pandaslib: ensure_float32 5x faster

## [0.1-0.89] - 2024-2026

### Summary
- Internal development versions
- Core functionality development for 31 modules
- Initial test suite creation
- Performance optimizations
- Bug fixes and improvements

---

## Module Categories

### Data Science & Analytics
- pandaslib, polarslib, numpylib, numbalib, matrix

### Database & Storage
- db, redislib, deltalakes, serialization

### Web & Cloud
- web, browser, cloud, graphql

### System & Infrastructure
- system, parallel, monitoring, distributed, scheduling

### Text & NLP
- strings, tokenizers, similarity

### Development Tools
- pythonlib, logginglib, benchmarking, dashlib

### Specialized
- image, filemaker, com, openai
