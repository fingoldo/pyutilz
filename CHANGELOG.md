# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-18

### Added - Hardware Detection Migration
- **~30 hardware detection functions** migrated from ml_perf_test project to `system.system` module
- **CPU Detection**:
  - `get_cpu_info()` - Enhanced CPU detection via py-cpuinfo with better filtering
  - `get_wmi_cpuinfo()` - Windows WMI CPU detection with detailed hardware info
  - `get_lscpu_info()` - Linux lscpu parser with automatic type conversion
  - `get_linux_board_info()` - Linux motherboard info from /sys/devices
  - `parse_dmidecode_info()` - Linux dmidecode parser (BIOS, memory, etc.)
- **GPU Detection** (replaces old functions):
  - `get_nvidia_smi_info()` - Rich nvidia-smi XML parsing with full GPU stats
  - `get_gpu_cuda_capabilities()` - CUDA device attributes via numba.cuda
  - `get_cuda_gpu_details()` - Combined nvidia-smi + CUDA capabilities
  - `CUDA_SM_TO_CORES` - Compute capability to CUDA cores mapping constant
- **Power & Large Pages**:
  - `check_large_pages_support()` - Cross-platform large pages detection (Windows/Linux/macOS)
  - `get_power_plan()` - Cross-platform power plan detection
  - `get_battery_info()` - Battery status and charge level
- **WMI Helpers** (Windows):
  - `get_wmi_obj_as_dict()` - WMI object to dict conversion with type handling
  - `summarize_devices()` - Hardware aggregation with counts
  - `dict_to_tuple()` - Dictionary hashing helper
  - `decode_memory_type()` - DDR type decoder (DDR3/DDR4/DDR5)
  - `decode_cpu_upgrade_method()` - CPU socket type decoder
  - `summarize_system_info()` - Complete Windows system summary (GPU, RAM, Cache, BIOS)
- **OS & Software**:
  - `get_os_info()` - Enhanced OS detection with detailed info
  - `get_python_info()` - Python implementation and version detection
- **Monitoring**:
  - `ensure_idle_devices()` - Wait for CPU/GPU idle before benchmarks
  - `system.hardware_monitor.UtilizationMonitor` - Background thread monitoring for CPU/GPU/RAM utilization
- **Utilities**:
  - `remove_nas()` - Recursive N/A removal from dicts with type conversion

### Changed
- **`get_system_info()` enhanced** with new fields while maintaining backward compatibility:
  - `cpu_wmi_info`, `cpu_lscpu_info`, `cpu_board_info` (platform-specific)
  - `gpu_nvidia_smi_info`, `gpu_cuda_capabilities` (replaces old gpuinfo)
  - `large_pages_support`, `power_plan`, `battery_info`
  - `system_wmi_summary`, `dmidecode_info` (detailed hardware)
  - **Backward compatible**: Existing fields (`host_name`, `os_machine_guid`, `os_serial`) preserved for `distributed.py`

### Removed
- **Old GPU functions** (replaced with superior ml_perf_test implementations):
  - `compute_total_gpus_ram()` - Use `get_nvidia_smi_info()` instead
  - `get_gpuinfo_gpu_info()` - Use `get_nvidia_smi_info()` instead
  - `get_gpuutil_gpu_info()` - Use `get_nvidia_smi_info()` instead
  - `get_pycuda_gpu_info()` - Use `get_gpu_cuda_capabilities()` instead

### Dependencies
- **New optional dependencies** in `[system]` extra:
  - `py-cpuinfo>=9.0` - Enhanced CPU detection
  - `GPUtil>=1.4` - GPU monitoring
  - `xmltodict>=0.13` - nvidia-smi XML parsing
  - `pypiwin32>=223` - WMI support on Windows

### Testing
- **20 new tests** for hardware detection functions (`test_hardware_detection.py`)
- **95% success rate** on Windows (19/20 tests passing, 1 skipped for Linux)
- **Full test coverage** for CPU, GPU, power, OS, and monitoring functions
- **Cross-platform testing** with platform-specific markers

### Documentation
- **MIGRATION_TEST_REPORT.md** - Comprehensive migration and testing documentation
- Complete test results with hardware-specific outputs
- Dependency installation guide
- Integration examples for ml_perf_test project

### Migration Notes
- **ml_perf_test integration**: Hardware detection functions now imported from pyutilz
- **~1500 lines removed** from ml_perf_test.py (migrated to pyutilz)
- **Graceful dependency handling**: Functions return None with warnings if optional deps unavailable
- **Platform compatibility**: Proper guards for Windows/Linux/macOS-specific functionality

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
