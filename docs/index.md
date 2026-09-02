# pyutilz

[![CI](https://github.com/fingoldo/pyutilz/workflows/CI/badge.svg)](https://github.com/fingoldo/pyutilz/actions)
[![MyPy](https://github.com/fingoldo/pyutilz/actions/workflows/mypy-full.yml/badge.svg)](https://github.com/fingoldo/pyutilz/actions/workflows/mypy-full.yml)
[![Black](https://github.com/fingoldo/pyutilz/workflows/Black/badge.svg)](https://github.com/fingoldo/pyutilz/actions)
[![codecov](https://codecov.io/gh/fingoldo/pyutilz/branch/master/graph/badge.svg)](https://codecov.io/gh/fingoldo/pyutilz)
[![codecov-numba](https://img.shields.io/codecov/c/github/fingoldo/pyutilz/master?flag=numba-disabled&label=codecov-numba)](https://codecov.io/gh/fingoldo/pyutilz/flags)
[![codecov-full](https://img.shields.io/codecov/c/github/fingoldo/pyutilz/master?flag=combined&label=codecov-full)](https://codecov.io/gh/fingoldo/pyutilz/flags)
[![PyPI](https://img.shields.io/pypi/v/pyutilz.svg)](https://pypi.org/project/pyutilz/)
[![Python](https://img.shields.io/pypi/pyversions/pyutilz.svg)](https://pypi.org/project/pyutilz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fingoldo/pyutilz/blob/master/LICENSE)
[![docs](https://github.com/fingoldo/pyutilz/actions/workflows/docs.yml/badge.svg)](https://fingoldo.github.io/pyutilz/)

A Python utilities library covering data-frame ops, databases, web/cloud, system monitoring, parallelism, and a unified async LLM-provider interface. The core installs nine hard dependencies -- `numba`, `numpy`, `joblib`, `portalocker`, `psutil`, `pandas`, `tqdm`, `pympler` (plus `tomli` below Python 3.11) -- because `pyutilz.core.pythonlib` and `pyutilz.system.system`, the modules nearly every other subpackage imports, use them unconditionally at import time; `pyproject.toml` records the rationale per dependency. Everything heavier than that (scipy, Pillow, selenium, the cloud SDKs, the LLM SDKs, spaCy, ...) is opt-in through an extras group, so you install only the domains you actually use.

See the [Modules](modules.md) reference for what each sub-package does, and the guides below for the subsystems with real design decisions behind them.

## Installation

```bash
pip install pyutilz[all,dev]          # full install (recommended -- see the note below)

pip install pyutilz                   # core only: numba, numpy, joblib, portalocker, psutil, pandas, tqdm, pympler
pip install pyutilz[dataframes]       # pandas + pyarrow + polars
pip install pyutilz[database]         # SQLAlchemy + psycopg2 + pymysql + redis
pip install pyutilz[web]              # selenium, requests, undetected-chromedriver, curl-cffi
pip install pyutilz[cloud]            # boto3 + google-cloud-storage
pip install pyutilz[nlp]              # spacy (>=3.10 only) + nltk + tiktoken + jellyfish + beautifulsoup4
pip install pyutilz[llm]              # anthropic + google-genai (>=3.9 only) + httpx + tenacity + pydantic + pydantic-settings + tiktoken
pip install pyutilz[system]           # scipy + Pillow + py-cpuinfo + GPUtil + xmltodict + jellyfish
pip install pyutilz[stats]            # documented empty alias -- pyutilz.stats needs only core numpy
pip install pyutilz[speedups]         # orjson -- drop-in accelerator, every use site falls back to stdlib json without it
pip install pyutilz[dash]             # flask + dash + dash-bootstrap-components (pyutilz.dev.dashlib)
pip install pyutilz[prefect]          # prefect (pyutilz.system.scheduling.prefect)
pip install pyutilz[tensorflow]       # tensorflow (system.parallel.set_tf_gpu only)
pip install pyutilz[gpu]              # cupy -- see the caveat below
pip install pyutilz[docs]             # mkdocs-material, to build this documentation site
pip install pyutilz[dev]              # pytest (+ cov/benchmark/asyncio/instafail/progress/timeout), ruff, black (>=3.10 only), mypy, bandit
```

`[all]` = `pandas,polars,database,web,cloud,nlp,llm,system,stats,speedups`. It deliberately leaves out four
groups, so `pip install pyutilz[all]` canNOT import `pyutilz.dev.dashlib` or
`pyutilz.system.scheduling.prefect` -- add the extra explicitly (`pip install "pyutilz[all,dash]"`):

- `[dash]` -- a whole flask/dash web-server stack for one notebook-dashboard helper.
- `[prefect]` -- a full workflow-orchestration runtime with a large transitive tree.
- `[tensorflow]` -- a multi-hundred-MB ML framework used by exactly one function.
- `[gpu]` -- PyPI's `cupy` is sdist-only, so this extra triggers a full local NVCC build. Install
  the matching binary wheel yourself instead (`pip install cupy-cuda12x`) and skip the extra;
  nothing breaks without it, since every cupy import site is `try/except`-guarded.

`pyproject.toml`'s `all`/`gpu` comments are the source of truth for these exclusions.

For development:

```bash
git clone https://github.com/fingoldo/pyutilz.git
cd pyutilz
pip install -e ".[all,dev]"
pip install -r requirements-dev.txt   # py-ci-shared -- a git checkout, so it cannot live in [dev]
pip install pre-commit vulture        # not declared in any extras group; the hooks need both
pre-commit install
pytest
```

To build this documentation site locally: `pip install -e ".[docs]" && mkdocs serve`.

Requires Python 3.8+. Tested on 3.8 through 3.14.

## Quick examples

**Shrink a DataFrame's memory** — auto-downcast every column to the
narrowest type that holds the data without precision loss; typical
50-80% reduction on real-world tabular data:

```python
from pyutilz.data.pandaslib import optimize_dtypes
df = optimize_dtypes(df)
```

**Unified LLM interface across 7 providers** — same `generate()` /
`generate_json()` surface (plus `generate_stream()` on the
OpenAI-compatible providers); switch by changing one string. See the
[LLM providers guide](guides/llm_providers.md) for the full picture.

```python
from pyutilz.llm import get_llm_provider

p = get_llm_provider("openrouter", model="anthropic/claude-sonnet-4.6")
text = await p.generate("Summarise this", system="You are concise.")
```

**Per-host kernel-tuning cache** — auto-tunes which CUDA/numba/cupy
variant of a hot kernel to run, per hardware fingerprint. See the
[kernel tuning cache guide](guides/kernel_tuning_cache.md).

```python
from pyutilz.performance.kernel_tuning import KernelTuningCache, hw_fingerprint

cache = KernelTuningCache.load_or_create()
print(hw_fingerprint())
```

**Sidecar-verified pickle load** — refuses to unpickle a payload whose
`.sha256` companion is missing or mismatched. See the
[safe pickle guide](guides/safe_pickle.md).

```python
from pyutilz.core.safe_pickle import safe_dump, safe_load

safe_dump(model, "model.pkl")   # writes model.pkl + model.pkl.sha256
model = safe_load("model.pkl")  # verifies the sidecar before unpickling
```

## Security

- Database operations use parameterised queries; `validate_sql_identifier`
  rejects identifiers that don't match `^[A-Za-z_][A-Za-z0-9_]*$`.
- `subprocess` calls never pass `shell=True`.
- Bandit (`bandit -ll`) and Vulture dead-code scans run as blocking
  gates both locally (pre-commit) and in CI, triaged to zero findings.
- LLM API keys are read from `.env` via `pydantic-settings`; the file is
  gitignored and a [detect-secrets](https://github.com/Yelp/detect-secrets)
  pre-commit hook blocks accidental in-source commits.

## License

MIT — see [LICENSE](https://github.com/fingoldo/pyutilz/blob/master/LICENSE).
